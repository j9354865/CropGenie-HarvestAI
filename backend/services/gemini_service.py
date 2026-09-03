"""
Gemini AI service — handles all communication with Google's Generative AI API.

Responsibilities:
    • Initialise the Gemini client with the API key from config
    • Maintain a system prompt tuned for agriculture advice
    • Accept conversation history and return AI responses
    • Fall back to an available model if the configured one is deprecated
    • Analyse plant images for disease detection (multimodal vision)
"""

import json
import logging
import re
import time
import threading
from typing import Callable, List, Optional

import google.generativeai as genai

from backend.config import settings
from backend.models.schemas import DiseaseDetectionResponse

logger = logging.getLogger(__name__)


# ── Global API request counter (thread-safe) ──────────────────────────
_api_request_counter = {"total": 0, "chat": 0, "success": 0, "retried": 0, "failed": 0}
_counter_lock = threading.Lock()


def _increment_counter(key: str) -> int:
    """Atomically increment a counter and return its new value."""
    with _counter_lock:
        _api_request_counter[key] = _api_request_counter.get(key, 0) + 1
        return _api_request_counter[key]


def _get_counter_snapshot() -> dict:
    """Return a snapshot of all counters for logging."""
    with _counter_lock:
        return dict(_api_request_counter)


# ── History size cap ──────────────────────────────────────────────────
# Gemini has a context window, but sending too many messages wastes tokens
# and makes us hit rate limits faster.  10 messages = ~5 exchanges which is
# enough for conversational context without bloating the request.
MAX_HISTORY_MESSAGES = 10


# ── Rate-limit detection & retry helpers (module-level) ──────────────

def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Return True if *exc* is a Gemini quota / rate-limit error.

    Detects via the exception class name (ResourceExhausted),
    the string representation, and common HTTP status codes.
    """
    cls_name = type(exc).__name__
    if cls_name in ("ResourceExhausted", "TooManyRequests", "RateLimitError"):
        return True
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "429",
            "quota",
            "resource has been exhausted",
            "rate limit",
            "resource_exhausted",
        )
    )


def _parse_retry_delay(exc_str: str) -> int:
    """Extract the retry delay (seconds) from a Gemini error message."""
    import re as _re

    m = _re.search(r"retry_delay.*?(\d+)", exc_str, _re.DOTALL)
    if m:
        return int(m.group(1))
    m = _re.search(r"seconds:\s*(\d+)", exc_str)
    if m:
        return int(m.group(1))
    return 0


def _call_gemini_api(call_fn: Callable, method_name: str = "Gemini"):
    """
    Call a Gemini API function with automatic rate-limit retry.

    Retry strategy (conservative to avoid wasting quota):
      • Max 2 attempts (1 original + 1 retry).
      • If the error includes a ``retry_delay`` ≤ 15 s, wait that long.
      • Otherwise use exponential back-off capped at 10 s.
      • If the retry delay exceeds 15 s, raise immediately — the user
        should try again later rather than wait a full minute.

    For non-rate-limit errors the exception is re-raised immediately.
    """
    max_attempts = 2
    max_wait_seconds = 15  # Do NOT make the user wait longer than this

    # Count this as ONE logical API call (retries don't add to the count)
    _increment_counter("total")
    _increment_counter(method_name.lower() if method_name != "Gemini" else "chat")
    request_num = _api_request_counter["total"]

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                "[%s] Gemini API request #%d (attempt %d/%d)",
                method_name, request_num, attempt, max_attempts,
            )
            result = call_fn()
            _increment_counter("success")
            return result

        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= max_attempts:
                _increment_counter("failed")
                logger.warning(
                    "[%s] Request #%d failed (attempt %d/%d): %s",
                    method_name, request_num, attempt, max_attempts,
                    str(exc)[:200],
                )
                raise

            # Determine how long to wait
            retry_after = _parse_retry_delay(str(exc))
            if 0 < retry_after <= max_wait_seconds:
                wait = retry_after
            elif retry_after > max_wait_seconds:
                # Server wants us to wait too long — fail fast with a
                # clear message so the user can try again later.
                _increment_counter("failed")
                logger.warning(
                    "[%s] Rate limited. Server wants %ds wait (>%ds cap). "
                    "Failing fast. Snapshot: %s",
                    method_name, retry_after, max_wait_seconds,
                    _get_counter_snapshot(),
                )
                raise RuntimeError(
                    f"AI is rate-limited. Please wait about "
                    f"{retry_after} seconds and try again."
                ) from exc
            else:
                wait = min(2 ** attempt * 3, max_wait_seconds)

            _increment_counter("retried")
            logger.warning(
                "[%s] Rate limit hit (attempt %d/%d). "
                "Waiting %ds before retry… | Snapshot: %s | %s",
                method_name, attempt, max_attempts, wait,
                _get_counter_snapshot(), str(exc)[:120],
            )
            time.sleep(wait)


# ── System prompt — shapes the AI's persona ─────────────────────────
SYSTEM_PROMPT = (
    "You are CropGenie, a friendly farming helper. "
    "You help farmers with easy, clear advice about:\n"
    "  \u2022 Growing crops and planning seasons\n"
    "  \u2022 Plant diseases, bugs, and how to treat them\n"
    "  \u2022 Soil health, fertilizer, and watering\n"
    "  \u2022 Weather and how it affects farming\n"
    "  \u2022 Taking care of animals and chickens\n\n"
    "VERY IMPORTANT RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Write like you are talking to a farmer "
    "who never went to school.\n"
    "  \u2013 Use SHORT sentences. One idea per sentence.\n"
    "  \u2013 Keep your answer SHORT — 2 to 4 sentences is best.\n"
    "  \u2013 Tell the farmer EXACTLY what to do next.\n"
    "  \u2013 Do NOT use big or difficult words.\n"
    "  \u2013 If you must use a hard word, explain it right away.\n"
    "  \u2013 Use bullet points only for steps (max 4-5 steps).\n"
    "  \u2013 If the question is not about farming, say so kindly.\n\n"
    "LANGUAGE RULES (very important):\n"
    "  \u2013 You will get a language instruction with EVERY message.\n"
    "    ALWAYS reply in that language. Never switch languages.\n"
    "  \u2013 For URDU: Use simple, everyday Urdu that village farmers speak. "
    "Short sentences. No bookish Urdu. Keep common English farming words "
    "(like 'fertilizer', 'spray', 'pesticide') if farmers know them better.\n"
    "  \u2013 For PASHTO: Use simple, everyday Pashto that village farmers speak. "
    "Short sentences. No bookish Pashto. Keep English or Urdu farming words "
    "if farmers know them better.\n"
    "  \u2013 For ENGLISH: Use very basic English. Short words. Short sentences. "
    "Like talking to a child.\n"
)

# ── Central language-instruction map ─────────────────────────────────
# Used by ALL generation methods to force a specific response language
# on every single request. This is the KEY mechanism that prevents
# Gemini from drifting to a different language mid-conversation.
LANGUAGE_INSTRUCTIONS = {
    "en-US": (
        "IMPORTANT: Reply ONLY in very simple English. "
        "Use short words and short sentences. "
        "Write like you are talking to a child or a farmer "
        "who did not go to school. No difficult words."
    ),
    "en-GB": (
        "IMPORTANT: Reply ONLY in very simple English. "
        "Use short words and short sentences. "
        "Write like you are talking to a child or a farmer "
        "who did not go to school. No difficult words."
    ),
    "en-IN": (
        "IMPORTANT: Reply ONLY in very simple English. "
        "Use short words and short sentences. "
        "Write like you are talking to a child or a farmer "
        "who did not go to school. No difficult words."
    ),
    "ur-PK": (
        "IMPORTANT: Reply ONLY in simple, everyday Urdu (\u0627\u0631\u062f\u0648). "
        "Use very short sentences. Use only common words that "
        "village farmers use every day. No bookish or difficult "
        "Urdu words. Keep English farming words like 'fertilizer', "
        "'spray', 'pesticide' if farmers know them better. "
        "Do not mix in English sentences."
    ),
    "ps-AF": (
        "IMPORTANT: Reply ONLY in simple, everyday Pashto (\u067e\u069a\u062a\u0648). "
        "Use very short sentences. Use only common words that "
        "village farmers use every day. No bookish or difficult "
        "Pashto words. Keep English or Urdu farming words if "
        "farmers know them better. Do not mix in English sentences."
    ),
}

# Ordered preference list for automatic fallback when the configured
# model is unavailable.  First match wins.
# NOTE: Google retires model names periodically.  If the top entry
# starts returning 404, move the model name from the error message
# to the top of this list (and update GEMINI_MODEL in .env).
_FALLBACK_MODELS = [
    "gemini-3.6-flash",       # Current recommended model
    "gemini-3.8-flash",       # Newest Flash
    "gemini-3.7-flash",       # Previous gen
    "gemini-3.5-flash",       # Older but still available
    "gemini-3.5-flash-lite",  # Lightweight alternative
]

# Weather-advisory system prompt — used by generate_weather_advice()
WEATHER_SYSTEM_PROMPT = (
    "You are CropGenie's weather helper. You get real weather data and "
    "give easy farming tips in short bullet points (max 5).\n\n"
    "Talk about:\n"
    "  \u2022 Watering \u2014 should the farmer water now or wait?\n"
    "  \u2022 Spraying \u2014 is it safe to spray pesticides or fertilizer?\n"
    "  \u2022 Warnings \u2014 heavy rain, very hot weather, frost, disease risk\n"
    "  \u2022 Other useful weather tips\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Do NOT repeat numbers from the weather data. Explain what they mean.\n"
    "  \u2013 Tell the farmer EXACTLY what to do.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language. Use simple everyday words."
)

# Disease-detection system prompt — used by analyze_disease()
# Forces JSON output so the backend can return a typed response.
DISEASE_SYSTEM_PROMPT = (
    "You are CropGenie's plant disease detection specialist. "
    "You MUST respond with ONLY valid JSON — no markdown, no explanation, "
    "just the raw JSON object.\n"
    "If the image is unclear or does not show a plant, set "
    'health_status to "unknown" and explain in the summary field.'
)

# Fertilizer-advisory system prompt — used by generate_fertilizer_explanation()
FERTILIZER_SYSTEM_PROMPT = (
    "You are CropGenie's fertilizer helper. You get fertilizer data from "
    "a smart model, plus crop name, soil type, and NPK info. "
    "Give a clear, easy explanation.\n\n"
    "Talk about:\n"
    "  \u2022 Why this fertilizer is good for their crop and soil\n"
    "  \u2022 How much to use per acre and total for their farm\n"
    "  \u2022 When and how to put it on the field\n"
    "  \u2022 Warnings \u2014 what happens if they use too much or too little\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Use bullet points for steps (max 5).\n"
    "  \u2013 Do NOT repeat raw numbers. Explain what they mean.\n"
    "  \u2013 Tell the farmer EXACTLY what to do.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language. Use simple everyday words."
)

# Location-insights system prompt — used by generate_location_insights()
LOCATION_SYSTEM_PROMPT = (
    "You are CropGenie's area farming guide. You get the farmer's "
    "location in Pakistan \u2014 their zone, climate, soil, and current "
    "weather. Give easy, useful tips for their area.\n\n"
    "Talk about:\n"
    "  \u2022 Which crops grow best in their area right now\n"
    "  \u2022 Simple summary of their soil and weather\n"
    "  \u2022 Watering tips for their area \u2014 is water easy or hard to get?\n"
    "  \u2022 Main problems in their area and what to do about them\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Use bullet points for tips (max 6).\n"
    "  \u2013 Do NOT repeat zone data. Explain it in easy words.\n"
    "  \u2013 Tell the farmer EXACTLY what to do.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language. Use simple everyday words."
)

# Market-advisory system prompt — used by generate_market_advice()
MARKET_SYSTEM_PROMPT = (
    "You are CropGenie's market price helper for Pakistani farmers. "
    "You get real mandi (market) price data and trends for a crop. "
    "Give easy selling advice.\n\n"
    "Talk about:\n"
    "  \u2022 Are prices going up, down, or staying the same?\n"
    "  \u2022 Best time to sell based on past patterns\n"
    "  \u2022 Should they sell now or wait?\n"
    "  \u2022 How to store the crop safely if they wait\n\n"
    "MONEY RULES (very important):\n"
    "  \u2022 Always show prices in PKR (Rs.) per maund (1 maund = 40 kg) "
    "or per bori (100 kg bag).\n"
    "  \u2022 NEVER use dollars, never per-kg, never Indian rupees.\n"
    "  \u2022 If the price is in a different unit (like per 100 pieces "
    "for kinnow), keep that unit and say so.\n\n"
    "If you know the farmer's area, mention:\n"
    "  \u2022 Is their area a big growing area for this crop?\n"
    "  \u2022 Any local market tips for their area\n\n"
    "At the end, remind them: prices are based on past data. "
    "Always check with their local mandi before selling.\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Use bullet points for tips (max 5).\n"
    "  \u2013 Do NOT repeat raw numbers. Explain what they mean.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language. Use simple everyday words."
)

# Irrigation-advisory system prompt — used by generate_irrigation_advice()
IRRIGATION_SYSTEM_PROMPT = (
    "You are a watering helper for farmers in Pakistan.\n\n"
    "You get the crop name, weather, rain forecast, and when they "
    "last watered. Give a SHORT, EASY watering tip.\n\n"
    "Your answer MUST have:\n"
    "  1. A clear answer: 'Water now' or 'Wait' or 'Water in X days'\n"
    "  2. One short sentence saying why\n"
    "  3. If watering: how much water \u2014 light, medium, or heavy\n\n"
    "RULES:\n"
    "  \u2013 Keep it under 60 words.\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Do NOT repeat weather numbers. Explain what they mean.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language."
)

# Farming-calendar system prompt — used by generate_calendar_advice()
CALENDAR_SYSTEM_PROMPT = (
    "You are a farming schedule helper for farmers in Pakistan.\n\n"
    "You get a crop name and area. Make a full, easy farming calendar "
    "from planting to harvesting.\n\n"
    "Your calendar MUST have these sections:\n"
    "  1. When to plant (sowing date or time)\n"
    "  2. When to water (key watering times \u2014 not every day, just main stages)\n"
    "  3. When to put fertilizer (what kind and roughly when)\n"
    "  4. When to spray for bugs and diseases (important times)\n"
    "  5. When to harvest\n\n"
    "Use simple timing like 'Week 2', 'Month 3' or real dates if you "
    "know the sowing date. Keep it practical for the farmer's area.\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Keep total answer under 200 words.\n"
    "  \u2013 Tell the farmer EXACTLY what to do at each step.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language."
)

# Government-support system prompt — used by generate_gov_support_advice()
GOV_SUPPORT_SYSTEM_PROMPT = (
    "You are a helper who explains government farming programs to "
    "Pakistani farmers in easy words.\n\n"
    "Farmers may ask about: subsidies (money help from the government), "
    "cheap loans, crop insurance, free seeds or fertilizer, Kisan Card, "
    "or other government help for farmers.\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 If you are NOT SURE about exact details (amounts, dates, "
    "who can apply), say so clearly. Tell the farmer to ask their "
    "local Agriculture Office or Zarai Taraqiati Bank (ZTBL).\n"
    "  \u2013 Never make up details you are not sure about.\n"
    "  \u2013 Keep your answer under 150 words.\n"
    "  \u2013 Tell the farmer EXACTLY what to do next.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language. Use simple everyday words."
)

# Crop-recommendation system prompt — used by generate_crop_recommendation_explanation()
CROP_RECOMMENDATION_SYSTEM_PROMPT = (
    "You are CropGenie's crop suggestion helper. You get data from a "
    "smart model about which crop is best for the farmer's soil and "
    "weather. Explain it in easy words.\n\n"
    "Talk about:\n"
    "  \u2022 Why this crop is a good choice for their soil and weather\n"
    "  \u2022 What the crop needs to grow well (water, sunlight, etc.)\n"
    "  \u2022 How to prepare the soil and plant the seeds\n"
    "  \u2022 Any warnings (like soil too acidic, not enough rain)\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Use bullet points for steps (max 5).\n"
    "  \u2013 Do NOT repeat raw numbers. Explain what they mean.\n"
    "  \u2013 Tell the farmer EXACTLY what to do.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language. Use simple everyday words."
)

# Grafting-advisor system prompt — used by generate_grafting_advice()
GRAFTING_SYSTEM_PROMPT = (
    "You are a plant grafting (paiwand) expert for farmers in Pakistan.\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Write like talking to a farmer "
    "who never went to school.\n"
    "  \u2013 If you use a hard word, explain it right away.\n"
    "  \u2013 Keep your answer under 200 words.\n"
    "  \u2013 Use SHORT sentences. One idea per sentence.\n\n"
    "Cover these five parts, each clearly labeled:\n"
    "  1. Which plants can be grafted together\n"
    "  2. Best way to do it \u2014 explain in easy steps\n"
    "  3. Best time of year to do it\n"
    "  4. What good it does for the farmer\n"
    "  5. Common mistakes to avoid"
)


# Harvest-loss predictor system prompt — used by generate_harvest_loss_advice()
HARVEST_LOSS_SYSTEM_PROMPT = (
    "You are a harvest timing helper for farmers in Pakistan.\n\n"
    "Your job: tell the farmer how much crop and money they will lose "
    "if they wait too long to harvest. Use the real weather and market "
    "data in the prompt. Do NOT make up fake numbers.\n\n"
    "For each of these three choices, give:\n"
    "  \u2022 Day 0 (harvest today): expected crop % and loss %\n"
    "  \u2022 Day 5 (wait 5 days): expected crop % and loss %\n"
    "  \u2022 Day 10 (wait 10 days): expected crop % and loss %\n\n"
    "Label each one clearly: 'Day 0:', 'Day 5:', 'Day 10:'\n\n"
    "Then give ONE final answer using EXACTLY one of these:\n"
    "  'HARVEST NOW' or 'WAIT 2 DAYS' or 'HARVEST BEFORE RAIN'\n\n"
    "Do NOT make up other labels.\n\n"
    "Tips:\n"
    "  \u2013 If rain is coming (>40% chance), say 'HARVEST BEFORE RAIN'\n"
    "  \u2013 If prices are going up and weather is clear, 'WAIT 2 DAYS' is OK\n"
    "  \u2013 If crop is fully ripe or too ripe, say 'HARVEST NOW'\n\n"
    "RULES:\n"
    "  \u2013 Use VERY SIMPLE words. Short sentences.\n"
    "  \u2013 Keep it under 150 words.\n"
    "  \u2013 Tell the farmer EXACTLY what to do.\n\n"
    "LANGUAGE: A language instruction will be added to each request. "
    "ALWAYS reply in that language. Use simple everyday words."
)


class GeminiService:
    """Wrapper around Google Generative AI for agriculture chat."""

    def __init__(self) -> None:
        """Configure the Gemini SDK and resolve a usable model."""
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        genai.configure(api_key=settings.gemini_api_key)
        self._model = self._resolve_model(
            settings.gemini_model, system_instruction=SYSTEM_PROMPT
        )
        self._weather_model = self._resolve_model(
            settings.gemini_model, system_instruction=WEATHER_SYSTEM_PROMPT
        )
        self._disease_model = self._resolve_model(
            settings.gemini_model, system_instruction=DISEASE_SYSTEM_PROMPT
        )
        self._fertilizer_model = self._resolve_model(
            settings.gemini_model, system_instruction=FERTILIZER_SYSTEM_PROMPT
        )
        self._location_model = self._resolve_model(
            settings.gemini_model, system_instruction=LOCATION_SYSTEM_PROMPT
        )
        self._market_model = self._resolve_model(
            settings.gemini_model, system_instruction=MARKET_SYSTEM_PROMPT
        )
        self._irrigation_model = self._resolve_model(
            settings.gemini_model, system_instruction=IRRIGATION_SYSTEM_PROMPT
        )
        self._calendar_model = self._resolve_model(
            settings.gemini_model, system_instruction=CALENDAR_SYSTEM_PROMPT
        )
        self._gov_support_model = self._resolve_model(
            settings.gemini_model, system_instruction=GOV_SUPPORT_SYSTEM_PROMPT
        )
        self._crop_recommendation_model = self._resolve_model(
            settings.gemini_model,
            system_instruction=CROP_RECOMMENDATION_SYSTEM_PROMPT,
        )
        self._grafting_model = self._resolve_model(
            settings.gemini_model,
            system_instruction=GRAFTING_SYSTEM_PROMPT,
        )
        self._harvest_loss_model = self._resolve_model(
            settings.gemini_model,
            system_instruction=HARVEST_LOSS_SYSTEM_PROMPT,
        )

    # ── Model resolution ────────────────────────────────────────────

    def _resolve_model(
        self,
        model_name: str,
        system_instruction: str = SYSTEM_PROMPT,
    ) -> genai.GenerativeModel:
        """
        Try to initialise the requested model.  If it fails (e.g. the
        model was deprecated), query the API for available models that
        support ``generateContent`` and pick the best fallback.
    
        Returns a ready-to-use GenerativeModel instance.
        """
        # 1. Try the configured model first
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
            )
            # Quick smoke-test: list the model to verify it exists.
            # genai.get_model() will raise if the model is not found.
            genai.get_model(f"models/{model_name}")
            logger.info("Gemini model loaded: %s", model_name)
            return model
    
        except Exception as exc:
            logger.warning(
                "Configured model '%s' is unavailable (%s). "
                "Searching for a fallback\u2026",
                model_name,
                exc,
            )
    
        # 2. Try the ordered fallback list
        fallback = self._find_fallback()
        if fallback:
            logger.info("Using fallback model: %s", fallback)
            return genai.GenerativeModel(
                model_name=fallback,
                system_instruction=system_instruction,
            )

        # 3. Nothing works — surface a clear error
        raise RuntimeError(
            f"No usable Gemini model found. Configured model '{model_name}' "
            f"is unavailable and no fallback could be resolved. "
            f"Check GEMINI_MODEL in your .env file and see "
            f"https://ai.google.dev/gemini-api/docs/models for current names."
        )

    def _find_fallback(self) -> Optional[str]:
        """
        Query the Gemini API for models that support ``generateContent``
        and return the first one that matches our preference list.
        """
        try:
            available = self._list_generate_content_models()
            logger.info(
                "Available generateContent models: %s",
                available,
            )
        except Exception as exc:
            logger.error("Failed to list Gemini models: %s", exc)
            return None

        for candidate in _FALLBACK_MODELS:
            if candidate in available:
                return candidate

        # If none of our preferred models matched, return the first
        # available one as a last resort.
        return available[0] if available else None

    @staticmethod
    def _list_generate_content_models() -> List[str]:
        """
        Return the short names (e.g. ``gemini-2.5-flash``) of every
        model that supports the ``generateContent`` method.
        """
        models: List[str] = []
        for m in genai.list_models():
            # m.name looks like "models/gemini-2.5-flash"
            short_name = m.name.removeprefix("models/")
            if "generateContent" in m.supported_generation_methods:
                models.append(short_name)
        return sorted(models)

    # ── Public API ───────────────────────────────────────────────────

    def generate_response(
        self,
        message: str,
        history: List[dict],
        language: str = "en-US",
    ) -> str:
        """
        Generate an AI reply for the farmer's message.

        Parameters
        ----------
        message : str
            The farmer's latest question / statement.
        history : list[dict]
            Prior messages as {"role": "user"|"assistant", "content": "..."}.
        language : str
            Locale code — the reply will be forced into this language
            via an explicit per-request instruction.

        Returns
        -------
        str
            The AI-generated response text.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails for any reason.
        """
        try:
            # Build a forced language instruction that overrides any
            # language drift from conversation history.
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(
                language, LANGUAGE_INSTRUCTIONS["en-US"]
            )
            message_with_lang = f"{lang_instruction}\n\n{message}"

            # Cap history to avoid wasting token quota on old messages.
            # Only the most recent MAX_HISTORY_MESSAGES are sent.
            trimmed_history = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
            if len(history) > MAX_HISTORY_MESSAGES:
                logger.info(
                    "[Chat] History trimmed: %d → %d messages (cap: %d)",
                    len(history), len(trimmed_history), MAX_HISTORY_MESSAGES,
                )

            # Convert our history dicts into the format Gemini expects
            gemini_history = [
                {"role": msg["role"], "parts": [msg["content"]]}
                for msg in trimmed_history
            ]

            logger.info(
                "[Chat] Gemini API request: msg_len=%d, history_len=%d, lang=%s",
                len(message), len(gemini_history), language,
            )

            chat = self._model.start_chat(history=gemini_history)
            response = _call_gemini_api(lambda: chat.send_message(message_with_lang), "Chat")

            logger.info(
                "[Chat] Gemini response received: %d chars",
                len(response.text.strip()),
            )
            return response.text.strip()

        except RuntimeError:
            raise  # re-raise RuntimeError (from _call_gemini_api) unchanged
        except Exception as exc:
            raise RuntimeError(
                f"AI service temporarily unavailable. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    # ── Weather Advisory ────────────────────────────────────────────

    def generate_weather_advice(
        self,
        weather_summary: str,
        language: str = "en-US",
    ) -> str:
        """
        Generate practical farming advice based on current and forecast
        weather data.

        Unlike ``generate_response``, this uses a dedicated weather-advice
        system prompt and does **not** carry conversation history — each
        weather recommendation is a standalone assessment.

        Parameters
        ----------
        weather_summary : str
            A formatted text block containing current conditions and
            forecast data (assembled by the weather route).

        Returns
        -------
        str
            Concise, farmer-friendly bullet-point advice.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        try:
            # Use the pre-built weather model (has its own system prompt)
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(
                language, LANGUAGE_INSTRUCTIONS["en-US"]
            )
            prompt = f"{lang_instruction}\n\n{weather_summary}"
            chat = self._weather_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Weather advice generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    # ── Disease Detection (Vision / Multimodal) ──────────────────────

    # Language instructions embedded in the analysis prompt
    _LANGUAGE_INSTRUCTIONS = {
        "en-US": "Respond in English.",
        "en-GB": "Respond in English.",
        "en-IN": "Respond in English.",
        "ur-PK": "Respond in Urdu (اردو).",
        "ps-AF": "Respond in Pashto (پښتو).",
    }

    def analyze_disease(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        language: str = "en-US",
    ) -> DiseaseDetectionResponse:
        """
        Analyse a plant image using Gemini's multimodal vision capability.

        Sends the raw image bytes alongside a structured prompt that asks
        Gemini to identify the plant and assess its health.  The model's
        JSON response is parsed and returned as a typed
        ``DiseaseDetectionResponse``.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (JPEG or PNG, already size-validated by the route).
        mime_type : str
            MIME type of the image (``image/jpeg`` or ``image/png``).
        language : str
            Language code — Gemini will return the analysis in this language.

        Returns
        -------
        DiseaseDetectionResponse
            Structured detection result.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails or the response cannot be parsed.
        """
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(
            language, LANGUAGE_INSTRUCTIONS["en-US"]
        )
        prompt = self._build_disease_prompt(lang_instruction)

        try:
            # Multimodal request: image bytes + text prompt in one call
            response = _call_gemini_api(
                lambda: self._disease_model.generate_content(
                    [
                        {"mime_type": mime_type, "data": image_bytes},
                        prompt,
                    ],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=2048,
                    ),
                ),
                "Disease",
            )

            raw_text = response.text.strip()
            logger.debug(
                "Disease raw response (first 200 chars): %s", raw_text[:200]
            )

            return self._parse_disease_response(raw_text)

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Disease analysis failed. Please try again with a "
                "clearer photo of the plant. (details: {})".format(exc)
            ) from exc

    @staticmethod
    def _build_disease_prompt(lang_instruction: str) -> str:
        """Return the structured text prompt sent alongside the image."""
        return (
            "Analyse this plant image for disease detection.\n\n"
            f"{lang_instruction}\n\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "plant_name": "identified crop or plant name",\n'
            '  "health_status": "healthy" | "diseased" | '
            '"nutrient_deficiency" | "unknown",\n'
            '  "disease_name": "disease or issue name, or null if healthy",\n'
            '  "cause": "likely cause of the disease or issue, or null",\n'
            '  "treatment": ["step 1", "step 2", "step 3"],\n'
            '  "prevention": ["tip 1", "tip 2", "tip 3"],\n'
            '  "summary": "brief farmer-friendly summary of findings"\n'
            "}\n\n"
            "If the image is unclear, does not show a plant, or you cannot "
            "confidently\n"
            "identify the issue, set health_status to 'unknown' and explain "
            "the\n"
            "limitation in the summary field.\n"
            "Use simple, farmer-friendly language.\n"
            "Return ONLY the JSON — no markdown, no backticks, no preamble."
        )

    @staticmethod
    def _parse_disease_response(raw_text: str) -> DiseaseDetectionResponse:
        """
        Parse the raw Gemini response into a DiseaseDetectionResponse.

        Gemini may wrap the JSON in ```json``` markdown blocks despite the
        system prompt asking it not to — this method strips those wrappers.
        """
        # Strip optional ```json ... ``` wrappers
        text = raw_text
        if "```" in text:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Fallback: return the raw text as the summary so the user
            # still sees useful output instead of an error.
            logger.warning(
                "Disease JSON parse failed (%s). Returning raw text as summary.",
                exc,
            )
            data = {
                "plant_name": "Unknown",
                "health_status": "unknown",
                "disease_name": None,
                "cause": None,
                "treatment": [],
                "prevention": [],
                "summary": raw_text,  # Gemini's raw analysis is still useful
            }

        return DiseaseDetectionResponse(**data)

    # ── Fertilizer Advisory ──────────────────────────────────────────

    def generate_fertilizer_explanation(
        self,
        fertilizer_name: str,
        crop: str,
        soil_type: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        acres: float,
        temperature: float,
        humidity: float,
        moisture: float,
        language: str = "en-US",
    ) -> str:
        """
        Generate a farmer-friendly explanation for an ML-predicted fertilizer.

        Unlike ``generate_response``, this uses a dedicated fertilizer-advisory
        system prompt and does **not** carry conversation history — each
        recommendation is a standalone assessment.

        Parameters
        ----------
        fertilizer_name : str
            The ML-predicted fertilizer (e.g. 'Urea', 'DAP', '14-35-14').
        crop : str
            Crop type.
        soil_type : str
            Soil type.
        nitrogen, phosphorus, potassium : float
            Current NPK levels.
        acres : float
            Farm size in acres.
        temperature, humidity, moisture : float
            Environmental conditions.

        Returns
        -------
        str
            Concise, farmer-friendly explanation.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(
            language, LANGUAGE_INSTRUCTIONS["en-US"]
        )
        prompt = (
            f"{lang_instruction}\n\n"
            f"Suggested fertilizer: {fertilizer_name}\n\n"
            f"Context:\n"
            f"  Crop: {crop}\n"
            f"  Soil: {soil_type}\n"
            f"  NPK: N={nitrogen}, P={phosphorus}, K={potassium}\n"
            f"  Farm size: {acres} acres\n"
            f"  Temperature: {temperature}°C\n"
            f"  Humidity: {humidity}%\n"
            f"  Soil moisture: {moisture}%\n\n"
            f"Explain why this fertilizer is good, how much to use "
            f"per acre and total, when and how to apply, and any warnings."
        )
        try:
            chat = self._fertilizer_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Fertilizer explanation generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    def generate_location_insights(
        self,
        zone_name: str,
        climate: str,
        soil: str,
        crops: str,
        irrigation: str,
        challenges: str,
        city: str,
        temperature: float | None = None,
        humidity: int | None = None,
        weather_desc: str | None = None,
        language: str = "en-US",
    ) -> str:
        """
        Generate region-specific farming insights for a Pakistan district.

        Uses a dedicated location-advisory system prompt and does **not**
        carry conversation history — each insight is standalone.

        Parameters
        ----------
        zone_name : str
            Full agro-ecological zone name.
        climate, soil, irrigation : str
            Zone characteristics from the reference data.
        crops : str
            Comma-separated list of typical crops.
        challenges : str
            Comma-separated list of key challenges.
        city : str
            The city/district entered by the user.
        temperature, humidity, weather_desc : optional
            Current weather data (may be None if weather fetch failed).

        Returns
        -------
        str
            Farmer-friendly regional farming guidance.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        weather_block = ""
        if temperature is not None:
            weather_block = (
                f"\nCurrent Weather in {city}:\n"
                f"  Temperature: {temperature}°C\n"
                f"  Humidity: {humidity}%\n"
            )
            if weather_desc:
                weather_block += f"  Conditions: {weather_desc}\n"

        lang_instruction = LANGUAGE_INSTRUCTIONS.get(
            language, LANGUAGE_INSTRUCTIONS["en-US"]
        )
        prompt = (
            f"{lang_instruction}\n\n"
            f"Location: {city}, Pakistan\n"
            f"Farming Zone: {zone_name}\n\n"
            f"Zone Characteristics:\n"
            f"  Climate: {climate}\n"
            f"  Soil: {soil}\n"
            f"  Typical Crops: {crops}\n"
            f"  Irrigation: {irrigation}\n"
            f"  Key Challenges: {challenges}\n"
            f"{weather_block}\n"
            f"Give farming advice for this area: what crops grow best, "
            f"tell about the soil and weather, and any problems to watch for."
        )
        try:
            chat = self._location_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Location insights generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    def generate_market_advice(
        self,
        commodity: str,
        trend: str,
        latest_price: float,
        pct_change: float,
        best_months: str,
        worst_months: str,
        seasonal_summary: str,
        location_context: str = "",
        unit_label: str = "Rs. / maund (40 kg)",
        language: str = "en-US",
    ) -> str:
        """
        Generate farmer-friendly market selling advice for a commodity.

        Uses a dedicated market-advisory system prompt and does **not**
        carry conversation history — each advice is standalone.

        Parameters
        ----------
        commodity : str
            Display name of the commodity.
        trend : str
            Trend direction (rising/falling/stable).
        latest_price : float
            Latest average price (Rs./maund PKR for weight commodities;
            native unit for count commodities like kinnow per 100 pcs).
        pct_change : float
            Recent 3-month percentage change.
        best_months, worst_months : str
            Comma-separated month names.
        seasonal_summary : str
            Formatted seasonal pattern description.
        location_context : str
            Optional zone/location context for personalised advice.
        unit_label : str
            Price unit label (e.g. "Rs. / maund (40 kg)") — passed
            through to the prompt so Gemini quotes the right unit.

        Returns
        -------
        str
            Concise, farmer-friendly market advice.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        location_block = ""
        if location_context:
            location_block = f"Farmer Location Context:\n{location_context}\n\n"

        lang_instruction = LANGUAGE_INSTRUCTIONS.get(
            language, LANGUAGE_INSTRUCTIONS["en-US"]
        )
        prompt = (
            f"{lang_instruction}\n\n"
            f"Commodity: {commodity}\n"
            f"Current Price Trend: {trend}\n"
            f"Latest Average Price: {latest_price:.0f} {unit_label}\n"
            f"Recent 3-Month Change: {pct_change:+.1f}%\n"
            f"Best Months to Sell (highest prices): {best_months}\n"
            f"Worst Months to Sell (lowest prices): {worst_months}\n"
            f"Seasonal Pattern: {seasonal_summary}\n\n"
            f"{location_block}"
            f"Give practical selling advice: should the farmer sell now "
            f"or wait? What months give the best prices? "
            f"If location is given, make advice fit that area."
        )
        try:
            chat = self._market_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Market advice generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    def generate_best_market_advice(
        self,
        crop: str,
        markets_summary: str,
        home_market: str,
        home_price: float,
        best_market: str,
        best_price: float,
        unit_label: str,
        best_gain: float | None,
        best_gain_pct: float | None,
        location: str,
        language: str = "en-US",
    ) -> str:
        """
        Generate a best-market selling recommendation with a rough
        transport-cost consideration.

        The prompt passes Pakistani district/market names to Gemini and
        asks it to estimate a reasonable intercity transport cost in PKR
        (per maund or per truckload) based on typical Pakistani trucking
        rates for that kind of distance, then judge whether travelling to
        the higher-price market is actually worth it.

        Parameters
        ----------
        crop : str
            Crop variety being sold (e.g. "Paddy Basmati").
        markets_summary : str
            Pre-formatted ranked list of markets with prices in Rs/maund.
        home_market, home_price : str, float
            The farmer's nearest market and its recent average price.
        best_market, best_price : str, float
            The top-ranked (highest price) market and its price.
        unit_label : str
            e.g. "Rs. / maund (40 kg)" — always passed so Gemini never
            invents other currencies or units.
        best_gain, best_gain_pct : float | None
            Price gap vs the home market (None if no home market found).
        location : str
            Farmer's stated city/district (used for distance estimates).
        language : str
            Response language (en-US, ur-PK, ps-AF).

        Returns
        -------
        str
            Farmer-friendly recommendation including a transport-cost
            estimate and a clear "worth it / not worth it" verdict.

        Raises
        ------

        RuntimeError
            If the Gemini API call fails.
        """
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(
            language, LANGUAGE_INSTRUCTIONS["en-US"]
        )

        gain_block = ""
        if best_gain is not None:
            gain_block = (
                f"Price gain at best market vs home market: "
                f"{best_gain:+.0f} {unit_label} ({best_gain_pct:+.1f}%)\n"
            )

        prompt = (
            f"A Pakistani farmer wants to sell: {crop}\n"
            f"Farmer's location: {location or home_market or 'unknown'}\n\n"
            f"Recent mandi prices ({unit_label}):\n{markets_summary}\n\n"
            f"Home market: {home_market or 'not identified'} "
            f"({home_price:.0f} {unit_label} if available)\n"
            f"Highest-price market: {best_market} "
            f"({best_price:.0f} {unit_label})\n"
            f"{gain_block}\n"
            f"TASK — give the farmer a practical recommendation:\n"
            f"1. Name the best mandi to sell at (or say 'sell at your "
            f"local market' if the gain is too small).\n"
            f"2. ESTIMATE the transport cost in PKR: based on the typical "
            f"intercity road distance between {location or 'the farm location'} "
            f"and {best_market} in Pakistan, use typical Pakistani truck / "
            f"mazda pickup rates (roughly Rs. 100–200 per km for a pickup "
            f"load, more for a full truck). Give the estimate as a range, "
            f"per trip AND per maund if the load is around 2–3 tons "
            f"(50–75 maunds).\n"
            f"3. Compare: does the price gain beat the transport cost? "
            f"Give a clear verdict — 'worth travelling' or 'sell locally'.\n"
            f"4. One short practical tip (loading, timing, or bargaining).\n\n"
            f"RULES:\n"
            f"- ALWAYS quote prices in PKR (Rs.) per maund or per bori "
            f"(100 kg bag) — never USD, never per-kg.\n"
            f"- Keep it under 150 words. {lang_instruction}\n"
            f"- Use simple bullet points."
        )
        try:
            chat = self._market_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Best-market advice generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    def generate_irrigation_advice(self, prompt: str, language: str = "en-US") -> str:
        """
        Generate a short irrigation recommendation from live weather data.

        The caller assembles the full prompt (crop, weather, rain forecast,
        soil type, last-irrigation info) and passes it in.  This method
        only handles the Gemini call.

        Uses a dedicated irrigation system prompt and does **not**
        carry conversation history — each recommendation is standalone.

        Parameters
        ----------
        prompt : str
            Pre-built irrigation prompt with real values filled in.

        Returns
        -------
        str
            Concise irrigation advice (under ~60 words).

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        try:
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(
                language, LANGUAGE_INSTRUCTIONS["en-US"]
            )
            full_prompt = f"{lang_instruction}\n\n{prompt}"
            chat = self._irrigation_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(full_prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Irrigation advice generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    def generate_calendar_advice(self, prompt: str, language: str = "en-US") -> str:
        """
        Generate a personalised farming calendar for a crop and region.

        The caller assembles the full prompt (crop, location, sowing date)
        and passes it in.  This method only handles the Gemini call.

        Uses a dedicated calendar system prompt and does **not** carry
        conversation history — each calendar is standalone.

        Parameters
        ----------
        prompt : str
            Pre-built calendar prompt with real values filled in.

        Returns
        -------
        str
            Structured farming calendar text (under ~200 words).

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        try:
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(
                language, LANGUAGE_INSTRUCTIONS["en-US"]
            )
            full_prompt = f"{lang_instruction}\n\n{prompt}"
            chat = self._calendar_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(full_prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Calendar generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    # ── Crop Recommendation Advisory ──────────────────────────────

    def generate_crop_recommendation_explanation(
        self,
        recommended_crop: str,
        confidence: float,
        top_3_alternatives: list[dict],
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        language: str = "en-US",
    ) -> str:
        """
        Generate a farmer-friendly explanation for an ML-predicted crop.

        Unlike ``generate_response``, this uses a dedicated crop-recommendation
        system prompt and does **not** carry conversation history — each
        recommendation is a standalone assessment.

        Parameters
        ----------
        recommended_crop : str
            The ML-predicted crop name (e.g. 'rice', 'wheat', 'cotton').
        confidence : float
            Model confidence for the top crop (0.0 – 1.0).
        top_3_alternatives : list[dict]
            Top 3 crops with confidence scores for context.
        nitrogen, phosphorus, potassium : float
            Soil NPK levels.
        temperature, humidity, ph, rainfall : float
            Climate and soil conditions.

        Returns
        -------
        str
            Concise, farmer-friendly explanation.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        # Build a readable alternatives line
        alt_lines = ", ".join(
            f"{a['crop']} ({a['confidence']:.0%})" for a in top_3_alternatives
        )

        lang_instruction = LANGUAGE_INSTRUCTIONS.get(
            language, LANGUAGE_INSTRUCTIONS["en-US"]
        )
        prompt = (
            f"{lang_instruction}\n\n"
            f"Suggested crop: {recommended_crop} "
            f"(confidence: {confidence:.0%})\n\n"
            f"Top 3 alternatives: {alt_lines}\n\n"
            f"Soil & Climate Conditions:\n"
            f"  Nitrogen (N): {nitrogen}\n"
            f"  Phosphorus (P): {phosphorus}\n"
            f"  Potassium (K): {potassium}\n"
            f"  Temperature: {temperature}°C\n"
            f"  Humidity: {humidity}%\n"
            f"  Soil pH: {ph}\n"
            f"  Rainfall: {rainfall}mm\n\n"
            f"Explain why {recommended_crop} is good for these "
            f"conditions, how to grow it, and any warnings."
        )
        try:
            chat = self._crop_recommendation_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Crop recommendation explanation generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    # ── Grafting Advisor ──────────────────────────────────────────

    def generate_grafting_advice(self, prompt: str, language: str = "en") -> str:
        """
        Generate farmer-friendly grafting advice for a given crop/tree.

        The caller assembles the full prompt (base_crop, goal, language)
        and passes it in.  This method only handles the Gemini call.

        Uses a dedicated grafting system prompt and does **not** carry
        conversation history — each advice is standalone.

        Parameters
        ----------
        prompt : str
            Pre-built grafting prompt with real values filled in.

        Returns
        -------
        str
            Concise, farmer-friendly grafting advice (under ~200 words).

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        try:
            # Map short codes to full locale codes for LANGUAGE_INSTRUCTIONS
            lang_map = {"en": "en-US", "ur": "ur-PK", "ps": "ps-AF"}
            full_locale = lang_map.get(language, "en-US")
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(
                full_locale, LANGUAGE_INSTRUCTIONS["en-US"]
            )
            full_prompt = f"{lang_instruction}\n\n{prompt}"
            chat = self._grafting_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(full_prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Grafting advice generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    def generate_gov_support_advice(self, prompt: str, language: str = "en-US") -> str:
        """
        Answer a farmer's question about government agriculture support.

        The caller assembles the full prompt (query, province) and passes
        it in.  This method only handles the Gemini call.

        Uses a dedicated gov-support system prompt and does **not** carry
        conversation history — each answer is standalone.

        Parameters
        ----------
        prompt : str
            Pre-built prompt with the farmer's question and province.

        Returns
        -------
        str
            Government support information (under ~150 words).

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        try:
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(
                language, LANGUAGE_INSTRUCTIONS["en-US"]
            )
            full_prompt = f"{lang_instruction}\n\n{prompt}"
            chat = self._gov_support_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(full_prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Government support advice generation failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

    # ── Harvest-loss prediction ───────────────────────────────────────

    def generate_harvest_loss_advice(self, prompt: str, language: str = "en-US") -> str:
        """
        Estimate yield/financial loss for different harvest-delay scenarios.

        The caller assembles the full prompt (crop, weather, market trend)
        and passes it in.  This method only handles the Gemini call.

        Parameters
        ----------
        prompt : str
            Pre-built prompt with crop, location, weather, and market data.
        language : str
            Full locale code ("en-US", "ur-PK", "ps-AF") or short code.

        Returns
        -------
        str
            Harvest-loss advice (under ~150 words) with scenario estimates.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        try:
            # Map short codes to full locale codes for LANGUAGE_INSTRUCTIONS
            lang_map = {"en": "en-US", "ur": "ur-PK", "ps": "ps-AF"}
            full_locale = lang_map.get(language, language)
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(
                full_locale, LANGUAGE_INSTRUCTIONS["en-US"]
            )
            full_prompt = f"{lang_instruction}\n\n{prompt}"
            chat = self._harvest_loss_model.start_chat(history=[])
            response = _call_gemini_api(lambda: chat.send_message(full_prompt), "Gemini")
            return response.text.strip()

        except RuntimeError:
            raise  # preserve rate-limit messages from _call_gemini_api
        except Exception as exc:
            raise RuntimeError(
                f"Harvest loss prediction failed. "
                f"Please try again in a moment. (details: {exc})"
            ) from exc

