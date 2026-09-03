"""
WhatsApp Bot webhook — POST /api/whatsapp/webhook
==================================================

Receives incoming WhatsApp messages via Twilio's webhook and routes
them to the appropriate CropGenie AI service:

  • **Text message**  → Chat service (Gemini conversation)
  • **Photo / image** → Disease Detection (Gemini vision)
  • **Voice note**    → Speech-to-Text → Chat service

Replies are sent back as WhatsApp text messages via TwiML (Twilio's
XML response format).  No Twilio SDK is needed for *receiving*
messages — Twilio simply POSTs form-encoded data to our endpoint.
The ``twilio`` package is used only to build TwiML responses cleanly.

Setup
-----
1. Add your Twilio credentials to ``.env`` (see ``.env.example``).
2. Expose your local backend via ngrok (see README).
3. Paste the ngrok URL + ``/api/whatsapp/webhook`` into the Twilio
   WhatsApp Sandbox webhook settings.

Flow
----
1. Twilio POSTs form-encoded data with fields like ``Body``, ``From``,
   ``NumMedia``, ``MediaUrl0``, ``MediaContentType0``.
2. We determine the message type and call the matching service.
3. We return a ``MessagingResponse`` (TwiML XML) that Twilio delivers
   back to the farmer's WhatsApp.
"""

import logging

import requests as http_requests
from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse

from backend.config import settings
from backend.services.gemini_service import GeminiService
from backend.services.speech_service import SpeechService

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy-initialised services ────────────────────────────────────────
_gemini_service: GeminiService | None = None
_speech_service: SpeechService | None = None


def _get_gemini_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


def _get_speech_service() -> SpeechService:
    """Lazy-init the speech service."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service


# ── Helpers ───────────────────────────────────────────────────────────

# Maximum WhatsApp reply length — longer replies get truncated with a note.
_MAX_REPLY_LENGTH = 1500


def _truncate(text: str, limit: int = _MAX_REPLY_LENGTH) -> str:
    """Trim a reply to fit within WhatsApp's practical message limit."""
    if len(text) <= limit:
        return text
    return text[:limit - 30] + "\n\n…(reply shortened)"


def _download_twilio_media(media_url: str) -> bytes | None:
    """
    Download a media file (image or audio) from Twilio's URL.

    Twilio media URLs require HTTP Basic authentication using the
    Account SID and Auth Token.  Returns raw bytes or None on failure.
    """
    try:
        resp = http_requests.get(
            media_url,
            auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.error("Failed to download Twilio media from %s: %s", media_url, exc)
        return None


def _detect_language(text: str) -> str:
    """
    Rough language detection from the message text.

    Returns a language code compatible with GeminiService and
    SpeechService (en-US, ur-PK, ps-AF).  Falls back to English.
    """
    # Check for common Urdu characters (Arabic script range)
    urdu_chars = set("اردوہےکایہھیںکیوںکہپرنسےمتعلقرہی")
    # Check for common Pashto-specific characters
    pashto_chars = set("ښګڼځړڅچډ")

    text_chars = set(text)

    if text_chars & pashto_chars:
        return "ps-AF"
    if text_chars & urdu_chars:
        return "ur-PK"

    # Simple keyword-based fallback for Roman Urdu
    roman_urdu = ["kya", "kaise", "kahan", "hai", "karo", "bhai", "mera"]
    lower = text.lower()
    if any(word in lower for word in roman_urdu):
        return "ur-PK"

    return "en-US"


# ── Error fallback messages ──────────────────────────────────────────

_ERROR_MESSAGES = {
    "en-US": "Sorry, I couldn't process that right now. Please try again in a moment. 🌾",
    "ur-PK": "معذرت، ابھی جواب نہیں دے سکا۔ براہ کرم دوبارہ کوشش کریں۔ 🌾",
    "ps-AF": "بخښنه غواړم، اوس ځواب ونه شو کولای. مهرباني وکړئ بیا هڅه وکړئ۔ 🌾",
}


# ── Main webhook endpoint ────────────────────────────────────────────

@router.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    """
    Handle an incoming WhatsApp message from Twilio.

    Twilio POSTs form-encoded data.  Key fields:
      - ``Body``: text content of the message
      - ``From``: sender's WhatsApp number (``whatsapp:+92...``)
      - ``NumMedia``: number of media attachments (``"0"``, ``"1"``, etc.)
      - ``MediaUrl0``: URL of the first media file (if any)
      - ``MediaContentType0``: MIME type of the first media (e.g. ``image/jpeg``)

    Returns a TwiML ``<Response>`` with one or more ``<Message>`` elements.
    """
    # ── Parse incoming form data ─────────────────────────────────
    form = await request.form()

    body = (form.get("Body") or "").strip()
    sender = form.get("From") or "unknown"
    num_media = int(form.get("NumMedia") or 0)
    media_url = form.get("MediaUrl0") or ""
    media_type = form.get("MediaContentType0") or ""

    logger.info(
        "WhatsApp message from %s — text=%r, media=%d (%s)",
        sender, body[:80] if body else "", num_media, media_type,
    )

    # Build the TwiML response — this is what Twilio sends back to
    # the farmer on WhatsApp.
    twiml = MessagingResponse()

    try:
        # ── Detect language from text (best-effort) ──────────────
        language = _detect_language(body) if body else "en-US"

        # ── Route by message type ────────────────────────────────

        if num_media > 0 and media_url:
            # Has media — check if image or audio
            reply = await _handle_media(
                media_url=media_url,
                media_type=media_type,
                body_text=body,
                language=language,
            )
        elif body:
            # Plain text message — route to chat
            reply = _handle_text(body, language)
        else:
            # No text and no media — ask the farmer to send something
            reply = (
                "🌾 Welcome to CropGenie!\n\n"
                "Send me:\n"
                "• A question about crops, weather, or farming\n"
                "• A photo of a plant for disease detection\n"
                "• A voice note with your question\n\n"
                "I'll help you right away!"
            )

        twiml.message(_truncate(reply))

    except Exception as exc:
        # Catch-all — never crash the webhook, always return valid TwiML
        logger.error("WhatsApp webhook error: %s", exc, exc_info=True)
        lang = _detect_language(body) if body else "en-US"
        twiml.message(_ERROR_MESSAGES.get(lang, _ERROR_MESSAGES["en-US"]))

    # Return TwiML XML as the HTTP response
    return Response(
        content=str(twiml),
        media_type="text/xml",
    )


# ── Message handlers ─────────────────────────────────────────────────

def _handle_text(body: str, language: str) -> str:
    """
    Handle a plain text message — route to the Gemini chat service.

    WhatsApp conversations are stateless (no history tracking in webhook
    mode), so each message is treated independently.
    """
    try:
        gemini_svc = _get_gemini_service()
        # Empty history — each WhatsApp message is standalone
        reply = gemini_svc.generate_response(body, history=[])
        return reply
    except Exception as exc:
        logger.error("Chat service error (WhatsApp): %s", exc)
        return _ERROR_MESSAGES.get(language, _ERROR_MESSAGES["en-US"])


async def _handle_media(
    media_url: str,
    media_type: str,
    body_text: str,
    language: str,
) -> str:
    """
    Handle a message with media (image or voice note).

    Images are routed to disease detection.
    Audio is transcribed first, then routed to chat.
    """
    # ── Image ─────────────────────────────────────────────────────
    if media_type.startswith("image/"):
        return _handle_image(media_url, media_type, body_text, language)

    # ── Audio / Voice note ────────────────────────────────────────
    if media_type.startswith("audio/"):
        return _handle_audio(media_url, language)

    # ── Unsupported media type ────────────────────────────────────
    logger.warning("Unsupported media type: %s", media_type)
    return (
        "I received a file, but I can only process:\n"
        "• **Photos** of plants (for disease detection)\n"
        "• **Voice notes** (I'll transcribe and answer)\n\n"
        "Please try sending one of those instead."
    )


def _handle_image(
    media_url: str,
    media_type: str,
    body_text: str,
    language: str,
) -> str:
    """
    Handle a plant photo — download and pass to disease detection.
    """
    image_bytes = _download_twilio_media(media_url)
    if image_bytes is None:
        return _ERROR_MESSAGES.get(language, _ERROR_MESSAGES["en-US"])

    try:
        gemini_svc = _get_gemini_service()
        result = gemini_svc.analyze_disease(
            image_bytes=image_bytes,
            mime_type=media_type,
            language=language,
        )

        # Format the structured disease result into a WhatsApp-friendly reply
        lines = [
            f"🌿 *Disease Analysis Result*",
            f"",
            f"*Plant:* {result.plant_name}",
            f"*Status:* {result.health_status}",
        ]

        if result.disease_name:
            lines.append(f"*Issue:* {result.disease_name}")

        if result.cause:
            lines.append(f"*Cause:* {result.cause}")

        if result.summary:
            lines.append(f"\n{result.summary}")

        if result.treatment:
            lines.append(f"\n*Treatment:*")
            for i, step in enumerate(result.treatment[:3], 1):
                lines.append(f"{i}. {step}")

        if result.prevention:
            lines.append(f"\n*Prevention:*")
            for i, tip in enumerate(result.prevention[:3], 1):
                lines.append(f"{i}. {tip}")

        return "\n".join(lines)

    except Exception as exc:
        logger.error("Disease detection error (WhatsApp): %s", exc)
        return _ERROR_MESSAGES.get(language, _ERROR_MESSAGES["en-US"])


def _handle_audio(media_url: str, language: str) -> str:
    """
    Handle a voice note — download, transcribe, then route to chat.

    Twilio WhatsApp voice notes are typically sent as .ogg files.
    The SpeechService expects WAV format, so we attempt to transcribe
    and fall back gracefully if the format isn't supported.
    """
    audio_bytes = _download_twilio_media(media_url)
    if audio_bytes is None:
        return _ERROR_MESSAGES.get(language, _ERROR_MESSAGES["en-US"])

    try:
        speech_svc = _get_speech_service()
        # Attempt transcription — SpeechService returns a friendly
        # error string if it can't understand the audio format.
        transcribed_text = speech_svc.speech_to_text(audio_bytes, language)

        # Check if transcription returned an error message
        error_indicators = [
            "couldn't understand",
            "could not understand",
            "unavailable",
            "failed",
            "Speech recognition",
        ]
        if any(ind in transcribed_text for ind in error_indicators):
            return (
                "🎤 I received your voice note but couldn't transcribe it.\n\n"
                "WhatsApp voice notes use a format (.ogg) that's tricky "
                "to process. Could you please **type your question** "
                "instead? I'll answer right away! 🌾"
            )

        logger.info("Voice note transcribed: %r", transcribed_text[:100])

        # Route the transcribed text to chat
        reply = _handle_text(transcribed_text, language)
        return f"🎤 _You said:_ \"{transcribed_text}\"\n\n{reply}"

    except Exception as exc:
        logger.error("Audio handling error (WhatsApp): %s", exc)
        return (
            "🎤 I received your voice note but had trouble processing it. "
            "Could you please **type your question** instead? 🌾"
        )
