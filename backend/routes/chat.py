"""
Chat route — /api/chat

Accepts a farmer's message + conversation history, forwards it to
the Gemini service, and returns the AI's reply.
"""

import asyncio
import hashlib
import logging
import time

from fastapi import APIRouter, HTTPException

from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.gemini_service import GeminiService, _get_counter_snapshot

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate once at module level (reused across requests)
_gemini_service: GeminiService | None = None


def _get_service() -> GeminiService:
    """Lazy-init the Gemini service so missing API keys surface clearly."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


# ── Request deduplication guard ─────────────────────────────────────────
# Prevents the same message from being processed twice (e.g. Streamlit
# rerun causing a second call for the same user input).
_last_request_fingerprint: str | None = None
_last_request_time: float = 0.0
_DEDUP_WINDOW_SECONDS = 3.0  # Ignore duplicate requests within 3 seconds


def _is_duplicate_request(message: str, history_len: int) -> bool:
    """Return True if this looks like a duplicate of the previous request."""
    global _last_request_fingerprint, _last_request_time

    fingerprint = hashlib.md5(f"{message}:{history_len}".encode()).hexdigest()
    now = time.monotonic()

    if (
        fingerprint == _last_request_fingerprint
        and (now - _last_request_time) < _DEDUP_WINDOW_SECONDS
    ):
        return True

    _last_request_fingerprint = fingerprint
    _last_request_time = now
    return False


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message through the Gemini agriculture assistant.

    Expects the full conversation history so the AI can maintain context.

    The Gemini API call (including any rate-limit retry waits) runs in a
    thread pool so it does NOT block the FastAPI event loop — other
    endpoints (health, TTS, STT) stay responsive.
    """
    try:
        service = _get_service()
        history_dicts = [msg.model_dump() for msg in request.history]

        # ── Deduplication: reject duplicate requests within 3 seconds ──
        if _is_duplicate_request(request.message, len(history_dicts)):
            logger.warning(
                "[Chat] Duplicate request detected (same message within %ds). Ignoring.",
                _DEDUP_WINDOW_SECONDS,
            )
            raise HTTPException(
                status_code=429,
                detail="Duplicate request. Please wait a moment before sending again.",
            )

        logger.info(
            "[Chat] User request received: lang=%s, history_len=%d, msg=%.80s | counters=%s",
            request.language,
            len(history_dicts),
            request.message,
            _get_counter_snapshot(),
        )

        # Run the blocking Gemini call (with retry sleeps) in a thread
        reply = await asyncio.to_thread(
            service.generate_response,
            request.message,
            history_dicts,
            request.language,
        )

        logger.info(
            "[Chat] Response OK: %d chars | counters=%s",
            len(reply),
            _get_counter_snapshot(),
        )
        return ChatResponse(response=reply)

    except HTTPException:
        raise  # re-raise our own HTTP exceptions unchanged

    except RuntimeError as exc:
        error_msg = str(exc)

        # ── Rate-limit with explicit wait time ─────────────────────
        # _call_gemini_api raises RuntimeError with:
        #   "AI is rate-limited. Please wait about Xs and try again."
        if "rate-limited" in error_msg.lower() or "rate limit" in error_msg.lower():
            logger.warning(
                "[Chat] Gemini rate-limited: %s | counters=%s",
                error_msg[:200],
                _get_counter_snapshot(),
            )
            raise HTTPException(
                status_code=503,
                detail=error_msg,  # pass the exact wait time to the user
            )

        # ── Quota / 429 / resource exhausted ────────────────────────
        if any(kw in error_msg.lower() for kw in ("429", "quota", "resource")):
            logger.warning(
                "[Chat] Gemini quota/rate-limit: %s | counters=%s",
                error_msg[:200],
                _get_counter_snapshot(),
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "AI is busy right now (rate limit). "
                    "Please wait a few seconds and try again."
                ),
            )

        # ── Other Gemini failures ───────────────────────────────────
        logger.error(
            "[Chat] Gemini API error: %s | counters=%s",
            error_msg[:200],
            _get_counter_snapshot(),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "AI service temporarily unavailable. "
                "Please try again in a moment."
            ),
        )

    except Exception as exc:
        logger.error("[Chat] Unexpected error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
