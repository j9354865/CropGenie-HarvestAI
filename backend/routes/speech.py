"""
Speech routes — /api/speech-to-text and /api/text-to-speech

• STT: accepts uploaded audio + language → returns transcribed text
  (audio is auto-converted from WebM/OGG to WAV inside the service)
• TTS: accepts text + language → returns MP3 audio bytes
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException
from fastapi.responses import Response

from backend.models.schemas import TTSRequest
from backend.services.speech_service import SpeechService

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate once at module level
_speech_service = SpeechService()


# ── Speech-to-Text ──────────────────────────────────────────────────

@router.post("/speech-to-text")
async def speech_to_text(
    audio: bytes = File(..., description="Audio from the microphone (WebM, OGG, or WAV)"),
    language: str = Form(
        default="en-US",
        description="Language locale code (en-US, ur-PK, ps-AF)",
    ),
):
    """
    Transcribe uploaded audio into text.

    The frontend sends raw audio bytes captured via the browser's
    MediaRecorder API (Streamlit's st.audio_input).  The speech
    service auto-converts WebM/OGG → WAV before recognition.

    STT runs in a thread pool so it does not block the event loop.
    """
    try:
        text = await _speech_service.speech_to_text(audio, language)
        return {"text": text}

    except Exception as exc:
        logger.error("STT route error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Speech recognition failed. Please try again or type your question.",
        )


# ── Text-to-Speech ──────────────────────────────────────────────────

@router.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """
    Synthesise speech audio from text.

    Returns an MP3 audio stream that the frontend can play directly.
    edge-tts retries up to 2× on transient failures before falling
    back to gTTS.
    """
    try:
        audio_bytes = await _speech_service.text_to_speech(
            request.text,
            request.language,
        )

        if audio_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="Text-to-speech synthesis failed. Please try again.",
            )

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
        )

    except HTTPException:
        raise  # re-raise HTTP exceptions unchanged

    except Exception as exc:
        logger.error("TTS route error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Text-to-speech failed. You can still read the text response.",
        )
