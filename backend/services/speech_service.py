"""
Speech service — Speech-to-Text (STT) and Text-to-Speech (TTS).

Libraries used:
    • SpeechRecognition  — wraps Google Web Speech API (free, no key needed)
    • edge-tts           — Microsoft Edge online TTS (free, no key needed,
                           neural voices for English, Urdu, and Pashto)
    • gTTS               — Google Text-to-Speech fallback (free, no key,
                           supports English and Urdu; no native Pashto,
                           so Pashto falls back to Urdu voice)
    • pydub              — audio format conversion (WebM/OGG → WAV)
    • imageio-ffmpeg     — bundled ffmpeg binary for pydub

Both libraries support Urdu (ur) and English (en).
edge-tts also supports Pashto (ps) via neural voices.
gTTS does NOT support Pashto — we fall back to Urdu voice for Pashto text.
"""

import asyncio
import io
import logging
import re
from typing import Optional

import edge_tts
import speech_recognition as sr
from gtts import gTTS
from gtts.lang import tts_langs

logger = logging.getLogger(__name__)

# ── Audio format conversion setup ────────────────────────────────────
# pydub + imageio-ffmpeg convert browser audio (WebM/OGG) to WAV
# so that SpeechRecognition can process it.
_HAS_PYDUB = False
try:
    from pydub import AudioSegment

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        AudioSegment.converter = get_ffmpeg_exe()
        AudioSegment.ffprobe = get_ffmpeg_exe()
    except ImportError:
        pass  # hope ffmpeg is on PATH

    _HAS_PYDUB = True
    logger.info("pydub loaded — audio format conversion enabled")
except ImportError:
    logger.warning(
        "pydub not installed — only WAV audio will work for STT. "
        "Install pydub + imageio-ffmpeg for full browser audio support."
    )

# ── Language helpers ─────────────────────────────────────────────────

# Maps full locale codes to short codes used by the STT library
_LOCALE_TO_SHORT = {
    "en-US": "en-US",
    "en-GB": "en-GB",
    "en-IN": "en-IN",
    "ur-PK": "ur-PK",
    "ps-AF": "ps-AF",
}

# Preferred neural voices per locale (edge-tts / Microsoft Azure quality).
# Female voices chosen for clarity in noisy outdoor environments.
_EDGE_VOICES = {
    "en-US": "en-US-JennyNeural",
    "en-GB": "en-GB-SoniaNeural",
    "en-IN": "en-IN-NeerjaNeural",
    "ur-PK": "ur-PK-UzmaNeural",
    "ps-AF": "ps-AF-LatifaNeural",
}

# Maps locale codes to ISO 639-1 short codes used by gTTS.
# NOTE: gTTS does NOT have a Pashto voice, so Pashto falls back
# to Urdu ("ur") — this is the closest available option and
# produces a reasonably understandable result for Pashto speakers
# familiar with Urdu.
_GTTS_LANG = {
    "en-US": "en",
    "en-GB": "en",
    "en-IN": "en",
    "ur-PK": "ur",
    "ps-AF": "ur",  # ← Pashto fallback: gTTS has no Pashto voice
}


# ── Text cleanup helper ──────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS speaks clean text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)       # **bold**
    text = re.sub(r"\*(.+?)\*", r"\1", text)             # *italic*
    text = re.sub(r"__(.+?)__", r"\1", text)             # __bold__
    text = re.sub(r"_(.+?)_", r"\1", text)               # _italic_
    text = re.sub(r"~~(.+?)~~", r"\1", text)             # ~~strike~~
    text = re.sub(r"`(.+?)`", r"\1", text)               # `code`
    text = re.sub(r"```[\s\S]*?```", "", text)           # code blocks
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # headers
    text = re.sub(r"^[\s]*[-*+]\s+", "• ", text, flags=re.M)  # bullets
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)     # links
    text = re.sub(r"^\s*>\s*", "", text, flags=re.M)     # blockquotes
    text = re.sub(r"\n{3,}", "\n\n", text)               # extra blank lines
    return text.strip()


class SpeechService:
    """Handles all speech conversion — audio → text and text → audio."""

    def __init__(self) -> None:
        self._recognizer = sr.Recognizer()
        # Tune for real-world noisy environments (farm, outdoors)
        self._recognizer.energy_threshold = 300
        self._recognizer.dynamic_energy_threshold = True

    # ── Audio normalisation ──────────────────────────────────────────

    def _normalise_audio(self, audio_bytes: bytes) -> bytes:
        """
        Convert audio bytes to WAV format.

        Streamlit's st.audio_input uses the browser MediaRecorder API,
        which typically produces WebM/Opus format (Chrome, Edge, Firefox)
        — NOT WAV.  SpeechRecognition only reads WAV, AIFF, or FLAC.

        This method uses pydub (+ ffmpeg via imageio-ffmpeg) to convert
        any supported audio format into WAV.  If pydub is unavailable,
        the raw bytes are returned as-is (will only work if the browser
        happened to produce WAV).
        """
        if not _HAS_PYDUB:
            return audio_bytes

        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_buf = io.BytesIO()
            audio_seg.export(wav_buf, format="wav")
            wav_bytes = wav_buf.getvalue()
            logger.info(
                "Audio normalised: %d bytes → %d bytes (WAV)",
                len(audio_bytes),
                len(wav_bytes),
            )
            return wav_bytes
        except Exception as exc:
            logger.warning(
                "pydub conversion failed (%s), trying raw bytes", exc
            )
            return audio_bytes

    # ── Internal sync STT (runs in thread pool) ─────────────────────

    def _recognize_sync(self, wav_bytes: bytes, locale: str) -> str:
        """
        Synchronous speech-to-text using Google Web Speech API.

        This is called inside ``asyncio.to_thread()`` so it does not
        block the FastAPI event loop.
        """
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = self._recognizer.record(source)

        text: str = self._recognizer.recognize_google(
            audio_data, language=locale
        )
        return text.strip()

    # ── Speech-to-Text (async) ──────────────────────────────────────

    async def speech_to_text(
        self,
        audio_bytes: bytes,
        language: str = "en-US",
    ) -> str:
        """
        Transcribe raw audio bytes into text.

        The audio is first normalised to WAV (converting WebM/OGG from
        the browser), then sent to Google Web Speech API for recognition.

        For Pashto: tries ``ps-AF`` first; if that fails, retries with
        ``ur`` (Urdu) since many Pashto speakers are bilingual.

        Returns the transcribed text, or a friendly error message.
        """
        try:
            # Step 1: Convert browser audio to WAV
            wav_bytes = self._normalise_audio(audio_bytes)
            logger.info(
                "STT: language=%s, input=%d bytes, wav=%d bytes",
                language,
                len(audio_bytes),
                len(wav_bytes),
            )

            # Step 2: Resolve locale code
            locale = _LOCALE_TO_SHORT.get(language, "en-US")

            # Step 3: Recognise speech (non-blocking)
            if locale == "ps-AF":
                # Pashto: try native first, then Urdu fallback
                try:
                    return await asyncio.to_thread(
                        self._recognize_sync, wav_bytes, "ps-AF"
                    )
                except (sr.UnknownValueError, sr.RequestError):
                    logger.info(
                        "Pashto STT failed, retrying with Urdu (ur)..."
                    )
                    try:
                        return await asyncio.to_thread(
                            self._recognize_sync, wav_bytes, "ur"
                        )
                    except sr.UnknownValueError:
                        return (
                            "Sorry, I couldn't understand that. "
                            "Please speak more clearly or type your question."
                        )
                    except sr.RequestError:
                        return (
                            "The speech recognition service is currently unavailable. "
                            "Please type your question instead."
                        )
            else:
                return await asyncio.to_thread(
                    self._recognize_sync, wav_bytes, locale
                )

        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return (
                "Sorry, I couldn't understand that. "
                "Please try speaking more clearly or type your question."
            )

        except sr.RequestError as exc:
            logger.error("Speech recognition service unavailable: %s", exc)
            return (
                "The speech recognition service is currently unavailable. "
                "Please type your question instead."
            )

        except Exception as exc:
            logger.error("Unexpected speech-to-text error: %s", exc)
            return (
                "Sorry, speech recognition failed. "
                "Please type your question instead."
            )

    # ── Text-to-Speech ──────────────────────────────────────────────

    async def text_to_speech(
        self,
        text: str,
        language: str = "en-US",
    ) -> Optional[bytes]:
        """
        Synthesise speech audio (MP3) from plain text.

        Strategy (two free TTS backends, no API key required):
          1. **edge-tts** — Microsoft neural voices (best quality,
             supports English, Urdu, AND Pashto).  Retries up to 2×
             on transient network failures.
          2. **gTTS** — Google TTS fallback (used if edge-tts fails;
             supports English and Urdu; Pashto falls back to Urdu).

        Markdown formatting is stripped before synthesis so the voice
        reads clean text without asterisks or hash marks.
        """
        # Clean the text for natural speech output
        clean_text = _strip_markdown(text)
        if not clean_text:
            return None

        # Truncate very long responses to keep audio short & fast
        if len(clean_text) > 600:
            clean_text = clean_text[:600].rsplit(" ", 1)[0] + "..."

        # Try edge-tts first (best quality, supports Pashto, with retries)
        audio = await self._tts_edge(clean_text, language)
        if audio:
            return audio

        # Fall back to gTTS if edge-tts failed
        logger.info("edge-tts failed after retries, trying gTTS fallback")
        audio = await self._tts_gtts(clean_text, language)
        if audio:
            return audio

        logger.error("All TTS methods failed for language=%s", language)
        return None

    # ── edge-tts (primary, with retry) ──────────────────────────────

    async def _tts_edge(
        self,
        text: str,
        language: str,
        max_retries: int = 2,
    ) -> Optional[bytes]:
        """
        Generate MP3 audio using Microsoft Edge TTS (neural voices).

        Retries up to *max_retries* times on transient failures
        (network timeouts, service hiccups) with short delays.
        Returns None on persistent failure so the caller can try gTTS.
        """
        voice = _EDGE_VOICES.get(language, "en-US-JennyNeural")

        for attempt in range(1, max_retries + 1):
            try:
                communicate = edge_tts.Communicate(text=text, voice=voice)

                # Collect audio chunks from the async stream
                audio_chunks: list[bytes] = []
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio" and "data" in chunk:
                        audio_chunks.append(chunk["data"])

                if not audio_chunks:
                    logger.warning(
                        "edge-tts attempt %d: stream returned no audio data",
                        attempt,
                    )
                else:
                    result = b"".join(audio_chunks)
                    logger.info(
                        "edge-tts OK: %d bytes, voice=%s (attempt %d)",
                        len(result),
                        voice,
                        attempt,
                    )
                    return result

            except Exception as exc:
                logger.warning(
                    "edge-tts attempt %d failed: %s", attempt, exc
                )

            # Brief pause before retry (avoids hammering the service)
            if attempt < max_retries:
                await asyncio.sleep(0.5)

        return None

    # ── gTTS (fallback, async wrapper) ──────────────────────────────

    async def _tts_gtts(
        self,
        text: str,
        language: str,
    ) -> Optional[bytes]:
        """
        Generate MP3 audio using Google Text-to-Speech.
        Used as fallback when edge-tts is unavailable.

        Runs in a thread pool so it doesn't block the event loop.

        Limitation: gTTS does NOT support Pashto ("ps").
        For Pashto text, we use Urdu ("ur") as the closest
        available language.
        """
        try:
            result = await asyncio.to_thread(self._tts_gtts_sync, text, language)
            return result
        except Exception as exc:
            logger.error("gTTS async wrapper failed: %s", exc)
            return None

    def _tts_gtts_sync(
        self,
        text: str,
        language: str,
    ) -> Optional[bytes]:
        """Synchronous gTTS generation (called via asyncio.to_thread)."""
        try:
            lang_code = _GTTS_LANG.get(language, "en")

            # Safety check: verify gTTS actually supports this language
            supported = tts_langs()
            if lang_code not in supported:
                logger.warning(
                    "gTTS does not support '%s', falling back to English",
                    lang_code,
                )
                lang_code = "en"

            tts = gTTS(text=text, lang=lang_code)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            result = buf.getvalue()

            if not result:
                logger.warning("gTTS returned empty audio")
                return None

            logger.info(
                "gTTS OK: %d bytes, lang=%s", len(result), lang_code
            )
            return result

        except Exception as exc:
            logger.error("gTTS synthesis also failed: %s", exc)
            return None
