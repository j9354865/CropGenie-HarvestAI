"""
Plant disease detection route — /api/detect-disease

Accepts an uploaded plant image (JPEG/PNG, max 5 MB) and a language code,
sends the image to Gemini's multimodal vision API for analysis, and returns
a structured disease detection result.
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.models.schemas import DiseaseDetectionResponse
from backend.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed MIME types and the maximum file size (bytes)
_ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Lazy-initialised service
_gemini_service: GeminiService | None = None


def _get_service() -> GeminiService:
    """Lazy-init the Gemini service (shared with other routes)."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


@router.post("/detect-disease", response_model=DiseaseDetectionResponse)
async def detect_disease(
    image: UploadFile = File(
        ...,
        description="Plant/leaf/fruit photo (JPEG or PNG, max 5 MB)",
    ),
    language: str = Form(
        default="en-US",
        description="Language code for the analysis response (en-US, ur-PK, ps-AF)",
    ),
):
    """
    Analyse a plant image for disease detection using Gemini vision.

    The frontend sends a single image file.  The route validates the
    file type and size, then delegates to GeminiService.analyze_disease()
    which returns a typed DiseaseDetectionResponse.
    """
    # ── Validate MIME type ──────────────────────────────────────────
    mime = (image.content_type or "").lower()
    if mime not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format '{mime}'. "
                "Please upload a JPEG, PNG, or WebP image."
            ),
        )

    # ── Read and validate size ──────────────────────────────────────
    try:
        image_bytes = await image.read()
    except Exception as exc:
        logger.error("Failed to read uploaded image: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image. Please try again.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty. Please select a valid photo.",
        )

    if len(image_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image is too large ({len(image_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum size is {_MAX_FILE_SIZE // 1024 // 1024} MB. "
                "Please use a smaller or compressed image."
            ),
        )

    # ── Run Gemini analysis ─────────────────────────────────────────
    try:
        service = _get_service()
        result = service.analyze_disease(
            image_bytes=image_bytes,
            mime_type=mime,
            language=language,
        )
        return result

    except RuntimeError as exc:
        # Gemini API failure — 503 for service unavailable
        logger.error("Disease detection Gemini error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error("Unexpected disease detection error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred during disease analysis. "
                "Please try again with a different photo."
            ),
        )
