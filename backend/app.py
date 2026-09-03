"""
CropGenie HarvestAI — FastAPI application entry point.

Run with:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.chat import router as chat_router
from backend.routes.speech import router as speech_router
from backend.routes.weather import router as weather_router
from backend.routes.predict import router as predict_router
from backend.routes.disease import router as disease_router
from backend.routes.fertilizer import router as fertilizer_router
from backend.routes.location import router as location_router
from backend.routes.market import router as market_router
from backend.routes.best_market import router as best_market_router
from backend.routes.irrigation import router as irrigation_router
from backend.routes.calendar import router as calendar_router
from backend.routes.gov_support import router as gov_support_router
from backend.routes.crop_recommend import router as crop_recommend_router
from backend.routes.grafting import router as grafting_router
from backend.routes.whatsapp import router as whatsapp_router
from backend.routes.harvest_loss import router as harvest_loss_router
from backend.services.gemini_service import _get_counter_snapshot

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Application ─────────────────────────────────────────────────────
app = FastAPI(
    title="CropGenie HarvestAI",
    description=(
        "AI Agriculture Assistant — helps farmers with crop advice, "
        "disease identification, weather guidance, and more."
    ),
    version="1.0.0",
)

# ── CORS ────────────────────────────────────────────────────────────
# Allow the Streamlit frontend (and any local dev client) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(speech_router, prefix="/api", tags=["Speech"])
app.include_router(weather_router, prefix="/api", tags=["Weather"])
app.include_router(predict_router, prefix="/api", tags=["Profit Prediction"])
app.include_router(disease_router, prefix="/api", tags=["Disease Detection"])
app.include_router(fertilizer_router, prefix="/api", tags=["Fertilizer"])
app.include_router(location_router, prefix="/api", tags=["Location Insights"])
app.include_router(market_router, prefix="/api", tags=["Market Trends"])
app.include_router(best_market_router, prefix="/api", tags=["Best Market"])
app.include_router(irrigation_router, prefix="/api", tags=["Smart Irrigation"])
app.include_router(calendar_router, prefix="/api", tags=["Farming Calendar"])
app.include_router(gov_support_router, prefix="/api", tags=["Government Support"])
app.include_router(crop_recommend_router, prefix="/api", tags=["Crop Recommendation"])
app.include_router(grafting_router, prefix="/api", tags=["Grafting Advisor"])
app.include_router(whatsapp_router, prefix="/api", tags=["WhatsApp Bot"])
app.include_router(harvest_loss_router, prefix="/api", tags=["Harvest Loss Predictor"])


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Lightweight endpoint for uptime monitors, load balancers, and frontend connectivity checks."""
    return {
        "status": "ok",
        "service": "CropGenie HarvestAI",
        "gemini_api_calls": _get_counter_snapshot(),
    }
