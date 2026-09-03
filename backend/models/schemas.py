"""
Pydantic request / response schemas shared across the backend.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Supported Languages ────────────────────────────────────────────

class LanguageCode:
    """
    ISO codes for the languages the voice pipeline supports.
    Used by both Speech-to-Text and Text-to-Speech services.
    """

    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    ENGLISH_IN = "en-IN"
    URDU = "ur-PK"       # Urdu (Pakistan)
    PASHTO = "ps-AF"     # Pashto (Afghanistan)

    DISPLAY_NAMES = {
        "en-US": "English (US)",
        "en-GB": "English (UK)",
        "en-IN": "English (India)",
        "ur-PK": "اردو (Urdu)",
        "ps-AF": "پښتو (Pashto)",
    }


# ── Chat ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message inside a conversation."""

    role: str = Field(
        ...,
        description="'user' or 'assistant'",
    )
    content: str = Field(
        ...,
        description="Plain-text message body",
    )


class ChatRequest(BaseModel):
    """Payload the frontend sends to /api/chat."""

    message: str = Field(
        ...,
        min_length=1,
        description="The farmer's latest question or statement",
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Full conversation history for context",
    )
    language: str = Field(
        default=LanguageCode.ENGLISH_US,
        description="Selected language code (for future locale-aware replies)",
    )


class ChatResponse(BaseModel):
    """Response returned by /api/chat."""

    response: str = Field(
        ...,
        description="AI-generated answer text",
    )


# ── Speech ─────────────────────────────────────────────────────────

class STTResponse(BaseModel):
    """Response returned by /api/speech-to-text."""

    text: str = Field(
        ...,
        description="Transcribed text from the audio",
    )


class TTSRequest(BaseModel):
    """Payload the frontend sends to /api/text-to-speech."""

    text: str = Field(
        ...,
        min_length=1,
        description="Text to convert into speech",
    )
    language: str = Field(
        default=LanguageCode.ENGLISH_US,
        description="Language code for voice synthesis",
    )


# ── Weather ────────────────────────────────────────────────────────

class WeatherRequest(BaseModel):
    """Payload the frontend sends to /api/weather."""

    city: Optional[str] = Field(
        default=None,
        description="City or town name (e.g. 'Lahore', 'Peshawar')",
    )
    lat: Optional[float] = Field(
        default=None,
        description="Latitude for GPS-based lookup",
    )
    lon: Optional[float] = Field(
        default=None,
        description="Longitude for GPS-based lookup",
    )


class ForecastDay(BaseModel):
    """A single day's forecast summary."""

    date: str = Field(..., description="Date (YYYY-MM-DD)")
    temp_min: float = Field(..., description="Minimum temperature (°C)")
    temp_max: float = Field(..., description="Maximum temperature (°C)")
    description: str = Field(..., description="Short weather description")
    rain_chance: float = Field(
        ..., description="Probability of precipitation (0.0 – 1.0)"
    )
    humidity: int = Field(..., description="Relative humidity (%)")


class CurrentWeather(BaseModel):
    """Real-time weather snapshot."""

    city: str = Field(..., description="Resolved city name")
    temperature: float = Field(..., description="Current temperature (°C)")
    feels_like: float = Field(..., description="Feels-like temperature (°C)")
    humidity: int = Field(..., description="Relative humidity (%)")
    wind_speed: float = Field(..., description="Wind speed (m/s)")
    description: str = Field(..., description="Short weather description")


class WeatherResponse(BaseModel):
    """Combined weather data + AI farming advice."""

    current: CurrentWeather = Field(
        ..., description="Current weather conditions"
    )
    forecast: List[ForecastDay] = Field(
        default_factory=list,
        description="Next 2-3 days forecast",
    )
    advice: str = Field(
        ..., description="AI-generated farming advice based on conditions"
    )


# ── Crop Yield & Profit Prediction ─────────────────────────────────

class PredictProfitRequest(BaseModel):
    """Payload the frontend sends to /api/predict-profit."""

    crop: str = Field(
        ..., description="Crop name (e.g. 'Wheat', 'Rice', 'Cotton')"
    )
    acres: float = Field(
        ..., gt=0, description="Farm size in acres"
    )
    city: str = Field(
        ..., min_length=1, description="City/town name for weather lookup"
    )
    fertilizer: float = Field(
        default=65, description="Fertilizer amount (kg/acre, approx.)"
    )
    nitrogen: float = Field(
        default=70, description="Soil Nitrogen level (N)"
    )
    phosphorus: float = Field(
        default=21, description="Soil Phosphorus level (P)"
    )
    potassium: float = Field(
        default=18, description="Soil Potassium level (K)"
    )


class ProfitResponse(BaseModel):
    """Full profit prediction result returned by /api/predict-profit."""

    crop: str = Field(..., description="Crop name")
    acres: float = Field(..., description="Farm size")
    predicted_yield: float = Field(
        ..., description="ML-predicted yield per acre"
    )
    total_yield: float = Field(
        ..., description="Total predicted yield (yield × acres)"
    )
    revenue: float = Field(
        ..., description="Estimated revenue (PKR)"
    )
    cost: float = Field(
        ..., description="Estimated farming cost (PKR)"
    )
    profit: float = Field(
        ..., description="Estimated profit (PKR)"
    )
    profit_margin: float = Field(
        ..., description="Profit margin (%)"
    )
    risk_warning: Optional[str] = Field(
        default=None,
        description="Weather-based risk warning (None if no risk)",
    )
    has_risk: bool = Field(
        default=False,
        description="True if a weather risk was detected",
    )
    weather_city: str = Field(
        ..., description="City used for weather lookup"
    )
    temperature: float = Field(
        ..., description="Current temperature for the location"
    )
    humidity: int = Field(
        ..., description="Current humidity for the location"
    )


# ── Plant Disease Detection ──────────────────────────────────────

class DiseaseDetectionResponse(BaseModel):
    """Structured result returned by /api/detect-disease.

    Gemini's vision model analyses a plant image and returns this
    structured response.  All string fields use farmer-friendly language.
    """

    plant_name: str = Field(
        ...,
        description="Identified plant/crop name (e.g. 'Tomato', 'Wheat', 'Cotton')",
    )
    health_status: str = Field(
        ...,
        description="One of: 'healthy', 'diseased', 'nutrient_deficiency', or 'unknown'",
    )
    disease_name: Optional[str] = Field(
        default=None,
        description="Name of identified disease/issue (None if healthy or unknown)",
    )
    cause: Optional[str] = Field(
        default=None,
        description="Likely cause of the disease or issue",
    )
    treatment: List[str] = Field(
        default_factory=list,
        description="Actionable treatment steps (each a short sentence)",
    )
    prevention: List[str] = Field(
        default_factory=list,
        description="Prevention tips to avoid recurrence",
    )
    summary: str = Field(
        ...,
        description="Brief overall summary of the analysis for the farmer",
    )


# ── Fertilizer Recommendation ────────────────────────────────────────

class FertilizerRequest(BaseModel):
    """Payload the frontend sends to /api/recommend-fertilizer."""

    crop: str = Field(
        ..., description="Crop type (e.g. 'Wheat', 'Cotton', 'Maize')"
    )
    soil_type: str = Field(
        ..., description="Soil type (e.g. 'Loamy', 'Clayey', 'Sandy')"
    )
    nitrogen: float = Field(
        ..., ge=0, description="Soil Nitrogen level"
    )
    phosphorus: float = Field(
        ..., ge=0, description="Soil Phosphorus level"
    )
    potassium: float = Field(
        ..., ge=0, description="Soil Potassium level"
    )
    acres: float = Field(
        default=5.0, gt=0, description="Farm size in acres"
    )
    city: Optional[str] = Field(
        default=None,
        description="City for weather lookup (optional; falls back to manual T/H/M)",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Current temperature °C (manual fallback if no weather)",
    )
    humidity: Optional[float] = Field(
        default=None,
        description="Current humidity % (manual fallback if no weather)",
    )
    moisture: float = Field(
        default=40.0,
        ge=0, le=100,
        description="Soil moisture % (manual or estimated)",
    )
    language: str = Field(
        default="en-US",
        description="Language code for the AI explanation",
    )


class FertilizerResponse(BaseModel):
    """Full fertilizer recommendation returned by /api/recommend-fertilizer."""

    fertilizer_name: str = Field(
        ..., description="ML-predicted fertilizer name (e.g. 'Urea', 'DAP')"
    )
    crop: str = Field(..., description="Crop type")
    soil_type: str = Field(..., description="Soil type")
    nitrogen: float = Field(..., description="Input nitrogen level")
    phosphorus: float = Field(..., description="Input phosphorus level")
    potassium: float = Field(..., description="Input potassium level")
    acres: float = Field(..., description="Farm size")
    temperature: float = Field(
        ..., description="Temperature used (°C)"
    )
    humidity: float = Field(
        ..., description="Humidity used (%)"
    )
    moisture: float = Field(
        ..., description="Moisture used (%)"
    )
    explanation: str = Field(
        ...,
        description="AI-generated farmer-friendly explanation (why + how much + when + warnings)",
    )


# ── Location-Based Insights ───────────────────────────────────────────

class LocationInsightsRequest(BaseModel):
    """Payload the frontend sends to /api/location-insights."""

    city: str = Field(
        ..., min_length=1,
        description="City or district name in Pakistan",
    )
    language: str = Field(
        default="en-US",
        description="Language code for the AI explanation",
    )


class LocationInsightsResponse(BaseModel):
    """Full regional guide returned by /api/location-insights."""

    district: str = Field(
        ..., description="Matched district name (title case)"
    )
    zone_key: str = Field(
        ..., description="Zone identifier (e.g. 'Zone 5')"
    )
    zone_name: str = Field(
        ..., description="Full zone name (e.g. 'Northern Irrigated Plain')"
    )
    zone_icon: str = Field(
        default="🌿",
        description="Emoji icon for the zone",
    )
    climate: str = Field(
        ..., description="Zone climate description"
    )
    soil: str = Field(
        ..., description="Zone soil description"
    )
    crops: List[str] = Field(
        default_factory=list,
        description="Typical crops for this zone",
    )
    irrigation: str = Field(
        ..., description="Irrigation availability and sources"
    )
    challenges: List[str] = Field(
        default_factory=list,
        description="Key agricultural challenges",
    )
    temperature: Optional[float] = Field(
        default=None, description="Current temperature (°C)"
    )
    humidity: Optional[int] = Field(
        default=None, description="Current humidity (%)"
    )
    weather_description: Optional[str] = Field(
        default=None, description="Current weather description"
    )
    insights: str = Field(
        ..., description="AI-generated regional farming insights"
    )


# ── Market Price Trends ────────────────────────────────────────────────

class MarketTrendRequest(BaseModel):
    """Payload the frontend sends to /api/market-trend."""

    commodity: str = Field(
        ..., min_length=1,
        description="Commodity name (e.g. 'Wheat', 'Cotton', 'Maize')",
    )
    city: str = Field(
        default="",
        description="Optional city/district for location-aware advice",
    )
    language: str = Field(
        default="en-US",
        description="Language code for the AI advice",
    )


class MonthlyStat(BaseModel):
    """Monthly price statistics for a commodity."""

    month: str = Field(..., description="Month abbreviation (Jan, Feb, ...)")
    avg_price: float = Field(..., description="Average modal price")
    min_price: float = Field(..., description="Minimum price recorded")
    max_price: float = Field(..., description="Maximum price recorded")


class ChartPoint(BaseModel):
    """A single data point for the price trend chart."""

    date: str = Field(..., description="Year-month (YYYY-MM)")
    price: float = Field(..., description="Average modal price")


class MarketTrendResponse(BaseModel):
    """Full market trend analysis returned by /api/market-trend."""

    commodity: str = Field(
        ..., description="Display name of the commodity"
    )
    trend: str = Field(
        ..., description="Trend direction: 'rising', 'falling', or 'stable'"
    )
    latest_price: float = Field(
        ...,
        description="Latest average price (Rs./maund PKR, or native unit for count crops)",
    )
    latest_date: str = Field(
        ..., description="Latest available month (YYYY-MM)"
    )
    price_unit: str = Field(
        default="Rs. / maund (40 kg)",
        description=(
            "Unit label for all prices (Rs. / maund, Rs. / 100 pcs, "
            "or Rs. / dozen)"
        ),
    )
    pct_change_3m: float = Field(
        default=0.0,
        description="Percentage price change (recent 3 months vs prior 3 months)",
    )
    best_months: List[str] = Field(
        default_factory=list,
        description="Top 3 months with highest average prices (best time to sell)",
    )
    worst_months: List[str] = Field(
        default_factory=list,
        description="Bottom 3 months with lowest average prices",
    )
    seasonal_stats: List[MonthlyStat] = Field(
        default_factory=list,
        description="Monthly seasonal statistics for charting",
    )
    chart_data: List[ChartPoint] = Field(
        default_factory=list,
        description="Monthly time series for the price line chart",
    )
    date_range_start: str = Field(
        ..., description="Dataset start date"
    )
    date_range_end: str = Field(
        ..., description="Dataset end date"
    )
    advice: str = Field(
        ..., description="AI-generated farmer-friendly market advice"
    )
    matched_city: Optional[str] = Field(
        default=None,
        description="Resolved city/district name (if location was provided)",
    )
    zone_name: Optional[str] = Field(
        default=None,
        description="Agro-Ecological Zone name (if location was resolved)",
    )
    zone_icon: Optional[str] = Field(
        default=None,
        description="Zone icon/emoji (if location was resolved)",
    )
    disclaimer: str = Field(
        default=(
            "Historical reference data — actual local mandi prices may vary. "
            "Always confirm with your local market before making decisions."
        ),
        description="Data source disclaimer",
    )


# ── Smart Irrigation ─────────────────────────────────────────────────

class IrrigationRequest(BaseModel):
    """Payload the frontend sends to /api/irrigation/recommend."""

    crop_name: str = Field(
        ..., min_length=1,
        description="Crop name (e.g. 'Wheat', 'Cotton', 'Rice')",
    )
    location: str = Field(
        ..., min_length=1,
        description="City or district name (e.g. 'Multan', 'Lahore')",
    )
    last_irrigation_days_ago: Optional[int] = Field(
        default=None, ge=0,
        description="Days since last irrigation (null if unknown)",
    )
    soil_type: Optional[str] = Field(
        default=None,
        description="Soil type (e.g. 'Loamy', 'Clayey', 'Sandy')",
    )


class IrrigationResponse(BaseModel):
    """Smart irrigation recommendation returned by /api/irrigation/recommend."""

    crop: str = Field(..., description="Crop name")
    location: str = Field(..., description="Location used for weather lookup")
    decision: str = Field(
        ..., description="Clear decision: 'Irrigate now', 'Wait', or 'Irrigate in X days'"
    )
    reason: str = Field(
        ..., description="One-sentence reason in simple language"
    )
    water_amount: str = Field(
        default="N/A",
        description="Water amount guidance: 'light', 'moderate', 'heavy', or 'N/A'",
    )
    raw_advice: str = Field(
        ..., description="Full AI-generated irrigation advice text"
    )
    temperature: float = Field(
        ..., description="Current temperature (°C) used for analysis"
    )
    humidity: int = Field(
        ..., description="Current humidity (%) used for analysis"
    )
    rain_forecast_text: str = Field(
        default="",
        description="Human-readable 3-day rain forecast summary",
    )


# ── Farming Calendar ───────────────────────────────────────────────────

class CalendarRequest(BaseModel):
    """Payload the frontend sends to /api/calendar/generate."""

    crop_name: str = Field(
        ..., min_length=1,
        description="Crop name (e.g. 'Wheat', 'Cotton', 'Rice')",
    )
    location: str = Field(
        ..., min_length=1,
        description="Region, district, or province (e.g. 'Punjab', 'Multan')",
    )
    sowing_date: Optional[str] = Field(
        default=None,
        description=(
            "Planned sowing date (e.g. '15 Nov 2026'). "
            "If not provided, Gemini will suggest the ideal sowing window."
        ),
    )


class CalendarSchedule(BaseModel):
    """Structured farming calendar broken into stages."""

    sowing: str = Field(
        ..., description="Sowing window or date"
    )
    watering_schedule: List[str] = Field(
        default_factory=list,
        description="Key irrigation stages (germination, flowering, etc.)",
    )
    fertilizer_schedule: List[str] = Field(
        default_factory=list,
        description="Fertilizer type and timing for each growth stage",
    )
    spray_reminders: List[str] = Field(
        default_factory=list,
        description="Pest/disease spray reminders at vulnerable stages",
    )
    harvesting: str = Field(
        ..., description="Expected harvesting date or window"
    )


class CalendarResponse(BaseModel):
    """Complete personalised farming calendar returned by /api/calendar/generate."""

    crop: str = Field(..., description="Crop name")
    location: str = Field(..., description="Region/location used")
    sowing_date_provided: bool = Field(
        default=False,
        description="True if the farmer provided a sowing date, False if Gemini suggested one",
    )
    calendar: CalendarSchedule = Field(
        ..., description="Structured farming calendar by stage"
    )
    raw_advice: str = Field(
        ..., description="Full unmodified AI-generated calendar text"
    )


# ── Government Support ──────────────────────────────────────────────

class GovSupportRequest(BaseModel):
    """Payload the frontend sends to /api/gov-support/ask."""

    query: str = Field(
        ..., min_length=1,
        description="Farmer's question about government schemes, subsidies, loans, etc.",
    )
    province: Optional[str] = Field(
        default=None,
        description=(
            "Province for region-specific info "
            "(Punjab, Sindh, KPK, Balochistan). "
            "If omitted, gives Pakistan-wide information."
        ),
    )


class GovSupportResponse(BaseModel):
    """Response from /api/gov-support/ask."""

    query: str = Field(
        ..., description="The farmer's original question"
    )
    province: str = Field(
        default="General (all provinces)",
        description="Province used for the answer, or 'General' if none specified",
    )
    answer: str = Field(
        ..., description="AI-generated answer about government support programs"
    )
    disclaimer: str = Field(
        default=(
            "Information may not reflect the latest scheme updates — "
            "please verify with your local Agriculture Extension Office "
            "or Zarai Taraqiati Bank (ZTBL)."
        ),
        description="Disclaimer reminding farmers to verify details locally",
    )


# ── Crop Recommendation ─────────────────────────────────────────────

class CropAlternative(BaseModel):
    """A single crop prediction with its confidence score."""

    crop: str = Field(..., description="Crop name")
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Model confidence for this crop (0.0 – 1.0)",
    )


class CropRecommendationRequest(BaseModel):
    """Payload the frontend sends to /api/crop/recommend."""

    nitrogen: float = Field(
        ..., ge=0, le=200,
        description="Soil Nitrogen level (N) in kg/ha",
    )
    phosphorus: float = Field(
        ..., ge=0, le=200,
        description="Soil Phosphorus level (P) in kg/ha",
    )
    potassium: float = Field(
        ..., ge=0, le=250,
        description="Soil Potassium level (K) in kg/ha",
    )
    temperature: float = Field(
        ..., ge=0, le=55,
        description="Average temperature in °C",
    )
    humidity: float = Field(
        ..., ge=0, le=100,
        description="Relative humidity percentage (0–100)",
    )
    ph: float = Field(
        ..., ge=0, le=14,
        description="Soil pH level (0–14)",
    )
    rainfall: float = Field(
        ..., ge=0, le=500,
        description="Annual rainfall in mm",
    )


class CropRecommendationResponse(BaseModel):
    """Full crop recommendation returned by /api/crop/recommend."""

    recommended_crop: str = Field(
        ..., description="Top recommended crop name"
    )
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Model confidence for the top crop (0.0 – 1.0)",
    )
    top_3_alternatives: List[CropAlternative] = Field(
        default_factory=list,
        description="Top 3 most likely crops with confidence scores",
    )
    explanation: str = Field(
        ...,
        description="AI-generated farmer-friendly explanation of why this crop suits the conditions",
    )


# ── Crop Grafting Advisor ──────────────────────────────────────────

class GraftingRequest(BaseModel):
    """Payload the frontend sends to /api/grafting/advise."""

    base_crop: str = Field(
        ..., min_length=1,
        description="The plant/tree the farmer currently has (e.g. 'Mango', 'Citrus')",
    )
    goal: Optional[str] = Field(
        default=None,
        description=(
            "What the farmer wants to achieve "
            "(e.g. 'better yield', 'disease resistance', 'faster growth'). "
            "If omitted, general grafting advice is given."
        ),
    )
    language: str = Field(
        default="en",
        description="Response language: 'en' (English), 'ur' (Urdu), or 'ps' (Pashto)",
    )


class GraftingResponse(BaseModel):
    """Response from /api/grafting/advise."""

    base_crop: str = Field(
        ..., description="The crop/tree the farmer asked about"
    )
    goal: str = Field(
        default="General improvement",
        description="The farmer's goal (or 'General improvement' if none specified)",
    )
    language: str = Field(
        default="en",
        description="Language the advice was generated in",
    )
    advice: str = Field(
        ..., description="AI-generated grafting advice in simple farmer-friendly language"
    )


# ── Best Market (Pakistani Mandi Prices) ──────────────────────────

class BestMarketRequest(BaseModel):
    """Payload the frontend sends to /api/best-market/recommend."""

    crop: str = Field(
        ..., min_length=1,
        description="Crop name (e.g. 'paddy', 'Paddy Basmati', 'mango', 'rice')",
    )
    location: str = Field(
        default="",
        description="Farmer's city/district — used for home-market comparison and transport estimates",
    )
    language: str = Field(
        default="en-US",
        description="Language code for the AI advice (en-US, ur-PK, ps-AF)",
    )


class MarketPrice(BaseModel):
    """One market's recent price summary for a crop."""

    market: str = Field(..., description="Mandi / market town name")
    avg_price: float = Field(
        ...,
        description="Recent average price — Rs/maund (weight crops) or native unit (count crops)",
    )
    price_per_bori: Optional[float] = Field(
        default=None,
        description="Price per bori/bag assuming 100 kg (weight crops only)",
    )
    unit_label: str = Field(
        default="Rs. / maund",
        description="Human-readable unit (e.g. 'Rs. / maund', 'Rs. / 100 pieces')",
    )
    min_price: float = Field(..., description="Recent minimum price")
    max_price: float = Field(..., description="Recent maximum price")
    records: int = Field(..., description="Number of price quotes behind the average")
    last_date: str = Field(..., description="Date of the most recent quote (YYYY-MM-DD)")


class BestMarketResponse(BaseModel):
    """Response from /api/best-market/recommend."""

    crop: str = Field(..., description="Matched crop variety (e.g. 'Paddy Basmati')")
    commodity: str = Field(..., description="Commodity group (e.g. 'paddy')")
    unit_label: str = Field(
        ..., description="Price unit label (e.g. 'Rs. / maund (40 kg)')"
    )
    bag_weight_kg: Optional[int] = Field(
        default=None,
        description="Assumed bori/bag weight in kg (100 kg default, weight crops only)",
    )
    markets: List[MarketPrice] = Field(
        default_factory=list,
        description="Top markets ranked by recent average price",
    )
    national_avg: float = Field(
        ..., description="Recent national average price in the same unit"
    )
    home_market: Optional[str] = Field(
        default=None,
        description="Farmer's nearest matched market (from their location)",
    )
    home_price: Optional[float] = Field(
        default=None,
        description="Recent average price at the home market (same unit)",
    )
    best_gain_vs_home: Optional[float] = Field(
        default=None,
        description="Price difference between best market and home market (Rs/maund)",
    )
    best_gain_pct: Optional[float] = Field(
        default=None,
        description="Price gain percentage vs home market",
    )
    last_date: str = Field(
        ..., description="Date of the most recent quote used (YYYY-MM-DD)"
    )
    advice: str = Field(
        ..., description="AI-generated selling advice with transport-cost consideration"
    )
    disclaimer: str = Field(
        default=(
            "Historical Pakistani mandi averages (2007–2022) — actual daily "
            "prices vary. Confirm with your local market before selling."
        ),
        description="Data-source disclaimer shown to the farmer",
    )


# ── Harvest Loss Predictor ───────────────────────────────────────────

class HarvestLossRequest(BaseModel):
    """Input for the harvest-loss prediction endpoint."""

    crop_name: str = Field(
        ..., description="Crop name, e.g. 'Wheat', 'Rice', 'Cotton'"
    )
    location: str = Field(
        ..., description="City or district for weather lookup, e.g. 'Multan'"
    )
    growth_stage: str = Field(
        ...,
        description=(
            "Current growth stage — one of: 'early grain filling', "
            "'nearing maturity', 'fully mature', 'overripe'"
        ),
    )
    planned_harvest_date: Optional[str] = Field(
        default=None,
        description="Planned harvest date (free text, e.g. '15 April 2027'). If blank, today = day 0.",
    )
    language: str = Field(
        default="en",
        description="Short language code: 'en', 'ur', or 'ps'",
    )


class HarvestLossScenario(BaseModel):
    """One harvest-delay scenario (e.g. Day 0, Day 5, Day 10)."""

    delay_days: int = Field(..., description="Days of delay from optimal harvest")
    expected_yield_pct: float = Field(
        ..., description="Expected yield as % of maximum (0-100)"
    )
    loss_pct: float = Field(
        ..., description="Estimated loss compared to optimal harvest (0-100)"
    )


class HarvestLossResponse(BaseModel):
    """Response from the harvest-loss prediction endpoint."""

    crop: str = Field(..., description="Echoed crop name")
    location: str = Field(..., description="Resolved location from weather API")
    language: str = Field(..., description="Language code used")
    scenarios: List[HarvestLossScenario] = Field(
        default_factory=list,
        description="Yield/loss estimates for each delay interval",
    )
    decision_badge: str = Field(
        default="HARVEST NOW",
        description="One of: 'HARVEST NOW', 'WAIT 2 DAYS', 'HARVEST BEFORE RAIN'",
    )
    advice_text: str = Field(
        ..., description="Full AI-generated advice in the selected language"
    )
    weather_used: bool = Field(
        default=True,
        description="Whether live weather data was successfully included",
    )
    market_used: bool = Field(
        default=True,
        description="Whether live market price data was successfully included",
    )

