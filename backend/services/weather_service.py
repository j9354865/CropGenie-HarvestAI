"""
Weather service — fetches real-time and forecast data from OpenWeatherMap.

Endpoints used (free tier):
    • /data/2.5/weather       — current conditions
    • /data/2.5/forecast      — 5-day / 3-hour forecast (we aggregate to daily)

The API key is read from config (OPENWEATHER_API_KEY in .env).
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import httpx

from backend.config import settings
from backend.models.schemas import CurrentWeather, ForecastDay

logger = logging.getLogger(__name__)

# OpenWeatherMap base URL
_BASE_URL = "https://api.openweathermap.org/data/2.5"


class WeatherService:
    """Fetches weather data from OpenWeatherMap and normalises it."""

    def __init__(self) -> None:
        if not settings.openweather_api_key:
            raise RuntimeError(
                "OPENWEATHER_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        self._api_key = settings.openweather_api_key

    # ── Public API ──────────────────────────────────────────────────

    def get_current_weather(
        self,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> CurrentWeather:
        """
        Fetch current weather for a city name **or** lat/lon coordinates.

        Raises
        ------
        ValueError
            If the location is invalid or not found.
        RuntimeError
            On API errors (network, rate-limit, server error).
        """
        params = self._build_params(city, lat, lon)
        data = self._get(f"{_BASE_URL}/weather", params)

        return CurrentWeather(
            city=data["name"],
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            humidity=data["main"]["humidity"],
            wind_speed=data["wind"]["speed"],
            description=data["weather"][0]["description"].title(),
        )

    def get_forecast(
        self,
        city: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        days: int = 3,
    ) -> List[ForecastDay]:
        """
        Fetch a short-term forecast (next *days* days).

        The OpenWeatherMap 5-day endpoint returns data in 3-hour blocks.
        We aggregate them into daily summaries (min/max temp, avg humidity,
        max precipitation probability).

        Returns at most *days* ForecastDay entries.
        """
        params = self._build_params(city, lat, lon)
        data = self._get(f"{_BASE_URL}/forecast", params)

        return self._aggregate_forecast(data["list"], days)

    # ── Internal helpers ────────────────────────────────────────────

    def _build_params(
        self,
        city: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
    ) -> dict:
        """Build the query parameters for an OpenWeatherMap request."""
        params: dict = {
            "appid": self._api_key,
            "units": "metric",   # Celsius
            "lang": "en",
        }

        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        elif city:
            params["q"] = city.strip()
        else:
            raise ValueError(
                "Provide either a city name or latitude/longitude coordinates."
            )

        return params

    @staticmethod
    def _get(url: str, params: dict) -> dict:
        """
        Perform a synchronous GET request with error handling.

        Raises
        ------
        ValueError
            If the API returns a 404 (location not found).
        RuntimeError
            For any other non-200 response.
        """
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(url, params=params)

            if resp.status_code == 404:
                location = params.get("q", f"{params.get('lat')},{params.get('lon')}")
                raise ValueError(
                    f"Location '{location}' not found. "
                    "Please check the spelling or try a nearby city."
                )

            if resp.status_code == 401:
                raise RuntimeError(
                    "OpenWeatherMap API key is invalid or expired. "
                    "Check OPENWEATHER_API_KEY in your .env file."
                )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Weather service returned status {resp.status_code}. "
                    "Please try again in a moment."
                )

            return resp.json()

        except httpx.TimeoutException:
            raise RuntimeError(
                "Weather service is taking too long to respond. "
                "Please try again."
            )
        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot reach the weather service. "
                "Check your internet connection."
            )

    @staticmethod
    def _aggregate_forecast(
        raw_blocks: List[dict],
        max_days: int,
    ) -> List[ForecastDay]:
        """
        Aggregate 3-hour forecast blocks into daily summaries.

        Groups by date, then computes:
            • min/max temperature
            • average humidity
            • max precipitation probability
            • most common weather description
        """
        # Group blocks by date
        daily: dict[str, list[dict]] = {}
        for block in raw_blocks:
            dt = datetime.fromtimestamp(block["dt"]).strftime("%Y-%m-%d")
            daily.setdefault(dt, []).append(block)

        forecast: List[ForecastDay] = []
        for date_str in sorted(daily.keys())[:max_days]:
            blocks = daily[date_str]

            temps = [b["main"]["temp"] for b in blocks]
            humidities = [b["main"]["humidity"] for b in blocks]

            # Pop (probability of precipitation) — may be absent
            pops = [b.get("pop", 0.0) for b in blocks]

            # Most common description
            descriptions = [b["weather"][0]["description"].title() for b in blocks]
            most_common_desc = max(set(descriptions), key=descriptions.count)

            forecast.append(
                ForecastDay(
                    date=date_str,
                    temp_min=round(min(temps), 1),
                    temp_max=round(max(temps), 1),
                    description=most_common_desc,
                    rain_chance=round(max(pops), 2),
                    humidity=round(sum(humidities) / len(humidities)),
                )
            )

        return forecast
