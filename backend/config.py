"""
Application configuration — loaded from environment variables via .env file.
Never hardcode secrets; always reference them through this config module.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central application settings.  All values are read from environment
    variables (or a .env file) at startup.
    """

    # ── AI Provider ────────────────────────────────────────────────
    gemini_api_key: str = ""
    # Google retires model names periodically.  If you see a 404 error
    # like "models/<name> is no longer available", just update this
    # default (or the GEMINI_MODEL value in .env) to the model name
    # mentioned in the error message.
    gemini_model: str = "gemini-3.6-flash"

    # ── Weather Data ─────────────────────────────────────────────────
    # Get a free key at https://openweathermap.org/api
    openweather_api_key: str = ""

    # ── Twilio (WhatsApp Bot) ─────────────────────────────────
    # Get credentials from https://console.twilio.com
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    # The Twilio WhatsApp Sandbox number (e.g. "+14155238886")
    twilio_whatsapp_number: str = ""

    # ── Server ─────────────────────────────────────────────────────
    backend_host: str = "http://localhost:8000"
    backend_port: int = 8000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Singleton — import this anywhere you need config values
settings = Settings()
