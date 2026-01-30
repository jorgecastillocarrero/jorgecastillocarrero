"""
Configuration management for Financial Data Project.
Uses pydantic-settings for environment variable handling.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # EODHD API Configuration
    eodhd_api_key: str = ""
    eodhd_base_url: str = "https://eodhd.com/api"

    # Database Configuration - absolute path to avoid issues with working directory
    database_url: str = "sqlite:///C:/Users/usuario/financial-data-project/data/financial_data.db"

    # Scheduler Configuration
    scheduler_enabled: bool = True
    download_interval_hours: int = 24

    # Rate Limiting
    api_rate_limit: int = 100  # requests per minute
    api_retry_attempts: int = 3
    api_retry_delay: float = 1.0  # seconds

    # OpenAI Configuration (optional)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Logging
    log_level: str = "INFO"

    # Data paths
    data_dir: Path = Path("data")

    @property
    def is_eodhd_configured(self) -> bool:
        """Check if EODHD API key is configured."""
        return bool(self.eodhd_api_key)

    @property
    def is_openai_configured(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Default symbols to track
DEFAULT_SYMBOLS = [
    "AAPL.US",
    "MSFT.US",
    "GOOGL.US",
    "AMZN.US",
    "TSLA.US",
    "META.US",
    "NVDA.US",
    "JPM.US",
    "V.US",
    "JNJ.US",
]

# Default exchanges
DEFAULT_EXCHANGES = [
    "US",
    "LSE",
    "XETRA",
    "PA",
    "MC",
]
