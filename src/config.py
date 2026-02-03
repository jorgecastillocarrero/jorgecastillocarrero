"""
Configuration management for Financial Data Project.
Uses pydantic-settings for environment variable handling.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


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

    # Database Configuration - can be overridden via DATABASE_URL env var
    database_url: str = ""

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

    # Anthropic (Claude) Configuration
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Google (Gemini) Configuration
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Groq Configuration
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Logging
    log_level: str = "INFO"

    # Data paths
    data_dir: Path = Path("data")

    # Dashboard Authentication
    dashboard_password: str = ""
    dashboard_auth_enabled: bool = False

    @property
    def effective_database_url(self) -> str:
        """Get database URL, defaulting to data/financial_data.db in project root."""
        if self.database_url:
            return self.database_url
        db_path = PROJECT_ROOT / self.data_dir / "financial_data.db"
        return f"sqlite:///{db_path}"

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
