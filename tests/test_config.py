"""
Tests for src/config.py - Configuration management.
"""

import os
import pytest
from unittest.mock import patch

from src.config import Settings, get_settings, DEFAULT_SYMBOLS, DEFAULT_EXCHANGES


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test that settings have correct default values."""
        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}, clear=False):
            settings = Settings()
            assert settings.eodhd_base_url == "https://eodhd.com/api"
            assert settings.scheduler_enabled is True
            assert settings.download_interval_hours == 24
            assert settings.api_rate_limit == 100
            assert settings.api_retry_attempts == 3
            assert settings.api_retry_delay == 1.0
            assert settings.log_level == "INFO"

    def test_eodhd_api_key_from_env(self):
        """Test API key is loaded from environment."""
        with patch.dict(os.environ, {"EODHD_API_KEY": "my_test_key"}, clear=False):
            settings = Settings()
            assert settings.eodhd_api_key == "my_test_key"

    def test_is_eodhd_configured_true(self):
        """Test is_eodhd_configured returns True when key is set."""
        with patch.dict(os.environ, {"EODHD_API_KEY": "valid_key"}, clear=False):
            settings = Settings()
            assert settings.is_eodhd_configured is True

    def test_is_eodhd_configured_false(self):
        """Test is_eodhd_configured returns False when key is empty."""
        with patch.dict(os.environ, {"EODHD_API_KEY": ""}, clear=False):
            settings = Settings()
            assert settings.is_eodhd_configured is False

    def test_is_openai_configured_true(self):
        """Test is_openai_configured returns True when key is set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key", "EODHD_API_KEY": "test"}, clear=False):
            settings = Settings()
            assert settings.is_openai_configured is True

    def test_is_openai_configured_false(self):
        """Test is_openai_configured returns False when key is empty."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "EODHD_API_KEY": "test"}, clear=False):
            settings = Settings()
            assert settings.is_openai_configured is False

    def test_openai_model_default(self):
        """Test default OpenAI model."""
        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}, clear=False):
            settings = Settings()
            assert settings.openai_model == "gpt-4o-mini"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        get_settings.cache_clear()  # Clear LRU cache
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached_result(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_symbols_not_empty(self):
        """Test that DEFAULT_SYMBOLS is not empty."""
        assert len(DEFAULT_SYMBOLS) > 0

    def test_default_symbols_format(self):
        """Test that DEFAULT_SYMBOLS have correct format (SYMBOL.EXCHANGE)."""
        for symbol in DEFAULT_SYMBOLS:
            assert "." in symbol, f"Symbol {symbol} missing exchange suffix"
            parts = symbol.split(".")
            assert len(parts) == 2
            assert len(parts[0]) > 0
            assert len(parts[1]) > 0

    def test_default_exchanges_not_empty(self):
        """Test that DEFAULT_EXCHANGES is not empty."""
        assert len(DEFAULT_EXCHANGES) > 0

    def test_us_exchange_in_defaults(self):
        """Test that US exchange is in defaults."""
        assert "US" in DEFAULT_EXCHANGES
