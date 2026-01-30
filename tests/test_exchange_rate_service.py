"""
Tests for src/exchange_rate_service.py - Exchange rate service.
"""

import pytest
from datetime import date
from unittest.mock import patch, MagicMock

from src.exchange_rate_service import (
    ExchangeRateService,
    ExchangeRateError,
    get_exchange_rate_service,
    MAX_LOOKBACK_DAYS,
)


class TestExchangeRateServiceInit:
    """Tests for service initialization."""

    def test_init_with_db_manager(self):
        """Test initialization with provided db_manager."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)
        assert service.db is mock_db

    def test_init_creates_empty_cache(self):
        """Test that initialization creates empty cache."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)
        assert service._cache == {}


class TestGetRate:
    """Tests for get_rate method."""

    def test_get_rate_found(self):
        """Test getting a rate that exists."""
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1.0850,)
        mock_session.execute.return_value = mock_result

        service = ExchangeRateService(db_manager=mock_db)
        rate = service.get_rate('EURUSD=X', date(2024, 1, 15))

        assert rate == 1.0850

    def test_get_rate_cached(self):
        """Test that cached rate is returned."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)
        service._cache['EURUSD=X_2024-01-15'] = 1.0900

        rate = service.get_rate('EURUSD=X', date(2024, 1, 15))

        assert rate == 1.0900
        mock_db.get_session.assert_not_called()

    def test_get_rate_lookback(self):
        """Test that lookback works when exact date not found."""
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        # First call returns None, second returns rate
        mock_result_none = MagicMock()
        mock_result_none.fetchone.return_value = None

        mock_result_found = MagicMock()
        mock_result_found.fetchone.return_value = (1.0800,)

        mock_session.execute.side_effect = [mock_result_none, mock_result_found]

        service = ExchangeRateService(db_manager=mock_db)
        rate = service.get_rate('EURUSD=X', date(2024, 1, 15))

        assert rate == 1.0800

    def test_get_rate_not_found_raises(self):
        """Test that ExchangeRateError is raised when not found."""
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        service = ExchangeRateService(db_manager=mock_db)

        with pytest.raises(ExchangeRateError) as exc_info:
            service.get_rate('EURUSD=X', date(2024, 1, 15))

        assert "No exchange rate found" in str(exc_info.value)


class TestGetRateSafe:
    """Tests for get_rate_safe method."""

    def test_get_rate_safe_returns_rate(self):
        """Test that rate is returned when found."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)
        service._cache['EURUSD=X_2024-01-15'] = 1.0850

        rate = service.get_rate_safe('EURUSD=X', date(2024, 1, 15))
        assert rate == 1.0850

    def test_get_rate_safe_returns_none_on_error(self):
        """Test that None is returned when not found."""
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        service = ExchangeRateService(db_manager=mock_db)
        rate = service.get_rate_safe('INVALID=X', date(2024, 1, 15))

        assert rate is None


class TestConvenienceMethods:
    """Tests for convenience methods (get_eur_usd, etc.)."""

    def test_get_eur_usd(self):
        """Test get_eur_usd method."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        with patch.object(service, 'get_rate', return_value=1.0850):
            rate = service.get_eur_usd(date(2024, 1, 15))

        assert rate == 1.0850

    def test_get_cad_eur_direct(self):
        """Test get_cad_eur when direct rate exists."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        with patch.object(service, 'get_rate_safe', return_value=0.68):
            rate = service.get_cad_eur(date(2024, 1, 15))

        assert rate == 0.68

    def test_get_cad_eur_calculated(self):
        """Test get_cad_eur when calculated from other pairs."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        with patch.object(service, 'get_rate_safe', return_value=None):
            with patch.object(service, 'get_rate', side_effect=[1.08, 1.35]):
                # CAD/EUR = 1 / (USD/CAD * EUR/USD) = 1 / (1.35 * 1.08)
                rate = service.get_cad_eur(date(2024, 1, 15))

        expected = 1 / (1.35 * 1.08)
        assert rate == pytest.approx(expected)

    def test_get_gbp_eur_calculated(self):
        """Test get_gbp_eur when calculated from inverse."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        with patch.object(service, 'get_rate_safe', return_value=None):
            with patch.object(service, 'get_rate', return_value=0.85):
                # GBP/EUR = 1 / EUR/GBP = 1 / 0.85
                rate = service.get_gbp_eur(date(2024, 1, 15))

        expected = 1 / 0.85
        assert rate == pytest.approx(expected)


class TestConvertToEur:
    """Tests for convert_to_eur method."""

    def test_convert_eur_passthrough(self):
        """Test that EUR stays as EUR."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        result = service.convert_to_eur(100.0, 'EUR', date(2024, 1, 15))
        assert result == 100.0

    def test_convert_usd_to_eur(self):
        """Test USD to EUR conversion."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        with patch.object(service, 'get_eur_usd', return_value=1.10):
            result = service.convert_to_eur(110.0, 'USD', date(2024, 1, 15))

        assert result == pytest.approx(100.0)

    def test_convert_cad_to_eur(self):
        """Test CAD to EUR conversion."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        with patch.object(service, 'get_cad_eur', return_value=0.70):
            result = service.convert_to_eur(100.0, 'CAD', date(2024, 1, 15))

        assert result == 70.0

    def test_convert_unsupported_raises(self):
        """Test that unsupported currency raises error."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)

        with pytest.raises(ExchangeRateError) as exc_info:
            service.convert_to_eur(100.0, 'XXX', date(2024, 1, 15))

        assert "Unsupported currency" in str(exc_info.value)


class TestClearCache:
    """Tests for cache clearing."""

    def test_clear_cache(self):
        """Test that clear_cache empties the cache."""
        mock_db = MagicMock()
        service = ExchangeRateService(db_manager=mock_db)
        service._cache['test'] = 1.0

        service.clear_cache()

        assert service._cache == {}


class TestGetExchangeRateService:
    """Tests for singleton getter."""

    def test_returns_service(self):
        """Test that function returns a service."""
        # Reset singleton for test
        import src.exchange_rate_service
        src.exchange_rate_service._rate_service = None

        with patch('src.exchange_rate_service.get_db_manager') as mock_get_db:
            mock_get_db.return_value = MagicMock()
            service = get_exchange_rate_service()

        assert isinstance(service, ExchangeRateService)

    def test_singleton_pattern(self):
        """Test that same instance is returned."""
        import src.exchange_rate_service
        src.exchange_rate_service._rate_service = None

        with patch('src.exchange_rate_service.get_db_manager') as mock_get_db:
            mock_get_db.return_value = MagicMock()
            service1 = get_exchange_rate_service()
            service2 = get_exchange_rate_service()

        assert service1 is service2
