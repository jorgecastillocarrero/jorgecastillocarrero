"""
Tests for src/daily_tracking.py - Daily portfolio tracking service.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from src.daily_tracking import DailyTrackingService, get_tracking_service
from src.database import (
    DatabaseManager, DailyHolding, StockTrade, DailyCash, CashMovement
)


class TestDailyTrackingServiceInit:
    """Tests for service initialization."""

    def test_init_with_db_manager(self, db_manager):
        """Test initialization with provided db_manager."""
        service = DailyTrackingService(db_manager=db_manager)
        assert service.db is db_manager

    def test_init_without_db_manager(self):
        """Test initialization without db_manager uses default."""
        with patch("src.daily_tracking.get_db_manager") as mock_get_db:
            mock_get_db.return_value = MagicMock()
            service = DailyTrackingService()
            mock_get_db.assert_called_once()


class TestHoldingsManagement:
    """Tests for holdings management."""

    def test_set_holdings_for_date(self, db_manager, sample_holdings_data):
        """Test setting holdings for a date."""
        service = DailyTrackingService(db_manager=db_manager)
        target_date = date(2024, 1, 15)

        count = service.set_holdings_for_date(target_date, sample_holdings_data)

        assert count == len(sample_holdings_data)

    def test_set_holdings_replaces_existing(self, db_manager):
        """Test that set_holdings replaces existing holdings."""
        service = DailyTrackingService(db_manager=db_manager)
        target_date = date(2024, 1, 16)

        # First set
        service.set_holdings_for_date(target_date, [
            {"account_code": "IB", "symbol": "AAPL", "shares": 100, "currency": "USD"}
        ])

        # Second set (should replace)
        count = service.set_holdings_for_date(target_date, [
            {"account_code": "IB", "symbol": "MSFT", "shares": 50, "currency": "USD"}
        ])

        holdings = service.get_holdings_for_date(target_date)
        assert count == 1
        assert len(holdings) == 1
        assert holdings[0]["symbol"] == "MSFT"

    def test_get_holdings_for_date(self, db_manager, sample_holdings_data):
        """Test getting holdings for a date."""
        service = DailyTrackingService(db_manager=db_manager)
        target_date = date(2024, 1, 17)

        service.set_holdings_for_date(target_date, sample_holdings_data)
        holdings = service.get_holdings_for_date(target_date)

        assert len(holdings) == len(sample_holdings_data)
        assert all("symbol" in h for h in holdings)

    def test_get_holdings_for_date_fallback(self, db_manager, sample_holdings_data):
        """Test that get_holdings falls back to previous date."""
        service = DailyTrackingService(db_manager=db_manager)

        # Set holdings for earlier date
        service.set_holdings_for_date(date(2024, 1, 10), sample_holdings_data)

        # Query later date with no data
        holdings = service.get_holdings_for_date(date(2024, 1, 11))

        assert len(holdings) == len(sample_holdings_data)

    def test_get_holdings_by_account(self, db_manager, sample_holdings_data):
        """Test filtering holdings by account."""
        service = DailyTrackingService(db_manager=db_manager)
        target_date = date(2024, 1, 18)

        service.set_holdings_for_date(target_date, sample_holdings_data)
        holdings = service.get_holdings_for_date(target_date, account_code="IB")

        assert all(h["account_code"] == "IB" for h in holdings)

    def test_propagate_holdings(self, db_manager, sample_holdings_data):
        """Test propagating holdings to another date."""
        service = DailyTrackingService(db_manager=db_manager)

        # Set holdings for source date
        source_date = date(2024, 1, 20)
        target_date = date(2024, 1, 21)

        service.set_holdings_for_date(source_date, sample_holdings_data)
        count = service.propagate_holdings(source_date, target_date)

        assert count == len(sample_holdings_data)

        target_holdings = service.get_holdings_for_date(target_date)
        assert len(target_holdings) == len(sample_holdings_data)

    def test_propagate_holdings_no_source(self, db_manager):
        """Test propagate with no source holdings."""
        service = DailyTrackingService(db_manager=db_manager)

        count = service.propagate_holdings(
            date(2024, 2, 1),  # No holdings exist
            date(2024, 2, 2)
        )

        assert count == 0


class TestStockTrades:
    """Tests for stock trade management."""

    def test_add_stock_trade_buy(self, db_manager, sample_holdings_data):
        """Test adding a buy trade."""
        service = DailyTrackingService(db_manager=db_manager)
        trade_date = date(2024, 1, 25)

        # Setup initial holdings
        service.set_holdings_for_date(trade_date, sample_holdings_data)

        # Add buy trade
        trade_id = service.add_stock_trade(
            account_code="IB",
            trade_date=trade_date,
            symbol="NVDA",
            trade_type="BUY",
            shares=25,
            price=500.0,
            currency="USD",
        )

        assert trade_id is not None

    def test_add_stock_trade_sell(self, db_manager):
        """Test adding a sell trade."""
        service = DailyTrackingService(db_manager=db_manager)
        trade_date = date(2024, 1, 26)

        # Setup initial holding
        service.set_holdings_for_date(trade_date, [
            {"account_code": "IB", "symbol": "AAPL", "shares": 100, "currency": "USD", "entry_price": 150}
        ])

        # Add sell trade
        trade_id = service.add_stock_trade(
            account_code="IB",
            trade_date=trade_date,
            symbol="AAPL",
            trade_type="SELL",
            shares=50,
            price=160.0,
        )

        assert trade_id is not None

    def test_get_trades(self, db_manager):
        """Test getting trades."""
        service = DailyTrackingService(db_manager=db_manager)
        trade_date = date(2024, 1, 27)

        service.set_holdings_for_date(trade_date, [])
        service.add_stock_trade(
            account_code="IB",
            trade_date=trade_date,
            symbol="TSLA",
            trade_type="BUY",
            shares=10,
            price=200.0,
        )

        trades = service.get_trades(start_date=trade_date, end_date=trade_date)

        assert len(trades) >= 1
        assert trades[0]["symbol"] == "TSLA"

    def test_get_trades_with_filters(self, db_manager):
        """Test getting trades with filters."""
        service = DailyTrackingService(db_manager=db_manager)
        trade_date = date(2024, 1, 28)

        service.set_holdings_for_date(trade_date, [])
        service.add_stock_trade("IB", trade_date, "AMZN", "BUY", 5, 150.0)
        service.add_stock_trade("CO3365", trade_date, "TEF.MC", "BUY", 100, 4.0, "EUR")

        trades_ib = service.get_trades(account_code="IB")
        trades_amzn = service.get_trades(symbol="AMZN")

        assert all(t["account"] == "IB" for t in trades_ib)
        assert all(t["symbol"] == "AMZN" for t in trades_amzn)


class TestCashManagement:
    """Tests for cash management."""

    def test_set_cash_for_date(self, db_manager, sample_cash_data):
        """Test setting cash balances."""
        service = DailyTrackingService(db_manager=db_manager)
        cash_date = date(2024, 2, 1)

        count = service.set_cash_for_date(cash_date, sample_cash_data)

        assert count == len(sample_cash_data)

    def test_get_cash_for_date(self, db_manager, sample_cash_data):
        """Test getting cash balances."""
        service = DailyTrackingService(db_manager=db_manager)
        cash_date = date(2024, 2, 2)

        service.set_cash_for_date(cash_date, sample_cash_data)
        cash = service.get_cash_for_date(cash_date)

        assert len(cash) == len(sample_cash_data)

    def test_get_cash_for_date_fallback(self, db_manager, sample_cash_data):
        """Test cash fallback to previous date."""
        service = DailyTrackingService(db_manager=db_manager)

        service.set_cash_for_date(date(2024, 2, 3), sample_cash_data)
        cash = service.get_cash_for_date(date(2024, 2, 4))  # No data for this date

        assert len(cash) == len(sample_cash_data)

    def test_get_cash_by_account(self, db_manager, sample_cash_data):
        """Test filtering cash by account."""
        service = DailyTrackingService(db_manager=db_manager)
        cash_date = date(2024, 2, 5)

        service.set_cash_for_date(cash_date, sample_cash_data)
        cash = service.get_cash_for_date(cash_date, account_code="IB")

        assert all(c["account"] == "IB" for c in cash)

    def test_add_cash_movement_deposit(self, db_manager):
        """Test adding a deposit."""
        service = DailyTrackingService(db_manager=db_manager)

        movement_id = service.add_cash_movement(
            account_code="IB",
            movement_date=date(2024, 2, 6),
            movement_type="DEPOSIT",
            amount=10000.0,
            currency="EUR",
        )

        assert movement_id is not None

    def test_add_cash_movement_withdrawal(self, db_manager):
        """Test adding a withdrawal."""
        service = DailyTrackingService(db_manager=db_manager)

        movement_id = service.add_cash_movement(
            account_code="IB",
            movement_date=date(2024, 2, 7),
            movement_type="WITHDRAWAL",
            amount=-5000.0,
            currency="EUR",
        )

        assert movement_id is not None

    def test_add_cash_movement_transfer(self, db_manager):
        """Test adding a transfer."""
        service = DailyTrackingService(db_manager=db_manager)

        movement_id = service.add_cash_movement(
            account_code="IB",
            movement_date=date(2024, 2, 8),
            movement_type="TRANSFER_IN",
            amount=2000.0,
            currency="EUR",
            counterpart_account="CO3365",
        )

        assert movement_id is not None

    def test_get_cash_movements(self, db_manager):
        """Test getting cash movements."""
        service = DailyTrackingService(db_manager=db_manager)
        movement_date = date(2024, 2, 9)

        service.add_cash_movement("IB", movement_date, "DEPOSIT", 1000.0)
        movements = service.get_cash_movements(start_date=movement_date)

        assert len(movements) >= 1

    def test_get_cash_movements_with_filters(self, db_manager):
        """Test getting movements with filters."""
        service = DailyTrackingService(db_manager=db_manager)
        movement_date = date(2024, 2, 10)

        service.add_cash_movement("IB", movement_date, "DEPOSIT", 1000.0)
        service.add_cash_movement("CO3365", movement_date, "DEPOSIT", 500.0)

        movements = service.get_cash_movements(account_code="IB")
        assert all(m["account"] == "IB" for m in movements)


class TestPortfolioValuation:
    """Tests for portfolio valuation."""

    def test_get_portfolio_value_structure(self, db_manager, sample_holdings_data, sample_cash_data):
        """Test portfolio value returns correct structure."""
        service = DailyTrackingService(db_manager=db_manager)
        target_date = date(2024, 2, 15)

        service.set_holdings_for_date(target_date, sample_holdings_data)
        service.set_cash_for_date(target_date, sample_cash_data)

        # Note: This will return empty holdings values because no prices exist
        result = service.get_portfolio_value(target_date)

        assert "date" in result
        assert "rates" in result
        assert "accounts" in result
        assert "total_holdings" in result
        assert "total_cash" in result
        assert "total" in result


class TestGetTrackingService:
    """Tests for get_tracking_service singleton."""

    def test_returns_service(self):
        """Test that function returns a service."""
        with patch("src.daily_tracking.get_db_manager") as mock_db:
            mock_db.return_value = MagicMock()
            service = get_tracking_service()
            assert isinstance(service, DailyTrackingService)

    def test_singleton_pattern(self):
        """Test that same instance is returned."""
        with patch("src.daily_tracking.get_db_manager") as mock_db:
            mock_manager = MagicMock()
            mock_db.return_value = mock_manager

            # Reset singleton for test
            import src.daily_tracking
            src.daily_tracking._tracking_service = None

            service1 = get_tracking_service()
            service2 = get_tracking_service()

            assert service1 is service2
