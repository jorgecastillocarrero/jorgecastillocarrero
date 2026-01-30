"""
Tests for src/database.py - Database models and operations.
"""

import pytest
import os
from datetime import datetime, date
from unittest.mock import patch, MagicMock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import (
    Base,
    DatabaseManager,
    Exchange,
    Symbol,
    PriceHistory,
    Portfolio,
    PortfolioHolding,
    DailyMetrics,
    AccountHolding,
    AssetType,
    PortfolioSnapshot,
    DailyHolding,
    StockTrade,
    DailyCash,
    CashMovement,
    DownloadLog,
    get_db_manager,
)


class TestExchangeModel:
    """Tests for Exchange model."""

    def test_exchange_creation(self, temp_db):
        """Test creating an exchange."""
        exchange = Exchange(
            code="US",
            name="US Stock Exchange",
            country="United States",
            currency="USD",
        )
        temp_db.add(exchange)
        temp_db.commit()

        assert exchange.id is not None
        assert exchange.code == "US"
        assert exchange.is_active is True

    def test_exchange_unique_code(self, temp_db):
        """Test that exchange code is unique."""
        exchange1 = Exchange(code="US", name="Exchange 1")
        temp_db.add(exchange1)
        temp_db.commit()

        exchange2 = Exchange(code="US", name="Exchange 2")
        temp_db.add(exchange2)

        with pytest.raises(Exception):  # IntegrityError
            temp_db.commit()


class TestSymbolModel:
    """Tests for Symbol model."""

    def test_symbol_creation(self, temp_db, sample_exchange):
        """Test creating a symbol."""
        symbol = Symbol(
            code="AAPL",
            exchange_id=sample_exchange.id,
            name="Apple Inc.",
            symbol_type="stock",
            currency="USD",
        )
        temp_db.add(symbol)
        temp_db.commit()

        assert symbol.id is not None
        assert symbol.code == "AAPL"

    def test_symbol_full_symbol_property(self, temp_db, sample_exchange):
        """Test full_symbol property."""
        symbol = Symbol(
            code="AAPL",
            exchange_id=sample_exchange.id,
            name="Apple Inc.",
        )
        temp_db.add(symbol)
        temp_db.commit()
        temp_db.refresh(symbol)
        temp_db.refresh(sample_exchange)

        assert symbol.full_symbol == "AAPL.US"

    def test_symbol_relationships(self, temp_db, sample_exchange):
        """Test symbol relationships."""
        symbol = Symbol(
            code="MSFT",
            exchange_id=sample_exchange.id,
            name="Microsoft Corp.",
        )
        temp_db.add(symbol)
        temp_db.commit()
        temp_db.refresh(sample_exchange)

        assert len(sample_exchange.symbols) >= 1


class TestPriceHistoryModel:
    """Tests for PriceHistory model."""

    def test_price_history_creation(self, temp_db, sample_symbol):
        """Test creating price history."""
        price = PriceHistory(
            symbol_id=sample_symbol.id,
            date=date(2024, 1, 2),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            adjusted_close=101.0,
            volume=1000000,
        )
        temp_db.add(price)
        temp_db.commit()

        assert price.id is not None
        assert price.close == 101.0


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_create_manager(self, db_manager):
        """Test creating DatabaseManager."""
        assert db_manager is not None

    def test_create_tables(self, db_manager):
        """Test that tables are created."""
        # Tables should already be created in fixture
        with db_manager.get_session() as session:
            # Should not raise
            session.query(Exchange).count()

    def test_get_session_context_manager(self, db_manager):
        """Test session context manager."""
        with db_manager.get_session() as session:
            assert session is not None

    def test_get_session_commits_on_success(self, db_manager):
        """Test that session commits on successful exit."""
        with db_manager.get_session() as session:
            exchange = Exchange(code="TEST", name="Test Exchange")
            session.add(exchange)

        with db_manager.get_session() as session:
            result = session.query(Exchange).filter(Exchange.code == "TEST").first()
            assert result is not None

    def test_get_session_rollback_on_error(self, db_manager):
        """Test that session rolls back on error."""
        try:
            with db_manager.get_session() as session:
                exchange = Exchange(code="TEST2", name="Test Exchange 2")
                session.add(exchange)
                raise ValueError("Test error")
        except ValueError:
            pass

        with db_manager.get_session() as session:
            result = session.query(Exchange).filter(Exchange.code == "TEST2").first()
            assert result is None


class TestExchangeOperations:
    """Tests for exchange operations."""

    def test_upsert_exchange_create(self, db_manager):
        """Test upsert creates new exchange."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {
                "code": "NEW",
                "name": "New Exchange",
                "country": "Test Country",
            })
            assert exchange.id is not None
            assert exchange.code == "NEW"

    def test_upsert_exchange_update(self, db_manager):
        """Test upsert updates existing exchange."""
        with db_manager.get_session() as session:
            # Create first
            db_manager.upsert_exchange(session, {"code": "UPD", "name": "Original"})

        with db_manager.get_session() as session:
            # Update
            exchange = db_manager.upsert_exchange(session, {"code": "UPD", "name": "Updated"})
            assert exchange.name == "Updated"


class TestSymbolOperations:
    """Tests for symbol operations."""

    def test_upsert_symbol_create(self, db_manager):
        """Test upsert creates new symbol."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US", "name": "US"})
            symbol = db_manager.upsert_symbol(session, {
                "code": "NEWSTOCK",
                "exchange_id": exchange.id,
                "name": "New Stock Inc.",
            })
            assert symbol.id is not None

    def test_get_symbol_by_code(self, db_manager):
        """Test getting symbol by code."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US2", "name": "US2"})
            db_manager.upsert_symbol(session, {
                "code": "FINDME",
                "exchange_id": exchange.id,
                "name": "Find Me Inc.",
            })

        with db_manager.get_session() as session:
            symbol = db_manager.get_symbol_by_code(session, "FINDME")
            assert symbol is not None
            assert symbol.name == "Find Me Inc."

    def test_get_symbol_by_code_not_found(self, db_manager):
        """Test getting non-existent symbol."""
        with db_manager.get_session() as session:
            symbol = db_manager.get_symbol_by_code(session, "NOTFOUND")
            assert symbol is None


class TestPriceOperations:
    """Tests for price history operations."""

    def test_bulk_insert_prices(self, db_manager, sample_prices_df):
        """Test bulk inserting prices."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US3", "name": "US3"})
            symbol = db_manager.upsert_symbol(session, {
                "code": "BULK",
                "exchange_id": exchange.id,
            })

            count = db_manager.bulk_insert_prices(session, symbol.id, sample_prices_df)
            assert count > 0

    def test_bulk_insert_prices_skips_duplicates(self, db_manager):
        """Test that duplicate dates are skipped."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US4", "name": "US4"})
            symbol = db_manager.upsert_symbol(session, {
                "code": "DUP",
                "exchange_id": exchange.id,
            })

            df = pd.DataFrame({
                "open": [100], "high": [102], "low": [99],
                "close": [101], "adjusted_close": [101], "volume": [1000000],
            }, index=pd.to_datetime(["2024-01-02"]))

            count1 = db_manager.bulk_insert_prices(session, symbol.id, df)
            count2 = db_manager.bulk_insert_prices(session, symbol.id, df)

            assert count1 == 1
            assert count2 == 0

    def test_get_price_history(self, db_manager):
        """Test getting price history as DataFrame."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US5", "name": "US5"})
            symbol = db_manager.upsert_symbol(session, {
                "code": "GETPX",
                "exchange_id": exchange.id,
            })

            # Insert some prices
            price = PriceHistory(
                symbol_id=symbol.id,
                date=date(2024, 1, 2),
                open=100, high=102, low=99, close=101, volume=1000000,
            )
            session.add(price)
            session.flush()

            df = db_manager.get_price_history(session, symbol.id)
            assert not df.empty
            assert "close" in df.columns

    def test_get_latest_price_date(self, db_manager):
        """Test getting latest price date."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US6", "name": "US6"})
            symbol = db_manager.upsert_symbol(session, {
                "code": "LATEST",
                "exchange_id": exchange.id,
            })

            for d in [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]:
                price = PriceHistory(symbol_id=symbol.id, date=d, close=100)
                session.add(price)
            session.flush()

            latest = db_manager.get_latest_price_date(session, symbol.id)
            assert latest == date(2024, 1, 4)


class TestPortfolioOperations:
    """Tests for portfolio operations."""

    def test_create_portfolio(self, db_manager):
        """Test creating a portfolio."""
        with db_manager.get_session() as session:
            portfolio = db_manager.create_portfolio(
                session,
                name="Test Portfolio",
                month=1,
                year=2024,
                initial_capital=100000,
            )
            assert portfolio.id is not None
            assert portfolio.name == "Test Portfolio"

    def test_get_portfolio(self, db_manager):
        """Test getting portfolio."""
        with db_manager.get_session() as session:
            db_manager.create_portfolio(session, name="Find Me", month=2, year=2024)

        with db_manager.get_session() as session:
            portfolio = db_manager.get_portfolio(session, name="Find Me", month=2, year=2024)
            assert portfolio is not None

    def test_get_portfolios(self, db_manager):
        """Test getting all portfolios."""
        with db_manager.get_session() as session:
            db_manager.create_portfolio(session, name="P1", month=1, year=2024)
            db_manager.create_portfolio(session, name="P2", month=2, year=2024)

        with db_manager.get_session() as session:
            portfolios = db_manager.get_portfolios(session, year=2024)
            assert len(portfolios) >= 2


class TestDailyMetricsOperations:
    """Tests for daily metrics operations."""

    def test_upsert_daily_metrics(self, db_manager):
        """Test upserting daily metrics."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US7", "name": "US7"})
            symbol = db_manager.upsert_symbol(session, {
                "code": "METRIC",
                "exchange_id": exchange.id,
            })

            metrics = db_manager.upsert_daily_metrics(session, symbol.id, date(2024, 1, 2), {
                "close_price": 100,
                "rsi_14": 55.5,
                "m200": 0.05,
            })
            assert metrics.rsi_14 == 55.5

    def test_get_daily_metrics(self, db_manager):
        """Test getting daily metrics."""
        with db_manager.get_session() as session:
            exchange = db_manager.upsert_exchange(session, {"code": "US8", "name": "US8"})
            symbol = db_manager.upsert_symbol(session, {
                "code": "GETMET",
                "exchange_id": exchange.id,
            })

            db_manager.upsert_daily_metrics(session, symbol.id, date(2024, 1, 2), {
                "close_price": 100, "rsi_14": 55,
            })

            df = db_manager.get_daily_metrics(session, symbol.id)
            assert not df.empty


class TestDownloadLogOperations:
    """Tests for download log operations."""

    def test_log_download(self, db_manager):
        """Test logging a download."""
        with db_manager.get_session() as session:
            log = db_manager.log_download(
                session,
                operation="eod",
                status="success",
                symbol="AAPL.US",
                records_downloaded=100,
            )
            assert log.id is not None
            assert log.status == "success"


class TestStatistics:
    """Tests for statistics."""

    def test_get_statistics(self, db_manager):
        """Test getting database statistics."""
        with db_manager.get_session() as session:
            stats = db_manager.get_statistics(session)

            assert "exchanges" in stats
            assert "symbols" in stats
            assert "price_records" in stats
            assert isinstance(stats["exchanges"], int)


class TestGetDbManager:
    """Tests for get_db_manager function."""

    def test_returns_manager(self):
        """Test that get_db_manager returns a manager."""
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
            manager = get_db_manager()
            assert manager is not None

    def test_creates_tables(self):
        """Test that get_db_manager creates tables."""
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
            manager = get_db_manager()
            with manager.get_session() as session:
                # Should not raise
                session.query(Symbol).count()
