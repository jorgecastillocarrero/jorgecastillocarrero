"""
Shared pytest fixtures for financial-data-project tests.
"""

import os
import pytest
import tempfile
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test environment before importing modules
os.environ["EODHD_API_KEY"] = "test_api_key"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from src.database import Base, DatabaseManager, Symbol, Exchange, PriceHistory


@pytest.fixture
def temp_db():
    """Create a temporary in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def db_manager():
    """Create a DatabaseManager with in-memory database."""
    with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        manager = DatabaseManager("sqlite:///:memory:")
        manager.create_tables()
        yield manager


@pytest.fixture
def sample_exchange(temp_db):
    """Create a sample exchange."""
    exchange = Exchange(
        code="US",
        name="US Stock Exchange",
        country="United States",
        currency="USD",
        timezone="America/New_York",
    )
    temp_db.add(exchange)
    temp_db.commit()
    return exchange


@pytest.fixture
def sample_symbol(temp_db, sample_exchange):
    """Create a sample symbol."""
    symbol = Symbol(
        code="AAPL",
        exchange_id=sample_exchange.id,
        name="Apple Inc.",
        symbol_type="stock",
        currency="USD",
    )
    temp_db.add(symbol)
    temp_db.commit()
    return symbol


@pytest.fixture
def sample_prices_df():
    """Create sample price DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=250, freq="B")
    data = {
        "open": [100 + i * 0.1 for i in range(250)],
        "high": [102 + i * 0.1 for i in range(250)],
        "low": [98 + i * 0.1 for i in range(250)],
        "close": [101 + i * 0.1 for i in range(250)],
        "adjusted_close": [101 + i * 0.1 for i in range(250)],
        "volume": [1000000 + i * 1000 for i in range(250)],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_prices_series():
    """Create sample price Series for technical calculations."""
    dates = pd.date_range(start="2024-01-01", periods=250, freq="B")
    prices = [100 + i * 0.1 + (i % 10) * 0.5 for i in range(250)]
    return pd.Series(prices, index=dates)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client for API testing."""
    with patch("httpx.Client") as mock:
        client_instance = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=client_instance)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield client_instance


@pytest.fixture
def mock_yfinance():
    """Mock yfinance module."""
    with patch("yfinance.Ticker") as mock_ticker:
        ticker_instance = MagicMock()
        mock_ticker.return_value = ticker_instance
        yield ticker_instance


@pytest.fixture
def sample_holdings_data():
    """Sample holdings data for daily tracking tests."""
    return [
        {"account_code": "IB", "symbol": "AAPL", "shares": 100, "currency": "USD", "entry_price": 150.0},
        {"account_code": "IB", "symbol": "MSFT", "shares": 50, "currency": "USD", "entry_price": 300.0},
        {"account_code": "CO3365", "symbol": "IAG.MC", "shares": 200, "currency": "EUR", "entry_price": 2.5},
    ]


@pytest.fixture
def sample_cash_data():
    """Sample cash data for daily tracking tests."""
    return [
        {"account_code": "IB", "currency": "USD", "amount": 5000.0},
        {"account_code": "IB", "currency": "EUR", "amount": 1000.0},
        {"account_code": "CO3365", "currency": "EUR", "amount": 2500.0},
    ]
