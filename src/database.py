"""
Database models and connection management using SQLAlchemy.
Supports SQLite (default) and PostgreSQL.
"""

import logging
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Date,
    Boolean,
    Text,
    BigInteger,
    ForeignKey,
    Index,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    sessionmaker,
    relationship,
)

from .config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


# =============================================================================
# Models
# =============================================================================


class Exchange(Base):
    """Exchange/Market information."""

    __tablename__ = "exchanges"

    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200))
    country = Column(String(100))
    currency = Column(String(10))
    timezone = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    symbols = relationship("Symbol", back_populates="exchange")


class Symbol(Base):
    """Stock/Instrument symbol information."""

    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True)
    code = Column(String(50), nullable=False)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"))
    name = Column(String(300))
    symbol_type = Column(String(50))  # stock, etf, fund, etc.
    currency = Column(String(10))
    isin = Column(String(20))

    # Company info (fixed data)
    sector = Column(String(100))
    industry = Column(String(200))
    country = Column(String(100))
    website = Column(String(300))
    description = Column(Text)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Full symbol like AAPL.US
    @property
    def full_symbol(self) -> str:
        if self.exchange:
            return f"{self.code}.{self.exchange.code}"
        return self.code

    # Relationships
    exchange = relationship("Exchange", back_populates="symbols")
    prices = relationship("PriceHistory", back_populates="symbol")
    fundamentals = relationship("Fundamental", back_populates="symbol")
    holdings = relationship("PortfolioHolding", back_populates="symbol")
    daily_metrics = relationship("DailyMetrics", back_populates="symbol")

    __table_args__ = (
        UniqueConstraint("code", "exchange_id", name="uq_symbol_exchange"),
        Index("ix_symbol_code", "code"),
    )


class PriceHistory(Base):
    """Historical OHLCV price data."""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adjusted_close = Column(Float)
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    symbol = relationship("Symbol", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("symbol_id", "date", name="uq_price_symbol_date"),
        Index("ix_price_date", "date"),
        Index("ix_price_symbol_date", "symbol_id", "date"),
    )


class Fundamental(Base):
    """Fundamental data for symbols."""

    __tablename__ = "fundamentals"

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    data_date = Column(Date, nullable=False)

    # General Info
    sector = Column(String(100))
    industry = Column(String(200))
    description = Column(Text)
    market_cap = Column(BigInteger)
    employees = Column(Integer)

    # Valuation Metrics
    pe_ratio = Column(Float)
    forward_pe = Column(Float)
    peg_ratio = Column(Float)
    price_to_book = Column(Float)
    price_to_sales = Column(Float)
    enterprise_value = Column(BigInteger)

    # Financials
    revenue = Column(BigInteger)
    gross_profit = Column(BigInteger)
    net_income = Column(BigInteger)
    ebitda = Column(BigInteger)

    # Per Share
    eps = Column(Float)
    book_value_per_share = Column(Float)
    dividend_per_share = Column(Float)
    dividend_yield = Column(Float)

    # Margins
    profit_margin = Column(Float)
    operating_margin = Column(Float)
    gross_margin = Column(Float)

    # Growth
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)

    # Earnings History - Last 4 quarters (Q0 = most recent)
    # Quarter 0 (most recent)
    q0_eps_actual = Column(Float)
    q0_eps_estimate = Column(Float)
    q0_eps_difference = Column(Float)
    q0_eps_surprise_pct = Column(Float)
    q0_quarter_end = Column(Date)

    # Quarter 1
    q1_eps_actual = Column(Float)
    q1_eps_estimate = Column(Float)
    q1_eps_difference = Column(Float)
    q1_eps_surprise_pct = Column(Float)
    q1_quarter_end = Column(Date)

    # Quarter 2
    q2_eps_actual = Column(Float)
    q2_eps_estimate = Column(Float)
    q2_eps_difference = Column(Float)
    q2_eps_surprise_pct = Column(Float)
    q2_quarter_end = Column(Date)

    # Quarter 3 (oldest)
    q3_eps_actual = Column(Float)
    q3_eps_estimate = Column(Float)
    q3_eps_difference = Column(Float)
    q3_eps_surprise_pct = Column(Float)
    q3_quarter_end = Column(Date)

    # Raw JSON data for additional fields
    raw_data = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    symbol = relationship("Symbol", back_populates="fundamentals")

    __table_args__ = (
        UniqueConstraint("symbol_id", "data_date", name="uq_fundamental_symbol_date"),
        Index("ix_fundamental_date", "data_date"),
    )


class Portfolio(Base):
    """Investment portfolio for tracking monthly selections."""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    month = Column(Integer, nullable=False)  # 1-12
    year = Column(Integer, nullable=False)
    description = Column(Text)
    initial_capital = Column(Float, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    holdings = relationship("PortfolioHolding", back_populates="portfolio")

    __table_args__ = (
        UniqueConstraint("name", "month", "year", name="uq_portfolio_name_month_year"),
        Index("ix_portfolio_year_month", "year", "month"),
    )


class PortfolioHolding(Base):
    """Individual stock holding in a portfolio."""

    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    entry_date = Column(Date, nullable=False)
    entry_price = Column(Float)
    shares = Column(Float, default=1)
    exit_date = Column(Date)
    exit_price = Column(Float)
    is_active = Column(Boolean, default=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")
    symbol = relationship("Symbol", back_populates="holdings")

    __table_args__ = (
        Index("ix_holding_portfolio", "portfolio_id"),
        Index("ix_holding_symbol", "symbol_id"),
    )


class DailyMetrics(Base):
    """Daily calculated metrics for symbols (RSI, M200, Sharpe, etc.)."""

    __tablename__ = "daily_metrics"

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Price data
    close_price = Column(Float)

    # Technical Indicators
    rsi_14 = Column(Float)  # 14-period RSI
    ma_50 = Column(Float)   # 50-day moving average
    ma_200 = Column(Float)  # 200-day moving average
    m200 = Column(Float)    # Distance from MA200: (price - MA200) / MA200
    m50 = Column(Float)     # Distance from MA50: (price - MA50) / MA50

    # Returns
    daily_return = Column(Float)      # Daily return
    weekly_return = Column(Float)     # 5-day return
    monthly_return = Column(Float)    # 21-day return
    ytd_return = Column(Float)        # Year-to-date return

    # Volatility & Risk
    volatility_21d = Column(Float)    # 21-day volatility (annualized)
    sharpe_21d = Column(Float)        # 21-day Sharpe ratio

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    symbol = relationship("Symbol", back_populates="daily_metrics")

    __table_args__ = (
        UniqueConstraint("symbol_id", "date", name="uq_metrics_symbol_date"),
        Index("ix_metrics_date", "date"),
        Index("ix_metrics_symbol_date", "symbol_id", "date"),
    )


class IBAccount(Base):
    """Interactive Brokers account information."""

    __tablename__ = "ib_accounts"

    id = Column(Integer, primary_key=True)
    account_id = Column(String(20), unique=True, nullable=False)  # e.g., U17236599
    name = Column(String(200))
    broker = Column(String(100), default="Interactive Brokers")
    base_currency = Column(String(10), default="EUR")
    account_type = Column(String(50))  # Margin, Cash, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trades = relationship("IBTrade", back_populates="account")
    futures_trades = relationship("IBFuturesTrade", back_populates="account")


class IBTrade(Base):
    """Interactive Brokers stock/ETF trades."""

    __tablename__ = "ib_trades"

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("ib_accounts.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    trade_date = Column(DateTime, nullable=False)
    quantity = Column(Float, nullable=False)  # Positive=buy, Negative=sell
    price = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    commission = Column(Float, default=0)
    total_cost = Column(Float)  # Price * Quantity + Commission
    trade_type = Column(String(10))  # BUY, SELL
    asset_type = Column(String(20), default="Stock")  # Stock, ETF, Cash
    is_cash_equivalent = Column(Boolean, default=False)  # True for TLT
    realized_pnl = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    account = relationship("IBAccount", back_populates="trades")

    __table_args__ = (
        Index("ix_ib_trade_date", "trade_date"),
        Index("ix_ib_trade_symbol", "symbol"),
    )


class IBFuturesTrade(Base):
    """Interactive Brokers futures trades (intraday)."""

    __tablename__ = "ib_futures_trades"

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("ib_accounts.id"), nullable=False)
    symbol = Column(String(20), nullable=False)  # e.g., GCH6
    underlying = Column(String(20))  # e.g., GC (Gold)
    expiry = Column(Date)  # Contract expiration
    trade_date = Column(DateTime, nullable=False)
    quantity = Column(Integer, nullable=False)  # Positive=long, Negative=short
    price = Column(Float, nullable=False)
    multiplier = Column(Integer, default=100)
    currency = Column(String(10), default="USD")
    commission = Column(Float, default=0)
    notional_value = Column(Float)  # Price * Quantity * Multiplier
    trade_type = Column(String(10))  # OPEN, CLOSE
    is_day_trade = Column(Boolean, default=True)  # All closed same day
    realized_pnl = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    account = relationship("IBAccount", back_populates="futures_trades")

    __table_args__ = (
        Index("ix_ib_futures_date", "trade_date"),
        Index("ix_ib_futures_symbol", "symbol"),
    )


class AccountHolding(Base):
    """Current holdings by account - the actual positions held."""

    __tablename__ = "account_holdings"

    id = Column(Integer, primary_key=True)
    account_code = Column(String(20), nullable=False)  # CO3365, RCO951, LACAIXA, IB
    symbol = Column(String(20), nullable=False)  # AAPL, IAG.MC, etc.
    shares = Column(Float, nullable=False)
    entry_date = Column(Date)  # When position was opened
    entry_price = Column(Float)  # Average entry price
    currency = Column(String(10))  # USD, EUR, CAD, CHF
    is_active = Column(Boolean, default=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("account_code", "symbol", name="uq_acct_holding_account_symbol"),
        Index("ix_acct_holding_account", "account_code"),
        Index("ix_acct_holding_symbol", "symbol"),
    )


class AssetType(Base):
    """Asset type categorization for symbols."""

    __tablename__ = "asset_types"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    asset_type = Column(String(50), nullable=False)  # Mensual, Quant, Value, Alpha Picks, Oro/Mineras, Cash/Monetario
    sub_type = Column(String(50))  # Optional sub-categorization
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_asset_type", "asset_type"),
    )


class AccountCash(Base):
    """Cash positions by account and currency."""

    __tablename__ = "account_cash"

    id = Column(Integer, primary_key=True)
    account_code = Column(String(20), nullable=False)  # CO3365, RCO951, LACAIXA, IB
    currency = Column(String(10), nullable=False)  # EUR, USD, CAD, etc.
    amount = Column(Float, nullable=False)
    as_of_date = Column(Date, nullable=False)  # Date of this cash balance
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("account_code", "currency", "as_of_date", name="uq_cash_account_currency_date"),
        Index("ix_cash_account", "account_code"),
        Index("ix_cash_date", "as_of_date"),
    )


class PortfolioSnapshot(Base):
    """Portfolio valuation snapshots by account and date."""

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True)
    account_code = Column(String(20), nullable=False)  # CO3365, RCO951, LACAIXA, IB
    snapshot_date = Column(Date, nullable=False)

    # Values in EUR
    stocks_value = Column(Float, default=0)      # Value of stock holdings
    etf_value = Column(Float, default=0)         # Value of ETF holdings
    cash_eur = Column(Float, default=0)          # Cash in EUR
    cash_usd_eur = Column(Float, default=0)      # Cash USD converted to EUR
    cash_other_eur = Column(Float, default=0)    # Other currencies converted to EUR
    futures_pnl = Column(Float, default=0)       # Futures P&L in EUR
    total_value = Column(Float, nullable=False)  # Total account value in EUR

    # Exchange rates used
    eur_usd_rate = Column(Float)
    eur_cad_rate = Column(Float)
    eur_chf_rate = Column(Float)

    # Source/notes
    source = Column(String(50))  # 'statement', 'calculated', 'manual'
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("account_code", "snapshot_date", name="uq_snapshot_account_date"),
        Index("ix_snapshot_date", "snapshot_date"),
        Index("ix_snapshot_account", "account_code"),
    )


class DailyHolding(Base):
    """Daily holdings snapshot - exact positions held on each date."""

    __tablename__ = "daily_holdings"

    id = Column(Integer, primary_key=True)
    holding_date = Column(Date, nullable=False)
    account_code = Column(String(20), nullable=False)  # CO3365, RCO951, LACAIXA, IB
    symbol = Column(String(20), nullable=False)
    shares = Column(Float, nullable=False)
    entry_price = Column(Float)  # Average cost basis
    currency = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("holding_date", "account_code", "symbol", name="uq_daily_holding"),
        Index("ix_daily_holding_date", "holding_date"),
        Index("ix_daily_holding_account", "account_code"),
        Index("ix_daily_holding_symbol", "symbol"),
    )


class StockTrade(Base):
    """Stock buy/sell trades."""

    __tablename__ = "stock_trades"

    id = Column(Integer, primary_key=True)
    account_code = Column(String(20), nullable=False)
    trade_date = Column(Date, nullable=False)
    symbol = Column(String(20), nullable=False)
    trade_type = Column(String(10), nullable=False)  # BUY, SELL
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    currency = Column(String(10), default='USD')
    commission = Column(Float, default=0)
    amount_local = Column(Float)  # Total in local currency (price * shares)
    amount_eur = Column(Float)  # Total converted to EUR
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_stock_trade_date", "trade_date"),
        Index("ix_stock_trade_account", "account_code"),
        Index("ix_stock_trade_symbol", "symbol"),
    )


class DailyCash(Base):
    """Daily cash balance snapshot by account and currency."""

    __tablename__ = "daily_cash"

    id = Column(Integer, primary_key=True)
    cash_date = Column(Date, nullable=False)
    account_code = Column(String(20), nullable=False)
    currency = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    amount_eur = Column(Float)  # Converted to EUR
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("cash_date", "account_code", "currency", name="uq_daily_cash"),
        Index("ix_daily_cash_date", "cash_date"),
        Index("ix_daily_cash_account", "account_code"),
    )


class CashMovement(Base):
    """Cash movements: deposits, withdrawals, transfers, dividends."""

    __tablename__ = "cash_movements"

    id = Column(Integer, primary_key=True)
    account_code = Column(String(20), nullable=False)
    movement_date = Column(Date, nullable=False)
    movement_type = Column(String(20), nullable=False)  # DEPOSIT, WITHDRAWAL, TRANSFER_IN, TRANSFER_OUT, DIVIDEND, FX_CONVERSION
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default='EUR')
    amount_eur = Column(Float)
    counterpart_account = Column(String(20))  # For transfers
    symbol = Column(String(20))  # For dividends
    reference = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_cash_movement_date", "movement_date"),
        Index("ix_cash_movement_account", "account_code"),
        Index("ix_cash_movement_type", "movement_type"),
    )


class Trabajadores(Base):
    """Daily employee count by symbol - Registro diario de empleados."""

    __tablename__ = "trabajadores"

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    fecha = Column(Date, nullable=False)
    employees = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    symbol = relationship("Symbol")

    __table_args__ = (
        UniqueConstraint("symbol_id", "fecha", name="uq_trabajadores_symbol_fecha"),
        Index("ix_trabajadores_symbol", "symbol_id"),
        Index("ix_trabajadores_fecha", "fecha"),
    )


class AccountMovement(Base):
    """Capital movements (deposits, withdrawals, transfers) by account."""

    __tablename__ = "account_movements"

    id = Column(Integer, primary_key=True)
    account_code = Column(String(20), nullable=False)  # CO3365, RCO951, LACAIXA, IB
    movement_date = Column(Date, nullable=False)

    # Movement details
    movement_type = Column(String(20), nullable=False)  # deposit, withdrawal, transfer_in, transfer_out, dividend
    amount = Column(Float, nullable=False)  # Positive for deposits, negative for withdrawals
    currency = Column(String(10), default='EUR')
    amount_eur = Column(Float)  # Amount converted to EUR

    # For transfers between accounts
    counterpart_account = Column(String(20))  # The other account in a transfer

    # Reference/tracking
    reference = Column(String(100))  # Bank reference, transfer ID, etc.
    description = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_movement_date", "movement_date"),
        Index("ix_movement_account", "account_code"),
        Index("ix_movement_type", "movement_type"),
    )


class DownloadLog(Base):
    """Log of data download operations."""

    __tablename__ = "download_logs"

    id = Column(Integer, primary_key=True)
    operation = Column(String(50), nullable=False)  # eod, fundamentals, bulk, etc.
    symbol = Column(String(50))
    exchange = Column(String(20))
    start_date = Column(Date)
    end_date = Column(Date)
    records_downloaded = Column(Integer, default=0)
    status = Column(String(20), nullable=False)  # success, error, partial
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)

    __table_args__ = (Index("ix_download_log_operation", "operation", "started_at"),)


# =============================================================================
# Database Manager
# =============================================================================


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, database_url: str | None = None):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL. Uses settings if not provided.
        """
        settings = get_settings()
        self.database_url = database_url or settings.effective_database_url

        # Ensure data directory exists for SQLite
        if self.database_url.startswith("sqlite"):
            (settings.data_dir if settings.data_dir.is_absolute() else Path(__file__).parent.parent / settings.data_dir).mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,
        )

        # Enable foreign keys for SQLite
        if self.database_url.startswith("sqlite"):
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
            event.listen(self.engine, "connect", set_sqlite_pragma)

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")

    @contextmanager
    def get_session(self):
        """Get a database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # =========================================================================
    # Exchange Operations
    # =========================================================================

    def upsert_exchange(self, session: Session, exchange_data: dict) -> Exchange:
        """Insert or update an exchange."""
        exchange = (
            session.query(Exchange)
            .filter(Exchange.code == exchange_data["code"])
            .first()
        )

        if exchange:
            for key, value in exchange_data.items():
                if hasattr(exchange, key):
                    setattr(exchange, key, value)
        else:
            exchange = Exchange(**exchange_data)
            session.add(exchange)

        session.flush()
        return exchange

    # =========================================================================
    # Symbol Operations
    # =========================================================================

    def upsert_symbol(self, session: Session, symbol_data: dict) -> Symbol:
        """Insert or update a symbol."""
        query = session.query(Symbol).filter(Symbol.code == symbol_data["code"])

        if "exchange_id" in symbol_data:
            query = query.filter(Symbol.exchange_id == symbol_data["exchange_id"])

        symbol = query.first()

        if symbol:
            for key, value in symbol_data.items():
                if hasattr(symbol, key):
                    setattr(symbol, key, value)
        else:
            symbol = Symbol(**symbol_data)
            session.add(symbol)

        session.flush()
        return symbol

    def get_symbol_by_code(
        self, session: Session, code: str, exchange_code: str | None = None
    ) -> Symbol | None:
        """Get a symbol by its code."""
        query = session.query(Symbol).filter(Symbol.code == code)

        if exchange_code:
            query = query.join(Exchange).filter(Exchange.code == exchange_code)

        return query.first()

    # =========================================================================
    # Price History Operations
    # =========================================================================

    def bulk_insert_prices(
        self, session: Session, symbol_id: int, prices_df: pd.DataFrame
    ) -> int:
        """
        Bulk insert price history data.

        Args:
            session: Database session
            symbol_id: Symbol ID
            prices_df: DataFrame with price data

        Returns:
            Number of records inserted
        """
        if prices_df.empty:
            return 0

        # Get existing dates for this symbol
        existing_dates = set(
            row[0]
            for row in session.query(PriceHistory.date)
            .filter(PriceHistory.symbol_id == symbol_id)
            .all()
        )

        records = []
        for idx, row in prices_df.iterrows():
            date_val = idx.date() if hasattr(idx, "date") else idx

            if date_val in existing_dates:
                continue

            records.append(
                PriceHistory(
                    symbol_id=symbol_id,
                    date=date_val,
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    adjusted_close=row.get("adjusted_close"),
                    volume=row.get("volume"),
                )
            )

        if records:
            session.add_all(records)
            session.flush()

        return len(records)

    def get_price_history(
        self,
        session: Session,
        symbol_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get price history as DataFrame."""
        query = session.query(PriceHistory).filter(
            PriceHistory.symbol_id == symbol_id
        )

        if start_date:
            query = query.filter(PriceHistory.date >= start_date)
        if end_date:
            query = query.filter(PriceHistory.date <= end_date)

        query = query.order_by(PriceHistory.date)

        records = query.all()

        if not records:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "adjusted_close": r.adjusted_close,
                "volume": r.volume,
            }
            for r in records
        ]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        return df

    def get_latest_price_date(
        self, session: Session, symbol_id: int
    ) -> datetime | None:
        """Get the most recent price date for a symbol."""
        result = (
            session.query(PriceHistory.date)
            .filter(PriceHistory.symbol_id == symbol_id)
            .order_by(PriceHistory.date.desc())
            .first()
        )

        return result[0] if result else None

    # =========================================================================
    # Download Log Operations
    # =========================================================================

    def log_download(
        self,
        session: Session,
        operation: str,
        status: str,
        symbol: str | None = None,
        exchange: str | None = None,
        records_downloaded: int = 0,
        error_message: str | None = None,
        started_at: datetime | None = None,
    ) -> DownloadLog:
        """Log a download operation."""
        completed_at = datetime.utcnow()
        started_at = started_at or completed_at

        duration = (completed_at - started_at).total_seconds()

        log = DownloadLog(
            operation=operation,
            symbol=symbol,
            exchange=exchange,
            records_downloaded=records_downloaded,
            status=status,
            error_message=error_message,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

        session.add(log)
        session.flush()

        return log

    # =========================================================================
    # Portfolio Operations
    # =========================================================================

    def create_portfolio(
        self,
        session: Session,
        name: str,
        month: int,
        year: int,
        initial_capital: float = 0,
        description: str = None,
    ) -> Portfolio:
        """Create a new portfolio."""
        portfolio = Portfolio(
            name=name,
            month=month,
            year=year,
            initial_capital=initial_capital,
            description=description,
        )
        session.add(portfolio)
        session.flush()
        return portfolio

    def get_portfolio(
        self, session: Session, name: str = None, month: int = None, year: int = None
    ) -> Portfolio | None:
        """Get a portfolio by name and/or month/year."""
        query = session.query(Portfolio)
        if name:
            query = query.filter(Portfolio.name == name)
        if month:
            query = query.filter(Portfolio.month == month)
        if year:
            query = query.filter(Portfolio.year == year)
        return query.first()

    def get_portfolios(self, session: Session, year: int = None) -> list[Portfolio]:
        """Get all portfolios, optionally filtered by year."""
        query = session.query(Portfolio).order_by(Portfolio.year.desc(), Portfolio.month.desc())
        if year:
            query = query.filter(Portfolio.year == year)
        return query.all()

    def add_holding_to_portfolio(
        self,
        session: Session,
        portfolio_id: int,
        symbol_id: int,
        entry_date,
        entry_price: float = None,
        shares: float = 1,
    ) -> PortfolioHolding:
        """Add a stock to a portfolio."""
        holding = PortfolioHolding(
            portfolio_id=portfolio_id,
            symbol_id=symbol_id,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
        )
        session.add(holding)
        session.flush()
        return holding

    def get_portfolio_holdings(
        self, session: Session, portfolio_id: int, active_only: bool = True
    ) -> list[PortfolioHolding]:
        """Get all holdings for a portfolio."""
        query = session.query(PortfolioHolding).filter(
            PortfolioHolding.portfolio_id == portfolio_id
        )
        if active_only:
            query = query.filter(PortfolioHolding.is_active == True)
        return query.all()

    # =========================================================================
    # Daily Metrics Operations
    # =========================================================================

    def upsert_daily_metrics(
        self, session: Session, symbol_id: int, date, metrics: dict
    ) -> DailyMetrics:
        """Insert or update daily metrics for a symbol."""
        existing = (
            session.query(DailyMetrics)
            .filter(DailyMetrics.symbol_id == symbol_id, DailyMetrics.date == date)
            .first()
        )

        if existing:
            for key, value in metrics.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            session.flush()
            return existing
        else:
            dm = DailyMetrics(symbol_id=symbol_id, date=date, **metrics)
            session.add(dm)
            session.flush()
            return dm

    def bulk_upsert_daily_metrics(
        self, session: Session, symbol_id: int, metrics_df: pd.DataFrame
    ) -> int:
        """Bulk insert/update daily metrics from a DataFrame."""
        if metrics_df.empty:
            return 0

        count = 0
        for idx, row in metrics_df.iterrows():
            date_val = idx.date() if hasattr(idx, "date") else idx
            metrics = row.to_dict()
            self.upsert_daily_metrics(session, symbol_id, date_val, metrics)
            count += 1

        return count

    def get_daily_metrics(
        self,
        session: Session,
        symbol_id: int,
        start_date=None,
        end_date=None,
        limit: int = None,
    ) -> pd.DataFrame:
        """Get daily metrics as DataFrame."""
        query = session.query(DailyMetrics).filter(DailyMetrics.symbol_id == symbol_id)

        if start_date:
            query = query.filter(DailyMetrics.date >= start_date)
        if end_date:
            query = query.filter(DailyMetrics.date <= end_date)

        query = query.order_by(DailyMetrics.date.desc())

        if limit:
            query = query.limit(limit)

        records = query.all()

        if not records:
            return pd.DataFrame()

        data = [
            {
                "date": r.date,
                "close_price": r.close_price,
                "rsi_14": r.rsi_14,
                "ma_50": r.ma_50,
                "ma_200": r.ma_200,
                "m200": r.m200,
                "m50": r.m50,
                "daily_return": r.daily_return,
                "weekly_return": r.weekly_return,
                "monthly_return": r.monthly_return,
                "ytd_return": r.ytd_return,
                "volatility_21d": r.volatility_21d,
                "sharpe_21d": r.sharpe_21d,
            }
            for r in records
        ]

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        return df

    def get_latest_metrics_for_symbols(
        self, session: Session, symbol_ids: list[int] = None
    ) -> pd.DataFrame:
        """Get the most recent metrics for multiple symbols."""
        from sqlalchemy import func

        # Subquery to get max date per symbol
        subq = (
            session.query(
                DailyMetrics.symbol_id,
                func.max(DailyMetrics.date).label("max_date"),
            )
            .group_by(DailyMetrics.symbol_id)
        )

        if symbol_ids:
            subq = subq.filter(DailyMetrics.symbol_id.in_(symbol_ids))

        subq = subq.subquery()

        # Join to get full records
        query = (
            session.query(DailyMetrics, Symbol.code, Symbol.name)
            .join(Symbol, DailyMetrics.symbol_id == Symbol.id)
            .join(
                subq,
                (DailyMetrics.symbol_id == subq.c.symbol_id)
                & (DailyMetrics.date == subq.c.max_date),
            )
        )

        results = query.all()

        if not results:
            return pd.DataFrame()

        data = []
        for dm, code, name in results:
            data.append({
                "symbol": code,
                "name": name,
                "date": dm.date,
                "close_price": dm.close_price,
                "rsi_14": dm.rsi_14,
                "m200": dm.m200,
                "m50": dm.m50,
                "daily_return": dm.daily_return,
                "weekly_return": dm.weekly_return,
                "monthly_return": dm.monthly_return,
                "sharpe_21d": dm.sharpe_21d,
            })

        return pd.DataFrame(data)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self, session: Session) -> dict:
        """Get database statistics."""
        return {
            "exchanges": session.query(Exchange).count(),
            "symbols": session.query(Symbol).count(),
            "price_records": session.query(PriceHistory).count(),
            "fundamentals": session.query(Fundamental).count(),
            "portfolios": session.query(Portfolio).count(),
            "portfolio_holdings": session.query(PortfolioHolding).count(),
            "daily_metrics": session.query(DailyMetrics).count(),
            "account_holdings": session.query(AccountHolding).count(),
            "asset_types": session.query(AssetType).count(),
            "account_cash": session.query(AccountCash).count(),
            "portfolio_snapshots": session.query(PortfolioSnapshot).count(),
            "account_movements": session.query(AccountMovement).count(),
            "download_logs": session.query(DownloadLog).count(),
        }


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager


# CLI test functionality
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Database ===\n")

    db = get_db_manager()
    print("Database tables created successfully")

    with db.get_session() as session:
        stats = db.get_statistics(session)
        print(f"Database statistics: {stats}")

    print("\n=== Database test completed ===")
