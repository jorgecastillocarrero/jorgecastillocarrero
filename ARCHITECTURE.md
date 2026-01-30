# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Streamlit Dashboard (web/app.py)            │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
│  │  │Posición │ │Composic.│ │Acciones │ │ Futuros │  ...   │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ PortfolioService │  │ExchangeRateServ. │  │ DailyTracking │  │
│  │ (portfolio_data) │  │(exchange_rate_s.)│  │(daily_tracking)│  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │   Validators     │  │  TechnicalCalc   │  │  AIAnalyzer   │  │
│  │  (validators.py) │  │  (technical.py)  │  │(ai_analyzer.py)│  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  DatabaseManager │  │   YahooClient    │  │  EODHDClient  │  │
│  │   (database.py)  │  │ (yahoo_client.py)│  │(eodhd_client) │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PERSISTENCE LAYER                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 SQLite / PostgreSQL                       │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │   │
│  │  │Symbols │ │ Prices │ │Holdings│ │ Trades │ │  Cash  │  │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
financial-data-project/
├── src/                      # Core application code
│   ├── config.py             # Configuration management (pydantic-settings)
│   ├── database.py           # SQLAlchemy models and DatabaseManager
│   ├── portfolio_data.py     # Portfolio calculations and service
│   ├── exchange_rate_service.py  # Centralized exchange rate handling
│   ├── daily_tracking.py     # Daily holdings and trades tracking
│   ├── validators.py         # Input validation utilities
│   ├── technical.py          # Technical indicators (RSI, MA, etc.)
│   ├── yahoo_client.py       # Yahoo Finance API client
│   ├── yahoo_downloader.py   # Batch price downloader
│   ├── eodhd_client.py       # EODHD API client
│   ├── scheduler.py          # APScheduler for automated downloads
│   └── analysis/             # Analysis modules
│       └── ai_analyzer.py    # AI-powered analysis (OpenAI)
│
├── web/                      # Web interface
│   └── app.py                # Streamlit dashboard
│
├── scripts/                  # Utility scripts
│   ├── import_ib_data.py     # Interactive Brokers data import
│   ├── import_portfolio.py   # Portfolio import utilities
│   ├── batch_download.py     # Batch price downloads
│   └── ...                   # Other maintenance scripts
│
├── tests/                    # Test suite
│   ├── conftest.py           # Shared fixtures
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_validators.py
│   ├── test_exchange_rate_service.py
│   └── ...
│
├── data/                     # Data directory (gitignored)
│   └── financial_data.db     # SQLite database
│
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── .env.example              # Environment template
└── CLAUDE.md                 # Code modification guidelines
```

## Data Flow

### Price Data Flow
```
Yahoo Finance API
       │
       ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ YahooClient  │───▶│YahooDownloader│───▶│  Database    │
└──────────────┘    └──────────────┘    │ (prices)     │
                                        └──────────────┘
```

### Portfolio Valuation Flow
```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Database   │───▶│PortfolioServ.│───▶│  Dashboard   │
│ (holdings,   │    │(calculations)│    │  (display)   │
│  prices)     │    └──────────────┘    └──────────────┘
└──────────────┘           │
                           ▼
                    ┌──────────────┐
                    │ExchangeRate  │
                    │   Service    │
                    └──────────────┘
```

## Key Design Patterns

### 1. Singleton Pattern
Services use singleton pattern for resource efficiency:
```python
_portfolio_service = None

def get_portfolio_service(db_manager=None) -> PortfolioDataService:
    global _portfolio_service
    if _portfolio_service is None:
        _portfolio_service = PortfolioDataService(db_manager)
    return _portfolio_service
```

### 2. Context Manager for Sessions
Database sessions use context managers for safe resource handling:
```python
with db.get_session() as session:
    # Operations automatically committed or rolled back
    result = session.query(Symbol).all()
```

### 3. Service Layer
Business logic is encapsulated in service classes:
- `PortfolioDataService` - Portfolio calculations
- `ExchangeRateService` - Currency conversion
- `DailyTrackingService` - Holdings tracking

### 4. Input Validation
Centralized validation in `validators.py`:
```python
code, exchange = validate_symbol("AAPL.US")
amount = validate_positive_number(100.5, "amount")
```

## Database Schema

### Core Tables
| Table | Description |
|-------|-------------|
| `exchanges` | Stock exchanges (US, MC, LSE, etc.) |
| `symbols` | Stock/ETF symbols with metadata |
| `price_history` | Daily OHLCV price data |
| `fundamentals` | Company fundamental data |

### Portfolio Tables
| Table | Description |
|-------|-------------|
| `portfolios` | Portfolio definitions |
| `portfolio_holdings` | Current holdings per portfolio |
| `daily_metrics` | Daily portfolio metrics |

### Trading Tables
| Table | Description |
|-------|-------------|
| `holding_diario` | Daily holdings snapshot |
| `stock_trades` | Stock trade history |
| `daily_cash` | Daily cash balances |
| `cash_movements` | Cash deposits/withdrawals |
| `ib_futures_trades` | Futures trades from IB |

## Security Measures

1. **SQL Injection Prevention**: All queries use parameterized statements
2. **Input Validation**: Centralized validation module
3. **Authentication**: Optional dashboard password protection
4. **Secrets Management**: Environment variables via `.env` (gitignored)
5. **Exception Handling**: Specific exceptions with logging

## Configuration

Configuration via environment variables (`.env`):
```
EODHD_API_KEY=your_key      # Market data API
DATABASE_URL=sqlite:///...   # Database connection
DASHBOARD_PASSWORD=xxx       # Optional auth
DASHBOARD_AUTH_ENABLED=true  # Enable auth
```

## Testing Strategy

- **Unit Tests**: Individual functions and classes
- **Integration Tests**: Database operations with in-memory SQLite
- **Fixtures**: Shared test fixtures in `conftest.py`
- **Mocking**: External APIs mocked for isolation

Run tests:
```bash
pytest tests/ -v --cov=src --cov-report=html
```
