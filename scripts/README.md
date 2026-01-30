# Utility Scripts

This directory contains utility scripts for data management and maintenance.

## Data Import Scripts

| Script | Description |
|--------|-------------|
| `import_ib_data.py` | Import trades and positions from Interactive Brokers |
| `import_portfolio.py` | Import portfolio data from various sources |
| `read_broker_reports.py` | Parse broker report files |
| `read_operations.py` | Read and process trade operations |

## Price Data Scripts

| Script | Description |
|--------|-------------|
| `batch_download.py` | Batch download price data for multiple symbols |
| `download_full_history.py` | Download complete price history for symbols |
| `fix_prices.py` | Fix or update price data issues |
| `fix_missing_symbols.py` | Add missing symbols to database |

## Analysis Scripts

| Script | Description |
|--------|-------------|
| `calculate_initial_correct.py` | Calculate initial portfolio values |
| `check_composition_v2.py` | Verify portfolio composition |
| `check_snapshots.py` | Validate portfolio snapshots |
| `check_spy.py` | Check SPY benchmark data |
| `analyze_r4_transfers.py` | Analyze R4 account transfers |

## Display Scripts

| Script | Description |
|--------|-------------|
| `show_all_days_by_type.py` | Show portfolio values by asset type |
| `show_daily_returns.py` | Display daily return calculations |
| `show_returns_by_type.py` | Show returns grouped by asset type |
| `show_variation_by_type.py` | Display value variations by type |

## Usage

Run scripts from project root:
```bash
python scripts/batch_download.py
python scripts/import_ib_data.py
```

Or with module syntax:
```bash
python -m scripts.batch_download
```
