"""
Interactive Brokers Report Parser

Watches a folder for new IB activity reports (CSV) and automatically
processes them to update the database with trades, holdings, and cash.

Usage:
    python -m src.ib_report_parser              # Process new files once
    python -m src.ib_report_parser --watch      # Watch folder continuously
    python -m src.ib_report_parser --file X.csv # Process specific file
"""

import os
import csv
import logging
import shutil
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import re

logger = logging.getLogger(__name__)

# Default folder to watch for IB reports
IB_REPORTS_FOLDER = Path("data/ib_reports")
IB_PROCESSED_FOLDER = Path("data/ib_reports/processed")


class IBReportParser:
    """Parser for Interactive Brokers activity reports."""

    def __init__(self, db_manager=None):
        if db_manager is None:
            from .database import get_db_manager
            self.db = get_db_manager()
        else:
            self.db = db_manager

    def parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse IB date/time format."""
        if not date_str or date_str == '--':
            return None

        # Try different formats
        formats = [
            "%Y-%m-%d, %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None

    def parse_number(self, value: str) -> Optional[float]:
        """Parse number from IB format (handles commas and negatives)."""
        if not value or value == '--' or value == '-':
            return None

        # Remove currency symbols and spaces
        clean = value.replace('$', '').replace('€', '').replace(' ', '')
        # Handle European format (1.234,56) vs US format (1,234.56)
        if ',' in clean and '.' in clean:
            # Determine format by position
            if clean.rfind(',') > clean.rfind('.'):
                # European: 1.234,56
                clean = clean.replace('.', '').replace(',', '.')
            else:
                # US: 1,234.56
                clean = clean.replace(',', '')
        elif ',' in clean and '.' not in clean:
            # Could be European decimal or US thousands
            # If only one comma and 2-3 digits after, treat as decimal
            parts = clean.split(',')
            if len(parts) == 2 and len(parts[1]) <= 3:
                clean = clean.replace(',', '.')
            else:
                clean = clean.replace(',', '')

        try:
            return float(clean)
        except ValueError:
            return None

    def get_account_id(self, session, account_number: str) -> int:
        """Get or create IB account."""
        from sqlalchemy import text

        result = session.execute(text(
            "SELECT id FROM ib_accounts WHERE account_id = :acc"
        ), {'acc': account_number})
        row = result.fetchone()

        if row:
            return row[0]

        # Create new account
        session.execute(text("""
            INSERT INTO ib_accounts (account_id, name, broker, base_currency, is_active, created_at)
            VALUES (:acc, :acc, 'Interactive Brokers', 'EUR', true, NOW())
        """), {'acc': account_number})
        session.commit()

        result = session.execute(text(
            "SELECT id FROM ib_accounts WHERE account_id = :acc"
        ), {'acc': account_number})
        return result.fetchone()[0]

    def parse_report(self, filepath: Path) -> dict:
        """
        Parse an IB activity report CSV.

        Returns dict with:
            - account_number: str
            - period_start: date
            - period_end: date
            - etf_trades: list of trades
            - futures_trades: list of trades
            - open_positions: list of positions
            - cash_balances: dict of currency -> amount
        """
        result = {
            'account_number': None,
            'period_start': None,
            'period_end': None,
            'etf_trades': [],
            'futures_trades': [],
            'open_positions': [],
            'cash_balances': {},
            'filename': filepath.name,
        }

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)

        current_section = None

        for row in rows:
            if len(row) < 2:
                continue

            section = row[0]
            data_type = row[1]

            # Account info
            if section == 'Información sobre la cuenta' and data_type == 'Data':
                if len(row) >= 4:
                    if row[2] == 'Cuenta':
                        result['account_number'] = row[3]

            # Period
            if section == 'Statement' and data_type == 'Data':
                if len(row) >= 4 and row[2] == 'Period':
                    period = row[3]
                    # Parse "Enero 30, 2026 - Febrero 6, 2026"
                    match = re.search(r'(\w+)\s+(\d+),\s+(\d{4})\s*-\s*(\w+)\s+(\d+),\s+(\d{4})', period)
                    if match:
                        # Spanish month names
                        months = {
                            'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4,
                            'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
                            'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
                        }
                        try:
                            start_month = months.get(match.group(1), 1)
                            start_day = int(match.group(2))
                            start_year = int(match.group(3))
                            end_month = months.get(match.group(4), 1)
                            end_day = int(match.group(5))
                            end_year = int(match.group(6))
                            result['period_start'] = date(start_year, start_month, start_day)
                            result['period_end'] = date(end_year, end_month, end_day)
                        except:
                            pass

            # Trades (Operaciones)
            if section == 'Operaciones' and data_type == 'Data':
                if len(row) >= 10 and row[2] == 'Order':
                    asset_type = row[3]  # Acciones or Futuros
                    currency = row[4]
                    symbol = row[5]
                    trade_date = self.parse_datetime(row[6])
                    quantity = self.parse_number(row[7])
                    price = self.parse_number(row[8])

                    if trade_date and quantity is not None and price is not None:
                        trade = {
                            'symbol': symbol,
                            'trade_date': trade_date,
                            'quantity': quantity,
                            'price': price,
                            'currency': currency,
                            'trade_type': 'BUY' if quantity > 0 else 'SELL',
                        }

                        if asset_type == 'Acciones':
                            trade['asset_type'] = 'ETF'
                            result['etf_trades'].append(trade)
                        elif asset_type == 'Futuros':
                            # Extract underlying and expiry from symbol
                            # ESH6 -> ES, March 2026
                            underlying = symbol[:2] if len(symbol) >= 2 else symbol
                            # Parse expiry from symbol (H=March, M=June, U=Sep, Z=Dec)
                            month_codes = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                                          'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
                            if len(symbol) >= 3:
                                month_code = symbol[2]
                                year_code = symbol[3] if len(symbol) >= 4 else '6'
                                month = month_codes.get(month_code, 3)
                                year = 2020 + int(year_code)
                                # Third Friday of the month (approximate)
                                expiry = date(year, month, 20)
                            else:
                                expiry = date(2026, 3, 20)

                            trade['underlying'] = underlying
                            trade['expiry'] = expiry
                            trade['multiplier'] = 50 if underlying == 'ES' else 20 if underlying == 'NQ' else 1
                            trade['trade_type'] = 'OPEN' if abs(quantity) == quantity else 'CLOSE'
                            # For futures, negative = short open, positive after short = close
                            result['futures_trades'].append(trade)

            # Open positions
            if section == 'Posiciones abiertas' and data_type == 'Data':
                if len(row) >= 12 and row[2] == 'Summary':
                    asset_type = row[3]
                    currency = row[4]
                    symbol = row[5]
                    quantity = self.parse_number(row[6])
                    cost_price = self.parse_number(row[9])

                    if quantity and cost_price:
                        result['open_positions'].append({
                            'symbol': symbol,
                            'quantity': quantity,
                            'cost_price': cost_price,
                            'currency': currency,
                            'asset_type': 'ETF' if asset_type == 'Acciones' else asset_type,
                        })

            # Cash balances (Saldos en fórex)
            if section == 'Saldos en fórex' and data_type == 'Data':
                if len(row) >= 6:
                    currency = row[3]
                    if currency in ('EUR', 'USD', 'GBP', 'CHF', 'CAD'):
                        quantity = self.parse_number(row[5])
                        if quantity is not None:
                            result['cash_balances'][currency] = quantity

        return result

    def process_report(self, filepath: Path) -> dict:
        """
        Process an IB report and update the database.

        Returns dict with processing results.
        """
        from sqlalchemy import text

        logger.info(f"Processing IB report: {filepath.name}")

        # Parse the report
        data = self.parse_report(filepath)

        if not data['account_number']:
            return {'success': False, 'error': 'Could not find account number'}

        results = {
            'success': True,
            'account': data['account_number'],
            'period': f"{data['period_start']} to {data['period_end']}",
            'etf_trades_added': 0,
            'futures_trades_added': 0,
            'holdings_updated': 0,
            'cash_updated': 0,
        }

        with self.db.get_session() as session:
            account_id = self.get_account_id(session, data['account_number'])

            # Fix sequences first
            session.execute(text("SELECT setval('ib_trades_id_seq', COALESCE((SELECT MAX(id) FROM ib_trades), 0) + 1)"))
            session.execute(text("SELECT setval('ib_futures_trades_id_seq', COALESCE((SELECT MAX(id) FROM ib_futures_trades), 0) + 1)"))

            # Insert ETF trades
            for trade in data['etf_trades']:
                # Check if exists
                exists = session.execute(text("""
                    SELECT id FROM ib_trades
                    WHERE symbol = :symbol AND trade_date = :trade_date
                """), {'symbol': trade['symbol'], 'trade_date': trade['trade_date']}).fetchone()

                if not exists:
                    session.execute(text("""
                        INSERT INTO ib_trades
                        (account_id, symbol, trade_date, quantity, price, currency, trade_type, asset_type, created_at)
                        VALUES (:acc_id, :symbol, :trade_date, :quantity, :price, :currency, :trade_type, :asset_type, NOW())
                    """), {
                        'acc_id': account_id,
                        'symbol': trade['symbol'],
                        'trade_date': trade['trade_date'],
                        'quantity': trade['quantity'],
                        'price': trade['price'],
                        'currency': trade['currency'],
                        'trade_type': trade['trade_type'],
                        'asset_type': trade['asset_type'],
                    })
                    results['etf_trades_added'] += 1

            # Insert futures trades
            for trade in data['futures_trades']:
                exists = session.execute(text("""
                    SELECT id FROM ib_futures_trades
                    WHERE symbol = :symbol AND trade_date = :trade_date
                """), {'symbol': trade['symbol'], 'trade_date': trade['trade_date']}).fetchone()

                if not exists:
                    session.execute(text("""
                        INSERT INTO ib_futures_trades
                        (account_id, symbol, underlying, expiry, trade_date, quantity, price, multiplier, currency, trade_type, created_at)
                        VALUES (:acc_id, :symbol, :underlying, :expiry, :trade_date, :quantity, :price, :multiplier, :currency, :trade_type, NOW())
                    """), {
                        'acc_id': account_id,
                        'symbol': trade['symbol'],
                        'underlying': trade['underlying'],
                        'expiry': trade['expiry'],
                        'trade_date': trade['trade_date'],
                        'quantity': trade['quantity'],
                        'price': trade['price'],
                        'multiplier': trade['multiplier'],
                        'currency': trade['currency'],
                        'trade_type': trade['trade_type'],
                    })
                    results['futures_trades_added'] += 1

            # Update holdings for period end date
            if data['open_positions'] and data['period_end']:
                end_date = data['period_end']

                # Delete existing holdings for this date
                session.execute(text("""
                    DELETE FROM holding_diario
                    WHERE account_code = 'IB' AND DATE(fecha) = :fecha
                """), {'fecha': end_date})

                for pos in data['open_positions']:
                    session.execute(text("""
                        INSERT INTO holding_diario
                        (fecha, account_code, symbol, shares, precio_entrada, currency, asset_type)
                        VALUES (:fecha, 'IB', :symbol, :shares, :precio, :currency, :asset_type)
                    """), {
                        'fecha': end_date,
                        'symbol': pos['symbol'],
                        'shares': pos['quantity'],
                        'precio': pos['cost_price'],
                        'currency': pos['currency'],
                        'asset_type': pos['asset_type'],
                    })
                    results['holdings_updated'] += 1

            # Update cash balances
            if data['cash_balances'] and data['period_end']:
                end_date = data['period_end']

                session.execute(text("""
                    DELETE FROM cash_diario
                    WHERE account_code = 'IB' AND DATE(fecha) = :fecha
                """), {'fecha': end_date})

                for currency, amount in data['cash_balances'].items():
                    session.execute(text("""
                        INSERT INTO cash_diario (fecha, account_code, currency, saldo)
                        VALUES (:fecha, 'IB', :currency, :saldo)
                    """), {
                        'fecha': end_date,
                        'currency': currency,
                        'saldo': amount,
                    })
                    results['cash_updated'] += 1

            session.commit()

        logger.info(f"Processed {filepath.name}: {results['etf_trades_added']} ETF trades, "
                   f"{results['futures_trades_added']} futures trades, "
                   f"{results['holdings_updated']} holdings, {results['cash_updated']} cash entries")

        return results

    def process_folder(self, folder: Path = None) -> list:
        """
        Process all unprocessed CSV files in the folder.

        Returns list of processing results.
        """
        if folder is None:
            folder = IB_REPORTS_FOLDER

        folder = Path(folder)
        processed_folder = folder / "processed"

        # Create folders if needed
        folder.mkdir(parents=True, exist_ok=True)
        processed_folder.mkdir(parents=True, exist_ok=True)

        results = []

        # Find CSV files
        for filepath in folder.glob("*.csv"):
            if filepath.is_file():
                try:
                    result = self.process_report(filepath)
                    results.append(result)

                    if result['success']:
                        # Move to processed folder
                        dest = processed_folder / filepath.name
                        shutil.move(str(filepath), str(dest))
                        logger.info(f"Moved {filepath.name} to processed/")

                except Exception as e:
                    logger.error(f"Error processing {filepath.name}: {e}")
                    results.append({
                        'success': False,
                        'filename': filepath.name,
                        'error': str(e),
                    })

        return results


def process_ib_reports(folder: Path = None) -> list:
    """Process all IB reports in folder. Called by scheduler."""
    parser = IBReportParser()
    return parser.process_folder(folder)


if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = IBReportParser()

    if len(sys.argv) > 1:
        if sys.argv[1] == '--watch':
            print(f"\nWatching folder: {IB_REPORTS_FOLDER}")
            print("Press Ctrl+C to stop\n")

            while True:
                results = parser.process_folder()
                if results:
                    for r in results:
                        if r['success']:
                            print(f"Processed: {r.get('account', 'unknown')} - "
                                  f"{r['etf_trades_added']} ETF, {r['futures_trades_added']} futures")
                        else:
                            print(f"Error: {r.get('error', 'unknown')}")
                time.sleep(60)  # Check every minute

        elif sys.argv[1] == '--file' and len(sys.argv) > 2:
            filepath = Path(sys.argv[2])
            if filepath.exists():
                result = parser.process_report(filepath)
                print(f"\nResult: {result}")
            else:
                print(f"File not found: {filepath}")
        else:
            print(__doc__)
    else:
        # Process once
        print(f"\nProcessing IB reports in: {IB_REPORTS_FOLDER}")
        results = parser.process_folder()

        if not results:
            print("No new reports found")
        else:
            for r in results:
                if r['success']:
                    print(f"  {r.get('account', '?')}: {r['etf_trades_added']} ETF trades, "
                          f"{r['futures_trades_added']} futures trades")
                else:
                    print(f"  Error: {r.get('error', 'unknown')}")
