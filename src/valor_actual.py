"""
Module for calculating portfolio positions (valor actual).

Logic:
1. Validate that >= 98% of symbols have prices for the date
2. Calculate holding_diario = yesterday + compras - ventas
3. Get prices only for exact date (no lookback)
4. Valor actual = sum(holdings * price) + cash
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Minimum percentage of symbols required to consider a date valid
MIN_PRICE_COVERAGE = 0.98  # 98%


class ValorActualCalculator:
    """Calculator for portfolio positions."""

    def __init__(self, db_manager=None):
        if db_manager is None:
            from .database import get_db_manager
            self.db = get_db_manager()
        else:
            self.db = db_manager

    # =========================================================================
    # VALIDATION: Check if date has enough price data
    # =========================================================================

    def get_total_symbols(self, session) -> int:
        """Get total number of symbols in database."""
        result = session.execute(text("SELECT COUNT(*) FROM symbols"))
        return result.fetchone()[0]

    def get_symbols_with_price(self, session, fecha: str) -> int:
        """Get count of symbols that have price data for exact date."""
        result = session.execute(text("""
            SELECT COUNT(DISTINCT symbol_id)
            FROM price_history
            WHERE DATE(date) = :fecha
        """), {'fecha': fecha})
        return result.fetchone()[0]

    def validate_price_coverage(self, session, fecha: str) -> Tuple[bool, float, int, int]:
        """
        Check if date has >= 98% price coverage.

        Returns:
            Tuple of (is_valid, coverage_pct, symbols_with_price, total_symbols)
        """
        total = self.get_total_symbols(session)
        with_price = self.get_symbols_with_price(session, fecha)
        coverage = with_price / total if total > 0 else 0
        is_valid = coverage >= MIN_PRICE_COVERAGE

        return is_valid, coverage * 100, with_price, total

    def get_valid_market_days(self, session, desde: str = None) -> list:
        """
        Get list of dates with >= 98% price coverage.

        Args:
            desde: Start date (optional, defaults to 2025-12-31)

        Returns:
            List of valid date strings
        """
        if desde is None:
            desde = '2025-12-31'

        total = self.get_total_symbols(session)
        min_count = int(total * MIN_PRICE_COVERAGE)

        result = session.execute(text("""
            SELECT DATE(date) as fecha, COUNT(DISTINCT symbol_id) as cnt
            FROM price_history
            WHERE DATE(date) >= :desde
            GROUP BY DATE(date)
            HAVING COUNT(DISTINCT symbol_id) >= :min_count
            ORDER BY fecha
        """), {'desde': desde, 'min_count': min_count})

        return [str(row[0]) for row in result.fetchall()]

    def get_last_valid_market_day(self, session) -> Optional[str]:
        """
        Get the most recent date with >= 98% price coverage.
        """
        total = self.get_total_symbols(session)
        min_count = int(total * MIN_PRICE_COVERAGE)

        result = session.execute(text("""
            SELECT DATE(date) as fecha
            FROM price_history
            GROUP BY DATE(date)
            HAVING COUNT(DISTINCT symbol_id) >= :min_count
            ORDER BY fecha DESC
            LIMIT 1
        """), {'min_count': min_count})

        row = result.fetchone()
        return str(row[0]) if row else None

    # =========================================================================
    # EXCHANGE RATES: Get rates for exact date only
    # =========================================================================

    def get_exchange_rate(self, session, symbol: str, fecha: str) -> Optional[float]:
        """Get exchange rate for exact date only."""
        result = session.execute(text("""
            SELECT ph.close
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE s.code = :symbol AND DATE(ph.date) = :fecha
            LIMIT 1
        """), {'symbol': symbol, 'fecha': fecha})
        row = result.fetchone()
        return row[0] if row else None

    def get_all_exchange_rates(self, session, fecha: str) -> dict:
        """
        Get all required exchange rates for a date.

        Returns:
            Dict with keys: eur_usd, cad_eur, chf_eur
        """
        rates = {
            'eur_usd': self.get_exchange_rate(session, 'EURUSD=X', fecha),
            'cad_eur': self.get_exchange_rate(session, 'CADEUR=X', fecha),
            'chf_eur': self.get_exchange_rate(session, 'CHFEUR=X', fecha),
        }
        return rates

    # =========================================================================
    # PRICES: Get price for exact date only (NO LOOKBACK)
    # =========================================================================

    def get_symbol_price(self, session, symbol: str, fecha: str) -> Optional[float]:
        """
        Get closing price for a symbol on EXACT date only.
        No lookback - returns None if no price for that date.
        """
        result = session.execute(text("""
            SELECT ph.close
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE s.code = :symbol AND DATE(ph.date) = :fecha
            LIMIT 1
        """), {'symbol': symbol, 'fecha': fecha})
        row = result.fetchone()
        return row[0] if row else None

    def get_source_currency(self, symbol: str) -> str:
        """Determine source currency based on symbol suffix."""
        if '.TO' in symbol:
            return 'CAD'
        elif '.MC' in symbol or '.MI' in symbol:
            return 'EUR'
        elif '.SW' in symbol:
            return 'CHF'
        else:
            return 'USD'

    # =========================================================================
    # HOLDING_DIARIO: Calculate holdings = yesterday + compras - ventas
    # =========================================================================

    def get_previous_holdings(self, session, fecha: str) -> dict:
        """
        Get holdings from the previous valid market day.

        Returns:
            Dict: {(account, symbol): {'shares': x, 'precio_entrada': y, 'currency': z, 'asset_type': t}}
        """
        # Find previous date with holdings
        result = session.execute(text("""
            SELECT MAX(fecha) FROM holding_diario WHERE fecha < :fecha
        """), {'fecha': fecha})
        row = result.fetchone()

        if not row or not row[0]:
            return {}

        prev_date = row[0]

        result = session.execute(text("""
            SELECT account_code, symbol, shares, precio_entrada, currency, asset_type
            FROM holding_diario
            WHERE fecha = :fecha
        """), {'fecha': prev_date})

        holdings = {}
        for account, symbol, shares, precio, currency, asset_type in result.fetchall():
            holdings[(account, symbol)] = {
                'shares': shares,
                'precio_entrada': precio,
                'currency': currency,
                'asset_type': asset_type
            }
        return holdings

    def get_compras_for_date(self, session, fecha: str) -> dict:
        """
        Get purchases for a date.

        Returns:
            Dict: {(account, symbol): {'shares': x, 'precio': y, 'currency': z, 'asset_type': t}}
        """
        result = session.execute(text("""
            SELECT account_code, symbol, SUM(shares) as shares,
                   AVG(precio) as precio, currency, asset_type
            FROM compras
            WHERE DATE(fecha) = :fecha
            GROUP BY account_code, symbol, currency, asset_type
        """), {'fecha': fecha})

        compras = {}
        for account, symbol, shares, precio, currency, asset_type in result.fetchall():
            compras[(account, symbol)] = {
                'shares': shares,
                'precio': precio,
                'currency': currency,
                'asset_type': asset_type
            }
        return compras

    def get_ventas_for_date(self, session, fecha: str) -> dict:
        """
        Get sales for a date.

        Returns:
            Dict: {(account, symbol): shares_sold}
        """
        result = session.execute(text("""
            SELECT account_code, symbol, SUM(shares) as shares
            FROM ventas
            WHERE DATE(fecha) = :fecha
            GROUP BY account_code, symbol
        """), {'fecha': fecha})

        ventas = {}
        for account, symbol, shares in result.fetchall():
            ventas[(account, symbol)] = shares
        return ventas

    def calculate_holding_diario(self, session, fecha: str) -> dict:
        """
        Calculate holdings for a date: yesterday + compras - ventas.

        Returns:
            Dict: {(account, symbol): {'shares': x, 'precio_entrada': y, 'currency': z, 'asset_type': t}}
        """
        # Start with previous day's holdings
        holdings = self.get_previous_holdings(session, fecha)

        # Add purchases
        compras = self.get_compras_for_date(session, fecha)
        for key, data in compras.items():
            if key in holdings:
                # Add shares, recalculate average price
                old_shares = holdings[key]['shares']
                old_precio = holdings[key]['precio_entrada'] or 0
                new_shares = data['shares']
                new_precio = data['precio']

                total_shares = old_shares + new_shares
                if total_shares > 0:
                    avg_precio = ((old_shares * old_precio) + (new_shares * new_precio)) / total_shares
                else:
                    avg_precio = new_precio

                holdings[key]['shares'] = total_shares
                holdings[key]['precio_entrada'] = avg_precio
            else:
                # New position
                holdings[key] = {
                    'shares': data['shares'],
                    'precio_entrada': data['precio'],
                    'currency': data['currency'],
                    'asset_type': data['asset_type']
                }

        # Subtract sales
        ventas = self.get_ventas_for_date(session, fecha)
        for key, shares_sold in ventas.items():
            if key in holdings:
                holdings[key]['shares'] -= shares_sold
                # Remove if no shares left
                if holdings[key]['shares'] <= 0:
                    del holdings[key]

        return holdings

    def save_holding_diario(self, session, fecha: str, holdings: dict) -> int:
        """
        Save calculated holdings to holding_diario table.

        Returns:
            Number of records saved
        """
        # Delete existing records for this date
        session.execute(text("""
            DELETE FROM holding_diario WHERE fecha = :fecha
        """), {'fecha': fecha})

        count = 0
        for (account, symbol), data in holdings.items():
            session.execute(text("""
                INSERT INTO holding_diario (fecha, account_code, symbol, shares, precio_entrada, currency, asset_type)
                VALUES (:fecha, :account, :symbol, :shares, :precio, :currency, :asset_type)
            """), {
                'fecha': fecha,
                'account': account,
                'symbol': symbol,
                'shares': data['shares'],
                'precio': data.get('precio_entrada'),
                'currency': data.get('currency', 'USD'),
                'asset_type': data.get('asset_type')
            })
            count += 1

        return count

    # =========================================================================
    # CASH_DIARIO: Calculate cash = yesterday + ventas - compras
    # =========================================================================

    def get_previous_cash(self, session, fecha: str) -> dict:
        """
        Get cash from the previous day.

        Returns:
            Dict: {(account, currency): saldo}
        """
        result = session.execute(text("""
            SELECT MAX(fecha) FROM cash_diario WHERE fecha < :fecha
        """), {'fecha': fecha})
        row = result.fetchone()

        if not row or not row[0]:
            return {}

        prev_date = row[0]

        result = session.execute(text("""
            SELECT account_code, currency, saldo
            FROM cash_diario
            WHERE fecha = :fecha
        """), {'fecha': prev_date})

        cash = {}
        for account, currency, saldo in result.fetchall():
            cash[(account, currency)] = saldo
        return cash

    def calculate_cash_diario(self, session, fecha: str) -> dict:
        """
        Calculate cash for a date: yesterday + ventas - compras.

        Returns:
            Dict: {(account, currency): saldo}
        """
        # Start with previous day's cash
        cash = self.get_previous_cash(session, fecha)

        # Add sales (cash inflow)
        result = session.execute(text("""
            SELECT account_code, currency, SUM(importe_total) as total
            FROM ventas
            WHERE DATE(fecha) = :fecha
            GROUP BY account_code, currency
        """), {'fecha': fecha})

        for account, currency, total in result.fetchall():
            key = (account, currency)
            cash[key] = cash.get(key, 0) + (total or 0)

        # Subtract purchases (cash outflow)
        result = session.execute(text("""
            SELECT account_code, currency, SUM(importe_total) as total
            FROM compras
            WHERE DATE(fecha) = :fecha
            GROUP BY account_code, currency
        """), {'fecha': fecha})

        for account, currency, total in result.fetchall():
            key = (account, currency)
            cash[key] = cash.get(key, 0) - (total or 0)

        return cash

    def save_cash_diario(self, session, fecha: str, cash: dict) -> int:
        """
        Save calculated cash to cash_diario table.

        Returns:
            Number of records saved
        """
        # Delete existing records for this date
        session.execute(text("""
            DELETE FROM cash_diario WHERE fecha = :fecha
        """), {'fecha': fecha})

        count = 0
        for (account, currency), saldo in cash.items():
            session.execute(text("""
                INSERT INTO cash_diario (fecha, account_code, currency, saldo)
                VALUES (:fecha, :account, :currency, :saldo)
            """), {
                'fecha': fecha,
                'account': account,
                'currency': currency,
                'saldo': saldo
            })
            count += 1

        return count

    # =========================================================================
    # VALOR ACTUAL: Main calculation
    # =========================================================================

    def calculate_valor_actual(self, fecha: str = None) -> dict:
        """
        Calculate valor actual for a date.

        Steps:
        1. Validate >= 98% price coverage
        2. Calculate holding_diario (yesterday + compras - ventas)
        3. Calculate cash_diario (yesterday + ventas - compras)
        4. Calculate valor = sum(holdings * price) + cash

        Args:
            fecha: Date to calculate (default: last valid market day)

        Returns:
            Dict with calculation results
        """
        results = {
            'fecha': None,
            'valid': False,
            'coverage_pct': 0,
            'holdings_count': 0,
            'holdings_eur': 0,
            'cash_eur': 0,
            'total_eur': 0,
            'by_account': {},
            'missing_prices': [],
            'error': None
        }

        with self.db.get_session() as session:
            # Determine date
            if fecha is None:
                fecha = self.get_last_valid_market_day(session)
                if not fecha:
                    results['error'] = 'No valid market day found'
                    return results

            results['fecha'] = fecha

            # Step 1: Validate price coverage
            is_valid, coverage, with_price, total = self.validate_price_coverage(session, fecha)
            results['coverage_pct'] = coverage

            if not is_valid:
                results['error'] = f'Insufficient price coverage: {coverage:.1f}% (need >= 98%)'
                return results

            results['valid'] = True

            # Step 2: Calculate and save holding_diario
            holdings = self.calculate_holding_diario(session, fecha)
            self.save_holding_diario(session, fecha, holdings)
            results['holdings_count'] = len(holdings)

            # Step 3: Calculate and save cash_diario
            cash = self.calculate_cash_diario(session, fecha)
            self.save_cash_diario(session, fecha, cash)

            # Step 4: Get exchange rates
            rates = self.get_all_exchange_rates(session, fecha)
            eur_usd = rates['eur_usd']
            cad_eur = rates['cad_eur']
            chf_eur = rates['chf_eur']

            if not eur_usd:
                results['error'] = 'Missing EUR/USD exchange rate'
                return results

            # Step 5: Calculate holdings value
            total_holdings_eur = 0
            by_account = {}

            for (account, symbol), data in holdings.items():
                price = self.get_symbol_price(session, symbol, fecha)

                if price is None:
                    results['missing_prices'].append(symbol)
                    continue

                shares = data['shares']
                value_local = shares * price

                # Convert to EUR
                currency = self.get_source_currency(symbol)
                if currency == 'EUR':
                    value_eur = value_local
                elif currency == 'USD':
                    value_eur = value_local / eur_usd
                elif currency == 'CAD':
                    value_eur = value_local * (cad_eur or 0.68)
                elif currency == 'CHF':
                    value_eur = value_local * (chf_eur or 1.08)
                else:
                    value_eur = value_local / eur_usd

                total_holdings_eur += value_eur

                if account not in by_account:
                    by_account[account] = {'holdings': 0, 'cash': 0, 'total': 0}
                by_account[account]['holdings'] += value_eur

            results['holdings_eur'] = total_holdings_eur

            # Step 6: Calculate cash value
            total_cash_eur = 0

            for (account, currency), saldo in cash.items():
                if currency == 'EUR':
                    cash_eur = saldo
                elif currency == 'USD':
                    cash_eur = saldo / eur_usd
                else:
                    cash_eur = saldo / eur_usd  # Default to USD

                total_cash_eur += cash_eur

                if account not in by_account:
                    by_account[account] = {'holdings': 0, 'cash': 0, 'total': 0}
                by_account[account]['cash'] += cash_eur

            results['cash_eur'] = total_cash_eur

            # Step 7: Calculate totals
            results['total_eur'] = total_holdings_eur + total_cash_eur

            for account in by_account:
                by_account[account]['total'] = by_account[account]['holdings'] + by_account[account]['cash']

            results['by_account'] = by_account

            # Commit changes
            session.commit()

            logger.info(f"Valor actual {fecha}: {results['total_eur']:,.0f} EUR "
                       f"(Holdings: {results['holdings_eur']:,.0f}, Cash: {results['cash_eur']:,.0f})")

        return results


# CLI for testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    calc = ValorActualCalculator()

    if len(sys.argv) > 1:
        fecha = sys.argv[1]
    else:
        fecha = None

    print("\n" + "=" * 60)
    print("CALCULO VALOR ACTUAL")
    print("=" * 60)

    results = calc.calculate_valor_actual(fecha)

    print(f"\nFecha: {results['fecha']}")
    print(f"Valido: {results['valid']}")
    print(f"Cobertura: {results['coverage_pct']:.1f}%")

    if results['error']:
        print(f"Error: {results['error']}")
    else:
        print(f"\nHoldings: {results['holdings_count']} posiciones = {results['holdings_eur']:,.0f} EUR")
        print(f"Cash: {results['cash_eur']:,.0f} EUR")
        print(f"TOTAL: {results['total_eur']:,.0f} EUR")

        print("\nPor cuenta:")
        for account, data in sorted(results['by_account'].items()):
            print(f"  {account}: {data['holdings']:,.0f} + {data['cash']:,.0f} = {data['total']:,.0f} EUR")

        if results['missing_prices']:
            print(f"\nPrecios faltantes: {', '.join(results['missing_prices'][:10])}")
