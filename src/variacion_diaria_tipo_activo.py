"""
Module for calculating daily variation by asset type (estrategia).

Logic:
1. Validate >= 98% price coverage
2. For each strategy: Position = Yesterday + Purchases - Sales
3. Cash = Yesterday + Sales - Purchases + FuturesP&L + Deposits - Withdrawals
"""

import logging
from datetime import date, datetime
from typing import Optional, Tuple
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Minimum percentage of symbols required to consider a date valid
MIN_PRICE_COVERAGE = 0.97  # 97%

# Strategy order for display
STRATEGY_ORDER = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'ETFs', 'Cash']


class VariacionDiariaTipoActivo:
    """Calculator for daily variation by asset type."""

    def __init__(self, db_manager=None):
        if db_manager is None:
            from .database import get_db_manager
            self.db = get_db_manager()
        else:
            self.db = db_manager

    # =========================================================================
    # VALIDATION: Check if date has enough price data (reused from valor_actual)
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

    def validate_price_coverage(self, session, fecha: str) -> Tuple[bool, float]:
        """Check if date has >= 98% price coverage."""
        total = self.get_total_symbols(session)
        with_price = self.get_symbols_with_price(session, fecha)
        coverage = with_price / total if total > 0 else 0
        return coverage >= MIN_PRICE_COVERAGE, coverage * 100

    def get_last_valid_market_day(self, session) -> Optional[str]:
        """Get the most recent date with >= 98% price coverage."""
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

    def get_previous_valid_market_day(self, session, fecha: str) -> Optional[str]:
        """Get the previous valid market day before the given date."""
        total = self.get_total_symbols(session)
        min_count = int(total * MIN_PRICE_COVERAGE)

        result = session.execute(text("""
            SELECT DATE(date) as fecha
            FROM price_history
            WHERE DATE(date) < :fecha
            GROUP BY DATE(date)
            HAVING COUNT(DISTINCT symbol_id) >= :min_count
            ORDER BY fecha DESC
            LIMIT 1
        """), {'fecha': fecha, 'min_count': min_count})

        row = result.fetchone()
        return str(row[0]) if row else None

    # =========================================================================
    # ASSET TYPE MAPPING
    # =========================================================================

    def get_asset_type_map(self, session) -> dict:
        """Get mapping of symbol -> asset_type from asset_types table."""
        result = session.execute(text('SELECT symbol, asset_type FROM asset_types'))
        return {r[0]: r[1] for r in result.fetchall()}

    def get_symbol_asset_type(self, asset_map: dict, account: str, symbol: str) -> str:
        """Get asset type for a symbol, checking account:symbol first, then symbol."""
        key = f'{account}:{symbol}'
        return asset_map.get(key) or asset_map.get(symbol) or 'Otros'

    # =========================================================================
    # EXCHANGE RATES
    # =========================================================================

    def get_exchange_rate(self, session, symbol: str, fecha: str) -> Optional[float]:
        """Get exchange rate for exact date."""
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
        """Get all required exchange rates for a date."""
        return {
            'eur_usd': self.get_exchange_rate(session, 'EURUSD=X', fecha),
            'cad_eur': self.get_exchange_rate(session, 'CADEUR=X', fecha),
            'chf_eur': self.get_exchange_rate(session, 'CHFEUR=X', fecha),
        }

    # =========================================================================
    # PRICE
    # =========================================================================

    def get_symbol_price(self, session, symbol: str, fecha: str) -> Optional[float]:
        """Get closing price for a symbol on exact date only."""
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
    # HOLDINGS BY STRATEGY
    # =========================================================================

    def get_holdings_for_date(self, session, fecha: str) -> list:
        """Get all holdings for a date from holding_diario."""
        result = session.execute(text("""
            SELECT account_code, symbol, shares, currency
            FROM holding_diario
            WHERE fecha = :fecha
        """), {'fecha': fecha})
        return result.fetchall()

    def get_compras_for_date(self, session, fecha: str) -> list:
        """Get all purchases for a date."""
        result = session.execute(text("""
            SELECT account_code, symbol, SUM(shares) as shares, currency
            FROM compras
            WHERE DATE(fecha) = :fecha
            GROUP BY account_code, symbol, currency
        """), {'fecha': fecha})
        return result.fetchall()

    def get_ventas_for_date(self, session, fecha: str) -> list:
        """Get all sales for a date."""
        result = session.execute(text("""
            SELECT account_code, symbol, SUM(shares) as shares, currency
            FROM ventas
            WHERE DATE(fecha) = :fecha
            GROUP BY account_code, symbol, currency
        """), {'fecha': fecha})
        return result.fetchall()

    def calculate_position_by_strategy(self, session, fecha: str, rates: dict, asset_map: dict) -> dict:
        """
        Calculate position value by strategy for a date.
        Position = Holdings value (shares * price) grouped by asset_type.

        Returns:
            Dict: {strategy: value_eur}
        """
        eur_usd = rates['eur_usd'] or 1.0
        cad_eur = rates['cad_eur'] or 0.68
        chf_eur = rates['chf_eur'] or 1.08

        holdings = self.get_holdings_for_date(session, fecha)

        by_strategy = {}

        for account, symbol, shares, currency in holdings:
            strategy = self.get_symbol_asset_type(asset_map, account, symbol)
            price = self.get_symbol_price(session, symbol, fecha)

            if price is None:
                continue

            value_local = shares * price

            # Convert to EUR
            src_currency = self.get_source_currency(symbol)
            if src_currency == 'EUR':
                value_eur = value_local
            elif src_currency == 'USD':
                value_eur = value_local / eur_usd
            elif src_currency == 'CAD':
                value_eur = value_local * cad_eur
            elif src_currency == 'CHF':
                value_eur = value_local * chf_eur
            else:
                value_eur = value_local / eur_usd

            if strategy not in by_strategy:
                by_strategy[strategy] = 0
            by_strategy[strategy] += value_eur

        return by_strategy

    # =========================================================================
    # CASH CALCULATION
    # =========================================================================

    def get_cash_for_date(self, session, fecha: str) -> dict:
        """Get cash by account and currency for a date."""
        result = session.execute(text("""
            SELECT account_code, currency, saldo
            FROM cash_diario
            WHERE fecha = :fecha
        """), {'fecha': fecha})

        cash = {}
        for account, currency, saldo in result.fetchall():
            if account not in cash:
                cash[account] = {}
            cash[account][currency] = saldo
        return cash

    def get_futures_pnl_for_date(self, session, fecha: str) -> float:
        """Get realized P&L from futures for a date."""
        result = session.execute(text("""
            SELECT COALESCE(SUM(realized_pnl), 0)
            FROM ib_futures_trades
            WHERE DATE(trade_date) = :fecha AND realized_pnl IS NOT NULL
        """), {'fecha': fecha})
        return result.fetchone()[0] or 0

    def get_deposits_withdrawals_for_date(self, session, fecha: str) -> Tuple[float, float]:
        """
        Get deposits and withdrawals for a date from movimientos_cash.

        Returns:
            Tuple of (deposits_eur, withdrawals_eur)
        """
        result = session.execute(text("""
            SELECT tipo, amount, currency
            FROM movimientos_cash
            WHERE DATE(fecha) = :fecha
        """), {'fecha': fecha})

        deposits = 0
        withdrawals = 0

        for tipo, amount, currency in result.fetchall():
            # Positive amount = deposit, negative = withdrawal
            if amount > 0:
                deposits += amount
            else:
                withdrawals += abs(amount)

        return deposits, withdrawals

    def calculate_total_cash_eur(self, session, fecha: str, rates: dict) -> float:
        """Calculate total cash in EUR for a date."""
        eur_usd = rates['eur_usd'] or 1.0

        cash_data = self.get_cash_for_date(session, fecha)
        total_eur = 0

        for account, currencies in cash_data.items():
            for currency, saldo in currencies.items():
                if currency == 'EUR':
                    total_eur += saldo
                elif currency == 'USD':
                    total_eur += saldo / eur_usd

        return total_eur

    # =========================================================================
    # MAIN CALCULATION: Daily Variation by Strategy
    # =========================================================================

    def calculate_variacion_diaria(self, fecha: str = None) -> dict:
        """
        Calculate daily variation by asset type (strategy).

        Steps:
        1. Validate >= 98% price coverage for both dates
        2. Calculate position by strategy for both dates
        3. Calculate cash for both dates
        4. Calculate variation (diff and %)

        Args:
            fecha: Target date (default: last valid market day)

        Returns:
            Dict with variation results
        """
        results = {
            'fecha_actual': None,
            'fecha_anterior': None,
            'valid': False,
            'coverage_actual': 0,
            'coverage_anterior': 0,
            'by_strategy': {},
            'total_actual': 0,
            'total_anterior': 0,
            'total_diff': 0,
            'total_pct': 0,
            'error': None
        }

        with self.db.get_session() as session:
            # Determine dates
            if fecha is None:
                fecha = self.get_last_valid_market_day(session)
                if not fecha:
                    results['error'] = 'No valid market day found'
                    return results

            fecha_anterior = self.get_previous_valid_market_day(session, fecha)
            if not fecha_anterior:
                results['error'] = 'No previous valid market day found'
                return results

            results['fecha_actual'] = fecha
            results['fecha_anterior'] = fecha_anterior

            # Validate price coverage for both dates
            is_valid_actual, coverage_actual = self.validate_price_coverage(session, fecha)
            is_valid_anterior, coverage_anterior = self.validate_price_coverage(session, fecha_anterior)

            results['coverage_actual'] = coverage_actual
            results['coverage_anterior'] = coverage_anterior

            if not is_valid_actual:
                results['error'] = f'Insufficient coverage for {fecha}: {coverage_actual:.1f}%'
                return results

            if not is_valid_anterior:
                results['error'] = f'Insufficient coverage for {fecha_anterior}: {coverage_anterior:.1f}%'
                return results

            results['valid'] = True

            # Get asset type mapping
            asset_map = self.get_asset_type_map(session)

            # Get exchange rates for both dates
            rates_actual = self.get_all_exchange_rates(session, fecha)
            rates_anterior = self.get_all_exchange_rates(session, fecha_anterior)

            # Calculate positions by strategy for both dates
            positions_actual = self.calculate_position_by_strategy(session, fecha, rates_actual, asset_map)
            positions_anterior = self.calculate_position_by_strategy(session, fecha_anterior, rates_anterior, asset_map)

            # Calculate cash for both dates
            cash_actual = self.calculate_total_cash_eur(session, fecha, rates_actual)
            cash_anterior = self.calculate_total_cash_eur(session, fecha_anterior, rates_anterior)

            # Add cash to positions
            positions_actual['Cash'] = cash_actual
            positions_anterior['Cash'] = cash_anterior

            # Calculate variation by strategy
            all_strategies = set(positions_actual.keys()) | set(positions_anterior.keys())

            # Sort strategies
            strategies_sorted = [s for s in STRATEGY_ORDER if s in all_strategies]
            strategies_sorted += [s for s in sorted(all_strategies) if s not in STRATEGY_ORDER]

            total_actual = 0
            total_anterior = 0

            for strategy in strategies_sorted:
                val_actual = positions_actual.get(strategy, 0)
                val_anterior = positions_anterior.get(strategy, 0)
                diff = val_actual - val_anterior
                pct = (diff / val_anterior * 100) if val_anterior > 0 else 0

                results['by_strategy'][strategy] = {
                    'actual': val_actual,
                    'anterior': val_anterior,
                    'diff': diff,
                    'pct': pct
                }

                total_actual += val_actual
                total_anterior += val_anterior

            results['total_actual'] = total_actual
            results['total_anterior'] = total_anterior
            results['total_diff'] = total_actual - total_anterior
            results['total_pct'] = (results['total_diff'] / total_anterior * 100) if total_anterior > 0 else 0

            logger.info(f"Variacion {fecha_anterior} -> {fecha}: "
                       f"{results['total_anterior']:,.0f} -> {results['total_actual']:,.0f} EUR "
                       f"({results['total_diff']:+,.0f}, {results['total_pct']:+.2f}%)")

        return results


# CLI for testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    calc = VariacionDiariaTipoActivo()

    if len(sys.argv) > 1:
        fecha = sys.argv[1]
    else:
        fecha = None

    print("\n" + "=" * 70)
    print("VARIACION DIARIA POR TIPO DE ACTIVO")
    print("=" * 70)

    results = calc.calculate_variacion_diaria(fecha)

    print(f"\nFecha anterior: {results['fecha_anterior']} (cobertura: {results['coverage_anterior']:.1f}%)")
    print(f"Fecha actual:   {results['fecha_actual']} (cobertura: {results['coverage_actual']:.1f}%)")
    print(f"Valido: {results['valid']}")

    if results['error']:
        print(f"Error: {results['error']}")
    else:
        print(f"\n{'Estrategia':<15} {'Anterior':>15} {'Actual':>15} {'Diferencia':>15} {'Var %':>10}")
        print("-" * 70)

        for strategy, data in results['by_strategy'].items():
            print(f"{strategy:<15} {data['anterior']:>15,.0f} {data['actual']:>15,.0f} "
                  f"{data['diff']:>+15,.0f} {data['pct']:>+9.2f}%")

        print("-" * 70)
        print(f"{'TOTAL':<15} {results['total_anterior']:>15,.0f} {results['total_actual']:>15,.0f} "
              f"{results['total_diff']:>+15,.0f} {results['total_pct']:>+9.2f}%")
