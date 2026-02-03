"""
Module for calculating and updating the posicion table.
Reads from holding_diario and cash_diario, gets current prices from price_history,
and updates/inserts into posicion.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Accounts to process
ACCOUNTS = ['CO3365', 'RCO951', 'LACAIXA', 'IB']


class PosicionCalculator:
    """Calculator for portfolio positions."""

    def __init__(self, db_manager=None):
        if db_manager is None:
            from .database import get_db_manager
            self.db = get_db_manager()
        else:
            self.db = db_manager

    def get_eur_usd_rate(self, session, fecha: str) -> Optional[float]:
        """Get EUR/USD exchange rate for a date. Returns None if not found."""
        result = session.execute(text("""
            SELECT ph.close
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE s.code = 'EURUSD=X' AND DATE(ph.date) <= :fecha
            ORDER BY ph.date DESC
            LIMIT 1
        """), {'fecha': fecha})
        row = result.fetchone()
        if not row:
            logger.warning(f"No EUR/USD rate found for {fecha}")
        return row[0] if row else None

    def get_cad_eur_rate(self, session, fecha: str) -> Optional[float]:
        """Get CAD/EUR exchange rate for a date. Returns None if not found."""
        result = session.execute(text("""
            SELECT ph.close
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE s.code = 'CADEUR=X' AND DATE(ph.date) <= :fecha
            ORDER BY ph.date DESC
            LIMIT 1
        """), {'fecha': fecha})
        row = result.fetchone()
        if not row:
            logger.warning(f"No CAD/EUR rate found for {fecha}")
        return row[0] if row else None

    def get_chf_eur_rate(self, session, fecha: str) -> Optional[float]:
        """Get CHF/EUR exchange rate for a date. Returns None if not found."""
        result = session.execute(text("""
            SELECT ph.close
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE s.code = 'CHFEUR=X' AND DATE(ph.date) <= :fecha
            ORDER BY ph.date DESC
            LIMIT 1
        """), {'fecha': fecha})
        row = result.fetchone()
        if not row:
            logger.warning(f"No CHF/EUR rate found for {fecha}")
        return row[0] if row else None

    def get_symbol_price(self, session, symbol: str, fecha: str) -> Optional[float]:
        """Get closing price for a symbol on a date (with lookback)."""
        for days_back in range(6):
            check_date = (datetime.strptime(fecha, '%Y-%m-%d') - timedelta(days=days_back)).strftime('%Y-%m-%d')
            result = session.execute(text("""
                SELECT ph.close
                FROM price_history ph
                JOIN symbols s ON ph.symbol_id = s.id
                WHERE s.code = :symbol AND DATE(ph.date) = :fecha
                LIMIT 1
            """), {'symbol': symbol, 'fecha': check_date})
            row = result.fetchone()
            if row:
                return row[0]
        return None

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

    def calculate_position_value_eur(
        self, session, symbol: str, shares: float, fecha: str,
        eur_usd: float, cad_eur: float, chf_eur: float
    ) -> Optional[float]:
        """Calculate position value in EUR."""
        price = self.get_symbol_price(session, symbol, fecha)
        if price is None:
            return None

        value_local = price * shares
        source_currency = self.get_source_currency(symbol)

        if source_currency == 'EUR':
            return value_local
        elif source_currency == 'USD':
            return value_local / eur_usd
        elif source_currency == 'CAD':
            return value_local * cad_eur
        elif source_currency == 'CHF':
            return value_local * chf_eur

        return value_local

    def get_holdings_for_account(self, session, account: str, fecha: str) -> list:
        """Get all holdings for an account on a date."""
        result = session.execute(text("""
            SELECT symbol, shares, currency
            FROM holding_diario
            WHERE account_code = :account AND fecha = :fecha
        """), {'account': account, 'fecha': fecha})
        return result.fetchall()

    def get_cash_for_account(self, session, account: str, fecha: str) -> dict:
        """Get cash for an account on a date."""
        result = session.execute(text("""
            SELECT currency, saldo
            FROM cash_diario
            WHERE account_code = :account AND fecha = :fecha
        """), {'account': account, 'fecha': fecha})
        return {row[0]: row[1] for row in result.fetchall()}

    def recalc_posicion_for_date(self, session, fecha: str) -> float:
        """Recalculate posicion for all accounts on a specific date."""
        logger.info(f"Calculating posicion for {fecha}")

        # Get exchange rates
        eur_usd = self.get_eur_usd_rate(session, fecha)
        cad_eur = self.get_cad_eur_rate(session, fecha)
        chf_eur = self.get_chf_eur_rate(session, fecha)

        # Check for missing exchange rates
        missing_rates = []
        if eur_usd is None:
            missing_rates.append('EUR/USD')
        if cad_eur is None:
            missing_rates.append('CAD/EUR')
        if chf_eur is None:
            missing_rates.append('CHF/EUR')

        if missing_rates:
            logger.error(f"Missing exchange rates for {fecha}: {', '.join(missing_rates)}")
            raise ValueError(f"No se encontraron tipos de cambio: {', '.join(missing_rates)}")

        total_general = 0

        for account in ACCOUNTS:
            # Calculate holdings value
            holdings = self.get_holdings_for_account(session, account, fecha)
            holding_eur = 0
            missing_prices = []

            for symbol, shares, currency in holdings:
                value = self.calculate_position_value_eur(
                    session, symbol, shares, fecha, eur_usd, cad_eur, chf_eur
                )
                if value:
                    holding_eur += value
                else:
                    missing_prices.append(symbol)

            if missing_prices:
                logger.warning(f"{account}: Missing prices for {', '.join(missing_prices[:5])}")

            # Calculate cash in EUR
            cash = self.get_cash_for_account(session, account, fecha)
            cash_eur = cash.get('EUR', 0)
            cash_usd = cash.get('USD', 0)
            cash_eur += cash_usd / eur_usd

            # Total
            total_eur = holding_eur + cash_eur
            total_general += total_eur

            # Insert/update posicion
            result = session.execute(text("""
                SELECT id FROM posicion WHERE fecha = :fecha AND account_code = :account
            """), {'fecha': fecha, 'account': account})
            exists = result.fetchone()

            if exists:
                session.execute(text("""
                    UPDATE posicion
                    SET holding_eur = :holding, cash_eur = :cash, total_eur = :total
                    WHERE fecha = :fecha AND account_code = :account
                """), {
                    'fecha': fecha, 'account': account,
                    'holding': holding_eur, 'cash': cash_eur, 'total': total_eur
                })
            else:
                session.execute(text("""
                    INSERT INTO posicion (fecha, account_code, holding_eur, cash_eur, total_eur)
                    VALUES (:fecha, :account, :holding, :cash, :total)
                """), {
                    'fecha': fecha, 'account': account,
                    'holding': holding_eur, 'cash': cash_eur, 'total': total_eur
                })

            logger.debug(f"{account}: {holding_eur:,.0f} + {cash_eur:,.0f} = {total_eur:,.0f} EUR")

        logger.info(f"Posicion {fecha}: Total = {total_general:,.0f} EUR")
        return total_general

    def get_dates_in_holding_diario(self, session) -> list:
        """Get all distinct dates in holding_diario."""
        result = session.execute(text("""
            SELECT DISTINCT fecha FROM holding_diario ORDER BY fecha
        """))
        return [row[0] for row in result.fetchall()]

    def get_dates_in_posicion(self, session) -> set:
        """Get all distinct dates in posicion."""
        result = session.execute(text("""
            SELECT DISTINCT fecha FROM posicion ORDER BY fecha
        """))
        return set(row[0] for row in result.fetchall())

    def recalc_missing_dates(self) -> dict:
        """Recalculate posicion for dates missing from the table."""
        results = {
            'processed': 0,
            'total_value': 0,
            'dates': []
        }

        with self.db.get_session() as session:
            holding_dates = self.get_dates_in_holding_diario(session)
            posicion_dates = self.get_dates_in_posicion(session)

            missing = [d for d in holding_dates if d not in posicion_dates]

            if not missing:
                logger.info("All holding_diario dates already have posicion data")
                return results

            logger.info(f"Found {len(missing)} dates without posicion data")

            for fecha in missing:
                total = self.recalc_posicion_for_date(session, fecha)
                results['processed'] += 1
                results['total_value'] = total
                results['dates'].append(fecha)

            session.commit()
            logger.info(f"Updated {len(missing)} dates in posicion")

        return results

    def get_last_market_day(self, session) -> Optional[str]:
        """
        Get the last market day (excludes weekends).
        Uses price_history to determine if market was open.
        """
        # Get most recent date with prices (indicates market was open)
        result = session.execute(text("""
            SELECT MAX(date) FROM price_history
            WHERE symbol_id = (SELECT id FROM symbols WHERE code = 'AAPL' LIMIT 1)
        """))
        row = result.fetchone()
        if row and row[0]:
            return row[0] if isinstance(row[0], str) else row[0].strftime('%Y-%m-%d')
        return None

    def propagate_holding_to_date(self, session, target_date: str) -> int:
        """
        Copy holdings from the most recent date to target_date.
        This assumes holdings don't change unless there are trades.

        Returns:
            Number of records copied
        """
        # Get most recent holding date
        result = session.execute(text("""
            SELECT MAX(fecha) FROM holding_diario WHERE fecha < :target
        """), {'target': target_date})
        row = result.fetchone()

        if not row or not row[0]:
            logger.warning("No previous holdings to copy from")
            return 0

        source_date = row[0]

        # Check if target already has data
        result = session.execute(text("""
            SELECT COUNT(*) FROM holding_diario WHERE fecha = :target
        """), {'target': target_date})
        if result.fetchone()[0] > 0:
            logger.info(f"Holdings for {target_date} already exist")
            return 0

        # Copy holdings from source to target
        result = session.execute(text("""
            INSERT INTO holding_diario (fecha, account_code, symbol, shares, precio_entrada, currency)
            SELECT :target, account_code, symbol, shares, precio_entrada, currency
            FROM holding_diario
            WHERE fecha = :source
        """), {'target': target_date, 'source': source_date})

        count = result.rowcount
        logger.info(f"Copied {count} holdings from {source_date} to {target_date}")
        return count

    def propagate_cash_to_date(self, session, target_date: str) -> int:
        """
        Calculate cash for target_date based on previous day's cash plus sales minus purchases.

        Formula: Cash nuevo = Cash anterior + Ventas - Compras

        Returns:
            Number of records created/updated
        """
        # Get most recent cash date
        result = session.execute(text("""
            SELECT MAX(fecha) FROM cash_diario WHERE fecha < :target
        """), {'target': target_date})
        row = result.fetchone()

        if not row or not row[0]:
            logger.warning("No previous cash to copy from")
            return 0

        source_date = row[0]

        # Get previous day's cash by account and currency
        result = session.execute(text("""
            SELECT account_code, currency, saldo
            FROM cash_diario
            WHERE fecha = :source
        """), {'source': source_date})
        prev_cash = {}
        for account, currency, saldo in result.fetchall():
            if account not in prev_cash:
                prev_cash[account] = {}
            prev_cash[account][currency] = saldo

        # Get sales for target_date (cash inflows) grouped by account and currency
        result = session.execute(text("""
            SELECT account_code, currency, SUM(importe_total) as total
            FROM ventas
            WHERE fecha = :target
            GROUP BY account_code, currency
        """), {'target': target_date})
        sales = {}
        for account, currency, total in result.fetchall():
            if account not in sales:
                sales[account] = {}
            sales[account][currency] = total or 0

        # Get purchases for target_date (cash outflows) grouped by account and currency
        result = session.execute(text("""
            SELECT account_code, currency, SUM(importe_total) as total
            FROM compras
            WHERE fecha = :target
            GROUP BY account_code, currency
        """), {'target': target_date})
        purchases = {}
        for account, currency, total in result.fetchall():
            if account not in purchases:
                purchases[account] = {}
            purchases[account][currency] = total or 0

        # Calculate new cash: previous + sales - purchases
        all_accounts = set(prev_cash.keys()) | set(sales.keys()) | set(purchases.keys())
        count = 0

        for account in all_accounts:
            account_prev = prev_cash.get(account, {})
            account_sales = sales.get(account, {})
            account_purchases = purchases.get(account, {})

            # Get all currencies for this account
            all_currencies = set(account_prev.keys()) | set(account_sales.keys()) | set(account_purchases.keys())

            for currency in all_currencies:
                prev_saldo = account_prev.get(currency, 0)
                sale_amount = account_sales.get(currency, 0)
                purchase_amount = account_purchases.get(currency, 0)

                new_saldo = prev_saldo + sale_amount - purchase_amount

                # Check if record exists for target date
                result = session.execute(text("""
                    SELECT id FROM cash_diario
                    WHERE fecha = :fecha AND account_code = :account AND currency = :currency
                """), {'fecha': target_date, 'account': account, 'currency': currency})
                exists = result.fetchone()

                if exists:
                    session.execute(text("""
                        UPDATE cash_diario SET saldo = :saldo
                        WHERE fecha = :fecha AND account_code = :account AND currency = :currency
                    """), {'fecha': target_date, 'account': account, 'currency': currency, 'saldo': new_saldo})
                else:
                    session.execute(text("""
                        INSERT INTO cash_diario (fecha, account_code, currency, saldo)
                        VALUES (:fecha, :account, :currency, :saldo)
                    """), {'fecha': target_date, 'account': account, 'currency': currency, 'saldo': new_saldo})

                count += 1
                if sale_amount != 0 or purchase_amount != 0:
                    logger.debug(f"{account} {currency}: {prev_saldo:,.2f} + {sale_amount:,.2f} - {purchase_amount:,.2f} = {new_saldo:,.2f}")

        logger.info(f"Updated {count} cash records for {target_date}")
        return count

    def recalc_today(self) -> dict:
        """
        Recalculate posicion for the last market day.
        If holdings don't exist for that day, propagates from the most recent day.
        """
        results = {
            'processed': 0,
            'total_value': 0,
            'date': None,
            'holdings_propagated': 0,
            'cash_propagated': 0
        }

        with self.db.get_session() as session:
            # Get last market day (based on price data)
            last_market_day = self.get_last_market_day(session)

            if not last_market_day:
                logger.warning("Could not determine last market day")
                # Fall back to most recent holding date
                result = session.execute(text("SELECT MAX(fecha) FROM holding_diario"))
                row = result.fetchone()
                if not row or not row[0]:
                    logger.warning("No data in holding_diario")
                    return results
                last_market_day = row[0]

            results['date'] = last_market_day

            # Propagate holdings if needed
            results['holdings_propagated'] = self.propagate_holding_to_date(session, last_market_day)
            results['cash_propagated'] = self.propagate_cash_to_date(session, last_market_day)

            if results['holdings_propagated'] > 0 or results['cash_propagated'] > 0:
                session.commit()

            # Now calculate posicion
            total = self.recalc_posicion_for_date(session, last_market_day)
            results['processed'] = 1
            results['total_value'] = total

            session.commit()

        return results
