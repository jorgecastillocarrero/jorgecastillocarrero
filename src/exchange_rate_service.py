"""
Centralized exchange rate service.
Fetches exchange rates from database with date fallback logic.
"""

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import text

from .database import get_db_manager

logger = logging.getLogger(__name__)

# Maximum days to look back for exchange rates
MAX_LOOKBACK_DAYS = 10


class ExchangeRateError(Exception):
    """Raised when exchange rate cannot be determined."""
    pass


class ExchangeRateService:
    """Service for fetching exchange rates from the database."""

    def __init__(self, db_manager=None):
        self.db = db_manager or get_db_manager()
        self._cache = {}

    def clear_cache(self):
        """Clear the exchange rate cache."""
        self._cache.clear()

    def get_rate(self, pair: str, target_date: date, max_lookback: int = MAX_LOOKBACK_DAYS) -> float:
        """
        Get exchange rate for a currency pair on a specific date.

        Looks back up to max_lookback days if rate not found on exact date.

        Args:
            pair: Currency pair code (e.g., 'EURUSD=X')
            target_date: Date for the rate
            max_lookback: Maximum days to look back (default 10)

        Returns:
            Exchange rate as float

        Raises:
            ExchangeRateError: If no rate found within lookback period
        """
        cache_key = f"{pair}_{target_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        with self.db.get_session() as session:
            for i in range(max_lookback):
                check_date = target_date - timedelta(days=i)
                result = session.execute(text("""
                    SELECT p.close FROM symbols s
                    JOIN price_history p ON s.id = p.symbol_id
                    WHERE s.code = :pair AND DATE(p.date) = :fecha
                """), {'pair': pair, 'fecha': check_date})

                row = result.fetchone()
                if row and row[0]:
                    rate = row[0]
                    self._cache[cache_key] = rate
                    if i > 0:
                        logger.debug(f"Using {pair} rate from {check_date} for {target_date}")
                    return rate

        raise ExchangeRateError(
            f"No exchange rate found for {pair} within {max_lookback} days of {target_date}"
        )

    def get_rate_safe(self, pair: str, target_date: date, max_lookback: int = MAX_LOOKBACK_DAYS) -> Optional[float]:
        """
        Get exchange rate, returning None instead of raising on failure.

        Args:
            pair: Currency pair code (e.g., 'EURUSD=X')
            target_date: Date for the rate
            max_lookback: Maximum days to look back

        Returns:
            Exchange rate or None if not found
        """
        try:
            return self.get_rate(pair, target_date, max_lookback)
        except ExchangeRateError:
            return None

    def get_eur_usd(self, target_date: date) -> float:
        """Get EUR/USD rate for a specific date."""
        # Debug: check what's in the database
        with self.db.get_session() as session:
            # Check if symbol exists
            result = session.execute(text("""
                SELECT id, code FROM symbols WHERE code LIKE '%EUR%' OR code LIKE '%USD%'
            """))
            symbols = result.fetchall()
            print(f"DEBUG CURRENCY SYMBOLS: {symbols}", flush=True)

            # Check price_history for EURUSD
            result = session.execute(text("""
                SELECT s.code, p.date, p.close FROM symbols s
                JOIN price_history p ON s.id = p.symbol_id
                WHERE s.code = 'EURUSD=X'
                ORDER BY p.date DESC LIMIT 5
            """))
            prices = result.fetchall()
            print(f"DEBUG EURUSD PRICES: {prices}", flush=True)

        return self.get_rate('EURUSD=X', target_date)

    def get_cad_eur(self, target_date: date) -> float:
        """
        Get CAD/EUR rate: how many EUR per 1 CAD.

        First tries CADEUR=X directly, then calculates from EURUSD and USDCAD.
        """
        # Try direct rate first
        direct_rate = self.get_rate_safe('CADEUR=X', target_date)
        if direct_rate:
            return direct_rate

        # Calculate from other pairs: CAD/EUR = 1 / (USD/CAD * EUR/USD)
        eur_usd = self.get_rate('EURUSD=X', target_date)
        usd_cad = self.get_rate('USDCAD=X', target_date)
        return 1 / (usd_cad * eur_usd)

    def get_chf_eur(self, target_date: date) -> float:
        """
        Get CHF/EUR rate: how many EUR per 1 CHF.

        First tries CHFEUR=X directly, then calculates from EURUSD and USDCHF.
        """
        # Try direct rate first
        direct_rate = self.get_rate_safe('CHFEUR=X', target_date)
        if direct_rate:
            return direct_rate

        # Calculate from other pairs: CHF/EUR = 1 / (USD/CHF * EUR/USD)
        eur_usd = self.get_rate('EURUSD=X', target_date)
        usd_chf = self.get_rate('USDCHF=X', target_date)
        return 1 / (usd_chf * eur_usd)

    def get_gbp_eur(self, target_date: date) -> float:
        """
        Get GBP/EUR rate: how many EUR per 1 GBP.

        First tries GBPEUR=X directly, then calculates from EURGBP.
        """
        # Try direct rate first
        direct_rate = self.get_rate_safe('GBPEUR=X', target_date)
        if direct_rate:
            return direct_rate

        # Calculate from inverse: GBP/EUR = 1 / EUR/GBP
        eur_gbp = self.get_rate('EURGBP=X', target_date)
        return 1 / eur_gbp

    def convert_to_eur(self, amount: float, currency: str, target_date: date) -> float:
        """
        Convert amount from any supported currency to EUR.

        Args:
            amount: Amount to convert
            currency: Source currency code (EUR, USD, CAD, CHF, GBP)
            target_date: Date for exchange rate

        Returns:
            Amount in EUR

        Raises:
            ExchangeRateError: If currency not supported or rate not found
        """
        if currency == 'EUR':
            return amount

        if currency == 'USD':
            eur_usd = self.get_eur_usd(target_date)
            return amount / eur_usd

        if currency == 'CAD':
            cad_eur = self.get_cad_eur(target_date)
            return amount * cad_eur

        if currency == 'CHF':
            chf_eur = self.get_chf_eur(target_date)
            return amount * chf_eur

        if currency == 'GBP':
            gbp_eur = self.get_gbp_eur(target_date)
            return amount * gbp_eur

        raise ExchangeRateError(f"Unsupported currency: {currency}")


# Singleton instance
_rate_service: Optional[ExchangeRateService] = None


def get_exchange_rate_service(db_manager=None) -> ExchangeRateService:
    """Get the singleton exchange rate service instance."""
    global _rate_service
    if _rate_service is None:
        _rate_service = ExchangeRateService(db_manager)
    return _rate_service
