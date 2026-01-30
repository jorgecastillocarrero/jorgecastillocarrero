"""
Yahoo Finance data downloader.
Downloads and stores historical prices and fundamentals.
"""

import json
import logging
from datetime import datetime, timedelta

from .config import get_settings
from .database import get_db_manager, Exchange, Symbol, PriceHistory, Fundamental, Trabajadores
from .yahoo_client import YahooFinanceClient

# Exchange suffix to currency mapping
EXCHANGE_CURRENCY_MAP = {
    "TO": "CAD",    # Toronto
    "V": "CAD",     # TSX Venture
    "SW": "CHF",    # Swiss
    "MI": "EUR",    # Milan
    "MC": "EUR",    # Madrid
    "PA": "EUR",    # Paris
    "AS": "EUR",    # Amsterdam
    "BR": "EUR",    # Brussels
    "VI": "EUR",    # Vienna
    "HE": "EUR",    # Helsinki
    "L": "GBP",     # London
    "LSE": "GBP",   # London
    "DE": "EUR",    # Germany (Xetra)
    "XETRA": "EUR", # Germany
    "F": "EUR",     # Frankfurt
    "ST": "SEK",    # Stockholm
    "OL": "NOK",    # Oslo
    "CO": "DKK",    # Copenhagen
    "T": "JPY",     # Tokyo
    "HK": "HKD",    # Hong Kong
    "SS": "CNY",    # Shanghai
    "SZ": "CNY",    # Shenzhen
    "AX": "AUD",    # Australia
    "NZ": "NZD",    # New Zealand
    "SA": "BRL",    # Sao Paulo
    "MX": "MXN",    # Mexico
}

logger = logging.getLogger(__name__)


# Default symbols to track - All portfolio symbols
DEFAULT_SYMBOLS = [
    # CO3365
    "AKAM", "VRTX", "PCAR", "BDX", "AMZN", "MCO", "HCA", "MA", "WST", "CRM",
    # La Caixa
    "JD", "BABA", "ATZ.TO", "IAG.MC", "AEM.TO", "NESN.SW",
    # RCO951 - All 93 symbols
    "B", "TFPM", "SSRM", "RGLD", "OR", "NEM", "KGC", "FNV", "BTG", "PAAS",
    "WPM", "CEF", "SBSW", "SLV", "GLD", "AEM", "AGI", "EQX", "MAG", "SILV",
    "PHYS",
    "EAT", "EZPW", "INCY", "MFC", "MU", "PARR", "STRL", "TIGO", "TTMI", "UNFI",
    "VSCO", "W",
    "BRK-B", "KMI", "OXY", "C", "PM", "ALLY", "USB", "CVS", "JXN", "KHC",
    "ATVI", "ORCL", "FND", "NU", "ABNB", "LEN", "FSLR", "HPQ",
    "ALIT", "ANET", "ANSS", "AVGO", "AXP", "BSY", "CDNS", "CRWD", "CSCO", "CSX",
    "DECK", "DPZ", "EBAY", "ELV", "ETN", "FI", "FICO", "HD", "IBKR", "IDXX",
    "INTU", "LMT", "LOW", "MELI", "MLM", "MMC", "MOH", "MPWR", "MSCI", "MSFT",
    "MTD", "NOC", "ODFL", "ORLY", "PHM", "PTC", "PWR", "ROP", "SHW", "SNPS",
    "TOST", "TPL", "TSCO", "TT", "TYL", "UHAL", "URI", "VLTO", "VMC", "ZTS",
    "AZO", "SPGI", "CMG", "CPRT", "NVR", "CTAS", "AAPL", "TDG", "FTNT", "V",
    "KNSL", "HLI", "DOCS", "DUOL",
    # Benchmarks
    "SPY", "QQQ", "TLT",
]


class YahooDownloader:
    """Downloads and stores data from Yahoo Finance."""

    def __init__(self):
        self.client = YahooFinanceClient()
        self.db = get_db_manager()

    def _ensure_exchange(self, session, exchange_code: str = "US") -> Exchange:
        """Ensure exchange exists in database."""
        exchange = (
            session.query(Exchange)
            .filter(Exchange.code == exchange_code)
            .first()
        )
        if not exchange:
            exchange = Exchange(
                code=exchange_code,
                name="United States" if exchange_code == "US" else exchange_code,
                country="USA" if exchange_code == "US" else None,
                currency="USD" if exchange_code == "US" else None,
            )
            session.add(exchange)
            session.flush()
        return exchange

    def _get_currency_for_symbol(self, symbol: str) -> str:
        """Get the trading currency based on symbol's exchange suffix."""
        if "." in symbol:
            suffix = symbol.split(".")[-1]
            return EXCHANGE_CURRENCY_MAP.get(suffix, "USD")
        return "USD"

    def _ensure_symbol(self, session, symbol: str, exchange: Exchange, info: dict = None) -> Symbol:
        """Ensure symbol exists in database with correct currency."""
        db_symbol = (
            session.query(Symbol)
            .filter(Symbol.code == symbol, Symbol.exchange_id == exchange.id)
            .first()
        )

        # Determine correct currency based on exchange suffix
        correct_currency = self._get_currency_for_symbol(symbol)

        if not db_symbol:
            db_symbol = Symbol(
                code=symbol,
                exchange_id=exchange.id,
                name=info.get("name") if info else symbol,
                symbol_type="stock",
                currency=correct_currency,
            )
            session.add(db_symbol)
            session.flush()
        else:
            # Update name if provided
            if info and info.get("name"):
                db_symbol.name = info.get("name")
            # Fix currency if it was wrong
            if db_symbol.currency != correct_currency:
                logger.info(f"{symbol}: Fixing currency {db_symbol.currency} -> {correct_currency}")
                db_symbol.currency = correct_currency
            session.flush()

        return db_symbol

    def download_historical_prices(
        self,
        symbol: str,
        period: str = "max",
        incremental: bool = True,
    ) -> int:
        """
        Download and store historical prices for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL' or 'AAPL.US')
            period: Data period (max, 10y, 5y, 2y, 1y, etc.)
            incremental: Only download new data if True

        Returns:
            Number of records inserted
        """
        # Clean symbol
        clean_symbol = symbol.replace(".US", "")
        started_at = datetime.utcnow()

        try:
            # Get basic info first
            ticker = self.client.get_ticker(clean_symbol)
            try:
                info = ticker.info
                name = info.get("longName") or info.get("shortName") or clean_symbol
            except:
                info = {}
                name = clean_symbol

            with self.db.get_session() as session:
                exchange = self._ensure_exchange(session, "US")
                db_symbol = self._ensure_symbol(
                    session, clean_symbol, exchange, {"name": name}
                )

                # Check for incremental download
                start_date = None
                if incremental:
                    latest = self.db.get_latest_price_date(session, db_symbol.id)
                    if latest:
                        start_date = latest + timedelta(days=1)
                        if start_date >= datetime.now().date():
                            logger.info(f"{symbol}: Already up to date")
                            return 0

            # Download data
            if start_date:
                df = self.client.get_historical_data(
                    clean_symbol, start=start_date.isoformat()
                )
            else:
                df = self.client.get_historical_data(clean_symbol, period=period)

            if df.empty:
                logger.warning(f"{symbol}: No data returned")
                return 0

            # Store in database
            with self.db.get_session() as session:
                exchange = self._ensure_exchange(session, "US")
                db_symbol = self._ensure_symbol(session, clean_symbol, exchange)

                count = self.db.bulk_insert_prices(session, db_symbol.id, df)

                self.db.log_download(
                    session,
                    operation="yahoo_eod",
                    status="success",
                    symbol=clean_symbol,
                    records_downloaded=count,
                    started_at=started_at,
                )

            logger.info(f"{symbol}: Downloaded {count} price records")
            return count

        except Exception as e:
            logger.error(f"{symbol}: Error - {e}")
            with self.db.get_session() as session:
                self.db.log_download(
                    session,
                    operation="yahoo_eod",
                    status="error",
                    symbol=symbol,
                    error_message=str(e),
                    started_at=started_at,
                )
            raise

    def download_fundamentals(self, symbol: str) -> bool:
        """
        Download and store fundamental data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if successful
        """
        clean_symbol = symbol.replace(".US", "")
        started_at = datetime.utcnow()

        try:
            data = self.client.get_fundamentals(clean_symbol)

            if "error" in data:
                raise Exception(data["error"])

            with self.db.get_session() as session:
                exchange = self._ensure_exchange(session, "US")
                db_symbol = self._ensure_symbol(
                    session, clean_symbol, exchange,
                    {"name": data["general"].get("name")}
                )

                # Check if we already have today's fundamentals
                today = datetime.utcnow().date()
                existing = (
                    session.query(Fundamental)
                    .filter(
                        Fundamental.symbol_id == db_symbol.id,
                        Fundamental.data_date == today,
                    )
                    .first()
                )

                if existing:
                    # Update existing
                    fundamental = existing
                else:
                    fundamental = Fundamental(
                        symbol_id=db_symbol.id,
                        data_date=today,
                    )
                    session.add(fundamental)

                # Update fields
                g = data["general"]
                m = data["market_data"]
                v = data["valuation"]
                f = data["financials"]
                ps = data["per_share"]
                mg = data["margins"]
                gr = data["growth"]
                r = data["returns"]

                fundamental.sector = g.get("sector")
                fundamental.industry = g.get("industry")
                fundamental.description = g.get("description")
                fundamental.market_cap = m.get("market_cap")
                fundamental.employees = g.get("employees")

                fundamental.pe_ratio = v.get("pe_ratio")
                fundamental.forward_pe = v.get("forward_pe")
                fundamental.peg_ratio = v.get("peg_ratio")
                fundamental.price_to_book = v.get("price_to_book")
                fundamental.price_to_sales = v.get("price_to_sales")
                fundamental.enterprise_value = m.get("enterprise_value")

                fundamental.revenue = f.get("revenue")
                fundamental.gross_profit = f.get("gross_profit")
                fundamental.net_income = f.get("net_income")
                fundamental.ebitda = f.get("ebitda")

                fundamental.eps = ps.get("eps_trailing")
                fundamental.book_value_per_share = ps.get("book_value")
                fundamental.dividend_per_share = data["dividend"].get("dividend_rate")
                fundamental.dividend_yield = data["dividend"].get("dividend_yield")

                fundamental.profit_margin = mg.get("profit_margin")
                fundamental.operating_margin = mg.get("operating_margin")
                fundamental.gross_margin = mg.get("gross_margin")

                fundamental.revenue_growth = gr.get("revenue_growth")
                fundamental.earnings_growth = gr.get("earnings_growth")

                # Fetch and store earnings history (EPS beat data)
                try:
                    ticker = self.client.get_ticker(clean_symbol)
                    earnings_hist = ticker.earnings_history
                    if earnings_hist is not None and not earnings_hist.empty:
                        # earnings_history is indexed by quarter date, most recent last
                        # Reverse to get most recent first
                        quarters = earnings_hist.iloc[::-1].head(4)

                        for i, (quarter_date, row) in enumerate(quarters.iterrows()):
                            q_date = quarter_date.date() if hasattr(quarter_date, 'date') else quarter_date
                            eps_actual = row.get('epsActual')
                            eps_estimate = row.get('epsEstimate')
                            eps_diff = row.get('epsDifference')
                            surprise_pct = row.get('surprisePercent')

                            if i == 0:
                                fundamental.q0_eps_actual = eps_actual
                                fundamental.q0_eps_estimate = eps_estimate
                                fundamental.q0_eps_difference = eps_diff
                                fundamental.q0_eps_surprise_pct = surprise_pct
                                fundamental.q0_quarter_end = q_date
                            elif i == 1:
                                fundamental.q1_eps_actual = eps_actual
                                fundamental.q1_eps_estimate = eps_estimate
                                fundamental.q1_eps_difference = eps_diff
                                fundamental.q1_eps_surprise_pct = surprise_pct
                                fundamental.q1_quarter_end = q_date
                            elif i == 2:
                                fundamental.q2_eps_actual = eps_actual
                                fundamental.q2_eps_estimate = eps_estimate
                                fundamental.q2_eps_difference = eps_diff
                                fundamental.q2_eps_surprise_pct = surprise_pct
                                fundamental.q2_quarter_end = q_date
                            elif i == 3:
                                fundamental.q3_eps_actual = eps_actual
                                fundamental.q3_eps_estimate = eps_estimate
                                fundamental.q3_eps_difference = eps_diff
                                fundamental.q3_eps_surprise_pct = surprise_pct
                                fundamental.q3_quarter_end = q_date
                except Exception as eh_error:
                    logger.warning(f"{symbol}: Could not fetch earnings history - {eh_error}")

                # Store raw JSON
                fundamental.raw_data = json.dumps(data)

                session.flush()

                # Save to trabajadores table
                employees_count = g.get("employees")
                if employees_count:
                    existing_trab = (
                        session.query(Trabajadores)
                        .filter(
                            Trabajadores.symbol_id == db_symbol.id,
                            Trabajadores.fecha == today,
                        )
                        .first()
                    )
                    if existing_trab:
                        existing_trab.employees = employees_count
                    else:
                        trab = Trabajadores(
                            symbol_id=db_symbol.id,
                            fecha=today,
                            employees=employees_count,
                        )
                        session.add(trab)
                    session.flush()

                self.db.log_download(
                    session,
                    operation="yahoo_fundamentals",
                    status="success",
                    symbol=clean_symbol,
                    records_downloaded=1,
                    started_at=started_at,
                )

            logger.info(f"{symbol}: Fundamentals updated")
            return True

        except Exception as e:
            logger.error(f"{symbol}: Fundamentals error - {e}")
            with self.db.get_session() as session:
                self.db.log_download(
                    session,
                    operation="yahoo_fundamentals",
                    status="error",
                    symbol=symbol,
                    error_message=str(e),
                    started_at=started_at,
                )
            return False

    def download_all(
        self,
        symbols: list[str] | None = None,
        include_fundamentals: bool = True,
        period: str = "max",
    ) -> dict:
        """
        Download all data for multiple symbols.

        Args:
            symbols: List of symbols (uses defaults if None)
            include_fundamentals: Also download fundamental data
            period: Historical data period

        Returns:
            Results summary
        """
        symbols = symbols or DEFAULT_SYMBOLS

        results = {
            "total_symbols": len(symbols),
            "prices_success": 0,
            "prices_failed": 0,
            "fundamentals_success": 0,
            "fundamentals_failed": 0,
            "total_price_records": 0,
            "errors": [],
        }

        for symbol in symbols:
            # Download prices
            try:
                count = self.download_historical_prices(symbol, period=period)
                results["prices_success"] += 1
                results["total_price_records"] += count
            except Exception as e:
                results["prices_failed"] += 1
                results["errors"].append({"symbol": symbol, "type": "prices", "error": str(e)})

            # Download fundamentals
            if include_fundamentals:
                try:
                    if self.download_fundamentals(symbol):
                        results["fundamentals_success"] += 1
                    else:
                        results["fundamentals_failed"] += 1
                except Exception as e:
                    results["fundamentals_failed"] += 1
                    results["errors"].append({"symbol": symbol, "type": "fundamentals", "error": str(e)})

        return results


# CLI functionality
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("\n=== Yahoo Finance Downloader ===\n")

    downloader = YahooDownloader()

    symbols = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SYMBOLS

    print(f"Downloading data for: {', '.join(symbols)}\n")

    results = downloader.download_all(symbols)

    print(f"\n=== Results ===")
    print(f"Prices: {results['prices_success']}/{results['total_symbols']} successful")
    print(f"Fundamentals: {results['fundamentals_success']}/{results['total_symbols']} successful")
    print(f"Total price records: {results['total_price_records']:,}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  - {err['symbol']} ({err['type']}): {err['error'][:50]}")
