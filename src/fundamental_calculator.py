"""
Fundamental Features Calculator - Daily Point-in-Time
Calculates daily market cap, ratios, margins, and growth metrics.

Usage:
    python -m src.fundamental_calculator --symbol AAPL
    python -m src.fundamental_calculator --all --limit 100
    python -m src.fundamental_calculator --test
"""

import logging
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"

# Index membership (simplified - ideally load from file)
SP500_SYMBOLS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
    'JNJ', 'JPM', 'V', 'PG', 'XOM', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY', 'PEP',
    'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT', 'DHR', 'NEE',
    'LIN', 'ADBE', 'CRM', 'NKE', 'ORCL', 'PM', 'TXN', 'RTX', 'COP', 'MS', 'IBM',
    'GE', 'CAT', 'AMGN', 'UPS', 'LOW', 'SPGI', 'BA', 'GS', 'BLK', 'INTU', 'AMD'
}

NASDAQ100_SYMBOLS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
    'PEP', 'CSCO', 'ADBE', 'NFLX', 'AMD', 'CMCSA', 'INTC', 'TXN', 'QCOM', 'TMUS',
    'AMGN', 'INTU', 'AMAT', 'ISRG', 'HON', 'SBUX', 'BKNG', 'VRTX', 'GILD', 'ADP',
    'MDLZ', 'ADI', 'REGN', 'LRCX', 'PANW', 'MU', 'KLAC', 'SNPS', 'CDNS', 'PYPL'
}


class FundamentalCalculator:
    """Calculate daily point-in-time fundamental features."""

    def __init__(self, db_url: str = FMP_DATABASE_URL):
        self.db_url = db_url

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def get_price_data(self, symbol: str) -> pd.DataFrame:
        """Get daily prices."""
        conn = self.get_connection()
        try:
            query = 'SELECT date, close FROM fmp_price_history WHERE symbol = %s ORDER BY date'
            df = pd.read_sql(query, conn, params=(symbol,))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    def get_profile(self, symbol: str) -> dict:
        """Get static profile data."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute('''
                SELECT sector, industry, country, exchange_short, employees, mkt_cap, price
                FROM fmp_profiles WHERE symbol = %s LIMIT 1
            ''', (symbol,))
            row = cur.fetchone()
            if row:
                shares = row[5] / row[6] if row[5] and row[6] and row[6] > 0 else None
                return {
                    'sector': row[0], 'industry': row[1], 'country': row[2],
                    'exchange': row[3], 'employees': row[4], 'shares': shares
                }
            return {}
        finally:
            conn.close()

    def get_income_statements(self, symbol: str) -> pd.DataFrame:
        """Get quarterly income statements."""
        conn = self.get_connection()
        try:
            query = '''
                SELECT date, period, revenue, gross_profit, operating_income, net_income,
                       eps_diluted, shares_outstanding
                FROM fmp_income_statements
                WHERE symbol = %s AND period IN ('Q1', 'Q2', 'Q3', 'Q4')
                ORDER BY date
            '''
            df = pd.read_sql(query, conn, params=(symbol,))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            return df
        finally:
            conn.close()

    def get_balance_sheets(self, symbol: str) -> pd.DataFrame:
        """Get quarterly balance sheets."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM fmp_balance_sheets WHERE symbol = %s", (symbol,))
            if cur.fetchone()[0] == 0:
                return pd.DataFrame()

            query = '''
                SELECT date, total_stockholders_equity, total_assets, total_debt,
                       total_current_assets, total_current_liabilities
                FROM fmp_balance_sheets
                WHERE symbol = %s AND period IN ('Q1', 'Q2', 'Q3', 'Q4')
                ORDER BY date
            '''
            df = pd.read_sql(query, conn, params=(symbol,))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except:
            return pd.DataFrame()
        finally:
            conn.close()

    def get_all_symbols(self, limit: Optional[int] = None) -> list:
        """Get symbols with price and profile data."""
        conn = self.get_connection()
        try:
            query = '''
                SELECT DISTINCT p.symbol
                FROM fmp_price_history p
                JOIN fmp_profiles pr ON p.symbol = pr.symbol
                GROUP BY p.symbol HAVING COUNT(*) >= 100
                ORDER BY p.symbol
            '''
            if limit:
                query = query.replace("ORDER BY p.symbol", f"ORDER BY p.symbol LIMIT {limit}")
            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    # ==================
    # CALCULATION HELPERS
    # ==================

    def get_ttm(self, df: pd.DataFrame, trade_date, column: str):
        """Get TTM (sum of last 4 quarters) for a column."""
        if df.empty or column not in df.columns:
            return None
        recent = df[df['date'] <= trade_date].tail(4)
        if len(recent) < 4:
            return None
        val = recent[column].sum()
        return val if pd.notna(val) else None

    def get_latest(self, df: pd.DataFrame, trade_date, column: str):
        """Get most recent value before trade_date."""
        if df.empty or column not in df.columns:
            return None
        recent = df[df['date'] <= trade_date]
        if recent.empty:
            return None
        val = recent[column].iloc[-1]
        return val if pd.notna(val) else None

    def get_growth(self, df: pd.DataFrame, trade_date, column: str, years: int):
        """Calculate CAGR over N years."""
        if df.empty or column not in df.columns:
            return None

        recent = df[df['date'] <= trade_date]
        if len(recent) < years * 4:  # Need enough quarters
            return None

        # Get TTM now and N years ago
        current_ttm = recent.tail(4)[column].sum()
        old_data = recent.head(len(recent) - (years - 1) * 4).tail(4)
        if len(old_data) < 4:
            return None
        old_ttm = old_data[column].sum()

        if old_ttm and old_ttm > 0 and current_ttm:
            cagr = (current_ttm / old_ttm) ** (1 / years) - 1
            return cagr
        return None

    def classify_market_cap(self, mkt_cap):
        """Classify market cap."""
        if mkt_cap is None:
            return None
        if mkt_cap < 300_000_000:
            return 'micro'
        elif mkt_cap < 2_000_000_000:
            return 'small'
        elif mkt_cap < 10_000_000_000:
            return 'mid'
        elif mkt_cap < 200_000_000_000:
            return 'large'
        return 'mega'

    def classify_pe(self, pe):
        """Classify P/E ratio."""
        if pe is None or np.isnan(pe):
            return None
        if pe < 0:
            return 'negative'
        elif pe < 15:
            return 'cheap'
        elif pe < 25:
            return 'fair'
        elif pe < 40:
            return 'expensive'
        return 'very_expensive'

    # ==================
    # MAIN CALCULATION
    # ==================

    def calculate_features(self, symbol: str) -> pd.DataFrame:
        """Calculate daily fundamental features."""
        prices = self.get_price_data(symbol)
        if prices.empty:
            return pd.DataFrame()

        profile = self.get_profile(symbol)
        if not profile:
            return pd.DataFrame()

        income_df = self.get_income_statements(symbol)
        balance_df = self.get_balance_sheets(symbol)
        shares = profile.get('shares')

        features = []

        for trade_date in prices.index:
            price = prices.loc[trade_date, 'close']
            td = pd.Timestamp(trade_date)

            # Market cap (daily)
            market_cap = int(price * shares) if shares else None

            # Revenue
            revenue_ttm = self.get_ttm(income_df, td, 'revenue')
            revenue_growth_3y = self.get_growth(income_df, td, 'revenue', 3)
            revenue_growth_5y = self.get_growth(income_df, td, 'revenue', 5)

            # EPS
            eps_ttm = self.get_ttm(income_df, td, 'eps_diluted')
            eps_growth_3y = self.get_growth(income_df, td, 'eps_diluted', 3)
            eps_growth_5y = self.get_growth(income_df, td, 'eps_diluted', 5)

            # Margins
            gross_profit_ttm = self.get_ttm(income_df, td, 'gross_profit')
            operating_income_ttm = self.get_ttm(income_df, td, 'operating_income')
            net_income_ttm = self.get_ttm(income_df, td, 'net_income')

            gross_margin = gross_profit_ttm / revenue_ttm if gross_profit_ttm and revenue_ttm else None
            operating_margin = operating_income_ttm / revenue_ttm if operating_income_ttm and revenue_ttm else None
            profit_margin = net_income_ttm / revenue_ttm if net_income_ttm and revenue_ttm else None

            # Balance sheet
            total_equity = self.get_latest(balance_df, td, 'total_stockholders_equity')
            total_debt = self.get_latest(balance_df, td, 'total_debt')
            total_assets = self.get_latest(balance_df, td, 'total_assets')
            current_assets = self.get_latest(balance_df, td, 'total_current_assets')
            current_liab = self.get_latest(balance_df, td, 'total_current_liabilities')

            # Ratios (daily - price dependent)
            pe_ratio = price / eps_ttm if eps_ttm and eps_ttm > 0 else None
            pb_ratio = market_cap / total_equity if market_cap and total_equity and total_equity > 0 else None
            ps_ratio = market_cap / revenue_ttm if market_cap and revenue_ttm and revenue_ttm > 0 else None
            debt_to_equity = total_debt / total_equity if total_debt and total_equity and total_equity > 0 else None
            current_ratio = current_assets / current_liab if current_assets and current_liab and current_liab > 0 else None

            # Profitability
            roe = net_income_ttm / total_equity if net_income_ttm and total_equity and total_equity > 0 else None
            roa = net_income_ttm / total_assets if net_income_ttm and total_assets and total_assets > 0 else None

            row = {
                'date': trade_date,
                # Static
                'sector': profile.get('sector'),
                'industry': profile.get('industry'),
                'country': profile.get('country'),
                'exchange': profile.get('exchange'),
                # Size
                'market_cap': market_cap,
                'market_cap_cat': self.classify_market_cap(market_cap),
                'employees': profile.get('employees'),
                # Revenue
                'revenue_ttm': int(revenue_ttm) if revenue_ttm else None,
                'revenue_fwd': None,  # Requires estimates data
                'revenue_growth_3y': revenue_growth_3y,
                'revenue_growth_5y': revenue_growth_5y,
                # EPS
                'eps_ttm': eps_ttm,
                'eps_fwd': None,  # Requires estimates data
                'eps_growth_3y': eps_growth_3y,
                'eps_growth_5y': eps_growth_5y,
                # Margins
                'gross_margin': gross_margin,
                'operating_margin': operating_margin,
                'profit_margin': profit_margin,
                # Valuation ratios
                'pe_ratio': pe_ratio,
                'pe_fwd': None,  # Requires estimates data
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'pe_zone': self.classify_pe(pe_ratio),
                # Balance sheet
                'total_debt': int(total_debt) if total_debt else None,
                'total_equity': int(total_equity) if total_equity else None,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                # Profitability
                'roe': roe,
                'roa': roa,
                # Indices
                'sp500_member': symbol in SP500_SYMBOLS,
                'nasdaq100_member': symbol in NASDAQ100_SYMBOLS,
            }
            features.append(row)

        return pd.DataFrame(features)

    def save_features(self, symbol: str, features: pd.DataFrame) -> int:
        """Save features to database."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM features_fundamental WHERE symbol = %s", (symbol,))

            columns = ['symbol', 'date', 'sector', 'industry', 'country', 'exchange',
                       'market_cap', 'market_cap_cat', 'employees',
                       'revenue_ttm', 'revenue_fwd', 'revenue_growth_3y', 'revenue_growth_5y',
                       'eps_ttm', 'eps_fwd', 'eps_growth_3y', 'eps_growth_5y',
                       'gross_margin', 'operating_margin', 'profit_margin',
                       'pe_ratio', 'pe_fwd', 'pb_ratio', 'ps_ratio', 'pe_zone',
                       'total_debt', 'total_equity', 'debt_to_equity', 'current_ratio',
                       'roe', 'roa', 'sp500_member', 'nasdaq100_member']

            values = []
            for _, row in features.iterrows():
                record = [symbol]
                for col in columns[1:]:
                    val = row.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        record.append(None)
                    elif isinstance(val, (np.bool_, bool)):
                        record.append(bool(val))
                    elif isinstance(val, (np.floating, float)):
                        record.append(float(val) if not np.isnan(val) else None)
                    elif isinstance(val, (np.integer, int)):
                        record.append(int(val))
                    else:
                        record.append(val)
                values.append(tuple(record))

            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO features_fundamental ({', '.join(columns)}) VALUES ({placeholders})"
            cur.executemany(query, values)
            conn.commit()
            return len(values)
        finally:
            conn.close()

    def process_symbol(self, symbol: str) -> bool:
        """Process single symbol."""
        try:
            features = self.calculate_features(symbol)
            if features.empty:
                logger.warning(f"{symbol}: No features")
                return False
            count = self.save_features(symbol, features)
            logger.info(f"{symbol}: Saved {count} records")
            return True
        except Exception as e:
            logger.error(f"{symbol}: {e}")
            return False

    def process_all(self, limit: Optional[int] = None, batch_log: int = 50):
        """Process all symbols."""
        symbols = self.get_all_symbols(limit)
        total, success, failed = len(symbols), 0, 0
        logger.info(f"Processing {total} symbols...")

        for i, symbol in enumerate(symbols, 1):
            if self.process_symbol(symbol):
                success += 1
            else:
                failed += 1
            if i % batch_log == 0:
                logger.info(f"Progress: {i}/{total} | OK: {success} | Failed: {failed}")

        logger.info(f"Done: {success} OK, {failed} failed")
        return {'total': total, 'success': success, 'failed': failed}


def main():
    parser = argparse.ArgumentParser(description='Calculate fundamental features')
    parser.add_argument('--symbol', type=str, help='Single symbol')
    parser.add_argument('--all', action='store_true', help='All symbols')
    parser.add_argument('--limit', type=int, help='Limit symbols')
    parser.add_argument('--test', action='store_true', help='Test with AAPL')

    args = parser.parse_args()
    calc = FundamentalCalculator()

    if args.test:
        calc.process_symbol('AAPL')
    elif args.symbol:
        calc.process_symbol(args.symbol)
    elif args.all:
        calc.process_all(limit=args.limit)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
