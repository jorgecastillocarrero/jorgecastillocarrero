"""
Market Breadth Calculator
Calculates McClellan Oscillator, Summation Index, and other breadth indicators.

Usage:
    python -m src.breadth_calculator
    python -m src.breadth_calculator --start 2023-01-01
"""

import logging
import argparse
from datetime import date, timedelta
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class BreadthCalculator:
    """Calculate market breadth indicators."""

    def __init__(self, db_url: str = FMP_DATABASE_URL):
        self.db_url = db_url

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def get_sp500_members(self) -> List[str]:
        """Get S&P 500 member symbols."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol FROM index_members
                WHERE index_name = 'SP500'
            """)
            symbols = [row[0] for row in cur.fetchall()]
            logger.info(f"S&P 500 members: {len(symbols)}")
            return symbols
        finally:
            conn.close()

    def get_nasdaq100_members(self) -> List[str]:
        """Get Nasdaq 100 member symbols."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol FROM index_members
                WHERE index_name = 'NASDAQ100'
            """)
            symbols = [row[0] for row in cur.fetchall()]
            logger.info(f"Nasdaq 100 members: {len(symbols)}")
            return symbols
        finally:
            conn.close()

    def get_all_symbols(self) -> List[str]:
        """Get all symbols with price data."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT symbol FROM fmp_price_history
                WHERE symbol NOT LIKE '%-%'
                AND symbol NOT LIKE '%.%'
                AND LENGTH(symbol) <= 5
            """)
            symbols = [row[0] for row in cur.fetchall()]
            logger.info(f"All market symbols: {len(symbols)}")
            return symbols
        finally:
            conn.close()

    def get_trading_dates(self, start_date: Optional[date] = None) -> List[date]:
        """Get all trading dates."""
        conn = self.get_connection()
        try:
            query = """
                SELECT DISTINCT date FROM fmp_price_history
                WHERE symbol = 'SPY'
            """
            if start_date:
                query += f" AND date >= '{start_date}'"
            query += " ORDER BY date"

            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_daily_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get daily price and technical data for symbols."""
        conn = self.get_connection()
        try:
            symbols_str = "','".join(symbols)
            query = f"""
                SELECT p.symbol, p.date, p.close,
                       LAG(p.close) OVER (PARTITION BY p.symbol ORDER BY p.date) as prev_close,
                       t.above_sma_200, t.above_sma_50, t.new_high, t.new_low
                FROM fmp_price_history p
                LEFT JOIN features_technical t ON p.symbol = t.symbol AND p.date = t.date
                WHERE p.symbol IN ('{symbols_str}')
                ORDER BY p.date, p.symbol
            """
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()

    def calculate_daily_breadth(self, df: pd.DataFrame, trade_date: date) -> Dict:
        """Calculate breadth metrics for a single day."""
        day_data = df[df['date'] == trade_date].copy()

        if day_data.empty:
            return None

        # Filter out rows without prev_close (first day for each symbol)
        valid_data = day_data[day_data['prev_close'].notna()]

        if valid_data.empty:
            return None

        # Calculate advances/declines
        advances = len(valid_data[valid_data['close'] > valid_data['prev_close']])
        declines = len(valid_data[valid_data['close'] < valid_data['prev_close']])
        unchanged = len(valid_data[valid_data['close'] == valid_data['prev_close']])
        total = advances + declines + unchanged

        if total == 0:
            return None

        # Net advances
        net_advances = advances - declines
        net_advances_pct = (net_advances / total) * 100 if total > 0 else 0

        # A/D Ratio
        adv_dec_ratio = advances / declines if declines > 0 else float('inf') if advances > 0 else 1

        # New highs/lows (use all day_data, not just valid_data)
        new_highs = int(day_data['new_high'].sum()) if 'new_high' in day_data.columns and day_data['new_high'].notna().any() else 0
        new_lows = int(day_data['new_low'].sum()) if 'new_low' in day_data.columns and day_data['new_low'].notna().any() else 0
        highs_lows_ratio = new_highs / new_lows if new_lows > 0 else float('inf') if new_highs > 0 else 1

        # % above SMAs (use all day_data)
        valid_sma200 = day_data[day_data['above_sma_200'].notna()]
        pct_above_sma200 = (valid_sma200['above_sma_200'].sum() / len(valid_sma200) * 100) if len(valid_sma200) > 0 else None

        valid_sma50 = day_data[day_data['above_sma_50'].notna()]
        pct_above_sma50 = (valid_sma50['above_sma_50'].sum() / len(valid_sma50) * 100) if len(valid_sma50) > 0 else None

        return {
            'advances': advances,
            'declines': declines,
            'unchanged': unchanged,
            'total': total,
            'adv_dec_ratio': adv_dec_ratio if adv_dec_ratio != float('inf') else None,
            'net_advances': net_advances,
            'net_advances_pct': net_advances_pct,
            'new_highs': int(new_highs) if pd.notna(new_highs) else 0,
            'new_lows': int(new_lows) if pd.notna(new_lows) else 0,
            'highs_lows_ratio': highs_lows_ratio if highs_lows_ratio != float('inf') else None,
            'pct_above_sma200': pct_above_sma200,
            'pct_above_sma50': pct_above_sma50,
        }

    def calculate_mcclellan(self, net_advances_series: pd.Series, pct_series: pd.Series) -> Dict:
        """Calculate McClellan Oscillator and Summation Index."""
        if len(net_advances_series) < 39:
            return {
                'mcclellan_osc_abs': None,
                'mcclellan_osc_pct': None,
                'mcclellan_sum_abs': None,
                'mcclellan_sum_pct': None,
            }

        # EMA calculation
        ema19_abs = net_advances_series.ewm(span=19, adjust=False).mean()
        ema39_abs = net_advances_series.ewm(span=39, adjust=False).mean()
        mcclellan_osc_abs = ema19_abs - ema39_abs

        ema19_pct = pct_series.ewm(span=19, adjust=False).mean()
        ema39_pct = pct_series.ewm(span=39, adjust=False).mean()
        mcclellan_osc_pct = ema19_pct - ema39_pct

        # Summation Index (cumulative sum of oscillator)
        mcclellan_sum_abs = mcclellan_osc_abs.cumsum()
        mcclellan_sum_pct = mcclellan_osc_pct.cumsum()

        return {
            'mcclellan_osc_abs': mcclellan_osc_abs.iloc[-1],
            'mcclellan_osc_pct': mcclellan_osc_pct.iloc[-1],
            'mcclellan_sum_abs': mcclellan_sum_abs.iloc[-1],
            'mcclellan_sum_pct': mcclellan_sum_pct.iloc[-1],
        }

    def calculate_all_breadth(self, start_date: Optional[date] = None) -> pd.DataFrame:
        """Calculate breadth for all indices."""
        trading_dates = self.get_trading_dates(start_date)
        if not trading_dates:
            logger.warning("No trading dates found")
            return pd.DataFrame()

        logger.info(f"Processing {len(trading_dates)} trading dates...")

        # Get symbol lists
        sp500_symbols = self.get_sp500_members()
        nasdaq_symbols = self.get_nasdaq100_members()
        all_symbols = self.get_all_symbols()

        # Get price data for all symbols
        logger.info("Loading price data...")
        all_unique_symbols = list(set(sp500_symbols + nasdaq_symbols + all_symbols))
        price_data = self.get_daily_data(all_unique_symbols)

        # Filter data by index
        sp500_data = price_data[price_data['symbol'].isin(sp500_symbols)]
        nasdaq_data = price_data[price_data['symbol'].isin(nasdaq_symbols)]

        results = []

        # Track net advances for McClellan calculation
        sp500_net_adv = []
        sp500_net_adv_pct = []
        nasdaq_net_adv = []
        nasdaq_net_adv_pct = []
        all_net_adv = []
        all_net_adv_pct = []

        for i, trade_date in enumerate(trading_dates):
            row = {'date': trade_date}

            # S&P 500
            sp500_breadth = self.calculate_daily_breadth(sp500_data, trade_date)
            if sp500_breadth:
                for key, value in sp500_breadth.items():
                    row[f'sp500_{key}'] = value
                sp500_net_adv.append(sp500_breadth['net_advances'])
                sp500_net_adv_pct.append(sp500_breadth['net_advances_pct'])

                # Calculate McClellan
                mcclellan = self.calculate_mcclellan(
                    pd.Series(sp500_net_adv),
                    pd.Series(sp500_net_adv_pct)
                )
                row['sp500_mcclellan_osc_abs'] = mcclellan['mcclellan_osc_abs']
                row['sp500_mcclellan_osc_pct'] = mcclellan['mcclellan_osc_pct']
                row['sp500_mcclellan_sum_abs'] = mcclellan['mcclellan_sum_abs']
                row['sp500_mcclellan_sum_pct'] = mcclellan['mcclellan_sum_pct']

            # Nasdaq 100
            nasdaq_breadth = self.calculate_daily_breadth(nasdaq_data, trade_date)
            if nasdaq_breadth:
                for key, value in nasdaq_breadth.items():
                    row[f'nasdaq_{key}'] = value
                nasdaq_net_adv.append(nasdaq_breadth['net_advances'])
                nasdaq_net_adv_pct.append(nasdaq_breadth['net_advances_pct'])

                mcclellan = self.calculate_mcclellan(
                    pd.Series(nasdaq_net_adv),
                    pd.Series(nasdaq_net_adv_pct)
                )
                row['nasdaq_mcclellan_osc_abs'] = mcclellan['mcclellan_osc_abs']
                row['nasdaq_mcclellan_osc_pct'] = mcclellan['mcclellan_osc_pct']
                row['nasdaq_mcclellan_sum_abs'] = mcclellan['mcclellan_sum_abs']
                row['nasdaq_mcclellan_sum_pct'] = mcclellan['mcclellan_sum_pct']

            # All market
            all_breadth = self.calculate_daily_breadth(price_data, trade_date)
            if all_breadth:
                for key, value in all_breadth.items():
                    row[f'all_{key}'] = value
                all_net_adv.append(all_breadth['net_advances'])
                all_net_adv_pct.append(all_breadth['net_advances_pct'])

                mcclellan = self.calculate_mcclellan(
                    pd.Series(all_net_adv),
                    pd.Series(all_net_adv_pct)
                )
                row['all_mcclellan_osc_abs'] = mcclellan['mcclellan_osc_abs']
                row['all_mcclellan_osc_pct'] = mcclellan['mcclellan_osc_pct']
                row['all_mcclellan_sum_abs'] = mcclellan['mcclellan_sum_abs']
                row['all_mcclellan_sum_pct'] = mcclellan['mcclellan_sum_pct']

            results.append(row)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(trading_dates)} dates")

        return pd.DataFrame(results)

    def save_breadth(self, df: pd.DataFrame) -> int:
        """Save breadth data to database."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Delete existing data
            dates = df['date'].tolist()
            if dates:
                cur.execute("DELETE FROM market_breadth WHERE date >= %s AND date <= %s",
                           (min(dates), max(dates)))

            count = 0
            for _, row in df.iterrows():
                columns = [col for col in row.index if col != 'date' and pd.notna(row[col])]
                columns.insert(0, 'date')

                values = [row['date']]
                for col in columns[1:]:
                    val = row[col]
                    if pd.isna(val):
                        values.append(None)
                    elif isinstance(val, (np.floating, float)):
                        values.append(float(val))
                    elif isinstance(val, (np.integer, int)):
                        values.append(int(val))
                    else:
                        values.append(val)

                placeholders = ', '.join(['%s'] * len(columns))
                query = f"INSERT INTO market_breadth ({', '.join(columns)}) VALUES ({placeholders})"
                cur.execute(query, values)
                count += 1

            conn.commit()
            return count
        finally:
            conn.close()

    def process(self, start_date: Optional[date] = None):
        """Process and save market breadth data."""
        df = self.calculate_all_breadth(start_date)
        if df.empty:
            logger.warning("No breadth data calculated")
            return 0

        count = self.save_breadth(df)
        logger.info(f"Saved {count} market breadth records")
        return count


def main():
    parser = argparse.ArgumentParser(description='Calculate market breadth indicators')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')

    args = parser.parse_args()

    start_date = None
    if args.start:
        start_date = date.fromisoformat(args.start)

    calc = BreadthCalculator()
    calc.process(start_date)


if __name__ == '__main__':
    main()
