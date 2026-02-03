"""
Database Analysis Tools for Financial AI Assistant
All queries use LOCAL database only - no external API calls

Architecture:
- symbols: code, name, exchange_id
- fundamentals: market_cap, PE, sector, industry, margins, growth
- price_history: OHLCV for calculating technical indicators
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import ta


class DatabaseAnalyzer:
    """
    Analysis tools using ONLY local database.
    No external API calls during queries.
    """

    def __init__(self, db_path: str = "data/financial_data.db"):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    # =========================================================================
    # BASIC QUERIES
    # =========================================================================

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get basic symbol info from symbols + fundamentals"""
        conn = self._get_connection()
        query = """
            SELECT
                s.id,
                s.code,
                s.name,
                f.sector,
                f.industry,
                f.market_cap,
                f.pe_ratio,
                f.forward_pe,
                f.peg_ratio,
                f.price_to_book,
                f.dividend_yield,
                f.profit_margin,
                f.operating_margin,
                f.revenue_growth,
                f.earnings_growth,
                f.employees,
                f.description
            FROM symbols s
            LEFT JOIN fundamentals f ON s.id = f.symbol_id
            WHERE UPPER(s.code) = UPPER(?)
        """
        cursor = conn.cursor()
        cursor.execute(query, (symbol,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row[0],
            "symbol": row[1],
            "name": row[2],
            "sector": row[3],
            "industry": row[4],
            "market_cap": row[5],
            "market_cap_b": round(row[5] / 1e9, 2) if row[5] else None,
            "pe_ratio": round(row[6], 2) if row[6] else None,
            "forward_pe": round(row[7], 2) if row[7] else None,
            "peg_ratio": round(row[8], 2) if row[8] else None,
            "price_to_book": round(row[9], 2) if row[9] else None,
            "dividend_yield_pct": round(row[10] * 100, 2) if row[10] else None,
            "profit_margin_pct": round(row[11] * 100, 2) if row[11] else None,
            "operating_margin_pct": round(row[12] * 100, 2) if row[12] else None,
            "revenue_growth_pct": round(row[13] * 100, 2) if row[13] else None,
            "earnings_growth_pct": round(row[14] * 100, 2) if row[14] else None,
            "employees": row[15],
            "description": row[16][:500] if row[16] else None
        }

    def get_price_history(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Get price history for technical analysis"""
        conn = self._get_connection()
        query = """
            SELECT
                ph.date,
                ph.open,
                ph.high,
                ph.low,
                ph.close,
                ph.volume
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE UPPER(s.code) = UPPER(?)
            ORDER BY ph.date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[symbol, days])
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df

    # =========================================================================
    # TECHNICAL INDICATORS (calculated from price_history)
    # =========================================================================

    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate RSI from price_history"""
        df = self.get_price_history(symbol, days=period * 4)

        if len(df) < period + 1:
            return None

        rsi = ta.momentum.RSIIndicator(df['close'], window=period).rsi()

        if rsi.empty or pd.isna(rsi.iloc[-1]):
            return None

        return round(rsi.iloc[-1], 2)

    def calculate_macd(self, symbol: str) -> Optional[Dict]:
        """Calculate MACD from price_history"""
        df = self.get_price_history(symbol, days=60)

        if len(df) < 26:
            return None

        macd_ind = ta.trend.MACD(df['close'])
        macd_val = macd_ind.macd().iloc[-1]
        signal_val = macd_ind.macd_signal().iloc[-1]

        if pd.isna(macd_val):
            return None

        return {
            "macd": round(macd_val, 4),
            "signal": round(signal_val, 4) if not pd.isna(signal_val) else None,
            "histogram": round(macd_ind.macd_diff().iloc[-1], 4)
        }

    def get_technical_indicators(self, symbol: str) -> Optional[Dict]:
        """Get all technical indicators for a symbol"""
        df = self.get_price_history(symbol, days=100)

        if len(df) < 26:
            return {"error": f"Insufficient data for {symbol}"}

        result = {
            "symbol": symbol,
            "last_price": round(df['close'].iloc[-1], 2),
            "last_date": df['date'].iloc[-1].strftime('%Y-%m-%d'),
        }

        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        result['rsi_14'] = round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else None

        # MACD
        macd = ta.trend.MACD(df['close'])
        result['macd'] = round(macd.macd().iloc[-1], 4) if not pd.isna(macd.macd().iloc[-1]) else None
        result['macd_signal'] = round(macd.macd_signal().iloc[-1], 4) if not pd.isna(macd.macd_signal().iloc[-1]) else None

        # Moving Averages
        result['sma_20'] = round(df['close'].rolling(20).mean().iloc[-1], 2)
        if len(df) >= 50:
            result['sma_50'] = round(df['close'].rolling(50).mean().iloc[-1], 2)

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        result['bb_upper'] = round(bb.bollinger_hband().iloc[-1], 2)
        result['bb_lower'] = round(bb.bollinger_lband().iloc[-1], 2)
        result['bb_mid'] = round(bb.bollinger_mavg().iloc[-1], 2)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        result['stoch_k'] = round(stoch.stoch().iloc[-1], 2) if not pd.isna(stoch.stoch().iloc[-1]) else None
        result['stoch_d'] = round(stoch.stoch_signal().iloc[-1], 2) if not pd.isna(stoch.stoch_signal().iloc[-1]) else None

        # ATR
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
        result['atr'] = round(atr.average_true_range().iloc[-1], 2) if not pd.isna(atr.average_true_range().iloc[-1]) else None

        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        result['adx'] = round(adx.adx().iloc[-1], 2) if not pd.isna(adx.adx().iloc[-1]) else None

        # Williams %R
        williams = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'])
        result['williams_r'] = round(williams.williams_r().iloc[-1], 2) if not pd.isna(williams.williams_r().iloc[-1]) else None

        # CCI
        cci = ta.trend.CCIIndicator(df['high'], df['low'], df['close'])
        result['cci'] = round(cci.cci().iloc[-1], 2) if not pd.isna(cci.cci().iloc[-1]) else None

        # OBV
        if 'volume' in df.columns and df['volume'].sum() > 0:
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            result['obv'] = int(obv.on_balance_volume().iloc[-1])

        return result

    # =========================================================================
    # SCREENING (fundamentals + technical)
    # =========================================================================

    def screen_stocks(
        self,
        min_market_cap_b: float = None,
        max_market_cap_b: float = None,
        sector: str = None,
        industry: str = None,
        rsi_below: float = None,
        rsi_above: float = None,
        pe_below: float = None,
        pe_above: float = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Screen stocks using fundamentals + technical indicators.
        All data from local database.
        """
        conn = self._get_connection()

        # Build query for fundamentals
        query = """
            SELECT
                s.code,
                s.name,
                f.sector,
                f.industry,
                f.market_cap,
                f.pe_ratio
            FROM symbols s
            JOIN fundamentals f ON s.id = f.symbol_id
            WHERE f.market_cap IS NOT NULL AND f.market_cap > 0
        """
        params = []

        if min_market_cap_b:
            query += " AND f.market_cap >= ?"
            params.append(min_market_cap_b * 1e9)

        if max_market_cap_b:
            query += " AND f.market_cap <= ?"
            params.append(max_market_cap_b * 1e9)

        if sector:
            query += " AND UPPER(f.sector) LIKE UPPER(?)"
            params.append(f"%{sector}%")

        if industry:
            query += " AND UPPER(f.industry) LIKE UPPER(?)"
            params.append(f"%{industry}%")

        if pe_below:
            query += " AND f.pe_ratio < ? AND f.pe_ratio > 0"
            params.append(pe_below)

        if pe_above:
            query += " AND f.pe_ratio > ?"
            params.append(pe_above)

        # For RSI filtering, scan all qualifying stocks (RSI outliers can be anywhere)
        # RSI < 30 or > 70 are rare conditions, so we need to scan comprehensively
        if rsi_below or rsi_above:
            # No limit when filtering by RSI - scan all matching fundamentals
            query += " ORDER BY f.market_cap DESC"
        else:
            query += " ORDER BY f.market_cap DESC LIMIT ?"
            params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # If no RSI filter, return fundamentals only
        if not rsi_below and not rsi_above:
            return [
                {
                    "symbol": row['code'],
                    "name": row['name'][:40] if row['name'] else None,
                    "sector": row['sector'],
                    "industry": row['industry'],
                    "market_cap_b": round(row['market_cap'] / 1e9, 2),
                    "pe_ratio": round(row['pe_ratio'], 2) if row['pe_ratio'] else None
                }
                for _, row in df.head(limit).iterrows()
            ]

        # Calculate RSI and filter
        results = []
        for _, row in df.iterrows():
            rsi = self.calculate_rsi(row['code'])

            if rsi is None:
                continue

            # Apply RSI filters
            if rsi_below and rsi >= rsi_below:
                continue
            if rsi_above and rsi <= rsi_above:
                continue

            results.append({
                "symbol": row['code'],
                "name": row['name'][:40] if row['name'] else None,
                "sector": row['sector'],
                "market_cap_b": round(row['market_cap'] / 1e9, 2),
                "pe_ratio": round(row['pe_ratio'], 2) if row['pe_ratio'] else None,
                "rsi": rsi
            })

            if len(results) >= limit:
                break

        return results

    def find_oversold(self, min_market_cap_b: float = 10, rsi_threshold: float = 30, limit: int = 20) -> List[Dict]:
        """Find oversold stocks (RSI below threshold)"""
        return self.screen_stocks(
            min_market_cap_b=min_market_cap_b,
            rsi_below=rsi_threshold,
            limit=limit
        )

    def find_overbought(self, min_market_cap_b: float = 10, rsi_threshold: float = 70, limit: int = 20) -> List[Dict]:
        """Find overbought stocks (RSI above threshold)"""
        return self.screen_stocks(
            min_market_cap_b=min_market_cap_b,
            rsi_above=rsi_threshold,
            limit=limit
        )

    def screen_by_performance(
        self,
        min_market_cap_b: float = None,
        year: int = None,
        period_days: int = None,
        min_return_pct: float = None,
        max_return_pct: float = None,
        sector: str = None,
        sort_by: str = 'return_desc',  # 'return_desc', 'return_asc', 'market_cap'
        limit: int = 50
    ) -> List[Dict]:
        """
        Screen stocks by performance (return) over a period.

        Args:
            min_market_cap_b: Minimum market cap in billions
            year: Specific year (e.g., 2025)
            period_days: Alternative to year - last N days
            min_return_pct: Minimum return percentage
            max_return_pct: Maximum return percentage
            sector: Filter by sector
            sort_by: Sort order
            limit: Max results
        """
        conn = self._get_connection()

        # Get symbols with fundamentals
        query = """
            SELECT s.id, s.code, s.name, f.sector, f.market_cap
            FROM symbols s
            JOIN fundamentals f ON s.id = f.symbol_id
            WHERE f.market_cap IS NOT NULL AND f.market_cap > 0
        """
        params = []

        if min_market_cap_b:
            query += " AND f.market_cap >= ?"
            params.append(min_market_cap_b * 1e9)

        if sector:
            query += " AND UPPER(f.sector) LIKE UPPER(?)"
            params.append(f"%{sector}%")

        query += " ORDER BY f.market_cap DESC"

        df_symbols = pd.read_sql_query(query, conn, params=params)

        # Determine date range
        if year:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
        elif period_days:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
        else:
            # Default: last 365 days
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        results = []

        for _, row in df_symbols.iterrows():
            symbol_id = row['id']

            # Get first and last price in the period
            price_query = """
                SELECT
                    (SELECT close FROM price_history
                     WHERE symbol_id = ? AND date >= ?
                     ORDER BY date ASC LIMIT 1) as start_price,
                    (SELECT close FROM price_history
                     WHERE symbol_id = ? AND date <= ?
                     ORDER BY date DESC LIMIT 1) as end_price
            """
            cursor = conn.cursor()
            cursor.execute(price_query, (symbol_id, start_date, symbol_id, end_date))
            price_row = cursor.fetchone()

            if not price_row or not price_row[0] or not price_row[1]:
                continue

            start_price, end_price = price_row
            return_pct = ((end_price - start_price) / start_price) * 100

            # Apply return filters
            if min_return_pct is not None and return_pct < min_return_pct:
                continue
            if max_return_pct is not None and return_pct > max_return_pct:
                continue

            results.append({
                'symbol': row['code'],
                'name': row['name'][:35] if row['name'] else None,
                'sector': row['sector'],
                'market_cap_b': round(row['market_cap'] / 1e9, 2),
                'start_price': round(start_price, 2),
                'end_price': round(end_price, 2),
                'return_pct': round(return_pct, 2)
            })

        conn.close()

        # Sort
        if sort_by == 'return_desc':
            results.sort(key=lambda x: x['return_pct'], reverse=True)
        elif sort_by == 'return_asc':
            results.sort(key=lambda x: x['return_pct'])
        elif sort_by == 'market_cap':
            results.sort(key=lambda x: x['market_cap_b'], reverse=True)

        return results[:limit]

    def get_top_gainers(self, year: int = None, period_days: int = 365, min_market_cap_b: float = 5, limit: int = 20) -> List[Dict]:
        """Get top gaining stocks"""
        return self.screen_by_performance(
            min_market_cap_b=min_market_cap_b,
            year=year,
            period_days=period_days if not year else None,
            min_return_pct=0,
            sort_by='return_desc',
            limit=limit
        )

    def get_top_losers(self, year: int = None, period_days: int = 365, min_market_cap_b: float = 5, limit: int = 20) -> List[Dict]:
        """Get top losing stocks"""
        return self.screen_by_performance(
            min_market_cap_b=min_market_cap_b,
            year=year,
            period_days=period_days if not year else None,
            max_return_pct=0,
            sort_by='return_asc',
            limit=limit
        )

    # =========================================================================
    # MARKET STATISTICS
    # =========================================================================

    def get_market_stats(self) -> Dict:
        """Get overall market statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Total counts
        cursor.execute("SELECT COUNT(*) FROM symbols")
        stats['total_symbols'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fundamentals WHERE market_cap > 0")
        stats['symbols_with_fundamentals'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT symbol_id) FROM price_history")
        stats['symbols_with_prices'] = cursor.fetchone()[0]

        # By sector
        query = """
            SELECT sector, COUNT(*) as count
            FROM fundamentals
            WHERE market_cap > 0 AND sector IS NOT NULL
            GROUP BY sector
            ORDER BY count DESC
        """
        df = pd.read_sql_query(query, conn)
        stats['by_sector'] = df.to_dict('records')

        # Market cap distribution
        query = """
            SELECT
                CASE
                    WHEN market_cap >= 200e9 THEN 'Mega (>200B)'
                    WHEN market_cap >= 10e9 THEN 'Large (10-200B)'
                    WHEN market_cap >= 2e9 THEN 'Mid (2-10B)'
                    ELSE 'Small (<2B)'
                END as category,
                COUNT(*) as count
            FROM fundamentals
            WHERE market_cap > 0
            GROUP BY category
        """
        df = pd.read_sql_query(query, conn)
        stats['by_market_cap'] = df.to_dict('records')

        conn.close()
        return stats

    # =========================================================================
    # PORTFOLIO QUERIES
    # =========================================================================

    def get_portfolio_holdings(self, fecha: str = None) -> List[Dict]:
        """Get portfolio holdings from holding_diario"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if not fecha:
            cursor.execute("SELECT MAX(fecha) FROM holding_diario")
            fecha = cursor.fetchone()[0]

        query = """
            SELECT account_code, symbol, shares, precio_entrada, currency
            FROM holding_diario
            WHERE fecha = ?
            ORDER BY account_code, ABS(shares * precio_entrada) DESC
        """
        cursor.execute(query, (fecha,))

        holdings = []
        for row in cursor.fetchall():
            holdings.append({
                "account": row[0],
                "symbol": row[1],
                "shares": row[2],
                "entry_price": row[3],
                "currency": row[4],
                "value": round(row[2] * row[3], 2) if row[2] and row[3] else 0
            })

        conn.close()
        return {"fecha": fecha, "holdings": holdings}

    def get_portfolio_summary(self, fecha: str = None) -> Dict:
        """Get portfolio summary from posicion"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if not fecha:
            cursor.execute("SELECT MAX(fecha) FROM posicion")
            fecha = cursor.fetchone()[0]

        cursor.execute("""
            SELECT account_code, holding_eur, cash_eur, total_eur
            FROM posicion WHERE fecha = ?
            ORDER BY total_eur DESC
        """, (fecha,))

        accounts = []
        total_h = total_c = total_t = 0
        for row in cursor.fetchall():
            accounts.append({
                "account": row[0],
                "holdings_eur": round(row[1], 2) if row[1] else 0,
                "cash_eur": round(row[2], 2) if row[2] else 0,
                "total_eur": round(row[3], 2) if row[3] else 0
            })
            total_h += row[1] or 0
            total_c += row[2] or 0
            total_t += row[3] or 0

        conn.close()
        return {
            "fecha": fecha,
            "accounts": accounts,
            "total_holdings_eur": round(total_h, 2),
            "total_cash_eur": round(total_c, 2),
            "total_portfolio_eur": round(total_t, 2)
        }

    def get_cash_positions(self, fecha: str = None) -> Dict:
        """Get cash positions from cash_diario"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if not fecha:
            cursor.execute("SELECT MAX(fecha) FROM cash_diario")
            fecha = cursor.fetchone()[0]

        cursor.execute("""
            SELECT account_code, currency, saldo
            FROM cash_diario WHERE fecha = ?
            ORDER BY account_code, currency
        """, (fecha,))

        cash = []
        for row in cursor.fetchall():
            cash.append({
                "account": row[0],
                "currency": row[1],
                "balance": round(row[2], 2) if row[2] else 0
            })

        conn.close()
        return {"fecha": fecha, "cash": cash}


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    analyzer = DatabaseAnalyzer()

    print("=== Market Stats ===")
    stats = analyzer.get_market_stats()
    print(f"Total symbols: {stats['total_symbols']}")
    print(f"With fundamentals: {stats['symbols_with_fundamentals']}")
    print(f"With prices: {stats['symbols_with_prices']}")

    print("\n=== Test: RSI < 20, Cap > 9.87B ===")
    results = analyzer.screen_stocks(min_market_cap_b=9.87, rsi_below=20, limit=20)
    print(f"Found: {len(results)}")
    for r in results:
        print(f"  {r['symbol']}: RSI={r['rsi']}, Cap=${r['market_cap_b']}B")

    print("\n=== Test: Technical Indicators AAPL ===")
    tech = analyzer.get_technical_indicators("AAPL")
    if tech:
        for k, v in tech.items():
            print(f"  {k}: {v}")
