"""
Technical Features Calculator - Simplified Version (60 features)
Calculates essential technical indicators for ML/AI Trading System.

Usage:
    python -m src.features_calculator --symbol AAPL
    python -m src.features_calculator --all --limit 100
    python -m src.features_calculator --test
"""

import logging
import argparse
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class TechnicalFeaturesCalculator:
    """Calculate 60 essential technical features for stocks."""

    def __init__(self, db_url: str = FMP_DATABASE_URL):
        self.db_url = db_url

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def get_price_data(self, symbol: str, min_days: int = 300) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol."""
        conn = self.get_connection()
        try:
            query = """
                SELECT date, open, high, low, close, volume
                FROM fmp_price_history
                WHERE symbol = %s
                ORDER BY date
            """
            df = pd.read_sql(query, conn, params=(symbol,))
            if len(df) < min_days:
                return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    def get_all_symbols(self, limit: Optional[int] = None) -> list:
        """Get symbols with sufficient price data."""
        conn = self.get_connection()
        try:
            query = """
                SELECT symbol, COUNT(*) as cnt
                FROM fmp_price_history
                GROUP BY symbol
                HAVING COUNT(*) >= 300
                ORDER BY symbol
            """
            if limit:
                query = query.replace("ORDER BY symbol", f"ORDER BY symbol LIMIT {limit}")
            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    # ===================
    # INDICATOR FUNCTIONS
    # ===================

    def sma(self, s: pd.Series, p: int) -> pd.Series:
        return s.rolling(window=p, min_periods=p).mean()

    def ema(self, s: pd.Series, p: int) -> pd.Series:
        return s.ewm(span=p, adjust=False).mean()

    def rsi(self, s: pd.Series, p: int = 14) -> pd.Series:
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=p, adjust=False).mean()
        loss = (-delta).where(delta < 0, 0.0).ewm(span=p, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd_histogram(self, s: pd.Series) -> pd.Series:
        macd = self.ema(s, 12) - self.ema(s, 26)
        signal = self.ema(macd, 9)
        return macd - signal

    def macd_cross_signal(self, s: pd.Series) -> pd.Series:
        macd = self.ema(s, 12) - self.ema(s, 26)
        signal = self.ema(macd, 9)
        cross = pd.Series(0, index=s.index)
        cross[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 1  # Cross up
        cross[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = -1  # Cross down
        return cross

    def stochastic(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14) -> pd.Series:
        lowest = l.rolling(window=p, min_periods=p).min()
        highest = h.rolling(window=p, min_periods=p).max()
        return 100 * (c - lowest) / (highest - lowest)

    def williams_r(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14) -> pd.Series:
        highest = h.rolling(window=p, min_periods=p).max()
        lowest = l.rolling(window=p, min_periods=p).min()
        return -100 * (highest - c) / (highest - lowest)

    def cci(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 20) -> pd.Series:
        tp = (h + l + c) / 3
        sma_tp = self.sma(tp, p)
        mad = tp.rolling(window=p, min_periods=p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma_tp) / (0.015 * mad)

    def mfi(self, h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, p: int = 14) -> pd.Series:
        tp = (h + l + c) / 3
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(window=p, min_periods=p).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(window=p, min_periods=p).sum()
        return 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

    def atr(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14) -> pd.Series:
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=p, min_periods=p).mean()

    def adx(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14):
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        up = h - h.shift(1)
        down = l.shift(1) - l
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        atr_val = tr.ewm(span=p, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=p, adjust=False).mean() / atr_val
        minus_di = 100 * minus_dm.ewm(span=p, adjust=False).mean() / atr_val
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.ewm(span=p, adjust=False).mean()
        return adx_val, plus_di, minus_di

    def bollinger(self, s: pd.Series, p: int = 20, std: float = 2.0):
        mid = self.sma(s, p)
        sd = s.rolling(window=p, min_periods=p).std()
        upper = mid + (sd * std)
        lower = mid - (sd * std)
        width = (upper - lower) / mid
        position = (s - lower) / (upper - lower)
        return width, position

    def keltner_position(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 20) -> pd.Series:
        mid = self.ema(c, p)
        atr_val = self.atr(h, l, c, p)
        upper = mid + (atr_val * 2)
        lower = mid - (atr_val * 2)
        return (c - lower) / (upper - lower)

    def obv(self, c: pd.Series, v: pd.Series) -> pd.Series:
        direction = np.sign(c.diff()).fillna(0)
        return (direction * v).cumsum()

    def obv_slope(self, c: pd.Series, v: pd.Series, p: int = 5) -> pd.Series:
        obv_val = self.obv(c, v)
        return (obv_val - obv_val.shift(p)) / p

    def adl(self, h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series) -> pd.Series:
        mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        return (mfm * v).cumsum()

    def adl_slope(self, h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, p: int = 5) -> pd.Series:
        adl_val = self.adl(h, l, c, v)
        return (adl_val - adl_val.shift(p)) / p

    def cmf(self, h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, p: int = 21) -> pd.Series:
        mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        mfv = mfm * v
        return mfv.rolling(window=p, min_periods=p).sum() / v.rolling(window=p, min_periods=p).sum()

    def psar_signal(self, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        """Simplified PSAR: 1=bullish, -1=bearish."""
        length = len(c)
        signal = pd.Series(0, index=c.index)
        af, af_step, af_max = 0.02, 0.02, 0.2
        is_long = True
        ep = l.iloc[0]
        hp, lp = h.iloc[0], l.iloc[0]
        psar = c.iloc[0]

        for i in range(1, length):
            if is_long:
                psar = psar + af * (hp - psar)
                psar = min(psar, l.iloc[i-1], l.iloc[max(0, i-2)])
                if l.iloc[i] < psar:
                    is_long = False
                    psar = hp
                    lp = l.iloc[i]
                    af = af_step
                elif h.iloc[i] > hp:
                    hp = h.iloc[i]
                    af = min(af + af_step, af_max)
            else:
                psar = psar + af * (lp - psar)
                psar = max(psar, h.iloc[i-1], h.iloc[max(0, i-2)])
                if h.iloc[i] > psar:
                    is_long = True
                    psar = lp
                    hp = h.iloc[i]
                    af = af_step
                elif l.iloc[i] < lp:
                    lp = l.iloc[i]
                    af = min(af + af_step, af_max)
            signal.iloc[i] = 1 if is_long else -1
        return signal

    def aroon_oscillator(self, h: pd.Series, l: pd.Series, p: int = 25) -> pd.Series:
        aroon_osc = pd.Series(index=h.index, dtype=float)
        for i in range(p, len(h)):
            hw = h.iloc[i-p:i+1]
            lw = l.iloc[i-p:i+1]
            days_high = p - hw.argmax()
            days_low = p - lw.argmin()
            up = 100 * (p - days_high) / p
            down = 100 * (p - days_low) / p
            aroon_osc.iloc[i] = up - down
        return aroon_osc

    def vortex_signal(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14) -> pd.Series:
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        vm_plus = abs(h - l.shift(1))
        vm_minus = abs(l - h.shift(1))
        tr_sum = tr.rolling(window=p, min_periods=p).sum()
        vi_plus = vm_plus.rolling(window=p, min_periods=p).sum() / tr_sum
        vi_minus = vm_minus.rolling(window=p, min_periods=p).sum() / tr_sum
        return vi_plus - vi_minus

    def ichimoku_signal(self, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        """1=above cloud, -1=below cloud, 0=inside."""
        senkou_a = ((h.rolling(9).max() + l.rolling(9).min()) / 2 +
                    (h.rolling(26).max() + l.rolling(26).min()) / 2) / 2
        senkou_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        signal = pd.Series(0, index=c.index)
        signal[c > cloud_top] = 1
        signal[c < cloud_bottom] = -1
        return signal

    def ttm_squeeze(self, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        """True when Bollinger inside Keltner."""
        bb_width, _ = self.bollinger(c, 20, 2.0)
        mid = self.ema(c, 20)
        atr_val = self.atr(h, l, c, 20)
        kc_width = (4 * atr_val) / mid
        return bb_width < kc_width

    def elder_signal(self, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        """Bull Power + Bear Power."""
        ema13 = self.ema(c, 13)
        return (h - ema13) + (l - ema13)

    def donchian_breakout(self, h: pd.Series, l: pd.Series, c: pd.Series, p: int = 20) -> pd.Series:
        """1=breakout up, -1=breakout down, 0=none."""
        upper = h.rolling(window=p, min_periods=p).max().shift(1)
        lower = l.rolling(window=p, min_periods=p).min().shift(1)
        signal = pd.Series(0, index=c.index)
        signal[c >= upper] = 1
        signal[c <= lower] = -1
        return signal

    def ultimate_oscillator(self, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        return 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7

    # ================
    # ZONE CLASSIFIERS
    # ================

    def zone_rsi(self, val: pd.Series) -> pd.Series:
        return pd.Series(np.select(
            [val < 30, val > 70],
            ['oversold', 'overbought'],
            default='neutral'
        ), index=val.index)

    def zone_stoch(self, val: pd.Series) -> pd.Series:
        return pd.Series(np.select(
            [val < 20, val > 80],
            ['oversold', 'overbought'],
            default='neutral'
        ), index=val.index)

    def zone_cci(self, val: pd.Series) -> pd.Series:
        return pd.Series(np.select(
            [val < -200, val < -100, val > 200, val > 100],
            ['extreme_oversold', 'oversold', 'extreme_overbought', 'overbought'],
            default='neutral'
        ), index=val.index)

    def zone_mfi(self, val: pd.Series) -> pd.Series:
        return self.zone_stoch(val)

    def zone_bb(self, position: pd.Series) -> pd.Series:
        return pd.Series(np.select(
            [position < 0, position > 1],
            ['below', 'above'],
            default='inside'
        ), index=position.index)

    def zone_trend(self, adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
        direction = np.sign(plus_di - minus_di)
        return pd.Series(np.select(
            [(adx > 25) & (direction > 0), (adx > 25) & (direction < 0),
             (adx > 40) & (direction > 0), (adx > 40) & (direction < 0)],
            ['up', 'down', 'strong_up', 'strong_down'],
            default='neutral'
        ), index=adx.index)

    def zone_volatility(self, vol: pd.Series) -> pd.Series:
        pct_20 = vol.rolling(window=252, min_periods=60).quantile(0.2)
        pct_80 = vol.rolling(window=252, min_periods=60).quantile(0.8)
        return pd.Series(np.select(
            [vol < pct_20, vol > pct_80],
            ['low', 'high'],
            default='normal'
        ), index=vol.index)

    def zone_volume(self, ratio: pd.Series) -> pd.Series:
        return pd.Series(np.select(
            [ratio < 0.5, ratio > 2.0],
            ['low', 'high'],
            default='normal'
        ), index=ratio.index)

    # =====================
    # MAIN CALCULATION
    # =====================

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 60 features."""
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume'].astype(float)

        f = pd.DataFrame(index=df.index)

        # Base
        f['open'], f['high'], f['low'], f['close'], f['volume'] = o, h, l, c, v.astype(int)

        # Tendencia
        f['sma_20'] = self.sma(c, 20)
        f['sma_50'] = self.sma(c, 50)
        f['sma_200'] = self.sma(c, 200)
        f['ema_12'] = self.ema(c, 12)
        f['ema_26'] = self.ema(c, 26)
        f['above_sma_20'] = c > f['sma_20']
        f['above_sma_50'] = c > f['sma_50']
        f['above_sma_200'] = c > f['sma_200']
        adx_val, plus_di, minus_di = self.adx(h, l, c)
        f['trend_strength'] = adx_val
        f['trend_direction'] = np.sign(plus_di - minus_di).astype(int)

        # Momentum
        f['rsi'] = self.rsi(c, 14)
        f['rsi_2'] = self.rsi(c, 2)
        f['macd_hist'] = self.macd_histogram(c)
        f['stoch'] = self.stochastic(h, l, c)
        f['williams'] = self.williams_r(h, l, c)
        f['cci'] = self.cci(h, l, c)
        f['mfi'] = self.mfi(h, l, c, v)
        f['ultimate_osc'] = self.ultimate_oscillator(h, l, c)

        # Zonas
        f['rsi_zone'] = self.zone_rsi(f['rsi'])
        f['stoch_zone'] = self.zone_stoch(f['stoch'])
        f['cci_zone'] = self.zone_cci(f['cci'])
        f['mfi_zone'] = self.zone_mfi(f['mfi'])
        bb_width, bb_pos = self.bollinger(c)
        f['bb_zone'] = self.zone_bb(bb_pos)
        f['trend_zone'] = self.zone_trend(adx_val, plus_di, minus_di)
        vol_20d = c.pct_change().rolling(20).std() * np.sqrt(252)
        f['volatility_zone'] = self.zone_volatility(vol_20d)
        vol_ratio = v / self.sma(v, 20)
        f['volume_zone'] = self.zone_volume(vol_ratio)

        # Volatilidad
        f['atr'] = self.atr(h, l, c)
        f['atr_pct'] = f['atr'] / c * 100
        f['bb_width'] = bb_width
        f['bb_position'] = bb_pos
        f['volatility_20d'] = vol_20d
        f['keltner_position'] = self.keltner_position(h, l, c)

        # Volumen
        f['volume_ratio'] = vol_ratio
        f['obv_slope'] = self.obv_slope(c, v)
        f['cmf'] = self.cmf(h, l, c, v)
        f['adl_slope'] = self.adl_slope(h, l, c, v)

        # Senales
        f['psar_signal'] = self.psar_signal(h, l, c)
        f['macd_signal'] = self.macd_cross_signal(c)
        f['aroon_signal'] = self.aroon_oscillator(h, l)
        f['vortex_signal'] = self.vortex_signal(h, l, c)
        f['ichimoku_signal'] = self.ichimoku_signal(h, l, c)
        f['squeeze_on'] = self.ttm_squeeze(h, l, c)
        f['elder_signal'] = self.elder_signal(h, l, c)
        f['donchian_breakout'] = self.donchian_breakout(h, l, c)

        # Price Action
        f['daily_return'] = c.pct_change()
        f['gap'] = (o - c.shift(1)) / c.shift(1)
        f['crash'] = f['daily_return'] < -0.05
        f['spike'] = f['daily_return'] > 0.05
        rolling_max = c.rolling(window=20, min_periods=1).max()
        f['drawdown_20d'] = (c - rolling_max) / rolling_max
        high_52w = h.rolling(window=252, min_periods=50).max()
        low_52w = l.rolling(window=252, min_periods=50).min()
        f['dist_52w_high'] = (c - high_52w) / high_52w
        f['dist_52w_low'] = (c - low_52w) / low_52w
        f['new_high'] = c >= high_52w
        f['new_low'] = c <= low_52w

        return f

    def save_features(self, symbol: str, features: pd.DataFrame) -> int:
        """Save features to database."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM features_technical WHERE symbol = %s", (symbol,))

            columns = ['symbol', 'date'] + list(features.columns)
            values = []

            for date, row in features.iterrows():
                record = [symbol, date.date() if hasattr(date, 'date') else date]
                for col in features.columns:
                    val = row[col]
                    if pd.isna(val):
                        record.append(None)
                    elif isinstance(val, (np.bool_, bool)):
                        record.append(bool(val))
                    elif isinstance(val, (np.floating, float)):
                        record.append(float(val) if not np.isnan(val) else None)
                    elif isinstance(val, (np.integer, int)):
                        record.append(int(val))
                    else:
                        record.append(str(val) if val else None)
                values.append(tuple(record))

            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO features_technical ({', '.join(columns)}) VALUES ({placeholders})"
            cur.executemany(query, values)
            conn.commit()
            return len(values)
        finally:
            conn.close()

    def process_symbol(self, symbol: str) -> bool:
        """Process single symbol."""
        try:
            df = self.get_price_data(symbol)
            if df.empty:
                logger.warning(f"{symbol}: Insufficient data")
                return False

            features = self.calculate_features(df)
            features = features.dropna(thresh=len(features.columns) * 0.5)

            if features.empty:
                logger.warning(f"{symbol}: No valid features")
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
    parser = argparse.ArgumentParser(description='Calculate technical features')
    parser.add_argument('--symbol', type=str, help='Single symbol')
    parser.add_argument('--all', action='store_true', help='All symbols')
    parser.add_argument('--limit', type=int, help='Limit symbols')
    parser.add_argument('--test', action='store_true', help='Test with AAPL')

    args = parser.parse_args()
    calc = TechnicalFeaturesCalculator()

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
