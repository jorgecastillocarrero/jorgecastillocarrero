"""
OPTIMIZACION 10L + 10S CON ESTRATEGIAS INDEPENDIENTES + NUEVOS INDICADORES
===========================================================================
Para cada composite score (1-10):
  - 10 LONGS ($25K c/u) con la mejor estrategia
  - 10 SHORTS ($25K c/u) con la mejor estrategia (puede ser diferente)
  Total: $250K longs + $250K shorts = $500K/semana

11 Estrategias (ambos lados):
  MR_1w, MR_2w, MOM_4w, MOM_12w, MR_RSI, VOL_UP, RANGE
  + PSAR     - Parabolic SAR
  + ST       - SuperTrend
  + BB       - Bollinger %B
  + STOCH    - Stochastic %K

Senal: viernes close | Entrada: lunes open | Salida: lunes open siguiente
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
import json
import warnings
import time
warnings.filterwarnings('ignore')

t0 = time.time()
engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

print("=" * 150)
print("  OPTIMIZACION 10L + 10S CON 11 ESTRATEGIAS INDEPENDIENTES")
print("  11 estrat x 10 scores x 2 lados | 10 longs ($25K) + 10 shorts ($25K) = $500K")
print("=" * 150)

# ============================================================
# [1/8] LOAD SPY + MARKET DATA
# ============================================================
print("\n[1/8] Cargando datos de mercado...")

with engine.connect() as conn:
    spy = pd.read_sql("""SELECT date, open, close FROM fmp_price_history
        WHERE symbol = 'SPY' AND date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])
    vix = pd.read_sql("SELECT date, close as vix FROM price_history_vix WHERE date >= '2003-01-01' ORDER BY date",
        conn, parse_dates=['date'])
    tip = pd.read_sql("""SELECT date, close as tip_close FROM fmp_price_history
        WHERE symbol = 'TIP' AND date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])
    ief = pd.read_sql("""SELECT date, close as ief_close FROM fmp_price_history
        WHERE symbol = 'IEF' AND date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])
    aaii = pd.read_sql("""SELECT date, bullish, bearish, bull_bear_spread
        FROM sentiment_aaii WHERE date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])

    with open('data/sp500_symbols.txt') as f:
        sp500_syms = [line.strip() for line in f if line.strip()]

    earnings = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s) AND eps_actual IS NOT NULL AND eps_estimated IS NOT NULL
        AND date >= '2004-01-01'
        ORDER BY date""", conn, params={"syms": sp500_syms}, parse_dates=['date'])

print(f"  SPY: {len(spy)} | VIX: {len(vix)} | TIP: {len(tip)} | IEF: {len(ief)} | AAII: {len(aaii)} | Earnings: {len(earnings)}")

# ============================================================
# [2/8] CALCULATE SPY INDICATORS + COMPOSITE SCORE
# ============================================================
print("\n[2/8] Calculando indicadores SPY y scores...")

spy = spy.sort_values('date').reset_index(drop=True)
c = spy['close']

spy['ma5'] = c.rolling(5).mean()
spy['ma10'] = c.rolling(10).mean()
spy['ma200'] = c.rolling(200).mean()
spy['dist_ma200'] = (c - spy['ma200']) / spy['ma200'] * 100
spy['dist_ma10'] = (c - spy['ma10']) / spy['ma10'] * 100
spy['dist_ma5'] = (c - spy['ma5']) / spy['ma5'] * 100

delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta).clip(lower=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
spy['rsi'] = 100 - (100 / (1 + rs))

spy = spy.merge(vix, on='date', how='left')
spy['vix'] = spy['vix'].ffill()
spy = spy.merge(tip, on='date', how='left')
spy['tip_close'] = spy['tip_close'].ffill()
spy = spy.merge(ief, on='date', how='left')
spy['ief_close'] = spy['ief_close'].ffill()
spy['tip_ief_ratio'] = spy['tip_close'] / spy['ief_close']
spy['tip_ief_change_20d'] = (spy['tip_ief_ratio'] / spy['tip_ief_ratio'].shift(20) - 1) * 100
spy = spy.merge(aaii[['date', 'bull_bear_spread']], on='date', how='left')
spy['bull_bear_spread'] = spy['bull_bear_spread'].ffill()
spy = spy.dropna(subset=['ma200', 'rsi']).reset_index(drop=True)

# Earnings quarterly
earnings['quarter'] = earnings['date'].dt.to_period('Q')
earnings['beat'] = (earnings['eps_actual'] > earnings['eps_estimated']).astype(int)
quarterly = earnings.groupby('quarter').agg(
    n_earnings=('beat', 'count'),
    beat_rate=('beat', 'mean'),
    total_eps=('eps_actual', 'sum'),
).reset_index()
quarterly = quarterly[quarterly['n_earnings'] >= 50].reset_index(drop=True)
quarterly['eps_yoy'] = quarterly['total_eps'] / quarterly['total_eps'].shift(4) - 1
quarterly['eps_yoy_pct'] = quarterly['eps_yoy'] * 100
quarterly['quarter_end'] = quarterly['quarter'].apply(lambda q: q.end_time.date())
quarterly = quarterly.sort_values('quarter_end').reset_index(drop=True)

def get_quarterly_data_for_date(target_date):
    td = target_date.date() if hasattr(target_date, 'date') else target_date
    valid = quarterly[quarterly['quarter_end'] <= td]
    return valid.iloc[-1] if len(valid) > 0 else None

# --- Score functions (identical to backtest_scoring_system.py) ---
def calc_ma_score(dist):
    if pd.isna(dist): return 5
    if dist < -30: return 1
    if dist < -20: return 2
    if dist < -10: return 3
    if dist < -7:  return 4
    if dist < 7:   return 5
    if dist < 10:  return 6
    if dist < 20:  return 7
    if dist < 30:  return 8
    if dist < 40:  return 9
    return 10

def calc_market_score(row):
    return int(round((calc_ma_score(row['dist_ma200']) + calc_ma_score(row['dist_ma10']) + calc_ma_score(row['dist_ma5'])) / 3))

def calc_vix_score(v):
    if pd.isna(v): return 5
    if v > 35: return 1
    if v > 25: return 2
    if v > 20: return 3
    if v > 18: return 4
    if v > 17: return 5
    if v > 16: return 6
    if v > 15: return 7
    if v > 14: return 8
    if v > 12: return 9
    return 10

def calc_rsi_score(rsi):
    if pd.isna(rsi): return 5
    if rsi < 20: return 1
    if rsi < 30: return 2
    if rsi < 40: return 3
    if rsi < 50: return 4
    if rsi < 55: return 5
    if rsi < 62: return 6
    if rsi < 70: return 7
    if rsi < 78: return 8
    if rsi < 85: return 9
    return 10

def calc_eps_growth_score(eps_yoy_pct):
    if pd.isna(eps_yoy_pct): return 5
    if eps_yoy_pct < -20: return 1
    if eps_yoy_pct < -10: return 2
    if eps_yoy_pct < -3:  return 3
    if eps_yoy_pct < 3:   return 4
    if eps_yoy_pct < 8:   return 5
    if eps_yoy_pct < 15:  return 6
    if eps_yoy_pct < 25:  return 7
    if eps_yoy_pct < 40:  return 8
    if eps_yoy_pct < 80:  return 9
    return 10

def calc_beat_rate_score(beat_rate):
    if pd.isna(beat_rate): return 5
    br = beat_rate * 100
    if br < 40: return 1
    if br < 50: return 2
    if br < 55: return 3
    if br < 60: return 4
    if br < 65: return 5
    if br < 70: return 6
    if br < 75: return 7
    if br < 80: return 8
    if br < 85: return 9
    return 10

def calc_inflation_score(tip_ief_chg_20d):
    if pd.isna(tip_ief_chg_20d): return 6
    if tip_ief_chg_20d > 5:   return 1
    if tip_ief_chg_20d > 3:   return 2
    if tip_ief_chg_20d > 2:   return 3
    if tip_ief_chg_20d > 1:   return 4
    if tip_ief_chg_20d > 0.3: return 5
    if tip_ief_chg_20d > -0.3: return 6
    if tip_ief_chg_20d > -1:  return 7
    if tip_ief_chg_20d > -2:  return 8
    if tip_ief_chg_20d > -3:  return 9
    return 10

def calc_sentiment_score(spread):
    if pd.isna(spread): return 5
    if spread < -25: return 1
    if spread < -15: return 2
    if spread < -5:  return 3
    if spread < 0:   return 4
    if spread < 5:   return 5
    if spread < 15:  return 6
    if spread < 25:  return 7
    if spread < 35:  return 8
    if spread < 50:  return 9
    return 10

# Apply all scores
spy['market_score'] = spy.apply(calc_market_score, axis=1)
spy['vix_score'] = spy['vix'].apply(calc_vix_score)
spy['rsi_score'] = spy['rsi'].apply(calc_rsi_score)
spy['inflation_score'] = spy['tip_ief_change_20d'].apply(calc_inflation_score)
spy['sentiment_score'] = spy['bull_bear_spread'].apply(calc_sentiment_score)

spy['eps_growth_score'] = 5
spy['beat_rate_score'] = 5
for i, row in spy.iterrows():
    qd = get_quarterly_data_for_date(row['date'])
    if qd is not None:
        spy.at[i, 'eps_growth_score'] = calc_eps_growth_score(qd['eps_yoy_pct'] if pd.notna(qd.get('eps_yoy_pct')) else np.nan)
        spy.at[i, 'beat_rate_score'] = calc_beat_rate_score(qd['beat_rate'])

# Composite score (weighted average, normalized to 1-10)
weights = {
    'market_score': 0.30, 'vix_score': 0.20, 'rsi_score': 0.10,
    'eps_growth_score': 0.15, 'beat_rate_score': 0.10,
    'inflation_score': 0.05, 'sentiment_score': 0.10,
}
spy['composite_raw'] = sum(spy[col] * w for col, w in weights.items())
comp_min = spy['composite_raw'].quantile(0.01)
comp_max = spy['composite_raw'].quantile(0.99)
spy['composite_score'] = ((spy['composite_raw'] - comp_min) / (comp_max - comp_min) * 9 + 1).clip(1, 10).round().astype(int)

print(f"  SPY con scores: {len(spy)} dias ({spy['date'].min().date()} a {spy['date'].max().date()})")
print(f"  Tiempo: {time.time()-t0:.0f}s")

# ============================================================
# [3/8] SP500 CONSTITUENTS (historical reconstruction)
# ============================================================
print("\n[3/8] Cargando constituyentes SP500...")

with open('data/sp500_constituents.json') as f:
    current_members = json.load(f)
with open('data/sp500_historical_changes.json') as f:
    all_changes = json.load(f)
all_changes.sort(key=lambda x: x.get('date', ''), reverse=True)
current_set = {d['symbol'] for d in current_members}
all_sp500_symbols = set(current_set)
for ch in all_changes:
    if ch.get('date', '') >= '2004-01-01':
        if ch.get('removedTicker'): all_sp500_symbols.add(ch['removedTicker'])
        if ch.get('symbol'): all_sp500_symbols.add(ch['symbol'])

def get_sp500_at_date(target_date):
    members = set(current_set)
    for ch in all_changes:
        if ch.get('date', '') > str(target_date):
            if ch.get('symbol') and ch['symbol'] in members: members.discard(ch['symbol'])
            if ch.get('removedTicker'): members.add(ch['removedTicker'])
    return members

sp500_cache = {}
def get_sp500_cached(d):
    k = str(d)[:7]
    if k not in sp500_cache: sp500_cache[k] = get_sp500_at_date(d)
    return sp500_cache[k]

print(f"  Simbolos historicos: {len(all_sp500_symbols)}")

# ============================================================
# [4/8] LOAD STOCK PRICES + EXTENDED INDICATORS
# ============================================================
print("\n[4/8] Cargando precios acciones SP500 (con high/low para RANGE)...")

with engine.connect() as conn:
    df = pd.read_sql("""SELECT symbol, date, open, high, low, close FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2003-01-01'
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

print(f"  Registros cargados: {len(df):,}")

def calc_parabolic_sar(high, low, close, af_start=0.02, af_max=0.20, af_step=0.02):
    """Parabolic SAR indicator. Returns (sar_values, trend: +1=bull, -1=bear)."""
    n = len(close)
    if n < 3:
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

    sar = np.full(n, np.nan)
    trend = np.ones(n)
    af_arr = np.zeros(n)
    ep = np.zeros(n)

    h = high.values
    l = low.values

    sar[0] = l[0]
    trend[0] = 1
    af_arr[0] = af_start
    ep[0] = h[0]

    for i in range(1, n):
        if trend[i-1] == 1:  # Bullish
            sar[i] = sar[i-1] + af_arr[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[i-2] if i >= 2 else l[i-1])
            if l[i] < sar[i]:  # Flip to bearish
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = l[i]
                af_arr[i] = af_start
            else:
                trend[i] = 1
                if h[i] > ep[i-1]:
                    ep[i] = h[i]
                    af_arr[i] = min(af_arr[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af_arr[i] = af_arr[i-1]
        else:  # Bearish
            sar[i] = sar[i-1] + af_arr[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[i-2] if i >= 2 else h[i-1])
            if h[i] > sar[i]:  # Flip to bullish
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = h[i]
                af_arr[i] = af_start
            else:
                trend[i] = -1
                if l[i] < ep[i-1]:
                    ep[i] = l[i]
                    af_arr[i] = min(af_arr[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af_arr[i] = af_arr[i-1]

    return pd.Series(sar, index=close.index), pd.Series(trend, index=close.index)


def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    """SuperTrend indicator. Returns (supertrend_values, direction: +1=bull, -1=bear)."""
    n = len(close)
    if n < period + 1:
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

    h = high.values
    l = low.values
    c = close.values

    # ATR
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    atr = pd.Series(tr).rolling(period).mean().values

    hl2 = (h + l) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st = np.full(n, np.nan)
    direction = np.ones(n)
    final_upper = upper.copy()
    final_lower = lower.copy()

    start = period
    st[start] = lower[start]
    direction[start] = 1

    for i in range(start + 1, n):
        # Adjust bands
        if lower[i] > final_lower[i-1] or c[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        if upper[i] < final_upper[i-1] or c[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Direction
        if direction[i-1] == 1:  # Was bullish
            if c[i] < final_lower[i]:
                direction[i] = -1
                st[i] = final_upper[i]
            else:
                direction[i] = 1
                st[i] = final_lower[i]
        else:  # Was bearish
            if c[i] > final_upper[i]:
                direction[i] = 1
                st[i] = final_lower[i]
            else:
                direction[i] = -1
                st[i] = final_upper[i]

    return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)


def calc_ind(g):
    """Extended indicators per stock for all strategies (7 original + 4 short-specific)."""
    g = g.sort_values('date').copy()
    c = g['close']
    h = g['high']
    l = g['low']

    # Multi-period returns
    g['ret_1w'] = c / c.shift(5) - 1
    g['ret_2w'] = c / c.shift(10) - 1
    g['ret_4w'] = c / c.shift(20) - 1
    g['ret_12w'] = c / c.shift(60) - 1

    # RSI(14) per stock
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g['rsi_14'] = 100 - (100 / (1 + rs))

    # Directional volatility: vol_5d * sign(ret_1w)
    daily_ret = c.pct_change()
    g['vol_5d'] = daily_ret.rolling(5).std()
    g['vol_dir'] = g['vol_5d'] * np.sign(g['ret_1w'])

    # Range position: 0 = weekly low, 1 = weekly high
    h5 = h.rolling(5).max()
    l5 = l.rolling(5).min()
    rng = h5 - l5
    g['range_pos'] = np.where(rng > 0, (c - l5) / rng, 0.5)

    # === NEW SHORT-SPECIFIC INDICATORS ===

    # Parabolic SAR distance: (close - SAR) / close * 100
    # Positive = price above SAR (bullish), Negative = below SAR (bearish)
    if len(g) >= 3:
        sar_vals, sar_trend = calc_parabolic_sar(h, l, c)
        g['psar_dist'] = (c - sar_vals) / c * 100
    else:
        g['psar_dist'] = np.nan

    # SuperTrend distance: (close - ST) / close * 100
    if len(g) >= 12:
        st_vals, st_dir = calc_supertrend(h, l, c, period=10, multiplier=3.0)
        g['st_dist'] = (c - st_vals) / c * 100
    else:
        g['st_dist'] = np.nan

    # Bollinger %B: (close - lower) / (upper - lower)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    g['bb_pctb'] = np.where(bb_width > 0, (c - bb_lower) / bb_width, 0.5)

    # Stochastic %K(14,3): (close - lowest_low_14) / (highest_high_14 - lowest_low_14) * 100
    low_14 = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    stoch_range = high_14 - low_14
    raw_k = np.where(stoch_range > 0, (c - low_14) / stoch_range * 100, 50)
    g['stoch_k'] = pd.Series(raw_k, index=c.index).rolling(3).mean()  # %K smoothed

    return g

print("  Calculando indicadores extendidos (originales + PSAR, SuperTrend, Bollinger, Stochastic)...")
df = df.groupby('symbol', group_keys=False).apply(calc_ind)
df_indexed = df.set_index(['symbol', 'date']).sort_index()
df['weekday'] = df['date'].dt.weekday

print(f"  Indicadores calculados. Tiempo: {time.time()-t0:.0f}s")

# ============================================================
# [5/8] BUILD WEEKLY FRAMEWORK
# ============================================================
print("\n[5/8] Construyendo estructura semanal...")

fridays = np.sort(df[df['weekday'] == 4]['date'].unique())
mondays = np.sort(df[df['weekday'] == 0]['date'].unique())
spy_indexed = spy.set_index('date')

weeks = []
for fri in fridays:
    next_mons = mondays[mondays > fri]
    if len(next_mons) < 2: continue
    fri_ts = pd.Timestamp(fri)
    if fri_ts not in spy_indexed.index: continue
    sr = spy_indexed.loc[fri_ts]
    if pd.isna(sr.get('market_score')): continue
    w = {
        'signal_date': fri_ts,
        'entry_date': pd.Timestamp(next_mons[0]),
        'exit_date': pd.Timestamp(next_mons[1]),
        'composite_score': int(sr['composite_score']),
    }
    weeks.append(w)

weeks_df = pd.DataFrame(weeks)
weeks_df = weeks_df[weeks_df['signal_date'] >= '2004-01-01'].reset_index(drop=True)

# Pre-build date snapshots for fast lookup
friday_data = {}
monday_data = {}
all_dates_in_index = set(df_indexed.index.get_level_values('date'))

for _, row in weeks_df.iterrows():
    sig = row['signal_date']
    if sig not in friday_data and sig in all_dates_in_index:
        friday_data[sig] = df_indexed.xs(sig, level='date', drop_level=True)
    for d in [row['entry_date'], row['exit_date']]:
        if d not in monday_data and d in all_dates_in_index:
            monday_data[d] = df_indexed.xs(d, level='date', drop_level=True)

print(f"  Semanas: {len(weeks_df)} ({weeks_df['signal_date'].min().date()} a {weeks_df['signal_date'].max().date()})")
print(f"  Snapshots: {len(friday_data)} viernes, {len(monday_data)} lunes")
print(f"  Tiempo: {time.time()-t0:.0f}s")

# ============================================================
# [6/8] STRATEGY DEFINITIONS
# ============================================================

# long_bottom=True  -> long = lowest values (oversold), short = top (overbought)
# long_bottom=False -> long = highest values (momentum), short = bottom (losers)

# 11 Estrategias comunes para ambos lados
STRATEGIES = {
    'MR_1w':   {'col': 'ret_1w',    'long_bottom': True,  'desc': 'Mean Rev 1sem'},
    'MR_2w':   {'col': 'ret_2w',    'long_bottom': True,  'desc': 'Mean Rev 2sem'},
    'MOM_4w':  {'col': 'ret_4w',    'long_bottom': False, 'desc': 'Momentum 4sem'},
    'MOM_12w': {'col': 'ret_12w',   'long_bottom': False, 'desc': 'Momentum 12sem'},
    'MR_RSI':  {'col': 'rsi_14',    'long_bottom': True,  'desc': 'Mean Rev RSI'},
    'VOL_UP':  {'col': 'vol_dir',   'long_bottom': False, 'desc': 'Vol Breakout Dir'},
    'RANGE':   {'col': 'range_pos', 'long_bottom': False, 'desc': 'Range Breakout'},
    'PSAR':    {'col': 'psar_dist', 'long_bottom': True,  'desc': 'Parabolic SAR'},
    'ST':      {'col': 'st_dist',   'long_bottom': True,  'desc': 'SuperTrend'},
    'BB':      {'col': 'bb_pctb',   'long_bottom': True,  'desc': 'Bollinger %B'},
    'STOCH':   {'col': 'stoch_k',   'long_bottom': True,  'desc': 'Stochastic %K'},
}

# Para longs y shorts se usan las mismas definiciones de estrategia
LONG_STRATEGIES = STRATEGIES
SHORT_STRATEGIES = STRATEGIES

N_LONG = 10   # 10 longs x $25K = $250K
N_SHORT = 10  # 10 shorts x $25K = $250K

print(f"\n  Estrategias: {list(STRATEGIES.keys())} ({len(STRATEGIES)})")
print(f"  Posiciones: {N_LONG} longs ($25K) + {N_SHORT} shorts ($25K) = $500K/semana")

# ============================================================
# [7/8] MAIN OPTIMIZATION LOOP (10L + 5S, estrategias separadas)
# ============================================================
print(f"\n[7/8] Ejecutando backtest 10L+10S ({len(weeks_df)} sem, {len(LONG_STRATEGIES)} estrat LONG + {len(SHORT_STRATEGIES)} estrat SHORT)...")

long_results = []
short_results = []
n_processed = 0

for _, wrow in weeks_df.iterrows():
    sig = wrow['signal_date']
    entry = wrow['entry_date']
    exit_d = wrow['exit_date']

    if sig not in friday_data or entry not in monday_data or exit_d not in monday_data:
        continue

    fri = friday_data[sig]
    en = monday_data[entry]
    ex = monday_data[exit_d]
    sp500 = get_sp500_cached(sig)
    elig = [s for s in fri.index if s in sp500]
    if len(elig) < 100:
        continue

    # Pre-compute individual stock returns (entry monday open -> exit monday open)
    stock_returns = {}
    for sym in elig:
        if sym in en.index and sym in ex.index:
            ep = en.loc[sym, 'open']
            xp = ex.loc[sym, 'open']
            if isinstance(ep, pd.Series): ep = ep.iloc[0]
            if isinstance(xp, pd.Series): xp = xp.iloc[0]
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                stock_returns[sym] = (xp - ep) / ep

    if len(stock_returns) < 20:
        continue

    available_syms = set(stock_returns.keys())
    score = wrow['composite_score']

    # --- LONG strategies (10 longs) ---
    for strat_name, strat_def in LONG_STRATEGIES.items():
        col = strat_def['col']
        long_bottom = strat_def['long_bottom']
        valid_syms = [s for s in available_syms if s in fri.index]
        if not valid_syms:
            continue
        vals = fri.loc[valid_syms, col]
        if isinstance(vals, pd.DataFrame):
            vals = vals.iloc[:, 0]
        ranking = vals.dropna().sort_values()
        if len(ranking) < 20:
            continue
        if long_bottom:
            long_syms = ranking.head(N_LONG).index.tolist()
        else:
            long_syms = ranking.tail(N_LONG).index.tolist()
        long_rets = [stock_returns[s] for s in long_syms if s in stock_returns]
        if len(long_rets) < 8:
            continue
        long_results.append({
            'date': sig, 'composite_score': score, 'strategy': strat_name,
            'ret': np.mean(long_rets),
        })

    # --- SHORT strategies (5 shorts) ---
    for strat_name, strat_def in SHORT_STRATEGIES.items():
        col = strat_def['col']
        long_bottom = strat_def['long_bottom']
        valid_syms = [s for s in available_syms if s in fri.index]
        if not valid_syms:
            continue
        vals = fri.loc[valid_syms, col]
        if isinstance(vals, pd.DataFrame):
            vals = vals.iloc[:, 0]
        ranking = vals.dropna().sort_values()
        if len(ranking) < 20:
            continue
        # For shorts: long_bottom=True -> short from TOP, long_bottom=False -> short from BOTTOM
        if long_bottom:
            short_syms = ranking.tail(N_SHORT).index.tolist()
        else:
            short_syms = ranking.head(N_SHORT).index.tolist()
        short_rets = [-stock_returns[s] for s in short_syms if s in stock_returns]
        if len(short_rets) < 8:
            continue
        short_results.append({
            'date': sig, 'composite_score': score, 'strategy': strat_name,
            'ret': np.mean(short_rets),
        })

    n_processed += 1
    if n_processed % 200 == 0:
        print(f"    Procesadas {n_processed} semanas... ({time.time()-t0:.0f}s)")

ldf = pd.DataFrame(long_results)
sdf = pd.DataFrame(short_results)
print(f"\n  Long results: {len(ldf):,} | Short results: {len(sdf):,} ({n_processed} semanas)")
print(f"  Tiempo backtest: {time.time()-t0:.0f}s")

# ============================================================
# [8/8] OUTPUT TABLES
# ============================================================

def calc_metrics(rets):
    """Calculate performance metrics for an array of weekly returns."""
    if len(rets) < 5:
        return None
    avg = np.mean(rets)
    std = np.std(rets)
    sharpe = (avg / std) * np.sqrt(52) if std > 0 else 0
    wr = np.mean(rets > 0) * 100
    annual = avg * 52 * 100
    pf = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0
    return {'n': len(rets), 'avg_ret': avg, 'annual': annual, 'sharpe': sharpe, 'wr': wr, 'pf': pf}

long_strat_names = list(LONG_STRATEGIES.keys())
short_strat_names = list(SHORT_STRATEGIES.keys())

# ================================================================
# TABLA A: Mejor estrategia LONG por Score (10 longs, Sharpe)
# ================================================================
print(f"\n\n{'='*160}")
print(f"  TABLA A: MEJOR ESTRATEGIA LONG POR SCORE ({N_LONG} longs x $25K)")
print(f"{'='*160}")

print(f"  {'Score':>5s} | {'Estrategia':>10s} | {'Sharpe':>7s} | {'Ret/sem':>8s} | {'Anual%':>7s} | {'WR%':>5s} | {'PF':>5s} | {'N':>5s} | {'2a Mejor':>10s} | {'2a Sh':>6s}")
print(f"  {'-'*5} | {'-'*10} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*10} | {'-'*6}")

long_table = {}

for score in range(1, 11):
    best_sharpe = -999
    best_combo = None
    second_sharpe = -999
    second_combo = None

    for sn in long_strat_names:
        mask = (ldf['composite_score'] == score) & (ldf['strategy'] == sn)
        rets = ldf[mask]['ret'].values
        if len(rets) < 5:
            continue
        m = calc_metrics(rets)
        if m is None:
            continue
        if m['sharpe'] > best_sharpe:
            second_sharpe = best_sharpe
            second_combo = best_combo
            best_sharpe = m['sharpe']
            best_combo = {'strategy': sn, **m}
        elif m['sharpe'] > second_sharpe:
            second_sharpe = m['sharpe']
            second_combo = {'strategy': sn, **m}

    if best_combo:
        long_table[score] = best_combo['strategy']
        sec_str = f"{second_combo['strategy']:>10s}" if second_combo else f"{'N/A':>10s}"
        sec_sh = f"{second_sharpe:>+6.2f}" if second_combo else f"{'N/A':>6s}"
        print(f"  {score:>5d} | {best_combo['strategy']:>10s} | "
              f"{best_combo['sharpe']:>+7.2f} | {best_combo['avg_ret']*100:>+7.3f}% | "
              f"{best_combo['annual']:>+6.1f}% | {best_combo['wr']:>4.0f}% | {best_combo['pf']:>5.2f} | "
              f"{best_combo['n']:>5d} | {sec_str} | {sec_sh}")
    else:
        long_table[score] = 'MR_1w'
        print(f"  {score:>5d} | {'N/A':>10s} | (fallback MR_1w)")

# ================================================================
# TABLA B: Mejor estrategia SHORT por Score (5 shorts, Sharpe)
# ================================================================
print(f"\n\n{'='*160}")
print(f"  TABLA B: MEJOR ESTRATEGIA SHORT POR SCORE ({N_SHORT} shorts x $50K)")
print(f"  11 estrategias: 7 originales + PSAR, SuperTrend, Bollinger, Stochastic")
print(f"{'='*160}")

print(f"  {'Score':>5s} | {'Estrategia':>10s} | {'Sharpe':>7s} | {'Ret/sem':>8s} | {'Anual%':>7s} | {'WR%':>5s} | {'PF':>5s} | {'N':>5s} | {'2a Mejor':>10s} | {'2a Sh':>6s} | {'3a Mejor':>10s} | {'3a Sh':>6s}")
print(f"  {'-'*5} | {'-'*10} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*10} | {'-'*6} | {'-'*10} | {'-'*6}")

short_table = {}

for score in range(1, 11):
    ranked = []
    for sn in short_strat_names:
        mask = (sdf['composite_score'] == score) & (sdf['strategy'] == sn)
        rets = sdf[mask]['ret'].values
        if len(rets) < 5:
            continue
        m = calc_metrics(rets)
        if m is None:
            continue
        ranked.append({'strategy': sn, **m})

    ranked.sort(key=lambda x: x['sharpe'], reverse=True)

    if ranked:
        b = ranked[0]
        short_table[score] = b['strategy']
        s2 = ranked[1] if len(ranked) > 1 else None
        s3 = ranked[2] if len(ranked) > 2 else None
        sec_str = f"{s2['strategy']:>10s}" if s2 else f"{'N/A':>10s}"
        sec_sh = f"{s2['sharpe']:>+6.2f}" if s2 else f"{'N/A':>6s}"
        trd_str = f"{s3['strategy']:>10s}" if s3 else f"{'N/A':>10s}"
        trd_sh = f"{s3['sharpe']:>+6.2f}" if s3 else f"{'N/A':>6s}"
        print(f"  {score:>5d} | {b['strategy']:>10s} | "
              f"{b['sharpe']:>+7.2f} | {b['avg_ret']*100:>+7.3f}% | "
              f"{b['annual']:>+6.1f}% | {b['wr']:>4.0f}% | {b['pf']:>5.2f} | "
              f"{b['n']:>5d} | {sec_str} | {sec_sh} | {trd_str} | {trd_sh}")
    else:
        short_table[score] = 'MR_1w'
        print(f"  {score:>5d} | {'N/A':>10s} | (fallback MR_1w)")

# ================================================================
# TABLA C: Decision final combinada
# ================================================================
latest = spy.iloc[-1]
current_score = int(latest['composite_score'])

print(f"\n\n{'='*160}")
print(f"  TABLA C: DECISION FINAL COMBINADA ({N_LONG}L x $25K + {N_SHORT}S x $50K = $500K)")
print(f"{'='*160}")

print(f"\n  {'Score':>5s} | {'Estrat LONG':>12s} | {'Estrat SHORT':>12s} | {'Nuevo?':>6s}")
print(f"  {'-'*5} | {'-'*12} | {'-'*12} | {'-'*6}")

new_short_strats = {'PSAR', 'ST', 'BB', 'STOCH'}
for score in range(1, 11):
    ls = long_table.get(score, 'N/A')
    ss = short_table.get(score, 'N/A')
    is_new = "NUEVO" if ss in new_short_strats else ""
    marker = " <<<" if score == current_score else ""
    print(f"  {score:>5d} | {ls:>12s} | {ss:>12s} | {is_new:>6s}{marker}")

print(f"\n  Fecha ultimo dato: {latest['date'].date()}")
print(f"  SPY: {latest['close']:.2f}")
print(f"  Composite Score actual: {current_score}/10")

ls_cur = long_table.get(current_score, 'N/A')
ss_cur = short_table.get(current_score, 'N/A')
print(f"\n  {'='*60}")
print(f"  >>> RECOMENDACION PARA SCORE {current_score}/10 <<<")
print(f"  {'='*60}")
print(f"  LONGS  ({N_LONG} pos x $25K): {ls_cur} ({LONG_STRATEGIES[ls_cur]['desc']})")
print(f"  SHORTS ({N_SHORT} pos x $25K): {ss_cur} ({SHORT_STRATEGIES[ss_cur]['desc']})")
print(f"  Total: {N_LONG + N_SHORT} posiciones x $25K = $500K/semana")

# ================================================================
# TABLA D: Heatmap Sharpe LONGS: Score x Estrategia (7 estrategias)
# ================================================================
print(f"\n\n{'='*160}")
print(f"  HEATMAP SHARPE LONGS: SCORE x ESTRATEGIA ({N_LONG} longs)")
print(f"{'='*160}")

hdr = f"  {'Score':>5s}"
for sn in long_strat_names:
    hdr += f" | {sn:>8s}"
hdr += f" | {'MEJOR':>8s}"
print(hdr)
sep = f"  {'-'*5}"
for _ in range(len(long_strat_names) + 1):
    sep += f" | {'-'*8}"
print(sep)

for score in range(1, 11):
    row_str = f"  {score:>5d}"
    best_in_row = -999
    best_sn = ""
    for sn in long_strat_names:
        mask = (ldf['composite_score'] == score) & (ldf['strategy'] == sn)
        rets = ldf[mask]['ret'].values
        if len(rets) >= 5:
            m = calc_metrics(rets)
            if m:
                row_str += f" | {m['sharpe']:>+8.2f}"
                if m['sharpe'] > best_in_row:
                    best_in_row = m['sharpe']
                    best_sn = sn
            else:
                row_str += f" | {'N/A':>8s}"
        else:
            row_str += f" | {'N/A':>8s}"
    row_str += f" | {best_sn:>8s}"
    marker = " <<<" if score == current_score else ""
    print(f"{row_str}{marker}")

# ================================================================
# TABLA E: Heatmap Sharpe SHORTS: Score x Estrategia (11 estrategias)
# ================================================================
print(f"\n\n{'='*160}")
print(f"  HEATMAP SHARPE SHORTS: SCORE x ESTRATEGIA ({N_SHORT} shorts)")
print(f"  * = nuevos indicadores (PSAR, ST, BB, STOCH)")
print(f"{'='*160}")

hdr = f"  {'Score':>5s}"
for sn in short_strat_names:
    label = f"*{sn}" if sn in new_short_strats else sn
    hdr += f" | {label:>8s}"
hdr += f" | {'MEJOR':>8s}"
print(hdr)
sep = f"  {'-'*5}"
for _ in range(len(short_strat_names) + 1):
    sep += f" | {'-'*8}"
print(sep)

for score in range(1, 11):
    row_str = f"  {score:>5d}"
    best_in_row = -999
    best_sn = ""
    for sn in short_strat_names:
        mask = (sdf['composite_score'] == score) & (sdf['strategy'] == sn)
        rets = sdf[mask]['ret'].values
        if len(rets) >= 5:
            m = calc_metrics(rets)
            if m:
                row_str += f" | {m['sharpe']:>+8.2f}"
                if m['sharpe'] > best_in_row:
                    best_in_row = m['sharpe']
                    best_sn = sn
            else:
                row_str += f" | {'N/A':>8s}"
        else:
            row_str += f" | {'N/A':>8s}"
    label = f"*{best_sn}" if best_sn in new_short_strats else best_sn
    row_str += f" | {label:>8s}"
    marker = " <<<" if score == current_score else ""
    print(f"{row_str}{marker}")

# ================================================================
# RESUMEN GENERAL
# ================================================================
print(f"\n\n{'='*160}")
print(f"  RESUMEN GENERAL")
print(f"{'='*160}")

print(f"\n  Distribucion de scores en el periodo de backtest:")
for score in range(1, 11):
    n = (weeks_df['composite_score'] == score).sum()
    pct = n / len(weeks_df) * 100
    bar = '#' * int(pct * 2)
    print(f"    Score {score:>2d}: {n:>4d} semanas ({pct:>5.1f}%) {bar}")

print(f"\n  Tablas de decision para backtest_regime_weekly_picks.py:")
print(f"\n  LONG_TABLE = {{")
for score in range(1, 11):
    ls = long_table.get(score, 'MR_1w')
    print(f"      {score}: '{ls}',")
print(f"  }}")

print(f"\n  SHORT_TABLE = {{")
for score in range(1, 11):
    ss = short_table.get(score, 'MR_1w')
    print(f"      {score}: '{ss}',")
print(f"  }}")

# Highlight which new strategies won
new_wins = [s for s in short_table.values() if s in new_short_strats]
if new_wins:
    from collections import Counter
    nw_counts = Counter(new_wins)
    print(f"\n  Nuevos indicadores ganadores: {dict(nw_counts)}")
else:
    print(f"\n  Ningun nuevo indicador gano (los 7 originales fueron mejores)")

print(f"\n  Modelo: {N_LONG} longs x $25K + {N_SHORT} shorts x $25K = $500K/semana")
print(f"  Senal viernes close -> Entrada lunes open -> Salida lunes open siguiente")

print(f"\n  Tiempo total: {time.time()-t0:.0f}s")
print(f"\n{'='*160}")
print(f"  FIN OPTIMIZACION {N_LONG}L + {N_SHORT}S")
print(f"{'='*160}")
