"""
WEEKLY PICKS REPORT - 10 Longs + 10 Shorts con 11 Estrategias Independientes
==============================================================================
20 acciones seleccionadas cada semana: 10 longs ($25K) + 10 shorts ($25K).
Cada lado usa su propia estrategia optima segun el composite score.
11 estrategias disponibles en ambos lados: MR_1w, MR_2w, MOM_4w, MOM_12w,
MR_RSI, VOL_UP, RANGE, PSAR, ST, BB, STOCH.

Reglas:
  - Senal: viernes close
  - Entrada: lunes open (siguiente al viernes)
  - Salida: lunes open (siguiente semana)
  - Long:  $25,000 x 10 = $250,000
  - Short: $25,000 x 10 = $250,000
  - Total: $500,000/semana
  - Costes: 0.3% por accion (comisiones + deslizamiento, round-trip)
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

LONG_POSITION_SIZE = 25_000   # $ por accion long (10 x $25K = $250K)
SHORT_POSITION_SIZE = 25_000  # $ por accion short (10 x $25K = $250K)
COST_PCT = 0.003              # 0.3% round-trip por accion
N_LONG = 10                   # 10 longs
N_SHORT = 10                  # 10 shorts
N_DETAIL_WEEKS = 20           # Semanas recientes a mostrar con detalle de picks

# Tablas de decision optimas (del backtest_regime_optimization.py - 10L+10S 11 estrat)
# Score -> estrategia para seleccionar 10 longs
LONG_TABLE = {
    1:  'ST',
    2:  'ST',
    3:  'BB',
    4:  'PSAR',
    5:  'MR_RSI',
    6:  'RANGE',
    7:  'MR_1w',
    8:  'MR_1w',
    9:  'MOM_12w',
    10: 'RANGE',
}
# Score -> estrategia para seleccionar 10 shorts
SHORT_TABLE = {
    1:  'MR_RSI',
    2:  'STOCH',
    3:  'BB',
    4:  'MR_RSI',
    5:  'STOCH',
    6:  'MOM_4w',
    7:  'MOM_12w',
    8:  'PSAR',
    9:  'STOCH',
    10: 'MOM_4w',
}

# 11 estrategias disponibles para ambos lados
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
LONG_STRATEGIES = STRATEGIES
SHORT_STRATEGIES = STRATEGIES

N_TOTAL = N_LONG + N_SHORT
TOTAL_CAPITAL = LONG_POSITION_SIZE * N_LONG + SHORT_POSITION_SIZE * N_SHORT
print("=" * 150)
print("  WEEKLY PICKS REPORT - 10 Longs + 10 Shorts con 11 Estrategias Independientes")
print(f"  Long: ${LONG_POSITION_SIZE:,} x {N_LONG} = ${LONG_POSITION_SIZE*N_LONG:,} | Short: ${SHORT_POSITION_SIZE:,} x {N_SHORT} = ${SHORT_POSITION_SIZE*N_SHORT:,} | Total: ${TOTAL_CAPITAL:,}/semana")
print("=" * 150)

# ============================================================
# [1/6] LOAD DATA + COMPUTE SCORES (same as optimization script)
# ============================================================
print("\n[1/6] Cargando datos de mercado...")

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
        AND date >= '2004-01-01' ORDER BY date""", conn, params={"syms": sp500_syms}, parse_dates=['date'])

print(f"  Datos cargados: SPY {len(spy)} dias")

# ============================================================
# [2/6] COMPUTE SCORES
# ============================================================
print("\n[2/6] Calculando scores...")

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
spy = spy.merge(vix, on='date', how='left'); spy['vix'] = spy['vix'].ffill()
spy = spy.merge(tip, on='date', how='left'); spy['tip_close'] = spy['tip_close'].ffill()
spy = spy.merge(ief, on='date', how='left'); spy['ief_close'] = spy['ief_close'].ffill()
spy['tip_ief_ratio'] = spy['tip_close'] / spy['ief_close']
spy['tip_ief_change_20d'] = (spy['tip_ief_ratio'] / spy['tip_ief_ratio'].shift(20) - 1) * 100
spy = spy.merge(aaii[['date', 'bull_bear_spread']], on='date', how='left')
spy['bull_bear_spread'] = spy['bull_bear_spread'].ffill()
spy = spy.dropna(subset=['ma200', 'rsi']).reset_index(drop=True)

earnings['quarter'] = earnings['date'].dt.to_period('Q')
earnings['beat'] = (earnings['eps_actual'] > earnings['eps_estimated']).astype(int)
quarterly = earnings.groupby('quarter').agg(
    n_earnings=('beat', 'count'), beat_rate=('beat', 'mean'), total_eps=('eps_actual', 'sum'),
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
    for threshold, score in [(35,1),(25,2),(20,3),(18,4),(17,5),(16,6),(15,7),(14,8),(12,9)]:
        if v > threshold: return score
    return 10

def calc_rsi_score(rsi):
    if pd.isna(rsi): return 5
    for threshold, score in [(20,1),(30,2),(40,3),(50,4),(55,5),(62,6),(70,7),(78,8),(85,9)]:
        if rsi < threshold: return score
    return 10

def calc_eps_growth_score(v):
    if pd.isna(v): return 5
    for threshold, score in [(-20,1),(-10,2),(-3,3),(3,4),(8,5),(15,6),(25,7),(40,8),(80,9)]:
        if v < threshold: return score
    return 10

def calc_beat_rate_score(br):
    if pd.isna(br): return 5
    br *= 100
    for threshold, score in [(40,1),(50,2),(55,3),(60,4),(65,5),(70,6),(75,7),(80,8),(85,9)]:
        if br < threshold: return score
    return 10

def calc_inflation_score(v):
    if pd.isna(v): return 6
    for threshold, score in [(5,1),(3,2),(2,3),(1,4),(0.3,5),(-0.3,6),(-1,7),(-2,8),(-3,9)]:
        if v > threshold: return score
    return 10

def calc_sentiment_score(v):
    if pd.isna(v): return 5
    for threshold, score in [(-25,1),(-15,2),(-5,3),(0,4),(5,5),(15,6),(25,7),(35,8),(50,9)]:
        if v < threshold: return score
    return 10

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

weights = {'market_score': 0.30, 'vix_score': 0.20, 'rsi_score': 0.10,
           'eps_growth_score': 0.15, 'beat_rate_score': 0.10,
           'inflation_score': 0.05, 'sentiment_score': 0.10}
spy['composite_raw'] = sum(spy[col] * w for col, w in weights.items())
comp_min = spy['composite_raw'].quantile(0.01)
comp_max = spy['composite_raw'].quantile(0.99)
spy['composite_score'] = ((spy['composite_raw'] - comp_min) / (comp_max - comp_min) * 9 + 1).clip(1, 10).round().astype(int)

print(f"  Scores calculados ({spy['date'].min().date()} a {spy['date'].max().date()})")

# ============================================================
# [3/6] SP500 + STOCK PRICES + INDICATORS
# ============================================================
print("\n[3/6] Cargando constituyentes SP500 y precios...")

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

with engine.connect() as conn:
    df = pd.read_sql("""SELECT symbol, date, open, high, low, close FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2003-01-01'
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

print(f"  Registros: {len(df):,}")

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
        if lower[i] > final_lower[i-1] or c[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        if upper[i] < final_upper[i-1] or c[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if direction[i-1] == 1:
            if c[i] < final_lower[i]:
                direction[i] = -1
                st[i] = final_upper[i]
            else:
                direction[i] = 1
                st[i] = final_lower[i]
        else:
            if c[i] > final_upper[i]:
                direction[i] = 1
                st[i] = final_lower[i]
            else:
                direction[i] = -1
                st[i] = final_upper[i]

    return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)


def calc_ind(g):
    g = g.sort_values('date').copy()
    c = g['close']
    h = g['high']
    l = g['low']

    g['ret_1w'] = c / c.shift(5) - 1
    g['ret_2w'] = c / c.shift(10) - 1
    g['ret_4w'] = c / c.shift(20) - 1
    g['ret_12w'] = c / c.shift(60) - 1
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g['rsi_14'] = 100 - (100 / (1 + rs))
    daily_ret = c.pct_change()
    g['vol_5d'] = daily_ret.rolling(5).std()
    g['vol_dir'] = g['vol_5d'] * np.sign(g['ret_1w'])
    h5 = h.rolling(5).max()
    l5 = l.rolling(5).min()
    rng = h5 - l5
    g['range_pos'] = np.where(rng > 0, (c - l5) / rng, 0.5)

    # Parabolic SAR distance: (close - SAR) / close * 100
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

    # Stochastic %K(14,3)
    low_14 = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    stoch_range = high_14 - low_14
    raw_k = np.where(stoch_range > 0, (c - low_14) / stoch_range * 100, 50)
    g['stoch_k'] = pd.Series(raw_k, index=c.index).rolling(3).mean()

    return g

print("  Calculando indicadores...")
df = df.groupby('symbol', group_keys=False).apply(calc_ind)
df_indexed = df.set_index(['symbol', 'date']).sort_index()
df['weekday'] = df['date'].dt.weekday
print(f"  Tiempo: {time.time()-t0:.0f}s")

# ============================================================
# [4/6] BUILD WEEKS + GENERATE PICKS
# ============================================================
print("\n[4/6] Generando picks semanales...")

fridays = np.sort(df[df['weekday'] == 4]['date'].unique())
mondays = np.sort(df[df['weekday'] == 0]['date'].unique())
spy_indexed = spy.set_index('date')
all_dates_in_index = set(df_indexed.index.get_level_values('date'))

weeks = []
for fri in fridays:
    next_mons = mondays[mondays > fri]
    if len(next_mons) < 1: continue  # Need at least 1 monday for entry
    fri_ts = pd.Timestamp(fri)
    if fri_ts not in spy_indexed.index: continue
    sr = spy_indexed.loc[fri_ts]
    if pd.isna(sr.get('market_score')): continue
    w = {
        'signal_date': fri_ts,
        'entry_date': pd.Timestamp(next_mons[0]),
        'exit_date': pd.Timestamp(next_mons[1]) if len(next_mons) >= 2 else None,
        'composite_score': int(sr['composite_score']),
    }
    weeks.append(w)

weeks_df = pd.DataFrame(weeks)
weeks_df = weeks_df[weeks_df['signal_date'] >= '2004-01-01'].reset_index(drop=True)

# Pre-build snapshots
friday_data = {}
monday_data = {}
for _, row in weeks_df.iterrows():
    sig = row['signal_date']
    if sig not in friday_data and sig in all_dates_in_index:
        friday_data[sig] = df_indexed.xs(sig, level='date', drop_level=True)
    for d in [row['entry_date'], row['exit_date']]:
        if d is not None and d not in monday_data and d in all_dates_in_index:
            monday_data[d] = df_indexed.xs(d, level='date', drop_level=True)

# Generate all weekly picks (5L + 5S with independent strategies)
all_weeks = []

def _rank_and_select(fri_snap, eligible, strat_def, n_long, n_short):
    """Rank stocks by strategy column and return (long_syms, short_syms, ranking)."""
    col = strat_def['col']
    long_bottom = strat_def['long_bottom']
    valid_syms = [s for s in eligible if s in fri_snap.index]
    if not valid_syms:
        return [], [], None
    vals = fri_snap.loc[valid_syms, col]
    if isinstance(vals, pd.DataFrame):
        vals = vals.iloc[:, 0]
    ranking = vals.dropna().sort_values()
    if len(ranking) < 20:
        return [], [], None
    if long_bottom:
        longs = ranking.head(n_long).index.tolist() if n_long > 0 else []
        shorts = ranking.tail(n_short).index.tolist() if n_short > 0 else []
    else:
        longs = ranking.tail(n_long).index.tolist() if n_long > 0 else []
        shorts = ranking.head(n_short).index.tolist() if n_short > 0 else []
    return longs, shorts, ranking

for _, wrow in weeks_df.iterrows():
    sig = wrow['signal_date']
    entry = wrow['entry_date']
    exit_d = wrow['exit_date']
    score = wrow['composite_score']

    if sig not in friday_data or entry not in monday_data:
        continue

    long_strat_name = LONG_TABLE[score]
    short_strat_name = SHORT_TABLE[score]
    long_strat_def = LONG_STRATEGIES[long_strat_name]
    short_strat_def = SHORT_STRATEGIES[short_strat_name]

    fri = friday_data[sig]
    en = monday_data[entry]
    sp500 = get_sp500_cached(sig)
    elig = [s for s in fri.index if s in sp500]
    if len(elig) < 100:
        continue

    # Select 10 longs using long strategy
    long_syms, _, long_ranking = _rank_and_select(fri, elig, long_strat_def, N_LONG, 0)
    # Select 10 shorts using short strategy
    _, short_syms, short_ranking = _rank_and_select(fri, elig, short_strat_def, 0, N_SHORT)

    if not long_syms or not short_syms:
        continue

    # Build picks with prices
    has_exit = exit_d is not None and exit_d in monday_data
    ex = monday_data[exit_d] if has_exit else None

    picks = []
    for sym in long_syms:
        if sym not in en.index:
            continue
        ep = en.loc[sym, 'open']
        if isinstance(ep, pd.Series): ep = ep.iloc[0]
        if pd.isna(ep) or ep <= 0:
            continue

        rank_val = long_ranking.loc[sym] if long_ranking is not None and sym in long_ranking.index else np.nan

        xp = np.nan
        gross_ret = np.nan
        net_ret = np.nan
        pnl = np.nan
        if has_exit and sym in ex.index:
            xp = ex.loc[sym, 'open']
            if isinstance(xp, pd.Series): xp = xp.iloc[0]
            if pd.notna(xp) and xp > 0:
                gross_ret = (xp - ep) / ep
                net_ret = gross_ret - COST_PCT
                pnl = LONG_POSITION_SIZE * net_ret

        picks.append({
            'symbol': sym, 'side': 'LONG', 'strategy': long_strat_name,
            'entry_price': ep, 'exit_price': xp,
            'rank_val': rank_val, 'gross_ret': gross_ret, 'net_ret': net_ret, 'pnl': pnl,
            'position_size': LONG_POSITION_SIZE,
        })

    for sym in short_syms:
        if sym not in en.index:
            continue
        ep = en.loc[sym, 'open']
        if isinstance(ep, pd.Series): ep = ep.iloc[0]
        if pd.isna(ep) or ep <= 0:
            continue

        rank_val = short_ranking.loc[sym] if short_ranking is not None and sym in short_ranking.index else np.nan

        xp = np.nan
        gross_ret = np.nan
        net_ret = np.nan
        pnl = np.nan
        if has_exit and sym in ex.index:
            xp = ex.loc[sym, 'open']
            if isinstance(xp, pd.Series): xp = xp.iloc[0]
            if pd.notna(xp) and xp > 0:
                gross_ret = (ep - xp) / ep
                net_ret = gross_ret - COST_PCT
                pnl = SHORT_POSITION_SIZE * net_ret

        picks.append({
            'symbol': sym, 'side': 'SHORT', 'strategy': short_strat_name,
            'entry_price': ep, 'exit_price': xp,
            'rank_val': rank_val, 'gross_ret': gross_ret, 'net_ret': net_ret, 'pnl': pnl,
            'position_size': SHORT_POSITION_SIZE,
        })

    # Week summary - compute long and short sides separately
    long_picks = [p for p in picks if p['side'] == 'LONG']
    short_picks = [p for p in picks if p['side'] == 'SHORT']
    valid_pnls = [p['pnl'] for p in picks if pd.notna(p['pnl'])]
    valid_nets = [p['net_ret'] for p in picks if pd.notna(p['net_ret'])]
    long_nets = [p['net_ret'] for p in long_picks if pd.notna(p['net_ret'])]
    short_nets = [p['net_ret'] for p in short_picks if pd.notna(p['net_ret'])]

    week_data = {
        'signal_date': sig,
        'entry_date': entry,
        'exit_date': exit_d,
        'composite_score': score,
        'long_strategy': long_strat_name,
        'short_strategy': short_strat_name,
        'n_long': len(long_picks),
        'n_short': len(short_picks),
        'picks': picks,
        'n_picks': len(picks),
        'week_pnl': sum(valid_pnls) if valid_pnls else np.nan,
        'week_ret': np.mean(valid_nets) if valid_nets else np.nan,
        'long_ret': np.mean(long_nets) if long_nets else np.nan,
        'short_ret': np.mean(short_nets) if short_nets else np.nan,
        'long_pnl': sum(p['pnl'] for p in long_picks if pd.notna(p['pnl'])) if long_picks else np.nan,
        'short_pnl': sum(p['pnl'] for p in short_picks if pd.notna(p['pnl'])) if short_picks else np.nan,
        'has_exit': has_exit,
    }
    all_weeks.append(week_data)

print(f"  Semanas generadas: {len(all_weeks)}")
print(f"  Tiempo: {time.time()-t0:.0f}s")

# ============================================================
# [5/6] TABLA DE DECISION (referencia)
# ============================================================
print(f"\n\n{'='*150}")
print(f"  TABLA DE DECISION POR REGIMEN - 10L + 10S Independientes (11 estrategias)")
print(f"{'='*150}")
print(f"  {'Score':>5s} | {'Estrat LONG':>12s} | {'Desc LONG':<20s} | {'Estrat SHORT':>12s} | {'Desc SHORT':<20s}")
print(f"  {'-'*5} | {'-'*12} | {'-'*20} | {'-'*12} | {'-'*20}")
for sc in range(1, 11):
    ls = LONG_TABLE[sc]
    ss = SHORT_TABLE[sc]
    print(f"  {sc:>5d} | {ls:>12s} | {LONG_STRATEGIES[ls]['desc']:<20s} | {ss:>12s} | {SHORT_STRATEGIES[ss]['desc']:<20s}")

print(f"\n  Entrada: LUNES OPEN (siguiente al viernes de senal)")
print(f"  Salida:  LUNES OPEN (siguiente semana)")
print(f"  Coste:   {COST_PCT*100:.1f}% por accion (comisiones + deslizamiento)")
print(f"  Tamano:  Long ${LONG_POSITION_SIZE:,} x {N_LONG} + Short ${SHORT_POSITION_SIZE:,} x {N_SHORT} = ${TOTAL_CAPITAL:,} total/semana")

# ============================================================
# [6/6] OUTPUT
# ============================================================

# --- SENAL ACTUAL / MAS RECIENTE ---
latest_week = all_weeks[-1]
print(f"\n\n{'='*150}")
print(f"  SENAL MAS RECIENTE")
print(f"{'='*150}")
print(f"  Senal (viernes):  {latest_week['signal_date'].date()}")
print(f"  Entrada (lunes):  {latest_week['entry_date'].date()} al OPEN")
if latest_week['exit_date'] is not None:
    print(f"  Salida (lunes):   {latest_week['exit_date'].date()} al OPEN")
else:
    print(f"  Salida (lunes):   Pendiente")
print(f"  Composite Score:  {latest_week['composite_score']}/10")
print(f"  Estrategia LONG:  {latest_week['long_strategy']} ({LONG_STRATEGIES[latest_week['long_strategy']]['desc']})")
print(f"  Estrategia SHORT: {latest_week['short_strategy']} ({SHORT_STRATEGIES[latest_week['short_strategy']]['desc']})")
print(f"  Posiciones:       {latest_week['n_long']}L + {latest_week['n_short']}S = {latest_week['n_picks']} total")

print(f"\n  {'#':>3s} | {'Accion':>6s} | {'Lado':>5s} | {'Estrategia':>8s} | {'Entrada$':>10s} | {'Salida$':>10s} | {'Bruto%':>7s} | {'Neto%':>7s} | {'P&L $':>10s}")
print(f"  {'-'*3} | {'-'*6} | {'-'*5} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*7} | {'-'*7} | {'-'*10}")

for i, p in enumerate(latest_week['picks'], 1):
    xp_str = f"${p['exit_price']:>8.2f}" if pd.notna(p['exit_price']) else "  Pendiente"
    gr_str = f"{p['gross_ret']*100:>+6.2f}%" if pd.notna(p['gross_ret']) else "    N/A"
    nr_str = f"{p['net_ret']*100:>+6.2f}%" if pd.notna(p['net_ret']) else "    N/A"
    pnl_str = f"${p['pnl']:>+9,.0f}" if pd.notna(p['pnl']) else "      N/A"
    print(f"  {i:>3d} | {p['symbol']:>6s} | {p['side']:>5s} | {p['strategy']:>8s} | ${p['entry_price']:>8.2f} | {xp_str} | {gr_str} | {nr_str} | {pnl_str}")

if pd.notna(latest_week['week_pnl']):
    long_pnl_str = f"${latest_week['long_pnl']:>+,.0f}" if pd.notna(latest_week['long_pnl']) else "N/A"
    short_pnl_str = f"${latest_week['short_pnl']:>+,.0f}" if pd.notna(latest_week['short_pnl']) else "N/A"
    print(f"\n  TOTAL SEMANA: P&L = ${latest_week['week_pnl']:>+,.0f} | Ret neto = {latest_week['week_ret']*100:>+.2f}%")
    print(f"    Longs:  {long_pnl_str} | Shorts: {short_pnl_str}")
else:
    print(f"\n  TOTAL SEMANA: Pendiente (posiciones abiertas)")

# --- ULTIMAS N SEMANAS CON DETALLE ---
print(f"\n\n{'='*150}")
print(f"  ULTIMAS {N_DETAIL_WEEKS} SEMANAS CON DETALLE DE PICKS")
print(f"{'='*150}")

start_idx = max(0, len(all_weeks) - N_DETAIL_WEEKS)
for w in all_weeks[start_idx:]:
    status = "CERRADA" if w['has_exit'] else "ABIERTA"
    exit_str = str(w['exit_date'].date()) if w['exit_date'] is not None else "Pendiente"

    print(f"\n  --- Senal {w['signal_date'].date()} | Entrada {w['entry_date'].date()} | Salida {exit_str} | Score {w['composite_score']} | L:{w['long_strategy']} S:{w['short_strategy']} | {status} ---")

    # Sort picks: longs first, then shorts
    longs = [p for p in w['picks'] if p['side'] == 'LONG']
    shorts = [p for p in w['picks'] if p['side'] == 'SHORT']

    for p in longs + shorts:
        xp_str = f"${p['exit_price']:>7.2f}" if pd.notna(p['exit_price']) else " Pend."
        nr_str = f"{p['net_ret']*100:>+5.1f}%" if pd.notna(p['net_ret']) else "  N/A"
        pnl_str = f"${p['pnl']:>+8,.0f}" if pd.notna(p['pnl']) else "     N/A"
        print(f"    {p['side']:>5s} {p['symbol']:<6s} ({p['strategy']:>7s}) entrada ${p['entry_price']:>7.2f} salida {xp_str} -> {nr_str} {pnl_str}")

    if pd.notna(w['week_pnl']):
        win = "WIN" if w['week_pnl'] > 0 else "LOSS"
        long_pnl_str = f"${w['long_pnl']:>+,.0f}" if pd.notna(w.get('long_pnl')) else "N/A"
        short_pnl_str = f"${w['short_pnl']:>+,.0f}" if pd.notna(w.get('short_pnl')) else "N/A"
        print(f"    TOTAL: ${w['week_pnl']:>+10,.0f} ({w['week_ret']*100:>+.2f}%) [{win}]  Longs: {long_pnl_str} | Shorts: {short_pnl_str}")

# --- RESUMEN TODAS LAS SEMANAS (condensado) ---
print(f"\n\n{'='*150}")
print(f"  RESUMEN HISTORICO COMPLETO (todas las semanas)")
print(f"{'='*150}")

print(f"\n  {'Senal':>12s} | {'Entrada':>12s} | {'Sc':>2s} | {'Estrat L':>8s} | {'Estrat S':>8s} | {'Ret%':>7s} | {'P&L $':>10s} | {'Acum $':>12s} | {'Picks'}")
print(f"  {'-'*12} | {'-'*12} | {'-'*2} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*10} | {'-'*12} | {'-'*40}")

cumulative_pnl = 0
yearly_pnl = {}
current_year = None
n_wins = 0
n_losses = 0
total_weeks_with_data = 0

for w in all_weeks:
    year = w['signal_date'].year
    if year not in yearly_pnl:
        yearly_pnl[year] = {'pnl': 0, 'weeks': 0, 'wins': 0}

    if pd.notna(w['week_pnl']):
        cumulative_pnl += w['week_pnl']
        yearly_pnl[year]['pnl'] += w['week_pnl']
        yearly_pnl[year]['weeks'] += 1
        total_weeks_with_data += 1
        if w['week_pnl'] > 0:
            n_wins += 1
            yearly_pnl[year]['wins'] += 1
        else:
            n_losses += 1

    # Print year separator
    if current_year != year:
        if current_year is not None and current_year in yearly_pnl:
            yp = yearly_pnl[current_year]
            wr = yp['wins']/yp['weeks']*100 if yp['weeks'] > 0 else 0
            print(f"  {'':>12s}   {'>>> AÑO '+str(current_year):>12s}: P&L = ${yp['pnl']:>+12,.0f} | {yp['weeks']} sem | WR {wr:.0f}%")
            print(f"  {'-'*12} | {'-'*12} | {'-'*2} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*10} | {'-'*12} | {'-'*40}")
        current_year = year

    ret_str = f"{w['week_ret']*100:>+6.2f}%" if pd.notna(w['week_ret']) else f"{'N/A':>7s}"
    pnl_str = f"${w['week_pnl']:>+9,.0f}" if pd.notna(w['week_pnl']) else f"{'N/A':>10s}"
    cum_str = f"${cumulative_pnl:>+11,.0f}"

    # Compact picks list
    longs = [p['symbol'] for p in w['picks'] if p['side'] == 'LONG']
    shorts = [p['symbol'] for p in w['picks'] if p['side'] == 'SHORT']
    picks_str = ""
    if longs:
        picks_str += "L:" + ",".join(longs[:5])
        if len(longs) > 5:
            picks_str += f"+{len(longs)-5}"
    if shorts:
        if picks_str: picks_str += " "
        picks_str += "S:" + ",".join(shorts[:3])
        if len(shorts) > 3:
            picks_str += f"+{len(shorts)-3}"

    print(f"  {str(w['signal_date'].date()):>12s} | {str(w['entry_date'].date()):>12s} | {w['composite_score']:>2d} | "
          f"{w['long_strategy']:>8s} | {w['short_strategy']:>8s} | {ret_str} | {pnl_str} | {cum_str} | {picks_str}")

# Print last year
if current_year is not None and current_year in yearly_pnl:
    yp = yearly_pnl[current_year]
    wr = yp['wins']/yp['weeks']*100 if yp['weeks'] > 0 else 0
    print(f"  {'':>12s}   {'>>> AÑO '+str(current_year):>12s}: P&L = ${yp['pnl']:>+12,.0f} | {yp['weeks']} sem | WR {wr:.0f}%")

# --- RESUMEN ANUAL ---
print(f"\n\n{'='*150}")
print(f"  RESUMEN POR AÑO")
print(f"{'='*150}")

print(f"\n  {'Año':>6s} | {'Semanas':>7s} | {'Wins':>5s} | {'WR%':>5s} | {'P&L $':>12s} | {'P&L/sem $':>10s} | {'Ret/sem%':>8s} | {'Acumulado $':>14s}")
print(f"  {'-'*6} | {'-'*7} | {'-'*5} | {'-'*5} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*14}")

cum = 0
for year in sorted(yearly_pnl.keys()):
    yp = yearly_pnl[year]
    cum += yp['pnl']
    wr = yp['wins']/yp['weeks']*100 if yp['weeks'] > 0 else 0
    pnl_per_week = yp['pnl'] / yp['weeks'] if yp['weeks'] > 0 else 0
    ret_per_week = pnl_per_week / TOTAL_CAPITAL * 100
    print(f"  {year:>6d} | {yp['weeks']:>7d} | {yp['wins']:>5d} | {wr:>4.0f}% | ${yp['pnl']:>+11,.0f} | ${pnl_per_week:>+9,.0f} | {ret_per_week:>+7.2f}% | ${cum:>+13,.0f}")

# --- ESTADISTICAS GLOBALES ---
print(f"\n\n{'='*150}")
print(f"  ESTADISTICAS GLOBALES")
print(f"{'='*150}")

total_capital = TOTAL_CAPITAL
all_pnls = [w['week_pnl'] for w in all_weeks if pd.notna(w['week_pnl'])]
all_rets = [w['week_ret'] for w in all_weeks if pd.notna(w['week_ret'])]
rets_arr = np.array(all_rets)

sharpe = (np.mean(rets_arr) / np.std(rets_arr)) * np.sqrt(52) if np.std(rets_arr) > 0 else 0
max_dd_pnl = 0
peak = 0
running = 0
for pnl in all_pnls:
    running += pnl
    if running > peak:
        peak = running
    dd = running - peak
    if dd < max_dd_pnl:
        max_dd_pnl = dd

print(f"""
  Capital por semana:     ${total_capital:>12,}
  Total semanas:          {total_weeks_with_data:>12,}
  Periodo:                {all_weeks[0]['signal_date'].date()} a {all_weeks[-1]['signal_date'].date()}

  P&L total:              ${cumulative_pnl:>+12,.0f}
  P&L medio/semana:       ${np.mean(all_pnls):>+12,.0f}
  Ret medio/semana:       {np.mean(rets_arr)*100:>+11.3f}%
  Ret anualizado:         {np.mean(rets_arr)*52*100:>+11.1f}%

  Sharpe (neto costes):   {sharpe:>+12.2f}
  Win Rate:               {n_wins/total_weeks_with_data*100:>11.0f}%
  Wins / Losses:          {n_wins:>5d} / {n_losses:>5d}

  Mejor semana:           ${max(all_pnls):>+12,.0f}
  Peor semana:            ${min(all_pnls):>+12,.0f}
  Max Drawdown:           ${max_dd_pnl:>+12,.0f}
""")

print(f"  Tiempo total: {time.time()-t0:.0f}s")

# ============================================================
# EXPORT TO EXCEL
# ============================================================
print(f"\n  Exportando a Excel...")

# --- Sheet 1: Weekly Summary ---
weekly_rows = []
cum_pnl = 0
for w in all_weeks:
    if pd.notna(w['week_pnl']):
        cum_pnl += w['week_pnl']
    longs = [p['symbol'] for p in w['picks'] if p['side'] == 'LONG']
    shorts = [p['symbol'] for p in w['picks'] if p['side'] == 'SHORT']
    weekly_rows.append({
        'Signal_Date': w['signal_date'].date(),
        'Entry_Date': w['entry_date'].date(),
        'Exit_Date': w['exit_date'].date() if w['exit_date'] is not None else None,
        'Year': w['signal_date'].year,
        'Score': w['composite_score'],
        'Long_Strategy': w['long_strategy'],
        'Short_Strategy': w['short_strategy'],
        'N_Long': w['n_long'],
        'N_Short': w['n_short'],
        'Long_Ret_Pct': round(w['long_ret'] * 100, 3) if pd.notna(w.get('long_ret')) else None,
        'Short_Ret_Pct': round(w['short_ret'] * 100, 3) if pd.notna(w.get('short_ret')) else None,
        'Ret_Pct': round(w['week_ret'] * 100, 3) if pd.notna(w['week_ret']) else None,
        'Long_PnL': round(w['long_pnl'], 0) if pd.notna(w.get('long_pnl')) else None,
        'Short_PnL': round(w['short_pnl'], 0) if pd.notna(w.get('short_pnl')) else None,
        'PnL_USD': round(w['week_pnl'], 0) if pd.notna(w['week_pnl']) else None,
        'Cumulative_PnL': round(cum_pnl, 0),
        'Win': 1 if pd.notna(w['week_pnl']) and w['week_pnl'] > 0 else (0 if pd.notna(w['week_pnl']) else None),
        'Long_Picks': ', '.join(longs),
        'Short_Picks': ', '.join(shorts),
    })
df_weeks = pd.DataFrame(weekly_rows)

# --- Sheet 2: Individual Picks ---
pick_rows = []
for w in all_weeks:
    for p in w['picks']:
        pick_rows.append({
            'Signal_Date': w['signal_date'].date(),
            'Entry_Date': w['entry_date'].date(),
            'Exit_Date': w['exit_date'].date() if w['exit_date'] is not None else None,
            'Year': w['signal_date'].year,
            'Score': w['composite_score'],
            'Strategy': p['strategy'],
            'Symbol': p['symbol'],
            'Side': p['side'],
            'Entry_Price': round(p['entry_price'], 2),
            'Exit_Price': round(p['exit_price'], 2) if pd.notna(p['exit_price']) else None,
            'Gross_Ret_Pct': round(p['gross_ret'] * 100, 3) if pd.notna(p['gross_ret']) else None,
            'Net_Ret_Pct': round(p['net_ret'] * 100, 3) if pd.notna(p['net_ret']) else None,
            'PnL_USD': round(p['pnl'], 0) if pd.notna(p['pnl']) else None,
            'Position_Size': p.get('position_size', LONG_POSITION_SIZE),
        })
df_picks = pd.DataFrame(pick_rows)

# --- Sheet 3: Annual Summary ---
annual_rows = []
cum_a = 0
for year in sorted(yearly_pnl.keys()):
    yp = yearly_pnl[year]
    cum_a += yp['pnl']
    wr = yp['wins'] / yp['weeks'] * 100 if yp['weeks'] > 0 else 0
    pnl_pw = yp['pnl'] / yp['weeks'] if yp['weeks'] > 0 else 0
    ret_pw = pnl_pw / TOTAL_CAPITAL * 100
    annual_rows.append({
        'Year': year,
        'Weeks': yp['weeks'],
        'Wins': yp['wins'],
        'Losses': yp['weeks'] - yp['wins'],
        'Win_Rate_Pct': round(wr, 1),
        'PnL_USD': round(yp['pnl'], 0),
        'PnL_per_Week': round(pnl_pw, 0),
        'Ret_per_Week_Pct': round(ret_pw, 2),
        'Ret_Annual_Pct': round(ret_pw * 52, 1),
        'Cumulative_PnL': round(cum_a, 0),
    })
df_annual = pd.DataFrame(annual_rows)

# --- Sheet 4: Decision Tables ---
decision_rows = []
for sc in range(1, 11):
    decision_rows.append({
        'Score': sc,
        'Long_Strategy': LONG_TABLE[sc],
        'Long_Desc': LONG_STRATEGIES[LONG_TABLE[sc]]['desc'],
        'Short_Strategy': SHORT_TABLE[sc],
        'Short_Desc': SHORT_STRATEGIES[SHORT_TABLE[sc]]['desc'],
        'Same': 'SI' if LONG_TABLE[sc] == SHORT_TABLE[sc] else 'NO',
    })
df_decision = pd.DataFrame(decision_rows)

# Write Excel
excel_path = 'data/regime_weekly_picks.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_weeks.to_excel(writer, sheet_name='Semanas', index=False)
    df_picks.to_excel(writer, sheet_name='Picks', index=False)
    df_annual.to_excel(writer, sheet_name='Resumen_Anual', index=False)
    df_decision.to_excel(writer, sheet_name='Decision_Tables', index=False)

print(f"  Exportado: {excel_path}")
print(f"    - Semanas: {len(df_weeks):,} filas")
print(f"    - Picks: {len(df_picks):,} filas")
print(f"    - Resumen Anual: {len(df_annual)} filas")
print(f"    - Decision Tables: {len(df_decision)} filas")

print(f"\n{'='*150}")
print(f"  FIN WEEKLY PICKS REPORT")
print(f"{'='*150}")
