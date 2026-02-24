"""
BACKTEST: Pattern-Based Multi-Factor Stock Picking
===================================================
5L + 5S | $50K/position | $500K/week | 0.3% cost
Mon open -> Mon open | Friday signal

LONG pattern: "Pullback in strong trend" (11 factors)
  Momentum 12w (20%), Pullback 1w (12%), PSAR BEAR (12%), RSI inv (10%),
  BB inv (8%), Stoch inv (8%), ST BEAR (8%), Margins (7%), Vol (5%),
  Beats (5%), EPS growth (5%)

SHORT pattern: "Overbought without fundamentals" (11 factors)
  PSAR BULL (15%), No momentum LP (15%), Rally 1w (12%), Stoch (12%),
  RSI (8%), Low margins (8%), ST BULL (8%), BB (7%), Div yield (5%),
  Payout (5%), Debt (5%)
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

# ============================================================
# CONFIGURATION
# ============================================================
LONG_POSITION_SIZE = 50_000
SHORT_POSITION_SIZE = 50_000
COST_PCT = 0.003
N_LONG = 5
N_SHORT = 5
N_TOTAL = N_LONG + N_SHORT
TOTAL_CAPITAL = LONG_POSITION_SIZE * N_LONG + SHORT_POSITION_SIZE * N_SHORT

# LONG scoring weights
LONG_W = {
    'sc_ret12w': 0.20,      # High 12w momentum
    'sc_rsi_inv': 0.10,     # Low RSI (oversold)
    'sc_psar_bear': 0.12,   # PSAR BEAR
    'sc_st_bear': 0.08,     # SuperTrend BEAR
    'sc_bb_inv': 0.08,      # Low BB%B
    'sc_stoch_inv': 0.08,   # Low Stochastic
    'sc_ret1w_inv': 0.12,   # Recent pullback
    'sc_vol': 0.05,         # High volatility
    'sc_margin': 0.07,      # High net margin
    'sc_beats': 0.05,       # Earnings beats
    'sc_epsgr': 0.05,       # EPS growth YoY
}

# SHORT scoring weights
SHORT_W = {
    'sc_psar_bull': 0.15,   # PSAR BULL
    'sc_ret12w_inv': 0.15,  # No long-term momentum
    'sc_ret1w': 0.12,       # Recent rally
    'sc_stoch': 0.12,       # High Stochastic
    'sc_rsi': 0.08,         # High RSI
    'sc_margin_inv': 0.08,  # Low margins
    'sc_st_bull': 0.08,     # SuperTrend BULL
    'sc_bb': 0.07,          # High BB%B
    'sc_divyield': 0.05,    # High dividend yield
    'sc_payout': 0.05,      # High payout ratio
    'sc_debt': 0.05,        # High debt/equity
}

print("=" * 150)
print("  BACKTEST: Pattern-Based Multi-Factor Stock Picking")
print(f"  {N_LONG}L + {N_SHORT}S | ${LONG_POSITION_SIZE:,}/pos | ${TOTAL_CAPITAL:,}/week | {COST_PCT*100:.1f}% cost")
print("=" * 150)

# ============================================================
# [1/7] S&P 500 MEMBERSHIP
# ============================================================
print("\n[1/7] Cargando constituyentes S&P 500...")

with open('data/sp500_constituents.json') as f:
    current_members = json.load(f)
with open('data/sp500_historical_changes.json') as f:
    all_changes = json.load(f)
all_changes.sort(key=lambda x: x.get('date', ''), reverse=True)
current_set = {d['symbol'] for d in current_members}
all_sp500_symbols = set(current_set)
for ch in all_changes:
    if ch.get('date', '') >= '2003-01-01':
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
# [2/7] LOAD PRICES + COMPUTE INDICATORS
# ============================================================
print("\n[2/7] Cargando precios y calculando indicadores...")

with engine.connect() as conn:
    df = pd.read_sql("""SELECT symbol, date, open, high, low, close, volume FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2003-01-01'
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

print(f"  Registros: {len(df):,} | t={time.time()-t0:.0f}s")


def calc_parabolic_sar(high, low, close, af_start=0.02, af_max=0.20, af_step=0.02):
    n = len(close)
    if n < 3:
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    sar = np.full(n, np.nan); trend = np.ones(n); af_arr = np.zeros(n); ep = np.zeros(n)
    h = high.values; l = low.values
    sar[0] = l[0]; trend[0] = 1; af_arr[0] = af_start; ep[0] = h[0]
    for i in range(1, n):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af_arr[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[i-2] if i >= 2 else l[i-1])
            if l[i] < sar[i]:
                trend[i] = -1; sar[i] = ep[i-1]; ep[i] = l[i]; af_arr[i] = af_start
            else:
                trend[i] = 1
                if h[i] > ep[i-1]: ep[i] = h[i]; af_arr[i] = min(af_arr[i-1] + af_step, af_max)
                else: ep[i] = ep[i-1]; af_arr[i] = af_arr[i-1]
        else:
            sar[i] = sar[i-1] + af_arr[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[i-2] if i >= 2 else h[i-1])
            if h[i] > sar[i]:
                trend[i] = 1; sar[i] = ep[i-1]; ep[i] = h[i]; af_arr[i] = af_start
            else:
                trend[i] = -1
                if l[i] < ep[i-1]: ep[i] = l[i]; af_arr[i] = min(af_arr[i-1] + af_step, af_max)
                else: ep[i] = ep[i-1]; af_arr[i] = af_arr[i-1]
    return pd.Series(sar, index=close.index), pd.Series(trend, index=close.index)


def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    n = len(close)
    if n < period + 1:
        return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)
    h = high.values; l = low.values; c = close.values
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    atr = pd.Series(tr).rolling(period).mean().values
    hl2 = (h + l) / 2; upper = hl2 + multiplier * atr; lower = hl2 - multiplier * atr
    st = np.full(n, np.nan); direction = np.ones(n)
    final_upper = upper.copy(); final_lower = lower.copy()
    start = period; st[start] = lower[start]; direction[start] = 1
    for i in range(start + 1, n):
        if lower[i] > final_lower[i-1] or c[i-1] < final_lower[i-1]: final_lower[i] = lower[i]
        else: final_lower[i] = final_lower[i-1]
        if upper[i] < final_upper[i-1] or c[i-1] > final_upper[i-1]: final_upper[i] = upper[i]
        else: final_upper[i] = final_upper[i-1]
        if direction[i-1] == 1:
            if c[i] < final_lower[i]: direction[i] = -1; st[i] = final_upper[i]
            else: direction[i] = 1; st[i] = final_lower[i]
        else:
            if c[i] > final_upper[i]: direction[i] = 1; st[i] = final_lower[i]
            else: direction[i] = -1; st[i] = final_upper[i]
    return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)


def calc_ind(g):
    g = g.sort_values('date').copy()
    c = g['close']; h = g['high']; l = g['low']

    g['ret_1w'] = c / c.shift(5) - 1
    g['ret_4w'] = c / c.shift(20) - 1
    g['ret_12w'] = c / c.shift(60) - 1

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss_s.replace(0, np.nan)
    g['rsi_14'] = 100 - (100 / (1 + rs))

    daily_ret = c.pct_change()
    g['vol_20d'] = daily_ret.rolling(20).std() * np.sqrt(252)

    # Parabolic SAR distance %
    if len(g) >= 3:
        sar_vals, _ = calc_parabolic_sar(h, l, c)
        g['psar_dist'] = (c - sar_vals) / c * 100
    else:
        g['psar_dist'] = np.nan

    # SuperTrend distance %
    if len(g) >= 12:
        st_vals, _ = calc_supertrend(h, l, c, period=10, multiplier=3.0)
        g['st_dist'] = (c - st_vals) / c * 100
    else:
        g['st_dist'] = np.nan

    # Bollinger %B
    bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std; bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    g['bb_pctb'] = np.where(bb_width > 0, (c - bb_lower) / bb_width, 0.5)

    # Stochastic %K(14,3)
    low_14 = l.rolling(14).min(); high_14 = h.rolling(14).max()
    stoch_range = high_14 - low_14
    raw_k = np.where(stoch_range > 0, (c - low_14) / stoch_range * 100, 50)
    g['stoch_k'] = pd.Series(raw_k, index=c.index).rolling(3).mean()

    return g


print("  Calculando indicadores (PSAR, SuperTrend, BB, Stoch, RSI)...")
df = df.groupby('symbol', group_keys=False).apply(calc_ind)
print(f"  Indicadores listos | t={time.time()-t0:.0f}s")

# ============================================================
# [3/7] LOAD FUNDAMENTALS
# ============================================================
print("\n[3/7] Cargando fundamentales...")

with engine.connect() as conn:
    ratios = pd.read_sql("""SELECT symbol, date, net_profit_margin, dividend_yield,
        payout_ratio, debt_equity_ratio
        FROM fmp_ratios
        WHERE symbol = ANY(%(syms)s) AND date >= '2002-01-01'
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

    earn = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s)
        AND eps_actual IS NOT NULL AND eps_estimated IS NOT NULL
        AND date >= '2002-01-01' ORDER BY symbol, date""",
        conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

print(f"  Ratios: {len(ratios):,} | Earnings: {len(earn):,}")

ratios = ratios.rename(columns={
    'net_profit_margin': 'net_margin', 'dividend_yield': 'div_yield',
    'payout_ratio': 'payout', 'debt_equity_ratio': 'de_ratio',
})

# Compute earnings stats per symbol
print("  Calculando stats de earnings...")
earn_stats_list = []
for sym, g in earn.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    for i in range(len(g)):
        start_idx = max(0, i - 3)
        window = g.iloc[start_idx:i+1]
        beats = int((window['eps_actual'] > window['eps_estimated']).sum())
        eps_growth = np.nan
        if i >= 4:
            curr = g.iloc[i]['eps_actual']
            prev = g.iloc[i-4]['eps_actual']
            if pd.notna(prev) and pd.notna(curr) and abs(prev) > 0.01:
                eps_growth = (curr - prev) / abs(prev) * 100
        earn_stats_list.append({'symbol': sym, 'date': g.iloc[i]['date'],
                                'beats_4q': beats, 'eps_growth_yoy': eps_growth})
earn_stats = pd.DataFrame(earn_stats_list)
print(f"  Earnings stats: {len(earn_stats):,} | t={time.time()-t0:.0f}s")

# ============================================================
# [4/7] MERGE FUNDAMENTALS INTO PRICE DATA
# ============================================================
print("\n[4/7] Merging fundamentales...")

df = df.sort_values('date')
ratios_m = ratios[['symbol', 'date', 'net_margin', 'div_yield', 'payout', 'de_ratio']].copy()
ratios_m = ratios_m.sort_values('date').drop_duplicates(subset=['symbol', 'date'], keep='last')
earn_m = earn_stats[['symbol', 'date', 'beats_4q', 'eps_growth_yoy']].copy()
earn_m = earn_m.sort_values('date').drop_duplicates(subset=['symbol', 'date'], keep='last')

df = pd.merge_asof(df, ratios_m, on='date', by='symbol', direction='backward')
print(f"  Ratios merged")
df = pd.merge_asof(df, earn_m, on='date', by='symbol', direction='backward')
print(f"  Earnings merged | t={time.time()-t0:.0f}s")

# Build indexed version
df_indexed = df.set_index(['symbol', 'date']).sort_index()
df['weekday'] = df['date'].dt.weekday
all_dates_in_index = set(df_indexed.index.get_level_values('date'))

# ============================================================
# [5/7] BUILD WEEKS + PATTERN SCORING LOOP
# ============================================================
print("\n[5/7] Generando picks semanales con pattern scoring...")

fridays = np.sort(df[df['weekday'] == 4]['date'].unique())
mondays = np.sort(df[df['weekday'] == 0]['date'].unique())

# Build week list
weeks_list = []
for fri in fridays:
    fri_ts = pd.Timestamp(fri)
    if fri_ts < pd.Timestamp('2004-01-01'):
        continue
    next_mons = mondays[mondays > fri]
    if len(next_mons) < 2:
        continue
    if fri_ts in all_dates_in_index:
        weeks_list.append({
            'signal': fri_ts,
            'entry': pd.Timestamp(next_mons[0]),
            'exit': pd.Timestamp(next_mons[1]),
        })

print(f"  Semanas a procesar: {len(weeks_list)}")

# Pre-cache date snapshots
date_snap = {}
needed_dates = set()
for w in weeks_list:
    needed_dates.update([w['signal'], w['entry'], w['exit']])
for d in sorted(needed_dates):
    if d in all_dates_in_index:
        date_snap[d] = df_indexed.xs(d, level='date', drop_level=True)
print(f"  Snapshots cacheados: {len(date_snap)} | t={time.time()-t0:.0f}s")


def score_cross_section(cross):
    """Compute LONG_SCORE and SHORT_SCORE for a cross-section of stocks."""
    n = len(cross)
    if n < 20:
        return cross

    # === LONG component scores (0-100, higher = better match for long) ===
    cross['sc_ret12w'] = cross['ret_12w'].rank(pct=True) * 100
    cross['sc_rsi_inv'] = (1 - cross['rsi_14'].rank(pct=True)) * 100
    cross['sc_psar_bear'] = np.where(cross['psar_dist'].isna(), 50,
                                      np.where(cross['psar_dist'] < 0, 100, 0))
    cross['sc_st_bear'] = np.where(cross['st_dist'].isna(), 50,
                                    np.where(cross['st_dist'] < 0, 100, 0))
    cross['sc_bb_inv'] = (1 - cross['bb_pctb'].rank(pct=True)) * 100
    cross['sc_stoch_inv'] = (1 - cross['stoch_k'].rank(pct=True)) * 100
    cross['sc_ret1w_inv'] = (1 - cross['ret_1w'].rank(pct=True, na_option='keep').fillna(0.5)) * 100
    cross['sc_vol'] = cross['vol_20d'].rank(pct=True, na_option='keep').fillna(0.5) * 100
    cross['sc_margin'] = cross['net_margin'].rank(pct=True, na_option='keep').fillna(0.5) * 100
    cross['sc_beats'] = cross['beats_4q'].fillna(2) / 4 * 100  # neutral = 2/4
    cross['sc_epsgr'] = cross['eps_growth_yoy'].rank(pct=True, na_option='keep').fillna(0.5) * 100

    # LONG composite
    cross['LONG_SCORE'] = sum(cross[col] * w for col, w in LONG_W.items())

    # === SHORT component scores (0-100, higher = better match for short) ===
    cross['sc_psar_bull'] = np.where(cross['psar_dist'].isna(), 50,
                                      np.where(cross['psar_dist'] > 0, 100, 0))
    cross['sc_st_bull'] = np.where(cross['st_dist'].isna(), 50,
                                    np.where(cross['st_dist'] > 0, 100, 0))
    cross['sc_stoch'] = cross['stoch_k'].rank(pct=True) * 100
    cross['sc_rsi'] = cross['rsi_14'].rank(pct=True) * 100
    cross['sc_bb'] = cross['bb_pctb'].rank(pct=True) * 100
    cross['sc_ret1w'] = cross['ret_1w'].rank(pct=True, na_option='keep').fillna(0.5) * 100
    cross['sc_ret12w_inv'] = (1 - cross['ret_12w'].rank(pct=True)) * 100
    cross['sc_divyield'] = cross['div_yield'].rank(pct=True, na_option='keep').fillna(0.5) * 100
    cross['sc_payout'] = cross['payout'].rank(pct=True, na_option='keep').fillna(0.5) * 100
    cross['sc_margin_inv'] = (1 - cross['net_margin'].rank(pct=True, na_option='keep').fillna(0.5)) * 100
    cross['sc_debt'] = cross['de_ratio'].rank(pct=True, na_option='keep').fillna(0.5) * 100

    # SHORT composite
    cross['SHORT_SCORE'] = sum(cross[col] * w for col, w in SHORT_W.items())

    return cross


# Main scoring loop
all_weeks = []
progress_interval = max(1, len(weeks_list) // 20)

for wi, w in enumerate(weeks_list):
    sig = w['signal']
    entry = w['entry']
    exit_d = w['exit']

    if wi % progress_interval == 0:
        print(f"  Procesando semana {wi+1}/{len(weeks_list)} ({sig.date()}) | t={time.time()-t0:.0f}s")

    if sig not in date_snap or entry not in date_snap:
        continue

    fri_snap = date_snap[sig]
    en_snap = date_snap[entry]
    has_exit = exit_d in date_snap
    ex_snap = date_snap[exit_d] if has_exit else None

    # Get S&P 500 members
    sp500 = get_sp500_cached(sig)
    eligible = [s for s in fri_snap.index if s in sp500]
    if len(eligible) < 100:
        continue

    # Cross-section for this Friday
    cross = fri_snap.loc[eligible].copy()
    cross = cross.dropna(subset=['ret_12w', 'rsi_14', 'bb_pctb', 'stoch_k'])
    if len(cross) < 50:
        continue

    # Score all stocks
    cross = score_cross_section(cross)

    # === SELECT LONG PICKS ===
    # Filter: ret_12w > 10% AND (PSAR BEAR OR ST BEAR)
    long_mask = (
        (cross['ret_12w'] > 0.10) &
        ((cross['psar_dist'].fillna(1) < 0) | (cross['st_dist'].fillna(1) < 0))
    )
    long_cands = cross[long_mask].dropna(subset=['LONG_SCORE'])

    # Fallback if too few: relax to any ret_12w > 0
    if len(long_cands) < N_LONG:
        long_cands2 = cross[cross['ret_12w'] > 0].dropna(subset=['LONG_SCORE'])
        if len(long_cands2) >= N_LONG:
            long_cands = long_cands2
        else:
            long_cands = cross.dropna(subset=['LONG_SCORE'])

    long_picks = long_cands.nlargest(min(N_LONG, len(long_cands)), 'LONG_SCORE')

    # === SELECT SHORT PICKS ===
    # Filter: PSAR BULL AND ret_12w < 15% AND stoch > 50
    short_mask = (
        (cross['psar_dist'].fillna(-1) > 0) &
        (cross['ret_12w'] < 0.15) &
        (cross['stoch_k'] > 50)
    )
    short_cands = cross[short_mask].dropna(subset=['SHORT_SCORE'])

    # Fallback if too few: relax stoch threshold
    if len(short_cands) < N_SHORT:
        short_cands2 = cross[(cross['psar_dist'].fillna(-1) > 0) & (cross['ret_12w'] < 0.20)].dropna(subset=['SHORT_SCORE'])
        if len(short_cands2) >= N_SHORT:
            short_cands = short_cands2
        else:
            short_cands = cross.dropna(subset=['SHORT_SCORE'])

    short_picks = short_cands.nlargest(min(N_SHORT, len(short_cands)), 'SHORT_SCORE')

    if len(long_picks) == 0 and len(short_picks) == 0:
        continue

    # === COMPUTE RETURNS ===
    picks = []

    for sym in long_picks.index:
        if sym not in en_snap.index:
            continue
        ep = en_snap.loc[sym, 'open']
        if isinstance(ep, pd.Series): ep = ep.iloc[0]
        if pd.isna(ep) or ep <= 0:
            continue

        xp = np.nan; gross_ret = np.nan; net_ret = np.nan; pnl = np.nan
        if has_exit and sym in ex_snap.index:
            xp = ex_snap.loc[sym, 'open']
            if isinstance(xp, pd.Series): xp = xp.iloc[0]
            if pd.notna(xp) and xp > 0:
                gross_ret = (xp - ep) / ep
                net_ret = gross_ret - COST_PCT
                pnl = LONG_POSITION_SIZE * net_ret

        picks.append({
            'symbol': sym, 'side': 'LONG',
            'entry_price': ep, 'exit_price': xp,
            'score': float(long_picks.loc[sym, 'LONG_SCORE']),
            'gross_ret': gross_ret, 'net_ret': net_ret, 'pnl': pnl,
            'position_size': LONG_POSITION_SIZE,
        })

    for sym in short_picks.index:
        if sym not in en_snap.index:
            continue
        ep = en_snap.loc[sym, 'open']
        if isinstance(ep, pd.Series): ep = ep.iloc[0]
        if pd.isna(ep) or ep <= 0:
            continue

        xp = np.nan; gross_ret = np.nan; net_ret = np.nan; pnl = np.nan
        if has_exit and sym in ex_snap.index:
            xp = ex_snap.loc[sym, 'open']
            if isinstance(xp, pd.Series): xp = xp.iloc[0]
            if pd.notna(xp) and xp > 0:
                gross_ret = (ep - xp) / ep  # short: profit when price drops
                net_ret = gross_ret - COST_PCT
                pnl = SHORT_POSITION_SIZE * net_ret

        picks.append({
            'symbol': sym, 'side': 'SHORT',
            'entry_price': ep, 'exit_price': xp,
            'score': float(short_picks.loc[sym, 'SHORT_SCORE']),
            'gross_ret': gross_ret, 'net_ret': net_ret, 'pnl': pnl,
            'position_size': SHORT_POSITION_SIZE,
        })

    # Week summary
    long_p = [p for p in picks if p['side'] == 'LONG']
    short_p = [p for p in picks if p['side'] == 'SHORT']
    valid_pnls = [p['pnl'] for p in picks if pd.notna(p['pnl'])]
    valid_nets = [p['net_ret'] for p in picks if pd.notna(p['net_ret'])]
    long_nets = [p['net_ret'] for p in long_p if pd.notna(p['net_ret'])]
    short_nets = [p['net_ret'] for p in short_p if pd.notna(p['net_ret'])]

    week_data = {
        'signal_date': sig,
        'entry_date': entry,
        'exit_date': exit_d,
        'n_long': len(long_p),
        'n_short': len(short_p),
        'picks': picks,
        'week_pnl': sum(valid_pnls) if valid_pnls else np.nan,
        'week_ret': np.mean(valid_nets) if valid_nets else np.nan,
        'long_ret': np.mean(long_nets) if long_nets else np.nan,
        'short_ret': np.mean(short_nets) if short_nets else np.nan,
        'long_pnl': sum(p['pnl'] for p in long_p if pd.notna(p['pnl'])) if long_p else np.nan,
        'short_pnl': sum(p['pnl'] for p in short_p if pd.notna(p['pnl'])) if short_p else np.nan,
        'has_exit': has_exit,
    }
    all_weeks.append(week_data)

print(f"\n  Semanas generadas: {len(all_weeks)} | t={time.time()-t0:.0f}s")

# ============================================================
# [6/7] RESULTS + STATISTICS
# ============================================================
print(f"\n\n{'='*150}")
print(f"  RESULTADOS DEL BACKTEST - Pattern-Based Multi-Factor Picking")
print(f"{'='*150}")

# --- SENAL MAS RECIENTE ---
latest = all_weeks[-1]
print(f"\n  SENAL MAS RECIENTE:")
print(f"  Senal: {latest['signal_date'].date()} | Entrada: {latest['entry_date'].date()} | Salida: {latest['exit_date'].date()}")
print(f"  Picks: {latest['n_long']}L + {latest['n_short']}S")
print(f"\n  {'#':>3s} | {'Sym':>6s} | {'Lado':>5s} | {'Score':>6s} | {'Entrada$':>10s} | {'Salida$':>10s} | {'Neto%':>7s} | {'P&L $':>10s}")
print(f"  {'-'*3} | {'-'*6} | {'-'*5} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*7} | {'-'*10}")
for i, p in enumerate(latest['picks'], 1):
    xp_str = f"${p['exit_price']:>8.2f}" if pd.notna(p['exit_price']) else "  Pendiente"
    nr_str = f"{p['net_ret']*100:>+6.2f}%" if pd.notna(p['net_ret']) else "    N/A"
    pnl_str = f"${p['pnl']:>+9,.0f}" if pd.notna(p['pnl']) else "      N/A"
    print(f"  {i:>3d} | {p['symbol']:>6s} | {p['side']:>5s} | {p['score']:>5.1f} | ${p['entry_price']:>8.2f} | {xp_str} | {nr_str} | {pnl_str}")
if pd.notna(latest['week_pnl']):
    print(f"\n  TOTAL: ${latest['week_pnl']:>+,.0f} | Longs: ${latest['long_pnl']:>+,.0f} | Shorts: ${latest['short_pnl']:>+,.0f}")

# --- RESUMEN POR ANO ---
print(f"\n\n{'='*150}")
print(f"  RESUMEN POR ANO")
print(f"{'='*150}")

yearly_pnl = {}
for w in all_weeks:
    year = w['signal_date'].year
    if year not in yearly_pnl:
        yearly_pnl[year] = {'pnl': 0, 'long_pnl': 0, 'short_pnl': 0, 'weeks': 0, 'wins': 0, 'rets': []}
    if pd.notna(w['week_pnl']):
        yearly_pnl[year]['pnl'] += w['week_pnl']
        yearly_pnl[year]['long_pnl'] += w['long_pnl'] if pd.notna(w.get('long_pnl')) else 0
        yearly_pnl[year]['short_pnl'] += w['short_pnl'] if pd.notna(w.get('short_pnl')) else 0
        yearly_pnl[year]['weeks'] += 1
        yearly_pnl[year]['rets'].append(w['week_ret'])
        if w['week_pnl'] > 0:
            yearly_pnl[year]['wins'] += 1

print(f"\n  {'Ano':>6s} | {'Sem':>4s} | {'WR%':>4s} | {'Long P&L':>12s} | {'Short P&L':>12s} | {'Total P&L':>12s} | {'P&L/sem':>10s} | {'Ret/sem%':>8s} | {'Sharpe':>6s} | {'Acumulado':>14s}")
print(f"  {'-'*6} | {'-'*4} | {'-'*4} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*6} | {'-'*14}")

cum = 0
for year in sorted(yearly_pnl.keys()):
    yp = yearly_pnl[year]
    cum += yp['pnl']
    wr = yp['wins'] / yp['weeks'] * 100 if yp['weeks'] > 0 else 0
    pnl_pw = yp['pnl'] / yp['weeks'] if yp['weeks'] > 0 else 0
    ret_pw = pnl_pw / TOTAL_CAPITAL * 100
    rets_arr = np.array(yp['rets'])
    sharpe_y = (np.mean(rets_arr) / np.std(rets_arr)) * np.sqrt(52) if len(rets_arr) > 1 and np.std(rets_arr) > 0 else 0
    print(f"  {year:>6d} | {yp['weeks']:>4d} | {wr:>3.0f}% | ${yp['long_pnl']:>+11,.0f} | ${yp['short_pnl']:>+11,.0f} | ${yp['pnl']:>+11,.0f} | ${pnl_pw:>+9,.0f} | {ret_pw:>+7.2f}% | {sharpe_y:>+5.2f} | ${cum:>+13,.0f}")

# Totals
all_pnls = [w['week_pnl'] for w in all_weeks if pd.notna(w['week_pnl'])]
all_rets = np.array([w['week_ret'] for w in all_weeks if pd.notna(w['week_ret'])])
all_long_pnls = [w['long_pnl'] for w in all_weeks if pd.notna(w.get('long_pnl'))]
all_short_pnls = [w['short_pnl'] for w in all_weeks if pd.notna(w.get('short_pnl'))]
total_long = sum(all_long_pnls)
total_short = sum(all_short_pnls)
total_pnl = sum(all_pnls)
total_weeks = len(all_pnls)
n_wins = sum(1 for p in all_pnls if p > 0)
sharpe = (np.mean(all_rets) / np.std(all_rets)) * np.sqrt(52) if np.std(all_rets) > 0 else 0
wr_pct = n_wins / total_weeks * 100 if total_weeks > 0 else 0
pnl_pw = total_pnl / total_weeks if total_weeks > 0 else 0
ret_pw = pnl_pw / TOTAL_CAPITAL * 100

print(f"  {'-'*6} | {'-'*4} | {'-'*4} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*6} | {'-'*14}")
print(f"  {'TOTAL':>6s} | {total_weeks:>4d} | {wr_pct:>3.0f}% | ${total_long:>+11,.0f} | ${total_short:>+11,.0f} | ${total_pnl:>+11,.0f} | ${pnl_pw:>+9,.0f} | {ret_pw:>+7.2f}% | {sharpe:>+5.2f} | ${cum:>+13,.0f}")

# Max Drawdown
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

# --- ESTADISTICAS GLOBALES ---
print(f"\n\n{'='*150}")
print(f"  ESTADISTICAS GLOBALES")
print(f"{'='*150}")

# Long-side stats
long_rets = np.array([w['long_ret'] for w in all_weeks if pd.notna(w.get('long_ret'))])
short_rets = np.array([w['short_ret'] for w in all_weeks if pd.notna(w.get('short_ret'))])
sharpe_l = (np.mean(long_rets) / np.std(long_rets)) * np.sqrt(52) if len(long_rets) > 0 and np.std(long_rets) > 0 else 0
sharpe_s = (np.mean(short_rets) / np.std(short_rets)) * np.sqrt(52) if len(short_rets) > 0 and np.std(short_rets) > 0 else 0

print(f"""
  Capital por semana:     ${TOTAL_CAPITAL:>12,}
  Total semanas:          {total_weeks:>12,}
  Periodo:                {all_weeks[0]['signal_date'].date()} a {all_weeks[-1]['signal_date'].date()}

  P&L total:              ${total_pnl:>+12,.0f}
  P&L medio/semana:       ${pnl_pw:>+12,.0f}
  Ret medio/semana:       {np.mean(all_rets)*100:>+11.3f}%
  Ret anualizado:         {np.mean(all_rets)*52*100:>+11.1f}%

  Sharpe combinado:       {sharpe:>+12.2f}
  Win Rate:               {wr_pct:>11.0f}%
  Wins / Losses:          {n_wins:>5d} / {total_weeks - n_wins:>5d}

  Mejor semana:           ${max(all_pnls):>+12,.0f}
  Peor semana:            ${min(all_pnls):>+12,.0f}
  Max Drawdown:           ${max_dd_pnl:>+12,.0f}

  --- Desglose por lado ---
  Long  P&L total:        ${total_long:>+12,.0f}  Sharpe: {sharpe_l:>+.2f}  Ret medio: {np.mean(long_rets)*100:>+.3f}%
  Short P&L total:        ${total_short:>+12,.0f}  Sharpe: {sharpe_s:>+.2f}  Ret medio: {np.mean(short_rets)*100:>+.3f}%
""")

# --- COMPARACION CON SISTEMA ANTERIOR ---
print(f"{'='*150}")
print(f"  COMPARACION: Pattern-Based vs Regime-Based (5L+5S)")
print(f"{'='*150}")
print(f"""
  Metrica                  | Pattern-Based  | Regime-Based (prev)
  -------------------------+----------------+--------------------
  P&L total                | ${total_pnl:>+12,.0f}  |   +$1,320,989
  Sharpe                   | {sharpe:>+12.2f}  |         +0.56
  Win Rate                 | {wr_pct:>11.0f}%  |           49%
  P&L/semana               | ${pnl_pw:>+12,.0f}  |      +$1,182
  Max Drawdown             | ${max_dd_pnl:>+12,.0f}  |    -$246,000
  Long P&L                 | ${total_long:>+12,.0f}  |  +$1,497,817
  Short P&L                | ${total_short:>+12,.0f}  |    -$176,828
  Long Sharpe              | {sharpe_l:>+12.2f}  |         +0.65
  Short Sharpe             | {sharpe_s:>+12.2f}  |         -0.13
""")

print(f"  Tiempo total: {time.time()-t0:.0f}s")

# ============================================================
# [7/7] EXPORT TO EXCEL
# ============================================================
print(f"\n  Exportando a Excel...")

# Sheet 1: Weekly Summary
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

# Sheet 2: Individual Picks
pick_rows = []
for w in all_weeks:
    for p in w['picks']:
        pick_rows.append({
            'Signal_Date': w['signal_date'].date(),
            'Entry_Date': w['entry_date'].date(),
            'Exit_Date': w['exit_date'].date() if w['exit_date'] is not None else None,
            'Year': w['signal_date'].year,
            'Symbol': p['symbol'],
            'Side': p['side'],
            'Score': round(p['score'], 1),
            'Entry_Price': round(p['entry_price'], 2),
            'Exit_Price': round(p['exit_price'], 2) if pd.notna(p['exit_price']) else None,
            'Gross_Ret_Pct': round(p['gross_ret'] * 100, 3) if pd.notna(p['gross_ret']) else None,
            'Net_Ret_Pct': round(p['net_ret'] * 100, 3) if pd.notna(p['net_ret']) else None,
            'PnL_USD': round(p['pnl'], 0) if pd.notna(p['pnl']) else None,
            'Position_Size': p.get('position_size', LONG_POSITION_SIZE),
        })
df_picks = pd.DataFrame(pick_rows)

# Sheet 3: Annual Summary
annual_rows = []
cum_a = 0
for year in sorted(yearly_pnl.keys()):
    yp = yearly_pnl[year]
    cum_a += yp['pnl']
    wr = yp['wins'] / yp['weeks'] * 100 if yp['weeks'] > 0 else 0
    pnl_pw_y = yp['pnl'] / yp['weeks'] if yp['weeks'] > 0 else 0
    ret_pw_y = pnl_pw_y / TOTAL_CAPITAL * 100
    rets_arr_y = np.array(yp['rets'])
    sharpe_y = (np.mean(rets_arr_y) / np.std(rets_arr_y)) * np.sqrt(52) if len(rets_arr_y) > 1 and np.std(rets_arr_y) > 0 else 0
    annual_rows.append({
        'Year': year,
        'Weeks': yp['weeks'],
        'Wins': yp['wins'],
        'Losses': yp['weeks'] - yp['wins'],
        'Win_Rate_Pct': round(wr, 1),
        'Long_PnL': round(yp['long_pnl'], 0),
        'Short_PnL': round(yp['short_pnl'], 0),
        'PnL_USD': round(yp['pnl'], 0),
        'PnL_per_Week': round(pnl_pw_y, 0),
        'Ret_per_Week_Pct': round(ret_pw_y, 2),
        'Ret_Annual_Pct': round(ret_pw_y * 52, 1),
        'Sharpe': round(sharpe_y, 2),
        'Cumulative_PnL': round(cum_a, 0),
    })
df_annual = pd.DataFrame(annual_rows)

# Sheet 4: Pattern Weights
weight_rows = []
for col, w in LONG_W.items():
    weight_rows.append({'Side': 'LONG', 'Factor': col, 'Weight': w})
for col, w in SHORT_W.items():
    weight_rows.append({'Side': 'SHORT', 'Factor': col, 'Weight': w})
df_weights = pd.DataFrame(weight_rows)

# Write Excel
excel_path = 'data/pattern_weekly_picks.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_weeks.to_excel(writer, sheet_name='Semanas', index=False)
    df_picks.to_excel(writer, sheet_name='Picks', index=False)
    df_annual.to_excel(writer, sheet_name='Resumen_Anual', index=False)
    df_weights.to_excel(writer, sheet_name='Pattern_Weights', index=False)

print(f"  Exportado: {excel_path}")
print(f"    - Semanas: {len(df_weeks):,} filas")
print(f"    - Picks: {len(df_picks):,} filas")
print(f"    - Resumen Anual: {len(df_annual)} filas")
print(f"    - Pattern Weights: {len(df_weights)} filas")

print(f"\n{'='*150}")
print(f"  FIN BACKTEST PATTERN-BASED")
print(f"{'='*150}")
