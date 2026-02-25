"""
Optimize: when to be Long-only, Short-only, or Both, based on market regime.
Regime signals: SPY SMA, ATR, momentum, combinations.
"""
import pandas as pd
import numpy as np
import psycopg2
from backtest_sector_events_v2 import load_etf_returns, backtest, build_weekly_events, SECTOR_ETFS

DB = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'

print("Loading data...")
etf_prices, etf_returns = load_etf_returns()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# Load SPY OHLC + sector ETF OHLC
conn = psycopg2.connect(DB)
symbols = ['SPY'] + SECTOR_ETFS
ph = ','.join(['%s'] * len(symbols))
ohlc = pd.read_sql(
    f"SELECT symbol, date, high, low, close FROM fmp_price_history "
    f"WHERE symbol IN ({ph}) ORDER BY date",
    conn, params=symbols, parse_dates=['date']
)
conn.close()

# ── Compute all signals ──
spy_df = ohlc[ohlc['symbol'] == 'SPY'].copy().sort_values('date').set_index('date')

# SMAs
for w in [10, 20, 30, 40, 50]:
    spy_df[f'sma{w}w'] = spy_df['close'].rolling(w * 5).mean()  # daily SMA

# Weekly
spy_w = spy_df['close'].resample('W-FRI').last()
spy_ret_w = spy_w.pct_change()

# Weekly SMAs
spy_w_df = spy_w.to_frame('close')
for w in [4, 8, 13, 20, 26, 40, 52]:
    spy_w_df[f'sma{w}'] = spy_w_df['close'].rolling(w).mean()

# Momentum
for w in [4, 8, 13, 26]:
    spy_w_df[f'mom{w}'] = spy_w_df['close'].pct_change(w)

# ATR
spy_daily = ohlc[ohlc['symbol'] == 'SPY'].copy().sort_values('date')
spy_daily['prev_close'] = spy_daily['close'].shift(1)
spy_daily['tr'] = np.maximum(spy_daily['high'] - spy_daily['low'],
    np.maximum(abs(spy_daily['high'] - spy_daily['prev_close']),
               abs(spy_daily['low'] - spy_daily['prev_close'])))
spy_daily['atr14'] = spy_daily['tr'].rolling(14).mean()
spy_daily['atr_pct'] = spy_daily['atr14'] / spy_daily['close'] * 100
spy_atr_w = spy_daily.set_index('date')['atr_pct'].resample('W-FRI').last()
spy_w_df['atr'] = spy_atr_w

# ETF ATRs
def compute_atr_pct(df, period=14):
    df = df.sort_values('date').copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['prev_close']), abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(period).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    return df.set_index('date')['atr_pct'].dropna()

etf_atr = {}
for sym in SECTOR_ETFS:
    sdf = ohlc[ohlc['symbol'] == sym].copy()
    if len(sdf) >= 20:
        etf_atr[sym] = compute_atr_pct(sdf, 14).resample('W-FRI').last().dropna()

# Run base backtest
res = backtest(weekly_events, etf_returns, n_long=3, n_short=3,
               momentum_decay=0.0, min_score=1.0)

# Merge signals (ALL LAGGED by 1 week for no look-ahead)
res = res.set_index('date').sort_index()

for col in spy_w_df.columns:
    res[f'spy_{col}'] = spy_w_df[col].shift(1)  # lagged

# Derived signals (from lagged data)
res['spy_above_sma13'] = res['spy_close'] > res['spy_sma13']
res['spy_above_sma20'] = res['spy_close'] > res['spy_sma20']
res['spy_above_sma26'] = res['spy_close'] > res['spy_sma26']
res['spy_above_sma40'] = res['spy_close'] > res['spy_sma40']
res['spy_above_sma52'] = res['spy_close'] > res['spy_sma52']
res['spy_mom4_pos'] = res['spy_mom4'] > 0
res['spy_mom13_pos'] = res['spy_mom13'] > 0
res['spy_mom26_pos'] = res['spy_mom26'] > 0
res['spy_atr_high'] = res['spy_atr'] >= 1.3

# Shorted ETF ATR (lagged)
atr_raw = []
for date, row in res.iterrows():
    if row['shorts'] and isinstance(row['shorts'], str):
        etfs = row['shorts'].split(',')
        atrs = [etf_atr[e].loc[date] for e in etfs if e in etf_atr and date in etf_atr[e].index]
        atr_raw.append(np.mean(atrs) if atrs else np.nan)
    else:
        atr_raw.append(np.nan)
res['short_etf_atr'] = pd.Series(atr_raw, index=res.index).shift(1)

res = res.dropna(subset=['spy_close']).reset_index()
print(f"  {len(res)} weeks with all signals")

# ═══════════════════════════════════════════════════════
# ANALYSIS 1: Long PnL by regime
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  LONG PnL BY MARKET REGIME")
print(f"{'=' * 90}")

active_long = res[res['n_longs'] > 0].copy()

for signal, label in [
    ('spy_above_sma13', 'SPY > SMA13w'),
    ('spy_above_sma20', 'SPY > SMA20w'),
    ('spy_above_sma26', 'SPY > SMA26w'),
    ('spy_above_sma40', 'SPY > SMA40w'),
    ('spy_above_sma52', 'SPY > SMA52w'),
    ('spy_mom4_pos', 'SPY Mom4w > 0'),
    ('spy_mom13_pos', 'SPY Mom13w > 0'),
    ('spy_mom26_pos', 'SPY Mom26w > 0'),
]:
    v = active_long[active_long[signal].notna()].copy()
    on = v[v[signal] == True]
    off = v[v[signal] == False]

    l_on = on['long_pnl'].sum()
    l_off = off['long_pnl'].sum()
    avg_on = on['long_pnl'].mean() if len(on) > 0 else 0
    avg_off = off['long_pnl'].mean() if len(off) > 0 else 0
    win_on = (on['long_pnl'] > 0).sum() / max(1, len(on)) * 100
    win_off = (off['long_pnl'] > 0).sum() / max(1, len(off)) * 100

    print(f"\n  {label}:")
    print(f"    YES: {len(on):>4d} wks  Long: ${l_on:>+10,.0f}  Avg: ${avg_on:>+7,.0f}  Win: {win_on:.0f}%")
    print(f"     NO: {len(off):>4d} wks  Long: ${l_off:>+10,.0f}  Avg: ${avg_off:>+7,.0f}  Win: {win_off:.0f}%")

# ═══════════════════════════════════════════════════════
# ANALYSIS 2: Short PnL by regime
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  SHORT PnL BY MARKET REGIME")
print(f"{'=' * 90}")

active_short = res[res['n_shorts'] > 0].copy()

for signal, label in [
    ('spy_above_sma13', 'SPY > SMA13w'),
    ('spy_above_sma20', 'SPY > SMA20w'),
    ('spy_above_sma26', 'SPY > SMA26w'),
    ('spy_above_sma40', 'SPY > SMA40w'),
    ('spy_above_sma52', 'SPY > SMA52w'),
    ('spy_mom4_pos', 'SPY Mom4w > 0'),
    ('spy_mom13_pos', 'SPY Mom13w > 0'),
    ('spy_atr_high', 'SPY ATR >= 1.3%'),
]:
    v = active_short[active_short[signal].notna()].copy()
    on = v[v[signal] == True]
    off = v[v[signal] == False]

    s_on = on['short_pnl'].sum()
    s_off = off['short_pnl'].sum()
    avg_on = on['short_pnl'].mean() if len(on) > 0 else 0
    avg_off = off['short_pnl'].mean() if len(off) > 0 else 0
    win_on = (on['short_pnl'] > 0).sum() / max(1, len(on)) * 100
    win_off = (off['short_pnl'] > 0).sum() / max(1, len(off)) * 100

    print(f"\n  {label}:")
    print(f"    YES: {len(on):>4d} wks  Short: ${s_on:>+10,.0f}  Avg: ${avg_on:>+7,.0f}  Win: {win_on:.0f}%")
    print(f"     NO: {len(off):>4d} wks  Short: ${s_off:>+10,.0f}  Avg: ${avg_off:>+7,.0f}  Win: {win_off:.0f}%")

# ═══════════════════════════════════════════════════════
# BACKTEST: Regime-adaptive strategies
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  BACKTEST: Regime-Adaptive Strategies")
print(f"{'=' * 115}")

def run_regime(res, long_fn, short_fn, label):
    """Apply regime: long_fn returns True to keep longs, short_fn for shorts."""
    sim = res.copy()
    for idx, row in sim.iterrows():
        if not long_fn(row):
            sim.at[idx, 'long_pnl'] = 0
            sim.at[idx, 'n_longs'] = 0
        if not short_fn(row):
            sim.at[idx, 'short_pnl'] = 0
            sim.at[idx, 'n_shorts'] = 0
    sim['total_pnl'] = sim['long_pnl'] + sim['short_pnl']
    return sim

def metrics(sim):
    aw = sim['total_pnl'].values
    tot = aw.sum()
    ltot = sim['long_pnl'].sum()
    stot = sim['short_pnl'].sum()
    sh = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    cum = np.cumsum(aw)
    dd = (cum - np.maximum.accumulate(cum)).min()
    yr_pnl = sim.groupby('year')['total_pnl'].sum()
    yp = (yr_pnl > 0).sum()
    yt = len(yr_pnl)
    lwks = (sim['n_longs'] > 0).sum()
    swks = (sim['n_shorts'] > 0).sum()
    return tot, ltot, stot, sh, dd, yp, yt, lwks, swks

print(f"\n  {'Rule':>45s} {'Total':>10} {'Long':>10} {'Short':>10} {'Sharpe':>7} {'MaxDD':>10} {'Yr+':>5} {'LWks':>5} {'SWks':>5}")
print("  " + "-" * 115)

# Helper lambdas
def always(r): return True
def never(r): return False
def above_sma(w):
    col = f'spy_above_sma{w}'
    return lambda r: r[col] if pd.notna(r.get(col)) else True
def below_sma(w):
    col = f'spy_above_sma{w}'
    return lambda r: not r[col] if pd.notna(r.get(col)) else False
def mom_pos(w):
    col = f'spy_mom{w}_pos'
    return lambda r: r[col] if pd.notna(r.get(col)) else True
def mom_neg(w):
    col = f'spy_mom{w}_pos'
    return lambda r: not r[col] if pd.notna(r.get(col)) else False
def atr_high(r): return r['spy_atr_high'] if pd.notna(r.get('spy_atr_high')) else False
def atr_low(r): return not r['spy_atr_high'] if pd.notna(r.get('spy_atr_high')) else True
def etf_atr_13(r): return r['short_etf_atr'] >= 1.3 if pd.notna(r.get('short_etf_atr')) else False

strategies = [
    # Baselines
    ("BASELINE: Always L+S",                always, always),
    ("Long Only",                           always, never),
    ("Short Only",                          never, always),

    # Long filter only (keep shorts always)
    ("Long: SPY>SMA20 | Short: always",     above_sma(20), always),
    ("Long: SPY>SMA26 | Short: always",     above_sma(26), always),
    ("Long: SPY>SMA40 | Short: always",     above_sma(40), always),
    ("Long: Mom13>0 | Short: always",       mom_pos(13), always),

    # Short filter only (keep longs always)
    ("Long: always | Short: SPY<SMA20",     always, below_sma(20)),
    ("Long: always | Short: SPY<SMA26",     always, below_sma(26)),
    ("Long: always | Short: SPY<SMA40",     always, below_sma(40)),
    ("Long: always | Short: Mom13<0",       always, mom_neg(13)),
    ("Long: always | Short: ATR>=1.3",      always, lambda r: atr_high(r)),
    ("Long: always | Short: ETF ATR>=1.3",  always, etf_atr_13),

    # Combined: Long when bull, Short when bear
    ("L:SPY>SMA20 | S:SPY<SMA20",          above_sma(20), below_sma(20)),
    ("L:SPY>SMA26 | S:SPY<SMA26",          above_sma(26), below_sma(26)),
    ("L:SPY>SMA40 | S:SPY<SMA40",          above_sma(40), below_sma(40)),
    ("L:Mom13>0 | S:Mom13<0",              mom_pos(13), mom_neg(13)),

    # Combined: Long when bull, Short when bear+vol
    ("L:SPY>SMA26 | S:SPY<SMA26+ATR1.3",   above_sma(26), lambda r: below_sma(26)(r) and atr_high(r)),
    ("L:SPY>SMA40 | S:SPY<SMA40+ATR1.3",   above_sma(40), lambda r: below_sma(40)(r) and atr_high(r)),
    ("L:Mom13>0 | S:Mom13<0+ATR1.3",        mom_pos(13), lambda r: mom_neg(13)(r) and atr_high(r)),

    # Always long + filtered short
    ("L:always | S:SPY<SMA26+ETF_ATR1.3",   always, lambda r: below_sma(26)(r) and etf_atr_13(r)),
    ("L:always | S:SPY<SMA40+ETF_ATR1.3",   always, lambda r: below_sma(40)(r) and etf_atr_13(r)),

    # Long filtered + filtered short
    ("L:SPY>SMA26 | S:ETF_ATR>=1.3",        above_sma(26), etf_atr_13),
    ("L:SPY>SMA40 | S:ETF_ATR>=1.3",        above_sma(40), etf_atr_13),
    ("L:Mom13>0 | S:ETF_ATR>=1.3",          mom_pos(13), etf_atr_13),

    # Momentum cross
    ("L:Mom4>0 | S:Mom4<0+ATR1.3",          mom_pos(4), lambda r: mom_neg(4)(r) and atr_high(r)),
    ("L:Mom4>0 | S:ETF_ATR>=1.3",           mom_pos(4), etf_atr_13),
]

results = []
for label, lfn, sfn in strategies:
    sim = run_regime(res, lfn, sfn, label)
    tot, ltot, stot, sh, dd, yp, yt, lwks, swks = metrics(sim)
    results.append((label, tot, ltot, stot, sh, dd, yp, yt, lwks, swks))
    print(f"  {label:>45s} ${tot/1000:>+8.0f}K ${ltot/1000:>+8.0f}K ${stot/1000:>+8.0f}K {sh:>+6.2f} ${dd/1000:>+8.0f}K {yp:>3d}/{yt:<2d} {lwks:>5d} {swks:>5d}")

# Sort by Sharpe
print(f"\n\n{'=' * 90}")
print(f"  TOP 10 BY SHARPE")
print(f"{'=' * 90}")
sorted_res = sorted(results, key=lambda x: x[4], reverse=True)
print(f"\n  {'#':>3} {'Rule':>45s} {'Total':>10} {'Sharpe':>7} {'MaxDD':>10} {'Yr+':>5}")
print("  " + "-" * 85)
for i, (label, tot, ltot, stot, sh, dd, yp, yt, lwks, swks) in enumerate(sorted_res[:10]):
    print(f"  {i+1:>3d} {label:>45s} ${tot/1000:>+8.0f}K {sh:>+6.2f} ${dd/1000:>+8.0f}K {yp:>3d}/{yt:<2d}")

# ═══════════════════════════════════════════════════════
# ANNUAL DETAIL for top 3
# ═══════════════════════════════════════════════════════
top3_labels = [r[0] for r in sorted_res[:3]]
print(f"\n\n{'=' * 120}")
print(f"  ANNUAL DETAIL: Top 3 strategies")
print(f"{'=' * 120}")

# Re-run top 3
top3_sims = {}
for label, lfn, sfn in strategies:
    if label in top3_labels:
        top3_sims[label] = run_regime(res, lfn, sfn, label)

# Also always short baseline
top3_sims["BASELINE: Always L+S"] = run_regime(res, always, always, "baseline")
if "BASELINE: Always L+S" not in top3_labels:
    top3_labels = ["BASELINE: Always L+S"] + top3_labels[:3]
else:
    top3_labels = top3_labels[:3]

capital = 1_000_000

years = sorted(res['year'].unique())
print(f"\n  {'Year':>6} {'SPY':>7}", end="")
for label in top3_labels:
    short_label = label[:22]
    print(f"  {short_label:>22s}", end="")
print()
print("  " + "-" * (14 + 24 * len(top3_labels)))

for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")
    for label in top3_labels:
        sim = top3_sims[label]
        yr = sim[sim['year'] == year]
        pnl = yr['total_pnl'].sum()
        ret = pnl / capital * 100
        print(f"  {ret:>+7.1f}% (${pnl/1000:>+5.0f}K)", end="")
    print()

print("  " + "-" * (14 + 24 * len(top3_labels)))
print(f"  {'TOTAL':>6} {'':>7}", end="")
for label in top3_labels:
    sim = top3_sims[label]
    pnl = sim['total_pnl'].sum()
    ret = pnl / capital * 100
    print(f"  {ret:>+7.1f}% (${pnl/1000:>+5.0f}K)", end="")
print()

print(f"\n{'=' * 120}")
