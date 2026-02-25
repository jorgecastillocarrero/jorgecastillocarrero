"""
Year-by-year comparison: Always Short vs Long Only vs ETF ATR>=1.3% vs ETF ATR>=1.6%
With Long/Short/Total for each + SPY benchmark
"""
import pandas as pd
import numpy as np
import psycopg2

DB = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'
from backtest_sector_events_v2 import load_etf_returns, backtest, build_weekly_events, SECTOR_ETFS

print("Loading data...")
etf_prices, etf_returns = load_etf_returns()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# Load OHLC for ATR
conn = psycopg2.connect(DB)
symbols = ['SPY'] + SECTOR_ETFS
ph = ','.join(['%s'] * len(symbols))
ohlc = pd.read_sql(
    f"SELECT symbol, date, high, low, close FROM fmp_price_history "
    f"WHERE symbol IN ({ph}) ORDER BY date",
    conn, params=symbols, parse_dates=['date']
)
conn.close()

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

# SPY benchmark
spy_df = ohlc[ohlc['symbol'] == 'SPY'].copy().sort_values('date').set_index('date')
spy_w = spy_df['close'].resample('W-FRI').last()
spy_ret_w = spy_w.pct_change()

# Backtest
res = backtest(weekly_events, etf_returns, n_long=3, n_short=3,
               momentum_decay=0.0, min_score=1.0)

# Merge shorted ETF ATR (lagged)
res = res.set_index('date').sort_index()
atr_raw = []
for date, row in res.iterrows():
    if row['shorts'] and isinstance(row['shorts'], str):
        etfs = row['shorts'].split(',')
        atrs = [etf_atr[e].loc[date] for e in etfs if e in etf_atr and date in etf_atr[e].index]
        atr_raw.append(np.mean(atrs) if atrs else np.nan)
    else:
        atr_raw.append(np.nan)
res['atr_raw'] = atr_raw
res['atr_lag'] = res['atr_raw'].shift(1)
res = res.reset_index()

# Build filtered versions
def make_filtered(res, fn):
    sim = res.copy()
    for idx, row in sim.iterrows():
        if not fn(row):
            sim.at[idx, 'short_pnl'] = 0
            sim.at[idx, 'n_shorts'] = 0
    sim['total_pnl'] = sim['long_pnl'] + sim['short_pnl']
    return sim

configs = {
    "Always Short": make_filtered(res, lambda r: True),
    "Long Only":    make_filtered(res, lambda r: False),
    "ATR>=1.3%":    make_filtered(res, lambda r: r['atr_lag'] >= 1.3 if pd.notna(r.get('atr_lag')) else False),
    "ATR>=1.6%":    make_filtered(res, lambda r: r['atr_lag'] >= 1.6 if pd.notna(r.get('atr_lag')) else False),
}

cnames = list(configs.keys())
years = sorted(res['year'].unique())

# ════════════════════════════════════════════════════════════════
# TABLA 1: Total PnL por año y configuración
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 105}")
print(f"  TOTAL PnL POR AÑO")
print(f"{'=' * 105}")
print(f"\n  {'Year':>6} {'SPY':>7}  {'Always Short':>14} {'Long Only':>14} {'ATR>=1.3%':>14} {'ATR>=1.6%':>14}  {'Best':>14}")
print("  " + "-" * 100)

cum = {c: 0 for c in cnames}
wins_count = {c: 0 for c in cnames}

for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0

    pnls = {}
    for c in cnames:
        yr = configs[c][configs[c]['year'] == year]
        pnls[c] = yr['total_pnl'].sum()
        cum[c] += pnls[c]
        if pnls[c] > 0:
            wins_count[c] += 1

    best = max(pnls, key=pnls.get)

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")
    for c in cnames:
        v = pnls[c] / 1000
        marker = " <<" if c == best else "   "
        print(f"  {v:>+8.0f}K{marker}", end="")
    print(f"  {best:>14s}")

print("  " + "-" * 100)
print(f"  {'TOTAL':>6} {'':>7}", end="")
for c in cnames:
    print(f"  {cum[c]/1000:>+8.0f}K   ", end="")
print()

# ════════════════════════════════════════════════════════════════
# TABLA 2: Long PnL por año (igual en todas, solo verificar)
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 105}")
print(f"  LONG PnL POR AÑO (igual en todas las configs)")
print(f"{'=' * 105}")
print(f"\n  {'Year':>6} {'SPY':>7} {'Long PnL':>12}")
print("  " + "-" * 30)

total_long = 0
for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0
    yr = configs["Always Short"][configs["Always Short"]['year'] == year]
    lpnl = yr['long_pnl'].sum()
    total_long += lpnl
    print(f"  {year:>6d} {spy_ann:>+6.1%} ${lpnl:>+11,.0f}")
print("  " + "-" * 30)
print(f"  {'TOTAL':>6} {'':>7} ${total_long:>+11,.0f}")

# ════════════════════════════════════════════════════════════════
# TABLA 3: Short PnL por año y configuración
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  SHORT PnL POR AÑO")
print(f"{'=' * 90}")
print(f"\n  {'Year':>6} {'SPY':>7}  {'Always Short':>14} {'ATR>=1.3%':>14} {'ATR>=1.6%':>14}  {'Mejor?':>8}")
print("  " + "-" * 75)

scum = {c: 0 for c in cnames}
for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0

    spnls = {}
    for c in ["Always Short", "ATR>=1.3%", "ATR>=1.6%"]:
        yr = configs[c][configs[c]['year'] == year]
        spnls[c] = yr['short_pnl'].sum()
        scum[c] += spnls[c]

    # Did filter improve short?
    imp13 = spnls["ATR>=1.3%"] - spnls["Always Short"]
    imp16 = spnls["ATR>=1.6%"] - spnls["Always Short"]
    best_imp = "1.3%" if imp13 > imp16 else "1.6%"
    if imp13 <= 0 and imp16 <= 0:
        best_imp = "none"

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")
    for c in ["Always Short", "ATR>=1.3%", "ATR>=1.6%"]:
        v = spnls[c] / 1000
        print(f"  {v:>+8.0f}K     ", end="")
    # Show improvement of 1.3% vs always
    print(f"  {imp13/1000:>+6.0f}K")

print("  " + "-" * 75)
print(f"  {'TOTAL':>6} {'':>7}", end="")
for c in ["Always Short", "ATR>=1.3%", "ATR>=1.6%"]:
    print(f"  {scum[c]/1000:>+8.0f}K     ", end="")
imp_total = scum["ATR>=1.3%"] - scum["Always Short"]
print(f"  {imp_total/1000:>+6.0f}K")

# ════════════════════════════════════════════════════════════════
# TABLA 4: Métricas resumen
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  MÉTRICAS RESUMEN")
print(f"{'=' * 90}")

print(f"\n  {'Métrica':>22s}", end="")
for c in cnames:
    print(f"  {c:>16s}", end="")
print()
print("  " + "-" * 88)

# Total PnL
print(f"  {'Total PnL':>22s}", end="")
for c in cnames:
    print(f"  ${cum[c]:>14,.0f}", end="")
print()

# Sharpe
print(f"  {'Sharpe':>22s}", end="")
for c in cnames:
    aw = configs[c]['total_pnl'].values
    sh = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    print(f"  {sh:>16.2f}", end="")
print()

# MaxDD
print(f"  {'Max Drawdown':>22s}", end="")
for c in cnames:
    aw = configs[c]['total_pnl'].values
    cum_arr = np.cumsum(aw)
    dd = (cum_arr - np.maximum.accumulate(cum_arr)).min()
    print(f"  ${dd:>14,.0f}", end="")
print()

# Positive years
print(f"  {'Años positivos':>22s}", end="")
for c in cnames:
    yr_pnl = configs[c].groupby('year')['total_pnl'].sum()
    yp = (yr_pnl > 0).sum()
    yt = len(yr_pnl)
    print(f"  {yp:>10d}/{yt:<4d}", end="")
print()

# Años negativos
print(f"  {'Años negativos':>22s}", end="")
for c in cnames:
    yr_pnl = configs[c].groupby('year')['total_pnl'].sum()
    yn = (yr_pnl < 0).sum()
    yt = len(yr_pnl)
    print(f"  {yn:>10d}/{yt:<4d}", end="")
print()

# Worst year
print(f"  {'Peor año':>22s}", end="")
for c in cnames:
    yr_pnl = configs[c].groupby('year')['total_pnl'].sum()
    wy = yr_pnl.idxmin()
    wv = yr_pnl.min()
    print(f"  {wy} ${wv/1000:>+6.0f}K", end="")
print()

# Best year
print(f"  {'Mejor año':>22s}", end="")
for c in cnames:
    yr_pnl = configs[c].groupby('year')['total_pnl'].sum()
    by = yr_pnl.idxmax()
    bv = yr_pnl.max()
    print(f"  {by} ${bv/1000:>+6.0f}K", end="")
print()

# Avg annual
print(f"  {'PnL anual medio':>22s}", end="")
for c in cnames:
    yr_pnl = configs[c].groupby('year')['total_pnl'].sum()
    print(f"  ${yr_pnl.mean():>14,.0f}", end="")
print()

# Short weeks
print(f"  {'Semanas con short':>22s}", end="")
for c in cnames:
    swks = (configs[c]['n_shorts'] > 0).sum()
    twks = len(configs[c])
    pct = swks / twks * 100
    print(f"  {swks:>8d} ({pct:>3.0f}%)", end="")
print()

# Win rate
print(f"  {'Win Rate semanal':>22s}", end="")
for c in cnames:
    aw = configs[c]['total_pnl'].values
    act = sum(1 for x in aw if x != 0)
    w = sum(1 for x in aw if x > 0)
    print(f"  {w/max(1,act)*100:>13.1f}%  ", end="")
print()

# Veces que es el mejor
print(f"\n  {'Veces mejor año':>22s}", end="")
for c in cnames:
    count = 0
    for year in years:
        pnls = {cn: configs[cn][configs[cn]['year'] == year]['total_pnl'].sum() for cn in cnames}
        if max(pnls, key=pnls.get) == c:
            count += 1
    print(f"  {count:>10d}/{len(years):<4d}", end="")
print()

print(f"\n{'=' * 90}")
