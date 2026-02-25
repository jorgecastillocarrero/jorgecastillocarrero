"""
Year-by-year RETURNS for ATR>=1.3% filter - capital base $500K total
$500K always invested: longs use $500K, shorts use $500K (same capital, not additive)
"""
import pandas as pd
import numpy as np
import psycopg2
from backtest_sector_events_v2 import load_etf_returns, backtest, build_weekly_events, SECTOR_ETFS

DB = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'

print("Loading data...")
etf_prices, etf_returns = load_etf_returns()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# ATR
conn = psycopg2.connect(DB)
symbols = ['SPY'] + SECTOR_ETFS
ph = ','.join(['%s'] * len(symbols))
ohlc = pd.read_sql(
    f"SELECT symbol, date, high, low, close FROM fmp_price_history "
    f"WHERE symbol IN ({ph}) ORDER BY date",
    conn, params=symbols, parse_dates=['date']
)
spy = pd.read_sql("SELECT date, close FROM fmp_price_history WHERE symbol='SPY' ORDER BY date",
                   conn, parse_dates=['date'])
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

spy = spy.set_index('date').sort_index()
spy_w = spy['close'].resample('W-FRI').last()
spy_ret_w = spy_w.pct_change()

# Backtest with $500K per side
res = backtest(weekly_events, etf_returns, n_long=3, n_short=3,
               capital_per_side=500_000,
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

# Apply ATR>=1.3% filter
for idx, row in res.iterrows():
    keep = row['atr_lag'] >= 1.3 if pd.notna(row.get('atr_lag')) else False
    if not keep:
        res.at[idx, 'short_pnl'] = 0
        res.at[idx, 'n_shorts'] = 0
res['total_pnl'] = res['long_pnl'] + res['short_pnl']

CAPITAL = 500_000  # Capital total invertido

years = sorted(res['year'].unique())
years = [y for y in years if y >= 2000]  # skip 1999

print(f"\n{'=' * 115}")
print(f"  RETORNOS ANUALES - ATR>=1.3% Filter 3L+3S (Capital: ${CAPITAL:,.0f})")
print(f"  $500K long always + $500K short cuando ETF ATR >= 1.3%")
print(f"{'=' * 115}")

print(f"\n  {'Año':>6} {'SPY':>8} {'Strat':>8} {'Long':>8} {'Short':>8} {'Alpha':>8} {'PnL':>12} {'Cumul PnL':>12} {'Cumul %':>8} {'ShWks':>6}")
print("  " + "-" * 105)

cumul_pnl = 0
cumul_strat = 1.0
cumul_spy = 1.0
yrs_pos = 0
yrs_neg = 0
yrs_alpha = 0
alphas = []
annual_rets = []

for year in years:
    yr = res[res['year'] == year]
    pnl = yr['total_pnl'].sum()
    lpnl = yr['long_pnl'].sum()
    spnl = yr['short_pnl'].sum()
    cumul_pnl += pnl
    swks = (yr['n_shorts'] > 0).sum()

    # Returns sobre $500K capital
    ret_total = pnl / CAPITAL * 100
    ret_long = lpnl / CAPITAL * 100
    ret_short = spnl / CAPITAL * 100
    annual_rets.append(ret_total)

    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1
    spy_pct = spy_ann * 100

    alpha = ret_total - spy_pct
    alphas.append(alpha)
    if pnl > 0: yrs_pos += 1
    if pnl < 0: yrs_neg += 1
    if alpha > 0: yrs_alpha += 1

    cumul_spy *= (1 + spy_ann)
    cumul_strat *= (1 + pnl / CAPITAL)
    cumul_ret = (cumul_strat - 1) * 100

    print(f"  {year:>6d} {spy_pct:>+7.1f}% {ret_total:>+7.1f}% {ret_long:>+7.1f}% {ret_short:>+7.1f}% {alpha:>+7.1f}% ${pnl:>+11,.0f} ${cumul_pnl:>+11,.0f} {cumul_ret:>+7.1f}% {swks:>5d}")

print("  " + "-" * 105)

n_years = len(years)
total_pnl = res[res['year'] >= 2000]['total_pnl'].sum()
total_ret = (cumul_strat - 1) * 100
spy_total_ret = (cumul_spy - 1) * 100

cagr_strat = (cumul_strat ** (1/n_years) - 1) * 100
cagr_spy = (cumul_spy ** (1/n_years) - 1) * 100

aw = res[res['year'] >= 2000]['total_pnl'].values
sharpe = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
cum_arr = np.cumsum(aw)
dd_abs = (cum_arr - np.maximum.accumulate(cum_arr)).min()
dd_pct = dd_abs / CAPITAL * 100

# Win rate
wins = (aw > 0).sum()
active = (aw != 0).sum()

print(f"\n  {'MÉTRICAS':>25s}")
print("  " + "-" * 60)
print(f"  {'Capital invertido':>25s}: ${CAPITAL:>,.0f}")
print(f"  {'Total PnL':>25s}: ${total_pnl:>+,.0f}")
print(f"  {'Retorno total':>25s}: {total_ret:>+.1f}%")
print(f"  {'SPY total':>25s}: {spy_total_ret:>+.1f}%")
print(f"  {'CAGR estrategia':>25s}: {cagr_strat:>+.1f}%")
print(f"  {'CAGR SPY':>25s}: {cagr_spy:>+.1f}%")
print(f"  {'Sharpe':>25s}: {sharpe:.2f}")
print(f"  {'Max Drawdown':>25s}: ${dd_abs:>,.0f} ({dd_pct:.1f}%)")
print(f"  {'Años positivos':>25s}: {yrs_pos}/{n_years}")
print(f"  {'Años negativos':>25s}: {yrs_neg}/{n_years}")
print(f"  {'Años con alpha':>25s}: {yrs_alpha}/{n_years}")
print(f"  {'Alpha medio anual':>25s}: {np.mean(alphas):>+.1f}%")
print(f"  {'Retorno medio anual':>25s}: {np.mean(annual_rets):>+.1f}%")
print(f"  {'Win rate semanal':>25s}: {wins/active*100:.1f}%")

print(f"\n  {'ACUMULADO':>25s}")
print("  " + "-" * 60)
print(f"  {'$500K en estrategia':>25s}: ${CAPITAL * cumul_strat:>,.0f}")
print(f"  {'$500K en SPY':>25s}: ${CAPITAL * cumul_spy:>,.0f}")

# Best/worst
best_i = np.argmax(annual_rets)
worst_i = np.argmin(annual_rets)
print(f"\n  {'Mejor año':>25s}: {years[best_i]} ({annual_rets[best_i]:>+.1f}%)")
print(f"  {'Peor año':>25s}: {years[worst_i]} ({annual_rets[worst_i]:>+.1f}%)")
print(f"  {'Mejor alpha':>25s}: {years[np.argmax(alphas)]} ({max(alphas):>+.1f}%)")
print(f"  {'Peor alpha':>25s}: {years[np.argmin(alphas)]} ({min(alphas):>+.1f}%)")

print(f"\n{'=' * 115}")
