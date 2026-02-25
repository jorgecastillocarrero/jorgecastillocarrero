"""
Year-by-year RETURNS for ATR>=1.3% filter (3L+3S, dec=0.0, thr=1.0)
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

# Apply ATR>=1.3% filter
for idx, row in res.iterrows():
    keep = row['atr_lag'] >= 1.3 if pd.notna(row.get('atr_lag')) else False
    if not keep:
        res.at[idx, 'short_pnl'] = 0
        res.at[idx, 'n_shorts'] = 0
res['total_pnl'] = res['long_pnl'] + res['short_pnl']

capital_per_side = 500_000
capital_total = capital_per_side * 2

years = sorted(res['year'].unique())

print(f"\n{'=' * 105}")
print(f"  RETORNOS ANUALES - ATR>=1.3% Filter 3L+3S ($500K/side = $1M total)")
print(f"{'=' * 105}")

print(f"\n  {'Year':>6} {'SPY':>8} {'Strat':>8} {'Long':>8} {'Short':>8} {'Alpha':>8} {'PnL':>12} {'Cumul PnL':>12} {'Cumul %':>8} {'ShWks':>6}")
print("  " + "-" * 100)

cumul_pnl = 0
cumul_strat = 1.0
cumul_spy = 1.0
yrs_pos = 0
yrs_alpha = 0
alphas = []

for year in years:
    yr = res[res['year'] == year]
    pnl = yr['total_pnl'].sum()
    lpnl = yr['long_pnl'].sum()
    spnl = yr['short_pnl'].sum()
    cumul_pnl += pnl
    swks = (yr['n_shorts'] > 0).sum()

    ret_total = pnl / capital_total * 100
    ret_long = lpnl / capital_per_side * 100
    ret_short = spnl / capital_per_side * 100

    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1
    spy_pct = spy_ann * 100

    alpha = ret_total - spy_pct
    alphas.append(alpha)
    if pnl > 0: yrs_pos += 1
    if alpha > 0: yrs_alpha += 1

    cumul_spy *= (1 + spy_ann)
    cumul_strat *= (1 + pnl / capital_total)
    cumul_ret = (cumul_strat - 1) * 100

    print(f"  {year:>6d} {spy_pct:>+7.1f}% {ret_total:>+7.1f}% {ret_long:>+7.1f}% {ret_short:>+7.1f}% {alpha:>+7.1f}% ${pnl:>+11,.0f} ${cumul_pnl:>+11,.0f} {cumul_ret:>+7.1f}% {swks:>5d}")

print("  " + "-" * 100)

total_pnl = res['total_pnl'].sum()
total_ret = (cumul_strat - 1) * 100
spy_total_ret = (cumul_spy - 1) * 100
n_years = len([y for y in years if y >= 2000 and y <= 2025])

cagr_strat = (cumul_strat ** (1/n_years) - 1) * 100
cagr_spy = (cumul_spy ** (1/n_years) - 1) * 100

aw = res['total_pnl'].values
sharpe = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
cum_arr = np.cumsum(aw)
dd = (cum_arr - np.maximum.accumulate(cum_arr)).min()
total_swks = (res['n_shorts'] > 0).sum()
total_wks = len(res)

print(f"\n  {'MÉTRICAS':>22s}")
print("  " + "-" * 55)
print(f"  {'Total PnL':>22s}: ${total_pnl:>+,.0f}")
print(f"  {'Retorno total':>22s}: {total_ret:>+.1f}%")
print(f"  {'SPY total':>22s}: {spy_total_ret:>+.1f}%")
print(f"  {'CAGR estrategia':>22s}: {cagr_strat:>+.1f}%")
print(f"  {'CAGR SPY':>22s}: {cagr_spy:>+.1f}%")
print(f"  {'Sharpe':>22s}: {sharpe:.2f}")
print(f"  {'Max Drawdown':>22s}: ${dd:>,.0f} ({dd/capital_total*100:.1f}%)")
print(f"  {'Años positivos':>22s}: {yrs_pos}/{len(years)}")
print(f"  {'Años con alpha':>22s}: {yrs_alpha}/{len(years)}")
print(f"  {'Alpha medio anual':>22s}: {np.mean(alphas):>+.1f}%")
print(f"  {'Semanas con short':>22s}: {total_swks}/{total_wks} ({total_swks/total_wks*100:.0f}%)")

print(f"\n  {'ACUMULADO':>22s}")
print("  " + "-" * 55)
print(f"  {'$1M en estrategia':>22s}: ${capital_total * cumul_strat:>,.0f}")
print(f"  {'$1M en SPY':>22s}: ${capital_total * cumul_spy:>,.0f}")

# Comparison vs Always Short
print(f"\n  {'VS ALWAYS SHORT':>22s}")
print("  " + "-" * 55)
# Re-run always short for comparison
res2 = backtest(weekly_events, etf_returns, n_long=3, n_short=3,
                momentum_decay=0.0, min_score=1.0)
as_pnl = res2['total_pnl'].sum()
as_aw = res2['total_pnl'].values
as_sh = as_aw.mean() / as_aw.std() * np.sqrt(52) if as_aw.std() > 0 else 0
as_cum = np.cumsum(as_aw)
as_dd = (as_cum - np.maximum.accumulate(as_cum)).min()

print(f"  {'':>22s} {'ATR>=1.3%':>14s} {'Always Short':>14s} {'Diff':>12s}")
print(f"  {'Total PnL':>22s} ${total_pnl:>+12,.0f} ${as_pnl:>+12,.0f} ${total_pnl-as_pnl:>+10,.0f}")
print(f"  {'Sharpe':>22s} {sharpe:>14.2f} {as_sh:>14.2f} {sharpe-as_sh:>+12.2f}")
print(f"  {'MaxDD':>22s} ${dd:>12,.0f} ${as_dd:>12,.0f} ${dd-as_dd:>+10,.0f}")
print(f"  {'Short PnL':>22s} ${res['short_pnl'].sum():>+12,.0f} ${res2['short_pnl'].sum():>+12,.0f} ${res['short_pnl'].sum()-res2['short_pnl'].sum():>+10,.0f}")

print(f"\n{'=' * 105}")
