"""
Year-by-year RETURNS for Always Short (3L+3S, dec=0.0, thr=1.0)
vs SPY Buy & Hold
"""
import pandas as pd
import numpy as np
import psycopg2
from backtest_sector_events_v2 import load_etf_returns, backtest, build_weekly_events

print("Loading data...")
etf_prices, etf_returns = load_etf_returns()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# SPY benchmark
conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
spy = pd.read_sql(
    "SELECT date, close FROM fmp_price_history WHERE symbol='SPY' ORDER BY date",
    conn, parse_dates=['date']
)
conn.close()
spy = spy.set_index('date').sort_index()
spy_w = spy['close'].resample('W-FRI').last()
spy_ret_w = spy_w.pct_change()

# Backtest
res = backtest(weekly_events, etf_returns, n_long=3, n_short=3,
               momentum_decay=0.0, min_score=1.0)

capital_per_side = 500_000
capital_total = capital_per_side * 2  # $1M deployed (500K long + 500K short)

years = sorted(res['year'].unique())

print(f"\n{'=' * 95}")
print(f"  RETORNOS ANUALES - Always Short 3L+3S ($500K/side = $1M total)")
print(f"{'=' * 95}")

print(f"\n  {'Year':>6} {'SPY':>8} {'Strat':>8} {'Long':>8} {'Short':>8} {'Alpha':>8} {'PnL':>12} {'Cumul PnL':>12} {'Cumul %':>8}")
print("  " + "-" * 90)

cumul_pnl = 0
cumul_spy = 1.0
cumul_strat = 1.0
yrs_pos = 0
yrs_alpha = 0
alphas = []

for year in years:
    yr = res[res['year'] == year]
    pnl = yr['total_pnl'].sum()
    lpnl = yr['long_pnl'].sum()
    spnl = yr['short_pnl'].sum()
    cumul_pnl += pnl

    # Returns as % of total capital
    ret_total = pnl / capital_total * 100
    ret_long = lpnl / capital_per_side * 100
    ret_short = spnl / capital_per_side * 100

    # SPY annual return
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

    print(f"  {year:>6d} {spy_pct:>+7.1f}% {ret_total:>+7.1f}% {ret_long:>+7.1f}% {ret_short:>+7.1f}% {alpha:>+7.1f}% ${pnl:>+11,.0f} ${cumul_pnl:>+11,.0f} {cumul_ret:>+7.1f}%")

print("  " + "-" * 90)

# Totals
total_pnl = res['total_pnl'].sum()
total_ret = (cumul_strat - 1) * 100
spy_total_ret = (cumul_spy - 1) * 100
n_years = len([y for y in years if y >= 2000 and y <= 2025])

# CAGR
cagr_strat = (cumul_strat ** (1/n_years) - 1) * 100
cagr_spy = (cumul_spy ** (1/n_years) - 1) * 100

aw = res['total_pnl'].values
sharpe = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
cum = np.cumsum(aw)
dd = (cum - np.maximum.accumulate(cum)).min()

print(f"\n  {'MÉTRICAS':>20s}")
print("  " + "-" * 50)
print(f"  {'Total PnL':>20s}: ${total_pnl:>+,.0f}")
print(f"  {'Retorno total':>20s}: {total_ret:>+.1f}%")
print(f"  {'SPY total':>20s}: {spy_total_ret:>+.1f}%")
print(f"  {'CAGR estrategia':>20s}: {cagr_strat:>+.1f}%")
print(f"  {'CAGR SPY':>20s}: {cagr_spy:>+.1f}%")
print(f"  {'Sharpe':>20s}: {sharpe:.2f}")
print(f"  {'Max Drawdown':>20s}: ${dd:>,.0f} ({dd/capital_total*100:.1f}%)")
print(f"  {'Años positivos':>20s}: {yrs_pos}/{len(years)}")
print(f"  {'Años con alpha':>20s}: {yrs_alpha}/{len(years)}")
print(f"  {'Alpha medio anual':>20s}: {np.mean(alphas):>+.1f}%")
print(f"  {'Mejor alpha':>20s}: {max(alphas):>+.1f}% ({years[alphas.index(max(alphas))]})")
print(f"  {'Peor alpha':>20s}: {min(alphas):>+.1f}% ({years[alphas.index(min(alphas))]})")

# Cumulative comparison
print(f"\n  {'ACUMULADO':>20s}")
print("  " + "-" * 50)
print(f"  {'$1M en estrategia':>20s}: ${capital_total * cumul_strat:>,.0f}")
print(f"  {'$1M en SPY':>20s}: ${capital_total * cumul_spy:>,.0f}")

print(f"\n{'=' * 95}")
