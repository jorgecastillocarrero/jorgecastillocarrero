"""
Detail week-by-week for a specific year (3L+3S dec=0.0 thr=1.0)
"""
import sys
import pandas as pd
import numpy as np
from backtest_sector_events_v2 import load_etf_returns, backtest, build_weekly_events

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2020

print("Loading data...")
etf_prices, etf_returns = load_etf_returns()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

res = backtest(weekly_events, etf_returns,
               n_long=3, n_short=3,
               momentum_decay=0.0, min_score=1.0)

yr = res[res['year'] == year].copy()

print(f"\n{'=' * 120}")
print(f"  DETALLE SEMANAL {year} - 3L+3S (dec=0.0, thr=1.0, $500K/side)")
print(f"{'=' * 120}")

print(f"\n  {'Week':>4} {'Date':>12} {'Events':>3} {'Longs':<20s} {'Shorts':<20s} {'Long PnL':>12} {'Short PnL':>12} {'Total':>12} {'Cumul':>12}")
print("  " + "-" * 115)

cumul = 0
for _, row in yr.iterrows():
    cumul += row['total_pnl']
    date_str = row['date'].strftime('%Y-%m-%d')
    longs = row['longs'] if row['longs'] else '---'
    shorts = row['shorts'] if row['shorts'] else '---'
    n_ev = row['n_events']

    s = '+' if row['total_pnl'] > 0 else '-' if row['total_pnl'] < 0 else ' '
    sc = '+' if cumul > 0 else '-' if cumul < 0 else ' '

    print(f"  {_+1:>4} {date_str:>12} {n_ev:>3} {longs:<20s} {shorts:<20s} "
          f"${row['long_pnl']:>+11,.0f} ${row['short_pnl']:>+11,.0f} "
          f"{s}${abs(row['total_pnl']):>10,.0f} {sc}${abs(cumul):>10,.0f}")

print("  " + "-" * 115)

# Summary
pnl = yr['total_pnl'].sum()
lpnl = yr['long_pnl'].sum()
spnl = yr['short_pnl'].sum()
act = len(yr[(yr['n_longs'] > 0) | (yr['n_shorts'] > 0)])
wins = len(yr[yr['total_pnl'] > 0])
wpct = wins / max(1, act) * 100
aw = yr['total_pnl'].values
cum = np.cumsum(aw)
dd = (cum - np.maximum.accumulate(cum)).min()

print(f"\n  Total PnL: ${pnl:>+,.0f}  (Long: ${lpnl:>+,.0f}  Short: ${spnl:>+,.0f})")
print(f"  Weeks: {len(yr)} total, {act} active, {wins} winning ({wpct:.1f}%)")
print(f"  Max Drawdown: ${dd:,.0f}")

# Events active
print(f"\n  Events active in {year}:")
from collections import Counter
ecounts = Counter()
for estr in yr['events']:
    if estr:
        for e in estr.split('|'):
            ecounts[e] += 1
for ev, cnt in ecounts.most_common():
    print(f"    {ev:40s}: {cnt:3d} weeks")
