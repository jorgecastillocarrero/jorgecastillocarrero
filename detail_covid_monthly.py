"""
Monthly returns during COVID period (Jan 2020 - Dec 2020) for 3L+3S
"""
import pandas as pd
import numpy as np
from backtest_sector_events_v2 import load_etf_returns, backtest, build_weekly_events

print("Loading data...")
etf_prices, etf_returns = load_etf_returns()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

res = backtest(weekly_events, etf_returns,
               n_long=3, n_short=3,
               momentum_decay=0.0, min_score=1.0)

# Filter 2020
yr = res[res['year'] == 2020].copy()
yr['month'] = yr['date'].dt.month
yr['month_name'] = yr['date'].dt.strftime('%b')

# Also get SPY for comparison
import psycopg2
conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
spy = pd.read_sql(
    "SELECT date, close FROM fmp_price_history WHERE symbol='SPY' ORDER BY date",
    conn, parse_dates=['date']
)
conn.close()
spy = spy.set_index('date')
spy_w = spy.resample('W-FRI').last()
spy_w['ret'] = spy_w['close'].pct_change()

print(f"\n{'=' * 100}")
print(f"  RETORNOS MENSUALES COVID 2020 - Strategy 3L+3S vs SPY")
print(f"{'=' * 100}")

print(f"\n  {'Mes':>6} {'Long PnL':>12} {'Short PnL':>12} {'Total PnL':>12} {'Cumul':>12} {'SPY ret':>8} {'Longs principales':>30s} {'Shorts principales':>30s}")
print("  " + "-" * 130)

cumul = 0
cumul_spy = 1.0

for month in range(1, 13):
    m = yr[yr['month'] == month]
    if m.empty:
        continue

    lpnl = m['long_pnl'].sum()
    spnl = m['short_pnl'].sum()
    pnl = m['total_pnl'].sum()
    cumul += pnl

    # SPY monthly return
    m_dates = m['date']
    spy_month = spy_w.loc[spy_w.index.isin(m_dates), 'ret']
    spy_ret = (1 + spy_month).prod() - 1
    cumul_spy *= (1 + spy_ret)

    # Most common longs/shorts this month
    from collections import Counter
    lc = Counter()
    sc = Counter()
    for _, row in m.iterrows():
        if row['longs']:
            for s in row['longs'].split(','):
                lc[s] += 1
        if row['shorts']:
            for s in row['shorts'].split(','):
                sc[s] += 1

    top_l = ','.join(f"{e}" for e, _ in lc.most_common(3))
    top_s = ','.join(f"{e}" for e, _ in sc.most_common(3))

    month_name = m.iloc[0]['date'].strftime('%b')
    s = '+' if pnl > 0 else '-' if pnl < 0 else ' '

    print(f"  {month_name:>6} ${lpnl:>+11,.0f} ${spnl:>+11,.0f} {s}${abs(pnl):>10,.0f} ${cumul:>+11,.0f} {spy_ret:>+7.1%} {top_l:>30s} {top_s:>30s}")

print("  " + "-" * 130)

total_pnl = yr['total_pnl'].sum()
total_lpnl = yr['long_pnl'].sum()
total_spnl = yr['short_pnl'].sum()
spy_total = cumul_spy - 1

print(f"\n  Strategy: ${total_pnl:>+,.0f}  ({total_pnl/500_000*100:>+.1f}% sobre $500K/side)")
print(f"    Long:   ${total_lpnl:>+,.0f}  ({total_lpnl/500_000*100:>+.1f}%)")
print(f"    Short:  ${total_spnl:>+,.0f}  ({total_spnl/500_000*100:>+.1f}%)")
print(f"  SPY B&H:  {spy_total:>+.1%}")
print(f"\n  Alpha vs SPY: {total_pnl/500_000*100 - spy_total*100:>+.1f} pp")

# Weekly detail for March (the crash month)
print(f"\n\n{'=' * 100}")
print(f"  DETALLE SEMANAL MARZO 2020 (crash COVID)")
print(f"{'=' * 100}")

march = yr[yr['month'] == 3]
print(f"\n  {'Date':>12} {'Events':>3} {'Longs':<18s} {'Shorts':<18s} {'Long PnL':>12} {'Short PnL':>12} {'Total':>12} {'SPY wk':>8}")
print("  " + "-" * 100)

cumul_march = 0
for _, row in march.iterrows():
    cumul_march += row['total_pnl']
    date_str = row['date'].strftime('%Y-%m-%d')
    longs = row['longs'] if row['longs'] else '---'
    shorts = row['shorts'] if row['shorts'] else '---'

    spy_ret_w = spy_w.loc[row['date'], 'ret'] if row['date'] in spy_w.index else 0

    # Active events
    events = row['events'].split('|') if row['events'] else []

    print(f"  {date_str:>12} {row['n_events']:>3} {longs:<18s} {shorts:<18s} "
          f"${row['long_pnl']:>+11,.0f} ${row['short_pnl']:>+11,.0f} "
          f"${row['total_pnl']:>+11,.0f} {spy_ret_w:>+7.1%}")

print("  " + "-" * 100)
print(f"  Marzo total: ${march['total_pnl'].sum():>+,.0f}")

# Events during March
print(f"\n  Eventos activos en Marzo 2020:")
ecounts = Counter()
for estr in march['events']:
    if estr:
        for e in estr.split('|'):
            ecounts[e] += 1
for ev, cnt in ecounts.most_common():
    print(f"    {ev:40s}: {cnt} semanas")
