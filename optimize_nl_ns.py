"""
Optimize N_longs x N_shorts independently (0-3 each side)
With and without ATR>=1.3% filter on shorts.
Capital: $500K total always invested.
"""
import pandas as pd
import numpy as np
import psycopg2
from backtest_sector_events_v2 import load_etf_returns, backtest, build_weekly_events, SECTOR_ETFS

DB = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'
CAPITAL = 500_000

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
etf_prices, etf_returns = load_etf_returns()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# ATR for short filter
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


def apply_atr_filter(res, etf_atr, atr_min=1.3):
    """Apply lagged ATR filter: zero out shorts when ATR < threshold."""
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
    res['atr_lag'] = res['atr_raw'].shift(1)  # lagged 1 week
    res = res.reset_index()

    for idx, row in res.iterrows():
        keep = row['atr_lag'] >= atr_min if pd.notna(row.get('atr_lag')) else False
        if not keep:
            res.at[idx, 'short_pnl'] = 0
            res.at[idx, 'n_shorts'] = 0
    res['total_pnl'] = res['long_pnl'] + res['short_pnl']
    return res


def compute_metrics(res, label=""):
    """Compute all key metrics for a backtest result."""
    r = res[res['year'] >= 2000].copy()
    aw = r['total_pnl'].values
    if len(aw) == 0:
        return None

    tot = aw.sum()
    sharpe = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    cum = np.cumsum(aw)
    dd = (cum - np.maximum.accumulate(cum)).min()
    dd_pct = dd / CAPITAL * 100

    yr_pnl = r.groupby('year')['total_pnl'].sum()
    yp = (yr_pnl > 0).sum()
    yt = len(yr_pnl)

    long_pnl = r['long_pnl'].sum()
    short_pnl = r['short_pnl'].sum()
    swks = (r['n_shorts'] > 0).sum()
    lwks = (r['n_longs'] > 0).sum()

    # CAGR
    cumul = 1.0
    for year in sorted(yr_pnl.index):
        cumul *= (1 + yr_pnl[year] / CAPITAL)
    n_years = len(yr_pnl)
    cagr = (cumul ** (1/n_years) - 1) * 100 if n_years > 0 else 0

    # Win rate
    active_weeks = aw[aw != 0]
    wins = (active_weeks > 0).sum()
    win_rate = wins / len(active_weeks) * 100 if len(active_weeks) > 0 else 0

    return {
        'label': label,
        'total': tot,
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'sharpe': sharpe,
        'maxdd': dd,
        'maxdd_pct': dd_pct,
        'yrs_pos': yp,
        'yrs_tot': yt,
        'cagr': cagr,
        'cumul': cumul,
        'win_rate': win_rate,
        'long_weeks': lwks,
        'short_weeks': swks,
    }


# ═══════════════════════════════════════════════════════════════
# TEST ALL NL x NS COMBINATIONS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 140}")
print(f"  OPTIMIZATION: N_longs x N_shorts (Capital: ${CAPITAL:,.0f})")
print(f"  Testing all combinations 0-3 longs x 0-3 shorts, with and without ATR>=1.3% filter")
print(f"{'=' * 140}")

# Header
print(f"\n  {'Config':>22s} {'Filter':>10s} {'Total':>10} {'Long':>10} {'Short':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>10} {'DD%':>7} {'Yr+':>7} {'Win%':>6} {'LWks':>6} {'SWks':>6}")
print("  " + "-" * 135)

all_metrics = []

for nl in range(4):
    for ns in range(4):
        if nl == 0 and ns == 0:
            continue

        # Run base backtest with max(nl,ns) to get proper ranking
        # The backtest function uses n_long and n_short to select top/bottom
        res_raw = backtest(weekly_events, etf_returns,
                           n_long=max(nl, 1), n_short=max(ns, 1),
                           capital_per_side=CAPITAL,
                           momentum_decay=0.0, min_score=1.0)

        if nl == 0:
            # No longs: zero out long side
            res_raw['long_pnl'] = 0
            res_raw['n_longs'] = 0
            res_raw['longs'] = ''
        elif nl < 3:
            # Need to re-run with correct n_long
            res_raw = backtest(weekly_events, etf_returns,
                               n_long=nl, n_short=max(ns, 1),
                               capital_per_side=CAPITAL,
                               momentum_decay=0.0, min_score=1.0)

        if ns == 0:
            # No shorts
            res_raw['short_pnl'] = 0
            res_raw['n_shorts'] = 0
            res_raw['shorts'] = ''
            res_raw['total_pnl'] = res_raw['long_pnl']

            label = f"{nl}L + 0S"
            m = compute_metrics(res_raw, label)
            if m:
                m['filter'] = 'N/A'
                all_metrics.append(m)
                print(f"  {label:>22s} {'N/A':>10s} ${m['total']/1000:>+8.0f}K ${m['long_pnl']/1000:>+8.0f}K ${m['short_pnl']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} ${m['maxdd']/1000:>+8.0f}K {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d} {m['win_rate']:>5.1f}% {m['long_weeks']:>5d} {m['short_weeks']:>5d}")
            continue

        # With shorts: test both Always Short and ATR>=1.3%
        for filter_name, apply_filter in [("Always", False), ("ATR>=1.3%", True)]:
            res = res_raw.copy()

            if ns == 0:
                continue

            if nl == 0:
                res['long_pnl'] = 0
                res['n_longs'] = 0
                res['longs'] = ''

            res['total_pnl'] = res['long_pnl'] + res['short_pnl']

            if apply_filter:
                res = apply_atr_filter(res, etf_atr, 1.3)

            label = f"{nl}L + {ns}S"
            m = compute_metrics(res, label)
            if m:
                m['filter'] = filter_name
                all_metrics.append(m)
                print(f"  {label:>22s} {filter_name:>10s} ${m['total']/1000:>+8.0f}K ${m['long_pnl']/1000:>+8.0f}K ${m['short_pnl']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} ${m['maxdd']/1000:>+8.0f}K {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d} {m['win_rate']:>5.1f}% {m['long_weeks']:>5d} {m['short_weeks']:>5d}")

    if nl < 3:
        print()  # separator between long groups


# ═══════════════════════════════════════════════════════════════
# RANKING BY SHARPE
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 100}")
print(f"  TOP 10 BY SHARPE RATIO")
print(f"{'=' * 100}")

ranked = sorted(all_metrics, key=lambda x: x['sharpe'], reverse=True)
print(f"\n  {'#':>3} {'Config':>22s} {'Filter':>10s} {'Total':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Yr+':>7}")
print("  " + "-" * 80)
for i, m in enumerate(ranked[:10]):
    print(f"  {i+1:>3d} {m['label']:>22s} {m['filter']:>10s} ${m['total']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d}")


# ═══════════════════════════════════════════════════════════════
# RANKING BY TOTAL PnL
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print(f"  TOP 10 BY TOTAL PnL")
print(f"{'=' * 100}")

ranked_pnl = sorted(all_metrics, key=lambda x: x['total'], reverse=True)
print(f"\n  {'#':>3} {'Config':>22s} {'Filter':>10s} {'Total':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Yr+':>7}")
print("  " + "-" * 80)
for i, m in enumerate(ranked_pnl[:10]):
    print(f"  {i+1:>3d} {m['label']:>22s} {m['filter']:>10s} ${m['total']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d}")


# ═══════════════════════════════════════════════════════════════
# RANKING BY CAGR / MaxDD (return per risk)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print(f"  TOP 10 BY CAGR / MaxDD% (Return per unit of risk)")
print(f"{'=' * 100}")

for m in all_metrics:
    m['cagr_dd'] = m['cagr'] / abs(m['maxdd_pct']) if m['maxdd_pct'] != 0 else 0

ranked_rr = sorted(all_metrics, key=lambda x: x['cagr_dd'], reverse=True)
print(f"\n  {'#':>3} {'Config':>22s} {'Filter':>10s} {'CAGR':>7} {'MaxDD%':>7} {'CAGR/DD':>8} {'Sharpe':>7} {'Total':>10}")
print("  " + "-" * 85)
for i, m in enumerate(ranked_rr[:10]):
    print(f"  {i+1:>3d} {m['label']:>22s} {m['filter']:>10s} {m['cagr']:>+6.1f}% {m['maxdd_pct']:>+6.1f}% {m['cagr_dd']:>7.2f} {m['sharpe']:>+6.2f} ${m['total']/1000:>+8.0f}K")


# ═══════════════════════════════════════════════════════════════
# YEAR-BY-YEAR: TOP 3 configs
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 140}")
print(f"  YEAR-BY-YEAR: Top 3 by Sharpe vs SPY")
print(f"{'=' * 140}")

# Get the raw results for top 3
top3_labels = [(ranked[i]['label'], ranked[i]['filter']) for i in range(3)]

# Re-run top 3 to get year-by-year
top3_res = {}
for label, filt in top3_labels:
    parts = label.split('L + ')
    nl = int(parts[0])
    ns = int(parts[1].replace('S', ''))

    res = backtest(weekly_events, etf_returns,
                   n_long=max(nl, 1), n_short=max(ns, 1),
                   capital_per_side=CAPITAL,
                   momentum_decay=0.0, min_score=1.0)

    if nl == 0:
        res['long_pnl'] = 0
        res['n_longs'] = 0
        res['longs'] = ''
    if ns == 0:
        res['short_pnl'] = 0
        res['n_shorts'] = 0
        res['shorts'] = ''

    res['total_pnl'] = res['long_pnl'] + res['short_pnl']

    if filt == "ATR>=1.3%":
        res = apply_atr_filter(res, etf_atr, 1.3)

    key = f"{label} ({filt})"
    top3_res[key] = res

col_w = 27
print(f"\n  {'Year':>6} {'SPY':>7}", end="")
for key in top3_res:
    short_key = key[:25]
    print(f"  {short_key:>{col_w}s}", end="")
print()
print("  " + "-" * (14 + (col_w + 2) * len(top3_res)))

years = sorted(set().union(*[set(r['year'].unique()) for r in top3_res.values()]))
years = [y for y in years if y >= 2000]

cumuls = {k: 1.0 for k in top3_res}
cumul_spy = 1.0

for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0
    cumul_spy *= (1 + spy_ann)

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")
    for key, res in top3_res.items():
        yr = res[res['year'] == year]
        pnl = yr['total_pnl'].sum()
        ret = pnl / CAPITAL * 100
        cumuls[key] *= (1 + pnl / CAPITAL)
        swks = (yr['n_shorts'] > 0).sum()
        print(f"  {ret:>+7.1f}% ({swks:>2d}s) ${pnl/1000:>+6.0f}K", end="")
    print()

print("  " + "-" * (14 + (col_w + 2) * len(top3_res)))
print(f"  {'TOTAL':>6} {'':>7}", end="")
for key in top3_res:
    res = top3_res[key]
    pnl = res[res['year'] >= 2000]['total_pnl'].sum()
    ret = pnl / CAPITAL * 100
    print(f"  {ret:>+7.1f}%       ${pnl/1000:>+6.0f}K", end="")
print()

print(f"  {'CAGR':>6} {'':>7}", end="")
for key in top3_res:
    cagr = (cumuls[key] ** (1/len(years)) - 1) * 100
    print(f"  {cagr:>+7.1f}%{' ':>20s}", end="")
print()

print(f"  {'$500K→':>6} {'':>7}", end="")
for key in top3_res:
    final = CAPITAL * cumuls[key]
    print(f"  ${final/1e6:>6.2f}M{' ':>19s}", end="")
print()

spy_final = CAPITAL * cumul_spy
print(f"  {'SPY→':>6} ${spy_final/1e6:.2f}M")

print(f"\n{'=' * 140}")
