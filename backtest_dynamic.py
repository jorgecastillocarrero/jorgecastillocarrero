"""
Dynamic NL x NS backtest: each week independently decides how many longs/shorts
based on score thresholds and per-ETF ATR filter.

Each week:
  - Longs: all sectors with score > long_threshold (up to max_long)
  - Shorts: all sectors with score < -short_threshold AND individual ETF ATR >= atr_min (up to max_short)

Capital $500K split equally among all active positions (longs + shorts).
"""
import pandas as pd
import numpy as np
import psycopg2
from collections import Counter
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

DB = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'
SECTOR_ETFS = ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
CAPITAL = 500_000


def load_data():
    conn = psycopg2.connect(DB)
    symbols = ['SPY'] + SECTOR_ETFS
    ph = ','.join(['%s'] * len(symbols))
    ohlc = pd.read_sql(
        f"SELECT symbol, date, high, low, close FROM fmp_price_history "
        f"WHERE symbol IN ({ph}) ORDER BY date",
        conn, params=symbols, parse_dates=['date']
    )
    conn.close()

    etf_prices = ohlc[ohlc['symbol'].isin(SECTOR_ETFS)].pivot_table(
        index='date', columns='symbol', values='close')
    etf_prices_w = etf_prices.resample('W-FRI').last()
    etf_returns_w = etf_prices_w.pct_change()

    spy_df = ohlc[ohlc['symbol'] == 'SPY'].copy().sort_values('date').set_index('date')
    spy_w = spy_df['close'].resample('W-FRI').last()
    spy_ret_w = spy_w.pct_change()

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

    return etf_prices_w, etf_returns_w, spy_w, spy_ret_w, etf_atr


def score_week(active_events):
    scores = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        impacts = EVENT_SUBSECTOR_MAP[evt_type]['impacto']
        for subsec, impact in impacts.items():
            scores[subsec] = scores.get(subsec, 0) + intensity * impact
    return scores


def aggregate_to_sectors(subsector_scores):
    sector_scores = {}
    for subsec_id, subsec_data in SUBSECTORS.items():
        etf = subsec_data['etf']
        score = subsector_scores.get(subsec_id, 0)
        if score != 0:
            sector_scores.setdefault(etf, []).append(score)
    result = {}
    for etf in SECTOR_ETFS:
        subscores = sector_scores.get(etf, [])
        result[etf] = sum(subscores)
    return result


def backtest_dynamic(weekly_events, etf_returns, etf_atr,
                     max_long=3, max_short=3,
                     long_threshold=1.0,
                     short_threshold=1.0,
                     atr_min=1.3,
                     capital_mode='split'):
    """
    Dynamic backtest: each week independently determines NL and NS.

    capital_mode:
      'split' = $500K split among ALL active positions (L+S)
      'per_side' = $500K for longs, $500K for shorts (independent)
      'long_base' = $500K always on longs, shorts get their own $500K
    """
    common = weekly_events.index.intersection(etf_returns.index).sort_values()
    results = []

    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]
        prev_date = common[i - 1] if i > 0 else None

        # 1. Get active events
        events_row = weekly_events.loc[date]
        active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

        # 2. Score sub-sectors -> sectors
        subsec_scores = score_week(active)
        sector_scores = aggregate_to_sectors(subsec_scores)

        # 3. Rank sectors
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. DYNAMIC LONG selection: all with score > threshold, up to max
        current_longs = [etf for etf, sc in ranked[:max_long] if sc > long_threshold]

        # 5. DYNAMIC SHORT selection: score < -threshold + individual ATR check
        short_candidates = [etf for etf, sc in ranked[-max_short:] if sc < -short_threshold]

        current_shorts = []
        if prev_date and short_candidates:
            for etf in short_candidates:
                if etf in etf_atr and prev_date in etf_atr[etf].index:
                    if etf_atr[etf].loc[prev_date] >= atr_min:
                        current_shorts.append(etf)

        # 6. Calculate PnL
        n_l = len(current_longs)
        n_s = len(current_shorts)
        long_pnl = 0
        short_pnl = 0

        if capital_mode == 'per_side':
            # $500K each side independently
            if n_l > 0:
                cap_l = CAPITAL / n_l
                for etf in current_longs:
                    r = etf_returns.loc[next_date].get(etf, 0)
                    if pd.isna(r): r = 0
                    long_pnl += cap_l * r
            if n_s > 0:
                cap_s = CAPITAL / n_s
                for etf in current_shorts:
                    r = etf_returns.loc[next_date].get(etf, 0)
                    if pd.isna(r): r = 0
                    short_pnl += cap_s * (-r)
        else:
            # 'split': $500K split among all positions
            n_total = n_l + n_s
            if n_total > 0:
                cap = CAPITAL / n_total
                for etf in current_longs:
                    r = etf_returns.loc[next_date].get(etf, 0)
                    if pd.isna(r): r = 0
                    long_pnl += cap * r
                for etf in current_shorts:
                    r = etf_returns.loc[next_date].get(etf, 0)
                    if pd.isna(r): r = 0
                    short_pnl += cap * (-r)

        results.append({
            'date': next_date,
            'year': next_date.year,
            'week': next_date.isocalendar()[1],
            'n_events': len(active),
            'n_longs': n_l,
            'n_shorts': n_s,
            'config': f"{n_l}L+{n_s}S",
            'longs': ','.join(current_longs),
            'shorts': ','.join(current_shorts),
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'total_pnl': long_pnl + short_pnl,
        })

    return pd.DataFrame(results)


# =====================================================================
# MAIN
# =====================================================================
print("Loading data...")
etf_prices_w, etf_returns_w, spy_w, spy_ret_w, etf_atr = load_data()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# Test configurations
configs = [
    # label, max_long, max_short, long_thr, short_thr, atr_min, capital_mode
    ("3L+3S fijo (baseline)",            3, 3, 1.0, 1.0, 0.0,  'per_side'),
    ("3L+3S fijo ATR>=1.3",             3, 3, 1.0, 1.0, 1.3,  'per_side'),
    ("Dinamico per_side ATR>=1.3",       3, 3, 1.0, 1.0, 1.3,  'per_side'),
    ("Dinamico split ATR>=1.3",          3, 3, 1.0, 1.0, 1.3,  'split'),
    ("Dinamico split ATR>=1.0",          3, 3, 1.0, 1.0, 1.0,  'split'),
    ("Dinamico split ATR>=1.5",          3, 3, 1.0, 1.0, 1.5,  'split'),
    ("Dinamico split ATR>=1.8",          3, 3, 1.0, 1.0, 1.8,  'split'),
    ("Dinamico split no ATR",            3, 3, 1.0, 1.0, 0.0,  'split'),
    ("Dinamico L_thr=2.0 split",         3, 3, 2.0, 1.0, 1.3,  'split'),
    ("Dinamico S_thr=2.0 split",         3, 3, 1.0, 2.0, 1.3,  'split'),
    ("Dinamico L2 S2 split",             3, 3, 2.0, 2.0, 1.3,  'split'),
    ("Dinamico max4L+4S split",          4, 4, 1.0, 1.0, 1.3,  'split'),
    ("Dinamico max5L+5S split",          5, 5, 1.0, 1.0, 1.3,  'split'),
]

print(f"\n{'=' * 150}")
print(f"  DYNAMIC NL x NS OPTIMIZATION (Capital: ${CAPITAL:,.0f})")
print(f"  Each week independently selects how many longs/shorts based on scores + ATR")
print(f"{'=' * 150}")

print(f"\n  {'Config':>35s} {'Total':>10} {'Long':>10} {'Short':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>10} {'DD%':>7} {'Yr+':>7} {'Win%':>6} {'AvgNL':>6} {'AvgNS':>6}")
print("  " + "-" * 140)

all_results = {}
all_metrics = []

for label, ml, ms, lt, st, atr, cmode in configs:
    if 'fijo' in label:
        # Fixed config: use the old backtest for comparison
        from backtest_sector_events_v2 import backtest as backtest_fixed, load_etf_returns
        res_fix = backtest_fixed(weekly_events, etf_returns_w,
                                  n_long=ml, n_short=ms,
                                  capital_per_side=CAPITAL,
                                  momentum_decay=0.0, min_score=lt)
        if 'ATR' in label:
            # Apply blanket ATR filter
            res_fix = res_fix.set_index('date').sort_index()
            atr_raw = []
            for date, row in res_fix.iterrows():
                if row['shorts'] and isinstance(row['shorts'], str):
                    etfs = row['shorts'].split(',')
                    atrs = [etf_atr[e].loc[date] for e in etfs if e in etf_atr and date in etf_atr[e].index]
                    atr_raw.append(np.mean(atrs) if atrs else np.nan)
                else:
                    atr_raw.append(np.nan)
            res_fix['atr_raw'] = atr_raw
            res_fix['atr_lag'] = res_fix['atr_raw'].shift(1)
            res_fix = res_fix.reset_index()
            for idx, row in res_fix.iterrows():
                keep = row['atr_lag'] >= 1.3 if pd.notna(row.get('atr_lag')) else False
                if not keep:
                    res_fix.at[idx, 'short_pnl'] = 0
                    res_fix.at[idx, 'n_shorts'] = 0
            res_fix['total_pnl'] = res_fix['long_pnl'] + res_fix['short_pnl']
        res = res_fix
    else:
        res = backtest_dynamic(weekly_events, etf_returns_w, etf_atr,
                               max_long=ml, max_short=ms,
                               long_threshold=lt, short_threshold=st,
                               atr_min=atr, capital_mode=cmode)

    all_results[label] = res

    # Metrics (2000+)
    r = res[res['year'] >= 2000]
    aw = r['total_pnl'].values
    tot = aw.sum()
    ltot = r['long_pnl'].sum()
    stot = r['short_pnl'].sum()
    sh = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    cum = np.cumsum(aw)
    dd = (cum - np.maximum.accumulate(cum)).min()
    dd_pct = dd / CAPITAL * 100
    yr_pnl = r.groupby('year')['total_pnl'].sum()
    yp = (yr_pnl > 0).sum()
    yt = len(yr_pnl)
    active_w = aw[aw != 0]
    wins = (active_w > 0).sum()
    wr = wins / len(active_w) * 100 if len(active_w) > 0 else 0
    avg_nl = r['n_longs'].mean()
    avg_ns = r['n_shorts'].mean()

    cumul = 1.0
    for year in sorted(yr_pnl.index):
        cumul *= (1 + yr_pnl[year] / CAPITAL)
    cagr = (cumul ** (1/len(yr_pnl)) - 1) * 100

    all_metrics.append({
        'label': label, 'total': tot, 'long_pnl': ltot, 'short_pnl': stot,
        'cagr': cagr, 'sharpe': sh, 'maxdd': dd, 'maxdd_pct': dd_pct,
        'yrs_pos': yp, 'yrs_tot': yt, 'win_rate': wr,
        'avg_nl': avg_nl, 'avg_ns': avg_ns, 'cumul': cumul,
    })

    print(f"  {label:>35s} ${tot/1000:>+8.0f}K ${ltot/1000:>+8.0f}K ${stot/1000:>+8.0f}K {cagr:>+6.1f}% {sh:>+6.2f} ${dd/1000:>+8.0f}K {dd_pct:>+6.1f}% {yp:>3d}/{yt:<2d} {wr:>5.1f}% {avg_nl:>5.1f} {avg_ns:>5.1f}")


# ═══════════════════════════════════════════════════════════════
# WEEKLY CONFIG DISTRIBUTION for best dynamic
# ═══════════════════════════════════════════════════════════════
best_label = "Dinamico split ATR>=1.3"
best = all_results[best_label]
br = best[best['year'] >= 2000]

print(f"\n\n{'=' * 100}")
print(f"  WEEKLY CONFIG DISTRIBUTION: {best_label}")
print(f"{'=' * 100}")

config_counts = br['config'].value_counts().sort_index()
print(f"\n  {'Config':>10s} {'Weeks':>6} {'%':>6} {'Avg PnL/wk':>12} {'Total PnL':>12} {'Win%':>6}")
print("  " + "-" * 60)

for cfg in sorted(config_counts.index):
    mask = br['config'] == cfg
    wks = mask.sum()
    pct = wks / len(br) * 100
    avg = br.loc[mask, 'total_pnl'].mean()
    tot = br.loc[mask, 'total_pnl'].sum()
    wins = (br.loc[mask, 'total_pnl'] > 0).sum()
    wr = wins / wks * 100
    print(f"  {cfg:>10s} {wks:>6d} {pct:>5.1f}% ${avg:>+11,.0f} ${tot/1000:>+10,.0f}K {wr:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# YEAR-BY-YEAR: dynamic vs fixed baselines
# ═══════════════════════════════════════════════════════════════
show_labels = [
    "3L+3S fijo (baseline)",
    "3L+3S fijo ATR>=1.3",
    "Dinamico split ATR>=1.3",
    "Dinamico split ATR>=1.0",
]
show_labels = [l for l in show_labels if l in all_results]

print(f"\n\n{'=' * 150}")
print(f"  YEAR-BY-YEAR COMPARISON")
print(f"{'=' * 150}")

col_w = 30
print(f"\n  {'Year':>6} {'SPY':>7}", end="")
for label in show_labels:
    short_l = label[:28]
    print(f"  {short_l:>{col_w}s}", end="")
print()
print("  " + "-" * (14 + (col_w + 2) * len(show_labels)))

years = sorted(set().union(*[set(r['year'].unique()) for r in all_results.values()]))
years = [y for y in years if y >= 2000]

cumuls = {l: 1.0 for l in show_labels}

for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")
    for label in show_labels:
        res = all_results[label]
        yr = res[res['year'] == year]
        pnl = yr['total_pnl'].sum()
        ret = pnl / CAPITAL * 100
        cumuls[label] *= (1 + pnl / CAPITAL)
        swks = (yr['n_shorts'] > 0).sum() if 'n_shorts' in yr.columns else 0
        avg_nl = yr['n_longs'].mean() if len(yr) > 0 else 0
        avg_ns = yr['n_shorts'].mean() if len(yr) > 0 else 0
        print(f"  {ret:>+6.1f}% {avg_nl:.1f}L {avg_ns:.1f}S ${pnl/1000:>+6.0f}K", end="")
    print()

print("  " + "-" * (14 + (col_w + 2) * len(show_labels)))
print(f"  {'TOTAL':>6} {'':>7}", end="")
for label in show_labels:
    res = all_results[label]
    pnl = res[res['year'] >= 2000]['total_pnl'].sum()
    print(f"  {pnl/CAPITAL*100:>+6.1f}%             ${pnl/1000:>+6.0f}K", end="")
print()

print(f"  {'CAGR':>6} {'':>7}", end="")
for label in show_labels:
    cagr = (cumuls[label] ** (1/len(years)) - 1) * 100
    print(f"  {cagr:>+6.1f}%{' ':>24s}", end="")
print()

print(f"  {'$500K':>6} {'':>7}", end="")
for label in show_labels:
    final = CAPITAL * cumuls[label]
    print(f"  ${final/1e6:>5.2f}M{' ':>23s}", end="")
print()

print(f"\n{'=' * 150}")
