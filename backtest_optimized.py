"""
Optimized Strategy: Persistent Longs + Tactical Shorts
- Longs: hold minimum 4 weeks before rotating (persistence)
- Shorts: max 3 weeks per position, only when ETF ATR >= 1.3%
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

    # ETF weekly prices/returns
    etf_prices = ohlc[ohlc['symbol'].isin(SECTOR_ETFS)].pivot_table(
        index='date', columns='symbol', values='close')
    etf_prices_w = etf_prices.resample('W-FRI').last()
    etf_returns_w = etf_prices_w.pct_change()

    # SPY
    spy_df = ohlc[ohlc['symbol'] == 'SPY'].copy().sort_values('date').set_index('date')
    spy_w = spy_df['close'].resample('W-FRI').last()
    spy_ret_w = spy_w.pct_change()

    # ATR per ETF
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


def backtest_optimized(weekly_events, etf_returns, etf_atr,
                       n_long=3, n_short=3,
                       min_score=1.0,
                       long_min_hold=4,    # min weeks to hold longs
                       short_max_hold=3,   # max weeks to hold shorts
                       short_atr_min=1.3): # min ATR to enable shorts
    """
    Optimized backtest:
    - Longs: hold for at least long_min_hold weeks before allowing rotation
    - Shorts: only when shorted ETFs ATR >= threshold, max short_max_hold weeks
    """
    common = weekly_events.index.intersection(etf_returns.index).sort_values()
    results = []

    current_longs = []       # list of ETF symbols
    long_hold_count = 0      # weeks held
    current_shorts = []
    short_hold_count = 0

    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]

        # 1. Get active events
        events_row = weekly_events.loc[date]
        active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

        # 2. Score sub-sectors -> sectors
        subsec_scores = score_week(active)
        sector_scores = aggregate_to_sectors(subsec_scores)

        # 3. Rank sectors
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_longs = [etf for etf, sc in ranked[:n_long] if sc > min_score]
        candidate_shorts = [etf for etf, sc in ranked[-n_short:] if sc < -min_score]

        # ── LONG LOGIC: persistent ──
        if long_hold_count >= long_min_hold or not current_longs:
            # Allowed to rotate
            if candidate_longs:
                if set(candidate_longs) != set(current_longs):
                    current_longs = candidate_longs
                    long_hold_count = 1
                else:
                    long_hold_count += 1
            else:
                current_longs = []
                long_hold_count = 0
        else:
            # Must hold current position
            long_hold_count += 1

        # ── SHORT LOGIC: tactical ──
        # Check ATR filter (lagged: use previous week's ATR)
        prev_date = common[i - 1] if i > 0 else None
        short_atr_ok = False
        if prev_date and candidate_shorts:
            atrs = []
            for etf in candidate_shorts:
                if etf in etf_atr and prev_date in etf_atr[etf].index:
                    atrs.append(etf_atr[etf].loc[prev_date])
            if atrs:
                short_atr_ok = np.mean(atrs) >= short_atr_min

        if short_hold_count >= short_max_hold:
            # Force close shorts after max hold
            current_shorts = []
            short_hold_count = 0

        if short_atr_ok and candidate_shorts:
            if not current_shorts:
                # Open new short
                current_shorts = candidate_shorts
                short_hold_count = 1
            elif set(candidate_shorts) != set(current_shorts):
                # Rotate short
                current_shorts = candidate_shorts
                short_hold_count = 1
            else:
                short_hold_count += 1
        elif not short_atr_ok:
            current_shorts = []
            short_hold_count = 0

        # 4. Calculate PnL
        long_pnl = 0
        short_pnl = 0
        n_l = len(current_longs)
        n_s = len(current_shorts)

        if n_l > 0:
            cap = CAPITAL / n_l
            for etf in current_longs:
                r = etf_returns.loc[next_date].get(etf, 0)
                if pd.isna(r): r = 0
                long_pnl += cap * r

        if n_s > 0:
            cap = CAPITAL / n_s
            for etf in current_shorts:
                r = etf_returns.loc[next_date].get(etf, 0)
                if pd.isna(r): r = 0
                short_pnl += cap * (-r)

        results.append({
            'date': next_date,
            'year': next_date.year,
            'n_events': len(active),
            'n_longs': n_l, 'n_shorts': n_s,
            'longs': ','.join(current_longs),
            'shorts': ','.join(current_shorts),
            'long_pnl': long_pnl, 'short_pnl': short_pnl,
            'total_pnl': long_pnl + short_pnl,
            'long_streak': long_hold_count,
            'short_streak': short_hold_count,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
etf_prices_w, etf_returns_w, spy_w, spy_ret_w, etf_atr = load_data()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# Test configurations
configs = [
    # label, long_min_hold, short_max_hold, short_atr_min
    ("Baseline (no persistence)",       1, 999, 0.0),
    ("ATR>=1.3 only (no persist)",      1, 999, 1.3),
    ("L:hold4 + S:tact3 + ATR1.3",     4, 3, 1.3),
    ("L:hold4 + S:tact2 + ATR1.3",     4, 2, 1.3),
    ("L:hold6 + S:tact3 + ATR1.3",     6, 3, 1.3),
    ("L:hold8 + S:tact3 + ATR1.3",     8, 3, 1.3),
    ("L:hold4 + S:tact4 + ATR1.3",     4, 4, 1.3),
    ("L:hold4 + S:tact3 + ATR1.6",     4, 3, 1.6),
    ("L:hold6 + S:tact2 + ATR1.3",     6, 2, 1.3),
    ("L:hold4 + no short",             4, 0, 9.9),
]

print(f"\n{'=' * 120}")
print(f"  OPTIMIZATION: Persistent Longs + Tactical Shorts (Capital: ${CAPITAL:,.0f})")
print(f"{'=' * 120}")

print(f"\n  {'Config':>35s} {'Total':>10} {'Long':>10} {'Short':>10} {'Sharpe':>7} {'MaxDD':>10} {'MaxDD%':>7} {'Yr+':>5} {'LStk':>5} {'SWks':>5}")
print("  " + "-" * 115)

all_results = {}
for label, lhold, shold, satr in configs:
    if shold == 0:
        # No shorts at all
        res = backtest_optimized(weekly_events, etf_returns_w, etf_atr,
                                 long_min_hold=lhold, short_max_hold=0, short_atr_min=99)
    else:
        res = backtest_optimized(weekly_events, etf_returns_w, etf_atr,
                                 long_min_hold=lhold, short_max_hold=shold, short_atr_min=satr)
    all_results[label] = res

    aw = res['total_pnl'].values
    tot = aw.sum()
    ltot = res['long_pnl'].sum()
    stot = res['short_pnl'].sum()
    sh = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    cum = np.cumsum(aw)
    dd = (cum - np.maximum.accumulate(cum)).min()
    yr_pnl = res.groupby('year')['total_pnl'].sum()
    yp = (yr_pnl > 0).sum()
    yt = len(yr_pnl)
    avg_lstk = res[res['n_longs'] > 0]['long_streak'].mean()
    swks = (res['n_shorts'] > 0).sum()

    print(f"  {label:>35s} ${tot/1000:>+8.0f}K ${ltot/1000:>+8.0f}K ${stot/1000:>+8.0f}K {sh:>+6.2f} ${dd/1000:>+8.0f}K {dd/CAPITAL*100:>+6.1f}% {yp:>3d}/{yt:<2d} {avg_lstk:>4.1f} {swks:>5d}")

# ═══════════════════════════════════════════════════════════════
# ANNUAL DETAIL: best config vs baselines
# ═══════════════════════════════════════════════════════════════
# Pick top 3 by Sharpe
ranked = sorted(all_results.items(), key=lambda x: x[1]['total_pnl'].values.mean() / x[1]['total_pnl'].values.std() * np.sqrt(52) if x[1]['total_pnl'].values.std() > 0 else 0, reverse=True)
show = ["Baseline (no persistence)", "ATR>=1.3 only (no persist)"] + [ranked[0][0], ranked[1][0]]
show = list(dict.fromkeys(show))  # deduplicate

print(f"\n\n{'=' * 130}")
print(f"  ANNUAL RETURNS (% of ${CAPITAL:,.0f})")
print(f"{'=' * 130}")

print(f"\n  {'Year':>6} {'SPY':>7}", end="")
for label in show:
    short_label = label[:25]
    print(f"  {short_label:>25s}", end="")
print()
print("  " + "-" * (14 + 27 * len(show)))

years = sorted(set().union(*[set(r['year'].unique()) for r in all_results.values()]))
years = [y for y in years if y >= 2000]

cumuls = {l: 1.0 for l in show}

for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")

    for label in show:
        res = all_results[label]
        yr = res[res['year'] == year]
        pnl = yr['total_pnl'].sum()
        ret = pnl / CAPITAL * 100
        cumuls[label] *= (1 + pnl / CAPITAL)
        swks = (yr['n_shorts'] > 0).sum()
        print(f"  {ret:>+7.1f}% ({swks:>2d}s) ${pnl/1000:>+5.0f}K", end="")
    print()

print("  " + "-" * (14 + 27 * len(show)))
print(f"  {'TOTAL':>6} {'':>7}", end="")
for label in show:
    res = all_results[label]
    pnl = res[res['year'] >= 2000]['total_pnl'].sum()
    ret = pnl / CAPITAL * 100
    cagr = (cumuls[label] ** (1/len(years)) - 1) * 100
    print(f"  {ret:>+7.1f}%       ${pnl/1000:>+5.0f}K", end="")
print()

print(f"\n  {'CAGR':>6} {'':>7}", end="")
for label in show:
    cagr = (cumuls[label] ** (1/len(years)) - 1) * 100
    print(f"  {cagr:>+7.1f}%{' ':>19s}", end="")
print()

print(f"\n{'=' * 130}")
