"""
Backtest V2: Event Calendar -> Sub-Sector Scoring -> Sector ETFs
=================================================================
Uses Wikipedia-sourced historical event calendar (ground truth)
instead of noisy price-proxy detection.

Flow:
  1. For each week, check which events are active (from calendar)
  2. Score sub-sectors using event x impact map
  3. Aggregate to sector ETF level
  4. Long best sectors, Short worst sectors
  5. Momentum persistence: don't rotate every week
"""

import pandas as pd
import numpy as np
import psycopg2
from collections import Counter
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import EVENT_CALENDAR, build_weekly_events

DB = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'
SECTOR_ETFS = ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']


def load_etf_returns():
    """Load weekly returns of sector ETFs."""
    conn = psycopg2.connect(DB)
    ph = ','.join(['%s'] * len(SECTOR_ETFS))
    df = pd.read_sql(
        f"SELECT symbol, date, close FROM fmp_price_history "
        f"WHERE symbol IN ({ph}) ORDER BY date",
        conn, params=SECTOR_ETFS, parse_dates=['date']
    )
    conn.close()
    prices = df.pivot_table(index='date', columns='symbol', values='close')
    prices_w = prices.resample('W-FRI').last()
    returns_w = prices_w.pct_change()
    return prices_w, returns_w


def score_week(active_events):
    """Score sub-sectors for a week based on active events.

    active_events: dict {event_type: intensity}
    Returns: dict {subsector_id: score}
    """
    scores = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        impacts = EVENT_SUBSECTOR_MAP[evt_type]['impacto']
        for subsec, impact in impacts.items():
            scores[subsec] = scores.get(subsec, 0) + intensity * impact
    return scores


def aggregate_to_sectors(subsector_scores):
    """Aggregate sub-sector scores to sector ETF level."""
    sector_scores = {}
    sector_details = {}

    for subsec_id, subsec_data in SUBSECTORS.items():
        etf = subsec_data['etf']
        score = subsector_scores.get(subsec_id, 0)
        if score != 0:
            sector_scores.setdefault(etf, []).append(score)
            sector_details.setdefault(etf, []).append(
                (subsec_id, subsec_data['label'], score)
            )

    result = {}
    for etf in SECTOR_ETFS:
        subscores = sector_scores.get(etf, [])
        result[etf] = sum(subscores)

    return result, sector_details


def backtest(weekly_events, etf_returns,
             n_long=3, n_short=3,
             capital_per_side=500_000,
             momentum_decay=0.5,
             min_score=1.0):
    """Run the backtest.

    Args:
        weekly_events: DataFrame with event intensities per week
        etf_returns: DataFrame with ETF weekly returns
        n_long/n_short: max positions each side
        capital_per_side: $ per side
        momentum_decay: how much of previous score persists (0=none, 1=full)
        min_score: minimum sector score to trigger position
    """
    common = weekly_events.index.intersection(etf_returns.index).sort_values()
    results = []
    prev_scores = {etf: 0 for etf in SECTOR_ETFS}

    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]

        # 1. Get active events for this week
        events_row = weekly_events.loc[date]
        active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

        # 2. Score sub-sectors
        subsec_scores = score_week(active)

        # 3. Aggregate to sectors
        sector_scores, details = aggregate_to_sectors(subsec_scores)

        # 4. Blend with momentum (persistence)
        blended = {}
        for etf in SECTOR_ETFS:
            raw = sector_scores.get(etf, 0)
            prev = prev_scores.get(etf, 0)
            blended[etf] = raw + momentum_decay * prev
        prev_scores = blended.copy()

        # 5. Rank and select
        ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        longs = [(etf, sc) for etf, sc in ranked[:n_long] if sc > min_score]
        shorts = [(etf, sc) for etf, sc in ranked[-n_short:] if sc < -min_score]

        # 6. Calculate PnL from next week's return
        long_pnl = 0
        short_pnl = 0
        n_l = len(longs)
        n_s = len(shorts)

        if n_l > 0:
            cap = capital_per_side / n_l
            for etf, _ in longs:
                r = etf_returns.loc[next_date].get(etf, 0)
                if pd.isna(r): r = 0
                long_pnl += cap * r

        if n_s > 0:
            cap = capital_per_side / n_s
            for etf, _ in shorts:
                r = etf_returns.loc[next_date].get(etf, 0)
                if pd.isna(r): r = 0
                short_pnl += cap * (-r)

        results.append({
            'date': next_date,
            'year': next_date.year,
            'n_events': len(active),
            'events': '|'.join(active.keys()),
            'n_longs': n_l, 'n_shorts': n_s,
            'longs': ','.join(e for e, _ in longs),
            'shorts': ','.join(e for e, _ in shorts),
            'long_pnl': long_pnl, 'short_pnl': short_pnl,
            'total_pnl': long_pnl + short_pnl,
            'sector_scores': blended.copy(),
        })

    return pd.DataFrame(results)


def print_results(results, label=""):
    """Print annual table."""
    if label:
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")

    print(f"\n  {'Year':>6} {'PnL':>12} {'Long':>10} {'Short':>10} {'Weeks':>6} {'Active':>6} {'Win%':>6}")
    print("  " + "-" * 64)

    total = 0
    yrs_pos = 0
    yrs_tot = 0
    all_w = []

    for year in sorted(results['year'].unique()):
        yr = results[results['year'] == year]
        pnl = yr['total_pnl'].sum()
        lpnl = yr['long_pnl'].sum()
        spnl = yr['short_pnl'].sum()
        n = len(yr)
        act = len(yr[(yr['n_longs'] > 0) | (yr['n_shorts'] > 0)])
        wins = len(yr[yr['total_pnl'] > 0])
        wpct = wins / max(1, act) * 100

        total += pnl
        yrs_tot += 1
        if pnl > 0: yrs_pos += 1
        all_w.extend(yr['total_pnl'].tolist())

        s = '+' if pnl > 0 else '-' if pnl < 0 else ' '
        print(f"  {year:>6d} {s}${abs(pnl):>10,.0f} ${lpnl:>+9,.0f} ${spnl:>+9,.0f} {n:>6d} {act:>6d} {wpct:>5.1f}%")

    print("  " + "-" * 64)
    aw = np.array(all_w)
    sharpe = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    cum = np.cumsum(aw)
    dd = (cum - np.maximum.accumulate(cum)).min()
    act_total = sum(1 for _, r in results.iterrows() if r['n_longs'] > 0 or r['n_shorts'] > 0)
    act_pct = act_total / len(results) * 100

    print(f"  {'TOTAL':>6}  ${total:>11,.0f}   Sharpe: {sharpe:.2f}   MaxDD: ${dd:,.0f}")
    print(f"  Positive years: {yrs_pos}/{yrs_tot}   Active: {act_pct:.0f}% of weeks")

    return total, sharpe, yrs_pos, yrs_tot


# ── BASELINE: Pure momentum for comparison ──

def backtest_momentum(etf_prices, etf_returns, n_long=3, n_short=3,
                       capital=500_000, lookback=4):
    """Simple sector momentum baseline."""
    common = etf_prices.dropna(how='all').index.intersection(etf_returns.index).sort_values()
    results = []

    for i in range(lookback, len(common) - 1):
        date = common[i]
        next_date = common[i + 1]
        prev_date = common[i - lookback]

        moms = {}
        for etf in SECTOR_ETFS:
            c = etf_prices.loc[date].get(etf)
            p = etf_prices.loc[prev_date].get(etf)
            if pd.notna(c) and pd.notna(p) and p > 0:
                moms[etf] = c / p - 1
        if len(moms) < 5:
            continue

        ranked = sorted(moms.items(), key=lambda x: x[1], reverse=True)
        longs = ranked[:n_long]
        shorts = ranked[-n_short:]

        lpnl = sum(capital / n_long * (etf_returns.loc[next_date].get(e, 0) if pd.notna(etf_returns.loc[next_date].get(e, 0)) else 0) for e, _ in longs)
        spnl = sum(capital / n_short * (-(etf_returns.loc[next_date].get(e, 0) if pd.notna(etf_returns.loc[next_date].get(e, 0)) else 0)) for e, _ in shorts)

        results.append({
            'date': next_date, 'year': next_date.year,
            'n_events': 0, 'events': '',
            'n_longs': n_long, 'n_shorts': n_short,
            'longs': ','.join(e for e, _ in longs),
            'shorts': ','.join(e for e, _ in shorts),
            'long_pnl': lpnl, 'short_pnl': spnl,
            'total_pnl': lpnl + spnl,
            'sector_scores': {},
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  BACKTEST V2: Event Calendar -> Sub-Sector -> Sector ETFs")
    print("  144 events (Wikipedia) x 30 types x 49 sub-sectors -> 9 ETFs")
    print("=" * 72)

    # 1. Load data
    print("\n1. Loading sector ETF prices...")
    etf_prices, etf_returns = load_etf_returns()
    print(f"   {len(etf_prices)} weeks, {etf_prices.index[0].strftime('%Y-%m-%d')} -> {etf_prices.index[-1].strftime('%Y-%m-%d')}")

    # 2. Build weekly events from calendar
    print("\n2. Building weekly event signals from calendar...")
    weekly_events = build_weekly_events('1999-01-01', '2026-03-01')
    n_active_weeks = (weekly_events.sum(axis=1) > 0).sum()
    print(f"   {len(weekly_events)} weeks, {n_active_weeks} with at least 1 event ({n_active_weeks/len(weekly_events)*100:.0f}%)")

    # Show weekly event count distribution
    event_counts = (weekly_events > 0).sum(axis=1)
    print(f"   Events per week: mean={event_counts.mean():.1f}, max={event_counts.max()}")

    # 3. Test multiple configurations
    print("\n3. Testing configurations...")
    configs = [
        ("3L+3S dec=0.5 thr=1.0",   3, 3, 0.5, 1.0),
        ("3L+3S dec=0.5 thr=2.0",   3, 3, 0.5, 2.0),
        ("3L+3S dec=0.5 thr=3.0",   3, 3, 0.5, 3.0),
        ("3L+3S dec=0.3 thr=2.0",   3, 3, 0.3, 2.0),
        ("3L+3S dec=0.7 thr=2.0",   3, 3, 0.7, 2.0),
        ("2L+2S dec=0.5 thr=2.0",   2, 2, 0.5, 2.0),
        ("4L+4S dec=0.5 thr=1.0",   4, 4, 0.5, 1.0),
        ("3L+3S dec=0.0 thr=1.0",   3, 3, 0.0, 1.0),  # No momentum
        ("3L+3S dec=0.0 thr=2.0",   3, 3, 0.0, 2.0),  # No momentum, higher threshold
    ]

    best_sharpe = -999
    best_label = ""
    best_res = None

    for label, nl, ns, decay, thr in configs:
        res = backtest(weekly_events, etf_returns,
                       n_long=nl, n_short=ns,
                       momentum_decay=decay, min_score=thr)
        if res.empty:
            continue

        aw = res['total_pnl'].values
        tot = aw.sum()
        sh = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
        act = len(res[(res['n_longs'] > 0) | (res['n_shorts'] > 0)])
        apct = act / len(res) * 100
        yp = sum(1 for y in res.groupby('year')['total_pnl'].sum() if y > 0)
        yt = len(res['year'].unique())

        print(f"   {label:30s}: ${tot:>+11,.0f}  Sharpe {sh:>+.2f}  Active {apct:>4.0f}%  Years+ {yp}/{yt}")

        if sh > best_sharpe:
            best_sharpe = sh
            best_label = label
            best_res = res

    # 4. Print best config details
    print(f"\n   >> Best: {best_label}")
    total, sharpe, yp, yt = print_results(best_res, f"EVENT CALENDAR STRATEGY: {best_label}")

    # 5. Baseline comparison
    print("\n4. Baseline: Pure 4-week Momentum 3L+3S...")
    base_res = backtest_momentum(etf_prices, etf_returns, n_long=3, n_short=3)
    if not base_res.empty:
        print_results(base_res, "BASELINE: 4-week Momentum 3L+3S")

    # 6. Event activity analysis
    print(f"\n5. Event Activity in Best Config:")
    ecounts = Counter()
    for estr in best_res['events']:
        if estr:
            for e in estr.split('|'):
                ecounts[e] += 1

    for ev, cnt in ecounts.most_common(20):
        n_impacts = len(EVENT_SUBSECTOR_MAP.get(ev, {}).get('impacto', {}))
        print(f"   {ev:35s}: {cnt:4d} weeks ({cnt/len(best_res)*100:5.1f}%)  -> {n_impacts} sub-sectors")

    # 7. Sector selection
    print(f"\n6. Sector Selection:")
    lc = Counter()
    sc = Counter()
    for _, row in best_res.iterrows():
        if row['longs']:
            for s in row['longs'].split(','):
                lc[s] += 1
        if row['shorts']:
            for s in row['shorts'].split(','):
                sc[s] += 1

    print(f"   {'LONG':20s}  {'SHORT':20s}")
    ll = lc.most_common(9)
    sl = sc.most_common(9)
    for i in range(max(len(ll), len(sl))):
        ls = f"{ll[i][0]:4s} ({ll[i][1]:3d})" if i < len(ll) else ""
        ss = f"{sl[i][0]:4s} ({sl[i][1]:3d})" if i < len(sl) else ""
        print(f"   {ls:20s}  {ss:20s}")

    # 8. Current signals
    print(f"\n7. Current Sector Scores (latest week):")
    last_date = weekly_events.index[-1]
    events_row = weekly_events.loc[last_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    if active:
        print(f"   Date: {last_date.strftime('%Y-%m-%d')}")
        print(f"   Active events: {', '.join(f'{k}({v:.1f})' for k, v in active.items())}")
        sscore = score_week(active)
        sagg, det = aggregate_to_sectors(sscore)
        ranked = sorted(sagg.items(), key=lambda x: x[1], reverse=True)
        print(f"\n   {'ETF':>5} {'Score':>8}  Sub-sectors")
        print("   " + "-" * 60)
        for etf, score in ranked:
            d = det.get(etf, [])
            dstr = ", ".join(f"{s[0]}({s[2]:+.1f})" for s in sorted(d, key=lambda x: -x[2]))
            marker = " << LONG" if score > 2 else " << SHORT" if score < -2 else ""
            print(f"   {etf:>5} {score:>+8.1f}  {dstr}{marker}")
    else:
        print(f"   No events active on {last_date.strftime('%Y-%m-%d')}")

    # 9. Best/worst periods
    print(f"\n8. Best/Worst 5 Weeks:")
    sorted_weeks = best_res.sort_values('total_pnl')
    print(f"   WORST:")
    for _, row in sorted_weeks.head(5).iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: ${row['total_pnl']:>+10,.0f}  L:[{row['longs']}] S:[{row['shorts']}]")
    print(f"   BEST:")
    for _, row in sorted_weeks.tail(5).iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: ${row['total_pnl']:>+10,.0f}  L:[{row['longs']}] S:[{row['shorts']}]")

    return best_res


if __name__ == '__main__':
    main()
