"""
Dynamic Backtest V2: Score-weighted allocation + crisis dampening.

Fixes:
1. Crisis dampening: when total negative intensity is high, positive
   "defensive" impacts are reduced (in 2008 even XLP dropped 15%)
2. Score-weighted NL/NS: use ratio of positive vs negative magnitude
   to decide how many longs vs shorts each week
3. Subsector-level scoring throughout

Constraint: max(NL, NS) = 3 always.
Capital: $500K split among all active positions.
"""
import pandas as pd
import numpy as np
import psycopg2
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


def score_week_v2(active_events, crisis_dampen=0.15):
    """
    Score sub-sectors with crisis dampening.

    When multiple negative events overlap (crisis), positive "defensive" impacts
    are reduced. In reality, severe crises drag everything down.

    crisis_dampen: how much to reduce positive impacts per unit of negative intensity.
    At dampen=0.15:
      - 1 event intensity 1: positives at 85%
      - 3 events intensity 2: positives at 10%
      - 5 events intensity 3: positives at ~0% (capped at 0)
    """
    # First pass: raw scores + total negative intensity
    raw_scores = {}
    total_neg_intensity = 0

    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        impacts = EVENT_SUBSECTOR_MAP[evt_type]['impacto']
        # Count negative intensity
        for subsec, impact in impacts.items():
            if impact < 0:
                total_neg_intensity += intensity * abs(impact)

    # Crisis dampening factor for positive impacts
    # More negative intensity = more dampening of positives
    pos_factor = max(0.0, 1.0 - crisis_dampen * total_neg_intensity / 10.0)

    # Second pass: apply scores with dampening
    scores = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        impacts = EVENT_SUBSECTOR_MAP[evt_type]['impacto']
        for subsec, impact in impacts.items():
            raw = intensity * impact
            if raw > 0:
                raw *= pos_factor  # Dampen positives in crisis
            scores[subsec] = scores.get(subsec, 0) + raw

    return scores, total_neg_intensity, pos_factor


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


def decide_allocation(sector_scores, short_ratio_threshold=0.5):
    """
    Use magnitude of positive vs negative scores to decide NL and NS.

    short_ratio = neg_magnitude / (pos_magnitude + neg_magnitude)
    - If short_ratio > 0.7: heavy short (0L+3S or 1L+3S)
    - If short_ratio > 0.5: balanced (2L+3S or 3L+3S)
    - If short_ratio > 0.3: light short (3L+1S or 3L+2S)
    - If short_ratio <= 0.3: long only (3L+0S)
    """
    pos_scores = {etf: sc for etf, sc in sector_scores.items() if sc > 0}
    neg_scores = {etf: sc for etf, sc in sector_scores.items() if sc < 0}

    pos_mag = sum(pos_scores.values())
    neg_mag = abs(sum(neg_scores.values()))
    total_mag = pos_mag + neg_mag

    if total_mag == 0:
        return 3, 0, 0.0  # No signal

    short_ratio = neg_mag / total_mag

    # Determine NL and NS based on ratio
    if short_ratio >= 0.70:
        nl, ns = 0, 3  # Pure short
    elif short_ratio >= 0.60:
        nl, ns = 1, 3  # Short dominant
    elif short_ratio >= 0.50:
        nl, ns = 2, 3  # Balanced-short
    elif short_ratio >= 0.40:
        nl, ns = 3, 3  # Balanced
    elif short_ratio >= 0.30:
        nl, ns = 3, 2  # Long dominant, some shorts
    elif short_ratio >= 0.15:
        nl, ns = 3, 1  # Long dominant
    else:
        nl, ns = 3, 0  # Pure long

    return nl, ns, short_ratio


def backtest_dynamic_v2(weekly_events, etf_returns, etf_atr,
                        long_threshold=1.0, short_threshold=1.0,
                        atr_min=1.3, crisis_dampen=0.15):
    """
    Dynamic backtest V2 with crisis dampening and score-weighted allocation.
    """
    common = weekly_events.index.intersection(etf_returns.index).sort_values()
    results = []

    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]
        prev_date = common[i - 1] if i > 0 else None

        events_row = weekly_events.loc[date]
        active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

        if not active:
            results.append({
                'date': next_date, 'year': next_date.year,
                'n_longs': 0, 'n_shorts': 0, 'config': '0L+0S',
                'longs': '', 'shorts': '',
                'long_pnl': 0, 'short_pnl': 0, 'total_pnl': 0,
                'short_ratio': 0, 'pos_factor': 1.0, 'neg_intensity': 0,
            })
            continue

        # Score with crisis dampening
        subsec_scores, neg_intensity, pos_factor = score_week_v2(active, crisis_dampen)
        sector_scores = aggregate_to_sectors(subsec_scores)

        # Determine allocation by score magnitude
        target_nl, target_ns, short_ratio = decide_allocation(sector_scores)

        # Rank sectors
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

        # Select longs: top target_nl with score > threshold (or forced if target says so)
        if target_nl > 0:
            candidates_l = [etf for etf, sc in ranked[:target_nl]]
            current_longs = candidates_l[:target_nl]
        else:
            current_longs = []

        # Select shorts: bottom target_ns with ATR filter per-ETF
        if target_ns > 0:
            candidates_s = [etf for etf, sc in ranked[-target_ns:] if sc < -short_threshold]
            current_shorts = []
            if prev_date:
                for etf in candidates_s:
                    if etf in etf_atr and prev_date in etf_atr[etf].index:
                        if atr_min <= 0 or etf_atr[etf].loc[prev_date] >= atr_min:
                            current_shorts.append(etf)
                    elif atr_min <= 0:
                        current_shorts.append(etf)
            # If ATR filtered out all shorts, but target says short-dominant,
            # still need max(NL,NS)=3, so force 3 longs
            if not current_shorts and target_nl < 3:
                current_longs = [etf for etf, _ in ranked[:3]]
        else:
            current_shorts = []

        # Remove overlaps
        overlap = set(current_longs) & set(current_shorts)
        for etf in overlap:
            current_shorts.remove(etf)

        # Enforce max(NL, NS) = 3
        n_l = len(current_longs)
        n_s = len(current_shorts)
        if max(n_l, n_s) < 3 and (n_l + n_s) > 0:
            if n_l >= n_s:
                current_longs = [etf for etf, _ in ranked[:3]]
            else:
                # Try to force 3 shorts
                extra_shorts = [etf for etf, sc in ranked[-3:] if etf not in current_shorts]
                if prev_date:
                    for etf in extra_shorts:
                        if len(current_shorts) >= 3:
                            break
                        if etf in etf_atr and prev_date in etf_atr[etf].index:
                            if atr_min <= 0 or etf_atr[etf].loc[prev_date] >= atr_min:
                                current_shorts.append(etf)
                if len(current_shorts) < 3:
                    current_longs = [etf for etf, _ in ranked[:3]]
            # Re-remove overlaps
            current_shorts = [e for e in current_shorts if e not in current_longs]
            n_l = len(current_longs)
            n_s = len(current_shorts)

        n_total = n_l + n_s
        long_pnl = 0
        short_pnl = 0

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
            'date': next_date, 'year': next_date.year,
            'n_longs': n_l, 'n_shorts': n_s,
            'config': f"{n_l}L+{n_s}S",
            'longs': ','.join(current_longs),
            'shorts': ','.join(current_shorts),
            'long_pnl': long_pnl, 'short_pnl': short_pnl,
            'total_pnl': long_pnl + short_pnl,
            'short_ratio': short_ratio,
            'pos_factor': pos_factor,
            'neg_intensity': neg_intensity,
        })

    return pd.DataFrame(results)


def compute_metrics(res, label=""):
    r = res[res['year'] >= 2000]
    aw = r['total_pnl'].values
    if len(aw) == 0:
        return None
    tot = aw.sum()
    sh = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    cum = np.cumsum(aw)
    dd = (cum - np.maximum.accumulate(cum)).min()
    dd_pct = dd / CAPITAL * 100
    yr_pnl = r.groupby('year')['total_pnl'].sum()
    yp = (yr_pnl > 0).sum()
    yt = len(yr_pnl)
    ltot = r['long_pnl'].sum()
    stot = r['short_pnl'].sum()
    active_w = aw[aw != 0]
    wins = (active_w > 0).sum()
    wr = wins / len(active_w) * 100 if len(active_w) > 0 else 0
    avg_nl = r['n_longs'].mean()
    avg_ns = r['n_shorts'].mean()
    cumul = 1.0
    for year in sorted(yr_pnl.index):
        cumul *= (1 + yr_pnl[year] / CAPITAL)
    cagr = (cumul ** (1/len(yr_pnl)) - 1) * 100
    return {
        'label': label, 'total': tot, 'long_pnl': ltot, 'short_pnl': stot,
        'cagr': cagr, 'sharpe': sh, 'maxdd': dd, 'maxdd_pct': dd_pct,
        'yrs_pos': yp, 'yrs_tot': yt, 'win_rate': wr,
        'avg_nl': avg_nl, 'avg_ns': avg_ns, 'cumul': cumul,
    }


# =====================================================================
print("Loading data...")
etf_prices_w, etf_returns_w, spy_w, spy_ret_w, etf_atr = load_data()
weekly_events = build_weekly_events('1999-01-01', '2026-03-01')


# ═══════════════════════════════════════════════════════════════
# 1. TEST DAMPENING LEVELS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 155}")
print(f"  DYNAMIC V2: Crisis Dampening + Score-Weighted Allocation")
print(f"  Capital: ${CAPITAL:,.0f} split among all | Per-ETF ATR filter | max(NL,NS)=3")
print(f"{'=' * 155}")

configs = [
    # label, long_thr, short_thr, atr_min, crisis_dampen
    ("No dampen, no ATR",             1.0, 1.0, 0.0, 0.0),
    ("No dampen, ATR>=1.3",           1.0, 1.0, 1.3, 0.0),
    ("Dampen=0.10, ATR>=1.3",         1.0, 1.0, 1.3, 0.10),
    ("Dampen=0.15, ATR>=1.3",         1.0, 1.0, 1.3, 0.15),
    ("Dampen=0.20, ATR>=1.3",         1.0, 1.0, 1.3, 0.20),
    ("Dampen=0.25, ATR>=1.3",         1.0, 1.0, 1.3, 0.25),
    ("Dampen=0.30, ATR>=1.3",         1.0, 1.0, 1.3, 0.30),
    ("Dampen=0.15, ATR>=1.0",         1.0, 1.0, 1.0, 0.15),
    ("Dampen=0.15, no ATR",           1.0, 1.0, 0.0, 0.15),
    ("Dampen=0.20, no ATR",           1.0, 1.0, 0.0, 0.20),
    ("Dampen=0.20, ATR>=1.0",         1.0, 1.0, 1.0, 0.20),
]

print(f"\n  {'Config':>30s} {'Total':>10} {'Long':>10} {'Short':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>10} {'DD%':>7} {'Yr+':>7} {'Win%':>6} {'AvgNL':>6} {'AvgNS':>6}")
print("  " + "-" * 140)

all_results = {}
all_metrics = []

for label, lt, st, atr, damp in configs:
    res = backtest_dynamic_v2(weekly_events, etf_returns_w, etf_atr,
                              long_threshold=lt, short_threshold=st,
                              atr_min=atr, crisis_dampen=damp)

    all_results[label] = res
    m = compute_metrics(res, label)
    if m:
        all_metrics.append(m)
        print(f"  {label:>30s} ${m['total']/1000:>+8.0f}K ${m['long_pnl']/1000:>+8.0f}K ${m['short_pnl']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} ${m['maxdd']/1000:>+8.0f}K {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d} {m['win_rate']:>5.1f}% {m['avg_nl']:>5.1f} {m['avg_ns']:>5.1f}")


# ═══════════════════════════════════════════════════════════════
# 2. CONFIG DISTRIBUTION for best configs
# ═══════════════════════════════════════════════════════════════
ranked_sh = sorted(all_metrics, key=lambda x: x['sharpe'], reverse=True)

print(f"\n\n{'=' * 110}")
print(f"  RANKING BY SHARPE")
print(f"{'=' * 110}")

print(f"\n  {'#':>3} {'Config':>30s} {'Total':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Yr+':>7} {'Short$':>10}")
print("  " + "-" * 95)
for i, m in enumerate(ranked_sh):
    print(f"  {i+1:>3d} {m['label']:>30s} ${m['total']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d} ${m['short_pnl']/1000:>+8.0f}K")


# Distribution for top 3
print(f"\n\n{'=' * 100}")
print(f"  WEEKLY CONFIG DISTRIBUTION - Top 3")
print(f"{'=' * 100}")

for idx in range(min(3, len(ranked_sh))):
    m = ranked_sh[idx]
    label = m['label']
    res = all_results[label]
    br = res[res['year'] >= 2000]

    print(f"\n  #{idx+1} {label} (Sharpe {m['sharpe']:+.2f}, CAGR {m['cagr']:+.1f}%)")
    config_counts = br['config'].value_counts().sort_index()
    print(f"  {'Config':>10s} {'Weeks':>6} {'%':>6} {'Avg PnL/wk':>12} {'Total PnL':>12} {'Win%':>6}")
    print("  " + "-" * 60)

    for cfg in sorted(config_counts.index):
        mask = br['config'] == cfg
        wks = mask.sum()
        pct = wks / len(br) * 100
        avg = br.loc[mask, 'total_pnl'].mean()
        tot_cfg = br.loc[mask, 'total_pnl'].sum()
        wins = (br.loc[mask, 'total_pnl'] > 0).sum()
        wr_cfg = wins / wks * 100
        print(f"  {cfg:>10s} {wks:>6d} {pct:>5.1f}% ${avg:>+11,.0f} ${tot_cfg/1000:>+10,.0f}K {wr_cfg:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# 3. 2008 DETAIL for best dampen config
# ═══════════════════════════════════════════════════════════════
best_label = ranked_sh[0]['label']
best = all_results[best_label]

print(f"\n\n{'=' * 130}")
print(f"  2007-2009 WEEKLY DETAIL: {best_label}")
print(f"{'=' * 130}")

detail = best[(best['year'] >= 2007) & (best['year'] <= 2009)].copy()
print(f"\n  {'Date':>12} {'Config':>8} {'Longs':>20} {'Shorts':>20} {'ShortR':>7} {'PosFact':>8} {'L PnL':>10} {'S PnL':>10} {'Total':>10}")
print("  " + "-" * 120)

for _, row in detail.iterrows():
    print(f"  {row['date'].strftime('%Y-%m-%d'):>12} {row['config']:>8} {row['longs']:>20} {row['shorts']:>20} {row['short_ratio']:>6.2f} {row['pos_factor']:>7.2f} ${row['long_pnl']:>+9,.0f} ${row['short_pnl']:>+9,.0f} ${row['total_pnl']:>+9,.0f}")


# ═══════════════════════════════════════════════════════════════
# 4. YEAR-BY-YEAR: Top 3
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 145}")
print(f"  YEAR-BY-YEAR: Top 3 by Sharpe")
print(f"{'=' * 145}")

top3 = [ranked_sh[i]['label'] for i in range(min(3, len(ranked_sh)))]

col_w = 32
print(f"\n  {'Year':>6} {'SPY':>7}", end="")
for label in top3:
    short_l = label[:30]
    print(f"  {short_l:>{col_w}s}", end="")
print()
print("  " + "-" * (14 + (col_w + 2) * len(top3)))

years = sorted(set().union(*[set(all_results[l]['year'].unique()) for l in top3]))
years = [y for y in years if y >= 2000]

cumuls_yr = {l: 1.0 for l in top3}

for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")
    for label in top3:
        res = all_results[label]
        yr = res[res['year'] == year]
        pnl = yr['total_pnl'].sum()
        ret = pnl / CAPITAL * 100
        cumuls_yr[label] *= (1 + pnl / CAPITAL)
        avg_nl = yr['n_longs'].mean() if len(yr) > 0 else 0
        avg_ns = yr['n_shorts'].mean() if len(yr) > 0 else 0
        print(f"  {ret:>+6.1f}% {avg_nl:.1f}L {avg_ns:.1f}S ${pnl/1000:>+6.0f}K", end="")
    print()

print("  " + "-" * (14 + (col_w + 2) * len(top3)))
print(f"  {'CAGR':>6} {'':>7}", end="")
for label in top3:
    cagr = (cumuls_yr[label] ** (1/len(years)) - 1) * 100
    print(f"  {cagr:>+6.1f}%{' ':>26s}", end="")
print()

print(f"  {'$500K':>6} {'':>7}", end="")
for label in top3:
    final = CAPITAL * cumuls_yr[label]
    print(f"  ${final/1e6:>5.2f}M{' ':>25s}", end="")
print()

print(f"\n{'=' * 145}")
