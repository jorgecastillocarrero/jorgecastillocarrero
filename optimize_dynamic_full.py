"""
Full dynamic optimization.
CONSTRAINT: max(n_longs, n_shorts) = 3 ALWAYS.
Valid weekly configs: 3L+0S, 3L+1S, 3L+2S, 3L+3S, 2L+3S, 1L+3S, 0L+3S.
Capital $500K split among all active positions.
Per-ETF ATR filter on shorts (lagged 1 week).
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
                     long_threshold=1.0, short_threshold=1.0,
                     atr_min=1.3,
                     dominance='auto'):
    """
    Dynamic backtest. CONSTRAINT: max(NL, NS) = 3 always.

    Logic each week:
    1. Score all sectors
    2. Qualifying longs = score > long_threshold (up to 3)
    3. Qualifying shorts = score < -short_threshold AND per-ETF ATR >= atr_min (up to 3)
    4. Enforce constraint: at least one side = 3
       - If qualifying_longs >= 3: take 3L + qualifying_shorts (0-3)
       - Elif qualifying_shorts == 3: take qualifying_longs (0-2) + 3S
       - Else: force dominant side to 3 (top 3 longs or bottom 3 shorts by score)

    dominance: 'auto' (signal strength), 'long' (always force 3L), 'short' (prefer 3S)
    """
    common = weekly_events.index.intersection(etf_returns.index).sort_values()
    results = []

    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]
        prev_date = common[i - 1] if i > 0 else None

        events_row = weekly_events.loc[date]
        active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

        subsec_scores = score_week(active)
        sector_scores = aggregate_to_sectors(subsec_scores)
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

        # Qualifying candidates
        qual_longs = [etf for etf, sc in ranked[:3] if sc > long_threshold]
        qual_shorts_score = [etf for etf, sc in ranked[-3:] if sc < -short_threshold]

        # Per-ETF ATR filter on shorts (lagged 1 week)
        qual_shorts = []
        if prev_date:
            for etf in qual_shorts_score:
                if etf in etf_atr and prev_date in etf_atr[etf].index:
                    if atr_min <= 0 or etf_atr[etf].loc[prev_date] >= atr_min:
                        qual_shorts.append(etf)
                elif atr_min <= 0:
                    qual_shorts.append(etf)

        # Forced top/bottom 3 (regardless of threshold)
        forced_top3 = [etf for etf, _ in ranked[:3]]
        forced_bot3_score = [etf for etf, _ in ranked[-3:]]
        forced_bot3 = []
        if prev_date:
            for etf in forced_bot3_score:
                if etf in etf_atr and prev_date in etf_atr[etf].index:
                    if atr_min <= 0 or etf_atr[etf].loc[prev_date] >= atr_min:
                        forced_bot3.append(etf)
                elif atr_min <= 0:
                    forced_bot3.append(etf)

        # ── ENFORCE CONSTRAINT: max(NL, NS) = 3 ──
        if not active:
            # No events active: default to 3L+0S (market neutral/long bias)
            current_longs = forced_top3[:3]
            current_shorts = []
        elif len(qual_longs) >= 3:
            # Longs qualified for 3 -> 3L + dynamic shorts
            current_longs = qual_longs[:3]
            current_shorts = qual_shorts[:3]
        elif len(qual_shorts) >= 3:
            # Shorts qualified for 3 -> dynamic longs + 3S
            current_longs = qual_longs[:3]
            current_shorts = qual_shorts[:3]
        else:
            # Neither side has 3 qualifying. Force dominant side to 3.
            if dominance == 'long':
                current_longs = forced_top3[:3]
                current_shorts = qual_shorts[:3]
            elif dominance == 'short':
                current_shorts = forced_bot3[:3] if len(forced_bot3) >= 3 else forced_bot3
                current_longs = qual_longs[:3]
                if len(current_shorts) < 3:
                    # Can't force 3 shorts (ATR), fallback to 3 longs
                    current_longs = forced_top3[:3]
            else:  # auto
                long_signal = sum(max(0, sc) for _, sc in ranked[:3])
                short_signal = sum(abs(min(0, sc)) for _, sc in ranked[-3:])
                if long_signal >= short_signal:
                    current_longs = forced_top3[:3]
                    current_shorts = qual_shorts[:3]
                else:
                    if len(forced_bot3) >= 3:
                        current_shorts = forced_bot3[:3]
                        current_longs = qual_longs[:3]
                    else:
                        current_longs = forced_top3[:3]
                        current_shorts = qual_shorts[:3]

        # Remove overlaps
        overlap = set(current_longs) & set(current_shorts)
        for etf in overlap:
            current_shorts.remove(etf)

        # FINAL VALIDATION: ensure max(NL, NS) = 3
        n_l = len(current_longs)
        n_s = len(current_shorts)
        if max(n_l, n_s) < 3 and (n_l + n_s) > 0:
            # Force longs to 3 as fallback
            current_longs = forced_top3[:3]
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
# 1. ALL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 155}")
print(f"  DYNAMIC OPTIMIZATION - max(NL, NS) = 3 always")
print(f"  Capital: ${CAPITAL:,.0f} split among all positions")
print(f"  Valid configs: 3L+0S, 3L+1S, 3L+2S, 3L+3S, 2L+3S, 1L+3S, 0L+3S")
print(f"{'=' * 155}")

configs = [
    # label, long_thr, short_thr, atr_min, dominance
    ("Auto ATR>=1.3 (default)",      1.0, 1.0, 1.3, 'auto'),
    ("Auto ATR>=1.0",                1.0, 1.0, 1.0, 'auto'),
    ("Auto ATR>=1.5",                1.0, 1.0, 1.5, 'auto'),
    ("Auto ATR>=1.8",                1.0, 1.0, 1.8, 'auto'),
    ("Auto no ATR",                  1.0, 1.0, 0.0, 'auto'),
    ("Auto L_thr=2.0 ATR1.3",       2.0, 1.0, 1.3, 'auto'),
    ("Auto S_thr=2.0 ATR1.3",       1.0, 2.0, 1.3, 'auto'),
    ("Auto L2 S2 ATR1.3",           2.0, 2.0, 1.3, 'auto'),
    ("Auto L_thr=0.5 ATR1.3",       0.5, 1.0, 1.3, 'auto'),
    ("Auto S_thr=0.5 ATR1.3",       1.0, 0.5, 1.3, 'auto'),
    ("",                             0, 0, 0, ''),  # separator
    ("Long bias ATR>=1.3",           1.0, 1.0, 1.3, 'long'),
    ("Long bias ATR>=1.0",           1.0, 1.0, 1.0, 'long'),
    ("Long bias no ATR",             1.0, 1.0, 0.0, 'long'),
    ("",                             0, 0, 0, ''),  # separator
    ("Short bias ATR>=1.3",          1.0, 1.0, 1.3, 'short'),
    ("Short bias ATR>=1.0",          1.0, 1.0, 1.0, 'short'),
]

print(f"\n  {'Config':>35s} {'Total':>10} {'Long':>10} {'Short':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>10} {'DD%':>7} {'Yr+':>7} {'Win%':>6} {'AvgNL':>6} {'AvgNS':>6}")
print("  " + "-" * 145)

all_results = {}
all_metrics = []

for label, lt, st, atr, dom in configs:
    if not dom:
        print()
        continue

    res = backtest_dynamic(weekly_events, etf_returns_w, etf_atr,
                           long_threshold=lt, short_threshold=st,
                           atr_min=atr, dominance=dom)

    all_results[label] = res
    m = compute_metrics(res, label)
    if m:
        all_metrics.append(m)
        print(f"  {label:>35s} ${m['total']/1000:>+8.0f}K ${m['long_pnl']/1000:>+8.0f}K ${m['short_pnl']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} ${m['maxdd']/1000:>+8.0f}K {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d} {m['win_rate']:>5.1f}% {m['avg_nl']:>5.1f} {m['avg_ns']:>5.1f}")


# ═══════════════════════════════════════════════════════════════
# 2. RANKING
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 110}")
print(f"  RANKING BY SHARPE")
print(f"{'=' * 110}")

ranked_sh = sorted(all_metrics, key=lambda x: x['sharpe'], reverse=True)
print(f"\n  {'#':>3} {'Config':>35s} {'Total':>10} {'CAGR':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Yr+':>7} {'AvgNL':>6} {'AvgNS':>6}")
print("  " + "-" * 100)
for i, m in enumerate(ranked_sh):
    print(f"  {i+1:>3d} {m['label']:>35s} ${m['total']/1000:>+8.0f}K {m['cagr']:>+6.1f}% {m['sharpe']:>+6.2f} {m['maxdd_pct']:>+6.1f}% {m['yrs_pos']:>3d}/{m['yrs_tot']:<2d} {m['avg_nl']:>5.1f} {m['avg_ns']:>5.1f}")


# ═══════════════════════════════════════════════════════════════
# 3. CONFIG DISTRIBUTION for top 4
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 100}")
print(f"  WEEKLY CONFIG DISTRIBUTION - Top 4 by Sharpe")
print(f"{'=' * 100}")

for idx in range(min(4, len(ranked_sh))):
    m = ranked_sh[idx]
    label = m['label']
    res = all_results[label]
    br = res[res['year'] >= 2000]

    print(f"\n  #{idx+1} {label} (Sharpe {m['sharpe']:+.2f}, CAGR {m['cagr']:+.1f}%, DD {m['maxdd_pct']:+.1f}%)")
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
# 4. YEAR-BY-YEAR: Top 4
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 155}")
print(f"  YEAR-BY-YEAR: Top 4 by Sharpe")
print(f"{'=' * 155}")

top4 = [ranked_sh[i]['label'] for i in range(min(4, len(ranked_sh)))]

col_w = 32
print(f"\n  {'Year':>6} {'SPY':>7}", end="")
for label in top4:
    short_l = label[:30]
    print(f"  {short_l:>{col_w}s}", end="")
print()
print("  " + "-" * (14 + (col_w + 2) * len(top4)))

years = sorted(set().union(*[set(all_results[l]['year'].unique()) for l in top4]))
years = [y for y in years if y >= 2000]

cumuls_yr = {l: 1.0 for l in top4}

for year in years:
    spy_yr = spy_ret_w[spy_ret_w.index.year == year].dropna()
    spy_ann = (1 + spy_yr).prod() - 1 if len(spy_yr) > 0 else 0

    print(f"  {year:>6d} {spy_ann:>+6.1%}", end="")
    for label in top4:
        res = all_results[label]
        yr = res[res['year'] == year]
        pnl = yr['total_pnl'].sum()
        ret = pnl / CAPITAL * 100
        cumuls_yr[label] *= (1 + pnl / CAPITAL)
        avg_nl = yr['n_longs'].mean() if len(yr) > 0 else 0
        avg_ns = yr['n_shorts'].mean() if len(yr) > 0 else 0
        print(f"  {ret:>+6.1f}% {avg_nl:.1f}L {avg_ns:.1f}S ${pnl/1000:>+6.0f}K", end="")
    print()

print("  " + "-" * (14 + (col_w + 2) * len(top4)))
print(f"  {'CAGR':>6} {'':>7}", end="")
for label in top4:
    cagr = (cumuls_yr[label] ** (1/len(years)) - 1) * 100
    print(f"  {cagr:>+6.1f}%{' ':>26s}", end="")
print()

print(f"  {'$500K':>6} {'':>7}", end="")
for label in top4:
    final = CAPITAL * cumuls_yr[label]
    print(f"  ${final/1e6:>5.2f}M{' ':>25s}", end="")
print()

print(f"\n{'=' * 155}")
