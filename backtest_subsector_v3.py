"""
Backtest V3: Subsector-level scoring + dynamic allocation + ATR filter
=======================================================================
Target: beat 170% ($1.36M over 24 years) from previous P1+P2 system.

Key improvements over V2:
- Trade at SUBSECTOR level (49 subsectors), not ETF level (9 ETFs)
- Recalibrated event map (408 impacts, all subsectors covered)
- Dynamic NL x NS with max(NL,NS)=3 constraint
- Score-weighted allocation: stronger scores get more capital
- Per-subsector ATR filter for shorts
- Crisis dampening: reduce positive scores when crisis intensity is high
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

# ── Config ────────────────────────────────────────────────────
FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

CAPITAL = 500_000
START_DATE = '2000-01-01'
END_DATE = '2026-02-21'


# ═══════════════════════════════════════════════════════════════
# 1. LOAD SUBSECTOR RETURNS (weekly, equal-weight of tickers)
# ═══════════════════════════════════════════════════════════════
print("Loading price data...")

# Build ticker -> subsector mapping
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id

all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

# Load weekly prices (Friday close)
price_query = f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '{START_DATE}' AND '{END_DATE}'
    AND EXTRACT(DOW FROM date) = 5
    ORDER BY symbol, date
"""
df_prices = pd.read_sql(price_query, engine)
df_prices['date'] = pd.to_datetime(df_prices['date'])
df_prices['subsector'] = df_prices['symbol'].map(ticker_to_sub)
df_prices = df_prices.dropna(subset=['subsector'])
print(f"  Loaded {len(df_prices)} weekly price records for {df_prices['symbol'].nunique()} tickers")

# If not enough Friday data, also get Thursday data as fallback
if len(df_prices) < 100000:
    print("  Also loading Thursday prices as fallback...")
    price_query2 = f"""
        SELECT symbol, date, close, high, low
        FROM fmp_price_history
        WHERE symbol IN ('{tlist}')
        AND date BETWEEN '{START_DATE}' AND '{END_DATE}'
        ORDER BY symbol, date
    """
    df_all = pd.read_sql(price_query2, engine)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
    df_all = df_all.dropna(subset=['subsector'])

    # Resample to weekly (last trading day of each week)
    df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
    df_all['year'] = df_all['date'].dt.year

    # Get last day of each week per ticker
    df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
    df_weekly['subsector'] = df_weekly['symbol'].map(ticker_to_sub)
    print(f"  Resampled to {len(df_weekly)} weekly records")
else:
    df_weekly = df_prices.copy()
    df_weekly['week'] = df_weekly['date'].dt.isocalendar().week.astype(int)
    df_weekly['year'] = df_weekly['date'].dt.year

# Calculate weekly returns per ticker
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = (df_weekly['close'] / df_weekly['prev_close'] - 1)
df_weekly = df_weekly.dropna(subset=['return'])

# Calculate ATR% per ticker (5-week rolling)
df_weekly['high_low_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['high_low_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100
)

# Aggregate to subsector level (equal-weight average)
subsec_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean'),
    n_tickers=('symbol', 'count')
).reset_index()

# Pivot to wide format
returns_wide = subsec_weekly.pivot(index='date', columns='subsector', values='avg_return')
atr_wide = subsec_weekly.pivot(index='date', columns='subsector', values='avg_atr')

# Lag ATR by 1 week (avoid look-ahead)
atr_wide_lagged = atr_wide.shift(1)

print(f"  Weekly returns: {returns_wide.shape[0]} weeks x {returns_wide.shape[1]} subsectors")
print(f"  Date range: {returns_wide.index.min()} to {returns_wide.index.max()}")


# ═══════════════════════════════════════════════════════════════
# 2. BUILD EVENT SCORES (weekly, subsector level)
# ═══════════════════════════════════════════════════════════════
print("\nBuilding weekly event scores...")
weekly_events = build_weekly_events(START_DATE, END_DATE)

def score_subsectors(active_events, dampen=0.0):
    """Score each subsector based on active events.
    dampen: reduce positive scores when negative intensity is high (crisis mode)
    """
    raw_scores = {}
    total_neg_intensity = 0

    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        impacts = EVENT_SUBSECTOR_MAP[evt_type]['impacto']
        for subsec, impact in impacts.items():
            raw_scores[subsec] = raw_scores.get(subsec, 0) + intensity * impact
            if impact < 0:
                total_neg_intensity += abs(intensity * impact)

    if dampen > 0 and total_neg_intensity > 0:
        # Reduce positive scores proportionally to crisis severity
        pos_factor = max(0.0, 1.0 - dampen * total_neg_intensity / 20.0)
        scores = {}
        for subsec, sc in raw_scores.items():
            if sc > 0:
                scores[subsec] = sc * pos_factor
            else:
                scores[subsec] = sc
        return scores

    return raw_scores


def decide_allocation(scores, max_positions=3):
    """Decide NL x NS based on score magnitudes.
    Constraint: max(NL, NS) = max_positions (default 3)
    """
    pos_scores = sorted([(s, sc) for s, sc in scores.items() if sc > 0], key=lambda x: -x[1])
    neg_scores = sorted([(s, sc) for s, sc in scores.items() if sc < 0], key=lambda x: x[1])

    total_pos = sum(sc for _, sc in pos_scores)
    total_neg = sum(abs(sc) for _, sc in neg_scores)
    total = total_pos + total_neg

    if total == 0:
        return [], []

    short_ratio = total_neg / total

    # Decide NL x NS based on relative magnitude
    if short_ratio >= 0.70:
        nl, ns = 0, max_positions
    elif short_ratio >= 0.60:
        nl, ns = 1, max_positions
    elif short_ratio >= 0.50:
        nl, ns = 2, max_positions
    elif short_ratio >= 0.40:
        nl, ns = max_positions, max_positions
    elif short_ratio >= 0.30:
        nl, ns = max_positions, 2
    elif short_ratio >= 0.20:
        nl, ns = max_positions, 1
    else:
        nl, ns = max_positions, 0

    longs = [s for s, _ in pos_scores[:nl]]
    shorts = [s for s, _ in neg_scores[:ns]]

    return longs, shorts


# ═══════════════════════════════════════════════════════════════
# 3. BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════
def backtest(dampen=0.0, atr_min_short=0.0, max_positions=3,
             score_weighted=False, min_score=0.5, capital=CAPITAL):
    """
    Run backtest with given parameters.

    Args:
        dampen: crisis dampening factor (0.0 = off)
        atr_min_short: minimum ATR% to allow short positions
        max_positions: max positions on dominant side
        score_weighted: weight positions by score magnitude
        min_score: minimum absolute score to qualify
        capital: capital per week
    """
    equity = 0.0
    weekly_pnl = []
    config_counts = {}
    yearly_pnl = {}

    for date in returns_wide.index:
        if date.year < 2002:  # Need warmup for ATR
            continue

        # Get active events for this week
        if date not in weekly_events.index:
            nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
            evt_date = weekly_events.index[nearest_idx]
        else:
            evt_date = date

        events_row = weekly_events.loc[evt_date]
        active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

        if not active:
            weekly_pnl.append({'date': date, 'pnl': 0, 'config': '0L+0S'})
            continue

        # Score subsectors
        scores = score_subsectors(active, dampen=dampen)

        # Filter by minimum score
        filtered = {s: sc for s, sc in scores.items() if abs(sc) >= min_score}

        if not filtered:
            weekly_pnl.append({'date': date, 'pnl': 0, 'config': '0L+0S'})
            continue

        # Decide allocation
        longs, shorts = decide_allocation(filtered, max_positions=max_positions)

        # Apply ATR filter to shorts
        if atr_min_short > 0 and date in atr_wide_lagged.index:
            atr_row = atr_wide_lagged.loc[date]
            shorts = [s for s in shorts if pd.notna(atr_row.get(s)) and atr_row[s] >= atr_min_short]

        # Get returns
        ret_row = returns_wide.loc[date]

        n_positions = len(longs) + len(shorts)
        if n_positions == 0:
            weekly_pnl.append({'date': date, 'pnl': 0, 'config': '0L+0S'})
            continue

        # Allocate capital
        if score_weighted and n_positions > 0:
            # Weight by score magnitude
            long_scores = {s: filtered.get(s, 0) for s in longs}
            short_scores = {s: abs(filtered.get(s, 0)) for s in shorts}
            total_score = sum(long_scores.values()) + sum(short_scores.values())

            if total_score > 0:
                week_pnl = 0
                for s in longs:
                    if pd.notna(ret_row.get(s)):
                        weight = long_scores[s] / total_score
                        week_pnl += capital * weight * ret_row[s]
                for s in shorts:
                    if pd.notna(ret_row.get(s)):
                        weight = short_scores[s] / total_score
                        week_pnl += capital * weight * (-ret_row[s])  # Short = negative return
            else:
                week_pnl = 0
        else:
            # Equal weight
            cap_per_pos = capital / n_positions
            week_pnl = 0
            for s in longs:
                if pd.notna(ret_row.get(s)):
                    week_pnl += cap_per_pos * ret_row[s]
            for s in shorts:
                if pd.notna(ret_row.get(s)):
                    week_pnl += cap_per_pos * (-ret_row[s])

        equity += week_pnl
        config = f"{len(longs)}L+{len(shorts)}S"
        config_counts[config] = config_counts.get(config, 0) + 1

        year = date.year
        yearly_pnl[year] = yearly_pnl.get(year, 0) + week_pnl

        weekly_pnl.append({'date': date, 'pnl': week_pnl, 'config': config, 'equity': equity})

    df_pnl = pd.DataFrame(weekly_pnl)

    if len(df_pnl) == 0:
        return None

    # Calculate metrics
    total_pnl = equity
    n_years = (df_pnl['date'].max() - df_pnl['date'].min()).days / 365.25
    cagr = (1 + total_pnl / capital) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    cum_pnl = df_pnl['pnl'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    max_dd_pct = max_dd / capital * 100 if capital > 0 else 0

    # Sharpe
    weekly_returns = df_pnl['pnl'] / capital
    sharpe = weekly_returns.mean() / weekly_returns.std() * np.sqrt(52) if weekly_returns.std() > 0 else 0

    # Win rate
    active_weeks = df_pnl[df_pnl['pnl'] != 0]
    win_rate = (active_weeks['pnl'] > 0).mean() * 100 if len(active_weeks) > 0 else 0

    # Profitable years
    n_profitable = sum(1 for y, p in yearly_pnl.items() if p > 0)
    n_total_years = len(yearly_pnl)

    return {
        'total_pnl': total_pnl,
        'cagr': cagr * 100,
        'max_dd': max_dd,
        'max_dd_pct': max_dd_pct,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'n_weeks': len(active_weeks),
        'config_counts': config_counts,
        'yearly_pnl': yearly_pnl,
        'n_profitable_years': n_profitable,
        'n_total_years': n_total_years,
        'df_pnl': df_pnl,
        'pnl_per_year': total_pnl / n_years if n_years > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# 4. OPTIMIZATION: TEST ALL PARAMETER COMBINATIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
print("  OPTIMIZATION: SUBSECTOR-LEVEL BACKTEST")
print("=" * 120)

configs = [
    # (name, dampen, atr_min_short, max_positions, score_weighted, min_score)
    ("Baseline 3x3 EW",          0.0,  0.0, 3, False, 0.5),
    ("ATR>=1.3 short",           0.0,  1.3, 3, False, 0.5),
    ("ATR>=1.5 short",           0.0,  1.5, 3, False, 0.5),
    ("ATR>=2.0 short",           0.0,  2.0, 3, False, 0.5),
    ("Score-weighted",           0.0,  0.0, 3, True,  0.5),
    ("Score-wtd + ATR1.3",       0.0,  1.3, 3, True,  0.5),
    ("Score-wtd + ATR1.5",       0.0,  1.5, 3, True,  0.5),
    ("Dampen 0.10",              0.10, 0.0, 3, False, 0.5),
    ("Dampen 0.10 + ATR1.3",    0.10, 1.3, 3, False, 0.5),
    ("Dampen 0.10 + SW",        0.10, 0.0, 3, True,  0.5),
    ("Dampen 0.10 + SW + ATR1.3", 0.10, 1.3, 3, True, 0.5),
    ("Dampen 0.15 + SW + ATR1.3", 0.15, 1.3, 3, True, 0.5),
    ("Dampen 0.20 + SW + ATR1.3", 0.20, 1.3, 3, True, 0.5),
    ("Dampen 0.10 + SW + ATR1.5", 0.10, 1.5, 3, True, 0.5),
    ("MinScore 1.0",             0.0,  0.0, 3, False, 1.0),
    ("MinScore 1.0 + ATR1.3",   0.0,  1.3, 3, False, 1.0),
    ("MinScore 1.0 + SW",       0.0,  1.3, 3, True,  1.0),
    ("Dampen 0.10 + MS1.0 + SW + ATR1.3", 0.10, 1.3, 3, True, 1.0),
    ("MaxPos 4",                 0.0,  1.3, 4, False, 0.5),
    ("MaxPos 5",                 0.0,  1.3, 5, False, 0.5),
    ("MaxPos 4 + SW",           0.0,  1.3, 4, True,  0.5),
    ("MaxPos 5 + SW",           0.0,  1.3, 5, True,  0.5),
    ("D0.10 + MP5 + SW + ATR1.3", 0.10, 1.3, 5, True, 0.5),
    ("D0.10 + MP4 + SW + ATR1.3", 0.10, 1.3, 4, True, 0.5),
]

print(f"\n  {'Config':>40s} {'Total$':>10s} {'CAGR%':>7s} {'MaxDD$':>10s} {'MaxDD%':>7s} {'Sharpe':>7s} {'WinR%':>6s} {'Weeks':>6s} {'Yr+':>4s} {'$/yr':>10s}")
print("  " + "-" * 120)

results = []
for name, dampen, atr, maxpos, sw, ms in configs:
    r = backtest(dampen=dampen, atr_min_short=atr, max_positions=maxpos,
                 score_weighted=sw, min_score=ms)
    if r is None:
        continue

    results.append((name, r))

    print(f"  {name:>40s} {r['total_pnl']:>+10,.0f} {r['cagr']:>6.1f}% {r['max_dd']:>+10,.0f} {r['max_dd_pct']:>6.1f}% "
          f"{r['sharpe']:>6.2f} {r['win_rate']:>5.1f}% {r['n_weeks']:>6d} "
          f"{r['n_profitable_years']}/{r['n_total_years']} {r['pnl_per_year']:>+10,.0f}")


# ═══════════════════════════════════════════════════════════════
# 5. BEST CONFIG DETAILS
# ═══════════════════════════════════════════════════════════════
# Sort by total PnL
results.sort(key=lambda x: x[1]['total_pnl'], reverse=True)

print(f"\n\n{'=' * 120}")
print(f"  TOP 5 CONFIGS BY TOTAL PNL")
print(f"{'=' * 120}")

for i, (name, r) in enumerate(results[:5]):
    print(f"\n  #{i+1}: {name}")
    print(f"  Total: ${r['total_pnl']:+,.0f} | CAGR: {r['cagr']:.1f}% | Sharpe: {r['sharpe']:.2f} | MaxDD: {r['max_dd_pct']:.1f}%")
    print(f"  Win rate: {r['win_rate']:.1f}% | Active weeks: {r['n_weeks']} | Profitable years: {r['n_profitable_years']}/{r['n_total_years']}")

    # Config distribution
    print(f"  Configs: ", end="")
    for cfg, cnt in sorted(r['config_counts'].items()):
        total_active = sum(r['config_counts'].values())
        pct = cnt / total_active * 100
        print(f"{cfg}={pct:.0f}% ", end="")
    print()

    # Yearly PnL
    print(f"  {'Year':>6s} {'PnL':>10s}")
    for year in sorted(r['yearly_pnl'].keys()):
        pnl = r['yearly_pnl'][year]
        marker = " <<<" if pnl < -20000 else " ***" if pnl > 100000 else ""
        print(f"  {year:>6d} {pnl:>+10,.0f}{marker}")


# ═══════════════════════════════════════════════════════════════
# 6. 2008 DEEP DIVE (best config)
# ═══════════════════════════════════════════════════════════════
best_name, best_r = results[0]
print(f"\n\n{'=' * 120}")
print(f"  2008 DEEP DIVE: {best_name}")
print(f"{'=' * 120}")

df_best = best_r['df_pnl']
df_2008 = df_best[(df_best['date'].dt.year >= 2007) & (df_best['date'].dt.year <= 2009)]

print(f"\n  {'Date':>12s} {'PnL':>10s} {'CumPnL':>10s} {'Config':>8s}")
print(f"  {'-' * 50}")

cum = 0
for _, row in df_2008.iterrows():
    cum += row['pnl']
    if row['pnl'] != 0:
        print(f"  {row['date'].strftime('%Y-%m-%d'):>12s} {row['pnl']:>+10,.0f} {cum:>+10,.0f} {row['config']:>8s}")

print(f"\n  2007 total: ${best_r['yearly_pnl'].get(2007, 0):+,.0f}")
print(f"  2008 total: ${best_r['yearly_pnl'].get(2008, 0):+,.0f}")
print(f"  2009 total: ${best_r['yearly_pnl'].get(2009, 0):+,.0f}")


# Sort by Sharpe ratio too
results_sharpe = sorted(results, key=lambda x: x[1]['sharpe'], reverse=True)
print(f"\n\n{'=' * 120}")
print(f"  TOP 5 BY SHARPE RATIO")
print(f"{'=' * 120}")
for i, (name, r) in enumerate(results_sharpe[:5]):
    print(f"  #{i+1}: {name:>40s} | Sharpe: {r['sharpe']:.2f} | Total: ${r['total_pnl']:+,.0f} | CAGR: {r['cagr']:.1f}% | MaxDD: {r['max_dd_pct']:.1f}%")

# CAGR/DD ratio
results_cagrdd = sorted(results, key=lambda x: abs(x[1]['cagr'] / x[1]['max_dd_pct']) if x[1]['max_dd_pct'] != 0 else 0, reverse=True)
print(f"\n  TOP 5 BY CAGR/DD RATIO")
print(f"  {'-' * 100}")
for i, (name, r) in enumerate(results_cagrdd[:5]):
    ratio = abs(r['cagr'] / r['max_dd_pct']) if r['max_dd_pct'] != 0 else 0
    print(f"  #{i+1}: {name:>40s} | CAGR/DD: {ratio:.2f} | CAGR: {r['cagr']:.1f}% | MaxDD: {r['max_dd_pct']:.1f}% | Total: ${r['total_pnl']:+,.0f}")

print(f"\n{'=' * 120}")
print(f"  Target: $1,000,000+ (200%+) with Sharpe > 1.0")
print(f"{'=' * 120}")
