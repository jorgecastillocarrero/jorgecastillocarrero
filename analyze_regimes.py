"""
Comparativa detallada BULLISH vs NEUTRAL vs BEARISH vs CRISIS
Volatilidad, scores, eventos, posiciones
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
sub_labels = {sid: sd['label'] for sid, sd in SUBSECTORS.items()}
MAX_CONTRIBUTION = 4.0

def score_fair(active_events):
    contributions = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        for subsec, impact in EVENT_SUBSECTOR_MAP[evt_type]['impacto'].items():
            if subsec not in contributions:
                contributions[subsec] = []
            contributions[subsec].append(intensity * impact)
    scores = {}
    for sub_id in SUBSECTORS:
        if sub_id not in contributions or len(contributions[sub_id]) == 0:
            scores[sub_id] = 5.0
        else:
            avg = np.mean(contributions[sub_id])
            scores[sub_id] = max(0.0, min(10.0, 5.0 + (avg / MAX_CONTRIBUTION) * 5.0))
    return scores

def adjust_score_by_price(scores, dd_row, rsi_row):
    adjusted = {}
    for sub_id, score in scores.items():
        dd_val = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi_val = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd_val): dd_val = 0
        if not pd.notna(rsi_val): rsi_val = 50
        if score < 5.0:
            dd_factor = np.clip((abs(dd_val) - 15) / 30, 0, 1)
            rsi_factor = np.clip((35 - rsi_val) / 20, 0, 1)
            oversold = max(dd_factor, rsi_factor)
            adjusted[sub_id] = score + (5.0 - score) * oversold * 0.5
        elif score > 5.0:
            rsi_factor = np.clip((rsi_val - 70) / 15, 0, 1)
            adjusted[sub_id] = score - (score - 5.0) * rsi_factor * 0.5
        else:
            adjusted[sub_id] = score
    return adjusted

# Load data
print("Cargando datos...")
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year

df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100
)

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'),
    avg_high=('high', 'mean'),
    avg_low=('low', 'mean'),
    avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean'),
).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_price_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)

returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# SPY weekly
spy = pd.read_sql("""
    SELECT date, close, high, low FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2001-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy['week'] = spy['date'].dt.isocalendar().week.astype(int)
spy['year'] = spy['date'].dt.year
spy_weekly = spy.sort_values('date').groupby(['year', 'week']).last().reset_index()
spy_weekly = spy_weekly.sort_values('date')
spy_weekly['spy_return'] = spy_weekly['close'].pct_change() * 100
spy_weekly['spy_hl'] = (spy_weekly['high'] - spy_weekly['low']) / spy_weekly['close'] * 100
spy_weekly = spy_weekly.set_index('date')

# VIX proxy: average ATR across all subsectors as volatility measure
atr_market = atr_wide.mean(axis=1)

# ---- Collect per-week metrics ----
print("Analizando regimenes...")
rows = []

for date in returns_wide.index:
    if date.year < 2002:
        continue

    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    if not active:
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_adj = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    # Pools
    longs_pool = [(s, sc) for s, sc in scores_adj.items() if sc > 6.5]
    shorts_pool = [(s, sc) for s, sc in scores_adj.items() if sc < 3.5]
    bear_count = len(shorts_pool)
    bull_count = len(longs_pool)
    total_pool = bear_count + bull_count
    bear_ratio = bear_count / total_pool if total_pool > 0 else 0.5

    # Regime
    if bear_ratio >= 0.60:
        regime = 'CRISIS'
    elif bear_ratio >= 0.45:
        regime = 'BEARISH'
    elif bear_ratio >= 0.30:
        regime = 'NEUTRAL'
    else:
        regime = 'BULLISH'

    # Score stats
    all_scores = list(scores_adj.values())
    scores_above_65 = [s for s in all_scores if s > 6.5]
    scores_below_35 = [s for s in all_scores if s < 3.5]
    scores_neutral = [s for s in all_scores if 3.5 <= s <= 6.5]

    # Market volatility (avg ATR of all subsectors)
    mkt_atr = atr_market.get(date, np.nan)

    # DD stats
    if dd_row is not None:
        dd_vals = dd_row.dropna()
        avg_dd = dd_vals.mean()
        min_dd = dd_vals.min()
        pct_dd_gt_20 = (dd_vals < -20).mean() * 100
    else:
        avg_dd = min_dd = pct_dd_gt_20 = np.nan

    # RSI stats
    if rsi_row is not None:
        rsi_vals = rsi_row.dropna()
        avg_rsi = rsi_vals.mean()
        pct_rsi_lt_30 = (rsi_vals < 30).mean() * 100
        pct_rsi_gt_70 = (rsi_vals > 70).mean() * 100
    else:
        avg_rsi = pct_rsi_lt_30 = pct_rsi_gt_70 = np.nan

    # SPY
    spy_idx = spy_weekly.index.get_indexer([date], method='nearest')[0]
    spy_date = spy_weekly.index[spy_idx]
    spy_ret = spy_weekly.iloc[spy_idx]['spy_return'] if abs((spy_date - date).days) < 5 else np.nan
    spy_hl = spy_weekly.iloc[spy_idx]['spy_hl'] if abs((spy_date - date).days) < 5 else np.nan

    # Event intensity sum
    total_intensity = sum(active.values())

    rows.append({
        'date': date,
        'regime': regime,
        'bear_ratio': bear_ratio,
        'n_events': len(active),
        'total_intensity': total_intensity,
        'n_bullish': bull_count,
        'n_bearish': bear_count,
        'n_neutral_subs': len(scores_neutral),
        'avg_score': np.mean(all_scores),
        'std_score': np.std(all_scores),
        'max_score': max(all_scores),
        'min_score': min(all_scores),
        'mkt_atr': mkt_atr,
        'avg_dd': avg_dd,
        'min_dd': min_dd,
        'pct_dd_gt_20': pct_dd_gt_20,
        'avg_rsi': avg_rsi,
        'pct_rsi_lt_30': pct_rsi_lt_30,
        'pct_rsi_gt_70': pct_rsi_gt_70,
        'spy_return': spy_ret,
        'spy_range': spy_hl,
        'events': ', '.join(sorted(active.keys())),
    })

df = pd.DataFrame(rows)

# ---- Print comparison ----
print(f"\n{'='*120}")
print(f"  COMPARATIVA POR REGIMEN DE MERCADO")
print(f"{'='*120}")

metrics = [
    ('Semanas', lambda g: len(g), '{:d}'),
    ('', None, ''),
    ('--- EVENTOS ---', None, ''),
    ('Num eventos activos', lambda g: g['n_events'].mean(), '{:.1f}'),
    ('Intensidad total', lambda g: g['total_intensity'].mean(), '{:.1f}'),
    ('', None, ''),
    ('--- SCORES ---', None, ''),
    ('Score promedio (49 sub)', lambda g: g['avg_score'].mean(), '{:.2f}'),
    ('Dispersion scores (std)', lambda g: g['std_score'].mean(), '{:.2f}'),
    ('Subsectores bullish (>6.5)', lambda g: g['n_bullish'].mean(), '{:.1f}'),
    ('Subsectores bearish (<3.5)', lambda g: g['n_bearish'].mean(), '{:.1f}'),
    ('Subsectores neutrales', lambda g: g['n_neutral_subs'].mean(), '{:.1f}'),
    ('Bear ratio', lambda g: g['bear_ratio'].mean(), '{:.2f}'),
    ('', None, ''),
    ('--- VOLATILIDAD ---', None, ''),
    ('ATR mercado (avg subs %)', lambda g: g['mkt_atr'].mean(), '{:.2f}%'),
    ('SPY rango semanal (H-L%)', lambda g: g['spy_range'].mean(), '{:.2f}%'),
    ('', None, ''),
    ('--- ESTADO PRECIO ---', None, ''),
    ('DD promedio mercado', lambda g: g['avg_dd'].mean(), '{:.1f}%'),
    ('Peor DD individual', lambda g: g['min_dd'].mean(), '{:.1f}%'),
    ('% subs con DD > -20%', lambda g: g['pct_dd_gt_20'].mean(), '{:.0f}%'),
    ('RSI promedio mercado', lambda g: g['avg_rsi'].mean(), '{:.1f}'),
    ('% subs RSI < 30', lambda g: g['pct_rsi_lt_30'].mean(), '{:.0f}%'),
    ('% subs RSI > 70', lambda g: g['pct_rsi_gt_70'].mean(), '{:.0f}%'),
    ('', None, ''),
    ('--- RESULTADO ---', None, ''),
    ('SPY return semanal', lambda g: g['spy_return'].mean(), '{:+.2f}%'),
]

regimes = ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']

# Header
print(f"\n  {'Metrica':<30s}", end='')
for r in regimes:
    print(f"  {r:>12s}", end='')
print()
print(f"  {'-'*82}")

for label, func, fmt in metrics:
    if func is None:
        if label:
            print(f"\n  {label}")
        continue
    print(f"  {label:<30s}", end='')
    for r in regimes:
        g = df[df['regime'] == r]
        val = func(g)
        print(f"  {fmt.format(val):>12s}", end='')
    print()

# ---- Eventos tipicos por regimen ----
print(f"\n\n{'='*120}")
print(f"  EVENTOS MAS FRECUENTES POR REGIMEN")
print(f"{'='*120}")

for regime in regimes:
    g = df[df['regime'] == regime]
    all_events = []
    for evts in g['events']:
        all_events.extend(evts.split(', '))
    from collections import Counter
    event_counts = Counter(all_events)
    n_weeks = len(g)
    print(f"\n  {regime} ({n_weeks} semanas):")
    for evt, count in event_counts.most_common(10):
        pct = count / n_weeks * 100
        print(f"    {evt:40s}  {count:3d}/{n_weeks}  ({pct:5.1f}%)")

# ---- Ejemplos tipicos ----
print(f"\n\n{'='*120}")
print(f"  EJEMPLO TIPICO DE CADA REGIMEN (semana mediana por bear_ratio)")
print(f"{'='*120}")

for regime in regimes:
    g = df[df['regime'] == regime].sort_values('bear_ratio')
    mid = g.iloc[len(g)//2]
    print(f"\n  {regime} - {mid['date'].strftime('%Y-%m-%d')}  br={mid['bear_ratio']:.2f}")
    print(f"    Eventos ({mid['n_events']}): {mid['events'][:90]}")
    print(f"    Scores: avg={mid['avg_score']:.2f} std={mid['std_score']:.2f} | {mid['n_bullish']:.0f} bull, {mid['n_bearish']:.0f} bear, {mid['n_neutral_subs']:.0f} neutral")
    print(f"    Volatilidad: ATR={mid['mkt_atr']:.2f}% SPY_range={mid['spy_range']:.2f}%")
    print(f"    Precio: DD_avg={mid['avg_dd']:.1f}% RSI_avg={mid['avg_rsi']:.1f} | {mid['pct_dd_gt_20']:.0f}% subs DD>-20%, {mid['pct_rsi_lt_30']:.0f}% RSI<30")
    print(f"    SPY: {mid['spy_return']:+.2f}%")
