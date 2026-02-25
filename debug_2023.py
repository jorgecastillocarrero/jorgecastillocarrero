"""
Debug 2023: Por que Fair V3 perdio -$49K cuando SPY gano +22.5%?
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

def decide_allocation(scores, max_pos=3):
    longs_pool = sorted([(s, sc) for s, sc in scores.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores.items() if sc < 3.5], key=lambda x: x[1])
    bear_count = len(shorts_pool)
    bull_count = len(longs_pool)
    if bear_count + bull_count == 0:
        bear_ratio = 0.5
    else:
        bear_ratio = bear_count / (bear_count + bull_count)
    if bear_ratio >= 0.70:   nl, ns = 0, max_pos
    elif bear_ratio >= 0.60: nl, ns = 1, max_pos
    elif bear_ratio >= 0.55: nl, ns = 2, max_pos
    elif bear_ratio >= 0.45: nl, ns = max_pos, max_pos
    elif bear_ratio >= 0.40: nl, ns = max_pos, 2
    elif bear_ratio >= 0.30: nl, ns = max_pos, 1
    else:                    nl, ns = max_pos, 0
    return [s for s, _ in longs_pool[:nl]], [s for s, _ in shorts_pool[:ns]], bear_ratio

# ---- Load data ----
print("Loading data...")
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
    AND date BETWEEN '2021-01-01' AND '2024-02-01'
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
atr_wide_lagged = atr_wide.shift(1)

weekly_events = build_weekly_events('2021-01-01', '2024-02-01')

CAPITAL = 500_000
ATR_MIN = 1.5

# ---- Que eventos estaban activos en 2023? ----
print(f"\n{'='*100}")
print(f"  EVENTOS ACTIVOS EN 2023")
print(f"{'='*100}")

weeks_2023 = returns_wide.index[
    (returns_wide.index >= '2023-01-01') & (returns_wide.index <= '2023-12-31')
]

# Contar eventos por semana
event_activity = {}
for date in weeks_2023:
    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    for evt, intensity in active.items():
        if evt not in event_activity:
            event_activity[evt] = []
        event_activity[evt].append((date, intensity))

for evt, weeks in sorted(event_activity.items(), key=lambda x: -len(x[1])):
    first = min(w[0] for w in weeks)
    last = max(w[0] for w in weeks)
    avg_int = np.mean([w[1] for w in weeks])
    print(f"  {evt:35s}  {len(weeks):2d} semanas  int={avg_int:.1f}  ({first.strftime('%m/%d')}-{last.strftime('%m/%d')})")


# ---- Semana a semana con detalle de posiciones ----
print(f"\n{'='*100}")
print(f"  DETALLE SEMANAL 2023 - Fair V3")
print(f"{'='*100}")

month_names = {1:'ENE', 2:'FEB', 3:'MAR', 4:'ABR', 5:'MAY', 6:'JUN',
               7:'JUL', 8:'AGO', 9:'SEP', 10:'OCT', 11:'NOV', 12:'DIC'}

monthly_pnl = {}
cum_pnl = 0

for date in weeks_2023:
    month = date.month

    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    if not active:
        if month not in monthly_pnl:
            monthly_pnl[month] = 0
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_adj = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    longs, shorts, br = decide_allocation(scores_adj)

    if date in atr_wide_lagged.index:
        atr_row = atr_wide_lagged.loc[date]
        shorts = [s for s in shorts if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    ret_row = returns_wide.loc[date]

    n = len(longs) + len(shorts)
    pnl = 0
    positions = []
    if n > 0:
        lw = {s: scores_adj[s] - 5.0 for s in longs}
        sw = {s: 5.0 - scores_adj[s] for s in shorts}
        tw = sum(lw.values()) + sum(sw.values())
        if tw > 0:
            for s in longs:
                if pd.notna(ret_row.get(s)):
                    w = lw[s] / tw
                    p = CAPITAL * w * ret_row[s]
                    pnl += p
                    positions.append(('L', s, scores_adj[s], ret_row[s], w, p))
            for s in shorts:
                if pd.notna(ret_row.get(s)):
                    w = sw[s] / tw
                    p = CAPITAL * w * (-ret_row[s])
                    pnl += p
                    positions.append(('S', s, scores_adj[s], ret_row[s], w, p))

    cum_pnl += pnl
    if month not in monthly_pnl:
        monthly_pnl[month] = 0
    monthly_pnl[month] += pnl

    cfg = f"{len(longs)}L+{len(shorts)}S"
    evt_str = ','.join(sorted(active.keys()))[:60]
    print(f"\n  {date.strftime('%Y-%m-%d')} {month_names[month]}  {cfg:6s}  br={br:.2f}  PnL=${pnl:+9,.0f}  Cum=${cum_pnl:+10,.0f}")
    print(f"    Eventos: {evt_str}")
    for side, s, sc, ret, w, p in positions:
        label = sub_labels.get(s, s)[:30]
        print(f"      {side} {label:30s}  sc={sc:.1f}  w={w:.0%}  ret={ret*100:+6.2f}%  ${p:+,.0f}")

# ---- Resumen mensual ----
print(f"\n\n{'='*100}")
print(f"  RESUMEN MENSUAL 2023")
print(f"{'='*100}")

cum = 0
for m in sorted(monthly_pnl.keys()):
    cum += monthly_pnl[m]
    print(f"  {month_names[m]:4s}  ${monthly_pnl[m]:+10,.0f}  Cum: ${cum:+10,.0f}")

print(f"\n  TOTAL: ${cum:+,.0f}")

# ---- Que subsectores subieron mas en 2023? (retorno anual) ----
print(f"\n\n{'='*100}")
print(f"  RETORNO ANUAL 2023 POR SUBSECTOR (top 15 y bottom 15)")
print(f"{'='*100}")

sub_annual = {}
for sub_id in SUBSECTORS:
    col = sub_id
    if col in returns_wide.columns:
        rets_2023 = returns_wide.loc[weeks_2023, col].dropna()
        if len(rets_2023) > 0:
            annual_ret = (1 + rets_2023).prod() - 1
            sub_annual[sub_id] = annual_ret * 100

sorted_subs = sorted(sub_annual.items(), key=lambda x: -x[1])

print(f"\n  TOP 15:")
for sub_id, ret in sorted_subs[:15]:
    label = sub_labels.get(sub_id, sub_id)
    print(f"    {label:35s}  {ret:+7.1f}%")

print(f"\n  BOTTOM 15:")
for sub_id, ret in sorted_subs[-15:]:
    label = sub_labels.get(sub_id, sub_id)
    print(f"    {label:35s}  {ret:+7.1f}%")

# ---- Que subsectores tenia el sistema como top longs? ----
print(f"\n\n{'='*100}")
print(f"  SCORES FAIR V3 - EJEMPLO MITAD DE 2023 (semana Jul)")
print(f"{'='*100}")

mid_date = weeks_2023[len(weeks_2023)//2]
if mid_date in weekly_events.index:
    evt_date = mid_date
else:
    nearest_idx = weekly_events.index.get_indexer([mid_date], method='nearest')[0]
    evt_date = weekly_events.index[nearest_idx]

events_row = weekly_events.loc[evt_date]
active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
scores = score_fair(active)

prev_dates = dd_wide.index[dd_wide.index < mid_date]
dd_r = dd_wide.loc[prev_dates[-1]]
rsi_r = rsi_wide.loc[prev_dates[-1]]
scores_adj = adjust_score_by_price(scores, dd_r, rsi_r)

sorted_scores = sorted(scores_adj.items(), key=lambda x: -x[1])

print(f"\n  Fecha: {mid_date.strftime('%Y-%m-%d')}")
print(f"  Eventos activos: {', '.join(active.keys())}")
print(f"\n  {'Subsector':35s}  {'Score Evt':>9s}  {'Score Adj':>9s}  {'Ret 2023':>9s}")
print(f"  {'-'*70}")
for sub_id, sc_adj in sorted_scores[:15]:
    label = sub_labels.get(sub_id, sub_id)[:35]
    sc_evt = scores.get(sub_id, 5.0)
    ret_yr = sub_annual.get(sub_id, 0)
    marker = " <-- TOP LONG" if sc_adj > 6.5 else ""
    print(f"  {label:35s}  {sc_evt:>9.2f}  {sc_adj:>9.2f}  {ret_yr:>+8.1f}%{marker}")

print(f"\n  BOTTOM 10:")
for sub_id, sc_adj in sorted_scores[-10:]:
    label = sub_labels.get(sub_id, sub_id)[:35]
    sc_evt = scores.get(sub_id, 5.0)
    ret_yr = sub_annual.get(sub_id, 0)
    marker = " <-- SHORT" if sc_adj < 3.5 else ""
    print(f"  {label:35s}  {sc_evt:>9.2f}  {sc_adj:>9.2f}  {ret_yr:>+8.1f}%{marker}")
