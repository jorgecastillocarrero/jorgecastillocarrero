"""
Fair V3 - Tabla semanal completa 2002-2025
==========================================
Exporta CSV con: fecha, regimen, config (NL+NS), bear_ratio,
longs, shorts, PnL V3, PnL acumulado, SPY return semanal
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
CAPITAL = 500_000
ATR_MIN = 1.5

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
atr_wide_lagged = atr_wide.shift(1)

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# SPY semanal
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2001-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy['week'] = spy['date'].dt.isocalendar().week.astype(int)
spy['year'] = spy['date'].dt.year
spy_weekly = spy.sort_values('date').groupby(['year', 'week']).last().reset_index()
spy_weekly = spy_weekly.sort_values('date')
spy_weekly['spy_prev'] = spy_weekly['close'].shift(1)
spy_weekly['spy_return'] = (spy_weekly['close'] / spy_weekly['spy_prev'] - 1) * 100
spy_weekly = spy_weekly.set_index('date')

# ---- Backtest loop ----
print("Ejecutando backtest...")

rows = []
cum_pnl = 0

for date in returns_wide.index:
    if date.year < 2002:
        continue

    # Eventos
    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    n_events = len(active)
    events_str = ', '.join(sorted(active.keys())) if active else ''

    # SPY return
    spy_idx = spy_weekly.index.get_indexer([date], method='nearest')[0]
    spy_date = spy_weekly.index[spy_idx]
    spy_ret = spy_weekly.iloc[spy_idx]['spy_return'] if abs((spy_date - date).days) < 5 else 0

    if not active:
        cum_pnl += 0
        rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'iso_week': date.isocalendar()[1],
            'n_events': 0,
            'events': '',
            'regime': 'NO_EVENTS',
            'bear_ratio': None,
            'config': '0L+0S',
            'n_longs': 0,
            'n_shorts': 0,
            'longs': '',
            'shorts': '',
            'pnl': 0,
            'cum_pnl': cum_pnl,
            'spy_return_pct': round(spy_ret, 2),
        })
        continue

    # Scores
    scores_evt = score_fair(active)

    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_adj = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    longs, shorts, bear_ratio = decide_allocation(scores_adj)

    # ATR filter
    if date in atr_wide_lagged.index:
        atr_row = atr_wide_lagged.loc[date]
        shorts = [s for s in shorts if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    ret_row = returns_wide.loc[date]

    # P&L
    n = len(longs) + len(shorts)
    pnl = 0
    if n > 0:
        lw = {s: scores_adj[s] - 5.0 for s in longs}
        sw = {s: 5.0 - scores_adj[s] for s in shorts}
        tw = sum(lw.values()) + sum(sw.values())
        if tw > 0:
            for s in longs:
                if pd.notna(ret_row.get(s)):
                    pnl += CAPITAL * (lw[s] / tw) * ret_row[s]
            for s in shorts:
                if pd.notna(ret_row.get(s)):
                    pnl += CAPITAL * (sw[s] / tw) * (-ret_row[s])

    cum_pnl += pnl

    # Regime
    if bear_ratio >= 0.60:
        regime = 'CRISIS'
    elif bear_ratio >= 0.45:
        regime = 'BEARISH'
    elif bear_ratio >= 0.30:
        regime = 'NEUTRAL'
    else:
        regime = 'BULLISH'

    cfg = f"{len(longs)}L+{len(shorts)}S"
    longs_str = ', '.join(sub_labels.get(s, s) for s in longs)
    shorts_str = ', '.join(sub_labels.get(s, s) for s in shorts)

    rows.append({
        'date': date.strftime('%Y-%m-%d'),
        'year': date.year,
        'iso_week': date.isocalendar()[1],
        'n_events': n_events,
        'events': events_str,
        'regime': regime,
        'bear_ratio': round(bear_ratio, 3),
        'config': cfg,
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'longs': longs_str,
        'shorts': shorts_str,
        'pnl': round(pnl, 0),
        'cum_pnl': round(cum_pnl, 0),
        'spy_return_pct': round(spy_ret, 2),
    })

df_out = pd.DataFrame(rows)

# Guardar CSV
csv_path = 'data/fair_v3_weekly_table.csv'
df_out.to_csv(csv_path, index=False)
print(f"\nCSV guardado: {csv_path} ({len(df_out)} semanas)")

# ---- Print resumen por consola ----
print(f"\n{'='*140}")
print(f"  TABLA SEMANAL COMPLETA - FAIR V3 ({len(df_out)} semanas, 2002-2025)")
print(f"{'='*140}")

# Resumen por regimen
print(f"\n  DISTRIBUCION POR REGIMEN:")
for regime in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS', 'NO_EVENTS']:
    subset = df_out[df_out['regime'] == regime]
    if len(subset) == 0:
        continue
    total_pnl = subset['pnl'].sum()
    avg_pnl = subset['pnl'].mean()
    n_weeks = len(subset)
    win_rate = (subset['pnl'] > 0).mean() * 100 if len(subset[subset['pnl'] != 0]) > 0 else 0
    avg_spy = subset['spy_return_pct'].mean()
    print(f"    {regime:12s}: {n_weeks:4d} semanas  PnL=${total_pnl:>+12,.0f}  Avg=${avg_pnl:>+8,.0f}/sem  WR={win_rate:5.1f}%  SPY avg={avg_spy:+.2f}%")

# Print tabla (primeras y ultimas semanas por ano)
print(f"\n  MUESTRA POR ANO (primera, media y ultima semana de cada ano):")
print(f"  {'Fecha':>12s} {'Regimen':>8s} {'Config':>7s} {'BR':>5s} {'#Evt':>4s} {'PnL':>10s} {'CumPnL':>12s} {'SPY%':>6s} {'Longs':>45s} {'Shorts':>45s}")
print(f"  {'-'*160}")

for year in sorted(df_out['year'].unique()):
    yr_data = df_out[df_out['year'] == year]
    if len(yr_data) == 0:
        continue
    # First, mid, last
    indices = [0, len(yr_data)//2, len(yr_data)-1]
    for idx in indices:
        r = yr_data.iloc[idx]
        br = f"{r['bear_ratio']:.2f}" if pd.notna(r['bear_ratio']) else "  -"
        longs_short = r['longs'][:42] + '...' if len(r['longs']) > 45 else r['longs']
        shorts_short = r['shorts'][:42] + '...' if len(r['shorts']) > 45 else r['shorts']
        print(f"  {r['date']:>12s} {r['regime']:>8s} {r['config']:>7s} {br:>5s} {r['n_events']:>4d} ${r['pnl']:>+9,.0f} ${r['cum_pnl']:>+11,.0f} {r['spy_return_pct']:>+5.1f}% {longs_short:>45s} {shorts_short:>45s}")

    yr_total = yr_data['pnl'].sum()
    print(f"  {'':>12s} {'':>8s} {'':>7s} {'':>5s} {'':>4s} ${yr_total:>+9,.0f} {'':>12s} {'':>6s} {'--- '+str(year)+' total ---':>45s}")
    print()

print(f"\n  Total: ${df_out['pnl'].sum():+,.0f} en {len(df_out)} semanas")
print(f"  CSV completo en: {csv_path}")
