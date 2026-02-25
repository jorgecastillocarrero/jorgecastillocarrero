"""
Validacion de regimenes: VIX real vs clasificacion del sistema
==============================================================
Usa VIX semanal como ground truth para validar el bear_ratio y regimen.

VIX levels (historicos):
  <15: Complacencia (ultra bull)
  15-20: Normal/Bull
  20-25: Elevado / transicion
  25-30: Miedo / Bear
  30-40: Panico / Crisis
  >40: Extremo (crash activo)

SPY weekly return como check adicional.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
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

# Load price data for DD/RSI
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

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
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

dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# VIX
vix_raw = pd.read_csv('data/vix_weekly.csv', header=[0, 1], index_col=0)
vix_raw.index = pd.to_datetime(vix_raw.index)
vix = vix_raw[('Close', '^VIX')].rename('vix_close')
vix_high = vix_raw[('High', '^VIX')].rename('vix_high')

# SPY
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2001-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy['week'] = spy['date'].dt.isocalendar().week.astype(int)
spy['year'] = spy['date'].dt.year
spy_weekly = spy.sort_values('date').groupby(['year', 'week']).last().reset_index()
spy_weekly = spy_weekly.sort_values('date').set_index('date')
spy_weekly['spy_ret'] = spy_weekly['close'].pct_change() * 100

# ---- Build comparison ----
print("Comparando regimenes...\n")

rows = []
for date in returns_wide.index:
    if date.year < 2002:
        continue

    # Events
    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    # Scores
    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    # Bear ratio
    longs_pool = [s for s, sc in scores_v3.items() if sc > 6.5]
    shorts_pool = [s for s, sc in scores_v3.items() if sc < 3.5]
    bc = len(shorts_pool)
    lc = len(longs_pool)
    br = bc / (bc + lc) if bc + lc > 0 else 0.5

    # Regime
    if br >= 0.60:     regime = 'CRISIS'
    elif br >= 0.50:   regime = 'BEARISH'
    elif br >= 0.30:   regime = 'NEUTRAL'
    else:              regime = 'BULLISH'

    # VIX
    vix_idx = vix.index.get_indexer([date], method='nearest')[0]
    vix_date = vix.index[vix_idx]
    if abs((vix_date - date).days) < 5:
        vix_val = vix.iloc[vix_idx]
        vix_hi = vix_high.iloc[vix_idx]
    else:
        vix_val = np.nan
        vix_hi = np.nan

    # VIX regime
    if pd.notna(vix_val):
        if vix_val >= 30:      vix_regime = 'CRISIS'
        elif vix_val >= 25:    vix_regime = 'BEARISH'
        elif vix_val >= 20:    vix_regime = 'NEUTRAL'
        else:                  vix_regime = 'BULLISH'
    else:
        vix_regime = None

    # SPY
    spy_idx = spy_weekly.index.get_indexer([date], method='nearest')[0]
    spy_date = spy_weekly.index[spy_idx]
    spy_ret = spy_weekly.iloc[spy_idx]['spy_ret'] if abs((spy_date - date).days) < 5 else np.nan

    # Events list
    events_str = ', '.join(sorted(active.keys())) if active else ''

    rows.append({
        'date': date,
        'year': date.year,
        'regime': regime,
        'br': br,
        'n_bull': lc,
        'n_bear': bc,
        'vix': vix_val,
        'vix_high': vix_hi,
        'vix_regime': vix_regime,
        'match': regime == vix_regime if vix_regime else None,
        'spy_ret': spy_ret,
        'n_events': len(active),
        'events': events_str,
        'avg_score': np.mean(list(scores_v3.values())),
    })

df = pd.DataFrame(rows)

# ================================================================
# RESULTADOS
# ================================================================
print(f"{'='*130}")
print(f"  VALIDACION REGIMEN vs VIX - {len(df)} semanas")
print(f"{'='*130}")

# 1. Match rate global
valid = df[df['vix_regime'].notna()]
match_rate = valid['match'].mean() * 100
print(f"\n  Match rate global: {match_rate:.1f}% ({valid['match'].sum()}/{len(valid)})")

# 2. Confusion matrix
print(f"\n  MATRIZ DE CONFUSION (filas=sistema, columnas=VIX):")
print(f"  {'':>12s} {'VIX_BULL':>10s} {'VIX_NEUT':>10s} {'VIX_BEAR':>10s} {'VIX_CRIS':>10s} {'Total':>8s}")
print(f"  {'-'*60}")
for sys_reg in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    sys_rows = valid[valid['regime'] == sys_reg]
    counts = []
    for vix_reg in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
        n = len(sys_rows[sys_rows['vix_regime'] == vix_reg])
        counts.append(n)
    total = sum(counts)
    print(f"  {sys_reg:>12s} {counts[0]:>10d} {counts[1]:>10d} {counts[2]:>10d} {counts[3]:>10d} {total:>8d}")

# 3. Per-year mismatch analysis
print(f"\n\n  MATCH RATE POR ANO:")
print(f"  {'Ano':>6s} {'Match%':>8s} {'Sys_BULL':>9s} {'Sys_NEUT':>9s} {'Sys_BEAR':>9s} {'Sys_CRIS':>9s} {'VIX avg':>8s} {'VIX_BULL':>9s} {'VIX_NEUT':>9s} {'VIX_BEAR':>9s} {'VIX_CRIS':>9s} {'SPY':>7s}")
print(f"  {'-'*110}")

for yr in sorted(df['year'].unique()):
    yr_data = df[(df['year'] == yr) & df['vix_regime'].notna()]
    if len(yr_data) == 0:
        continue
    match = yr_data['match'].mean() * 100
    # System counts
    sb = len(yr_data[yr_data['regime'] == 'BULLISH'])
    sn = len(yr_data[yr_data['regime'] == 'NEUTRAL'])
    sbe = len(yr_data[yr_data['regime'] == 'BEARISH'])
    sc = len(yr_data[yr_data['regime'] == 'CRISIS'])
    # VIX counts
    vb = len(yr_data[yr_data['vix_regime'] == 'BULLISH'])
    vn = len(yr_data[yr_data['vix_regime'] == 'NEUTRAL'])
    vbe = len(yr_data[yr_data['vix_regime'] == 'BEARISH'])
    vc = len(yr_data[yr_data['vix_regime'] == 'CRISIS'])
    # VIX avg
    vix_avg = yr_data['vix'].mean()
    spy_yr = yr_data['spy_ret'].sum()
    marker = " <<<" if match < 40 else ""
    print(f"  {yr:>6d} {match:>7.0f}% {sb:>9d} {sn:>9d} {sbe:>9d} {sc:>9d} {vix_avg:>7.1f} {vb:>9d} {vn:>9d} {vbe:>9d} {vc:>9d} {spy_yr:>+6.1f}%{marker}")

# 4. Biggest mismatches: system says BULLISH but VIX says CRISIS/BEARISH
print(f"\n\n  PEORES DISCREPANCIAS: Sistema=BULLISH, VIX>=25 (BEARISH/CRISIS)")
print(f"  {'Fecha':>12s} {'VIX':>6s} {'VIX_reg':>8s} {'BR':>5s} {'#Bull':>5s} {'#Bear':>5s} {'SPY%':>6s} {'Eventos'}")
print(f"  {'-'*120}")
mismatches = valid[(valid['regime'] == 'BULLISH') & (valid['vix'] >= 25)].sort_values('vix', ascending=False)
for _, r in mismatches.head(30).iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d'):>12s} {r['vix']:>5.1f} {r['vix_regime']:>8s} {r['br']:.2f} {r['n_bull']:>5d} {r['n_bear']:>5d} {r['spy_ret']:>+5.1f}% {r['events'][:80]}")

# 5. System says CRISIS but VIX < 20 (BULLISH)
print(f"\n\n  DISCREPANCIAS INVERSAS: Sistema=CRISIS, VIX<20 (BULLISH)")
print(f"  {'Fecha':>12s} {'VIX':>6s} {'VIX_reg':>8s} {'BR':>5s} {'#Bull':>5s} {'#Bear':>5s} {'SPY%':>6s} {'Eventos'}")
print(f"  {'-'*120}")
mismatches2 = valid[(valid['regime'] == 'CRISIS') & (valid['vix'] < 20)].sort_values('vix')
for _, r in mismatches2.head(30).iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d'):>12s} {r['vix']:>5.1f} {r['vix_regime']:>8s} {r['br']:.2f} {r['n_bull']:>5d} {r['n_bear']:>5d} {r['spy_ret']:>+5.1f}% {r['events'][:80]}")

# 6. Year 2002 detail
print(f"\n\n  DETALLE 2002 (semana a semana):")
print(f"  {'Fecha':>12s} {'Sys':>8s} {'VIX':>6s} {'VIX_reg':>8s} {'BR':>5s} {'#B':>3s} {'#S':>3s} {'SPY%':>6s} {'Match':>5s} {'Eventos'}")
print(f"  {'-'*120}")
yr2002 = df[df['year'] == 2002]
for _, r in yr2002.iterrows():
    m = 'OK' if r['match'] else 'MISS' if r['match'] is not None else '?'
    print(f"  {r['date'].strftime('%Y-%m-%d'):>12s} {r['regime']:>8s} {r['vix']:>5.1f} {r['vix_regime'] or '?':>8s} {r['br']:.2f} {r['n_bull']:>3d} {r['n_bear']:>3d} {r['spy_ret']:>+5.1f}% {m:>5s} {r['events'][:70]}")
