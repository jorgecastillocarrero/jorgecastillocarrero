"""
Test VIX como veto: solo corrige discrepancias absurdas
========================================================
- CRISIS pero VIX < 20 (normal) → BEARISH
- BULLISH pero VIX >= 25 (miedo) → NEUTRAL
El resto no se toca.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
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

def decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row):
    long_cands, short_cands = [], []
    for sub_id in SUBSECTORS:
        score = scores_evt.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        dd_factor = np.clip((abs(dd) - 15) / 30, 0, 1)
        rsi_os = np.clip((40 - rsi) / 25, 0, 1)
        oversold = max(dd_factor, rsi_os)
        if oversold > 0.1 and score >= 4.0:
            long_cands.append((sub_id, oversold))
        rsi_ob = np.clip((rsi - 65) / 20, 0, 1)
        if rsi_ob > 0.1 and dd > -8 and score <= 6.0 and atr >= ATR_MIN:
            short_cands.append((sub_id, rsi_ob))
    long_cands.sort(key=lambda x: -x[1])
    short_cands.sort(key=lambda x: -x[1])
    longs = [c[0] for c in long_cands[:3]]
    shorts = [c[0] for c in short_cands[:2]]
    weights = {}
    for s, w in long_cands[:3]: weights[s] = w
    for s, w in short_cands[:2]: weights[s] = w
    return longs, shorts, weights

def calc_pnl(longs, shorts, scores, ret_row, capital):
    n = len(longs) + len(shorts)
    if n == 0: return 0.0
    lw = {s: scores[s] - 5.0 for s in longs}
    sw = {s: 5.0 - scores[s] for s in shorts}
    tw = sum(lw.values()) + sum(sw.values())
    if tw <= 0: return 0.0
    pnl = 0.0
    for s in longs:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (lw[s] / tw) * ret_row[s]
    for s in shorts:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (sw[s] / tw) * (-ret_row[s])
    return pnl

def calc_pnl_meanrev(longs, shorts, weights, ret_row, capital):
    n = len(longs) + len(shorts)
    if n == 0: return 0.0
    lw = {s: weights.get(s, 0) for s in longs}
    sw = {s: weights.get(s, 0) for s in shorts}
    tw = sum(lw.values()) + sum(sw.values())
    if tw <= 0: return 0.0
    pnl = 0.0
    for s in longs:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (lw[s] / tw) * ret_row[s]
    for s in shorts:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (sw[s] / tw) * (-ret_row[s])
    return pnl

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
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
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

# VIX (lagged = semana anterior)
vix_raw = pd.read_csv('data/vix_weekly.csv', header=[0, 1], index_col=0)
vix_raw.index = pd.to_datetime(vix_raw.index)
vix_close = vix_raw[('Close', '^VIX')]
vix_lagged = vix_close.shift(1)

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

# ---- Backtest ----
print("Ejecutando backtest...\n")

rows = []
yearly_veto = {}
yearly_base = {}
n_veto_applied = {'CRISIS->BEARISH': 0, 'BULLISH->NEUTRAL': 0}

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
        rows.append({'date': date, 'year': date.year, 'pnl_veto': 0, 'pnl_base': 0,
                     'regime_veto': 'NO_EVT', 'regime_base': 'NO_EVT', 'vix': np.nan, 'br': 0.5,
                     'vetoed': False})
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])
    bc = len(shorts_pool)
    lc = len(longs_pool)
    br = bc / (bc + lc) if bc + lc > 0 else 0.5

    # Regimen base (solo BR, igual que el sistema actual)
    if br >= 0.60:     regime_base = 'CRISIS'
    elif br >= 0.50:   regime_base = 'BEARISH'
    elif br >= 0.30:   regime_base = 'NEUTRAL'
    else:              regime_base = 'BULLISH'

    # VIX lagged
    vix_idx = vix_lagged.index.get_indexer([date], method='nearest')[0]
    vix_date = vix_lagged.index[vix_idx]
    vix_val = vix_lagged.iloc[vix_idx] if abs((vix_date - date).days) < 8 else np.nan

    # VIX VETO (2 niveles):
    # <12 complacencia, 12-15 bull, 15-19 normal, 20-25 alerta, 25+ miedo
    regime_veto = regime_base
    vetoed = False
    if pd.notna(vix_val):
        if regime_base == 'CRISIS':
            if vix_val < 15:
                # Bull calm: crisis es absurdo → NEUTRAL
                regime_veto = 'NEUTRAL'
                vetoed = True
                n_veto_applied['CRISIS->NEUTRAL (VIX<15)'] = n_veto_applied.get('CRISIS->NEUTRAL (VIX<15)', 0) + 1
            elif vix_val < 20:
                # Normal: no es crisis pero eventos negativos → BEARISH
                regime_veto = 'BEARISH'
                vetoed = True
                n_veto_applied['CRISIS->BEARISH (VIX 15-20)'] = n_veto_applied.get('CRISIS->BEARISH (VIX 15-20)', 0) + 1
        elif regime_base == 'BULLISH':
            if vix_val >= 25:
                # Miedo: no puede ser bull → NEUTRAL
                regime_veto = 'NEUTRAL'
                vetoed = True
                n_veto_applied['BULLISH->NEUTRAL (VIX>=25)'] = n_veto_applied.get('BULLISH->NEUTRAL (VIX>=25)', 0) + 1
            elif vix_val >= 20:
                # Alerta: no es bull puro → NEUTRAL
                regime_veto = 'NEUTRAL'
                vetoed = True
                n_veto_applied['BULLISH->NEUTRAL (VIX 20-25)'] = n_veto_applied.get('BULLISH->NEUTRAL (VIX 20-25)', 0) + 1

    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None
    ret_row = returns_wide.loc[date]

    # --- PnL con VETO ---
    if regime_veto == 'NEUTRAL':
        l_v, s_v, w_v = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_veto = calc_pnl_meanrev(l_v, s_v, w_v, ret_row, CAPITAL)
    else:
        if regime_veto == 'CRISIS':
            if br >= 0.70:   nl, ns = 0, 3
            elif br >= 0.60: nl, ns = 1, 3
            elif br >= 0.55: nl, ns = 2, 3
            else:            nl, ns = 3, 3
        elif regime_veto == 'BEARISH':
            nl, ns = 2, 3
        else:  # BULLISH
            nl, ns = 3, 0
        l_v = [s for s, _ in longs_pool[:nl]]
        s_v = [s for s, _ in shorts_pool[:ns]]
        if atr_row is not None:
            s_v = [s for s in s_v if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
        pnl_veto = calc_pnl(l_v, s_v, scores_v3, ret_row, CAPITAL)

    # --- PnL BASE (sistema actual con BR thresholds 0.30/0.50/0.60) ---
    if regime_base == 'NEUTRAL':
        l_b, s_b, w_b = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_base = calc_pnl_meanrev(l_b, s_b, w_b, ret_row, CAPITAL)
    else:
        if br >= 0.70:   nl_b, ns_b = 0, 3
        elif br >= 0.60: nl_b, ns_b = 1, 3
        elif br >= 0.55: nl_b, ns_b = 2, 3
        elif br >= 0.50: nl_b, ns_b = 3, 3
        elif br < 0.30:  nl_b, ns_b = 3, 0
        else:            nl_b, ns_b = 3, 0  # shouldn't happen, NEUTRAL handled above
        l_b = [s for s, _ in longs_pool[:nl_b]]
        s_b = [s for s, _ in shorts_pool[:ns_b]]
        if atr_row is not None:
            s_b = [s for s in s_b if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
        pnl_base = calc_pnl(l_b, s_b, scores_v3, ret_row, CAPITAL)

    year = date.year
    yearly_veto[year] = yearly_veto.get(year, 0) + pnl_veto
    yearly_base[year] = yearly_base.get(year, 0) + pnl_base

    spy_idx = spy_weekly.index.get_indexer([date], method='nearest')[0]
    spy_date = spy_weekly.index[spy_idx]
    spy_ret = spy_weekly.iloc[spy_idx]['spy_ret'] if abs((spy_date - date).days) < 5 else 0

    rows.append({
        'date': date, 'year': year,
        'pnl_veto': pnl_veto, 'pnl_base': pnl_base,
        'regime_veto': regime_veto, 'regime_base': regime_base,
        'vix': vix_val, 'br': br, 'spy_ret': spy_ret,
        'vetoed': vetoed,
    })

df = pd.DataFrame(rows)

# ================================================================
# RESULTADOS
# ================================================================
print(f"{'='*130}")
print(f"  VIX VETO: CRISIS->BEARISH si VIX<20, BULLISH->NEUTRAL si VIX>=25")
print(f"{'='*130}")

total_veto = df['pnl_veto'].sum()
total_base = df['pnl_base'].sum()

active_v = df[df['pnl_veto'] != 0]['pnl_veto']
active_b = df[df['pnl_base'] != 0]['pnl_base']
sh_v = (active_v/CAPITAL).mean() / (active_v/CAPITAL).std() * np.sqrt(52)
sh_b = (active_b/CAPITAL).mean() / (active_b/CAPITAL).std() * np.sqrt(52)
wr_v = (active_v > 0).mean() * 100
wr_b = (active_b > 0).mean() * 100
cum_v = df['pnl_veto'].cumsum(); dd_v = (cum_v - cum_v.cummax()).min()
cum_b = df['pnl_base'].cumsum(); dd_b = (cum_b - cum_b.cummax()).min()

print(f"\n  Vetos aplicados: {sum(n_veto_applied.values())} semanas ({sum(n_veto_applied.values())/len(df)*100:.1f}%)")
for k, v in n_veto_applied.items():
    print(f"    {k}: {v} semanas")

print(f"\n  {'Metrica':<25s} {'Con VIX veto':>16s} {'Sin veto (base)':>16s} {'Diferencia':>14s}")
print(f"  {'-'*75}")
print(f"  {'Total PnL':<25s} ${total_veto:>+14,.0f} ${total_base:>+14,.0f} ${total_veto-total_base:>+12,.0f}")
print(f"  {'Sharpe':<25s} {sh_v:>16.2f} {sh_b:>16.2f} {sh_v-sh_b:>+14.2f}")
print(f"  {'Win Rate':<25s} {wr_v:>15.1f}% {wr_b:>15.1f}%")
print(f"  {'Max DD':<25s} ${dd_v:>+14,.0f} ${dd_b:>+14,.0f} ${dd_v-dd_b:>+12,.0f}")

# Ano a ano
print(f"\n  {'Ano':>6s} {'Con veto':>12s} {'Sin veto':>12s} {'Dif':>12s} {'SPY':>7s} {'Vetos':>6s}  Cambios regimen")
print(f"  {'-'*100}")

all_years = sorted(set(yearly_veto.keys()))
for yr in all_years:
    pv = yearly_veto.get(yr, 0)
    pb = yearly_base.get(yr, 0)
    yr_data = df[df['year'] == yr]
    spy_yr = yr_data['spy_ret'].sum()
    n_vetoed = yr_data['vetoed'].sum()

    # What changed
    changes = yr_data[yr_data['vetoed']]
    if len(changes) > 0:
        change_counts = {}
        for _, r in changes.iterrows():
            key = f"{r['regime_base']}->{r['regime_veto']}"
            change_counts[key] = change_counts.get(key, 0) + 1
        change_str = ', '.join(f"{k}:{v}" for k, v in change_counts.items())
    else:
        change_str = ""

    marker = ""
    if pv - pb > 30000: marker = " +++"
    elif pb - pv > 30000: marker = " ---"

    print(f"  {yr:>6d} ${pv:>+10,.0f} ${pb:>+10,.0f} ${pv-pb:>+10,.0f} {spy_yr:>+6.1f}% {n_vetoed:>5d}  {change_str}{marker}")

print(f"  {'-'*100}")
print(f"  {'TOTAL':>6s} ${total_veto:>+10,.0f} ${total_base:>+10,.0f} ${total_veto-total_base:>+10,.0f}")

# Por regimen
print(f"\n  DISTRIBUCION REGIMEN CON VETO:")
for reg in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    nv = len(df[df['regime_veto'] == reg])
    nb = len(df[df['regime_base'] == reg])
    rg = df[df['regime_veto'] == reg]
    pnl_sum = rg['pnl_veto'].sum()
    act = rg[rg['pnl_veto'] != 0]['pnl_veto']
    sh = (act/CAPITAL).mean()/(act/CAPITAL).std()*np.sqrt(52) if len(act)>1 and (act/CAPITAL).std()>0 else 0
    wr = (act>0).mean()*100 if len(act)>0 else 0
    print(f"    {reg:<10s}: {nv:>5d} sem (era {nb:>5d})  PnL=${pnl_sum:>+12,.0f}  WR={wr:5.1f}%  Sh={sh:.2f}")

# Detalle semanas vetoed
print(f"\n  DETALLE SEMANAS VETADAS (todas):")
vetoed_weeks = df[df['vetoed']].sort_values('date')
print(f"  {'Fecha':>12s} {'VIX':>6s} {'BR':>5s} {'Base':>8s} {'Veto':>8s} {'PnL base':>12s} {'PnL veto':>12s} {'Dif':>12s}")
print(f"  {'-'*85}")
for _, r in vetoed_weeks.iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d'):>12s} {r['vix']:>5.1f} {r['br']:.2f} {r['regime_base']:>8s} {r['regime_veto']:>8s} ${r['pnl_base']:>+10,.0f} ${r['pnl_veto']:>+10,.0f} ${r['pnl_veto']-r['pnl_base']:>+10,.0f}")
