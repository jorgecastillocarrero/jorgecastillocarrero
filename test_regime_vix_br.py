"""
Test regimen combinado: VIX + bear_ratio
==========================================
Prueba diferentes combinaciones de thresholds para definir el regimen
usando VIX (mercado real) + bear_ratio (eventos fundamentales).
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

def decide_allocation(scores, regime, max_pos=3):
    """Allocation basada en regimen (VIX+BR)."""
    longs_pool = sorted([(s, sc) for s, sc in scores.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores.items() if sc < 3.5], key=lambda x: x[1])

    if regime == 'CRISIS':    nl, ns = 1, max_pos
    elif regime == 'BEARISH': nl, ns = 2, max_pos
    elif regime == 'NEUTRAL': nl, ns = max_pos, 1
    else:                     nl, ns = max_pos, 0  # BULLISH

    return [s for s, _ in longs_pool[:nl]], [s for s, _ in shorts_pool[:ns]]

def decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row):
    long_cands = []
    short_cands = []
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

# VIX semanal
vix_raw = pd.read_csv('data/vix_weekly.csv', header=[0, 1], index_col=0)
vix_raw.index = pd.to_datetime(vix_raw.index)
vix_close = vix_raw[('Close', '^VIX')]
# VIX lagged (usamos VIX de la semana anterior, no puede ser look-ahead)
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


def classify_regime_vix_br(vix_val, bear_ratio):
    """Regimen combinado VIX + bear_ratio.
    VIX: miedo real del mercado
    BR: direccion fundamental segun eventos
    """
    if pd.isna(vix_val):
        # Fallback: solo bear_ratio
        if bear_ratio >= 0.55: return 'CRISIS'
        elif bear_ratio >= 0.45: return 'BEARISH'
        elif bear_ratio >= 0.30: return 'NEUTRAL'
        else: return 'BULLISH'

    # VIX bajo (<20): mercado tranquilo
    if vix_val < 20:
        if bear_ratio < 0.30:     return 'BULLISH'
        elif bear_ratio < 0.50:   return 'NEUTRAL'
        else:                     return 'BEARISH'

    # VIX medio (20-25): mercado nervioso
    elif vix_val < 25:
        if bear_ratio < 0.30:     return 'NEUTRAL'
        elif bear_ratio < 0.50:   return 'BEARISH'
        else:                     return 'CRISIS'

    # VIX alto (>=25): mercado con miedo
    else:
        if bear_ratio < 0.30:     return 'NEUTRAL'
        elif bear_ratio < 0.50:   return 'CRISIS'
        else:                     return 'CRISIS'


# ---- Backtest con regimen VIX+BR ----
print("Ejecutando backtest con regimen VIX+BR...\n")

rows = []
yearly_new = {}
yearly_old = {}

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

    if not active:
        rows.append({'date': date, 'year': date.year, 'pnl_new': 0, 'pnl_old': 0,
                     'regime_new': 'NO_EVT', 'regime_old': 'NO_EVT', 'vix': np.nan, 'br': 0.5})
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    # Bear ratio (from scores)
    longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])
    bc = len(shorts_pool)
    lc = len(longs_pool)
    br = bc / (bc + lc) if bc + lc > 0 else 0.5

    # VIX (lagged = semana anterior)
    vix_idx = vix_lagged.index.get_indexer([date], method='nearest')[0]
    vix_date = vix_lagged.index[vix_idx]
    vix_val = vix_lagged.iloc[vix_idx] if abs((vix_date - date).days) < 8 else np.nan

    # ATR row
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None
    ret_row = returns_wide.loc[date]

    # === NEW: Regimen VIX+BR ===
    regime_new = classify_regime_vix_br(vix_val, br)

    if regime_new == 'NEUTRAL':
        longs_new, shorts_new, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_new = calc_pnl_meanrev(longs_new, shorts_new, weights_mr, ret_row, CAPITAL)
    else:
        longs_new, shorts_new = decide_allocation(scores_v3, regime_new)
        if atr_row is not None:
            shorts_new = [s for s in shorts_new if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
        pnl_new = calc_pnl(longs_new, shorts_new, scores_v3, ret_row, CAPITAL)

    # === OLD: Regimen solo BR ===
    if br >= 0.60:     regime_old = 'CRISIS'
    elif br >= 0.50:   regime_old = 'BEARISH'
    elif br >= 0.30:   regime_old = 'NEUTRAL'
    else:              regime_old = 'BULLISH'

    if regime_old == 'NEUTRAL':
        longs_old, shorts_old, weights_mr_old = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_old = calc_pnl_meanrev(longs_old, shorts_old, weights_mr_old, ret_row, CAPITAL)
    else:
        # Old allocation logic
        if br >= 0.70:   nl_o, ns_o = 0, 3
        elif br >= 0.60: nl_o, ns_o = 1, 3
        elif br >= 0.55: nl_o, ns_o = 2, 3
        elif br >= 0.50: nl_o, ns_o = 3, 3
        else:            nl_o, ns_o = 3, 0
        longs_old = [s for s, _ in longs_pool[:nl_o]]
        shorts_old = [s for s, _ in shorts_pool[:ns_o]]
        if atr_row is not None:
            shorts_old = [s for s in shorts_old if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
        pnl_old = calc_pnl(longs_old, shorts_old, scores_v3, ret_row, CAPITAL)

    year = date.year
    yearly_new[year] = yearly_new.get(year, 0) + pnl_new
    yearly_old[year] = yearly_old.get(year, 0) + pnl_old

    # SPY
    spy_idx = spy_weekly.index.get_indexer([date], method='nearest')[0]
    spy_date = spy_weekly.index[spy_idx]
    spy_ret = spy_weekly.iloc[spy_idx]['spy_ret'] if abs((spy_date - date).days) < 5 else 0

    rows.append({
        'date': date, 'year': year,
        'pnl_new': pnl_new, 'pnl_old': pnl_old,
        'regime_new': regime_new, 'regime_old': regime_old,
        'vix': vix_val, 'br': br, 'spy_ret': spy_ret,
    })

df = pd.DataFrame(rows)

# ================================================================
# RESULTADOS
# ================================================================
print(f"{'='*130}")
print(f"  COMPARACION: Regimen VIX+BR vs Regimen solo BR")
print(f"{'='*130}")

# Totales
total_new = df['pnl_new'].sum()
total_old = df['pnl_old'].sum()

active_new = df[df['pnl_new'] != 0]['pnl_new']
active_old = df[df['pnl_old'] != 0]['pnl_old']
sh_new = (active_new/CAPITAL).mean() / (active_new/CAPITAL).std() * np.sqrt(52) if len(active_new) > 1 else 0
sh_old = (active_old/CAPITAL).mean() / (active_old/CAPITAL).std() * np.sqrt(52) if len(active_old) > 1 else 0
wr_new = (active_new > 0).mean() * 100
wr_old = (active_old > 0).mean() * 100

# Max DD
cum_new = df['pnl_new'].cumsum()
cum_old = df['pnl_old'].cumsum()
dd_new = (cum_new - cum_new.cummax()).min()
dd_old = (cum_old - cum_old.cummax()).min()

print(f"\n  {'Metrica':<25s} {'VIX+BR (nuevo)':>18s} {'Solo BR (actual)':>18s} {'Diferencia':>14s}")
print(f"  {'-'*80}")
print(f"  {'Total PnL':<25s} ${total_new:>+16,.0f} ${total_old:>+16,.0f} ${total_new-total_old:>+12,.0f}")
print(f"  {'Sharpe':<25s} {sh_new:>18.2f} {sh_old:>18.2f} {sh_new-sh_old:>+14.2f}")
print(f"  {'Win Rate':<25s} {wr_new:>17.1f}% {wr_old:>17.1f}%")
print(f"  {'Max DD':<25s} ${dd_new:>+16,.0f} ${dd_old:>+16,.0f} ${dd_new-dd_old:>+12,.0f}")

# Ano a ano
print(f"\n\n  {'Ano':>6s} {'VIX+BR':>12s} {'Solo BR':>12s} {'Dif':>12s} {'SPY':>7s}  {'NEW regimes':>50s} {'OLD regimes':>50s}")
print(f"  {'-'*155}")

all_years = sorted(set(yearly_new.keys()) | set(yearly_old.keys()))
for yr in all_years:
    pn = yearly_new.get(yr, 0)
    po = yearly_old.get(yr, 0)
    yr_data = df[df['year'] == yr]
    spy_yr = yr_data['spy_ret'].sum()

    # Regime counts
    rc_new = yr_data['regime_new'].value_counts().to_dict()
    rc_old = yr_data['regime_old'].value_counts().to_dict()
    rn_str = ' '.join(f"{r[0]}:{rc_new.get(r, 0)}" for r in ['BULLISH','NEUTRAL','BEARISH','CRISIS'])
    ro_str = ' '.join(f"{r[0]}:{rc_old.get(r, 0)}" for r in ['BULLISH','NEUTRAL','BEARISH','CRISIS'])

    marker = ""
    if pn - po > 30000: marker = " +++"
    elif po - pn > 30000: marker = " ---"

    print(f"  {yr:>6d} ${pn:>+10,.0f} ${po:>+10,.0f} ${pn-po:>+10,.0f} {spy_yr:>+6.1f}%  {rn_str:>50s} {ro_str:>50s}{marker}")

print(f"  {'-'*155}")
print(f"  {'TOTAL':>6s} ${total_new:>+10,.0f} ${total_old:>+10,.0f} ${total_new-total_old:>+10,.0f}")

# Por regimen nuevo
print(f"\n\n  ESTADISTICAS POR REGIMEN (VIX+BR):")
print(f"  {'Regimen':<10s} {'Semanas':>8s} {'PnL Total':>14s} {'Avg/sem':>10s} {'WR':>7s} {'Sharpe':>8s}")
print(f"  {'-'*65}")
for reg in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    rg = df[df['regime_new'] == reg]
    if len(rg) == 0: continue
    pnls = rg['pnl_new']
    act = pnls[pnls != 0]
    sh = (act/CAPITAL).mean() / (act/CAPITAL).std() * np.sqrt(52) if len(act) > 1 and (act/CAPITAL).std() > 0 else 0
    wr = (act > 0).mean() * 100 if len(act) > 0 else 0
    print(f"  {reg:<10s} {len(rg):>8d} ${pnls.sum():>+12,.0f} ${pnls.mean():>+8,.0f} {wr:>6.1f}% {sh:>7.2f}")

# Distribucion regimen nuevo
print(f"\n  DISTRIBUCION REGIMEN NUEVO:")
for reg in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    n = len(df[df['regime_new'] == reg])
    pct = n / len(df) * 100
    n_old = len(df[df['regime_old'] == reg])
    pct_old = n_old / len(df) * 100
    print(f"    {reg:<10s}: {n:>5d} ({pct:>5.1f}%)  vs old {n_old:>5d} ({pct_old:>5.1f}%)")

# Match con VIX
print(f"\n  MATCH VIX por regimen nuevo:")
for reg in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    rg = df[(df['regime_new'] == reg) & df['vix'].notna()]
    if len(rg) == 0: continue
    vix_avg = rg['vix'].mean()
    vix_med = rg['vix'].median()
    print(f"    {reg:<10s}: VIX avg={vix_avg:.1f}  median={vix_med:.1f}  range=[{rg['vix'].min():.0f}-{rg['vix'].max():.0f}]")
