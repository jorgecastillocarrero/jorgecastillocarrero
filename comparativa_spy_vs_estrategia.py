"""
Comparativa: SPY vs Subsectores seleccionados por la estrategia, por regimen
============================================================================
Para cada semana del backtest:
  - Retorno SPY (buy & hold)
  - Retorno de los subsectores seleccionados por la estrategia (sin costes)
  - Diferencia (alpha de la seleccion)
Agrupado por regimen de mercado.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0
ATR_MIN = 1.5

# ================================================================
# FUNCIONES (copiadas de report_compound.py)
# ================================================================
def score_fair(active_events):
    contributions = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP: continue
        for subsec, impact in EVENT_SUBSECTOR_MAP[evt_type]['impacto'].items():
            if subsec not in contributions: contributions[subsec] = []
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

def decide_burbuja_aggressive(scores_v3, dd_row, rsi_row):
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if score <= 6.0: continue
        if dd < -8: continue
        if rsi < 55: continue
        momentum_score = 0.0
        momentum_score += np.clip((score - 6.0) / 2.5, 0, 1) * 2.5
        momentum_score += np.clip((8 + dd) / 8, 0, 1) * 2.0
        momentum_score += np.clip((rsi - 55) / 25, 0, 1) * 1.5
        candidates.append((sub_id, momentum_score))
    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def decide_neutral_oversold(scores_v3, dd_row, rsi_row):
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if dd > -15 or rsi > 35: continue
        if score < 3.5: continue
        w = np.clip((abs(dd) - 15) / 20, 0, 1) + np.clip((35 - rsi) / 15, 0, 1)
        candidates.append((sub_id, w))
    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def decide_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row):
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        if score >= 4.5: continue
        if dd < -25: continue
        if rsi < 25: continue
        if atr < ATR_MIN: continue
        breakdown_score = 0.0
        breakdown_score += np.clip((5.0 - score) / 3.0, 0, 1) * 2.0
        breakdown_score += np.clip(abs(dd) / 20.0, 0, 1) * 1.5
        breakdown_score += np.clip((50 - rsi) / 25.0, 0, 1) * 1.5
        breakdown_score += np.clip((atr - ATR_MIN) / 3.0, 0, 1) * 1.0
        candidates.append((sub_id, breakdown_score))
    candidates.sort(key=lambda x: -x[1])
    shorts = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return shorts, weights

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

def classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: return 'NEUTRAL', {}
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL', {}
    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50
    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) > 0:
        spy_last = spy_w.loc[spy_dates[-1]]
        spy_above_ma200 = spy_last.get('above_ma200', 0.5)
        spy_mom_10w = spy_last.get('mom_10w', 0)
        spy_dist = spy_last.get('dist_ma200', 0)
    else:
        spy_above_ma200 = 0.5; spy_mom_10w = 0; spy_dist = 0
    if not pd.notna(spy_mom_10w): spy_mom_10w = 0
    if not pd.notna(spy_dist): spy_dist = 0
    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    if pct_dd_healthy >= 75: score_bdd = 2.0
    elif pct_dd_healthy >= 60: score_bdd = 1.0
    elif pct_dd_healthy >= 45: score_bdd = 0.0
    elif pct_dd_healthy >= 30: score_bdd = -1.0
    elif pct_dd_healthy >= 15: score_bdd = -2.0
    else: score_bdd = -3.0
    if pct_rsi_above55 >= 75: score_brsi = 2.0
    elif pct_rsi_above55 >= 60: score_brsi = 1.0
    elif pct_rsi_above55 >= 45: score_brsi = 0.0
    elif pct_rsi_above55 >= 30: score_brsi = -1.0
    elif pct_rsi_above55 >= 15: score_brsi = -2.0
    else: score_brsi = -3.0
    if pct_dd_deep <= 5: score_ddp = 1.5
    elif pct_dd_deep <= 15: score_ddp = 0.5
    elif pct_dd_deep <= 30: score_ddp = -0.5
    elif pct_dd_deep <= 50: score_ddp = -1.5
    else: score_ddp = -2.5
    if spy_above_ma200 and spy_dist > 5: score_spy = 1.5
    elif spy_above_ma200: score_spy = 0.5
    elif spy_dist > -5: score_spy = -0.5
    elif spy_dist > -15: score_spy = -1.5
    else: score_spy = -2.5
    if spy_mom_10w > 5: score_mom = 1.0
    elif spy_mom_10w > 0: score_mom = 0.5
    elif spy_mom_10w > -5: score_mom = -0.5
    elif spy_mom_10w > -15: score_mom = -1.0
    else: score_mom = -1.5
    total = score_bdd + score_brsi + score_ddp + score_spy + score_mom
    is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'CAUTIOUS'
    return regime, {'score_total': total}

# ================================================================
# CARGAR DATOS
# ================================================================
print("Cargando datos...")

ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

df_all = pd.read_sql(f"""
    SELECT symbol, date, open, close, high, low
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
    lambda x: x.rolling(5, min_periods=3).mean() * 100)

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean')).reset_index()
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

# SPY
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100
spy_w['ret_spy'] = spy_w['close'].pct_change()

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

weekly_events = build_weekly_events('2000-01-01', '2026-02-28')

# ================================================================
# BACKTEST SEMANAL
# ================================================================
print("Calculando retornos semanales...")

weekly_data = []

for date in returns_wide.index:
    if date.year < 2001:
        continue

    # SPY return for this week
    spy_ret = spy_w.loc[date, 'ret_spy'] if date in spy_w.index else np.nan
    if not pd.notna(spy_ret):
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
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    ret_row = returns_wide.loc[date]
    regime, _ = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    # Determine strategy: longs and shorts
    if regime == 'BURBUJA':
        longs_sel, weights = decide_burbuja_aggressive(scores_v3, dd_row, rsi_row)
        shorts_sel = []
        if longs_sel:
            strat_ret = calc_pnl_meanrev(longs_sel, [], weights, ret_row, 1.0)
        else:
            longs_sel = []
            strat_ret = 0.0
    elif regime == 'GOLDILOCKS':
        top3 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
        longs_sel = [s for s, _ in top3]
        shorts_sel = []
        strat_ret = calc_pnl(longs_sel, [], scores_v3, ret_row, 1.0)
    elif regime == 'ALCISTA':
        top3 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
        longs_sel = [s for s, _ in top3]
        shorts_sel = []
        strat_ret = calc_pnl(longs_sel, [], scores_v3, ret_row, 1.0)
    elif regime in ('NEUTRAL', 'CAUTIOUS'):
        longs_sel, weights = decide_neutral_oversold(scores_v3, dd_row, rsi_row)
        shorts_sel = []
        if longs_sel:
            strat_ret = calc_pnl_meanrev(longs_sel, [], weights, ret_row, 1.0)
        else:
            strat_ret = 0.0
    elif regime in ('BEARISH', 'CRISIS', 'PANICO'):
        shorts_sel, weights = decide_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row)
        longs_sel = []
        if shorts_sel:
            strat_ret = calc_pnl_meanrev([], shorts_sel, weights, ret_row, 1.0)
        else:
            strat_ret = 0.0
    else:
        longs_sel = []
        shorts_sel = []
        strat_ret = 0.0

    n_pos = len(longs_sel) + len(shorts_sel)

    # Subsectores seleccionados
    selected_names = []
    for s in longs_sel:
        selected_names.append(f"{s}(L)")
    for s in shorts_sel:
        selected_names.append(f"{s}(S)")

    # Retorno medio de los subsectores seleccionados (sin ponderacion FV, puro equal-weight)
    if longs_sel or shorts_sel:
        rets_selected = []
        for s in longs_sel:
            r = ret_row.get(s)
            if pd.notna(r):
                rets_selected.append(r)
        for s in shorts_sel:
            r = ret_row.get(s)
            if pd.notna(r):
                rets_selected.append(-r)  # short = -return
        avg_selected_ew = np.mean(rets_selected) if rets_selected else 0.0
    else:
        avg_selected_ew = 0.0

    weekly_data.append({
        'date': date,
        'year': date.year,
        'regime': regime,
        'spy_ret': spy_ret,
        'strat_ret': strat_ret,  # FV-weighted
        'strat_ew_ret': avg_selected_ew,  # Equal-weight
        'n_pos': n_pos,
        'n_longs': len(longs_sel),
        'n_shorts': len(shorts_sel),
        'selected': ', '.join(selected_names),
    })

df = pd.DataFrame(weekly_data)

# ================================================================
# TABLA COMPARATIVA POR REGIMEN
# ================================================================
print("\n" + "=" * 140)
print("  COMPARATIVA: SPY vs ESTRATEGIA SEMANAL_SUBSECTORES POR REGIMEN")
print("  Retornos semanales brutos (sin costes), periodo 2001-2025")
print("=" * 140)

regime_order = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO']

# Tabla 1: SPY vs Estrategia (media, mediana, std)
print(f"\n{'1. RETORNO MEDIO SEMANAL':>35}")
print("-" * 130)
print(f"  {'Regimen':<12} {'N':>5} {'Dir':>5}  |  {'SPY avg':>8} {'SPY med':>8} {'SPY std':>8} {'SPY WR':>7}  |  "
      f"{'Estr avg':>8} {'Estr med':>9} {'Estr std':>9} {'Estr WR':>8}  |  {'Alpha':>7}")
print("-" * 130)

totals_spy = []
totals_strat = []

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]
    n = len(sub)

    # Direccion
    if reg in ('BEARISH', 'CRISIS', 'PANICO'):
        direction = 'SHORT'
    elif reg in ('NEUTRAL', 'CAUTIOUS'):
        direction = 'OVSLD'
    else:
        direction = 'LONG'

    # SPY stats
    spy_avg = sub['spy_ret'].mean() * 100
    spy_med = sub['spy_ret'].median() * 100
    spy_std = sub['spy_ret'].std() * 100
    spy_wr = (sub['spy_ret'] > 0).mean() * 100

    # Strategy stats
    strat_avg = sub['strat_ret'].mean() * 100
    strat_med = sub['strat_ret'].median() * 100
    strat_std = sub['strat_ret'].std() * 100
    strat_wr = (sub['strat_ret'] > 0).mean() * 100

    alpha = strat_avg - spy_avg

    totals_spy.extend(sub['spy_ret'].tolist())
    totals_strat.extend(sub['strat_ret'].tolist())

    marker = " <<" if abs(alpha) > 0.15 else ""
    print(f"  {reg:<12} {n:>5} {direction:>5}  |  {spy_avg:>+7.2f}% {spy_med:>+7.2f}% {spy_std:>7.2f}% {spy_wr:>6.1f}%  |  "
          f"{strat_avg:>+7.2f}% {strat_med:>+8.2f}% {strat_std:>8.2f}% {strat_wr:>7.1f}%  |  {alpha:>+6.2f}%{marker}")

# Total
spy_total_avg = np.mean(totals_spy) * 100
strat_total_avg = np.mean(totals_strat) * 100
print("-" * 130)
print(f"  {'TOTAL':<12} {len(df):>5} {'':>5}  |  {spy_total_avg:>+7.2f}% {'':>8} {'':>8} {'':>7}  |  "
      f"{strat_total_avg:>+7.2f}% {'':>9} {'':>9} {'':>8}  |  {strat_total_avg - spy_total_avg:>+6.2f}%")

# Tabla 2: Desglose LONGS vs SHORTS vs SPY
print(f"\n\n{'2. DESGLOSE POR DIRECCION: retorno semanal medio':>50}")
print("-" * 110)
print(f"  {'Regimen':<12} {'N':>5}  |  {'SPY avg%':>8}  |  {'L avg%':>8} {'L WR%':>7}  |  {'S avg%':>8} {'S WR%':>7}  |  "
      f"{'Total%':>8}  |  {'vs SPY':>7}")
print("-" * 110)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]
    n = len(sub)

    spy_avg = sub['spy_ret'].mean() * 100
    strat_avg = sub['strat_ret'].mean() * 100

    # Longs return (from ret_row for selected longs only)
    # We need to recalculate from the weekly_data, but we stored strat_ret as combined
    # Let's use n_longs/n_shorts to infer direction
    has_longs = sub['n_longs'] > 0
    has_shorts = sub['n_shorts'] > 0

    if has_longs.any():
        # We can approximate: when n_longs > 0 and n_shorts == 0, strat_ret = long return
        long_only = sub[(sub['n_longs'] > 0) & (sub['n_shorts'] == 0)]
        l_avg = long_only['strat_ret'].mean() * 100 if len(long_only) > 0 else 0
        l_wr = (long_only['strat_ret'] > 0).mean() * 100 if len(long_only) > 0 else 0
    else:
        l_avg = 0
        l_wr = 0

    if has_shorts.any():
        short_only = sub[(sub['n_shorts'] > 0) & (sub['n_longs'] == 0)]
        s_avg = short_only['strat_ret'].mean() * 100 if len(short_only) > 0 else 0
        s_wr = (short_only['strat_ret'] > 0).mean() * 100 if len(short_only) > 0 else 0
    else:
        s_avg = 0
        s_wr = 0

    l_str = f"{l_avg:>+7.2f}% {l_wr:>6.1f}%" if has_longs.any() else f"{'---':>8} {'---':>7}"
    s_str = f"{s_avg:>+7.2f}% {s_wr:>6.1f}%" if has_shorts.any() else f"{'---':>8} {'---':>7}"

    alpha = strat_avg - spy_avg
    print(f"  {reg:<12} {n:>5}  |  {spy_avg:>+7.2f}%  |  {l_str}  |  {s_str}  |  "
          f"{strat_avg:>+7.2f}%  |  {alpha:>+6.2f}%")


# Tabla 3: Sharpe semanal por regimen
print(f"\n\n{'3. EFICIENCIA (Sharpe semanal) SPY vs Estrategia':>50}")
print("-" * 90)
print(f"  {'Regimen':<12} {'N':>5}  |  {'SPY Sharpe':>11} {'SPY Best':>9} {'SPY Worst':>10}  |  "
      f"{'Est Sharpe':>11} {'Est Best':>9} {'Est Worst':>10}")
print("-" * 90)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]

    spy_vals = sub['spy_ret'].values
    strat_vals = sub['strat_ret'].values

    spy_sharpe = spy_vals.mean() / spy_vals.std() * np.sqrt(52) if spy_vals.std() > 0 else 0
    strat_sharpe = strat_vals.mean() / strat_vals.std() * np.sqrt(52) if strat_vals.std() > 0 else 0

    spy_best = spy_vals.max() * 100
    spy_worst = spy_vals.min() * 100
    strat_best = strat_vals.max() * 100
    strat_worst = strat_vals.min() * 100

    winner = "EST" if strat_sharpe > spy_sharpe else "SPY"
    print(f"  {reg:<12} {mask.sum():>5}  |  {spy_sharpe:>+10.2f} {spy_best:>+8.1f}% {spy_worst:>+9.1f}%  |  "
          f"{strat_sharpe:>+10.2f} {strat_best:>+8.1f}% {strat_worst:>+9.1f}%  [{winner}]")


# Tabla 4: Subsectores mas frecuentes por regimen
print(f"\n\n{'4. SUBSECTORES MAS SELECCIONADOS POR REGIMEN':>50}")
print("-" * 100)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]

    # Count subsector frequency
    from collections import Counter
    subsec_counts = Counter()
    for _, row in sub.iterrows():
        if row['selected']:
            for item in row['selected'].split(', '):
                if item:
                    subsec_counts[item] += 1

    if not subsec_counts:
        no_op = (sub['n_pos'] == 0).sum()
        print(f"\n  {reg} ({mask.sum()} sem): Sin seleccion en {no_op} semanas")
        continue

    top_subs = subsec_counts.most_common(8)
    total_weeks = mask.sum()
    print(f"\n  {reg} ({total_weeks} sem):")
    for sub_name, count in top_subs:
        pct = count / total_weeks * 100
        bar = '#' * int(pct / 2)
        print(f"    {sub_name:<30s}: {count:>4} ({pct:>5.1f}%) {bar}")

    # Semanas sin operacion
    no_op = (sub['n_pos'] == 0).sum()
    if no_op > 0:
        print(f"    {'(sin operar)':<30s}: {no_op:>4} ({no_op/total_weeks*100:>5.1f}%)")


# Tabla 5: Correlacion SPY vs Estrategia por regimen
print(f"\n\n{'5. CORRELACION SPY vs ESTRATEGIA':>40}")
print("-" * 70)
print(f"  {'Regimen':<12} {'N':>5}  |  {'Corr':>6}  |  {'Cuando SPY-':>12} {'Est avg':>8}  |  {'Cuando SPY+':>12} {'Est avg':>8}")
print("-" * 70)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() < 5:
        continue
    sub = df[mask]

    corr = sub['spy_ret'].corr(sub['strat_ret'])

    spy_down = sub[sub['spy_ret'] < 0]
    spy_up = sub[sub['spy_ret'] >= 0]

    strat_when_spy_down = spy_down['strat_ret'].mean() * 100 if len(spy_down) > 0 else 0
    strat_when_spy_up = spy_up['strat_ret'].mean() * 100 if len(spy_up) > 0 else 0

    print(f"  {reg:<12} {mask.sum():>5}  |  {corr:>+5.2f}  |  {len(spy_down):>5} sem    {strat_when_spy_down:>+7.2f}%  |  "
          f"{len(spy_up):>5} sem    {strat_when_spy_up:>+7.2f}%")


# Tabla 6: Impacto anualizado
print(f"\n\n{'6. IMPACTO ANUALIZADO (base $500K)':>40}")
print("-" * 100)
print(f"  {'Regimen':<12} {'N/ano':>6}  |  {'SPY $/sem':>10} {'SPY $/ano':>11}  |  {'Est $/sem':>10} {'Est $/ano':>11}  |  {'Alpha $/ano':>12}")
print("-" * 100)

BASE = 500_000
n_years = 25  # 2001-2025

total_spy_yr = 0
total_strat_yr = 0

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]

    spy_per_week = sub['spy_ret'].mean() * BASE
    strat_per_week = sub['strat_ret'].mean() * BASE

    weeks_per_year = mask.sum() / n_years
    spy_per_year = spy_per_week * weeks_per_year
    strat_per_year = strat_per_week * weeks_per_year
    alpha_year = strat_per_year - spy_per_year

    total_spy_yr += spy_per_year
    total_strat_yr += strat_per_year

    print(f"  {reg:<12} {weeks_per_year:>5.1f}  |  ${spy_per_week:>9,.0f} ${spy_per_year:>10,.0f}  |  "
          f"${strat_per_week:>9,.0f} ${strat_per_year:>10,.0f}  |  ${alpha_year:>+11,.0f}")

print("-" * 100)
print(f"  {'TOTAL':<12} {len(df)/n_years:>5.1f}  |  {'':>10} ${total_spy_yr:>10,.0f}  |  "
      f"{'':>10} ${total_strat_yr:>10,.0f}  |  ${total_strat_yr - total_spy_yr:>+11,.0f}")

print(f"\n  Nota: SPY $/ano = retorno semanal medio * semanas/ano por regimen * $500K")
print(f"  La estrategia opera con subsectores FV-weighted, no con SPY directamente")
