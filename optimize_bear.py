"""
Optimizador de configuracion BEAR_LEVE y BEAR_MODERADO
Pre-computa datos una vez, luego testea matriz de (nl, ns) para cada bear regime
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0
CAPITAL_INICIAL = 500_000
ATR_MIN = 1.5
COST_PER_TRADE = 0.0010

# ================================================================
# FUNCIONES (copiadas de report_compound.py)
# ================================================================
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

def classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        return 'NEUTRAL', {}
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0:
        return 'NEUTRAL', {}
    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above50 = (rsi_row > 50).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50
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
    else: score_bdd = -2.0
    if pct_rsi_above50 >= 75: score_brsi = 2.0
    elif pct_rsi_above50 >= 60: score_brsi = 1.0
    elif pct_rsi_above50 >= 45: score_brsi = 0.0
    elif pct_rsi_above50 >= 30: score_brsi = -1.0
    else: score_brsi = -2.0
    if pct_dd_deep <= 5: score_ddp = 1.5
    elif pct_dd_deep <= 15: score_ddp = 0.5
    elif pct_dd_deep <= 30: score_ddp = -0.5
    else: score_ddp = -1.5
    if spy_above_ma200 and spy_dist > 5: score_spy = 1.5
    elif spy_above_ma200: score_spy = 0.5
    elif spy_dist > -5: score_spy = -0.5
    else: score_spy = -1.5
    if spy_mom_10w > 5: score_mom = 1.0
    elif spy_mom_10w > 0: score_mom = 0.5
    elif spy_mom_10w > -5: score_mom = -0.5
    else: score_mom = -1.0

    total = score_bdd + score_brsi + score_ddp + score_spy + score_mom
    if total >= 4.0: regime = 'BULLISH'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -1.5: regime = 'BEAR_LEVE'
    elif total >= -3.0: regime = 'BEAR_MODERADO'
    else: regime = 'CRISIS'
    if vix_val >= 30 and regime == 'BULLISH': regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL': regime = 'BEAR_LEVE'
    return regime, {'score_total': total}

# ================================================================
# CARGAR DATOS (una sola vez)
# ================================================================
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

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# ================================================================
# PRE-COMPUTAR DATOS POR SEMANA (regimen, pools, scores, returns)
# ================================================================
print("Pre-computando semanas...")

week_data = []
for date in returns_wide.index:
    if date.year < 2001:
        continue

    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    regime, _ = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    if not active:
        week_data.append({'date': date, 'regime': regime, 'active': False,
                          'longs_pool': [], 'shorts_pool': [], 'scores_v3': {},
                          'ret_row': None, 'atr_row': None})
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None
    ret_row = returns_wide.loc[date]

    longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])

    # Filtrar shorts por ATR
    if atr_row is not None:
        shorts_pool = [(s, sc) for s, sc in shorts_pool if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    week_data.append({
        'date': date, 'regime': regime, 'active': True,
        'longs_pool': longs_pool, 'shorts_pool': shorts_pool,
        'scores_v3': scores_v3, 'ret_row': ret_row, 'atr_row': atr_row,
    })

print(f"  {len(week_data)} semanas pre-computadas")

# Contar semanas bear
n_bl = sum(1 for w in week_data if w['regime'] == 'BEAR_LEVE')
n_bm = sum(1 for w in week_data if w['regime'] == 'BEAR_MODERADO')
print(f"  BEAR_LEVE: {n_bl} semanas, BEAR_MODERADO: {n_bm} semanas")

# ================================================================
# FUNCION: replay con config bear
# ================================================================
def replay_config(bear_leve_cfg, bear_mod_cfg):
    """Replay todas las semanas con config bear especifica.
    Config = (nl, ns) para cada bear regime.
    Retorna: (capital_final, sharpe, max_dd, avg_ret_bl, avg_ret_bm, wr_bl, wr_bm)
    """
    bl_nl, bl_ns = bear_leve_cfg
    bm_nl, bm_ns = bear_mod_cfg

    rets_all = []
    rets_bl = []
    rets_bm = []

    for w in week_data:
        if not w['active']:
            rets_all.append(0.0)
            continue

        regime = w['regime']
        longs_pool = w['longs_pool']
        shorts_pool = w['shorts_pool']
        scores_v3 = w['scores_v3']
        ret_row = w['ret_row']

        # Asignar config segun regimen
        if regime == 'CRISIS':         nl, ns = 0, 3
        elif regime == 'BEAR_MODERADO': nl, ns = bm_nl, bm_ns
        elif regime == 'BEAR_LEVE':     nl, ns = bl_nl, bl_ns
        elif regime == 'NEUTRAL':       nl, ns = 3, 3
        else:                           nl, ns = 3, 0  # BULLISH

        longs = [s for s, _ in longs_pool[:nl]]
        shorts = [s for s, _ in shorts_pool[:ns]]

        pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)
        n_pos = len(longs) + len(shorts)
        cost = COST_PER_TRADE * 2 if n_pos > 0 else 0
        ret_net = pnl_unit - cost

        rets_all.append(ret_net)
        if regime == 'BEAR_LEVE':
            rets_bl.append(ret_net)
        elif regime == 'BEAR_MODERADO':
            rets_bm.append(ret_net)

    rets_arr = np.array(rets_all)

    # Compound anual
    capital = CAPITAL_INICIAL
    years = sorted(set(w['date'].year for w in week_data))
    for year in years:
        yr_rets = [rets_all[i] for i, w in enumerate(week_data) if w['date'].year == year]
        pnl_yr = sum(capital * r for r in yr_rets)
        capital += pnl_yr

    # Sharpe
    sharpe = rets_arr.mean() / rets_arr.std() * np.sqrt(52) if rets_arr.std() > 0 else 0

    # Max DD
    equity = CAPITAL_INICIAL * np.cumprod(1 + rets_arr)
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min() * 100

    # Bear stats
    bl_arr = np.array(rets_bl) if rets_bl else np.array([0.0])
    bm_arr = np.array(rets_bm) if rets_bm else np.array([0.0])

    avg_bl = bl_arr.mean() * 100
    avg_bm = bm_arr.mean() * 100
    wr_bl = (bl_arr > 0).mean() * 100 if len(rets_bl) > 0 else 0
    wr_bm = (bm_arr > 0).mean() * 100 if len(rets_bm) > 0 else 0
    avg_bl_dollar = bl_arr.mean() * CAPITAL_INICIAL
    avg_bm_dollar = bm_arr.mean() * CAPITAL_INICIAL

    return {
        'capital': capital,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'avg_bl': avg_bl, 'avg_bm': avg_bm,
        'wr_bl': wr_bl, 'wr_bm': wr_bm,
        'avg_bl_$': avg_bl_dollar, 'avg_bm_$': avg_bm_dollar,
    }

# ================================================================
# TESTEAR MATRIZ DE CONFIGURACIONES
# ================================================================
print("\nTesteando configuraciones bear...\n")

# Configs a testear: (nl, ns)
configs = [
    (0, 0),  # sit out
    (0, 1),  # solo 1 short
    (0, 2),  # solo 2 shorts
    (0, 3),  # solo 3 shorts (= CRISIS)
    (1, 1),
    (1, 2),
    (1, 3),  # actual BEAR_MOD
    (2, 2),
    (2, 3),  # actual BEAR_LEVE
    (3, 3),  # = como NEUTRAL (sin meanrev)
]

# Primero: optimizar BEAR_LEVE con BEAR_MOD fijo en (1,3)
print("=" * 100)
print("OPTIMIZACION BEAR_LEVE (BEAR_MODERADO fijo = 1L+3S)")
print("=" * 100)
print(f"  {'Config':<10} {'Cap.Final':>14} {'Sharpe':>8} {'MaxDD':>8} {'Avg%/sem':>10} {'$/sem':>10} {'WR%':>6}")
print("-" * 70)

best_bl = None
best_bl_capital = 0
for cfg in configs:
    r = replay_config(bear_leve_cfg=cfg, bear_mod_cfg=(1, 3))
    label = f"{cfg[0]}L+{cfg[1]}S"
    marker = " <-- actual" if cfg == (2, 3) else ""
    if r['capital'] > best_bl_capital:
        best_bl_capital = r['capital']
        best_bl = cfg
    print(f"  {label:<10} ${r['capital']:>13,.0f} {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}% "
          f"{r['avg_bl']:>+9.3f}% ${r['avg_bl_$']:>8,.0f} {r['wr_bl']:>5.1f}%{marker}")

print(f"\n  >> Mejor BEAR_LEVE: {best_bl[0]}L+{best_bl[1]}S (${best_bl_capital:,.0f})")

# Segundo: optimizar BEAR_MODERADO con BEAR_LEVE en su mejor config
print(f"\n{'=' * 100}")
print(f"OPTIMIZACION BEAR_MODERADO (BEAR_LEVE fijo = {best_bl[0]}L+{best_bl[1]}S)")
print("=" * 100)
print(f"  {'Config':<10} {'Cap.Final':>14} {'Sharpe':>8} {'MaxDD':>8} {'Avg%/sem':>10} {'$/sem':>10} {'WR%':>6}")
print("-" * 70)

best_bm = None
best_bm_capital = 0
for cfg in configs:
    r = replay_config(bear_leve_cfg=best_bl, bear_mod_cfg=cfg)
    label = f"{cfg[0]}L+{cfg[1]}S"
    marker = " <-- actual" if cfg == (1, 3) else ""
    if r['capital'] > best_bm_capital:
        best_bm_capital = r['capital']
        best_bm = cfg
    print(f"  {label:<10} ${r['capital']:>13,.0f} {r['sharpe']:>7.2f} {r['max_dd']:>7.1f}% "
          f"{r['avg_bm']:>+9.3f}% ${r['avg_bm_$']:>8,.0f} {r['wr_bm']:>5.1f}%{marker}")

print(f"\n  >> Mejor BEAR_MODERADO: {best_bm[0]}L+{best_bm[1]}S (${best_bm_capital:,.0f})")

# Tercero: comparacion ANTES vs DESPUES
print(f"\n{'=' * 100}")
print("COMPARACION: ANTES vs OPTIMIZADO")
print("=" * 100)

r_antes = replay_config(bear_leve_cfg=(2, 3), bear_mod_cfg=(1, 3))
r_optim = replay_config(bear_leve_cfg=best_bl, bear_mod_cfg=best_bm)

print(f"\n  {'':>20} {'ANTES':>15} {'OPTIMIZADO':>15} {'Diferencia':>15}")
print(f"  {'BEAR_LEVE config':>20} {'2L+3S':>15} {f'{best_bl[0]}L+{best_bl[1]}S':>15}")
print(f"  {'BEAR_MOD config':>20} {'1L+3S':>15} {f'{best_bm[0]}L+{best_bm[1]}S':>15}")
print(f"  {'Capital Final':>20} ${r_antes['capital']:>14,.0f} ${r_optim['capital']:>14,.0f} ${r_optim['capital']-r_antes['capital']:>+14,.0f}")
print(f"  {'Sharpe':>20} {r_antes['sharpe']:>14.2f} {r_optim['sharpe']:>14.2f} {r_optim['sharpe']-r_antes['sharpe']:>+14.2f}")
print(f"  {'Max Drawdown':>20} {r_antes['max_dd']:>13.1f}% {r_optim['max_dd']:>13.1f}% {r_optim['max_dd']-r_antes['max_dd']:>+13.1f}%")
print(f"  {'BEAR_LEVE avg/sem':>20} ${r_antes['avg_bl_$']:>13,.0f} ${r_optim['avg_bl_$']:>13,.0f}")
print(f"  {'BEAR_MOD avg/sem':>20} ${r_antes['avg_bm_$']:>13,.0f} ${r_optim['avg_bm_$']:>13,.0f}")
