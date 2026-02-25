"""
Reporte con reinversion anual de beneficios + costes conservadores
Capital inicial: $500,000 - se reinvierte todo cada ano
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
COST_PER_TRADE = 0.0010  # 0.10% por trade

# ================================================================
# FUNCIONES (mismas)
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
    weights = {s: w for s, w in long_cands[:3]}
    weights.update({s: w for s, w in short_cands[:2]})
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
        spy_above_ma200 = 0.5
        spy_mom_10w = 0
        spy_dist = 0
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
    elif total >= -1.5: regime = 'BEAR_LEVE'      # -1.5 a 0.5: bearish suave
    elif total >= -3.0: regime = 'BEAR_MODERADO'   # -3.0 a -1.5: bearish fuerte
    else: regime = 'CRISIS'                        # < -3.0: crisis

    # VIX como alerta: calm -> less stable
    if vix_val >= 30 and regime == 'BULLISH':
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'BEAR_LEVE'

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
# PASO 1: Calcular retornos semanales % (con capital fijo para obtener %)
# ================================================================
print("Calculando retornos semanales...")

weekly_returns = []  # lista de (date, ret_pct, regime)

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

    if not active:
        weekly_returns.append({'date': date, 'ret_gross': 0, 'cost_pct': 0, 'regime': 'NEUTRAL', 'n_pos': 0})
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None
    ret_row = returns_wide.loc[date]

    regime, _ = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])

    if regime == 'CRISIS':         nl, ns = 0, 3   # solo shorts
    elif regime == 'BEAR_MODERADO': nl, ns = 1, 3   # defensivo fuerte
    elif regime == 'BEAR_LEVE':     nl, ns = 2, 3   # defensivo suave
    elif regime == 'NEUTRAL':       nl, ns = 3, 3   # mean reversion
    else:                           nl, ns = 3, 0   # BULLISH solo longs

    longs = [s for s, _ in longs_pool[:nl]]
    shorts = [s for s, _ in shorts_pool[:ns]]

    if atr_row is not None:
        shorts = [s for s in shorts if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    # Calcular con capital unitario ($1) para obtener retorno %
    if regime == 'NEUTRAL':
        longs, shorts, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_unit = calc_pnl_meanrev(longs, shorts, weights_mr, ret_row, 1.0)
    else:
        pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)

    n_pos = len(longs) + len(shorts)
    cost_pct = COST_PER_TRADE * 2 if n_pos > 0 else 0  # 0.20% round-trip

    weekly_returns.append({
        'date': date,
        'ret_gross': pnl_unit,      # retorno % bruto (como fraccion de 1)
        'cost_pct': cost_pct,        # coste % (como fraccion de 1)
        'regime': regime,
        'n_pos': n_pos,
    })

df_ret = pd.DataFrame(weekly_returns)
df_ret['ret_net'] = df_ret['ret_gross'] - df_ret['cost_pct']

# SPY annual (compuesto)
spy_annual = {}
spy_compound = {}
spy_capital = CAPITAL_INICIAL
for year in range(2001, 2026):
    spy_yr = spy_w[(spy_w.index.year == year) & spy_w['ret_spy'].notna()]
    if len(spy_yr) > 0:
        spy_yr_ret = (1 + spy_yr['ret_spy']).prod() - 1
        spy_annual[year] = spy_yr_ret * 100
        spy_capital = spy_capital * (1 + spy_yr_ret)
        spy_compound[year] = spy_capital

# ================================================================
# PASO 2: Compounding anual
# ================================================================
print("Calculando compounding anual...")

capital = CAPITAL_INICIAL
results = []

for year in sorted(df_ret['date'].dt.year.unique()):
    yr = df_ret[df_ret['date'].dt.year == year]

    # Capital al inicio del ano
    capital_inicio = capital

    # P&L del ano con el capital actual
    pnl_gross_yr = 0
    pnl_net_yr = 0
    for _, row in yr.iterrows():
        pnl_gross_yr += capital * row['ret_gross']
        pnl_net_yr += capital * row['ret_net']

    # Rentabilidad neta del ano
    rent_net = pnl_net_yr / capital * 100 if capital > 0 else 0
    rent_gross = pnl_gross_yr / capital * 100 if capital > 0 else 0
    cost_yr = pnl_gross_yr - pnl_net_yr

    # Capital al final del ano (reinvertir)
    capital = capital + pnl_net_yr

    # Regimenes
    rc = yr['regime'].value_counts()

    results.append({
        'year': year,
        'capital_inicio': capital_inicio,
        'pnl_gross': pnl_gross_yr,
        'pnl_net': pnl_net_yr,
        'cost': cost_yr,
        'rent_gross': rent_gross,
        'rent_net': rent_net,
        'capital_fin': capital,
        'spy_ret': spy_annual.get(year, 0),
        'bull': rc.get('BULLISH', 0),
        'neut': rc.get('NEUTRAL', 0),
        'bear_l': rc.get('BEAR_LEVE', 0),
        'bear_m': rc.get('BEAR_MODERADO', 0),
        'cris': rc.get('CRISIS', 0),
    })

df_r = pd.DataFrame(results)

# SPY compound
spy_cap = CAPITAL_INICIAL
spy_caps = []
for _, row in df_r.iterrows():
    spy_ret = spy_annual.get(row['year'], 0) / 100
    spy_cap = spy_cap * (1 + spy_ret)
    spy_caps.append(spy_cap)
df_r['spy_capital'] = spy_caps

# ================================================================
# REPORTE
# ================================================================
print("\n" + "=" * 120)
print(f"SISTEMA MARKET CON REINVERSION ANUAL - Capital inicial: ${CAPITAL_INICIAL:,.0f}")
print(f"Costes: {COST_PER_TRADE*100:.2f}% por trade (conservador)")
print("=" * 120)

print(f"\n{'Ano':>5} {'Cap.Inicio':>12} {'S&P500':>8} {'Sist%':>8} {'Alpha':>8} "
      f"{'P&L Neto':>12} {'Costes':>10} {'Cap.Final':>14}  "
      f"{'BULL':>5} {'NEUT':>5} {'BR_L':>5} {'BR_M':>5} {'CRIS':>5}  {'SPY Cap':>14}")
print("-" * 145)

for _, row in df_r.iterrows():
    alpha = row['rent_net'] - row['spy_ret']
    print(f"{int(row['year']):>5} ${row['capital_inicio']:>11,.0f} {row['spy_ret']:>7.1f}% "
          f"{row['rent_net']:>7.1f}% {alpha:>+7.1f}% "
          f"${row['pnl_net']:>11,.0f} ${row['cost']:>9,.0f} ${row['capital_fin']:>13,.0f}  "
          f"{int(row['bull']):>5} {int(row['neut']):>5} {int(row['bear_l']):>5} {int(row['bear_m']):>5} {int(row['cris']):>5}  "
          f"${row['spy_capital']:>13,.0f}")

# Resumen final
capital_final = df_r['capital_fin'].iloc[-1]
spy_final = df_r['spy_capital'].iloc[-1]
n_years = len(df_r)
cagr_sys = (capital_final / CAPITAL_INICIAL) ** (1 / n_years) - 1
cagr_spy = (spy_final / CAPITAL_INICIAL) ** (1 / n_years) - 1
total_costs = df_r['cost'].sum()
multiple_sys = capital_final / CAPITAL_INICIAL
multiple_spy = spy_final / CAPITAL_INICIAL

print("-" * 145)

# Estadisticas por regimen
print(f"\n{'Regimen':<16} {'Semanas':>7} {'Avg ret%':>10} {'WinRate':>8} {'Config':>8}")
print("-" * 55)
for reg, cfg in [('BULLISH', '3L+0S'), ('NEUTRAL', '3L+3S*'), ('BEAR_LEVE', '2L+3S'), ('BEAR_MODERADO', '1L+3S'), ('CRISIS', '0L+3S')]:
    mask = df_ret['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df_ret[mask]
    avg_ret = sub['ret_net'].mean() * 100
    wr = (sub['ret_net'] > 0).mean() * 100
    print(f"{reg:<16} {mask.sum():>7} {avg_ret:>9.2f}% {wr:>7.1f}% {cfg:>8}")

print(f"\n{'RESUMEN FINAL':>20}")
print(f"  {'':>20} {'SISTEMA':>15} {'S&P 500':>15} {'Diferencia':>15}")
print(f"  {'Capital Final':>20} ${capital_final:>14,.0f} ${spy_final:>14,.0f} ${capital_final - spy_final:>+14,.0f}")
print(f"  {'Multiplicador':>20} {multiple_sys:>14.1f}x {multiple_spy:>14.1f}x")
print(f"  {'CAGR':>20} {cagr_sys*100:>13.1f}% {cagr_spy*100:>13.1f}% {(cagr_sys-cagr_spy)*100:>+13.1f}%")
print(f"  {'Costes totales':>20} ${total_costs:>14,.0f}")
print(f"  {'Anos positivos':>20} {(df_r['pnl_net'] > 0).sum():>10}/{n_years}")
print(f"  {'Anos alpha > 0':>20} {((df_r['rent_net'] - df_r['spy_ret']) > 0).sum():>10}/{n_years}")
