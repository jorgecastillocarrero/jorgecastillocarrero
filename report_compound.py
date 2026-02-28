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
COST_PER_TRADE = 0.0010  # 0.10% por trade (comisiones + spread)
SLIPPAGE_PER_SIDE = 0.0005  # 0.05% slippage por lado (gap weekend + ejecucion)

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

def decide_alcista_pullback(scores_v3, dd_row, rsi_row):
    """ALCISTA (4.0-5.5): Pullback Trading.
    Comprar sectores alcistas que han corregido a soporte.
    - Score >= 5.0: fundamentales sanos (sector en tendencia alcista)
    - DD entre -5% y -15%: ha corregido pero no roto (pullback a soporte)
    - RSI entre 30-50: viniendo de sobreventa, recuperandose (no sobrecomprado)
    """
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50

        if score < 5.0: continue        # solo sectores con fundamentales sanos
        if dd > -5 or dd < -15: continue # pullback moderado, no colapso
        if rsi > 50 or rsi < 30: continue # recuperandose de oversold

        # Scoring: mejor pullback = mas profundo + RSI mas bajo + mejores fundamentales
        pullback_score = 0.0
        pullback_score += np.clip((abs(dd) - 5) / 10, 0, 1) * 2.0   # profundidad pullback
        pullback_score += np.clip((50 - rsi) / 20, 0, 1) * 2.0      # RSI recuperandose
        pullback_score += np.clip((score - 5.0) / 3.0, 0, 1) * 1.5  # calidad fundamental

        candidates.append((sub_id, pullback_score))

    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def decide_goldilocks_breakout(scores_v3, dd_row, rsi_row):
    """GOLDILOCKS (5.5-7.0): Breakout Trading.
    Comprar rupturas de resistencia: sectores near ATH con tendencia confirmada.
    - Score >= 5.5: fundamentales fuertes
    - DD > -8%: cerca de maximos (breakout zone)
    - RSI > 50: momentum positivo confirmado (tendencia fuerte)
    """
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50

        if score < 5.5: continue        # solo sectores fuertes
        if dd < -8: continue             # debe estar cerca de ATH
        if rsi < 50: continue            # momentum confirmado

        # Scoring: mas cerca de ATH + RSI mas fuerte + mejor score
        breakout_score = 0.0
        breakout_score += np.clip((8 + dd) / 8, 0, 1) * 2.5        # nearness to ATH
        breakout_score += np.clip((rsi - 50) / 30, 0, 1) * 2.0     # fuerza tendencia
        breakout_score += np.clip((score - 5.5) / 3.0, 0, 1) * 1.5 # calidad fundamental

        candidates.append((sub_id, breakout_score))

    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def decide_burbuja_aggressive(scores_v3, dd_row, rsi_row):
    """BURBUJA: longs agresivos en sectores con maximo momentum.
    - Near ATH (DD > -5%): sector en maximos, momentum intacto
    - RSI > 60: fuerza confirmada (no sobreventa)
    - Score > 6.0: fundamentales fuertes
    Concentrar en los 3 mejores por momentum score.
    """
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50

        # Filtros: sector fuerte, near ATH
        if score <= 6.0: continue            # solo fundamentales fuertes
        if dd < -8: continue                  # no en correccion
        if rsi < 55: continue                 # momentum confirmado

        # Scoring: cuanto mas cerca de ATH y mas RSI, mas agresivo
        momentum_score = 0.0
        momentum_score += np.clip((score - 6.0) / 2.5, 0, 1) * 2.5    # fair value alto
        momentum_score += np.clip((8 + dd) / 8, 0, 1) * 2.0            # near ATH (dd > -8 -> 0..1)
        momentum_score += np.clip((rsi - 55) / 25, 0, 1) * 1.5         # RSI fuerte
        candidates.append((sub_id, momentum_score))

    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def decide_neutral_oversold(scores_v3, dd_row, rsi_row):
    """NEUTRAL: Oversold Deep (solo longs, sin shorts).
    Comprar sectores en soporte extremo: DD < -15%, RSI < 35, score >= 3.5.
    Si no hay candidatos, no operar (0L+0S).
    """
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
    """Shorts agresivos para BEARISH: sectores en inicio de ruptura.
    - DD entre -5% y -20% (empezando a caer, NO ya colapsados)
    - RSI 30-50 (debilitandose, pero sin sobreventa extrema = sin squeeze)
    - ATR alto (mas recorrido por posicion)
    - Score < 4.5 (fundamentales debilitandose)
    """
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0

        # Filtros: sector en ruptura inicial, no colapsado
        if score >= 4.5: continue          # fundamentales no suficientemente debiles
        if dd < -25: continue              # ya colapsado = squeeze risk
        if rsi < 25: continue              # sobreventa extrema = squeeze risk
        if atr < ATR_MIN: continue         # sin volatilidad = sin recorrido

        # Scoring: mas agresivo = mas recorrido potencial
        breakdown_score = 0.0
        breakdown_score += np.clip((5.0 - score) / 3.0, 0, 1) * 2.0    # peor fair value = mejor short
        breakdown_score += np.clip(abs(dd) / 20.0, 0, 1) * 1.5          # drawdown moderado
        breakdown_score += np.clip((50 - rsi) / 25.0, 0, 1) * 1.5       # RSI debilitandose
        breakdown_score += np.clip((atr - ATR_MIN) / 3.0, 0, 1) * 1.0   # mas ATR = mas recorrido

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
    pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50

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

    # Scoring extendido: mas resolucion en zona negativa (rango -12.5 a +8.0)
    # Zona positiva sin cambios, zona negativa añade 1 escalon extra por indicador

    # Breadth DD (% subsectores con DD > -10%)
    if pct_dd_healthy >= 75: score_bdd = 2.0
    elif pct_dd_healthy >= 60: score_bdd = 1.0
    elif pct_dd_healthy >= 45: score_bdd = 0.0
    elif pct_dd_healthy >= 30: score_bdd = -1.0
    elif pct_dd_healthy >= 15: score_bdd = -2.0
    else: score_bdd = -3.0

    # Breadth RSI (% subsectores con RSI > 55)
    if pct_rsi_above55 >= 75: score_brsi = 2.0
    elif pct_rsi_above55 >= 60: score_brsi = 1.0
    elif pct_rsi_above55 >= 45: score_brsi = 0.0
    elif pct_rsi_above55 >= 30: score_brsi = -1.0
    elif pct_rsi_above55 >= 15: score_brsi = -2.0
    else: score_brsi = -3.0

    # DD deep (% subsectores con DD < -20%)
    if pct_dd_deep <= 5: score_ddp = 1.5
    elif pct_dd_deep <= 15: score_ddp = 0.5
    elif pct_dd_deep <= 30: score_ddp = -0.5
    elif pct_dd_deep <= 50: score_ddp = -1.5
    else: score_ddp = -2.5

    # SPY vs MA200
    if spy_above_ma200 and spy_dist > 5: score_spy = 1.5
    elif spy_above_ma200: score_spy = 0.5
    elif spy_dist > -5: score_spy = -0.5
    elif spy_dist > -15: score_spy = -1.5
    else: score_spy = -2.5

    # SPY momentum 10 semanas
    if spy_mom_10w > 5: score_mom = 1.0
    elif spy_mom_10w > 0: score_mom = 0.5
    elif spy_mom_10w > -5: score_mom = -0.5
    elif spy_mom_10w > -15: score_mom = -1.0
    else: score_mom = -1.5

    total = score_bdd + score_brsi + score_ddp + score_spy + score_mom

    # BURBUJA: condiciones extremas (DD healthy >= 85% Y RSI>55 >= 90%)
    is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)

    # Regimenes: positivos sin cambios, negativos con 4 niveles
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'

    # VIX como alerta
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'CAUTIOUS'

    # VIX bajando vs semana anterior → señal de rebote/reversion
    vix_dates_all = vix_df.index[vix_df.index <= date]
    if len(vix_dates_all) >= 2:
        prev_vix = vix_df.loc[vix_dates_all[-2], 'vix']
        if pd.notna(prev_vix) and vix_val < prev_vix:
            if regime == 'PANICO':
                regime = 'CAPITULACION'
            elif regime == 'BEARISH':
                regime = 'RECOVERY'

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

# Datos viernes (ultimo dia de la semana): para senales (DD, RSI, regimen)
df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100)

# Datos lunes (primer dia de la semana): para retornos reales de trading
df_monday = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).first().reset_index()
df_monday = df_monday.sort_values(['symbol', 'date'])
df_monday['prev_open'] = df_monday.groupby('symbol')['open'].shift(1)
df_monday['return_mon'] = df_monday['open'] / df_monday['prev_open'] - 1
df_monday = df_monday.dropna(subset=['return_mon'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

# Retornos lunes open -> lunes open (1 semana exacta de holding)
sub_monday = df_monday.groupby(['subsector', 'date']).agg(
    avg_return_mon=('return_mon', 'mean')).reset_index()
sub_monday = sub_monday.sort_values(['subsector', 'date'])

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

# Retornos lunes open -> lunes open (1 semana exacta de holding)
# Flujo: senal viernes W -> compra lunes W+1 open -> venta lunes W+2 open
# return_mon en df_monday = open(this_mon) / open(prev_mon) - 1  (backward-looking)
# Para senal viernes W, el retorno de trading es return_mon del lunes W+2:
#   open(lun W+2) / open(lun W+1) - 1 = retorno de la semana operada
# fri + 10 dias = lunes de 2 semanas despues = donde esta almacenado el retorno correcto
returns_mon_wide = sub_monday.pivot(index='date', columns='subsector', values='avg_return_mon')
mon_dates = returns_mon_wide.index.tolist()
fri_dates = returns_wide.index.tolist()

fri_to_mon_ret = {}
for fri in fri_dates:
    target = fri + pd.Timedelta(days=10)  # viernes + 10 = lunes W+2 (donde esta el retorno del trade)
    diffs = [abs((d - target).days) for d in mon_dates]
    if diffs:
        closest_mon = mon_dates[diffs.index(min(diffs))]
        if abs((closest_mon - target).days) <= 3:
            fri_to_mon_ret[fri] = closest_mon

returns_trade_wide = pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns, dtype=float)
for fri, mon in fri_to_mon_ret.items():
    if mon in returns_mon_wide.index:
        returns_trade_wide.loc[fri] = returns_mon_wide.loc[mon]
print(f"Retornos lunes->lunes mapeados: {returns_trade_wide.notna().any(axis=1).sum()}/{len(fri_dates)} semanas")

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
# Senal disponible lunes open (eventos programados + regimen de semana anterior)
# Retorno = viernes W-1 close -> viernes W close (aprox. lunes open -> viernes close)
# Gap weekend + slippage ejecucion modelados como coste adicional
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
    prev_dates = dd_wide.index[dd_wide.index <= date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    # Retorno real: lunes open -> lunes open (1 semana exacta)
    if date in returns_trade_wide.index and returns_trade_wide.loc[date].notna().any():
        ret_row = returns_trade_wide.loc[date]
    else:
        ret_row = returns_wide.loc[date]  # fallback viernes->viernes

    regime, _ = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])

    if regime == 'PANICO':          nl, ns = 0, 3   # solo shorts (extremo)
    elif regime == 'CRISIS':         nl, ns = 0, 3   # solo shorts
    elif regime == 'BEARISH':       nl, ns = 0, 3   # shorts agresivos
    elif regime == 'CAUTIOUS':      nl, ns = 3, 0   # oversold deep (solo longs)
    elif regime == 'NEUTRAL':       nl, ns = 3, 0   # oversold deep (solo longs)
    elif regime == 'ALCISTA':    nl, ns = 3, 0   # longs standard
    elif regime == 'GOLDILOCKS':       nl, ns = 3, 0   # longs standard
    elif regime == 'BURBUJA':       nl, ns = 3, 0   # longs agresivos (override abajo)
    elif regime == 'CAPITULACION':  nl, ns = 3, 0   # longs agresivos (rebote desde panico)
    elif regime == 'RECOVERY': nl, ns = 3, 0   # longs (reversion desde bearish)
    else:                           nl, ns = 3, 0

    longs = [s for s, _ in longs_pool[:nl]]
    shorts = [s for s, _ in shorts_pool[:ns]]

    if atr_row is not None:
        shorts = [s for s in shorts if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    # Calcular con capital unitario ($1) para obtener retorno %
    if regime == 'BURBUJA':
        longs_bub, weights_bub = decide_burbuja_aggressive(scores_v3, dd_row, rsi_row)
        if longs_bub:
            longs = longs_bub
            shorts = []
            pnl_unit = calc_pnl_meanrev(longs_bub, [], weights_bub, ret_row, 1.0)
        else:
            pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)
    elif regime == 'GOLDILOCKS':
        # Top3 FairValue: los 3 subsectores con mayor score, ponderados por score-5.0
        top3_gold = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
        longs = [s for s, _ in top3_gold]
        shorts = []
        pnl_unit = calc_pnl(longs, [], scores_v3, ret_row, 1.0)
    elif regime == 'ALCISTA':
        # Top3 FairValue: los 3 subsectores con mayor score, ponderados por score-5.0
        top3_alc = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
        longs = [s for s, _ in top3_alc]
        shorts = []
        pnl_unit = calc_pnl(longs, [], scores_v3, ret_row, 1.0)
    elif regime == 'CAUTIOUS':
        longs_caut, weights_caut = decide_neutral_oversold(scores_v3, dd_row, rsi_row)
        if longs_caut:
            longs = longs_caut
            shorts = []
            pnl_unit = calc_pnl_meanrev(longs_caut, [], weights_caut, ret_row, 1.0)
        else:
            longs = []
            shorts = []
            pnl_unit = 0.0
    elif regime == 'NEUTRAL':
        longs_neut, weights_neut = decide_neutral_oversold(scores_v3, dd_row, rsi_row)
        if longs_neut:
            longs = longs_neut
            shorts = []
            pnl_unit = calc_pnl_meanrev(longs_neut, [], weights_neut, ret_row, 1.0)
        else:
            longs = []
            shorts = []
            pnl_unit = 0.0
    elif regime == 'BEARISH':
        shorts_bear, weights_bear = decide_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row)
        if shorts_bear:
            longs = []
            shorts = shorts_bear
            pnl_unit = calc_pnl_meanrev([], shorts_bear, weights_bear, ret_row, 1.0)
        else:
            pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)
    elif regime == 'CRISIS':
        shorts_cri, weights_cri = decide_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row)
        if shorts_cri:
            longs = []
            shorts = shorts_cri
            pnl_unit = calc_pnl_meanrev([], shorts_cri, weights_cri, ret_row, 1.0)
        else:
            pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)
    elif regime == 'PANICO':
        shorts_pan, weights_pan = decide_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row)
        if shorts_pan:
            longs = []
            shorts = shorts_pan
            pnl_unit = calc_pnl_meanrev([], shorts_pan, weights_pan, ret_row, 1.0)
        else:
            pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)
    elif regime == 'CAPITULACION':
        # Top3 FairValue: rebote desde panico, longs agresivos
        top3_cap = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
        longs = [s for s, _ in top3_cap]
        shorts = []
        pnl_unit = calc_pnl(longs, [], scores_v3, ret_row, 1.0)
    elif regime == 'RECOVERY':
        # Top3 FairValue: reversion desde bearish, longs
        top3_br = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
        longs = [s for s, _ in top3_br]
        shorts = []
        pnl_unit = calc_pnl(longs, [], scores_v3, ret_row, 1.0)
    else:
        pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)

    n_pos = len(longs) + len(shorts)
    # Costes: comisiones (0.10%/lado) + slippage (0.05%/lado) = 0.15%/lado, 0.30% round-trip
    cost_pct = (COST_PER_TRADE + SLIPPAGE_PER_SIDE) * 2 if n_pos > 0 else 0

    # Separar P&L longs vs shorts para diagnostico
    pnl_longs = calc_pnl(longs, [], scores_v3, ret_row, 1.0) if longs else 0
    pnl_shorts = calc_pnl([], shorts, scores_v3, ret_row, 1.0) if shorts else 0

    weekly_returns.append({
        'date': date,
        'ret_gross': pnl_unit,      # retorno % bruto (como fraccion de 1)
        'cost_pct': cost_pct,        # coste % (como fraccion de 1)
        'regime': regime,
        'n_pos': n_pos,
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'pnl_longs': pnl_longs,
        'pnl_shorts': pnl_shorts,
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
        'burb': rc.get('BURBUJA', 0),
        'gold': rc.get('GOLDILOCKS', 0),
        'alci': rc.get('ALCISTA', 0),
        'neut': rc.get('NEUTRAL', 0),
        'caut': rc.get('CAUTIOUS', 0),
        'bear': rc.get('BEARISH', 0),
        'cris': rc.get('CRISIS', 0),
        'pani': rc.get('PANICO', 0),
        'capi': rc.get('CAPITULACION', 0),
        'reco': rc.get('RECOVERY', 0),
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
print(f"Costes: {COST_PER_TRADE*100:.2f}% comision + {SLIPPAGE_PER_SIDE*100:.2f}% slippage por lado = {(COST_PER_TRADE+SLIPPAGE_PER_SIDE)*2*100:.2f}% round-trip")
print(f"Timing: senal viernes close -> compra lunes open -> retorno semanal (gap+slippage en costes)")
print("=" * 120)

print(f"\n{'Ano':>5} {'Cap.Inicio':>12} {'S&P500':>8} {'Sist%':>8} {'Alpha':>8} "
      f"{'P&L Neto':>12} {'Costes':>10} {'Cap.Final':>14}  "
      f"{'BURB':>5} {'ALCI':>5} {'GOLD':>5} {'NEUT':>5} {'CAUT':>5} {'BEAR':>5} {'RECO':>5} {'CRIS':>5} {'PANI':>5} {'CAPI':>5}  {'SPY Cap':>14}")
print("-" * 180)

for _, row in df_r.iterrows():
    alpha = row['rent_net'] - row['spy_ret']
    print(f"{int(row['year']):>5} ${row['capital_inicio']:>11,.0f} {row['spy_ret']:>7.1f}% "
          f"{row['rent_net']:>7.1f}% {alpha:>+7.1f}% "
          f"${row['pnl_net']:>11,.0f} ${row['cost']:>9,.0f} ${row['capital_fin']:>13,.0f}  "
          f"{int(row['burb']):>5} {int(row['alci']):>5} {int(row['gold']):>5} "
          f"{int(row['neut']):>5} {int(row['caut']):>5} {int(row['bear']):>5} {int(row['reco']):>5} {int(row['cris']):>5} {int(row['pani']):>5} {int(row['capi']):>5}  "
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

print("-" * 180)

# Estadisticas por regimen
print(f"\n{'Regimen':<16} {'Semanas':>7} {'Avg ret%':>10} {'WinRate':>8} {'Config':>8}")
print("-" * 55)
for reg, cfg in [('BURBUJA', '3L+0S*'), ('GOLDILOCKS', '3L+0S'), ('ALCISTA', '3L+0S'), ('NEUTRAL', 'oversold'), ('CAUTIOUS', 'oversold'), ('BEARISH', '0L+3S*'), ('RECOVERY', '3L+0S'), ('CRISIS', '0L+3S*'), ('PANICO', '0L+3S*'), ('CAPITULACION', '3L+0S')]:
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

# Impacto por regimen (base $500K)
BASE = 500_000
print(f"\n{'IMPACTO POR REGIMEN (base $500K)':>45}")
print("-" * 75)
print(f"  {'Regimen':<16} {'N':>5} {'Avg bruto':>12} {'Avg neto':>12} {'Coste/sem':>10} {'Margen':>12}")
print("-" * 75)
for reg, cfg in [('BURBUJA', '3L+0S*'), ('GOLDILOCKS', '3L+0S'), ('ALCISTA', '3L+0S'), ('NEUTRAL', 'oversold'), ('CAUTIOUS', 'oversold'), ('BEARISH', '0L+3S*'), ('RECOVERY', '3L+0S'), ('CRISIS', '0L+3S*'), ('PANICO', '0L+3S*'), ('CAPITULACION', '3L+0S')]:
    mask = df_ret['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df_ret[mask]
    avg_gross = sub['ret_gross'].mean() * BASE
    avg_cost = sub['cost_pct'].mean() * BASE
    avg_net = sub['ret_net'].mean() * BASE
    ratio = avg_net / avg_cost if avg_cost > 0 else 0
    margen = "Holgado" if ratio > 2.5 else "OK" if ratio > 1.0 else "Justo" if ratio > 0.2 else "NEGATIVO"
    print(f"  {reg:<16} {mask.sum():>5} ${avg_gross:>10,.0f} ${avg_net:>10,.0f} ${avg_cost:>8,.0f}   {margen:<10} ({ratio:.1f}x)")

# ================================================================
# ESTADISTICOS DETALLADOS
# ================================================================
print("\n" + "=" * 80)
print("ESTADISTICOS DETALLADOS")
print("=" * 80)

# --- Nivel semanal ---
rets = df_ret['ret_net'].values
rets_spy = spy_w['ret_spy'].reindex(df_ret['date'].values).values

# Sharpe (anualizado, 52 semanas)
sharpe_sys = rets.mean() / rets.std() * np.sqrt(52) if rets.std() > 0 else 0
sharpe_spy = np.nanmean(rets_spy) / np.nanstd(rets_spy) * np.sqrt(52) if np.nanstd(rets_spy) > 0 else 0

# Sortino (solo downside deviation)
downside = rets[rets < 0]
downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-9
sortino_sys = rets.mean() / downside_std * np.sqrt(52)

downside_spy = rets_spy[~np.isnan(rets_spy) & (rets_spy < 0)]
downside_std_spy = np.sqrt(np.mean(downside_spy**2)) if len(downside_spy) > 0 else 1e-9
sortino_spy = np.nanmean(rets_spy) / downside_std_spy * np.sqrt(52)

# Max Drawdown semanal (sobre equity curve compuesta)
equity = CAPITAL_INICIAL * np.cumprod(1 + rets)
peak = np.maximum.accumulate(equity)
dd_pct = (equity - peak) / peak
max_dd = dd_pct.min() * 100

equity_spy = CAPITAL_INICIAL * np.cumprod(1 + np.nan_to_num(rets_spy))
peak_spy = np.maximum.accumulate(equity_spy)
dd_spy = (equity_spy - peak_spy) / peak_spy
max_dd_spy = dd_spy.min() * 100

# Calmar = CAGR / |MaxDD|
calmar_sys = (cagr_sys * 100) / abs(max_dd) if max_dd != 0 else 0
calmar_spy = (cagr_spy * 100) / abs(max_dd_spy) if max_dd_spy != 0 else 0

# Profit Factor
gross_wins = rets[rets > 0].sum()
gross_losses = abs(rets[rets < 0].sum())
profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

# Win/Loss stats
avg_win = rets[rets > 0].mean() * 100 if (rets > 0).any() else 0
avg_loss = rets[rets < 0].mean() * 100 if (rets < 0).any() else 0
best_week = rets.max() * 100
worst_week = rets.min() * 100
win_rate = (rets > 0).mean() * 100
n_weeks = len(rets)
n_active = (df_ret['n_pos'] > 0).sum()

# Streaks
streaks_win, streaks_loss = [], []
curr = 0
for r in rets:
    if r > 0:
        if curr > 0: curr += 1
        else: curr = 1
    elif r < 0:
        if curr < 0: curr -= 1
        else:
            if curr > 0: streaks_win.append(curr)
            curr = -1
    else:
        if curr > 0: streaks_win.append(curr)
        elif curr < 0: streaks_loss.append(abs(curr))
        curr = 0
if curr > 0: streaks_win.append(curr)
elif curr < 0: streaks_loss.append(abs(curr))

max_win_streak = max(streaks_win) if streaks_win else 0
max_loss_streak = max(streaks_loss) if streaks_loss else 0

# Volatilidad anualizada
vol_sys = rets.std() * np.sqrt(52) * 100
vol_spy = np.nanstd(rets_spy) * np.sqrt(52) * 100

# Skew y Kurtosis (manual, sin scipy)
n = len(rets)
m = rets.mean()
s = rets.std()
skew_sys = (np.sum((rets - m)**3) / n) / s**3 if s > 0 else 0
kurt_sys = (np.sum((rets - m)**4) / n) / s**4 - 3 if s > 0 else 0  # excess kurtosis

# Max DD duration
dd_start = None
max_dd_dur = 0
curr_dur = 0
for i, eq in enumerate(equity):
    if eq < peak[i]:
        curr_dur += 1
        max_dd_dur = max(max_dd_dur, curr_dur)
    else:
        curr_dur = 0

# Peor ano
worst_year_row = df_r.loc[df_r['rent_net'].idxmin()]
best_year_row = df_r.loc[df_r['rent_net'].idxmax()]

# Años con perdida
loss_years = df_r[df_r['pnl_net'] < 0]

print(f"\n{'METRICAS DE RIESGO-RETORNO':>35} {'SISTEMA':>12} {'S&P 500':>12}")
print("-" * 62)
print(f"  {'Sharpe Ratio (anual.)':>33} {sharpe_sys:>11.2f} {sharpe_spy:>11.2f}")
print(f"  {'Sortino Ratio (anual.)':>33} {sortino_sys:>11.2f} {sortino_spy:>11.2f}")
print(f"  {'Calmar Ratio':>33} {calmar_sys:>11.2f} {calmar_spy:>11.2f}")
print(f"  {'Volatilidad Anualizada':>33} {vol_sys:>10.1f}% {vol_spy:>10.1f}%")
print(f"  {'Max Drawdown':>33} {max_dd:>10.1f}% {max_dd_spy:>10.1f}%")
print(f"  {'Max DD Duracion (semanas)':>33} {max_dd_dur:>11} {'':>12}")
print(f"  {'Profit Factor':>33} {profit_factor:>11.2f} {'':>12}")
print(f"  {'Skewness':>33} {skew_sys:>+11.3f} {'':>12}")
print(f"  {'Kurtosis (excess)':>33} {kurt_sys:>11.2f} {'':>12}")

print(f"\n{'DISTRIBUCION SEMANAL':>35}")
print("-" * 62)
print(f"  {'Semanas totales':>33} {n_weeks:>11}")
print(f"  {'Semanas activas':>33} {n_active:>11}")
print(f"  {'Win Rate':>33} {win_rate:>10.1f}%")
print(f"  {'Avg Win':>33} {avg_win:>+10.3f}%")
print(f"  {'Avg Loss':>33} {avg_loss:>+10.3f}%")
print(f"  {'Win/Loss Ratio':>33} {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>11.2f}")
print(f"  {'Mejor Semana':>33} {best_week:>+10.2f}%")
print(f"  {'Peor Semana':>33} {worst_week:>+10.2f}%")
print(f"  {'Max Racha Ganadora':>33} {max_win_streak:>8} sem")
print(f"  {'Max Racha Perdedora':>33} {max_loss_streak:>8} sem")

print(f"\n{'DISTRIBUCION ANUAL':>35}")
print("-" * 62)
print(f"  {'Mejor Ano':>33}  {int(best_year_row['year'])} ({best_year_row['rent_net']:+.1f}%)")
print(f"  {'Peor Ano':>33}  {int(worst_year_row['year'])} ({worst_year_row['rent_net']:+.1f}%)")
print(f"  {'Anos con perdida':>33} {len(loss_years):>11}")
if len(loss_years) > 0:
    for _, ly in loss_years.iterrows():
        print(f"  {'':>35} {int(ly['year'])}: {ly['rent_net']:+.1f}%")

# Estadisticos por regimen detallados
print(f"\n{'DETALLE POR REGIMEN':>35}")
print("-" * 95)
print(f"  {'Regimen':<16} {'N':>5} {'Avg%':>7} {'Med%':>7} {'Std%':>7} {'WR%':>6} {'Sharpe':>7} {'Best%':>8} {'Worst%':>8} {'PF':>6}")
print("-" * 95)
for reg, cfg in [('BURBUJA', '3L+0S*'), ('GOLDILOCKS', '3L+0S'), ('ALCISTA', '3L+0S'), ('NEUTRAL', 'oversold'), ('CAUTIOUS', 'oversold'), ('BEARISH', '0L+3S*'), ('RECOVERY', '3L+0S'), ('CRISIS', '0L+3S*'), ('PANICO', '0L+3S*'), ('CAPITULACION', '3L+0S')]:
    mask = df_ret['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df_ret[mask]['ret_net'].values
    avg = sub.mean() * 100
    med = np.median(sub) * 100
    std = sub.std() * 100
    wr = (sub > 0).mean() * 100
    sh = sub.mean() / sub.std() * np.sqrt(52) if sub.std() > 0 else 0
    bst = sub.max() * 100
    wst = sub.min() * 100
    gw = sub[sub > 0].sum()
    gl = abs(sub[sub < 0].sum())
    pf = gw / gl if gl > 0 else float('inf')
    print(f"  {reg:<16} {mask.sum():>5} {avg:>+6.2f} {med:>+6.2f} {std:>6.2f} {wr:>5.1f} {sh:>+6.2f} {bst:>+7.2f} {wst:>+7.2f} {pf:>5.2f}")

# Diagnostico LONGS vs SHORTS por regimen
print(f"\n{'DIAGNOSTICO LONGS vs SHORTS POR REGIMEN':>45}")
print("-" * 85)
print(f"  {'Regimen':<16} {'N':>5}  {'Avg L%':>8} {'WR L%':>7}  {'Avg S%':>8} {'WR S%':>7}  {'L $/sem':>10} {'S $/sem':>10}")
print("-" * 85)
BASE = 500_000
for reg in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'RECOVERY', 'CRISIS', 'PANICO', 'CAPITULACION']:
    mask = df_ret['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df_ret[mask]
    avg_l = sub['pnl_longs'].mean() * 100
    avg_s = sub['pnl_shorts'].mean() * 100
    wr_l = (sub['pnl_longs'] > 0).mean() * 100 if (sub['n_longs'] > 0).any() else 0
    wr_s = (sub['pnl_shorts'] > 0).mean() * 100 if (sub['n_shorts'] > 0).any() else 0
    dl = sub['pnl_longs'].mean() * BASE
    ds = sub['pnl_shorts'].mean() * BASE
    print(f"  {reg:<16} {mask.sum():>5}  {avg_l:>+7.3f} {wr_l:>6.1f}  {avg_s:>+7.3f} {wr_s:>6.1f}  ${dl:>9,.0f} ${ds:>9,.0f}")

# Caracteristicas detalladas LONGS y SHORTS por regimen
print(f"\n{'CARACTERISTICAS DETALLADAS POR LADO Y REGIMEN':>50}")
print("=" * 100)
for reg in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'RECOVERY', 'CRISIS', 'PANICO', 'CAPITULACION']:
    mask = df_ret['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df_ret[mask]

    print(f"\n  {reg} ({mask.sum()} semanas)")
    print(f"  {'-'*96}")

    for side, col in [('LONGS', 'pnl_longs'), ('SHORTS', 'pnl_shorts')]:
        vals = sub[col].values
        n_active = (sub[f'n_{side.lower()}'] > 0).sum()
        if n_active == 0:
            print(f"    {side:<8} No activo")
            continue

        vals_active = vals[sub[f'n_{side.lower()}'] > 0]
        avg = vals_active.mean() * 100
        med = np.median(vals_active) * 100
        std = vals_active.std() * 100
        wr = (vals_active > 0).mean() * 100
        best = vals_active.max() * 100
        worst = vals_active.min() * 100
        sharpe = vals_active.mean() / vals_active.std() * np.sqrt(52) if vals_active.std() > 0 else 0
        gw = vals_active[vals_active > 0].sum()
        gl = abs(vals_active[vals_active < 0].sum())
        pf = gw / gl if gl > 0 else float('inf')
        avg_n = sub.loc[sub[f'n_{side.lower()}'] > 0, f'n_{side.lower()}'].mean()

        # Streaks
        wins = vals_active > 0
        max_wstreak = max_lstreak = curr = 0
        for w in wins:
            if w: curr = max(0, curr) + 1; max_wstreak = max(max_wstreak, curr)
            else: curr = min(0, curr) - 1; max_lstreak = max(max_lstreak, abs(curr))

        # Distribucion por quintiles
        q25 = np.percentile(vals_active, 25) * 100
        q75 = np.percentile(vals_active, 75) * 100

        print(f"    {side:<8} N={n_active:>4} | Avg={avg:>+6.3f}% Med={med:>+6.3f}% Std={std:>5.3f}% | "
              f"WR={wr:>5.1f}% Sharpe={sharpe:>+5.2f} PF={pf:>4.2f}")
        print(f"    {'':8} Avg pos={avg_n:>3.1f} | Best={best:>+6.2f}% Worst={worst:>+6.2f}% | "
              f"Q25={q25:>+6.3f}% Q75={q75:>+6.3f}%")
        print(f"    {'':8} Racha W={max_wstreak} L={max_lstreak} | "
              f"$/sem=${vals_active.mean()*BASE:>+8,.0f}")
