"""
Test Regime basado en datos de mercado reales
==============================================
Reemplaza bear_ratio como clasificador de regimen con indicadores de mercado:

PRIMARIO (Breadth interno):
- % subsectores con DD > -10% (salud del mercado)
- % subsectores con RSI > 50 (amplitud)

CONFIRMACION (SPY Trend):
- SPY vs MA200 (tendencia)
- SPY momentum 10 semanas (direccion)

ALERTA (VIX):
- Solo para pasar de calm -> menos estable (nunca al reves)

El scoring de subsectores sigue siendo event-based (funciona para seleccion).
Solo cambia COMO se determina el regimen.
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

# ================================================================
# FUNCIONES DEL SISTEMA (mismas que fair_v3)
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


def decide_allocation_by_regime(regime, scores):
    """Decide longs/shorts basado en regimen de mercado + scores para seleccion."""
    longs_pool = sorted([(s, sc) for s, sc in scores.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores.items() if sc < 3.5], key=lambda x: x[1])

    max_pos = 3
    if regime == 'CRISIS':
        nl, ns = 0, max_pos      # solo shorts
    elif regime == 'BEARISH':
        nl, ns = 1, max_pos      # defensivo
    elif regime == 'NEUTRAL':
        nl, ns = max_pos, max_pos # equilibrado (o mean reversion)
    else:  # BULLISH
        nl, ns = max_pos, 0      # solo longs

    longs = [s for s, _ in longs_pool[:nl]]
    shorts = [s for s, _ in shorts_pool[:ns]]

    # bear_ratio para referencia
    bc = len(shorts_pool)
    blc = len(longs_pool)
    br = bc / (bc + blc) if (bc + blc) > 0 else 0.5

    return longs, shorts, br


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
    for s, w in long_cands[:3]:
        weights[s] = w
    for s, w in short_cands[:2]:
        weights[s] = w
    return longs, shorts, weights


def calc_pnl(longs, shorts, scores, ret_row, capital):
    n = len(longs) + len(shorts)
    if n == 0:
        return 0.0
    lw = {s: scores[s] - 5.0 for s in longs}
    sw = {s: 5.0 - scores[s] for s in shorts}
    tw = sum(lw.values()) + sum(sw.values())
    if tw <= 0:
        return 0.0
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
    if n == 0:
        return 0.0
    lw = {s: weights.get(s, 0) for s in longs}
    sw = {s: weights.get(s, 0) for s in shorts}
    tw = sum(lw.values()) + sum(sw.values())
    if tw <= 0:
        return 0.0
    pnl = 0.0
    for s in longs:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (lw[s] / tw) * ret_row[s]
    for s in shorts:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (sw[s] / tw) * (-ret_row[s])
    return pnl


# ================================================================
# CARGAR DATOS
# ================================================================
print("=" * 70)
print("TEST: REGIMEN BASADO EN DATOS DE MERCADO REALES")
print("=" * 70)

print("\nCargando datos de precio...")
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

# SPY con MA200
print("Cargando SPY...")
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

# VIX
vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

# Eventos
weekly_events = build_weekly_events('2000-01-01', '2026-02-21')


# ================================================================
# FUNCION DE REGIMEN DE MERCADO (datos reales)
# ================================================================

def classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df):
    """
    Clasifica el regimen basado en datos de mercado reales.

    Indicadores:
    1. Breadth DD: % subsectores con DD > -10% (cerca de maximos)
    2. Breadth RSI: % subsectores con RSI > 50 (momento positivo)
    3. SPY vs MA200: tendencia macro
    4. SPY momentum 10w: direccion
    5. VIX: solo como alerta (calm -> less stable)
    """
    # 1. Breadth
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        return 'NEUTRAL', {}

    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]

    n_total = dd_row.notna().sum()
    if n_total == 0:
        return 'NEUTRAL', {}

    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100   # cerca de maximos
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100      # en caida profunda
    pct_rsi_above50 = (rsi_row > 50).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50

    # 2. SPY trend
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

    # 3. VIX
    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # ================================================================
    # SCORING COMPUESTO
    # ================================================================
    # Cada componente da un score de -2 (crisis) a +2 (bull)

    # Breadth DD healthy (PESO ALTO - mejor separador)
    if pct_dd_healthy >= 75:
        score_breadth_dd = 2.0
    elif pct_dd_healthy >= 60:
        score_breadth_dd = 1.0
    elif pct_dd_healthy >= 45:
        score_breadth_dd = 0.0
    elif pct_dd_healthy >= 30:
        score_breadth_dd = -1.0
    else:
        score_breadth_dd = -2.0

    # Breadth RSI (PESO ALTO)
    if pct_rsi_above50 >= 75:
        score_breadth_rsi = 2.0
    elif pct_rsi_above50 >= 60:
        score_breadth_rsi = 1.0
    elif pct_rsi_above50 >= 45:
        score_breadth_rsi = 0.0
    elif pct_rsi_above50 >= 30:
        score_breadth_rsi = -1.0
    else:
        score_breadth_rsi = -2.0

    # DD deep (PESO MEDIO - inverso)
    if pct_dd_deep <= 5:
        score_dd_deep = 1.5
    elif pct_dd_deep <= 15:
        score_dd_deep = 0.5
    elif pct_dd_deep <= 30:
        score_dd_deep = -0.5
    else:
        score_dd_deep = -1.5

    # SPY trend (PESO MEDIO)
    if spy_above_ma200 and spy_dist > 5:
        score_spy = 1.5
    elif spy_above_ma200:
        score_spy = 0.5
    elif spy_dist > -5:
        score_spy = -0.5
    else:
        score_spy = -1.5

    # SPY momentum (PESO BAJO)
    if spy_mom_10w > 5:
        score_mom = 1.0
    elif spy_mom_10w > 0:
        score_mom = 0.5
    elif spy_mom_10w > -5:
        score_mom = -0.5
    else:
        score_mom = -1.0

    # Score total (sin VIX)
    total_score = score_breadth_dd + score_breadth_rsi + score_dd_deep + score_spy + score_mom
    # Rango: -8.0 a +8.0

    # Clasificacion base
    if total_score >= 4.0:
        regime = 'BULLISH'
    elif total_score >= 0.5:
        regime = 'NEUTRAL'
    elif total_score >= -3.0:
        regime = 'BEARISH'
    else:
        regime = 'CRISIS'

    # VIX como ALERTA ONLY: si VIX alto, sube un nivel de cautela
    if vix_val >= 30 and regime == 'BULLISH':
        regime = 'NEUTRAL'    # miedo, no puede ser bull
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'BEARISH'    # panico, sube cautela

    details = {
        'pct_dd_healthy': pct_dd_healthy,
        'pct_rsi_above50': pct_rsi_above50,
        'pct_dd_deep': pct_dd_deep,
        'spy_above_ma200': spy_above_ma200,
        'spy_mom_10w': spy_mom_10w,
        'vix': vix_val,
        'score_total': total_score,
        'score_bdd': score_breadth_dd,
        'score_brsi': score_breadth_rsi,
        'score_ddp': score_dd_deep,
        'score_spy': score_spy,
        'score_mom': score_mom,
    }

    return regime, details


# ================================================================
# BACKTEST
# ================================================================
print("\nEjecutando backtest 2002-2025...")
print("  Sistema BASE: bear_ratio (eventos)")
print("  Sistema MARKET: breadth + SPY trend + VIX alerta")

yearly_base = {}
yearly_market = {}
all_pnl_base = []
all_pnl_market = []

for date in returns_wide.index:
    if date.year < 2001:
        continue

    # Eventos activos
    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    if not active:
        all_pnl_base.append({'date': date, 'pnl': 0, 'regime': 'NEUTRAL'})
        all_pnl_market.append({'date': date, 'pnl': 0, 'regime': 'NEUTRAL'})
        continue

    # Scores (comunes)
    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    ret_row = returns_wide.loc[date]

    # === SISTEMA BASE (bear_ratio) ===
    longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])
    bc = len(shorts_pool)
    blc = len(longs_pool)
    br = bc / (bc + blc) if (bc + blc) > 0 else 0.5

    if br >= 0.60:     regime_base = 'CRISIS'
    elif br >= 0.50:   regime_base = 'BEARISH'
    elif br >= 0.30:   regime_base = 'NEUTRAL'
    else:              regime_base = 'BULLISH'

    # Allocation base
    if br >= 0.70:   nl_b, ns_b = 0, 3
    elif br >= 0.60: nl_b, ns_b = 1, 3
    elif br >= 0.55: nl_b, ns_b = 2, 3
    elif br >= 0.45: nl_b, ns_b = 3, 3
    elif br >= 0.40: nl_b, ns_b = 3, 2
    elif br >= 0.30: nl_b, ns_b = 3, 1
    else:            nl_b, ns_b = 3, 0
    longs_b = [s for s, _ in longs_pool[:nl_b]]
    shorts_b = [s for s, _ in shorts_pool[:ns_b]]

    # Filtro ATR base
    if atr_row is not None:
        shorts_b = [s for s in shorts_b if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    is_neutral_b = 0.30 <= br < 0.50
    if is_neutral_b:
        longs_b, shorts_b, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_base = calc_pnl_meanrev(longs_b, shorts_b, weights_mr, ret_row, CAPITAL)
    else:
        pnl_base = calc_pnl(longs_b, shorts_b, scores_v3, ret_row, CAPITAL)

    year = date.year
    yearly_base[year] = yearly_base.get(year, 0) + pnl_base
    all_pnl_base.append({'date': date, 'pnl': pnl_base, 'regime': regime_base, 'bear_ratio': br})

    # === SISTEMA MARKET (datos de mercado) ===
    regime_mkt, details = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    # Allocation basada en regimen de mercado pero usando scores para seleccion
    if regime_mkt == 'CRISIS':
        nl_m, ns_m = 0, 3
    elif regime_mkt == 'BEARISH':
        nl_m, ns_m = 1, 3
    elif regime_mkt == 'NEUTRAL':
        nl_m, ns_m = 3, 3  # mean reversion
    else:  # BULLISH
        nl_m, ns_m = 3, 0

    longs_m = [s for s, _ in longs_pool[:nl_m]]
    shorts_m = [s for s, _ in shorts_pool[:ns_m]]

    # Filtro ATR
    if atr_row is not None:
        shorts_m = [s for s in shorts_m if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    # NEUTRAL usa mean reversion, resto score-based
    if regime_mkt == 'NEUTRAL':
        longs_m, shorts_m, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_mkt = calc_pnl_meanrev(longs_m, shorts_m, weights_mr, ret_row, CAPITAL)
    else:
        pnl_mkt = calc_pnl(longs_m, shorts_m, scores_v3, ret_row, CAPITAL)

    yearly_market[year] = yearly_market.get(year, 0) + pnl_mkt
    all_pnl_market.append({
        'date': date, 'pnl': pnl_mkt, 'regime': regime_mkt,
        'score_total': details.get('score_total', 0),
        'pct_dd_healthy': details.get('pct_dd_healthy', 0),
        'pct_rsi_above50': details.get('pct_rsi_above50', 0),
        'vix': details.get('vix', 0),
    })

df_base = pd.DataFrame(all_pnl_base)
df_mkt = pd.DataFrame(all_pnl_market)

# ================================================================
# METRICAS
# ================================================================
def calc_metrics(df_pnl, yearly_pnl, label):
    total = df_pnl['pnl'].sum()
    n_years = (df_pnl['date'].max() - df_pnl['date'].min()).days / 365.25
    cum = df_pnl['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    active = df_pnl[df_pnl['pnl'] != 0]
    weekly_ret = active['pnl'] / CAPITAL
    sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52) if len(active) > 0 and weekly_ret.std() > 0 else 0
    win_rate = (active['pnl'] > 0).mean() * 100 if len(active) > 0 else 0
    n_profitable = sum(1 for p in yearly_pnl.values() if p > 0)
    return {
        'label': label, 'total': total, 'max_dd': max_dd, 'sharpe': sharpe,
        'win_rate': win_rate, 'n_profitable': n_profitable, 'n_years': len(yearly_pnl),
    }

m_base = calc_metrics(df_base, yearly_base, "BASE (bear_ratio)")
m_mkt = calc_metrics(df_mkt, yearly_market, "MARKET (breadth+SPY+VIX)")

# ================================================================
# RESULTADOS
# ================================================================
print("\n" + "=" * 70)
print("RESULTADOS COMPARATIVOS")
print("=" * 70)

for m in [m_base, m_mkt]:
    print(f"\n{m['label']}:")
    print(f"  P&L Total:    ${m['total']:>12,.0f}")
    print(f"  Max Drawdown: ${m['max_dd']:>12,.0f}")
    print(f"  Sharpe:       {m['sharpe']:>8.2f}")
    print(f"  Win Rate:     {m['win_rate']:>7.1f}%")
    print(f"  Anos +: {m['n_profitable']}/{m['n_years']}")

# Tabla anual
print("\n" + "=" * 70)
print("TABLA ANUAL COMPARATIVA")
print("=" * 70)

# SPY annual returns
spy_annual = spy_w['ret_spy'].groupby(spy_w.index.year).sum() * 100

print(f"\n{'Ano':>5} {'SPY':>8} {'BASE':>10} {'MARKET':>10} {'Diff':>10}  {'Reg BASE':>10} {'Reg MKT':>10}")
print("-" * 80)

for year in sorted(set(list(yearly_base.keys()) + list(yearly_market.keys()))):
    pnl_b = yearly_base.get(year, 0)
    pnl_m = yearly_market.get(year, 0)
    spy_r = spy_annual.get(year, 0)
    diff = pnl_m - pnl_b

    # Regime dominante
    rb = df_base[df_base['date'].dt.year == year]['regime'].mode()
    rm = df_mkt[df_mkt['date'].dt.year == year]['regime'].mode()
    rb_str = rb.iloc[0] if len(rb) > 0 else '?'
    rm_str = rm.iloc[0] if len(rm) > 0 else '?'

    marker = " <---" if abs(diff) > 50000 else ""
    print(f"{year:>5} {spy_r:>7.1f}% ${pnl_b:>9,.0f} ${pnl_m:>9,.0f} ${diff:>+9,.0f}  {rb_str:>10} {rm_str:>10}{marker}")

# Distribucion de regimenes por ano
print("\n" + "=" * 70)
print("DISTRIBUCION DE REGIMENES POR ANO")
print("=" * 70)

print(f"\n{'Ano':>5} {'SPY':>7}  {'--- BASE (bear_ratio) ---':>35}  {'--- MARKET (datos reales) ---':>40}")
print(f"{'':>5} {'':>7}  {'BULL':>6} {'NEUT':>6} {'BEAR':>6} {'CRIS':>6}  {'BULL':>6} {'NEUT':>6} {'BEAR':>6} {'CRIS':>6}")
print("-" * 95)

for year in range(2001, 2026):
    yr_b = df_base[df_base['date'].dt.year == year]
    yr_m = df_mkt[df_mkt['date'].dt.year == year]
    if len(yr_b) == 0:
        continue

    spy_r = spy_annual.get(year, 0)

    # Counts base
    cb = yr_b['regime'].value_counts()
    # Counts market
    cm = yr_m['regime'].value_counts()

    print(f"{year:>5} {spy_r:>6.1f}%  "
          f"{cb.get('BULLISH',0):>6} {cb.get('NEUTRAL',0):>6} {cb.get('BEARISH',0):>6} {cb.get('CRISIS',0):>6}  "
          f"{cm.get('BULLISH',0):>6} {cm.get('NEUTRAL',0):>6} {cm.get('BEARISH',0):>6} {cm.get('CRISIS',0):>6}")

# Estadisticas por regimen MARKET
print("\n" + "=" * 70)
print("ESTADISTICAS POR REGIMEN (MARKET)")
print("=" * 70)

for regime in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    mask = df_mkt['regime'] == regime
    sub = df_mkt[mask]
    if len(sub) == 0:
        continue
    active = sub[sub['pnl'] != 0]
    avg_pnl = active['pnl'].mean() if len(active) > 0 else 0
    wr = (active['pnl'] > 0).mean() * 100 if len(active) > 0 else 0
    total = sub['pnl'].sum()
    print(f"\n  {regime}: {len(sub)} semanas, P&L total ${total:,.0f}, "
          f"avg ${avg_pnl:,.0f}/sem, WR {wr:.0f}%")

print("\n" + "=" * 70)
print("ANOS PROBLEMATICOS: detalle de regimen MARKET")
print("=" * 70)

for year in [2001, 2002, 2003, 2007, 2008, 2009, 2011, 2015, 2018, 2019, 2020, 2022]:
    yr = df_mkt[df_mkt['date'].dt.year == year]
    if len(yr) == 0:
        continue
    spy_r = spy_annual.get(year, 0)
    cm = yr['regime'].value_counts()
    avg_score = yr['score_total'].mean()
    avg_dd_h = yr['pct_dd_healthy'].mean()
    avg_rsi = yr['pct_rsi_above50'].mean()
    avg_vix = yr['vix'].mean()

    print(f"\n  {year} (SPY {spy_r:+.1f}%): score={avg_score:.1f}, "
          f"DD_healthy={avg_dd_h:.0f}%, RSI>50={avg_rsi:.0f}%, VIX={avg_vix:.0f}")
    print(f"    Regimenes: {dict(cm)}")
