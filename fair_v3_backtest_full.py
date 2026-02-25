"""
Fair V3 Backtest Completo 2003-2025
====================================
Compara:
- Fair V3: score eventos (promedio) + ajuste precio (DD/RSI) + bear_ratio
- Fair V2: score eventos (promedio) + bear_ratio (sin ajuste precio)
- Referencia: V3 original (616%, score raw + short_ratio)

Capital: $500,000/semana
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

# ================================================================
# FUNCIONES DEL SISTEMA FAIR V3
# ================================================================

def score_fair(active_events):
    """Score 0-10 basado en promedio de contribuciones."""
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
    """Ajuste por estado del precio: oversold (shorts) y overbought (longs)."""
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
    """Asignacion basada en bear_ratio (conteo bearish vs bullish)."""
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


def calc_pnl(longs, shorts, scores, ret_row, capital):
    """Calcula P&L con pesos por distancia al neutral (5.0)."""
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


def decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row):
    """Mean Reversion para NEUTRAL (br 0.30-0.45): 3L+2S basado en precio.
    LONGS: oversold (DD<-15% o RSI<40) + eventos no negativos (score>=4.0)
    SHORTS: overbought (RSI>65, DD>-8%) + eventos no positivos (score<=6.0) + ATR>=1.5%
    """
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
        # Longs oversold
        dd_factor = np.clip((abs(dd) - 15) / 30, 0, 1)
        rsi_os = np.clip((40 - rsi) / 25, 0, 1)
        oversold = max(dd_factor, rsi_os)
        if oversold > 0.1 and score >= 4.0:
            long_cands.append((sub_id, oversold))
        # Shorts overbought
        rsi_ob = np.clip((rsi - 65) / 20, 0, 1)
        if rsi_ob > 0.1 and dd > -8 and score <= 6.0 and atr >= ATR_MIN:
            short_cands.append((sub_id, rsi_ob))
    long_cands.sort(key=lambda x: -x[1])
    short_cands.sort(key=lambda x: -x[1])
    longs = [c[0] for c in long_cands[:3]]
    shorts = [c[0] for c in short_cands[:2]]
    # Weights dict for calc_pnl_meanrev
    weights = {}
    for s, w in long_cands[:3]:
        weights[s] = w
    for s, w in short_cands[:2]:
        weights[s] = w
    return longs, shorts, weights


def calc_pnl_meanrev(longs, shorts, weights, ret_row, capital):
    """P&L con pesos por oversold/overbought strength."""
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
print("Cargando datos de precio...")

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
print(f"  {len(df_all):,} registros diarios, {df_all['symbol'].nunique()} tickers")

# Resample semanal
df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100
)

# Agregar a nivel subsector
sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'),
    avg_high=('high', 'mean'),
    avg_low=('low', 'mean'),
    avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean'),
).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

# Fix KKR bug: filtrar fechas con pocos subsectores
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]
print(f"  Fechas validas (>=40 subsectores): {len(valid_dates)}")

# Calcular metricas de precio (DD 52w, RSI 14w)
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

print("  Calculando DD 52w y RSI 14w...")
sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)

# Pivotar a formato wide
returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide_lagged = atr_wide.shift(1)

print(f"  {returns_wide.shape[0]} semanas x {returns_wide.shape[1]} subsectores")
print(f"  Rango: {returns_wide.index.min().date()} a {returns_wide.index.max().date()}")

# Eventos
print("  Construyendo calendario de eventos...")
weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# SPY para referencia
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2001-12-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy_weekly = spy.set_index('date').resample('W-FRI').last().dropna()
spy_weekly['return'] = spy_weekly['close'].pct_change()

# ================================================================
# BACKTEST LOOP
# ================================================================
print("\nEjecutando backtest 2002-2025...")

yearly_v3 = {}
yearly_v2 = {}
all_pnl_v3 = []
all_pnl_v2 = []
config_counts_v3 = {}
config_counts_v2 = {}

for date in returns_wide.index:
    if date.year < 2002:
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
        all_pnl_v3.append({'date': date, 'pnl': 0, 'config': '0L+0S'})
        all_pnl_v2.append({'date': date, 'pnl': 0, 'config': '0L+0S'})
        continue

    # Score Fair (comun a V2 y V3)
    scores_evt = score_fair(active)

    # --- V3: con ajuste de precio ---
    prev_dates = dd_wide.index[dd_wide.index < date]
    if len(prev_dates) > 0:
        dd_row = dd_wide.loc[prev_dates[-1]]
        rsi_row = rsi_wide.loc[prev_dates[-1]]
    else:
        dd_row = None
        rsi_row = None

    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    longs_v3, shorts_v3, br_v3 = decide_allocation(scores_v3)

    # --- V2: solo eventos ---
    longs_v2, shorts_v2, br_v2 = decide_allocation(scores_evt)

    # ATR row (needed for both ATR filter and mean reversion)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    # Filtro ATR para shorts
    if atr_row is not None:
        shorts_v3 = [s for s in shorts_v3 if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
        shorts_v2 = [s for s in shorts_v2 if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    ret_row = returns_wide.loc[date]

    # P&L V3: Mean Reversion para NEUTRAL (br 0.30-0.50), score-based para el resto
    is_neutral = 0.30 <= br_v3 < 0.50
    if is_neutral:
        longs_v3, shorts_v3, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_v3 = calc_pnl_meanrev(longs_v3, shorts_v3, weights_mr, ret_row, CAPITAL)
        cfg_v3 = f"{len(longs_v3)}L+{len(shorts_v3)}S*"  # asterisco = mean reversion
    else:
        pnl_v3 = calc_pnl(longs_v3, shorts_v3, scores_v3, ret_row, CAPITAL)
        cfg_v3 = f"{len(longs_v3)}L+{len(shorts_v3)}S"
    config_counts_v3[cfg_v3] = config_counts_v3.get(cfg_v3, 0) + 1

    # P&L V2
    pnl_v2 = calc_pnl(longs_v2, shorts_v2, scores_evt, ret_row, CAPITAL)
    cfg_v2 = f"{len(longs_v2)}L+{len(shorts_v2)}S"
    config_counts_v2[cfg_v2] = config_counts_v2.get(cfg_v2, 0) + 1

    # Regime
    if br_v3 >= 0.60:     regime = 'CRISIS'
    elif br_v3 >= 0.50:   regime = 'BEARISH'
    elif br_v3 >= 0.30:   regime = 'NEUTRAL'
    else:                  regime = 'BULLISH'

    year = date.year
    yearly_v3[year] = yearly_v3.get(year, 0) + pnl_v3
    yearly_v2[year] = yearly_v2.get(year, 0) + pnl_v2

    all_pnl_v3.append({'date': date, 'pnl': pnl_v3, 'config': cfg_v3, 'regime': regime})
    all_pnl_v2.append({'date': date, 'pnl': pnl_v2, 'config': cfg_v2})

df_v3 = pd.DataFrame(all_pnl_v3)
df_v2 = pd.DataFrame(all_pnl_v2)


# ================================================================
# METRICAS
# ================================================================
def calc_metrics(df_pnl, yearly_pnl, label):
    total = df_pnl['pnl'].sum()
    n_years = (df_pnl['date'].max() - df_pnl['date'].min()).days / 365.25
    cagr = (1 + total / CAPITAL) ** (1 / n_years) - 1 if n_years > 0 else 0

    cum_pnl = df_pnl['pnl'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    active = df_pnl[df_pnl['pnl'] != 0]
    weekly_ret = active['pnl'] / CAPITAL
    sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52) if len(active) > 0 and weekly_ret.std() > 0 else 0
    win_rate = (active['pnl'] > 0).mean() * 100 if len(active) > 0 else 0
    n_profitable = sum(1 for p in yearly_pnl.values() if p > 0)

    return {
        'label': label,
        'total': total,
        'cagr': cagr * 100,
        'max_dd': max_dd,
        'max_dd_pct': max_dd / CAPITAL * 100,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'n_active': len(active),
        'n_profitable_years': n_profitable,
        'n_years': len(yearly_pnl),
        'pnl_per_year': total / n_years if n_years > 0 else 0,
    }


m_v3 = calc_metrics(df_v3, yearly_v3, "Fair V3 (eventos+precio)")
m_v2 = calc_metrics(df_v2, yearly_v2, "Fair V2 (solo eventos)")


# ================================================================
# RESULTADOS
# ================================================================
print(f"\n{'='*120}")
print(f"  BACKTEST COMPLETO 2002-2025 - SISTEMA FAIR")
print(f"{'='*120}")

print(f"\n  {'Metrica':<28s} {'Fair V3 (evt+precio)':>22s} {'Fair V2 (solo evt)':>22s} {'Diferencia':>14s}")
print(f"  {'-'*90}")

for key, fmt in [
    ('total', '${:+,.0f}'),
    ('cagr', '{:.1f}%'),
    ('max_dd', '${:+,.0f}'),
    ('max_dd_pct', '{:.1f}%'),
    ('sharpe', '{:.2f}'),
    ('win_rate', '{:.1f}%'),
    ('n_active', '{}'),
]:
    v3_val = m_v3[key]
    v2_val = m_v2[key]
    labels = {
        'total': 'Total P&L',
        'cagr': 'CAGR',
        'max_dd': 'Max Drawdown $',
        'max_dd_pct': 'Max Drawdown %',
        'sharpe': 'Sharpe Ratio',
        'win_rate': 'Win Rate',
        'n_active': 'Semanas activas',
    }
    lbl = labels[key]

    if isinstance(v3_val, float):
        diff = v3_val - v2_val
        if 'pct' in key or key == 'cagr':
            diff_str = f"{diff:+.1f}pp"
        elif '$' in fmt:
            diff_str = f"${diff:+,.0f}"
        else:
            diff_str = f"{diff:+.2f}"
    else:
        diff_str = ""

    print(f"  {lbl:<28s} {fmt.format(v3_val):>22s} {fmt.format(v2_val):>22s} {diff_str:>14s}")

print(f"  {'Anos rentables':<28s} {m_v3['n_profitable_years']}/{m_v3['n_years']:>19} {m_v2['n_profitable_years']}/{m_v2['n_years']:>19}")
print(f"  {'P&L/ano promedio':<28s} ${m_v3['pnl_per_year']:>+19,.0f} ${m_v2['pnl_per_year']:>+19,.0f}")


# ---- PnL por año ----
print(f"\n\n{'='*120}")
print(f"  PNL POR AÑO")
print(f"{'='*120}")

all_years = sorted(set(yearly_v3.keys()) | set(yearly_v2.keys()))

# SPY annual returns
spy_annual = {}
for yr in all_years:
    spy_yr = spy_weekly[(spy_weekly.index.year == yr)]
    if len(spy_yr) >= 2:
        spy_annual[yr] = (spy_yr.iloc[-1]['close'] / spy_yr.iloc[0]['close'] - 1) * 100
    else:
        spy_annual[yr] = 0

# Contar regimenes por ano
regime_counts_yr = {}
for r in all_pnl_v3:
    yr = r['date'].year
    reg = r.get('regime', 'NO_EVT')
    if yr not in regime_counts_yr:
        regime_counts_yr[yr] = {'BULLISH': 0, 'NEUTRAL': 0, 'BEARISH': 0, 'CRISIS': 0}
    if reg in regime_counts_yr[yr]:
        regime_counts_yr[yr][reg] += 1

print(f"\n  {'Ano':>6s} {'Fair V3':>12s} {'%Cap':>7s} {'SPY':>8s} {'Alpha':>8s} {'BULL':>5s} {'NEUT':>5s} {'BEAR':>5s} {'CRIS':>5s} {'Total':>5s}")
print(f"  {'-'*80}")

cum_v3 = 0
cum_v2 = 0
tot_bull = tot_neut = tot_bear = tot_cris = 0
for yr in all_years:
    pv3 = yearly_v3.get(yr, 0)
    pv2 = yearly_v2.get(yr, 0)
    spy_r = spy_annual.get(yr, 0)
    cum_v3 += pv3
    cum_v2 += pv2

    v3_ret = pv3 / CAPITAL * 100
    alpha = v3_ret - spy_r

    rc = regime_counts_yr.get(yr, {'BULLISH': 0, 'NEUTRAL': 0, 'BEARISH': 0, 'CRISIS': 0})
    nb, nn, nbe, nc = rc['BULLISH'], rc['NEUTRAL'], rc['BEARISH'], rc['CRISIS']
    tot_sem = nb + nn + nbe + nc
    tot_bull += nb; tot_neut += nn; tot_bear += nbe; tot_cris += nc

    print(f"  {yr:>6d} ${pv3:>+10,.0f} {v3_ret:>+6.1f}% {spy_r:>+7.1f}% {alpha:>+6.1f}pp {nb:>5d} {nn:>5d} {nbe:>5d} {nc:>5d} {tot_sem:>5d}")

print(f"  {'-'*80}")
print(f"  {'TOTAL':>6s} ${cum_v3:>+10,.0f} {cum_v3/CAPITAL*100:>+6.0f}%  {'':>8s} {'':>8s} {tot_bull:>5d} {tot_neut:>5d} {tot_bear:>5d} {tot_cris:>5d} {tot_bull+tot_neut+tot_bear+tot_cris:>5d}")
print(f"  {'':>6s} {'':>12s} {'':>7s}  {'':>8s} {'':>8s} {tot_bull/(tot_bull+tot_neut+tot_bear+tot_cris)*100:>4.0f}% {tot_neut/(tot_bull+tot_neut+tot_bear+tot_cris)*100:>4.0f}% {tot_bear/(tot_bull+tot_neut+tot_bear+tot_cris)*100:>4.0f}% {tot_cris/(tot_bull+tot_neut+tot_bear+tot_cris)*100:>4.0f}%")


# ---- Configs ----
print(f"\n\n{'='*120}")
print(f"  DISTRIBUCION DE CONFIGURACIONES")
print(f"{'='*120}")

total_v3 = sum(config_counts_v3.values())
total_v2 = sum(config_counts_v2.values())

print(f"\n  {'Config':>8s} {'Fair V3':>10s} {'%V3':>6s} {'Fair V2':>10s} {'%V2':>6s}")
print(f"  {'-'*45}")
all_cfgs = sorted(set(config_counts_v3.keys()) | set(config_counts_v2.keys()))
for cfg in all_cfgs:
    cv3 = config_counts_v3.get(cfg, 0)
    cv2 = config_counts_v2.get(cfg, 0)
    print(f"  {cfg:>8s} {cv3:>10d} {cv3/total_v3*100:>5.1f}% {cv2:>10d} {cv2/total_v2*100:>5.1f}%")


# ---- Anos perdedores detalle ----
print(f"\n\n{'='*120}")
print(f"  DETALLE AÑOS PERDEDORES (Fair V3)")
print(f"{'='*120}")

for yr in all_years:
    if yearly_v3.get(yr, 0) < 0:
        yr_weeks = [r for r in all_pnl_v3 if r['date'].year == yr and r['pnl'] != 0]
        n_win = sum(1 for w in yr_weeks if w['pnl'] > 0)
        n_loss = sum(1 for w in yr_weeks if w['pnl'] < 0)
        worst = min(yr_weeks, key=lambda x: x['pnl']) if yr_weeks else None
        best = max(yr_weeks, key=lambda x: x['pnl']) if yr_weeks else None
        print(f"\n  {yr}: ${yearly_v3[yr]:+,.0f} | SPY: {spy_annual.get(yr,0):+.1f}%")
        print(f"    Semanas: {len(yr_weeks)} activas, {n_win} ganadoras, {n_loss} perdedoras")
        if worst:
            print(f"    Peor semana:  {worst['date'].strftime('%Y-%m-%d')} ${worst['pnl']:+,.0f} ({worst['config']})")
        if best:
            print(f"    Mejor semana: {best['date'].strftime('%Y-%m-%d')} ${best['pnl']:+,.0f} ({best['config']})")


# ---- Estadisticas por regimen ----
print(f"\n\n{'='*120}")
print(f"  ESTADISTICAS POR REGIMEN (Fair V3)")
print(f"{'='*120}")

print(f"\n  {'Regimen':<10s} {'Semanas':>8s} {'PnL Total':>14s} {'Avg/sem':>10s} {'WR':>7s} {'Sharpe':>8s} {'Best sem':>12s} {'Worst sem':>12s}")
print(f"  {'-'*90}")

for regime in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    rg_weeks = [r for r in all_pnl_v3 if r.get('regime') == regime and r['pnl'] != 0]
    rg_all = [r for r in all_pnl_v3 if r.get('regime') == regime]
    if not rg_all:
        continue
    pnls = [r['pnl'] for r in rg_all]
    total = sum(pnls)
    avg = np.mean(pnls)
    wr = sum(1 for p in pnls if p > 0) / len([p for p in pnls if p != 0]) * 100 if any(p != 0 for p in pnls) else 0
    rets = [p / CAPITAL for p in pnls if p != 0]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(52) if len(rets) > 1 and np.std(rets) > 0 else 0
    best = max(pnls)
    worst = min(pnls)
    print(f"  {regime:<10s} {len(rg_all):>8d} ${total:>+12,.0f} ${avg:>+8,.0f} {wr:>6.1f}% {sharpe:>7.2f} ${best:>+10,.0f} ${worst:>+10,.0f}")

# Desglose NEUTRAL por ano
print(f"\n  NEUTRAL por ano:")
for yr in all_years:
    nr = [r for r in all_pnl_v3 if r.get('regime') == 'NEUTRAL' and r['date'].year == yr]
    if not nr:
        continue
    pnl_yr = sum(r['pnl'] for r in nr)
    cfgs = {}
    for r in nr:
        cfgs[r['config']] = cfgs.get(r['config'], 0) + 1
    cfg_str = ', '.join(f"{k}:{v}" for k, v in sorted(cfgs.items()))
    print(f"    {yr}: {len(nr):2d} sem  PnL=${pnl_yr:>+10,.0f}  [{cfg_str}]")


# ---- Comparacion con sistemas anteriores ----
print(f"\n\n{'='*120}")
print(f"  COMPARACION CON SISTEMAS ANTERIORES")
print(f"{'='*120}")

print(f"""
  Sistema                          Total P&L    %Capital    Sharpe    Tipo
  ---------------------------------------------------------------------------
  V3 Original (raw+dampen)         ~$3,080,000   ~616%      ~0.70     Score raw, short_ratio
  P1+P2 (3-Tier, 24 sem)          ~$1,360,000   ~272%      ~0.90     Seasonal patterns
  Fair V2 (solo eventos)           ${cum_v2:>+10,.0f}   {cum_v2/CAPITAL*100:>+5.0f}%      {m_v2['sharpe']:.2f}     Score fair, bear_ratio
  Fair V3 (evt+precio+meanrev)     ${cum_v3:>+10,.0f}   {cum_v3/CAPITAL*100:>+5.0f}%      {m_v3['sharpe']:.2f}     Score fair+DD/RSI+MR neutral
""")

print(f"{'='*120}")
print(f"  FIN DEL BACKTEST")
print(f"{'='*120}")
