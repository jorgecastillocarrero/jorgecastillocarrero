"""
Backtest: 10 acciones individuales (longs + shorts) por regimen
================================================================
En vez de operar subsectores, selecciona las 10 mejores acciones
individuales del S&P 500 cada semana segun:
  1. Fair Value del subsector (evento macro)
  2. Momentum individual (12w return)
  3. Fuerza tecnica individual (RSI)

Asignacion long/short por regimen:
  BURBUJA:    10L + 0S
  GOLDILOCKS:  7L + 3S
  ALCISTA:     7L + 3S
  NEUTRAL:     5L + 5S
  CAUTIOUS:    5L + 5S
  BEARISH:     3L + 7S
  CRISIS:      0L + 10S
  PANICO:      0L + 10S
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0

# Configuracion long/short por regimen
REGIME_ALLOC = {
    'BURBUJA':       (10, 0),
    'GOLDILOCKS':    (7, 3),
    'ALCISTA':       (7, 3),
    'NEUTRAL':       (5, 5),
    'CAUTIOUS':      (5, 5),
    'BEARISH':       (3, 7),
    'CRISIS':        (0, 10),
    'PANICO':        (0, 10),
    'CAPITULACION':  (10, 0),
}

# ================================================================
# FUNCIONES
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

def classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: return 'NEUTRAL'
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL'
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
    # CAPITULACION: PANICO + VIX bajando vs semana anterior
    if regime == 'PANICO':
        vix_dates_all = vix_df.index[vix_df.index <= date]
        if len(vix_dates_all) >= 2:
            prev_vix = vix_df.loc[vix_dates_all[-2], 'vix']
            if pd.notna(prev_vix) and vix_val < prev_vix:
                regime = 'CAPITULACION'
    return regime

# ================================================================
# CARGAR DATOS
# ================================================================
print("Cargando datos de acciones individuales...")

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

print(f"  Acciones cargadas: {df_all['symbol'].nunique()} tickers, {len(df_all):,} registros")

# Datos semanales por accion individual
df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])

# Metricas individuales por accion
print("Calculando metricas individuales (DD, RSI, momentum)...")

def calc_stock_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['high'].rolling(52, min_periods=26).max()
    g['dd_52w'] = (g['close'] / g['high_52w'] - 1) * 100
    # RSI 14 semanas
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    # Momentum 12 semanas
    g['mom_12w'] = g['close'].pct_change(12) * 100
    # ATR % (5 semanas)
    g['hl_range'] = (g['high'] - g['low']) / g['close']
    g['atr_pct'] = g['hl_range'].rolling(5, min_periods=3).mean() * 100
    return g

df_weekly = df_weekly.groupby('symbol', group_keys=False).apply(calc_stock_metrics)

# Subsector aggregates (para regimen classification)
sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_return=('return', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index

def calc_sub_metrics(g):
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

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_sub_metrics)
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

# SPY
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21' ORDER BY date
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

# Pivotar retornos individuales por fecha
# Para cada viernes, tener el retorno semanal de cada accion
stock_ret = df_weekly.pivot_table(index='date', columns='symbol', values='return')
stock_dd = df_weekly.pivot_table(index='date', columns='symbol', values='dd_52w')
stock_rsi = df_weekly.pivot_table(index='date', columns='symbol', values='rsi_14w')
stock_mom = df_weekly.pivot_table(index='date', columns='symbol', values='mom_12w')
stock_atr = df_weekly.pivot_table(index='date', columns='symbol', values='atr_pct')

# Fechas validas (al menos 40 subsectores con datos)
valid_fridays = sorted(set(stock_ret.index) & set(valid_dates))

print(f"  Semanas validas: {len(valid_fridays)}")

# ================================================================
# BACKTEST: 10 ACCIONES POR SEMANA
# ================================================================
print("Ejecutando backtest 10 acciones...")

results = []

for date in valid_fridays:
    if date.year < 2001:
        continue

    # SPY return
    spy_ret = spy_w.loc[date, 'ret_spy'] if date in spy_w.index else np.nan
    if not pd.notna(spy_ret):
        continue

    # Events -> FV scores
    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    if not active:
        continue

    fv_scores = score_fair(active)

    # Regime
    regime = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)
    n_long, n_short = REGIME_ALLOC.get(regime, (5, 5))

    # Datos individuales de esta semana (semana ANTERIOR para señales)
    prev_dates = stock_ret.index[stock_ret.index < date]
    if len(prev_dates) == 0:
        continue
    signal_date = prev_dates[-1]  # viernes anterior para señales tecnicas

    # Retornos de ESTA semana (lo que ganamos/perdemos)
    ret_this_week = stock_ret.loc[date] if date in stock_ret.index else None
    if ret_this_week is None:
        continue

    # Señales tecnicas del viernes anterior
    dd_prev = stock_dd.loc[signal_date] if signal_date in stock_dd.index else None
    rsi_prev = stock_rsi.loc[signal_date] if signal_date in stock_rsi.index else None
    mom_prev = stock_mom.loc[signal_date] if signal_date in stock_mom.index else None
    atr_prev = stock_atr.loc[signal_date] if signal_date in stock_atr.index else None

    # ── SELECCION DE LONGS ──
    long_candidates = []
    if n_long > 0:
        # Subsectores con FV > 5.5 (favorables)
        good_subs = {s: fv for s, fv in fv_scores.items() if fv > 5.5}
        for sub_id, fv in good_subs.items():
            tickers = SUBSECTORS[sub_id]['tickers']
            for ticker in tickers:
                if ticker not in ret_this_week.index: continue
                ret = ret_this_week.get(ticker)
                if not pd.notna(ret): continue
                mom = mom_prev.get(ticker, 0) if mom_prev is not None else 0
                rsi = rsi_prev.get(ticker, 50) if rsi_prev is not None else 50
                dd = dd_prev.get(ticker, 0) if dd_prev is not None else 0
                if not pd.notna(mom): mom = 0
                if not pd.notna(rsi): rsi = 50
                if not pd.notna(dd): dd = 0

                # Score compuesto: FV subsector + momentum individual + fuerza tecnica
                composite = 0.0
                composite += np.clip((fv - 5.0) / 4.0, 0, 1) * 3.0      # FV subsector (0-3)
                composite += np.clip((mom + 20) / 60, 0, 1) * 3.0        # Momentum 12w (-20 a +40 -> 0-3)
                composite += np.clip((rsi - 30) / 50, 0, 1) * 2.0        # RSI (30-80 -> 0-2)
                composite += np.clip((dd + 30) / 30, 0, 1) * 2.0         # Near ATH (DD -30 a 0 -> 0-2)

                long_candidates.append({
                    'ticker': ticker, 'sub': sub_id, 'fv': fv,
                    'mom': mom, 'rsi': rsi, 'dd': dd,
                    'score': composite, 'ret': ret,
                })

    # ── SELECCION DE SHORTS ──
    short_candidates = []
    if n_short > 0:
        # Subsectores con FV < 4.5 (desfavorables)
        bad_subs = {s: fv for s, fv in fv_scores.items() if fv < 4.5}
        for sub_id, fv in bad_subs.items():
            tickers = SUBSECTORS[sub_id]['tickers']
            for ticker in tickers:
                if ticker not in ret_this_week.index: continue
                ret = ret_this_week.get(ticker)
                if not pd.notna(ret): continue
                mom = mom_prev.get(ticker, 0) if mom_prev is not None else 0
                rsi = rsi_prev.get(ticker, 50) if rsi_prev is not None else 50
                dd = dd_prev.get(ticker, 0) if dd_prev is not None else 0
                atr = atr_prev.get(ticker, 0) if atr_prev is not None else 0
                if not pd.notna(mom): mom = 0
                if not pd.notna(rsi): rsi = 50
                if not pd.notna(dd): dd = 0
                if not pd.notna(atr): atr = 0

                # Filtros para shorts: no demasiado colapsado (squeeze risk)
                if dd < -40: continue       # ya colapsado
                if rsi < 15: continue       # sobreventa extrema
                if atr < 1.5: continue      # sin volatilidad

                # Score: peor FV + peor momentum + debilidad tecnica + alta volatilidad
                composite = 0.0
                composite += np.clip((5.0 - fv) / 4.0, 0, 1) * 3.0      # Peor FV (0-3)
                composite += np.clip((-mom + 20) / 60, 0, 1) * 3.0       # Peor momentum (0-3)
                composite += np.clip((70 - rsi) / 50, 0, 1) * 2.0        # RSI debil (0-2)
                composite += np.clip((atr - 1.5) / 5.0, 0, 1) * 2.0     # Alta ATR = recorrido (0-2)

                short_candidates.append({
                    'ticker': ticker, 'sub': sub_id, 'fv': fv,
                    'mom': mom, 'rsi': rsi, 'dd': dd,
                    'score': composite, 'ret': ret,
                })

    # Ordenar y seleccionar
    long_candidates.sort(key=lambda x: -x['score'])
    short_candidates.sort(key=lambda x: -x['score'])

    # Diversificar: max 2 acciones del mismo subsector
    def diversified_pick(candidates, n, max_per_sub=2):
        picked = []
        sub_count = {}
        for c in candidates:
            s = c['sub']
            if sub_count.get(s, 0) >= max_per_sub:
                continue
            picked.append(c)
            sub_count[s] = sub_count.get(s, 0) + 1
            if len(picked) >= n:
                break
        return picked

    longs_sel = diversified_pick(long_candidates, n_long)
    shorts_sel = diversified_pick(short_candidates, n_short)

    # Retorno equal-weight del portfolio de 10 acciones
    port_rets = []
    long_tickers = []
    short_tickers = []

    for c in longs_sel:
        port_rets.append(c['ret'])       # long = positive return
        long_tickers.append(c['ticker'])
    for c in shorts_sel:
        port_rets.append(-c['ret'])      # short = inverted return
        short_tickers.append(c['ticker'])

    if port_rets:
        port_ret = np.mean(port_rets)
    else:
        port_ret = 0.0

    n_total = len(longs_sel) + len(shorts_sel)

    results.append({
        'date': date,
        'year': date.year,
        'regime': regime,
        'spy_ret': spy_ret,
        'port_ret': port_ret,
        'n_long': len(longs_sel),
        'n_short': len(shorts_sel),
        'n_total': n_total,
        'long_ret': np.mean([c['ret'] for c in longs_sel]) if longs_sel else 0,
        'short_ret': np.mean([-c['ret'] for c in shorts_sel]) if shorts_sel else 0,
        'longs': ', '.join(long_tickers[:5]),
        'shorts': ', '.join(short_tickers[:5]),
    })

df = pd.DataFrame(results)
print(f"  Semanas procesadas: {len(df)}")

# ================================================================
# RESULTADOS
# ================================================================
print("\n" + "=" * 140)
print("  BACKTEST: 10 ACCIONES INDIVIDUALES (LONGS + SHORTS) POR REGIMEN")
print("  Periodo: 2001-2025 | Equal-weight | Sin costes")
print("=" * 140)

regime_order = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO', 'CAPITULACION']

# ── Tabla 1: Comparativa principal ──
print(f"\n{'1. SPY vs 10 ACCIONES: retorno semanal medio':>50}")
print("-" * 135)
print(f"  {'Regimen':<12} {'N':>5} {'Alloc':>6}  |  "
      f"{'SPY avg':>8} {'SPY med':>8} {'SPY WR':>7}  |  "
      f"{'Port avg':>8} {'Port med':>9} {'Port WR':>8}  |  "
      f"{'Alpha':>7} {'Alpha $':>9}")
print("-" * 135)

BASE = 500_000
n_years = 25

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]
    nl, ns = REGIME_ALLOC[reg]
    alloc = f"{nl}L+{ns}S"

    spy_avg = sub['spy_ret'].mean() * 100
    spy_med = sub['spy_ret'].median() * 100
    spy_wr = (sub['spy_ret'] > 0).mean() * 100

    port_avg = sub['port_ret'].mean() * 100
    port_med = sub['port_ret'].median() * 100
    port_wr = (sub['port_ret'] > 0).mean() * 100

    alpha = port_avg - spy_avg
    alpha_yr = alpha / 100 * BASE * mask.sum() / n_years

    print(f"  {reg:<12} {mask.sum():>5} {alloc:>6}  |  "
          f"{spy_avg:>+7.2f}% {spy_med:>+7.2f}% {spy_wr:>6.1f}%  |  "
          f"{port_avg:>+7.2f}% {port_med:>+8.2f}% {port_wr:>7.1f}%  |  "
          f"{alpha:>+6.2f}% ${alpha_yr:>+8,.0f}")

# Total
spy_total = df['spy_ret'].mean() * 100
port_total = df['port_ret'].mean() * 100
alpha_total = port_total - spy_total
alpha_yr_total = alpha_total / 100 * BASE * len(df) / n_years
print("-" * 135)
print(f"  {'TOTAL':<12} {len(df):>5} {'':>6}  |  "
      f"{spy_total:>+7.2f}% {'':>8} {'':>7}  |  "
      f"{port_total:>+7.2f}% {'':>9} {'':>8}  |  "
      f"{alpha_total:>+6.2f}% ${alpha_yr_total:>+8,.0f}")


# ── Tabla 2: Desglose LONGS vs SHORTS ──
print(f"\n\n{'2. DESGLOSE LONGS vs SHORTS POR REGIMEN':>45}")
print("-" * 100)
print(f"  {'Regimen':<12} {'N':>5}  |  {'L avg%':>7} {'L WR%':>7} {'L best':>7} {'L wrst':>7}  |  "
      f"{'S avg%':>7} {'S WR%':>7} {'S best':>7} {'S wrst':>7}")
print("-" * 100)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]
    nl, ns = REGIME_ALLOC[reg]

    if nl > 0:
        l_avg = sub['long_ret'].mean() * 100
        l_wr = (sub['long_ret'] > 0).mean() * 100
        l_best = sub['long_ret'].max() * 100
        l_worst = sub['long_ret'].min() * 100
        l_str = f"{l_avg:>+6.2f}% {l_wr:>6.1f}% {l_best:>+6.1f}% {l_worst:>+6.1f}%"
    else:
        l_str = f"{'---':>7} {'---':>7} {'---':>7} {'---':>7}"

    if ns > 0:
        s_avg = sub['short_ret'].mean() * 100
        s_wr = (sub['short_ret'] > 0).mean() * 100
        s_best = sub['short_ret'].max() * 100
        s_worst = sub['short_ret'].min() * 100
        s_str = f"{s_avg:>+6.2f}% {s_wr:>6.1f}% {s_best:>+6.1f}% {s_worst:>+6.1f}%"
    else:
        s_str = f"{'---':>7} {'---':>7} {'---':>7} {'---':>7}"

    print(f"  {reg:<12} {mask.sum():>5}  |  {l_str}  |  {s_str}")


# ── Tabla 3: Sharpe ──
print(f"\n\n{'3. EFICIENCIA (Sharpe semanal anualizado)':>45}")
print("-" * 90)
print(f"  {'Regimen':<12} {'N':>5}  |  {'SPY Sharpe':>11}  |  {'Port Sharpe':>12}  |  {'Ganador':>8}")
print("-" * 90)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() < 5:
        continue
    sub = df[mask]

    spy_sh = sub['spy_ret'].mean() / sub['spy_ret'].std() * np.sqrt(52) if sub['spy_ret'].std() > 0 else 0
    port_sh = sub['port_ret'].mean() / sub['port_ret'].std() * np.sqrt(52) if sub['port_ret'].std() > 0 else 0

    winner = "PORT" if port_sh > spy_sh else "SPY"
    print(f"  {reg:<12} {mask.sum():>5}  |  {spy_sh:>+10.2f}  |  {port_sh:>+11.2f}  |  [{winner:>5}]")

# Global
spy_sh_g = df['spy_ret'].mean() / df['spy_ret'].std() * np.sqrt(52)
port_sh_g = df['port_ret'].mean() / df['port_ret'].std() * np.sqrt(52)
print("-" * 90)
print(f"  {'GLOBAL':<12} {len(df):>5}  |  {spy_sh_g:>+10.2f}  |  {port_sh_g:>+11.2f}  |  [{'PORT' if port_sh_g > spy_sh_g else 'SPY':>5}]")


# ── Tabla 4: Correlacion ──
print(f"\n\n{'4. CORRELACION SPY vs PORTFOLIO':>40}")
print("-" * 75)
print(f"  {'Regimen':<12} {'N':>5}  |  {'Corr':>6}  |  {'SPY- -> Port':>13}  |  {'SPY+ -> Port':>13}")
print("-" * 75)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() < 5:
        continue
    sub = df[mask]
    corr = sub['spy_ret'].corr(sub['port_ret'])
    spy_down = sub[sub['spy_ret'] < 0]
    spy_up = sub[sub['spy_ret'] >= 0]
    port_when_down = spy_down['port_ret'].mean() * 100 if len(spy_down) > 0 else 0
    port_when_up = spy_up['port_ret'].mean() * 100 if len(spy_up) > 0 else 0
    print(f"  {reg:<12} {mask.sum():>5}  |  {corr:>+5.2f}  |  {port_when_down:>+12.2f}%  |  {port_when_up:>+12.2f}%")


# ── Tabla 5: Acciones mas seleccionadas ──
from collections import Counter
print(f"\n\n{'5. TOP 10 ACCIONES MAS SELECCIONADAS POR REGIMEN':>55}")
print("=" * 100)

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]

    long_counts = Counter()
    short_counts = Counter()
    for _, row in sub.iterrows():
        if row['longs']:
            for t in row['longs'].split(', '):
                if t: long_counts[t] += 1
        if row['shorts']:
            for t in row['shorts'].split(', '):
                if t: short_counts[t] += 1

    nl, ns = REGIME_ALLOC[reg]
    print(f"\n  {reg} ({mask.sum()} sem, {nl}L+{ns}S):")

    if long_counts:
        top_l = long_counts.most_common(5)
        names = [f"{t}({c})" for t, c in top_l]
        print(f"    LONGS:  {', '.join(names)}")
    if short_counts:
        top_s = short_counts.most_common(5)
        names = [f"{t}({c})" for t, c in top_s]
        print(f"    SHORTS: {', '.join(names)}")


# ── Tabla 6: Comparativa con estrategia subsectores ──
print(f"\n\n{'6. COMPARATIVA: SPY vs SUBSECTORES vs 10 ACCIONES':>55}")
print("=" * 120)
print(f"  {'Regimen':<12} {'N':>5}  |  {'SPY avg%':>9}  |  {'SubSec avg%':>12} {'Alpha sub':>10}  |  "
      f"{'10Acc avg%':>11} {'Alpha 10a':>10}  |  {'Mejor':>6}")
print("-" * 120)

# Nota: para subsectores, usamos los datos de comparativa_spy_vs_estrategia (hardcoded de la ejecucion anterior)
subsec_avgs = {
    'BURBUJA': 0.80, 'GOLDILOCKS': 1.04, 'ALCISTA': 1.11,
    'NEUTRAL': 0.50, 'CAUTIOUS': 0.60,
    'BEARISH': 0.79, 'CRISIS': 0.63, 'PANICO': 1.79,
}

for reg in regime_order:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]

    spy_avg = sub['spy_ret'].mean() * 100
    port_avg = sub['port_ret'].mean() * 100
    sub_avg = subsec_avgs.get(reg, 0)

    alpha_sub = sub_avg - spy_avg
    alpha_10a = port_avg - spy_avg

    if port_avg > sub_avg and port_avg > spy_avg:
        mejor = "10Acc"
    elif sub_avg > port_avg and sub_avg > spy_avg:
        mejor = "SubSec"
    else:
        mejor = "SPY"

    print(f"  {reg:<12} {mask.sum():>5}  |  {spy_avg:>+8.2f}%  |  {sub_avg:>+11.2f}% {alpha_sub:>+9.2f}%  |  "
          f"{port_avg:>+10.2f}% {alpha_10a:>+9.2f}%  |  {mejor:>6}")


# ── Tabla 7: Compounding anual ──
print(f"\n\n{'7. RENTABILIDAD ANUAL COMPUESTA':>40}")
print("-" * 100)
print(f"  {'Ano':>5} {'SPY%':>7} {'Port%':>8} {'Alpha%':>8} {'N sem':>6}  |  "
      f"{'SPY acum':>10} {'Port acum':>11}")
print("-" * 100)

spy_cap = BASE
port_cap = BASE

for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year]
    spy_yr = (1 + yr['spy_ret']).prod() - 1
    port_yr = (1 + yr['port_ret']).prod() - 1

    spy_cap *= (1 + spy_yr)
    port_cap *= (1 + port_yr)

    alpha = (port_yr - spy_yr) * 100
    print(f"  {year:>5} {spy_yr*100:>+6.1f}% {port_yr*100:>+7.1f}% {alpha:>+7.1f}% {len(yr):>6}  |  "
          f"${spy_cap:>9,.0f} ${port_cap:>10,.0f}")

print("-" * 100)
mult_spy = spy_cap / BASE
mult_port = port_cap / BASE
cagr_spy = (spy_cap / BASE) ** (1/n_years) - 1
cagr_port = (port_cap / BASE) ** (1/n_years) - 1
print(f"\n  Capital final:   SPY ${spy_cap:>12,.0f} ({mult_spy:.1f}x)  |  Portfolio ${port_cap:>12,.0f} ({mult_port:.1f}x)")
print(f"  CAGR:            SPY {cagr_spy*100:>5.1f}%            |  Portfolio {cagr_port*100:>5.1f}%")
print(f"  Alpha CAGR:      {(cagr_port - cagr_spy)*100:>+5.1f}%")

# Sharpe global
spy_sharpe = df['spy_ret'].mean() / df['spy_ret'].std() * np.sqrt(52)
port_sharpe = df['port_ret'].mean() / df['port_ret'].std() * np.sqrt(52)
print(f"  Sharpe:          SPY {spy_sharpe:.2f}             |  Portfolio {port_sharpe:.2f}")

# Win rate annual
spy_pos_years = sum(1 for y in sorted(df['year'].unique()) if (1 + df[df['year']==y]['spy_ret']).prod() > 1)
port_pos_years = sum(1 for y in sorted(df['year'].unique()) if (1 + df[df['year']==y]['port_ret']).prod() > 1)
print(f"  Anos positivos:  SPY {spy_pos_years}/{n_years}             |  Portfolio {port_pos_years}/{n_years}")
