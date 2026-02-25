"""
Análisis de Indicadores Complementarios para Clasificación de Régimen
=====================================================================
Evalúa qué indicadores de mercado (además del VIX) ayudan a mejorar
la clasificación del régimen BULLISH/NEUTRAL/BEARISH/CRISIS.

Indicadores evaluados:
1. SPY Trend: SPY vs 200-day MA (tendencia)
2. SPY Momentum: retorno 10 semanas (momentum)
3. Market Breadth: % subsectores con RSI > 50 (amplitud)
4. Credit Spread: LQD vs IEF performance relativo (crédito)
5. Defensive Rotation: defensivos vs cíclicos (rotación)
6. Yield Curve Proxy: TLT vs SHY (curva de tipos)
7. VIX: volatilidad implícita
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0

# ================================================================
# 1. CARGAR DATOS DEL SISTEMA (misma lógica que fair_v3_backtest_full)
# ================================================================
print("=" * 70)
print("ANÁLISIS DE INDICADORES COMPLEMENTARIOS PARA RÉGIMEN")
print("=" * 70)

print("\n1. Cargando datos del sistema...")

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

# Weekly subsector aggregation
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year
df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'),
    avg_high=('high', 'mean'),
    avg_return=('return', 'mean'),
).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

# RSI semanal por subsector (para breadth)
def calc_rsi_weekly(g):
    g = g.sort_values('date').copy()
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    # High 52w y DD
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    return g

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_rsi_weekly)
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')

# Score y bear_ratio del sistema
print("  Calculando scores y bear_ratio semanal...")
weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

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

system_data = []
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
        system_data.append({'date': date, 'bear_ratio': 0.5, 'regime': 'NEUTRAL', 'n_bull': 0, 'n_bear': 0})
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    longs_pool = [s for s, sc in scores_v3.items() if sc > 6.5]
    shorts_pool = [s for s, sc in scores_v3.items() if sc < 3.5]
    bc = len(shorts_pool)
    blc = len(longs_pool)
    br = bc / (bc + blc) if (bc + blc) > 0 else 0.5

    if br >= 0.60:     regime = 'CRISIS'
    elif br >= 0.50:   regime = 'BEARISH'
    elif br >= 0.30:   regime = 'NEUTRAL'
    else:              regime = 'BULLISH'

    system_data.append({'date': date, 'bear_ratio': br, 'regime': regime, 'n_bull': blc, 'n_bear': bc})

df_system = pd.DataFrame(system_data).set_index('date')
print(f"  {len(df_system)} semanas con datos del sistema")

# ================================================================
# 2. CARGAR INDICADORES DE MERCADO
# ================================================================
print("\n2. Cargando indicadores de mercado...")

# --- SPY ---
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()
spy['ma200'] = spy['close'].rolling(200).mean()
spy['above_ma200'] = (spy['close'] > spy['ma200']).astype(int)
spy['spy_dist_ma200'] = (spy['close'] / spy['ma200'] - 1) * 100  # % distance from MA200
spy_weekly = spy.resample('W-FRI').last().dropna(subset=['ma200'])
spy_weekly['ret_spy'] = spy_weekly['close'].pct_change()
spy_weekly['mom_10w'] = spy_weekly['close'].pct_change(10) * 100  # 10-week momentum %
spy_weekly['mom_20w'] = spy_weekly['close'].pct_change(20) * 100  # 20-week momentum %
print(f"  SPY: {spy_weekly.index.min().date()} a {spy_weekly.index.max().date()}")

# --- Credit Spread Proxy: LQD vs IEF ---
credit = pd.read_sql("""
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol IN ('LQD', 'IEF') AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
credit['date'] = pd.to_datetime(credit['date'])
credit_piv = credit.pivot(index='date', columns='symbol', values='close').sort_index()
credit_weekly = credit_piv.resample('W-FRI').last().dropna()
# Ratio LQD/IEF: baja = stress (crédito peor que treasuries)
credit_weekly['lqd_ief_ratio'] = credit_weekly['LQD'] / credit_weekly['IEF']
credit_weekly['credit_mom_4w'] = credit_weekly['lqd_ief_ratio'].pct_change(4) * 100
credit_weekly['credit_z'] = (
    (credit_weekly['lqd_ief_ratio'] - credit_weekly['lqd_ief_ratio'].rolling(52).mean()) /
    credit_weekly['lqd_ief_ratio'].rolling(52).std()
)
print(f"  Credit (LQD/IEF): {credit_weekly.index.min().date()} a {credit_weekly.index.max().date()}")

# --- Defensive vs Cyclical Rotation ---
sectors = pd.read_sql("""
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol IN ('XLP', 'XLU', 'XLV', 'XLY', 'XLK', 'XLI', 'XLE', 'XLF')
    AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
sectors['date'] = pd.to_datetime(sectors['date'])
sect_piv = sectors.pivot(index='date', columns='symbol', values='close').sort_index()
sect_weekly = sect_piv.resample('W-FRI').last().dropna()
# Retorno 4 semanas
for col in sect_weekly.columns:
    sect_weekly[f'{col}_ret4w'] = sect_weekly[col].pct_change(4)

# Ratio defensivos vs cíclicos (4w returns)
def calc_def_cyc(row):
    defensive = np.nanmean([row.get('XLP_ret4w', np.nan), row.get('XLU_ret4w', np.nan), row.get('XLV_ret4w', np.nan)])
    cyclical = np.nanmean([row.get('XLY_ret4w', np.nan), row.get('XLK_ret4w', np.nan), row.get('XLI_ret4w', np.nan)])
    if pd.notna(defensive) and pd.notna(cyclical) and cyclical != 0:
        return (defensive - cyclical) * 100  # positivo = defensivo gana = risk-off
    return np.nan

sect_weekly['def_cyc_spread'] = sect_weekly.apply(calc_def_cyc, axis=1)
print(f"  Sectors: {sect_weekly.index.min().date()} a {sect_weekly.index.max().date()}")

# --- Yield Curve Proxy: TLT vs SHY ---
bonds = pd.read_sql("""
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol IN ('TLT', 'SHY', 'IEF') AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
bonds['date'] = pd.to_datetime(bonds['date'])
bonds_piv = bonds.pivot(index='date', columns='symbol', values='close').sort_index()
bonds_weekly = bonds_piv.resample('W-FRI').last().dropna()
# TLT/SHY ratio: sube = curva se aplana/invierte (flight to quality), baja = curva empina (risk-on)
bonds_weekly['tlt_shy_ratio'] = bonds_weekly['TLT'] / bonds_weekly['SHY']
bonds_weekly['curve_mom_4w'] = bonds_weekly['tlt_shy_ratio'].pct_change(4) * 100
print(f"  Bonds (TLT/SHY): {bonds_weekly.index.min().date()} a {bonds_weekly.index.max().date()}")

# --- VIX ---
vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
print(f"  VIX: {vix_df.index.min().date()} a {vix_df.index.max().date()}")

# --- Market Breadth: % subsectores con RSI > 50 ---
print("  Calculando Market Breadth (% subsectores RSI > 50)...")
breadth = pd.DataFrame(index=rsi_wide.index)
breadth['pct_rsi_above50'] = (rsi_wide > 50).sum(axis=1) / rsi_wide.notna().sum(axis=1) * 100
breadth['pct_rsi_above60'] = (rsi_wide > 60).sum(axis=1) / rsi_wide.notna().sum(axis=1) * 100
breadth['pct_rsi_below40'] = (rsi_wide < 40).sum(axis=1) / rsi_wide.notna().sum(axis=1) * 100
# DD breadth: % subsectores con DD > -10% (cerca de máximos)
breadth['pct_dd_healthy'] = (dd_wide > -10).sum(axis=1) / dd_wide.notna().sum(axis=1) * 100
breadth['pct_dd_deep'] = (dd_wide < -20).sum(axis=1) / dd_wide.notna().sum(axis=1) * 100

# ================================================================
# 3. COMBINAR TODO
# ================================================================
print("\n3. Combinando indicadores...")

# Merge all on date
df = df_system.copy()
df = df.join(spy_weekly[['above_ma200', 'spy_dist_ma200', 'mom_10w', 'mom_20w', 'ret_spy']], how='left')
df = df.join(vix_df[['close']].rename(columns={'close': 'vix'}), how='left')
df = df.join(credit_weekly[['credit_mom_4w', 'credit_z']], how='left')
df = df.join(sect_weekly[['def_cyc_spread']], how='left')
df = df.join(bonds_weekly[['curve_mom_4w']], how='left')
df = df.join(breadth, how='left')

# Forward fill VIX y otros para semanas que no coinciden exactamente
df['vix'] = df['vix'].ffill()
df['above_ma200'] = df['above_ma200'].ffill()
df['spy_dist_ma200'] = df['spy_dist_ma200'].ffill()
df['mom_10w'] = df['mom_10w'].ffill()
df['credit_mom_4w'] = df['credit_mom_4w'].ffill()
df['credit_z'] = df['credit_z'].ffill()
df['def_cyc_spread'] = df['def_cyc_spread'].ffill()
df['curve_mom_4w'] = df['curve_mom_4w'].ffill()

df = df.dropna(subset=['vix', 'above_ma200', 'pct_rsi_above50'])
print(f"  {len(df)} semanas con todos los indicadores")

# ================================================================
# 4. ANÁLISIS POR RÉGIMEN
# ================================================================
print("\n" + "=" * 70)
print("4. MEDIA DE CADA INDICADOR POR RÉGIMEN DEL SISTEMA")
print("=" * 70)

indicators = ['vix', 'above_ma200', 'spy_dist_ma200', 'mom_10w',
              'credit_z', 'def_cyc_spread', 'curve_mom_4w',
              'pct_rsi_above50', 'pct_dd_healthy', 'pct_dd_deep']

regime_order = ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']
print(f"\n{'Indicador':<22} {'BULLISH':>10} {'NEUTRAL':>10} {'BEARISH':>10} {'CRISIS':>10}  {'Separación':>10}")
print("-" * 82)

for ind in indicators:
    means = {}
    for r in regime_order:
        mask = df['regime'] == r
        means[r] = df.loc[mask, ind].mean() if mask.sum() > 0 else np.nan

    # Separación = diferencia entre BULLISH y CRISIS (cuanto mayor, mejor discrimina)
    sep = abs(means.get('CRISIS', 0) - means.get('BULLISH', 0))

    print(f"{ind:<22} {means.get('BULLISH',0):>10.2f} {means.get('NEUTRAL',0):>10.2f} "
          f"{means.get('BEARISH',0):>10.2f} {means.get('CRISIS',0):>10.2f}  {sep:>10.2f}")

# Conteo de semanas
print(f"\n{'N semanas':<22}", end="")
for r in regime_order:
    print(f" {(df['regime']==r).sum():>10d}", end="")
print()

# ================================================================
# 5. MISMATCHES: cuando el sistema falla
# ================================================================
print("\n" + "=" * 70)
print("5. ANÁLISIS DE MISMATCHES (régimen vs realidad SPY)")
print("=" * 70)

# Clasificar SPY return semanal
df['spy_actual'] = 'FLAT'
df.loc[df['ret_spy'] > 0.01, 'spy_actual'] = 'UP'
df.loc[df['ret_spy'] < -0.01, 'spy_actual'] = 'DOWN'
df.loc[df['ret_spy'] < -0.03, 'spy_actual'] = 'CRASH'

# Semanas problemáticas: sistema dice BULLISH pero SPY baja >1%
bull_bad = df[(df['regime'] == 'BULLISH') & (df['ret_spy'] < -0.01)]
crisis_good = df[(df['regime'] == 'CRISIS') & (df['ret_spy'] > 0.01)]

print(f"\nBULLISH pero SPY baja >1%: {len(bull_bad)} semanas")
print(f"  VIX medio: {bull_bad['vix'].mean():.1f}")
print(f"  SPY dist MA200: {bull_bad['spy_dist_ma200'].mean():.1f}%")
print(f"  Momentum 10w: {bull_bad['mom_10w'].mean():.1f}%")
print(f"  Credit z-score: {bull_bad['credit_z'].mean():.2f}")
print(f"  Breadth RSI>50: {bull_bad['pct_rsi_above50'].mean():.1f}%")
print(f"  DD healthy: {bull_bad['pct_dd_healthy'].mean():.1f}%")

print(f"\nCRISIS pero SPY sube >1%: {len(crisis_good)} semanas")
print(f"  VIX medio: {crisis_good['vix'].mean():.1f}")
print(f"  SPY dist MA200: {crisis_good['spy_dist_ma200'].mean():.1f}%")
print(f"  Momentum 10w: {crisis_good['mom_10w'].mean():.1f}%")
print(f"  Credit z-score: {crisis_good['credit_z'].mean():.2f}")
print(f"  Breadth RSI>50: {crisis_good['pct_rsi_above50'].mean():.1f}%")
print(f"  DD healthy: {crisis_good['pct_dd_healthy'].mean():.1f}%")

# ================================================================
# 6. PODER DISCRIMINATORIO DE CADA INDICADOR
# ================================================================
print("\n" + "=" * 70)
print("6. PODER DISCRIMINATORIO: cada indicador vs SPY futuro 4 semanas")
print("=" * 70)

df['spy_fwd_4w'] = df['ret_spy'].rolling(4).sum().shift(-4)

print(f"\n{'Indicador':<22} {'Corr':>8} {'Info ratio':>12}  Interpretación")
print("-" * 75)

for ind in indicators:
    valid = df[[ind, 'spy_fwd_4w']].dropna()
    if len(valid) < 50:
        continue
    corr = valid[ind].corr(valid['spy_fwd_4w'])

    # Split en quintiles y ver diferencia de SPY return
    valid['q'] = pd.qcut(valid[ind], 5, labels=False, duplicates='drop')
    q_means = valid.groupby('q')['spy_fwd_4w'].mean()
    info_ratio = (q_means.iloc[-1] - q_means.iloc[0]) if len(q_means) >= 5 else 0

    # Interpretación
    if abs(corr) > 0.15:
        interp = "FUERTE"
    elif abs(corr) > 0.08:
        interp = "MODERADO"
    else:
        interp = "DÉBIL"

    print(f"{ind:<22} {corr:>8.3f} {info_ratio:>12.4f}  {interp}")

# ================================================================
# 7. COMBINACIÓN ÓPTIMA: qué indicadores corrigen mejor el régimen
# ================================================================
print("\n" + "=" * 70)
print("7. INDICADORES POR AÑO: señal vs régimen del sistema")
print("=" * 70)

# Para cada año, mostrar: régimen dominante del sistema, señal de cada indicador, SPY return
print(f"\n{'Año':>5} {'SPY%':>7} {'Regime':>9} {'VIX':>6} {'MA200':>6} {'Mom10':>7} {'CrdtZ':>7} "
      f"{'DefCyc':>7} {'Brdth':>7} {'DDhlth':>7} {'DDdeep':>7}")
print("-" * 95)

for year in range(2002, 2026):
    ym = df[df.index.year == year]
    if len(ym) == 0:
        continue

    spy_yr = ym['ret_spy'].sum() * 100
    dominant_regime = ym['regime'].mode().iloc[0] if len(ym) > 0 else '?'

    print(f"{year:>5} {spy_yr:>6.1f}% {dominant_regime:>9} "
          f"{ym['vix'].mean():>6.1f} {ym['above_ma200'].mean():>6.2f} "
          f"{ym['mom_10w'].mean():>7.1f} {ym['credit_z'].mean():>7.2f} "
          f"{ym['def_cyc_spread'].mean():>7.2f} {ym['pct_rsi_above50'].mean():>7.1f} "
          f"{ym['pct_dd_healthy'].mean():>7.1f} {ym['pct_dd_deep'].mean():>7.1f}")

# ================================================================
# 8. AÑOS PROBLEMÁTICOS: detalle de indicadores
# ================================================================
print("\n" + "=" * 70)
print("8. AÑOS PROBLEMÁTICOS: sistema vs indicadores")
print("=" * 70)

problem_years = [2002, 2007, 2008, 2011, 2015, 2018, 2020, 2022]
for year in problem_years:
    ym = df[df.index.year == year]
    if len(ym) == 0:
        continue

    spy_yr = ym['ret_spy'].sum() * 100
    regime_counts = ym['regime'].value_counts()

    print(f"\n--- {year} (SPY: {spy_yr:+.1f}%) ---")
    print(f"  Regímenes: {dict(regime_counts)}")
    print(f"  VIX medio: {ym['vix'].mean():.1f} (min {ym['vix'].min():.1f}, max {ym['vix'].max():.1f})")
    print(f"  SPY vs MA200: {ym['above_ma200'].mean()*100:.0f}% del tiempo encima")
    print(f"  Momentum 10w: {ym['mom_10w'].mean():.1f}%")
    print(f"  Credit z: {ym['credit_z'].mean():.2f}")
    print(f"  Def vs Cyc: {ym['def_cyc_spread'].mean():.2f}")
    print(f"  Breadth (RSI>50): {ym['pct_rsi_above50'].mean():.1f}%")
    print(f"  DD healthy: {ym['pct_dd_healthy'].mean():.1f}%, DD deep: {ym['pct_dd_deep'].mean():.1f}%")

    # Señal que debería haber dado
    if spy_yr < -15:
        expected = "CRISIS/BEARISH"
    elif spy_yr < -5:
        expected = "BEARISH/NEUTRAL"
    elif spy_yr > 15:
        expected = "BULLISH"
    else:
        expected = "NEUTRAL/MIXTO"
    print(f"  -> Señal esperada: {expected}")

# ================================================================
# 9. RESUMEN: RANKING DE INDICADORES
# ================================================================
print("\n" + "=" * 70)
print("9. RESUMEN: RANKING DE UTILIDAD PARA RÉGIMEN")
print("=" * 70)

print("""
Los indicadores se evalúan en 3 dimensiones:
- Separación: ¿cuánto difiere entre BULLISH y CRISIS?
- Correlación: ¿predice el retorno futuro de SPY?
- Disponibilidad: ¿tenemos datos desde 2002?

INDICADOR                TIPO           SEPARACIÓN  CORRELACIÓN  DISPONIBILIDAD
---------------------------------------------------------------------------------
""")

# Calculate and rank
ranking = []
for ind in indicators:
    means = {}
    for r in regime_order:
        mask = df['regime'] == r
        means[r] = df.loc[mask, ind].mean() if mask.sum() > 0 else 0
    sep = abs(means.get('CRISIS', 0) - means.get('BULLISH', 0))

    valid = df[[ind, 'spy_fwd_4w']].dropna()
    corr = valid[ind].corr(valid['spy_fwd_4w']) if len(valid) > 50 else 0

    tipo = {
        'vix': 'Volatilidad',
        'above_ma200': 'Tendencia',
        'spy_dist_ma200': 'Tendencia',
        'mom_10w': 'Momentum',
        'credit_z': 'Crédito',
        'def_cyc_spread': 'Rotación',
        'curve_mom_4w': 'Tipos',
        'pct_rsi_above50': 'Amplitud',
        'pct_dd_healthy': 'Amplitud',
        'pct_dd_deep': 'Amplitud',
    }.get(ind, '?')

    ranking.append({
        'ind': ind, 'tipo': tipo, 'sep': sep, 'corr': corr,
    })

ranking.sort(key=lambda x: x['sep'] + abs(x['corr']) * 100, reverse=True)

for r in ranking:
    stars_sep = '*' * min(5, int(r['sep'] / 5))
    stars_corr = '*' * min(5, int(abs(r['corr']) * 30))
    print(f"  {r['ind']:<22} {r['tipo']:<15} {stars_sep:<8} {stars_corr:<8}")

print("""
Leyenda: * = bajo, ** = medio, *** = alto, **** = muy alto, ***** = excelente
""")
