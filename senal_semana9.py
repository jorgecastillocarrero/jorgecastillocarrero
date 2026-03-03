"""
Senal para semana 9 de 2026: regimen de mercado + top 10 acciones S&P 500
Datos al viernes 21 feb 2026 -> compra lunes 24 feb open -> venta lunes 3 mar open
"""
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# ================================================================
# 1. REGIMEN DE MERCADO (viernes 21 feb 2026)
# ================================================================
print("=" * 80)
print("  SENAL SEMANAL - SEMANA 9 (compra lunes 24/02, venta lunes 03/03)")
print("=" * 80)

from sector_event_map import SUBSECTORS

ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

# Subsectores para regimen
df_sub = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2024-01-01' AND '2026-02-21' ORDER BY symbol, date
""", engine)
df_sub['date'] = pd.to_datetime(df_sub['date'])
df_sub['subsector'] = df_sub['symbol'].map(ticker_to_sub)
df_sub = df_sub.dropna(subset=['subsector'])
df_sub['week'] = df_sub['date'].dt.isocalendar().week.astype(int)
df_sub['year'] = df_sub['date'].dt.year

sub_weekly = df_sub.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
sub_weekly = sub_weekly.sort_values(['symbol', 'date'])
sub_agg = sub_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'), avg_low=('low', 'mean')).reset_index()
sub_agg = sub_agg.sort_values(['subsector', 'date'])

# Necesitamos 52 semanas de datos para metricas, cargar desde 2000
df_sub_full = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21' ORDER BY symbol, date
""", engine)
df_sub_full['date'] = pd.to_datetime(df_sub_full['date'])
df_sub_full['subsector'] = df_sub_full['symbol'].map(ticker_to_sub)
df_sub_full = df_sub_full.dropna(subset=['subsector'])
df_sub_full['week'] = df_sub_full['date'].dt.isocalendar().week.astype(int)
df_sub_full['year'] = df_sub_full['date'].dt.year

sub_w_full = df_sub_full.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
sub_agg_full = sub_w_full.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'), avg_low=('low', 'mean')).reset_index()
sub_agg_full = sub_agg_full.sort_values(['subsector', 'date'])

date_counts = sub_agg_full.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_agg_full = sub_agg_full[sub_agg_full['date'].isin(valid_dates)]

def calc_price_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

sub_agg_full = sub_agg_full.groupby('subsector', group_keys=False).apply(calc_price_metrics)
dd_wide = sub_agg_full.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_agg_full.pivot(index='date', columns='subsector', values='rsi_14w')

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

# VIX
vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
    skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

# Clasificar regimen para viernes 21 feb 2026
target_date = pd.Timestamp('2026-02-20')  # viernes mas cercano
prev_dates = dd_wide.index[dd_wide.index <= target_date]
last_date = prev_dates[-1]
print(f"\n  Fecha senal: {last_date.strftime('%Y-%m-%d')} (viernes)")

dd_row = dd_wide.loc[last_date]
rsi_row = rsi_wide.loc[last_date]
n_total = dd_row.notna().sum()
pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
pct_dd_deep = (dd_row < -20).sum() / n_total * 100
pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100

spy_last = spy_w.loc[spy_w.index[spy_w.index <= target_date][-1]]
spy_above_ma200 = spy_last['above_ma200']
spy_mom_10w = spy_last['mom_10w'] if pd.notna(spy_last['mom_10w']) else 0
spy_dist = spy_last['dist_ma200'] if pd.notna(spy_last['dist_ma200']) else 0
spy_close = spy_last['close']

vix_dates = vix_df.index[vix_df.index <= target_date]
vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
if not pd.notna(vix_val): vix_val = 20

# Scoring
if pct_dd_healthy >= 75: s1 = 2.0
elif pct_dd_healthy >= 60: s1 = 1.0
elif pct_dd_healthy >= 45: s1 = 0.0
elif pct_dd_healthy >= 30: s1 = -1.0
elif pct_dd_healthy >= 15: s1 = -2.0
else: s1 = -3.0

if pct_rsi_above55 >= 75: s2 = 2.0
elif pct_rsi_above55 >= 60: s2 = 1.0
elif pct_rsi_above55 >= 45: s2 = 0.0
elif pct_rsi_above55 >= 30: s2 = -1.0
elif pct_rsi_above55 >= 15: s2 = -2.0
else: s2 = -3.0

if pct_dd_deep <= 5: s3 = 1.5
elif pct_dd_deep <= 15: s3 = 0.5
elif pct_dd_deep <= 30: s3 = -0.5
elif pct_dd_deep <= 50: s3 = -1.5
else: s3 = -2.5

if spy_above_ma200 and spy_dist > 5: s4 = 1.5
elif spy_above_ma200: s4 = 0.5
elif spy_dist > -5: s4 = -0.5
elif spy_dist > -15: s4 = -1.5
else: s4 = -2.5

if spy_mom_10w > 5: s5 = 1.0
elif spy_mom_10w > 0: s5 = 0.5
elif spy_mom_10w > -5: s5 = -0.5
elif spy_mom_10w > -15: s5 = -1.0
else: s5 = -1.5

total = s1 + s2 + s3 + s4 + s5
is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)
if is_burbuja: regime = 'BURBUJA'
elif total >= 7.0: regime = 'GOLDILOCKS'
elif total >= 4.0: regime = 'ALCISTA'
elif total >= 0.5: regime = 'NEUTRAL'
elif total >= -2.0: regime = 'CAUTIOUS'
elif total >= -5.0: regime = 'BEARISH'
elif total >= -9.0: regime = 'CRISIS'
else: regime = 'PANICO'

if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'): regime = 'NEUTRAL'
elif vix_val >= 35 and regime == 'NEUTRAL': regime = 'CAUTIOUS'

print(f"\n  REGIMEN DE MERCADO: {regime}")
print(f"  Score total: {total:+.1f}")
print(f"  Componentes: BDD={s1:+.1f} BRSI={s2:+.1f} DDP={s3:+.1f} SPY={s4:+.1f} MOM={s5:+.1f}")
print(f"\n  Indicadores:")
print(f"    SPY: ${spy_close:.2f}  dist MA200: {spy_dist:+.1f}%  mom 10w: {spy_mom_10w:+.1f}%  {'> MA200' if spy_above_ma200 else '< MA200'}")
print(f"    VIX: {vix_val:.1f}")
print(f"    Breadth DD healthy (>-10%): {pct_dd_healthy:.0f}%")
print(f"    Breadth DD deep (<-20%): {pct_dd_deep:.0f}%")
print(f"    Breadth RSI > 55: {pct_rsi_above55:.0f}%")

# Estrategia segun regimen
print(f"\n  ESTRATEGIA: ", end="")
if regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
    print(f"LONG - Comprar acciones con mejor momentum y fundamentales")
    strategy = 'LONG_MOMENTUM'
elif regime == 'NEUTRAL':
    print(f"LONG SELECTIVO - Solo comprar acciones en soporte extremo (oversold deep)")
    strategy = 'LONG_OVERSOLD'
elif regime == 'CAUTIOUS':
    print(f"LONG SELECTIVO - Solo comprar acciones en soporte extremo (oversold deep)")
    strategy = 'LONG_OVERSOLD'
elif regime in ('BEARISH', 'CRISIS', 'PANICO'):
    print(f"SHORT - Vender en corto acciones debiles")
    strategy = 'SHORT'

# ================================================================
# 2. SELECCION DE ACCIONES S&P 500
# ================================================================
print(f"\n{'='*80}")
print(f"  TOP 10 ACCIONES S&P 500 - Regimen: {regime}")
print(f"{'='*80}")

# Cargar S&P 500 constituents
with open('C:/Users/usuario/financial-data-project/data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
sp500_tickers = [s['symbol'] for s in sp500]
print(f"\n  S&P 500 constituents: {len(sp500_tickers)} acciones")

# Ticker -> sector/industry del JSON
ticker_info = {s['symbol']: {'name': s.get('name',''), 'sector': s.get('sector',''),
               'industry': s.get('subSector', s.get('industry',''))} for s in sp500}

# Cargar precios de acciones S&P 500 (ultimo ano para metricas)
sp_list = "','".join(sp500_tickers)
print("  Cargando precios S&P 500...")
df_sp = pd.read_sql(f"""
    SELECT symbol, date, open, close, high, low, volume
    FROM fmp_price_history WHERE symbol IN ('{sp_list}')
    AND date BETWEEN '2025-01-01' AND '2026-02-21' ORDER BY symbol, date
""", engine)
df_sp['date'] = pd.to_datetime(df_sp['date'])

# Cargar tambien datos de 52 semanas para high_52w
df_sp_52w = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history WHERE symbol IN ('{sp_list}')
    AND date BETWEEN '2024-02-01' AND '2026-02-21' ORDER BY symbol, date
""", engine)
df_sp_52w['date'] = pd.to_datetime(df_sp_52w['date'])

# Calcular metricas por accion
print("  Calculando metricas...")
stock_metrics = []

for ticker in sp500_tickers:
    df_t = df_sp[df_sp['symbol'] == ticker].sort_values('date')
    df_t52 = df_sp_52w[df_sp_52w['symbol'] == ticker].sort_values('date')
    if len(df_t) < 20 or len(df_t52) < 100:
        continue

    last = df_t.iloc[-1]
    close = last['close']

    # High 52 semanas
    high_52w = df_t52['high'].rolling(252, min_periods=126).max().iloc[-1]
    dd_52w = (close / high_52w - 1) * 100 if pd.notna(high_52w) and high_52w > 0 else 0

    # RSI 14 dias
    delta = df_t['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss_s = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean().iloc[-1]
    avg_loss = loss_s.rolling(14, min_periods=7).mean().iloc[-1]
    if pd.notna(avg_gain) and pd.notna(avg_loss) and avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))
    else:
        rsi_14 = 50

    # Momentum: retorno 1 mes, 3 meses
    ret_1m = (close / df_t['close'].iloc[-22] - 1) * 100 if len(df_t) >= 22 else 0
    ret_3m = (close / df_t['close'].iloc[-66] - 1) * 100 if len(df_t) >= 66 else 0

    # MA50 y MA200
    ma50 = df_t['close'].rolling(50, min_periods=25).mean().iloc[-1]
    ma200 = df_t52['close'].rolling(200, min_periods=100).mean().iloc[-1] if len(df_t52) >= 100 else np.nan
    above_ma50 = close > ma50 if pd.notna(ma50) else False
    above_ma200 = close > ma200 if pd.notna(ma200) else False
    dist_ma200 = (close / ma200 - 1) * 100 if pd.notna(ma200) and ma200 > 0 else 0

    # Volumen relativo (vs media 20 dias)
    vol_avg = df_t['volume'].rolling(20).mean().iloc[-1]
    vol_rel = last['volume'] / vol_avg if pd.notna(vol_avg) and vol_avg > 0 else 1.0

    # ATR% (volatilidad)
    df_t_copy = df_t.copy()
    df_t_copy['hl_range'] = (df_t_copy['high'] - df_t_copy['low']) / df_t_copy['close']
    atr_pct = df_t_copy['hl_range'].rolling(20, min_periods=10).mean().iloc[-1] * 100

    info = ticker_info.get(ticker, {})
    stock_metrics.append({
        'ticker': ticker, 'name': info.get('name', ''),
        'sector': info.get('sector', ''), 'industry': info.get('industry', ''),
        'close': close, 'dd_52w': dd_52w, 'rsi_14': rsi_14,
        'ret_1m': ret_1m, 'ret_3m': ret_3m,
        'above_ma50': above_ma50, 'above_ma200': above_ma200,
        'dist_ma200': dist_ma200, 'vol_rel': vol_rel, 'atr_pct': atr_pct,
    })

df_stocks = pd.DataFrame(stock_metrics)
print(f"  Acciones con datos suficientes: {len(df_stocks)}")

# ================================================================
# 3. RANKING SEGUN ESTRATEGIA
# ================================================================
if strategy == 'LONG_MOMENTUM':
    # Regimen alcista: comprar los mas fuertes
    # Filtros: > MA200, > MA50, RSI > 50, DD > -10% (near highs)
    candidates = df_stocks[
        (df_stocks['above_ma200']) &
        (df_stocks['above_ma50']) &
        (df_stocks['rsi_14'] > 50) &
        (df_stocks['dd_52w'] > -10)
    ].copy()

    # Scoring: momentum + near ATH + RSI fuerte
    candidates['score'] = (
        candidates['ret_3m'].clip(-20, 30) / 30 * 3.0 +       # momentum 3m (peso 3)
        candidates['ret_1m'].clip(-10, 15) / 15 * 2.0 +       # momentum 1m (peso 2)
        (candidates['dd_52w'] + 10) / 10 * 2.0 +               # nearness ATH (peso 2)
        (candidates['rsi_14'] - 50) / 30 * 1.5 +               # RSI fuerte (peso 1.5)
        candidates['dist_ma200'].clip(0, 20) / 20 * 1.5        # dist MA200 (peso 1.5)
    )
    candidates = candidates.sort_values('score', ascending=False)
    label = "LONG MOMENTUM (> MA200, > MA50, RSI > 50, near ATH)"

elif strategy == 'LONG_OVERSOLD':
    # Regimen neutral/cautious: comprar solo oversold profundo
    # Filtros: DD < -15%, RSI < 40, pero no destruido (> MA200 o cerca)
    candidates = df_stocks[
        (df_stocks['dd_52w'] < -15) &
        (df_stocks['rsi_14'] < 40)
    ].copy()

    if len(candidates) < 5:
        # Relajar filtros si no hay suficientes
        candidates = df_stocks[
            (df_stocks['dd_52w'] < -10) &
            (df_stocks['rsi_14'] < 45)
        ].copy()

    # Scoring: cuanto mas oversold + mejor rebote potencial
    candidates['score'] = (
        (-candidates['dd_52w'] - 10) / 20 * 3.0 +            # DD profundo (peso 3)
        (40 - candidates['rsi_14']).clip(0, 30) / 30 * 2.5 +  # RSI oversold (peso 2.5)
        candidates['atr_pct'].clip(0, 5) / 5 * 1.5 +          # volatilidad = recorrido (peso 1.5)
        candidates['vol_rel'].clip(0, 3) / 3 * 1.0            # volumen relativo (peso 1)
    )
    candidates = candidates.sort_values('score', ascending=False)
    label = "LONG OVERSOLD (DD < -15%, RSI < 40)"

elif strategy == 'SHORT':
    # Regimen bear: vender en corto los mas debiles
    # Filtros: < MA200, RSI < 50, DD significativo
    candidates = df_stocks[
        (~df_stocks['above_ma200']) &
        (df_stocks['rsi_14'] < 50) &
        (df_stocks['dd_52w'] < -5)
    ].copy()

    if len(candidates) < 5:
        candidates = df_stocks[
            (df_stocks['rsi_14'] < 50) &
            (df_stocks['dd_52w'] < -5)
        ].copy()

    # Scoring: mas debil = mejor short
    candidates['score'] = (
        (-candidates['dd_52w'] - 5) / 25 * 2.5 +              # DD profundo (peso 2.5)
        (50 - candidates['rsi_14']) / 30 * 2.0 +               # RSI debil (peso 2)
        (-candidates['ret_1m']).clip(0, 20) / 20 * 2.0 +       # momentum negativo 1m (peso 2)
        (-candidates['ret_3m']).clip(0, 30) / 30 * 1.5 +       # momentum negativo 3m (peso 1.5)
        (-candidates['dist_ma200']).clip(0, 20) / 20 * 1.0     # debajo MA200 (peso 1)
    )
    candidates = candidates.sort_values('score', ascending=False)
    label = "SHORT (< MA200, RSI < 50, debilitandose)"

print(f"\n  Estrategia: {label}")
print(f"  Candidatos que pasan filtros: {len(candidates)}")

# Top 10
top10 = candidates.head(10)
print(f"\n  {'#':>3} {'Ticker':<7} {'Nombre':<30} {'Sector':<25} {'Precio':>8} {'DD 52w':>8} {'RSI':>6} "
      f"{'Ret 1m':>8} {'Ret 3m':>8} {'Dist200':>8} {'ATR%':>6} {'Score':>7}")
print(f"  {'-'*145}")

for i, (_, row) in enumerate(top10.iterrows(), 1):
    name = row['name'][:28] if len(str(row['name'])) > 28 else row['name']
    sector = row['sector'][:23] if len(str(row['sector'])) > 23 else row['sector']
    print(f"  {i:>3} {row['ticker']:<7} {name:<30} {sector:<25} ${row['close']:>7.2f} "
          f"{row['dd_52w']:>+7.1f}% {row['rsi_14']:>5.0f} "
          f"{row['ret_1m']:>+7.1f}% {row['ret_3m']:>+7.1f}% "
          f"{row['dist_ma200']:>+7.1f}% {row['atr_pct']:>5.1f} {row['score']:>6.2f}")

# Resumen adicional
print(f"\n  {'='*80}")
print(f"  RESUMEN DE LA SENAL")
print(f"  {'='*80}")
print(f"  Regimen: {regime} (score {total:+.1f})")
print(f"  Accion: {'COMPRAR (LONG)' if strategy != 'SHORT' else 'VENDER EN CORTO (SHORT)'}")
print(f"  Timing: Compra lunes 24/02 open -> Venta lunes 03/03 open")
print(f"  Acciones seleccionadas:")
for i, (_, row) in enumerate(top10.iterrows(), 1):
    print(f"    {i:>2}. {row['ticker']:<6} ${row['close']:>7.2f}  ({row['name']})")

# Diversificacion por sector
print(f"\n  Diversificacion por sector:")
sector_counts = top10['sector'].value_counts()
for sector, count in sector_counts.items():
    print(f"    {sector}: {count} acciones")
