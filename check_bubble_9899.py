"""Verificar si 1998-1999 (burbuja .com) se clasifica como BURBUJA"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

print("Cargando datos 1996-2000...")
df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '1996-01-01' AND '2000-12-31'
    ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year

print(f"Registros: {len(df_all):,}")
print(f"Symbols: {df_all['symbol'].nunique()}")
print(f"Subsectors: {df_all['subsector'].nunique()}")
print(f"Rango fechas: {df_all['date'].min().date()} a {df_all['date'].max().date()}")

# Verificar cobertura por ano
for yr in [1996, 1997, 1998, 1999, 2000]:
    n_sym = df_all[df_all['year'] == yr]['symbol'].nunique()
    n_sub = df_all[df_all['year'] == yr]['subsector'].nunique()
    print(f"  {yr}: {n_sym} symbols, {n_sub} subsectors")

df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

# Filtrar fechas con al menos 30 subsectores (puede haber menos en los 90s)
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 30].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]
print(f"\nFechas validas (>=30 subsectors): {len(valid_dates)}")

def calc_metrics(g):
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

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_metrics)
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

# SPY
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '1996-01-01' AND '2000-12-31'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

print(f"SPY rango: {spy_daily.index.min().date()} a {spy_daily.index.max().date()}")

# Calcular score para cada semana 1998-1999
print("\n" + "=" * 100)
print("SCORES SEMANALES 1998-1999 (config: RSI>55, S4 max1.5, BURB>=8.0)")
print("=" * 100)

dates_98_99 = dd_wide.index[(dd_wide.index >= '1998-01-01') & (dd_wide.index <= '1999-12-31')]

results = []
for date in dates_98_99:
    dd_row = dd_wide.loc[date]
    rsi_row = rsi_wide.loc[date]
    n_total = dd_row.notna().sum()
    if n_total < 20:
        continue

    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    n_rsi = rsi_row.notna().sum()
    pct_rsi55 = (rsi_row > 55).sum() / n_rsi * 100 if n_rsi > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) == 0:
        continue
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_above = spy_last.get('above_ma200', 0)
    spy_mom = spy_last.get('mom_10w', 0)
    spy_dist = spy_last.get('dist_ma200', 0)
    if not pd.notna(spy_mom): spy_mom = 0
    if not pd.notna(spy_dist): spy_dist = 0

    # S1
    if pct_dd_healthy >= 75: s1 = 2.0
    elif pct_dd_healthy >= 60: s1 = 1.0
    elif pct_dd_healthy >= 45: s1 = 0.0
    elif pct_dd_healthy >= 30: s1 = -1.0
    else: s1 = -2.0

    # S2 con RSI>55
    if pct_rsi55 >= 75: s2 = 2.0
    elif pct_rsi55 >= 60: s2 = 1.0
    elif pct_rsi55 >= 45: s2 = 0.0
    elif pct_rsi55 >= 30: s2 = -1.0
    else: s2 = -2.0

    # S3
    if pct_dd_deep <= 5: s3 = 1.5
    elif pct_dd_deep <= 15: s3 = 0.5
    elif pct_dd_deep <= 30: s3 = -0.5
    else: s3 = -1.5

    # S4 max 1.5
    if spy_above and spy_dist > 5: s4 = 1.5
    elif spy_above: s4 = 0.5
    elif spy_dist > -5: s4 = -0.5
    else: s4 = -1.5

    # S5
    if spy_mom > 5: s5 = 1.0
    elif spy_mom > 0: s5 = 0.5
    elif spy_mom > -5: s5 = -0.5
    else: s5 = -1.0

    total = s1 + s2 + s3 + s4 + s5

    if total >= 8.0: regime = 'BURBUJA'
    elif total >= 5.5: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -1.5: regime = 'CAUTIOUS'
    elif total >= -3.0: regime = 'BEARISH'
    else: regime = 'CRISIS'

    results.append({
        'date': date, 'score': total, 'regime': regime,
        's1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5,
        'dd_h': pct_dd_healthy, 'rsi55': pct_rsi55, 'dd_d': pct_dd_deep,
        'spy_dist': spy_dist, 'spy_mom': spy_mom,
        'n_sub': n_total,
    })

df = pd.DataFrame(results)

if len(df) == 0:
    print("No hay datos suficientes para 1998-1999!")
else:
    # Mostrar todas las semanas con score >= 6
    high_scores = df[df['score'] >= 6.0].sort_values('date')
    print(f"\nSemanas con score >= 6.0: {len(high_scores)}")
    print(f"\n{'Fecha':>12} {'Score':>6} {'Regimen':>12} {'S1':>5} {'S2':>5} {'S3':>5} {'S4':>5} {'S5':>5} "
          f"{'DD_h%':>6} {'RSI55%':>7} {'DD_d%':>6} {'SPY_d':>6} {'SPY_m':>6} {'N':>4}")
    print("-" * 110)
    for _, r in high_scores.iterrows():
        print(f"{r['date'].strftime('%Y-%m-%d'):>12} {r['score']:>+5.1f} {r['regime']:>12} "
              f"{r['s1']:>+4.1f} {r['s2']:>+4.1f} {r['s3']:>+4.1f} {r['s4']:>+4.1f} {r['s5']:>+4.1f} "
              f"{r['dd_h']:>5.0f}% {r['rsi55']:>5.0f}% {r['dd_d']:>5.0f}% "
              f"{r['spy_dist']:>+5.1f} {r['spy_mom']:>+5.1f} {int(r['n_sub']):>4}")

    # Resumen por regimen
    print(f"\n{'DISTRIBUCION 1998-1999':>30}")
    print("-" * 40)
    rc = df['regime'].value_counts()
    for reg in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'CRISIS']:
        n = rc.get(reg, 0)
        print(f"  {reg:<12} {n:>5} ({n/len(df)*100:.1f}%)")

    # Score maximo
    max_row = df.loc[df['score'].idxmax()]
    print(f"\n  Score MAXIMO: {max_row['score']:+.1f} el {max_row['date'].strftime('%Y-%m-%d')}")
    print(f"  S1={max_row['s1']:+.1f} S2={max_row['s2']:+.1f} S3={max_row['s3']:+.1f} "
          f"S4={max_row['s4']:+.1f} S5={max_row['s5']:+.1f}")
