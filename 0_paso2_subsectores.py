"""
Paso 2: Score de subsectores - drawdown 52w + RSI 14w
Solo carga datos recientes (2 anos) para calcular el estado actual
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
tickers = [s['symbol'] for s in sp500]

profiles = pd.read_sql(
    "SELECT symbol, industry, sector FROM fmp_profiles WHERE symbol IN ('"
    + "','".join(tickers) + "')", engine)
sub_map = profiles.groupby('industry')['symbol'].apply(list).to_dict()
sub_map = {k: v for k, v in sub_map.items() if len(v) >= 3 and k is not None}
ticker_to_sub = {}
for sub, tks in sub_map.items():
    for t in tks:
        ticker_to_sub[t] = sub
sub_to_sector = {}
for _, row in profiles.iterrows():
    if row['industry'] in sub_map:
        sub_to_sector[row['industry']] = row['sector']

all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)
n_subs = len(sub_map)
print(f'Subsectores: {n_subs}, Tickers: {len(all_tickers)}')

# Solo ultimos 2 anos (suficiente para 52w drawdown + 14w RSI)
df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2024-01-01' AND '2026-02-28'
    ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['dow'] = df_all['date'].dt.dayofweek
print(f'Registros cargados: {len(df_all):,}')

# Semanal jueves
df_thu = df_all[df_all['dow'] <= 3].copy()
df_thu['iso_year'] = df_thu['date'].dt.isocalendar().year.astype(int)
df_thu['week'] = df_thu['date'].dt.isocalendar().week.astype(int)
df_weekly = df_thu.sort_values('date').groupby(['symbol', 'iso_year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])

# Promedios por subsector
sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

# Calcular drawdown 52w y RSI 14w
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

# Ultima fecha
last_date = sub_weekly['date'].max()
latest = sub_weekly[sub_weekly['date'] == last_date][['subsector', 'drawdown_52w', 'rsi_14w']].copy()
latest['sector'] = latest['subsector'].map(sub_to_sector)
latest = latest.dropna(subset=['drawdown_52w', 'rsi_14w'])

# Indicadores globales para el regimen
n = len(latest)
pct_dd_h = (latest['drawdown_52w'] > -10).sum() / n * 100
pct_dd_d = (latest['drawdown_52w'] < -20).sum() / n * 100
pct_rsi = (latest['rsi_14w'] > 55).sum() / n * 100

print(f'\n{"="*100}')
print(f'PASO 2: ESTADO DE SUBSECTORES - {last_date.strftime("%Y-%m-%d")}')
print(f'{"="*100}')
print(f'  DD saludable (>-10%):  {pct_dd_h:.0f}% de subsectores')
print(f'  DD profundo (<-20%):   {pct_dd_d:.0f}% de subsectores')
print(f'  RSI > 55:              {pct_rsi:.0f}% de subsectores')

# Tabla ordenada por drawdown
latest = latest.sort_values('drawdown_52w', ascending=True)

print(f'\n{"Rank":>4} {"Subsector":<40} {"Sector":<25} {"DD52w":>7} {"RSI14":>6}')
print('-' * 95)
for rank, (_, row) in enumerate(latest.iterrows(), 1):
    dd = row['drawdown_52w']
    rsi = row['rsi_14w']
    tag = ''
    if dd < -20:
        tag = ' << HUNDIDO'
    elif dd < -10:
        tag = ' < DEBIL'
    elif dd > -3:
        tag = ' >> SANO'
    elif dd > -5:
        tag = ' > FUERTE'
    print(f'{rank:>4} {row["subsector"][:39]:<40} {row["sector"][:24]:<25} {dd:>+6.1f}% {rsi:>5.1f}{tag}')

# Resumen por sector
print(f'\n{"="*70}')
print('RESUMEN POR SECTOR')
print(f'{"="*70}')
sec_stats = latest.groupby('sector').agg(
    n=('drawdown_52w', 'count'),
    dd_avg=('drawdown_52w', 'mean'),
    dd_worst=('drawdown_52w', 'min'),
    rsi_avg=('rsi_14w', 'mean'),
).sort_values('dd_avg', ascending=True)

print(f'{"Sector":<25} {"N":>3} {"DD Avg":>8} {"DD Peor":>9} {"RSI Avg":>8}')
print('-' * 60)
for sector, row in sec_stats.iterrows():
    print(f'{sector[:24]:<25} {row["n"]:>3} {row["dd_avg"]:>+7.1f}% {row["dd_worst"]:>+8.1f}% {row["rsi_avg"]:>7.1f}')
