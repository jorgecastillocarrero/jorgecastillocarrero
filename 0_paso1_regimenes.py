"""
Paso 1: Regimenes de mercado - Senal Jueves, Trade Fri->Fri
Muestra SPY return por regimen con todos los indicadores
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# Cargar regimenes jueves
df = pd.read_csv('data/regimenes_jueves.csv')
df['fecha_senal'] = pd.to_datetime(df['fecha_senal'])

# SPY diario
spy = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()
spy_dates = set(spy.index.tolist())

# Calcular retorno Fri open -> Fri open para cada jueves
spy_rets = []
for _, row in df.iterrows():
    thu = row['fecha_senal']

    fri_entry = None
    for d in range(1, 5):
        c = thu + pd.Timedelta(days=d)
        if c in spy_dates:
            fri_entry = c
            break

    fri_exit = None
    for d in range(8, 12):
        c = thu + pd.Timedelta(days=d)
        if c in spy_dates:
            fri_exit = c
            break

    if fri_entry and fri_exit:
        ret = (spy.loc[fri_exit, 'open'] / spy.loc[fri_entry, 'open'] - 1) * 100
        spy_rets.append(ret)
    else:
        spy_rets.append(np.nan)

df['spy_ret'] = spy_rets
df = df.dropna(subset=['spy_ret'])

print('=' * 95)
print('PASO 1: REGIMENES DE MERCADO - SENAL JUEVES')
print('Senal: Jueves close | Trade: Viernes open -> Viernes open siguiente')
print(f'Periodo: {df.fecha_senal.min().strftime("%Y-%m-%d")} a {df.fecha_senal.max().strftime("%Y-%m-%d")}')
print(f'Semanas: {len(df)}')
print('=' * 95)

# Tabla por regimen
regs = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS',
        'BEARISH', 'RECOVERY', 'CRISIS', 'PANICO', 'CAPITULACION']

print(f'\n{"Regimen":<16} {"N":>5} {"%Sem":>6} {"Avg%":>7} {"Med%":>7} {"Std%":>7} {"WR%":>6} {"Best%":>8} {"Worst%":>9} {"Total%":>9}')
print('-' * 95)
for reg in regs:
    mask = df['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = df[mask]['spy_ret']
    n = len(sub)
    pct = n / len(df) * 100
    print(f'{reg:<16} {n:>5} {pct:>5.1f}% {sub.mean():>+6.2f} {sub.median():>+6.2f} '
          f'{sub.std():>6.2f} {(sub > 0).mean() * 100:>5.1f} '
          f'{sub.max():>+7.2f} {sub.min():>+8.2f} {sub.sum():>+8.1f}')

print('-' * 95)
all_r = df['spy_ret']
print(f'{"TOTAL":<16} {len(all_r):>5} {100:>5.1f}% {all_r.mean():>+6.2f} {all_r.median():>+6.2f} '
      f'{all_r.std():>6.2f} {(all_r > 0).mean() * 100:>5.1f} '
      f'{all_r.max():>+7.2f} {all_r.min():>+8.2f} {all_r.sum():>+8.1f}')

# Por zona de accion
print('\n' + '=' * 60)
print('POR ZONA DE ACCION')
print('=' * 60)
zones = {
    'LONG':       ['BURBUJA', 'GOLDILOCKS', 'ALCISTA'],
    'NEUTRAL':    ['NEUTRAL'],
    'CAUTIOUS':   ['CAUTIOUS'],
    'DEFENSIVE':  ['BEARISH', 'RECOVERY'],
    'CRISIS':     ['CRISIS', 'PANICO', 'CAPITULACION'],
}

for zone, zone_regs in zones.items():
    mask = df['regime'].isin(zone_regs)
    sub = df[mask]['spy_ret']
    if len(sub) == 0:
        continue
    print(f'  {zone:<12} N:{len(sub):>4} ({len(sub)/len(df)*100:>4.1f}%)  '
          f'Avg:{sub.mean():>+5.2f}%  WR:{(sub > 0).mean() * 100:>5.1f}%  '
          f'Total:{sub.sum():>+7.1f}%')

# Simulacion compuesta
print('\n' + '=' * 60)
print('SIMULACION $100K: invertir SPY solo en ciertas zonas')
print('=' * 60)

for zone, zone_regs in zones.items():
    cap = 100000
    weeks = 0
    for _, row in df.iterrows():
        if row['regime'] in zone_regs:
            cap *= (1 + row['spy_ret'] / 100)
            weeks += 1
    print(f'  Solo {zone:<12}: $100K -> ${cap:>10,.0f} ({weeks:>4} sem, {(cap / 100000 - 1) * 100:>+6.1f}%)')

# Buy & Hold
cap_bh = 100000
for _, row in df.iterrows():
    cap_bh *= (1 + row['spy_ret'] / 100)
print(f'  Buy & Hold SPY  : $100K -> ${cap_bh:>10,.0f} ({len(df):>4} sem, {(cap_bh / 100000 - 1) * 100:>+6.1f}%)')

# Solo LONG zone
cap_long = 100000
for _, row in df.iterrows():
    if row['regime'] in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA']:
        cap_long *= (1 + row['spy_ret'] / 100)
print(f'  Solo LONG zone  : $100K -> ${cap_long:>10,.0f}')

# Evitar CRISIS
cap_no_crisis = 100000
for _, row in df.iterrows():
    if row['regime'] not in ['CRISIS', 'PANICO', 'CAPITULACION']:
        cap_no_crisis *= (1 + row['spy_ret'] / 100)
print(f'  Evitar CRISIS   : $100K -> ${cap_no_crisis:>10,.0f}')
