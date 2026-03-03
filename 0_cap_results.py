"""Resultados completos Estrategia C por regimen"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
from collections import Counter, defaultdict

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
tickers = [s['symbol'] for s in sp500]

profiles = pd.read_sql(
    "SELECT symbol, industry, sector FROM fmp_profiles WHERE symbol IN ('"
    + "','".join(tickers) + "')", engine)
ticker_to_ind = dict(zip(profiles['symbol'], profiles['industry']))
all_tickers = profiles['symbol'].tolist()
tlist = "','".join(all_tickers)

# Subsectores
LONG_SUBS = {
    'Oil & Gas Exploration & Production', 'Oil & Gas Refining & Marketing',
    'Oil & Gas Equipment & Services', 'Oil & Gas Midstream',
    'Residential Construction', 'Semiconductors',
    'Auto - Manufacturers', 'Auto - Parts',
    'Specialty Retail', 'Software - Infrastructure',
    'Communication Equipment', 'Computer Hardware',
    'Electronic Gaming & Multimedia', 'Engineering & Construction',
    'Restaurants',
}
SHORT_SUBS = {
    'Grocery Stores', 'Tobacco', 'Food Confectioners', 'Packaged Foods',
    'Household & Personal Products', 'Drug Manufacturers - General',
    'Beverages - Non-Alcoholic', 'Discount Stores',
    'Medical - Distribution', 'Regulated Water', 'Regulated Gas',
    'Agricultural Farm Products',
}

# Regimenes jueves
df_thu = pd.read_csv('data/regimenes_jueves.csv')
df_thu['fecha_senal'] = pd.to_datetime(df_thu['fecha_senal'])

# SPY
spy = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()
spy_dates = set(spy.index.tolist())

import sys
REGIME = sys.argv[1] if len(sys.argv) > 1 else 'PANICO'
cap_rows = df_thu[df_thu['regime'] == REGIME].copy()
cap_dates = cap_rows['fecha_senal'].tolist()

# Trade periods
trade_periods = []
for thu in cap_dates:
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
        trade_periods.append((thu, fri_entry, fri_exit))

print(f'ESTRATEGIA C SUBSECTORES - REGIMEN: {REGIME}')
print(f'LONG: ciclicos oversold | SHORT: defensivos baja beta')
print(f'Semanas {REGIME}: {len(cap_dates)} | Periodos validos: {len(trade_periods)}')
print('=' * 85)

results = []
long_counter = Counter()
short_counter = Counter()

for i, (thu, entry, exit_d) in enumerate(trade_periods):
    cap_info = cap_rows[cap_rows['fecha_senal'] == thu].iloc[0]
    vix = cap_info['vix']

    dd_start = thu - pd.Timedelta(days=380)
    hist = pd.read_sql(f"""
        SELECT symbol, date, close, high FROM fmp_price_history
        WHERE symbol IN ('{tlist}')
        AND date BETWEEN '{dd_start.strftime('%Y-%m-%d')}' AND '{thu.strftime('%Y-%m-%d')}'
        ORDER BY symbol, date
    """, engine)
    hist['date'] = pd.to_datetime(hist['date'])
    max_52w = hist.groupby('symbol')['high'].max()
    last_close = hist.sort_values('date').groupby('symbol')['close'].last()
    common = max_52w.index.intersection(last_close.index)
    drawdown = (last_close[common] / max_52w[common] - 1) * 100

    p_entry = pd.read_sql(f"""
        SELECT symbol, open FROM fmp_price_history
        WHERE date = '{entry.strftime('%Y-%m-%d')}' AND symbol IN ('{tlist}')
    """, engine).set_index('symbol')['open']
    p_exit = pd.read_sql(f"""
        SELECT symbol, open FROM fmp_price_history
        WHERE date = '{exit_d.strftime('%Y-%m-%d')}' AND symbol IN ('{tlist}')
    """, engine).set_index('symbol')['open']

    tradeable = p_entry.index.intersection(p_exit.index).intersection(drawdown.index)
    ret = (p_exit[tradeable] / p_entry[tradeable] - 1) * 100
    dd = drawdown[tradeable]
    spy_ret = (spy.loc[exit_d, 'open'] / spy.loc[entry, 'open'] - 1) * 100

    long_pool = [s for s in tradeable if ticker_to_ind.get(s) in LONG_SUBS]
    if len(long_pool) >= 5:
        long_syms = dd[long_pool].sort_values(ascending=True).head(5).index.tolist()
    else:
        long_syms = dd[list(tradeable)].sort_values(ascending=True).head(5).index.tolist()
    long_ret = ret[long_syms].mean()

    short_pool = [s for s in tradeable if ticker_to_ind.get(s) in SHORT_SUBS]
    if len(short_pool) >= 5:
        short_syms = dd[short_pool].sort_values(ascending=False).head(5).index.tolist()
    else:
        short_syms = dd[list(tradeable)].sort_values(ascending=False).head(5).index.tolist()
    short_ret = -ret[short_syms].mean()

    total = (long_ret + short_ret) / 2

    for s in long_syms:
        long_counter[s] += 1
    for s in short_syms:
        short_counter[s] += 1

    results.append({
        'fecha': thu, 'vix': vix, 'spy_ret': spy_ret,
        'long_ret': long_ret, 'short_ret': short_ret, 'total': total,
        'long_syms': long_syms, 'short_syms': short_syms,
    })

# TABLA SEMANA A SEMANA
cap = 100000
peak = cap
max_dd = 0
print(f'\n{"#":>2} {"Fecha":>11} {"VIX":>4} {"Long":>7} {"Short":>7} {"Total":>7} {"SPY":>7} {"vsSpY":>7} {"Equity":>10}')
print('-' * 75)
for i, r in enumerate(results):
    cap *= (1 + r['total'] / 100)
    if cap > peak:
        peak = cap
    dd = (cap - peak) / peak * 100
    if dd < max_dd:
        max_dd = dd
    vs = r['total'] - r['spy_ret']
    w = '*' if r['total'] > 0 else ' '
    print(f'{i+1:>2} {r["fecha"].strftime("%Y-%m-%d"):>11} {r["vix"]:>4.0f} '
          f'{r["long_ret"]:>+6.1f}% {r["short_ret"]:>+6.1f}% {r["total"]:>+6.1f}% '
          f'{r["spy_ret"]:>+6.1f}% {vs:>+6.1f}% ${cap:>9,.0f}{w}')
print('-' * 75)

# RESUMEN
t = np.array([r['total'] for r in results])
l = np.array([r['long_ret'] for r in results])
s = np.array([r['short_ret'] for r in results])
sp = np.array([r['spy_ret'] for r in results])

cap_final = 100000
for r in t:
    cap_final *= (1 + r / 100)
cap_spy = 100000
for r in sp:
    cap_spy *= (1 + r / 100)

print(f'\n{"="*60}')
print(f'RESUMEN {REGIME}')
print(f'{"="*60}')
print(f'  Semanas operadas:     {len(t)}')
print(f'  Capital: $100,000 -> ${cap_final:,.0f} (SPY: ${cap_spy:,.0f})')
print(f'  Rentabilidad:         {(cap_final/100000-1)*100:+.1f}% (SPY: {(cap_spy/100000-1)*100:+.1f}%)')
print(f'  Avg semanal:          {t.mean():+.2f}%')
print(f'  Mediana:              {np.median(t):+.2f}%')
print(f'  Win Rate:             {(t>0).mean()*100:.1f}%')
sh = t.mean()/t.std()*np.sqrt(52) if t.std() > 0 else 0
print(f'  Sharpe:               {sh:.2f}')
print(f'  Mejor:                {t.max():+.2f}%')
print(f'  Peor:                 {t.min():+.2f}%')
print(f'  Max Drawdown:         {max_dd:.1f}%')
print(f'  Long avg:             {l.mean():+.2f}% (WR {(l>0).mean()*100:.0f}%)')
print(f'  Short avg:            {s.mean():+.2f}% (WR {(s>0).mean()*100:.0f}%)')

# PROTECCION
neg = sp < 0
if neg.sum() > 0:
    print(f'\n  PROTECCION (SPY<0): {neg.sum()} semanas')
    print(f'    SPY:     {sp[neg].mean():+.2f}%')
    print(f'    C:       {t[neg].mean():+.2f}%')
    print(f'    Ventaja: {t[neg].mean()-sp[neg].mean():+.2f}%')
    print(f'    C>0:     {(t[neg]>0).sum()}/{neg.sum()}')

# POR AÑO
print(f'\n{"Ano":>6} {"N":>3} {"C Total":>9} {"SPY Tot":>9} {"C Avg":>8} {"WR":>6} {"Mejor":>8} {"Peor":>8}')
print('-' * 60)
yearly = defaultdict(list)
yearly_spy = defaultdict(list)
for r in results:
    y = r['fecha'].year
    yearly[y].append(r['total'])
    yearly_spy[y].append(r['spy_ret'])
for y in sorted(yearly.keys()):
    a = np.array(yearly[y])
    sa = np.array(yearly_spy[y])
    print(f'{y:>6} {len(a):>3} {a.sum():>+8.1f}% {sa.sum():>+8.1f}% '
          f'{a.mean():>+7.2f}% {(a>0).mean()*100:>5.0f}% {a.max():>+7.1f}% {a.min():>+7.1f}%')

# POR VIX
print(f'\nPor nivel VIX:')
vix_arr = np.array([r['vix'] for r in results])
for lo, hi, label in [(15,25,'VIX 15-25'), (25,35,'VIX 25-35'), (35,50,'VIX 35-50'), (50,100,'VIX 50+')]:
    mask = (vix_arr >= lo) & (vix_arr < hi)
    if mask.sum() > 0:
        print(f'  {label:<12} N:{mask.sum():>2}  Avg:{t[mask].mean():>+5.2f}%  '
              f'WR:{(t[mask]>0).mean()*100:>5.1f}%  Total:{t[mask].sum():>+6.1f}%')

# ACCIONES MAS FRECUENTES
print(f'\nLONG mas frecuentes:')
for sym, count in long_counter.most_common(10):
    ind = ticker_to_ind.get(sym, '?')
    print(f'  {sym:<6} {count:>2}x  {ind[:35]}')
print(f'\nSHORT mas frecuentes:')
for sym, count in short_counter.most_common(10):
    ind = ticker_to_ind.get(sym, '?')
    print(f'  {sym:<6} {count:>2}x  {ind[:35]}')
