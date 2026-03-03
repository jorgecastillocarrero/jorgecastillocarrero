"""Backtest CAPITULACION - Estrategia C: por subsectores
LONG: subsectores ciclicos que mas rebotan en CAPITULACION
SHORT: subsectores defensivos baja beta
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
ticker_to_sector = dict(zip(profiles['symbol'], profiles['sector']))
ticker_to_ind = dict(zip(profiles['symbol'], profiles['industry']))
all_tickers = profiles['symbol'].tolist()
tlist = "','".join(all_tickers)

# Subsectores para LONG: ciclicos que mas rebotan
LONG_SUBS = {
    'Oil & Gas Exploration & Production',
    'Oil & Gas Refining & Marketing',
    'Oil & Gas Equipment & Services',
    'Oil & Gas Midstream',
    'Residential Construction',
    'Semiconductors',
    'Auto - Manufacturers',
    'Auto - Parts',
    'Specialty Retail',
    'Software - Infrastructure',
    'Communication Equipment',
    'Computer Hardware',
    'Electronic Gaming & Multimedia',
    'Engineering & Construction',
    'Restaurants',
}

# Subsectores para SHORT: defensivos baja beta
SHORT_SUBS = {
    'Grocery Stores',
    'Tobacco',
    'Food Confectioners',
    'Packaged Foods',
    'Household & Personal Products',
    'Drug Manufacturers - General',
    'Beverages - Non-Alcoholic',
    'Discount Stores',
    'Medical - Distribution',
    'Regulated Water',
    'Regulated Gas',
    'Agricultural Farm Products',
}

# Fechas CAPITULACION
df_thu = pd.read_csv('data/regimenes_jueves.csv')
df_thu['fecha_senal'] = pd.to_datetime(df_thu['fecha_senal'])
cap_rows = df_thu[df_thu['regime'] == 'CAPITULACION'].copy()
cap_dates = cap_rows['fecha_senal'].tolist()

# SPY
spy = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()
spy_dates = set(spy.index.tolist())

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

print(f'BACKTEST CAPITULACION - Estrategia C: Subsectores')
print(f'LONG: {len(LONG_SUBS)} subsectores ciclicos (Oil&Gas, Auto, Semis, Retail...)')
print(f'SHORT: {len(SHORT_SUBS)} subsectores defensivos baja beta (Grocery, Tobacco, Pharma...)')
print(f'Seleccion: 5 mas oversold del grupo LONG, 5 mas resistentes del grupo SHORT')
print(f'Periodos: {len(trade_periods)}')
print('=' * 140)

results = []
for i, (thu, entry, exit_d) in enumerate(trade_periods):
    cap_info = cap_rows[cap_rows['fecha_senal'] == thu].iloc[0]
    vix = cap_info['vix']

    # Drawdown 52w
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

    # Precios entry/exit
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

    # LONG 5: mas oversold de subsectores ciclicos
    long_pool = [s for s in tradeable if ticker_to_ind.get(s) in LONG_SUBS]
    if len(long_pool) >= 5:
        long_syms = dd[long_pool].sort_values(ascending=True).head(5).index.tolist()
    else:
        long_syms = dd[list(tradeable)].sort_values(ascending=True).head(5).index.tolist()
    long_ret = ret[long_syms].mean()

    # SHORT 5: mas resistentes de subsectores defensivos
    short_pool = [s for s in tradeable if ticker_to_ind.get(s) in SHORT_SUBS]
    if len(short_pool) >= 5:
        short_syms = dd[short_pool].sort_values(ascending=False).head(5).index.tolist()
    else:
        short_syms = dd[list(tradeable)].sort_values(ascending=False).head(5).index.tolist()
    short_ret = -ret[short_syms].mean()

    total = (long_ret + short_ret) / 2

    results.append({
        'fecha': thu, 'vix': vix, 'spy_ret': spy_ret,
        'long_ret': long_ret, 'short_ret': short_ret, 'total': total,
        'long_syms': long_syms, 'short_syms': short_syms,
    })

    ls = ' '.join([f'{s}({ticker_to_ind.get(s, "?")[:15]})' for s in long_syms])
    ss = ' '.join([f'{s}({ticker_to_ind.get(s, "?")[:15]})' for s in short_syms])
    w = '+' if total > 0 else '-'
    print(f'{i+1:>2}. {thu.strftime("%Y-%m-%d")} VIX:{vix:>4.0f} | '
          f'L:{long_ret:>+6.1f}% S:{short_ret:>+6.1f}% = {w}{abs(total):>5.1f}% | SPY:{spy_ret:>+6.1f}% | '
          f'L:[{ls[:70]}] S:[{ss[:70]}]')

df_r = pd.DataFrame(results)

print()
print('=' * 70)
print('RESUMEN ESTRATEGIA C - SUBSECTORES')
print('=' * 70)
print(f'Semanas: {len(df_r)}')
print(f'  Avg:     {df_r["total"].mean():+.2f}%')
print(f'  Median:  {df_r["total"].median():+.2f}%')
print(f'  WR:      {(df_r["total"] > 0).mean() * 100:.1f}%')
print(f'  Best:    {df_r["total"].max():+.2f}%')
print(f'  Worst:   {df_r["total"].min():+.2f}%')
print(f'  Total:   {df_r["total"].sum():+.1f}%')
sharpe = df_r["total"].mean() / df_r["total"].std() * np.sqrt(52) if df_r["total"].std() > 0 else 0
print(f'  Sharpe:  {sharpe:.2f}')
print(f'  L avg:   {df_r["long_ret"].mean():+.2f}%  S avg: {df_r["short_ret"].mean():+.2f}%')
print(f'  SPY avg: {df_r["spy_ret"].mean():+.2f}%')

# Compuesto
capital = 100000
for r in df_r['total']:
    capital *= (1 + r / 100)
cap_spy = 100000
for r in df_r['spy_ret']:
    cap_spy *= (1 + r / 100)
print(f'  $100K -> ${capital:,.0f} (SPY: ${cap_spy:,.0f})')

# Proteccion
neg = df_r['spy_ret'] < 0
if neg.sum() > 0:
    print(f'\n  PROTECCION (SPY<0): {neg.sum()} semanas')
    print(f'    SPY medio:  {df_r.loc[neg, "spy_ret"].mean():+.2f}%')
    print(f'    C medio:    {df_r.loc[neg, "total"].mean():+.2f}%')
    print(f'    Ventaja:    {df_r.loc[neg, "total"].mean() - df_r.loc[neg, "spy_ret"].mean():+.2f}%')
    print(f'    C positivo: {(df_r.loc[neg, "total"] > 0).sum()}/{neg.sum()}')

# Comparativa A vs B vs C
a_rets = [+4.5, +10.3, -1.0, +1.3, +22.0, -1.6, -4.6, +9.2, -3.4, +0.3, +1.0,
          +5.4, -1.1, +0.0, +7.3, +4.4, +3.1, +6.2, +2.4, -6.5, -1.6, +4.2,
          +1.9, +1.9, +6.2, +1.9, -12.2, -14.1, -14.7, +32.8, +13.1, +5.7,
          +3.9, +11.7, +6.2, +19.0, -5.7, +11.5, -0.7, -4.4, +4.6, -3.1, -2.5]
b_rets = [+3.1, +10.3, -1.3, +7.2, +2.8, -2.3, -1.5, +6.7, -0.4, -0.2, +0.6,
          +6.2, +1.6, +0.1, -0.8, +1.8, -1.3, +4.6, -2.2, -6.7, -1.3, +1.1,
          -0.6, +1.6, +7.2, +1.4, -3.8, -4.7, -8.8, +4.9, +3.1, +1.7,
          +2.5, +17.4, +3.1, +17.7, -10.7, +12.8, +0.2, -1.3, +2.0, -4.1, -1.2]
c_rets = df_r['total'].tolist()
spy_r = df_r['spy_ret'].tolist()

a = np.array(a_rets)
b = np.array(b_rets)
c = np.array(c_rets)
s = np.array(spy_r)

print()
print('=' * 80)
print('COMPARATIVA FINAL: A vs B vs C')
print('=' * 80)
print(f'{"Metrica":<20} {"A:Baseline":>12} {"B:Sector":>12} {"C:Subsector":>12} {"SPY":>12}')
print('-' * 72)
print(f'{"Avg %":<20} {a.mean():>+11.2f}% {b.mean():>+11.2f}% {c.mean():>+11.2f}% {s.mean():>+11.2f}%')
print(f'{"Median %":<20} {np.median(a):>+11.2f}% {np.median(b):>+11.2f}% {np.median(c):>+11.2f}% {np.median(s):>+11.2f}%')
print(f'{"WR %":<20} {(a>0).mean()*100:>10.1f}% {(b>0).mean()*100:>10.1f}% {(c>0).mean()*100:>10.1f}% {(s>0).mean()*100:>10.1f}%')
print(f'{"Worst %":<20} {a.min():>+11.2f}% {b.min():>+11.2f}% {c.min():>+11.2f}% {s.min():>+11.2f}%')
print(f'{"Total acum %":<20} {a.sum():>+11.1f}% {b.sum():>+11.1f}% {c.sum():>+11.1f}% {s.sum():>+11.1f}%')

sha = a.mean()/a.std()*np.sqrt(52) if a.std()>0 else 0
shb = b.mean()/b.std()*np.sqrt(52) if b.std()>0 else 0
shc = c.mean()/c.std()*np.sqrt(52) if c.std()>0 else 0
print(f'{"Sharpe":<20} {sha:>11.2f}  {shb:>11.2f}  {shc:>11.2f}')

for name, arr in [('A', a), ('B', b), ('C', c), ('SPY', s)]:
    cap = 100000
    for r in arr:
        cap *= (1 + r / 100)
    print(f'  $100K {name}: ${cap:,.0f}')

neg_mask = s < 0
if neg_mask.sum() > 0:
    print(f'\n  Proteccion SPY<0:')
    print(f'    A: {a[neg_mask].mean():+.2f}%  B: {b[neg_mask].mean():+.2f}%  C: {c[neg_mask].mean():+.2f}%  SPY: {s[neg_mask].mean():+.2f}%')
