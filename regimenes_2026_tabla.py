"""
Tabla completa de regimenes 2026 semana a semana con todos los datos y calculos.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
symbols = [s['symbol'] for s in sp500]

profiles = pd.read_sql("SELECT symbol, industry FROM fmp_profiles WHERE symbol = ANY(%(syms)s)",
                        engine, params={'syms': symbols})
sym_to_sub = dict(zip(profiles['symbol'], profiles['industry']))

print("Cargando precios...")
prices = pd.read_sql("""
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol = ANY(%(syms)s) AND date BETWEEN '2023-01-01' AND '2026-02-28'
    ORDER BY date
""", engine, params={'syms': symbols})
prices['date'] = pd.to_datetime(prices['date'])
prices['subsector'] = prices['symbol'].map(sym_to_sub)
prices = prices.dropna(subset=['subsector'])
sub_counts = prices.groupby('subsector')['symbol'].nunique()
valid_subs = sub_counts[sub_counts >= 3].index
prices = prices[prices['subsector'].isin(valid_subs)]

weekly = prices.set_index('date').groupby('subsector').resample('W-FRI')['close'].mean().reset_index()
weekly = weekly.rename(columns={'close': 'avg_close'}).sort_values(['subsector', 'date'])

def calc_metrics(grp):
    grp = grp.sort_values('date')
    high_52 = grp['avg_close'].rolling(52, min_periods=10).max()
    grp['drawdown_52w'] = (grp['avg_close'] / high_52 - 1) * 100
    delta = grp['avg_close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    grp['rsi_14w'] = 100 - (100 / (1 + rs))
    return grp

weekly = weekly.groupby('subsector', group_keys=False).apply(calc_metrics)
dd_wide = weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = weekly.pivot(index='date', columns='subsector', values='rsi_14w')

spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2023-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

vix_df = pd.read_csv('data/vix_weekly.csv', skiprows=3, header=None,
                      names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.set_index('date').sort_index()
vix_df['vix'] = vix_df['close']

fridays_2026 = pd.date_range('2026-01-02', '2026-02-27', freq='W-FRI')

rows = []
for fri in fridays_2026:
    prev_dates = dd_wide.index[dd_wide.index <= fri]
    if len(prev_dates) == 0: continue
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: continue

    n_dd_h = int((dd_row > -10).sum())
    n_dd_d = int((dd_row < -20).sum())
    n_rsi_t = int(rsi_row.notna().sum())
    n_rsi_55 = int((rsi_row > 55).sum())
    pct_dd_h = n_dd_h / n_total * 100
    pct_dd_d = n_dd_d / n_total * 100
    pct_rsi = n_rsi_55 / n_rsi_t * 100 if n_rsi_t > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= fri]
    sl = spy_w.loc[spy_dates[-1]]
    spy_c = sl['close']
    spy_ma = sl['ma200']
    spy_ab = sl['above_ma200']
    spy_dist = sl['dist_ma200']
    spy_mom = sl['mom_10w']
    if not pd.notna(spy_mom): spy_mom = 0

    vix_dates = vix_df.index[vix_df.index <= fri]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scores
    if pct_dd_h >= 75: s_bdd = 2.0
    elif pct_dd_h >= 60: s_bdd = 1.0
    elif pct_dd_h >= 45: s_bdd = 0.0
    elif pct_dd_h >= 30: s_bdd = -1.0
    elif pct_dd_h >= 15: s_bdd = -2.0
    else: s_bdd = -3.0

    if pct_rsi >= 75: s_brsi = 2.0
    elif pct_rsi >= 60: s_brsi = 1.0
    elif pct_rsi >= 45: s_brsi = 0.0
    elif pct_rsi >= 30: s_brsi = -1.0
    elif pct_rsi >= 15: s_brsi = -2.0
    else: s_brsi = -3.0

    if pct_dd_d <= 5: s_ddp = 1.5
    elif pct_dd_d <= 15: s_ddp = 0.5
    elif pct_dd_d <= 30: s_ddp = -0.5
    elif pct_dd_d <= 50: s_ddp = -1.5
    else: s_ddp = -2.5

    if spy_ab and spy_dist > 5: s_spy = 1.5
    elif spy_ab: s_spy = 0.5
    elif spy_dist > -5: s_spy = -0.5
    elif spy_dist > -15: s_spy = -1.5
    else: s_spy = -2.5

    if spy_mom > 5: s_mom = 1.0
    elif spy_mom > 0: s_mom = 0.5
    elif spy_mom > -5: s_mom = -0.5
    elif spy_mom > -15: s_mom = -1.0
    else: s_mom = -1.5

    total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

    is_burbuja = (total >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90)
    if is_burbuja: reg = 'BURBUJA'
    elif total >= 7.0: reg = 'GOLDILOCKS'
    elif total >= 4.0: reg = 'ALCISTA'
    elif total >= 0.5: reg = 'NEUTRAL'
    elif total >= -2.0: reg = 'CAUTIOUS'
    elif total >= -5.0: reg = 'BEARISH'
    elif total >= -9.0: reg = 'CRISIS'
    else: reg = 'PANICO'

    vix_note = ''
    if vix_val >= 30 and reg in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        vix_note = f'VIX->NEUTRAL'
        reg = 'NEUTRAL'
    elif vix_val >= 35 and reg == 'NEUTRAL':
        vix_note = f'VIX->CAUTIOUS'
        reg = 'CAUTIOUS'

    rows.append({
        'sem': fri.isocalendar()[1],
        'fecha': fri.strftime('%d/%m'),
        'n': n_total,
        'dd_h': n_dd_h,
        'dd_h_pct': pct_dd_h,
        'dd_d': n_dd_d,
        'dd_d_pct': pct_dd_d,
        'rsi55': n_rsi_55,
        'rsi55_pct': pct_rsi,
        'spy_c': spy_c,
        'spy_ma': spy_ma,
        'spy_ab': 'SI' if spy_ab else 'NO',
        'spy_dist': spy_dist,
        'spy_mom': spy_mom,
        'vix': vix_val,
        's_bdd': s_bdd,
        's_brsi': s_brsi,
        's_ddp': s_ddp,
        's_spy': s_spy,
        's_mom': s_mom,
        'total': total,
        'reg': reg,
        'vix_note': vix_note,
    })

df = pd.DataFrame(rows)

# --- Imprimir tabla con TODOS los datos ---
print(f"\n{'='*160}")
print(f"  TABLA COMPLETA REGIMENES 2026 - CALCULO SEMANA A SEMANA")
print(f"{'='*160}")

# Parte 1: Datos brutos
print(f"\n  PARTE 1: DATOS BRUTOS")
print(f"  {'-'*120}")
h1 = f"  {'Sem':>3} {'Fecha':>5} | {'N':>2} | {'DD>-10':>6} {'%':>5} | {'DD<-20':>6} {'%':>5} | {'RSI>55':>6} {'%':>5} | {'SPY':>7} {'MA200':>7} {'>MA':>3} {'Dist%':>6} {'Mom10':>6} | {'VIX':>5}"
print(h1)
print(f"  {'-'*3} {'-'*5} | {'-'*2} | {'-'*6} {'-'*5} | {'-'*6} {'-'*5} | {'-'*6} {'-'*5} | {'-'*7} {'-'*7} {'-'*3} {'-'*6} {'-'*6} | {'-'*5}")
for r in rows:
    print(f"  {r['sem']:>3} {r['fecha']:>5} | {r['n']:>2} | {r['dd_h']:>3}/66 {r['dd_h_pct']:>5.1f} | {r['dd_d']:>3}/66 {r['dd_d_pct']:>5.1f} | {r['rsi55']:>3}/66 {r['rsi55_pct']:>5.1f} | {r['spy_c']:>7.1f} {r['spy_ma']:>7.1f} {r['spy_ab']:>3} {r['spy_dist']:>+6.1f} {r['spy_mom']:>+6.1f} | {r['vix']:>5.1f}")

# Parte 2: Scores y regimen
print(f"\n  PARTE 2: SCORES Y REGIMEN")
print(f"  {'-'*100}")
h2 = f"  {'Sem':>3} {'Fecha':>5} | {'BDD':>5} | {'BRSI':>5} | {'DDP':>5} | {'SPY':>5} | {'MOM':>5} | {'TOTAL':>6} | {'REGIMEN':>10} | {'Nota'}"
print(h2)
print(f"  {'-'*3} {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*6} | {'-'*10} | {'-'*15}")
for r in rows:
    print(f"  {r['sem']:>3} {r['fecha']:>5} | {r['s_bdd']:>+5.1f} | {r['s_brsi']:>+5.1f} | {r['s_ddp']:>+5.1f} | {r['s_spy']:>+5.1f} | {r['s_mom']:>+5.1f} | {r['total']:>+6.1f} | {r['reg']:>10} | {r['vix_note']}")

# Parte 3: Referencia umbrales
print(f"\n  REFERENCIA UMBRALES:")
print(f"  {'-'*100}")
print(f"  BDD  (% DD>-10%):  >=75->+2.0  >=60->+1.0  >=45->0.0   >=30->-1.0  >=15->-2.0  <15->-3.0")
print(f"  BRSI (% RSI>55):   >=75->+2.0  >=60->+1.0  >=45->0.0   >=30->-1.0  >=15->-2.0  <15->-3.0")
print(f"  DDP  (% DD<-20%):  <=5->+1.5   <=15->+0.5  <=30->-0.5  <=50->-1.5  >50->-2.5")
print(f"  SPY  (vs MA200):   >MA & d>5%->+1.5  >MA & d<=5%->+0.5  <MA & d>-5%->-0.5  <MA & d>-15%->-1.5  <MA & d<=-15%->-2.5")
print(f"  MOM  (mom 10w):    >5%->+1.0   >0%->+0.5   >-5%->-0.5  >-15%->-1.0  <=-15%->-1.5")
print(f"")
print(f"  REGIMENES (por score total):")
print(f"  BURBUJA >=8.0 (+ DD_H>=85% + RSI>55>=90%) | GOLDILOCKS >=7.0 | ALCISTA >=4.0 | NEUTRAL >=0.5")
print(f"  CAUTIOUS >=-2.0 | BEARISH >=-5.0 | CRISIS >=-9.0 | PANICO <-9.0")
print(f"  VIX VETO: VIX>=30 rebaja BURBUJA/GOLDILOCKS/ALCISTA->NEUTRAL | VIX>=35 rebaja NEUTRAL->CAUTIOUS")
