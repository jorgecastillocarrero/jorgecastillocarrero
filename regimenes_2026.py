"""
Calcula y muestra el regimen de mercado semana a semana para 2026,
con todos los indicadores intermedios.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# --- Cargar constituyentes S&P 500 ---
import json
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
symbols = [s['symbol'] for s in sp500]

# --- Subsectores ---
profiles = pd.read_sql("""
    SELECT symbol, sector, industry
    FROM fmp_profiles
    WHERE symbol = ANY(%(syms)s)
""", engine, params={'syms': symbols})
sym_to_sub = dict(zip(profiles['symbol'], profiles['industry']))

# --- Precios semanales subsectores ---
print("Cargando precios...")
prices = pd.read_sql("""
    SELECT symbol, date, close, open
    FROM fmp_price_history
    WHERE symbol = ANY(%(syms)s) AND date BETWEEN '2023-01-01' AND '2026-02-28'
    ORDER BY date
""", engine, params={'syms': symbols})
prices['date'] = pd.to_datetime(prices['date'])
prices['subsector'] = prices['symbol'].map(sym_to_sub)
prices = prices.dropna(subset=['subsector'])

# Subsectores con >= 3 acciones
sub_counts = prices.groupby('subsector')['symbol'].nunique()
valid_subs = sub_counts[sub_counts >= 3].index
prices = prices[prices['subsector'].isin(valid_subs)]

# Precio medio semanal por subsector (viernes close)
weekly = prices.set_index('date').groupby('subsector').resample('W-FRI')['close'].mean().reset_index()
weekly = weekly.rename(columns={'close': 'avg_close'})
weekly = weekly.sort_values(['subsector', 'date'])

# Calcular metricas tecnicas por subsector
def calc_metrics(grp):
    grp = grp.sort_values('date')
    grp['avg_return'] = grp['avg_close'].pct_change()
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

# --- SPY ---
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2023-01-01' AND '2026-02-28'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

# --- VIX ---
vix_df = pd.read_csv('data/vix_weekly.csv', skiprows=3, header=None,
                      names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.set_index('date').sort_index()
vix_df['vix'] = vix_df['close']

# --- Clasificar regimen ---
def classify_regime(date):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        return None
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0:
        return None

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
        spy_above_ma200 = 0.5
        spy_mom_10w = 0
        spy_dist = 0
    if not pd.notna(spy_mom_10w): spy_mom_10w = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scoring
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

    # VIX override
    vix_override = ''
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        vix_override = f' (VIX {vix_val:.1f} -> rebajado de {regime})'
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        vix_override = f' (VIX {vix_val:.1f} -> rebajado de NEUTRAL)'
        regime = 'CAUTIOUS'

    return {
        'date': date,
        'pct_dd_healthy': pct_dd_healthy,
        'pct_dd_deep': pct_dd_deep,
        'pct_rsi_above55': pct_rsi_above55,
        'spy_above_ma200': spy_above_ma200,
        'spy_dist_ma200': spy_dist,
        'spy_mom_10w': spy_mom_10w,
        'vix': vix_val,
        'score_bdd': score_bdd,
        'score_brsi': score_brsi,
        'score_ddp': score_ddp,
        'score_spy': score_spy,
        'score_mom': score_mom,
        'total': total,
        'regime': regime,
        'vix_override': vix_override,
        'n_subsectors': n_total,
    }

# --- Generar para 2026 semana a semana ---
fridays_2026 = pd.date_range('2026-01-02', '2026-02-27', freq='W-FRI')
print(f"\nViernes 2026 disponibles: {len(fridays_2026)}")

results = []
for fri in fridays_2026:
    r = classify_regime(fri)
    if r:
        results.append(r)

# --- Imprimir tabla ---
print(f"\n{'='*140}")
print(f"  REGIMEN DE MERCADO 2026 - SEMANA A SEMANA")
print(f"{'='*140}")
print(f"  {'Sem':>3} {'Viernes':>12} | {'DD_H%':>5} {'DD_D%':>5} {'RSI>55%':>7} | {'SPY>MA':>6} {'Dist%':>6} {'Mom10':>6} {'VIX':>5} | {'BDD':>4} {'BRSI':>5} {'DDP':>4} {'SPY':>4} {'MOM':>4} | {'TOTAL':>6} | {'REGIMEN':>12} | {'Nota'}")
print(f"  {'-'*3} {'-'*12} | {'-'*5} {'-'*5} {'-'*7} | {'-'*6} {'-'*6} {'-'*6} {'-'*5} | {'-'*4} {'-'*5} {'-'*4} {'-'*4} {'-'*4} | {'-'*6} | {'-'*12} | {'-'*20}")

for i, r in enumerate(results):
    week_num = r['date'].isocalendar()[1]
    print(f"  {week_num:>3} {r['date'].strftime('%Y-%m-%d'):>12} | "
          f"{r['pct_dd_healthy']:5.1f} {r['pct_dd_deep']:5.1f} {r['pct_rsi_above55']:7.1f} | "
          f"{'SI' if r['spy_above_ma200'] else 'NO':>6} {r['spy_dist_ma200']:6.1f} {r['spy_mom_10w']:6.1f} {r['vix']:5.1f} | "
          f"{r['score_bdd']:4.1f} {r['score_brsi']:5.1f} {r['score_ddp']:4.1f} {r['score_spy']:4.1f} {r['score_mom']:4.1f} | "
          f"{r['total']:6.1f} | {r['regime']:>12} | {r['vix_override']}")

print(f"\n  LEYENDA:")
print(f"  DD_H% = % subsectores con drawdown > -10% (saludables)")
print(f"  DD_D% = % subsectores con drawdown < -20% (profundos)")
print(f"  RSI>55% = % subsectores con RSI 14w > 55")
print(f"  SPY>MA = SPY por encima de MA200")
print(f"  Dist% = distancia SPY a MA200 en %")
print(f"  Mom10 = momentum SPY 10 semanas en %")
print(f"  VIX = valor VIX semanal")
print(f"  BDD/BRSI/DDP/SPY/MOM = scores individuales")
print(f"  TOTAL = suma scores (rango -12.5 a +8.0)")
print(f"\n  UMBRALES: BURBUJA >= 8.0 (+ DD_H>=85 + RSI>55>=90) | GOLDILOCKS >= 7.0 | ALCISTA >= 4.0")
print(f"            NEUTRAL >= 0.5 | CAUTIOUS >= -2.0 | BEARISH >= -5.0 | CRISIS >= -9.0 | PANICO < -9.0")
print(f"  VIX VETO: VIX>=30 rebaja BURBUJA/GOLDILOCKS/ALCISTA a NEUTRAL | VIX>=35 rebaja NEUTRAL a CAUTIOUS")
