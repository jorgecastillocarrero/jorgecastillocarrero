"""
Semana 1 de 2026 - Proceso completo detallado:
1. Senal: Viernes 26/12/2025 -> Viernes 02/01/2026
2. Con datos del viernes 02/01/2026, calcular regimen
3. Trading: Lunes 05/01/2026 open -> Lunes 12/01/2026 open
4. Rentabilidad SPY real del trade
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

print("="*80)
print("  SEMANA 1 - 2026 - PROCESO COMPLETO")
print("="*80)

print(f"""
  FLUJO TEMPORAL:
  -----------------------------------------------------------------------
  Senal:   Vie 26/12/2025 close -> Vie 02/01/2026 close
           (con estos datos se calcula regimen y se genera la senal)
  Trading: Lun 05/01/2026 open  -> Lun 12/01/2026 open
           (se ejecuta el trade con el gap del fin de semana)
  -----------------------------------------------------------------------
""")

# =====================================================================
# PASO 1: DATOS DISPONIBLES AL VIERNES 02/01/2026
# =====================================================================
print("="*80)
print("  PASO 1: DATOS DISPONIBLES AL VIERNES 02/01/2026")
print("="*80)

# --- SPY ---
spy_daily = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
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

# Datos SPY al viernes 02/01/2026
target = pd.Timestamp('2026-01-02')
spy_dates = spy_w.index[spy_w.index <= target]
spy_fri = spy_w.loc[spy_dates[-1]]

print(f"\n  SPY al viernes {spy_dates[-1].strftime('%d/%m/%Y')}:")
print(f"    Close:              {spy_fri['close']:.2f}")
print(f"    MA200:              {spy_fri['ma200']:.2f}")
print(f"    SPY > MA200:        {'SI' if spy_fri['above_ma200'] else 'NO'}")
print(f"    Distancia MA200:    {spy_fri['dist_ma200']:+.2f}%")
print(f"    Momentum 10 sem:    {spy_fri['mom_10w']:+.2f}%")

# --- VIX ---
vix_df = pd.read_csv('data/vix_weekly.csv', skiprows=3, header=None,
                      names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.set_index('date').sort_index()
vix_df['vix'] = vix_df['close']

vix_dates = vix_df.index[vix_df.index <= target]
vix_val = vix_df.loc[vix_dates[-1], 'vix']
print(f"\n  VIX al {vix_dates[-1].strftime('%d/%m/%Y')}:")
print(f"    VIX:                {vix_val:.2f}")

# --- Subsectores ---
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
symbols = [s['symbol'] for s in sp500]

profiles = pd.read_sql("SELECT symbol, industry FROM fmp_profiles WHERE symbol = ANY(%(syms)s)",
                        engine, params={'syms': symbols})
sym_to_sub = dict(zip(profiles['symbol'], profiles['industry']))

prices = pd.read_sql("""
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol = ANY(%(syms)s) AND date BETWEEN '2023-01-01' AND '2026-01-03'
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

prev_dates = dd_wide.index[dd_wide.index <= target]
last_date = prev_dates[-1]
dd_row = dd_wide.loc[last_date]
rsi_row = rsi_wide.loc[last_date]

n_total = dd_row.notna().sum()
n_dd_healthy = int((dd_row > -10).sum())
n_dd_deep = int((dd_row < -20).sum())
n_rsi_total = int(rsi_row.notna().sum())
n_rsi_55 = int((rsi_row > 55).sum())

pct_dd_h = n_dd_healthy / n_total * 100
pct_dd_d = n_dd_deep / n_total * 100
pct_rsi = n_rsi_55 / n_rsi_total * 100

print(f"\n  Subsectores al viernes {last_date.strftime('%d/%m/%Y')} ({n_total} subsectores):")
print(f"    DD saludables (>-10%):  {n_dd_healthy}/{n_total} = {pct_dd_h:.1f}%")
print(f"    DD profundos (<-20%):   {n_dd_deep}/{n_total} = {pct_dd_d:.1f}%")
print(f"    RSI > 55:               {n_rsi_55}/{n_rsi_total} = {pct_rsi:.1f}%")

# Listar subsectores en cada categoria
dd_healthy_subs = dd_row[dd_row > -10].index.tolist()
dd_deep_subs = dd_row[dd_row < -20].sort_values().index.tolist()
rsi_above_subs = rsi_row[rsi_row > 55].index.tolist()

print(f"\n  Subsectores con DD < -20% (profundos):")
for s in dd_deep_subs:
    print(f"    {s}: DD={dd_row[s]:.1f}%  RSI={rsi_row[s]:.1f}")

# =====================================================================
# PASO 2: CALCULO DE REGIMEN
# =====================================================================
print(f"\n{'='*80}")
print("  PASO 2: CALCULO DE REGIMEN")
print("="*80)

spy_above = spy_fri['above_ma200']
spy_dist = spy_fri['dist_ma200']
spy_mom = spy_fri['mom_10w'] if pd.notna(spy_fri['mom_10w']) else 0

# BDD
print(f"\n  1. BDD (Breadth Drawdown):")
print(f"     Valor: {pct_dd_h:.1f}% de subsectores con DD > -10%")
if pct_dd_h >= 75: s_bdd = 2.0; print(f"     {pct_dd_h:.1f}% >= 75% --> score = +2.0")
elif pct_dd_h >= 60: s_bdd = 1.0; print(f"     60% <= {pct_dd_h:.1f}% < 75% --> score = +1.0")
elif pct_dd_h >= 45: s_bdd = 0.0; print(f"     45% <= {pct_dd_h:.1f}% < 60% --> score = 0.0")
elif pct_dd_h >= 30: s_bdd = -1.0; print(f"     30% <= {pct_dd_h:.1f}% < 45% --> score = -1.0")
elif pct_dd_h >= 15: s_bdd = -2.0; print(f"     15% <= {pct_dd_h:.1f}% < 30% --> score = -2.0")
else: s_bdd = -3.0; print(f"     {pct_dd_h:.1f}% < 15% --> score = -3.0")

# BRSI
print(f"\n  2. BRSI (Breadth RSI):")
print(f"     Valor: {pct_rsi:.1f}% de subsectores con RSI > 55")
if pct_rsi >= 75: s_brsi = 2.0; print(f"     {pct_rsi:.1f}% >= 75% --> score = +2.0")
elif pct_rsi >= 60: s_brsi = 1.0; print(f"     60% <= {pct_rsi:.1f}% < 75% --> score = +1.0")
elif pct_rsi >= 45: s_brsi = 0.0; print(f"     45% <= {pct_rsi:.1f}% < 60% --> score = 0.0")
elif pct_rsi >= 30: s_brsi = -1.0; print(f"     30% <= {pct_rsi:.1f}% < 45% --> score = -1.0")
elif pct_rsi >= 15: s_brsi = -2.0; print(f"     15% <= {pct_rsi:.1f}% < 30% --> score = -2.0")
else: s_brsi = -3.0; print(f"     {pct_rsi:.1f}% < 15% --> score = -3.0")

# DDP
print(f"\n  3. DDP (Deep Drawdown):")
print(f"     Valor: {pct_dd_d:.1f}% de subsectores con DD < -20%")
if pct_dd_d <= 5: s_ddp = 1.5; print(f"     {pct_dd_d:.1f}% <= 5% --> score = +1.5")
elif pct_dd_d <= 15: s_ddp = 0.5; print(f"     5% < {pct_dd_d:.1f}% <= 15% --> score = +0.5")
elif pct_dd_d <= 30: s_ddp = -0.5; print(f"     15% < {pct_dd_d:.1f}% <= 30% --> score = -0.5")
elif pct_dd_d <= 50: s_ddp = -1.5; print(f"     30% < {pct_dd_d:.1f}% <= 50% --> score = -1.5")
else: s_ddp = -2.5; print(f"     {pct_dd_d:.1f}% > 50% --> score = -2.5")

# SPY
print(f"\n  4. SPY (vs MA200):")
print(f"     SPY={spy_fri['close']:.2f}  MA200={spy_fri['ma200']:.2f}  >MA200={'SI' if spy_above else 'NO'}  Dist={spy_dist:+.1f}%")
if spy_above and spy_dist > 5: s_spy = 1.5; print(f"     >MA200 Y dist {spy_dist:+.1f}% > 5% --> score = +1.5")
elif spy_above: s_spy = 0.5; print(f"     >MA200 Y dist {spy_dist:+.1f}% <= 5% --> score = +0.5")
elif spy_dist > -5: s_spy = -0.5; print(f"     <MA200 Y dist {spy_dist:+.1f}% > -5% --> score = -0.5")
elif spy_dist > -15: s_spy = -1.5; print(f"     <MA200 Y dist {spy_dist:+.1f}% > -15% --> score = -1.5")
else: s_spy = -2.5; print(f"     <MA200 Y dist {spy_dist:+.1f}% <= -15% --> score = -2.5")

# MOM
print(f"\n  5. MOM (Momentum SPY 10 semanas):")
print(f"     Valor: {spy_mom:+.2f}%")
if spy_mom > 5: s_mom = 1.0; print(f"     {spy_mom:+.1f}% > 5% --> score = +1.0")
elif spy_mom > 0: s_mom = 0.5; print(f"     0% < {spy_mom:+.1f}% <= 5% --> score = +0.5")
elif spy_mom > -5: s_mom = -0.5; print(f"     -5% < {spy_mom:+.1f}% <= 0% --> score = -0.5")
elif spy_mom > -15: s_mom = -1.0; print(f"     -15% < {spy_mom:+.1f}% <= -5% --> score = -1.0")
else: s_mom = -1.5; print(f"     {spy_mom:+.1f}% <= -15% --> score = -1.5")

total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

print(f"\n  SUMA TOTAL:")
print(f"     BDD({s_bdd:+.1f}) + BRSI({s_brsi:+.1f}) + DDP({s_ddp:+.1f}) + SPY({s_spy:+.1f}) + MOM({s_mom:+.1f}) = {total:+.1f}")

# Regimen
is_burbuja = (total >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90)
print(f"\n  CLASIFICACION:")
if is_burbuja:
    regime = 'BURBUJA'
    print(f"     {total:+.1f} >= 8.0 Y DD_H={pct_dd_h:.0f}%>=85% Y RSI55={pct_rsi:.0f}%>=90% --> BURBUJA")
elif total >= 7.0:
    regime = 'GOLDILOCKS'
    print(f"     {total:+.1f} >= 7.0 --> GOLDILOCKS")
elif total >= 4.0:
    regime = 'ALCISTA'
    print(f"     {total:+.1f} >= 4.0 --> ALCISTA")
elif total >= 0.5:
    regime = 'NEUTRAL'
    print(f"     {total:+.1f} >= 0.5 --> NEUTRAL")
elif total >= -2.0:
    regime = 'CAUTIOUS'
    print(f"     {total:+.1f} >= -2.0 --> CAUTIOUS")
elif total >= -5.0:
    regime = 'BEARISH'
    print(f"     {total:+.1f} >= -5.0 --> BEARISH")
elif total >= -9.0:
    regime = 'CRISIS'
    print(f"     {total:+.1f} >= -9.0 --> CRISIS")
else:
    regime = 'PANICO'
    print(f"     {total:+.1f} < -9.0 --> PANICO")

# VIX veto
if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
    print(f"     VIX={vix_val:.1f} >= 30 --> rebaja {regime} a NEUTRAL")
    regime = 'NEUTRAL'
elif vix_val >= 35 and regime == 'NEUTRAL':
    print(f"     VIX={vix_val:.1f} >= 35 --> rebaja NEUTRAL a CAUTIOUS")
    regime = 'CAUTIOUS'
else:
    print(f"     VIX={vix_val:.1f} < 30 --> sin veto VIX")

print(f"\n  >>> REGIMEN FINAL SEMANA 1: {regime} (score {total:+.1f}) <<<")

# =====================================================================
# PASO 3: RENTABILIDAD SPY DEL TRADE
# =====================================================================
print(f"\n{'='*80}")
print("  PASO 3: RENTABILIDAD SPY DEL TRADE")
print("="*80)

# Buscar precios exactos
dates_trading = ['2026-01-02', '2026-01-05', '2026-01-09', '2026-01-12']
spy_prices = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2025-12-22' AND '2026-01-16'
    ORDER BY date
""", engine)
spy_prices['date'] = pd.to_datetime(spy_prices['date'])
spy_prices = spy_prices.set_index('date')

print(f"\n  Precios SPY disponibles:")
for idx, row in spy_prices.iterrows():
    marker = ""
    if idx == pd.Timestamp('2025-12-26'):
        marker = " <-- Senal inicio (Vie W-1)"
    elif idx == pd.Timestamp('2026-01-02'):
        marker = " <-- Senal fin (Vie W) = fecha regimen"
    elif idx == pd.Timestamp('2026-01-05'):
        marker = " <-- ENTRADA trade (Lun W+1 open)"
    elif idx == pd.Timestamp('2026-01-12'):
        marker = " <-- SALIDA trade (Lun W+2 open)"
    print(f"    {idx.strftime('%a %d/%m/%Y')}: open={row['open']:.2f}  close={row['close']:.2f}{marker}")

# Calcular rentabilidades
fri_prev = pd.Timestamp('2025-12-26')
fri_curr = pd.Timestamp('2026-01-02')
mon_entry = pd.Timestamp('2026-01-05')
mon_exit = pd.Timestamp('2026-01-12')

if fri_prev in spy_prices.index and fri_curr in spy_prices.index:
    ret_senal = (spy_prices.loc[fri_curr, 'close'] / spy_prices.loc[fri_prev, 'close'] - 1) * 100
    print(f"\n  SENAL (Vie->Vie close):")
    print(f"    {spy_prices.loc[fri_prev, 'close']:.2f} -> {spy_prices.loc[fri_curr, 'close']:.2f} = {ret_senal:+.2f}%")

if fri_curr in spy_prices.index and mon_entry in spy_prices.index:
    gap = (spy_prices.loc[mon_entry, 'open'] / spy_prices.loc[fri_curr, 'close'] - 1) * 100
    print(f"\n  GAP FIN DE SEMANA (Vie close -> Lun open):")
    print(f"    {spy_prices.loc[fri_curr, 'close']:.2f} -> {spy_prices.loc[mon_entry, 'open']:.2f} = {gap:+.2f}%")

if mon_entry in spy_prices.index and mon_exit in spy_prices.index:
    ret_trade = (spy_prices.loc[mon_exit, 'open'] / spy_prices.loc[mon_entry, 'open'] - 1) * 100
    print(f"\n  RENTABILIDAD TRADE (Lun open -> Lun open):")
    print(f"    {spy_prices.loc[mon_entry, 'open']:.2f} -> {spy_prices.loc[mon_exit, 'open']:.2f} = {ret_trade:+.2f}%")
    print(f"\n  >>> RENTABILIDAD SPY SEMANA 1: {ret_trade:+.2f}% <<<")
else:
    print(f"\n  No se encontraron precios para Lun 05/01 y/o Lun 12/01")
    # Buscar lunes mas cercanos
    for d in pd.date_range('2026-01-05', '2026-01-07'):
        if d in spy_prices.index:
            mon_entry = d
            break
    for d in pd.date_range('2026-01-12', '2026-01-14'):
        if d in spy_prices.index:
            mon_exit = d
            break
    if mon_entry in spy_prices.index and mon_exit in spy_prices.index:
        ret_trade = (spy_prices.loc[mon_exit, 'open'] / spy_prices.loc[mon_entry, 'open'] - 1) * 100
        print(f"\n  RENTABILIDAD TRADE ({mon_entry.strftime('%d/%m')} open -> {mon_exit.strftime('%d/%m')} open):")
        print(f"    {spy_prices.loc[mon_entry, 'open']:.2f} -> {spy_prices.loc[mon_exit, 'open']:.2f} = {ret_trade:+.2f}%")
        print(f"\n  >>> RENTABILIDAD SPY SEMANA 1: {ret_trade:+.2f}% <<<")
