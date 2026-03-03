"""
Detalle completo del calculo de regimen semana a semana 2026.
Para cada semana muestra los datos brutos y como se llega al score.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# --- Constituyentes S&P 500 ---
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
symbols = [s['symbol'] for s in sp500]

# --- Subsectores ---
profiles = pd.read_sql("""
    SELECT symbol, industry FROM fmp_profiles WHERE symbol = ANY(%(syms)s)
""", engine, params={'syms': symbols})
sym_to_sub = dict(zip(profiles['symbol'], profiles['industry']))

# --- Precios ---
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
n_subsectors = len(valid_subs)
print(f"Subsectores validos: {n_subsectors}")

# Semanal viernes
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

# --- Semana a semana 2026 ---
fridays_2026 = pd.date_range('2026-01-02', '2026-02-27', freq='W-FRI')

for fri in fridays_2026:
    week_num = fri.isocalendar()[1]

    # Datos brutos
    prev_dates = dd_wide.index[dd_wide.index <= fri]
    if len(prev_dates) == 0:
        continue
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]

    n_total = dd_row.notna().sum()
    n_dd_healthy = (dd_row > -10).sum()
    n_dd_deep = (dd_row < -20).sum()
    n_rsi_total = rsi_row.notna().sum()
    n_rsi_above55 = (rsi_row > 55).sum()

    pct_dd_healthy = n_dd_healthy / n_total * 100
    pct_dd_deep = n_dd_deep / n_total * 100
    pct_rsi_above55 = n_rsi_above55 / n_rsi_total * 100 if n_rsi_total > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= fri]
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_close = spy_last['close']
    spy_ma200 = spy_last['ma200']
    spy_above = spy_last['above_ma200']
    spy_dist = spy_last['dist_ma200']
    spy_mom = spy_last['mom_10w']
    if not pd.notna(spy_mom): spy_mom = 0

    vix_dates = vix_df.index[vix_df.index <= fri]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scores
    if pct_dd_healthy >= 75: score_bdd = 2.0; umbral_bdd = '>=75'
    elif pct_dd_healthy >= 60: score_bdd = 1.0; umbral_bdd = '>=60'
    elif pct_dd_healthy >= 45: score_bdd = 0.0; umbral_bdd = '>=45'
    elif pct_dd_healthy >= 30: score_bdd = -1.0; umbral_bdd = '>=30'
    elif pct_dd_healthy >= 15: score_bdd = -2.0; umbral_bdd = '>=15'
    else: score_bdd = -3.0; umbral_bdd = '<15'

    if pct_rsi_above55 >= 75: score_brsi = 2.0; umbral_brsi = '>=75'
    elif pct_rsi_above55 >= 60: score_brsi = 1.0; umbral_brsi = '>=60'
    elif pct_rsi_above55 >= 45: score_brsi = 0.0; umbral_brsi = '>=45'
    elif pct_rsi_above55 >= 30: score_brsi = -1.0; umbral_brsi = '>=30'
    elif pct_rsi_above55 >= 15: score_brsi = -2.0; umbral_brsi = '>=15'
    else: score_brsi = -3.0; umbral_brsi = '<15'

    if pct_dd_deep <= 5: score_ddp = 1.5; umbral_ddp = '<=5'
    elif pct_dd_deep <= 15: score_ddp = 0.5; umbral_ddp = '<=15'
    elif pct_dd_deep <= 30: score_ddp = -0.5; umbral_ddp = '<=30'
    elif pct_dd_deep <= 50: score_ddp = -1.5; umbral_ddp = '<=50'
    else: score_ddp = -2.5; umbral_ddp = '>50'

    if spy_above and spy_dist > 5: score_spy = 1.5; umbral_spy = '>MA200 & dist>5%'
    elif spy_above: score_spy = 0.5; umbral_spy = '>MA200 & dist<=5%'
    elif spy_dist > -5: score_spy = -0.5; umbral_spy = '<MA200 & dist>-5%'
    elif spy_dist > -15: score_spy = -1.5; umbral_spy = '<MA200 & dist>-15%'
    else: score_spy = -2.5; umbral_spy = '<MA200 & dist<=-15%'

    if spy_mom > 5: score_mom = 1.0; umbral_mom = '>5%'
    elif spy_mom > 0: score_mom = 0.5; umbral_mom = '>0%'
    elif spy_mom > -5: score_mom = -0.5; umbral_mom = '>-5%'
    elif spy_mom > -15: score_mom = -1.0; umbral_mom = '>-15%'
    else: score_mom = -1.5; umbral_mom = '<=-15%'

    total = score_bdd + score_brsi + score_ddp + score_spy + score_mom

    # Regimen
    is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'

    vix_note = ''
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        vix_note = f'VIX={vix_val:.1f}>=30 rebaja {regime}->NEUTRAL'
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        vix_note = f'VIX={vix_val:.1f}>=35 rebaja NEUTRAL->CAUTIOUS'
        regime = 'CAUTIOUS'

    # Imprimir
    print(f"\n{'='*80}")
    print(f"  SEMANA {week_num} | Viernes {fri.strftime('%Y-%m-%d')} | REGIMEN: {regime}")
    print(f"{'='*80}")
    print(f"")
    print(f"  DATOS BRUTOS:")
    print(f"    Subsectores totales: {n_total}")
    print(f"    DD saludables (>-10%): {n_dd_healthy}/{n_total} = {pct_dd_healthy:.1f}%")
    print(f"    DD profundos (<-20%):  {n_dd_deep}/{n_total} = {pct_dd_deep:.1f}%")
    print(f"    RSI > 55:              {n_rsi_above55}/{n_rsi_total} = {pct_rsi_above55:.1f}%")
    print(f"    SPY close:             {spy_close:.2f}")
    print(f"    SPY MA200:             {spy_ma200:.2f}")
    print(f"    SPY > MA200:           {'SI' if spy_above else 'NO'}")
    print(f"    SPY dist MA200:        {spy_dist:+.1f}%")
    print(f"    SPY mom 10w:           {spy_mom:+.1f}%")
    print(f"    VIX:                   {vix_val:.1f}")
    print(f"")
    print(f"  CALCULO SCORES:")
    print(f"    BDD:  {pct_dd_healthy:5.1f}% -> umbral {umbral_bdd:>8} -> score = {score_bdd:+.1f}")
    print(f"    BRSI: {pct_rsi_above55:5.1f}% -> umbral {umbral_brsi:>8} -> score = {score_brsi:+.1f}")
    print(f"    DDP:  {pct_dd_deep:5.1f}% -> umbral {umbral_ddp:>8} -> score = {score_ddp:+.1f}")
    print(f"    SPY:  dist={spy_dist:+.1f}% -> {umbral_spy:>20} -> score = {score_spy:+.1f}")
    print(f"    MOM:  mom={spy_mom:+.1f}% -> umbral {umbral_mom:>8} -> score = {score_mom:+.1f}")
    print(f"")
    print(f"  TOTAL = {score_bdd:+.1f} + {score_brsi:+.1f} + {score_ddp:+.1f} + {score_spy:+.1f} + {score_mom:+.1f} = {total:+.1f}")
    if total >= 7.0:
        print(f"  {total:+.1f} >= 7.0 -> GOLDILOCKS (o BURBUJA si DD_H>=85 y RSI>55>=90)")
    elif total >= 4.0:
        print(f"  {total:+.1f} >= 4.0 -> ALCISTA")
    elif total >= 0.5:
        print(f"  {total:+.1f} >= 0.5 -> NEUTRAL")
    elif total >= -2.0:
        print(f"  {total:+.1f} >= -2.0 -> CAUTIOUS")
    elif total >= -5.0:
        print(f"  {total:+.1f} >= -5.0 -> BEARISH")
    elif total >= -9.0:
        print(f"  {total:+.1f} >= -9.0 -> CRISIS")
    else:
        print(f"  {total:+.1f} < -9.0 -> PANICO")
    if vix_note:
        print(f"  !! {vix_note}")
    print(f"  -> REGIMEN FINAL: {regime}")
