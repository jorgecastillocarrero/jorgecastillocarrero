"""Diagnostico: que regimen tiene la semana del 21/02/2026"""
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

# Precios con 52 semanas de historia para DD y RSI
df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2023-01-01' AND '2026-02-21'
    ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year

df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

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

# Fecha mas cercana al 21/02/2026
target = pd.Timestamp('2026-02-21')
dates_avail = pd.to_datetime(sub_weekly['date'].unique())
closest = dates_avail[dates_avail <= target].max()
print(f"Fecha mas cercana: {closest.date()}")

# DD y RSI para esa fecha
row_data = sub_weekly[sub_weekly['date'] == closest]
dd_vals = row_data.set_index('subsector')['drawdown_52w']
rsi_vals = row_data.set_index('subsector')['rsi_14w']

n_total = dd_vals.notna().sum()
pct_dd_healthy = (dd_vals > -10).sum() / n_total * 100
pct_dd_deep = (dd_vals < -20).sum() / n_total * 100
n_rsi = rsi_vals.notna().sum()
pct_rsi_above50 = (rsi_vals > 50).sum() / n_rsi * 100

print(f"\n{'='*60}")
print(f"BREADTH SUBSECTORES ({n_total} subsectores)")
print(f"{'='*60}")
print(f"  DD healthy (>-10%): {(dd_vals > -10).sum()}/{n_total} = {pct_dd_healthy:.1f}%")
print(f"  DD deep (<-20%):    {(dd_vals < -20).sum()}/{n_total} = {pct_dd_deep:.1f}%")
print(f"  RSI > 50:           {(rsi_vals > 50).sum()}/{n_rsi} = {pct_rsi_above50:.1f}%")

# SPY
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2024-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()
spy['ma200'] = spy['close'].rolling(200).mean()
spy['above_ma200'] = (spy['close'] > spy['ma200']).astype(int)
spy['dist_ma200'] = (spy['close'] / spy['ma200'] - 1) * 100
spy_w = spy.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

spy_dates = spy_w.index[spy_w.index <= target]
spy_last = spy_w.loc[spy_dates[-1]]
print(f"\n{'='*60}")
print(f"SPY (fecha: {spy_dates[-1].date()})")
print(f"{'='*60}")
print(f"  Close: {spy_last['close']:.2f}")
print(f"  MA200: {spy_last['ma200']:.2f}")
print(f"  Above MA200: {bool(spy_last['above_ma200'])}")
print(f"  Dist MA200: {spy_last['dist_ma200']:+.1f}%")
print(f"  Mom 10w: {spy_last['mom_10w']:+.1f}%")

# VIX
vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_dates = vix_df.index[vix_df.index <= target]
vix_val = vix_df.loc[vix_dates[-1], 'close']
print(f"\n{'='*60}")
print(f"VIX: {vix_val:.1f}")
print(f"{'='*60}")

# Calcular score componente a componente
print(f"\n{'='*60}")
print(f"CALCULO SCORE DETALLADO")
print(f"{'='*60}")

if pct_dd_healthy >= 75: s1 = 2.0; s1_l = ">=75%"
elif pct_dd_healthy >= 60: s1 = 1.0; s1_l = ">=60%"
elif pct_dd_healthy >= 45: s1 = 0.0; s1_l = ">=45%"
elif pct_dd_healthy >= 30: s1 = -1.0; s1_l = ">=30%"
else: s1 = -2.0; s1_l = "<30%"

if pct_rsi_above50 >= 75: s2 = 2.0; s2_l = ">=75%"
elif pct_rsi_above50 >= 60: s2 = 1.0; s2_l = ">=60%"
elif pct_rsi_above50 >= 45: s2 = 0.0; s2_l = ">=45%"
elif pct_rsi_above50 >= 30: s2 = -1.0; s2_l = ">=30%"
else: s2 = -2.0; s2_l = "<30%"

if pct_dd_deep <= 5: s3 = 1.5; s3_l = "<=5%"
elif pct_dd_deep <= 15: s3 = 0.5; s3_l = "<=15%"
elif pct_dd_deep <= 30: s3 = -0.5; s3_l = "<=30%"
else: s3 = -1.5; s3_l = ">30%"

spy_above = spy_last['above_ma200']
spy_dist = spy_last['dist_ma200']
spy_mom = spy_last['mom_10w']

if spy_above and spy_dist > 5: s4 = 1.5; s4_l = "above+dist>5%"
elif spy_above: s4 = 0.5; s4_l = "above MA200"
elif spy_dist > -5: s4 = -0.5; s4_l = "below,dist>-5%"
else: s4 = -1.5; s4_l = "below,dist<-5%"

if spy_mom > 5: s5 = 1.0; s5_l = "mom>5%"
elif spy_mom > 0: s5 = 0.5; s5_l = "mom>0%"
elif spy_mom > -5: s5 = -0.5; s5_l = "mom>-5%"
else: s5 = -1.0; s5_l = "mom<-5%"

total = s1 + s2 + s3 + s4 + s5

print(f"  S1 Breadth DD healthy ({pct_dd_healthy:.0f}% {s1_l}):  {s1:+.1f}")
print(f"  S2 Breadth RSI>50     ({pct_rsi_above50:.0f}% {s2_l}):  {s2:+.1f}")
print(f"  S3 DD deep            ({pct_dd_deep:.0f}% {s3_l}):      {s3:+.1f}")
print(f"  S4 SPY vs MA200       ({spy_dist:+.1f}% {s4_l}): {s4:+.1f}")
print(f"  S5 SPY momentum 10w   ({spy_mom:+.1f}% {s5_l}):  {s5:+.1f}")
print(f"  {'-'*50}")
print(f"  TOTAL SCORE: {total:+.1f}")

if total >= 7.0: regime = 'BURBUJA (7.0+)'
elif total >= 5.5: regime = 'GOLDILOCKS (5.5-7.0)'
elif total >= 4.0: regime = 'ALCISTA (4.0-5.5)'
elif total >= 0.5: regime = 'NEUTRAL (0.5-4.0)'
elif total >= -1.5: regime = 'CAUTIOUS (-1.5-0.5)'
elif total >= -3.0: regime = 'BEARISH (-3.0 a -1.5)'
else: regime = 'CRISIS (<-3.0)'

override = ""
if vix_val >= 30 and total >= 4.0:
    override = " -> NEUTRAL (VIX >= 30 override!)"
elif vix_val >= 35 and total >= 0.5 and total < 4.0:
    override = " -> CAUTIOUS (VIX >= 35 override!)"

print(f"\n  >>> REGIMEN: {regime}{override} <<<")

# Detalle subsectores problematicos
print(f"\n{'='*60}")
print(f"SUBSECTORES CON DD < -10% (NO saludables)")
print(f"{'='*60}")
bad_dd = dd_vals[dd_vals < -10].sort_values()
for sub, dd in bad_dd.items():
    rsi = rsi_vals.get(sub, 0)
    if pd.notna(rsi):
        print(f"  {sub:<35} DD={dd:>+6.1f}%  RSI={rsi:>5.1f}")

print(f"\n{'='*60}")
print(f"SUBSECTORES CON RSI < 50 (sin momentum)")
print(f"{'='*60}")
bad_rsi = rsi_vals[rsi_vals < 50].sort_values()
for sub, rsi in bad_rsi.items():
    dd = dd_vals.get(sub, 0)
    if pd.notna(dd):
        print(f"  {sub:<35} RSI={rsi:>5.1f}  DD={dd:>+6.1f}%")

# Resumen visual
print(f"\n{'='*60}")
print(f"RESUMEN VISUAL")
print(f"{'='*60}")
n_dd_healthy = (dd_vals > -10).sum()
n_dd_warn = ((dd_vals <= -10) & (dd_vals > -20)).sum()
n_dd_deep = (dd_vals < -20).sum()
n_rsi_strong = (rsi_vals > 60).sum()
n_rsi_ok = ((rsi_vals >= 50) & (rsi_vals <= 60)).sum()
n_rsi_weak = ((rsi_vals >= 40) & (rsi_vals < 50)).sum()
n_rsi_bad = (rsi_vals < 40).sum()

print(f"  DD:  {n_dd_healthy} healthy | {n_dd_warn} warning (-10/-20%) | {n_dd_deep} deep (<-20%)")
print(f"  RSI: {n_rsi_strong} strong (>60) | {n_rsi_ok} ok (50-60) | {n_rsi_weak} weak (40-50) | {n_rsi_bad} bad (<40)")
