"""Calibrar score: probar distintos umbrales RSI y pesos S4"""
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

target = pd.Timestamp('2026-02-21')
dates_avail = pd.to_datetime(sub_weekly['date'].unique())
closest = dates_avail[dates_avail <= target].max()

row_data = sub_weekly[sub_weekly['date'] == closest]
dd_vals = row_data.set_index('subsector')['drawdown_52w']
rsi_vals = row_data.set_index('subsector')['rsi_14w']

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
spy_above = spy_last['above_ma200']
spy_dist = spy_last['dist_ma200']
spy_mom = spy_last['mom_10w']

n_total = dd_vals.notna().sum()
pct_dd_healthy = (dd_vals > -10).sum() / n_total * 100
pct_dd_deep = (dd_vals < -20).sum() / n_total * 100

# S1 y S3 no cambian
if pct_dd_healthy >= 75: s1 = 2.0
elif pct_dd_healthy >= 60: s1 = 1.0
elif pct_dd_healthy >= 45: s1 = 0.0
elif pct_dd_healthy >= 30: s1 = -1.0
else: s1 = -2.0

if pct_dd_deep <= 5: s3 = 1.5
elif pct_dd_deep <= 15: s3 = 0.5
elif pct_dd_deep <= 30: s3 = -0.5
else: s3 = -1.5

# S5 no cambia
if spy_mom > 5: s5 = 1.0
elif spy_mom > 0: s5 = 0.5
elif spy_mom > -5: s5 = -0.5
else: s5 = -1.0

print(f"Fecha: {closest.date()}")
print(f"Fijos: S1(DD healthy)={s1:+.1f}  S3(DD deep)={s3:+.1f}  S5(mom)={s5:+.1f}")
print(f"SPY: dist={spy_dist:+.1f}%, mom={spy_mom:+.1f}%, above={bool(spy_above)}")
print()

# Contar sectores por RSI threshold
n_rsi = rsi_vals.notna().sum()
for th in [50, 55, 60, 65]:
    n = (rsi_vals > th).sum()
    pct = n / n_rsi * 100
    print(f"  RSI > {th}: {n}/{n_rsi} = {pct:.1f}%")

print()
print("=" * 100)
print(f"{'RSI_th':>6} {'%RSI':>6} {'S2':>5} | {'S4 config':>20} {'S4':>5} | {'TOTAL':>6} {'REGIMEN':>15}")
print("=" * 100)

# Probar combinaciones
rsi_thresholds = [50, 55, 60]
s4_configs = [
    ("ACTUAL (max 1.5)", lambda d, a: 1.5 if (a and d > 5) else (0.5 if a else (-0.5 if d > -5 else -1.5))),
    ("max 1.0, dist>5%", lambda d, a: 1.0 if (a and d > 5) else (0.5 if a else (-0.5 if d > -5 else -1.0))),
    ("max 1.0, dist>8%", lambda d, a: 1.0 if (a and d > 8) else (0.5 if a else (-0.5 if d > -5 else -1.0))),
    ("max 0.5 (solo above)", lambda d, a: 0.5 if a else (-0.5 if d > -5 else -1.0)),
]

for rsi_th in rsi_thresholds:
    n_above = (rsi_vals > rsi_th).sum()
    pct_above = n_above / n_rsi * 100

    if pct_above >= 75: s2 = 2.0
    elif pct_above >= 60: s2 = 1.0
    elif pct_above >= 45: s2 = 0.0
    elif pct_above >= 30: s2 = -1.0
    else: s2 = -2.0

    for s4_name, s4_fn in s4_configs:
        s4 = s4_fn(spy_dist, spy_above)
        total = s1 + s2 + s3 + s4 + s5

        if total >= 7.0: regime = 'BURBUJA'
        elif total >= 5.5: regime = 'GOLDILOCKS'
        elif total >= 4.0: regime = 'ALCISTA'
        elif total >= 0.5: regime = 'NEUTRAL'
        elif total >= -1.5: regime = 'CAUTIOUS'
        elif total >= -3.0: regime = 'BEARISH'
        else: regime = 'CRISIS'

        marker = " <-- ACTUAL" if rsi_th == 50 and s4_name.startswith("ACTUAL") else ""
        print(f"  RSI>{rsi_th} {pct_above:>5.1f}% S2={s2:+.1f} | {s4_name:>20} S4={s4:+.1f} | {total:>+5.1f}  {regime:<15}{marker}")
    print()

# Tambien probar cambiando los brackets de S2
print("\n" + "=" * 100)
print("ALTERNATIVA: cambiar brackets de S2 (mas exigentes)")
print("=" * 100)

brackets_options = [
    ("ACTUAL: 75/60/45/30", [75, 60, 45, 30]),
    ("Opcion A: 80/65/50/35", [80, 65, 50, 35]),
    ("Opcion B: 85/70/55/40", [85, 70, 55, 40]),
]

for rsi_th in [50, 55, 60]:
    n_above = (rsi_vals > rsi_th).sum()
    pct_above = n_above / n_rsi * 100
    print(f"\n  RSI > {rsi_th}: {pct_above:.1f}%")

    for bracket_name, brackets in brackets_options:
        if pct_above >= brackets[0]: s2 = 2.0
        elif pct_above >= brackets[1]: s2 = 1.0
        elif pct_above >= brackets[2]: s2 = 0.0
        elif pct_above >= brackets[3]: s2 = -1.0
        else: s2 = -2.0

        # Con S4 actual y S4 reducido
        s4_actual = 1.5 if (spy_above and spy_dist > 5) else (0.5 if spy_above else (-0.5 if spy_dist > -5 else -1.5))
        s4_reduced = 1.0 if (spy_above and spy_dist > 8) else (0.5 if spy_above else (-0.5 if spy_dist > -5 else -1.0))

        t1 = s1 + s2 + s3 + s4_actual + s5
        t2 = s1 + s2 + s3 + s4_reduced + s5

        r1 = 'BURB' if t1>=7 else 'GOLD' if t1>=5.5 else 'ALCI' if t1>=4 else 'NEUT' if t1>=0.5 else 'CAUT'
        r2 = 'BURB' if t2>=7 else 'GOLD' if t2>=5.5 else 'ALCI' if t2>=4 else 'NEUT' if t2>=0.5 else 'CAUT'

        print(f"    {bracket_name}: S2={s2:+.1f} | S4 actual={t1:+.1f} ({r1}) | S4 red.={t2:+.1f} ({r2})")
