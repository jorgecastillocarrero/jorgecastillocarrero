"""Calibrar score: ver distribucion historica de regimenes con distintas configs"""
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

print("Cargando datos...")
df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21'
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

date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

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

dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

print("Calculando scores historicos...")

def calc_score(date, rsi_threshold=50, s4_max=1.5, s4_dist_th=5):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        return None, None
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0:
        return None, None

    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    n_rsi = rsi_row.notna().sum()
    pct_rsi_above = (rsi_row > rsi_threshold).sum() / n_rsi * 100 if n_rsi > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) == 0:
        return None, None
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_above = spy_last.get('above_ma200', 0.5)
    spy_mom = spy_last.get('mom_10w', 0)
    spy_dist = spy_last.get('dist_ma200', 0)
    if not pd.notna(spy_mom): spy_mom = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # S1
    if pct_dd_healthy >= 75: s1 = 2.0
    elif pct_dd_healthy >= 60: s1 = 1.0
    elif pct_dd_healthy >= 45: s1 = 0.0
    elif pct_dd_healthy >= 30: s1 = -1.0
    else: s1 = -2.0

    # S2 (con RSI threshold variable)
    if pct_rsi_above >= 75: s2 = 2.0
    elif pct_rsi_above >= 60: s2 = 1.0
    elif pct_rsi_above >= 45: s2 = 0.0
    elif pct_rsi_above >= 30: s2 = -1.0
    else: s2 = -2.0

    # S3
    if pct_dd_deep <= 5: s3 = 1.5
    elif pct_dd_deep <= 15: s3 = 0.5
    elif pct_dd_deep <= 30: s3 = -0.5
    else: s3 = -1.5

    # S4 (con max y dist threshold variables)
    if spy_above and spy_dist > s4_dist_th:
        s4 = s4_max
    elif spy_above:
        s4 = 0.5
    elif spy_dist > -5:
        s4 = -0.5
    else:
        s4 = -1.0 if s4_max <= 1.0 else -1.5

    # S5
    if spy_mom > 5: s5 = 1.0
    elif spy_mom > 0: s5 = 0.5
    elif spy_mom > -5: s5 = -0.5
    else: s5 = -1.0

    total = s1 + s2 + s3 + s4 + s5
    return total, vix_val


# Configuraciones a probar
configs = [
    ("ACTUAL (RSI>50, S4 max1.5, BURB>=7.0)", 50, 1.5, 5, 7.0),
    ("RSI>55, S4 max1.5, BURB>=8.0",          55, 1.5, 5, 8.0),
    ("RSI>55, S4 max1.0, BURB>=8.0",          55, 1.0, 5, 8.0),
    ("RSI>60, S4 max1.5, BURB>=8.0",          60, 1.5, 5, 8.0),
    ("RSI>60, S4 max1.0, BURB>=8.0",          60, 1.0, 5, 8.0),
    ("RSI>55, S4 max1.0 d>8%, BURB>=8.0",     55, 1.0, 8, 8.0),
]

dates_to_check = dd_wide.index[dd_wide.index >= '2001-01-01']

for config_name, rsi_th, s4_max, s4_dist, burb_th in configs:
    regime_counts = {'BURBUJA': 0, 'GOLDILOCKS': 0, 'ALCISTA': 0, 'NEUTRAL': 0,
                     'CAUTIOUS': 0, 'BEARISH': 0, 'CRISIS': 0}

    for date in dates_to_check:
        total, vix_val = calc_score(date, rsi_th, s4_max, s4_dist)
        if total is None:
            continue

        if total >= burb_th: regime = 'BURBUJA'
        elif total >= 5.5: regime = 'GOLDILOCKS'
        elif total >= 4.0: regime = 'ALCISTA'
        elif total >= 0.5: regime = 'NEUTRAL'
        elif total >= -1.5: regime = 'CAUTIOUS'
        elif total >= -3.0: regime = 'BEARISH'
        else: regime = 'CRISIS'

        # VIX override
        if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
            regime = 'NEUTRAL'
        elif vix_val >= 35 and regime == 'NEUTRAL':
            regime = 'CAUTIOUS'

        regime_counts[regime] += 1

    total_weeks = sum(regime_counts.values())
    print(f"\n{config_name}")
    print(f"  {'Regimen':<12} {'N':>5} {'%':>6}")
    print(f"  {'-'*28}")
    for reg in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'CRISIS']:
        n = regime_counts[reg]
        pct = n / total_weeks * 100 if total_weeks > 0 else 0
        marker = ""
        if reg == 'BURBUJA' and n < 100: marker = " (pocas = OK)"
        elif reg == 'BURBUJA' and n > 200: marker = " (demasiadas!)"
        print(f"  {reg:<12} {n:>5} {pct:>5.1f}%{marker}")

# Score de la semana actual con cada config
print("\n" + "=" * 80)
print("SCORE SEMANA 21/02/2026 CON CADA CONFIG")
print("=" * 80)

target = pd.Timestamp('2026-02-21')
for config_name, rsi_th, s4_max, s4_dist, burb_th in configs:
    total, vix_val = calc_score(target, rsi_th, s4_max, s4_dist)
    if total is None:
        continue

    if total >= burb_th: regime = 'BURBUJA'
    elif total >= 5.5: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -1.5: regime = 'CAUTIOUS'
    elif total >= -3.0: regime = 'BEARISH'
    else: regime = 'CRISIS'

    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        regime = 'NEUTRAL (VIX)'

    print(f"  {config_name:<45} Score={total:+.1f} -> {regime}")
