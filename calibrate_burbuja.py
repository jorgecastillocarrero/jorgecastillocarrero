"""Calibrar cuantas semanas BURBUJA con distintos brackets S1/S2"""
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

dates = dd_wide.index[dd_wide.index >= '2001-01-01']

# Pre-calcular componentes para cada fecha
print("Pre-calculando componentes...")
records = []
for date in dates:
    dd_row = dd_wide.loc[date]
    rsi_row = rsi_wide.loc[date]
    n_total = dd_row.notna().sum()
    if n_total == 0:
        continue
    n_rsi = rsi_row.notna().sum()
    if n_rsi == 0:
        continue

    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi55 = (rsi_row > 55).sum() / n_rsi * 100

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) == 0:
        continue
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_above = spy_last.get('above_ma200', 0)
    spy_mom = spy_last.get('mom_10w', 0)
    spy_dist = spy_last.get('dist_ma200', 0)
    if not pd.notna(spy_mom): spy_mom = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # S3 fijo
    if pct_dd_deep <= 5: s3 = 1.5
    elif pct_dd_deep <= 15: s3 = 0.5
    elif pct_dd_deep <= 30: s3 = -0.5
    else: s3 = -1.5

    # S4 fijo (max 1.5)
    if spy_above and spy_dist > 5: s4 = 1.5
    elif spy_above: s4 = 0.5
    elif spy_dist > -5: s4 = -0.5
    else: s4 = -1.5

    # S5 fijo
    if spy_mom > 5: s5 = 1.0
    elif spy_mom > 0: s5 = 0.5
    elif spy_mom > -5: s5 = -0.5
    else: s5 = -1.0

    records.append({
        'date': date, 'dd_h': pct_dd_healthy, 'rsi55': pct_rsi55,
        's3': s3, 's4': s4, 's5': s5, 'vix': vix_val,
    })

df = pd.DataFrame(records)
print(f"Semanas totales: {len(df)}")

# Probar combinaciones de brackets S1/S2
print("\n" + "=" * 90)
print("BURBUJA (score=8.0) con distintos brackets S1/S2")
print("=" * 90)
print(f"\n{'S1 bracket':>20} {'S2 bracket':>20} {'N BURB':>8} {'Anos':>30}")
print("-" * 90)

for s1_th in [75, 80, 85, 90]:
    for s2_th in [75, 80, 85, 90]:
        burb_dates = []
        for _, r in df.iterrows():
            # S1 con bracket variable
            if r['dd_h'] >= s1_th: s1 = 2.0
            elif r['dd_h'] >= s1_th - 15: s1 = 1.0
            else: s1 = 0.0  # simplificado

            # S2 con bracket variable
            if r['rsi55'] >= s2_th: s2 = 2.0
            elif r['rsi55'] >= s2_th - 15: s2 = 1.0
            else: s2 = 0.0

            total = s1 + s2 + r['s3'] + r['s4'] + r['s5']

            # VIX override
            if r['vix'] >= 30 and total >= 8.0:
                continue

            if total >= 8.0:
                burb_dates.append(r['date'])

        if burb_dates:
            years = pd.Series([d.year for d in burb_dates]).value_counts().sort_index()
            yr_str = ", ".join([f"{y}({n})" for y, n in years.items()])
        else:
            yr_str = "-"

        marker = " <<<" if 5 <= len(burb_dates) <= 15 else ""
        print(f"  DD_h>={s1_th}% for +2.0  RSI55>={s2_th}% for +2.0  {len(burb_dates):>5}   {yr_str}{marker}")

# Detalle de las mejores opciones (~10 semanas)
print("\n" + "=" * 90)
print("DETALLE: opciones con ~10 semanas BURBUJA")
print("=" * 90)

for s1_th, s2_th in [(85, 80), (80, 85), (85, 85), (90, 80), (80, 90), (85, 90), (90, 85), (90, 90)]:
    burb_weeks = []
    for _, r in df.iterrows():
        if r['dd_h'] >= s1_th: s1 = 2.0
        elif r['dd_h'] >= s1_th - 15: s1 = 1.0
        else: s1 = 0.0

        if r['rsi55'] >= s2_th: s2 = 2.0
        elif r['rsi55'] >= s2_th - 15: s2 = 1.0
        else: s2 = 0.0

        total = s1 + s2 + r['s3'] + r['s4'] + r['s5']
        if r['vix'] >= 30 and total >= 8.0:
            continue
        if total >= 8.0:
            burb_weeks.append((r['date'], r['dd_h'], r['rsi55'], r['s4'], r['s5'], r['vix']))

    if 0 < len(burb_weeks) <= 20:
        print(f"\n  S1>={s1_th}%, S2>={s2_th}%: {len(burb_weeks)} semanas")
        for d, ddh, rsi, s4, s5, vix in burb_weeks:
            print(f"    {d.strftime('%Y-%m-%d')} DD_h={ddh:.0f}% RSI55={rsi:.0f}% S4={s4:+.1f} S5={s5:+.1f} VIX={vix:.0f}")
