"""Analizar distribucion de BULLISH para dividir en 3 sub-niveles"""
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
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_price_metrics(g):
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

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)
returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')
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
spy_w['ret_spy'] = spy_w['close'].pct_change()

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

# Calcular score y metricas para semanas BULLISH
print("Analizando semanas BULLISH...\n")
results = []
for date in returns_wide.index:
    if date.year < 2001:
        continue
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        continue
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0:
        continue

    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    n_rsi = rsi_row.notna().sum()
    pct_rsi_above50 = (rsi_row > 50).sum() / n_rsi * 100 if n_rsi > 0 else 50
    pct_rsi_above70 = (rsi_row > 70).sum() / n_rsi * 100 if n_rsi > 0 else 0
    pct_rsi_above80 = (rsi_row > 80).sum() / n_rsi * 100 if n_rsi > 0 else 0
    pct_dd_near0 = (dd_row > -3).sum() / n_total * 100  # cerca de maximos

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) == 0:
        continue
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_above = spy_last.get('above_ma200', 0.5)
    spy_mom = spy_last.get('mom_10w', 0)
    spy_dist = spy_last.get('dist_ma200', 0)
    if not pd.notna(spy_mom): spy_mom = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Score compuesto
    if pct_dd_healthy >= 75: s1 = 2.0
    elif pct_dd_healthy >= 60: s1 = 1.0
    elif pct_dd_healthy >= 45: s1 = 0.0
    elif pct_dd_healthy >= 30: s1 = -1.0
    else: s1 = -2.0
    if pct_rsi_above50 >= 75: s2 = 2.0
    elif pct_rsi_above50 >= 60: s2 = 1.0
    elif pct_rsi_above50 >= 45: s2 = 0.0
    elif pct_rsi_above50 >= 30: s2 = -1.0
    else: s2 = -2.0
    if pct_dd_deep <= 5: s3 = 1.5
    elif pct_dd_deep <= 15: s3 = 0.5
    elif pct_dd_deep <= 30: s3 = -0.5
    else: s3 = -1.5
    if spy_above and spy_dist > 5: s4 = 1.5
    elif spy_above: s4 = 0.5
    elif spy_dist > -5: s4 = -0.5
    else: s4 = -1.5
    if spy_mom > 5: s5 = 1.0
    elif spy_mom > 0: s5 = 0.5
    elif spy_mom > -5: s5 = -0.5
    else: s5 = -1.0

    total = s1 + s2 + s3 + s4 + s5

    # VIX override
    if vix_val >= 30 and total >= 4.0:
        continue  # se reclasifica a NEUTRAL

    if total >= 4.0:
        # Retorno semanal
        ret_row = returns_wide.loc[date] if date in returns_wide.index else None
        avg_ret = ret_row.mean() if ret_row is not None else 0

        results.append({
            'date': date, 'year': date.year, 'score': total,
            'dd_healthy': pct_dd_healthy, 'dd_deep': pct_dd_deep,
            'dd_near0': pct_dd_near0,
            'rsi_above50': pct_rsi_above50, 'rsi_above70': pct_rsi_above70,
            'rsi_above80': pct_rsi_above80,
            'spy_dist': spy_dist, 'spy_mom': spy_mom, 'vix': vix_val,
            'avg_ret': avg_ret,
        })

df = pd.DataFrame(results)
print(f"Total semanas BULLISH: {len(df)}")
print(f"Score range: {df['score'].min():.1f} a {df['score'].max():.1f}")

# Distribucion por score
print(f"\n{'DISTRIBUCION POR SCORE':>30}")
print("-" * 40)
for s in sorted(df['score'].unique()):
    n = (df['score'] == s).sum()
    print(f"  Score {s:.1f}: {n:>4} semanas ({n/len(df)*100:.1f}%)")

# Tres niveles propuestos
print(f"\n{'=' * 100}")
print("PROPUESTA: 3 NIVELES DE BULLISH")
print("=" * 100)

levels = [
    (4.0, 5.5, 'GOLDILOCKS (4.0-5.5)', 'Bull sano, subida sostenible'),
    (5.5, 7.0, 'BULLISH (5.5-7.0)', 'Bull fuerte, momentum claro'),
    (7.0, 9.0, 'BURBUJA (7.0+)', 'Todo en maximos, complacencia'),
]

for lo, hi, label, desc in levels:
    mask = (df['score'] >= lo) & (df['score'] < hi)
    sub = df[mask]
    if len(sub) == 0:
        print(f"\n  {label}: 0 semanas")
        continue

    avg_ret = sub['avg_ret'].mean() * 100
    wr = (sub['avg_ret'] > 0).mean() * 100

    print(f"\n  {label} - {desc}")
    print(f"  {'-' * 90}")
    print(f"  Semanas: {len(sub)} ({len(sub)/len(df)*100:.0f}% del bull)")
    print(f"  Avg ret mercado: {avg_ret:+.3f}%  WR: {wr:.0f}%")
    print(f"  DD healthy: {sub['dd_healthy'].mean():.0f}%  DD deep: {sub['dd_deep'].mean():.1f}%  DD near 0: {sub['dd_near0'].mean():.0f}%")
    print(f"  RSI>50: {sub['rsi_above50'].mean():.0f}%  RSI>70: {sub['rsi_above70'].mean():.0f}%  RSI>80: {sub['rsi_above80'].mean():.0f}%")
    print(f"  SPY dist MA200: +{sub['spy_dist'].mean():.1f}%  SPY mom 10w: +{sub['spy_mom'].mean():.1f}%  VIX: {sub['vix'].mean():.1f}")

    year_counts = sub['year'].value_counts().sort_index()
    top_years = year_counts[year_counts >= 5].index.tolist()
    print(f"  Anos principales (5+ sem): {top_years}")

# Deteccion BURBUJA alternativa: no solo score, tambien indicadores extremos
print(f"\n{'=' * 100}")
print("DETECCION BURBUJA POR INDICADORES EXTREMOS")
print("=" * 100)

# Burbuja = RSI>70 en >40% sectores + SPY >10% MA200 + VIX < 18
for rsi_th, spy_th, vix_th, label in [
    (40, 10, 18, 'Estricta: RSI70>40% + SPY>10% + VIX<18'),
    (30, 8, 20, 'Media: RSI70>30% + SPY>8% + VIX<20'),
    (50, 12, 16, 'Extrema: RSI70>50% + SPY>12% + VIX<16'),
]:
    mask = (df['rsi_above70'] > rsi_th) & (df['spy_dist'] > spy_th) & (df['vix'] < vix_th)
    n = mask.sum()
    if n > 0:
        sub = df[mask]
        years = dict(sub['year'].value_counts().sort_index())
        avg_ret = sub['avg_ret'].mean() * 100
        print(f"\n  {label}")
        print(f"  Semanas: {n} | Avg ret: {avg_ret:+.3f}% | Por ano: {years}")
    else:
        print(f"\n  {label}: 0 semanas")
