"""Verificar sector tecnologico en burbuja .com 1997-2000"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# Identificar subsectores tech
tech_subs = [s for s in SUBSECTORS if any(k in s for k in
    ['software', 'semi', 'cloud', 'cyber', 'it_', 'ecommerce', 'internet', 'media_entertainment'])]
print("Subsectores TECH encontrados:")
for s in sorted(tech_subs):
    tickers = SUBSECTORS[s]['tickers'][:5]
    print(f"  {s:<30} tickers: {tickers}")

# Tickers tech
tech_tickers = []
for s in tech_subs:
    tech_tickers.extend(SUBSECTORS[s]['tickers'])

# Tambien mirar los grandes tech de la epoca
iconic_90s = ['MSFT', 'INTC', 'CSCO', 'ORCL', 'DELL', 'AAPL', 'IBM', 'QCOM', 'YHOO', 'AMZN']
print(f"\nIconicos 90s a verificar: {iconic_90s}")

# Verificar cuales existen en la DB en esa epoca
all_check = list(set(tech_tickers + iconic_90s))
tlist = "','".join(all_check)

df = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '1996-01-01' AND '2001-12-31'
    ORDER BY symbol, date
""", engine)
df['date'] = pd.to_datetime(df['date'])

print(f"\nRegistros totales: {len(df):,}")
print(f"Symbols encontrados: {df['symbol'].nunique()}")

# Cobertura por ano
for yr in range(1996, 2002):
    syms = sorted(df[df['date'].dt.year == yr]['symbol'].unique())
    print(f"  {yr}: {len(syms)} symbols")

# Verificar iconicos
print("\nIconicos 90s en DB:")
for t in iconic_90s:
    sub = df[df['symbol'] == t]
    if len(sub) > 0:
        print(f"  {t}: {sub['date'].min().date()} a {sub['date'].max().date()} ({len(sub)} registros)")
    else:
        print(f"  {t}: NO encontrado")

# SPY como referencia
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '1996-01-01' AND '2001-12-31'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()

# QQQ (Nasdaq 100) - empezo en 1999
qqq = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'QQQ' AND date BETWEEN '1996-01-01' AND '2001-12-31'
    ORDER BY date
""", engine)
qqq['date'] = pd.to_datetime(qqq['date'])
if len(qqq) > 0:
    qqq = qqq.set_index('date').sort_index()
    print(f"\nQQQ rango: {qqq.index.min().date()} a {qqq.index.max().date()}")

# Calcular retornos anuales de los iconicos
print("\n" + "=" * 80)
print("RETORNOS ANUALES TECH ICONICOS (burbuja .com)")
print("=" * 80)
print(f"\n{'Symbol':<8}", end="")
for yr in range(1997, 2002):
    print(f"  {yr:>8}", end="")
print()
print("-" * 58)

for t in iconic_90s:
    sub = df[df['symbol'] == t].set_index('date').sort_index()
    if len(sub) < 10:
        continue
    print(f"{t:<8}", end="")
    for yr in range(1997, 2002):
        yr_data = sub[sub.index.year == yr]
        if len(yr_data) >= 10:
            ret = (yr_data['close'].iloc[-1] / yr_data['close'].iloc[0] - 1) * 100
            print(f"  {ret:>+7.0f}%", end="")
        else:
            print(f"  {'N/A':>8}", end="")
    print()

# SPY y QQQ
print(f"{'SPY':<8}", end="")
for yr in range(1997, 2002):
    yr_data = spy[spy.index.year == yr]
    if len(yr_data) >= 10:
        ret = (yr_data['close'].iloc[-1] / yr_data['close'].iloc[0] - 1) * 100
        print(f"  {ret:>+7.0f}%", end="")
    else:
        print(f"  {'N/A':>8}", end="")
print()

if len(qqq) > 0:
    print(f"{'QQQ':<8}", end="")
    for yr in range(1997, 2002):
        yr_data = qqq[qqq.index.year == yr]
        if len(yr_data) >= 10:
            ret = (yr_data['close'].iloc[-1] / yr_data['close'].iloc[0] - 1) * 100
            print(f"  {ret:>+7.0f}%", end="")
        else:
            print(f"  {'N/A':>8}", end="")
    print()

# Ahora calcular DD y RSI de subsectores tech vs mercado general
print("\n" + "=" * 80)
print("SUBSECTORES TECH: DD y RSI trimestrales 1998-2000")
print("=" * 80)

ticker_to_sub = {}
for sub_id in tech_subs:
    for t in SUBSECTORS[sub_id]['tickers']:
        ticker_to_sub[t] = sub_id

df['subsector'] = df['symbol'].map(ticker_to_sub)
df_tech = df.dropna(subset=['subsector'])
df_tech['week'] = df_tech['date'].dt.isocalendar().week.astype(int)
df_tech['year'] = df_tech['date'].dt.year

df_weekly = df_tech.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
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

# Mostrar snapshots trimestrales
for quarter_date in ['1998-03-31', '1998-06-30', '1998-09-30', '1998-12-31',
                      '1999-03-31', '1999-06-30', '1999-09-30', '1999-12-31',
                      '2000-03-31', '2000-06-30', '2000-09-30', '2000-12-31']:
    target = pd.Timestamp(quarter_date)
    avail = sub_weekly['date'].unique()
    avail = pd.to_datetime(avail)
    closest_dates = avail[avail <= target]
    if len(closest_dates) == 0:
        continue
    closest = closest_dates.max()

    snap = sub_weekly[sub_weekly['date'] == closest]
    if len(snap) == 0:
        continue

    print(f"\n  {quarter_date} (data: {closest.date()})")
    print(f"  {'Subsector':<30} {'DD_52w':>8} {'RSI_14w':>8} {'Close':>10}")
    for _, row in snap.sort_values('subsector').iterrows():
        dd = row['drawdown_52w']
        rsi = row['rsi_14w']
        cl = row['avg_close']
        dd_str = f"{dd:+.1f}%" if pd.notna(dd) else "N/A"
        rsi_str = f"{rsi:.1f}" if pd.notna(rsi) else "N/A"
        marker = " <<<" if pd.notna(dd) and dd > -3 and pd.notna(rsi) and rsi > 70 else ""
        print(f"  {row['subsector']:<30} {dd_str:>8} {rsi_str:>8} {cl:>10.2f}{marker}")
