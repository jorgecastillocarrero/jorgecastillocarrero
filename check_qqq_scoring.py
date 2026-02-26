"""Scoring usando QQQ/Nasdaq100 en vez de SPY para 1998-1999"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# QQQ empezo en marzo 1999, pero podemos usar ^IXIC (Nasdaq Composite) o MSFT+INTC+CSCO como proxy
# Primero verificar que tenemos

for sym in ['QQQ', 'TQQQ', '^IXIC', 'ONEQ']:
    df = pd.read_sql(f"""
        SELECT MIN(date) as min_d, MAX(date) as max_d, COUNT(*) as n
        FROM fmp_price_history
        WHERE symbol = '{sym}'
    """, engine)
    print(f"  {sym}: {df.iloc[0]['min_d']} a {df.iloc[0]['max_d']} ({df.iloc[0]['n']} registros)")

# Crear proxy Nasdaq con los iconicos tech de los 90s
print("\nCreando proxy Nasdaq100 con tech leaders...")
tech_leaders = ['MSFT', 'INTC', 'CSCO', 'ORCL', 'AAPL', 'IBM', 'QCOM', 'AMZN']
tlist = "','".join(tech_leaders)

df_tech = pd.read_sql(f"""
    SELECT symbol, date, close
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '1996-01-01' AND '2001-12-31'
    ORDER BY date
""", engine)
df_tech['date'] = pd.to_datetime(df_tech['date'])

# Equal-weight index de tech leaders
tech_daily = df_tech.pivot(index='date', columns='symbol', values='close')
# Normalizar cada uno a base 100 desde su primer dato disponible
for col in tech_daily.columns:
    first_valid = tech_daily[col].first_valid_index()
    tech_daily[col] = tech_daily[col] / tech_daily[col].loc[first_valid] * 100
tech_daily['tech_index'] = tech_daily.mean(axis=1)

# SPY para comparar
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '1996-01-01' AND '2001-12-31'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()

# QQQ cuando existe
qqq = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'QQQ' AND date BETWEEN '1996-01-01' AND '2001-12-31'
    ORDER BY date
""", engine)
qqq['date'] = pd.to_datetime(qqq['date'])
if len(qqq) > 0:
    qqq = qqq.set_index('date').sort_index()

# Calcular metricas para tech_index, SPY y QQQ
def calc_index_metrics(series, name):
    df = pd.DataFrame({'close': series}).dropna()
    df['ma200'] = df['close'].rolling(200).mean()
    df['above_ma200'] = (df['close'] > df['ma200']).astype(int)
    df['dist_ma200'] = (df['close'] / df['ma200'] - 1) * 100
    weekly = df.resample('W-FRI').last().dropna(subset=['ma200'])
    weekly['mom_10w'] = weekly['close'].pct_change(10) * 100
    weekly['high_52w'] = weekly['close'].rolling(52, min_periods=26).max()
    weekly['dd_52w'] = (weekly['close'] / weekly['high_52w'] - 1) * 100

    # RSI
    delta = weekly['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    weekly['rsi'] = 100 - (100 / (1 + rs))

    return weekly

tech_weekly = calc_index_metrics(tech_daily['tech_index'], 'TECH')
spy_weekly = calc_index_metrics(spy['close'], 'SPY')
if len(qqq) > 0:
    qqq_weekly = calc_index_metrics(qqq['close'], 'QQQ')

# Mostrar scoring semanal para 1998-1999
print("\n" + "=" * 130)
print("SCORING SEMANAL 1998-1999: TECH INDEX vs SPY")
print("=" * 130)

# Filtrar 1998-1999
tech_98_99 = tech_weekly[(tech_weekly.index >= '1998-01-01') & (tech_weekly.index <= '1999-12-31')]

print(f"\n{'Fecha':>12} | {'TECH INDEX':^50} | {'SPY':^35}")
print(f"{'':>12} | {'Close':>8} {'MA200':>8} {'Dist%':>7} {'Mom10w':>8} {'DD52w':>7} {'RSI':>6} | {'Close':>8} {'Dist%':>7} {'Mom10w':>8} {'DD52w':>7} {'RSI':>6}")
print("-" * 130)

for date in tech_98_99.index:
    t = tech_weekly.loc[date]

    spy_dates = spy_weekly.index[spy_weekly.index <= date]
    if len(spy_dates) == 0:
        continue
    s = spy_weekly.loc[spy_dates[-1]]

    # Marcar semanas extremas
    marker = ""
    if pd.notna(t.get('dist_ma200')) and t['dist_ma200'] > 20:
        marker = " <<< BURBUJA TECH"
    elif pd.notna(t.get('dist_ma200')) and t['dist_ma200'] > 10:
        marker = " << BULL FUERTE"

    t_dist = t['dist_ma200'] if pd.notna(t.get('dist_ma200')) else 0
    t_mom = t['mom_10w'] if pd.notna(t.get('mom_10w')) else 0
    t_dd = t['dd_52w'] if pd.notna(t.get('dd_52w')) else 0
    t_rsi = t['rsi'] if pd.notna(t.get('rsi')) else 0
    t_ma = t['ma200'] if pd.notna(t.get('ma200')) else 0

    s_dist = s['dist_ma200'] if pd.notna(s.get('dist_ma200')) else 0
    s_mom = s['mom_10w'] if pd.notna(s.get('mom_10w')) else 0
    s_dd = s['dd_52w'] if pd.notna(s.get('dd_52w')) else 0
    s_rsi = s['rsi'] if pd.notna(s.get('rsi')) else 0

    print(f"{date.strftime('%Y-%m-%d'):>12} | {t['close']:>8.1f} {t_ma:>8.1f} {t_dist:>+6.1f}% {t_mom:>+7.1f}% {t_dd:>+6.1f}% {t_rsi:>5.1f} | "
          f"{s['close']:>8.2f} {s_dist:>+6.1f}% {s_mom:>+7.1f}% {s_dd:>+6.1f}% {s_rsi:>5.1f}{marker}")

# Resumen de maximos
print("\n" + "=" * 80)
print("MAXIMOS TECH INDEX 1998-1999")
print("=" * 80)
max_dist = tech_98_99['dist_ma200'].max()
max_dist_date = tech_98_99['dist_ma200'].idxmax()
max_mom = tech_98_99['mom_10w'].max()
max_mom_date = tech_98_99['mom_10w'].idxmax()
max_rsi = tech_98_99['rsi'].max()
max_rsi_date = tech_98_99['rsi'].idxmax()
print(f"  Max dist MA200: {max_dist:+.1f}% el {max_dist_date.strftime('%Y-%m-%d')}")
print(f"  Max mom 10w:    {max_mom:+.1f}% el {max_mom_date.strftime('%Y-%m-%d')}")
print(f"  Max RSI:        {max_rsi:.1f} el {max_rsi_date.strftime('%Y-%m-%d')}")

# Comparar con SPY
spy_98_99 = spy_weekly[(spy_weekly.index >= '1998-01-01') & (spy_weekly.index <= '1999-12-31')]
print(f"\nMAXIMOS SPY 1998-1999")
print(f"  Max dist MA200: {spy_98_99['dist_ma200'].max():+.1f}%")
print(f"  Max mom 10w:    {spy_98_99['mom_10w'].max():+.1f}%")
print(f"  Max RSI:        {spy_98_99['rsi'].max():.1f}")

# Scoring S4 si usaramos tech_index en vez de SPY
print("\n" + "=" * 80)
print("IMPACTO: S4 con TECH INDEX vs SPY")
print("=" * 80)
print(f"\n  Semanas donde TECH dist>20% (burbuja clara):")
bubble_weeks = tech_98_99[tech_98_99['dist_ma200'] > 20]
for date in bubble_weeks.index:
    t = tech_weekly.loc[date]
    s = spy_weekly.loc[spy_weekly.index[spy_weekly.index <= date][-1]]
    t_dist = t['dist_ma200']
    s_dist = s['dist_ma200']
    print(f"    {date.strftime('%Y-%m-%d')}: TECH dist={t_dist:+.1f}%  SPY dist={s_dist:+.1f}%  "
          f"TECH mom={t['mom_10w']:+.1f}%  SPY mom={s['mom_10w']:+.1f}%")
