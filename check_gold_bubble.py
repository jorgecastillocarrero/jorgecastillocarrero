"""Verificar si ha habido burbuja en oro/gold miners"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# Buscar simbolos relacionados con oro
gold_symbols = ['GLD', 'GDX', 'GDXJ', 'NEM', 'GOLD', 'AEM', 'FNV', 'WPM', 'KGC', 'AU', 'IAU']
print("Verificando disponibilidad de simbolos gold...")
for sym in gold_symbols:
    df = pd.read_sql(f"""
        SELECT MIN(date) as min_d, MAX(date) as max_d, COUNT(*) as n
        FROM fmp_price_history WHERE symbol = '{sym}'
    """, engine)
    r = df.iloc[0]
    if r['n'] > 0:
        print(f"  {sym}: {r['min_d']} a {r['max_d']} ({r['n']} registros)")
    else:
        print(f"  {sym}: NO encontrado")

# Usar GLD (Gold ETF, desde 2004) y gold miners
print("\nCargando datos...")
gold_etfs = ['GLD', 'GDX']
miners = ['NEM', 'GOLD', 'AEM', 'FNV', 'WPM', 'KGC']
all_gold = gold_etfs + miners
tlist = "','".join(all_gold)

df = pd.read_sql(f"""
    SELECT symbol, date, close
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY symbol, date
""", engine)
df['date'] = pd.to_datetime(df['date'])

def calc_index_metrics(series):
    d = pd.DataFrame({'close': series}).dropna()
    d['ma200'] = d['close'].rolling(200).mean()
    d['dist_ma200'] = (d['close'] / d['ma200'] - 1) * 100
    w = d.resample('W-FRI').last().dropna(subset=['ma200'])
    w['mom_10w'] = w['close'].pct_change(10) * 100
    w['high_52w'] = w['close'].rolling(52, min_periods=26).max()
    w['dd_52w'] = (w['close'] / w['high_52w'] - 1) * 100
    delta = w['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    w['rsi'] = 100 - (100 / (1 + rs))
    return w

# GLD
gld = df[df['symbol'] == 'GLD'].set_index('date').sort_index()
if len(gld) > 200:
    gld_w = calc_index_metrics(gld['close'])
    print(f"\nGLD rango: {gld.index.min().date()} a {gld.index.max().date()}")

    # Periodos con dist > 15% (burbuja potencial)
    print("\n" + "=" * 100)
    print("GLD: SEMANAS CON DIST MA200 > 15%")
    print("=" * 100)
    bubble = gld_w[gld_w['dist_ma200'] > 15]
    if len(bubble) > 0:
        print(f"\nTotal semanas: {len(bubble)}")
        print(f"\n{'Fecha':>12} {'Close':>8} {'Dist%':>7} {'Mom10w':>8} {'DD52w':>7} {'RSI':>6}")
        print("-" * 55)
        for date, r in bubble.iterrows():
            print(f"{date.strftime('%Y-%m-%d'):>12} {r['close']:>8.2f} {r['dist_ma200']:>+6.1f}% "
                  f"{r['mom_10w']:>+7.1f}% {r['dd_52w']:>+6.1f}% {r['rsi']:>5.1f}")
    else:
        print("  Ninguna semana con dist > 15%")

    # Maximos historicos
    print("\n" + "=" * 100)
    print("GLD: MAXIMOS POR PERIODO")
    print("=" * 100)
    for period, start, end in [
        ('2005-2007 (pre-crisis)', '2005-01-01', '2007-12-31'),
        ('2008-2009 (crisis)', '2008-01-01', '2009-12-31'),
        ('2010-2011 (BURBUJA ORO)', '2010-01-01', '2011-12-31'),
        ('2012-2015 (post-burbuja)', '2012-01-01', '2015-12-31'),
        ('2016-2019', '2016-01-01', '2019-12-31'),
        ('2020-2021 (COVID)', '2020-01-01', '2021-12-31'),
        ('2022-2023', '2022-01-01', '2023-12-31'),
        ('2024-2026', '2024-01-01', '2026-12-31'),
    ]:
        p = gld_w[(gld_w.index >= start) & (gld_w.index <= end)]
        if len(p) == 0:
            continue
        max_dist = p['dist_ma200'].max()
        max_mom = p['mom_10w'].max()
        max_rsi = p['rsi'].max()
        max_close = p['close'].max()
        print(f"  {period:<30} Close max={max_close:>8.2f}  Dist={max_dist:>+6.1f}%  "
              f"Mom={max_mom:>+6.1f}%  RSI={max_rsi:>5.1f}")

# Gold Miners index (equal weight)
print("\n" + "=" * 100)
print("GOLD MINERS INDEX (NEM, GOLD, AEM, FNV, WPM, KGC)")
print("=" * 100)
miners_daily = df[df['symbol'].isin(miners)].pivot(index='date', columns='symbol', values='close')
# Normalizar
for col in miners_daily.columns:
    first = miners_daily[col].first_valid_index()
    if first is not None:
        miners_daily[col] = miners_daily[col] / miners_daily[col].loc[first] * 100
miners_daily['index'] = miners_daily.mean(axis=1)
miners_daily = miners_daily.dropna(subset=['index'])

if len(miners_daily) > 200:
    miners_w = calc_index_metrics(miners_daily['index'])
    print(f"Rango: {miners_daily.index.min().date()} a {miners_daily.index.max().date()}")

    print(f"\n{'Periodo':<30} {'Max Dist%':>10} {'Max Mom%':>10} {'Max RSI':>8}")
    print("-" * 65)
    for period, start, end in [
        ('2005-2007 (pre-crisis)', '2005-01-01', '2007-12-31'),
        ('2008-2009 (crisis)', '2008-01-01', '2009-12-31'),
        ('2010-2011 (BURBUJA ORO)', '2010-01-01', '2011-12-31'),
        ('2012-2015 (post-burbuja)', '2012-01-01', '2015-12-31'),
        ('2016-2019', '2016-01-01', '2019-12-31'),
        ('2020-2021 (COVID)', '2020-01-01', '2021-12-31'),
        ('2022-2023', '2022-01-01', '2023-12-31'),
        ('2024-2026', '2024-01-01', '2026-12-31'),
    ]:
        p = miners_w[(miners_w.index >= start) & (miners_w.index <= end)]
        if len(p) == 0:
            continue
        print(f"  {period:<30} {p['dist_ma200'].max():>+9.1f}% {p['mom_10w'].max():>+9.1f}% "
              f"{p['rsi'].max():>7.1f}")

    # Detalle 2010-2011 (burbuja oro conocida)
    print("\n  Detalle 2010-2011 (burbuja oro):")
    p = miners_w[(miners_w.index >= '2010-06-01') & (miners_w.index <= '2011-09-30')]
    high_weeks = p[p['dist_ma200'] > 15]
    if len(high_weeks) > 0:
        print(f"  Semanas con dist > 15%: {len(high_weeks)}")
        print(f"\n  {'Fecha':>12} {'Dist%':>7} {'Mom10w':>8} {'DD52w':>7} {'RSI':>6}")
        print(f"  {'-'*45}")
        for date, r in high_weeks.iterrows():
            marker = " <<< BURBUJA" if r['dist_ma200'] > 25 else ""
            print(f"  {date.strftime('%Y-%m-%d'):>12} {r['dist_ma200']:>+6.1f}% "
                  f"{r['mom_10w']:>+7.1f}% {r['dd_52w']:>+6.1f}% {r['rsi']:>5.1f}{marker}")

# 2024-2026 actual
print("\n" + "=" * 100)
print("GLD ACTUAL 2024-2026: Estamos en burbuja de oro?")
print("=" * 100)
if len(gld) > 200:
    recent = gld_w[gld_w.index >= '2024-01-01']
    high_recent = recent[recent['dist_ma200'] > 10]
    if len(high_recent) > 0:
        print(f"  Semanas con dist > 10%: {len(high_recent)}")
        print(f"\n  {'Fecha':>12} {'Close':>8} {'Dist%':>7} {'Mom10w':>8} {'DD52w':>7} {'RSI':>6}")
        print(f"  {'-'*55}")
        for date, r in high_recent.iterrows():
            marker = " <<< BURBUJA" if r['dist_ma200'] > 20 else " << FUERTE" if r['dist_ma200'] > 15 else ""
            print(f"  {date.strftime('%Y-%m-%d'):>12} {r['close']:>8.2f} {r['dist_ma200']:>+6.1f}% "
                  f"{r['mom_10w']:>+7.1f}% {r['dd_52w']:>+6.1f}% {r['rsi']:>5.1f}{marker}")
    else:
        print("  Ninguna semana con dist > 10%")
