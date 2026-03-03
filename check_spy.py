import pandas as pd, sqlalchemy, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

r = pd.read_sql("SELECT date, close FROM fmp_price_history WHERE symbol='SPY' AND date BETWEEN '2026-02-19' AND '2026-02-27' ORDER BY date", engine)
print('SPY precios diarios:')
print(r.to_string(index=False))

r2 = pd.read_sql("SELECT AVG(close) as ma200 FROM (SELECT close FROM fmp_price_history WHERE symbol='SPY' ORDER BY date DESC LIMIT 200) sub", engine)
ma200 = float(r2.iloc[0]['ma200'])
print(f'\nMA200 actual: {ma200:.2f}')
for _, row in r.iterrows():
    dist = (row['close'] / ma200 - 1) * 100
    print(f"  {row['date']}: ${row['close']:.2f}  dist MA200={dist:+.1f}%  {'> 5% -> +1.5' if dist > 5 else '<= 5% -> +0.5'}")

# Check SPY resample W-FRI
spy_daily = pd.read_sql("SELECT date, close FROM fmp_price_history WHERE symbol='SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
print(f"\nSPY resample W-FRI ultimas 3 semanas:")
for _, row in spy_w.tail(3).iterrows():
    dist = (row['close'] / row['ma200'] - 1) * 100
    print(f"  {_.strftime('%Y-%m-%d')}: ${row['close']:.2f}  MA200=${row['ma200']:.2f}  dist={dist:+.1f}%")

# El problema: target_date=2026-02-26 es jueves, W-FRI resample coge el viernes 2026-02-21 como ultimo viernes completo
# O bien coge los datos parciales de la semana en curso
target = pd.Timestamp('2026-02-26')
spy_w_target = spy_w[spy_w.index <= target]
last_w = spy_w_target.iloc[-1]
print(f"\nEl script usa spy_w <= 2026-02-26:")
print(f"  Fecha: {spy_w_target.index[-1].strftime('%Y-%m-%d')}")
print(f"  Close: ${last_w['close']:.2f}")
print(f"  MA200: ${last_w['ma200']:.2f}")
dist_used = (last_w['close'] / last_w['ma200'] - 1) * 100
print(f"  Dist: {dist_used:+.1f}% -> s_spy = {'+1.5' if dist_used > 5 else '+0.5'}")
print(f"\n  ESTO ES EL BUG: resample W-FRI con datos hasta jueves 26/02")
print(f"  coge el close del JUEVES como si fuera el cierre semanal del viernes")
print(f"  Pero el SPY real del jueves 26/02 fue $682.39, no ${last_w['close']:.2f}")
