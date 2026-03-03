"""3DH OPT: señales del ultimo dia disponible para entrar al dia siguiente."""
import json, pandas as pd, numpy as np, gc
from sqlalchemy import create_engine
import yfinance as yf

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

SLIP = 0.3

def np_sma(arr, window):
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    sma = (cs[window:] - cs[:-window]) / window
    result = np.full(len(arr), np.nan)
    result[window-1:] = sma
    return result

# 1. Load FMP prices (history for SMAs)
print('Loading FMP prices...')
prc = []
for i in range(0, len(tickers), 25):
    b = tickers[i:i+25]
    prc.append(pd.read_sql("""SELECT symbol, date, open, high, low, close
       FROM fmp_price_history WHERE symbol = ANY(%(t)s) AND date >= '2024-01-01'
       ORDER BY symbol, date""", engine, params={'t': b}))
    if (i // 25) % 5 == 0:
        print(f'  Batch {i//25+1}/{(len(tickers)+24)//25}...')
df_fmp = pd.concat(prc, ignore_index=True)
df_fmp['date'] = pd.to_datetime(df_fmp['date'])
del prc
gc.collect()
fmp_max = df_fmp['date'].max()
print(f'  FMP data through: {fmp_max.strftime("%Y-%m-%d")}')

# 2. Download recent from yfinance (last 10 days to fill gap)
print('\nDownloading recent prices from yfinance...')
yf_data = yf.download(tickers, period='10d', group_by='ticker', progress=False, threads=True)
print(f'  yfinance dates: {yf_data.index.min().strftime("%Y-%m-%d")} to {yf_data.index.max().strftime("%Y-%m-%d")}')

# Build combined dataframe per symbol
# Parse yfinance into rows
yf_rows = []
for sym in tickers:
    try:
        if len(tickers) > 1:
            sdf = yf_data[sym].dropna(subset=['Close'])
        else:
            sdf = yf_data.dropna(subset=['Close'])
        for dt, row in sdf.iterrows():
            if pd.Timestamp(dt) > fmp_max:
                yf_rows.append({
                    'symbol': sym, 'date': pd.Timestamp(dt),
                    'open': row['Open'], 'high': row['High'],
                    'low': row['Low'], 'close': row['Close']
                })
    except:
        pass

df_yf = pd.DataFrame(yf_rows)
print(f'  New rows from yfinance: {len(df_yf):,}')

# Combine
df = pd.concat([df_fmp, df_yf], ignore_index=True)
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
del df_fmp, df_yf
gc.collect()

latest_date = df['date'].max()
print(f'  Combined data through: {latest_date.strftime("%Y-%m-%d")}')

# 3. Scan for signals on the LAST available date
print(f'\nScanning signals...')
signals = []

for sym, g in df.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < 210:
        continue
    c = g['close'].values.astype(np.float64)
    o = g['open'].values.astype(np.float64)
    h = g['high'].values.astype(np.float64)
    lo = g['low'].values.astype(np.float64)
    dates = g['date'].values
    s5 = np_sma(c, 5)
    s50 = np_sma(c, 50)
    s200 = np_sma(c, 200)

    # Last complete close: 2026-03-02 (Monday)
    target = pd.Timestamp('2026-03-02')
    valid_idx = [j for j in range(n) if pd.Timestamp(dates[j]) <= target]
    if not valid_idx:
        continue
    i = valid_idx[-1]
    if pd.Timestamp(dates[i]) != target:
        continue
    if np.isnan(s5[i]) or np.isnan(s50[i]) or np.isnan(s200[i]):
        continue

    sig_date = pd.Timestamp(dates[i])

    d5 = (c[i] / s5[i] - 1) * 100
    d50 = (c[i] / s50[i] - 1) * 100
    d200 = (c[i] / s200[i] - 1) * 100

    # Filters
    if d5 <= 2.0:
        continue
    if d50 >= -5.0:
        continue
    if c[i] >= s200[i]:
        continue

    # 3 consecutive higher highs AND higher lows
    if i < 3:
        continue
    if h[i-2] <= h[i-3] or lo[i-2] <= lo[i-3]:
        continue
    if h[i-1] <= h[i-2] or lo[i-1] <= lo[i-2]:
        continue
    if h[i] <= h[i-1] or lo[i] <= lo[i-1]:
        continue

    signals.append({
        'symbol': sym,
        'signal_date': sig_date.strftime('%Y-%m-%d'),
        'close': round(c[i], 2),
        'sma5': round(s5[i], 2),
        'sma50': round(s50[i], 2),
        'sma200': round(s200[i], 2),
        'd5': round(d5, 2),
        'd50': round(d50, 2),
        'd200': round(d200, 2),
        'h_3': round(h[i-3], 2), 'l_3': round(lo[i-3], 2),
        'h_2': round(h[i-2], 2), 'l_2': round(lo[i-2], 2),
        'h_1': round(h[i-1], 2), 'l_1': round(lo[i-1], 2),
        'h_0': round(h[i], 2),   'l_0': round(lo[i], 2),
    })

signals.sort(key=lambda x: x['d50'])

print(f'\n{"="*120}')
print(f'  SENALES 3DH OPT - Fecha senal: 2026-03-02 - ENTRAR SHORT HOY 03/03 A LA APERTURA')
print(f'{"="*120}')
print(f'  Reglas: Close > SMA5 (+2%) | Close < SMA50 (-5%) | Close < SMA200 | 3 dias HH+HL consecutivos')
print(f'  Accion: SHORT al open del dia siguiente, cubrir al open 4 dias despues')
print(f'  Slippage: {SLIP}%')
print(f'\n  Total senales: {len(signals)}')

if signals:
    print(f'\n  {"#":>3s} {"Symbol":>8s} {"Close":>8s} {"SMA5":>8s} {"SMA50":>8s} {"SMA200":>8s} {"d5%":>7s} {"d50%":>7s} {"d200%":>7s} | {"H-3":>7s} {"L-3":>7s} {"H-2":>7s} {"L-2":>7s} {"H-1":>7s} {"L-1":>7s} {"H-0":>7s} {"L-0":>7s}')
    print(f'  {"-"*140}')
    for idx, s in enumerate(signals, 1):
        print(f'  {idx:3d} {s["symbol"]:>8s} {s["close"]:8.2f} {s["sma5"]:8.2f} {s["sma50"]:8.2f} {s["sma200"]:8.2f} {s["d5"]:+6.1f}% {s["d50"]:+6.1f}% {s["d200"]:+6.1f}% | {s["h_3"]:7.2f} {s["l_3"]:7.2f} {s["h_2"]:7.2f} {s["l_2"]:7.2f} {s["h_1"]:7.2f} {s["l_1"]:7.2f} {s["h_0"]:7.2f} {s["l_0"]:7.2f}')
else:
    print('\n  >>> NO hay senales para hoy')
