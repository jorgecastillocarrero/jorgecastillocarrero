"""3DH OPT: señales del ultimo cierre con regla no-overlap."""
import json, pandas as pd, numpy as np, gc
from sqlalchemy import create_engine
import yfinance as yf

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

SLIP = 0.3
HD = 4  # hold days

def np_sma(arr, window):
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    sma = (cs[window:] - cs[:-window]) / window
    result = np.full(len(arr), np.nan)
    result[window-1:] = sma
    return result

# 1. Load FMP prices
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

# 2. Download recent from yfinance
print('\nDownloading recent prices from yfinance...')
yf_data = yf.download(tickers, period='10d', group_by='ticker', progress=False, threads=True)
print(f'  yfinance dates: {yf_data.index.min().strftime("%Y-%m-%d")} to {yf_data.index.max().strftime("%Y-%m-%d")}')

yf_rows = []
for sym in tickers:
    try:
        sdf = yf_data[sym].dropna(subset=['Close'])
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

df = pd.concat([df_fmp, df_yf], ignore_index=True)
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
del df_fmp, df_yf
gc.collect()

# Target date: last complete close (yesterday, skip weekends)
target = pd.Timestamp('today').normalize() - pd.Timedelta(days=1)
while target.weekday() >= 5:
    target -= pd.Timedelta(days=1)
print(f'  Target signal date: {target.strftime("%Y-%m-%d")}')

# 3. Scan with NO-OVERLAP rule
print(f'\nScanning signals with no-overlap (HD={HD})...')
signals = []
blocked = []
open_positions = []

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

    # Find target index
    target_idx = None
    for j in range(n-1, max(n-15, 0), -1):
        if pd.Timestamp(dates[j]) == target:
            target_idx = j
            break
    if target_idx is None:
        continue

    # Scan from ~15 trading days before target to build no-overlap state
    scan_start = max(203, target_idx - 15)
    busy = -1

    for i in range(scan_start, target_idx + 1):
        if np.isnan(s5[i]) or np.isnan(s50[i]) or np.isnan(s200[i]):
            continue
        d5 = (c[i] / s5[i] - 1) * 100
        if d5 <= 2.0:
            continue
        d50 = (c[i] / s50[i] - 1) * 100
        if d50 >= -5.0:
            continue
        if c[i] >= s200[i]:
            continue
        if i < 3:
            continue
        if h[i-2] <= h[i-3] or lo[i-2] <= lo[i-3]:
            continue
        if h[i-1] <= h[i-2] or lo[i-1] <= lo[i-2]:
            continue
        if h[i] <= h[i-1] or lo[i] <= lo[i-1]:
            continue

        # Entry = next day after signal
        bi = i + 1
        # Exit = entry + HD
        si = bi + HD

        # No-overlap check: entry must be AFTER previous exit
        is_blocked = (bi <= busy)

        sig_date = pd.Timestamp(dates[i])

        if i == target_idx:
            # Target date signal
            info = {
                'symbol': sym,
                'signal_date': sig_date.strftime('%Y-%m-%d'),
                'close': round(c[i], 2),
                'sma5': round(s5[i], 2),
                'sma50': round(s50[i], 2),
                'sma200': round(s200[i], 2),
                'd5': round(d5, 2),
                'd50': round(d50, 2),
                'd200': round(float(c[i] / s200[i] - 1) * 100, 2),
            }
            if is_blocked:
                blocked.append(info)
            else:
                signals.append(info)

        # Update busy (even if si >= n, position is still open)
        if not is_blocked:
            busy = si
            # Track if position is still open (exit not yet reached)
            if si >= target_idx:
                entry_d = pd.Timestamp(dates[bi]).strftime('%Y-%m-%d') if bi < n else 'PENDING'
                exit_d = pd.Timestamp(dates[si]).strftime('%Y-%m-%d') if si < n else 'PENDING'
                entry_p = round(float(o[bi]), 2) if bi < n else 0
                open_positions.append({
                    'symbol': sym,
                    'signal': sig_date.strftime('%Y-%m-%d'),
                    'entry': entry_d,
                    'exit': exit_d,
                    'entry_p': entry_p,
                    'is_today': (i == target_idx),
                })

signals.sort(key=lambda x: x['d50'])

# Print results
print(f'\n{"="*130}')
print(f'  SENALES 3DH OPT - Senal: {target.strftime("%Y-%m-%d")} - ENTRAR SHORT HOY A LA APERTURA')
print(f'{"="*130}')
print(f'  Reglas: Close > SMA5 (+2%) | Close < SMA50 (-5%) | Close < SMA200 | 3 dias HH+HL | No-overlap')
print(f'  Hold: {HD} dias | Slippage: {SLIP}%')

# Open positions from recent signals (not today)
prev_open = [p for p in open_positions if not p['is_today']]
if prev_open:
    print(f'\n  POSICIONES ABIERTAS de senales anteriores ({len(prev_open)}):')
    print(f'  {"Symbol":>8s} {"Signal":>12s} {"Entry":>12s} {"Exit":>12s} {"Entry$":>9s}')
    print(f'  {"-"*58}')
    for p in sorted(prev_open, key=lambda x: x['signal']):
        print(f'  {p["symbol"]:>8s} {p["signal"]:>12s} {p["entry"]:>12s} {p["exit"]:>12s} {p["entry_p"]:9.2f}')

# Blocked signals
if blocked:
    print(f'\n  BLOQUEADAS por no-overlap ({len(blocked)}):')
    for b in blocked:
        print(f'    {b["symbol"]:>8s} - cumple reglas pero tiene posicion abierta')

# New signals
print(f'\n  SENALES NUEVAS: {len(signals)}')
if signals:
    print(f'\n  {"#":>3s} {"Symbol":>8s} {"Close":>8s} {"SMA5":>8s} {"SMA50":>8s} {"SMA200":>8s} {"d5%":>7s} {"d50%":>7s} {"d200%":>7s}')
    print(f'  {"-"*85}')
    for idx, s in enumerate(signals, 1):
        print(f'  {idx:3d} {s["symbol"]:>8s} {s["close"]:8.2f} {s["sma5"]:8.2f} {s["sma50"]:8.2f} {s["sma200"]:8.2f} {s["d5"]:+6.1f}% {s["d50"]:+6.1f}% {s["d200"]:+6.1f}%')
else:
    print('\n  >>> NO hay senales nuevas para hoy')
