"""3DH OPT 4d: listado completo de trades y rentabilidad año a año."""
import json, pandas as pd, numpy as np, gc
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

SLIP = 0.3
TRADE_SIZE = 25000
HD = 4

def np_sma(arr, window):
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    sma = (cs[window:] - cs[:-window]) / window
    result = np.full(len(arr), np.nan)
    result[window-1:] = sma
    return result

print('Loading prices...')
prc = []
for i in range(0, len(tickers), 25):
    b = tickers[i:i+25]
    prc.append(pd.read_sql("""SELECT symbol, date, open, high, low, close
       FROM fmp_price_history WHERE symbol = ANY(%(t)s) AND date >= '2005-01-01'
       ORDER BY symbol, date""", engine, params={'t': b}))
    if (i // 25) % 5 == 0:
        print(f'  Batch {i//25+1}/{(len(tickers)+24)//25}...')
df = pd.concat(prc, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
del prc
gc.collect()

print('Precomputing...')
sym_data = {}
for sym, g in df.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < 210:
        continue
    c = g['close'].values.astype(np.float32)
    o = g['open'].values.astype(np.float32)
    h = g['high'].values.astype(np.float32)
    lo = g['low'].values.astype(np.float32)
    dates = g['date'].values
    s5 = np_sma(c, 5)
    s50 = np_sma(c, 50)
    s200 = np_sma(c, 200)
    sym_data[sym] = (c, o, h, lo, dates, s5, s50, s200, n)
del df
gc.collect()

print('Scanning trades...')
trades = []
for sym, (c, o, h, lo, dates, s5, s50, s200, n) in sym_data.items():
    busy = -1
    for i in range(203, n - 1):
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
        if h[i-2] <= h[i-3] or lo[i-2] <= lo[i-3]:
            continue
        if h[i-1] <= h[i-2] or lo[i-1] <= lo[i-2]:
            continue
        if h[i] <= h[i-1] or lo[i] <= lo[i-1]:
            continue
        bi = i + 1
        if bi >= n or bi <= busy:
            continue
        bp = o[bi]
        if bp <= 0:
            continue
        si = bi + HD
        if si >= n:
            continue
        cv = o[si]
        ret = round(float(bp / cv - 1) * 100 - SLIP, 2)
        pnl = round(TRADE_SIZE * ret / 100, 2)
        busy = si
        sig_d = pd.Timestamp(dates[i])
        entry_d = pd.Timestamp(dates[bi])
        exit_d = pd.Timestamp(dates[si])
        trades.append({
            'symbol': sym, 'year': sig_d.year,
            'signal': sig_d.strftime('%Y-%m-%d'),
            'entry': entry_d.strftime('%Y-%m-%d'),
            'exit': exit_d.strftime('%Y-%m-%d'),
            'entry_p': round(float(bp), 2),
            'exit_p': round(float(cv), 2),
            'ret': ret, 'pnl': pnl,
        })

trades.sort(key=lambda t: t['entry'])
print(f'Total trades: {len(trades):,}')

# ========== YEAR BY YEAR ==========
all_years = sorted(set(t['year'] for t in trades))

print(f'\n{"="*100}')
print(f'  3DH OPT (D50<-5% D5>+2%) | 4d hold | ${TRADE_SIZE:,}/trade | {SLIP}% slip')
print(f'{"="*100}')
print(f'  {"Ano":>5s} | {"N":>5s} {"WR":>5s} {"Avg":>7s} {"Med":>7s} {"PF":>5s} {"PnL$":>12s} {"PnL/Tr":>8s} | {"Cum$":>12s}')
print(f'  {"-"*80}')

cum = 0
for yr in all_years:
    yr_t = [t for t in trades if t['year'] == yr]
    rets = pd.Series([t['ret'] for t in yr_t])
    pnl = sum(t['pnl'] for t in yr_t)
    cum += pnl
    w = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    pf = w / l if l > 0 else float('inf')
    pfs = f'{pf:5.2f}' if pf < 99 else '  inf'
    ppt = pnl / len(yr_t)
    print(f'  {yr:5d} | {len(yr_t):5d} {(rets>0).mean()*100:4.1f}% {rets.mean():+6.2f}% {rets.median():+6.2f}% {pfs} {pnl:+12,.0f} {ppt:+8,.0f} | {cum:+12,.0f}')

print(f'  {"-"*80}')
rets_all = pd.Series([t['ret'] for t in trades])
pnl_all = sum(t['pnl'] for t in trades)
w = rets_all[rets_all > 0].sum()
l = abs(rets_all[rets_all < 0].sum())
pf = w / l if l > 0 else 0
print(f'  {"TOTAL":>5s} | {len(trades):5d} {(rets_all>0).mean()*100:4.1f}% {rets_all.mean():+6.2f}% {rets_all.median():+6.2f}% {pf:5.2f} {pnl_all:+12,.0f} {pnl_all/len(trades):+8,.0f} |')

# ========== LAST 50 TRADES ==========
print(f'\n{"="*120}')
print(f'  ULTIMOS 50 TRADES')
print(f'{"="*120}')
print(f'  {"#":>4s} {"Symbol":>8s} {"Signal":>12s} {"Entry":>12s} {"Exit":>12s} {"Entry$":>9s} {"Exit$":>9s} {"Ret%":>7s} {"PnL$":>9s}')
print(f'  {"-"*95}')
last50 = trades[-50:]
for idx, t in enumerate(last50, len(trades)-49):
    color = '+' if t['ret'] > 0 else ''
    print(f'  {idx:4d} {t["symbol"]:>8s} {t["signal"]:>12s} {t["entry"]:>12s} {t["exit"]:>12s} {t["entry_p"]:9.2f} {t["exit_p"]:9.2f} {t["ret"]:+6.2f}% {t["pnl"]:+9,.0f}')

# ========== 2025-2026 ALL TRADES ==========
print(f'\n{"="*120}')
print(f'  TODOS LOS TRADES 2025-2026')
print(f'{"="*120}')
print(f'  {"#":>4s} {"Symbol":>8s} {"Signal":>12s} {"Entry":>12s} {"Exit":>12s} {"Entry$":>9s} {"Exit$":>9s} {"Ret%":>7s} {"PnL$":>9s}')
print(f'  {"-"*95}')
recent = [t for t in trades if t['year'] >= 2025]
for idx, t in enumerate(recent, 1):
    print(f'  {idx:4d} {t["symbol"]:>8s} {t["signal"]:>12s} {t["entry"]:>12s} {t["exit"]:>12s} {t["entry_p"]:9.2f} {t["exit_p"]:9.2f} {t["ret"]:+6.2f}% {t["pnl"]:+9,.0f}')

print(f'\n  2025: {sum(1 for t in recent if t["year"]==2025)} trades, PnL: ${sum(t["pnl"] for t in recent if t["year"]==2025):+,.0f}')
print(f'  2026: {sum(1 for t in recent if t["year"]==2026)} trades, PnL: ${sum(t["pnl"] for t in recent if t["year"]==2026):+,.0f}')

# Save JSON cache for panel_control
cache = [{'sym': t['symbol'], 'sig': t['signal'], 'entry': t['entry'],
          'exit': t['exit'], 'ret': t['ret'], 'pnl': t['pnl']} for t in trades]
with open('data/3dh_opt_4d_trades.json', 'w') as f:
    json.dump(cache, f, separators=(',', ':'))
print(f'\nSaved {len(cache)} trades to data/3dh_opt_4d_trades.json')
