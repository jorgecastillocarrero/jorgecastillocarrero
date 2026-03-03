import json, pandas as pd, numpy as np, gc
from sqlalchemy import create_engine
from collections import Counter

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

# ========== LOAD PRICES ==========
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

# Precompute
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
print(f'  {len(sym_data)} symbols')

# ========== SCAN: D50<-5% + D5>+2% ==========
print('\nScanning 3DH D50<-5% D5>+2% (4d hold)...')
trades = []
for sym, (c, o, h, lo, dates, s5, s50, s200, n) in sym_data.items():
    busy = -1
    for i in range(203, n - 1):
        if np.isnan(s5[i]) or np.isnan(s50[i]) or np.isnan(s200[i]):
            continue
        # close > SMA5 by at least 2%
        d5 = (c[i] / s5[i] - 1) * 100
        if d5 <= 2.0:
            continue
        # close < SMA50 by at least 5%
        d50 = (c[i] / s50[i] - 1) * 100
        if d50 >= -5.0:
            continue
        # close < SMA200
        if c[i] >= s200[i]:
            continue
        # 3 consecutive higher highs AND higher lows
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
        sig_date = pd.Timestamp(dates[i])
        entry_date = pd.Timestamp(dates[bi])
        exit_date = pd.Timestamp(dates[si])
        trades.append({
            'symbol': sym, 'year': sig_date.year, 'month': sig_date.month,
            'signal': sig_date, 'entry': entry_date, 'exit': exit_date,
            'entry_p': float(bp), 'exit_p': float(cv),
            'd5': round(float(d5), 1), 'd50': round(float(d50), 1),
            'ret': ret, 'pnl': pnl
        })

print(f'  Total trades: {len(trades):,}')

# ========== 1. YEAR BY YEAR ==========
all_years = sorted(set(t['year'] for t in trades))
print(f'\n{"="*100}')
print(f'  3DH OPTIMIZED: D50<-5% + D5>+2% | 4d hold | ${TRADE_SIZE:,}/trade | {SLIP}% slip')
print(f'{"="*100}')
print(f'  {"Ano":>5s} | {"N":>5s} {"WR":>5s} {"Avg":>7s} {"Med":>7s} {"PF":>5s} {"PnL$":>12s} | {"MaxWin":>8s} {"MaxLoss":>8s} {"Cum$":>12s}')
print(f'  {"-"*90}')

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
    print(f'  {yr:5d} | {len(yr_t):5d} {(rets>0).mean()*100:4.1f}% {rets.mean():+6.2f}% {rets.median():+6.2f}% {pfs} {pnl:+12,.0f} | {rets.max():+7.2f}% {rets.min():+7.2f}% {cum:+12,.0f}')

print(f'  {"-"*90}')
rets_all = pd.Series([t['ret'] for t in trades])
pnl_all = sum(t['pnl'] for t in trades)
w = rets_all[rets_all > 0].sum()
l = abs(rets_all[rets_all < 0].sum())
pf = w / l if l > 0 else 0
print(f'  {"TOTAL":>5s} | {len(trades):5d} {(rets_all>0).mean()*100:4.1f}% {rets_all.mean():+6.2f}% {rets_all.median():+6.2f}% {pf:5.2f} {pnl_all:+12,.0f} |')

# ========== 2. MONTHLY DISTRIBUTION ==========
print(f'\n{"="*100}')
print(f'  DISTRIBUCION MENSUAL')
print(f'{"="*100}')
month_names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
print(f'  {"Mes":>5s} | {"N":>5s} {"WR":>5s} {"Avg":>7s} {"PF":>5s} {"PnL$":>12s} | {"N/yr":>5s}')
print(f'  {"-"*55}')
for m in range(1, 13):
    mt = [t for t in trades if t['month'] == m]
    if mt:
        rets = pd.Series([t['ret'] for t in mt])
        pnl = sum(t['pnl'] for t in mt)
        w = rets[rets > 0].sum()
        l = abs(rets[rets < 0].sum())
        pf = w / l if l > 0 else float('inf')
        pfs = f'{pf:5.2f}' if pf < 99 else '  inf'
        n_yr = len(mt) / len(all_years)
        print(f'  {month_names[m-1]:>5s} | {len(mt):5d} {(rets>0).mean()*100:4.1f}% {rets.mean():+6.2f}% {pfs} {pnl:+12,.0f} | {n_yr:5.1f}')

# ========== 3. CONCURRENT POSITIONS ==========
print(f'\n{"="*100}')
print(f'  CAPITAL CONCURRENTE (posiciones simultaneas)')
print(f'{"="*100}')

# Build daily position count
from datetime import timedelta
all_dates = sorted(set(t['entry'].date() for t in trades) | set(t['exit'].date() for t in trades))
if all_dates:
    date_range = pd.date_range(all_dates[0], all_dates[-1], freq='B')
    daily_count = pd.Series(0, index=date_range)
    for t in trades:
        mask = (date_range >= t['entry']) & (date_range < t['exit'])
        daily_count[mask] += 1

    print(f'  Media:   {daily_count.mean():5.1f} posiciones = ${daily_count.mean()*TRADE_SIZE:,.0f}')
    print(f'  Mediana: {daily_count.median():5.0f} posiciones = ${daily_count.median()*TRADE_SIZE:,.0f}')
    print(f'  P75:     {daily_count.quantile(0.75):5.0f} posiciones = ${daily_count.quantile(0.75)*TRADE_SIZE:,.0f}')
    print(f'  P90:     {daily_count.quantile(0.90):5.0f} posiciones = ${daily_count.quantile(0.90)*TRADE_SIZE:,.0f}')
    print(f'  P95:     {daily_count.quantile(0.95):5.0f} posiciones = ${daily_count.quantile(0.95)*TRADE_SIZE:,.0f}')
    print(f'  Max:     {daily_count.max():5.0f} posiciones = ${daily_count.max()*TRADE_SIZE:,.0f}')

    # By year
    print(f'\n  {"Ano":>5s} | {"Media":>6s} {"Med":>4s} {"Max":>4s} | {"Cap media":>12s} {"Cap max":>12s}')
    print(f'  {"-"*60}')
    for yr in all_years:
        yr_mask = daily_count.index.year == yr
        yr_dc = daily_count[yr_mask]
        if len(yr_dc) > 0:
            print(f'  {yr:5d} | {yr_dc.mean():6.1f} {yr_dc.median():4.0f} {yr_dc.max():4.0f} | ${yr_dc.mean()*TRADE_SIZE:>11,.0f} ${yr_dc.max()*TRADE_SIZE:>11,.0f}')

# ========== 4. TOP SYMBOLS ==========
print(f'\n{"="*100}')
print(f'  TOP 20 SYMBOLS (mas trades)')
print(f'{"="*100}')
sym_stats = {}
for t in trades:
    s = t['symbol']
    if s not in sym_stats:
        sym_stats[s] = []
    sym_stats[s].append(t)

sym_list = []
for s, st in sym_stats.items():
    rets = pd.Series([t['ret'] for t in st])
    pnl = sum(t['pnl'] for t in st)
    w = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    pf = w / l if l > 0 else 99
    sym_list.append((s, len(st), (rets>0).mean()*100, rets.mean(), pf, pnl))

sym_list.sort(key=lambda x: x[1], reverse=True)
print(f'  {"Symbol":>8s} | {"N":>4s} {"WR":>5s} {"Avg":>7s} {"PF":>5s} {"PnL$":>10s}')
print(f'  {"-"*50}')
for s, n, wr, avg, pf, pnl in sym_list[:20]:
    pfs = f'{pf:5.2f}' if pf < 99 else '  inf'
    print(f'  {s:>8s} | {n:4d} {wr:4.1f}% {avg:+6.2f}% {pfs} {pnl:+10,.0f}')

print(f'\n  Total symbols traded: {len(sym_stats)}')

# ========== 5. DRAWDOWN ANALYSIS ==========
print(f'\n{"="*100}')
print(f'  DRAWDOWN ANALYSIS (PnL acumulado)')
print(f'{"="*100}')

# Sort trades by entry date
trades_sorted = sorted(trades, key=lambda t: t['entry'])
cum_pnl = []
running = 0
for t in trades_sorted:
    running += t['pnl']
    cum_pnl.append(running)

cum_s = pd.Series(cum_pnl)
peak = cum_s.cummax()
dd = cum_s - peak
max_dd = dd.min()
max_dd_idx = dd.idxmin()
peak_at = peak[max_dd_idx]
print(f'  Max Drawdown: ${max_dd:,.0f} (desde peak ${peak_at:,.0f})')
print(f'  Trade #{max_dd_idx} de {len(trades_sorted)}')
if max_dd_idx < len(trades_sorted):
    t = trades_sorted[max_dd_idx]
    print(f'  Fecha: {t["entry"].strftime("%Y-%m-%d")}')

# Longest losing streak
streak = 0
max_streak = 0
for t in trades_sorted:
    if t['ret'] < 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0
print(f'  Max losing streak: {max_streak} trades consecutivos')

# ========== 6. COMPARATIVA CON RELAX5 ==========
print(f'\n{"="*100}')
print(f'  RESUMEN FINAL: 3DH OPT vs RELAX5 15d')
print(f'{"="*100}')

# RELAX5 known results from memory
print(f'\n  {"Metrica":<25s} {"3DH OPT 4d":>15s} {"RELAX5 15d":>15s}')
print(f'  {"-"*55}')
print(f'  {"Trades":<25s} {len(trades):>15,} {"2,108":>15s}')
print(f'  {"Trades/year":<25s} {len(trades)/len(all_years):>15,.0f} {"100":>15s}')
print(f'  {"Trades/month":<25s} {len(trades)/len(all_years)/12:>15,.0f} {"8":>15s}')
print(f'  {"Win Rate":<25s} {(rets_all>0).mean()*100:>14.1f}% {"47.2%":>15s}')
print(f'  {"Avg Return":<25s} {rets_all.mean():>+14.2f}% {"+0.59%":>15s}')
print(f'  {"Profit Factor":<25s} {pf:>15.2f} {"1.18":>15s}')
pnl_s = f'${pnl_all:+,.0f}'
print(f'  {"PnL Total":<25s} {pnl_s:>15s} {"$+313K":>15s}')
ppt_s = f'${pnl_all/len(trades):+,.0f}'
print(f'  {"PnL/Trade":<25s} {ppt_s:>15s} {"$+148":>15s}')
print(f'  {"Hold period":<25s} {"4 days":>15s} {"15 days":>15s}')
print(f'  {"Capital/trade":<25s} {"$25,000":>15s} {"$25,000":>15s}')
if all_dates:
    cap_s = f'${daily_count.median()*TRADE_SIZE:,.0f}'
    print(f'  {"Capital mediana":<25s} {cap_s:>15s} {"$100,000":>15s}')
    cap95 = f'${daily_count.quantile(0.95)*TRADE_SIZE:,.0f}'
    print(f'  {"Capital P95":<25s} {cap95:>15s} {"$625,000":>15s}')
ann_s = f'${pnl_all/len(all_years):+,.0f}'
print(f'  {"PnL/year":<25s} {ann_s:>15s} {"$+14,209":>15s}')
print(f'  {"Max DD":<25s} ${max_dd:>+14,.0f} {"":>15s}')
