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
print(f'  {len(df):,} rows, {df["symbol"].nunique()} symbols')

# Precompute per symbol
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
    s10 = np_sma(c, 10)
    s50 = np_sma(c, 50)
    s100 = np_sma(c, 100)
    s200 = np_sma(c, 200)
    sym_data[sym] = (c, o, h, lo, dates, s5, s10, s50, s100, s200, n)
del df
gc.collect()
print(f'  {len(sym_data)} symbols ready')

# Configurations to test
# (name, days_pattern, dist_sma5_min, dist_sma50_max, dist_sma200_max, dist_sma100_max, close_above_sma10)
# dist_sma50_max: close must be < SMA50 by at least this % (e.g., -5 means close < 95% of SMA50)
# dist_sma200_max: close must be < SMA200 by at least this %

configs = [
    # name, n_days, d5_min, d50_max, d200_max, d100_max, need_below_s10
    ('BASE (original)',           3, 0,    0,     0,     None, False),
    ('D50<-3%',                   3, 0,   -3,     0,     None, False),
    ('D50<-5%',                   3, 0,   -5,     0,     None, False),
    ('D200<-3%',                  3, 0,    0,    -3,     None, False),
    ('D200<-5%',                  3, 0,    0,    -5,     None, False),
    ('D50<-3% D200<-3%',          3, 0,   -3,    -3,     None, False),
    ('D50<-5% D200<-5%',          3, 0,   -5,    -5,     None, False),
    ('D5>+2%',                    3, 2,    0,     0,     None, False),
    ('D5>+3%',                    3, 3,    0,     0,     None, False),
    ('4 days pattern',            4, 0,    0,     0,     None, False),
    ('5 days pattern',            5, 0,    0,     0,     None, False),
    ('D50<-5% + D5>+2%',         3, 2,   -5,     0,     None, False),
    ('D50<-3% + 4days',           4, 0,   -3,     0,     None, False),
    ('D50<-5% + 4days',           4, 0,   -5,     0,     None, False),
    ('D200<-5% + 4days',          4, 0,    0,    -5,     None, False),
    ('D50<-5% D200<-5% + 4d',    4, 0,   -5,    -5,     None, False),
    ('<SMA100',                   3, 0,    0,     0,       0,  False),
    ('D50<-5% + <SMA100',        3, 0,   -5,     0,       0,  False),
    ('<SMA10 + D50<-5%',          3, 0,   -5,     0,     None, True),
]

results = []

for cfg_name, n_days, d5_min, d50_max, d200_max, d100_max, need_below_s10 in configs:
    trades = []
    start_idx = 200 + n_days
    for sym, (c, o, h, lo, dates, s5, s10, s50, s100, s200, n) in sym_data.items():
        busy = -1
        for i in range(start_idx, n - 1):
            if np.isnan(s5[i]) or np.isnan(s50[i]) or np.isnan(s200[i]):
                continue
            # close > SMA5
            d5 = (c[i] / s5[i] - 1) * 100
            if d5 <= d5_min:
                continue
            # close < SMA50 (by at least d50_max %)
            d50 = (c[i] / s50[i] - 1) * 100
            if d50 >= d50_max:
                continue
            # close < SMA200 (by at least d200_max %)
            d200 = (c[i] / s200[i] - 1) * 100
            if d200 >= d200_max:
                continue
            # Optional: close < SMA100
            if d100_max is not None:
                if np.isnan(s100[i]):
                    continue
                d100 = (c[i] / s100[i] - 1) * 100
                if d100 >= d100_max:
                    continue
            # Optional: close < SMA10
            if need_below_s10:
                if np.isnan(s10[i]) or c[i] >= s10[i]:
                    continue
            # N consecutive higher highs AND higher lows
            ok = True
            for k in range(n_days):
                day = i - (n_days - 1 - k)
                if h[day] <= h[day-1] or lo[day] <= lo[day-1]:
                    ok = False
                    break
            if not ok:
                continue
            # Entry/exit
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
            yr = pd.Timestamp(dates[i]).year
            trades.append({'year': yr, 'ret': ret, 'pnl': pnl})

    results.append((cfg_name, trades))

# ========== PRINT RESULTS ==========
print(f'\n{"="*140}')
print(f'  3 DAY HIGH SHORT OPTIMIZATION - {HD}d hold - ${TRADE_SIZE:,}/trade - {SLIP}% slippage')
print(f'{"="*140}')
print(f'\n{"Config":<30s} | {"Trades":>7s} | {"WR":>5s} | {"Avg":>7s} | {"PF":>5s} | {"PnL Total":>12s} | {"PnL/Trade":>10s} | {"PnL/Year":>10s} | {"Best Year":>20s} | {"Worst Year":>20s}')
print('-' * 140)

for cfg_name, trades in results:
    if not trades:
        print(f'{cfg_name:<30s} |       0 |    - |       - |     - |            - |          - |          - |')
        continue
    rets = pd.Series([t['ret'] for t in trades])
    pnls = pd.Series([t['pnl'] for t in trades])
    n = len(rets)
    wr = (rets > 0).mean() * 100
    avg = rets.mean()
    w = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    pf = w / l if l > 0 else 0
    ppt = pnls.sum() / n
    years = sorted(set(t['year'] for t in trades))
    n_yrs = len(years)
    ann = pnls.sum() / n_yrs if n_yrs > 0 else 0

    best_yr, best_pnl = None, -1e9
    worst_yr, worst_pnl = None, 1e9
    for yr in years:
        yp = sum(t['pnl'] for t in trades if t['year'] == yr)
        if yp > best_pnl:
            best_yr, best_pnl = yr, yp
        if yp < worst_pnl:
            worst_yr, worst_pnl = yr, yp

    best_s = f'{best_yr}: ${best_pnl:+,.0f}'
    worst_s = f'{worst_yr}: ${worst_pnl:+,.0f}'
    print(f'{cfg_name:<30s} | {n:>7,} | {wr:4.1f}% | {avg:+6.2f}% | {pf:5.2f} | {pnls.sum():>+12,.0f} | {ppt:>+10,.0f} | {ann:>+10,.0f} | {best_s:>20s} | {worst_s:>20s}')

# ========== TOP 3: YEAR BY YEAR ==========
# Sort by PF descending, take top 3 with >500 trades
ranked = [(name, trades) for name, trades in results if len(trades) >= 300]
ranked.sort(key=lambda x: (
    pd.Series([t['ret'] for t in x[1]])[pd.Series([t['ret'] for t in x[1]]) > 0].sum() /
    max(abs(pd.Series([t['ret'] for t in x[1]])[pd.Series([t['ret'] for t in x[1]]) < 0].sum()), 0.01)
), reverse=True)

top3 = ranked[:3]
print(f'\n{"="*140}')
print(f'  TOP 3 CONFIGS (min 300 trades) - YEAR BY YEAR')
print(f'{"="*140}')

all_years = sorted(set(t['year'] for _, trades in top3 for t in trades))

print(f'{"":>5s}', end='')
for name, _ in top3:
    short = name[:18]
    print(f' | {"--- " + short + " ---":^38s}', end='')
print()
print(f'{"Ano":>5s}', end='')
for _ in top3:
    print(f' | {"N":>5s} {"WR":>5s} {"Avg":>7s} {"PF":>5s} {"PnL$":>12s}', end='')
print()
print('-' * 130)

for yr in all_years:
    print(f'{yr:5d}', end='')
    for name, trades in top3:
        yr_t = [t for t in trades if t['year'] == yr]
        if yr_t:
            rets = pd.Series([t['ret'] for t in yr_t])
            pnl = sum(t['pnl'] for t in yr_t)
            w = rets[rets > 0].sum()
            l = abs(rets[rets < 0].sum())
            pf = w / l if l > 0 else float('inf')
            pfs = f'{pf:5.2f}' if pf < 99 else '  inf'
            print(f' | {len(yr_t):5d} {(rets>0).mean()*100:4.1f}% {rets.mean():+6.2f}% {pfs} {pnl:+12,.0f}', end='')
        else:
            print(f' |     0    -       -     -            -', end='')
    print()

print('-' * 130)
print(f'{"TOTAL":>5s}', end='')
for name, trades in top3:
    rets = pd.Series([t['ret'] for t in trades])
    pnl = sum(t['pnl'] for t in trades)
    w = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    pf = w / l if l > 0 else 0
    print(f' | {len(trades):5d} {(rets>0).mean()*100:4.1f}% {rets.mean():+6.2f}% {pf:5.2f} {pnl:+12,.0f}', end='')
print()

# Trades per year avg
print(f'\n  Trades/year promedio:')
for name, trades in top3:
    yrs = set(t['year'] for t in trades)
    print(f'    {name:<30s}: {len(trades)/len(yrs):.0f} trades/year ({len(trades)/len(yrs)/12:.0f}/month)')
