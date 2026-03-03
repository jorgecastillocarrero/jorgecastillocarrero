import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json','r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

# Load earnings
print("Loading earnings...")
earn_chunks = []
for i in range(0, len(tickers), 50):
    batch = tickers[i:i+50]
    q = """SELECT symbol, date, eps_actual, eps_estimated
           FROM fmp_earnings WHERE symbol = ANY(%(t)s) AND eps_actual IS NOT NULL
           AND date >= '2018-01-01' ORDER BY symbol, date"""
    earn_chunks.append(pd.read_sql(q, engine, params={'t': batch}))
earn = pd.concat(earn_chunks, ignore_index=True)
earn['date'] = pd.to_datetime(earn['date'])
earn['eps_actual'] = earn['eps_actual'].astype(float)
earn['eps_estimated'] = earn['eps_estimated'].astype(float)

def compute_earn_status(grp):
    grp = grp.sort_values('date').reset_index(drop=True)
    grp['beat'] = grp['eps_actual'] > grp['eps_estimated']
    grp['beat_4q'] = grp['beat'].rolling(4, min_periods=4).sum() == 4
    grp['eps_ttm'] = grp['eps_actual'].rolling(4, min_periods=4).sum()
    grp['eps_ttm_prev'] = grp['eps_ttm'].shift(4)
    grp['eps_growth'] = np.where(grp['eps_ttm_prev'] > 0, (grp['eps_ttm'] / grp['eps_ttm_prev'] - 1) * 100, np.nan)
    grp['fundamental_ok'] = grp['beat_4q'] & (grp['eps_growth'] >= 8.0)
    return grp[['symbol','date','fundamental_ok']]

earn_status = earn.groupby('symbol', group_keys=False).apply(compute_earn_status)
earn_lookup = {}
for sym, grp in earn_status.groupby('symbol'):
    grp = grp.sort_values('date')
    earn_lookup[sym] = list(zip(grp['date'].values, grp['fundamental_ok'].values))

# Load prices
print("Loading prices...")
price_chunks = []
for i in range(0, len(tickers), 25):
    batch = tickers[i:i+25]
    q = """SELECT symbol, date, open, high, low, close
           FROM fmp_price_history WHERE symbol = ANY(%(t)s) AND date >= '2019-01-01'
           ORDER BY symbol, date"""
    price_chunks.append(pd.read_sql(q, engine, params={'t': batch}))
    if (i // 25) % 5 == 0:
        print(f"  Batch {i//25+1}/{(len(tickers)+24)//25}...")

df = pd.concat(price_chunks, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
print(f"Prices: {len(df):,} rows")

# Compute signals with stop loss
print("Computing signals with stop loss...")
SLIP = 0.3
STOP_LEVELS = [-3, -5, -7]
signals = []

for sym, g in df.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    if len(g) < 210:
        continue

    close = g['close'].values
    high = g['high'].values
    low = g['low'].values
    opn = g['open'].values
    dates = g['date'].values
    n = len(g)

    sma200 = pd.Series(close).rolling(200).mean().values
    sma5 = pd.Series(close).rolling(5).mean().values

    el = earn_lookup.get(sym, [])

    for i in range(203, n - 6):
        if pd.Timestamp(dates[i]).year < 2020:
            continue
        if close[i] <= sma200[i] or close[i] >= sma5[i]:
            continue
        if not (high[i-2] < high[i-3] and low[i-2] < low[i-3]):
            continue
        if not (high[i-1] < high[i-2] and low[i-1] < low[i-2]):
            continue
        if not (high[i] < high[i-1] and low[i] < low[i-1]):
            continue

        sig_date = dates[i]
        fok = False
        for ed, ok in el:
            if ed < sig_date:
                fok = ok
            else:
                break
        if not fok:
            continue

        buy_price = opn[i+1]
        if buy_price <= 0:
            continue

        row = {
            'symbol': sym,
            'signal_date': sig_date,
            'year': pd.Timestamp(sig_date).year,
        }

        # No stop loss: sell at open day+4
        sell_idx = i + 5
        if sell_idx < n:
            row['ret_no_sl'] = round((opn[sell_idx] / buy_price - 1) * 100 - SLIP, 2)
        else:
            row['ret_no_sl'] = None

        # With stop loss
        for sl in STOP_LEVELS:
            stop_price = buy_price * (1 + sl / 100)
            stopped = False
            for day in range(1, 5):
                di = i + 1 + day
                if di >= n:
                    break
                if low[di] <= stop_price:
                    row[f'ret_sl{sl}'] = round(sl - SLIP, 2)
                    stopped = True
                    break
            if not stopped:
                if sell_idx < n:
                    row[f'ret_sl{sl}'] = round((opn[sell_idx] / buy_price - 1) * 100 - SLIP, 2)
                else:
                    row[f'ret_sl{sl}'] = None

        signals.append(row)

sdf = pd.DataFrame(signals)
print(f"\nTotal signals 2020-2026: {len(sdf):,}")

# Results
cols = ['ret_no_sl', 'ret_sl-3', 'ret_sl-5', 'ret_sl-7']
labels = ['Sin SL', 'SL -3%', 'SL -5%', 'SL -7%']

print(f"\n{'='*70}")
print(f"=== 2020-2026 CON FILTRO - COMPARATIVA STOP LOSS ===")
print(f"{'='*70}")

print(f"\n{'Metrica':>15s} | {labels[0]:>12s} | {labels[1]:>12s} | {labels[2]:>12s} | {labels[3]:>12s}")
print('-'*70)
for name, fn in [
    ('Trades', lambda c: f"{sdf[c].dropna().shape[0]:,}"),
    ('Win rate', lambda c: f"{(sdf[c].dropna()>0).mean()*100:.1f}%"),
    ('Avg neto', lambda c: f"{sdf[c].dropna().mean():+.2f}%"),
    ('Median', lambda c: f"{sdf[c].dropna().median():+.2f}%"),
    ('Sum neta', lambda c: f"{sdf[c].dropna().sum():+,.1f}%"),
    ('PF', lambda c: f"{sdf[c].dropna()[sdf[c].dropna()>0].sum() / abs(sdf[c].dropna()[sdf[c].dropna()<0].sum()):.2f}" if abs(sdf[c].dropna()[sdf[c].dropna()<0].sum()) > 0 else '-'),
    ('Max loss', lambda c: f"{sdf[c].dropna().min():+.2f}%"),
]:
    vals = [fn(c) for c in cols]
    print(f"{name:>15s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s}")

# Stops hit count
for sl in STOP_LEVELS:
    col = f'ret_sl{sl}'
    stop_ret = round(sl - SLIP, 2)
    stopped = (sdf[col].dropna() == stop_ret).sum()
    total = sdf[col].dropna().shape[0]
    print(f"  SL {sl}%: {stopped} stops hit ({stopped/total*100:.1f}% de trades)")

# By year
print(f"\n=== POR ANO ===")
print(f"{'Ano':>5s} | {'--- Sin SL ---':^18s} | {'--- SL -3% ---':^18s} | {'--- SL -5% ---':^18s} | {'--- SL -7% ---':^18s}")
print(f"{'':>5s} | {'N':>4s} {'WR':>5s} {'Avg':>6s} | {'N':>4s} {'WR':>5s} {'Avg':>6s} | {'N':>4s} {'WR':>5s} {'Avg':>6s} | {'N':>4s} {'WR':>5s} {'Avg':>6s}")
print('-'*85)
for year in sorted(sdf['year'].unique()):
    yg = sdf[sdf['year']==year]
    parts = []
    for c in cols:
        v = yg[c].dropna()
        nn = len(v)
        if nn > 0:
            parts.append(f"{nn:4d} {(v>0).mean()*100:4.1f}% {v.mean():+5.2f}%")
        else:
            parts.append(f"   0    -     -")
    print(f"{year:5d} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]}")

parts = []
for c in cols:
    v = sdf[c].dropna()
    parts.append(f"{len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+5.2f}%")
print('-'*85)
print(f"{'TOTAL':>5s} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]}")

# Sum by year
print(f"\n=== SUM NETA POR ANO ===")
print(f"{'Ano':>5s} | {'Sin SL':>10s} | {'SL -3%':>10s} | {'SL -5%':>10s} | {'SL -7%':>10s}")
print('-'*50)
for year in sorted(sdf['year'].unique()):
    yg = sdf[sdf['year']==year]
    vals = []
    for c in cols:
        v = yg[c].dropna()
        vals.append(f"{v.sum():+9.1f}%")
    print(f"{year:5d} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s} | {vals[3]:>10s}")
vals = [f"{sdf[c].dropna().sum():+9.1f}%" for c in cols]
print('-'*50)
print(f"{'TOTAL':>5s} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s} | {vals[3]:>10s}")
