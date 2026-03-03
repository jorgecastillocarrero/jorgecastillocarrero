import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json','r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))
print(f"S&P 500 tickers: {len(tickers)}")

SLIP = 0.3
HOLD_DAYS = [5, 10, 15, 22, 68]

# ── 1. Load earnings (EPS surprise) ──
print("Loading earnings...")
earn_chunks = []
for i in range(0, len(tickers), 50):
    batch = tickers[i:i+50]
    q = """SELECT symbol, date, eps_actual, eps_estimated
           FROM fmp_earnings WHERE symbol = ANY(%(t)s) AND eps_actual IS NOT NULL
           AND date >= '2017-01-01' ORDER BY symbol, date"""
    earn_chunks.append(pd.read_sql(q, engine, params={'t': batch}))
earn = pd.concat(earn_chunks, ignore_index=True)
earn['date'] = pd.to_datetime(earn['date'])
earn['eps_actual'] = earn['eps_actual'].astype(float)
earn['eps_estimated'] = earn['eps_estimated'].astype(float)
earn['surprise_pos'] = earn['eps_actual'] > earn['eps_estimated']
print(f"  Earnings: {len(earn):,} rows, {earn['symbol'].nunique()} symbols")

# Build lookup: symbol -> list of (date, surprise_pos)
earn_lookup = {}
for sym, grp in earn.groupby('symbol'):
    grp = grp.sort_values('date')
    earn_lookup[sym] = list(zip(grp['date'].values, grp['surprise_pos'].values))

# ── 2. Load P/E and EPS growth from peg_weekly ──
print("Loading P/E and EPS growth from peg_weekly...")
peg_chunks = []
for i in range(0, len(tickers), 50):
    batch = tickers[i:i+50]
    q = """SELECT symbol, week_ending, pe_ratio, eps_growth_yoy
           FROM peg_weekly WHERE symbol = ANY(%(t)s)
           AND week_ending >= '2019-01-01'
           ORDER BY symbol, week_ending"""
    peg_chunks.append(pd.read_sql(q, engine, params={'t': batch}))
peg = pd.concat(peg_chunks, ignore_index=True)
peg['week_ending'] = pd.to_datetime(peg['week_ending'])
peg['pe_ratio'] = pd.to_numeric(peg['pe_ratio'], errors='coerce')
peg['eps_growth_yoy'] = pd.to_numeric(peg['eps_growth_yoy'], errors='coerce')
print(f"  PEG weekly: {len(peg):,} rows, {peg['symbol'].nunique()} symbols")

# Build lookup: symbol -> sorted list of (week_ending, pe_ratio, eps_growth_yoy)
peg_lookup = {}
for sym, grp in peg.groupby('symbol'):
    grp = grp.sort_values('week_ending')
    peg_lookup[sym] = list(zip(
        grp['week_ending'].values,
        grp['pe_ratio'].values,
        grp['eps_growth_yoy'].values
    ))

# ── 3. Load prices ──
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
print(f"  Prices: {len(df):,} rows, {df['symbol'].nunique()} symbols")

# ── 4. Helper: get latest value from sorted lookup before a date ──
def get_latest_earn(lookup_list, sig_date):
    """Get most recent earnings surprise before sig_date"""
    result = None
    for d, val in lookup_list:
        if d < sig_date:
            result = val
        else:
            break
    return result

def get_latest_peg(lookup_list, sig_date):
    """Get most recent P/E and EPS growth on or before sig_date"""
    pe = None
    epsg = None
    for d, p, e in lookup_list:
        if d <= sig_date:
            pe = p
            epsg = e
        else:
            break
    return pe, epsg

# ── 5. Compute signals ──
print("Computing signals...")
max_hold = max(HOLD_DAYS)
signals = []
filter_stats = {'total_checked': 0, 'pass_sma': 0, 'pass_earn': 0,
                'pass_pe': 0, 'pass_epsg': 0, 'pass_all': 0}

for sym, g in df.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < 210:
        continue

    close = g['close'].values.astype(float)
    opn = g['open'].values.astype(float)
    dates = g['date'].values

    # Compute SMAs
    sma50 = pd.Series(close).rolling(50).mean().values
    sma100 = pd.Series(close).rolling(100).mean().values
    sma200 = pd.Series(close).rolling(200).mean().values

    el = earn_lookup.get(sym, [])
    pl = peg_lookup.get(sym, [])

    # Track open position end index per holding period (no overlap)
    busy_until = {hd: -1 for hd in HOLD_DAYS}

    for i in range(200, n - 1):
        if pd.Timestamp(dates[i]).year < 2020:
            continue
        if np.isnan(sma50[i]) or np.isnan(sma100[i]) or np.isnan(sma200[i]):
            continue
        if sma50[i] <= 0 or sma100[i] <= 0 or sma200[i] <= 0:
            continue

        filter_stats['total_checked'] += 1

        # Price filters
        dist_sma50 = (close[i] / sma50[i] - 1) * 100
        dist_sma100 = (close[i] / sma100[i] - 1) * 100
        dist_sma200 = (close[i] / sma200[i] - 1) * 100

        # Price < -8% below SMA50
        if dist_sma50 >= -8.0:
            continue
        # Price < -4% below SMA100
        if dist_sma100 >= -4.0:
            continue
        # Price between -2.5% and +2.5% of SMA200
        if dist_sma200 < -2.5 or dist_sma200 > 2.5:
            continue

        filter_stats['pass_sma'] += 1

        sig_date = dates[i]

        # EPS surprise positive (last quarter)
        surprise = get_latest_earn(el, sig_date)
        if surprise is None or not surprise:
            continue
        filter_stats['pass_earn'] += 1

        # P/E > 0 and EPS growth > 12%
        pe, epsg = get_latest_peg(pl, sig_date)
        if pe is None or np.isnan(pe) or pe <= 0:
            continue
        filter_stats['pass_pe'] += 1

        if epsg is None or np.isnan(epsg) or epsg <= 12.0:
            continue
        filter_stats['pass_epsg'] += 1

        filter_stats['pass_all'] += 1

        # Buy at next day's open
        buy_idx = i + 1
        if buy_idx >= n:
            continue
        buy_price = opn[buy_idx]
        if buy_price <= 0:
            continue

        row = {
            'symbol': sym,
            'signal_date': sig_date,
            'buy_date': dates[buy_idx],
            'buy_price': round(buy_price, 2),
            'close_at_signal': round(close[i], 2),
            'sma50': round(sma50[i], 2),
            'sma100': round(sma100[i], 2),
            'sma200': round(sma200[i], 2),
            'dist_sma50': round(dist_sma50, 1),
            'dist_sma100': round(dist_sma100, 1),
            'dist_sma200': round(dist_sma200, 1),
            'pe_ratio': round(pe, 1),
            'eps_growth': round(epsg, 1),
            'year': pd.Timestamp(sig_date).year,
        }

        # Holding periods - only if not already in a position for that period
        any_trade = False
        for hd in HOLD_DAYS:
            if buy_idx <= busy_until[hd]:
                # Already in a position, skip this holding period
                row[f'ret_{hd}d'] = None
                row[f'skip_{hd}d'] = True
            else:
                sell_idx = buy_idx + hd
                if sell_idx < n:
                    sell_price = opn[sell_idx]
                    ret = round((sell_price / buy_price - 1) * 100 - SLIP, 2)
                    row[f'ret_{hd}d'] = ret
                    row[f'skip_{hd}d'] = False
                    busy_until[hd] = sell_idx  # Block until sell day
                    any_trade = True
                else:
                    row[f'ret_{hd}d'] = None
                    row[f'skip_{hd}d'] = False

        if not any_trade and all(row.get(f'skip_{hd}d', False) for hd in HOLD_DAYS):
            continue  # All periods blocked, skip entirely

        signals.append(row)

sdf = pd.DataFrame(signals)
print(f"\n=== FILTER FUNNEL ===")
print(f"  Days checked:    {filter_stats['total_checked']:>10,}")
print(f"  Pass SMA filter: {filter_stats['pass_sma']:>10,}")
print(f"  Pass earnings:   {filter_stats['pass_earn']:>10,}")
print(f"  Pass P/E > 0:    {filter_stats['pass_pe']:>10,}")
print(f"  Pass EPS g >12%: {filter_stats['pass_epsg']:>10,}")
print(f"  Final signals:   {filter_stats['pass_all']:>10,}")

if len(sdf) == 0:
    print("\nNo signals found!")
    exit()

print(f"\nTotal signal rows: {len(sdf):,}")
print(f"Unique symbols: {sdf['symbol'].nunique()}")

# ── 6. Results ──
ret_cols = [f'ret_{hd}d' for hd in HOLD_DAYS]
skip_cols = [f'skip_{hd}d' for hd in HOLD_DAYS]
labels = [f'{hd}d' for hd in HOLD_DAYS]

print(f"\n{'='*80}")
print(f"=== ESTRATEGIA 2 - SIN SOLAPAMIENTO (con 0.3% slippage) ===")
print(f"{'='*80}")

# Show skipped count
print(f"\n{'':>15s}", end='')
for hd in HOLD_DAYS:
    sc = f'skip_{hd}d'
    skipped = sdf[sc].sum() if sc in sdf.columns else 0
    total = filter_stats['pass_all']
    print(f" | {hd}d: {int(total-skipped)} trades ({int(skipped)} skip)", end='')
print()

print(f"\n{'Metrica':>15s}", end='')
for lb in labels:
    print(f" | {lb:>12s}", end='')
print()
print('-' * (16 + 15 * len(labels)))

for name, fn in [
    ('Trades', lambda c: f"{sdf[c].dropna().shape[0]:,}"),
    ('Win rate', lambda c: f"{(sdf[c].dropna()>0).mean()*100:.1f}%"),
    ('Avg neto', lambda c: f"{sdf[c].dropna().mean():+.2f}%"),
    ('Median', lambda c: f"{sdf[c].dropna().median():+.2f}%"),
    ('Sum neta', lambda c: f"{sdf[c].dropna().sum():+,.1f}%"),
    ('PF', lambda c: f"{sdf[c].dropna()[sdf[c].dropna()>0].sum() / abs(sdf[c].dropna()[sdf[c].dropna()<0].sum()):.2f}" if abs(sdf[c].dropna()[sdf[c].dropna()<0].sum()) > 0 else '-'),
    ('Max gain', lambda c: f"{sdf[c].dropna().max():+.2f}%"),
    ('Max loss', lambda c: f"{sdf[c].dropna().min():+.2f}%"),
    ('Std', lambda c: f"{sdf[c].dropna().std():.2f}%"),
]:
    print(f"{name:>15s}", end='')
    for c in ret_cols:
        print(f" | {fn(c):>12s}", end='')
    print()

# ── 7. By year ──
print(f"\n=== POR AÑO ===")
for hd, col in zip(HOLD_DAYS, ret_cols):
    print(f"\n--- Holding {hd} dias ---")
    print(f"{'Año':>5s} | {'N':>5s} | {'WR':>6s} | {'Avg':>8s} | {'Med':>8s} | {'Sum':>10s} | {'PF':>6s}")
    print('-' * 55)
    for year in sorted(sdf['year'].unique()):
        yg = sdf[sdf['year'] == year]
        v = yg[col].dropna()
        nn = len(v)
        if nn > 0:
            wr = (v > 0).mean() * 100
            avg = v.mean()
            med = v.median()
            sm = v.sum()
            wins = v[v > 0].sum()
            losses = abs(v[v < 0].sum())
            pf = wins / losses if losses > 0 else float('inf')
            print(f"{year:5d} | {nn:5d} | {wr:5.1f}% | {avg:+7.2f}% | {med:+7.2f}% | {sm:+9.1f}% | {pf:5.2f}")
        else:
            print(f"{year:5d} |     0 |     - |       - |       - |        - |     -")
    # Total
    v = sdf[col].dropna()
    wr = (v > 0).mean() * 100
    avg = v.mean()
    med = v.median()
    sm = v.sum()
    wins = v[v > 0].sum()
    losses = abs(v[v < 0].sum())
    pf = wins / losses if losses > 0 else float('inf')
    print('-' * 55)
    print(f"TOTAL | {len(v):5d} | {wr:5.1f}% | {avg:+7.2f}% | {med:+7.2f}% | {sm:+9.1f}% | {pf:5.2f}")

# ── 8. Top signals by return (68d) ──
best_col = 'ret_68d'
if best_col in sdf.columns and sdf[best_col].dropna().shape[0] > 0:
    print(f"\n=== TOP 20 MEJORES TRADES (68d) ===")
    top = sdf.dropna(subset=[best_col]).nlargest(20, best_col)
    print(f"{'Symbol':>8s} | {'Signal Date':>12s} | {'Buy$':>8s} | {'Dist50':>7s} | {'Dist200':>7s} | {'P/E':>6s} | {'EPSg':>6s} | {'Ret68d':>8s}")
    print('-' * 75)
    for _, r in top.iterrows():
        print(f"{r['symbol']:>8s} | {str(r['signal_date'])[:10]:>12s} | {r['buy_price']:>8.2f} | {r['dist_sma50']:>6.1f}% | {r['dist_sma200']:>6.1f}% | {r['pe_ratio']:>6.1f} | {r['eps_growth']:>5.1f}% | {r[best_col]:>+7.2f}%")

    print(f"\n=== TOP 20 PEORES TRADES (68d) ===")
    bot = sdf.dropna(subset=[best_col]).nsmallest(20, best_col)
    print(f"{'Symbol':>8s} | {'Signal Date':>12s} | {'Buy$':>8s} | {'Dist50':>7s} | {'Dist200':>7s} | {'P/E':>6s} | {'EPSg':>6s} | {'Ret68d':>8s}")
    print('-' * 75)
    for _, r in bot.iterrows():
        print(f"{r['symbol']:>8s} | {str(r['signal_date'])[:10]:>12s} | {r['buy_price']:>8.2f} | {r['dist_sma50']:>6.1f}% | {r['dist_sma200']:>6.1f}% | {r['pe_ratio']:>6.1f} | {r['eps_growth']:>5.1f}% | {r[best_col]:>+7.2f}%")
