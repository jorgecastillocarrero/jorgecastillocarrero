import json, pandas as pd, numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))
print(f'Tickers: {len(tickers)}')

SLIP = 0.3
HOLD_DAYS = [5, 10, 15, 22, 68]

# 1. Earnings - compute surprise, PE proxy, EPS growth ourselves
print('Loading earnings...')
ec = []
for i in range(0, len(tickers), 50):
    b = tickers[i:i+50]
    ec.append(pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated
       FROM fmp_earnings WHERE symbol = ANY(%(t)s) AND eps_actual IS NOT NULL
       ORDER BY symbol, date""", engine, params={'t': b}))
earn = pd.concat(ec, ignore_index=True)
earn['date'] = pd.to_datetime(earn['date'])
earn['eps_actual'] = earn['eps_actual'].astype(float)
earn['eps_estimated'] = earn['eps_estimated'].astype(float)
print(f'  Earnings: {len(earn):,} rows, {earn["symbol"].nunique()} symbols')

# Compute per symbol: surprise, EPS TTM, EPS growth YoY
def compute_earn_features(grp):
    grp = grp.sort_values('date').reset_index(drop=True)
    grp['surprise_pos'] = grp['eps_actual'] > grp['eps_estimated']
    grp['eps_ttm'] = grp['eps_actual'].rolling(4, min_periods=4).sum()
    grp['eps_ttm_prev'] = grp['eps_ttm'].shift(4)
    grp['eps_growth_yoy'] = np.where(
        grp['eps_ttm_prev'].abs() > 0.01,
        (grp['eps_ttm'] / grp['eps_ttm_prev'] - 1) * 100,
        np.nan
    )
    return grp[['symbol', 'date', 'surprise_pos', 'eps_ttm', 'eps_growth_yoy']]

print('  Computing EPS TTM and growth...')
earn_feat = earn.groupby('symbol', group_keys=False).apply(compute_earn_features)

# Build lookup: symbol -> list of (date, surprise_pos, eps_ttm, eps_growth_yoy)
earn_lookup = {}
for sym, grp in earn_feat.groupby('symbol'):
    grp = grp.sort_values('date')
    earn_lookup[sym] = list(zip(
        grp['date'].values,
        grp['surprise_pos'].values,
        grp['eps_ttm'].values,
        grp['eps_growth_yoy'].values
    ))

# 2. Prices desde 2005
print('Loading prices (desde 2005)...')
prc = []
for i in range(0, len(tickers), 25):
    b = tickers[i:i+25]
    prc.append(pd.read_sql("""SELECT symbol, date, open, close
       FROM fmp_price_history WHERE symbol = ANY(%(t)s) AND date >= '2005-01-01'
       ORDER BY symbol, date""", engine, params={'t': b}))
    if (i // 25) % 5 == 0:
        print(f'  Batch {i//25+1}/{(len(tickers)+24)//25}...')
df = pd.concat(prc, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
print(f'  Prices: {len(df):,} rows, {df["symbol"].nunique()} symbols')

def get_earn_data(ll, sig_date):
    """Get most recent earnings data before sig_date.
    Returns (surprise_pos, eps_ttm, eps_growth_yoy) or (None,None,None)"""
    surp, epsttm, epsg = None, None, None
    for d, s, e, g in ll:
        if d < sig_date:
            surp, epsttm, epsg = s, e, g
        else:
            break
    return surp, epsttm, epsg

# 3. Signals
print('Computing signals...')
signals = {hd: [] for hd in HOLD_DAYS}

for sym, g in df.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < 210:
        continue
    close = g['close'].values.astype(float)
    opn = g['open'].values.astype(float)
    dates = g['date'].values
    sma50 = pd.Series(close).rolling(50).mean().values
    sma100 = pd.Series(close).rolling(100).mean().values
    sma200 = pd.Series(close).rolling(200).mean().values
    el = earn_lookup.get(sym, [])
    busy = {hd: -1 for hd in HOLD_DAYS}

    for i in range(200, n - 1):
        if np.isnan(sma50[i]) or np.isnan(sma100[i]) or np.isnan(sma200[i]):
            continue
        if sma50[i] <= 0 or sma100[i] <= 0 or sma200[i] <= 0:
            continue
        d50 = (close[i] / sma50[i] - 1) * 100
        d100 = (close[i] / sma100[i] - 1) * 100
        d200 = (close[i] / sma200[i] - 1) * 100
        # Price < -8% below SMA50
        if d50 >= -8.0:
            continue
        # Price < -4% below SMA100
        if d100 >= -4.0:
            continue
        # Price between -2.5% and +2.5% of SMA200
        if d200 < -2.5 or d200 > 2.5:
            continue

        sd = dates[i]

        # Earnings filters
        surp, epsttm, epsg = get_earn_data(el, sd)
        # EPS surprise positive
        if surp is None or not surp:
            continue
        # P/E > 0: price / EPS TTM > 0 => EPS TTM > 0 (price always > 0)
        if epsttm is None or np.isnan(epsttm) or epsttm <= 0:
            continue
        # PE = close / eps_ttm
        pe = close[i] / epsttm
        if pe <= 0:
            continue
        # EPS growth > 12%
        if epsg is None or np.isnan(epsg) or epsg <= 12.0:
            continue

        bi = i + 1
        if bi >= n:
            continue
        bp = opn[bi]
        if bp <= 0:
            continue
        yr = pd.Timestamp(sd).year

        for hd in HOLD_DAYS:
            if bi <= busy[hd]:
                continue
            si = bi + hd
            if si >= n:
                continue
            sp = opn[si]
            ret = round((sp / bp - 1) * 100 - SLIP, 2)
            busy[hd] = si
            signals[hd].append({'symbol': sym, 'year': yr, 'ret': ret})

# 4. Results
print(f'\n{"="*90}')
print(f'=== ESTRATEGIA 2 - HISTORICO COMPLETO (PE y EPS growth calculados) ===')
print(f'=== Sin solapamiento, 0.3% slippage ===')
print(f'{"="*90}')

print(f'\n{"Metrica":>15s}', end='')
for hd in HOLD_DAYS:
    print(f' |     {hd:>2d}d       ', end='')
print()
print('-' * 95)

for name in ['Trades', 'WR', 'Avg', 'Med', 'Sum', 'PF', 'MaxG', 'MaxL']:
    print(f'{name:>15s}', end='')
    for hd in HOLD_DAYS:
        v = pd.Series([x['ret'] for x in signals[hd]])
        if len(v) == 0:
            print(f' |        -      ', end='')
            continue
        if name == 'Trades':
            print(f' | {len(v):>13,}', end='')
        elif name == 'WR':
            print(f' | {(v>0).mean()*100:>12.1f}%', end='')
        elif name == 'Avg':
            print(f' | {v.mean():>+12.2f}%', end='')
        elif name == 'Med':
            print(f' | {v.median():>+12.2f}%', end='')
        elif name == 'Sum':
            print(f' | {v.sum():>+11.1f}%', end='')
        elif name == 'PF':
            w = v[v > 0].sum()
            l = abs(v[v < 0].sum())
            print(f' | {w/l if l > 0 else 0:>13.2f}', end='')
        elif name == 'MaxG':
            print(f' | {v.max():>+12.2f}%', end='')
        elif name == 'MaxL':
            print(f' | {v.min():>+12.2f}%', end='')
    print()

# By year
all_years = sorted(set(x['year'] for hd in HOLD_DAYS for x in signals[hd]))

print(f'\n=== POR ANO ===')
print(f'{"Ano":>5s}', end='')
for hd in HOLD_DAYS:
    print(f' |  --- {hd:2d}d ---         ', end='')
print()
print(f'{"":>5s}', end='')
for hd in HOLD_DAYS:
    print(f' | {"N":>4s} {"WR":>5s} {"Avg":>7s} {"Sum":>9s}', end='')
print()
print('-' * 145)

for yr in all_years:
    print(f'{yr:5d}', end='')
    for hd in HOLD_DAYS:
        rets = [x['ret'] for x in signals[hd] if x['year'] == yr]
        if len(rets) > 0:
            v = pd.Series(rets)
            print(f' | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+8.1f}%', end='')
        else:
            print(f' |    0    -       -        -', end='')
    print()

print('-' * 145)
print(f'{"TOTAL":>5s}', end='')
for hd in HOLD_DAYS:
    v = pd.Series([x['ret'] for x in signals[hd]])
    print(f' | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+8.1f}%', end='')
print()

# PF by year
print(f'\n=== PROFIT FACTOR POR ANO ===')
print(f'{"Ano":>5s}', end='')
for hd in HOLD_DAYS:
    print(f' | {hd:>3d}d PF', end='')
print()
print('-' * 55)
for yr in all_years:
    print(f'{yr:5d}', end='')
    for hd in HOLD_DAYS:
        rets = [x['ret'] for x in signals[hd] if x['year'] == yr]
        if len(rets) > 0:
            v = pd.Series(rets)
            w = v[v > 0].sum()
            l = abs(v[v < 0].sum())
            pf = w / l if l > 0 else float('inf')
            if pf > 99:
                print(f' |   inf', end='')
            else:
                print(f' | {pf:>5.2f}', end='')
        else:
            print(f' |     -', end='')
    print()
print('-' * 55)
print(f'{"TOTAL":>5s}', end='')
for hd in HOLD_DAYS:
    v = pd.Series([x['ret'] for x in signals[hd]])
    w = v[v > 0].sum()
    l = abs(v[v < 0].sum())
    print(f' | {w/l if l > 0 else 0:>5.2f}', end='')
print()

# Years with negative sum
print(f'\n=== ANOS CON SUM NEGATIVA ===')
for hd in HOLD_DAYS:
    neg_years = []
    for yr in all_years:
        rets = [x['ret'] for x in signals[hd] if x['year'] == yr]
        if len(rets) > 0:
            s = sum(rets)
            if s < 0:
                neg_years.append(f'{yr}({s:+.0f}%)')
    print(f'  {hd:2d}d: {", ".join(neg_years) if neg_years else "ninguno"}')

# Decades summary
print(f'\n=== POR DECADA ===')
decades = {}
for yr in all_years:
    dec = (yr // 10) * 10
    if dec not in decades:
        decades[dec] = {}
    decades[dec][yr] = True

for dec in sorted(decades.keys()):
    print(f'\n  {dec}s:', end='')
    for hd in HOLD_DAYS:
        rets = [x['ret'] for x in signals[hd] if (x['year'] // 10) * 10 == dec]
        if len(rets) > 0:
            v = pd.Series(rets)
            w = v[v > 0].sum()
            l = abs(v[v < 0].sum())
            pf = w / l if l > 0 else 0
            print(f'  {hd}d: {len(v)}t WR{(v>0).mean()*100:.0f}% Avg{v.mean():+.2f}% PF{pf:.2f}', end=' |')
        else:
            print(f'  {hd}d: 0t', end=' |')
    print()
