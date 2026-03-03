import json, pandas as pd, numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

SLIP = 0.3
HOLD_DAYS = [5, 10, 15, 22, 68]

# 1. Earnings
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

def compute_earn_features(grp):
    grp = grp.sort_values('date').reset_index(drop=True)
    grp['surprise_neg'] = grp['eps_actual'] < grp['eps_estimated']
    grp['eps_ttm'] = grp['eps_actual'].rolling(4, min_periods=4).sum()
    grp['eps_ttm_prev'] = grp['eps_ttm'].shift(4)
    grp['eps_growth_yoy'] = np.where(
        grp['eps_ttm_prev'].abs() > 0.01,
        (grp['eps_ttm'] / grp['eps_ttm_prev'] - 1) * 100,
        np.nan
    )
    return grp[['symbol', 'date', 'surprise_neg', 'eps_ttm', 'eps_growth_yoy']]

earn_feat = earn.groupby('symbol', group_keys=False).apply(compute_earn_features)
earn_lookup = {}
for sym, grp in earn_feat.groupby('symbol'):
    grp = grp.sort_values('date')
    earn_lookup[sym] = list(zip(
        grp['date'].values, grp['surprise_neg'].values,
        grp['eps_ttm'].values, grp['eps_growth_yoy'].values
    ))

# 2. Prices
print('Loading prices...')
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
print(f'  Prices: {len(df):,} rows')

def get_earn(ll, sd):
    sn, et, eg = None, None, None
    for d, s, e, g in ll:
        if d < sd:
            sn, et, eg = s, e, g
        else:
            break
    return sn, et, eg

# 3. Test multiple filter combos
configs = [
    # name, d50_min, d100_min, d200_range, need_surprise_neg, eps_ttm_rule, epsg_rule
    ('ORIGINAL: d50>8 d100>4 d200+-2.5 surp- epsttm<0 epsg<-12',
     8, 4, 2.5, True, 'neg', -12),
    ('RELAX1: d50>8 d100>4 d200+-5 surp- epsg<-12',
     8, 4, 5.0, True, 'any', -12),
    ('RELAX2: d50>8 d100>4 d200+-5 epsg<-12 (sin surprise)',
     8, 4, 5.0, False, 'any', -12),
    ('RELAX3: d50>5 d100>3 d200+-5 surp- epsg<-8',
     5, 3, 5.0, True, 'any', -8),
    ('RELAX4: d50>5 d100>3 d200+-5 epsg<-8 (sin surprise)',
     5, 3, 5.0, False, 'any', -8),
    ('RELAX5: d50>8 d100>4 d200+-5 epsg<0 (sin surprise)',
     8, 4, 5.0, False, 'any', 0),
    ('RELAX6: d50>5 d100>3 d200+-5 epsg<0 (sin surprise)',
     5, 3, 5.0, False, 'any', 0),
    ('SOLO_PRECIO: d50>8 d100>4 d200+-2.5 (sin filtros fund.)',
     8, 4, 2.5, False, 'any', None),
    ('SOLO_PRECIO2: d50>5 d100>3 d200+-5 (sin filtros fund.)',
     5, 3, 5.0, False, 'any', None),
]

# Precompute per-symbol arrays
print('Precomputing...')
sym_data = {}
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
    sym_data[sym] = (close, opn, dates, sma50, sma100, sma200, n)

print(f'Symbols with data: {len(sym_data)}')

for cfg_name, d50_min, d100_min, d200_range, need_surp, eps_rule, epsg_thresh in configs:
    signals = {hd: [] for hd in HOLD_DAYS}

    for sym, (close, opn, dates, sma50, sma100, sma200, n) in sym_data.items():
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

            if d50 < d50_min:
                continue
            if d100 < d100_min:
                continue
            if d200 < -d200_range or d200 > d200_range:
                continue

            sd = dates[i]
            sn, et, eg = get_earn(el, sd)

            if need_surp:
                if sn is None or not sn:
                    continue

            if eps_rule == 'neg':
                if et is None or np.isnan(et) or et >= 0:
                    continue

            if epsg_thresh is not None:
                if eg is None or np.isnan(eg) or eg >= epsg_thresh:
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
                cover = opn[si]
                ret = round((bp / cover - 1) * 100 - SLIP, 2)
                busy[hd] = si
                signals[hd].append({'symbol': sym, 'year': yr, 'ret': ret})

    # Print summary
    print(f'\n{"="*100}')
    print(f'  {cfg_name}')
    print(f'{"="*100}')

    has_any = any(len(signals[hd]) > 0 for hd in HOLD_DAYS)
    if not has_any:
        print('  ** 0 trades en todos los periodos **')
        continue

    print(f'{"":>8s}', end='')
    for hd in HOLD_DAYS:
        print(f' | {"":>3s}{hd:>2d}d{"":>3s}', end='')
    print()

    for name in ['N', 'WR', 'Avg', 'Sum', 'PF']:
        print(f'  {name:>5s}', end='')
        for hd in HOLD_DAYS:
            rets = [x['ret'] for x in signals[hd]]
            if len(rets) == 0:
                print(f' | {"--":>8s}', end='')
                continue
            v = pd.Series(rets)
            if name == 'N':
                print(f' | {len(v):>8,}', end='')
            elif name == 'WR':
                print(f' | {(v>0).mean()*100:>7.1f}%', end='')
            elif name == 'Avg':
                print(f' | {v.mean():>+7.2f}%', end='')
            elif name == 'Sum':
                print(f' | {v.sum():>+7.0f}%', end='')
            elif name == 'PF':
                w = v[v > 0].sum()
                l = abs(v[v < 0].sum())
                print(f' | {w/l if l > 0 else 0:>8.2f}', end='')
        print()

    # By year for best period (22d)
    hd_best = 22
    rets_best = [x['ret'] for x in signals[hd_best]]
    if len(rets_best) > 0:
        all_years = sorted(set(x['year'] for x in signals[hd_best]))
        print(f'\n  Por ano ({hd_best}d):')
        print(f'  {"Ano":>5s} | {"N":>4s} {"WR":>5s} {"Avg":>7s} {"Sum":>9s} {"PF":>6s}')
        print(f'  {"-"*45}')
        for yr in all_years:
            r = [x['ret'] for x in signals[hd_best] if x['year'] == yr]
            if len(r) > 0:
                v = pd.Series(r)
                w = v[v > 0].sum()
                l = abs(v[v < 0].sum())
                pf = w / l if l > 0 else float('inf')
                pfstr = f'{pf:5.2f}' if pf < 99 else '  inf'
                print(f'  {yr:5d} | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+8.1f}% {pfstr}')
        v = pd.Series(rets_best)
        w = v[v > 0].sum()
        l = abs(v[v < 0].sum())
        pf = w / l if l > 0 else 0
        print(f'  {"-"*45}')
        print(f'  {"TOTAL":>5s} | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+8.1f}% {pf:5.2f}')
