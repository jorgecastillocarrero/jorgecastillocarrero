import json, pandas as pd, numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

SLIP = 0.3
HOLD_DAYS = [5, 10, 15, 22, 68]

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

def compute_ef(grp):
    grp = grp.sort_values('date').reset_index(drop=True)
    grp['surprise_neg'] = grp['eps_actual'] < grp['eps_estimated']
    grp['eps_ttm'] = grp['eps_actual'].rolling(4, min_periods=4).sum()
    grp['eps_ttm_prev'] = grp['eps_ttm'].shift(4)
    grp['eps_growth_yoy'] = np.where(
        grp['eps_ttm_prev'].abs() > 0.01,
        (grp['eps_ttm'] / grp['eps_ttm_prev'] - 1) * 100, np.nan)
    return grp[['symbol', 'date', 'surprise_neg', 'eps_ttm', 'eps_growth_yoy']]

ef = earn.groupby('symbol', group_keys=False).apply(compute_ef)
earn_lk = {}
for sym, grp in ef.groupby('symbol'):
    grp = grp.sort_values('date')
    earn_lk[sym] = list(zip(grp['date'].values, grp['surprise_neg'].values,
                            grp['eps_ttm'].values, grp['eps_growth_yoy'].values))

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

# Precompute
sym_data = {}
for sym, g in df.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < 210:
        continue
    c = g['close'].values.astype(float)
    o = g['open'].values.astype(float)
    d = g['date'].values
    s50 = pd.Series(c).rolling(50).mean().values
    s100 = pd.Series(c).rolling(100).mean().values
    s200 = pd.Series(c).rolling(200).mean().values
    sym_data[sym] = (c, o, d, s50, s100, s200, n)

def get_e(ll, sig):
    sn, et, eg = None, None, None
    for dd, s, e, g in ll:
        if dd < sig:
            sn, et, eg = s, e, g
        else:
            break
    return sn, et, eg

configs = [
    ('SOLO_PRECIO (d50>8 d100>4 d200+-2.5)', 8, 4, 2.5, False, None),
    ('RELAX1 (d50>8 d100>4 d200+-5 surp- epsg<-12)', 8, 4, 5.0, True, -12),
    ('RELAX5 (d50>8 d100>4 d200+-5 epsg<0 sin surp)', 8, 4, 5.0, False, 0),
]

for cfg_name, d50m, d100m, d200r, need_surp, epsg_th in configs:
    signals = {hd: [] for hd in HOLD_DAYS}
    for sym, (c, o, dates, s50, s100, s200, n) in sym_data.items():
        ell = earn_lk.get(sym, [])
        busy = {hd: -1 for hd in HOLD_DAYS}
        for i in range(200, n - 1):
            if np.isnan(s50[i]) or np.isnan(s100[i]) or np.isnan(s200[i]):
                continue
            if s50[i] <= 0 or s100[i] <= 0 or s200[i] <= 0:
                continue
            dd50 = (c[i] / s50[i] - 1) * 100
            dd100 = (c[i] / s100[i] - 1) * 100
            dd200 = (c[i] / s200[i] - 1) * 100
            if dd50 < d50m:
                continue
            if dd100 < d100m:
                continue
            if dd200 < -d200r or dd200 > d200r:
                continue
            sig = dates[i]
            if epsg_th is not None or need_surp:
                sn, et, eg = get_e(ell, sig)
                if need_surp and (sn is None or not sn):
                    continue
                if epsg_th is not None and (eg is None or np.isnan(eg) or eg >= epsg_th):
                    continue
            bi = i + 1
            if bi >= n:
                continue
            bp = o[bi]
            if bp <= 0:
                continue
            yr = pd.Timestamp(sig).year
            for hd in HOLD_DAYS:
                if bi <= busy[hd]:
                    continue
                si = bi + hd
                if si >= n:
                    continue
                cv = o[si]
                ret = round((bp / cv - 1) * 100 - SLIP, 2)
                busy[hd] = si
                signals[hd].append({'year': yr, 'ret': ret})

    print(f'\n{"="*165}')
    print(f'  SHORT {cfg_name}')
    print(f'{"="*165}')
    print(f'{"Ano":>5s}', end='')
    for hd in HOLD_DAYS:
        print(f' |  ------ {hd:2d}d ------', end='')
    print()
    print(f'{"":>5s}', end='')
    for hd in HOLD_DAYS:
        print(f' | {"N":>4s} {"WR":>5s} {"Avg":>7s} {"Sum":>8s} {"PF":>5s}', end='')
    print()
    print('-' * 165)

    all_years = sorted(set(x['year'] for hd in HOLD_DAYS for x in signals[hd]))
    for yr in all_years:
        print(f'{yr:5d}', end='')
        for hd in HOLD_DAYS:
            r = [x['ret'] for x in signals[hd] if x['year'] == yr]
            if len(r) > 0:
                v = pd.Series(r)
                w = v[v > 0].sum()
                l = abs(v[v < 0].sum())
                pf = w / l if l > 0 else float('inf')
                pfs = f'{pf:5.2f}' if pf < 99 else '  inf'
                print(f' | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+7.0f}% {pfs}', end='')
            else:
                print(f' |    0    -       -       -     -', end='')
        print()
    print('-' * 165)
    print(f'{"TOTAL":>5s}', end='')
    for hd in HOLD_DAYS:
        v = pd.Series([x['ret'] for x in signals[hd]])
        w = v[v > 0].sum()
        l = abs(v[v < 0].sum())
        pf = w / l if l > 0 else 0
        print(f' | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+7.0f}% {pf:5.2f}', end='')
    print()
