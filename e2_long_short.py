import json, pandas as pd, numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))
print(f'Tickers: {len(tickers)}')

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
print(f'  Earnings: {len(earn):,} rows')

def compute_earn_features(grp):
    grp = grp.sort_values('date').reset_index(drop=True)
    grp['surprise_pos'] = grp['eps_actual'] > grp['eps_estimated']
    grp['surprise_neg'] = grp['eps_actual'] < grp['eps_estimated']
    grp['eps_ttm'] = grp['eps_actual'].rolling(4, min_periods=4).sum()
    grp['eps_ttm_prev'] = grp['eps_ttm'].shift(4)
    grp['eps_growth_yoy'] = np.where(
        grp['eps_ttm_prev'].abs() > 0.01,
        (grp['eps_ttm'] / grp['eps_ttm_prev'] - 1) * 100,
        np.nan
    )
    return grp[['symbol', 'date', 'surprise_pos', 'surprise_neg', 'eps_ttm', 'eps_growth_yoy']]

print('  Computing EPS TTM and growth...')
earn_feat = earn.groupby('symbol', group_keys=False).apply(compute_earn_features)

earn_lookup = {}
for sym, grp in earn_feat.groupby('symbol'):
    grp = grp.sort_values('date')
    earn_lookup[sym] = list(zip(
        grp['date'].values,
        grp['surprise_pos'].values,
        grp['surprise_neg'].values,
        grp['eps_ttm'].values,
        grp['eps_growth_yoy'].values
    ))

# 2. Prices
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
    surp_pos, surp_neg, epsttm, epsg = None, None, None, None
    for d, sp, sn, e, g in ll:
        if d < sig_date:
            surp_pos, surp_neg, epsttm, epsg = sp, sn, e, g
        else:
            break
    return surp_pos, surp_neg, epsttm, epsg

# 3. Signals LONG y SHORT
print('Computing signals LONG y SHORT...')
sig_long = {hd: [] for hd in HOLD_DAYS}
sig_short = {hd: [] for hd in HOLD_DAYS}

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
    busy_l = {hd: -1 for hd in HOLD_DAYS}
    busy_s = {hd: -1 for hd in HOLD_DAYS}

    for i in range(200, n - 1):
        if np.isnan(sma50[i]) or np.isnan(sma100[i]) or np.isnan(sma200[i]):
            continue
        if sma50[i] <= 0 or sma100[i] <= 0 or sma200[i] <= 0:
            continue

        d50 = (close[i] / sma50[i] - 1) * 100
        d100 = (close[i] / sma100[i] - 1) * 100
        d200 = (close[i] / sma200[i] - 1) * 100

        sd = dates[i]
        surp_pos, surp_neg, epsttm, epsg = get_earn_data(el, sd)

        bi = i + 1
        if bi >= n:
            continue
        bp = opn[bi]
        if bp <= 0:
            continue
        yr = pd.Timestamp(sd).year

        # === LONG: precio hundido, fundamentales fuertes ===
        is_long = (
            d50 < -8.0 and
            d100 < -4.0 and
            -2.5 <= d200 <= 2.5 and
            surp_pos is not None and surp_pos and
            epsttm is not None and not np.isnan(epsttm) and epsttm > 0 and
            epsg is not None and not np.isnan(epsg) and epsg > 12.0
        )

        if is_long:
            for hd in HOLD_DAYS:
                if bi <= busy_l[hd]:
                    continue
                si = bi + hd
                if si >= n:
                    continue
                sp = opn[si]
                ret = round((sp / bp - 1) * 100 - SLIP, 2)
                busy_l[hd] = si
                sig_long[hd].append({'symbol': sym, 'year': yr, 'ret': ret})

        # === SHORT: precio inflado, fundamentales debiles ===
        is_short = (
            d50 > 8.0 and
            d100 > 4.0 and
            -2.5 <= d200 <= 2.5 and
            surp_neg is not None and surp_neg and
            epsttm is not None and not np.isnan(epsttm) and epsttm < 0 and
            epsg is not None and not np.isnan(epsg) and epsg < -12.0
        )

        if is_short:
            for hd in HOLD_DAYS:
                if bi <= busy_s[hd]:
                    continue
                si = bi + hd
                if si >= n:
                    continue
                cover = opn[si]
                ret = round((bp / cover - 1) * 100 - SLIP, 2)  # short profit
                busy_s[hd] = si
                sig_short[hd].append({'symbol': sym, 'year': yr, 'ret': ret})


# 4. Results
def print_results(title, signals_dict):
    all_years = sorted(set(x['year'] for hd in HOLD_DAYS for x in signals_dict[hd])) if any(signals_dict[hd] for hd in HOLD_DAYS) else []

    print(f'\n{"="*95}')
    print(f'=== {title} ===')
    print(f'{"="*95}')

    print(f'\n{"Metrica":>15s}', end='')
    for hd in HOLD_DAYS:
        print(f' |     {hd:>2d}d       ', end='')
    print()
    print('-' * 95)

    for name in ['Trades', 'WR', 'Avg', 'Med', 'Sum', 'PF', 'MaxG', 'MaxL']:
        print(f'{name:>15s}', end='')
        for hd in HOLD_DAYS:
            rets = [x['ret'] for x in signals_dict[hd]]
            if len(rets) == 0:
                print(f' |        -      ', end='')
                continue
            v = pd.Series(rets)
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

    if not all_years:
        return

    # By year
    print(f'\n{"Ano":>5s}', end='')
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
            rets = [x['ret'] for x in signals_dict[hd] if x['year'] == yr]
            if len(rets) > 0:
                v = pd.Series(rets)
                print(f' | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+8.1f}%', end='')
            else:
                print(f' |    0    -       -        -', end='')
        print()

    print('-' * 145)
    print(f'{"TOTAL":>5s}', end='')
    for hd in HOLD_DAYS:
        v = pd.Series([x['ret'] for x in signals_dict[hd]])
        if len(v) > 0:
            print(f' | {len(v):4d} {(v>0).mean()*100:4.1f}% {v.mean():+6.2f}% {v.sum():+8.1f}%', end='')
        else:
            print(f' |    0    -       -        -', end='')
    print()

    # PF by year
    print(f'\n{"Ano":>5s}', end='')
    for hd in HOLD_DAYS:
        print(f' | {hd:>3d}d PF', end='')
    print()
    print('-' * 55)
    for yr in all_years:
        print(f'{yr:5d}', end='')
        for hd in HOLD_DAYS:
            rets = [x['ret'] for x in signals_dict[hd] if x['year'] == yr]
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
        v = pd.Series([x['ret'] for x in signals_dict[hd]])
        if len(v) > 0:
            w = v[v > 0].sum()
            l = abs(v[v < 0].sum())
            print(f' | {w/l if l > 0 else 0:>5.2f}', end='')
        else:
            print(f' |     -', end='')
    print()


print_results('LONG: precio hundido + fundamentales fuertes (comprar dip)', sig_long)
print_results('SHORT: precio inflado + fundamentales debiles (shortear top)', sig_short)

# 5. Comparativa directa LONG vs SHORT para 15d y 22d
print(f'\n{"="*80}')
print(f'=== COMPARATIVA DIRECTA LONG vs SHORT ===')
print(f'{"="*80}')
for hd in HOLD_DAYS:
    vl = pd.Series([x['ret'] for x in sig_long[hd]])
    vs = pd.Series([x['ret'] for x in sig_short[hd]])
    if len(vl) == 0 and len(vs) == 0:
        continue
    print(f'\n  {hd}d:')
    for label, v in [('LONG ', vl), ('SHORT', vs)]:
        if len(v) > 0:
            w = v[v > 0].sum()
            l = abs(v[v < 0].sum())
            pf = w / l if l > 0 else 0
            print(f'    {label}: {len(v):>5d} trades | WR {(v>0).mean()*100:5.1f}% | Avg {v.mean():+6.2f}% | Med {v.median():+6.2f}% | Sum {v.sum():+9.1f}% | PF {pf:.2f}')
        else:
            print(f'    {label}:     0 trades')
