import json, pandas as pd, numpy as np, gc
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

SLIP = 0.3
TRADE_SIZE = 25000

def np_sma(arr, window):
    """Compute SMA using numpy cumsum (no pandas overhead)."""
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    sma = (cs[window:] - cs[:-window]) / window
    result = np.full(len(arr), np.nan)
    result[window-1:] = sma
    return result

# ========== PASS 1: RELAX5 SHORT 15d ==========
print('=== PASS 1: RELAX5 SHORT 15d ===')
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
    grp['eps_ttm'] = grp['eps_actual'].rolling(4, min_periods=4).sum()
    grp['eps_ttm_prev'] = grp['eps_ttm'].shift(4)
    grp['eps_growth_yoy'] = np.where(
        grp['eps_ttm_prev'].abs() > 0.01,
        (grp['eps_ttm'] / grp['eps_ttm_prev'] - 1) * 100, np.nan)
    return grp[['symbol', 'date', 'eps_growth_yoy']]

ef = earn.groupby('symbol', group_keys=False).apply(compute_ef)
earn_lk = {}
for sym, grp in ef.groupby('symbol'):
    grp = grp.sort_values('date')
    earn_lk[sym] = list(zip(grp['date'].values, grp['eps_growth_yoy'].values))
del earn, ec, ef
gc.collect()

def get_epsg(ll, sig):
    eg = None
    for dd, g in ll:
        if dd < sig:
            eg = g
        else:
            break
    return eg

print('Loading prices (close+open only)...')
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
del prc
gc.collect()

r5_trades = []
R5_HD = 15
for sym, g in df.groupby('symbol'):
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < 210:
        continue
    c = g['close'].values.astype(np.float32)
    o = g['open'].values.astype(np.float32)
    dates = g['date'].values
    s50 = np_sma(c, 50)
    s100 = np_sma(c, 100)
    s200 = np_sma(c, 200)
    ell = earn_lk.get(sym, [])
    busy = -1
    for i in range(200, n - 1):
        if np.isnan(s50[i]) or np.isnan(s100[i]) or np.isnan(s200[i]):
            continue
        if s50[i] <= 0 or s100[i] <= 0 or s200[i] <= 0:
            continue
        dd50 = (c[i] / s50[i] - 1) * 100
        dd100 = (c[i] / s100[i] - 1) * 100
        dd200 = (c[i] / s200[i] - 1) * 100
        if dd50 < 8.0 or dd100 < 4.0:
            continue
        if dd200 < -5.0 or dd200 > 5.0:
            continue
        sig = dates[i]
        eg = get_epsg(ell, sig)
        if eg is None or np.isnan(eg) or eg >= 0:
            continue
        bi = i + 1
        if bi >= n or bi <= busy:
            continue
        bp = o[bi]
        if bp <= 0:
            continue
        si = bi + R5_HD
        if si >= n:
            continue
        cv = o[si]
        ret = round(float(bp / cv - 1) * 100 - SLIP, 2)
        pnl = round(TRADE_SIZE * ret / 100, 2)
        busy = si
        yr = pd.Timestamp(sig).year
        r5_trades.append({'year': yr, 'ret': ret, 'pnl': pnl})

del df, earn_lk
gc.collect()
print(f'  RELAX5 15d: {len(r5_trades):,} trades')

# ========== PASS 2: 3 DAY HIGH SHORT 4d ==========
print('\n=== PASS 2: 3 DAY HIGH SHORT 4d ===')
print('Loading prices (with high/low)...')
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

h3_trades = []
H3_HD = 4
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
    busy = -1
    for i in range(203, n - 1):
        if np.isnan(s5[i]) or np.isnan(s50[i]) or np.isnan(s200[i]):
            continue
        # close > SMA5
        if c[i] <= s5[i]:
            continue
        # close < SMA50
        if c[i] >= s50[i]:
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
        si = bi + H3_HD
        if si >= n:
            continue
        cv = o[si]
        ret = round(float(bp / cv - 1) * 100 - SLIP, 2)
        pnl = round(TRADE_SIZE * ret / 100, 2)
        busy = si
        yr = pd.Timestamp(dates[i]).year
        h3_trades.append({'year': yr, 'ret': ret, 'pnl': pnl})

del df
gc.collect()
print(f'  3 Day High 4d: {len(h3_trades):,} trades')

# ========== COMPARATIVA ==========
all_years = sorted(set(
    [t['year'] for t in r5_trades] + [t['year'] for t in h3_trades]
))

def stats(trades):
    if not trades:
        return 0, 0, 0, 0, 0
    rets = pd.Series([t['ret'] for t in trades])
    pnls = pd.Series([t['pnl'] for t in trades])
    n = len(rets)
    wr = (rets > 0).mean() * 100
    avg = rets.mean()
    w = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    pf = w / l if l > 0 else float('inf')
    return n, wr, avg, pf, pnls.sum()

print(f'\n{"="*120}')
print(f'  COMPARATIVA SHORT: RELAX5 15d vs 3 DAY HIGH 4d   (${TRADE_SIZE:,}/trade, {SLIP}% slippage)')
print(f'{"="*120}')
print(f'{"":>5s} | {"-------- RELAX5 15d --------":^35s} | {"------- 3 DAY HIGH 4d ------":^35s} | {"-- COMBINED --":^18s}')
print(f'{"Ano":>5s} | {"N":>5s} {"WR":>5s} {"Avg":>7s} {"PF":>5s} {"PnL$":>10s} | {"N":>5s} {"WR":>5s} {"Avg":>7s} {"PF":>5s} {"PnL$":>10s} | {"PnL$":>10s} {"N":>6s}')
print('-' * 120)

tot_r5 = {'n': 0, 'wins': 0, 'sum_w': 0, 'sum_l': 0, 'sum_ret': 0, 'pnl': 0}
tot_h3 = {'n': 0, 'wins': 0, 'sum_w': 0, 'sum_l': 0, 'sum_ret': 0, 'pnl': 0}
yr_both_wins = 0
yr_both_losses = 0

for yr in all_years:
    r5_yr = [t for t in r5_trades if t['year'] == yr]
    h3_yr = [t for t in h3_trades if t['year'] == yr]

    n1, wr1, avg1, pf1, pnl1 = stats(r5_yr)
    if n1 > 0:
        pfs1 = f'{pf1:5.2f}' if pf1 < 99 else '  inf'
        r5_str = f'{n1:5d} {wr1:4.1f}% {avg1:+6.2f}% {pfs1} {pnl1:+10,.0f}'
        tot_r5['n'] += n1
        tot_r5['wins'] += sum(1 for t in r5_yr if t['ret'] > 0)
        tot_r5['sum_w'] += sum(t['ret'] for t in r5_yr if t['ret'] > 0)
        tot_r5['sum_l'] += abs(sum(t['ret'] for t in r5_yr if t['ret'] < 0))
        tot_r5['sum_ret'] += sum(t['ret'] for t in r5_yr)
        tot_r5['pnl'] += pnl1
    else:
        r5_str = f'    0    -       -     -          -'

    n2, wr2, avg2, pf2, pnl2 = stats(h3_yr)
    if n2 > 0:
        pfs2 = f'{pf2:5.2f}' if pf2 < 99 else '  inf'
        h3_str = f'{n2:5d} {wr2:4.1f}% {avg2:+6.2f}% {pfs2} {pnl2:+10,.0f}'
        tot_h3['n'] += n2
        tot_h3['wins'] += sum(1 for t in h3_yr if t['ret'] > 0)
        tot_h3['sum_w'] += sum(t['ret'] for t in h3_yr if t['ret'] > 0)
        tot_h3['sum_l'] += abs(sum(t['ret'] for t in h3_yr if t['ret'] < 0))
        tot_h3['sum_ret'] += sum(t['ret'] for t in h3_yr)
        tot_h3['pnl'] += pnl2
    else:
        h3_str = f'    0    -       -     -          -'

    both_pnl = pnl1 + pnl2
    both_n = n1 + n2
    if both_pnl > 0:
        yr_both_wins += 1
    elif both_pnl < 0:
        yr_both_losses += 1

    print(f'{yr:5d} | {r5_str} | {h3_str} | {both_pnl:+10,.0f} {both_n:6d}')

print('-' * 120)

def tot_str(t):
    if t['n'] > 0:
        wr = t['wins'] / t['n'] * 100
        avg = t['sum_ret'] / t['n']
        pf = t['sum_w'] / t['sum_l'] if t['sum_l'] > 0 else 0
        return f'{t["n"]:5d} {wr:4.1f}% {avg:+6.2f}% {pf:5.2f} {t["pnl"]:+10,.0f}'
    return f'    0    -       -     -          -'

both_tot = tot_r5['pnl'] + tot_h3['pnl']
both_n = tot_r5['n'] + tot_h3['n']
print(f'{"TOTAL":>5s} | {tot_str(tot_r5)} | {tot_str(tot_h3)} | {both_tot:+10,.0f} {both_n:6d}')

# ========== RESUMEN ==========
print(f'\n{"="*80}')
print(f'  RESUMEN COMPARATIVO')
print(f'{"="*80}')

print(f'\n  {"Metrica":<25s} {"RELAX5 15d":>15s} {"3DH 4d":>15s} {"COMBINED":>15s}')
print(f'  {"-"*70}')
print(f'  {"Trades":<25s} {tot_r5["n"]:>15,} {tot_h3["n"]:>15,} {both_n:>15,}')

wr_r5 = tot_r5['wins'] / tot_r5['n'] * 100 if tot_r5['n'] > 0 else 0
wr_h3 = tot_h3['wins'] / tot_h3['n'] * 100 if tot_h3['n'] > 0 else 0
wins_both = tot_r5['wins'] + tot_h3['wins']
wr_both = wins_both / both_n * 100 if both_n > 0 else 0
print(f'  {"Win Rate":<25s} {wr_r5:>14.1f}% {wr_h3:>14.1f}% {wr_both:>14.1f}%')

avg_r5 = tot_r5['sum_ret'] / tot_r5['n'] if tot_r5['n'] > 0 else 0
avg_h3 = tot_h3['sum_ret'] / tot_h3['n'] if tot_h3['n'] > 0 else 0
avg_both = (tot_r5['sum_ret'] + tot_h3['sum_ret']) / both_n if both_n > 0 else 0
print(f'  {"Avg Return":<25s} {avg_r5:>+14.2f}% {avg_h3:>+14.2f}% {avg_both:>+14.2f}%')

pf_r5 = tot_r5['sum_w'] / tot_r5['sum_l'] if tot_r5['sum_l'] > 0 else 0
pf_h3 = tot_h3['sum_w'] / tot_h3['sum_l'] if tot_h3['sum_l'] > 0 else 0
sum_w_both = tot_r5['sum_w'] + tot_h3['sum_w']
sum_l_both = tot_r5['sum_l'] + tot_h3['sum_l']
pf_both = sum_w_both / sum_l_both if sum_l_both > 0 else 0
print(f'  {"Profit Factor":<25s} {pf_r5:>15.2f} {pf_h3:>15.2f} {pf_both:>15.2f}')

pnl_r5_s = f'${tot_r5["pnl"]:+,.0f}'
pnl_h3_s = f'${tot_h3["pnl"]:+,.0f}'
pnl_both_s = f'${both_tot:+,.0f}'
print(f'  {"PnL Total":<25s} {pnl_r5_s:>15s} {pnl_h3_s:>15s} {pnl_both_s:>15s}')

ppt_r5 = f'${tot_r5["pnl"]/tot_r5["n"]:+,.0f}' if tot_r5["n"] > 0 else '$0'
ppt_h3 = f'${tot_h3["pnl"]/tot_h3["n"]:+,.0f}' if tot_h3["n"] > 0 else '$0'
ppt_both = f'${both_tot/both_n:+,.0f}' if both_n > 0 else '$0'
print(f'  {"PnL / Trade":<25s} {ppt_r5:>15s} {ppt_h3:>15s} {ppt_both:>15s}')

win_yr_r5 = sum(1 for yr in all_years if sum(t['pnl'] for t in r5_trades if t['year']==yr)>0)
win_yr_h3 = sum(1 for yr in all_years if sum(t['pnl'] for t in h3_trades if t['year']==yr)>0)
loss_yr_r5 = sum(1 for yr in all_years if sum(t['pnl'] for t in r5_trades if t['year']==yr)<0)
loss_yr_h3 = sum(1 for yr in all_years if sum(t['pnl'] for t in h3_trades if t['year']==yr)<0)
print(f'  {"Anos ganadores":<25s} {win_yr_r5:>15d} {win_yr_h3:>15d} {yr_both_wins:>15d}')
print(f'  {"Anos perdedores":<25s} {loss_yr_r5:>15d} {loss_yr_h3:>15d} {yr_both_losses:>15d}')

print(f'  {"Hold period":<25s} {"15 days":>15s} {"4 days":>15s}')
print(f'  {"Capital/trade":<25s} {"$25,000":>15s} {"$25,000":>15s}')

n_years = len(all_years)
ann_r5 = tot_r5['pnl'] / n_years if n_years > 0 else 0
ann_h3 = tot_h3['pnl'] / n_years if n_years > 0 else 0
ann_both = both_tot / n_years if n_years > 0 else 0
ann_r5_s = f'${ann_r5:+,.0f}'
ann_h3_s = f'${ann_h3:+,.0f}'
ann_both_s = f'${ann_both:+,.0f}'
print(f'  {"PnL/year (avg)":<25s} {ann_r5_s:>15s} {ann_h3_s:>15s} {ann_both_s:>15s}')

# ========== CORRELACION ==========
print(f'\n{"="*80}')
print(f'  CORRELACION ANUAL')
print(f'{"="*80}')
r5_by_yr = {}
h3_by_yr = {}
for yr in all_years:
    r5_by_yr[yr] = sum(t['pnl'] for t in r5_trades if t['year'] == yr)
    h3_by_yr[yr] = sum(t['pnl'] for t in h3_trades if t['year'] == yr)

s1 = pd.Series(r5_by_yr)
s2 = pd.Series(h3_by_yr)
corr = s1.corr(s2)
print(f'\n  Correlacion PnL anual: {corr:+.3f}')
if corr < 0:
    print(f'  >>> NEGATIVA: las estrategias se complementan (diversificacion)')
elif corr < 0.3:
    print(f'  >>> BAJA: poca relacion, buena diversificacion')
elif corr < 0.7:
    print(f'  >>> MODERADA: cierta relacion')
else:
    print(f'  >>> ALTA: se mueven juntas, poca diversificacion')

print(f'\n  {"Ano":<6s} {"R5 PnL":>10s} {"3DH PnL":>10s} {"COMBINED":>10s} {"Complementa?":>15s}')
print(f'  {"-"*55}')
for yr in all_years:
    p1 = r5_by_yr[yr]
    p2 = h3_by_yr[yr]
    comb = p1 + p2
    if (p1 > 0 and p2 < 0) or (p1 < 0 and p2 > 0):
        comp = 'SI (hedge)'
    elif p1 > 0 and p2 > 0:
        comp = 'AMBOS +'
    else:
        comp = 'AMBOS -'
    print(f'  {yr:<6d} {p1:+10,.0f} {p2:+10,.0f} {comb:+10,.0f} {comp:>15s}')
