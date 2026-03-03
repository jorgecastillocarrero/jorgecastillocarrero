"""Verificar que NO hay solapamiento de trades por simbolo en ambas estrategias."""
import json, pandas as pd, numpy as np, gc
from sqlalchemy import create_engine

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

with open('data/sp500_constituents.json', 'r') as f:
    sp500 = json.load(f)
tickers = sorted(set(s['symbol'] for s in sp500))

SLIP = 0.3
TRADE_SIZE = 25000

def np_sma(arr, window):
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    sma = (cs[window:] - cs[:-window]) / window
    result = np.full(len(arr), np.nan)
    result[window-1:] = sma
    return result

# ========== EARNINGS (RELAX5) ==========
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

# ========== PRICES ==========
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

# ========== SCAN BOTH STRATEGIES WITH FULL TRADE DETAIL ==========
print('\nScanning both strategies...')

r5_trades = {}  # {symbol: [(entry_idx, exit_idx, entry_date, exit_date), ...]}
h3_trades = {}

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
    s100 = np_sma(c, 100)
    s200 = np_sma(c, 200)
    ell = earn_lk.get(sym, [])

    # --- RELAX5 15d ---
    busy_r5 = -1
    sym_r5 = []
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
        if bi >= n or bi <= busy_r5:
            continue
        bp = o[bi]
        if bp <= 0:
            continue
        si = bi + 15
        if si >= n:
            continue
        busy_r5 = si
        sym_r5.append((bi, si, pd.Timestamp(dates[bi]), pd.Timestamp(dates[si])))
    if sym_r5:
        r5_trades[sym] = sym_r5

    # --- 3DH OPT 4d ---
    busy_h3 = -1
    sym_h3 = []
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
        if bi >= n or bi <= busy_h3:
            continue
        bp = o[bi]
        if bp <= 0:
            continue
        si = bi + 4
        if si >= n:
            continue
        busy_h3 = si
        sym_h3.append((bi, si, pd.Timestamp(dates[bi]), pd.Timestamp(dates[si])))
    if sym_h3:
        h3_trades[sym] = sym_h3

del df
gc.collect()

# ========== CHECK OVERLAP ==========
def check_overlap(name, trades_dict):
    total_trades = 0
    overlaps = 0
    overlap_examples = []
    for sym, tlist in trades_dict.items():
        total_trades += len(tlist)
        tlist_sorted = sorted(tlist, key=lambda x: x[0])
        for j in range(1, len(tlist_sorted)):
            prev_entry, prev_exit, prev_ed, prev_xd = tlist_sorted[j-1]
            curr_entry, curr_exit, curr_ed, curr_xd = tlist_sorted[j]
            # Overlap: current entry <= previous exit (still holding when new trade starts)
            if curr_entry <= prev_exit:
                overlaps += 1
                if len(overlap_examples) < 10:
                    overlap_examples.append(
                        f'    {sym}: trade {j-1} entry_idx={prev_entry} exit_idx={prev_exit} '
                        f'({prev_ed.strftime("%Y-%m-%d")} -> {prev_xd.strftime("%Y-%m-%d")}) '
                        f'vs trade {j} entry_idx={curr_entry} ({curr_ed.strftime("%Y-%m-%d")})'
                    )
    return total_trades, overlaps, overlap_examples

print(f'\n{"="*80}')
print(f'  VERIFICACION DE SOLAPAMIENTO')
print(f'{"="*80}')

for name, trades_dict in [('RELAX5 15d', r5_trades), ('3DH OPT 4d', h3_trades)]:
    total, overlaps, examples = check_overlap(name, trades_dict)
    syms = len(trades_dict)
    print(f'\n  {name}:')
    print(f'    Simbolos: {syms}')
    print(f'    Trades totales: {total:,}')
    print(f'    Solapamientos: {overlaps}')
    if overlaps == 0:
        print(f'    >>> OK: NO HAY SOLAPAMIENTO')
    else:
        print(f'    >>> ERROR: {overlaps} SOLAPAMIENTOS DETECTADOS')
        for ex in examples:
            print(ex)

# ========== VERIFY BUSY LOGIC DETAIL ==========
# Also check: is busy = exit_idx correct?
# Entry at bi, exit at si = bi + HD. Next trade entry must be bi > busy (= si).
# That means next entry must be AFTER exit day.
# But the check is `bi <= busy` -> skip. So bi must be > busy = si.
# Entry bi = si+1 means: next day after exit. Exit is "open at si", so position closed on day si.
# Next trade enters on si+1 (open). No overlap. CORRECT.

print(f'\n  Logica busy:')
print(f'    busy = exit_index (dia de cierre)')
print(f'    Condicion skip: entry_index <= busy')
print(f'    => Siguiente trade entra en exit_index + 1 (dia despues del cierre)')
print(f'    => El dia del cierre (exit) se usa open para cerrar,')
print(f'       el dia siguiente se usa open para abrir nueva posicion.')
print(f'    >>> CORRECTO: no hay dia con 2 posiciones abiertas del mismo simbolo')

# ========== ALSO CHECK: does entry_idx == exit_idx ever happen? ==========
print(f'\n  Verificacion adicional:')
for name, trades_dict in [('RELAX5 15d', r5_trades), ('3DH OPT 4d', h3_trades)]:
    zero_hold = 0
    same_day = 0
    for sym, tlist in trades_dict.items():
        for entry_i, exit_i, ed, xd in tlist:
            if entry_i >= exit_i:
                zero_hold += 1
            if ed.date() == xd.date():
                same_day += 1
    print(f'    {name}: entry >= exit: {zero_hold}, same day entry/exit: {same_day}')
