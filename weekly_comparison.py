"""
Comparativa semanal Original vs Min DD vs SPY
- Lista semana a semana con regimen y PnL
- Correlacion de cada estrategia con SPY
"""
import re, json, sys, io, csv, bisect
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import groupby

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
BASE = Path(__file__).parent

COST_3DH = 25000
COST_E2 = 20000
SLIP = 0.003
START_YEAR = 2005
MAX_3DH = 30

H3_ACTIVE = {'CAUTIOUS', 'RECOVERY', 'CRISIS', 'PANICO'}
E2_ACTIVE = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'BEARISH', 'CAPITULACION', 'RECOVERY'}

STRAT_E2_ALL = {
    'BURBUJA':      [(0, 10, 'long'),  (-20, -10, 'short')],
    'GOLDILOCKS':   [(0, 10, 'long'),  (-20, -10, 'short')],
    'ALCISTA':      [(0, 10, 'long'),  (-10, None, 'short')],
    'NEUTRAL':      [(10, 20, 'long'), (-20, -10, 'short')],
    'CAUTIOUS':     [(-10, None, 'short'), (-20, -10, 'short')],
    'BEARISH':      [(-10, None, 'short'), (-20, -10, 'short')],
    'RECOVERY':     [(0, 10, 'long'),  (-20, -10, 'long')],
    'CAPITULACION': [(10, 20, 'long'), (20, 30, 'long')],
    'CRISIS':       [(-10, None, 'short'), (-20, -10, 'short')],
    'PANICO':       [(10, 20, 'long'), (20, 30, 'long')],
}

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data...")

# 3DH trades with cap30
with open(BASE / 'data' / '3dh_opt_4d_trades.json', 'r') as f:
    h3_raw = json.load(f)
h3_all = []
for t in h3_raw:
    h3_all.append({'sym': t['sym'], 'sig': t['sig'], 'entry': t['entry'], 'exit': t['exit'],
        'ret': t['ret'], 'pnl': t['pnl'], 'year': int(t['sig'][:4])})
h3_all.sort(key=lambda t: t['entry'])

# Apply cap30
h3_trades = []
open_by_exit = defaultdict(int)
for entry_date, group in groupby(h3_all, key=lambda t: t['entry']):
    day_trades = list(group)
    expired = [ex for ex in open_by_exit if ex <= entry_date]
    for ex in expired:
        del open_by_exit[ex]
    currently_open = sum(open_by_exit.values())
    available = max(0, MAX_3DH - currently_open)
    n_signals = len(day_trades)
    if available <= 0:
        continue
    if n_signals <= available:
        for t in day_trades:
            h3_trades.append(t)
            open_by_exit[t['exit']] += 1
    else:
        avg_ret = round(float(np.mean([t['ret'] for t in day_trades])), 2)
        avg_pnl = round(COST_3DH * avg_ret / 100, 2)
        day_trades.sort(key=lambda t: t['sym'])
        for i in range(available):
            h3_trades.append({**day_trades[i], 'ret': avg_ret, 'pnl': avg_pnl})
            open_by_exit[day_trades[i]['exit']] += 1

print(f"  3DH cap30: {len(h3_trades)} trades")

# E2 stock data
with open(BASE / 'acciones_navegable.html', 'r', encoding='utf-8') as f:
    html_text = f.read()
m2 = re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
WEEKS = json.loads(m2.group(1))
del html_text
print(f"  E2: {len(WEEKS)} weeks")

# Regime lookups
def build_regime_lookup(csv_path):
    entries = []
    with open(BASE / csv_path) as f:
        for row in csv.DictReader(f):
            d = str(row['fecha_senal'])[:10]
            r = row['regime']
            spy_ret = row.get('spy_ret_pct', '')
            entries.append((d, r, spy_ret))
    entries.sort()
    dates = [e[0] for e in entries]
    regimes = [e[1] for e in entries]
    spy_rets = [e[2] for e in entries]
    def get(sig_date):
        idx = bisect.bisect_right(dates, str(sig_date)[:10]) - 1
        if idx >= 0:
            sr = spy_rets[idx]
            sr_val = float(sr) if sr and sr != '' else None
            return regimes[idx], sr_val
        return 'UNKNOWN', None
    return get

get_orig = build_regime_lookup('data/regimenes_historico.csv')
get_mindd = build_regime_lookup('data/regimenes_mindd.csv')

# ═══════════════════════════════════════════════════════════════
# 2. COMPUTE WEEKLY PNL FOR EACH MODE
# ═══════════════════════════════════════════════════════════════
print("Computing weekly PnL...")

# E2 dates (sorted)
e2_dates = sorted(set(w['d'] for w in WEEKS if w['y'] >= START_YEAR))

def compute_weekly(get_regime):
    """Returns dict {date: {regime, e2_pnl, h3_pnl, dual_pnl, spy_ret}}"""
    result = {}

    # E2 weekly
    for w in WEEKS:
        date, year = w['d'], w['y']
        if year < START_YEAR:
            continue
        stocks = w['s']
        regime, spy_ret = get_regime(date)

        strat = STRAT_E2_ALL.get(regime, [])
        e2_pnl = 0
        n_pos = 0
        for start, end, direction in strat:
            selected = stocks[start:end] if end is not None else stocks[start:]
            for s in selected:
                rv = s[8]
                if rv is None:
                    continue
                rv = max(-50, min(50, rv))
                if direction == 'long':
                    e2_pnl += COST_E2 * (rv / 100 - SLIP)
                else:
                    e2_pnl += COST_E2 * (-rv / 100 - SLIP)
                n_pos += 1

        # Only count E2 if regime is in E2_ACTIVE
        e2_active = regime in E2_ACTIVE
        result[date] = {
            'regime': regime,
            'e2_pnl': round(e2_pnl, 2) if e2_active else 0,
            'e2_raw': round(e2_pnl, 2),
            'e2_active': e2_active,
            'h3_pnl': 0,
            'spy_ret': spy_ret,
            'n_pos': n_pos if e2_active else 0,
        }

    # 3DH by week (assign to nearest E2 week)
    for t in h3_trades:
        regime, _ = get_regime(t['sig'])
        if regime not in H3_ACTIVE:
            continue
        sig = t['sig']
        idx = bisect.bisect_right(e2_dates, sig) - 1
        if idx >= 0:
            week_date = e2_dates[idx]
            if week_date in result:
                result[week_date]['h3_pnl'] += t['pnl']

    # Compute dual
    for d in result:
        result[d]['h3_pnl'] = round(result[d]['h3_pnl'], 2)
        result[d]['dual_pnl'] = round(result[d]['e2_pnl'] + result[d]['h3_pnl'], 2)

    return result

orig_weekly = compute_weekly(get_orig)
mindd_weekly = compute_weekly(get_mindd)

# ═══════════════════════════════════════════════════════════════
# 3. WEEKLY LIST + CORRELATION
# ═══════════════════════════════════════════════════════════════
all_dates = sorted(set(list(orig_weekly.keys()) + list(mindd_weekly.keys())))

# Build arrays for correlation
orig_pnl_arr = []
mindd_pnl_arr = []
spy_ret_arr = []
orig_e2_arr = []
mindd_e2_arr = []
orig_h3_arr = []
mindd_h3_arr = []

rows = []
for d in all_dates:
    o = orig_weekly.get(d, {'regime': '-', 'e2_pnl': 0, 'h3_pnl': 0, 'dual_pnl': 0, 'spy_ret': None})
    m = mindd_weekly.get(d, {'regime': '-', 'e2_pnl': 0, 'h3_pnl': 0, 'dual_pnl': 0, 'spy_ret': None})
    spy = o['spy_ret'] if o['spy_ret'] is not None else m['spy_ret']

    rows.append({
        'date': d,
        'orig_reg': o['regime'],
        'mindd_reg': m['regime'],
        'same_reg': o['regime'] == m['regime'],
        'orig_e2': o['e2_pnl'],
        'orig_h3': o['h3_pnl'],
        'orig_dual': o['dual_pnl'],
        'mindd_e2': m['e2_pnl'],
        'mindd_h3': m['h3_pnl'],
        'mindd_dual': m['dual_pnl'],
        'spy_ret': spy,
    })

    if spy is not None:
        orig_pnl_arr.append(o['dual_pnl'])
        mindd_pnl_arr.append(m['dual_pnl'])
        spy_ret_arr.append(spy)
        orig_e2_arr.append(o['e2_pnl'])
        mindd_e2_arr.append(m['e2_pnl'])
        orig_h3_arr.append(o['h3_pnl'])
        mindd_h3_arr.append(m['h3_pnl'])

# Correlations
spy_pnl_proxy = [s * COST_E2 * 20 / 100 for s in spy_ret_arr]  # SPY scaled to similar capital

corr_orig_spy = float(np.corrcoef(orig_pnl_arr, spy_ret_arr)[0, 1])
corr_mindd_spy = float(np.corrcoef(mindd_pnl_arr, spy_ret_arr)[0, 1])
corr_orig_e2_spy = float(np.corrcoef(orig_e2_arr, spy_ret_arr)[0, 1])
corr_mindd_e2_spy = float(np.corrcoef(mindd_e2_arr, spy_ret_arr)[0, 1])
corr_orig_h3_spy = float(np.corrcoef(orig_h3_arr, spy_ret_arr)[0, 1])
corr_mindd_h3_spy = float(np.corrcoef(mindd_h3_arr, spy_ret_arr)[0, 1])
corr_orig_mindd = float(np.corrcoef(orig_pnl_arr, mindd_pnl_arr)[0, 1])

# ═══════════════════════════════════════════════════════════════
# 4. OUTPUT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 130)
print("  CORRELACIONES")
print("=" * 130)
print(f"  {'Comparacion':<40} {'Pearson r':>10}")
print(f"  {'-'*40} {'-'*10}")
print(f"  {'Original DUAL vs SPY':<40} {corr_orig_spy:>+10.4f}")
print(f"  {'Min DD DUAL vs SPY':<40} {corr_mindd_spy:>+10.4f}")
print(f"  {'Original E2 vs SPY':<40} {corr_orig_e2_spy:>+10.4f}")
print(f"  {'Min DD E2 vs SPY':<40} {corr_mindd_e2_spy:>+10.4f}")
print(f"  {'Original 3DH vs SPY':<40} {corr_orig_h3_spy:>+10.4f}")
print(f"  {'Min DD 3DH vs SPY':<40} {corr_mindd_h3_spy:>+10.4f}")
print(f"  {'Original DUAL vs Min DD DUAL':<40} {corr_orig_mindd:>+10.4f}")

# By regime correlation
print("\n" + "=" * 130)
print("  CORRELACION POR REGIMEN (DUAL vs SPY)")
print("=" * 130)
print(f"  {'Regimen':<14} | {'--- ORIGINAL ---':^30} | {'--- MIN DD ---':^30}")
print(f"  {'':14} | {'N':>5} {'Corr':>8} {'AvgPnL':>10} {'AvgSPY':>8} | {'N':>5} {'Corr':>8} {'AvgPnL':>10} {'AvgSPY':>8}")

from collections import Counter
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
                'RECOVERY','CRISIS','PANICO','CAPITULACION']

for reg in REGIME_ORDER:
    for label, weekly_data in [('ORIG', orig_weekly), ('MINDD', mindd_weekly)]:
        reg_rows = [(d, weekly_data[d]) for d in sorted(weekly_data) if weekly_data[d]['regime'] == reg and weekly_data[d]['spy_ret'] is not None]
        if label == 'ORIG':
            if len(reg_rows) >= 3:
                pnls = [r[1]['dual_pnl'] for r in reg_rows]
                spys = [r[1]['spy_ret'] for r in reg_rows]
                corr = float(np.corrcoef(pnls, spys)[0, 1])
                avg_pnl = np.mean(pnls)
                avg_spy = np.mean(spys)
                print(f"  {reg:<14} | {len(reg_rows):>5} {corr:>+8.4f} {avg_pnl:>10,.0f} {avg_spy:>+7.3f}%", end='')
            else:
                print(f"  {reg:<14} | {'':>5} {'n/a':>8} {'':>10} {'':>8}", end='')
        else:
            if len(reg_rows) >= 3:
                pnls = [r[1]['dual_pnl'] for r in reg_rows]
                spys = [r[1]['spy_ret'] for r in reg_rows]
                corr = float(np.corrcoef(pnls, spys)[0, 1])
                avg_pnl = np.mean(pnls)
                avg_spy = np.mean(spys)
                print(f" | {len(reg_rows):>5} {corr:>+8.4f} {avg_pnl:>10,.0f} {avg_spy:>+7.3f}%")
            else:
                print(f" | {'':>5} {'n/a':>8} {'':>10} {'':>8}")

# Regime distribution comparison
print("\n" + "=" * 130)
print("  DISTRIBUCION DE REGIMENES")
print("=" * 130)
orig_regs = Counter(r['orig_reg'] for r in rows)
mindd_regs = Counter(r['mindd_reg'] for r in rows)
diff_count = sum(1 for r in rows if not r['same_reg'])
print(f"  Total semanas: {len(rows)} | Diferentes: {diff_count} ({diff_count/len(rows)*100:.1f}%)\n")
print(f"  {'Regimen':<14} {'Original':>8} {'Min DD':>8} {'Diff':>6}")
for reg in REGIME_ORDER:
    o = orig_regs.get(reg, 0)
    m = mindd_regs.get(reg, 0)
    print(f"  {reg:<14} {o:>8} {m:>8} {m-o:>+6}")

# Annual correlation
print("\n" + "=" * 130)
print("  CORRELACION ANUAL (DUAL vs SPY)")
print("=" * 130)
print(f"  {'Año':>6} | {'--- ORIGINAL ---':^25} | {'--- MIN DD ---':^25}")
print(f"  {'':>6} | {'N':>4} {'Corr':>8} {'PnL':>10} | {'N':>4} {'Corr':>8} {'PnL':>10}")

years = sorted(set(int(d[:4]) for d in all_dates))
for y in years:
    yr_rows = [r for r in rows if int(r['date'][:4]) == y and r['spy_ret'] is not None]
    if len(yr_rows) < 5:
        continue
    o_pnl = [r['orig_dual'] for r in yr_rows]
    m_pnl = [r['mindd_dual'] for r in yr_rows]
    spy = [r['spy_ret'] for r in yr_rows]
    co = float(np.corrcoef(o_pnl, spy)[0, 1])
    cm = float(np.corrcoef(m_pnl, spy)[0, 1])
    o_tot = sum(o_pnl)
    m_tot = sum(m_pnl)
    print(f"  {y:>6} | {len(yr_rows):>4} {co:>+8.4f} {o_tot:>10,.0f} | {len(yr_rows):>4} {cm:>+8.4f} {m_tot:>10,.0f}")

# ═══════════════════════════════════════════════════════════════
# 5. WEEKLY LIST (CSV + console)
# ═══════════════════════════════════════════════════════════════
csv_path = BASE / 'data' / 'weekly_comparison.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['fecha', 'orig_reg', 'mindd_reg', 'mismo',
                     'orig_e2', 'orig_h3', 'orig_dual',
                     'mindd_e2', 'mindd_h3', 'mindd_dual',
                     'spy_ret_pct'])
    for r in rows:
        writer.writerow([r['date'], r['orig_reg'], r['mindd_reg'],
                         'SI' if r['same_reg'] else 'NO',
                         r['orig_e2'], r['orig_h3'], r['orig_dual'],
                         r['mindd_e2'], r['mindd_h3'], r['mindd_dual'],
                         round(r['spy_ret'], 4) if r['spy_ret'] is not None else ''])

print(f"\n  CSV guardado: {csv_path}")

# Console: first 30 + last 10
print("\n" + "=" * 150)
print("  LISTA SEMANAL (primeras 30 + ultimas 10)")
print("=" * 150)
print(f"  {'Fecha':<12} {'OrigReg':>14} {'MinDDReg':>14} {'=':>2} "
      f"{'OrigE2':>10} {'OrigH3':>10} {'OrigDual':>10} "
      f"{'MinE2':>10} {'MinH3':>10} {'MinDual':>10} "
      f"{'SPY%':>8}")
print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*2} "
      f"{'-'*10} {'-'*10} {'-'*10} "
      f"{'-'*10} {'-'*10} {'-'*10} "
      f"{'-'*8}")

def print_row(r):
    eq = '=' if r['same_reg'] else '!'
    spy_str = f"{r['spy_ret']:>+7.3f}%" if r['spy_ret'] is not None else '    n/a'
    print(f"  {r['date']:<12} {r['orig_reg']:>14} {r['mindd_reg']:>14} {eq:>2} "
          f"{r['orig_e2']:>10,.0f} {r['orig_h3']:>10,.0f} {r['orig_dual']:>10,.0f} "
          f"{r['mindd_e2']:>10,.0f} {r['mindd_h3']:>10,.0f} {r['mindd_dual']:>10,.0f} "
          f"{spy_str}")

for r in rows[:30]:
    print_row(r)
print(f"  {'... ('+str(len(rows)-40)+' filas mas) ...':^150}")
for r in rows[-10:]:
    print_row(r)

# Summary
print("\n" + "=" * 130)
print("  RESUMEN FINAL")
print("=" * 130)
o_total = sum(r['orig_dual'] for r in rows)
m_total = sum(r['mindd_dual'] for r in rows)
print(f"  Original DUAL total: EUR {o_total:>12,.0f}")
print(f"  Min DD DUAL total:   EUR {m_total:>12,.0f}")
print(f"  Diferencia:          EUR {o_total - m_total:>12,.0f} ({(o_total-m_total)/abs(m_total)*100:+.1f}%)")
print(f"\n  Correlacion Original vs SPY:  {corr_orig_spy:+.4f}")
print(f"  Correlacion Min DD vs SPY:    {corr_mindd_spy:+.4f}")
print(f"  Correlacion Original vs MinDD:{corr_orig_mindd:+.4f}")
