"""
Estadisticas E2 detalladas por año y por regimen para los 3 modos
Muestra el PnL E2, con desglose long/short, por periodo
"""
import re, json, csv, bisect, sys, io
import numpy as np
import pandas as pd
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

COST_E2, SLIP = 20000, 0.003
START_YEAR = 2005

E2_ACTIVE = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'BEARISH', 'CAPITULACION', 'RECOVERY'}

STRAT_E2 = {
    'BURBUJA':      [(0,10,'long'),(-20,-10,'short')],
    'GOLDILOCKS':   [(0,10,'long'),(-20,-10,'short')],
    'ALCISTA':      [(0,10,'long'),(-10,None,'short')],
    'NEUTRAL':      [(10,20,'long'),(-20,-10,'short')],
    'CAUTIOUS':     [(-10,None,'short'),(-20,-10,'short')],
    'BEARISH':      [(-10,None,'short'),(-20,-10,'short')],
    'RECOVERY':     [(0,10,'long'),(-20,-10,'long')],
    'CAPITULACION': [(10,20,'long'),(20,30,'long')],
    'CRISIS':       [(-10,None,'short'),(-20,-10,'short')],
    'PANICO':       [(10,20,'long'),(20,30,'long')],
}

REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
                'RECOVERY','CRISIS','PANICO','CAPITULACION']

MODE_LABELS = ['Original', 'Hybrid', 'Min DD']
MODE_CSVS = ['data/regimenes_historico.csv', 'data/regimenes_hybrid.csv', 'data/regimenes_mindd.csv']

# Load E2
print("Loading E2 data...")
with open('acciones_navegable.html', 'r', encoding='utf-8') as f:
    html = f.read()
WEEKS = json.loads(re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))
print(f"  {len(WEEKS)} weeks")

# Process each mode
for mi, (label, csv_path) in enumerate(zip(MODE_LABELS, MODE_CSVS)):
    entries = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            entries.append((str(row['fecha_senal'])[:10], row['regime']))
    entries.sort()
    rd = [e[0] for e in entries]
    rv = [e[1] for e in entries]
    def get_r(d):
        idx = bisect.bisect_right(rd, str(d)[:10]) - 1
        return rv[idx] if idx >= 0 else 'UNKNOWN'

    # Per-year and per-regime stats
    year_stats = defaultdict(lambda: {'pnl_long': 0, 'pnl_short': 0, 'n_long': 0, 'n_short': 0, 'n_weeks': 0})
    regime_stats = defaultdict(lambda: {'pnl_long': 0, 'pnl_short': 0, 'n_long': 0, 'n_short': 0, 'n_weeks': 0})

    for w in WEEKS:
        if w['y'] < START_YEAR: continue
        reg = get_r(w['d'])
        if reg not in E2_ACTIVE: continue

        year = w['y']
        strat = STRAT_E2.get(reg, [])
        wk_long = 0
        wk_short = 0
        n_l = 0
        n_s = 0

        for start, end, direction in strat:
            sel = w['s'][start:end] if end else w['s'][start:]
            for s in sel:
                rv2 = s[8]
                if rv2 is None: continue
                rv2 = max(-50, min(50, rv2))
                if direction == 'long':
                    wk_long += COST_E2 * (rv2/100 - SLIP)
                    n_l += 1
                else:
                    wk_short += COST_E2 * (-rv2/100 - SLIP)
                    n_s += 1

        year_stats[year]['pnl_long'] += wk_long
        year_stats[year]['pnl_short'] += wk_short
        year_stats[year]['n_long'] += n_l
        year_stats[year]['n_short'] += n_s
        year_stats[year]['n_weeks'] += 1

        regime_stats[reg]['pnl_long'] += wk_long
        regime_stats[reg]['pnl_short'] += wk_short
        regime_stats[reg]['n_long'] += n_l
        regime_stats[reg]['n_short'] += n_s
        regime_stats[reg]['n_weeks'] += 1

    # Print year table
    print(f"\n{'='*100}")
    print(f"  E2 - {label} - POR AÑO (Long vs Short)")
    print(f"{'='*100}")
    print(f"  {'Año':>5s}  {'Sem':>4s}  {'PnL Long':>12s}  {'PnL Short':>12s}  {'PnL TOTAL':>12s}  {'#Long':>6s}  {'#Short':>7s}  {'%Long':>6s}")
    print(f"  {'-'*5}  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*7}  {'-'*6}")

    total_l = total_s = total_nl = total_ns = total_w = 0
    for yr in sorted(year_stats.keys()):
        ys = year_stats[yr]
        total = ys['pnl_long'] + ys['pnl_short']
        pct_l = ys['pnl_long'] / total * 100 if total != 0 else 0
        print(f"  {yr:>5d}  {ys['n_weeks']:>4d}  {ys['pnl_long']:>+12,.0f}  {ys['pnl_short']:>+12,.0f}  {total:>+12,.0f}  {ys['n_long']:>6d}  {ys['n_short']:>7d}  {pct_l:>+5.0f}%")
        total_l += ys['pnl_long']
        total_s += ys['pnl_short']
        total_nl += ys['n_long']
        total_ns += ys['n_short']
        total_w += ys['n_weeks']

    total_t = total_l + total_s
    pct = total_l / total_t * 100 if total_t != 0 else 0
    print(f"  {'TOTAL':>5s}  {total_w:>4d}  {total_l:>+12,.0f}  {total_s:>+12,.0f}  {total_t:>+12,.0f}  {total_nl:>6d}  {total_ns:>7d}  {pct:>+5.0f}%")

    # Print regime table
    print(f"\n  E2 - {label} - POR REGIMEN (Long vs Short)")
    print(f"  {'Regimen':>15s}  {'Sem':>4s}  {'PnL Long':>12s}  {'PnL Short':>12s}  {'PnL TOTAL':>12s}  {'#Long':>6s}  {'#Short':>7s}")
    print(f"  {'-'*15}  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*7}")

    for reg in REGIME_ORDER:
        if reg not in regime_stats: continue
        rs = regime_stats[reg]
        if rs['n_weeks'] == 0: continue
        total = rs['pnl_long'] + rs['pnl_short']
        print(f"  {reg:>15s}  {rs['n_weeks']:>4d}  {rs['pnl_long']:>+12,.0f}  {rs['pnl_short']:>+12,.0f}  {total:>+12,.0f}  {rs['n_long']:>6d}  {rs['n_short']:>7d}")

# Now show the DIFFERENCE between modes for key years
print(f"\n{'='*100}")
print(f"  DIFERENCIA E2 Original vs Hybrid vs MinDD - AÑOS CRITICOS")
print(f"{'='*100}")

# Collect all mode data for comparison
all_mode_years = []
for mi, (label, csv_path) in enumerate(zip(MODE_LABELS, MODE_CSVS)):
    entries = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            entries.append((str(row['fecha_senal'])[:10], row['regime']))
    entries.sort()
    rd = [e[0] for e in entries]
    rv = [e[1] for e in entries]
    def get_r(d):
        idx = bisect.bisect_right(rd, str(d)[:10]) - 1
        return rv[idx] if idx >= 0 else 'UNKNOWN'

    year_data = defaultdict(lambda: {'pnl': 0, 'long': 0, 'short': 0, 'regimes': defaultdict(int)})
    for w in WEEKS:
        if w['y'] < START_YEAR: continue
        reg = get_r(w['d'])
        if reg not in E2_ACTIVE: continue

        year = w['y']
        year_data[year]['regimes'][reg] += 1
        strat = STRAT_E2.get(reg, [])
        for start, end, direction in strat:
            sel = w['s'][start:end] if end else w['s'][start:]
            for s in sel:
                rv2 = s[8]
                if rv2 is None: continue
                rv2 = max(-50, min(50, rv2))
                if direction == 'long':
                    year_data[year]['long'] += COST_E2 * (rv2/100 - SLIP)
                else:
                    year_data[year]['short'] += COST_E2 * (-rv2/100 - SLIP)
        year_data[year]['pnl'] = year_data[year]['long'] + year_data[year]['short']

    all_mode_years.append(year_data)

# Compare key years
for yr in [2008, 2009, 2020, 2021, 2022]:
    print(f"\n  --- {yr} ---")
    for mi, label in enumerate(MODE_LABELS):
        yd = all_mode_years[mi][yr]
        regs = ', '.join(f"{r}:{n}" for r, n in sorted(yd['regimes'].items(), key=lambda x: -x[1]))
        print(f"    {label:10s}: PnL {yd['pnl']:>+10,.0f}  (L:{yd['long']:>+10,.0f} S:{yd['short']:>+10,.0f})  Regimenes: {regs}")
