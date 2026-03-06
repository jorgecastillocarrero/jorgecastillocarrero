"""
Media de E2 PnL de los 3 modos por regimen - base para decidir estrategia
Si un resultado es overfitting (solo en 1 modo), la media lo diluye
"""
import re, json, csv, bisect, sys, io
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

COST_E2, SLIP = 20000, 0.003
START_YEAR = 2005

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

MODE_LABELS = ['Original', 'Hybrid', 'MinDD']
MODE_CSVS = ['data/regimenes_historico.csv', 'data/regimenes_hybrid.csv', 'data/regimenes_mindd.csv']

# Load E2
with open('acciones_navegable.html', 'r', encoding='utf-8') as f:
    html = f.read()
WEEKS = json.loads(re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))

# Process each mode
all_mode_stats = []

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

    regime_data = defaultdict(lambda: {
        'n': 0, 'pnl_long': 0, 'pnl_short': 0, 'n_long': 0, 'n_short': 0,
        'weekly_pnls': [], 'long_rets': [], 'short_rets': []
    })

    for w in WEEKS:
        if w['y'] < START_YEAR: continue
        reg = get_r(w['d'])
        rd2 = regime_data[reg]
        rd2['n'] += 1

        strat = STRAT_E2.get(reg, [])
        wk_l = wk_s = 0
        n_l = n_s = 0

        for start, end, direction in strat:
            sel = w['s'][start:end] if end else w['s'][start:]
            for s in sel:
                rv2 = s[8]
                if rv2 is None: continue
                rv2 = max(-50, min(50, rv2))
                if direction == 'long':
                    wk_l += COST_E2 * (rv2/100 - SLIP)
                    n_l += 1
                    rd2['long_rets'].append(rv2)
                else:
                    wk_s += COST_E2 * (-rv2/100 - SLIP)
                    n_s += 1
                    rd2['short_rets'].append(-rv2)

        rd2['pnl_long'] += wk_l
        rd2['pnl_short'] += wk_s
        rd2['n_long'] += n_l
        rd2['n_short'] += n_s
        rd2['weekly_pnls'].append(wk_l + wk_s)

    all_mode_stats.append(regime_data)

# ═══════════════════════════════════════════════════════════════
# TABLA COMPARATIVA + MEDIA
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*160}")
print(f"  E2 POR REGIMEN: 3 modos + MEDIA (base para estrategia)")
print(f"{'='*160}")

print(f"\n  {'Regimen':>14s} │ {'Estrategia':>22s} │ {'Orig PnL':>10s} {'€/s':>7s} {'WR':>5s} │ {'Hyb PnL':>10s} {'€/s':>7s} {'WR':>5s} │ {'MinDD PnL':>10s} {'€/s':>7s} {'WR':>5s} │ {'MEDIA PnL':>10s} {'€/s':>7s} {'WR':>5s} │ {'Long%':>6s} │ {'OF?':>4s}")
print(f"  {'-'*14} │ {'-'*22} │ {'-'*10} {'-'*7} {'-'*5} │ {'-'*10} {'-'*7} {'-'*5} │ {'-'*10} {'-'*7} {'-'*5} │ {'-'*10} {'-'*7} {'-'*5} │ {'-'*6} │ {'-'*4}")

total_pnls = [0, 0, 0]
total_avg = 0
total_n = 0

for reg in REGIME_ORDER:
    strat = STRAT_E2.get(reg, [])
    strat_desc = ' + '.join(f"{'T' if s>=0 else 'B'}{abs(s)}-{abs(e) if e else 'N'} {'L' if d=='long' else 'S'}" for s,e,d in strat)

    vals = []
    for mi in range(3):
        rs = all_mode_stats[mi][reg]
        pnl = rs['pnl_long'] + rs['pnl_short']
        per_wk = pnl / rs['n'] if rs['n'] > 0 else 0
        wr = sum(1 for v in rs['weekly_pnls'] if v > 0) / len(rs['weekly_pnls']) * 100 if rs['weekly_pnls'] else 0
        vals.append((rs['n'], pnl, per_wk, wr))
        total_pnls[mi] += pnl

    # Media
    avg_pnl = np.mean([v[1] for v in vals])
    avg_n = np.mean([v[0] for v in vals])
    avg_per_wk = avg_pnl / avg_n if avg_n > 0 else 0
    avg_wr = np.mean([v[3] for v in vals])
    total_avg += avg_pnl
    total_n += avg_n

    # Long vs short contribution in average
    avg_l = np.mean([all_mode_stats[mi][reg]['pnl_long'] for mi in range(3)])
    avg_s = np.mean([all_mode_stats[mi][reg]['pnl_short'] for mi in range(3)])
    pct_long = avg_l / (avg_l + avg_s) * 100 if (avg_l + avg_s) != 0 else 0

    # Overfitting flag
    max_diff = max(v[1] for v in vals) - min(v[1] for v in vals)
    of_flag = 'SI' if max_diff > 100000 else 'med' if max_diff > 50000 else ''

    print(f"  {reg:>14s} │ {strat_desc:>22s} │ {vals[0][1]:>+10,.0f} {vals[0][2]:>+7,.0f} {vals[0][3]:>4.0f}% │ {vals[1][1]:>+10,.0f} {vals[1][2]:>+7,.0f} {vals[1][3]:>4.0f}% │ {vals[2][1]:>+10,.0f} {vals[2][2]:>+7,.0f} {vals[2][3]:>4.0f}% │ {avg_pnl:>+10,.0f} {avg_per_wk:>+7,.0f} {avg_wr:>4.0f}% │ {pct_long:>+5.0f}% │ {of_flag:>4s}")

print(f"  {'TOTAL':>14s} │ {'':>22s} │ {total_pnls[0]:>+10,.0f} {'':>7s} {'':>5s} │ {total_pnls[1]:>+10,.0f} {'':>7s} {'':>5s} │ {total_pnls[2]:>+10,.0f} {'':>7s} {'':>5s} │ {total_avg:>+10,.0f} {'':>7s} {'':>5s} │ {'':>6s} │ {'':>4s}")

# ═══════════════════════════════════════════════════════════════
# DESGLOSE LONG vs SHORT por regimen (media 3 modos)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"  DESGLOSE LONG vs SHORT (media 3 modos)")
print(f"{'='*120}")
print(f"\n  {'Regimen':>14s} │ {'PnL Long':>10s} │ {'Avg L%':>7s} │ {'#Long':>6s} │ {'PnL Short':>10s} │ {'Avg S%':>7s} │ {'#Short':>7s} │ {'TOTAL':>10s} │ {'Veredicto':>30s}")
print(f"  {'-'*14} │ {'-'*10} │ {'-'*7} │ {'-'*6} │ {'-'*10} │ {'-'*7} │ {'-'*7} │ {'-'*10} │ {'-'*30}")

for reg in REGIME_ORDER:
    avg_l = np.mean([all_mode_stats[mi][reg]['pnl_long'] for mi in range(3)])
    avg_s = np.mean([all_mode_stats[mi][reg]['pnl_short'] for mi in range(3)])
    avg_nl = np.mean([all_mode_stats[mi][reg]['n_long'] for mi in range(3)])
    avg_ns = np.mean([all_mode_stats[mi][reg]['n_short'] for mi in range(3)])

    # Average return per trade
    all_l_rets = []
    all_s_rets = []
    for mi in range(3):
        all_l_rets.extend(all_mode_stats[mi][reg]['long_rets'])
        all_s_rets.extend(all_mode_stats[mi][reg]['short_rets'])
    avg_l_ret = np.mean(all_l_rets) if all_l_rets else 0
    avg_s_ret = np.mean(all_s_rets) if all_s_rets else 0

    total = avg_l + avg_s

    # Verdict
    if avg_l > 0 and avg_s > 0:
        verdict = 'Long OK + Short OK'
    elif avg_l > 0 and avg_s <= 0:
        if abs(avg_s) > avg_l * 0.5:
            verdict = 'Long OK, Short PIERDE mucho'
        else:
            verdict = 'Long OK, Short pierde poco'
    elif avg_l <= 0 and avg_s > 0:
        if abs(avg_l) > avg_s * 0.5:
            verdict = 'Short OK, Long PIERDE mucho'
        else:
            verdict = 'Short OK, Long pierde poco'
    else:
        verdict = 'AMBOS PIERDEN'

    print(f"  {reg:>14s} │ {avg_l:>+10,.0f} │ {avg_l_ret:>+6.2f}% │ {avg_nl:>5.0f}  │ {avg_s:>+10,.0f} │ {avg_s_ret:>+6.2f}% │ {avg_ns:>6.0f}  │ {total:>+10,.0f} │ {verdict}")

print(f"\n{'='*120}")
print(f"  RECOMENDACION BASADA EN MEDIA 3 MODOS")
print(f"{'='*120}")
print(f"\n  Regimenes donde E2 aporta valor (media positiva):")
for reg in REGIME_ORDER:
    avg_pnl = np.mean([all_mode_stats[mi][reg]['pnl_long'] + all_mode_stats[mi][reg]['pnl_short'] for mi in range(3)])
    if avg_pnl > 0:
        avg_l = np.mean([all_mode_stats[mi][reg]['pnl_long'] for mi in range(3)])
        avg_s = np.mean([all_mode_stats[mi][reg]['pnl_short'] for mi in range(3)])
        n = np.mean([all_mode_stats[mi][reg]['n'] for mi in range(3)])
        print(f"    {reg:>14s}: Media {avg_pnl:>+10,.0f} ({n:.0f} sem)  Long:{avg_l:>+10,.0f}  Short:{avg_s:>+10,.0f}")

print(f"\n  Regimenes donde E2 NO aporta valor (media negativa o ~0):")
for reg in REGIME_ORDER:
    avg_pnl = np.mean([all_mode_stats[mi][reg]['pnl_long'] + all_mode_stats[mi][reg]['pnl_short'] for mi in range(3)])
    if avg_pnl <= 0:
        avg_l = np.mean([all_mode_stats[mi][reg]['pnl_long'] for mi in range(3)])
        avg_s = np.mean([all_mode_stats[mi][reg]['pnl_short'] for mi in range(3)])
        n = np.mean([all_mode_stats[mi][reg]['n'] for mi in range(3)])
        print(f"    {reg:>14s}: Media {avg_pnl:>+10,.0f} ({n:.0f} sem)  Long:{avg_l:>+10,.0f}  Short:{avg_s:>+10,.0f}")
