"""
Simular TODAS las combinaciones de estrategia E2 para cada regimen
Para encontrar qué funciona realmente (media 3 modos)
"""
import re, json, csv, bisect, sys, io
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

COST_E2, SLIP = 20000, 0.003
START_YEAR = 2005

# All possible legs
ALL_LEGS = [
    (0, 10, 'long',  'T0-10 L'),    # Top 10 long
    (10, 20, 'long', 'T10-20 L'),   # Top 11-20 long
    (20, 30, 'long', 'T20-30 L'),   # Top 21-30 long
    (-10, None, 'long', 'B10-N L'), # Bot 10 long
    (-20, -10, 'long', 'B11-20 L'), # Bot 11-20 long
    (0, 10, 'short', 'T0-10 S'),    # Top 10 short
    (10, 20, 'short', 'T10-20 S'),  # Top 11-20 short
    (-10, None, 'short', 'B10 S'),  # Bot 10 short
    (-20, -10, 'short', 'B11-20 S'),# Bot 11-20 short
]

REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
                'RECOVERY','CRISIS','PANICO','CAPITULACION']

MODE_LABELS = ['Original', 'Hybrid', 'MinDD']
MODE_CSVS = ['data/regimenes_historico.csv', 'data/regimenes_hybrid.csv', 'data/regimenes_mindd.csv']

# Load E2
with open('acciones_navegable.html', 'r', encoding='utf-8') as f:
    html = f.read()
WEEKS = json.loads(re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))

# For each mode, build {date: regime}
mode_regimes = []
for csv_path in MODE_CSVS:
    entries = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            entries.append((str(row['fecha_senal'])[:10], row['regime']))
    entries.sort()
    mode_regimes.append(entries)

def get_regime(mode_entries, d):
    rd = [e[0] for e in mode_entries]
    rv = [e[1] for e in mode_entries]
    idx = bisect.bisect_right(rd, str(d)[:10]) - 1
    return rv[idx] if idx >= 0 else 'UNKNOWN'

# Pre-compute: for each week, compute PnL for each individual leg
print("Computing per-leg PnL for all weeks...")

# week_leg_pnl[week_idx][leg_idx] = pnl
week_data = []  # [{year, date, regimes: [r0,r1,r2], leg_pnls: [pnl_per_leg]}]

for wi, w in enumerate(WEEKS):
    if w['y'] < START_YEAR: continue
    regs = [get_regime(mode_regimes[mi], w['d']) for mi in range(3)]

    leg_pnls = []
    for start, end, direction, label in ALL_LEGS:
        sel = w['s'][start:end] if end else w['s'][start:]
        pnl = 0
        for s in sel:
            rv2 = s[8]
            if rv2 is None: continue
            rv2 = max(-50, min(50, rv2))
            if direction == 'long':
                pnl += COST_E2 * (rv2/100 - SLIP)
            else:
                pnl += COST_E2 * (-rv2/100 - SLIP)
        leg_pnls.append(pnl)

    week_data.append({'y': w['y'], 'd': w['d'], 'regs': regs, 'legs': leg_pnls})

print(f"  {len(week_data)} weeks processed")

# For each regime, compute each individual leg's PnL (media 3 modos)
print(f"\n{'='*140}")
print(f"  PnL POR LEG INDIVIDUAL POR REGIMEN (media 3 modos)")
print(f"{'='*140}")

for reg in REGIME_ORDER:
    # Collect weeks for this regime in each mode
    mode_weeks = [[], [], []]
    for wd in week_data:
        for mi in range(3):
            if wd['regs'][mi] == reg:
                mode_weeks[mi].append(wd)

    avg_n = np.mean([len(mw) for mw in mode_weeks])
    if avg_n < 5:
        continue

    print(f"\n  ┌─── {reg} ({avg_n:.0f} sem media) {'─'*90}")
    print(f"  │ {'Leg':>12s} │ {'Orig PnL':>10s} {'€/s':>7s} {'WR':>5s} │ {'Hyb PnL':>10s} {'€/s':>7s} {'WR':>5s} │ {'MinDD PnL':>10s} {'€/s':>7s} {'WR':>5s} │ {'MEDIA PnL':>10s} {'€/s':>7s} {'WR':>5s} │")
    print(f"  │ {'-'*12} │ {'-'*10} {'-'*7} {'-'*5} │ {'-'*10} {'-'*7} {'-'*5} │ {'-'*10} {'-'*7} {'-'*5} │ {'-'*10} {'-'*7} {'-'*5} │")

    for li, (start, end, direction, label) in enumerate(ALL_LEGS):
        mode_vals = []
        for mi in range(3):
            pnls = [wd['legs'][li] for wd in mode_weeks[mi]]
            if len(pnls) == 0:
                mode_vals.append((0, 0, 0))
                continue
            total = sum(pnls)
            per_wk = total / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            mode_vals.append((total, per_wk, wr))

        avg_pnl = np.mean([v[0] for v in mode_vals])
        avg_per_wk = np.mean([v[1] for v in mode_vals])
        avg_wr = np.mean([v[2] for v in mode_vals])

        # Highlight good ones
        marker = ' <<<' if avg_pnl > 20000 else ' !!!' if avg_pnl < -50000 else ''

        print(f"  │ {label:>12s} │ {mode_vals[0][0]:>+10,.0f} {mode_vals[0][1]:>+7,.0f} {mode_vals[0][2]:>4.0f}% │ {mode_vals[1][0]:>+10,.0f} {mode_vals[1][1]:>+7,.0f} {mode_vals[1][2]:>4.0f}% │ {mode_vals[2][0]:>+10,.0f} {mode_vals[2][1]:>+7,.0f} {mode_vals[2][2]:>4.0f}% │ {avg_pnl:>+10,.0f} {avg_per_wk:>+7,.0f} {avg_wr:>4.0f}% │{marker}")

    print(f"  └{'─'*135}")

# ═══════════════════════════════════════════════════════════════
# Propuesta de estrategia óptima
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*140}")
print(f"  MEJORES 2 LEGS POR REGIMEN (media 3 modos, top por €/semana)")
print(f"{'='*140}")

for reg in REGIME_ORDER:
    mode_weeks = [[], [], []]
    for wd in week_data:
        for mi in range(3):
            if wd['regs'][mi] == reg:
                mode_weeks[mi].append(wd)

    avg_n = np.mean([len(mw) for mw in mode_weeks])
    if avg_n < 5: continue

    # Score each leg
    leg_scores = []
    for li, (start, end, direction, label) in enumerate(ALL_LEGS):
        mode_pnls = []
        for mi in range(3):
            pnls = [wd['legs'][li] for wd in mode_weeks[mi]]
            if pnls:
                mode_pnls.append(sum(pnls) / len(pnls))  # per_wk
            else:
                mode_pnls.append(0)
        avg_per_wk = np.mean(mode_pnls)
        avg_pnl = np.mean([sum(wd['legs'][li] for wd in mode_weeks[mi]) for mi in range(3)])
        # Consistency: all 3 modes positive?
        all_pos = all(p > 0 for p in mode_pnls)
        leg_scores.append((label, avg_per_wk, avg_pnl, all_pos, mode_pnls))

    # Sort by avg per_wk
    leg_scores.sort(key=lambda x: -x[1])

    print(f"\n  {reg} ({avg_n:.0f} sem):")
    for rank, (label, per_wk, total_pnl, all_pos, mode_pnls) in enumerate(leg_scores):
        if rank >= 4 and per_wk <= 0: break
        consist = '3/3' if all_pos else f"{sum(1 for p in mode_pnls if p > 0)}/3"
        flag = ' ★' if all_pos and per_wk > 0 else ''
        print(f"    {rank+1}. {label:>12s}  €/sem: {per_wk:>+7,.0f}  Total: {total_pnl:>+10,.0f}  Consist: {consist}{flag}")
