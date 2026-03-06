"""
Tabla completa de retornos E2 por regimen x 3 modos
Con desglose long/short, PnL/semana, y comparativa para detectar overfitting
"""
import re, json, csv, bisect, sys, io
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

COST_E2, SLIP = 20000, 0.003

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
START_YEAR = 2005

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
        'weekly_rets': [], 'long_rets': [], 'short_rets': [],
        'spy_rets': []
    })

    for w in WEEKS:
        if w['y'] < START_YEAR: continue
        reg = get_r(w['d'])
        rd2 = regime_data[reg]
        rd2['n'] += 1

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
                    pnl = COST_E2 * (rv2/100 - SLIP)
                    wk_long += pnl
                    n_l += 1
                    rd2['long_rets'].append(rv2)
                else:
                    pnl = COST_E2 * (-rv2/100 - SLIP)
                    wk_short += pnl
                    n_s += 1
                    rd2['short_rets'].append(-rv2)

        rd2['pnl_long'] += wk_long
        rd2['pnl_short'] += wk_short
        rd2['n_long'] += n_l
        rd2['n_short'] += n_s
        rd2['weekly_rets'].append(wk_long + wk_short)

    all_mode_stats.append(regime_data)

# ═══════════════════════════════════════════════════════════════
# TABLA PRINCIPAL: E2 por regimen x 3 modos
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*140}")
print(f"  E2 RETORNOS POR REGIMEN - COMPARATIVA 3 MODOS (2005-2026)")
print(f"{'='*140}")

for mi, label in enumerate(MODE_LABELS):
    rd = all_mode_stats[mi]
    print(f"\n  ┌─── {label} {'─'*120}")
    print(f"  │ {'Regimen':>14s} │ {'Sem':>4s} │ {'PnL Long':>10s} │ {'PnL Short':>10s} │ {'PnL TOTAL':>10s} │ {'€/sem':>7s} │ {'WR':>5s} │ {'AvgL%':>7s} │ {'AvgS%':>7s} │ {'#L':>5s} │ {'#S':>5s} │ {'Estrategia':>30s} │")
    print(f"  │ {'-'*14} │ {'-'*4} │ {'-'*10} │ {'-'*10} │ {'-'*10} │ {'-'*7} │ {'-'*5} │ {'-'*7} │ {'-'*7} │ {'-'*5} │ {'-'*5} │ {'-'*30} │")

    total_n = total_l = total_s = 0
    all_wr = []
    for reg in REGIME_ORDER:
        rs = rd[reg]
        if rs['n'] == 0: continue
        total = rs['pnl_long'] + rs['pnl_short']
        per_wk = total / rs['n'] if rs['n'] > 0 else 0
        wr_vals = rs['weekly_rets']
        wr = sum(1 for v in wr_vals if v > 0) / len(wr_vals) * 100 if wr_vals else 0
        avg_l = np.mean(rs['long_rets']) if rs['long_rets'] else 0
        avg_s = np.mean(rs['short_rets']) if rs['short_rets'] else 0

        strat = STRAT_E2.get(reg, [])
        strat_desc = ' + '.join(f"{'T' if s>=0 else 'B'}{abs(s)}-{abs(e) if e else 'N'} {'L' if d=='long' else 'S'}" for s,e,d in strat)

        print(f"  │ {reg:>14s} │ {rs['n']:>4d} │ {rs['pnl_long']:>+10,.0f} │ {rs['pnl_short']:>+10,.0f} │ {total:>+10,.0f} │ {per_wk:>+7,.0f} │ {wr:>4.0f}% │ {avg_l:>+6.2f}% │ {avg_s:>+6.2f}% │ {rs['n_long']:>5d} │ {rs['n_short']:>5d} │ {strat_desc:>30s} │")

        total_n += rs['n']
        total_l += rs['pnl_long']
        total_s += rs['pnl_short']
        all_wr.extend(wr_vals)

    total_t = total_l + total_s
    wr_total = sum(1 for v in all_wr if v > 0) / len(all_wr) * 100 if all_wr else 0
    print(f"  │ {'TOTAL':>14s} │ {total_n:>4d} │ {total_l:>+10,.0f} │ {total_s:>+10,.0f} │ {total_t:>+10,.0f} │ {total_t/total_n:>+7,.0f} │ {wr_total:>4.0f}% │ {'':>7s} │ {'':>7s} │ {'':>5s} │ {'':>5s} │ {'':>30s} │")
    print(f"  └{'─'*137}")

# ═══════════════════════════════════════════════════════════════
# TABLA COMPARATIVA: mismo regimen, 3 modos lado a lado
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*140}")
print(f"  COMPARATIVA POR REGIMEN: PnL E2 por modo (EUR)")
print(f"{'='*140}")
print(f"\n  {'Regimen':>14s} │ {'Orig Sem':>8s} │ {'Orig PnL':>10s} │ {'Orig €/s':>8s} │ {'Hyb Sem':>8s} │ {'Hyb PnL':>10s} │ {'Hyb €/s':>8s} │ {'MinDD Sem':>9s} │ {'MinDD PnL':>10s} │ {'MinDD €/s':>9s} │ {'Max Diff':>10s}")
print(f"  {'-'*14} │ {'-'*8} │ {'-'*10} │ {'-'*8} │ {'-'*8} │ {'-'*10} │ {'-'*8} │ {'-'*9} │ {'-'*10} │ {'-'*9} │ {'-'*10}")

total_pnls = [0, 0, 0]
for reg in REGIME_ORDER:
    vals = []
    for mi in range(3):
        rs = all_mode_stats[mi][reg]
        pnl = rs['pnl_long'] + rs['pnl_short']
        per_wk = pnl / rs['n'] if rs['n'] > 0 else 0
        vals.append((rs['n'], pnl, per_wk))
        total_pnls[mi] += pnl

    pnls = [v[1] for v in vals]
    max_diff = max(pnls) - min(pnls)

    # Flag if max_diff is large relative to average
    avg_pnl = np.mean([abs(p) for p in pnls])
    flag = ' <<<' if max_diff > 50000 else ''

    print(f"  {reg:>14s} │ {vals[0][0]:>8d} │ {vals[0][1]:>+10,.0f} │ {vals[0][2]:>+8,.0f} │ {vals[1][0]:>8d} │ {vals[1][1]:>+10,.0f} │ {vals[1][2]:>+8,.0f} │ {vals[2][0]:>9d} │ {vals[2][1]:>+10,.0f} │ {vals[2][2]:>+9,.0f} │ {max_diff:>+10,.0f}{flag}")

print(f"\n  {'TOTAL':>14s} │ {'':>8s} │ {total_pnls[0]:>+10,.0f} │ {'':>8s} │ {'':>8s} │ {total_pnls[1]:>+10,.0f} │ {'':>8s} │ {'':>9s} │ {total_pnls[2]:>+10,.0f} │ {'':>9s} │ {max(total_pnls)-min(total_pnls):>+10,.0f}")

# ═══════════════════════════════════════════════════════════════
# RETORNO MEDIO SPY POR REGIMEN (para ver si el regimen discrimina)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*140}")
print(f"  RETORNO MEDIO SPY POR REGIMEN x MODO (para validar discriminación)")
print(f"{'='*140}")

# Need SPY returns - load from regimenes CSV
spy_by_date = {}
with open('data/regimenes_historico.csv') as f:
    for row in csv.DictReader(f):
        if 'spy_ret_pct' in row and row['spy_ret_pct']:
            spy_by_date[row['fecha_senal'][:10]] = float(row['spy_ret_pct'])

print(f"\n  {'Regimen':>14s} │ {'Orig':>5s} │ {'Orig SPY%':>10s} │ {'Hyb':>5s} │ {'Hyb SPY%':>10s} │ {'MinDD':>5s} │ {'MinDD SPY%':>10s} │ {'Rango SPY%':>10s} │ Nota")
print(f"  {'-'*14} │ {'-'*5} │ {'-'*10} │ {'-'*5} │ {'-'*10} │ {'-'*5} │ {'-'*10} │ {'-'*10} │ {'-'*30}")

for reg in REGIME_ORDER:
    spy_vals = [[], [], []]
    for mi, csv_path in enumerate(MODE_CSVS):
        entries = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if int(row['year']) < START_YEAR: continue
                if row['regime'] == reg:
                    d = row['fecha_senal'][:10]
                    if d in spy_by_date:
                        spy_vals[mi].append(spy_by_date[d])

    avgs = [np.mean(v) if v else 0 for v in spy_vals]
    ns = [len(v) for v in spy_vals]
    rango = max(avgs) - min(avgs)

    # Note for overfitting
    note = ''
    if rango > 0.3:
        note = 'ALTA VARIANZA entre modos'
    elif rango > 0.15:
        note = 'varianza moderada'

    print(f"  {reg:>14s} │ {ns[0]:>5d} │ {avgs[0]:>+9.3f}% │ {ns[1]:>5d} │ {avgs[1]:>+9.3f}% │ {ns[2]:>5d} │ {avgs[2]:>+9.3f}% │ {rango:>9.3f}% │ {note}")

# ═══════════════════════════════════════════════════════════════
# CONCLUSION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*140}")
print(f"  DIAGNOSTICO DE OVERFITTING")
print(f"{'='*140}")

print(f"\n  Regimenes con MaxDiff E2 > EUR 50K entre modos = OVERFITTING:")
for reg in REGIME_ORDER:
    pnls = []
    for mi in range(3):
        rs = all_mode_stats[mi][reg]
        pnls.append(rs['pnl_long'] + rs['pnl_short'])
    diff = max(pnls) - min(pnls)
    if diff > 50000:
        best_mi = pnls.index(max(pnls))
        worst_mi = pnls.index(min(pnls))
        print(f"    {reg:>14s}: Diff EUR {diff:>+10,.0f}  (mejor: {MODE_LABELS[best_mi]} {max(pnls):>+10,.0f} / peor: {MODE_LABELS[worst_mi]} {min(pnls):>+10,.0f})")

print(f"\n  Regimenes ESTABLES (MaxDiff < EUR 20K):")
for reg in REGIME_ORDER:
    pnls = []
    for mi in range(3):
        rs = all_mode_stats[mi][reg]
        pnls.append(rs['pnl_long'] + rs['pnl_short'])
    diff = max(pnls) - min(pnls)
    if diff < 20000:
        print(f"    {reg:>14s}: Diff EUR {diff:>+10,.0f}  ({', '.join(f'{MODE_LABELS[mi]}:{p:>+,.0f}' for mi, p in enumerate(pnls))})")
