"""
Analisis de overfitting: 3 modos de regimen
- Distribucion de semanas por regimen
- Transiciones entre modos (que cambia y cuando)
- Correlaciones de clasificacion
- Impacto en PnL por cada cambio de regimen
"""
import csv, sys, io
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MODE_LABELS = ['Original', 'Hybrid', 'Min DD']
MODE_CSVS = ['data/regimenes_historico.csv', 'data/regimenes_hybrid.csv', 'data/regimenes_mindd.csv']
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
                'RECOVERY','CRISIS','PANICO','CAPITULACION']

# Load all 3 CSVs
modes = []
for csv_path in MODE_CSVS:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'fecha': r['fecha_senal'][:10],
                'year': int(r['year']),
                'regime': r['regime'],
                'total': float(r['total']),
                'spy_ret': float(r['spy_ret_pct']) if 'spy_ret_pct' in r and r['spy_ret_pct'] else 0,
            })
    modes.append(rows)

# Align by fecha_senal
dates_common = set(r['fecha'] for r in modes[0])
for m in modes[1:]:
    dates_common &= set(r['fecha'] for r in m)
dates_common = sorted(dates_common)

# Build aligned dicts
aligned = []
for d in dates_common:
    row = {'fecha': d}
    for mi in range(3):
        for r in modes[mi]:
            if r['fecha'] == d:
                row[f'r{mi}'] = r['regime']
                row[f's{mi}'] = r['total']
                row[f'y'] = r['year']
                row[f'spy'] = r['spy_ret']
                break
    if all(f'r{mi}' in row for mi in range(3)):
        aligned.append(row)

print(f"Semanas alineadas: {len(aligned)}")
START = 2005
aligned = [a for a in aligned if a['y'] >= START]
print(f"Desde {START}: {len(aligned)}")

# ═══════════════════════════════════════════════════════════════
# 1. DISTRIBUCION DE SEMANAS POR REGIMEN
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  1. DISTRIBUCION DE SEMANAS POR REGIMEN")
print(f"{'='*110}")
print(f"\n  {'Regimen':>15s}  {'Original':>10s}  {'Hybrid':>10s}  {'MinDD':>10s}  {'Diff O-H':>10s}  {'Diff O-M':>10s}  {'Diff H-M':>10s}")
print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

counts = [{}, {}, {}]
for mi in range(3):
    for reg in REGIME_ORDER:
        counts[mi][reg] = sum(1 for a in aligned if a[f'r{mi}'] == reg)

for reg in REGIME_ORDER:
    o, h, m = counts[0][reg], counts[1][reg], counts[2][reg]
    print(f"  {reg:>15s}  {o:>7d} ({o/len(aligned)*100:4.1f}%)  {h:>7d} ({h/len(aligned)*100:4.1f}%)  {m:>7d} ({m/len(aligned)*100:4.1f}%)  {o-h:>+10d}  {o-m:>+10d}  {h-m:>+10d}")

# ═══════════════════════════════════════════════════════════════
# 2. ACUERDO ENTRE MODOS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  2. ACUERDO ENTRE MODOS")
print(f"{'='*110}")

agree_oh = sum(1 for a in aligned if a['r0'] == a['r1'])
agree_om = sum(1 for a in aligned if a['r0'] == a['r2'])
agree_hm = sum(1 for a in aligned if a['r1'] == a['r2'])
agree_all = sum(1 for a in aligned if a['r0'] == a['r1'] == a['r2'])
n = len(aligned)

print(f"\n  Original = Hybrid:  {agree_oh:>5d} / {n} = {agree_oh/n*100:.1f}%")
print(f"  Original = MinDD:   {agree_om:>5d} / {n} = {agree_om/n*100:.1f}%")
print(f"  Hybrid   = MinDD:   {agree_hm:>5d} / {n} = {agree_hm/n*100:.1f}%")
print(f"  Los 3 iguales:      {agree_all:>5d} / {n} = {agree_all/n*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 3. MATRIZ DE CONFUSION: Original vs Hybrid
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  3. MATRIZ DE CONFUSION: Original vs Hybrid (filas=Original, cols=Hybrid)")
print(f"{'='*110}")

conf_oh = defaultdict(lambda: defaultdict(int))
for a in aligned:
    conf_oh[a['r0']][a['r1']] += 1

# Only show regimes with differences
diff_regs = [r for r in REGIME_ORDER if any(conf_oh[r][r2] > 0 and r != r2 for r2 in REGIME_ORDER)]
show_regs = REGIME_ORDER  # show all

print(f"\n  {'':>15s}", end='')
for r2 in show_regs:
    print(f"  {r2[:6]:>6s}", end='')
print()
print(f"  {'-'*15}", end='')
for _ in show_regs:
    print(f"  {'-'*6}", end='')
print()

for r1 in show_regs:
    total_r1 = sum(conf_oh[r1][r2] for r2 in show_regs)
    if total_r1 == 0: continue
    print(f"  {r1:>15s}", end='')
    for r2 in show_regs:
        v = conf_oh[r1][r2]
        if v > 0:
            mark = ' *' if r1 == r2 else '  '
            print(f"  {v:>4d}{mark}", end='')
        else:
            print(f"  {'':>6s}", end='')
    print(f"  | {total_r1}")

# ═══════════════════════════════════════════════════════════════
# 4. MATRIZ DE CONFUSION: Original vs MinDD
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  4. MATRIZ DE CONFUSION: Original vs MinDD (filas=Original, cols=MinDD)")
print(f"{'='*110}")

conf_om = defaultdict(lambda: defaultdict(int))
for a in aligned:
    conf_om[a['r0']][a['r2']] += 1

print(f"\n  {'':>15s}", end='')
for r2 in show_regs:
    print(f"  {r2[:6]:>6s}", end='')
print()
print(f"  {'-'*15}", end='')
for _ in show_regs:
    print(f"  {'-'*6}", end='')
print()

for r1 in show_regs:
    total_r1 = sum(conf_om[r1][r2] for r2 in show_regs)
    if total_r1 == 0: continue
    print(f"  {r1:>15s}", end='')
    for r2 in show_regs:
        v = conf_om[r1][r2]
        if v > 0:
            mark = ' *' if r1 == r2 else '  '
            print(f"  {v:>4d}{mark}", end='')
        else:
            print(f"  {'':>6s}", end='')
    print(f"  | {total_r1}")

# ═══════════════════════════════════════════════════════════════
# 5. SEMANAS DONDE DIFIEREN - POR AÑO
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  5. SEMANAS DONDE ORIGINAL != HYBRID - POR AÑO")
print(f"{'='*110}")

diffs_by_year = defaultdict(list)
for a in aligned:
    if a['r0'] != a['r1']:
        diffs_by_year[a['y']].append(a)

print(f"\n  {'Año':>5s}  {'Diff':>5s}  {'Total':>5s}  {'%Diff':>6s}  {'Transiciones más frecuentes':>50s}")
print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*50}")

for yr in sorted(diffs_by_year.keys()):
    diffs = diffs_by_year[yr]
    total_yr = sum(1 for a in aligned if a['y'] == yr)
    # Count transitions
    trans = Counter()
    for d in diffs:
        trans[f"{d['r0']}→{d['r1']}"] += 1
    top3 = ', '.join(f"{t}({n})" for t, n in trans.most_common(3))
    print(f"  {yr:>5d}  {len(diffs):>5d}  {total_yr:>5d}  {len(diffs)/total_yr*100:>5.1f}%  {top3}")

# ═══════════════════════════════════════════════════════════════
# 6. CORRELACION DE SCORES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  6. CORRELACION DE SCORES TOTALES")
print(f"{'='*110}")

s0 = np.array([a['s0'] for a in aligned])
s1 = np.array([a['s1'] for a in aligned])
s2 = np.array([a['s2'] for a in aligned])

print(f"\n  Score medio:  Orig {np.mean(s0):+.2f} | Hybrid {np.mean(s1):+.2f} | MinDD {np.mean(s2):+.2f}")
print(f"  Score std:    Orig {np.std(s0):.2f}  | Hybrid {np.std(s1):.2f}  | MinDD {np.std(s2):.2f}")
print(f"  Score min:    Orig {np.min(s0):+.1f} | Hybrid {np.min(s1):+.1f} | MinDD {np.min(s2):+.1f}")
print(f"  Score max:    Orig {np.max(s0):+.1f} | Hybrid {np.max(s1):+.1f} | MinDD {np.max(s2):+.1f}")

corr_01 = np.corrcoef(s0, s1)[0,1]
corr_02 = np.corrcoef(s0, s2)[0,1]
corr_12 = np.corrcoef(s1, s2)[0,1]
print(f"\n  Corr Orig-Hybrid: {corr_01:.4f}")
print(f"  Corr Orig-MinDD:  {corr_02:.4f}")
print(f"  Corr Hybrid-MinDD:{corr_12:.4f}")

# Score difference stats
diff_oh = s0 - s1
diff_om = s0 - s2
print(f"\n  Diff score Orig-Hybrid: mean {np.mean(diff_oh):+.3f}, std {np.std(diff_oh):.3f}, max {np.max(np.abs(diff_oh)):.1f}")
print(f"  Diff score Orig-MinDD:  mean {np.mean(diff_om):+.3f}, std {np.std(diff_om):.3f}, max {np.max(np.abs(diff_om)):.1f}")

# ═══════════════════════════════════════════════════════════════
# 7. SENSIBILIDAD: semanas donde 1 punto de score cambia regimen
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  7. SENSIBILIDAD: semanas cerca de umbrales (score ±0.5 del umbral)")
print(f"{'='*110}")

THRESHOLDS = [8.0, 7.0, 4.0, 0.5, -2.0, -5.0, -9.0]
THRESH_NAMES = ['BURBUJA/GOLD', 'GOLD/ALCISTA', 'ALCISTA/NEUTRAL', 'NEUTRAL/CAUTIOUS',
                'CAUTIOUS/BEARISH', 'BEARISH/CRISIS', 'CRISIS/PANICO']

for mi, label in enumerate(MODE_LABELS):
    scores = [a[f's{mi}'] for a in aligned]
    print(f"\n  {label}:")
    print(f"    {'Umbral':>20s}  {'Valor':>6s}  {'±0.5':>6s}  {'±1.0':>6s}  {'±1.5':>6s}")
    for th, name in zip(THRESHOLDS, THRESH_NAMES):
        n05 = sum(1 for s in scores if abs(s - th) <= 0.5)
        n10 = sum(1 for s in scores if abs(s - th) <= 1.0)
        n15 = sum(1 for s in scores if abs(s - th) <= 1.5)
        print(f"    {name:>20s}  {th:>+5.1f}  {n05:>5d}  {n10:>5d}  {n15:>5d}")

# ═══════════════════════════════════════════════════════════════
# 8. OVERFITTING TEST: retorno SPY futuro por regimen
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  8. TEST DE OVERFITTING: retorno SPY medio por regimen (debe discriminar bien)")
print(f"{'='*110}")

# Group regimes into bullish/bearish
BULLISH = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA'}
BEARISH_SET = {'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO'}
NEUTRAL_SET = {'NEUTRAL', 'RECOVERY', 'CAPITULACION'}

print(f"\n  {'Regimen':>15s}  {'Orig SPY%':>10s}  {'Hyb SPY%':>10s}  {'MinDD SPY%':>10s}  {'Orig N':>7s}  {'Hyb N':>7s}  {'MinDD N':>7s}")
print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}")

for reg in REGIME_ORDER:
    vals = []
    for mi in range(3):
        sub = [a['spy'] for a in aligned if a[f'r{mi}'] == reg]
        if len(sub) > 0:
            vals.append((np.mean(sub), len(sub)))
        else:
            vals.append((0, 0))
    print(f"  {reg:>15s}  {vals[0][0]:>+9.3f}%  {vals[1][0]:>+9.3f}%  {vals[2][0]:>+9.3f}%  {vals[0][1]:>7d}  {vals[1][1]:>7d}  {vals[2][1]:>7d}")

# Discrimination power: avg return in bullish regimes vs bearish regimes
print(f"\n  PODER DISCRIMINATORIO (avg SPY ret):")
for mi, label in enumerate(MODE_LABELS):
    bull_ret = [a['spy'] for a in aligned if a[f'r{mi}'] in BULLISH]
    bear_ret = [a['spy'] for a in aligned if a[f'r{mi}'] in BEARISH_SET]
    neut_ret = [a['spy'] for a in aligned if a[f'r{mi}'] in NEUTRAL_SET]
    spread = np.mean(bull_ret) - np.mean(bear_ret)
    print(f"    {label:10s}: Bull {np.mean(bull_ret):+.3f}% ({len(bull_ret)}) | Bear {np.mean(bear_ret):+.3f}% ({len(bear_ret)}) | Neutral {np.mean(neut_ret):+.3f}% ({len(neut_ret)}) | Spread: {spread:+.3f}%")

# ═══════════════════════════════════════════════════════════════
# 9. ESTABILIDAD TEMPORAL: poder discriminatorio por periodo
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  9. ESTABILIDAD TEMPORAL: spread Bull-Bear por periodo de 5 años")
print(f"{'='*110}")

periods = [(2005,2009), (2010,2014), (2015,2019), (2020,2026)]
print(f"\n  {'Periodo':>12s}", end='')
for label in MODE_LABELS:
    print(f"  {'Spread '+label:>18s}", end='')
print()
print(f"  {'-'*12}", end='')
for _ in MODE_LABELS:
    print(f"  {'-'*18}", end='')
print()

for y1, y2 in periods:
    sub = [a for a in aligned if y1 <= a['y'] <= y2]
    print(f"  {y1}-{y2:>4d}   ", end='')
    for mi in range(3):
        bull = [a['spy'] for a in sub if a[f'r{mi}'] in BULLISH]
        bear = [a['spy'] for a in sub if a[f'r{mi}'] in BEARISH_SET]
        if len(bull) > 0 and len(bear) > 0:
            spread = np.mean(bull) - np.mean(bear)
            print(f"  {spread:>+.3f}% ({len(bull):3d}B/{len(bear):3d}b)", end='')
        else:
            print(f"  {'N/A':>18s}", end='')
    print()

# ═══════════════════════════════════════════════════════════════
# 10. CONCLUSION OVERFITTING
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  10. RESUMEN Y CONCLUSION")
print(f"{'='*110}")

# Varianza del spread por periodo
for mi, label in enumerate(MODE_LABELS):
    spreads = []
    for y1, y2 in periods:
        sub = [a for a in aligned if y1 <= a['y'] <= y2]
        bull = [a['spy'] for a in sub if a[f'r{mi}'] in BULLISH]
        bear = [a['spy'] for a in sub if a[f'r{mi}'] in BEARISH_SET]
        if len(bull) > 0 and len(bear) > 0:
            spreads.append(np.mean(bull) - np.mean(bear))
    cv = np.std(spreads) / abs(np.mean(spreads)) * 100 if np.mean(spreads) != 0 else 999
    print(f"\n  {label}:")
    print(f"    Spreads por periodo: {', '.join(f'{s:+.3f}%' for s in spreads)}")
    print(f"    Media spread: {np.mean(spreads):+.3f}%, Std: {np.std(spreads):.3f}%, CV: {cv:.0f}%")
    print(f"    Acuerdo con otros: O-H={agree_oh/n*100:.0f}%, O-M={agree_om/n*100:.0f}%, H-M={agree_hm/n*100:.0f}%")

print(f"\n  SEÑALES DE OVERFITTING:")
print(f"  - CV alto = spread inconsistente entre periodos = posible overfitting")
print(f"  - Diferencia grande entre modos en años especificos = sensibilidad a umbrales")
print(f"  - Menor acuerdo entre modos = mayor zona gris en clasificación")
