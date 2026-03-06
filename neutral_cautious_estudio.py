"""
Estudio detallado NEUTRAL vs CAUTIOUS para homogeneizacion
- Distribucion de scores
- Rendimiento por buckets de score
- Frontera optima
- Analisis por los 3 modos
"""
import csv, sys, io
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

modes = [
    ('Original', 'data/regimenes_historico.csv'),
    ('Hybrid',   'data/regimenes_hybrid.csv'),
    ('MinDD',    'data/regimenes_mindd.csv'),
]

# Load all data with scores
mode_rows = {}
for label, path in modes:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                row['_ret'] = float(row['spy_ret_pct'])
            except:
                row['_ret'] = None
            try:
                row['_score'] = float(row['total'])
            except:
                row['_score'] = None
            rows.append(row)
    mode_rows[label] = rows

# ═══════════════════════════════════════════════════════════════
# 1. DISTRIBUCION DE SCORES EN NEUTRAL Y CAUTIOUS
# ═══════════════════════════════════════════════════════════════
print('='*100)
print('  1. DISTRIBUCION DE SCORES EN NEUTRAL Y CAUTIOUS')
print('='*100)

for label in ['Original', 'Hybrid', 'MinDD']:
    print(f'\n  {label}:')
    for reg in ['NEUTRAL', 'CAUTIOUS']:
        scores = [r['_score'] for r in mode_rows[label] if r['regime'] == reg and r['_score'] is not None]
        if scores:
            print(f'    {reg:<10s}: N={len(scores):>4d}  Score: min={min(scores):+.1f}  max={max(scores):+.1f}  avg={np.mean(scores):+.1f}  med={np.median(scores):+.1f}')
            # Histogram
            buckets = defaultdict(int)
            for s in scores:
                b = round(s * 2) / 2  # bucket by 0.5
                buckets[b] += 1
            for b in sorted(buckets.keys()):
                bar = '#' * buckets[b]
                print(f'      Score {b:+5.1f}: {buckets[b]:>4d} {bar}')

# ═══════════════════════════════════════════════════════════════
# 2. RENDIMIENTO SPY POR BUCKET DE SCORE (zona NEUTRAL-CAUTIOUS)
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"="*100}')
print(f'  2. RENDIMIENTO SPY POR BUCKET DE SCORE (todos los modos combinados)')
print(f'{"="*100}')

# Combine all 3 modes
all_data = []
for label in ['Original', 'Hybrid', 'MinDD']:
    for r in mode_rows[label]:
        if r['_score'] is not None and r['_ret'] is not None:
            all_data.append((r['_score'], r['_ret'], r['regime'], label))

# Buckets from -3.0 to +5.0 (CAUTIOUS to ALCISTA range)
print(f'\n  {"Score":>6s}  {"Regime":>10s}  {"N":>5s}  {"Avg%":>7s}  {"Med%":>7s}  {"WR%":>5s}  {"Tot%":>8s}  {"Std%":>6s}  {"Visual"}')
print(f'  {"-"*6}  {"-"*10}  {"-"*5}  {"-"*7}  {"-"*7}  {"-"*5}  {"-"*8}  {"-"*6}  {"-"*30}')

for score_bucket in np.arange(-3.0, 5.5, 0.5):
    bucket_rets = [ret for sc, ret, reg, lbl in all_data
                   if abs(sc - score_bucket) < 0.01]
    if not bucket_rets:
        continue

    # Determine regime for this score
    if score_bucket >= 4.0:
        reg_label = 'ALCISTA'
    elif score_bucket >= 0.5:
        reg_label = 'NEUTRAL'
    elif score_bucket >= -2.0:
        reg_label = 'CAUTIOUS'
    else:
        reg_label = 'BEARISH'

    avg = np.mean(bucket_rets)
    med = np.median(bucket_rets)
    wr = sum(1 for r in bucket_rets if r > 0) / len(bucket_rets) * 100
    tot = sum(bucket_rets)
    std = np.std(bucket_rets)

    # Visual bar
    bar_len = int(avg / 0.05) if avg > 0 else int(avg / 0.05)
    if avg >= 0:
        bar = ' ' * 10 + '|' + '+' * min(bar_len, 20)
    else:
        padding = max(0, 10 + bar_len)
        bar = ' ' * padding + '-' * min(abs(bar_len), 10) + '|'

    marker = ' <<<' if score_bucket == 0.5 else ''  # frontera
    print(f'  {score_bucket:>+5.1f}  {reg_label:>10s}  {len(bucket_rets):>5d}  {avg:>+7.3f}  {med:>+7.3f}  {wr:>4.1f}%  {tot:>+8.1f}  {std:>6.2f}  {bar}{marker}')

# ═══════════════════════════════════════════════════════════════
# 3. RENDIMIENTO POR MODO Y BUCKET (solo zona critica -2.0 a +4.0)
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"="*100}')
print(f'  3. RENDIMIENTO POR MODO - ZONA NEUTRAL/CAUTIOUS (score -2.0 a +4.0)')
print(f'{"="*100}')

for label in ['Original', 'Hybrid', 'MinDD']:
    print(f'\n  {label}:')
    print(f'  {"Score":>6s}  {"Regime":>10s}  {"N":>4s}  {"Avg%":>7s}  {"WR%":>5s}  {"Tot%":>7s}')
    print(f'  {"-"*6}  {"-"*10}  {"-"*4}  {"-"*7}  {"-"*5}  {"-"*7}')

    for score_bucket in np.arange(-2.0, 4.5, 0.5):
        rets = [r['_ret'] for r in mode_rows[label]
                if r['_score'] is not None and r['_ret'] is not None
                and abs(r['_score'] - score_bucket) < 0.01]
        if not rets:
            continue

        if score_bucket >= 4.0:
            reg_label = 'ALCISTA'
        elif score_bucket >= 0.5:
            reg_label = 'NEUTRAL'
        elif score_bucket >= -2.0:
            reg_label = 'CAUTIOUS'
        else:
            reg_label = 'BEARISH'

        avg = np.mean(rets)
        wr = sum(1 for r in rets if r > 0) / len(rets) * 100
        tot = sum(rets)

        sep = '  <-- frontera' if score_bucket == 0.5 else ''
        print(f'  {score_bucket:>+5.1f}  {reg_label:>10s}  {len(rets):>4d}  {avg:>+7.3f}  {wr:>4.1f}%  {tot:>+7.1f}{sep}')

# ═══════════════════════════════════════════════════════════════
# 4. ANALISIS DE FRONTERAS ALTERNATIVAS
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"="*100}')
print(f'  4. FRONTERAS ALTERNATIVAS: si movemos el corte NEUTRAL/CAUTIOUS')
print(f'{"="*100}')

print(f'\n  Actual: NEUTRAL >= 0.5, CAUTIOUS >= -2.0')
print(f'\n  {"Frontera":>9s}  |  {"--- NEUTRAL (>= frontera) ---":>35s}  |  {"--- CAUTIOUS (< frontera) ---":>35s}')
print(f'  {"-"*9}  |  {"N":>4s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s} {"Std":>5s}  |  {"N":>4s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s} {"Std":>5s}')

for frontier in np.arange(-1.5, 2.5, 0.5):
    # Combined 3 modes
    neu_rets = []
    cau_rets = []
    for label in ['Original', 'Hybrid', 'MinDD']:
        for r in mode_rows[label]:
            if r['_score'] is None or r['_ret'] is None:
                continue
            sc = r['_score']
            # Only in the NEUTRAL-CAUTIOUS zone (exclude ALCISTA and BEARISH)
            if sc >= 4.0 or sc < -2.0:
                continue
            if sc >= frontier:
                neu_rets.append(r['_ret'])
            else:
                cau_rets.append(r['_ret'])

    if not neu_rets or not cau_rets:
        continue

    n_avg = np.mean(neu_rets)
    n_wr = sum(1 for r in neu_rets if r > 0) / len(neu_rets) * 100
    n_tot = sum(neu_rets)
    n_std = np.std(neu_rets)

    c_avg = np.mean(cau_rets)
    c_wr = sum(1 for r in cau_rets if r > 0) / len(cau_rets) * 100
    c_tot = sum(cau_rets)
    c_std = np.std(cau_rets)

    diff = n_avg - c_avg
    marker = ' <<<' if frontier == 0.5 else ''

    print(f'  Score {frontier:>+4.1f}  |  {len(neu_rets):>4d} {n_avg:>+7.3f} {n_wr:>4.1f}% {n_tot:>+7.1f} {n_std:>5.2f}  |  {len(cau_rets):>4d} {c_avg:>+7.3f} {c_wr:>4.1f}% {c_tot:>+7.1f} {c_std:>5.2f}  diff={diff:>+.3f}{marker}')

# ═══════════════════════════════════════════════════════════════
# 5. PROPUESTA: JUNTAR O SEPARAR?
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"="*100}')
print(f'  5. NEUTRAL + CAUTIOUS JUNTOS vs SEPARADOS')
print(f'{"="*100}')

for label in ['Original', 'Hybrid', 'MinDD']:
    neu = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'NEUTRAL' and r['_ret'] is not None]
    cau = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'CAUTIOUS' and r['_ret'] is not None]
    juntos = neu + cau

    print(f'\n  {label}:')
    print(f'    NEUTRAL:          N={len(neu):>4d}  Avg={np.mean(neu):>+.3f}%  WR={sum(1 for r in neu if r>0)/len(neu)*100:.1f}%  Tot={sum(neu):>+.1f}%')
    print(f'    CAUTIOUS:         N={len(cau):>4d}  Avg={np.mean(cau):>+.3f}%  WR={sum(1 for r in cau if r>0)/len(cau)*100:.1f}%  Tot={sum(cau):>+.1f}%')
    print(f'    JUNTOS (N+C):     N={len(juntos):>4d}  Avg={np.mean(juntos):>+.3f}%  WR={sum(1 for r in juntos if r>0)/len(juntos)*100:.1f}%  Tot={sum(juntos):>+.1f}%')
    print(f'    Diferencia N-C:   {np.mean(neu)-np.mean(cau):>+.3f}% por semana')
