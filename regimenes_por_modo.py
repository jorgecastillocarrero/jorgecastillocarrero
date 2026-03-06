"""Comparativa de regimenes SPY por modo (Original, Hybrid, MinDD)"""
import csv, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

modes = [
    ('Original', 'data/regimenes_historico.csv'),
    ('Hybrid',   'data/regimenes_hybrid.csv'),
    ('MinDD',    'data/regimenes_mindd.csv'),
]

REGS = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
        'RECOVERY','CRISIS','PANICO','CAPITULACION']

# Load data per mode
mode_data = {}
for label, path in modes:
    by_reg = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            reg = row['regime']
            ret_str = row.get('spy_ret_pct', '')
            try:
                ret = float(ret_str)
            except:
                ret = None
            if reg not in by_reg:
                by_reg[reg] = []
            by_reg[reg].append(ret)
    mode_data[label] = by_reg

def fmt(rets):
    n = len(rets)
    if n == 0:
        return '                                '
    avg = np.mean(rets)
    wr = sum(1 for r in rets if r > 0) / n * 100
    tot = sum(rets)
    std = np.std(rets)
    return f'{n:>4d} {avg:>+6.2f} {wr:>5.1f}% {tot:>+7.1f} {std:>5.2f}'

print(f'              |  -------- Original --------  |  --------- Hybrid ---------  |  ---------- MinDD ----------')
print(f'Regimen       |    N   Avg%    WR%    Tot%  Std |    N   Avg%    WR%    Tot%  Std |    N   Avg%    WR%    Tot%  Std')
print(f'{"="*110}')

for reg in REGS:
    parts = []
    for label in ['Original', 'Hybrid', 'MinDD']:
        rets = [r for r in mode_data[label].get(reg, []) if r is not None]
        parts.append(fmt(rets))
    print(f'{reg:<14s}| {parts[0]} | {parts[1]} | {parts[2]}')

# Totals
print(f'{"="*110}')
parts = []
for label in ['Original', 'Hybrid', 'MinDD']:
    all_rets = []
    for reg in REGS:
        all_rets.extend([r for r in mode_data[label].get(reg, []) if r is not None])
    parts.append(fmt(all_rets))
print(f'{"TOTAL":<14s}| {parts[0]} | {parts[1]} | {parts[2]}')

# Media de los 3 modos por regimen
print(f'\n\n{"="*80}')
print(f'  MEDIA DE LOS 3 MODOS POR REGIMEN')
print(f'{"="*80}')
print(f'{"Regimen":<14s}  {"N med":>5s}  {"Avg%":>6s}  {"WR%":>5s}  {"Tot%":>7s}  {"Std":>5s}  {"Discrimina?":>12s}')
print(f'{"-"*14}  {"-"*5}  {"-"*6}  {"-"*5}  {"-"*7}  {"-"*5}  {"-"*12}')

for reg in REGS:
    avgs, wrs, tots, ns = [], [], [], []
    for label in ['Original', 'Hybrid', 'MinDD']:
        rets = [r for r in mode_data[label].get(reg, []) if r is not None]
        if len(rets) > 0:
            ns.append(len(rets))
            avgs.append(np.mean(rets))
            wrs.append(sum(1 for r in rets if r > 0) / len(rets) * 100)
            tots.append(sum(rets))
    if not ns:
        continue
    avg_n = np.mean(ns)
    avg_avg = np.mean(avgs)
    avg_wr = np.mean(wrs)
    avg_tot = np.mean(tots)

    # Consistency: all 3 same sign?
    all_pos = all(a > 0 for a in avgs)
    all_neg = all(a < 0 for a in avgs)
    if all_pos:
        disc = 'ALCISTA 3/3'
    elif all_neg:
        disc = 'BAJISTA 3/3'
    else:
        disc = 'MIXTO'

    print(f'{reg:<14s}  {avg_n:>5.0f}  {avg_avg:>+6.2f}  {avg_wr:>4.1f}%  {avg_tot:>+7.1f}  {"":>5s}  {disc}')
