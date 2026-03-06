"""
Estudio frontera CAUTIOUS vs BEARISH
- Semanas que cambian de regimen entre modos
- Scores cerca de la frontera (-2.0)
- Indicadores medios
- Fronteras alternativas
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

# Load by date
by_date = {}
for label, path in modes:
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row['fecha_senal'][:10]
            if d not in by_date:
                by_date[d] = {}
            try:
                ret = float(row['spy_ret_pct'])
            except:
                ret = None
            try:
                score = float(row['total'])
            except:
                score = None
            by_date[d][label] = {
                'regime': row['regime'],
                'score': score,
                'ret': ret,
                'pct_dd_h': float(row.get('pct_dd_h', 0)),
                'pct_rsi': float(row.get('pct_rsi', 0)),
                'spy_dist': float(row.get('spy_dist', 0)),
                'spy_mom': float(row.get('spy_mom', 0)),
                'vix': float(row.get('vix', 0)),
            }

# ===================================================================
# 1. SEMANAS DONDE UN MODO DICE CAUTIOUS Y OTRO BEARISH
# ===================================================================
print('='*120)
print('  1. SEMANAS DONDE UN MODO DICE CAUTIOUS Y OTRO BEARISH')
print('='*120)

discrepantes = []
for d in sorted(by_date.keys()):
    data = by_date[d]
    if len(data) < 3:
        continue
    regs = [data[m]['regime'] for m in ['Original', 'Hybrid', 'MinDD']]
    has_cau = 'CAUTIOUS' in regs
    has_bear = 'BEARISH' in regs
    if has_cau and has_bear:
        discrepantes.append(d)

print(f'\n  Total semanas con discrepancia C/B: {len(discrepantes)}')

print(f'\n  {"Fecha":>10s}  {"Orig Reg":>10s} {"Sc":>5s}  {"Hyb Reg":>10s} {"Sc":>5s}  {"MinDD Reg":>10s} {"Sc":>5s}  {"SPY Ret%":>8s}')
print(f'  {"-"*10}  {"-"*10} {"-"*5}  {"-"*10} {"-"*5}  {"-"*10} {"-"*5}  {"-"*8}')

rets_disc = []
for d in discrepantes:
    data = by_date[d]
    o = data.get('Original', {})
    h = data.get('Hybrid', {})
    m = data.get('MinDD', {})
    ret = o.get('ret')
    if ret is not None:
        rets_disc.append(ret)
    ret_str = f'{ret:>+8.2f}' if ret is not None else '     ---'

    os = f"{o.get('score', 0):>+5.1f}" if o.get('score') is not None else '    ?'
    hs = f"{h.get('score', 0):>+5.1f}" if h.get('score') is not None else '    ?'
    ms = f"{m.get('score', 0):>+5.1f}" if m.get('score') is not None else '    ?'

    print(f'  {d:>10s}  {o.get("regime","?"):>10s} {os}  {h.get("regime","?"):>10s} {hs}  {m.get("regime","?"):>10s} {ms}  {ret_str}')

if rets_disc:
    print(f'\n  Estas {len(rets_disc)} semanas discrepantes:')
    print(f'    Avg: {np.mean(rets_disc):>+.3f}%  WR: {sum(1 for r in rets_disc if r>0)/len(rets_disc)*100:.1f}%  Tot: {sum(rets_disc):>+.1f}%')

# ===================================================================
# 2. SCORES CERCA DE LA FRONTERA (-2.0)
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  2. SCORES CERCA DE LA FRONTERA: que pasa entre score -3.0 y -1.0')
print(f'{"="*120}')

for label in ['Original', 'Hybrid', 'MinDD']:
    frontera_weeks = []
    for d in sorted(by_date.keys()):
        data = by_date[d]
        if label not in data:
            continue
        sc = data[label].get('score')
        ret = data[label].get('ret')
        reg = data[label].get('regime')
        if sc is not None and -3.5 <= sc <= -0.5 and ret is not None:
            frontera_weeks.append((d, sc, ret, reg))

    print(f'\n  {label} - Semanas en zona frontera:')
    print(f'    {"Score":>6s}  {"Regimen":>10s}  {"N":>4s}  {"Avg%":>7s}  {"WR%":>5s}  {"Tot%":>7s}')
    print(f'    {"-"*6}  {"-"*10}  {"-"*4}  {"-"*7}  {"-"*5}  {"-"*7}')

    for sc_val in np.arange(-3.5, 0.0, 0.5):
        weeks_sc = [(d, r, reg) for d, sc, r, reg in frontera_weeks if abs(sc - sc_val) < 0.01]
        if not weeks_sc:
            continue
        rets_sc = [r for _, r, _ in weeks_sc]
        reg_label = weeks_sc[0][2]
        avg = np.mean(rets_sc)
        wr = sum(1 for r in rets_sc if r > 0) / len(rets_sc) * 100
        tot = sum(rets_sc)
        marker = '  <-- frontera' if abs(sc_val - (-2.0)) < 0.01 else ''
        print(f'    {sc_val:>+5.1f}  {reg_label:>10s}  {len(rets_sc):>4d}  {avg:>+7.3f}  {wr:>4.1f}%  {tot:>+7.1f}{marker}')

# ===================================================================
# 3. QUE INDICADOR CAUSA EL CAMBIO
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  3. QUE DIFERENCIA HAY ENTRE CAUTIOUS Y BEARISH (indicadores medios)')
print(f'{"="*120}')

for label in ['Original', 'Hybrid', 'MinDD']:
    print(f'\n  {label}:')
    print(f'    {"Indicador":>15s}  {"CAUTIOUS":>10s}  {"BEARISH":>10s}  {"Diff":>10s}')
    print(f'    {"-"*15}  {"-"*10}  {"-"*10}  {"-"*10}')

    for ind_name, ind_key in [('DD healthy %', 'pct_dd_h'), ('RSI>55 %', 'pct_rsi'),
                               ('SPY dist %', 'spy_dist'), ('SPY mom %', 'spy_mom'),
                               ('VIX', 'vix'), ('Score', 'score')]:
        c_vals = [by_date[d][label][ind_key] for d in sorted(by_date.keys())
                  if label in by_date[d] and by_date[d][label]['regime'] == 'CAUTIOUS'
                  and by_date[d][label][ind_key] is not None]
        b_vals = [by_date[d][label][ind_key] for d in sorted(by_date.keys())
                  if label in by_date[d] and by_date[d][label]['regime'] == 'BEARISH'
                  and by_date[d][label][ind_key] is not None]

        if c_vals and b_vals:
            c_avg = np.mean(c_vals)
            b_avg = np.mean(b_vals)
            print(f'    {ind_name:>15s}  {c_avg:>+10.2f}  {b_avg:>+10.2f}  {c_avg-b_avg:>+10.2f}')

# ===================================================================
# 4. FRONTERAS ALTERNATIVAS
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  4. FRONTERAS ALTERNATIVAS: si movemos el corte CAUTIOUS/BEARISH')
print(f'{"="*120}')

# Load mode rows for combined analysis
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

print(f'\n  Actual: CAUTIOUS >= -2.0, BEARISH >= -5.0')
print(f'\n  {"Frontera":>9s}  |  {"--- CAUTIOUS (>= frontera) ---":>35s}  |  {"--- BEARISH (< frontera) ---":>35s}')
print(f'  {"-"*9}  |  {"N":>4s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s} {"Std":>5s}  |  {"N":>4s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s} {"Std":>5s}')

for frontier in np.arange(-4.0, -0.5, 0.5):
    cau_rets = []
    bear_rets = []
    for label in ['Original', 'Hybrid', 'MinDD']:
        for r in mode_rows[label]:
            if r['_score'] is None or r['_ret'] is None:
                continue
            sc = r['_score']
            # Only in CAUTIOUS-BEARISH zone (exclude NEUTRAL and CRISIS)
            if sc >= 0.5 or sc < -5.0:
                continue
            if sc >= frontier:
                cau_rets.append(r['_ret'])
            else:
                bear_rets.append(r['_ret'])

    if not cau_rets or not bear_rets:
        continue

    c_avg = np.mean(cau_rets)
    c_wr = sum(1 for r in cau_rets if r > 0) / len(cau_rets) * 100
    c_tot = sum(cau_rets)
    c_std = np.std(cau_rets)

    b_avg = np.mean(bear_rets)
    b_wr = sum(1 for r in bear_rets if r > 0) / len(bear_rets) * 100
    b_tot = sum(bear_rets)
    b_std = np.std(bear_rets)

    diff = c_avg - b_avg
    marker = ' <<<' if abs(frontier - (-2.0)) < 0.01 else ''

    print(f'  Score {frontier:>+4.1f}  |  {len(cau_rets):>4d} {c_avg:>+7.3f} {c_wr:>4.1f}% {c_tot:>+7.1f} {c_std:>5.2f}  |  {len(bear_rets):>4d} {b_avg:>+7.3f} {b_wr:>4.1f}% {b_tot:>+7.1f} {b_std:>5.2f}  diff={diff:>+.3f}{marker}')

# ===================================================================
# 5. CAUTIOUS + BEARISH JUNTOS vs SEPARADOS
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  5. CAUTIOUS + BEARISH JUNTOS vs SEPARADOS')
print(f'{"="*120}')

for label in ['Original', 'Hybrid', 'MinDD']:
    cau = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'CAUTIOUS' and r['_ret'] is not None]
    bear = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'BEARISH' and r['_ret'] is not None]
    juntos = cau + bear

    print(f'\n  {label}:')
    if cau:
        print(f'    CAUTIOUS:         N={len(cau):>4d}  Avg={np.mean(cau):>+.3f}%  WR={sum(1 for r in cau if r>0)/len(cau)*100:.1f}%  Tot={sum(cau):>+.1f}%')
    if bear:
        print(f'    BEARISH:          N={len(bear):>4d}  Avg={np.mean(bear):>+.3f}%  WR={sum(1 for r in bear if r>0)/len(bear)*100:.1f}%  Tot={sum(bear):>+.1f}%')
    if juntos:
        print(f'    JUNTOS (C+B):     N={len(juntos):>4d}  Avg={np.mean(juntos):>+.3f}%  WR={sum(1 for r in juntos if r>0)/len(juntos)*100:.1f}%  Tot={sum(juntos):>+.1f}%')
    if cau and bear:
        print(f'    Diferencia C-B:   {np.mean(cau)-np.mean(bear):>+.3f}% por semana')

# ===================================================================
# 6. POR AÑO: CAUTIOUS vs BEARISH
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  6. POR AÑO: semanas CAUTIOUS vs BEARISH y rendimiento (media 3 modos)')
print(f'{"="*120}')

years_data = defaultdict(lambda: {'ret_cau': [], 'ret_bear': []})

for label in ['Original', 'Hybrid', 'MinDD']:
    for d in sorted(by_date.keys()):
        if label not in by_date[d]:
            continue
        data = by_date[d][label]
        year = int(d[:4])
        ret = data.get('ret')
        if ret is None:
            continue
        if data['regime'] == 'CAUTIOUS':
            years_data[year]['ret_cau'].append(ret)
        elif data['regime'] == 'BEARISH':
            years_data[year]['ret_bear'].append(ret)

print(f'\n  {"Ano":>4s}  {"N_CAU":>5s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s}  {"N_BEAR":>5s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s}  {"Diff":>7s}')
print(f'  {"-"*4}  {"-"*5} {"-"*7} {"-"*5} {"-"*7}  {"-"*5} {"-"*7} {"-"*5} {"-"*7}  {"-"*7}')

for year in sorted(years_data.keys()):
    rc = years_data[year]['ret_cau']
    rb = years_data[year]['ret_bear']

    if rc:
        c_avg = np.mean(rc)
        c_wr = sum(1 for r in rc if r > 0) / len(rc) * 100
        c_tot = sum(rc)
        c_str = f'{len(rc):>5d} {c_avg:>+7.3f} {c_wr:>4.1f}% {c_tot:>+7.1f}'
    else:
        c_str = f'{"":>5s} {"":>7s} {"":>5s} {"":>7s}'
        c_avg = 0

    if rb:
        b_avg = np.mean(rb)
        b_wr = sum(1 for r in rb if r > 0) / len(rb) * 100
        b_tot = sum(rb)
        b_str = f'{len(rb):>5d} {b_avg:>+7.3f} {b_wr:>4.1f}% {b_tot:>+7.1f}'
    else:
        b_str = f'{"":>5s} {"":>7s} {"":>5s} {"":>7s}'
        b_avg = 0

    if rc and rb:
        diff = f'{c_avg - b_avg:>+7.3f}'
    else:
        diff = f'{"":>7s}'

    if not rc and not rb:
        continue

    print(f'  {year:>4d}  {c_str}  {b_str}  {diff}')
