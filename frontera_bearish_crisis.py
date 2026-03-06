"""
Estudio frontera BEARISH vs CRISIS
- Semanas que cambian de regimen entre modos
- Scores cerca de la frontera (-5.0)
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
# 1. SEMANAS DONDE UN MODO DICE BEARISH Y OTRO CRISIS
# ===================================================================
print('='*120)
print('  1. SEMANAS DONDE UN MODO DICE BEARISH Y OTRO CRISIS')
print('='*120)

discrepantes = []
for d in sorted(by_date.keys()):
    data = by_date[d]
    if len(data) < 3:
        continue
    regs = [data[m]['regime'] for m in ['Original', 'Hybrid', 'MinDD']]
    has_bear = 'BEARISH' in regs
    has_crisis = 'CRISIS' in regs
    if has_bear and has_crisis:
        discrepantes.append(d)

print(f'\n  Total semanas con discrepancia B/Cr: {len(discrepantes)}')

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
# 2. SCORES CERCA DE LA FRONTERA (-5.0)
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  2. SCORES CERCA DE LA FRONTERA: que pasa entre score -7.0 y -3.0')
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
        if sc is not None and -7.5 <= sc <= -2.5 and ret is not None:
            frontera_weeks.append((d, sc, ret, reg))

    print(f'\n  {label} - Semanas en zona frontera:')
    print(f'    {"Score":>6s}  {"Regimen":>10s}  {"N":>4s}  {"Avg%":>7s}  {"WR%":>5s}  {"Tot%":>7s}')
    print(f'    {"-"*6}  {"-"*10}  {"-"*4}  {"-"*7}  {"-"*5}  {"-"*7}')

    for sc_val in np.arange(-7.5, -2.0, 0.5):
        weeks_sc = [(d, r, reg) for d, sc, r, reg in frontera_weeks if abs(sc - sc_val) < 0.01]
        if not weeks_sc:
            continue
        rets_sc = [r for _, r, _ in weeks_sc]
        reg_label = weeks_sc[0][2]
        avg = np.mean(rets_sc)
        wr = sum(1 for r in rets_sc if r > 0) / len(rets_sc) * 100
        tot = sum(rets_sc)
        marker = '  <-- frontera' if abs(sc_val - (-5.0)) < 0.01 else ''
        print(f'    {sc_val:>+5.1f}  {reg_label:>10s}  {len(rets_sc):>4d}  {avg:>+7.3f}  {wr:>4.1f}%  {tot:>+7.1f}{marker}')

# ===================================================================
# 3. QUE INDICADOR CAUSA EL CAMBIO
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  3. QUE DIFERENCIA HAY ENTRE BEARISH Y CRISIS (indicadores medios)')
print(f'{"="*120}')

for label in ['Original', 'Hybrid', 'MinDD']:
    print(f'\n  {label}:')
    print(f'    {"Indicador":>15s}  {"BEARISH":>10s}  {"CRISIS":>10s}  {"Diff":>10s}')
    print(f'    {"-"*15}  {"-"*10}  {"-"*10}  {"-"*10}')

    for ind_name, ind_key in [('DD healthy %', 'pct_dd_h'), ('RSI>55 %', 'pct_rsi'),
                               ('SPY dist %', 'spy_dist'), ('SPY mom %', 'spy_mom'),
                               ('VIX', 'vix'), ('Score', 'score')]:
        b_vals = [by_date[d][label][ind_key] for d in sorted(by_date.keys())
                  if label in by_date[d] and by_date[d][label]['regime'] == 'BEARISH'
                  and by_date[d][label][ind_key] is not None]
        cr_vals = [by_date[d][label][ind_key] for d in sorted(by_date.keys())
                  if label in by_date[d] and by_date[d][label]['regime'] == 'CRISIS'
                  and by_date[d][label][ind_key] is not None]

        if b_vals and cr_vals:
            b_avg = np.mean(b_vals)
            cr_avg = np.mean(cr_vals)
            print(f'    {ind_name:>15s}  {b_avg:>+10.2f}  {cr_avg:>+10.2f}  {b_avg-cr_avg:>+10.2f}')

# ===================================================================
# 4. FRONTERAS ALTERNATIVAS
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  4. FRONTERAS ALTERNATIVAS: si movemos el corte BEARISH/CRISIS')
print(f'{"="*120}')

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

print(f'\n  Actual: BEARISH >= -5.0, CRISIS >= -9.0')
print(f'\n  {"Frontera":>9s}  |  {"--- BEARISH (>= frontera) ---":>35s}  |  {"--- CRISIS (< frontera) ---":>35s}')
print(f'  {"-"*9}  |  {"N":>4s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s} {"Std":>5s}  |  {"N":>4s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s} {"Std":>5s}')

for frontier in np.arange(-7.0, -3.0, 0.5):
    bear_rets = []
    crisis_rets = []
    for label in ['Original', 'Hybrid', 'MinDD']:
        for r in mode_rows[label]:
            if r['_score'] is None or r['_ret'] is None:
                continue
            sc = r['_score']
            # Only in BEARISH-CRISIS zone (exclude CAUTIOUS and PANICO)
            if sc >= -2.0 or sc < -9.0:
                continue
            if sc >= frontier:
                bear_rets.append(r['_ret'])
            else:
                crisis_rets.append(r['_ret'])

    if not bear_rets or not crisis_rets:
        continue

    b_avg = np.mean(bear_rets)
    b_wr = sum(1 for r in bear_rets if r > 0) / len(bear_rets) * 100
    b_tot = sum(bear_rets)
    b_std = np.std(bear_rets)

    cr_avg = np.mean(crisis_rets)
    cr_wr = sum(1 for r in crisis_rets if r > 0) / len(crisis_rets) * 100
    cr_tot = sum(crisis_rets)
    cr_std = np.std(crisis_rets)

    diff = b_avg - cr_avg
    marker = ' <<<' if abs(frontier - (-5.0)) < 0.01 else ''

    print(f'  Score {frontier:>+4.1f}  |  {len(bear_rets):>4d} {b_avg:>+7.3f} {b_wr:>4.1f}% {b_tot:>+7.1f} {b_std:>5.2f}  |  {len(crisis_rets):>4d} {cr_avg:>+7.3f} {cr_wr:>4.1f}% {cr_tot:>+7.1f} {cr_std:>5.2f}  diff={diff:>+.3f}{marker}')

# ===================================================================
# 5. BEARISH + CRISIS JUNTOS vs SEPARADOS
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  5. BEARISH + CRISIS JUNTOS vs SEPARADOS')
print(f'{"="*120}')

for label in ['Original', 'Hybrid', 'MinDD']:
    bear = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'BEARISH' and r['_ret'] is not None]
    crisis = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'CRISIS' and r['_ret'] is not None]
    juntos = bear + crisis

    print(f'\n  {label}:')
    if bear:
        print(f'    BEARISH:          N={len(bear):>4d}  Avg={np.mean(bear):>+.3f}%  WR={sum(1 for r in bear if r>0)/len(bear)*100:.1f}%  Tot={sum(bear):>+.1f}%')
    if crisis:
        print(f'    CRISIS:           N={len(crisis):>4d}  Avg={np.mean(crisis):>+.3f}%  WR={sum(1 for r in crisis if r>0)/len(crisis)*100:.1f}%  Tot={sum(crisis):>+.1f}%')
    if juntos:
        print(f'    JUNTOS (B+Cr):    N={len(juntos):>4d}  Avg={np.mean(juntos):>+.3f}%  WR={sum(1 for r in juntos if r>0)/len(juntos)*100:.1f}%  Tot={sum(juntos):>+.1f}%')
    if bear and crisis:
        print(f'    Diferencia B-Cr:  {np.mean(bear)-np.mean(crisis):>+.3f}% por semana')

# ===================================================================
# 6. TAMBIÉN: CRISIS vs PANICO
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  6. BONUS: CRISIS vs PANICO')
print(f'{"="*120}')

for label in ['Original', 'Hybrid', 'MinDD']:
    crisis = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'CRISIS' and r['_ret'] is not None]
    panico = [r['_ret'] for r in mode_rows[label] if r['regime'] == 'PANICO' and r['_ret'] is not None]

    print(f'\n  {label}:')
    if crisis:
        print(f'    CRISIS:           N={len(crisis):>4d}  Avg={np.mean(crisis):>+.3f}%  WR={sum(1 for r in crisis if r>0)/len(crisis)*100:.1f}%  Tot={sum(crisis):>+.1f}%')
    if panico:
        print(f'    PANICO:           N={len(panico):>4d}  Avg={np.mean(panico):>+.3f}%  WR={sum(1 for r in panico if r>0)/len(panico)*100:.1f}%  Tot={sum(panico):>+.1f}%')
    if crisis and panico:
        print(f'    Diferencia Cr-P:  {np.mean(crisis)-np.mean(panico):>+.3f}% por semana')

# ===================================================================
# 7. DISCREPANCIAS B/Cr POR AÑO
# ===================================================================
print(f'\n\n{"="*120}')
print(f'  7. POR AÑO: semanas BEARISH vs CRISIS y rendimiento (media 3 modos)')
print(f'{"="*120}')

years_data = defaultdict(lambda: {'ret_bear': [], 'ret_crisis': []})

for label in ['Original', 'Hybrid', 'MinDD']:
    for d in sorted(by_date.keys()):
        if label not in by_date[d]:
            continue
        data = by_date[d][label]
        year = int(d[:4])
        ret = data.get('ret')
        if ret is None:
            continue
        if data['regime'] == 'BEARISH':
            years_data[year]['ret_bear'].append(ret)
        elif data['regime'] == 'CRISIS':
            years_data[year]['ret_crisis'].append(ret)

print(f'\n  {"Ano":>4s}  {"N_BEAR":>5s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s}  {"N_CRISIS":>5s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s}  {"Diff":>7s}')
print(f'  {"-"*4}  {"-"*5} {"-"*7} {"-"*5} {"-"*7}  {"-"*5} {"-"*7} {"-"*5} {"-"*7}  {"-"*7}')

for year in sorted(years_data.keys()):
    rb = years_data[year]['ret_bear']
    rcr = years_data[year]['ret_crisis']

    if rb:
        b_avg = np.mean(rb)
        b_wr = sum(1 for r in rb if r > 0) / len(rb) * 100
        b_tot = sum(rb)
        b_str = f'{len(rb):>5d} {b_avg:>+7.3f} {b_wr:>4.1f}% {b_tot:>+7.1f}'
    else:
        b_str = f'{"":>5s} {"":>7s} {"":>5s} {"":>7s}'
        b_avg = 0

    if rcr:
        cr_avg = np.mean(rcr)
        cr_wr = sum(1 for r in rcr if r > 0) / len(rcr) * 100
        cr_tot = sum(rcr)
        cr_str = f'{len(rcr):>5d} {cr_avg:>+7.3f} {cr_wr:>4.1f}% {cr_tot:>+7.1f}'
    else:
        cr_str = f'{"":>5s} {"":>7s} {"":>5s} {"":>7s}'
        cr_avg = 0

    if rb and rcr:
        diff = f'{b_avg - cr_avg:>+7.3f}'
    else:
        diff = f'{"":>7s}'

    if not rb and not rcr:
        continue

    print(f'  {year:>4d}  {b_str}  {cr_str}  {diff}')
