"""
Estudio frontera NEUTRAL vs CAUTIOUS
- Semanas que cambian de regimen entre modos
- Que score tienen en cada modo
- Como rinden realmente esas semanas frontera
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

# ═══════════════════════════════════════════════════════════════
# 1. SEMANAS DONDE LOS MODOS DISCREPAN EN NEUTRAL/CAUTIOUS
# ═══════════════════════════════════════════════════════════════
print('='*120)
print('  1. SEMANAS DONDE UN MODO DICE NEUTRAL Y OTRO CAUTIOUS')
print('='*120)

discrepantes = []
for d in sorted(by_date.keys()):
    data = by_date[d]
    if len(data) < 3:
        continue
    regs = [data[m]['regime'] for m in ['Original', 'Hybrid', 'MinDD']]
    # Al menos uno NEUTRAL y al menos uno CAUTIOUS
    has_n = 'NEUTRAL' in regs
    has_c = 'CAUTIOUS' in regs
    if has_n and has_c:
        discrepantes.append(d)

print(f'\n  Total semanas con discrepancia N/C: {len(discrepantes)}')

# Show them
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

# ═══════════════════════════════════════════════════════════════
# 2. SCORES CERCA DE LA FRONTERA (0.5)
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"="*120}')
print(f'  2. SCORES CERCA DE LA FRONTERA: que pasa entre score 0.0 y +1.0')
print(f'{"="*120}')

# Semanas con score entre -0.5 y +1.5 en algun modo
for label in ['Original', 'Hybrid', 'MinDD']:
    frontera_weeks = []
    for d in sorted(by_date.keys()):
        data = by_date[d]
        if label not in data:
            continue
        sc = data[label].get('score')
        ret = data[label].get('ret')
        reg = data[label].get('regime')
        if sc is not None and -0.5 <= sc <= 1.0 and ret is not None:
            frontera_weeks.append((d, sc, ret, reg))

    # Group by score
    print(f'\n  {label} - Semanas en zona frontera:')
    print(f'    {"Score":>6s}  {"Regimen":>10s}  {"N":>4s}  {"Avg%":>7s}  {"WR%":>5s}  {"Tot%":>7s}')
    print(f'    {"-"*6}  {"-"*10}  {"-"*4}  {"-"*7}  {"-"*5}  {"-"*7}')

    for sc_val in [-0.5, 0.0, 0.5, 1.0]:
        weeks_sc = [(d, r, reg) for d, sc, r, reg in frontera_weeks if abs(sc - sc_val) < 0.01]
        if not weeks_sc:
            continue
        rets_sc = [r for _, r, _ in weeks_sc]
        reg_label = weeks_sc[0][2]
        avg = np.mean(rets_sc)
        wr = sum(1 for r in rets_sc if r > 0) / len(rets_sc) * 100
        tot = sum(rets_sc)
        marker = '  <-- frontera' if sc_val == 0.5 else ''
        print(f'    {sc_val:>+5.1f}  {reg_label:>10s}  {len(rets_sc):>4d}  {avg:>+7.3f}  {wr:>4.1f}%  {tot:>+7.1f}{marker}')

# ═══════════════════════════════════════════════════════════════
# 3. QUE INDICADOR CAUSA EL CAMBIO
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"="*120}')
print(f'  3. QUE DIFERENCIA HAY ENTRE NEUTRAL Y CAUTIOUS (indicadores medios)')
print(f'{"="*120}')

for label in ['Original', 'Hybrid', 'MinDD']:
    print(f'\n  {label}:')
    print(f'    {"Indicador":>15s}  {"NEUTRAL":>10s}  {"CAUTIOUS":>10s}  {"Diff":>10s}')
    print(f'    {"-"*15}  {"-"*10}  {"-"*10}  {"-"*10}')

    for ind_name, ind_key in [('DD healthy %', 'pct_dd_h'), ('RSI>55 %', 'pct_rsi'),
                               ('SPY dist %', 'spy_dist'), ('SPY mom %', 'spy_mom'),
                               ('VIX', 'vix'), ('Score', 'score')]:
        n_vals = [by_date[d][label][ind_key] for d in sorted(by_date.keys())
                  if label in by_date[d] and by_date[d][label]['regime'] == 'NEUTRAL'
                  and by_date[d][label][ind_key] is not None]
        c_vals = [by_date[d][label][ind_key] for d in sorted(by_date.keys())
                  if label in by_date[d] and by_date[d][label]['regime'] == 'CAUTIOUS'
                  and by_date[d][label][ind_key] is not None]

        if n_vals and c_vals:
            n_avg = np.mean(n_vals)
            c_avg = np.mean(c_vals)
            print(f'    {ind_name:>15s}  {n_avg:>+10.2f}  {c_avg:>+10.2f}  {n_avg-c_avg:>+10.2f}')

# ═══════════════════════════════════════════════════════════════
# 4. ANALISIS POR AÑO: NEUTRAL vs CAUTIOUS en cada modo
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"="*120}')
print(f'  4. POR AÑO: semanas NEUTRAL vs CAUTIOUS y rendimiento (media 3 modos)')
print(f'{"="*120}')

years_data = defaultdict(lambda: {'n_neu': [], 'n_cau': [], 'ret_neu': [], 'ret_cau': []})

for label in ['Original', 'Hybrid', 'MinDD']:
    for d in sorted(by_date.keys()):
        if label not in by_date[d]:
            continue
        data = by_date[d][label]
        year = int(d[:4])
        ret = data.get('ret')
        if ret is None:
            continue
        if data['regime'] == 'NEUTRAL':
            years_data[year]['ret_neu'].append(ret)
        elif data['regime'] == 'CAUTIOUS':
            years_data[year]['ret_cau'].append(ret)

print(f'\n  {"Ano":>4s}  {"N_NEU":>5s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s}  {"N_CAU":>5s} {"Avg%":>7s} {"WR%":>5s} {"Tot%":>7s}  {"Diff":>7s}')
print(f'  {"-"*4}  {"-"*5} {"-"*7} {"-"*5} {"-"*7}  {"-"*5} {"-"*7} {"-"*5} {"-"*7}  {"-"*7}')

for year in sorted(years_data.keys()):
    rn = years_data[year]['ret_neu']
    rc = years_data[year]['ret_cau']

    if rn:
        n_avg = np.mean(rn)
        n_wr = sum(1 for r in rn if r > 0) / len(rn) * 100
        n_tot = sum(rn)
        n_str = f'{len(rn):>5d} {n_avg:>+7.3f} {n_wr:>4.1f}% {n_tot:>+7.1f}'
    else:
        n_str = f'{"":>5s} {"":>7s} {"":>5s} {"":>7s}'
        n_avg = 0

    if rc:
        c_avg = np.mean(rc)
        c_wr = sum(1 for r in rc if r > 0) / len(rc) * 100
        c_tot = sum(rc)
        c_str = f'{len(rc):>5d} {c_avg:>+7.3f} {c_wr:>4.1f}% {c_tot:>+7.1f}'
    else:
        c_str = f'{"":>5s} {"":>7s} {"":>5s} {"":>7s}'
        c_avg = 0

    if rn and rc:
        diff = f'{n_avg - c_avg:>+7.3f}'
    else:
        diff = f'{"":>7s}'

    if not rn and not rc:
        continue

    print(f'  {year:>4d}  {n_str}  {c_str}  {diff}')
