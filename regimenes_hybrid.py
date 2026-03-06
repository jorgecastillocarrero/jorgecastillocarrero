"""
Genera regimen HYBRID: promedio de scores Original + MinDD
Aplica umbrales Original sobre el score promediado
"""
import pandas as pd, numpy as np, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

orig = pd.read_csv('data/regimenes_historico.csv')
mindd = pd.read_csv('data/regimenes_mindd.csv')

print(f"Original: {len(orig)} weeks")
print(f"MinDD: {len(mindd)} weeks")

# Merge on fecha_senal
m = orig.merge(mindd[['fecha_senal', 'total', 'regime', 'vix_delta']],
               on='fecha_senal', suffixes=('', '_mindd'))

# Average score
m['total_hybrid'] = ((m['total'] + m['total_mindd']) / 2).round(1)

# Apply Original thresholds on hybrid score
def assign_regime(row):
    score = row['total_hybrid']
    vix = row['vix']
    vix_delta = row['vix_delta']
    pct_dd_h = row['pct_dd_h']
    pct_rsi = row['pct_rsi']

    # Base regime from score
    if score >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90:
        regime = 'BURBUJA'
    elif score >= 7.0:
        regime = 'GOLDILOCKS'
    elif score >= 4.0:
        regime = 'ALCISTA'
    elif score >= 0.5:
        regime = 'NEUTRAL'
    elif score >= -2.0:
        regime = 'CAUTIOUS'
    elif score >= -5.0:
        regime = 'BEARISH'
    elif score >= -9.0:
        regime = 'CRISIS'
    else:
        regime = 'PANICO'

    # VIX veto
    if vix >= 35 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL'):
        regime = 'CAUTIOUS'
    elif vix >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        regime = 'NEUTRAL'

    # RECOVERY: BEARISH + VIX bajando
    if regime == 'BEARISH' and vix_delta < 0:
        regime = 'RECOVERY'

    # CAPITULACION: PANICO + VIX bajando
    if regime == 'PANICO' and vix_delta < 0:
        regime = 'CAPITULACION'

    return regime

m['regime_hybrid'] = m.apply(assign_regime, axis=1)

# Build output CSV (same format as Original)
out = m[['fecha_senal', 'year', 'sem', 'n_sub', 'dd_h', 'pct_dd_h', 'dd_d', 'pct_dd_d',
         'rsi55', 'pct_rsi', 'spy_close', 'spy_ma200', 'spy_dist', 'spy_mom',
         'vix', 'vix_delta', 's_bdd', 's_brsi', 's_ddp', 's_spy', 's_mom']].copy()
out['total'] = m['total_hybrid']
out['regime'] = m['regime_hybrid']
out['vix_veto'] = m['vix_veto']
out['fri_entry'] = m['fri_entry']
out['fri_exit'] = m['fri_exit']
out['spy_entry_open'] = m['spy_entry_open']
out['spy_exit_open'] = m['spy_exit_open']
out['spy_ret_pct'] = m['spy_ret_pct']

out.to_csv('data/regimenes_hybrid.csv', index=False)
print(f"\nSaved: data/regimenes_hybrid.csv ({len(out)} weeks)")

# Compare all three
print(f"\n{'='*80}")
print(f"  COMPARATIVA REGIMENES: Original vs Hybrid vs MinDD")
print(f"{'='*80}")
print(f"\n  {'Regimen':15s} {'Original':>8s} {'Hybrid':>8s} {'MinDD':>8s}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")

regimes = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
           'RECOVERY','CRISIS','PANICO','CAPITULACION']

for reg in regimes:
    n_orig = (m['regime'] == reg).sum()
    n_hyb = (m['regime_hybrid'] == reg).sum()
    n_mindd = (m['regime_mindd'] == reg).sum()
    print(f"  {reg:15s} {n_orig:>8d} {n_hyb:>8d} {n_mindd:>8d}")

# Agreement rates
agree_oh = (m['regime'] == m['regime_hybrid']).sum()
agree_hm = (m['regime_hybrid'] == m['regime_mindd']).sum()
agree_om = (m['regime'] == m['regime_mindd']).sum()
print(f"\n  Acuerdo Orig-Hybrid: {agree_oh}/{len(m)} ({agree_oh/len(m)*100:.1f}%)")
print(f"  Acuerdo Hybrid-MinDD: {agree_hm}/{len(m)} ({agree_hm/len(m)*100:.1f}%)")
print(f"  Acuerdo Orig-MinDD:  {agree_om}/{len(m)} ({agree_om/len(m)*100:.1f}%)")

# Score stats
print(f"\n  Score medio: Orig {m['total'].mean():+.2f} | Hybrid {m['total_hybrid'].mean():+.2f} | MinDD {m['total_mindd'].mean():+.2f}")
print(f"  Score std:   Orig {m['total'].std():.2f} | Hybrid {m['total_hybrid'].std():.2f} | MinDD {m['total_mindd'].std():.2f}")
