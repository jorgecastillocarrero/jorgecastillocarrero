"""
Analisis del regimen NEUTRAL (bear_ratio 0.30-0.45) - 140 semanas perdedoras
Objetivo: entender por que pierde y como optimizar
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/fair_v3_weekly_table.csv')
neutral = df[df['regime'] == 'NEUTRAL'].copy()

print(f"{'='*100}")
print(f"  ANALISIS REGIMEN NEUTRAL - {len(neutral)} semanas, PnL=${neutral['pnl'].sum():+,.0f}")
print(f"{'='*100}")

# 1. Distribucion por config
print(f"\n  1. DISTRIBUCION POR CONFIGURACION:")
for cfg in sorted(neutral['config'].unique()):
    subset = neutral[neutral['config'] == cfg]
    total = subset['pnl'].sum()
    avg = subset['pnl'].mean()
    wr = (subset['pnl'] > 0).mean() * 100
    spy_avg = subset['spy_return_pct'].mean()
    print(f"    {cfg:8s}: {len(subset):3d} sem  PnL=${total:>+10,.0f}  Avg=${avg:>+6,.0f}  WR={wr:5.1f}%  SPY={spy_avg:+.2f}%")

# 2. Desglose longs vs shorts PnL
# Para cada semana NEUTRAL, ver si los shorts ayudan o perjudican
print(f"\n  2. BEAR_RATIO DETALLE:")
for br_range, label in [
    ((0.30, 0.35), "0.30-0.35 (casi bullish)"),
    ((0.35, 0.40), "0.35-0.40"),
    ((0.40, 0.45), "0.40-0.45 (casi bearish)"),
]:
    subset = neutral[(neutral['bear_ratio'] >= br_range[0]) & (neutral['bear_ratio'] < br_range[1])]
    if len(subset) == 0:
        continue
    total = subset['pnl'].sum()
    avg = subset['pnl'].mean()
    wr = (subset['pnl'] > 0).mean() * 100
    spy_avg = subset['spy_return_pct'].mean()
    n_short = (subset['n_shorts'] > 0).sum()
    print(f"    br {label:25s}: {len(subset):3d} sem  PnL=${total:>+10,.0f}  Avg=${avg:>+6,.0f}  WR={wr:5.1f}%  SPY={spy_avg:+.2f}%  con_shorts={n_short}")

# 3. Cuando tiene shorts vs no
print(f"\n  3. CON SHORTS vs SIN SHORTS:")
with_shorts = neutral[neutral['n_shorts'] > 0]
no_shorts = neutral[neutral['n_shorts'] == 0]
print(f"    Con shorts:  {len(with_shorts):3d} sem  PnL=${with_shorts['pnl'].sum():>+10,.0f}  Avg=${with_shorts['pnl'].mean():>+6,.0f}  WR={(with_shorts['pnl'] > 0).mean()*100:5.1f}%  SPY={(with_shorts['spy_return_pct'].mean()):+.2f}%")
if len(no_shorts) > 0:
    print(f"    Sin shorts:  {len(no_shorts):3d} sem  PnL=${no_shorts['pnl'].sum():>+10,.0f}  Avg=${no_shorts['pnl'].mean():>+6,.0f}  WR={(no_shorts['pnl'] > 0).mean()*100:5.1f}%  SPY={(no_shorts['spy_return_pct'].mean()):+.2f}%")

# 4. Semanas SPY positivas vs negativas
print(f"\n  4. CUANDO SPY SUBE vs BAJA:")
spy_up = neutral[neutral['spy_return_pct'] > 0]
spy_down = neutral[neutral['spy_return_pct'] <= 0]
print(f"    SPY sube:  {len(spy_up):3d} sem  PnL=${spy_up['pnl'].sum():>+10,.0f}  Avg=${spy_up['pnl'].mean():>+6,.0f}")
print(f"    SPY baja:  {len(spy_down):3d} sem  PnL=${spy_down['pnl'].sum():>+10,.0f}  Avg=${spy_down['pnl'].mean():>+6,.0f}")

# 5. Distribucion por ano
print(f"\n  5. PNL NEUTRAL POR ANO:")
for year in sorted(neutral['year'].unique()):
    yr = neutral[neutral['year'] == year]
    cfgs = yr['config'].value_counts().to_dict()
    cfg_str = ', '.join(f"{k}:{v}" for k, v in sorted(cfgs.items()))
    print(f"    {year}: {len(yr):2d} sem  PnL=${yr['pnl'].sum():>+10,.0f}  SPY={yr['spy_return_pct'].mean():+.2f}%  [{cfg_str}]")

# 6. Las 10 peores semanas NEUTRAL
print(f"\n  6. 10 PEORES SEMANAS NEUTRAL:")
worst = neutral.nsmallest(10, 'pnl')
for _, r in worst.iterrows():
    print(f"    {r['date']}  {r['config']:6s}  br={r['bear_ratio']:.2f}  PnL=${r['pnl']:>+9,.0f}  SPY={r['spy_return_pct']:+.1f}%  L:{r['longs'][:40]}  S:{r['shorts'][:40]}")

# 7. Las 10 mejores semanas NEUTRAL
print(f"\n  7. 10 MEJORES SEMANAS NEUTRAL:")
best = neutral.nlargest(10, 'pnl')
for _, r in best.iterrows():
    print(f"    {r['date']}  {r['config']:6s}  br={r['bear_ratio']:.2f}  PnL=${r['pnl']:>+9,.0f}  SPY={r['spy_return_pct']:+.1f}%  L:{r['longs'][:40]}  S:{r['shorts'][:40]}")

# 8. Simulacion: que pasa si en NEUTRAL eliminamos los shorts?
print(f"\n\n{'='*100}")
print(f"  SIMULACIONES DE OPTIMIZACION")
print(f"{'='*100}")

# Opcion A: No operar en NEUTRAL (0L+0S)
print(f"\n  A) No operar en NEUTRAL:")
print(f"     Ahorro: ${-neutral['pnl'].sum():+,.0f} (evitamos -$42K de perdidas)")
print(f"     Semanas inactivas: +{len(neutral)} semanas sin operar")

# Para las demas opciones necesitamos re-simular
# Cargar los datos completos para re-calcular
print(f"\n  B) En NEUTRAL, solo longs (eliminar shorts):")
print(f"     Requiere re-simulacion (ver siguiente paso)")

print(f"\n  C) En NEUTRAL, subir threshold de shorts (score < 2.5 en vez de < 3.5):")
print(f"     Requiere re-simulacion")

print(f"\n  D) Reclasificar: br 0.30-0.37 = BULLISH (3L+0S), br 0.37-0.45 = NEUTRAL (3L+0S):")
print(f"     Simplemente: no shortear cuando br < 0.45")
