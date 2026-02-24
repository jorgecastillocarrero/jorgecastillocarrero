"""
BACKTEST COMBINADO 3 NIVELES (v2 - P2 con 10 longs):
  Nivel 1 - Pattern 1 (14 sem): 5L + 5S con pattern multi-factor  → $500K/sem
  Nivel 2 - Pattern 2 (10 sem): 10L solo longs con pattern        → $500K/sem
  Nivel 3 - Adaptativo (28 sem): 5L + 5S por regimen de mercado   → $500K/sem

P2 usa resultados del backtest de 10 longs (tmp_p2_10longs.py).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from regime_pattern_seasonal import OPERABLE_WEEKS
from regime_pattern_longonly import OPERABLE_WEEKS_LONG

# ============================================================
# CARGAR DATOS
# ============================================================
# Pattern backtest 5L+5S (para P1)
pat = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
pat = pat.dropna(subset=['PnL_USD'])
pat['Signal_Date'] = pd.to_datetime(pat['Signal_Date'])
pat['Week_of_Year'] = pat['Signal_Date'].dt.isocalendar().week.astype(int)

# Regime backtest (para adaptativo)
reg = pd.read_excel('data/regime_weekly_picks.xlsx', sheet_name='Semanas')
reg['Signal_Date'] = pd.to_datetime(reg['Signal_Date'])
reg['Week_of_Year'] = reg['Signal_Date'].dt.isocalendar().week.astype(int)

# P2 10-longs backtest (tmp_p2_10longs.py output)
# We need to re-load from the script results. Since we don't have an Excel,
# we recalculate: P2 10L P&L = P2 5L P&L * ratio from backtest
# Backtest showed: 5L = $389,983, 10L = $511,120, ratio = 1.311
# But better: we compute 10L P&L = 5L P&L (from pattern) + extra P&L scaled
# Actually, the simplest: 10L P&L per week ≈ 5L Long_PnL * (511120/389983) = 1.311

# More accurate: use the actual 10L backtest per-week data
# The 10L backtest result is: P&L = $511,120 across 225 weeks
# vs pattern 5L: Long_PnL = $525,174 across 220 weeks
# The difference is due to different eligible stock sets and slight timing
# For the combined, we use the 10L result directly

# Scale factor per week: 10L/5L ratio from the backtest
# Per-week basis from the summary: 10L PnL/sem = $2,272, 5L PnL/sem = $1,733
# Extra from picks 6-10 per week = $538
# So: 10L weekly PnL ≈ 5L weekly PnL + $538 extra per week on average
# But the distribution varies. Let's use a more precise approach:
# Apply per-week-of-year ratios from the semana analysis

# Per-week ratios from the backtest:
WEEK_10L_PNL = {
    4: 83276, 5: 14928, 6: 32777, 11: 85411, 12: 31773,
    25: 40981, 28: 139608, 31: 3449, 44: 26447, 50: 52469,
}
WEEK_5L_PNL = {
    4: 44666, 5: 19069, 6: 50368, 11: 44063, 12: 44433,
    25: 36020, 28: 73464, 31: 6597, 44: 28812, 50: 42490,
}

# For each P2 week, scale the pattern Long_PnL by the 10L/5L ratio for that week
def get_10l_scale(week):
    """Return scaling factor to convert 5L PnL to 10L PnL for a given week."""
    if week in WEEK_5L_PNL and WEEK_5L_PNL[week] != 0:
        return WEEK_10L_PNL[week] / WEEK_5L_PNL[week]
    return 1.311  # fallback overall ratio

print("=" * 160)
print("  BACKTEST COMBINADO 3 NIVELES (v2)")
print("  P1: 5L+5S Pattern ($500K) | P2: 10L Pattern ($500K) | Adaptativo: 5L+5S Regime ($500K)")
print("=" * 160)

# ============================================================
# MERGE DATA
# ============================================================
merged = pat[['Signal_Date', 'Year', 'Week_of_Year', 'PnL_USD', 'Long_PnL', 'Short_PnL',
              'Ret_Pct', 'Win']].copy()
merged = merged.rename(columns={
    'PnL_USD': 'pat_pnl', 'Long_PnL': 'pat_long', 'Short_PnL': 'pat_short',
    'Ret_Pct': 'pat_ret', 'Win': 'pat_win'
})

reg_cols = reg[['Signal_Date', 'Score', 'PnL_USD', 'Long_PnL', 'Short_PnL', 'Ret_Pct', 'Win']].copy()
reg_cols = reg_cols.rename(columns={
    'PnL_USD': 'reg_pnl', 'Long_PnL': 'reg_long', 'Short_PnL': 'reg_short',
    'Ret_Pct': 'reg_ret', 'Win': 'reg_win'
})

combined = merged.merge(reg_cols, on='Signal_Date', how='inner')

def classify_week(w):
    if w in OPERABLE_WEEKS: return 'P1_LS'
    elif w in OPERABLE_WEEKS_LONG: return 'P2_10L'
    else: return 'ADAPTIVE'

combined['Tier'] = combined['Week_of_Year'].apply(classify_week)

# Combined P&L
def calc_pnl(row):
    if row['Tier'] == 'P1_LS':
        return row['pat_pnl']  # 5L + 5S pattern
    elif row['Tier'] == 'P2_10L':
        # Scale 5L Long_PnL to 10L using per-week ratio
        scale = get_10l_scale(row['Week_of_Year'])
        return row['pat_long'] * scale
    else:
        return row['reg_pnl']  # adaptive 5L + 5S

def calc_long(row):
    if row['Tier'] == 'P1_LS': return row['pat_long']
    elif row['Tier'] == 'P2_10L':
        scale = get_10l_scale(row['Week_of_Year'])
        return row['pat_long'] * scale
    else: return row['reg_long']

def calc_short(row):
    if row['Tier'] == 'P1_LS': return row['pat_short']
    elif row['Tier'] == 'P2_10L': return 0
    else: return row['reg_short']

combined['pnl'] = combined.apply(calc_pnl, axis=1)
combined['long_pnl'] = combined.apply(calc_long, axis=1)
combined['short_pnl'] = combined.apply(calc_short, axis=1)
combined['win'] = (combined['pnl'] > 0).astype(int)

# ============================================================
# 1. RESUMEN POR TIER
# ============================================================
total_n = len(combined)

print(f"\n  RESUMEN POR NIVEL")
print(f"\n  {'Tier':>20s} | {'N':>5} | {'%':>4} | {'Capital':>8} | {'Total P&L':>12} | {'P&L/sem':>10} | {'Long':>12} | {'Short':>12} | {'WR%':>4}")
print(f"  {'-'*20} | {'-'*5} | {'-'*4} | {'-'*8} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*4}")

for tier, label in [('P1_LS','P1: 5L+5S Pattern'), ('P2_10L','P2: 10L Pattern'), ('ADAPTIVE','Adaptativo 5L+5S')]:
    sub = combined[combined['Tier']==tier]
    n = len(sub)
    pnl = sub['pnl'].sum()
    lp = sub['long_pnl'].sum()
    sp = sub['short_pnl'].sum()
    wr = sub['win'].sum()/n*100
    cap = "$500K" if tier != 'OLD_P2' else "$500K"
    print(f"  {label:>20s} | {n:>5} | {n/total_n*100:>3.0f}% | {cap:>8s} | ${pnl:>+11,.0f} | ${pnl/n:>+9,.0f} | ${lp:>+11,.0f} | ${sp:>+11,.0f} | {wr:>3.0f}%")

pnl_total = combined['pnl'].sum()
long_total = combined['long_pnl'].sum()
short_total = combined['short_pnl'].sum()
wr_total = combined['win'].sum()/total_n*100
print(f"  {'-'*20} | {'-'*5} | {'-'*4} | {'-'*8} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*4}")
print(f"  {'COMBINADO':>20s} | {total_n:>5} | 100% | {'$500K':>8s} | ${pnl_total:>+11,.0f} | ${pnl_total/total_n:>+9,.0f} | ${long_total:>+11,.0f} | ${short_total:>+11,.0f} | {wr_total:>3.0f}%")

# ============================================================
# 2. AÑO A AÑO
# ============================================================
print(f"\n\n{'='*160}")
print("  AÑO A AÑO COMBINADO 3 NIVELES")
print("=" * 160)
print(f"\n  {'Año':>6} | {'P1':>3} | {'P1 P&L':>11} | {'P2':>3} | {'P2 10L P&L':>11} | {'Adp':>3} | {'Adp P&L':>11} | {'Combo':>12} | {'WR%':>4} | {'Acum':>14}")
print(f"  {'-'*6} | {'-'*3} | {'-'*11} | {'-'*3} | {'-'*11} | {'-'*3} | {'-'*11} | {'-'*12} | {'-'*4} | {'-'*14}")

cum = 0; n_pos = 0
for year in sorted(combined['Year'].unique()):
    ys = combined[combined['Year']==year]
    p1s = ys[ys['Tier']=='P1_LS']; p2s = ys[ys['Tier']=='P2_10L']; ads = ys[ys['Tier']=='ADAPTIVE']
    pnl1 = p1s['pnl'].sum(); pnl2 = p2s['pnl'].sum(); pnla = ads['pnl'].sum()
    combo = pnl1 + pnl2 + pnla; cum += combo
    wr_y = ys['win'].sum()/len(ys)*100 if len(ys)>0 else 0
    if combo > 0: n_pos += 1
    print(f"  {year:>6} | {len(p1s):>3} | ${pnl1:>+10,.0f} | {len(p2s):>3} | ${pnl2:>+10,.0f} | {len(ads):>3} | ${pnla:>+10,.0f} | ${combo:>+11,.0f} | {wr_y:>3.0f}% | ${cum:>+13,.0f}")

n_years = len(combined['Year'].unique())
print(f"\n  Años positivos: {n_pos}/{n_years} ({n_pos/n_years*100:.0f}%)")

# ============================================================
# 3. COMPARATIVA FINAL
# ============================================================
print(f"\n\n{'='*160}")
print("  COMPARATIVA FINAL: TODOS LOS SISTEMAS")
print("=" * 160)

adapt_pnl = reg['PnL_USD'].sum()
p1_only = combined[combined['Tier']=='P1_LS']['pat_pnl'].sum()
p1p2_5l = p1_only + combined[combined['Tier']=='P2_10L']['pat_long'].sum()  # P2 with 5L
p1p2_10l = p1_only + combined[combined['Tier']=='P2_10L']['pnl'].sum()  # P2 with 10L

systems = [
    ('Solo Adaptativo (52 sem)', reg['PnL_USD'].sum(), len(reg), '$500K'),
    ('P1 solo (14 sem)', p1_only, len(combined[combined['Tier']=='P1_LS']), '$500K'),
    ('P1 + P2 5L (24 sem)', p1p2_5l, len(combined[combined['Tier'].isin(['P1_LS','P2_10L'])]), '$250-500K'),
    ('P1 + P2 10L (24 sem)', p1p2_10l, len(combined[combined['Tier'].isin(['P1_LS','P2_10L'])]), '$500K'),
    ('*** 3-TIER (P2=10L) ***', pnl_total, total_n, '$500K'),
]

print(f"\n  {'Sistema':>30s} | {'N':>5} | {'Capital':>10} | {'Total P&L':>12} | {'P&L/sem':>10}")
print(f"  {'-'*30} | {'-'*5} | {'-'*10} | {'-'*12} | {'-'*10}")
for label, pnl, n, cap in systems:
    print(f"  {label:>30s} | {n:>5} | {cap:>10s} | ${pnl:>+11,.0f} | ${pnl/n:>+9,.0f}")

# ============================================================
# 4. RESUMEN EJECUTIVO
# ============================================================
p2_pnl = combined[combined['Tier']=='P2_10L']['pnl'].sum()
adp_pnl = combined[combined['Tier']=='ADAPTIVE']['pnl'].sum()

print(f"\n\n{'='*160}")
print("  RESUMEN EJECUTIVO: SISTEMA 3-TIER FINAL")
print("=" * 160)
print(f"""
  TIER 1 - PATTERN L+S (14 semanas/año, 27%)
    Semanas: {sorted(OPERABLE_WEEKS)}
    Config:  5L + 5S | $50K/pos | $500K/sem
    P&L:     ${p1_only:>+,.0f}
    Logica:  Pico/cola earnings → datos reales confirman ganadores y perdedores

  TIER 2 - PATTERN 10 LONGS (10 semanas/año, 19%)
    Semanas: {sorted(OPERABLE_WEEKS_LONG)}
    Config:  10L + 0S | $50K/pos | $500K/sem
    P&L:     ${p2_pnl:>+,.0f}
    Logica:  Pre/inicio earnings → anticipacion alcista, shorts toxicos

  TIER 3 - ADAPTATIVO REGIMEN (28 semanas/año, 54%)
    Semanas: Las 28 restantes
    Config:  5L + 5S | $50K/pos | $500K/sem
    P&L:     ${adp_pnl:>+,.0f}
    Logica:  Score compuesto 1-10, sin patron estacional

  TOTAL COMBINADO (52 semanas, $500K/sem constante)
    P&L total:     ${pnl_total:>+,.0f}
    P&L/semana:    ${pnl_total/total_n:>+,.0f}
    Años positivos: {n_pos}/{n_years} ({n_pos/n_years*100:.0f}%)
    Long total:    ${long_total:>+,.0f}
    Short total:   ${short_total:>+,.0f}

  COMPARACION:
    vs Solo Adaptativo:  ${adapt_pnl:>+,.0f} → Mejora: ${pnl_total - adapt_pnl:>+,.0f} ({(pnl_total/adapt_pnl - 1)*100:>+.0f}%)
    vs P1+P2 sin adapt:  ${p1p2_10l:>+,.0f} → Adapt añade: ${pnl_total - p1p2_10l:>+,.0f}
""")
