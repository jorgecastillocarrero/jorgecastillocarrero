"""
BACKTEST COMBINADO 3 NIVELES:
  Nivel 1 - Pattern 1 (14 sem): 5L + 5S con pattern multi-factor  → $748K
  Nivel 2 - Pattern 2 (10 sem): 5L solo longs con pattern         → $525K
  Nivel 3 - Adaptativo (28 sem): 5L + 5S por regimen de mercado   → ???

Las semanas Pattern 1 y 2 usan los resultados del pattern backtest.
Las semanas restantes usan los resultados del regime backtest adaptativo.
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
# Pattern backtest (para P1 y P2)
pat = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
pat = pat.dropna(subset=['PnL_USD'])
pat['Signal_Date'] = pd.to_datetime(pat['Signal_Date'])
pat['Week_of_Year'] = pat['Signal_Date'].dt.isocalendar().week.astype(int)

# Regime backtest (para adaptativo)
reg = pd.read_excel('data/regime_weekly_picks.xlsx', sheet_name='Semanas')
reg['Signal_Date'] = pd.to_datetime(reg['Signal_Date'])
reg['Week_of_Year'] = reg['Signal_Date'].dt.isocalendar().week.astype(int)

print("=" * 180)
print("  BACKTEST COMBINADO 3 NIVELES")
print("  P1: Pattern L+S (14 sem) | P2: Pattern Long-Only (10 sem) | Adaptativo: Regime Score (28 sem)")
print("=" * 180)

# ============================================================
# CONSTRUIR SEMANA A SEMANA
# ============================================================
# Merge por Signal_Date
merged = pat[['Signal_Date', 'Year', 'Week_of_Year', 'PnL_USD', 'Long_PnL', 'Short_PnL',
              'Ret_Pct', 'Long_Ret_Pct', 'Short_Ret_Pct', 'Win']].copy()
merged = merged.rename(columns={
    'PnL_USD': 'pat_pnl', 'Long_PnL': 'pat_long', 'Short_PnL': 'pat_short',
    'Ret_Pct': 'pat_ret', 'Long_Ret_Pct': 'pat_long_ret', 'Short_Ret_Pct': 'pat_short_ret',
    'Win': 'pat_win'
})

reg_cols = reg[['Signal_Date', 'Score', 'Long_Strategy', 'Short_Strategy',
                'PnL_USD', 'Long_PnL', 'Short_PnL', 'Ret_Pct', 'Win']].copy()
reg_cols = reg_cols.rename(columns={
    'PnL_USD': 'reg_pnl', 'Long_PnL': 'reg_long', 'Short_PnL': 'reg_short',
    'Ret_Pct': 'reg_ret', 'Win': 'reg_win'
})

combined = merged.merge(reg_cols, on='Signal_Date', how='inner')

# Clasificar cada semana
def classify_week(w):
    if w in OPERABLE_WEEKS:
        return 'P1_LS'
    elif w in OPERABLE_WEEKS_LONG:
        return 'P2_LONG'
    else:
        return 'ADAPTIVE'

combined['Tier'] = combined['Week_of_Year'].apply(classify_week)

# Calcular P&L combinado segun tier
def calc_combined_pnl(row):
    if row['Tier'] == 'P1_LS':
        # Pattern 1: 5L + 5S (full pattern P&L)
        return row['pat_pnl']
    elif row['Tier'] == 'P2_LONG':
        # Pattern 2: solo 5 longs del pattern (sin shorts)
        return row['pat_long']
    else:
        # Adaptativo: 5L + 5S del regime backtest
        return row['reg_pnl']

def calc_combined_long(row):
    if row['Tier'] == 'P1_LS':
        return row['pat_long']
    elif row['Tier'] == 'P2_LONG':
        return row['pat_long']
    else:
        return row['reg_long']

def calc_combined_short(row):
    if row['Tier'] == 'P1_LS':
        return row['pat_short']
    elif row['Tier'] == 'P2_LONG':
        return 0  # No shorts en P2
    else:
        return row['reg_short']

combined['comb_pnl'] = combined.apply(calc_combined_pnl, axis=1)
combined['comb_long'] = combined.apply(calc_combined_long, axis=1)
combined['comb_short'] = combined.apply(calc_combined_short, axis=1)
combined['comb_win'] = (combined['comb_pnl'] > 0).astype(int)

# Capital deployed per tier
def calc_capital(row):
    if row['Tier'] == 'P1_LS':
        return 500_000   # 5L + 5S × $50K
    elif row['Tier'] == 'P2_LONG':
        return 250_000   # 5L × $50K
    else:
        return 500_000   # 5L + 5S × $50K

combined['capital'] = combined.apply(calc_capital, axis=1)

# ============================================================
# 1. RESUMEN POR TIER
# ============================================================
print(f"\n  RESUMEN POR NIVEL")
print(f"  {'-'*160}")
print(f"\n  {'Tier':>15s} | {'N':>5} | {'%Tiempo':>8} | {'Total P&L':>12} | {'P&L/sem':>10} | {'Long P&L':>12} | {'Short P&L':>12} | {'WR%':>4} | {'Sharpe':>6}")
print(f"  {'-'*15} | {'-'*5} | {'-'*8} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*4} | {'-'*6}")

total_n = len(combined)
for tier, label in [('P1_LS', 'Pattern 1 (L+S)'), ('P2_LONG', 'Pattern 2 (Long)'), ('ADAPTIVE', 'Adaptativo')]:
    sub = combined[combined['Tier'] == tier]
    n = len(sub)
    pnl = sub['comb_pnl'].sum()
    long_pnl = sub['comb_long'].sum()
    short_pnl = sub['comb_short'].sum()
    wr = sub['comb_win'].sum() / n * 100
    pnl_pw = pnl / n

    # Sharpe: need returns
    if tier == 'P2_LONG':
        rets = sub['pat_long_ret'].dropna().values / 100
    elif tier == 'P1_LS':
        rets = sub['pat_ret'].values / 100
    else:
        rets = sub['reg_ret'].values / 100

    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if len(rets) > 1 and np.std(rets) > 0 else 0

    print(f"  {label:>15s} | {n:>5} | {n/total_n*100:>7.0f}% | ${pnl:>+11,.0f} | ${pnl_pw:>+9,.0f} | ${long_pnl:>+11,.0f} | ${short_pnl:>+11,.0f} | {wr:>3.0f}% | {sharpe:>+5.2f}")

# Total combinado
pnl_total = combined['comb_pnl'].sum()
long_total = combined['comb_long'].sum()
short_total = combined['comb_short'].sum()
wr_total = combined['comb_win'].sum() / total_n * 100
pnl_pw_total = pnl_total / total_n

print(f"  {'-'*15} | {'-'*5} | {'-'*8} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*4} | {'-'*6}")
print(f"  {'COMBINADO':>15s} | {total_n:>5} | {'100':>7s}% | ${pnl_total:>+11,.0f} | ${pnl_pw_total:>+9,.0f} | ${long_total:>+11,.0f} | ${short_total:>+11,.0f} | {wr_total:>3.0f}% |")

# ============================================================
# 2. COMPARATIVA CON ALTERNATIVAS
# ============================================================
print(f"\n\n{'='*180}")
print(f"  COMPARATIVA CON ALTERNATIVAS")
print(f"{'='*180}")

alternatives = []

# A) Solo adaptativo (todas las semanas)
a_pnl = reg['PnL_USD'].sum()
a_n = len(reg)
a_wr = reg['Win'].sum() / a_n * 100
a_rets = reg['Ret_Pct'].values / 100
a_sh = (np.mean(a_rets) / np.std(a_rets)) * np.sqrt(52) if np.std(a_rets) > 0 else 0
alternatives.append(('Solo Adaptativo (52 sem)', a_n, a_pnl, a_pnl/a_n, a_wr, a_sh, reg['Long_PnL'].sum(), reg['Short_PnL'].sum()))

# B) Solo Pattern (todas las semanas)
b_pnl = pat['pat_pnl' if 'pat_pnl' in pat.columns else 'PnL_USD'].sum() if 'pat_pnl' in pat.columns else pat['PnL_USD'].sum()
# Recalculate from original
pat2 = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
pat2 = pat2.dropna(subset=['PnL_USD'])
b_pnl = pat2['PnL_USD'].sum()
b_n = len(pat2)
b_wr = pat2['Win'].sum() / b_n * 100
b_rets = pat2['Ret_Pct'].values / 100
b_sh = (np.mean(b_rets) / np.std(b_rets)) * np.sqrt(52) if np.std(b_rets) > 0 else 0
alternatives.append(('Solo Pattern (52 sem)', b_n, b_pnl, b_pnl/b_n, b_wr, b_sh, pat2['Long_PnL'].sum(), pat2['Short_PnL'].sum()))

# C) Pattern 1 solo (14 sem)
c_sub = combined[combined['Tier'] == 'P1_LS']
c_pnl = c_sub['pat_pnl'].sum()
c_n = len(c_sub)
alternatives.append(('P1 solo (14 sem, 27%)', c_n, c_pnl, c_pnl/c_n if c_n > 0 else 0, 0, 0, c_sub['pat_long'].sum(), c_sub['pat_short'].sum()))

# D) P1 + P2 (24 sem, sin adaptativo)
d_sub = combined[combined['Tier'].isin(['P1_LS', 'P2_LONG'])]
d_pnl = d_sub['comb_pnl'].sum()
d_n = len(d_sub)
alternatives.append(('P1+P2 (24 sem, 46%)', d_n, d_pnl, d_pnl/d_n if d_n > 0 else 0, 0, 0, d_sub['comb_long'].sum(), d_sub['comb_short'].sum()))

# E) COMBINADO 3 NIVELES
alternatives.append(('*** COMBINADO 3 NIV ***', total_n, pnl_total, pnl_pw_total, wr_total, 0, long_total, short_total))

print(f"\n  {'Sistema':>30s} | {'N':>5} | {'Total P&L':>12} | {'P&L/sem':>10} | {'WR%':>4} | {'Sharpe':>6} | {'Long P&L':>12} | {'Short P&L':>12}")
print(f"  {'-'*30} | {'-'*5} | {'-'*12} | {'-'*10} | {'-'*4} | {'-'*6} | {'-'*12} | {'-'*12}")
for label, n, pnl, ppw, wr, sh, lp, sp in alternatives:
    print(f"  {label:>30s} | {n:>5} | ${pnl:>+11,.0f} | ${ppw:>+9,.0f} | {wr:>3.0f}% | {sh:>+5.2f} | ${lp:>+11,.0f} | ${sp:>+11,.0f}")

# ============================================================
# 3. AÑO A AÑO COMBINADO
# ============================================================
print(f"\n\n{'='*180}")
print(f"  AÑO A AÑO: COMBINADO 3 NIVELES")
print(f"{'='*180}")

print(f"\n  {'Año':>6} | {'P1':>3} | {'P1 P&L':>11} | {'P2':>3} | {'P2 Long':>11} | {'Adp':>3} | {'Adp P&L':>11} | {'Combo P&L':>12} | {'WR%':>4} | {'Acum':>14}")
print(f"  {'-'*6} | {'-'*3} | {'-'*11} | {'-'*3} | {'-'*11} | {'-'*3} | {'-'*11} | {'-'*12} | {'-'*4} | {'-'*14}")

cum = 0
n_pos = 0
for year in sorted(combined['Year'].unique()):
    ys = combined[combined['Year'] == year]

    p1s = ys[ys['Tier'] == 'P1_LS']
    p2s = ys[ys['Tier'] == 'P2_LONG']
    ads = ys[ys['Tier'] == 'ADAPTIVE']

    n1 = len(p1s); pnl1 = p1s['comb_pnl'].sum()
    n2 = len(p2s); pnl2 = p2s['comb_pnl'].sum()
    na = len(ads); pnla = ads['comb_pnl'].sum()

    combo = pnl1 + pnl2 + pnla
    cum += combo
    n_y = len(ys)
    wr_y = ys['comb_win'].sum() / n_y * 100 if n_y > 0 else 0
    if combo > 0: n_pos += 1

    print(f"  {year:>6} | {n1:>3} | ${pnl1:>+10,.0f} | {n2:>3} | ${pnl2:>+10,.0f} | {na:>3} | ${pnla:>+10,.0f} | ${combo:>+11,.0f} | {wr_y:>3.0f}% | ${cum:>+13,.0f}")

n_years = len(combined['Year'].unique())
print(f"  {'-'*6} | {'-'*3} | {'-'*11} | {'-'*3} | {'-'*11} | {'-'*3} | {'-'*11} | {'-'*12} | {'-'*4} | {'-'*14}")
print(f"  {'TOTAL':>6} |     | ${combined[combined['Tier']=='P1_LS']['comb_pnl'].sum():>+10,.0f} |     | ${combined[combined['Tier']=='P2_LONG']['comb_pnl'].sum():>+10,.0f} |     | ${combined[combined['Tier']=='ADAPTIVE']['comb_pnl'].sum():>+10,.0f} | ${pnl_total:>+11,.0f} |      | ${cum:>+13,.0f}")
print(f"\n  Años positivos: {n_pos}/{n_years} ({n_pos/n_years*100:.0f}%)")

# ============================================================
# 4. DRAWDOWN ANALYSIS
# ============================================================
print(f"\n\n{'='*180}")
print(f"  DRAWDOWN ANALYSIS")
print(f"{'='*180}")

# Equity curve
combined_sorted = combined.sort_values('Signal_Date')
equity = combined_sorted['comb_pnl'].cumsum().values
peak = np.maximum.accumulate(equity)
drawdown = equity - peak

max_dd = drawdown.min()
max_dd_idx = np.argmin(drawdown)
max_dd_date = combined_sorted.iloc[max_dd_idx]['Signal_Date']

# Find peak before max dd
peak_val = peak[max_dd_idx]
peak_idx = np.where(equity == peak_val)[0]
if len(peak_idx) > 0:
    peak_date = combined_sorted.iloc[peak_idx[0]]['Signal_Date']
else:
    peak_date = 'N/A'

print(f"\n  Max Drawdown: ${max_dd:>+,.0f}")
print(f"  Drawdown date: {max_dd_date}")
print(f"  Peak before DD: ${peak_val:>+,.0f} ({peak_date})")

# Recovery
if max_dd_idx < len(equity) - 1:
    recovery_idx = None
    for i in range(max_dd_idx, len(equity)):
        if equity[i] >= peak_val:
            recovery_idx = i
            break
    if recovery_idx:
        recovery_date = combined_sorted.iloc[recovery_idx]['Signal_Date']
        print(f"  Recovery date: {recovery_date}")
    else:
        print(f"  Recovery: NOT YET")

# Max consecutive losses
consec_loss = 0
max_consec_loss = 0
for pnl in combined_sorted['comb_pnl'].values:
    if pnl < 0:
        consec_loss += 1
        max_consec_loss = max(max_consec_loss, consec_loss)
    else:
        consec_loss = 0
print(f"  Max consecutive losses: {max_consec_loss} semanas")

# Win rate by tier
print(f"\n  Win Rate por tier:")
for tier in ['P1_LS', 'P2_LONG', 'ADAPTIVE']:
    sub = combined[combined['Tier'] == tier]
    n = len(sub)
    w = sub['comb_win'].sum()
    print(f"    {tier:>12s}: {w}/{n} ({w/n*100:.0f}%)")

# ============================================================
# 5. VALOR AÑADIDO DEL SISTEMA 3-TIER vs ALTERNATIVAS
# ============================================================
print(f"\n\n{'='*180}")
print(f"  VALOR AÑADIDO: 3-TIER vs ALTERNATIVAS")
print(f"{'='*180}")

adapt_only_pnl = reg['PnL_USD'].sum()
pattern_14_pnl = combined[combined['Tier'] == 'P1_LS']['pat_pnl'].sum()
p1p2_pnl = combined[combined['Tier'].isin(['P1_LS', 'P2_LONG'])]['comb_pnl'].sum()

print(f"\n  Solo Adaptativo (52 sem):    ${adapt_only_pnl:>+12,.0f}")
print(f"  3-Tier Combinado:            ${pnl_total:>+12,.0f}")
print(f"  Mejora vs Adaptativo:        ${pnl_total - adapt_only_pnl:>+12,.0f} ({(pnl_total/adapt_only_pnl - 1)*100:>+.0f}%)")
print(f"\n  Desglose del valor añadido:")

# What the adaptive would have gotten in P1 weeks
adapt_in_p1_weeks = combined[combined['Tier'] == 'P1_LS']['reg_pnl'].sum()
adapt_in_p2_weeks = combined[combined['Tier'] == 'P2_LONG']['reg_pnl'].sum()
adapt_in_adapt_weeks = combined[combined['Tier'] == 'ADAPTIVE']['reg_pnl'].sum()

pattern_in_p1_weeks = combined[combined['Tier'] == 'P1_LS']['comb_pnl'].sum()
pattern_in_p2_weeks = combined[combined['Tier'] == 'P2_LONG']['comb_pnl'].sum()

print(f"    En semanas P1: Adaptativo ${adapt_in_p1_weeks:>+10,.0f} → Pattern L+S ${pattern_in_p1_weeks:>+10,.0f} | Delta: ${pattern_in_p1_weeks - adapt_in_p1_weeks:>+10,.0f}")
print(f"    En semanas P2: Adaptativo ${adapt_in_p2_weeks:>+10,.0f} → Long Only  ${pattern_in_p2_weeks:>+10,.0f} | Delta: ${pattern_in_p2_weeks - adapt_in_p2_weeks:>+10,.0f}")
print(f"    En semanas Adp: (sin cambio)                          ${adapt_in_adapt_weeks:>+10,.0f}")

# ============================================================
# 6. ADAPTATIVO SOLO: DETALLE DE LAS 28 SEMANAS
# ============================================================
print(f"\n\n{'='*180}")
print(f"  DETALLE: ADAPTATIVO EN LAS 28 SEMANAS RESTANTES")
print(f"{'='*180}")

adapt_sub = combined[combined['Tier'] == 'ADAPTIVE']
n_adapt = len(adapt_sub)
adapt_pnl = adapt_sub['comb_pnl'].sum()
adapt_long = adapt_sub['comb_long'].sum()
adapt_short = adapt_sub['comb_short'].sum()
adapt_wr = adapt_sub['comb_win'].sum() / n_adapt * 100
adapt_rets = adapt_sub['reg_ret'].values / 100
adapt_sh = (np.mean(adapt_rets) / np.std(adapt_rets)) * np.sqrt(52) if np.std(adapt_rets) > 0 else 0

print(f"\n  Semanas adaptativas: {n_adapt} ({n_adapt/total_n*100:.0f}% del total)")
print(f"  P&L: ${adapt_pnl:>+,.0f} | P&L/sem: ${adapt_pnl/n_adapt:>+,.0f} | Sharpe: {adapt_sh:>+.2f} | WR: {adapt_wr:.0f}%")
print(f"  Long: ${adapt_long:>+,.0f} | Short: ${adapt_short:>+,.0f}")

# Score distribution in adaptive weeks
print(f"\n  Distribucion de Scores en semanas adaptativas:")
print(f"  {'Score':>6} | {'N':>4} | {'%':>5} | {'P&L':>11} | {'P&L/sem':>10} | {'WR%':>4}")
print(f"  {'-'*6} | {'-'*4} | {'-'*5} | {'-'*11} | {'-'*10} | {'-'*4}")
for score in sorted(adapt_sub['Score'].unique()):
    ss = adapt_sub[adapt_sub['Score'] == score]
    n_s = len(ss)
    pnl_s = ss['comb_pnl'].sum()
    wr_s = ss['comb_win'].sum() / n_s * 100
    print(f"  {score:>6} | {n_s:>4} | {n_s/n_adapt*100:>4.0f}% | ${pnl_s:>+10,.0f} | ${pnl_s/n_s:>+9,.0f} | {wr_s:>3.0f}%")

# ============================================================
# 7. RESUMEN EJECUTIVO FINAL
# ============================================================
print(f"\n\n{'='*180}")
print(f"  RESUMEN EJECUTIVO FINAL")
print(f"{'='*180}")
print(f"""
  SISTEMA 3-TIER DE TRADING SEMANAL S&P 500
  ==========================================

  TIER 1 - PATTERN L+S (14 semanas/año, 27% del tiempo)
    Semanas: {sorted(OPERABLE_WEEKS)}
    Config:  5L + 5S | $50K/pos | $500K/sem
    Pattern: Multi-factor scoring (pullback en tendencia + overbought sin fundamento)
    P&L:     ${combined[combined['Tier']=='P1_LS']['comb_pnl'].sum():>+,.0f}

  TIER 2 - PATTERN LONG-ONLY (10 semanas/año, 19% del tiempo)
    Semanas: {sorted(OPERABLE_WEEKS_LONG)}
    Config:  5L + 0S | $50K/pos | $250K/sem
    Pattern: Mismo scoring, solo lado long (shorts toxicos)
    P&L:     ${combined[combined['Tier']=='P2_LONG']['comb_pnl'].sum():>+,.0f}

  TIER 3 - ADAPTATIVO POR REGIMEN (28 semanas/año, 54% del tiempo)
    Semanas: Las 28 restantes
    Config:  5L + 5S | $50K/pos | $500K/sem
    Score:   Composite 1-10 (Market 30%, VIX 20%, EPS 15%, RSI/Sent/Beat 10% ea, Infl 5%)
    Strat:   LONG_TABLE + SHORT_TABLE segun score
    P&L:     ${combined[combined['Tier']=='ADAPTIVE']['comb_pnl'].sum():>+,.0f}

  TOTAL COMBINADO
    Semanas: {total_n} ({total_n/52:.0f} años)
    P&L:     ${pnl_total:>+,.0f}
    P&L/sem: ${pnl_pw_total:>+,.0f}
    Años positivos: {n_pos}/{n_years} ({n_pos/n_years*100:.0f}%)
    Long:    ${long_total:>+,.0f}
    Short:   ${short_total:>+,.0f}

  vs Solo Adaptativo: ${adapt_only_pnl:>+,.0f} → Mejora: ${pnl_total - adapt_only_pnl:>+,.0f}
""")
