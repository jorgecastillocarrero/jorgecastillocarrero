"""Estadisticos anuales detallados del regimen Pattern 2: Long-Only."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from regime_pattern_seasonal import OPERABLE_WEEKS
from regime_pattern_longonly import OPERABLE_WEEKS_LONG, CLUSTERS_LONG

df = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
df = df.dropna(subset=['PnL_USD'])
df['Signal_Date'] = pd.to_datetime(df['Signal_Date'])
df['Week_of_Year'] = df['Signal_Date'].dt.isocalendar().week.astype(int)

# Solo semanas Long-Only (Pattern 2)
df_lo = df[df['Week_of_Year'].isin(OPERABLE_WEEKS_LONG)].copy()

print("=" * 160)
print("  ESTADISTICOS ANUALES - REGIMEN PATTERN 2: LONG-ONLY (19% del tiempo, 10 semanas/año)")
print("  5L + 0S | $50K/posicion | $250K/semana | 0.3% cost")
print("=" * 160)

# ============================================================
# 1. AÑO A AÑO
# ============================================================
print(f"\n  {'Año':>6} | {'Sem':>4} | {'W':>3} | {'WR%':>4} | {'Long P&L':>12} | {'Sharpe L':>8} | {'Short P&L':>12} | {'Sh S':>6} | {'Total P&L':>12} | {'Solo Long':>12} | {'Acum Long':>14}")
print(f"  {'-'*6} | {'-'*4} | {'-'*3} | {'-'*4} | {'-'*12} | {'-'*8} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*12} | {'-'*14}")

cum_long = 0
cum_total = 0
n_years_long_pos = 0

yearly_data = {}

for year in sorted(df_lo['Year'].unique()):
    sub = df_lo[df_lo['Year'] == year]
    n = len(sub)
    long_pnl = sub['Long_PnL'].sum()
    short_pnl = sub['Short_PnL'].sum()
    total_pnl = sub['PnL_USD'].sum()
    wins = int(sub['Win'].sum())
    wr = wins / n * 100 if n > 0 else 0

    # Long win rate
    long_wins = (sub['Long_PnL'] > 0).sum()
    long_wr = long_wins / n * 100

    long_rets = sub['Long_Ret_Pct'].dropna().values / 100
    sharpe_l = (np.mean(long_rets) / np.std(long_rets)) * np.sqrt(52) if len(long_rets) > 1 and np.std(long_rets) > 0 else 0

    short_rets = sub['Short_Ret_Pct'].dropna().values / 100
    sharpe_s = (np.mean(short_rets) / np.std(short_rets)) * np.sqrt(52) if len(short_rets) > 1 and np.std(short_rets) > 0 else 0

    cum_long += long_pnl
    cum_total += total_pnl
    if long_pnl > 0: n_years_long_pos += 1

    yearly_data[year] = {'n': n, 'long_pnl': long_pnl, 'short_pnl': short_pnl,
                          'total_pnl': total_pnl, 'wr': wr, 'long_wr': long_wr,
                          'sharpe_l': sharpe_l, 'sharpe_s': sharpe_s, 'wins': wins}

    print(f"  {year:>6} | {n:>4} | {wins:>3} | {wr:>3.0f}% | ${long_pnl:>+11,.0f} | {sharpe_l:>+7.2f} | ${short_pnl:>+11,.0f} | {sharpe_s:>+5.2f} | ${total_pnl:>+11,.0f} | ${long_pnl:>+11,.0f} | ${cum_long:>+13,.0f}")

# Totals
n_total = len(df_lo)
long_total = df_lo['Long_PnL'].sum()
short_total = df_lo['Short_PnL'].sum()
total_total = df_lo['PnL_USD'].sum()
wins_total = int(df_lo['Win'].sum())
wr_total = wins_total / n_total * 100
long_rets_all = df_lo['Long_Ret_Pct'].dropna().values / 100
sharpe_l_all = (np.mean(long_rets_all) / np.std(long_rets_all)) * np.sqrt(52)
short_rets_all = df_lo['Short_Ret_Pct'].dropna().values / 100
sharpe_s_all = (np.mean(short_rets_all) / np.std(short_rets_all)) * np.sqrt(52)

print(f"  {'-'*6} | {'-'*4} | {'-'*3} | {'-'*4} | {'-'*12} | {'-'*8} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*12} | {'-'*14}")
print(f"  {'TOTAL':>6} | {n_total:>4} | {wins_total:>3} | {wr_total:>3.0f}% | ${long_total:>+11,.0f} | {sharpe_l_all:>+7.2f} | ${short_total:>+11,.0f} | {sharpe_s_all:>+5.2f} | ${total_total:>+11,.0f} | ${long_total:>+11,.0f} | ${cum_long:>+13,.0f}")

n_years = len(yearly_data)
print(f"\n  Resumen Pattern 2 (Solo Longs):")
print(f"    Años positivos (longs): {n_years_long_pos}/{n_years} ({n_years_long_pos/n_years*100:.0f}%)")
print(f"    Long P&L total:  ${long_total:>+,.0f}")
print(f"    Short P&L total: ${short_total:>+,.0f}  <-- TOXICO, por eso NO operamos shorts")
print(f"    Ahorro evitando shorts: ${abs(short_total):>,.0f}")
print(f"    Mejor año (longs):  {max(yearly_data.keys(), key=lambda y: yearly_data[y]['long_pnl'])} (${max(yearly_data[y]['long_pnl'] for y in yearly_data):>+,.0f})")
print(f"    Peor año (longs):   {min(yearly_data.keys(), key=lambda y: yearly_data[y]['long_pnl'])} (${min(yearly_data[y]['long_pnl'] for y in yearly_data):>+,.0f})")

# ============================================================
# 2. COMPARATIVA COMBINADA: P1 + P2
# ============================================================
print(f"\n\n{'='*160}")
print("  COMPARATIVA: Pattern 1 (L+S) + Pattern 2 (Solo Longs) + Combinado")
print("=" * 160)

df_p1 = df[df['Week_of_Year'].isin(OPERABLE_WEEKS)]
df_p2 = df[df['Week_of_Year'].isin(OPERABLE_WEEKS_LONG)]
all_weeks = OPERABLE_WEEKS | OPERABLE_WEEKS_LONG
df_all = df[df['Week_of_Year'].isin(all_weeks)]

# Para combinado: en P1 usamos L+S, en P2 solo L
# Simulacion: P&L combinado = P&L_P1 (L+S) + Long_PnL_P2
df_combo_p1 = df[df['Week_of_Year'].isin(OPERABLE_WEEKS)]['PnL_USD'].sum()  # L+S
df_combo_p2 = df[df['Week_of_Year'].isin(OPERABLE_WEEKS_LONG)]['Long_PnL'].sum()  # solo L
combo_pnl = df_combo_p1 + df_combo_p2

n_p1 = len(df_p1)
n_p2 = len(df_p2)

print(f"\n  {'Regimen':>35s} | {'N sem':>6} | {'%Tiempo':>8} | {'Total P&L':>12} | {'P&L/sem':>10} | {'Long P&L':>12} | {'Short P&L':>12}")
print(f"  {'-'*35} | {'-'*6} | {'-'*8} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12}")

# P1
p1_pnl = df_p1['PnL_USD'].sum()
p1_long = df_p1['Long_PnL'].sum()
p1_short = df_p1['Short_PnL'].sum()
print(f"  {'Pattern 1: L+S (14 sem)':>35s} | {n_p1:>6} | {n_p1/len(df)*100:>7.0f}% | ${p1_pnl:>+11,.0f} | ${p1_pnl/n_p1:>+9,.0f} | ${p1_long:>+11,.0f} | ${p1_short:>+11,.0f}")

# P2 (solo longs)
p2_long = df_p2['Long_PnL'].sum()
print(f"  {'Pattern 2: Solo Longs (10 sem)':>35s} | {n_p2:>6} | {n_p2/len(df)*100:>7.0f}% | ${p2_long:>+11,.0f} | ${p2_long/n_p2:>+9,.0f} | ${p2_long:>+11,.0f} | ${'0':>10s}")

# Combinado
n_combo = n_p1 + n_p2
print(f"  {'COMBINADO P1+P2':>35s} | {n_combo:>6} | {n_combo/len(df)*100:>7.0f}% | ${combo_pnl:>+11,.0f} | ${combo_pnl/n_combo:>+9,.0f} | ${p1_long+p2_long:>+11,.0f} | ${p1_short:>+11,.0f}")

# Todas las semanas (referencia)
all_pnl = df['PnL_USD'].sum()
all_long = df['Long_PnL'].sum()
all_short = df['Short_PnL'].sum()
print(f"  {'Todas (52 sem, referencia)':>35s} | {len(df):>6} | {'100':>7s}% | ${all_pnl:>+11,.0f} | ${all_pnl/len(df):>+9,.0f} | ${all_long:>+11,.0f} | ${all_short:>+11,.0f}")

# ============================================================
# 3. AÑO A AÑO COMBINADO
# ============================================================
print(f"\n\n{'='*160}")
print("  AÑO A AÑO COMBINADO: P1 (L+S) + P2 (Solo Longs)")
print("=" * 160)

print(f"\n  {'Año':>6} | {'P1 sem':>6} | {'P1 P&L':>12} | {'P2 sem':>6} | {'P2 Long':>12} | {'Combo P&L':>12} | {'Acum':>14}")
print(f"  {'-'*6} | {'-'*6} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*12} | {'-'*14}")

cum_combo = 0
n_combo_pos = 0
all_years = sorted(set(df_p1['Year'].unique()) | set(df_p2['Year'].unique()))

for year in all_years:
    s1 = df_p1[df_p1['Year'] == year]
    s2 = df_p2[df_p2['Year'] == year]
    n1 = len(s1)
    n2 = len(s2)
    pnl1 = s1['PnL_USD'].sum() if n1 > 0 else 0
    pnl2_long = s2['Long_PnL'].sum() if n2 > 0 else 0
    combo = pnl1 + pnl2_long
    cum_combo += combo
    if combo > 0: n_combo_pos += 1

    print(f"  {year:>6} | {n1:>6} | ${pnl1:>+11,.0f} | {n2:>6} | ${pnl2_long:>+11,.0f} | ${combo:>+11,.0f} | ${cum_combo:>+13,.0f}")

print(f"  {'-'*6} | {'-'*6} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*12} | {'-'*14}")
print(f"  {'TOTAL':>6} | {n_p1:>6} | ${p1_pnl:>+11,.0f} | {n_p2:>6} | ${p2_long:>+11,.0f} | ${combo_pnl:>+11,.0f} | ${cum_combo:>+13,.0f}")
print(f"\n  Años combinado positivos: {n_combo_pos}/{len(all_years)} ({n_combo_pos/len(all_years)*100:.0f}%)")

# ============================================================
# 4. CALENDARIO VISUAL
# ============================================================
print(f"\n\n{'='*160}")
print("  CALENDARIO SEMANAL: QUE OPERAR CADA SEMANA DEL AÑO")
print("=" * 160)

print(f"\n  {'Sem':>4} | {'Accion':>12} | {'Longs':>6} | {'Shorts':>6} | {'Capital':>10} | Notas")
print(f"  {'-'*4} | {'-'*12} | {'-'*6} | {'-'*6} | {'-'*10} | {'-'*40}")

for w in range(1, 54):
    if w in OPERABLE_WEEKS:
        action = "L+S (P1)"
        longs = "5"
        shorts = "5"
        capital = "$500K"
        notes = f"Pattern 1 - {next((c for c,d in __import__('regime_pattern_seasonal', fromlist=['CLUSTERS']).CLUSTERS.items() if w in d['weeks']), ('',''))[0]}"
    elif w in OPERABLE_WEEKS_LONG:
        action = "LONG (P2)"
        longs = "5"
        shorts = "0"
        capital = "$250K"
        notes = f"Pattern 2 - {next((c for c,d in CLUSTERS_LONG.items() if w in d['weeks']), ('',''))[0]}"
    else:
        action = "NO OPERAR"
        longs = "0"
        shorts = "0"
        capital = "$0"
        notes = ""
    print(f"  {w:>4} | {action:>12} | {longs:>6} | {shorts:>6} | {capital:>10} | {notes}")
