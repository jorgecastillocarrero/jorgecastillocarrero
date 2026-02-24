"""Comparativa año a año: Adaptativo (52 sem) vs P1+P2 (24 sem, sin adaptativo)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from regime_pattern_seasonal import OPERABLE_WEEKS
from regime_pattern_longonly import OPERABLE_WEEKS_LONG

# Ratios 10L/5L por semana (del backtest tmp_p2_10longs.py)
WEEK_10L_PNL = {4:83276, 5:14928, 6:32777, 11:85411, 12:31773, 25:40981, 28:139608, 31:3449, 44:26447, 50:52469}
WEEK_5L_PNL  = {4:44666, 5:19069, 6:50368, 11:44063, 12:44433, 25:36020, 28:73464, 31:6597, 44:28812, 50:42490}

def scale_10l(week, pnl_5l):
    if week in WEEK_5L_PNL and WEEK_5L_PNL[week] != 0:
        return pnl_5l * (WEEK_10L_PNL[week] / WEEK_5L_PNL[week])
    return pnl_5l * 1.311

# Cargar datos
pat = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
pat = pat.dropna(subset=['PnL_USD'])
pat['Signal_Date'] = pd.to_datetime(pat['Signal_Date'])
pat['Week_of_Year'] = pat['Signal_Date'].dt.isocalendar().week.astype(int)

reg = pd.read_excel('data/regime_weekly_picks.xlsx', sheet_name='Semanas')
reg['Signal_Date'] = pd.to_datetime(reg['Signal_Date'])
reg['Week_of_Year'] = reg['Signal_Date'].dt.isocalendar().week.astype(int)

CAPITAL = 500_000

print("=" * 180)
print("  COMPARATIVA AÑO A AÑO: ADAPTATIVO (52 sem) vs P1+P2 (24 sem)")
print("  Adaptativo: 5L+5S regime score, $500K/sem, TODAS las semanas")
print("  P1+P2: P1=5L+5S pattern (14 sem) + P2=10L pattern (10 sem), $500K/sem, 24 sem/año")
print("=" * 180)

# ============================================================
# Calcular por año
# ============================================================
all_years = sorted(set(reg['Year'].unique()) | set(pat['Year'].unique()))

print(f"\n  {'':>6} |{'ADAPTATIVO (52 sem)':^42s}|{'P1+P2 (24 sem)':^54s}| {'Comp':^20s}")
print(f"  {'Año':>6} | {'Sem':>4} | {'P&L':>11} | {'Ret%':>7} | {'WR%':>4} | {'Sharpe':>6} | {'Sem':>4} | {'P1 P&L':>11} | {'P2 10L':>11} | {'Total':>11} | {'Ret%':>7} | {'WR%':>4} | {'Mejor':>8} | {'Delta':>11}")
print(f"  {'-'*6} | {'-'*4} | {'-'*11} | {'-'*7} | {'-'*4} | {'-'*6} | {'-'*4} | {'-'*11} | {'-'*11} | {'-'*11} | {'-'*7} | {'-'*4} | {'-'*8} | {'-'*11}")

cum_adapt = 0
cum_p1p2 = 0
n_adapt_wins = 0
n_p1p2_wins = 0
yearly_adapt = []
yearly_p1p2 = []

for year in all_years:
    # --- ADAPTATIVO ---
    ra = reg[reg['Year'] == year]
    na = len(ra)
    pnl_a = ra['PnL_USD'].sum() if na > 0 else 0
    wr_a = ra['Win'].sum() / na * 100 if na > 0 else 0
    ret_a = pnl_a / CAPITAL * 100  # return on capital per year
    rets_a = ra['Ret_Pct'].values / 100 if na > 0 else np.array([0])
    sh_a = (np.mean(rets_a) / np.std(rets_a)) * np.sqrt(52) if na > 1 and np.std(rets_a) > 0 else 0
    cum_adapt += pnl_a
    yearly_adapt.append(pnl_a)

    # --- P1 + P2 ---
    pa = pat[pat['Year'] == year]

    # P1: semanas Pattern 1, usar PnL completo (L+S)
    p1 = pa[pa['Week_of_Year'].isin(OPERABLE_WEEKS)]
    pnl_p1 = p1['PnL_USD'].sum() if len(p1) > 0 else 0
    n_p1 = len(p1)

    # P2: semanas Pattern 2, escalar Long_PnL a 10 longs
    p2 = pa[pa['Week_of_Year'].isin(OPERABLE_WEEKS_LONG)]
    pnl_p2 = 0
    for _, row in p2.iterrows():
        pnl_p2 += scale_10l(row['Week_of_Year'], row['Long_PnL'])
    n_p2 = len(p2)

    pnl_p1p2 = pnl_p1 + pnl_p2
    n_p1p2 = n_p1 + n_p2
    ret_p1p2 = pnl_p1p2 / CAPITAL * 100

    # Win rate P1+P2
    wins_p1 = p1['Win'].sum() if len(p1) > 0 else 0
    wins_p2 = (p2['Long_PnL'] > 0).sum() if len(p2) > 0 else 0  # approx for 10L
    wr_p1p2 = (wins_p1 + wins_p2) / n_p1p2 * 100 if n_p1p2 > 0 else 0

    cum_p1p2 += pnl_p1p2
    yearly_p1p2.append(pnl_p1p2)

    # Quien gana
    if pnl_p1p2 > pnl_a:
        mejor = "P1+P2"
        n_p1p2_wins += 1
    else:
        mejor = "ADAPT"
        n_adapt_wins += 1

    delta = pnl_p1p2 - pnl_a

    print(f"  {year:>6} | {na:>4} | ${pnl_a:>+10,.0f} | {ret_a:>+6.1f}% | {wr_a:>3.0f}% | {sh_a:>+5.2f} | {n_p1p2:>4} | ${pnl_p1:>+10,.0f} | ${pnl_p2:>+10,.0f} | ${pnl_p1p2:>+10,.0f} | {ret_p1p2:>+6.1f}% | {wr_p1p2:>3.0f}% | {mejor:>8s} | ${delta:>+10,.0f}")

# Totales
pnl_a_total = sum(yearly_adapt)
pnl_p_total = sum(yearly_p1p2)
n_years = len(all_years)

print(f"  {'-'*6} | {'-'*4} | {'-'*11} | {'-'*7} | {'-'*4} | {'-'*6} | {'-'*4} | {'-'*11} | {'-'*11} | {'-'*11} | {'-'*7} | {'-'*4} | {'-'*8} | {'-'*11}")
print(f"  {'TOTAL':>6} |      | ${pnl_a_total:>+10,.0f} |         |      |        |      |             |             | ${pnl_p_total:>+10,.0f} |         |      |          | ${pnl_p_total-pnl_a_total:>+10,.0f}")

print(f"\n  Años que gana Adaptativo: {n_adapt_wins}/{n_years} ({n_adapt_wins/n_years*100:.0f}%)")
print(f"  Años que gana P1+P2:      {n_p1p2_wins}/{n_years} ({n_p1p2_wins/n_years*100:.0f}%)")

# ============================================================
# ACUMULADOS
# ============================================================
print(f"\n\n{'='*180}")
print("  CURVA ACUMULADA")
print("=" * 180)

print(f"\n  {'Año':>6} | {'Acum Adapt':>14} | {'Acum P1+P2':>14} | {'Delta acum':>14}")
print(f"  {'-'*6} | {'-'*14} | {'-'*14} | {'-'*14}")

ca = 0; cp = 0
for i, year in enumerate(all_years):
    ca += yearly_adapt[i]
    cp += yearly_p1p2[i]
    print(f"  {year:>6} | ${ca:>+13,.0f} | ${cp:>+13,.0f} | ${cp-ca:>+13,.0f}")

# ============================================================
# ESTADISTICAS
# ============================================================
print(f"\n\n{'='*180}")
print("  ESTADISTICAS COMPARATIVAS")
print("=" * 180)

ya = np.array(yearly_adapt)
yp = np.array(yearly_p1p2)

# Annual returns on $500K capital
ret_a_ann = ya / CAPITAL * 100
ret_p_ann = yp / CAPITAL * 100

print(f"\n  {'Metrica':>30s} | {'Adaptativo':>14s} | {'P1+P2':>14s}")
print(f"  {'-'*30} | {'-'*14} | {'-'*14}")
print(f"  {'P&L total':>30s} | ${ya.sum():>+13,.0f} | ${yp.sum():>+13,.0f}")
print(f"  {'P&L medio anual':>30s} | ${ya.mean():>+13,.0f} | ${yp.mean():>+13,.0f}")
print(f"  {'Mejor año':>30s} | ${ya.max():>+13,.0f} | ${yp.max():>+13,.0f}")
print(f"  {'Peor año':>30s} | ${ya.min():>+13,.0f} | ${yp.min():>+13,.0f}")
print(f"  {'Mediana anual':>30s} | ${np.median(ya):>+13,.0f} | ${np.median(yp):>+13,.0f}")
print(f"  {'Desv std anual':>30s} | ${np.std(ya):>+13,.0f} | ${np.std(yp):>+13,.0f}")
print(f"  {'Años positivos':>30s} | {(ya>0).sum():>10}/{n_years:<3} | {(yp>0).sum():>10}/{n_years:<3}")
print(f"  {'Años > $50K':>30s} | {(ya>50000).sum():>10}/{n_years:<3} | {(yp>50000).sum():>10}/{n_years:<3}")
print(f"  {'Años negativos':>30s} | {(ya<0).sum():>10}/{n_years:<3} | {(yp<0).sum():>10}/{n_years:<3}")
print(f"  {'Peor drawdown anual':>30s} | ${ya.min():>+13,.0f} | ${yp.min():>+13,.0f}")
print(f"  {'Ret% medio anual':>30s} | {ret_a_ann.mean():>+12.1f}% | {ret_p_ann.mean():>+12.1f}%")
print(f"  {'Ret% mejor año':>30s} | {ret_a_ann.max():>+12.1f}% | {ret_p_ann.max():>+12.1f}%")
print(f"  {'Ret% peor año':>30s} | {ret_a_ann.min():>+12.1f}% | {ret_p_ann.min():>+12.1f}%")
print(f"  {'Sharpe anual (ret/vol)':>30s} | {np.mean(ret_a_ann)/np.std(ret_a_ann):>+13.2f} | {np.mean(ret_p_ann)/np.std(ret_p_ann):>+13.2f}")
print(f"  {'Semanas operadas/año':>30s} | {'~50':>14s} | {'~24':>14s}")
print(f"  {'Capital en riesgo':>30s} | {'$500K×50sem':>14s} | {'$500K×24sem':>14s}")
print(f"  {'Eficiencia (P&L/capital exp)':>30s} | {ya.sum()/(CAPITAL*50*n_years)*100:>+12.3f}% | {yp.sum()/(CAPITAL*24*n_years)*100:>+12.3f}%")

# Max cumulative drawdown
ca_arr = np.cumsum(ya)
cp_arr = np.cumsum(yp)
ca_peak = np.maximum.accumulate(ca_arr)
cp_peak = np.maximum.accumulate(cp_arr)
ca_dd = (ca_arr - ca_peak).min()
cp_dd = (cp_arr - cp_peak).min()

print(f"  {'Max DD acumulado':>30s} | ${ca_dd:>+13,.0f} | ${cp_dd:>+13,.0f}")

# Consecutive losing years
def max_consec_neg(arr):
    m = 0; c = 0
    for v in arr:
        if v < 0: c += 1; m = max(m, c)
        else: c = 0
    return m

print(f"  {'Años negativos seguidos':>30s} | {max_consec_neg(ya):>14} | {max_consec_neg(yp):>14}")
