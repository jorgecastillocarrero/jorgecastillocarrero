"""Estadisticos anuales detallados del regimen pattern estacional (27% del tiempo)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from regime_pattern_seasonal import OPERABLE_WEEKS

df = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
df = df.dropna(subset=['PnL_USD'])
df['Signal_Date'] = pd.to_datetime(df['Signal_Date'])
df['Week_of_Year'] = df['Signal_Date'].dt.isocalendar().week.astype(int)

# Solo semanas operables
df = df[df['Week_of_Year'].isin(OPERABLE_WEEKS)].copy()

print("=" * 160)
print("  ESTADISTICOS ANUALES - REGIMEN PATTERN ESTACIONAL (27% del tiempo, 14 semanas/año)")
print("  5L + 5S | $50K/posicion | $500K/semana | 0.3% cost")
print("=" * 160)

yearly = {}
for year in sorted(df['Year'].unique()):
    sub = df[df['Year'] == year]
    n = len(sub)
    wins = int(sub['Win'].sum())
    wr = wins / n * 100 if n > 0 else 0
    pnl = sub['PnL_USD'].sum()
    long_pnl = sub['Long_PnL'].sum()
    short_pnl = sub['Short_PnL'].sum()
    pnl_pw = pnl / n if n > 0 else 0
    ret_pw = pnl_pw / 500_000 * 100

    rets = sub['Ret_Pct'].values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if len(rets) > 1 and np.std(rets) > 0 else 0

    long_rets = sub['Long_Ret_Pct'].dropna().values / 100
    short_rets = sub['Short_Ret_Pct'].dropna().values / 100
    sharpe_l = (np.mean(long_rets) / np.std(long_rets)) * np.sqrt(52) if len(long_rets) > 1 and np.std(long_rets) > 0 else 0
    sharpe_s = (np.mean(short_rets) / np.std(short_rets)) * np.sqrt(52) if len(short_rets) > 1 and np.std(short_rets) > 0 else 0

    # Max DD within year
    peak = 0; running = 0; max_dd = 0
    for p in sub['PnL_USD'].values:
        running += p
        if running > peak: peak = running
        dd = running - peak
        if dd < max_dd: max_dd = dd

    yearly[year] = {
        'n': n, 'wins': wins, 'wr': wr, 'pnl': pnl,
        'long_pnl': long_pnl, 'short_pnl': short_pnl,
        'pnl_pw': pnl_pw, 'ret_pw': ret_pw, 'sharpe': sharpe,
        'sharpe_l': sharpe_l, 'sharpe_s': sharpe_s, 'max_dd': max_dd,
    }

# Print table
print(f"\n  {'Ano':>6} | {'Sem':>4} | {'W':>3} | {'WR%':>4} | {'Long P&L':>12} | {'Sh L':>6} | {'Short P&L':>12} | {'Sh S':>6} | {'Total P&L':>12} | {'P&L/sem':>10} | {'Ret/sem':>8} | {'Sharpe':>6} | {'MaxDD':>10} | {'Acum':>14}")
print(f"  {'-'*6} | {'-'*4} | {'-'*3} | {'-'*4} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*6} | {'-'*10} | {'-'*14}")

cum = 0
n_years_pos = 0
n_years_long_pos = 0
n_years_short_pos = 0

for year in sorted(yearly.keys()):
    y = yearly[year]
    cum += y['pnl']
    if y['pnl'] > 0: n_years_pos += 1
    if y['long_pnl'] > 0: n_years_long_pos += 1
    if y['short_pnl'] > 0: n_years_short_pos += 1
    print(f"  {year:>6} | {y['n']:>4} | {y['wins']:>3} | {y['wr']:>3.0f}% | ${y['long_pnl']:>+11,.0f} | {y['sharpe_l']:>+5.2f} | ${y['short_pnl']:>+11,.0f} | {y['sharpe_s']:>+5.2f} | ${y['pnl']:>+11,.0f} | ${y['pnl_pw']:>+9,.0f} | {y['ret_pw']:>+7.2f}% | {y['sharpe']:>+5.2f} | ${y['max_dd']:>+9,.0f} | ${cum:>+13,.0f}")

# Totals
n_total = len(df)
wins_total = int(df['Win'].sum())
wr_total = wins_total / n_total * 100
pnl_total = df['PnL_USD'].sum()
long_total = df['Long_PnL'].sum()
short_total = df['Short_PnL'].sum()
pnl_pw_total = pnl_total / n_total
ret_pw_total = pnl_pw_total / 500_000 * 100
rets_all = df['Ret_Pct'].values / 100
sharpe_all = (np.mean(rets_all) / np.std(rets_all)) * np.sqrt(52)
long_rets_all = df['Long_Ret_Pct'].dropna().values / 100
short_rets_all = df['Short_Ret_Pct'].dropna().values / 100
sharpe_l_all = (np.mean(long_rets_all) / np.std(long_rets_all)) * np.sqrt(52)
sharpe_s_all = (np.mean(short_rets_all) / np.std(short_rets_all)) * np.sqrt(52)

# Max DD global
peak = 0; running = 0; max_dd_global = 0
for p in df.sort_values('Signal_Date')['PnL_USD'].values:
    running += p
    if running > peak: peak = running
    dd = running - peak
    if dd < max_dd_global: max_dd_global = dd

print(f"  {'-'*6} | {'-'*4} | {'-'*3} | {'-'*4} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*6} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*6} | {'-'*10} | {'-'*14}")
print(f"  {'TOTAL':>6} | {n_total:>4} | {wins_total:>3} | {wr_total:>3.0f}% | ${long_total:>+11,.0f} | {sharpe_l_all:>+5.2f} | ${short_total:>+11,.0f} | {sharpe_s_all:>+5.2f} | ${pnl_total:>+11,.0f} | ${pnl_pw_total:>+9,.0f} | {ret_pw_total:>+7.2f}% | {sharpe_all:>+5.2f} | ${max_dd_global:>+9,.0f} | ${cum:>+13,.0f}")

n_years = len(yearly)
print(f"\n  Resumen:")
print(f"    Anos positivos (total):  {n_years_pos}/{n_years} ({n_years_pos/n_years*100:.0f}%)")
print(f"    Anos positivos (longs):  {n_years_long_pos}/{n_years} ({n_years_long_pos/n_years*100:.0f}%)")
print(f"    Anos positivos (shorts): {n_years_short_pos}/{n_years} ({n_years_short_pos/n_years*100:.0f}%)")
print(f"    Mejor ano:  {max(yearly.keys(), key=lambda y: yearly[y]['pnl'])} (${max(yearly[y]['pnl'] for y in yearly):>+,.0f})")
print(f"    Peor ano:   {min(yearly.keys(), key=lambda y: yearly[y]['pnl'])} (${min(yearly[y]['pnl'] for y in yearly):>+,.0f})")
print(f"    Ret anualiz: {ret_pw_total * 52:.1f}%")
print(f"    Max DD glob: ${max_dd_global:>+,.0f}")
