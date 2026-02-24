"""Export seasonal regime data to Excel for reference."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import datetime
from regime_pattern_seasonal import (OPERABLE_WEEKS, WEEK_DETAIL, CLUSTERS,
                                      LONG_WEIGHTS, SHORT_WEIGHTS, BACKTEST_STATS)

# Load backtest results
df = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
df = df.dropna(subset=['PnL_USD'])
df['Signal_Date'] = pd.to_datetime(df['Signal_Date'])
df['Week_of_Year'] = df['Signal_Date'].dt.isocalendar().week.astype(int)
df['Operable'] = df['Week_of_Year'].isin(OPERABLE_WEEKS)

picks = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Picks')
picks['Signal_Date'] = pd.to_datetime(picks['Signal_Date'])
picks['Week_of_Year'] = picks['Signal_Date'].dt.isocalendar().week.astype(int)
picks = picks.dropna(subset=['PnL_USD'])

# ============================================================
# Sheet 1: Semanas operables (solo las 302 buenas)
# ============================================================
df_op = df[df['Operable']].copy()
df_op['Cluster'] = df_op['Week_of_Year'].apply(
    lambda w: next((c for c, d in CLUSTERS.items() if w in d['weeks']), 'unknown'))

# ============================================================
# Sheet 2: Picks de semanas operables
# ============================================================
picks_op = picks[picks['Week_of_Year'].isin(OPERABLE_WEEKS)].copy()

# ============================================================
# Sheet 3: Resumen por semana del año (las 14 operables)
# ============================================================
week_rows = []
for w in sorted(OPERABLE_WEEKS):
    detail = WEEK_DETAIL[w]
    sub = df_op[df_op['Week_of_Year'] == w]
    n = len(sub)
    wins = sub['Win'].sum()
    wr = wins / n * 100 if n > 0 else 0
    week_rows.append({
        'Week': w,
        'Approx_Date': detail['approx'],
        'Cluster': detail['cluster'],
        'N_Years': n,
        'Wins': int(wins),
        'Win_Rate': round(wr, 0),
        'Consistency': detail['consistency'],
        'Total_PnL': round(detail['pnl'], 0),
        'Sharpe': detail['sharpe'],
        'Long_PnL': round(sub['Long_PnL'].sum(), 0),
        'Short_PnL': round(sub['Short_PnL'].sum(), 0),
        'Avg_Long_Ret': round(sub['Long_Ret_Pct'].mean(), 2),
        'Avg_Short_Ret': round(sub['Short_Ret_Pct'].mean(), 2),
    })
df_weeks_detail = pd.DataFrame(week_rows)

# ============================================================
# Sheet 4: Resumen anual (solo semanas operables)
# ============================================================
annual_rows = []
cum = 0
for year in sorted(df_op['Year'].unique()):
    sub = df_op[df_op['Year'] == year]
    n = len(sub)
    wins = sub['Win'].sum()
    wr = wins / n * 100 if n > 0 else 0
    pnl = sub['PnL_USD'].sum()
    cum += pnl
    long_pnl = sub['Long_PnL'].sum()
    short_pnl = sub['Short_PnL'].sum()
    rets = sub['Ret_Pct'].values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if len(rets) > 1 and np.std(rets) > 0 else 0
    annual_rows.append({
        'Year': year,
        'Weeks_Traded': n,
        'Wins': int(wins),
        'Win_Rate': round(wr, 0),
        'Long_PnL': round(long_pnl, 0),
        'Short_PnL': round(short_pnl, 0),
        'Total_PnL': round(pnl, 0),
        'PnL_per_Week': round(pnl / n, 0) if n > 0 else 0,
        'Sharpe': round(sharpe, 2),
        'Cumulative_PnL': round(cum, 0),
    })
df_annual = pd.DataFrame(annual_rows)

# ============================================================
# Sheet 5: Scoring weights
# ============================================================
weight_rows = []
for col, w in LONG_WEIGHTS.items():
    weight_rows.append({'Side': 'LONG', 'Factor': col, 'Weight': w, 'Pattern': 'Pullback en tendencia fuerte'})
for col, w in SHORT_WEIGHTS.items():
    weight_rows.append({'Side': 'SHORT', 'Factor': col, 'Weight': w, 'Pattern': 'Overbought sin fundamento'})
df_weights = pd.DataFrame(weight_rows)

# ============================================================
# Sheet 6: Clusters
# ============================================================
cluster_rows = []
for cname, cdata in CLUSTERS.items():
    total_pnl = sum(WEEK_DETAIL[w]['pnl'] for w in cdata['weeks'])
    avg_sharpe = np.mean([WEEK_DETAIL[w]['sharpe'] for w in cdata['weeks']])
    avg_consist = np.mean([WEEK_DETAIL[w]['consistency'] for w in cdata['weeks']])
    cluster_rows.append({
        'Cluster': cname,
        'Description': cdata['desc'],
        'Weeks': ','.join(str(w) for w in cdata['weeks']),
        'N_Weeks': len(cdata['weeks']),
        'Total_PnL': total_pnl,
        'Avg_Sharpe': round(avg_sharpe, 2),
        'Avg_Consistency': round(avg_consist, 0),
    })
df_clusters = pd.DataFrame(cluster_rows)

# ============================================================
# Sheet 7: Comparativa 3 regimenes
# ============================================================
df_rest = df[~df['Operable']]
comp_rows = []

for label, subset in [("Pattern Estacional (27%)", df_op), ("Resto semanas (73%)", df_rest), ("Todas (100%)", df)]:
    n = len(subset)
    pnl = subset['PnL_USD'].sum()
    wins = subset['Win'].sum()
    wr = wins / n * 100 if n > 0 else 0
    rets = subset['Ret_Pct'].values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if len(rets) > 1 and np.std(rets) > 0 else 0
    comp_rows.append({
        'Regimen': label,
        'N_Weeks': n,
        'Win_Rate': round(wr, 0),
        'Total_PnL': round(pnl, 0),
        'PnL_per_Week': round(pnl / n, 0) if n > 0 else 0,
        'Sharpe': round(sharpe, 2),
        'Long_PnL': round(subset['Long_PnL'].sum(), 0),
        'Short_PnL': round(subset['Short_PnL'].sum(), 0),
    })
# Add regime-based for comparison
comp_rows.append({
    'Regimen': 'Regime-Based 5L+5S (ref)',
    'N_Weeks': 1118,
    'Win_Rate': 49,
    'Total_PnL': 1_320_989,
    'PnL_per_Week': 1_182,
    'Sharpe': 0.56,
    'Long_PnL': 1_497_817,
    'Short_PnL': -176_828,
})
df_comp = pd.DataFrame(comp_rows)

# Write Excel
excel_path = 'data/regime_pattern_seasonal.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_op.to_excel(writer, sheet_name='Semanas_Operables', index=False)
    picks_op.to_excel(writer, sheet_name='Picks_Operables', index=False)
    df_weeks_detail.to_excel(writer, sheet_name='Semanas_del_Ano', index=False)
    df_annual.to_excel(writer, sheet_name='Resumen_Anual', index=False)
    df_weights.to_excel(writer, sheet_name='Pattern_Weights', index=False)
    df_clusters.to_excel(writer, sheet_name='Clusters', index=False)
    df_comp.to_excel(writer, sheet_name='Comparativa', index=False)

print(f"Exportado: {excel_path}")
print(f"  - Semanas_Operables: {len(df_op)} filas (302 semanas)")
print(f"  - Picks_Operables: {len(picks_op)} filas")
print(f"  - Semanas_del_Ano: {len(df_weeks_detail)} filas (14 semanas)")
print(f"  - Resumen_Anual: {len(df_annual)} filas")
print(f"  - Pattern_Weights: {len(df_weights)} filas")
print(f"  - Clusters: {len(df_clusters)} filas")
print(f"  - Comparativa: {len(df_comp)} filas")

# Print summary comparison
print(f"\n{'='*100}")
print(f"  COMPARATIVA REGIMENES")
print(f"{'='*100}")
print(f"\n  {'Regimen':>35s} | {'N':>5} | {'WR%':>4} | {'P&L':>12} | {'P&L/sem':>10} | {'Sharpe':>6} | {'Long':>12} | {'Short':>12}")
print(f"  {'-'*35} | {'-'*5} | {'-'*4} | {'-'*12} | {'-'*10} | {'-'*6} | {'-'*12} | {'-'*12}")
for _, r in df_comp.iterrows():
    print(f"  {r['Regimen']:>35s} | {int(r['N_Weeks']):>5} | {int(r['Win_Rate']):>3}% | ${r['Total_PnL']:>+11,.0f} | ${r['PnL_per_Week']:>+9,.0f} | {r['Sharpe']:>+5.2f} | ${r['Long_PnL']:>+11,.0f} | ${r['Short_PnL']:>+11,.0f}")
