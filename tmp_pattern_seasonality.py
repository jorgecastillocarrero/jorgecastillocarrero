"""
Analisis de estacionalidad + patrones en el backtest pattern-based.
Busca semanas del año que consistentemente funcionan bien.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

df = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
df = df.dropna(subset=['PnL_USD'])
df['Signal_Date'] = pd.to_datetime(df['Signal_Date'])
df['Week_of_Year'] = df['Signal_Date'].dt.isocalendar().week.astype(int)
df['Month'] = df['Signal_Date'].dt.month
df['Year'] = df['Signal_Date'].dt.year

# ============================================================
# 1. RENDIMIENTO POR SEMANA DEL AÑO (1-52)
# ============================================================
print("=" * 160)
print("  1. RENDIMIENTO POR SEMANA DEL AÑO (estacionalidad)")
print("=" * 160)

by_week = df.groupby('Week_of_Year').agg(
    N=('PnL_USD', 'count'),
    Wins=('Win', 'sum'),
    Total_PnL=('PnL_USD', 'sum'),
    Mean_PnL=('PnL_USD', 'mean'),
    Std_PnL=('PnL_USD', 'std'),
    Long_PnL=('Long_PnL', 'sum'),
    Short_PnL=('Short_PnL', 'sum'),
    Mean_Ret=('Ret_Pct', 'mean'),
    Std_Ret=('Ret_Pct', 'std'),
    Long_Ret=('Long_Ret_Pct', 'mean'),
    Short_Ret=('Short_Ret_Pct', 'mean'),
).reset_index()
by_week['WR'] = (by_week['Wins'] / by_week['N'] * 100).round(0).astype(int)
by_week['Sharpe'] = np.where(by_week['Std_Ret'] > 0, by_week['Mean_Ret'] / by_week['Std_Ret'] * np.sqrt(52), 0)
by_week['PnL_per_week'] = by_week['Total_PnL'] / by_week['N']

# Consistency: how many years was this week positive?
consistency = df.groupby(['Week_of_Year', 'Year']).agg(PnL=('PnL_USD', 'sum')).reset_index()
consistency['Pos'] = (consistency['PnL'] > 0).astype(int)
cons_by_week = consistency.groupby('Week_of_Year').agg(
    N_years=('Pos', 'count'),
    Pos_years=('Pos', 'sum'),
).reset_index()
cons_by_week['Consistency'] = (cons_by_week['Pos_years'] / cons_by_week['N_years'] * 100).round(0).astype(int)
by_week = by_week.merge(cons_by_week[['Week_of_Year', 'N_years', 'Pos_years', 'Consistency']], on='Week_of_Year')

print(f"\n  {'Sem':>4} | {'N':>3} | {'WR%':>4} | {'Total P&L':>11} | {'P&L/sem':>9} | {'Ret%':>7} | {'Sharpe':>6} | {'Long P&L':>11} | {'Short P&L':>11} | {'L ret%':>7} | {'S ret%':>7} | {'Consist':>8} | {'Pos/Tot':>7}")
print(f"  {'-'*4} | {'-'*3} | {'-'*4} | {'-'*11} | {'-'*9} | {'-'*7} | {'-'*6} | {'-'*11} | {'-'*11} | {'-'*7} | {'-'*7} | {'-'*8} | {'-'*7}")

for _, r in by_week.sort_values('Week_of_Year').iterrows():
    w = int(r['Week_of_Year'])
    flag = " ***" if r['Consistency'] >= 60 and r['Total_PnL'] > 0 else (" !!!" if r['Consistency'] >= 60 and r['Total_PnL'] < 0 else "")
    print(f"  {w:>4} | {int(r['N']):>3} | {int(r['WR']):>3}% | ${r['Total_PnL']:>+10,.0f} | ${r['PnL_per_week']:>+8,.0f} | {r['Mean_Ret']:>+6.2f}% | {r['Sharpe']:>+5.2f} | ${r['Long_PnL']:>+10,.0f} | ${r['Short_PnL']:>+10,.0f} | {r['Long_Ret']:>+6.2f}% | {r['Short_Ret']:>+6.2f}% | {int(r['Consistency']):>6}% | {int(r['Pos_years']):>2}/{int(r['N_years']):>2}{flag}")

# ============================================================
# 2. TOP SEMANAS CONSISTENTEMENTE POSITIVAS
# ============================================================
print(f"\n\n{'='*160}")
print("  2. SEMANAS CONSISTENTEMENTE POSITIVAS (>=55% anos positivos AND P&L > 0)")
print("=" * 160)

good_weeks = by_week[(by_week['Consistency'] >= 55) & (by_week['Total_PnL'] > 0)].sort_values('Total_PnL', ascending=False)
print(f"\n  {len(good_weeks)} semanas encontradas:\n")
print(f"  {'Sem':>4} | {'Meses':>10} | {'N':>3} | {'WR%':>4} | {'Total P&L':>11} | {'Sharpe':>6} | {'Consist':>8} | {'Long P&L':>11} | {'Short P&L':>11}")
print(f"  {'-'*4} | {'-'*10} | {'-'*3} | {'-'*4} | {'-'*11} | {'-'*6} | {'-'*8} | {'-'*11} | {'-'*11}")

import datetime
for _, r in good_weeks.iterrows():
    w = int(r['Week_of_Year'])
    # Approximate month for this week
    approx = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=w-1)
    month_str = approx.strftime('%b %d')
    print(f"  {w:>4} | ~{month_str:>8} | {int(r['N']):>3} | {int(r['WR']):>3}% | ${r['Total_PnL']:>+10,.0f} | {r['Sharpe']:>+5.2f} | {int(r['Consistency']):>6}% | ${r['Long_PnL']:>+10,.0f} | ${r['Short_PnL']:>+10,.0f}")

# ============================================================
# 3. RENDIMIENTO POR MES
# ============================================================
print(f"\n\n{'='*160}")
print("  3. RENDIMIENTO POR MES")
print("=" * 160)

by_month = df.groupby('Month').agg(
    N=('PnL_USD', 'count'),
    Wins=('Win', 'sum'),
    Total_PnL=('PnL_USD', 'sum'),
    Mean_PnL=('PnL_USD', 'mean'),
    Long_PnL=('Long_PnL', 'sum'),
    Short_PnL=('Short_PnL', 'sum'),
    Mean_Ret=('Ret_Pct', 'mean'),
    Std_Ret=('Ret_Pct', 'std'),
).reset_index()
by_month['WR'] = (by_month['Wins'] / by_month['N'] * 100).round(0).astype(int)
by_month['Sharpe'] = np.where(by_month['Std_Ret'] > 0, by_month['Mean_Ret'] / by_month['Std_Ret'] * np.sqrt(52), 0)

month_names = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',
               7:'Julio',8:'Agosto',9:'Sep',10:'Octubre',11:'Nov',12:'Dic'}

print(f"\n  {'Mes':>3} {'Nombre':>10} | {'N':>4} | {'WR%':>4} | {'Total P&L':>12} | {'Long P&L':>12} | {'Short P&L':>12} | {'Ret%':>7} | {'Sharpe':>6}")
print(f"  {'-'*3} {'-'*10} | {'-'*4} | {'-'*4} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*7} | {'-'*6}")
for _, r in by_month.iterrows():
    m = int(r['Month'])
    print(f"  {m:>3} {month_names[m]:>10} | {int(r['N']):>4} | {int(r['WR']):>3}% | ${r['Total_PnL']:>+11,.0f} | ${r['Long_PnL']:>+11,.0f} | ${r['Short_PnL']:>+11,.0f} | {r['Mean_Ret']:>+6.2f}% | {r['Sharpe']:>+5.2f}")

# ============================================================
# 4. HEATMAP: MES x AÑO
# ============================================================
print(f"\n\n{'='*160}")
print("  4. HEATMAP P&L: MES x AÑO")
print("=" * 160)

pivot = df.groupby(['Year', 'Month'])['PnL_USD'].sum().unstack(fill_value=0)
months_present = sorted(pivot.columns)

header = f"  {'Ano':>6}"
for m in months_present:
    header += f" | {month_names.get(m, str(m)):>8}"
header += f" | {'TOTAL':>10}"
print(f"\n{header}")
print(f"  {'-'*6}" + "".join(f" | {'-'*8}" for _ in months_present) + f" | {'-'*10}")

for year in sorted(pivot.index):
    row_str = f"  {year:>6}"
    total = 0
    for m in months_present:
        val = pivot.loc[year, m] if m in pivot.columns else 0
        total += val
        row_str += f" | ${val:>+7,.0f}"
    row_str += f" | ${total:>+9,.0f}"
    print(row_str)

# ============================================================
# 5. DETALLE POR AÑO DE LAS SEMANAS "BUENAS"
# ============================================================
print(f"\n\n{'='*160}")
print("  5. DETALLE: SEMANAS BUENAS POR AÑO (las de consistencia >= 55%)")
print("=" * 160)

good_week_nums = set(good_weeks['Week_of_Year'].astype(int).values)
if len(good_week_nums) > 0:
    df_good = df[df['Week_of_Year'].isin(good_week_nums)]
    df_bad = df[~df['Week_of_Year'].isin(good_week_nums)]

    print(f"\n  Semanas 'buenas' (estacionales): {sorted(good_week_nums)}")
    print(f"  Total semanas buenas: {len(df_good)} ({len(df_good)/len(df)*100:.0f}%)")
    print(f"  Total semanas restantes: {len(df_bad)} ({len(df_bad)/len(df)*100:.0f}%)")

    for label, subset in [("SEMANAS BUENAS (estacionales)", df_good), ("SEMANAS RESTANTES", df_bad)]:
        total_pnl = subset['PnL_USD'].sum()
        wins = subset['Win'].sum()
        n = len(subset)
        wr = wins / n * 100 if n > 0 else 0
        rets = subset['Ret_Pct'].values / 100
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
        long_pnl = subset['Long_PnL'].sum()
        short_pnl = subset['Short_PnL'].sum()

        print(f"\n  --- {label} ---")
        print(f"  N={n} | WR={wr:.0f}% | P&L=${total_pnl:>+,.0f} | Long=${long_pnl:>+,.0f} | Short=${short_pnl:>+,.0f} | Sharpe={sharpe:>+.2f}")

        # Year by year
        by_year = subset.groupby('Year').agg(
            N=('PnL_USD', 'count'), Wins=('Win', 'sum'),
            PnL=('PnL_USD', 'sum'), Long=('Long_PnL', 'sum'), Short=('Short_PnL', 'sum'),
        ).reset_index()
        by_year['WR'] = (by_year['Wins'] / by_year['N'] * 100).round(0).astype(int)

        print(f"  {'Ano':>6} | {'N':>3} | {'WR%':>4} | {'P&L':>11} | {'Long':>11} | {'Short':>11}")
        for _, r in by_year.iterrows():
            print(f"  {int(r['Year']):>6} | {int(r['N']):>3} | {int(r['WR']):>3}% | ${r['PnL']:>+10,.0f} | ${r['Long']:>+10,.0f} | ${r['Short']:>+10,.0f}")

# ============================================================
# 6. EARNINGS SEASON vs NO-EARNINGS
# ============================================================
print(f"\n\n{'='*160}")
print("  6. EARNINGS SEASON vs FUERA DE EARNINGS")
print("=" * 160)

# Earnings season: weeks ~3-8 (Jan-mid Feb) and ~14-20 (Apr-May) and ~27-33 (Jul-Aug) and ~40-46 (Oct-Nov)
def is_earnings_season(week):
    return (3 <= week <= 8) or (14 <= week <= 20) or (27 <= week <= 33) or (40 <= week <= 46)

df['Earnings_Season'] = df['Week_of_Year'].apply(is_earnings_season)

for label, mask in [("EARNINGS SEASON", df['Earnings_Season']), ("FUERA EARNINGS", ~df['Earnings_Season'])]:
    subset = df[mask]
    total_pnl = subset['PnL_USD'].sum()
    wins = subset['Win'].sum()
    n = len(subset)
    wr = wins / n * 100 if n > 0 else 0
    rets = subset['Ret_Pct'].values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
    long_pnl = subset['Long_PnL'].sum()
    short_pnl = subset['Short_PnL'].sum()
    print(f"\n  {label}: N={n} ({n/len(df)*100:.0f}%) | WR={wr:.0f}% | P&L=${total_pnl:>+,.0f} | Long=${long_pnl:>+,.0f} | Short=${short_pnl:>+,.0f} | Sharpe={sharpe:>+.2f}")

# ============================================================
# 7. ANALISIS LADO LONG vs SHORT POR PERIODO
# ============================================================
print(f"\n\n{'='*160}")
print("  7. LADO LONG SOLO vs SHORT SOLO (que pasa si eliminamos un lado?)")
print("=" * 160)

for label, col in [("SOLO LONGS", "Long_PnL"), ("SOLO SHORTS", "Short_PnL")]:
    pnls = df[col].dropna()
    total = pnls.sum()
    rets_col = 'Long_Ret_Pct' if 'Long' in label else 'Short_Ret_Pct'
    rets = df[rets_col].dropna().values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
    wins = (pnls > 0).sum()
    wr = wins / len(pnls) * 100
    print(f"  {label}: P&L=${total:>+,.0f} | Sharpe={sharpe:>+.2f} | WR={wr:.0f}%")

# ============================================================
# 8. CORRELACION ENTRE LADOS: semanas donde AMBOS ganan
# ============================================================
print(f"\n\n{'='*160}")
print("  8. CORRELACION LONG-SHORT: cuando ambos lados ganan")
print("=" * 160)

df['Long_Win'] = df['Long_PnL'] > 0
df['Short_Win'] = df['Short_PnL'] > 0
df['Both_Win'] = df['Long_Win'] & df['Short_Win']
df['Both_Lose'] = (~df['Long_Win']) & (~df['Short_Win'])
df['Long_Only_Win'] = df['Long_Win'] & (~df['Short_Win'])
df['Short_Only_Win'] = (~df['Long_Win']) & df['Short_Win']

n = len(df)
bw = df['Both_Win'].sum()
bl = df['Both_Lose'].sum()
low = df['Long_Only_Win'].sum()
sow = df['Short_Only_Win'].sum()

print(f"\n  Ambos ganan:       {bw:>5} ({bw/n*100:.1f}%) -> P&L medio: ${df[df['Both_Win']]['PnL_USD'].mean():>+,.0f}")
print(f"  Solo Long gana:    {low:>5} ({low/n*100:.1f}%) -> P&L medio: ${df[df['Long_Only_Win']]['PnL_USD'].mean():>+,.0f}")
print(f"  Solo Short gana:   {sow:>5} ({sow/n*100:.1f}%) -> P&L medio: ${df[df['Short_Only_Win']]['PnL_USD'].mean():>+,.0f}")
print(f"  Ambos pierden:     {bl:>5} ({bl/n*100:.1f}%) -> P&L medio: ${df[df['Both_Lose']]['PnL_USD'].mean():>+,.0f}")

# Cuando ambos ganan - que semanas del año son?
both_win_weeks = df[df['Both_Win']].groupby('Week_of_Year').size().reset_index(name='Count')
both_win_weeks = both_win_weeks.sort_values('Count', ascending=False).head(15)
print(f"\n  TOP 15 semanas del año donde ambos lados ganan mas frecuentemente:")
for _, r in both_win_weeks.iterrows():
    w = int(r['Week_of_Year'])
    approx = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=w-1)
    total_bw = df[(df['Both_Win']) & (df['Week_of_Year'] == w)]
    total_all = df[df['Week_of_Year'] == w]
    rate = len(total_bw) / len(total_all) * 100
    avg_pnl = total_bw['PnL_USD'].mean()
    print(f"    Sem {w:>2} (~{approx.strftime('%b %d')}): {int(r['Count']):>2} veces de {len(total_all)} ({rate:.0f}%) | P&L medio: ${avg_pnl:>+,.0f}")

# ============================================================
# 9. CLUSTERS TEMPORALES: periodos multi-semana consecutivos
# ============================================================
print(f"\n\n{'='*160}")
print("  9. MEJORES PERIODOS DE 4 SEMANAS CONSECUTIVAS (ventana rodante)")
print("=" * 160)

# Rolling 4-week P&L by week_of_year
weekly_avg = by_week.set_index('Week_of_Year').sort_index()
rolling4 = []
for start_w in range(1, 50):
    weeks_in_window = list(range(start_w, start_w + 4))
    subset = weekly_avg.loc[weekly_avg.index.isin(weeks_in_window)]
    if len(subset) >= 3:
        total = subset['Total_PnL'].sum()
        avg_sharpe = subset['Sharpe'].mean()
        avg_wr = subset['WR'].mean()
        avg_consist = subset['Consistency'].mean()
        rolling4.append({
            'Start_Week': start_w, 'End_Week': start_w + 3,
            'PnL': total, 'Avg_Sharpe': avg_sharpe, 'Avg_WR': avg_wr,
            'Avg_Consistency': avg_consist,
        })

r4 = pd.DataFrame(rolling4).sort_values('PnL', ascending=False)

print(f"\n  TOP 10 mejores periodos de 4 semanas:")
print(f"  {'Sem':>8} | {'Aprox':>16} | {'P&L':>12} | {'Sharpe':>7} | {'WR%':>5} | {'Consist%':>9}")
print(f"  {'-'*8} | {'-'*16} | {'-'*12} | {'-'*7} | {'-'*5} | {'-'*9}")
for _, r in r4.head(10).iterrows():
    s = int(r['Start_Week']); e = int(r['End_Week'])
    d1 = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=s-1)
    d2 = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=e-1)
    print(f"  {s:>3}-{e:<3} | {d1.strftime('%b %d')}-{d2.strftime('%b %d')} | ${r['PnL']:>+11,.0f} | {r['Avg_Sharpe']:>+6.2f} | {r['Avg_WR']:>4.0f}% | {r['Avg_Consistency']:>7.0f}%")

print(f"\n  BOTTOM 10 peores periodos de 4 semanas:")
print(f"  {'Sem':>8} | {'Aprox':>16} | {'P&L':>12} | {'Sharpe':>7} | {'WR%':>5} | {'Consist%':>9}")
print(f"  {'-'*8} | {'-'*16} | {'-'*12} | {'-'*7} | {'-'*5} | {'-'*9}")
for _, r in r4.tail(10).iterrows():
    s = int(r['Start_Week']); e = int(r['End_Week'])
    d1 = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=s-1)
    d2 = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=e-1)
    print(f"  {s:>3}-{e:<3} | {d1.strftime('%b %d')}-{d2.strftime('%b %d')} | ${r['PnL']:>+11,.0f} | {r['Avg_Sharpe']:>+6.2f} | {r['Avg_WR']:>4.0f}% | {r['Avg_Consistency']:>7.0f}%")

# ============================================================
# 10. RESUMEN: que semanas OPERAR y cuales NO
# ============================================================
print(f"\n\n{'='*160}")
print("  10. RESUMEN: SEMANAS PARA OPERAR vs EVITAR")
print("=" * 160)

# Define "operable" weeks: consistency >= 50% AND positive total P&L AND Sharpe > 0
operable = by_week[(by_week['Consistency'] >= 50) & (by_week['Total_PnL'] > 0) & (by_week['Sharpe'] > 0)]
avoid = by_week[(by_week['Consistency'] <= 35) | (by_week['Total_PnL'] < -5000)]

op_weeks = set(operable['Week_of_Year'].astype(int))
av_weeks = set(avoid['Week_of_Year'].astype(int))
neutral = set(range(1, 53)) - op_weeks - av_weeks

df_op = df[df['Week_of_Year'].isin(op_weeks)]
df_av = df[df['Week_of_Year'].isin(av_weeks)]
df_ne = df[df['Week_of_Year'].isin(neutral)]

print(f"\n  OPERAR ({len(op_weeks)} semanas del año, {len(df_op)} obs, {len(df_op)/len(df)*100:.0f}% del tiempo):")
print(f"    Semanas: {sorted(op_weeks)}")
rets_op = df_op['Ret_Pct'].values / 100
sh_op = (np.mean(rets_op) / np.std(rets_op)) * np.sqrt(52) if np.std(rets_op) > 0 else 0
print(f"    P&L: ${df_op['PnL_USD'].sum():>+,.0f} | WR: {df_op['Win'].sum()/len(df_op)*100:.0f}% | Sharpe: {sh_op:>+.2f}")
print(f"    Long: ${df_op['Long_PnL'].sum():>+,.0f} | Short: ${df_op['Short_PnL'].sum():>+,.0f}")

print(f"\n  EVITAR ({len(av_weeks)} semanas del año, {len(df_av)} obs, {len(df_av)/len(df)*100:.0f}% del tiempo):")
print(f"    Semanas: {sorted(av_weeks)}")
rets_av = df_av['Ret_Pct'].values / 100
sh_av = (np.mean(rets_av) / np.std(rets_av)) * np.sqrt(52) if np.std(rets_av) > 0 else 0
print(f"    P&L: ${df_av['PnL_USD'].sum():>+,.0f} | WR: {df_av['Win'].sum()/len(df_av)*100:.0f}% | Sharpe: {sh_av:>+.2f}")

print(f"\n  NEUTRAL ({len(neutral)} semanas del año, {len(df_ne)} obs, {len(df_ne)/len(df)*100:.0f}% del tiempo):")
rets_ne = df_ne['Ret_Pct'].values / 100
sh_ne = (np.mean(rets_ne) / np.std(rets_ne)) * np.sqrt(52) if np.std(rets_ne) > 0 else 0
print(f"    P&L: ${df_ne['PnL_USD'].sum():>+,.0f} | WR: {df_ne['Win'].sum()/len(df_ne)*100:.0f}% | Sharpe: {sh_ne:>+.2f}")

# What if we ONLY traded "operable" weeks?
print(f"\n\n  >>> SI SOLO OPERAMOS LAS SEMANAS 'BUENAS':")
if len(df_op) > 0:
    pnl_op = df_op['PnL_USD'].sum()
    weeks_op = len(df_op)
    pnl_pw = pnl_op / weeks_op
    ret_pw = pnl_pw / 500000 * 100
    print(f"      P&L total: ${pnl_op:>+,.0f} en {weeks_op} semanas ({weeks_op/len(df)*100:.0f}% del tiempo)")
    print(f"      P&L/semana: ${pnl_pw:>+,.0f} | Ret/sem: {ret_pw:>+.2f}% | Sharpe: {sh_op:>+.2f}")
    print(f"      Long: ${df_op['Long_PnL'].sum():>+,.0f} | Short: ${df_op['Short_PnL'].sum():>+,.0f}")

# ============================================================
# 11. SOLO LONGS EN SEMANAS BUENAS (eliminar shorts destructivos)
# ============================================================
print(f"\n\n{'='*160}")
print("  11. SOLO LONGS en semanas buenas (sin shorts)")
print("=" * 160)

if len(df_op) > 0:
    long_only_pnl = df_op['Long_PnL'].sum()
    long_only_rets = df_op['Long_Ret_Pct'].dropna().values / 100
    sh_lo = (np.mean(long_only_rets) / np.std(long_only_rets)) * np.sqrt(52) if np.std(long_only_rets) > 0 else 0
    wr_lo = (df_op['Long_PnL'] > 0).sum() / len(df_op) * 100
    print(f"  P&L: ${long_only_pnl:>+,.0f} | Sharpe: {sh_lo:>+.2f} | WR: {wr_lo:.0f}%")
    print(f"  P&L/sem: ${long_only_pnl/len(df_op):>+,.0f}")

    # Year by year
    print(f"\n  {'Ano':>6} | {'N':>3} | {'WR%':>4} | {'Long P&L':>11}")
    print(f"  {'-'*6} | {'-'*3} | {'-'*4} | {'-'*11}")
    by_year_lo = df_op.groupby('Year').agg(N=('Long_PnL', 'count'), Wins=('Long_Win', 'sum'), PnL=('Long_PnL', 'sum')).reset_index()
    by_year_lo['WR'] = (by_year_lo['Wins'] / by_year_lo['N'] * 100).round(0).astype(int)
    for _, r in by_year_lo.iterrows():
        print(f"  {int(r['Year']):>6} | {int(r['N']):>3} | {int(r['WR']):>3}% | ${r['PnL']:>+10,.0f}")

# ============================================================
# 12. ANALISIS PICKS: que simbolos aparecen mas en semanas buenas
# ============================================================
print(f"\n\n{'='*160}")
print("  12. TOP SIMBOLOS en semanas buenas")
print("=" * 160)

picks = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Picks')
picks['Signal_Date'] = pd.to_datetime(picks['Signal_Date'])
picks['Week_of_Year'] = picks['Signal_Date'].dt.isocalendar().week.astype(int)
picks = picks.dropna(subset=['PnL_USD'])

good_picks = picks[picks['Week_of_Year'].isin(op_weeks)]

for side in ['LONG', 'SHORT']:
    sp = good_picks[good_picks['Side'] == side]
    by_sym = sp.groupby('Symbol').agg(
        N=('PnL_USD', 'count'),
        Total_PnL=('PnL_USD', 'sum'),
        Avg_Ret=('Net_Ret_Pct', 'mean'),
        Wins=('PnL_USD', lambda x: (x > 0).sum()),
    ).reset_index()
    by_sym['WR'] = (by_sym['Wins'] / by_sym['N'] * 100).round(0)
    by_sym = by_sym[by_sym['N'] >= 3].sort_values('Total_PnL', ascending=False)

    print(f"\n  --- TOP 15 {side} (>=3 apariciones) ---")
    print(f"  {'Symbol':>8} | {'N':>4} | {'WR%':>4} | {'Total P&L':>11} | {'P&L/pick':>10} | {'Ret%':>7}")
    for _, r in by_sym.head(15).iterrows():
        pp = r['Total_PnL'] / r['N']
        print(f"  {r['Symbol']:>8} | {int(r['N']):>4} | {int(r['WR']):>3}% | ${r['Total_PnL']:>+10,.0f} | ${pp:>+9,.0f} | {r['Avg_Ret']:>+6.2f}%")

print(f"\n\n  Tiempo: {pd.Timestamp.now().strftime('%H:%M:%S')}")
