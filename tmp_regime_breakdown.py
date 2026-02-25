import pandas as pd
import numpy as np

df = pd.read_excel('data/regime_weekly_picks.xlsx', sheet_name='Semanas')
df = df.dropna(subset=['PnL_USD'])

# =============================================
# POR SCORE (situacion de mercado)
# =============================================
print("=" * 140)
print("  RESULTADOS POR REGIMEN DE MERCADO (Score 1=bearish ... 10=bullish)")
print("=" * 140)

by_score = df.groupby('Score').agg(
    Sem=('PnL_USD', 'count'),
    Wins=('Win', 'sum'),
    Long_PnL=('Long_PnL', 'sum'),
    Short_PnL=('Short_PnL', 'sum'),
    Total_PnL=('PnL_USD', 'sum'),
    Long_Ret_Avg=('Long_Ret_Pct', 'mean'),
    Short_Ret_Avg=('Short_Ret_Pct', 'mean'),
    Total_Ret_Avg=('Ret_Pct', 'mean'),
).reset_index()

by_score['WR'] = (by_score['Wins'] / by_score['Sem'] * 100).round(0).astype(int)

# Sharpe por score
sharpe_data = []
for sc in range(1, 11):
    mask = df['Score'] == sc
    lr = df.loc[mask, 'Long_Ret_Pct'].dropna() / 100
    sr = df.loc[mask, 'Short_Ret_Pct'].dropna() / 100
    tr = df.loc[mask, 'Ret_Pct'].dropna() / 100
    sl = lr.mean() / lr.std() * np.sqrt(52) if lr.std() > 0 else 0
    ss = sr.mean() / sr.std() * np.sqrt(52) if sr.std() > 0 else 0
    st = tr.mean() / tr.std() * np.sqrt(52) if tr.std() > 0 else 0
    sharpe_data.append({'Score': sc, 'Sharpe_L': sl, 'Sharpe_S': ss, 'Sharpe_T': st})
sharpe_df = pd.DataFrame(sharpe_data)

hdr = f"{'Score':>5} | {'Estrat L':>8} | {'Estrat S':>8} | {'Sem':>4} | {'WR%':>4} | {'Long P&L':>11} | {'L avg%':>7} | {'Sh L':>6} | {'Short P&L':>11} | {'S avg%':>7} | {'Sh S':>6} | {'Total P&L':>11} | {'T avg%':>7} | {'Sh T':>6}"
print(hdr)
print("-" * 140)

from backtest_regime_weekly_picks import LONG_TABLE, SHORT_TABLE

for _, r in by_score.iterrows():
    sc = int(r['Score'])
    sh = sharpe_df[sharpe_df['Score'] == sc].iloc[0]
    ls = LONG_TABLE[sc]
    ss = SHORT_TABLE[sc]
    sem = int(r['Sem'])
    wr = int(r['WR'])
    lp = r['Long_PnL']
    sp = r['Short_PnL']
    tp = r['Total_PnL']
    la = r['Long_Ret_Avg']
    sa = r['Short_Ret_Avg']
    ta = r['Total_Ret_Avg']
    print(f"{sc:>5} | {ls:>8} | {ss:>8} | {sem:>4} | {wr:>3}% | ${lp:>+10,.0f} | {la:>+6.2f}% | {sh['Sharpe_L']:>+5.2f} | ${sp:>+10,.0f} | {sa:>+6.2f}% | {sh['Sharpe_S']:>+5.2f} | ${tp:>+10,.0f} | {ta:>+6.2f}% | {sh['Sharpe_T']:>+5.2f}")

# Totales
print("-" * 140)
tl = by_score['Long_PnL'].sum()
ts = by_score['Short_PnL'].sum()
tt = by_score['Total_PnL'].sum()
la_all = df['Long_Ret_Pct'].mean()
sa_all = df['Short_Ret_Pct'].mean()
ta_all = df['Ret_Pct'].mean()
lr_all = df['Long_Ret_Pct'].dropna() / 100
sr_all = df['Short_Ret_Pct'].dropna() / 100
tr_all = df['Ret_Pct'].dropna() / 100
sl_all = lr_all.mean() / lr_all.std() * np.sqrt(52) if lr_all.std() > 0 else 0
ss_all = sr_all.mean() / sr_all.std() * np.sqrt(52) if sr_all.std() > 0 else 0
st_all = tr_all.mean() / tr_all.std() * np.sqrt(52) if tr_all.std() > 0 else 0
n_all = len(df)
wr_all = int(df['Win'].sum() / n_all * 100)
print(f"{'TOTAL':>5} | {'':>8} | {'':>8} | {n_all:>4} | {wr_all:>3}% | ${tl:>+10,.0f} | {la_all:>+6.2f}% | {sl_all:>+5.2f} | ${ts:>+10,.0f} | {sa_all:>+6.2f}% | {ss_all:>+5.2f} | ${tt:>+10,.0f} | {ta_all:>+6.2f}% | {st_all:>+5.2f}")

# =============================================
# POR GRUPO DE REGIMEN
# =============================================
print(f"\n\n{'=' * 140}")
print("  RESULTADOS POR GRUPO DE REGIMEN")
print("=" * 140)

def assign_group(score):
    if score <= 3: return '1. Bearish (1-3)'
    elif score <= 6: return '2. Neutral (4-6)'
    else: return '3. Bullish (7-10)'

df['Grupo'] = df['Score'].apply(assign_group)

by_group = df.groupby('Grupo').agg(
    Sem=('PnL_USD', 'count'),
    Wins=('Win', 'sum'),
    Long_PnL=('Long_PnL', 'sum'),
    Short_PnL=('Short_PnL', 'sum'),
    Total_PnL=('PnL_USD', 'sum'),
    Long_Ret_Avg=('Long_Ret_Pct', 'mean'),
    Short_Ret_Avg=('Short_Ret_Pct', 'mean'),
    Total_Ret_Avg=('Ret_Pct', 'mean'),
).reset_index()

by_group['WR'] = (by_group['Wins'] / by_group['Sem'] * 100).round(0).astype(int)

print(f"{'Grupo':>22} | {'Sem':>4} | {'WR%':>4} | {'Long P&L':>12} | {'L avg%':>7} | {'Short P&L':>12} | {'S avg%':>7} | {'Total P&L':>12} | {'T avg%':>7}")
print("-" * 120)

for _, r in by_group.iterrows():
    g = r['Grupo']
    sem = int(r['Sem'])
    wr = int(r['WR'])
    lp = r['Long_PnL']
    sp = r['Short_PnL']
    tp = r['Total_PnL']
    la = r['Long_Ret_Avg']
    sa = r['Short_Ret_Avg']
    ta = r['Total_Ret_Avg']
    print(f"{g:>22} | {sem:>4} | {wr:>3}% | ${lp:>+11,.0f} | {la:>+6.2f}% | ${sp:>+11,.0f} | {sa:>+6.2f}% | ${tp:>+11,.0f} | {ta:>+6.2f}%")

print("-" * 120)
print(f"{'TOTAL':>22} | {n_all:>4} | {wr_all:>3}% | ${tl:>+11,.0f} | {la_all:>+6.2f}% | ${ts:>+11,.0f} | {sa_all:>+6.2f}% | ${tt:>+11,.0f} | {ta_all:>+6.2f}%")

# =============================================
# CONTRIBUCION POR ESTRATEGIA
# =============================================
print(f"\n\n{'=' * 140}")
print("  CONTRIBUCION POR ESTRATEGIA (como fue usada)")
print("=" * 140)

# Long strategies used
picks = pd.read_excel('data/regime_weekly_picks.xlsx', sheet_name='Picks')
picks = picks.dropna(subset=['PnL_USD'])

for side in ['LONG', 'SHORT']:
    side_picks = picks[picks['Side'] == side]
    by_strat = side_picks.groupby('Strategy').agg(
        N_picks=('PnL_USD', 'count'),
        Total_PnL=('PnL_USD', 'sum'),
        Avg_Ret=('Net_Ret_Pct', 'mean'),
        Wins=('PnL_USD', lambda x: (x > 0).sum()),
    ).reset_index()
    by_strat['WR'] = (by_strat['Wins'] / by_strat['N_picks'] * 100).round(0).astype(int)
    by_strat = by_strat.sort_values('Total_PnL', ascending=False)

    print(f"\n  --- {side} ---")
    print(f"  {'Estrategia':>10} | {'Picks':>6} | {'WR%':>4} | {'P&L Total':>12} | {'P&L/pick':>10} | {'Ret avg%':>8}")
    print(f"  {'-'*10} | {'-'*6} | {'-'*4} | {'-'*12} | {'-'*10} | {'-'*8}")
    for _, r in by_strat.iterrows():
        n = int(r['N_picks'])
        wr = int(r['WR'])
        tp = r['Total_PnL']
        pp = tp / n
        ar = r['Avg_Ret']
        print(f"  {r['Strategy']:>10} | {n:>6} | {wr:>3}% | ${tp:>+11,.0f} | ${pp:>+9,.0f} | {ar:>+7.2f}%")
