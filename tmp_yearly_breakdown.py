import pandas as pd
import numpy as np

df = pd.read_excel('data/regime_weekly_picks.xlsx', sheet_name='Semanas')

yearly = df.groupby('Year').agg(
    Sem=('PnL_USD', 'count'),
    Wins=('Win', 'sum'),
    Long_PnL=('Long_PnL', 'sum'),
    Short_PnL=('Short_PnL', 'sum'),
    Total_PnL=('PnL_USD', 'sum'),
).reset_index()

yearly['WR'] = (yearly['Wins'] / yearly['Sem'] * 100).round(0).astype(int)
yearly['Long_Ret'] = (yearly['Long_PnL'] / (yearly['Sem'] * 250000) * 100).round(1)
yearly['Short_Ret'] = (yearly['Short_PnL'] / (yearly['Sem'] * 250000) * 100).round(1)
yearly['Total_Ret'] = (yearly['Total_PnL'] / (yearly['Sem'] * 500000) * 100).round(1)

cum = 0
hdr = f"{'Año':>5} | {'Sem':>3} | {'WR%':>4} | {'Long P&L':>12} | {'Long%':>7} | {'Short P&L':>12} | {'Short%':>7} | {'Total P&L':>12} | {'Total%':>7} | {'Acumulado':>14}"
sep = f"{'-'*5} | {'-'*3} | {'-'*4} | {'-'*12} | {'-'*7} | {'-'*12} | {'-'*7} | {'-'*12} | {'-'*7} | {'-'*14}"
print(hdr)
print(sep)

for _, r in yearly.iterrows():
    cum += r['Total_PnL']
    yr = int(r['Year'])
    sem = int(r['Sem'])
    wr = int(r['WR'])
    lp = r['Long_PnL']
    lr = r['Long_Ret']
    sp = r['Short_PnL']
    sr = r['Short_Ret']
    tp = r['Total_PnL']
    tr = r['Total_Ret']
    print(f"{yr:>5} | {sem:>3} | {wr:>3}% | ${lp:>+11,.0f} | {lr:>+6.1f}% | ${sp:>+11,.0f} | {sr:>+6.1f}% | ${tp:>+11,.0f} | {tr:>+6.1f}% | ${cum:>+13,.0f}")

print()
tl = yearly['Long_PnL'].sum()
ts = yearly['Short_PnL'].sum()
tt = yearly['Total_PnL'].sum()
print(f"TOTAL Long:  ${tl:>+12,.0f}  ({tl/tt*100:.0f}% del total)")
print(f"TOTAL Short: ${ts:>+12,.0f}  ({ts/tt*100:.0f}% del total)")
print(f"TOTAL:       ${tt:>+12,.0f}")

# Sharpe por lado
long_weekly = df['Long_Ret_Pct'].dropna() / 100
short_weekly = df['Short_Ret_Pct'].dropna() / 100
total_weekly = df['Ret_Pct'].dropna() / 100

sharpe_l = long_weekly.mean() / long_weekly.std() * np.sqrt(52) if long_weekly.std() > 0 else 0
sharpe_s = short_weekly.mean() / short_weekly.std() * np.sqrt(52) if short_weekly.std() > 0 else 0
sharpe_t = total_weekly.mean() / total_weekly.std() * np.sqrt(52) if total_weekly.std() > 0 else 0

print(f"\nSharpe Long:  {sharpe_l:>+.2f}")
print(f"Sharpe Short: {sharpe_s:>+.2f}")
print(f"Sharpe Total: {sharpe_t:>+.2f}")

# Años positivos por lado
long_pos = (yearly['Long_PnL'] > 0).sum()
short_pos = (yearly['Short_PnL'] > 0).sum()
total_pos = (yearly['Total_PnL'] > 0).sum()
n = len(yearly)
print(f"\nAños positivos Long:  {long_pos}/{n}")
print(f"Años positivos Short: {short_pos}/{n}")
print(f"Años positivos Total: {total_pos}/{n}")
