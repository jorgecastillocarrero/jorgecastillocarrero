"""
Buscar semanas adicionales que se puedan añadir al regimen estacional.
Analiza las 38 semanas NO incluidas para ver si alguna merece entrar.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from regime_pattern_seasonal import OPERABLE_WEEKS

df = pd.read_excel('data/pattern_weekly_picks.xlsx', sheet_name='Semanas')
df = df.dropna(subset=['PnL_USD'])
df['Signal_Date'] = pd.to_datetime(df['Signal_Date'])
df['Week_of_Year'] = df['Signal_Date'].dt.isocalendar().week.astype(int)
df['Year'] = df['Signal_Date'].dt.year

# Semanas NO incluidas actualmente
excluded_weeks = sorted(set(range(1, 54)) - OPERABLE_WEEKS)

print("=" * 180)
print("  ANALISIS DE SEMANAS CANDIDATAS PARA AMPLIAR EL REGIMEN")
print(f"  Actualmente: {sorted(OPERABLE_WEEKS)} (14 semanas)")
print(f"  Excluidas a analizar: {len(excluded_weeks)} semanas")
print("=" * 180)

# ============================================================
# 1. ANALISIS DETALLADO DE CADA SEMANA EXCLUIDA
# ============================================================
print(f"\n  {'Sem':>4} | {'N':>3} | {'WR%':>4} | {'Total P&L':>11} | {'P&L/sem':>9} | {'Ret%':>7} | {'Sharpe':>6} | {'Long P&L':>11} | {'Sh L':>6} | {'Short P&L':>11} | {'Sh S':>6} | {'Consist':>8} | {'Pos/Tot':>7} | {'Stable':>6} | Veredicto")
print(f"  {'-'*4} | {'-'*3} | {'-'*4} | {'-'*11} | {'-'*9} | {'-'*7} | {'-'*6} | {'-'*11} | {'-'*6} | {'-'*11} | {'-'*6} | {'-'*8} | {'-'*7} | {'-'*6} | {'-'*30}")

candidates = []

for w in range(1, 54):
    sub = df[df['Week_of_Year'] == w]
    if len(sub) < 5:
        continue

    n = len(sub)
    wins = int(sub['Win'].sum())
    wr = wins / n * 100
    pnl = sub['PnL_USD'].sum()
    pnl_pw = pnl / n
    long_pnl = sub['Long_PnL'].sum()
    short_pnl = sub['Short_PnL'].sum()

    rets = sub['Ret_Pct'].values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0

    long_rets = sub['Long_Ret_Pct'].dropna().values / 100
    short_rets = sub['Short_Ret_Pct'].dropna().values / 100
    sharpe_l = (np.mean(long_rets) / np.std(long_rets)) * np.sqrt(52) if len(long_rets) > 1 and np.std(long_rets) > 0 else 0
    sharpe_s = (np.mean(short_rets) / np.std(short_rets)) * np.sqrt(52) if len(short_rets) > 1 and np.std(short_rets) > 0 else 0

    # Consistency: % of years positive
    by_year = sub.groupby('Year')['PnL_USD'].sum()
    pos_years = (by_year > 0).sum()
    total_years = len(by_year)
    consistency = pos_years / total_years * 100

    # Stability: check recent vs old performance
    # Split at 2015
    old = sub[sub['Year'] <= 2015]
    recent = sub[sub['Year'] > 2015]
    old_pnl = old['PnL_USD'].sum() if len(old) > 0 else 0
    recent_pnl = recent['PnL_USD'].sum() if len(recent) > 0 else 0
    # Stable if both halves are positive, or recent is strongly positive
    stable = "YES" if (recent_pnl > 0 and (old_pnl > -10000 or pnl > 30000)) else "no"

    # Max consecutive losses
    results = sub.sort_values('Signal_Date')['PnL_USD'].values
    max_consec_loss = 0
    consec = 0
    for r in results:
        if r < 0:
            consec += 1
            max_consec_loss = max(max_consec_loss, consec)
        else:
            consec = 0

    in_regime = "<<< YA" if w in OPERABLE_WEEKS else ""

    # Scoring for candidacy
    score = 0
    if pnl > 0: score += 2
    if pnl > 20000: score += 1
    if sharpe > 0.5: score += 2
    if sharpe > 1.0: score += 1
    if wr >= 50: score += 1
    if consistency >= 50: score += 2
    if consistency >= 60: score += 1
    if stable == "YES": score += 2
    if long_pnl > 0 and short_pnl > 0: score += 2  # both sides positive
    if max_consec_loss <= 4: score += 1

    verdict = ""
    if w in OPERABLE_WEEKS:
        verdict = "YA INCLUIDA"
    elif score >= 8:
        verdict = "*** INCLUIR ***"
    elif score >= 6:
        verdict = "** CANDIDATA **"
    elif score >= 4:
        verdict = "* POSIBLE *"
    else:
        verdict = "Descartar"

    candidates.append({
        'week': w, 'n': n, 'wr': wr, 'pnl': pnl, 'pnl_pw': pnl_pw,
        'sharpe': sharpe, 'long_pnl': long_pnl, 'short_pnl': short_pnl,
        'sharpe_l': sharpe_l, 'sharpe_s': sharpe_s,
        'consistency': consistency, 'pos_years': pos_years, 'total_years': total_years,
        'stable': stable, 'score': score, 'verdict': verdict,
        'in_regime': w in OPERABLE_WEEKS, 'recent_pnl': recent_pnl, 'old_pnl': old_pnl,
    })

    mean_ret = np.mean(rets) * 100

    print(f"  {w:>4} | {n:>3} | {wr:>3.0f}% | ${pnl:>+10,.0f} | ${pnl_pw:>+8,.0f} | {mean_ret:>+6.2f}% | {sharpe:>+5.2f} | ${long_pnl:>+10,.0f} | {sharpe_l:>+5.2f} | ${short_pnl:>+10,.0f} | {sharpe_s:>+5.2f} | {consistency:>6.0f}% | {pos_years:>2}/{total_years:>2} | {stable:>6} | {verdict}")

# ============================================================
# 2. RANKING DE CANDIDATAS (no incluidas)
# ============================================================
print(f"\n\n{'='*180}")
print("  RANKING DE CANDIDATAS (semanas no incluidas, ordenadas por score)")
print("=" * 180)

cands = [c for c in candidates if not c['in_regime']]
cands.sort(key=lambda x: (-x['score'], -x['sharpe']))

print(f"\n  {'#':>3} | {'Sem':>4} | {'Score':>5} | {'N':>3} | {'WR%':>4} | {'Total P&L':>11} | {'Sharpe':>6} | {'Long P&L':>11} | {'Short P&L':>11} | {'Consist':>8} | {'Stable':>6} | {'Recent P&L':>11} | Veredicto")
print(f"  {'-'*3} | {'-'*4} | {'-'*5} | {'-'*3} | {'-'*4} | {'-'*11} | {'-'*6} | {'-'*11} | {'-'*11} | {'-'*8} | {'-'*6} | {'-'*11} | {'-'*30}")

for i, c in enumerate(cands, 1):
    print(f"  {i:>3} | {c['week']:>4} | {c['score']:>5} | {c['n']:>3} | {c['wr']:>3.0f}% | ${c['pnl']:>+10,.0f} | {c['sharpe']:>+5.2f} | ${c['long_pnl']:>+10,.0f} | ${c['short_pnl']:>+10,.0f} | {c['consistency']:>6.0f}% | {c['stable']:>6} | ${c['recent_pnl']:>+10,.0f} | {c['verdict']}")

# ============================================================
# 3. DETALLE AÑO A AÑO DE LAS MEJORES CANDIDATAS
# ============================================================
top_cands = [c for c in cands if c['score'] >= 6]

if top_cands:
    print(f"\n\n{'='*180}")
    print(f"  DETALLE AÑO A AÑO DE LAS {len(top_cands)} MEJORES CANDIDATAS")
    print("=" * 180)

    for c in top_cands:
        w = c['week']
        sub = df[df['Week_of_Year'] == w].sort_values('Year')

        import datetime
        approx = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=w-1)

        print(f"\n  --- Semana {w} (~{approx.strftime('%b %d')}) | Score={c['score']} | {c['verdict']} ---")
        print(f"  Total: P&L=${c['pnl']:>+,.0f} | Sharpe={c['sharpe']:>+.2f} | WR={c['wr']:.0f}% | Consist={c['consistency']:.0f}%")
        print(f"  Long: ${c['long_pnl']:>+,.0f} (Sh {c['sharpe_l']:>+.2f}) | Short: ${c['short_pnl']:>+,.0f} (Sh {c['sharpe_s']:>+.2f})")
        print(f"\n  {'Ano':>6} | {'P&L':>10} | {'Long':>10} | {'Short':>10} | {'Ret%':>7}")
        print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*7}")

        for _, r in sub.iterrows():
            ret = r['Ret_Pct'] if pd.notna(r['Ret_Pct']) else 0
            lp = r['Long_PnL'] if pd.notna(r['Long_PnL']) else 0
            sp = r['Short_PnL'] if pd.notna(r['Short_PnL']) else 0
            print(f"  {int(r['Year']):>6} | ${r['PnL_USD']:>+9,.0f} | ${lp:>+9,.0f} | ${sp:>+9,.0f} | {ret:>+6.2f}%")

# ============================================================
# 4. SIMULACION: REGIMEN EXPANDIDO
# ============================================================
print(f"\n\n{'='*180}")
print("  SIMULACION: REGIMEN EXPANDIDO (original + candidatas con score >= 6)")
print("=" * 180)

new_weeks_to_add = set(c['week'] for c in cands if c['score'] >= 6)
expanded = OPERABLE_WEEKS | new_weeks_to_add

print(f"\n  Original:  {sorted(OPERABLE_WEEKS)} ({len(OPERABLE_WEEKS)} sem)")
print(f"  Añadir:    {sorted(new_weeks_to_add)} ({len(new_weeks_to_add)} sem)")
print(f"  Expandido: {sorted(expanded)} ({len(expanded)} sem)")

for label, week_set in [("ORIGINAL (14 sem)", OPERABLE_WEEKS),
                         ("EXPANDIDO", expanded),
                         ("SOLO NUEVAS", new_weeks_to_add)]:
    sub = df[df['Week_of_Year'].isin(week_set)]
    n = len(sub)
    if n == 0:
        continue
    pnl = sub['PnL_USD'].sum()
    wins = sub['Win'].sum()
    wr = wins / n * 100
    pnl_pw = pnl / n
    rets = sub['Ret_Pct'].values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
    long_pnl = sub['Long_PnL'].sum()
    short_pnl = sub['Short_PnL'].sum()
    pct_time = n / len(df) * 100

    print(f"\n  {label} ({len(week_set)} sem, {n} obs, {pct_time:.0f}% tiempo):")
    print(f"    P&L: ${pnl:>+,.0f} | P&L/sem: ${pnl_pw:>+,.0f} | Sharpe: {sharpe:>+.2f} | WR: {wr:.0f}%")
    print(f"    Long: ${long_pnl:>+,.0f} | Short: ${short_pnl:>+,.0f}")

    # Year by year for expanded
    if label == "EXPANDIDO":
        print(f"\n    {'Ano':>6} | {'Sem':>4} | {'WR%':>4} | {'Long P&L':>11} | {'Short P&L':>11} | {'Total P&L':>11} | {'Sharpe':>6}")
        print(f"    {'-'*6} | {'-'*4} | {'-'*4} | {'-'*11} | {'-'*11} | {'-'*11} | {'-'*6}")
        for year in sorted(sub['Year'].unique()):
            ys = sub[sub['Year'] == year]
            yn = len(ys)
            yw = int(ys['Win'].sum())
            ywr = yw / yn * 100
            ypnl = ys['PnL_USD'].sum()
            ylp = ys['Long_PnL'].sum()
            ysp = ys['Short_PnL'].sum()
            yr = ys['Ret_Pct'].values / 100
            ysh = (np.mean(yr) / np.std(yr)) * np.sqrt(52) if len(yr) > 1 and np.std(yr) > 0 else 0
            print(f"    {year:>6} | {yn:>4} | {ywr:>3.0f}% | ${ylp:>+10,.0f} | ${ysp:>+10,.0f} | ${ypnl:>+10,.0f} | {ysh:>+5.2f}")

# ============================================================
# 5. ANALISIS: SOLO LONGS EN SEMANAS EXCLUIDAS (hay alpha long escondido?)
# ============================================================
print(f"\n\n{'='*180}")
print("  BONUS: SOLO LONGS por semana del año (sin shorts)")
print("=" * 180)

print(f"\n  {'Sem':>4} | {'N':>3} | {'WR%':>4} | {'Long P&L':>11} | {'Sharpe L':>8} | {'Consist':>8} | {'En regimen':>10}")
print(f"  {'-'*4} | {'-'*3} | {'-'*4} | {'-'*11} | {'-'*8} | {'-'*8} | {'-'*10}")

long_only_good = []
for w in range(1, 54):
    sub = df[df['Week_of_Year'] == w]
    if len(sub) < 5:
        continue
    n = len(sub)
    lp = sub['Long_PnL'].sum()
    lw = (sub['Long_PnL'] > 0).sum()
    lwr = lw / n * 100
    lr = sub['Long_Ret_Pct'].dropna().values / 100
    lsh = (np.mean(lr) / np.std(lr)) * np.sqrt(52) if len(lr) > 1 and np.std(lr) > 0 else 0

    by_year = sub.groupby('Year')['Long_PnL'].sum()
    lcons = (by_year > 0).sum() / len(by_year) * 100

    in_reg = "SI" if w in OPERABLE_WEEKS else ""
    flag = " ***" if lp > 20000 and lsh > 0.5 and lcons >= 50 and w not in OPERABLE_WEEKS else ""

    print(f"  {w:>4} | {n:>3} | {lwr:>3.0f}% | ${lp:>+10,.0f} | {lsh:>+7.2f} | {lcons:>6.0f}% | {in_reg:>10}{flag}")

    if lp > 10000 and lsh > 0.3 and lcons >= 45 and w not in OPERABLE_WEEKS:
        long_only_good.append({'week': w, 'long_pnl': lp, 'sharpe': lsh, 'consistency': lcons})

if long_only_good:
    print(f"\n  Semanas con alpha LONG escondido (no en regimen): {[x['week'] for x in long_only_good]}")
