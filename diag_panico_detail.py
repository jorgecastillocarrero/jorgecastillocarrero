"""
Por que hay 245 semanas con score < -4.0?
Desglosar que indicadores los empujan ahi
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# Cargar SPY
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21' ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

# Cargar subsectores
from sector_event_map import SUBSECTORS
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21' ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year

df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_price_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

# Calcular scores para todas las semanas
print("Calculando scores para todas las semanas...")

all_weeks = []
for date in dd_wide.index:
    if date.year < 2001: continue
    dd_row = dd_wide.loc[date]
    rsi_row = rsi_wide.loc[date]
    n_total = dd_row.notna().sum()
    if n_total == 0: continue
    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) == 0: continue
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_above_ma200 = spy_last.get('above_ma200', 0.5)
    spy_mom_10w = spy_last.get('mom_10w', 0)
    spy_dist = spy_last.get('dist_ma200', 0)
    if not pd.notna(spy_mom_10w): spy_mom_10w = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scores individuales
    if pct_dd_healthy >= 75: s_bdd = 2.0
    elif pct_dd_healthy >= 60: s_bdd = 1.0
    elif pct_dd_healthy >= 45: s_bdd = 0.0
    elif pct_dd_healthy >= 30: s_bdd = -1.0
    else: s_bdd = -2.0

    if pct_rsi_above55 >= 75: s_brsi = 2.0
    elif pct_rsi_above55 >= 60: s_brsi = 1.0
    elif pct_rsi_above55 >= 45: s_brsi = 0.0
    elif pct_rsi_above55 >= 30: s_brsi = -1.0
    else: s_brsi = -2.0

    if pct_dd_deep <= 5: s_ddp = 1.5
    elif pct_dd_deep <= 15: s_ddp = 0.5
    elif pct_dd_deep <= 30: s_ddp = -0.5
    else: s_ddp = -1.5

    if spy_above_ma200 and spy_dist > 5: s_spy = 1.5
    elif spy_above_ma200: s_spy = 0.5
    elif spy_dist > -5: s_spy = -0.5
    else: s_spy = -1.5

    if spy_mom_10w > 5: s_mom = 1.0
    elif spy_mom_10w > 0: s_mom = 0.5
    elif spy_mom_10w > -5: s_mom = -0.5
    else: s_mom = -1.0

    total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

    all_weeks.append({
        'date': date, 'year': date.year, 'score_total': total,
        's_bdd': s_bdd, 's_brsi': s_brsi, 's_ddp': s_ddp, 's_spy': s_spy, 's_mom': s_mom,
        'vix': vix_val, 'pct_dd_healthy': pct_dd_healthy, 'pct_dd_deep': pct_dd_deep,
        'pct_rsi_above55': pct_rsi_above55, 'spy_dist': spy_dist, 'spy_mom': spy_mom_10w,
    })

df = pd.DataFrame(all_weeks)
panico = df[df['score_total'] < -4.0].copy()

# ================================================================
# 1. Valores posibles del score total (es discreto!)
# ================================================================
print(f"\n{'='*100}")
print(f"  VALORES UNICOS DE SCORE TOTAL (todas las semanas)")
print(f"{'='*100}")
vc = df['score_total'].value_counts().sort_index()
print(f"\n  {'Score':>6s} {'N total':>8s} {'% total':>8s}")
print(f"  {'-'*25}")
for score, count in vc.items():
    pct = count / len(df) * 100
    if count >= 5:
        print(f"  {score:>+6.1f} {count:>8d} {pct:>7.1f}%")

# ================================================================
# 2. Por que score < -4.0? Que combinaciones de indicadores?
# ================================================================
print(f"\n{'='*100}")
print(f"  POR QUE SCORE < -4.0? ({len(panico)} semanas)")
print(f"{'='*100}")

print(f"\n  Distribucion de cada componente en PANICO vs TODO:")
for col, label in [('s_bdd', 'Breadth DD'), ('s_brsi', 'Breadth RSI'), ('s_ddp', 'DD deep'),
                    ('s_spy', 'SPY vs MA200'), ('s_mom', 'SPY mom 10w')]:
    print(f"\n  {label}:")
    for val in sorted(df[col].unique()):
        n_all = (df[col] == val).sum()
        n_pan = (panico[col] == val).sum()
        pct_all = n_all / len(df) * 100
        pct_pan = n_pan / len(panico) * 100 if len(panico) > 0 else 0
        if n_pan > 0:
            print(f"    {val:>+5.1f}: ALL={n_all:>4d} ({pct_all:>5.1f}%)  PANICO={n_pan:>4d} ({pct_pan:>5.1f}%)")

# ================================================================
# 3. Score minimo posible y combinaciones comunes
# ================================================================
print(f"\n{'='*100}")
print(f"  COMBINACIONES MAS COMUNES EN PANICO (score < -4.0)")
print(f"{'='*100}")

panico['combo'] = panico.apply(lambda r: f"bdd={r['s_bdd']:+.1f} brsi={r['s_brsi']:+.1f} ddp={r['s_ddp']:+.1f} spy={r['s_spy']:+.1f} mom={r['s_mom']:+.1f}", axis=1)
combos = panico['combo'].value_counts().head(15)
print(f"\n  {'Combinacion':<60s} {'N':>4s} {'Score':>6s}")
print(f"  {'-'*75}")
for combo, n in combos.items():
    sub = panico[panico['combo'] == combo]
    score = sub['score_total'].iloc[0]
    print(f"  {combo:<60s} {n:>4d} {score:>+6.1f}")

# ================================================================
# 4. El problema: cuantos indicadores en minimo simultaneamente?
# ================================================================
print(f"\n{'='*100}")
print(f"  CUANTOS INDICADORES EN SU MINIMO SIMULTANEAMENTE?")
print(f"{'='*100}")

mins = {'s_bdd': -2.0, 's_brsi': -2.0, 's_ddp': -1.5, 's_spy': -1.5, 's_mom': -1.0}
df['n_at_min'] = sum((df[col] == mins[col]).astype(int) for col in mins)
panico_with_mins = df[df['score_total'] < -4.0].copy()
panico_with_mins['n_at_min'] = sum((panico_with_mins[col] == mins[col]).astype(int) for col in mins)

print(f"\n  PANICO (score < -4.0):")
for n_min in range(6):
    sub = panico_with_mins[panico_with_mins['n_at_min'] == n_min]
    if len(sub) == 0: continue
    print(f"    {n_min} indicadores en minimo: {len(sub)} semanas ({len(sub)/len(panico_with_mins)*100:.1f}%)")

print(f"\n  TODAS las semanas:")
for n_min in range(6):
    sub = df[df['n_at_min'] == n_min]
    if len(sub) == 0: continue
    print(f"    {n_min} indicadores en minimo: {len(sub)} semanas ({len(sub)/len(df)*100:.1f}%)")

# ================================================================
# 5. Score total range (-8 es realmente el minimo?)
# ================================================================
print(f"\n{'='*100}")
print(f"  RANGO TEORICO DEL SCORE TOTAL")
print(f"{'='*100}")
print(f"  Minimo teorico: {-2 + -2 + -1.5 + -1.5 + -1.0} (todos en minimo)")
print(f"  Maximo teorico: {+2 + +2 + +1.5 + +1.5 + +1.0}")
print(f"  Rango real: [{df['score_total'].min():.1f}, {df['score_total'].max():.1f}]")
print(f"  Rango PANICO: [{panico['score_total'].min():.1f}, {panico['score_total'].max():.1f}]")

# ================================================================
# 6. Distribucion completa de regimenes con el score discreto
# ================================================================
print(f"\n{'='*100}")
print(f"  HISTOGRAMA COMPLETO DE SCORE TOTAL")
print(f"{'='*100}")

print(f"\n  {'Score':>6s} {'N':>5s} {'Barra':<50s} {'Regimen actual'}")
print(f"  {'-'*75}")
for score in sorted(df['score_total'].unique()):
    n = (df['score_total'] == score).sum()
    bar = '#' * min(int(n / 3), 50)
    if score >= 8.0: reg = 'BURBUJA'
    elif score >= 7.0: reg = 'GOLDILOCKS'
    elif score >= 4.0: reg = 'ALCISTA'
    elif score >= 0.5: reg = 'NEUTRAL'
    elif score >= -1.5: reg = 'CAUTIOUS'
    elif score >= -3.0: reg = 'BEARISH'
    else: reg = 'CRISIS'
    print(f"  {score:>+6.1f} {n:>5d} {bar:<50s} {reg}")

# ================================================================
# 7. Por ano: semanas en cada nivel de score
# ================================================================
print(f"\n{'='*100}")
print(f"  SEMANAS CON SCORE < -4.0 POR ANO (PANICO actual)")
print(f"{'='*100}")
for year in sorted(panico['year'].unique()):
    yr = panico[panico['year'] == year]
    scores = yr['score_total'].value_counts().sort_index()
    score_str = ' '.join(f"{s:+.0f}:{n}" for s, n in scores.items())
    vix_avg = yr['vix'].mean()
    dd_deep = yr['pct_dd_deep'].mean()
    print(f"  {year}: {len(yr):>3d} sem  VIX={vix_avg:.0f}  DD_deep={dd_deep:.0f}%  Scores: {score_str}")
