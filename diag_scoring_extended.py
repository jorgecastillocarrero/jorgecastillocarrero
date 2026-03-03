"""
Ampliar scoring negativo: mas granularidad para separar BEARISH/CRISIS/PANICO
Solo se extiende la zona negativa, la positiva queda igual.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# --- Cargar datos (rapido, solo lo necesario) ---
print("Cargando datos...")
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
spy_w['ret_spy'] = spy_w['close'].pct_change()

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

# ================================================================
# SCORING ANTIGUO vs NUEVO
# ================================================================

def score_old(pct_dd_healthy, pct_dd_deep, pct_rsi_above55, spy_above_ma200, spy_dist, spy_mom_10w):
    """Scoring actual (rango -8 a +8)"""
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

    return s_bdd + s_brsi + s_ddp + s_spy + s_mom, s_bdd, s_brsi, s_ddp, s_spy, s_mom

def score_new(pct_dd_healthy, pct_dd_deep, pct_rsi_above55, spy_above_ma200, spy_dist, spy_mom_10w):
    """Scoring extendido: mas resolucion en zona negativa (rango -12.5 a +8)"""
    # Breadth DD: añadir <15% = -3
    if pct_dd_healthy >= 75: s_bdd = 2.0
    elif pct_dd_healthy >= 60: s_bdd = 1.0
    elif pct_dd_healthy >= 45: s_bdd = 0.0
    elif pct_dd_healthy >= 30: s_bdd = -1.0
    elif pct_dd_healthy >= 15: s_bdd = -2.0
    else: s_bdd = -3.0

    # Breadth RSI: añadir <15% = -3
    if pct_rsi_above55 >= 75: s_brsi = 2.0
    elif pct_rsi_above55 >= 60: s_brsi = 1.0
    elif pct_rsi_above55 >= 45: s_brsi = 0.0
    elif pct_rsi_above55 >= 30: s_brsi = -1.0
    elif pct_rsi_above55 >= 15: s_brsi = -2.0
    else: s_brsi = -3.0

    # DD deep: añadir >50% = -2.5
    if pct_dd_deep <= 5: s_ddp = 1.5
    elif pct_dd_deep <= 15: s_ddp = 0.5
    elif pct_dd_deep <= 30: s_ddp = -0.5
    elif pct_dd_deep <= 50: s_ddp = -1.5
    else: s_ddp = -2.5

    # SPY vs MA200: añadir dist < -15% = -2.5
    if spy_above_ma200 and spy_dist > 5: s_spy = 1.5
    elif spy_above_ma200: s_spy = 0.5
    elif spy_dist > -5: s_spy = -0.5
    elif spy_dist > -15: s_spy = -1.5
    else: s_spy = -2.5

    # SPY mom 10w: añadir < -15% = -1.5
    if spy_mom_10w > 5: s_mom = 1.0
    elif spy_mom_10w > 0: s_mom = 0.5
    elif spy_mom_10w > -5: s_mom = -0.5
    elif spy_mom_10w > -15: s_mom = -1.0
    else: s_mom = -1.5

    return s_bdd + s_brsi + s_ddp + s_spy + s_mom, s_bdd, s_brsi, s_ddp, s_spy, s_mom

# ================================================================
# CALCULAR AMBOS SCORES PARA TODAS LAS SEMANAS
# ================================================================
print("Calculando scores old vs new...")

rows = []
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

    spy_ret = spy_w.loc[spy_dates[-1], 'ret_spy'] if 'ret_spy' in spy_w.columns else 0
    if not pd.notna(spy_ret): spy_ret = 0

    old_total, o1, o2, o3, o4, o5 = score_old(pct_dd_healthy, pct_dd_deep, pct_rsi_above55, spy_above_ma200, spy_dist, spy_mom_10w)
    new_total, n1, n2, n3, n4, n5 = score_new(pct_dd_healthy, pct_dd_deep, pct_rsi_above55, spy_above_ma200, spy_dist, spy_mom_10w)

    rows.append({
        'date': date, 'year': date.year, 'old': old_total, 'new': new_total,
        'vix': vix_val, 'spy_ret': spy_ret,
        'pct_dd_healthy': pct_dd_healthy, 'pct_dd_deep': pct_dd_deep,
        'pct_rsi_above55': pct_rsi_above55, 'spy_dist': spy_dist, 'spy_mom': spy_mom_10w,
        'n_bdd': n1, 'n_brsi': n2, 'n_ddp': n3, 'n_spy': n4, 'n_mom': n5,
    })

df = pd.DataFrame(rows)

# ================================================================
# 1. COMPARACION RANGES
# ================================================================
print(f"\n{'='*110}")
print(f"  RANGO SCORING")
print(f"{'='*110}")
print(f"  Old: [{df['old'].min():.1f}, {df['old'].max():.1f}]  teorico [-8.0, +8.0]")
print(f"  New: [{df['new'].min():.1f}, {df['new'].max():.1f}]  teorico [-12.5, +8.0]")

# ================================================================
# 2. HISTOGRAMA NEW SCORING (zona negativa)
# ================================================================
print(f"\n{'='*110}")
print(f"  HISTOGRAMA NUEVO SCORING (solo zona negativa)")
print(f"{'='*110}")
neg = df[df['new'] < 0.5].copy()
vc = neg['new'].value_counts().sort_index()
print(f"\n  {'Score':>7s} {'N':>5s} {'Barra':<50s} {'Old score(s)'}")
print(f"  {'-'*85}")
for score in sorted(neg['new'].unique()):
    n = (neg['new'] == score).sum()
    bar = '#' * min(int(n / 2), 50)
    # Que old scores mapean a este new score?
    old_scores = df[df['new'] == score]['old'].value_counts().sort_index()
    old_str = ', '.join(f"{s:+.1f}({n})" for s, n in old_scores.items())
    print(f"  {score:>+7.1f} {n:>5d} {bar:<50s} {old_str}")

# ================================================================
# 3. DONDE ESTABAN LOS 106 de score -8.0 OLD?
# ================================================================
print(f"\n{'='*110}")
print(f"  REDISTRIBUCION DE OLD -8.0 (106 semanas) EN NUEVO SCORING")
print(f"{'='*110}")
old_minus8 = df[df['old'] == -8.0].copy()
vc_new = old_minus8['new'].value_counts().sort_index()
print(f"\n  {'New score':>10s} {'N':>5s} {'VIX avg':>8s} {'SPY ret':>9s} {'DD deep':>8s}")
print(f"  {'-'*45}")
for score, n in vc_new.items():
    sub = old_minus8[old_minus8['new'] == score]
    print(f"  {score:>+10.1f} {n:>5d} {sub['vix'].mean():>7.1f} {sub['spy_ret'].mean()*100:>+8.2f}% {sub['pct_dd_deep'].mean():>7.1f}%")

# ================================================================
# 4. PROPUESTA DE REGIMENES CON NUEVO SCORING
# ================================================================
print(f"\n{'='*110}")
print(f"  PROPUESTA REGIMENES CON NUEVO SCORING")
print(f"{'='*110}")

# Los regimenes positivos no cambian (mismo score en zona positiva)
# Solo definimos los negativos
proposals = {
    'A': [('CAUTIOUS', -2.0, 0.5), ('BEARISH', -5.0, -2.0), ('CRISIS', -8.0, -5.0), ('PANICO', -99, -8.0)],
    'B': [('CAUTIOUS', -2.0, 0.5), ('BEARISH', -4.5, -2.0), ('CRISIS', -7.5, -4.5), ('PANICO', -99, -7.5)],
    'C': [('CAUTIOUS', -2.0, 0.5), ('BEARISH', -5.0, -2.0), ('CRISIS', -9.0, -5.0), ('PANICO', -99, -9.0)],
    'D': [('CAUTIOUS', -1.5, 0.5), ('BEARISH', -4.0, -1.5), ('CRISIS', -8.0, -4.0), ('PANICO', -99, -8.0)],
    'E': [('CAUTIOUS', -1.5, 0.5), ('BEARISH', -5.0, -1.5), ('CRISIS', -8.0, -5.0), ('PANICO', -99, -8.0)],
}

for pname, levels in proposals.items():
    print(f"\n  --- Propuesta {pname} ---")
    print(f"  {'Regimen':<12s} {'Score range':<16s} {'N':>5s} {'SPY avg':>8s} {'VIX avg':>8s} {'DD deep':>8s} {'DD hlth':>8s}")
    print(f"  {'-'*75}")
    for reg, lo, hi in levels:
        sub = df[(df['new'] >= lo) & (df['new'] < hi)]
        if len(sub) == 0:
            print(f"  {reg:<12s} [{lo:>+6.1f},{hi:>+5.1f}) {0:>5d}   ---")
            continue
        print(f"  {reg:<12s} [{lo:>+6.1f},{hi:>+5.1f}) {len(sub):>5d} {sub['spy_ret'].mean()*100:>+7.2f}% "
              f"{sub['vix'].mean():>7.1f} {sub['pct_dd_deep'].mean():>7.1f}% {sub['pct_dd_healthy'].mean():>7.1f}%")

# ================================================================
# 5. DETALLE POR AÑO PARA LA MEJOR PROPUESTA
# ================================================================
for pname, levels in proposals.items():
    print(f"\n  --- {pname} por ano ---")
    years = sorted(df['year'].unique())
    header = f"  {'Ano':>5s}"
    for reg, _, _ in levels:
        header += f" {reg:>8s}"
    print(header)
    print(f"  {'-'*45}")
    for year in years:
        yr = df[df['year'] == year]
        line = f"  {year:>5d}"
        for reg, lo, hi in levels:
            n = len(yr[(yr['new'] >= lo) & (yr['new'] < hi)])
            line += f" {n:>8d}"
        print(line)
    line = f"  {'TOTAL':>5s}"
    for reg, lo, hi in levels:
        n = len(df[(df['new'] >= lo) & (df['new'] < hi)])
        line += f" {n:>8d}"
    print(line)

# ================================================================
# 6. VERIFICAR QUE ZONA POSITIVA NO CAMBIA
# ================================================================
print(f"\n{'='*110}")
print(f"  VERIFICACION: ZONA POSITIVA SIN CAMBIOS")
print(f"{'='*110}")
pos = df[df['old'] >= 0.5]
same = (pos['old'] == pos['new']).all()
print(f"  Semanas con old >= 0.5: {len(pos)}")
print(f"  Score identico old==new: {same}")
if not same:
    diff = pos[pos['old'] != pos['new']]
    print(f"  DIFERENCIAS: {len(diff)} semanas")
    print(diff[['date', 'old', 'new']].head(10))
