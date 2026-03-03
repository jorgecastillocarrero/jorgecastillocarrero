"""
Comparativa Jueves vs Viernes - Paso a paso
Paso 1: Cargar datos y clasificar regimenes con senal jueves
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# ============================================================
# 1. CONSTITUYENTES S&P 500
# ============================================================
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
tickers = [s['symbol'] for s in sp500]

profiles = pd.read_sql(
    "SELECT symbol, industry FROM fmp_profiles WHERE symbol IN ('"
    + "','".join(tickers) + "')", engine)

sub_map = profiles.groupby('industry')['symbol'].apply(list).to_dict()
sub_map = {k: v for k, v in sub_map.items() if len(v) >= 3 and k is not None}
ticker_to_sub = {}
for sub, tks in sub_map.items():
    for t in tks:
        ticker_to_sub[t] = sub

all_tickers = list(ticker_to_sub.keys())
print(f'Subsectores: {len(sub_map)}, Tickers: {len(all_tickers)}')

# ============================================================
# 2. PRECIOS DIARIOS
# ============================================================
tlist = "','".join(all_tickers)
df_all = pd.read_sql(f"""
    SELECT symbol, date, open, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-28'
    ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['dow'] = df_all['date'].dt.dayofweek  # 0=lun, 3=jue, 4=vie
print(f'Registros precio: {len(df_all):,}')

# ============================================================
# 3. WEEKLY JUEVES: ultimo dia de trading hasta jueves (dow <= 3)
# ============================================================
df_thu = df_all[df_all['dow'] <= 3].copy()
df_thu['iso_year'] = df_thu['date'].dt.isocalendar().year.astype(int)
df_thu['week'] = df_thu['date'].dt.isocalendar().week.astype(int)

df_weekly_thu = df_thu.sort_values('date').groupby(['symbol', 'iso_year', 'week']).last().reset_index()
df_weekly_thu = df_weekly_thu.sort_values(['symbol', 'date'])
df_weekly_thu['prev_close'] = df_weekly_thu.groupby('symbol')['close'].shift(1)
df_weekly_thu['return'] = df_weekly_thu['close'] / df_weekly_thu['prev_close'] - 1
df_weekly_thu = df_weekly_thu.dropna(subset=['return'])

# Subsector semanal jueves
sub_thu = df_weekly_thu.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean')).reset_index()
sub_thu = sub_thu.sort_values(['subsector', 'date'])

date_counts = sub_thu.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_thu = sub_thu[sub_thu['date'].isin(valid_dates)]

def calc_metrics(g):
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

sub_thu = sub_thu.groupby('subsector', group_keys=False).apply(calc_metrics)
dd_wide = sub_thu.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_thu.pivot(index='date', columns='subsector', values='rsi_14w')
print(f'Semanas jueves validas: {len(dd_wide)}')

# ============================================================
# 4. SPY DIARIO + MA200 (calculado sobre todos los dias, leido hasta jueves)
# ============================================================
spy_daily = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()

# SPY semanal jueves: ultimo dia de trading hasta jueves
spy_thu_daily = spy_daily[spy_daily.index.dayofweek <= 3]
spy_w = spy_thu_daily.resample('W-THU').last().dropna(subset=['ma200'])
spy_w['above_ma200'] = (spy_w['close'] > spy_w['ma200']).astype(int)
spy_w['dist_ma200'] = (spy_w['close'] / spy_w['ma200'] - 1) * 100
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

# ============================================================
# 5. VIX DIARIO desde price_history_vix
# ============================================================
vix_daily = pd.read_sql("""
    SELECT date, close FROM price_history_vix
    WHERE date BETWEEN '2000-01-01' AND '2026-02-28'
    ORDER BY date
""", engine)
vix_daily['date'] = pd.to_datetime(vix_daily['date'])
vix_daily = vix_daily.set_index('date').sort_index()
vix_daily = vix_daily.rename(columns={'close': 'vix'})
vix_daily['vix'] = vix_daily['vix'].astype(float)
print(f'VIX diario: {len(vix_daily)} registros, {vix_daily.index.min()} a {vix_daily.index.max()}')

# ============================================================
# 6. CLASIFICAR REGIMENES (JUEVES)
# ============================================================
def classify(date):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        return 'NEUTRAL', 0, 20, {}
    last = prev_dates[-1]
    dd_row = dd_wide.loc[last]
    rsi_row = rsi_wide.loc[last]
    n = dd_row.notna().sum()
    if n == 0:
        return 'NEUTRAL', 0, 20, {}

    pct_dd_h = (dd_row > -10).sum() / n * 100
    pct_dd_d = (dd_row < -20).sum() / n * 100
    pct_rsi = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) > 0:
        sl = spy_w.loc[spy_dates[-1]]
        above = sl.get('above_ma200', 0.5)
        mom = sl.get('mom_10w', 0)
        dist = sl.get('dist_ma200', 0)
    else:
        above, mom, dist = 0.5, 0, 0
    if not pd.notna(mom): mom = 0
    if not pd.notna(dist): dist = 0

    # VIX del jueves real (dia exacto o ultimo dia disponible)
    vix_dates = vix_daily.index[vix_daily.index <= date]
    vix_val = float(vix_daily.loc[vix_dates[-1], 'vix']) if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scores
    if pct_dd_h >= 75: s1 = 2.0
    elif pct_dd_h >= 60: s1 = 1.0
    elif pct_dd_h >= 45: s1 = 0.0
    elif pct_dd_h >= 30: s1 = -1.0
    elif pct_dd_h >= 15: s1 = -2.0
    else: s1 = -3.0

    if pct_rsi >= 75: s2 = 2.0
    elif pct_rsi >= 60: s2 = 1.0
    elif pct_rsi >= 45: s2 = 0.0
    elif pct_rsi >= 30: s2 = -1.0
    elif pct_rsi >= 15: s2 = -2.0
    else: s2 = -3.0

    if pct_dd_d <= 5: s3 = 1.5
    elif pct_dd_d <= 15: s3 = 0.5
    elif pct_dd_d <= 30: s3 = -0.5
    elif pct_dd_d <= 50: s3 = -1.5
    else: s3 = -2.5

    if above and dist > 5: s4 = 1.5
    elif above: s4 = 0.5
    elif dist > -5: s4 = -0.5
    elif dist > -15: s4 = -1.5
    else: s4 = -2.5

    if mom > 5: s5 = 1.0
    elif mom > 0: s5 = 0.5
    elif mom > -5: s5 = -0.5
    elif mom > -15: s5 = -1.0
    else: s5 = -1.5

    total = s1 + s2 + s3 + s4 + s5

    is_burbuja = (total >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'

    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'CAUTIOUS'

    spy_close = spy_w.loc[spy_dates[-1], 'close'] if len(spy_dates) > 0 else 0
    return regime, total, vix_val, {
        'dd_healthy': pct_dd_h, 'dd_deep': pct_dd_d, 'rsi_broad': pct_rsi,
        'spy_close': spy_close, 'spy_dist': dist, 'spy_mom': mom
    }

# Clasificar todas las semanas
thu_dates = sorted(dd_wide.index)
prev_vix_val = None
results = []

for thu in thu_dates:
    if thu.year < 2001:
        _, _, vix_val, _ = classify(thu)
        prev_vix_val = vix_val
        continue

    regime, total, vix_val, details = classify(thu)

    # CAPITULACION / RECOVERY: VIX bajando vs semana anterior
    if prev_vix_val is not None and vix_val < prev_vix_val:
        if regime == 'PANICO':
            regime = 'CAPITULACION'
        elif regime == 'BEARISH':
            regime = 'RECOVERY'

    results.append({
        'fecha_senal': thu,
        'regime': regime,
        'total': total,
        'vix': vix_val,
        'prev_vix': prev_vix_val,
        'year': thu.year,
        'month': thu.month,
        'week_num': thu.isocalendar()[1],
        **details,
    })

    prev_vix_val = vix_val

dfr = pd.DataFrame(results)

# Guardar CSV jueves para comparativas
dfr.to_csv('data/regimenes_jueves.csv', index=False)
print(f'CSV guardado: data/regimenes_jueves.csv ({len(dfr)} semanas)')

# ============================================================
# VALIDACION: Distribucion de regimenes
# ============================================================
print('\n' + '='*60)
print('PASO 1: DISTRIBUCION DE REGIMENES (senal jueves, VIX diario)')
print('='*60)
rc = dfr['regime'].value_counts()
print(f'\n{"Regimen":<16} {"N":>5} {"%":>7}')
print('-'*30)
for reg in ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']:
    n = rc.get(reg, 0)
    pct = n / len(dfr) * 100
    print(f'{reg:<16} {n:>5} {pct:>6.1f}%')
print('-'*30)
print(f'{"TOTAL":<16} {len(dfr):>5}')

# Comparar con viernes (CSV)
df_fri = pd.read_csv('data/regimenes_historico.csv')
df_fri['prev_vix'] = df_fri['vix'].shift(1)
mask_cap = (df_fri['regime'] == 'PANICO') & (df_fri['vix'] < df_fri['prev_vix'])
df_fri.loc[mask_cap, 'regime'] = 'CAPITULACION'
mask_rec = (df_fri['regime'] == 'BEARISH') & (df_fri['vix'] < df_fri['prev_vix'])
df_fri.loc[mask_rec, 'regime'] = 'RECOVERY'

rc_fri = df_fri['regime'].value_counts()

print('\n' + '='*60)
print('COMPARATIVA DISTRIBUCION: Viernes vs Jueves')
print('='*60)
print(f'\n{"Regimen":<16} {"Viernes":>8} {"Jueves":>8} {"Diff":>6}')
print('-'*42)
for reg in ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']:
    nv = rc_fri.get(reg, 0)
    nj = rc.get(reg, 0)
    print(f'{reg:<16} {nv:>8} {nj:>8} {nj-nv:>+6}')
print('-'*42)
print(f'{"TOTAL":<16} {len(df_fri):>8} {len(dfr):>8}')

# ============================================================
# PASO 2: RETORNOS SPY Fri open -> Fri open (senal jueves)
# ============================================================
# Para cada jueves de senal, buscar:
#   - Viernes siguiente (thu+1): entrada a open
#   - Viernes de la semana siguiente (thu+8): salida a open

spy_open = spy_daily[['open']].copy()

for i, row in dfr.iterrows():
    thu = row['fecha_senal']

    # Viernes entrada: primer dia de trading despues del jueves
    fri_entry = None
    for d in range(1, 4):
        candidate = thu + pd.Timedelta(days=d)
        if candidate in spy_open.index:
            fri_entry = candidate
            break

    # Viernes salida: primer dia de trading despues de thu+7
    fri_exit = None
    for d in range(8, 11):
        candidate = thu + pd.Timedelta(days=d)
        if candidate in spy_open.index:
            fri_exit = candidate
            break

    if fri_entry and fri_exit:
        ret = (spy_open.loc[fri_exit, 'open'] / spy_open.loc[fri_entry, 'open'] - 1) * 100
        dfr.at[i, 'spy_ret_pct'] = ret
        dfr.at[i, 'fri_entry'] = fri_entry
        dfr.at[i, 'fri_exit'] = fri_exit

dfr_valid = dfr.dropna(subset=['spy_ret_pct'])

print('\n' + '='*90)
print('PASO 2: RETORNOS SPY POR REGIMEN - SENAL JUEVES')
print('Senal: Jueves close | Trade: Vie open -> Vie open | 2001-2026')
print('='*90)
print(f'\n{"Regimen":<16} {"N":>5} {"Avg%":>8} {"Med%":>8} {"Std%":>8} {"WR%":>7} {"Best%":>8} {"Worst%":>9} {"Total%":>9}')
print('-'*90)

for reg in ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']:
    mask = dfr_valid['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = dfr_valid[mask]['spy_ret_pct']
    print(f'{reg:<16} {len(sub):>5} {sub.mean():>+7.2f} {sub.median():>+7.2f} {sub.std():>7.2f} {(sub>0).mean()*100:>6.1f} {sub.max():>+7.2f} {sub.min():>+8.2f} {sub.sum():>+8.1f}')

print('-'*90)
all_ret = dfr_valid['spy_ret_pct']
print(f'{"TOTAL":<16} {len(all_ret):>5} {all_ret.mean():>+7.2f} {all_ret.median():>+7.2f} {all_ret.std():>7.2f} {(all_ret>0).mean()*100:>6.1f} {all_ret.max():>+7.2f} {all_ret.min():>+8.2f} {all_ret.sum():>+8.1f}')

# Verificacion: mostrar primeras entradas/salidas
print('\nVerificacion primeras 5 semanas:')
for _, r in dfr_valid.head().iterrows():
    print(f'  Senal: {r["fecha_senal"].strftime("%Y-%m-%d")} ({r["fecha_senal"].strftime("%a")}) | Entry: {r["fri_entry"].strftime("%Y-%m-%d")} ({r["fri_entry"].strftime("%a")}) | Exit: {r["fri_exit"].strftime("%Y-%m-%d")} ({r["fri_exit"].strftime("%a")}) | Ret: {r["spy_ret_pct"]:+.2f}%')

# Comparativa con viernes (Mon->Mon del CSV)
print('\n' + '='*90)
print('COMPARATIVA: Viernes (Mon-Mon) vs Jueves (Fri-Fri)')
print('='*90)
print(f'\n{"Regimen":<16} {"N_vie":>5} {"Vie Avg%":>9} {"WR%":>6}  {"N_jue":>5} {"Jue Avg%":>9} {"WR%":>6}  {"Delta":>7}')
print('-'*80)

for reg in ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']:
    mask_v = df_fri['regime'] == reg
    mask_j = dfr_valid['regime'] == reg
    n_v = mask_v.sum()
    n_j = mask_j.sum()
    avg_v = df_fri[mask_v]['spy_ret_pct'].mean() if n_v > 0 else 0
    wr_v = (df_fri[mask_v]['spy_ret_pct'] > 0).mean() * 100 if n_v > 0 else 0
    avg_j = dfr_valid[mask_j]['spy_ret_pct'].mean() if n_j > 0 else 0
    wr_j = (dfr_valid[mask_j]['spy_ret_pct'] > 0).mean() * 100 if n_j > 0 else 0
    delta = avg_j - avg_v
    print(f'{reg:<16} {n_v:>5} {avg_v:>+8.2f}% {wr_v:>5.1f}  {n_j:>5} {avg_j:>+8.2f}% {wr_j:>5.1f}  {delta:>+6.2f}%')

print('-'*80)
avg_v_all = df_fri['spy_ret_pct'].dropna().mean()
avg_j_all = dfr_valid['spy_ret_pct'].mean()
print(f'{"TOTAL":<16} {len(df_fri):>5} {avg_v_all:>+8.2f}%        {len(dfr_valid):>5} {avg_j_all:>+8.2f}%        {avg_j_all-avg_v_all:>+6.2f}%')

# ============================================================
# PASO 3: HTML REGIMENES ANO A ANO (SENAL JUEVES)
# ============================================================
print('\nGenerando HTML...')

REGIME_ORDER = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'RECOVERY', 'CRISIS', 'PANICO', 'CAPITULACION']
REGIME_COLOR = {
    'BURBUJA': '#ff6600', 'GOLDILOCKS': '#00aa00', 'ALCISTA': '#33cc33',
    'NEUTRAL': '#888888', 'CAUTIOUS': '#cc9900', 'BEARISH': '#cc3333',
    'RECOVERY': '#0088cc', 'CRISIS': '#990000', 'PANICO': '#660066',
    'CAPITULACION': '#0066cc'
}
REGIME_BG = {
    'BURBUJA': '#fff3e0', 'GOLDILOCKS': '#e8f5e9', 'ALCISTA': '#f1f8e9',
    'NEUTRAL': '#f5f5f5', 'CAUTIOUS': '#fff8e1', 'BEARISH': '#ffebee',
    'RECOVERY': '#e3f2fd', 'CRISIS': '#fce4ec', 'PANICO': '#f3e5f5',
    'CAPITULACION': '#e1f5fe'
}
MONTHS = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

# SPY retorno semanal para calcular anual
spy_w_ret = spy_daily[['close']].resample('W-THU').last().dropna()
spy_w_ret['ret_spy'] = spy_w_ret['close'].pct_change()

html = []
html.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Regimenes de Mercado - Senal Jueves (2001-2026)</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #fafafa; }
h1 { color: #333; border-bottom: 3px solid #333; padding-bottom: 10px; }
h2 { color: #555; margin-top: 30px; }
.year-card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 5px solid #333; }
.year-title { font-size: 24px; font-weight: bold; color: #333; }
.spy-info { font-size: 16px; color: #666; margin: 5px 0; }
.spy-pos { color: #00aa00; font-weight: bold; }
.spy-neg { color: #cc3333; font-weight: bold; }
.timeline { display: flex; gap: 2px; margin: 10px 0; }
.month-cell { width: 60px; height: 40px; display: flex; align-items: center; justify-content: center;
              border-radius: 4px; font-weight: bold; font-size: 12px; color: white; }
.month-labels { display: flex; gap: 2px; margin-bottom: 2px; }
.month-label { width: 60px; text-align: center; font-size: 11px; color: #999; }
.regime-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; margin: 2px;
                font-size: 12px; font-weight: bold; color: white; }
.transitions { margin: 8px 0; font-size: 13px; color: #666; }
.trans-arrow { color: #333; font-weight: bold; }
.week-strip { display: flex; gap: 1px; margin: 8px 0; flex-wrap: wrap; }
.week-dot { width: 14px; height: 14px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th { background: #333; color: white; padding: 8px 12px; text-align: center; font-size: 13px; }
td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; font-size: 13px; }
tr:hover { background: #f0f0f0; }
.legend { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 13px; }
.legend-dot { width: 16px; height: 16px; border-radius: 50%; }
.section-neg { border-left-color: #cc3333; }
.section-pos { border-left-color: #00aa00; }
.section-mix { border-left-color: #cc9900; }
.quarter { margin: 5px 0; padding: 8px 12px; background: #f8f8f8; border-radius: 4px; font-size: 13px; }
.quarter-name { font-weight: bold; color: #555; }
</style></head><body>
<h1>Regimenes de Mercado - Senal JUEVES (2001-2026)</h1>
<p style="color:#666">Senal: Jueves close | Trade: Viernes open &rarr; Viernes open siguiente<br>
5 indicadores: Breadth DD, Breadth RSI, DD Deep%, SPY vs MA200, SPY Mom 10w. Score [-12.5, +8.0]<br>
VIX diario (price_history_vix). ISO year para agrupacion semanal.</p>
<div class="legend">""")

for r in REGIME_ORDER:
    html.append(f'<div class="legend-item"><div class="legend-dot" style="background:{REGIME_COLOR[r]}"></div>{r}</div>')
html.append('</div>')

positive = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'RECOVERY', 'CAPITULACION'}
negative = {'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO'}
def zone(r):
    if r in positive: return 'POS'
    if r == 'NEUTRAL': return 'NEU'
    return 'NEG'

for year in sorted(dfr['year'].unique()):
    yr = dfr[dfr['year'] == year].sort_values('fecha_senal')
    if len(yr) == 0: continue

    spy_yr = spy_w_ret[(spy_w_ret.index.year == year) & spy_w_ret['ret_spy'].notna()]
    spy_annual = ((1 + spy_yr['ret_spy']).prod() - 1) * 100 if len(spy_yr) > 0 else 0
    spy_start = spy_yr.iloc[0]['close'] if len(spy_yr) > 0 else 0
    spy_end = spy_yr.iloc[-1]['close'] if len(spy_yr) > 0 else 0
    rc = yr['regime'].value_counts()
    avg_score = yr['total'].mean()
    avg_vix = yr['vix'].mean()

    n_pos = sum(rc.get(r, 0) for r in positive)
    n_neg = sum(rc.get(r, 0) for r in negative)
    if n_neg > n_pos * 1.5: section_class = 'section-neg'
    elif n_pos > n_neg * 1.5: section_class = 'section-pos'
    else: section_class = 'section-mix'

    spy_class = 'spy-pos' if spy_annual >= 0 else 'spy-neg'

    html.append(f'<div class="year-card {section_class}">')
    html.append(f'<div class="year-title">{year}</div>')
    html.append(f'<div class="spy-info">S&P 500: <span class="{spy_class}">{spy_annual:+.1f}%</span> '
                f'({spy_start:.0f} &rarr; {spy_end:.0f}) &nbsp;|&nbsp; '
                f'Score avg: {avg_score:+.1f} &nbsp;|&nbsp; VIX avg: {avg_vix:.0f} &nbsp;|&nbsp; '
                f'{len(yr)} semanas</div>')

    # Timeline mensual
    html.append('<div class="month-labels">')
    for m in MONTHS:
        html.append(f'<div class="month-label">{m}</div>')
    html.append('</div><div class="timeline">')
    for month in range(1, 13):
        m_data = yr[yr['month'] == month]
        if len(m_data) == 0:
            html.append('<div class="month-cell" style="background:#ddd">&nbsp;</div>')
        else:
            dom = m_data['regime'].value_counts().index[0]
            html.append(f'<div class="month-cell" style="background:{REGIME_COLOR[dom]}">{dom[:3]}</div>')
    html.append('</div>')

    # Tira semanal
    html.append('<div class="week-strip">')
    for _, row in yr.iterrows():
        r = row['regime']
        title = f"Sem {row['week_num']}: {r} (score {row['total']:+.1f}, VIX {row['vix']:.0f})"
        html.append(f'<div class="week-dot" style="background:{REGIME_COLOR[r]}" title="{title}"></div>')
    html.append('</div>')

    # Badges
    html.append('<div style="margin:8px 0">')
    for r in REGIME_ORDER:
        n = rc.get(r, 0)
        if n > 0:
            html.append(f'<span class="regime-badge" style="background:{REGIME_COLOR[r]}">{r}: {n}</span>')
    html.append('</div>')

    # Transiciones
    transitions = []
    prev_reg = None
    for _, row in yr.iterrows():
        if row['regime'] != prev_reg:
            if prev_reg is not None:
                transitions.append((row['fecha_senal'].strftime('%d/%m'), prev_reg, row['regime']))
            prev_reg = row['regime']
    sig_trans = []
    for date_str, fr, to in transitions:
        if zone(fr) != zone(to) or (fr in negative and to in negative and fr != to):
            sig_trans.append(f'{date_str}: <span style="color:{REGIME_COLOR[fr]}">{fr}</span> '
                           f'<span class="trans-arrow">&rarr;</span> '
                           f'<span style="color:{REGIME_COLOR[to]}">{to}</span>')
    if sig_trans:
        html.append(f'<div class="transitions"><b>Transiciones:</b> {" &nbsp;|&nbsp; ".join(sig_trans)}</div>')

    # Trimestres
    for q in range(1, 5):
        q_data = yr[yr['month'].between((q-1)*3+1, q*3)]
        if len(q_data) == 0: continue
        q_rc = q_data['regime'].value_counts()
        q_vix = q_data['vix'].mean()
        q_name = ['Q1 (Ene-Mar)', 'Q2 (Abr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dic)'][q-1]
        badges = ''
        for r in REGIME_ORDER:
            n = q_rc.get(r, 0)
            if n > 0:
                badges += f'<span class="regime-badge" style="background:{REGIME_COLOR[r]};font-size:11px">{r[:3]}:{n}</span>'
        html.append(f'<div class="quarter"><span class="quarter-name">{q_name}:</span> '
                    f'VIX {q_vix:.0f} &nbsp; {badges}</div>')

    html.append('</div>')

# Tabla resumen
html.append('<h2>Tabla Resumen: Semanas por Regimen y Ano</h2>')
html.append('<table><tr><th>Ano</th><th>SPY%</th>')
for r in REGIME_ORDER:
    html.append(f'<th style="background:{REGIME_COLOR[r]}">{r[:4]}</th>')
html.append('<th>TOTAL</th><th>Zona+</th><th>Neutral</th><th>Zona-</th></tr>')

totals = {r: 0 for r in REGIME_ORDER}
for year in sorted(dfr['year'].unique()):
    yr = dfr[dfr['year'] == year]
    rc = yr['regime'].value_counts()
    spy_yr = spy_w_ret[(spy_w_ret.index.year == year) & spy_w_ret['ret_spy'].notna()]
    spy_annual = ((1 + spy_yr['ret_spy']).prod() - 1) * 100 if len(spy_yr) > 0 else 0
    spy_class = 'spy-pos' if spy_annual >= 0 else 'spy-neg'

    html.append(f'<tr><td><b>{year}</b></td><td class="{spy_class}">{spy_annual:+.1f}%</td>')
    for r in REGIME_ORDER:
        n = rc.get(r, 0)
        totals[r] += n
        if n > 0:
            html.append(f'<td style="background:{REGIME_BG[r]};font-weight:bold">{n}</td>')
        else:
            html.append('<td style="color:#ddd">-</td>')
    n_total = len(yr)
    n_pos = sum(rc.get(r, 0) for r in positive)
    n_neu = rc.get('NEUTRAL', 0)
    n_neg = sum(rc.get(r, 0) for r in negative)
    html.append(f'<td><b>{n_total}</b></td><td style="color:#00aa00">{n_pos}</td>'
                f'<td style="color:#888">{n_neu}</td><td style="color:#cc3333">{n_neg}</td></tr>')

n_all = sum(totals.values())
html.append('<tr style="font-weight:bold;background:#eee"><td>TOTAL</td><td></td>')
for r in REGIME_ORDER:
    pct = totals[r] / n_all * 100
    html.append(f'<td>{totals[r]}<br><small>{pct:.1f}%</small></td>')
n_pos = sum(totals[r] for r in positive)
n_neu = totals['NEUTRAL']
n_neg = sum(totals[r] for r in negative)
html.append(f'<td>{n_all}</td><td style="color:#00aa00">{n_pos}<br><small>{n_pos/n_all*100:.1f}%</small></td>'
            f'<td style="color:#888">{n_neu}<br><small>{n_neu/n_all*100:.1f}%</small></td>'
            f'<td style="color:#cc3333">{n_neg}<br><small>{n_neg/n_all*100:.1f}%</small></td></tr>')
html.append('</table>')

# Detalle semanal por ano
html.append('<h2>Detalle Semanal por Ano</h2>')
for year in sorted(dfr['year'].unique()):
    yr = dfr[dfr['year'] == year].sort_values('fecha_senal')
    if len(yr) == 0: continue
    html.append(f'<h3>{year}</h3>')
    html.append('<table><tr><th>Sem</th><th>Fecha (Jue)</th><th>Regimen</th><th>Score</th>'
                '<th>VIX</th><th>SPY</th><th>SPY dist MA200</th><th>SPY mom 10w</th>'
                '<th>DD Healthy%</th><th>DD Deep%</th><th>RSI Broad%</th></tr>')
    for _, row in yr.iterrows():
        r = row['regime']
        html.append(f'<tr style="background:{REGIME_BG[r]}">'
                    f'<td>{row["week_num"]}</td>'
                    f'<td>{row["fecha_senal"].strftime("%d/%m/%Y")}</td>'
                    f'<td style="color:{REGIME_COLOR[r]};font-weight:bold">{r}</td>'
                    f'<td>{row["total"]:+.1f}</td>'
                    f'<td>{row["vix"]:.0f}</td>'
                    f'<td>{row["spy_close"]:.0f}</td>'
                    f'<td>{row["spy_dist"]:+.1f}%</td>'
                    f'<td>{row["spy_mom"]:+.1f}%</td>'
                    f'<td>{row["dd_healthy"]:.0f}%</td>'
                    f'<td>{row["dd_deep"]:.0f}%</td>'
                    f'<td>{row["rsi_broad"]:.0f}%</td></tr>')
    html.append('</table>')

html.append('</body></html>')

output_path = 'C:/Users/usuario/financial-data-project/regimenes_jueves.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))
print(f'HTML generado: {output_path}')
