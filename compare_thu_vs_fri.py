"""
Comparativa de regimenes: Senal Jueves vs Senal Viernes
- Jueves: recalcula indicadores con datos hasta jueves, trade Vie open -> Vie open
- Viernes: datos del CSV existente, trade Mon open -> Mon open
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# ============================================================
# 1. CARGAR CONSTITUYENTES S&P 500
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
# 2. CARGAR PRECIOS DIARIOS
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

print(f'Registros cargados: {len(df_all):,}')

# ============================================================
# 3. DATOS SEMANALES JUEVES (ultimo dia hasta jueves incluido)
# ============================================================
df_thu = df_all[df_all['dow'] <= 3].copy()
df_thu['year'] = df_thu['date'].dt.year
df_thu['week'] = df_thu['date'].dt.isocalendar().week.astype(int)

df_weekly_thu = df_thu.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly_thu = df_weekly_thu.sort_values(['symbol', 'date'])
df_weekly_thu['prev_close'] = df_weekly_thu.groupby('symbol')['close'].shift(1)
df_weekly_thu['return'] = df_weekly_thu['close'] / df_weekly_thu['prev_close'] - 1
df_weekly_thu = df_weekly_thu.dropna(subset=['return'])

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
# 4. SPY DIARIO + MA200
# ============================================================
spy_daily = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()

# SPY semanal jueves
spy_thu = spy_daily[spy_daily.index.dayofweek <= 3].copy()
spy_w = spy_thu.resample('W-THU').last().dropna(subset=['ma200'])
spy_w['above_ma200'] = (spy_w['close'] > spy_w['ma200']).astype(int)
spy_w['dist_ma200'] = (spy_w['close'] / spy_w['ma200'] - 1) * 100
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

# ============================================================
# 5. VIX SEMANAL (jueves)
# ============================================================
vix_df = pd.read_csv('data/vix_weekly.csv', skiprows=3, header=None,
                       names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})
# Resamplear a jueves
vix_daily = vix_df['vix'].resample('D').ffill()
vix_thu = vix_daily.resample('W-THU').last().to_frame()

# ============================================================
# 6. CLASIFICAR REGIMENES (JUEVES)
# ============================================================
def classify(date):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        return 'NEUTRAL', 0, 20
    last = prev_dates[-1]
    dd_row = dd_wide.loc[last]
    rsi_row = rsi_wide.loc[last]
    n = dd_row.notna().sum()
    if n == 0:
        return 'NEUTRAL', 0, 20

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

    vix_dates = vix_thu.index[vix_thu.index <= date]
    vix_val = vix_thu.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
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

    return regime, total, vix_val

# ============================================================
# 7. CALCULAR RETORNOS Fri open -> Fri open
# ============================================================
thu_dates = sorted(dd_wide.index)
prev_vix_val = None
results = []

for thu in thu_dates:
    if thu.year < 2001:
        _, _, vix_val = classify(thu)
        prev_vix_val = vix_val
        continue

    regime, total, vix_val = classify(thu)

    # CAPITULACION / RECOVERY
    if prev_vix_val is not None and vix_val < prev_vix_val:
        if regime == 'PANICO':
            regime = 'CAPITULACION'
        elif regime == 'BEARISH':
            regime = 'RECOVERY'

    # Viernes siguiente al jueves = thu + 1 dia
    fri_entry = None
    for i in range(3):
        d = thu + pd.Timedelta(days=1+i)
        if d in spy_daily.index:
            fri_entry = d
            break

    # Viernes siguiente = thu + 8 dias
    fri_exit = None
    for i in range(3):
        d = thu + pd.Timedelta(days=8+i)
        if d in spy_daily.index:
            fri_exit = d
            break

    spy_ret = None
    if fri_entry and fri_exit:
        spy_ret = (spy_daily.loc[fri_exit, 'open'] / spy_daily.loc[fri_entry, 'open'] - 1) * 100

    results.append({
        'fecha_senal': thu,
        'regime': regime,
        'total': total,
        'vix': vix_val,
        'fri_entry': fri_entry,
        'fri_exit': fri_exit,
        'spy_ret_pct': spy_ret
    })

    prev_vix_val = vix_val

dfr = pd.DataFrame(results)
dfr = dfr.dropna(subset=['spy_ret_pct'])

print(f'\nSemanas con retorno: {len(dfr)}')

# ============================================================
# 8. TABLA DE RESULTADOS
# ============================================================
print('\n' + '='*90)
print('RETORNO SPY POR REGIMEN (10) - SENAL JUEVES (recalculado)')
print('Senal: Jueves close | Trade: Vie open -> Vie open | 2001-2026')
print('='*90)
print(f'{"Regimen":<16} {"N":>5} {"Avg%":>8} {"Med%":>8} {"Std%":>8} {"WR%":>7} {"Best%":>8} {"Worst%":>9} {"Total%":>9}')
print('-'*90)

for reg in ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']:
    mask = dfr['regime'] == reg
    if mask.sum() == 0:
        continue
    sub = dfr[mask]['spy_ret_pct']
    print(f'{reg:<16} {len(sub):>5} {sub.mean():>+7.2f} {sub.median():>+7.2f} {sub.std():>7.2f} {(sub>0).mean()*100:>6.1f} {sub.max():>+7.2f} {sub.min():>+8.2f} {sub.sum():>+8.1f}')

print('-'*90)
all_ret = dfr['spy_ret_pct']
print(f'{"TOTAL":<16} {len(all_ret):>5} {all_ret.mean():>+7.2f} {all_ret.median():>+7.2f} {all_ret.std():>7.2f} {(all_ret>0).mean()*100:>6.1f} {all_ret.max():>+7.2f} {all_ret.min():>+8.2f} {all_ret.sum():>+8.1f}')

# ============================================================
# 9. COMPARATIVA CON VIERNES (CSV)
# ============================================================
df_fri = pd.read_csv('data/regimenes_historico.csv')
df_fri['prev_vix'] = df_fri['vix'].shift(1)
mask_cap = (df_fri['regime'] == 'PANICO') & (df_fri['vix'] < df_fri['prev_vix'])
df_fri.loc[mask_cap, 'regime'] = 'CAPITULACION'
mask_rec = (df_fri['regime'] == 'BEARISH') & (df_fri['vix'] < df_fri['prev_vix'])
df_fri.loc[mask_rec, 'regime'] = 'RECOVERY'

print('\n' + '='*90)
print('COMPARATIVA: Senal Viernes (Mon-Mon) vs Senal Jueves (Fri-Fri)')
print('='*90)
print(f'{"Regimen":<16} {"N_vie":>5} {"Vie Avg%":>9} {"WR%":>6}  {"N_jue":>5} {"Jue Avg%":>9} {"WR%":>6}  {"Delta":>7}')
print('-'*75)

for reg in ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']:
    mask_v = df_fri['regime'] == reg
    mask_j = dfr['regime'] == reg

    n_v = mask_v.sum()
    n_j = mask_j.sum()

    avg_v = df_fri[mask_v]['spy_ret_pct'].mean() if n_v > 0 else 0
    wr_v = (df_fri[mask_v]['spy_ret_pct'] > 0).mean() * 100 if n_v > 0 else 0
    avg_j = dfr[mask_j]['spy_ret_pct'].mean() if n_j > 0 else 0
    wr_j = (dfr[mask_j]['spy_ret_pct'] > 0).mean() * 100 if n_j > 0 else 0
    delta = avg_j - avg_v

    print(f'{reg:<16} {n_v:>5} {avg_v:>+8.2f}% {wr_v:>5.1f}  {n_j:>5} {avg_j:>+8.2f}% {wr_j:>5.1f}  {delta:>+6.2f}%')

print('-'*75)
avg_v_all = df_fri['spy_ret_pct'].dropna().mean()
avg_j_all = dfr['spy_ret_pct'].mean()
print(f'{"TOTAL":<16} {len(df_fri):>5} {avg_v_all:>+8.2f}%        {len(dfr):>5} {avg_j_all:>+8.2f}%        {avg_j_all-avg_v_all:>+6.2f}%')
