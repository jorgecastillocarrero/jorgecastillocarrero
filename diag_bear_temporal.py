"""
Verificar estructura temporal: bear markets son cortos e intensos
Cuanto duran los episodios? Son consecutivos? Transiciones?
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# --- Cargar datos (solo lo necesario) ---
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

# Calcular nuevo score para todas las semanas
print("Calculando scores...")
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

    # Nuevo scoring extendido
    if pct_dd_healthy >= 75: s_bdd = 2.0
    elif pct_dd_healthy >= 60: s_bdd = 1.0
    elif pct_dd_healthy >= 45: s_bdd = 0.0
    elif pct_dd_healthy >= 30: s_bdd = -1.0
    elif pct_dd_healthy >= 15: s_bdd = -2.0
    else: s_bdd = -3.0

    if pct_rsi_above55 >= 75: s_brsi = 2.0
    elif pct_rsi_above55 >= 60: s_brsi = 1.0
    elif pct_rsi_above55 >= 45: s_brsi = 0.0
    elif pct_rsi_above55 >= 30: s_brsi = -1.0
    elif pct_rsi_above55 >= 15: s_brsi = -2.0
    else: s_brsi = -3.0

    if pct_dd_deep <= 5: s_ddp = 1.5
    elif pct_dd_deep <= 15: s_ddp = 0.5
    elif pct_dd_deep <= 30: s_ddp = -0.5
    elif pct_dd_deep <= 50: s_ddp = -1.5
    else: s_ddp = -2.5

    if spy_above_ma200 and spy_dist > 5: s_spy = 1.5
    elif spy_above_ma200: s_spy = 0.5
    elif spy_dist > -5: s_spy = -0.5
    elif spy_dist > -15: s_spy = -1.5
    else: s_spy = -2.5

    if spy_mom_10w > 5: s_mom = 1.0
    elif spy_mom_10w > 0: s_mom = 0.5
    elif spy_mom_10w > -5: s_mom = -0.5
    elif spy_mom_10w > -15: s_mom = -1.0
    else: s_mom = -1.5

    total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

    # Regimen con propuesta C: -2/-5/-9
    if total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90: reg = 'BURBUJA'
    elif total >= 7.0: reg = 'GOLDILOCKS'
    elif total >= 4.0: reg = 'ALCISTA'
    elif total >= 0.5: reg = 'NEUTRAL'
    elif total >= -2.0: reg = 'CAUTIOUS'
    elif total >= -5.0: reg = 'BEARISH'
    elif total >= -9.0: reg = 'CRISIS'
    else: reg = 'PANICO'

    # VIX override
    if vix_val >= 30 and reg in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        reg = 'NEUTRAL'
    elif vix_val >= 35 and reg == 'NEUTRAL':
        reg = 'CAUTIOUS'

    rows.append({
        'date': date, 'year': date.year, 'score': total, 'regime': reg,
        'vix': vix_val, 'spy_ret': spy_ret, 'spy_dist': spy_dist,
        'pct_dd_deep': pct_dd_deep, 'pct_dd_healthy': pct_dd_healthy,
    })

df = pd.DataFrame(rows)
df = df.sort_values('date').reset_index(drop=True)

# ================================================================
# 1. TIMELINE: secuencia de regimenes por ano
# ================================================================
print(f"\n{'='*130}")
print(f"  TIMELINE: SECUENCIA DE REGIMENES (Propuesta C)")
print(f"{'='*130}")

regime_map = {'BURBUJA': 'B', 'GOLDILOCKS': 'G', 'ALCISTA': 'A', 'NEUTRAL': 'N',
              'CAUTIOUS': 'c', 'BEARISH': 'b', 'CRISIS': 'X', 'PANICO': 'P'}

for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year]
    seq = ''.join(regime_map.get(r, '?') for r in yr['regime'])
    counts = yr['regime'].value_counts().to_dict()
    cnt_str = ' '.join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"  {year}: {seq}  [{cnt_str}]")

print(f"\n  Leyenda: B=BURBUJA G=GOLDILOCKS A=ALCISTA N=NEUTRAL c=CAUTIOUS b=BEARISH X=CRISIS P=PANICO")

# ================================================================
# 2. DURACION DE EPISODIOS BEAR/CRISIS/PANICO
# ================================================================
print(f"\n{'='*130}")
print(f"  DURACION DE EPISODIOS NEGATIVOS (secuencias consecutivas)")
print(f"{'='*130}")

negative_regs = {'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO'}
episodes = []
current_ep = None

for _, row in df.iterrows():
    is_neg = row['regime'] in negative_regs
    if is_neg:
        if current_ep is None:
            current_ep = {'start': row['date'], 'end': row['date'], 'weeks': 1,
                         'regimes': [row['regime']], 'scores': [row['score']],
                         'spy_rets': [row['spy_ret']], 'vix_vals': [row['vix']]}
        else:
            current_ep['end'] = row['date']
            current_ep['weeks'] += 1
            current_ep['regimes'].append(row['regime'])
            current_ep['scores'].append(row['score'])
            current_ep['spy_rets'].append(row['spy_ret'])
            current_ep['vix_vals'].append(row['vix'])
    else:
        if current_ep is not None:
            episodes.append(current_ep)
            current_ep = None

if current_ep is not None:
    episodes.append(current_ep)

print(f"\n  Total episodios negativos: {len(episodes)}")
print(f"  Duracion media: {np.mean([e['weeks'] for e in episodes]):.1f} sem")
print(f"  Duracion mediana: {np.median([e['weeks'] for e in episodes]):.0f} sem")
print(f"  Min/Max: {min(e['weeks'] for e in episodes)}/{max(e['weeks'] for e in episodes)} sem")

print(f"\n  Distribucion de duracion:")
dur_bins = [(1, 1), (2, 3), (4, 8), (9, 20), (21, 52), (53, 999)]
dur_labels = ['1 sem', '2-3 sem', '4-8 sem', '9-20 sem', '21-52 sem', '>52 sem']
for (lo, hi), label in zip(dur_bins, dur_labels):
    eps = [e for e in episodes if lo <= e['weeks'] <= hi]
    if not eps: continue
    print(f"    {label:>10s}: {len(eps):>3d} episodios")

# ================================================================
# 3. EPISODIOS LARGOS (>8 semanas) - detalle
# ================================================================
print(f"\n{'='*130}")
print(f"  EPISODIOS NEGATIVOS > 8 SEMANAS")
print(f"{'='*130}")
print(f"  {'Inicio':<12s} {'Fin':<12s} {'Sem':>4s} {'Score min':>10s} {'VIX max':>8s} {'SPY tot':>8s} {'Secuencia regimenes'}")
print(f"  {'-'*110}")

long_eps = sorted([e for e in episodes if e['weeks'] > 8], key=lambda x: x['start'])
for ep in long_eps:
    seq = ''.join(regime_map.get(r, '?') for r in ep['regimes'])
    # Condensar secuencia
    if len(seq) > 50:
        seq = seq[:25] + '...' + seq[-25:]
    spy_total = sum(ep['spy_rets']) * 100
    print(f"  {ep['start'].strftime('%Y-%m-%d'):<12s} {ep['end'].strftime('%Y-%m-%d'):<12s} {ep['weeks']:>4d} "
          f"{min(ep['scores']):>+10.1f} {max(ep['vix_vals']):>7.1f} {spy_total:>+7.1f}% {seq}")

# ================================================================
# 4. TRANSICIONES: de que regimen se entra y a que se sale
# ================================================================
print(f"\n{'='*130}")
print(f"  TRANSICIONES ENTRE REGIMENES (semana a semana)")
print(f"{'='*130}")

transitions = {}
for i in range(1, len(df)):
    prev_reg = df.iloc[i-1]['regime']
    curr_reg = df.iloc[i]['regime']
    if prev_reg != curr_reg:
        key = f"{prev_reg} -> {curr_reg}"
        transitions[key] = transitions.get(key, 0) + 1

print(f"\n  Transiciones mas comunes:")
for key, count in sorted(transitions.items(), key=lambda x: -x[1])[:25]:
    print(f"    {key:<35s}: {count:>4d}")

# ================================================================
# 5. PATRON TIPICO DE UN BEAR MARKET
# ================================================================
print(f"\n{'='*130}")
print(f"  PATRON TIPICO: SECUENCIA DE REGIMENES EN BEAR MARKETS CONOCIDOS")
print(f"{'='*130}")

bear_periods = [
    ('2001-02', '2002-12', 'Dot-com crash'),
    ('2007-10', '2009-06', 'GFC'),
    ('2020-02', '2020-06', 'COVID'),
    ('2022-01', '2022-12', 'Rate hike bear'),
]

for start, end, name in bear_periods:
    sub = df[(df['date'] >= start) & (df['date'] <= end)]
    seq = ''.join(regime_map.get(r, '?') for r in sub['regime'])
    counts = sub['regime'].value_counts().to_dict()
    cnt_str = ' '.join(f"{k}:{v}" for k, v in sorted(counts.items()))
    spy_total = sub['spy_ret'].sum() * 100

    # Fase de entrada (primeras 4 sem negativas)
    neg_start = sub[sub['regime'].isin(negative_regs)].head(4)
    entry_regs = ', '.join(neg_start['regime'].tolist()) if len(neg_start) > 0 else 'N/A'

    # Fase de salida (ultimas 4 sem negativas)
    neg_end = sub[sub['regime'].isin(negative_regs)].tail(4)
    exit_regs = ', '.join(neg_end['regime'].tolist()) if len(neg_end) > 0 else 'N/A'

    print(f"\n  {name} ({start} a {end}): {len(sub)} sem, SPY={spy_total:+.1f}%")
    print(f"    Secuencia: {seq}")
    print(f"    Conteo: {cnt_str}")
    print(f"    Entrada: {entry_regs}")
    print(f"    Salida:  {exit_regs}")
    print(f"    Score: min={sub['score'].min():+.1f} max={sub['score'].max():+.1f} avg={sub['score'].mean():+.1f}")

# ================================================================
# 6. VELOCIDAD: cuantas semanas de CAUTIOUS antes de PANICO?
# ================================================================
print(f"\n{'='*130}")
print(f"  VELOCIDAD DE DETERIORO: semanas desde primera señal negativa hasta PANICO")
print(f"{'='*130}")

for ep in long_eps:
    if 'PANICO' not in ep['regimes']: continue
    first_neg_idx = 0  # ya es negativo
    first_panic_idx = ep['regimes'].index('PANICO')
    weeks_to_panic = first_panic_idx
    entry_seq = '->'.join(dict.fromkeys(ep['regimes'][:first_panic_idx+1]))  # unique regimes in order
    print(f"  {ep['start'].strftime('%Y-%m-%d')}: {weeks_to_panic} sem hasta PANICO  "
          f"[{entry_seq}]  VIX max={max(ep['vix_vals']):>.0f}")
