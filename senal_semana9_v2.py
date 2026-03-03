"""
Senal semana 9 de 2026 - Sistema Semanal_Subsectores
Viernes 20/02 al cierre -> compra lunes 24/02 open -> venta lunes 03/03 open
Usa la logica real del sistema: regimen + Fair Value + seleccion por regimen
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0
ATR_MIN = 1.5

def score_fair(active_events):
    contributions = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP: continue
        for subsec, impact in EVENT_SUBSECTOR_MAP[evt_type]['impacto'].items():
            if subsec not in contributions: contributions[subsec] = []
            contributions[subsec].append(intensity * impact)
    scores = {}
    for sub_id in SUBSECTORS:
        if sub_id not in contributions or len(contributions[sub_id]) == 0:
            scores[sub_id] = 5.0
        else:
            avg = np.mean(contributions[sub_id])
            scores[sub_id] = max(0.0, min(10.0, 5.0 + (avg / MAX_CONTRIBUTION) * 5.0))
    return scores

def adjust_score_by_price(scores, dd_row, rsi_row):
    adjusted = {}
    for sub_id, score in scores.items():
        dd_val = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi_val = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd_val): dd_val = 0
        if not pd.notna(rsi_val): rsi_val = 50
        if score < 5.0:
            dd_factor = np.clip((abs(dd_val) - 15) / 30, 0, 1)
            rsi_factor = np.clip((35 - rsi_val) / 20, 0, 1)
            oversold = max(dd_factor, rsi_factor)
            adjusted[sub_id] = score + (5.0 - score) * oversold * 0.5
        elif score > 5.0:
            rsi_factor = np.clip((rsi_val - 70) / 15, 0, 1)
            adjusted[sub_id] = score - (score - 5.0) * rsi_factor * 0.5
        else:
            adjusted[sub_id] = score
    return adjusted

# ================================================================
# CARGAR DATOS
# ================================================================
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
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100)

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_atr=('atr_pct', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_price_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')

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

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# ================================================================
# SENAL PARA VIERNES 20 FEB 2026
# ================================================================
target_date = pd.Timestamp('2026-02-20')

# Buscar fecha mas cercana en nuestros datos
all_dates = dd_wide.index[dd_wide.index <= target_date]
signal_date = all_dates[-1]
print(f"\nFecha senal: {signal_date.strftime('%Y-%m-%d')}")

# 1. REGIMEN
dd_row_reg = dd_wide.loc[signal_date]
rsi_row_reg = rsi_wide.loc[signal_date]
n_total = dd_row_reg.notna().sum()
pct_dd_healthy = (dd_row_reg > -10).sum() / n_total * 100
pct_dd_deep = (dd_row_reg < -20).sum() / n_total * 100
pct_rsi_above55 = (rsi_row_reg > 55).sum() / rsi_row_reg.notna().sum() * 100

spy_last = spy_w.loc[spy_w.index[spy_w.index <= target_date][-1]]
spy_above_ma200 = spy_last['above_ma200']
spy_mom_10w = spy_last['mom_10w'] if pd.notna(spy_last['mom_10w']) else 0
spy_dist = spy_last['dist_ma200'] if pd.notna(spy_last['dist_ma200']) else 0

vix_dates = vix_df.index[vix_df.index <= target_date]
vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
if not pd.notna(vix_val): vix_val = 20

if pct_dd_healthy >= 75: s1 = 2.0
elif pct_dd_healthy >= 60: s1 = 1.0
elif pct_dd_healthy >= 45: s1 = 0.0
elif pct_dd_healthy >= 30: s1 = -1.0
elif pct_dd_healthy >= 15: s1 = -2.0
else: s1 = -3.0
if pct_rsi_above55 >= 75: s2 = 2.0
elif pct_rsi_above55 >= 60: s2 = 1.0
elif pct_rsi_above55 >= 45: s2 = 0.0
elif pct_rsi_above55 >= 30: s2 = -1.0
elif pct_rsi_above55 >= 15: s2 = -2.0
else: s2 = -3.0
if pct_dd_deep <= 5: s3 = 1.5
elif pct_dd_deep <= 15: s3 = 0.5
elif pct_dd_deep <= 30: s3 = -0.5
elif pct_dd_deep <= 50: s3 = -1.5
else: s3 = -2.5
if spy_above_ma200 and spy_dist > 5: s4 = 1.5
elif spy_above_ma200: s4 = 0.5
elif spy_dist > -5: s4 = -0.5
elif spy_dist > -15: s4 = -1.5
else: s4 = -2.5
if spy_mom_10w > 5: s5 = 1.0
elif spy_mom_10w > 0: s5 = 0.5
elif spy_mom_10w > -5: s5 = -0.5
elif spy_mom_10w > -15: s5 = -1.0
else: s5 = -1.5

total = s1 + s2 + s3 + s4 + s5
is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)
if is_burbuja: regime = 'BURBUJA'
elif total >= 7.0: regime = 'GOLDILOCKS'
elif total >= 4.0: regime = 'ALCISTA'
elif total >= 0.5: regime = 'NEUTRAL'
elif total >= -2.0: regime = 'CAUTIOUS'
elif total >= -5.0: regime = 'BEARISH'
elif total >= -9.0: regime = 'CRISIS'
else: regime = 'PANICO'
if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'): regime = 'NEUTRAL'
elif vix_val >= 35 and regime == 'NEUTRAL': regime = 'CAUTIOUS'

print(f"\n{'='*100}")
print(f"  FASE 1 - REGIMEN DE MERCADO: {regime} (score {total:+.1f})")
print(f"{'='*100}")
print(f"  Componentes: BDD={s1:+.1f} BRSI={s2:+.1f} DDP={s3:+.1f} SPY={s4:+.1f} MOM={s5:+.1f}")
print(f"  SPY: ${spy_last['close']:.2f}  dist MA200: {spy_dist:+.1f}%  mom 10w: {spy_mom_10w:+.1f}%")
print(f"  VIX: {vix_val:.1f}  |  Breadth: DD healthy {pct_dd_healthy:.0f}%, DD deep {pct_dd_deep:.0f}%, RSI>55 {pct_rsi_above55:.0f}%")

# 2. FAIR VALUE + SELECCION POR REGIMEN
if signal_date in weekly_events.index:
    evt_date = signal_date
else:
    nearest_idx = weekly_events.index.get_indexer([signal_date], method='nearest')[0]
    evt_date = weekly_events.index[nearest_idx]

events_row = weekly_events.loc[evt_date]
active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

scores_evt = score_fair(active)
prev_dates = dd_wide.index[dd_wide.index < signal_date]
dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
atr_row = atr_wide.loc[signal_date] if signal_date in atr_wide.index else None
scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)

print(f"\n{'='*100}")
print(f"  FASE 2 - ESTRATEGIA: Semanal_Subsectores")
print(f"{'='*100}")

# Eventos activos
print(f"\n  Eventos macro activos ({len(active)}):")
for evt, intensity in sorted(active.items(), key=lambda x: -x[1]):
    print(f"    {evt}: intensidad {intensity:.1f}")

# Scores FV de todos los subsectores
print(f"\n  {'Subsector':<40} {'FV raw':>7} {'FV adj':>7} {'DD 52w':>8} {'RSI 14w':>8} {'ATR%':>6}")
print(f"  {'-'*75}")

sub_data_list = []
for sub_id in sorted(SUBSECTORS.keys()):
    fv_raw = scores_evt.get(sub_id, 5.0)
    fv_adj = scores_v3.get(sub_id, 5.0)
    dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
    rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
    atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
    if not pd.notna(dd): dd = 0
    if not pd.notna(rsi): rsi = 50
    if not pd.notna(atr): atr = 0
    name = SUBSECTORS[sub_id]['label']
    sub_data_list.append({
        'id': sub_id, 'name': name, 'fv_raw': fv_raw, 'fv_adj': fv_adj,
        'dd': dd, 'rsi': rsi, 'atr': atr,
        'tickers': SUBSECTORS[sub_id]['tickers']
    })

sub_data_list.sort(key=lambda x: -x['fv_adj'])
for s in sub_data_list:
    marker = ''
    if s['fv_adj'] > 6.5: marker = ' <<< LONG pool'
    elif s['fv_adj'] < 3.5: marker = ' <<< SHORT pool'
    print(f"  {s['name']:<40} {s['fv_raw']:>6.2f} {s['fv_adj']:>6.2f} {s['dd']:>+7.1f}% {s['rsi']:>7.1f} {s['atr']:>5.1f}{marker}")

# Seleccion segun regimen
print(f"\n{'='*100}")
print(f"  FASE 2 - SELECCION (regimen {regime})")
print(f"{'='*100}")

if regime in ('ALCISTA', 'GOLDILOCKS'):
    # Top3 FairValue: los 3 subsectores con mayor score > 5.5
    top3 = sorted([(s['id'], s['fv_adj'], s['name']) for s in sub_data_list if s['fv_adj'] > 5.5],
                  key=lambda x: -x[1])[:3]
    print(f"\n  Estrategia: Top 3 Fair Value (score > 5.5)")
    print(f"  Posicion: 3L + 0S")
    print(f"\n  LONGS seleccionados:")
    total_w = sum(x[1] - 5.0 for x in top3)
    for i, (sub_id, score, name) in enumerate(top3, 1):
        weight = (score - 5.0) / total_w * 100
        tickers = SUBSECTORS[sub_id]['tickers']
        print(f"    {i}. {name} (FV={score:.2f}, peso={weight:.0f}%)")
        print(f"       Tickers: {', '.join(tickers)}")
    selected = top3
    direction = 'LONG'

elif regime == 'BURBUJA':
    # Burbuja aggressive
    from report_compound import decide_burbuja_aggressive
    # Inline
    candidates = []
    for s in sub_data_list:
        if s['fv_adj'] <= 6.0 or s['dd'] < -8 or s['rsi'] < 55: continue
        ms = np.clip((s['fv_adj'] - 6.0) / 2.5, 0, 1) * 2.5 + np.clip((8 + s['dd']) / 8, 0, 1) * 2.0 + np.clip((s['rsi'] - 55) / 25, 0, 1) * 1.5
        candidates.append((s['id'], ms, s['name'], s['fv_adj']))
    candidates.sort(key=lambda x: -x[1])
    top3 = candidates[:3]
    print(f"\n  Estrategia: Burbuja Agresiva (FV > 6.0, near ATH, RSI > 55)")
    print(f"  Posicion: 3L + 0S")
    print(f"\n  LONGS seleccionados:")
    for i, (sub_id, ms, name, fv) in enumerate(top3, 1):
        tickers = SUBSECTORS[sub_id]['tickers']
        print(f"    {i}. {name} (FV={fv:.2f}, momentum_score={ms:.2f})")
        print(f"       Tickers: {', '.join(tickers)}")
    selected = top3
    direction = 'LONG'

elif regime in ('NEUTRAL', 'CAUTIOUS'):
    # Oversold deep: DD < -15%, RSI < 35, score >= 3.5
    candidates = []
    for s in sub_data_list:
        if s['dd'] > -15 or s['rsi'] > 35 or s['fv_adj'] < 3.5: continue
        w = np.clip((abs(s['dd']) - 15) / 20, 0, 1) + np.clip((35 - s['rsi']) / 15, 0, 1)
        candidates.append((s['id'], w, s['name'], s['fv_adj'], s['dd'], s['rsi']))
    candidates.sort(key=lambda x: -x[1])
    top3 = candidates[:3]
    if top3:
        print(f"\n  Estrategia: Oversold Deep (DD < -15%, RSI < 35, FV >= 3.5)")
        print(f"  Posicion: {len(top3)}L + 0S")
        print(f"\n  LONGS seleccionados:")
        for i, (sub_id, w, name, fv, dd, rsi) in enumerate(top3, 1):
            tickers = SUBSECTORS[sub_id]['tickers']
            print(f"    {i}. {name} (FV={fv:.2f}, DD={dd:+.1f}%, RSI={rsi:.0f})")
            print(f"       Tickers: {', '.join(tickers)}")
    else:
        print(f"\n  Estrategia: Oversold Deep -> NO HAY CANDIDATOS")
        print(f"  Posicion: 0L + 0S (no operar)")
    selected = top3
    direction = 'LONG'

elif regime in ('BEARISH', 'CRISIS', 'PANICO'):
    # Bear Aggressive: score < 4.5, DD > -25, RSI > 25, ATR >= 1.5
    candidates = []
    for s in sub_data_list:
        if s['fv_adj'] >= 4.5 or s['dd'] < -25 or s['rsi'] < 25 or s['atr'] < ATR_MIN: continue
        w = (np.clip((5.0 - s['fv_adj']) / 3.0, 0, 1) * 2.0 +
             np.clip(abs(s['dd']) / 20.0, 0, 1) * 1.5 +
             np.clip((50 - s['rsi']) / 25.0, 0, 1) * 1.5 +
             np.clip((s['atr'] - ATR_MIN) / 3.0, 0, 1) * 1.0)
        candidates.append((s['id'], w, s['name'], s['fv_adj'], s['dd'], s['rsi'], s['atr']))
    candidates.sort(key=lambda x: -x[1])
    top3 = candidates[:3]
    if top3:
        print(f"\n  Estrategia: Bear Aggressive (FV < 4.5, DD > -25, RSI > 25, ATR >= 1.5)")
        print(f"  Posicion: 0L + {len(top3)}S")
        print(f"\n  SHORTS seleccionados:")
        for i, (sub_id, w, name, fv, dd, rsi, atr) in enumerate(top3, 1):
            tickers = SUBSECTORS[sub_id]['tickers']
            print(f"    {i}. {name} (FV={fv:.2f}, DD={dd:+.1f}%, RSI={rsi:.0f}, ATR={atr:.1f}%)")
            print(f"       Tickers: {', '.join(tickers)}")
    else:
        print(f"\n  Estrategia: Bear Aggressive -> NO HAY CANDIDATOS")
        print(f"  Posicion: 0L + 0S (fallback a bottom 3 FV)")
    selected = top3
    direction = 'SHORT'

# ================================================================
# RESUMEN FINAL
# ================================================================
print(f"\n{'='*100}")
print(f"  RESUMEN SENAL SEMANA 9")
print(f"{'='*100}")
print(f"  Regimen: {regime} (score {total:+.1f})")
print(f"  Timing: compra lunes 24/02 open -> venta lunes 03/03 open")
print(f"  Direccion: {direction}")
if selected:
    print(f"  Posiciones:")
    for i, item in enumerate(selected, 1):
        sub_id = item[0]
        name = item[2]
        tickers = SUBSECTORS[sub_id]['tickers']
        print(f"    {i}. {direction} {name}")
        print(f"       Tickers: {', '.join(tickers)}")
else:
    print(f"  NO OPERAR esta semana")
