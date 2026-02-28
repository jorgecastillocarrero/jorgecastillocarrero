"""
Senal Semana 9 de 2026: Subsectores + 10 Acciones
===================================================
Viernes 21/02 cierre -> Compra lunes 24/02 open -> Venta lunes 03/03 open
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

REGIME_ALLOC = {
    'BURBUJA': (10, 0), 'GOLDILOCKS': (7, 3), 'ALCISTA': (7, 3),
    'NEUTRAL': (5, 5), 'CAUTIOUS': (5, 5),
    'BEARISH': (3, 7), 'CRISIS': (0, 10), 'PANICO': (0, 10),
    'CAPITULACION': (10, 0),
}

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
    AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year

# Weekly por accion
df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100)

# Metricas individuales
def calc_stock_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['high'].rolling(52, min_periods=26).max()
    g['dd_52w'] = (g['close'] / g['high_52w'] - 1) * 100
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    g['mom_12w'] = g['close'].pct_change(12) * 100
    return g

df_weekly = df_weekly.groupby('symbol', group_keys=False).apply(calc_stock_metrics)

# Subsector aggregates
sub_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
sub_weekly = sub_weekly.sort_values(['symbol', 'date'])
sub_agg = sub_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean')).reset_index()
sub_agg = sub_agg.sort_values(['subsector', 'date'])

# ATR subsector
sub_weekly_ret = df_weekly.groupby(['subsector', 'date']).agg(
    avg_atr=('atr_pct', 'mean')).reset_index()
sub_agg = sub_agg.merge(sub_weekly_ret, on=['subsector', 'date'], how='left')

date_counts = sub_agg.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_agg = sub_agg[sub_agg['date'].isin(valid_dates)]

def calc_sub_metrics(g):
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

sub_agg = sub_agg.groupby('subsector', group_keys=False).apply(calc_sub_metrics)
dd_wide = sub_agg.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_agg.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide = sub_agg.pivot(index='date', columns='subsector', values='avg_atr')

# SPY
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

vix_df = pd.read_sql("""
    SELECT date, close as vix FROM price_history_vix
    WHERE symbol='^VIX' ORDER BY date
""", engine)
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.set_index('date')

weekly_events = build_weekly_events('2000-01-01', '2026-02-28')

# ================================================================
# FECHA SENAL
# ================================================================
target_date = pd.Timestamp('2026-02-26')
all_dates = dd_wide.index[dd_wide.index <= target_date]
signal_date = all_dates[-1]
print(f"Fecha senal: {signal_date.strftime('%Y-%m-%d')}")
print(f"Ultimo dia datos SPY: {spy_daily.index[-1].strftime('%Y-%m-%d')}")

# ================================================================
# FASE 1: REGIMEN DE MERCADO
# ================================================================
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

# Scoring
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
# CAPITULACION: PANICO + VIX bajando vs semana anterior
if regime == 'PANICO':
    vix_dates_prev = vix_df.index[vix_df.index <= target_date]
    if len(vix_dates_prev) >= 2:
        prev_vix = vix_df.loc[vix_dates_prev[-2], 'vix']
        if pd.notna(prev_vix) and vix_val < prev_vix:
            regime = 'CAPITULACION'

print(f"\n{'='*100}")
print(f"  FASE 1 - REGIMEN DE MERCADO: {regime} (score {total:+.1f})")
print(f"{'='*100}")
print(f"  BDD={s1:+.1f} BRSI={s2:+.1f} DDP={s3:+.1f} SPY={s4:+.1f} MOM={s5:+.1f}")
print(f"  SPY: ${spy_last['close']:.2f} | dist MA200: {spy_dist:+.1f}% | mom 10w: {spy_mom_10w:+.1f}%")
print(f"  VIX: {vix_val:.1f} | Breadth: DD>-10%={pct_dd_healthy:.0f}% DD<-20%={pct_dd_deep:.0f}% RSI>55={pct_rsi_above55:.0f}%")

# ================================================================
# FASE 2: FAIR VALUE
# ================================================================
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

print(f"\n  Eventos activos ({len(active)}):")
for evt, intensity in sorted(active.items(), key=lambda x: -x[1]):
    print(f"    {evt}: {intensity:.1f}")

# Subsector data
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
    sub_data_list.append({
        'id': sub_id, 'name': SUBSECTORS[sub_id]['label'],
        'fv_raw': fv_raw, 'fv_adj': fv_adj,
        'dd': dd, 'rsi': rsi, 'atr': atr,
    })
sub_data_list.sort(key=lambda x: -x['fv_adj'])

# ================================================================
# ESTRATEGIA A: SUBSECTORES (3 posiciones)
# ================================================================
print(f"\n{'='*100}")
print(f"  ESTRATEGIA A: SEMANAL_SUBSECTORES (regimen {regime})")
print(f"{'='*100}")

if regime in ('ALCISTA', 'GOLDILOCKS', 'CAPITULACION'):
    top3 = sorted([(s['id'], s['fv_adj'], s['name']) for s in sub_data_list if s['fv_adj'] > 5.5],
                  key=lambda x: -x[1])[:3]
    print(f"\n  Top 3 Fair Value (3L + 0S):")
    total_w = sum(x[1] - 5.0 for x in top3) if top3 else 1
    for i, (sub_id, score, name) in enumerate(top3, 1):
        weight = (score - 5.0) / total_w * 100
        tickers = SUBSECTORS[sub_id]['tickers']
        print(f"    {i}. LONG {name} (FV={score:.2f}, peso={weight:.0f}%)")
        print(f"       Tickers: {', '.join(tickers)}")

elif regime == 'BURBUJA':
    candidates = []
    for s in sub_data_list:
        if s['fv_adj'] <= 6.0 or s['dd'] < -8 or s['rsi'] < 55: continue
        ms = (np.clip((s['fv_adj'] - 6.0) / 2.5, 0, 1) * 2.5 +
              np.clip((8 + s['dd']) / 8, 0, 1) * 2.0 +
              np.clip((s['rsi'] - 55) / 25, 0, 1) * 1.5)
        candidates.append((s['id'], ms, s['name'], s['fv_adj']))
    candidates.sort(key=lambda x: -x[1])
    top3 = candidates[:3]
    print(f"\n  Burbuja Agresiva (3L + 0S):")
    for i, (sub_id, ms, name, fv) in enumerate(top3, 1):
        tickers = SUBSECTORS[sub_id]['tickers']
        print(f"    {i}. LONG {name} (FV={fv:.2f}, score={ms:.2f})")
        print(f"       Tickers: {', '.join(tickers)}")

elif regime in ('NEUTRAL', 'CAUTIOUS'):
    candidates = []
    for s in sub_data_list:
        if s['dd'] > -15 or s['rsi'] > 35 or s['fv_adj'] < 3.5: continue
        w = np.clip((abs(s['dd']) - 15) / 20, 0, 1) + np.clip((35 - s['rsi']) / 15, 0, 1)
        candidates.append((s['id'], w, s['name'], s['fv_adj'], s['dd'], s['rsi']))
    candidates.sort(key=lambda x: -x[1])
    top3 = candidates[:3]
    if top3:
        print(f"\n  Oversold Deep ({len(top3)}L + 0S):")
        for i, (sub_id, w, name, fv, dd, rsi) in enumerate(top3, 1):
            tickers = SUBSECTORS[sub_id]['tickers']
            print(f"    {i}. LONG {name} (FV={fv:.2f}, DD={dd:+.1f}%, RSI={rsi:.0f})")
            print(f"       Tickers: {', '.join(tickers)}")
    else:
        print(f"\n  Oversold Deep: NO HAY CANDIDATOS -> NO OPERAR")

elif regime in ('BEARISH', 'CRISIS', 'PANICO'):
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
        print(f"\n  Bear Aggressive (0L + {len(top3)}S):")
        for i, (sub_id, w, name, fv, dd, rsi, atr) in enumerate(top3, 1):
            tickers = SUBSECTORS[sub_id]['tickers']
            print(f"    {i}. SHORT {name} (FV={fv:.2f}, DD={dd:+.1f}%, RSI={rsi:.0f}, ATR={atr:.1f}%)")
            print(f"       Tickers: {', '.join(tickers)}")
    else:
        print(f"\n  Bear Aggressive: NO HAY CANDIDATOS -> NO OPERAR")

# ================================================================
# ESTRATEGIA B: 10 ACCIONES INDIVIDUALES
# ================================================================
print(f"\n{'='*100}")
print(f"  ESTRATEGIA B: 10 ACCIONES INDIVIDUALES (regimen {regime})")
print(f"{'='*100}")

n_long, n_short = REGIME_ALLOC[regime]
print(f"  Asignacion: {n_long}L + {n_short}S")

# Datos individuales para la fecha de senal
stock_data = df_weekly[df_weekly['date'] <= signal_date].sort_values(['symbol', 'date'])
latest_stocks = stock_data.groupby('symbol').last().reset_index()

# LONGS
long_candidates = []
if n_long > 0:
    good_subs = {s['id']: s['fv_adj'] for s in sub_data_list if s['fv_adj'] > 5.5}
    for _, stock in latest_stocks.iterrows():
        ticker = stock['symbol']
        sub = ticker_to_sub.get(ticker)
        if sub not in good_subs: continue
        fv = good_subs[sub]
        mom = stock.get('mom_12w', 0)
        rsi = stock.get('rsi_14w', 50)
        dd = stock.get('dd_52w', 0)
        price = stock.get('close', 0)
        if not pd.notna(mom): mom = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(dd): dd = 0

        composite = 0.0
        composite += np.clip((fv - 5.0) / 4.0, 0, 1) * 3.0
        composite += np.clip((mom + 20) / 60, 0, 1) * 3.0
        composite += np.clip((rsi - 30) / 50, 0, 1) * 2.0
        composite += np.clip((dd + 30) / 30, 0, 1) * 2.0

        long_candidates.append({
            'ticker': ticker, 'sub': sub, 'sub_name': SUBSECTORS[sub]['label'],
            'fv': fv, 'mom': mom, 'rsi': rsi, 'dd': dd, 'price': price,
            'score': composite,
        })

# SHORTS
short_candidates = []
if n_short > 0:
    bad_subs = {s['id']: s['fv_adj'] for s in sub_data_list if s['fv_adj'] < 4.5}
    for _, stock in latest_stocks.iterrows():
        ticker = stock['symbol']
        sub = ticker_to_sub.get(ticker)
        if sub not in bad_subs: continue
        fv = bad_subs[sub]
        mom = stock.get('mom_12w', 0)
        rsi = stock.get('rsi_14w', 50)
        dd = stock.get('dd_52w', 0)
        atr = stock.get('atr_pct', 0)
        price = stock.get('close', 0)
        if not pd.notna(mom): mom = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(atr): atr = 0

        if dd < -40 or rsi < 15 or atr < 1.5: continue

        composite = 0.0
        composite += np.clip((5.0 - fv) / 4.0, 0, 1) * 3.0
        composite += np.clip((-mom + 20) / 60, 0, 1) * 3.0
        composite += np.clip((70 - rsi) / 50, 0, 1) * 2.0
        composite += np.clip((atr - 1.5) / 5.0, 0, 1) * 2.0

        short_candidates.append({
            'ticker': ticker, 'sub': sub, 'sub_name': SUBSECTORS[sub]['label'],
            'fv': fv, 'mom': mom, 'rsi': rsi, 'dd': dd, 'price': price,
            'atr': atr, 'score': composite,
        })

# Ordenar
long_candidates.sort(key=lambda x: -x['score'])
short_candidates.sort(key=lambda x: -x['score'])

# Diversificar: max 2 por subsector
def diversified_pick(candidates, n, max_per_sub=2):
    picked = []
    sub_count = {}
    for c in candidates:
        s = c['sub']
        if sub_count.get(s, 0) >= max_per_sub: continue
        picked.append(c)
        sub_count[s] = sub_count.get(s, 0) + 1
        if len(picked) >= n: break
    return picked

longs_sel = diversified_pick(long_candidates, n_long)
shorts_sel = diversified_pick(short_candidates, n_short)

if longs_sel:
    print(f"\n  LONGS ({len(longs_sel)}):")
    print(f"  {'#':>3} {'Ticker':<8} {'Precio':>8} {'Subsector':<30} {'FV':>5} {'Mom12w':>8} {'RSI':>5} {'DD52w':>7} {'Score':>6}")
    print(f"  {'-'*90}")
    for i, c in enumerate(longs_sel, 1):
        print(f"  {i:>3} {c['ticker']:<8} ${c['price']:>7.2f} {c['sub_name']:<30} {c['fv']:>4.1f} {c['mom']:>+7.1f}% {c['rsi']:>4.0f} {c['dd']:>+6.1f}% {c['score']:>5.1f}")

if shorts_sel:
    print(f"\n  SHORTS ({len(shorts_sel)}):")
    print(f"  {'#':>3} {'Ticker':<8} {'Precio':>8} {'Subsector':<30} {'FV':>5} {'Mom12w':>8} {'RSI':>5} {'DD52w':>7} {'ATR%':>5} {'Score':>6}")
    print(f"  {'-'*98}")
    for i, c in enumerate(shorts_sel, 1):
        print(f"  {i:>3} {c['ticker']:<8} ${c['price']:>7.2f} {c['sub_name']:<30} {c['fv']:>4.1f} {c['mom']:>+7.1f}% {c['rsi']:>4.0f} {c['dd']:>+6.1f}% {c['atr']:>4.1f} {c['score']:>5.1f}")

if not longs_sel and not shorts_sel:
    print(f"\n  NO HAY CANDIDATOS -> NO OPERAR")

# ================================================================
# RESUMEN
# ================================================================
print(f"\n{'='*100}")
print(f"  RESUMEN SENAL SEMANA 9 (2026)")
print(f"  Compra: lunes 24/02 open | Venta: lunes 03/03 open")
print(f"{'='*100}")
print(f"\n  Regimen: {regime} (score {total:+.1f})")
print(f"\n  ESTRATEGIA A - Subsectores:")
if regime in ('ALCISTA', 'GOLDILOCKS', 'CAPITULACION') and top3:
    for i, (sub_id, score, name) in enumerate(top3, 1):
        print(f"    {i}. LONG {name} ({SUBSECTORS[sub_id]['etf']})")
elif regime in ('NEUTRAL', 'CAUTIOUS'):
    if 'top3' in dir() and top3:
        for i, item in enumerate(top3, 1):
            print(f"    {i}. LONG {item[2]}")
    else:
        print(f"    NO OPERAR")
elif regime in ('BEARISH', 'CRISIS', 'PANICO'):
    if 'top3' in dir() and top3:
        for i, item in enumerate(top3, 1):
            print(f"    {i}. SHORT {item[2]}")
    else:
        print(f"    NO OPERAR")

print(f"\n  ESTRATEGIA B - 10 Acciones ({n_long}L+{n_short}S):")
if longs_sel:
    tickers_l = [c['ticker'] for c in longs_sel]
    print(f"    LONGS:  {', '.join(tickers_l)}")
if shorts_sel:
    tickers_s = [c['ticker'] for c in shorts_sel]
    print(f"    SHORTS: {', '.join(tickers_s)}")
if not longs_sel and not shorts_sel:
    print(f"    NO OPERAR")
