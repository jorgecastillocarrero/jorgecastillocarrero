"""
Diagnostico NEUTRAL: optimizar pata LARGA y pata CORTA por separado
Luego decidir: mejor combinacion o eliminar un lado
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
COST_PER_TRADE = 0.0010
SLIPPAGE_PER_SIDE = 0.0005
COST_RT = (COST_PER_TRADE + SLIPPAGE_PER_SIDE) * 2  # 0.30%

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

def classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: return 'NEUTRAL', 0.0
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL', 0.0
    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50
    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) > 0:
        spy_last = spy_w.loc[spy_dates[-1]]
        spy_above_ma200 = spy_last.get('above_ma200', 0.5)
        spy_mom_10w = spy_last.get('mom_10w', 0)
        spy_dist = spy_last.get('dist_ma200', 0)
    else:
        spy_above_ma200 = 0.5; spy_mom_10w = 0; spy_dist = 0
    if not pd.notna(spy_mom_10w): spy_mom_10w = 0
    if not pd.notna(spy_dist): spy_dist = 0
    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    if pct_dd_healthy >= 75: score_bdd = 2.0
    elif pct_dd_healthy >= 60: score_bdd = 1.0
    elif pct_dd_healthy >= 45: score_bdd = 0.0
    elif pct_dd_healthy >= 30: score_bdd = -1.0
    else: score_bdd = -2.0
    if pct_rsi_above55 >= 75: score_brsi = 2.0
    elif pct_rsi_above55 >= 60: score_brsi = 1.0
    elif pct_rsi_above55 >= 45: score_brsi = 0.0
    elif pct_rsi_above55 >= 30: score_brsi = -1.0
    else: score_brsi = -2.0
    if pct_dd_deep <= 5: score_ddp = 1.5
    elif pct_dd_deep <= 15: score_ddp = 0.5
    elif pct_dd_deep <= 30: score_ddp = -0.5
    else: score_ddp = -1.5
    if spy_above_ma200 and spy_dist > 5: score_spy = 1.5
    elif spy_above_ma200: score_spy = 0.5
    elif spy_dist > -5: score_spy = -0.5
    else: score_spy = -1.5
    if spy_mom_10w > 5: score_mom = 1.0
    elif spy_mom_10w > 0: score_mom = 0.5
    elif spy_mom_10w > -5: score_mom = -0.5
    else: score_mom = -1.0

    total = score_bdd + score_brsi + score_ddp + score_spy + score_mom
    is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -1.5: regime = 'CAUTIOUS'
    elif total >= -3.0: regime = 'BEARISH'
    else: regime = 'CRISIS'
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'CAUTIOUS'
    return regime, total

# ================================================================
# CARGAR DATOS (identico a report_compound)
# ================================================================
print("Cargando datos...")

ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

df_all = pd.read_sql(f"""
    SELECT symbol, date, open, close, high, low
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

df_monday = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).first().reset_index()
df_monday = df_monday.sort_values(['symbol', 'date'])
df_monday['prev_open'] = df_monday.groupby('symbol')['open'].shift(1)
df_monday['return_mon'] = df_monday['open'] / df_monday['prev_open'] - 1
df_monday = df_monday.dropna(subset=['return_mon'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

sub_monday = df_monday.groupby(['subsector', 'date']).agg(
    avg_return_mon=('return_mon', 'mean')).reset_index()
sub_monday = sub_monday.sort_values(['subsector', 'date'])

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

returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide_lagged = atr_wide.shift(1)

returns_mon_wide = sub_monday.pivot(index='date', columns='subsector', values='avg_return_mon')
mon_dates = returns_mon_wide.index.tolist()
fri_dates = returns_wide.index.tolist()
fri_to_mon_ret = {}
for fri in fri_dates:
    target = fri + pd.Timedelta(days=3)
    diffs = [abs((d - target).days) for d in mon_dates]
    if diffs:
        closest_mon = mon_dates[diffs.index(min(diffs))]
        if abs((closest_mon - target).days) <= 3:
            fri_to_mon_ret[fri] = closest_mon
returns_trade_wide = pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns, dtype=float)
for fri, mon in fri_to_mon_ret.items():
    if mon in returns_mon_wide.index:
        returns_trade_wide.loc[fri] = returns_mon_wide.loc[mon]

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

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# ================================================================
# ESTRATEGIAS PATA LARGA (cada una opera independiente)
# ================================================================
# Cada estrategia devuelve lista de (subsector, weight)

def long_current_meanrev(scores_evt, scores_v3, dd_row, rsi_row):
    """Actual: oversold (DD profundo o RSI bajo) + score >= 4.0"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_evt.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        dd_factor = np.clip((abs(dd) - 15) / 30, 0, 1)
        rsi_os = np.clip((40 - rsi) / 25, 0, 1)
        oversold = max(dd_factor, rsi_os)
        if oversold > 0.1 and score >= 4.0:
            cands.append((sub_id, oversold))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

def long_top3_fv(scores_v3):
    """Top 3 por Fair Value score > 5.0 (como ALCISTA)"""
    cands = sorted([(s, sc - 5.0) for s, sc in scores_v3.items() if sc > 5.0], key=lambda x: -x[1])
    return cands[:3]

def long_top3_fv_strict(scores_v3):
    """Top 3 por Fair Value score > 5.5"""
    cands = sorted([(s, sc - 5.0) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])
    return cands[:3]

def long_top3_fv_relaxed(scores_v3):
    """Top 3 por Fair Value score > 4.5 (incluye ligeramente negativos)"""
    cands = sorted([(s, sc - 4.0) for s, sc in scores_v3.items() if sc > 4.5], key=lambda x: -x[1])
    return cands[:3]

def long_momentum(scores_v3, dd_row, rsi_row):
    """Momentum: DD > -5% + RSI > 55 + score > 5.0 (near ATH, tendencia)"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if score <= 5.0 or dd < -5 or rsi < 55: continue
        w = (score - 5.0) * 0.5 + np.clip((rsi - 55) / 25, 0, 1) * 0.3 + np.clip((5 + dd) / 5, 0, 1) * 0.2
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

def long_pullback(scores_v3, dd_row, rsi_row):
    """Pullback: score >= 5.0 + DD entre -5% y -15% + RSI 30-50"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if score < 5.0 or dd > -5 or dd < -15 or rsi > 50 or rsi < 30: continue
        w = np.clip((abs(dd) - 5) / 10, 0, 1) + np.clip((50 - rsi) / 20, 0, 1) + np.clip((score - 5) / 3, 0, 1)
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

def long_oversold_deep(scores_v3, dd_row, rsi_row):
    """Oversold profundo: DD < -15% + RSI < 35 (extremo, soporte)"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if dd > -15 or rsi > 35: continue
        if score < 3.5: continue  # fundamentales no destruidos
        w = np.clip((abs(dd) - 15) / 20, 0, 1) + np.clip((35 - rsi) / 15, 0, 1)
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

# ================================================================
# ESTRATEGIAS PATA CORTA
# ================================================================

def short_current_meanrev(scores_evt, scores_v3, dd_row, rsi_row, atr_row):
    """Actual: RSI > 65 + DD > -8% + score <= 6.0 + ATR >= 1.5%"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_evt.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        rsi_ob = np.clip((rsi - 65) / 20, 0, 1)
        if rsi_ob > 0.1 and dd > -8 and score <= 6.0 and atr >= ATR_MIN:
            cands.append((sub_id, rsi_ob))
    cands.sort(key=lambda x: -x[1])
    return cands[:2]

def short_overbought_strict(scores_v3, dd_row, rsi_row, atr_row):
    """Overbought estricto: RSI > 70 + DD > -3% + score <= 5.5 + ATR >= 1.5"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        if rsi <= 70 or dd < -3 or score > 5.5 or atr < ATR_MIN: continue
        w = np.clip((rsi - 70) / 15, 0, 1) + np.clip((5.5 - score) / 2.5, 0, 1)
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:2]

def short_weak_momentum(scores_v3, dd_row, rsi_row, atr_row):
    """Momentum debil: score < 4.0 + RSI < 45 + DD < -10% + ATR >= 1.5 (inicio ruptura)"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        if score >= 4.0 or rsi >= 45 or dd > -10 or atr < ATR_MIN: continue
        w = np.clip((4.0 - score) / 2.0, 0, 1) + np.clip((45 - rsi) / 20, 0, 1) + np.clip((abs(dd) - 10) / 15, 0, 1)
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:2]

def short_bottom3_fv(scores_v3, atr_row):
    """Bottom 3 FV: los 3 peores scores + ATR >= 1.5"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(atr): atr = 0
        if score >= 4.5 or atr < ATR_MIN: continue
        cands.append((sub_id, 5.0 - score))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

def short_none():
    """No shortear"""
    return []

def calc_side_pnl(positions, ret_row, is_short=False, capital=0.5):
    """PnL de un lado. Capital = fraccion del total (0.5 = mitad)"""
    if not positions: return 0.0
    tw = sum(w for _, w in positions)
    if tw <= 0: return 0.0
    pnl = 0.0
    for s, w in positions:
        r = ret_row.get(s)
        if pd.notna(r):
            alloc = capital * (w / tw)
            pnl += alloc * (-r if is_short else r)
    return pnl

# ================================================================
# SIMULACION
# ================================================================
print("Simulando patas LARGA y CORTA por separado en NEUTRAL...")

long_strats = {
    'L1_MeanRev': [],        # actual
    'L2_Top3FV': [],         # como ALCISTA
    'L3_Top3FV_strict': [],  # score > 5.5
    'L4_Top3FV_relaxed': [], # score > 4.5
    'L5_Momentum': [],       # near ATH + RSI alto
    'L6_Pullback': [],       # correccion en tendencia
    'L7_OversoldDeep': [],   # soporte extremo
    'L0_NoLongs': [],        # no operar longs
}

short_strats = {
    'S1_MeanRev': [],       # actual
    'S2_OB_strict': [],     # RSI>70 + DD>-3%
    'S3_WeakMom': [],       # score<4 + RSI<45
    'S4_Bottom3FV': [],     # peores FV scores
    'S0_NoShorts': [],      # no shortear
}

neutral_meta = []

for date in returns_wide.index:
    if date.year < 2001: continue

    regime, score_total = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)
    if regime != 'NEUTRAL': continue

    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    if not active: continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    if date in returns_trade_wide.index and returns_trade_wide.loc[date].notna().any():
        ret_row = returns_trade_wide.loc[date]
    else:
        ret_row = returns_wide.loc[date]

    spy_ret = spy_w.loc[spy_w.index[spy_w.index <= date][-1], 'ret_spy'] if len(spy_w.index[spy_w.index <= date]) > 0 else 0
    if not pd.notna(spy_ret): spy_ret = 0

    neutral_meta.append({'date': date, 'year': date.year, 'score_total': score_total, 'spy_ret': spy_ret})

    # --- PATA LARGA ---
    for name, func in [
        ('L1_MeanRev', lambda: long_current_meanrev(scores_evt, scores_v3, dd_row, rsi_row)),
        ('L2_Top3FV', lambda: long_top3_fv(scores_v3)),
        ('L3_Top3FV_strict', lambda: long_top3_fv_strict(scores_v3)),
        ('L4_Top3FV_relaxed', lambda: long_top3_fv_relaxed(scores_v3)),
        ('L5_Momentum', lambda: long_momentum(scores_v3, dd_row, rsi_row)),
        ('L6_Pullback', lambda: long_pullback(scores_v3, dd_row, rsi_row)),
        ('L7_OversoldDeep', lambda: long_oversold_deep(scores_v3, dd_row, rsi_row)),
        ('L0_NoLongs', lambda: []),
    ]:
        pos = func()
        pnl = calc_side_pnl(pos, ret_row, is_short=False, capital=1.0)
        cost = COST_RT if pos else 0
        long_strats[name].append({'date': date, 'pnl': pnl, 'cost': cost, 'n': len(pos),
                                   'picks': ','.join(s for s, _ in pos) if pos else ''})

    # --- PATA CORTA ---
    for name, func in [
        ('S1_MeanRev', lambda: short_current_meanrev(scores_evt, scores_v3, dd_row, rsi_row, atr_row)),
        ('S2_OB_strict', lambda: short_overbought_strict(scores_v3, dd_row, rsi_row, atr_row)),
        ('S3_WeakMom', lambda: short_weak_momentum(scores_v3, dd_row, rsi_row, atr_row)),
        ('S4_Bottom3FV', lambda: short_bottom3_fv(scores_v3, atr_row)),
        ('S0_NoShorts', lambda: []),
    ]:
        pos = func()
        pnl = calc_side_pnl(pos, ret_row, is_short=True, capital=1.0)
        cost = COST_RT if pos else 0
        short_strats[name].append({'date': date, 'pnl': pnl, 'cost': cost, 'n': len(pos),
                                    'picks': ','.join(s for s, _ in pos) if pos else ''})

# ================================================================
# RESULTADOS PATA LARGA
# ================================================================
def print_side_results(strats, side_label, capital=500_000):
    print(f"\n{'='*120}")
    print(f"  PATA {side_label} - {len(next(iter(strats.values())))} semanas NEUTRAL")
    print(f"{'='*120}")
    print(f"  {'Estrategia':<22s} {'N':>4s} {'Actv':>5s} {'Avg%':>8s} {'Med%':>8s} {'Std%':>8s} "
          f"{'WR%':>6s} {'Sharpe':>7s} {'PF':>6s} {'Total$':>12s} {'$/sem':>9s} {'AvgPos':>6s}")
    print(f"  {'-'*112}")

    for name, records in strats.items():
        df = pd.DataFrame(records)
        df['ret_net'] = df['pnl'] - df['cost']
        n_total = len(df)
        n_active = (df['n'] > 0).sum()
        if n_active == 0:
            total_pnl = 0
            print(f"  {name:<22s} {n_total:>4d} {n_active:>5d}     --- NO OPERA ---"
                  f"                               ${total_pnl:>+11,.0f} ${0:>+8,.0f}")
            continue
        active_df = df[df['n'] > 0]
        avg_ret = active_df['ret_net'].mean() * 100
        med_ret = active_df['ret_net'].median() * 100
        std_ret = active_df['ret_net'].std() * 100
        wr = (active_df['ret_net'] > 0).mean() * 100
        sharpe = avg_ret / std_ret * np.sqrt(52) if std_ret > 0 else 0
        wins = active_df[active_df['ret_net'] > 0]['ret_net'].sum()
        losses = abs(active_df[active_df['ret_net'] < 0]['ret_net'].sum())
        pf = wins / losses if losses > 0 else 999
        total_pnl = df['ret_net'].sum() * capital
        avg_pnl = total_pnl / n_total
        avg_pos = active_df['n'].mean()
        print(f"  {name:<22s} {n_total:>4d} {n_active:>5d} {avg_ret:>+8.3f} {med_ret:>+8.3f} {std_ret:>8.3f} "
              f"{wr:>5.1f}% {sharpe:>+7.2f} {pf:>6.2f} ${total_pnl:>+11,.0f} ${avg_pnl:>+8,.0f} {avg_pos:>5.1f}")

print_side_results(long_strats, "LARGA")
print_side_results(short_strats, "CORTA")

# ================================================================
# DESGLOSE POR SUB-RANGO SCORE
# ================================================================
print(f"\n{'='*120}")
print(f"  PATA LARGA POR SUB-RANGO DE SCORE NEUTRAL")
print(f"{'='*120}")

for lo, hi, label in [(0.5, 2.0, 'LOW (0.5-2.0)'), (2.0, 3.0, 'MID (2.0-3.0)'), (3.0, 4.0, 'HIGH (3.0-4.0)')]:
    idx_range = [i for i, m in enumerate(neutral_meta) if lo <= m['score_total'] < hi]
    if not idx_range: continue
    spy_avg = np.mean([neutral_meta[i]['spy_ret'] for i in idx_range]) * 100
    print(f"\n  {label} ({len(idx_range)} sem, SPY avg={spy_avg:+.2f}%):")
    print(f"  {'Estrategia':<22s} {'Actv':>5s} {'Avg%':>8s} {'WR%':>6s} {'Sharpe':>7s} {'Total$':>12s}")
    print(f"  {'-'*65}")
    for name, records in long_strats.items():
        sub = [records[i] for i in idx_range]
        df = pd.DataFrame(sub)
        df['ret_net'] = df['pnl'] - df['cost']
        n_active = (df['n'] > 0).sum()
        if n_active == 0:
            print(f"  {name:<22s} {n_active:>5d}     --- NO OPERA ---")
            continue
        active_df = df[df['n'] > 0]
        avg_ret = active_df['ret_net'].mean() * 100
        std_ret = active_df['ret_net'].std() * 100
        wr = (active_df['ret_net'] > 0).mean() * 100
        sharpe = avg_ret / std_ret * np.sqrt(52) if std_ret > 0 else 0
        total_pnl = df['ret_net'].sum() * 500_000
        print(f"  {name:<22s} {n_active:>5d} {avg_ret:>+8.3f} {wr:>5.1f}% {sharpe:>+7.2f} ${total_pnl:>+11,.0f}")

print(f"\n{'='*120}")
print(f"  PATA CORTA POR SUB-RANGO DE SCORE NEUTRAL")
print(f"{'='*120}")

for lo, hi, label in [(0.5, 2.0, 'LOW (0.5-2.0)'), (2.0, 3.0, 'MID (2.0-3.0)'), (3.0, 4.0, 'HIGH (3.0-4.0)')]:
    idx_range = [i for i, m in enumerate(neutral_meta) if lo <= m['score_total'] < hi]
    if not idx_range: continue
    print(f"\n  {label} ({len(idx_range)} sem):")
    print(f"  {'Estrategia':<22s} {'Actv':>5s} {'Avg%':>8s} {'WR%':>6s} {'Sharpe':>7s} {'Total$':>12s}")
    print(f"  {'-'*65}")
    for name, records in short_strats.items():
        sub = [records[i] for i in idx_range]
        df = pd.DataFrame(sub)
        df['ret_net'] = df['pnl'] - df['cost']
        n_active = (df['n'] > 0).sum()
        if n_active == 0:
            print(f"  {name:<22s} {n_active:>5d}     --- NO OPERA ---")
            continue
        active_df = df[df['n'] > 0]
        avg_ret = active_df['ret_net'].mean() * 100
        std_ret = active_df['ret_net'].std() * 100
        wr = (active_df['ret_net'] > 0).mean() * 100
        sharpe = avg_ret / std_ret * np.sqrt(52) if std_ret > 0 else 0
        total_pnl = df['ret_net'].sum() * 500_000
        print(f"  {name:<22s} {n_active:>5d} {avg_ret:>+8.3f} {wr:>5.1f}% {sharpe:>+7.2f} ${total_pnl:>+11,.0f}")

# ================================================================
# MEJORES COMBINACIONES L+S
# ================================================================
print(f"\n{'='*120}")
print(f"  MEJORES COMBINACIONES LARGA + CORTA (capital 50/50)")
print(f"{'='*120}")
print(f"  {'Combinacion':<40s} {'Avg%':>8s} {'WR%':>6s} {'Sharpe':>7s} {'Total$':>12s} {'$/sem':>9s}")
print(f"  {'-'*86}")

combos = []
for lname, lrecs in long_strats.items():
    for sname, srecs in short_strats.items():
        rets = []
        for i in range(len(lrecs)):
            # Capital split: 50% longs, 50% shorts (o 100% si solo 1 lado)
            l_active = lrecs[i]['n'] > 0
            s_active = srecs[i]['n'] > 0
            if l_active and s_active:
                ret = (lrecs[i]['pnl'] * 0.5 + srecs[i]['pnl'] * 0.5
                       - lrecs[i]['cost'] * 0.5 - srecs[i]['cost'] * 0.5)
            elif l_active:
                ret = lrecs[i]['pnl'] - lrecs[i]['cost']
            elif s_active:
                ret = srecs[i]['pnl'] - srecs[i]['cost']
            else:
                ret = 0
            rets.append(ret)
        arr = np.array(rets)
        n_active = np.sum(np.array([lrecs[i]['n'] + srecs[i]['n'] for i in range(len(lrecs))]) > 0)
        if n_active == 0: continue
        active_rets = arr[np.array([lrecs[i]['n'] + srecs[i]['n'] for i in range(len(lrecs))]) > 0]
        avg_ret = active_rets.mean() * 100
        std_ret = active_rets.std() * 100
        wr = (active_rets > 0).mean() * 100
        sharpe = avg_ret / std_ret * np.sqrt(52) if std_ret > 0 else 0
        total_pnl = arr.sum() * 500_000
        avg_pnl = total_pnl / len(arr)
        combos.append((f"{lname} + {sname}", avg_ret, wr, sharpe, total_pnl, avg_pnl, n_active))

combos.sort(key=lambda x: -x[3])  # sort by Sharpe
for combo, avg_ret, wr, sharpe, total_pnl, avg_pnl, n_active in combos[:15]:
    print(f"  {combo:<40s} {avg_ret:>+8.3f} {wr:>5.1f}% {sharpe:>+7.2f} ${total_pnl:>+11,.0f} ${avg_pnl:>+8,.0f}")

# ================================================================
# POR ANO: ACTUAL vs MEJOR LONG vs NO OPERAR
# ================================================================
print(f"\n{'='*120}")
print(f"  PNL POR ANO (base $500K)")
print(f"{'='*120}")
# Identificar mejores estrategias para el desglose
print(f"  {'Ano':>5s} {'N':>3s}  {'L1actual$':>11s}  {'L2Top3FV$':>11s}  {'L3strict$':>11s}  {'S1actual$':>11s}  {'S0none$':>11s}  {'SPY%':>6s}")
print(f"  {'-'*80}")

years = sorted(set(m['year'] for m in neutral_meta))
for year in years:
    idx_year = [i for i, m in enumerate(neutral_meta) if m['year'] == year]
    n_sem = len(idx_year)
    spy_avg = np.mean([neutral_meta[i]['spy_ret'] for i in idx_year]) * 100

    def year_pnl(strat_recs):
        return sum((strat_recs[i]['pnl'] - strat_recs[i]['cost']) for i in idx_year) * 500_000

    print(f"  {year:>5d} {n_sem:>3d}  ${year_pnl(long_strats['L1_MeanRev']):>+10,.0f}  "
          f"${year_pnl(long_strats['L2_Top3FV']):>+10,.0f}  "
          f"${year_pnl(long_strats['L3_Top3FV_strict']):>+10,.0f}  "
          f"${year_pnl(short_strats['S1_MeanRev']):>+10,.0f}  "
          f"${year_pnl(short_strats['S0_NoShorts']):>+10,.0f}  "
          f"{spy_avg:>+5.2f}%")
print(f"  {'-'*80}")
def total_pnl(strat_recs):
    return sum((r['pnl'] - r['cost']) for r in strat_recs) * 500_000
print(f"  {'TOTAL':>5s} {len(neutral_meta):>3d}  ${total_pnl(long_strats['L1_MeanRev']):>+10,.0f}  "
      f"${total_pnl(long_strats['L2_Top3FV']):>+10,.0f}  "
      f"${total_pnl(long_strats['L3_Top3FV_strict']):>+10,.0f}  "
      f"${total_pnl(short_strats['S1_MeanRev']):>+10,.0f}  "
      f"${total_pnl(short_strats['S0_NoShorts']):>+10,.0f}")
