"""
Optimizar los 4 regimenes negativos: CAUTIOUS, BEARISH, CRISIS, PANICO
Probar diferentes estrategias de shorts y longs para cada uno
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
COST_RT = 0.0030  # 0.30% round-trip

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

def classify_regime(date, dd_wide, rsi_wide, spy_w, vix_df):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: return 'NEUTRAL', 0.0
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]; rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL', 0.0
    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50
    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) > 0:
        spy_last = spy_w.loc[spy_dates[-1]]
        spy_above_ma200 = spy_last.get('above_ma200', 0.5)
        spy_mom_10w = spy_last.get('mom_10w', 0); spy_dist = spy_last.get('dist_ma200', 0)
    else:
        spy_above_ma200 = 0.5; spy_mom_10w = 0; spy_dist = 0
    if not pd.notna(spy_mom_10w): spy_mom_10w = 0
    if not pd.notna(spy_dist): spy_dist = 0
    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20
    # Scoring extendido
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
    if is_burbuja: reg = 'BURBUJA'
    elif total >= 7.0: reg = 'GOLDILOCKS'
    elif total >= 4.0: reg = 'ALCISTA'
    elif total >= 0.5: reg = 'NEUTRAL'
    elif total >= -2.0: reg = 'CAUTIOUS'
    elif total >= -5.0: reg = 'BEARISH'
    elif total >= -9.0: reg = 'CRISIS'
    else: reg = 'PANICO'
    if vix_val >= 30 and reg in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'): reg = 'NEUTRAL'
    elif vix_val >= 35 and reg == 'NEUTRAL': reg = 'CAUTIOUS'
    return reg, total

# --- Estrategias ---

def s_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row):
    """Actual BEARISH: inicio ruptura, score<4.5, DD>-25, RSI>25, ATR>=1.5"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        if score >= 4.5 or dd < -25 or rsi < 25 or atr < ATR_MIN: continue
        w = np.clip((5.0 - score) / 3.0, 0, 1) * 2.0 + np.clip(abs(dd) / 20.0, 0, 1) * 1.5 + np.clip((50 - rsi) / 25.0, 0, 1) * 1.5 + np.clip((atr - ATR_MIN) / 3.0, 0, 1) * 1.0
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

def s_bottom3_fv(scores_v3, atr_row):
    """Bottom 3 FV: peores scores con ATR>=1.5"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(atr): atr = 0
        if score >= 4.5 or atr < ATR_MIN: continue
        cands.append((sub_id, 5.0 - score))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

def s_bottom3_fv_noatr(scores_v3):
    """Bottom 3 FV sin filtro ATR"""
    cands = sorted([(s, 5.0 - sc) for s, sc in scores_v3.items() if sc < 4.5], key=lambda x: -x[1])
    return cands[:3]

def s_bottom3_fv_standard(scores_v3):
    """Bottom 3 FV: score < 3.5 (como CRISIS actual)"""
    cands = sorted([(s, 5.0 - sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: -x[1])
    return cands[:3]

def s_weak_momentum(scores_v3, dd_row, rsi_row, atr_row):
    """Weak momentum: score<4.0 + RSI<45 + DD<-10% + ATR>=1.5"""
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
    return cands[:3]

def s_none():
    return []

def l_oversold_deep(scores_v3, dd_row, rsi_row):
    """Oversold profundo: DD < -15%, RSI < 35, score >= 3.5"""
    cands = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if dd > -15 or rsi > 35 or score < 3.5: continue
        w = np.clip((abs(dd) - 15) / 20, 0, 1) + np.clip((35 - rsi) / 15, 0, 1)
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:3]

def l_support(scores_v3, dd_row, rsi_row):
    """Soporte: DD -12% a -25%, RSI < 45, score >= 4.0 (CAUTIOUS actual)"""
    cands = []
    for sub_id in SUBSECTORS:
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        score = scores_v3.get(sub_id, 5.0)
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if dd > -12 or dd < -25 or rsi > 45 or rsi < 20 or score < 4.0: continue
        w = np.clip((abs(dd) - 12) / 8, 0, 1) * 2.0 + np.clip((45 - rsi) / 20, 0, 1) * 1.5 + np.clip((score - 4.0) / 2.0, 0, 1) * 1.0
        cands.append((sub_id, w))
    cands.sort(key=lambda x: -x[1])
    return cands[:2]

def l_none():
    return []

def calc_side_pnl(positions, ret_row, is_short=False):
    if not positions: return 0.0
    tw = sum(w for _, w in positions)
    if tw <= 0: return 0.0
    pnl = 0.0
    for s, w in positions:
        r = ret_row.get(s)
        if pd.notna(r):
            pnl += (w / tw) * (-r if is_short else r)
    return pnl

# ================================================================
# CARGAR DATOS
# ================================================================
print("Cargando datos...")
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']: ticker_to_sub[t] = sub_id
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
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(lambda x: x.rolling(5, min_periods=3).mean() * 100)

df_monday = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).first().reset_index()
df_monday = df_monday.sort_values(['symbol', 'date'])
df_monday['prev_open'] = df_monday.groupby('symbol')['open'].shift(1)
df_monday['return_mon'] = df_monday['open'] / df_monday['prev_open'] - 1
df_monday = df_monday.dropna(subset=['return_mon'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'), avg_low=('low', 'mean'),
    avg_return=('return', 'mean'), avg_atr=('atr_pct', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])
sub_monday = df_monday.groupby(['subsector', 'date']).agg(avg_return_mon=('return_mon', 'mean')).reset_index()
sub_monday = sub_monday.sort_values(['subsector', 'date'])
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_price_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean(); avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)
returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide_lagged = atr_wide.shift(1)

returns_mon_wide = sub_monday.pivot(index='date', columns='subsector', values='avg_return_mon')
mon_dates = returns_mon_wide.index.tolist(); fri_dates = returns_wide.index.tolist()
fri_to_mon_ret = {}
for fri in fri_dates:
    target = fri + pd.Timedelta(days=3)
    diffs = [abs((d - target).days) for d in mon_dates]
    if diffs:
        cm = mon_dates[diffs.index(min(diffs))]
        if abs((cm - target).days) <= 3: fri_to_mon_ret[fri] = cm
returns_trade_wide = pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns, dtype=float)
for fri, mon in fri_to_mon_ret.items():
    if mon in returns_mon_wide.index: returns_trade_wide.loc[fri] = returns_mon_wide.loc[mon]

spy_daily = pd.read_sql("SELECT date, close FROM fmp_price_history WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21' ORDER BY date", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date']); spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100; spy_w['ret_spy'] = spy_w['close'].pct_change()

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
    skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date'); vix_df = vix_df.rename(columns={'close': 'vix'})
weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# ================================================================
# SIMULACION POR REGIMEN
# ================================================================
print("Simulando estrategias por regimen negativo...")

target_regimes = ['CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO']

# Estrategias a probar (nombre, funcion_longs, funcion_shorts)
strat_configs = [
    ('S_BearAggr',      None, 'bear_aggr'),
    ('S_Bot3FV',        None, 'bot3_fv'),
    ('S_Bot3FV_noATR',  None, 'bot3_noatr'),
    ('S_Bot3FV_std',    None, 'bot3_std'),
    ('S_WeakMom',       None, 'weak_mom'),
    ('S_None',          None, 'none'),
    ('L_Oversold',      'oversold', None),
    ('L_Support',       'support', None),
    ('L_None',          'none', None),
    ('LS_BearAggr+Ovs', 'oversold', 'bear_aggr'),
    ('LS_Bot3+Ovs',     'oversold', 'bot3_std'),
]

results_by_regime = {reg: {s[0]: [] for s in strat_configs} for reg in target_regimes}

for date in returns_wide.index:
    if date.year < 2001: continue
    regime, score_total = classify_regime(date, dd_wide, rsi_wide, spy_w, vix_df)
    if regime not in target_regimes: continue

    if date in weekly_events.index: evt_date = date
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

    for sname, long_type, short_type in strat_configs:
        # Longs
        if long_type == 'oversold': lpos = l_oversold_deep(scores_v3, dd_row, rsi_row)
        elif long_type == 'support': lpos = l_support(scores_v3, dd_row, rsi_row)
        elif long_type == 'none' or long_type is None: lpos = []
        else: lpos = []

        # Shorts
        if short_type == 'bear_aggr': spos = s_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row)
        elif short_type == 'bot3_fv': spos = s_bottom3_fv(scores_v3, atr_row)
        elif short_type == 'bot3_noatr': spos = s_bottom3_fv_noatr(scores_v3)
        elif short_type == 'bot3_std': spos = s_bottom3_fv_standard(scores_v3)
        elif short_type == 'weak_mom': spos = s_weak_momentum(scores_v3, dd_row, rsi_row, atr_row)
        elif short_type == 'none' or short_type is None: spos = []
        else: spos = []

        n = len(lpos) + len(spos)
        if n > 0 and lpos and spos:
            pnl = calc_side_pnl(lpos, ret_row, False) * 0.5 + calc_side_pnl(spos, ret_row, True) * 0.5
        elif lpos:
            pnl = calc_side_pnl(lpos, ret_row, False)
        elif spos:
            pnl = calc_side_pnl(spos, ret_row, True)
        else:
            pnl = 0.0
        cost = COST_RT if n > 0 else 0

        results_by_regime[regime][sname].append({
            'date': date, 'pnl': pnl, 'cost': cost, 'n': n, 'nl': len(lpos), 'ns': len(spos)
        })

# ================================================================
# RESULTADOS
# ================================================================
for regime in target_regimes:
    strats = results_by_regime[regime]
    n_sem = len(next(v for v in strats.values() if v))
    print(f"\n{'='*120}")
    print(f"  {regime} ({n_sem} semanas)")
    print(f"{'='*120}")
    print(f"  {'Estrategia':<22s} {'Actv':>5s} {'Avg%':>8s} {'Med%':>8s} {'Std%':>8s} {'WR%':>6s} {'Sharpe':>7s} {'PF':>6s} {'Total$':>12s} {'$/sem':>9s}")
    print(f"  {'-'*95}")

    for sname in [s[0] for s in strat_configs]:
        recs = strats[sname]
        if not recs: continue
        df = pd.DataFrame(recs)
        df['ret_net'] = df['pnl'] - df['cost']
        n_active = (df['n'] > 0).sum()
        if n_active == 0:
            total_pnl = 0
            print(f"  {sname:<22s} {n_active:>5d}     --- NO OPERA ---                              ${total_pnl:>+11,.0f} ${0:>+8,.0f}")
            continue
        active = df[df['n'] > 0]
        avg = active['ret_net'].mean() * 100
        med = active['ret_net'].median() * 100
        std = active['ret_net'].std() * 100
        wr = (active['ret_net'] > 0).mean() * 100
        sharpe = avg / std * np.sqrt(52) if std > 0 else 0
        wins = active[active['ret_net'] > 0]['ret_net'].sum()
        losses = abs(active[active['ret_net'] < 0]['ret_net'].sum())
        pf = wins / losses if losses > 0 else 999
        total_pnl = df['ret_net'].sum() * 500_000
        avg_pnl = total_pnl / len(df)
        print(f"  {sname:<22s} {n_active:>5d} {avg:>+8.3f} {med:>+8.3f} {std:>8.3f} {wr:>5.1f}% {sharpe:>+7.2f} {pf:>6.2f} ${total_pnl:>+11,.0f} ${avg_pnl:>+8,.0f}")

# ================================================================
# RESUMEN: MEJOR ESTRATEGIA POR REGIMEN
# ================================================================
print(f"\n{'='*120}")
print(f"  RESUMEN: MEJOR ESTRATEGIA POR REGIMEN (por Sharpe)")
print(f"{'='*120}")
print(f"  {'Regimen':<12s} {'N':>4s} {'Mejor estrategia':<22s} {'Avg%':>8s} {'WR%':>6s} {'Sharpe':>7s} {'Total$':>12s}")
print(f"  {'-'*75}")

for regime in target_regimes:
    strats = results_by_regime[regime]
    best_name = None; best_sharpe = -999
    for sname in [s[0] for s in strat_configs]:
        recs = strats[sname]
        if not recs: continue
        df = pd.DataFrame(recs); df['ret_net'] = df['pnl'] - df['cost']
        active = df[df['n'] > 0]
        if len(active) == 0: continue
        avg = active['ret_net'].mean() * 100
        std = active['ret_net'].std() * 100
        sharpe = avg / std * np.sqrt(52) if std > 0 else 0
        if sharpe > best_sharpe:
            best_sharpe = sharpe; best_name = sname
            best_avg = avg; best_wr = (active['ret_net'] > 0).mean() * 100
            best_total = df['ret_net'].sum() * 500_000
    n_sem = len(next(v for v in strats.values() if v))
    if best_name:
        print(f"  {regime:<12s} {n_sem:>4d} {best_name:<22s} {best_avg:>+8.3f} {best_wr:>5.1f}% {best_sharpe:>+7.2f} ${best_total:>+11,.0f}")
