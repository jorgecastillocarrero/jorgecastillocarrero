"""Comparar retornos: Monday open->open vs Monday close->close"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0
CAPITAL_INICIAL = 500_000
COST_PER_TRADE = 0.0010
SLIPPAGE_PER_SIDE = 0.0005

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
    if len(prev_dates) == 0: return 'NEUTRAL', {}
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL', {}
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
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'): regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL': regime = 'CAUTIOUS'
    return regime, {'score_total': total}

ATR_MIN = 1.5

def decide_burbuja_aggressive(scores_v3, dd_row, rsi_row):
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if score <= 6.0: continue
        if dd < -8: continue
        if rsi < 55: continue
        momentum_score = np.clip((score - 6.0) / 2.5, 0, 1) * 2.5 + np.clip((8 + dd) / 8, 0, 1) * 2.0 + np.clip((rsi - 55) / 25, 0, 1) * 1.5
        candidates.append((sub_id, momentum_score))
    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def decide_cautious_support(scores_v3, dd_row, rsi_row):
    candidates = []
    for sub_id in SUBSECTORS:
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        score = scores_v3.get(sub_id, 5.0)
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if dd > -12 or dd < -25: continue
        if rsi > 45 or rsi < 20: continue
        if score < 4.0: continue
        support_score = np.clip((abs(dd) - 12) / 8, 0, 1) * 2.0 + np.clip((45 - rsi) / 20, 0, 1) * 1.5 + np.clip((score - 4.0) / 2.0, 0, 1) * 1.0
        candidates.append((sub_id, support_score))
    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:2]]
    weights = {s: w for s, w in candidates[:2]}
    return longs, weights

def decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row):
    long_cands, short_cands = [], []
    for sub_id in SUBSECTORS:
        score = scores_evt.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        dd_factor = np.clip((abs(dd) - 15) / 30, 0, 1)
        rsi_os = np.clip((40 - rsi) / 25, 0, 1)
        oversold = max(dd_factor, rsi_os)
        if oversold > 0.1 and score >= 4.0: long_cands.append((sub_id, oversold))
        rsi_ob = np.clip((rsi - 65) / 20, 0, 1)
        if rsi_ob > 0.1 and dd > -8 and score <= 6.0 and atr >= ATR_MIN: short_cands.append((sub_id, rsi_ob))
    long_cands.sort(key=lambda x: -x[1])
    short_cands.sort(key=lambda x: -x[1])
    longs = [c[0] for c in long_cands[:3]]
    shorts = [c[0] for c in short_cands[:2]]
    weights = {s: w for s, w in long_cands[:3]}
    weights.update({s: w for s, w in short_cands[:2]})
    return longs, shorts, weights

def decide_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row):
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0
        if score >= 4.5: continue
        if dd < -25: continue
        if rsi < 25: continue
        if atr < ATR_MIN: continue
        breakdown_score = np.clip((5.0 - score) / 3.0, 0, 1) * 2.0 + np.clip(abs(dd) / 20.0, 0, 1) * 1.5 + np.clip((50 - rsi) / 25.0, 0, 1) * 1.5 + np.clip((atr - ATR_MIN) / 3.0, 0, 1) * 1.0
        candidates.append((sub_id, breakdown_score))
    candidates.sort(key=lambda x: -x[1])
    shorts = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return shorts, weights

def calc_pnl(longs, shorts, scores, ret_row, capital):
    n = len(longs) + len(shorts)
    if n == 0: return 0.0
    lw = {s: scores[s] - 5.0 for s in longs}
    sw = {s: 5.0 - scores[s] for s in shorts}
    tw = sum(lw.values()) + sum(sw.values())
    if tw <= 0: return 0.0
    pnl = 0.0
    for s in longs:
        if pd.notna(ret_row.get(s)): pnl += capital * (lw[s] / tw) * ret_row[s]
    for s in shorts:
        if pd.notna(ret_row.get(s)): pnl += capital * (sw[s] / tw) * (-ret_row[s])
    return pnl

def calc_pnl_meanrev(longs, shorts, weights, ret_row, capital):
    n = len(longs) + len(shorts)
    if n == 0: return 0.0
    lw = {s: weights.get(s, 0) for s in longs}
    sw = {s: weights.get(s, 0) for s in shorts}
    tw = sum(lw.values()) + sum(sw.values())
    if tw <= 0: return 0.0
    pnl = 0.0
    for s in longs:
        if pd.notna(ret_row.get(s)): pnl += capital * (lw[s] / tw) * ret_row[s]
    for s in shorts:
        if pd.notna(ret_row.get(s)): pnl += capital * (sw[s] / tw) * (-ret_row[s])
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
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year

# Friday data (signals)
df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100)

# Monday data (first day of week)
df_monday = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).first().reset_index()
df_monday = df_monday.sort_values(['symbol', 'date'])

# OPEN returns: open(W+1) / open(W) - 1
df_monday['prev_open'] = df_monday.groupby('symbol')['open'].shift(1)
df_monday['return_mon_open'] = df_monday['open'] / df_monday['prev_open'] - 1

# CLOSE returns: close(monday W+1) / close(monday W) - 1
df_monday['prev_close_mon'] = df_monday.groupby('symbol')['close'].shift(1)
df_monday['return_mon_close'] = df_monday['close'] / df_monday['prev_close_mon'] - 1

df_monday = df_monday.dropna(subset=['return_mon_open', 'return_mon_close'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

# Monday subsector returns - both open and close
sub_monday_open = df_monday.groupby(['subsector', 'date']).agg(
    avg_return=('return_mon_open', 'mean')).reset_index()
sub_monday_close = df_monday.groupby(['subsector', 'date']).agg(
    avg_return=('return_mon_close', 'mean')).reset_index()

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

# Map Friday dates to Monday returns for both open and close
def map_fri_to_mon(sub_mon_df, col_name):
    ret_wide = sub_mon_df.pivot(index='date', columns='subsector', values='avg_return')
    mon_dates = ret_wide.index.tolist()
    fri_dates = returns_wide.index.tolist()
    fri_to_mon = {}
    for fri in fri_dates:
        target = fri + pd.Timedelta(days=3)
        diffs = [abs((d - target).days) for d in mon_dates]
        if diffs:
            closest = mon_dates[diffs.index(min(diffs))]
            if abs((closest - target).days) <= 3:
                fri_to_mon[fri] = closest
    result = pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns, dtype=float)
    for fri, mon in fri_to_mon.items():
        if mon in ret_wide.index:
            result.loc[fri] = ret_wide.loc[mon]
    return result

returns_open_wide = map_fri_to_mon(sub_monday_open, 'open')
returns_close_wide = map_fri_to_mon(sub_monday_close, 'close')

print(f"Open mapped: {returns_open_wide.notna().any(axis=1).sum()} weeks")
print(f"Close mapped: {returns_close_wide.notna().any(axis=1).sum()} weeks")

# SPY and VIX
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
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
# RUN SYSTEM WITH BOTH RETURN TYPES
# ================================================================
def run_system(ret_trade_wide, label):
    weekly_returns = []
    for date in returns_wide.index:
        if date.year < 2001: continue
        if date in weekly_events.index: evt_date = date
        else:
            nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
            evt_date = weekly_events.index[nearest_idx]
        events_row = weekly_events.loc[evt_date]
        active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
        if not active:
            weekly_returns.append({'date': date, 'ret_net': 0, 'regime': 'NEUTRAL'})
            continue
        scores_evt = score_fair(active)
        prev_dates = dd_wide.index[dd_wide.index < date]
        dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
        rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
        scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
        atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

        if date in ret_trade_wide.index and ret_trade_wide.loc[date].notna().any():
            ret_row = ret_trade_wide.loc[date]
        else:
            ret_row = returns_wide.loc[date]

        regime, _ = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)
        longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
        shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])

        if regime == 'CRISIS': nl, ns = 0, 3
        elif regime == 'BEARISH': nl, ns = 0, 3
        elif regime == 'CAUTIOUS': nl, ns = 0, 0
        elif regime == 'NEUTRAL': nl, ns = 3, 3
        elif regime == 'ALCISTA': nl, ns = 3, 0
        elif regime == 'GOLDILOCKS': nl, ns = 3, 0
        elif regime == 'BURBUJA': nl, ns = 3, 0
        else: nl, ns = 3, 0

        longs = [s for s, _ in longs_pool[:nl]]
        shorts = [s for s, _ in shorts_pool[:ns]]
        if atr_row is not None:
            shorts = [s for s in shorts if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

        if regime == 'BURBUJA':
            longs_bub, weights_bub = decide_burbuja_aggressive(scores_v3, dd_row, rsi_row)
            if longs_bub:
                longs = longs_bub; shorts = []
                pnl_unit = calc_pnl_meanrev(longs_bub, [], weights_bub, ret_row, 1.0)
            else:
                pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)
        elif regime == 'GOLDILOCKS':
            top3 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
            longs = [s for s, _ in top3]; shorts = []
            pnl_unit = calc_pnl(longs, [], scores_v3, ret_row, 1.0)
        elif regime == 'ALCISTA':
            top3 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
            longs = [s for s, _ in top3]; shorts = []
            pnl_unit = calc_pnl(longs, [], scores_v3, ret_row, 1.0)
        elif regime == 'CAUTIOUS':
            longs_sup, weights_sup = decide_cautious_support(scores_v3, dd_row, rsi_row)
            if longs_sup:
                longs = longs_sup; shorts = []
                pnl_unit = calc_pnl_meanrev(longs_sup, [], weights_sup, ret_row, 1.0)
            else:
                longs = []; shorts = []; pnl_unit = 0.0
        elif regime == 'NEUTRAL':
            longs, shorts, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
            pnl_unit = calc_pnl_meanrev(longs, shorts, weights_mr, ret_row, 1.0)
        elif regime == 'BEARISH':
            shorts_bear, weights_bear = decide_bear_aggressive(scores_v3, dd_row, rsi_row, atr_row)
            if shorts_bear:
                longs = []; shorts = shorts_bear
                pnl_unit = calc_pnl_meanrev([], shorts_bear, weights_bear, ret_row, 1.0)
            else:
                pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)
        else:
            pnl_unit = calc_pnl(longs, shorts, scores_v3, ret_row, 1.0)

        n_pos = len(longs) + len(shorts)
        cost_pct = (COST_PER_TRADE + SLIPPAGE_PER_SIDE) * 2 if n_pos > 0 else 0
        weekly_returns.append({'date': date, 'ret_net': pnl_unit - cost_pct, 'regime': regime})

    df_ret = pd.DataFrame(weekly_returns)

    # Compound
    capital = CAPITAL_INICIAL
    for year in sorted(df_ret['date'].dt.year.unique()):
        yr = df_ret[df_ret['date'].dt.year == year]
        pnl = sum(capital * row['ret_net'] for _, row in yr.iterrows())
        capital += pnl

    rets = df_ret['ret_net'].values
    n_years = df_ret['date'].dt.year.nunique()
    cagr = (capital / CAPITAL_INICIAL) ** (1 / n_years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(52) if rets.std() > 0 else 0
    downside = rets[rets < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-9
    sortino = rets.mean() / downside_std * np.sqrt(52)
    equity = CAPITAL_INICIAL * np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak).min() * 100
    wr = (rets > 0).mean() * 100
    avg_ret = rets.mean() * 100

    # Per-regime stats
    regime_stats = {}
    for reg in df_ret['regime'].unique():
        sub = df_ret[df_ret['regime'] == reg]['ret_net'].values
        if len(sub) > 0:
            regime_stats[reg] = {
                'n': len(sub),
                'avg': sub.mean() * 100,
                'wr': (sub > 0).mean() * 100,
                'sharpe': sub.mean() / sub.std() * np.sqrt(52) if sub.std() > 0 else 0
            }

    return {
        'label': label, 'capital_final': capital, 'cagr': cagr * 100,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'wr': wr, 'avg_ret': avg_ret, 'regime_stats': regime_stats,
        'n_years': n_years, 'multiple': capital / CAPITAL_INICIAL
    }

# Run both
print("\nEjecutando sistema con Monday OPEN -> OPEN...")
res_open = run_system(returns_open_wide, "Monday OPEN -> OPEN")
print("Ejecutando sistema con Monday CLOSE -> CLOSE...")
res_close = run_system(returns_close_wide, "Monday CLOSE -> CLOSE")

# ================================================================
# COMPARAR
# ================================================================
print("\n" + "=" * 100)
print("COMPARACION: MONDAY OPEN vs MONDAY CLOSE")
print("=" * 100)

print(f"\n{'Metrica':<25} {'OPEN->OPEN':>15} {'CLOSE->CLOSE':>15} {'Diferencia':>15}")
print("-" * 75)
for name, key, fmt in [
    ('Capital Final', 'capital_final', '${:>13,.0f}'),
    ('Multiplicador', 'multiple', '{:>13.1f}x'),
    ('CAGR', 'cagr', '{:>12.1f}%'),
    ('Sharpe', 'sharpe', '{:>13.2f}'),
    ('Sortino', 'sortino', '{:>13.2f}'),
    ('Max Drawdown', 'max_dd', '{:>12.1f}%'),
    ('Win Rate', 'wr', '{:>12.1f}%'),
    ('Avg Ret/sem', 'avg_ret', '{:>+12.3f}%'),
]:
    vo = res_open[key]
    vc = res_close[key]
    diff = vc - vo
    if 'capital' in key.lower():
        print(f"  {name:<23} ${vo:>13,.0f} ${vc:>13,.0f} ${diff:>+13,.0f}")
    elif 'multiple' in key.lower():
        print(f"  {name:<23} {vo:>13.1f}x {vc:>13.1f}x {diff:>+13.1f}x")
    elif '%' in fmt:
        print(f"  {name:<23} {vo:>12.1f}% {vc:>12.1f}% {diff:>+12.1f}%")
    else:
        print(f"  {name:<23} {vo:>13.2f} {vc:>13.2f} {diff:>+13.2f}")

# Regime comparison
print(f"\n{'DETALLE POR REGIMEN':>35}")
print("-" * 100)
print(f"  {'Regimen':<16} {'N':>5} | {'Avg OPEN%':>10} {'WR O%':>7} {'Sh O':>7} | {'Avg CLOSE%':>10} {'WR C%':>7} {'Sh C':>7}")
print("-" * 100)
for reg in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'CRISIS']:
    so = res_open['regime_stats'].get(reg)
    sc = res_close['regime_stats'].get(reg)
    if not so and not sc: continue
    n = so['n'] if so else (sc['n'] if sc else 0)
    ao = so['avg'] if so else 0
    wo = so['wr'] if so else 0
    sho = so['sharpe'] if so else 0
    ac = sc['avg'] if sc else 0
    wc = sc['wr'] if sc else 0
    shc = sc['sharpe'] if sc else 0
    print(f"  {reg:<16} {n:>5} | {ao:>+9.3f}% {wo:>6.1f} {sho:>+6.2f} | {ac:>+9.3f}% {wc:>6.1f} {shc:>+6.2f}")

winner = "OPEN" if res_open['sharpe'] > res_close['sharpe'] else "CLOSE"
print(f"\n  >>> GANADOR: Monday {winner} (mayor Sharpe)")
print(f"  >>> Open Sharpe={res_open['sharpe']:.2f} CAGR={res_open['cagr']:.1f}%")
print(f"  >>> Close Sharpe={res_close['sharpe']:.2f} CAGR={res_close['cagr']:.1f}%")
