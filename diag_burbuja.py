"""Diagnostico detallado del regimen BURBUJA"""
import pandas as pd
import numpy as np
from collections import Counter
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0

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
    return total, is_burbuja, {'pct_dd_healthy': pct_dd_healthy, 'pct_rsi_above55': pct_rsi_above55,
                                'pct_dd_deep': pct_dd_deep, 'spy_dist': spy_dist,
                                'spy_mom_10w': spy_mom_10w, 'vix': vix_val,
                                'score_bdd': score_bdd, 'score_brsi': score_brsi,
                                'score_ddp': score_ddp, 'score_spy': score_spy, 'score_mom': score_mom}

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
        momentum_score = 0.0
        momentum_score += np.clip((score - 6.0) / 2.5, 0, 1) * 2.5
        momentum_score += np.clip((8 + dd) / 8, 0, 1) * 2.0
        momentum_score += np.clip((rsi - 55) / 25, 0, 1) * 1.5
        candidates.append((sub_id, momentum_score))
    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def calc_pnl(longs, shorts, scores, ret_row, capital):
    n = len(longs) + len(shorts)
    if n == 0: return 0.0
    lw = {s: scores[s] - 5.0 for s in longs}
    sw = {s: 5.0 - scores[s] for s in shorts}
    tw = sum(lw.values()) + sum(sw.values())
    if tw <= 0: return 0.0
    pnl = 0.0
    for s in longs:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (lw[s] / tw) * ret_row[s]
    for s in shorts:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (sw[s] / tw) * (-ret_row[s])
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
        if pd.notna(ret_row.get(s)):
            pnl += capital * (lw[s] / tw) * ret_row[s]
    for s in shorts:
        if pd.notna(ret_row.get(s)):
            pnl += capital * (sw[s] / tw) * (-ret_row[s])
    return pnl

# --- Cargar datos ---
print("Cargando datos...")
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
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

df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])

# Monday returns
df_monday = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).first().reset_index()
df_monday = df_monday.sort_values(['symbol', 'date'])
df_monday['prev_open'] = df_monday.groupby('symbol')['open'].shift(1)
df_monday['return_mon'] = df_monday['open'] / df_monday['prev_open'] - 1
df_monday = df_monday.dropna(subset=['return_mon'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

sub_monday = df_monday.groupby(['subsector', 'date']).agg(
    avg_return_mon=('return_mon', 'mean')).reset_index()

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
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

# Monday returns aligned to Friday
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

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# ================================================================
print("\n" + "=" * 100)
print("DIAGNOSTICO BURBUJA")
print("=" * 100)

bub_weeks = []
high_score_weeks = []  # semanas con score >= 7.5

for date in returns_wide.index:
    if date.year < 2001: continue
    total, is_burbuja, details = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    if date in weekly_events.index: evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    scores_evt = score_fair(active) if active else {s: 5.0 for s in SUBSECTORS}
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    # Monday return
    if date in returns_trade_wide.index and returns_trade_wide.loc[date].notna().any():
        ret_row = returns_trade_wide.loc[date]
    else:
        ret_row = returns_wide.loc[date]

    # Estrategia 1: Burbuja aggressive (actual)
    lb, wb = decide_burbuja_aggressive(scores_v3, dd_row, rsi_row)
    if lb:
        pnl_bub = calc_pnl_meanrev(lb, [], wb, ret_row, 1.0)
    else:
        top3 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])[:3]
        pnl_bub = calc_pnl([s for s, _ in top3], [], scores_v3, ret_row, 1.0)

    # Estrategia 2: Top3 FairValue (>5.5)
    top3_fv = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
    pnl_t3 = calc_pnl([s for s, _ in top3_fv], [], scores_v3, ret_row, 1.0)

    # Estrategia 3: Top5 FairValue
    top5_fv = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:5]
    pnl_t5 = calc_pnl([s for s, _ in top5_fv], [], scores_v3, ret_row, 1.0)

    # Determinar regimen actual
    vix_val = details.get('vix', 20)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    else: regime = 'OTHER'
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'): regime = 'NEUTRAL_VIX'

    if total >= 7.5:  # high score weeks (BURBUJA + top GOLDILOCKS)
        high_score_weeks.append({
            'date': date, 'year': date.year, 'regime': regime,
            'score_total': total, 'is_burbuja': is_burbuja,
            'pnl_bub': pnl_bub * 100, 'pnl_t3': pnl_t3 * 100, 'pnl_t5': pnl_t5 * 100,
            'dd_healthy': details['pct_dd_healthy'], 'rsi55': details['pct_rsi_above55'],
            'vix': details['vix'], 'spy_dist': details['spy_dist'],
        })

    if is_burbuja:
        bub_weeks.append({
            'date': date, 'year': date.year,
            'pnl_bub': pnl_bub * 100, 'pnl_t3': pnl_t3 * 100, 'pnl_t5': pnl_t5 * 100,
            'score_total': total,
            'dd_healthy': details['pct_dd_healthy'], 'rsi55': details['pct_rsi_above55'],
            'spy_dist': details['spy_dist'], 'spy_mom': details['spy_mom_10w'],
            'vix': details['vix'],
            'longs_bub': ','.join(lb) if lb else 'none',
            'longs_t3': ','.join([s for s, _ in top3_fv]),
        })

bdf = pd.DataFrame(bub_weeks)
hsdf = pd.DataFrame(high_score_weeks)

# 1. Las 9 semanas BURBUJA
print("\n1. TODAS LAS SEMANAS BURBUJA (detalle)")
print("-" * 130)
for _, r in bdf.iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d')} | Score={r['score_total']:.1f} "
          f"DD_H={r['dd_healthy']:.0f}% RSI55={r['rsi55']:.0f}% SPY_dist={r['spy_dist']:.1f}% VIX={r['vix']:.1f}")
    print(f"    Burbuja: {r['pnl_bub']:>+6.2f}%  Top3FV: {r['pnl_t3']:>+6.2f}%  Top5FV: {r['pnl_t5']:>+6.2f}%")
    print(f"    Longs bub: {r['longs_bub']}")
    print(f"    Longs t3:  {r['longs_t3']}")

# 2. Comparar estrategias
print("\n2. COMPARAR ESTRATEGIAS EN BURBUJA (9 semanas)")
print("-" * 80)
for name, col in [('Burbuja Aggressive', 'pnl_bub'), ('Top3 FV', 'pnl_t3'), ('Top5 FV', 'pnl_t5')]:
    vals = bdf[col].values / 100
    avg = vals.mean() * 100
    wr = (vals > 0).mean() * 100
    print(f"  {name:<22} Avg={avg:>+.3f}% WR={wr:.1f}%")

# 3. Semanas con score >= 7.5 (BURBUJA + GOLDILOCKS top)
print(f"\n3. SEMANAS CON SCORE >= 7.5 ({len(hsdf)} total)")
print("-" * 110)
print(f"  {'Regime':<12} {'N':>4}  {'Avg Bub%':>9} {'Avg T3%':>9} {'Avg T5%':>9}")
for reg in hsdf['regime'].unique():
    sub = hsdf[hsdf['regime'] == reg]
    print(f"  {reg:<12} {len(sub):>4}  {sub['pnl_bub'].mean():>+8.3f} {sub['pnl_t3'].mean():>+8.3f} {sub['pnl_t5'].mean():>+8.3f}")
print(f"\n  Total:       {len(hsdf):>4}  {hsdf['pnl_bub'].mean():>+8.3f} {hsdf['pnl_t3'].mean():>+8.3f} {hsdf['pnl_t5'].mean():>+8.3f}")

# 4. Por ano
print(f"\n4. SEMANAS SCORE >= 7.5 POR ANO")
print("-" * 80)
for year in sorted(hsdf['year'].unique()):
    yr = hsdf[hsdf['year'] == year]
    print(f"  {year}: {len(yr):>2} sem | regimenes: {dict(yr['regime'].value_counts())}")

# 5. Que pasa si eliminamos BURBUJA y lo tratamos como GOLDILOCKS?
print(f"\n5. SIMULACION: FUSIONAR BURBUJA -> GOLDILOCKS (Top3 FV)")
print("-" * 80)
# Solo las 9 semanas burbuja con estrategia Top3 FV
vals_bub = bdf['pnl_bub'].values / 100
vals_t3 = bdf['pnl_t3'].values / 100
print(f"  Como BURBUJA aggressive: Avg={vals_bub.mean()*100:>+.3f}% WR={(vals_bub>0).mean()*100:.1f}%")
print(f"  Como GOLDILOCKS (Top3):  Avg={vals_t3.mean()*100:>+.3f}% WR={(vals_t3>0).mean()*100:.1f}%")
diff = (vals_t3.mean() - vals_bub.mean()) * 100
print(f"  Diferencia: {diff:>+.3f}% por semana ({'+' if diff > 0 else ''}mejor con {'Top3' if diff > 0 else 'Burbuja'})")

# 6. Criterios BURBUJA: relajar para tener mas muestra?
print(f"\n6. CRITERIOS ALTERNATIVOS PARA BURBUJA")
print("-" * 90)
for label, crit in [
    ("Actual (>=8.0, DD_H>=85, RSI55>=90)", lambda r: r['score_total'] >= 8.0 and r['dd_healthy'] >= 85 and r['rsi55'] >= 90),
    ("Score>=8.0 (sin DD_H/RSI55)", lambda r: r['score_total'] >= 8.0),
    ("Score>=7.5 + DD_H>=80", lambda r: r['score_total'] >= 7.5 and r['dd_healthy'] >= 80),
    ("Score>=7.5 + DD_H>=75 + RSI55>=75", lambda r: r['score_total'] >= 7.5 and r['dd_healthy'] >= 75 and r['rsi55'] >= 75),
]:
    mask = hsdf.apply(crit, axis=1)
    if mask.sum() > 0:
        sub = hsdf[mask]
        avg_t3 = sub['pnl_t3'].mean()
        wr_t3 = (sub['pnl_t3'] > 0).mean() * 100
        print(f"  {label:<50} N={mask.sum():>3} | T3 Avg={avg_t3:>+.3f}% WR={wr_t3:.1f}%")
