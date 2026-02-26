"""Diagnostico detallado del regimen ALCISTA"""
import pandas as pd
import numpy as np
from collections import Counter
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
    return regime, {'score_total': total, 'pct_dd_healthy': pct_dd_healthy,
                    'pct_rsi_above55': pct_rsi_above55, 'pct_dd_deep': pct_dd_deep,
                    'spy_dist': spy_dist, 'spy_mom_10w': spy_mom_10w, 'vix': vix_val}

def decide_alcista_pullback(scores_v3, dd_row, rsi_row):
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if score < 5.0: continue
        if dd > -5 or dd < -15: continue
        if rsi > 50 or rsi < 30: continue
        pullback_score = 0.0
        pullback_score += np.clip((abs(dd) - 5) / 10, 0, 1) * 2.0
        pullback_score += np.clip((50 - rsi) / 20, 0, 1) * 2.0
        pullback_score += np.clip((score - 5.0) / 3.0, 0, 1) * 1.5
        candidates.append((sub_id, pullback_score))
    candidates.sort(key=lambda x: -x[1])
    longs = [c[0] for c in candidates[:3]]
    weights = {s: w for s, w in candidates[:3]}
    return longs, weights

def decide_goldilocks_breakout(scores_v3, dd_row, rsi_row):
    candidates = []
    for sub_id in SUBSECTORS:
        score = scores_v3.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if score < 5.5: continue
        if dd < -8: continue
        if rsi < 50: continue
        breakout_score = 0.0
        breakout_score += np.clip((8 + dd) / 8, 0, 1) * 2.5
        breakout_score += np.clip((rsi - 50) / 30, 0, 1) * 2.0
        breakout_score += np.clip((score - 5.5) / 3.0, 0, 1) * 1.5
        candidates.append((sub_id, breakout_score))
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
    SELECT symbol, date, close, high, low
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
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100)

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean')).reset_index()
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
returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')
atr_wide_lagged = atr_wide.shift(1)

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
print("DIAGNOSTICO ALCISTA")
print("=" * 100)

alc_weeks = []
for date in returns_wide.index:
    if date.year < 2001: continue
    regime, details = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)
    if regime != 'ALCISTA': continue

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
    ret_row = returns_wide.loc[date]
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    # Estrategia 1: Pullback (actual)
    lp, wp = decide_alcista_pullback(scores_v3, dd_row, rsi_row)
    if lp:
        pnl_pb = calc_pnl_meanrev(lp, [], wp, ret_row, 1.0)
        pb_str = ','.join(lp)
    else:
        # Fallback: top3 por score > 6.5
        pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])[:3]
        longs_fb = [s for s, _ in pool]
        pnl_pb = calc_pnl(longs_fb, [], scores_v3, ret_row, 1.0)
        pb_str = 'fallback'

    # Estrategia 2: Top3 FairValue (>5.5)
    top3 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
    longs_t3 = [s for s, _ in top3]
    pnl_t3 = calc_pnl(longs_t3, [], scores_v3, ret_row, 1.0)

    # Estrategia 3: Top3 FairValue (>6.5 - mas selectivo)
    top3_strict = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])[:3]
    longs_t3s = [s for s, _ in top3_strict]
    pnl_t3s = calc_pnl(longs_t3s, [], scores_v3, ret_row, 1.0)

    # Estrategia 4: Breakout (la que quitamos de GOLDILOCKS)
    lb, wb = decide_goldilocks_breakout(scores_v3, dd_row, rsi_row)
    if lb:
        pnl_brk = calc_pnl_meanrev(lb, [], wb, ret_row, 1.0)
    else:
        pnl_brk = pnl_t3  # fallback

    # Estrategia 5: Top5 FairValue (mas diversificado)
    top5 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:5]
    longs_t5 = [s for s, _ in top5]
    pnl_t5 = calc_pnl(longs_t5, [], scores_v3, ret_row, 1.0)

    # Contar candidatos pullback
    n_pb = len(lp) if lp else 0

    alc_weeks.append({
        'date': date, 'year': date.year, 'week': date.isocalendar()[1],
        'score_total': details.get('score_total', 0),
        'pnl_pullback': pnl_pb * 100,
        'pnl_top3': pnl_t3 * 100,
        'pnl_top3_strict': pnl_t3s * 100,
        'pnl_breakout': pnl_brk * 100,
        'pnl_top5': pnl_t5 * 100,
        'n_pullback': n_pb,
        'pb_str': pb_str,
        'longs_t3': ','.join(longs_t3),
    })

adf = pd.DataFrame(alc_weeks)

# 1. Distribucion por ano
print("\n1. DISTRIBUCION ALCISTA POR ANO")
print("-" * 110)
for year in sorted(adf['year'].unique()):
    yr = adf[adf['year'] == year]
    avg = yr['pnl_pullback'].mean()
    wr = (yr['pnl_pullback'] > 0).mean() * 100
    avg_t3 = yr['pnl_top3'].mean()
    wr_t3 = (yr['pnl_top3'] > 0).mean() * 100
    print(f"  {year}: {len(yr):>3} sem | Pullback: Avg={avg:>+6.2f}% WR={wr:>5.1f}% | "
          f"Top3FV: Avg={avg_t3:>+6.2f}% WR={wr_t3:>5.1f}% | Score={yr['score_total'].mean():.1f}")

# 2. Comparar todas las estrategias
print("\n2. COMPARAR ESTRATEGIAS EN SEMANAS ALCISTA")
print("-" * 90)
for name, col in [('Pullback (actual)', 'pnl_pullback'),
                  ('Top3 FV (>5.5)', 'pnl_top3'),
                  ('Top3 FV (>6.5)', 'pnl_top3_strict'),
                  ('Breakout', 'pnl_breakout'),
                  ('Top5 FV (>5.5)', 'pnl_top5')]:
    vals = adf[col].values / 100
    avg = vals.mean() * 100
    wr = (vals > 0).mean() * 100
    sh = vals.mean() / vals.std() * np.sqrt(52) if vals.std() > 0 else 0
    gw = vals[vals > 0].sum()
    gl = abs(vals[vals < 0].sum())
    pf = gw / gl if gl > 0 else float('inf')
    print(f"  {name:<22} {len(vals):>3} sem | Avg={avg:>+.3f}% WR={wr:>5.1f}% | "
          f"Sharpe={sh:.2f} PF={pf:.2f}")

# 3. Pullback: con candidatos vs fallback
print("\n3. PULLBACK: CON CANDIDATOS vs FALLBACK")
print("-" * 80)
has_pb = adf[adf['pb_str'] != 'fallback']
no_pb = adf[adf['pb_str'] == 'fallback']
print(f"  Con pullback:  {len(has_pb):>3} sem | Avg={has_pb['pnl_pullback'].mean():>+.3f}% "
      f"WR={(has_pb['pnl_pullback'] > 0).mean() * 100:.1f}%")
print(f"  Sin (fallback): {len(no_pb):>3} sem | Avg={no_pb['pnl_pullback'].mean():>+.3f}% "
      f"WR={(no_pb['pnl_pullback'] > 0).mean() * 100:.1f}%")

# 4. PnL por score_total
print("\n4. PNL POR SCORE TOTAL (Pullback vs Top3)")
print("-" * 90)
for lo, hi in [(4.0, 5.0), (5.0, 6.0), (6.0, 7.0)]:
    mask = (adf['score_total'] >= lo) & (adf['score_total'] < hi)
    if mask.sum() > 0:
        sub = adf[mask]
        print(f"  Score [{lo:.1f}-{hi:.1f}): {mask.sum():>3} sem | "
              f"Pullback={sub['pnl_pullback'].mean():>+.3f}% "
              f"Top3={sub['pnl_top3'].mean():>+.3f}% "
              f"Top3strict={sub['pnl_top3_strict'].mean():>+.3f}%")

# 5. Peores semanas
print("\n5. TOP 10 PEORES SEMANAS ALCISTA (pullback actual)")
print("-" * 110)
worst = adf.nsmallest(10, 'pnl_pullback')
for _, r in worst.iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d')} w{int(r['week']):>2} | "
          f"PB={r['pnl_pullback']:>+6.2f}% T3={r['pnl_top3']:>+6.2f}% "
          f"Score={r['score_total']:.1f} N_pb={r['n_pullback']} | {r['pb_str']}")

# 6. Mejores semanas
print("\n6. TOP 10 MEJORES SEMANAS ALCISTA (pullback actual)")
print("-" * 110)
best = adf.nlargest(10, 'pnl_pullback')
for _, r in best.iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d')} w{int(r['week']):>2} | "
          f"PB={r['pnl_pullback']:>+6.2f}% T3={r['pnl_top3']:>+6.2f}% "
          f"Score={r['score_total']:.1f} N_pb={r['n_pullback']} | {r['pb_str']}")

# 7. Sectores mas frecuentes
print("\n7. SECTORES MAS FRECUENTES EN LONGS Top3")
print("-" * 80)
sector_counts = Counter()
for _, r in adf.iterrows():
    for s in r['longs_t3'].split(','):
        if s: sector_counts[s] += 1
for s, c in sector_counts.most_common(15):
    print(f"  {s:<25} {c:>4} veces")

# 8. N candidatos pullback
print("\n8. N CANDIDATOS PULLBACK DISPONIBLES")
print("-" * 80)
for n in sorted(adf['n_pullback'].unique()):
    mask = adf['n_pullback'] == n
    if mask.sum() > 0:
        sub = adf[mask]
        print(f"  N={n}: {mask.sum():>3} sem | "
              f"PB Avg={sub['pnl_pullback'].mean():>+.3f}% "
              f"T3 Avg={sub['pnl_top3'].mean():>+.3f}%")

# 9. Semanas donde Top3 >> Pullback y viceversa
print("\n9. DIFERENCIA Top3 - Pullback")
print("-" * 80)
adf['diff'] = adf['pnl_top3'] - adf['pnl_pullback']
print(f"  Avg diff: {adf['diff'].mean():>+.3f}% (positivo = Top3 mejor)")
print(f"  Top3 gana: {(adf['diff'] > 0).sum()}/{len(adf)} semanas ({(adf['diff'] > 0).mean()*100:.1f}%)")
print(f"  Pullback gana: {(adf['diff'] < 0).sum()}/{len(adf)} semanas ({(adf['diff'] < 0).mean()*100:.1f}%)")
print(f"  Empate: {(adf['diff'] == 0).sum()}/{len(adf)} semanas")
