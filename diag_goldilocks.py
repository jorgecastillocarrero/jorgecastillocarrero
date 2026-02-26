"""Diagnostico detallado del regimen GOLDILOCKS"""
import pandas as pd
import numpy as np
from collections import Counter
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events
from report_compound import (score_fair, adjust_score_by_price, classify_regime_market,
                              decide_goldilocks_breakout, calc_pnl_meanrev, calc_pnl)

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

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
# ANALISIS GOLDILOCKS
# ================================================================
print("\n" + "=" * 100)
print("DIAGNOSTICO GOLDILOCKS")
print("=" * 100)

gold_weeks = []
all_regime_data = []

for date in returns_wide.index:
    if date.year < 2001:
        continue

    regime, details = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    if date in weekly_events.index:
        evt_date = date
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

    # Contar candidatos breakout
    n_breakout = 0
    if dd_row is not None and rsi_row is not None:
        for sub_id in SUBSECTORS:
            score = scores_v3.get(sub_id, 5.0)
            dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
            rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
            if not pd.notna(dd): dd = 0
            if not pd.notna(rsi): rsi = 50
            if score >= 5.5 and dd > -8 and rsi > 50:
                n_breakout += 1

    all_regime_data.append({
        'date': date, 'regime': regime,
        'score_total': details.get('score_total', 0),
        'n_breakout': n_breakout
    })

    if regime == 'GOLDILOCKS':
        longs_brk, weights_brk = decide_goldilocks_breakout(scores_v3, dd_row, rsi_row)
        if longs_brk:
            pnl = calc_pnl_meanrev(longs_brk, [], weights_brk, ret_row, 1.0)
            longs_str = ','.join(longs_brk)
        else:
            longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
            longs = [s for s, _ in longs_pool[:3]]
            pnl = calc_pnl(longs, [], scores_v3, ret_row, 1.0)
            longs_str = 'fallback'

        gold_weeks.append({
            'date': date, 'year': date.year, 'week': date.isocalendar()[1],
            'pnl_pct': pnl * 100, 'score_total': details.get('score_total', 0),
            'dd_healthy': details.get('pct_dd_healthy', 0),
            'rsi55': details.get('pct_rsi_above55', 0),
            'spy_dist': details.get('spy_dist', 0),
            'spy_mom': details.get('spy_mom_10w', 0),
            'vix': details.get('vix', 20),
            'n_breakout': n_breakout,
            'n_longs': len(longs_brk) if longs_brk else 3,
            'longs': longs_str,
        })

gdf = pd.DataFrame(gold_weeks)

# 1. Distribucion por ano
print("\n1. DISTRIBUCION GOLDILOCKS POR ANO")
print("-" * 100)
for year in sorted(gdf['year'].unique()):
    yr = gdf[gdf['year'] == year]
    avg = yr['pnl_pct'].mean()
    wr = (yr['pnl_pct'] > 0).mean() * 100
    tot = yr['pnl_pct'].sum()
    print(f"  {year}: {len(yr):>3} sem | Avg={avg:>+6.2f}% WR={wr:>5.1f}% | "
          f"Total={tot:>+7.2f}% | ScoreAvg={yr['score_total'].mean():.1f} VIX={yr['vix'].mean():.1f}")

# 2. Semanas con/sin candidatos breakout
print("\n2. EFICACIA BREAKOUT")
print("-" * 80)
has_brk = gdf[gdf['longs'] != 'fallback']
no_brk = gdf[gdf['longs'] == 'fallback']
print(f"  Con breakout:  {len(has_brk)} sem | "
      f"Avg={has_brk['pnl_pct'].mean():>+.3f}% WR={(has_brk['pnl_pct'] > 0).mean() * 100:.1f}%")
if len(no_brk) > 0:
    print(f"  Sin breakout:  {len(no_brk)} sem | "
          f"Avg={no_brk['pnl_pct'].mean():>+.3f}% WR={(no_brk['pnl_pct'] > 0).mean() * 100:.1f}%")
else:
    print("  Sin breakout:  0 sem")

# 3. PnL por score_total
print("\n3. PNL POR SCORE TOTAL")
print("-" * 80)
for lo, hi in [(7.0, 7.5), (7.5, 8.0), (8.0, 9.0)]:
    mask = (gdf['score_total'] >= lo) & (gdf['score_total'] < hi)
    if mask.sum() > 0:
        sub = gdf[mask]
        print(f"  Score [{lo:.1f}-{hi:.1f}): {mask.sum():>3} sem | "
              f"Avg={sub['pnl_pct'].mean():>+.3f}% WR={(sub['pnl_pct'] > 0).mean() * 100:.1f}%")

# 4. Peores semanas
print("\n4. TOP 10 PEORES SEMANAS GOLDILOCKS")
print("-" * 110)
worst = gdf.nsmallest(10, 'pnl_pct')
for _, r in worst.iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d')} w{int(r['week']):>2} | "
          f"PnL={r['pnl_pct']:>+6.2f}% Score={r['score_total']:.1f} "
          f"DD_H={r['dd_healthy']:.0f}% RSI55={r['rsi55']:.0f}% VIX={r['vix']:.1f} | {r['longs']}")

# 5. Mejores semanas
print("\n5. TOP 10 MEJORES SEMANAS GOLDILOCKS")
print("-" * 110)
best = gdf.nlargest(10, 'pnl_pct')
for _, r in best.iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d')} w{int(r['week']):>2} | "
          f"PnL={r['pnl_pct']:>+6.2f}% Score={r['score_total']:.1f} "
          f"DD_H={r['dd_healthy']:.0f}% RSI55={r['rsi55']:.0f}% VIX={r['vix']:.1f} | {r['longs']}")

# 6. Sectores mas frecuentes como longs
print("\n6. SECTORES MAS FRECUENTES EN LONGS GOLDILOCKS")
print("-" * 80)
sector_counts = Counter()
sector_pnls = {}
for _, r in gdf.iterrows():
    if r['longs'] == 'fallback':
        continue
    for s in r['longs'].split(','):
        sector_counts[s] += 1
for s, c in sector_counts.most_common(15):
    print(f"  {s:<25} {c:>4} veces")

# 7. Score total que CASI llega a GOLDILOCKS (6.0-7.0)
print("\n7. SEMANAS ALCISTA ALTAS (score 6.0-7.0) - potencial GOLDILOCKS")
print("-" * 80)
ardf = pd.DataFrame(all_regime_data)
near_gold = ardf[(ardf['score_total'] >= 6.0) & (ardf['score_total'] < 7.0)]
print(f"  {len(near_gold)} semanas con score 6.0-7.0 (actualmente ALCISTA)")
by_year = near_gold.groupby(near_gold['date'].dt.year).size()
for y, n in by_year.items():
    print(f"    {y}: {n} semanas")

# 8. N breakout candidatos en GOLDILOCKS
print("\n8. N CANDIDATOS BREAKOUT EN GOLDILOCKS")
print("-" * 80)
for n in sorted(gdf['n_breakout'].unique()):
    mask = gdf['n_breakout'] == n
    if mask.sum() > 0:
        sub = gdf[mask]
        print(f"  N={n:>2}: {mask.sum():>3} sem | "
              f"Avg={sub['pnl_pct'].mean():>+.3f}% WR={(sub['pnl_pct'] > 0).mean() * 100:.1f}%")

# 9. VIX distribution en GOLDILOCKS
print("\n9. GOLDILOCKS POR NIVEL DE VIX")
print("-" * 80)
for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 30)]:
    mask = (gdf['vix'] >= lo) & (gdf['vix'] < hi)
    if mask.sum() > 0:
        sub = gdf[mask]
        print(f"  VIX [{lo:>2}-{hi:>2}): {mask.sum():>3} sem | "
              f"Avg={sub['pnl_pct'].mean():>+.3f}% WR={(sub['pnl_pct'] > 0).mean() * 100:.1f}%")

# 10. SPY momentum en GOLDILOCKS
print("\n10. GOLDILOCKS POR SPY MOMENTUM 10w")
print("-" * 80)
for lo, hi in [(-5, 0), (0, 5), (5, 10), (10, 20), (20, 50)]:
    mask = (gdf['spy_mom'] >= lo) & (gdf['spy_mom'] < hi)
    if mask.sum() > 0:
        sub = gdf[mask]
        print(f"  Mom [{lo:>3}-{hi:>3}%): {mask.sum():>3} sem | "
              f"Avg={sub['pnl_pct'].mean():>+.3f}% WR={(sub['pnl_pct'] > 0).mean() * 100:.1f}%")

# 11. Comparar GOLDILOCKS breakout vs ALCISTA pullback vs simple top3
print("\n11. COMPARAR ESTRATEGIAS EN SEMANAS GOLDILOCKS")
print("-" * 80)
from report_compound import decide_alcista_pullback
pnl_breakout = []
pnl_pullback = []
pnl_top3 = []
for _, r in gdf.iterrows():
    date = r['date']
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

    # Breakout
    lb, wb = decide_goldilocks_breakout(scores_v3, dd_row, rsi_row)
    if lb:
        pnl_breakout.append(calc_pnl_meanrev(lb, [], wb, ret_row, 1.0) * 100)
    else:
        pnl_breakout.append(np.nan)

    # Pullback
    lp, wp = decide_alcista_pullback(scores_v3, dd_row, rsi_row)
    if lp:
        pnl_pullback.append(calc_pnl_meanrev(lp, [], wp, ret_row, 1.0) * 100)
    else:
        pnl_pullback.append(np.nan)

    # Top3 por score
    top3 = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 5.5], key=lambda x: -x[1])[:3]
    longs_t3 = [s for s, _ in top3]
    pnl_top3.append(calc_pnl(longs_t3, [], scores_v3, ret_row, 1.0) * 100)

pb = pd.Series(pnl_breakout).dropna()
pp = pd.Series(pnl_pullback).dropna()
pt = pd.Series(pnl_top3)

print(f"  Breakout (actual): {len(pb):>3} sem | Avg={pb.mean():>+.3f}% WR={(pb > 0).mean() * 100:.1f}% | Sharpe={(pb.mean()/pb.std()*np.sqrt(52)):.2f}")
if len(pp) > 0:
    print(f"  Pullback:          {len(pp):>3} sem | Avg={pp.mean():>+.3f}% WR={(pp > 0).mean() * 100:.1f}% | Sharpe={(pp.mean()/pp.std()*np.sqrt(52)):.2f}")
print(f"  Top3 FairValue:    {len(pt):>3} sem | Avg={pt.mean():>+.3f}% WR={(pt > 0).mean() * 100:.1f}% | Sharpe={(pt.mean()/pt.std()*np.sqrt(52)):.2f}")
