"""
Diagnostico BEARISH (65 sem) + CRISIS (295 sem)
Problema: muy pocas BEARISH, demasiadas CRISIS
Objetivo: redistribuir + añadir PANICO
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
COST_RT = (COST_PER_TRADE + SLIPPAGE_PER_SIDE) * 2

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
# RECOPILAR SEMANAS BEARISH+CRISIS CON TODOS LOS INDICADORES
# ================================================================
print("Analizando semanas con score < -1.5 (BEARISH+CRISIS actuales)...")

bear_crisis_weeks = []

for date in returns_wide.index:
    if date.year < 2001: continue

    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: continue
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: continue
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

    # Calcular score total
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

    # Solo semanas negativas (CAUTIOUS o peor: total < 0.5)
    # Pero centrandonos en BEARISH+CRISIS: total < -1.5
    if total >= 0.5: continue  # skip NEUTRAL y bullish

    # VIX override
    regime_orig = 'NEUTRAL' if total >= 0.5 else 'CAUTIOUS' if total >= -1.5 else 'BEARISH' if total >= -3.0 else 'CRISIS'
    if vix_val >= 35 and regime_orig == 'NEUTRAL':
        regime_orig = 'CAUTIOUS'

    # SPY return de esa semana
    spy_ret = spy_w.loc[spy_w.index[spy_w.index <= date][-1], 'ret_spy'] if len(spy_w.index[spy_w.index <= date]) > 0 else 0
    if not pd.notna(spy_ret): spy_ret = 0

    # Retornos de subsectores
    if date in returns_trade_wide.index and returns_trade_wide.loc[date].notna().any():
        ret_row = returns_trade_wide.loc[date]
    else:
        ret_row = returns_wide.loc[date]

    # FV scores
    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    scores_evt = score_fair(active) if active else {s: 5.0 for s in SUBSECTORS}
    prev_dd_dates = dd_wide.index[dd_wide.index < date]
    dd_row_prev = dd_wide.loc[prev_dd_dates[-1]] if len(prev_dd_dates) > 0 else None
    rsi_row_prev = rsi_wide.loc[prev_dd_dates[-1]] if len(prev_dd_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row_prev, rsi_row_prev)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    # Shorts pool
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])
    n_shorts_avail = len(shorts_pool)

    # SPY retorno
    bear_crisis_weeks.append({
        'date': date, 'year': date.year, 'score_total': total, 'regime': regime_orig,
        'vix': vix_val, 'spy_ret': spy_ret, 'spy_dist': spy_dist, 'spy_mom': spy_mom_10w,
        'pct_dd_healthy': pct_dd_healthy, 'pct_dd_deep': pct_dd_deep,
        'pct_rsi_above55': pct_rsi_above55, 'n_shorts_avail': n_shorts_avail,
        'score_bdd': score_bdd, 'score_brsi': score_brsi, 'score_ddp': score_ddp,
        'score_spy': score_spy, 'score_mom': score_mom,
    })

df = pd.DataFrame(bear_crisis_weeks)

# ================================================================
# 1. DISTRIBUCION DE SCORE TOTAL (para entender donde cortar)
# ================================================================
print(f"\n{'='*120}")
print(f"  DISTRIBUCION SCORE TOTAL - Semanas con score < 0.5 ({len(df)} semanas)")
print(f"{'='*120}")

bins = [(-10, -6), (-6, -5), (-5, -4), (-4, -3.5), (-3.5, -3), (-3, -2.5), (-2.5, -2), (-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 0.5)]
print(f"\n  {'Score range':<15s} {'N':>4s} {'Reg actual':>12s} {'SPY avg%':>9s} {'VIX avg':>8s} {'DD deep%':>9s} {'RSI>55%':>8s} {'SPY dist':>9s}")
print(f"  {'-'*80}")
for lo, hi in bins:
    sub = df[(df['score_total'] >= lo) & (df['score_total'] < hi)]
    if len(sub) == 0: continue
    regimes = sub['regime'].value_counts().to_dict()
    reg_str = '/'.join(f"{r}:{n}" for r, n in sorted(regimes.items()))
    print(f"  [{lo:>+5.1f},{hi:>+5.1f}) {len(sub):>4d} {reg_str:>12s} {sub['spy_ret'].mean()*100:>+8.2f}% "
          f"{sub['vix'].mean():>7.1f} {sub['pct_dd_deep'].mean():>8.1f}% {sub['pct_rsi_above55'].mean():>7.1f}% "
          f"{sub['spy_dist'].mean():>+8.1f}%")

# ================================================================
# 2. DISTRIBUCION ACTUAL: CAUTIOUS vs BEARISH vs CRISIS
# ================================================================
print(f"\n{'='*120}")
print(f"  REGIMEN ACTUAL")
print(f"{'='*120}")
for reg in ['CAUTIOUS', 'BEARISH', 'CRISIS']:
    sub = df[df['regime'] == reg]
    if len(sub) == 0: continue
    print(f"\n  {reg} ({len(sub)} sem):")
    print(f"    Score total: min={sub['score_total'].min():.1f} max={sub['score_total'].max():.1f} "
          f"avg={sub['score_total'].mean():.1f} med={sub['score_total'].median():.1f}")
    print(f"    VIX: avg={sub['vix'].mean():.1f} med={sub['vix'].median():.1f} max={sub['vix'].max():.1f}")
    print(f"    SPY ret: avg={sub['spy_ret'].mean()*100:+.2f}% WR={(sub['spy_ret']>0).mean()*100:.1f}%")
    print(f"    DD deep%: avg={sub['pct_dd_deep'].mean():.1f}% DD healthy%: avg={sub['pct_dd_healthy'].mean():.1f}%")
    print(f"    SPY dist MA200: avg={sub['spy_dist'].mean():+.1f}% SPY mom 10w: avg={sub['spy_mom'].mean():+.1f}%")
    # Por ano
    print(f"    Por ano: ", end='')
    for year in sorted(sub['year'].unique()):
        n = (sub['year'] == year).sum()
        print(f"{year}:{n} ", end='')
    print()

# ================================================================
# 3. PROPUESTA: 4 niveles (CAUTIOUS / BEARISH / CRISIS / PANICO)
# ================================================================
print(f"\n{'='*120}")
print(f"  PROPUESTA: 4 NIVELES NEGATIVOS")
print(f"{'='*120}")

# Propuesta A: redistribuir umbrales
proposals = {
    'A_actual':    [('CAUTIOUS', -1.5, 0.5), ('BEARISH', -3.0, -1.5), ('CRISIS', -99, -3.0)],
    'B_rebalance': [('CAUTIOUS', -1.0, 0.5), ('BEARISH', -2.5, -1.0), ('CRISIS', -4.0, -2.5), ('PANICO', -99, -4.0)],
    'C_equil':     [('CAUTIOUS', -1.0, 0.5), ('BEARISH', -2.0, -1.0), ('CRISIS', -3.5, -2.0), ('PANICO', -99, -3.5)],
    'D_aggr':      [('CAUTIOUS', -0.5, 0.5), ('BEARISH', -2.0, -0.5), ('CRISIS', -4.0, -2.0), ('PANICO', -99, -4.0)],
}

for pname, levels in proposals.items():
    print(f"\n  --- {pname} ---")
    print(f"  {'Regimen':<12s} {'Score range':<16s} {'N':>4s} {'SPY avg%':>9s} {'VIX avg':>8s} {'DD deep':>8s} {'Shorts':>7s}")
    print(f"  {'-'*70}")
    for reg, lo, hi in levels:
        sub = df[(df['score_total'] >= lo) & (df['score_total'] < hi)]
        if len(sub) == 0:
            print(f"  {reg:<12s} [{lo:>+5.1f},{hi:>+5.1f}) {0:>4d}   ---")
            continue
        print(f"  {reg:<12s} [{lo:>+5.1f},{hi:>+5.1f}) {len(sub):>4d} {sub['spy_ret'].mean()*100:>+8.2f}% "
              f"{sub['vix'].mean():>7.1f} {sub['pct_dd_deep'].mean():>7.1f}% {sub['n_shorts_avail'].mean():>6.1f}")

# ================================================================
# 4. DETALLE POR AÑO para cada propuesta
# ================================================================
print(f"\n{'='*120}")
print(f"  DISTRIBUCION POR ANO - Propuesta B vs Propuesta C")
print(f"{'='*120}")

for pname, levels in [('B_rebalance', proposals['B_rebalance']), ('C_equil', proposals['C_equil'])]:
    print(f"\n  --- {pname} ---")
    years = sorted(df['year'].unique())
    header = f"  {'Ano':>5s}"
    for reg, _, _ in levels:
        header += f" {reg:>8s}"
    header += "   Total"
    print(header)
    print(f"  {'-'*60}")
    for year in years:
        yr = df[df['year'] == year]
        line = f"  {year:>5d}"
        total_yr = 0
        for reg, lo, hi in levels:
            n = len(yr[(yr['score_total'] >= lo) & (yr['score_total'] < hi)])
            line += f" {n:>8d}"
            total_yr += n
        line += f"   {total_yr:>5d}"
        print(line)
    # Totals
    line = f"  {'TOTAL':>5s}"
    for reg, lo, hi in levels:
        n = len(df[(df['score_total'] >= lo) & (df['score_total'] < hi)])
        line += f" {n:>8d}"
    line += f"   {len(df):>5d}"
    print(line)

# ================================================================
# 5. VIX como indicador de PANICO
# ================================================================
print(f"\n{'='*120}")
print(f"  VIX COMO INDICADOR DE PANICO")
print(f"{'='*120}")

for vix_thresh in [25, 30, 35, 40, 45]:
    sub = df[df['vix'] >= vix_thresh]
    if len(sub) == 0: continue
    print(f"\n  VIX >= {vix_thresh}: {len(sub)} semanas")
    print(f"    Score total: avg={sub['score_total'].mean():.1f} min={sub['score_total'].min():.1f}")
    print(f"    SPY ret: avg={sub['spy_ret'].mean()*100:+.2f}%  WR={(sub['spy_ret']>0).mean()*100:.1f}%")
    print(f"    DD deep: avg={sub['pct_dd_deep'].mean():.1f}%  DD healthy: avg={sub['pct_dd_healthy'].mean():.1f}%")
    by_year = sub.groupby('year').size()
    print(f"    Anos: {dict(by_year)}")

# ================================================================
# 6. COMBINACION SCORE + VIX para definir PANICO
# ================================================================
print(f"\n{'='*120}")
print(f"  COMBINACION SCORE + VIX PARA PANICO")
print(f"{'='*120}")

for score_thresh, vix_thresh in [(-4.0, 30), (-3.5, 35), (-4.0, 35), (-3.0, 40), (-5.0, 30), (-4.0, 40)]:
    sub = df[(df['score_total'] < score_thresh) | (df['vix'] >= vix_thresh)]
    sub2 = df[(df['score_total'] < score_thresh) & (df['vix'] >= vix_thresh)]
    if len(sub) == 0: continue
    print(f"\n  score<{score_thresh} OR VIX>={vix_thresh}: {len(sub)} sem (AND: {len(sub2)} sem)")
    print(f"    SPY ret: avg={sub['spy_ret'].mean()*100:+.2f}%  WR={(sub['spy_ret']>0).mean()*100:.1f}%")
    print(f"    DD deep: avg={sub['pct_dd_deep'].mean():.1f}%")
