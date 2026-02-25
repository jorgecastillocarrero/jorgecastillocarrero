"""
Analisis BEARISH: puede beneficiarse de Mean Reversion?
=========================================================
Compara el sistema actual (score-based) vs mean reversion para br 0.45-0.60
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
sub_labels = {sid: sd['label'] for sid, sd in SUBSECTORS.items()}
MAX_CONTRIBUTION = 4.0
CAPITAL = 500_000
ATR_MIN = 1.5

def score_fair(active_events):
    contributions = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        for subsec, impact in EVENT_SUBSECTOR_MAP[evt_type]['impacto'].items():
            if subsec not in contributions:
                contributions[subsec] = []
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

def decide_allocation(scores, max_pos=3):
    longs_pool = sorted([(s, sc) for s, sc in scores.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores.items() if sc < 3.5], key=lambda x: x[1])
    bear_count = len(shorts_pool)
    bull_count = len(longs_pool)
    if bear_count + bull_count == 0:
        bear_ratio = 0.5
    else:
        bear_ratio = bear_count / (bear_count + bull_count)
    if bear_ratio >= 0.70:   nl, ns = 0, max_pos
    elif bear_ratio >= 0.60: nl, ns = 1, max_pos
    elif bear_ratio >= 0.55: nl, ns = 2, max_pos
    elif bear_ratio >= 0.45: nl, ns = max_pos, max_pos
    elif bear_ratio >= 0.40: nl, ns = max_pos, 2
    elif bear_ratio >= 0.30: nl, ns = max_pos, 1
    else:                    nl, ns = max_pos, 0
    return [s for s, _ in longs_pool[:nl]], [s for s, _ in shorts_pool[:ns]], bear_ratio

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

def decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row):
    long_cands = []
    short_cands = []
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
        if oversold > 0.1 and score >= 4.0:
            long_cands.append((sub_id, oversold))
        rsi_ob = np.clip((rsi - 65) / 20, 0, 1)
        if rsi_ob > 0.1 and dd > -8 and score <= 6.0 and atr >= ATR_MIN:
            short_cands.append((sub_id, rsi_ob))
    long_cands.sort(key=lambda x: -x[1])
    short_cands.sort(key=lambda x: -x[1])
    longs = [c[0] for c in long_cands[:3]]
    shorts = [c[0] for c in short_cands[:2]]
    weights = {}
    for s, w in long_cands[:3]: weights[s] = w
    for s, w in short_cands[:2]: weights[s] = w
    return longs, shorts, weights

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

# ---- Load data ----
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
    lambda x: x.rolling(5, min_periods=3).mean() * 100
)

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean'),
    avg_atr=('atr_pct', 'mean'),
).reset_index()
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
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide_lagged = atr_wide.shift(1)

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# ---- Comparar BEARISH: actual vs mean reversion ----
print("\nAnalizando semanas BEARISH (br 0.45-0.60)...\n")

rows = []

for date in returns_wide.index:
    if date.year < 2002:
        continue

    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    if not active:
        continue

    scores_evt = score_fair(active)

    prev_dates = dd_wide.index[dd_wide.index < date]
    if len(prev_dates) > 0:
        dd_row = dd_wide.loc[prev_dates[-1]]
        rsi_row = rsi_wide.loc[prev_dates[-1]]
    else:
        dd_row = None
        rsi_row = None

    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    longs_v3, shorts_v3, br_v3 = decide_allocation(scores_v3)

    # Solo BEARISH (0.45 <= br < 0.60)
    if br_v3 < 0.45 or br_v3 >= 0.60:
        continue

    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    # Filtro ATR
    if atr_row is not None:
        shorts_v3_filt = [s for s in shorts_v3 if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
    else:
        shorts_v3_filt = shorts_v3

    ret_row = returns_wide.loc[date]

    # PnL actual (score-based)
    pnl_actual = calc_pnl(longs_v3, shorts_v3_filt, scores_v3, ret_row, CAPITAL)

    # PnL con mean reversion
    longs_mr, shorts_mr, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
    pnl_mr = calc_pnl_meanrev(longs_mr, shorts_mr, weights_mr, ret_row, CAPITAL)

    # PnL solo longs actuales (sin shorts)
    pnl_lonly = calc_pnl(longs_v3, [], scores_v3, ret_row, CAPITAL)

    # PnL solo shorts actuales (sin longs)
    pnl_sonly = calc_pnl([], shorts_v3_filt, scores_v3, ret_row, CAPITAL)

    cfg_actual = f"{len(longs_v3)}L+{len(shorts_v3_filt)}S"
    cfg_mr = f"{len(longs_mr)}L+{len(shorts_mr)}S*"

    # Desglose: que retorno tuvo cada posicion
    long_labels = [sub_labels.get(s, s) for s in longs_v3]
    short_labels = [sub_labels.get(s, s) for s in shorts_v3_filt]
    long_rets = [ret_row.get(s, 0)*100 for s in longs_v3 if pd.notna(ret_row.get(s))]
    short_rets = [-ret_row.get(s, 0)*100 for s in shorts_v3_filt if pd.notna(ret_row.get(s))]

    rows.append({
        'date': date,
        'year': date.year,
        'br': br_v3,
        'cfg_actual': cfg_actual,
        'cfg_mr': cfg_mr,
        'pnl_actual': pnl_actual,
        'pnl_mr': pnl_mr,
        'pnl_lonly': pnl_lonly,
        'pnl_sonly': pnl_sonly,
        'n_longs': len(longs_v3),
        'n_shorts': len(shorts_v3_filt),
        'n_longs_mr': len(longs_mr),
        'n_shorts_mr': len(shorts_mr),
        'long_labels': ', '.join(long_labels),
        'short_labels': ', '.join(short_labels),
        'avg_long_ret': np.mean(long_rets) if long_rets else 0,
        'avg_short_ret': np.mean(short_rets) if short_rets else 0,
    })

df = pd.DataFrame(rows)

print(f"{'='*120}")
print(f"  ANALISIS BEARISH - {len(df)} semanas (br 0.45-0.60)")
print(f"{'='*120}")

# 1. Totales
print(f"\n  1. COMPARACION GLOBAL:")
print(f"     Score-based (actual):  PnL=${df['pnl_actual'].sum():>+12,.0f}  Avg=${df['pnl_actual'].mean():>+8,.0f}  WR={(df['pnl_actual']>0).mean()*100:5.1f}%")
print(f"     Mean Reversion:        PnL=${df['pnl_mr'].sum():>+12,.0f}  Avg=${df['pnl_mr'].mean():>+8,.0f}  WR={(df['pnl_mr']>0).mean()*100:5.1f}%")
print(f"     Diferencia MR-actual:  ${df['pnl_mr'].sum()-df['pnl_actual'].sum():>+12,.0f}")

# 2. Desglose longs vs shorts
print(f"\n  2. LONGS vs SHORTS (actual score-based):")
print(f"     Solo longs:   PnL=${df['pnl_lonly'].sum():>+12,.0f}  Avg=${df['pnl_lonly'].mean():>+8,.0f}")
print(f"     Solo shorts:  PnL=${df['pnl_sonly'].sum():>+12,.0f}  Avg=${df['pnl_sonly'].mean():>+8,.0f}")
print(f"     Combinado:    PnL=${df['pnl_actual'].sum():>+12,.0f}")
print(f"     Los longs contribuyen {df['pnl_lonly'].sum()/(df['pnl_lonly'].sum()+df['pnl_sonly'].sum())*100:.0f}% del PnL total" if df['pnl_lonly'].sum()+df['pnl_sonly'].sum() != 0 else "")

# 3. Por config
print(f"\n  3. POR CONFIGURACION (actual):")
for cfg in sorted(df['cfg_actual'].unique()):
    subset = df[df['cfg_actual'] == cfg]
    print(f"     {cfg:8s}: {len(subset):3d} sem  PnL=${subset['pnl_actual'].sum():>+10,.0f}  Avg=${subset['pnl_actual'].mean():>+6,.0f}  WR={(subset['pnl_actual']>0).mean()*100:5.1f}%")

# 4. Por bear_ratio band
print(f"\n  4. POR BEAR_RATIO BAND:")
for lo, hi, label in [(0.45, 0.50, "0.45-0.50 (mild)"), (0.50, 0.55, "0.50-0.55"), (0.55, 0.60, "0.55-0.60 (harsh)")]:
    subset = df[(df['br'] >= lo) & (df['br'] < hi)]
    if len(subset) == 0:
        continue
    print(f"     br {label:20s}: {len(subset):3d} sem")
    print(f"       Actual: PnL=${subset['pnl_actual'].sum():>+10,.0f}  Avg=${subset['pnl_actual'].mean():>+6,.0f}  WR={(subset['pnl_actual']>0).mean()*100:5.1f}%")
    print(f"       MeanRv: PnL=${subset['pnl_mr'].sum():>+10,.0f}  Avg=${subset['pnl_mr'].mean():>+6,.0f}  WR={(subset['pnl_mr']>0).mean()*100:5.1f}%")

# 5. Por ano
print(f"\n  5. POR ANO:")
print(f"     {'Ano':>4s} {'Sem':>4s} {'Actual':>12s} {'MeanRev':>12s} {'Dif':>12s}")
print(f"     {'-'*50}")
for yr in sorted(df['year'].unique()):
    subset = df[df['year'] == yr]
    pa = subset['pnl_actual'].sum()
    pm = subset['pnl_mr'].sum()
    marker = ""
    if pm - pa > 20000: marker = " <-- MR mejor"
    elif pa - pm > 20000: marker = " <-- Actual mejor"
    print(f"     {yr:>4d} {len(subset):>4d} ${pa:>+10,.0f} ${pm:>+10,.0f} ${pm-pa:>+10,.0f}{marker}")
total_a = df['pnl_actual'].sum()
total_m = df['pnl_mr'].sum()
print(f"     {'-'*50}")
print(f"     {'TOT':>4s} {len(df):>4d} ${total_a:>+10,.0f} ${total_m:>+10,.0f} ${total_m-total_a:>+10,.0f}")

# 6. Sharpe comparison
rets_actual = df['pnl_actual'] / CAPITAL
rets_mr = df['pnl_mr'] / CAPITAL
sh_a = rets_actual.mean() / rets_actual.std() * np.sqrt(52) if rets_actual.std() > 0 else 0
sh_m = rets_mr.mean() / rets_mr.std() * np.sqrt(52) if rets_mr.std() > 0 else 0
print(f"\n  6. SHARPE:")
print(f"     Actual:   {sh_a:.2f}")
print(f"     MeanRev:  {sh_m:.2f}")

# 7. Contexto: que pasa en BEARISH
print(f"\n  7. CONTEXTO BEARISH:")
# DD y RSI medios
dd_vals = []
rsi_vals = []
for date in df['date']:
    prev_dates = dd_wide.index[dd_wide.index < date]
    if len(prev_dates) > 0:
        dd_vals.append(dd_wide.loc[prev_dates[-1]].mean())
        rsi_vals.append(rsi_wide.loc[prev_dates[-1]].mean())
print(f"     DD promedio mercado: {np.mean(dd_vals):.1f}%")
print(f"     RSI promedio mercado: {np.mean(rsi_vals):.1f}")
print(f"     Avg long return/sem (actual): {df['avg_long_ret'].mean():+.2f}%")
print(f"     Avg short return/sem (actual): {df['avg_short_ret'].mean():+.2f}%")

# 8. 10 peores semanas actual vs mean reversion
print(f"\n  8. 10 PEORES SEMANAS ACTUAL vs MEAN REVERSION:")
worst = df.nsmallest(10, 'pnl_actual')
for _, r in worst.iterrows():
    print(f"     {r['date'].strftime('%Y-%m-%d')}  br={r['br']:.2f}  {r['cfg_actual']:6s}  Actual=${r['pnl_actual']:>+9,.0f}  MR=${r['pnl_mr']:>+9,.0f}  Dif=${r['pnl_mr']-r['pnl_actual']:>+9,.0f}")

# 9. 10 mejores semanas
print(f"\n  9. 10 MEJORES SEMANAS ACTUAL vs MEAN REVERSION:")
best = df.nlargest(10, 'pnl_actual')
for _, r in best.iterrows():
    print(f"     {r['date'].strftime('%Y-%m-%d')}  br={r['br']:.2f}  {r['cfg_actual']:6s}  Actual=${r['pnl_actual']:>+9,.0f}  MR=${r['pnl_mr']:>+9,.0f}  Dif=${r['pnl_mr']-r['pnl_actual']:>+9,.0f}")

# 10. Weekly diff histogram
diffs = df['pnl_mr'] - df['pnl_actual']
print(f"\n  10. DISTRIBUCION DIFERENCIAS (MR - Actual):")
print(f"     MR gana:  {(diffs > 0).sum()} semanas ({(diffs>0).mean()*100:.0f}%)")
print(f"     MR pierde: {(diffs < 0).sum()} semanas ({(diffs<0).mean()*100:.0f}%)")
print(f"     Empate:    {(diffs == 0).sum()} semanas")
print(f"     Media dif: ${diffs.mean():+,.0f}/sem")
print(f"     Mediana:   ${diffs.median():+,.0f}/sem")
