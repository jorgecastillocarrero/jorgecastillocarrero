"""
Test NEUTRAL Mean Reversion: 3L+2S basado en price extremes
===========================================================
En NEUTRAL, los event scores son mediocres (mayoria en zona 3.5-6.5).
En vez de confiar en scores con baja conviccion, usar el ESTADO DEL PRECIO:
- LONGS: subsectores oversold (DD alto, RSI bajo) que los eventos no castigan
- SHORTS: subsectores overbought (RSI alto, near highs) que los eventos no favorecen
- Evitar la zona media: solo operar extremos del rango
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

def decide_allocation_standard(scores, max_pos=3):
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


def decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row):
    """
    Mean Reversion para NEUTRAL: 3L+2S

    LONGS: subsectores oversold que eventos no castigan
    - DD < -15% OR RSI < 40 (en soporte / zona baja del rango)
    - Event score >= 4.0 (eventos no fuertemente negativos)
    - Ranking por "oversold_strength" = max(dd_factor, rsi_factor)

    SHORTS: subsectores overbought que eventos no favorecen
    - RSI > 65 AND DD > -8% (en resistencia / zona alta del rango)
    - Event score <= 6.0 (eventos no fuertemente positivos)
    - ATR >= 1.5%
    - Ranking por rsi (mas overbought primero)

    EVITAR ZONA MEDIA: no operar subsectores en rango intermedio
    """
    long_candidates = []
    short_candidates = []

    for sub_id in SUBSECTORS:
        score = scores_evt.get(sub_id, 5.0)
        dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        atr = atr_row.get(sub_id, 0) if atr_row is not None else 0

        if not pd.notna(dd): dd = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(atr): atr = 0

        # LONG candidates: oversold + eventos no negativos
        dd_factor = np.clip((abs(dd) - 15) / 30, 0, 1)
        rsi_factor_os = np.clip((40 - rsi) / 25, 0, 1)  # 40 -> 15
        oversold_str = max(dd_factor, rsi_factor_os)

        if oversold_str > 0.1 and score >= 4.0:
            long_candidates.append((sub_id, oversold_str, score, dd, rsi))

        # SHORT candidates: overbought + eventos no positivos
        rsi_factor_ob = np.clip((rsi - 65) / 20, 0, 1)  # 65 -> 85
        near_highs = dd > -8  # within 8% of 52w high

        if rsi_factor_ob > 0.1 and near_highs and score <= 6.0 and atr >= ATR_MIN:
            short_candidates.append((sub_id, rsi_factor_ob, score, dd, rsi))

    # Sort: longs by oversold strength (desc), shorts by overbought strength (desc)
    long_candidates.sort(key=lambda x: -x[1])
    short_candidates.sort(key=lambda x: -x[1])

    longs = [c[0] for c in long_candidates[:3]]
    shorts = [c[0] for c in short_candidates[:2]]

    return longs, shorts, long_candidates[:5], short_candidates[:5]


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
    avg_close=('close', 'mean'),
    avg_high=('high', 'mean'),
    avg_low=('low', 'mean'),
    avg_return=('return', 'mean'),
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


# ---- Backtest: comparar ORIGINAL vs MEAN REVERSION en NEUTRAL ----
print("Ejecutando backtest comparativo...")

results_original = []
results_meanrev = []

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
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None

    scores_adj = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    # Determine regime
    longs_std, shorts_std, bear_ratio = decide_allocation_standard(scores_adj)

    if bear_ratio >= 0.60:
        regime = 'CRISIS'
    elif bear_ratio >= 0.45:
        regime = 'BEARISH'
    elif bear_ratio >= 0.30:
        regime = 'NEUTRAL'
    else:
        regime = 'BULLISH'

    ret_row = returns_wide.loc[date]

    # ---- ORIGINAL (score-based) ----
    # ATR filter
    if atr_row is not None:
        shorts_std_f = [s for s in shorts_std if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
    else:
        shorts_std_f = shorts_std

    n = len(longs_std) + len(shorts_std_f)
    pnl_orig = 0
    if n > 0:
        lw = {s: scores_adj[s] - 5.0 for s in longs_std}
        sw = {s: 5.0 - scores_adj[s] for s in shorts_std_f}
        tw = sum(lw.values()) + sum(sw.values())
        if tw > 0:
            for s in longs_std:
                if pd.notna(ret_row.get(s)):
                    pnl_orig += CAPITAL * (lw[s] / tw) * ret_row[s]
            for s in shorts_std_f:
                if pd.notna(ret_row.get(s)):
                    pnl_orig += CAPITAL * (sw[s] / tw) * (-ret_row[s])

    results_original.append({'date': date, 'pnl': pnl_orig, 'regime': regime,
                            'config': f"{len(longs_std)}L+{len(shorts_std_f)}S",
                            'bear_ratio': bear_ratio})

    # ---- MEAN REVERSION (only for NEUTRAL) ----
    if regime == 'NEUTRAL':
        longs_mr, shorts_mr, long_cands, short_cands = decide_neutral_meanrev(
            scores_evt, dd_row, rsi_row, atr_row)

        n_mr = len(longs_mr) + len(shorts_mr)
        pnl_mr = 0
        if n_mr > 0:
            # Weight by oversold/overbought strength
            lw_mr = {}
            for c in long_cands:
                if c[0] in longs_mr:
                    lw_mr[c[0]] = c[1]  # oversold_strength
            sw_mr = {}
            for c in short_cands:
                if c[0] in shorts_mr:
                    sw_mr[c[0]] = c[1]  # overbought_strength
            tw_mr = sum(lw_mr.values()) + sum(sw_mr.values())
            if tw_mr > 0:
                for s in longs_mr:
                    if pd.notna(ret_row.get(s)):
                        pnl_mr += CAPITAL * (lw_mr.get(s, 0) / tw_mr) * ret_row[s]
                for s in shorts_mr:
                    if pd.notna(ret_row.get(s)):
                        pnl_mr += CAPITAL * (sw_mr.get(s, 0) / tw_mr) * (-ret_row[s])

        results_meanrev.append({'date': date, 'pnl': pnl_mr,
                               'longs': ', '.join(sub_labels.get(s, s)[:25] for s in longs_mr),
                               'shorts': ', '.join(sub_labels.get(s, s)[:25] for s in shorts_mr),
                               'config': f"{len(longs_mr)}L+{len(shorts_mr)}S",
                               'pnl_orig': pnl_orig,
                               'bear_ratio': bear_ratio,
                               'n_long_cands': len(long_cands),
                               'n_short_cands': len(short_cands)})
    else:
        results_meanrev.append({'date': date, 'pnl': pnl_orig,
                               'longs': '', 'shorts': '',
                               'config': f"{len(longs_std)}L+{len(shorts_std_f)}S",
                               'pnl_orig': pnl_orig,
                               'bear_ratio': bear_ratio,
                               'n_long_cands': 0, 'n_short_cands': 0})

df_orig = pd.DataFrame(results_original)
df_mr = pd.DataFrame(results_meanrev)

# ---- Results ----
print(f"\n{'='*120}")
print(f"  NEUTRAL: ORIGINAL (score-based) vs MEAN REVERSION (price-based)")
print(f"{'='*120}")

# Overall
neutral_orig = df_orig[df_orig['regime'] == 'NEUTRAL']
neutral_mr = df_mr[df_mr.index.isin(neutral_orig.index)]

pnl_o = neutral_orig['pnl'].sum()
pnl_m = neutral_mr['pnl'].sum()
wr_o = (neutral_orig['pnl'] > 0).mean() * 100
wr_m = (neutral_mr['pnl'] > 0).mean() * 100

print(f"\n  {'Metrica':<25s} {'Original':>12s} {'Mean Rev':>12s} {'Diferencia':>12s}")
print(f"  {'-'*65}")
print(f"  {'Semanas':<25s} {len(neutral_orig):>12d} {len(neutral_mr):>12d}")
print(f"  {'P&L Total':<25s} ${pnl_o:>+10,.0f} ${pnl_m:>+10,.0f} ${pnl_m-pnl_o:>+10,.0f}")
print(f"  {'P&L Avg/semana':<25s} ${pnl_o/len(neutral_orig):>+10,.0f} ${pnl_m/len(neutral_mr):>+10,.0f}")
print(f"  {'Win Rate':<25s} {wr_o:>11.1f}% {wr_m:>11.1f}%")

# Total system impact
total_orig = df_orig['pnl'].sum()
total_mr = df_mr['pnl'].sum()
print(f"\n  {'TOTAL SISTEMA':<25s} ${total_orig:>+10,.0f} ${total_mr:>+10,.0f} ${total_mr-total_orig:>+10,.0f}")

# By year
print(f"\n  PNL NEUTRAL POR ANO:")
print(f"  {'Ano':>6s} {'Original':>10s} {'Mean Rev':>10s} {'Diff':>10s} {'Sem':>4s}")
print(f"  {'-'*45}")
for year in sorted(neutral_orig['date'].dt.year.unique()):
    no = neutral_orig[neutral_orig['date'].dt.year == year]
    nm = neutral_mr[neutral_mr.index.isin(no.index)]
    po = no['pnl'].sum()
    pm = nm['pnl'].sum()
    print(f"  {year:>6d} ${po:>+9,.0f} ${pm:>+9,.0f} ${pm-po:>+9,.0f} {len(no):>4d}")

# Config distribution in mean rev
print(f"\n  CONFIGS MEAN REVERSION EN NEUTRAL:")
neutral_mr_only = df_mr[df_mr.index.isin(neutral_orig.index)]
cfgs = neutral_mr_only['config'].value_counts()
for cfg, cnt in cfgs.items():
    sub = neutral_mr_only[neutral_mr_only['config'] == cfg]
    print(f"    {cfg:8s}: {cnt:3d} sem  PnL=${sub['pnl'].sum():>+10,.0f}  Avg=${sub['pnl'].mean():>+6,.0f}")

# Candidates availability
n_lc = neutral_mr_only['n_long_cands'].mean()
n_sc = neutral_mr_only['n_short_cands'].mean()
print(f"\n  Candidatos promedio: {n_lc:.1f} longs oversold, {n_sc:.1f} shorts overbought")
print(f"  Semanas sin longs: {(neutral_mr_only['n_long_cands'] == 0).sum()}")
print(f"  Semanas sin shorts: {(neutral_mr_only['n_short_cands'] == 0).sum()}")

# Detail: weeks where mean rev differs most from original
print(f"\n  TOP 10 SEMANAS CON MAYOR MEJORA (Mean Rev vs Original):")
neutral_mr_only = neutral_mr_only.copy()
neutral_mr_only['diff'] = neutral_mr_only['pnl'] - neutral_mr_only['pnl_orig']
best_diff = neutral_mr_only.nlargest(10, 'diff')
for _, r in best_diff.iterrows():
    print(f"    {r['date'].strftime('%Y-%m-%d')}  MR=${r['pnl']:>+8,.0f}  Orig=${r['pnl_orig']:>+8,.0f}  Diff=${r['diff']:>+8,.0f}  {r['config']}  L:{str(r['longs'])[:35]}  S:{str(r['shorts'])[:35]}")

print(f"\n  TOP 10 SEMANAS CON PEOR RESULTADO (Mean Rev vs Original):")
worst_diff = neutral_mr_only.nsmallest(10, 'diff')
for _, r in worst_diff.iterrows():
    print(f"    {r['date'].strftime('%Y-%m-%d')}  MR=${r['pnl']:>+8,.0f}  Orig=${r['pnl_orig']:>+8,.0f}  Diff=${r['diff']:>+8,.0f}  {r['config']}  L:{str(r['longs'])[:35]}  S:{str(r['shorts'])[:35]}")
