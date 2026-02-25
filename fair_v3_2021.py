"""
Fair V3: Score Eventos + Ajuste Precio - AÑO 2021 (superalcista)
================================================================
SPY 2021: +26.9%. Verificar que el sistema funciona en bull market.
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

# ---- Load data ----
print("Loading data...")
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
    AND date BETWEEN '2019-01-01' AND '2022-02-01'
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

weekly_events = build_weekly_events('2019-01-01', '2022-02-01')

CAPITAL = 500_000
ATR_MIN = 1.5

# SPY
spy = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2020-12-20' AND '2022-01-10'
    ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date')

# ---- 2021 ----
weeks_2021 = returns_wide.index[
    (returns_wide.index >= '2021-01-01') & (returns_wide.index <= '2021-12-31')
]

month_names = {1:'ENERO', 2:'FEBRERO', 3:'MARZO', 4:'ABRIL', 5:'MAYO', 6:'JUNIO',
               7:'JULIO', 8:'AGOSTO', 9:'SEPTIEMBRE', 10:'OCTUBRE', 11:'NOVIEMBRE', 12:'DICIEMBRE'}

print(f"\n{'='*120}")
print(f"  2021 COMPLETO - FAIR V3 (score eventos + ajuste precio) vs V2 (solo eventos)")
print(f"  SPY 2021: ~+27%  |  Año superalcista post-COVID")
print(f"{'='*120}")

monthly_v3 = {}
monthly_v2 = {}
cum_v3 = 0
cum_v2 = 0
all_configs_v3 = []
all_configs_v2 = []
week_details = []

for date in weeks_2021:
    month = date.month

    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    if not active:
        if month not in monthly_v3:
            monthly_v3[month] = 0
            monthly_v2[month] = 0
        all_configs_v3.append('0L+0S')
        all_configs_v2.append('0L+0S')
        week_details.append({'date': date, 'month': month, 'pnl_v3': 0, 'pnl_v2': 0,
                            'cfg_v3': '0L+0S', 'cfg_v2': '0L+0S', 'positions': [],
                            'bear_ratio': 0.5, 'events': 0})
        continue

    scores_evt = score_fair(active)

    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None

    scores_adj = adjust_score_by_price(scores_evt, dd_row, rsi_row)

    longs_v3, shorts_v3, br_v3 = decide_allocation(scores_adj)
    longs_v2, shorts_v2, br_v2 = decide_allocation(scores_evt)

    if date in atr_wide_lagged.index:
        atr_row = atr_wide_lagged.loc[date]
        shorts_v3 = [s for s in shorts_v3 if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]
        shorts_v2 = [s for s in shorts_v2 if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    ret_row = returns_wide.loc[date]

    # V3 P&L
    pnl_v3 = 0
    n = len(longs_v3) + len(shorts_v3)
    v3_pos = []
    if n > 0:
        lw = {s: scores_adj[s] - 5.0 for s in longs_v3}
        sw = {s: 5.0 - scores_adj[s] for s in shorts_v3}
        tw = sum(lw.values()) + sum(sw.values())
        if tw > 0:
            for s in longs_v3:
                if pd.notna(ret_row.get(s)):
                    w = lw[s] / tw
                    p = CAPITAL * w * ret_row[s]
                    pnl_v3 += p
                    v3_pos.append(('L', s, scores_evt[s], scores_adj[s], ret_row[s], w, p))
            for s in shorts_v3:
                if pd.notna(ret_row.get(s)):
                    w = sw[s] / tw
                    p = CAPITAL * w * (-ret_row[s])
                    pnl_v3 += p
                    v3_pos.append(('S', s, scores_evt[s], scores_adj[s], ret_row[s], w, p))

    # V2 P&L
    pnl_v2 = 0
    n2 = len(longs_v2) + len(shorts_v2)
    if n2 > 0:
        lw2 = {s: scores_evt[s] - 5.0 for s in longs_v2}
        sw2 = {s: 5.0 - scores_evt[s] for s in shorts_v2}
        tw2 = sum(lw2.values()) + sum(sw2.values())
        if tw2 > 0:
            for s in longs_v2:
                if pd.notna(ret_row.get(s)):
                    pnl_v2 += CAPITAL * (lw2[s] / tw2) * ret_row[s]
            for s in shorts_v2:
                if pd.notna(ret_row.get(s)):
                    pnl_v2 += CAPITAL * (sw2[s] / tw2) * (-ret_row[s])

    cum_v3 += pnl_v3
    cum_v2 += pnl_v2

    if month not in monthly_v3:
        monthly_v3[month] = 0
        monthly_v2[month] = 0
    monthly_v3[month] += pnl_v3
    monthly_v2[month] += pnl_v2

    cfg_v3 = f"{len(longs_v3)}L+{len(shorts_v3)}S"
    cfg_v2 = f"{len(longs_v2)}L+{len(shorts_v2)}S"
    all_configs_v3.append(cfg_v3)
    all_configs_v2.append(cfg_v2)

    week_details.append({'date': date, 'month': month, 'pnl_v3': pnl_v3, 'pnl_v2': pnl_v2,
                        'cfg_v3': cfg_v3, 'cfg_v2': cfg_v2, 'positions': v3_pos,
                        'bear_ratio': br_v3, 'events': len(active)})

# ---- Print mes a mes ----
cum_v3_p = 0
cum_v2_p = 0

for month in sorted(monthly_v3.keys()):
    mp_v3 = monthly_v3[month]
    mp_v2 = monthly_v2[month]
    cum_v3_p += mp_v3
    cum_v2_p += mp_v2

    first_day = pd.Timestamp(f'2021-{month:02d}-01')
    last_day = pd.Timestamp(f'2021-{month+1:02d}-01') - pd.Timedelta(days=1) if month < 12 else pd.Timestamp('2021-12-31')
    spy_m = spy[(spy.index >= first_day) & (spy.index <= last_day)]
    spy_chg = (spy_m.iloc[-1]['close'] / spy_m.iloc[0]['close'] - 1) * 100 if len(spy_m) >= 2 else 0

    print(f"\n{'='*120}")
    print(f"  {month_names[month]} 2021  |  V3: ${mp_v3:+,.0f}  V2: ${mp_v2:+,.0f}  Dif: ${mp_v3-mp_v2:+,.0f}  |  SPY: {spy_chg:+.1f}%")
    print(f"  Acumulado V3: ${cum_v3_p:+,.0f}  V2: ${cum_v2_p:+,.0f}")
    print(f"{'='*120}")

    month_weeks = [w for w in week_details if w['month'] == month]
    for w in month_weeks:
        ds = w['date'].strftime('%Y-%m-%d')
        evt_str = f"{w['events']}evt" if w['events'] > 0 else "sin eventos"
        print(f"\n  {ds}  {w['cfg_v3']:8s}  br={w['bear_ratio']:.2f}  {evt_str}  V3: ${w['pnl_v3']:+9,.0f}  V2: ${w['pnl_v2']:+9,.0f}")
        for side, s, sc_evt, sc_adj, ret, weight, pnl in w.get('positions', []):
            label = sub_labels.get(s, s)[:28]
            adj_str = f"({sc_evt:.1f}->{sc_adj:.1f})" if abs(sc_evt - sc_adj) > 0.01 else f"({sc_adj:.1f})"
            print(f"      {side} {label:28s}  {adj_str:>14s}  w={weight:.0%}  ret={ret*100:+6.2f}%  ${pnl:+,.0f}")

# ---- Resumen ----
print(f"\n\n{'='*120}")
print(f"  RESUMEN 2021 COMPLETO")
print(f"{'='*120}")

print(f"\n  {'Mes':12s} {'V3 (precio)':>12s} {'V2 (eventos)':>12s} {'Diferencia':>12s} {'SPY':>8s}")
print(f"  {'-'*60}")

cum_v3_s = 0
cum_v2_s = 0
for month in sorted(monthly_v3.keys()):
    cum_v3_s += monthly_v3[month]
    cum_v2_s += monthly_v2[month]

    first_day = pd.Timestamp(f'2021-{month:02d}-01')
    last_day = pd.Timestamp(f'2021-{month+1:02d}-01') - pd.Timedelta(days=1) if month < 12 else pd.Timestamp('2021-12-31')
    spy_m = spy[(spy.index >= first_day) & (spy.index <= last_day)]
    spy_chg = (spy_m.iloc[-1]['close'] / spy_m.iloc[0]['close'] - 1) * 100 if len(spy_m) >= 2 else 0

    print(f"  {month_names[month]:12s} ${monthly_v3[month]:+10,.0f} ${monthly_v2[month]:+10,.0f} ${monthly_v3[month]-monthly_v2[month]:+10,.0f}  {spy_chg:+.1f}%")

print(f"  {'-'*60}")
print(f"  {'TOTAL':12s} ${cum_v3_s:+10,.0f} ${cum_v2_s:+10,.0f} ${cum_v3_s-cum_v2_s:+10,.0f}")

spy_start = spy[spy.index >= '2021-01-04'].iloc[0]['close']
spy_end = spy[spy.index <= '2021-12-31'].iloc[-1]['close']
print(f"\n  SPY 2021: {(spy_end/spy_start-1)*100:+.1f}%")
print(f"  V3 (eventos + precio): ${cum_v3_s:+,.0f} ({cum_v3_s/CAPITAL*100:+.1f}%)")
print(f"  V2 (solo eventos):     ${cum_v2_s:+,.0f} ({cum_v2_s/CAPITAL*100:+.1f}%)")

# Config distribution
from collections import Counter
cc_v3 = Counter(all_configs_v3)
cc_v2 = Counter(all_configs_v2)
print(f"\n  Configs V3: {dict(sorted(cc_v3.items(), key=lambda x: -x[1]))}")
print(f"  Configs V2: {dict(sorted(cc_v2.items(), key=lambda x: -x[1]))}")

# Active weeks
active_v3 = sum(1 for w in week_details if w['pnl_v3'] != 0)
active_v2 = sum(1 for w in week_details if w['pnl_v2'] != 0)
print(f"  Semanas activas V3: {active_v3}  V2: {active_v2}  Total: {len(week_details)}")
