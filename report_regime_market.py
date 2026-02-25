"""
Reporte anual: Sistema MARKET con regimenes por ano
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0
CAPITAL = 500_000
ATR_MIN = 1.5
COST_PER_TRADE = 0.0010  # 0.10% por trade (conservador: comision + spread + slippage)

# ================================================================
# FUNCIONES
# ================================================================
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
        if oversold > 0.1 and score >= 4.0:
            long_cands.append((sub_id, oversold))
        rsi_ob = np.clip((rsi - 65) / 20, 0, 1)
        if rsi_ob > 0.1 and dd > -8 and score <= 6.0 and atr >= ATR_MIN:
            short_cands.append((sub_id, rsi_ob))
    long_cands.sort(key=lambda x: -x[1])
    short_cands.sort(key=lambda x: -x[1])
    longs = [c[0] for c in long_cands[:3]]
    shorts = [c[0] for c in short_cands[:2]]
    weights = {s: w for s, w in long_cands[:3]}
    weights.update({s: w for s, w in short_cands[:2]})
    return longs, shorts, weights

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

def classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0:
        return 'NEUTRAL', {}
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0:
        return 'NEUTRAL', {}
    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above50 = (rsi_row > 50).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) > 0:
        spy_last = spy_w.loc[spy_dates[-1]]
        spy_above_ma200 = spy_last.get('above_ma200', 0.5)
        spy_mom_10w = spy_last.get('mom_10w', 0)
        spy_dist = spy_last.get('dist_ma200', 0)
    else:
        spy_above_ma200 = 0.5
        spy_mom_10w = 0
        spy_dist = 0
    if not pd.notna(spy_mom_10w): spy_mom_10w = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scoring
    if pct_dd_healthy >= 75: score_bdd = 2.0
    elif pct_dd_healthy >= 60: score_bdd = 1.0
    elif pct_dd_healthy >= 45: score_bdd = 0.0
    elif pct_dd_healthy >= 30: score_bdd = -1.0
    else: score_bdd = -2.0

    if pct_rsi_above50 >= 75: score_brsi = 2.0
    elif pct_rsi_above50 >= 60: score_brsi = 1.0
    elif pct_rsi_above50 >= 45: score_brsi = 0.0
    elif pct_rsi_above50 >= 30: score_brsi = -1.0
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

    if total >= 4.0: regime = 'BULLISH'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -3.0: regime = 'BEARISH'
    else: regime = 'CRISIS'

    if vix_val >= 30 and regime == 'BULLISH':
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'BEARISH'

    return regime, {'pct_dd_healthy': pct_dd_healthy, 'pct_rsi_above50': pct_rsi_above50,
                    'pct_dd_deep': pct_dd_deep, 'vix': vix_val, 'score_total': total}

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
atr_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_atr')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide_lagged = atr_wide.shift(1)

# SPY
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

# VIX
vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

# Eventos
weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

# ================================================================
# BACKTEST
# ================================================================
print("Ejecutando backtest...")

yearly_pnl = {}
all_records = []

for date in returns_wide.index:
    if date.year < 2001:
        continue

    if date in weekly_events.index:
        evt_date = date
    else:
        nearest_idx = weekly_events.index.get_indexer([date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]

    events_row = weekly_events.loc[evt_date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    if not active:
        all_records.append({'date': date, 'pnl': 0, 'regime': 'NEUTRAL'})
        continue

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < date]
    dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)
    atr_row = atr_wide_lagged.loc[date] if date in atr_wide_lagged.index else None
    ret_row = returns_wide.loc[date]

    # Regimen de mercado
    regime, details = classify_regime_market(date, dd_wide, rsi_wide, spy_w, vix_df)

    # Allocation
    longs_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc > 6.5], key=lambda x: -x[1])
    shorts_pool = sorted([(s, sc) for s, sc in scores_v3.items() if sc < 3.5], key=lambda x: x[1])

    if regime == 'CRISIS':    nl, ns = 0, 3
    elif regime == 'BEARISH': nl, ns = 1, 3
    elif regime == 'NEUTRAL': nl, ns = 3, 3
    else:                     nl, ns = 3, 0

    longs = [s for s, _ in longs_pool[:nl]]
    shorts = [s for s, _ in shorts_pool[:ns]]

    if atr_row is not None:
        shorts = [s for s in shorts if pd.notna(atr_row.get(s)) and atr_row[s] >= ATR_MIN]

    if regime == 'NEUTRAL':
        longs, shorts, weights_mr = decide_neutral_meanrev(scores_evt, dd_row, rsi_row, atr_row)
        pnl_gross = calc_pnl_meanrev(longs, shorts, weights_mr, ret_row, CAPITAL)
    else:
        pnl_gross = calc_pnl(longs, shorts, scores_v3, ret_row, CAPITAL)

    # Costes: cada posicion paga ida + vuelta (0.10% x 2 = 0.20% del capital asignado)
    n_positions = len(longs) + len(shorts)
    if n_positions > 0:
        cost = CAPITAL * COST_PER_TRADE * 2 * n_positions / max(n_positions, 1)
        # cost = capital_total * 0.10% * 2 (ida+vuelta) = 0.20% de $500K = $1,000/semana
        cost = CAPITAL * COST_PER_TRADE * 2  # 0.20% sobre el capital total desplegado
    else:
        cost = 0
    pnl = pnl_gross - cost

    year = date.year
    yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
    all_records.append({'date': date, 'pnl': pnl, 'pnl_gross': pnl_gross, 'cost': cost, 'regime': regime, 'n_pos': n_positions})

df = pd.DataFrame(all_records)

# SPY annual
spy_annual = {}
for year in range(2001, 2026):
    spy_yr = spy_w[(spy_w.index.year == year) & spy_w['ret_spy'].notna()]
    if len(spy_yr) > 0:
        spy_annual[year] = ((1 + spy_yr['ret_spy']).prod() - 1) * 100

# ================================================================
# METRICAS GLOBALES
# ================================================================
total_pnl = df['pnl'].sum()
cum = df['pnl'].cumsum()
max_dd = (cum - cum.cummax()).min()
active = df[df['pnl'] != 0]
weekly_ret = active['pnl'] / CAPITAL
sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52) if weekly_ret.std() > 0 else 0
win_rate = (active['pnl'] > 0).mean() * 100
n_profitable = sum(1 for p in yearly_pnl.values() if p > 0)

# CAGR del sistema
n_years = (df['date'].max() - df['date'].min()).days / 365.25
cagr_sys = (1 + total_pnl / CAPITAL) ** (1 / n_years) - 1 if n_years > 0 else 0

# CAGR del SPY
spy_first = spy_w.loc[spy_w.index >= df['date'].min(), 'close'].iloc[0]
spy_last = spy_w.loc[spy_w.index <= df['date'].max(), 'close'].iloc[-1]
cagr_spy = (spy_last / spy_first) ** (1 / n_years) - 1

# Alpha anualizado
alpha_total = cagr_sys - cagr_spy

print("\n" + "=" * 115)
print("SISTEMA MARKET: REGIMEN BASADO EN DATOS REALES (breadth + SPY trend + VIX alerta)")
print("=" * 115)
print(f"\nP&L Total: ${total_pnl:,.0f} ({total_pnl/CAPITAL*100:.0f}%)  |  "
      f"CAGR: {cagr_sys*100:.1f}%  |  Sharpe: {sharpe:.2f}  |  Win Rate: {win_rate:.1f}%  |  "
      f"Max DD: ${max_dd:,.0f}  |  Anos +: {n_profitable}/{len(yearly_pnl)}")
print(f"SPY CAGR: {cagr_spy*100:.1f}%  |  Alpha anualizado: {alpha_total*100:+.1f}%")

# ================================================================
# TABLA ANUAL
# ================================================================
print("\n" + "=" * 115)
print("RESULTADOS ANUALES")
print("=" * 115)
print(f"\n{'Ano':>5} {'S&P500':>8} {'Sist%':>8} {'Alpha':>8} {'P&L Sist':>12} {'Acum':>14}  "
      f"{'BULL':>5} {'NEUT':>5} {'BEAR':>5} {'CRIS':>5}  {'Sharpe':>7}")
print("-" * 115)

acum = 0
sum_alpha = 0
for year in sorted(yearly_pnl.keys()):
    pnl = yearly_pnl[year]
    acum += pnl
    spy_r = spy_annual.get(year, 0)
    rent = pnl / CAPITAL * 100
    alpha = rent - spy_r
    sum_alpha += alpha

    yr_df = df[df['date'].dt.year == year]
    rc = yr_df['regime'].value_counts()
    nb = rc.get('BULLISH', 0)
    nn = rc.get('NEUTRAL', 0)
    nbr = rc.get('BEARISH', 0)
    nc = rc.get('CRISIS', 0)

    # Sharpe anual
    yr_active = yr_df[yr_df['pnl'] != 0]
    if len(yr_active) > 5:
        yr_ret = yr_active['pnl'] / CAPITAL
        yr_sharpe = yr_ret.mean() / yr_ret.std() * np.sqrt(52) if yr_ret.std() > 0 else 0
    else:
        yr_sharpe = 0

    # Marker
    if alpha > 30:
        marker = " **"
    elif alpha < -15:
        marker = " !!"
    else:
        marker = ""

    print(f"{year:>5} {spy_r:>7.1f}% {rent:>7.1f}% {alpha:>+7.1f}% ${pnl:>11,.0f} ${acum:>13,.0f}  "
          f"{nb:>5} {nn:>5} {nbr:>5} {nc:>5}  {yr_sharpe:>7.2f}{marker}")

avg_alpha = sum_alpha / len(yearly_pnl)
avg_rent = sum(yearly_pnl.values()) / len(yearly_pnl) / CAPITAL * 100
avg_spy = sum(spy_annual.get(y, 0) for y in yearly_pnl.keys()) / len(yearly_pnl)
alpha_positive = sum(1 for y in yearly_pnl.keys() if (yearly_pnl[y]/CAPITAL*100 - spy_annual.get(y, 0)) > 0)

print("-" * 115)
print(f"{'MEDIA':>5} {avg_spy:>7.1f}% {avg_rent:>7.1f}% {avg_alpha:>+7.1f}% ${sum(yearly_pnl.values())/len(yearly_pnl):>11,.0f}")
print(f"\nAlpha positivo: {alpha_positive}/{len(yearly_pnl)} anos ({alpha_positive/len(yearly_pnl)*100:.0f}%)")

# ================================================================
# ESTADISTICAS POR REGIMEN
# ================================================================
print("\n" + "=" * 100)
print("ESTADISTICAS POR REGIMEN")
print("=" * 100)

print(f"\n{'Regimen':>10} {'Semanas':>8} {'P&L Total':>12} {'Avg/sem':>10} {'WinRate':>8} {'Sharpe':>8}")
print("-" * 60)

for regime in ['BULLISH', 'NEUTRAL', 'BEARISH', 'CRISIS']:
    mask = df['regime'] == regime
    sub = df[mask]
    if len(sub) == 0:
        continue
    act = sub[sub['pnl'] != 0]
    total_r = sub['pnl'].sum()
    avg_r = act['pnl'].mean() if len(act) > 0 else 0
    wr = (act['pnl'] > 0).mean() * 100 if len(act) > 0 else 0
    ret_r = act['pnl'] / CAPITAL
    sh_r = ret_r.mean() / ret_r.std() * np.sqrt(52) if len(act) > 5 and ret_r.std() > 0 else 0

    print(f"{regime:>10} {len(sub):>8} ${total_r:>11,.0f} ${avg_r:>9,.0f} {wr:>7.1f}% {sh_r:>8.2f}")

# ================================================================
# RESUMEN DE COSTES
# ================================================================
print("\n" + "=" * 100)
print("DESGLOSE DE COSTES (CONSERVADOR: 0.10% por trade)")
print("=" * 100)

total_gross = df['pnl_gross'].sum()
total_costs = df['cost'].sum()
total_net = df['pnl'].sum()
weeks_active = (df['n_pos'] > 0).sum()
avg_cost_week = total_costs / weeks_active if weeks_active > 0 else 0
avg_positions = df.loc[df['n_pos'] > 0, 'n_pos'].mean()

print(f"\n  Modelo de costes: {COST_PER_TRADE*100:.2f}% por trade (ida) + {COST_PER_TRADE*100:.2f}% (vuelta) = {COST_PER_TRADE*2*100:.2f}% round-trip")
print(f"  Posiciones medias/semana: {avg_positions:.1f}")
print(f"  Semanas activas: {weeks_active}")
print(f"  Coste medio/semana: ${avg_cost_week:,.0f}")
print(f"\n  P&L Bruto:  ${total_gross:>12,.0f}")
print(f"  Costes:     ${total_costs:>12,.0f} ({total_costs/total_gross*100:.1f}% del bruto)")
print(f"  P&L Neto:   ${total_net:>12,.0f}")
print(f"  Impacto:    {(total_net/total_gross - 1)*100:+.1f}% sobre P&L bruto")
