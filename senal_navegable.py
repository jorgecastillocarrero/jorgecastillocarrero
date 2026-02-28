"""
Genera HTML navegable con señales de TODAS las semanas.
Selector interactivo: el usuario elige la semana y ve régimen, subsectores, picks.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0

REGIME_ALLOC = {
    'BURBUJA': (10, 0), 'GOLDILOCKS': (7, 3), 'ALCISTA': (7, 3),
    'NEUTRAL': (5, 5), 'CAUTIOUS': (5, 5),
    'BEARISH': (3, 7), 'CRISIS': (0, 10), 'PANICO': (0, 10),
    'CAPITULACION': (10, 0), 'RECOVERY': (7, 3),
}

REGIME_COLORS = {
    'BURBUJA': '#e91e63', 'GOLDILOCKS': '#4caf50', 'ALCISTA': '#2196f3',
    'NEUTRAL': '#ff9800', 'CAUTIOUS': '#ff5722', 'BEARISH': '#795548',
    'CRISIS': '#9c27b0', 'PANICO': '#f44336', 'CAPITULACION': '#00bcd4',
    'RECOVERY': '#8bc34a',
}

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
print("Cargando datos base...")
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

df_all = pd.read_sql(f"""
    SELECT symbol, date, open, close, high, low
    FROM fmp_price_history WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-12-31' ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])

df_weekly = df_all.sort_values('date').groupby(['symbol', pd.Grouper(key='date', freq='W-FRI')]).last().reset_index()
df_weekly = df_weekly.dropna(subset=['close'])
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])
df_weekly['hl_range'] = (df_weekly['high'] - df_weekly['low']) / df_weekly['close']
df_weekly['atr_pct'] = df_weekly.groupby('symbol')['hl_range'].transform(
    lambda x: x.rolling(5, min_periods=3).mean() * 100)

def calc_stock_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['high'].rolling(52, min_periods=26).max()
    g['dd_52w'] = (g['close'] / g['high_52w'] - 1) * 100
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    g['mom_12w'] = g['close'].pct_change(12) * 100
    return g

print("Calculando metricas acciones...")
df_weekly = df_weekly.groupby('symbol', group_keys=False).apply(calc_stock_metrics)

sub_agg = df_all.sort_values('date').groupby(['subsector', pd.Grouper(key='date', freq='W-FRI')]).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'), avg_low=('low', 'mean')
).reset_index().dropna(subset=['avg_close'])
sub_agg = sub_agg.sort_values(['subsector', 'date'])

sub_weekly_ret = df_weekly.groupby(['subsector', 'date']).agg(avg_atr=('atr_pct', 'mean')).reset_index()
sub_agg = sub_agg.merge(sub_weekly_ret, on=['subsector', 'date'], how='left')

def calc_sub_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

print("Calculando metricas subsectores...")
sub_agg = sub_agg.groupby('subsector', group_keys=False).apply(calc_sub_metrics)
dd_wide = sub_agg.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_agg.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide = sub_agg.pivot(index='date', columns='subsector', values='avg_atr')

# --- Retornos Vie open -> Vie open por subsector ---
print("Calculando retornos subsectores (Fri open -> Fri open)...")
# Calcular open promedio por subsector y viernes
df_all_daily = df_all[['symbol', 'date', 'open', 'subsector']].dropna(subset=['open', 'subsector']).copy()
df_all_daily = df_all_daily.set_index('date').sort_index()

# Para cada viernes del resample, buscar el open del dia (o siguiente habil)
sub_fri_open = df_all_daily.groupby([pd.Grouper(freq='W-FRI'), 'subsector'])['open'].mean()
sub_fri_open = sub_fri_open.reset_index()
sub_fri_open.columns = ['date', 'subsector', 'avg_open']
sub_fri_open = sub_fri_open.sort_values(['subsector', 'date'])
sub_fri_open['next_open'] = sub_fri_open.groupby('subsector')['avg_open'].shift(-1)
sub_fri_open['ret_pct'] = (sub_fri_open['next_open'] / sub_fri_open['avg_open'] - 1) * 100
sub_ret_wide = sub_fri_open.pivot(index='date', columns='subsector', values='ret_pct')
print(f"  Subsector returns: {sub_ret_wide.shape[0]} semanas x {sub_ret_wide.shape[1]} subsectores")

weekly_events = build_weekly_events('2000-01-01', '2026-12-31')

# CSV regimenes
reg_df = pd.read_csv('data/regimenes_historico.csv')
reg_df['fecha_senal'] = pd.to_datetime(reg_df['fecha_senal'])

# ================================================================
# Calcular señal para cada semana del CSV
# ================================================================
print("Generando datos por semana...")
all_weeks_data = []
prev_ranks = {}  # sub_id -> rank (1-based) de la semana anterior

for _, reg_row in reg_df.iterrows():
    signal_date = reg_row['fecha_senal']
    if signal_date not in dd_wide.index:
        continue

    regime = reg_row['regime']
    total = float(reg_row['total'])
    s_bdd = float(reg_row['s_bdd'])
    s_brsi = float(reg_row['s_brsi'])
    s_ddp = float(reg_row['s_ddp'])
    s_spy = float(reg_row['s_spy'])
    s_mom = float(reg_row['s_mom'])
    spy_close = float(reg_row['spy_close'])
    spy_ma200 = float(reg_row['spy_ma200'])
    spy_dist = float(reg_row['spy_dist'])
    spy_mom = float(reg_row['spy_mom'])
    vix_val = float(reg_row['vix'])
    pct_dd_h = float(reg_row['pct_dd_h'])
    pct_dd_d = float(reg_row['pct_dd_d'])
    pct_rsi = float(reg_row['pct_rsi'])
    n_sub = int(reg_row['n_sub'])
    spy_ret = float(reg_row['spy_ret_pct']) if pd.notna(reg_row['spy_ret_pct']) else None

    year = int(reg_row['year'])
    sem = int(reg_row['sem'])

    # Fair Value
    dd_row_cur = dd_wide.loc[signal_date]
    rsi_row_cur = rsi_wide.loc[signal_date]

    if signal_date in weekly_events.index:
        evt_date = signal_date
    else:
        nearest_idx = weekly_events.index.get_indexer([signal_date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: float(events_row[col]) for col in events_row.index if events_row[col] > 0}

    scores_evt = score_fair(active)

    prev_dates = dd_wide.index[dd_wide.index < signal_date]
    dd_row_prev = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row_prev = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    atr_row = atr_wide.loc[signal_date] if signal_date in atr_wide.index else None
    scores_adj = adjust_score_by_price(scores_evt, dd_row_prev, rsi_row_prev)

    # Subsector data + returns
    sub_ret_row = sub_ret_wide.loc[signal_date] if signal_date in sub_ret_wide.index else None
    subs = []
    for sub_id in sorted(SUBSECTORS.keys()):
        fv_raw = scores_evt.get(sub_id, 5.0)
        fv_adj = scores_adj.get(sub_id, 5.0)
        dd = float(dd_row_cur.get(sub_id, 0)) if pd.notna(dd_row_cur.get(sub_id, 0)) else 0
        rsi = float(rsi_row_cur.get(sub_id, 50)) if pd.notna(rsi_row_cur.get(sub_id, 50)) else 50
        atr = float(atr_row.get(sub_id, 0)) if atr_row is not None and pd.notna(atr_row.get(sub_id, 0)) else 0
        ret = None
        if sub_ret_row is not None and sub_id in sub_ret_row.index and pd.notna(sub_ret_row[sub_id]):
            ret = round(float(sub_ret_row[sub_id]), 2)
        subs.append({
            'id': sub_id, 'name': SUBSECTORS[sub_id]['label'],
            'etf': SUBSECTORS[sub_id].get('etf', ''),
            'n': len(SUBSECTORS[sub_id]['tickers']),
            'fv': round(fv_raw, 2), 'fva': round(fv_adj, 2),
            'dd': round(dd, 1), 'rsi': round(rsi, 0), 'atr': round(atr, 1),
            'ret': ret,
        })
    subs.sort(key=lambda x: -x['fva'])

    # Calcular cambio de posicion vs semana anterior
    cur_ranks = {}
    for rank, s in enumerate(subs, 1):
        cur_ranks[s['id']] = rank
        if s['id'] in prev_ranks:
            s['chg'] = prev_ranks[s['id']] - rank  # positivo = subio
        else:
            s['chg'] = None
    prev_ranks = cur_ranks

    # Picks
    n_long, n_short = REGIME_ALLOC.get(regime, (5, 5))
    stock_data = df_weekly[df_weekly['date'] <= signal_date].sort_values(['symbol', 'date'])
    latest_stocks = stock_data.groupby('symbol').last().reset_index()

    long_cands = []
    good_subs = {s['id']: s['fva'] for s in subs if s['fva'] > 5.5}
    for _, stock in latest_stocks.iterrows():
        sub = ticker_to_sub.get(stock['symbol'])
        if sub not in good_subs: continue
        fv = good_subs[sub]
        mom = float(stock.get('mom_12w', 0)) if pd.notna(stock.get('mom_12w', 0)) else 0
        rsi = float(stock.get('rsi_14w', 50)) if pd.notna(stock.get('rsi_14w', 50)) else 50
        dd = float(stock.get('dd_52w', 0)) if pd.notna(stock.get('dd_52w', 0)) else 0
        price = float(stock.get('close', 0))
        composite = (np.clip((fv - 5.0) / 4.0, 0, 1) * 3.0 +
                     np.clip((mom + 20) / 60, 0, 1) * 3.0 +
                     np.clip((rsi - 30) / 50, 0, 1) * 2.0 +
                     np.clip((dd + 30) / 30, 0, 1) * 2.0)
        long_cands.append({
            't': stock['symbol'], 'sub': SUBSECTORS[sub]['label'],
            'fv': round(fv, 1), 'mom': round(mom, 1), 'rsi': round(rsi, 0),
            'dd': round(dd, 1), 'p': round(price, 2), 'sc': round(composite, 1),
        })

    short_cands = []
    bad_subs = {s['id']: s['fva'] for s in subs if s['fva'] < 4.5}
    for _, stock in latest_stocks.iterrows():
        sub = ticker_to_sub.get(stock['symbol'])
        if sub not in bad_subs: continue
        fv = bad_subs[sub]
        mom = float(stock.get('mom_12w', 0)) if pd.notna(stock.get('mom_12w', 0)) else 0
        rsi = float(stock.get('rsi_14w', 50)) if pd.notna(stock.get('rsi_14w', 50)) else 50
        dd = float(stock.get('dd_52w', 0)) if pd.notna(stock.get('dd_52w', 0)) else 0
        atr = float(stock.get('atr_pct', 0)) if pd.notna(stock.get('atr_pct', 0)) else 0
        price = float(stock.get('close', 0))
        if dd < -40 or rsi < 15 or atr < 1.5: continue
        composite = (np.clip((5.0 - fv) / 4.0, 0, 1) * 3.0 +
                     np.clip((-mom + 20) / 60, 0, 1) * 3.0 +
                     np.clip((70 - rsi) / 50, 0, 1) * 2.0 +
                     np.clip((atr - 1.5) / 5.0, 0, 1) * 2.0)
        short_cands.append({
            't': stock['symbol'], 'sub': SUBSECTORS[sub]['label'],
            'fv': round(fv, 1), 'mom': round(mom, 1), 'rsi': round(rsi, 0),
            'dd': round(dd, 1), 'atr': round(atr, 1), 'p': round(price, 2), 'sc': round(composite, 1),
        })

    long_cands.sort(key=lambda x: -x['sc'])
    short_cands.sort(key=lambda x: -x['sc'])

    # Diversified picks
    def div_pick(cands, n, max_per=2):
        picked = []; sub_cnt = {}
        for c in cands:
            s = c['sub']
            if sub_cnt.get(s, 0) >= max_per: continue
            picked.append(c); sub_cnt[s] = sub_cnt.get(s, 0) + 1
            if len(picked) >= n: break
        return picked

    longs_sel = div_pick(long_cands, n_long)
    shorts_sel = div_pick(short_cands, n_short)

    # Events
    events_list = [{'name': k, 'intensity': round(v, 1)} for k, v in sorted(active.items(), key=lambda x: -x[1])]

    week_key = f"{year}-W{sem:02d}"
    all_weeks_data.append({
        'key': week_key,
        'date': signal_date.strftime('%Y-%m-%d'),
        'year': year, 'sem': sem,
        'regime': regime, 'total': total,
        'scores': [s_bdd, s_brsi, s_ddp, s_spy, s_mom],
        'spy': round(spy_close, 2), 'ma200': round(spy_ma200, 2),
        'dist': round(spy_dist, 1), 'mom': round(spy_mom, 1),
        'vix': round(vix_val, 1),
        'pct_dd_h': round(pct_dd_h, 1), 'pct_dd_d': round(pct_dd_d, 1),
        'pct_rsi': round(pct_rsi, 1), 'n_sub': n_sub,
        'spy_ret': round(spy_ret, 2) if spy_ret is not None else None,
        'n_long': n_long, 'n_short': n_short,
        'subs': subs,
        'events': events_list,
        'longs': long_cands[:20],
        'shorts': short_cands[:20],
        'longs_sel': longs_sel,
        'shorts_sel': shorts_sel,
    })

print(f"  Total semanas procesadas: {len(all_weeks_data)}")

# ================================================================
# Calcular estadisticas por posicion y regimen
# ================================================================
print("Calculando estadisticas por posicion y regimen...")
rank_regime_data = {}
for week in all_weeks_data:
    regime = week['regime']
    if regime not in rank_regime_data:
        rank_regime_data[regime] = {}
    for rank_idx, sub in enumerate(week['subs'], 1):
        if rank_idx not in rank_regime_data[regime]:
            rank_regime_data[regime][rank_idx] = []
        if sub['ret'] is not None:
            rank_regime_data[regime][rank_idx].append(sub['ret'])

rank_stats_out = {}
for regime in rank_regime_data:
    rank_stats_out[regime] = {}
    for rank_idx in sorted(rank_regime_data[regime].keys()):
        rets = rank_regime_data[regime][rank_idx]
        if len(rets) > 0:
            rank_stats_out[regime][str(rank_idx)] = {
                'avg': round(float(np.mean(rets)), 3),
                'n': len(rets),
                'wr': round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                'total': round(float(np.sum(rets)), 1),
            }
print(f"  Regimenes con stats: {list(rank_stats_out.keys())}")

# ================================================================
# Generar HTML
# ================================================================
print("Generando HTML...")

regime_colors_json = json.dumps(REGIME_COLORS)
weeks_json = json.dumps(all_weeks_data)
rank_stats_json = json.dumps(rank_stats_out)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Senales Semanales - Navegable</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #fff; color: #222; margin: 20px; }}
h1 {{ color: #1565c0; text-align: center; margin-bottom: 5px; }}
h2 {{ color: #333; margin-top: 25px; margin-bottom: 10px; border-bottom: 2px solid #1565c0; padding-bottom: 5px; }}
h3 {{ color: #555; margin-top: 15px; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 15px; font-size: 14px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 15px; font-size: 12px; }}
th {{ background: #1565c0; color: #fff; padding: 6px 5px; text-align: center; border: 1px solid #ccc; cursor: pointer; }}
th:hover {{ background: #0d47a1; }}
td {{ padding: 5px; text-align: center; border: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #f5f7fa; }}
tr:hover {{ background: #e3f2fd; }}
.pos {{ color: #2e7d32; font-weight: bold; }}
.neg {{ color: #c62828; font-weight: bold; }}
.neutral {{ color: #999; }}
td.left {{ text-align: left; }}
.summary-box {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 15px; }}
.summary-card {{ background: #f5f7fa; border: 1px solid #ddd; border-radius: 8px; padding: 12px; flex: 1; min-width: 120px; text-align: center; }}
.summary-card h4 {{ margin: 0 0 5px 0; color: #1565c0; font-size: 11px; }}
.summary-card .value {{ font-size: 20px; font-weight: bold; }}
.regime-badge {{ display: inline-block; padding: 5px 18px; border-radius: 5px; font-size: 18px; font-weight: bold; color: #fff; }}
.note {{ background: #fffde7; padding: 10px 15px; border-radius: 8px; border-left: 4px solid #ffd600; margin-bottom: 15px; font-size: 13px; }}
.indicator-table {{ width: auto !important; margin: 0 auto 15px auto !important; min-width: 700px; }}
.indicator-table td {{ padding: 7px 10px; }}
.indicator-table th {{ background: #455a64; }}
.pick-long {{ background: #e8f5e9 !important; }}
.pick-long:hover {{ background: #c8e6c9 !important; }}
.pick-short {{ background: #ffebee !important; }}
.pick-short:hover {{ background: #ffcdd2 !important; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }}
.badge-long {{ background: #2e7d32; color: #fff; }}
.badge-short {{ background: #c62828; color: #fff; }}
.fv-high {{ background: #e8f5e9; }}
.fv-low {{ background: #ffebee; }}

/* Navigator */
.nav-bar {{ text-align: center; margin: 15px 0 20px 0; padding: 15px; background: #f5f7fa; border-radius: 10px; border: 1px solid #ddd; }}
.nav-bar select {{ font-size: 16px; padding: 8px 15px; border-radius: 5px; border: 1px solid #ccc; margin: 0 10px; }}
.nav-bar button {{ font-size: 14px; padding: 8px 20px; border-radius: 5px; border: 1px solid #1565c0; background: #1565c0; color: #fff; cursor: pointer; margin: 0 5px; }}
.nav-bar button:hover {{ background: #0d47a1; }}
.nav-bar .year-btns {{ margin-top: 10px; }}
.nav-bar .year-btn {{ font-size: 12px; padding: 4px 10px; border-radius: 3px; border: 1px solid #999; background: #fff; color: #333; cursor: pointer; margin: 2px; }}
.nav-bar .year-btn:hover, .nav-bar .year-btn.active {{ background: #1565c0; color: #fff; border-color: #1565c0; }}
.ret-badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 10px; }}
.chg-up {{ color: #2e7d32; font-weight: bold; font-size: 11px; }}
.chg-down {{ color: #c62828; font-weight: bold; font-size: 11px; }}
.chg-same {{ color: #bbb; font-size: 11px; }}
.rank-stats-table td {{ padding: 6px 8px; font-size: 12px; }}
.rank-stats-table th {{ font-size: 11px; padding: 5px 6px; white-space: nowrap; }}
</style>
</head>
<body>

<h1>Senales Semanales - Navegador</h1>
<p class="subtitle">Senal: jueves cierre &rarr; jueves cierre | Trading: viernes apertura &rarr; viernes apertura</p>

<div class="nav-bar">
    <button onclick="prevWeek()">&larr; Anterior</button>
    <select id="weekSelect" onchange="loadWeek()"></select>
    <button onclick="nextWeek()">Siguiente &rarr;</button>
    <div class="year-btns" id="yearBtns"></div>
    <div style="margin-top:8px;" id="regimeBtns"></div>
</div>

<div id="content"></div>

<script>
const WEEKS = {weeks_json};
const REGIME_COLORS = {regime_colors_json};
const RANK_STATS = {rank_stats_json};

const sel = document.getElementById('weekSelect');
const yearBtns = document.getElementById('yearBtns');

// Populate selector
WEEKS.forEach((w, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    const retStr = w.spy_ret !== null ? (w.spy_ret >= 0 ? '+' : '') + w.spy_ret.toFixed(2) + '%' : '-';
    opt.text = w.year + ' S' + w.sem + ' (' + w.date + ') - ' + w.regime + ' ' + retStr;
    sel.appendChild(opt);
}});

// Year buttons
const years = [...new Set(WEEKS.map(w => w.year))].sort();
years.forEach(y => {{
    const btn = document.createElement('button');
    btn.className = 'year-btn';
    btn.textContent = y;
    btn.onclick = () => filterByYear(y);
    yearBtns.appendChild(btn);
}});

// Regime buttons
const regimeBtns = document.getElementById('regimeBtns');
const regimeOrder = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION'];
regimeOrder.forEach(r => {{
    const btn = document.createElement('button');
    btn.className = 'year-btn';
    btn.style.background = REGIME_COLORS[r];
    btn.style.color = '#fff';
    btn.style.borderColor = REGIME_COLORS[r];
    btn.textContent = r;
    btn.onclick = () => filterByRegime(r);
    regimeBtns.appendChild(btn);
}});
const allBtn = document.createElement('button');
allBtn.className = 'year-btn';
allBtn.textContent = 'TODOS';
allBtn.onclick = () => resetFilter();
regimeBtns.insertBefore(allBtn, regimeBtns.firstChild);

let activeFilter = null; // null = all, or {{type: 'year'|'regime', value: X}}
let filteredIndices = WEEKS.map((_, i) => i);

function applyFilter() {{
    sel.innerHTML = '';
    filteredIndices.forEach(i => {{
        const w = WEEKS[i];
        const opt = document.createElement('option');
        opt.value = i;
        const retStr = w.spy_ret !== null ? (w.spy_ret >= 0 ? '+' : '') + w.spy_ret.toFixed(2) + '%' : '-';
        opt.text = w.year + ' S' + w.sem + ' (' + w.date + ') - ' + w.regime + ' ' + retStr;
        sel.appendChild(opt);
    }});
    if (filteredIndices.length > 0) {{
        sel.value = filteredIndices[filteredIndices.length - 1];
        loadWeek();
    }}
    // Highlight buttons
    document.querySelectorAll('#yearBtns .year-btn').forEach(b => {{
        b.classList.toggle('active', activeFilter && activeFilter.type === 'year' && parseInt(b.textContent) === activeFilter.value);
    }});
    document.querySelectorAll('#regimeBtns .year-btn').forEach(b => {{
        if (b.textContent === 'TODOS') {{
            b.classList.toggle('active', !activeFilter);
        }} else {{
            b.classList.toggle('active', activeFilter && activeFilter.type === 'regime' && b.textContent === activeFilter.value);
        }}
    }});
}}

function filterByYear(y) {{
    activeFilter = {{type: 'year', value: y}};
    filteredIndices = WEEKS.map((w, i) => w.year === y ? i : -1).filter(i => i >= 0);
    applyFilter();
}}

function filterByRegime(r) {{
    activeFilter = {{type: 'regime', value: r}};
    filteredIndices = WEEKS.map((w, i) => w.regime === r ? i : -1).filter(i => i >= 0);
    applyFilter();
}}

function resetFilter() {{
    activeFilter = null;
    filteredIndices = WEEKS.map((_, i) => i);
    applyFilter();
}}

// Start at last week
sel.value = WEEKS.length - 1;
loadWeek();

function prevWeek() {{
    const curIdx = filteredIndices.indexOf(parseInt(sel.value));
    if (curIdx > 0) {{ sel.value = filteredIndices[curIdx - 1]; loadWeek(); }}
}}
function nextWeek() {{
    const curIdx = filteredIndices.indexOf(parseInt(sel.value));
    if (curIdx < filteredIndices.length - 1) {{ sel.value = filteredIndices[curIdx + 1]; loadWeek(); }}
}}

function vc(v) {{ return v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral'; }}
function fmt(v, decimals=1) {{ return (v >= 0 ? '+' : '') + v.toFixed(decimals); }}

function fvBar(v) {{
    const pct = v / 10 * 100;
    let color = v >= 7 ? '#2e7d32' : v >= 5.5 ? '#1565c0' : v >= 4.5 ? '#ff9800' : '#c62828';
    return '<div style="background:#eee;border-radius:3px;height:14px;width:80px;display:inline-block;vertical-align:middle;">' +
           '<div style="background:' + color + ';height:14px;border-radius:3px;width:' + pct.toFixed(0) + 'px;"></div></div> <b>' + v.toFixed(2) + '</b>';
}}

function loadWeek() {{
    const w = WEEKS[sel.value];
    const rc = REGIME_COLORS[w.regime] || '#666';
    const retHtml = w.spy_ret !== null
        ? '<span class="ret-badge" style="background:' + (w.spy_ret >= 0 ? '#e8f5e9;color:#2e7d32' : '#ffebee;color:#c62828') + ';">SPY: ' + fmt(w.spy_ret, 2) + '%</span>'
        : '<span class="ret-badge" style="background:#f5f5f5;color:#999;">SPY: pendiente</span>';

    const labels = ['BDD - Breadth Drawdown', 'BRSI - Breadth RSI', 'DDP - Deep Drawdown %', 'SPY - vs MA200', 'MOM - Momentum 10w'];
    const ranges = ['-3.0 a +2.0', '-3.0 a +2.0', '-2.5 a +1.5', '-2.5 a +1.5', '-1.5 a +1.0'];
    const vals = [w.pct_dd_h+'%', w.pct_rsi+'%', w.pct_dd_d+'%', fmt(w.dist)+'%', fmt(w.mom)+'%'];

    let h = '<h2>1. Regimen de Mercado</h2>';
    h += '<div style="text-align:center;margin:15px 0;">';
    h += '<span class="regime-badge" style="background:'+rc+';">'+w.regime+'</span>';
    h += ' <span style="font-size:20px;margin-left:15px;font-weight:bold;">Score: '+fmt(w.total)+'</span>';
    h += retHtml + '</div>';

    h += '<div class="summary-box">';
    h += '<div class="summary-card"><h4>SPY</h4><div class="value">$'+w.spy+'</div></div>';
    h += '<div class="summary-card"><h4>MA200</h4><div class="value">$'+w.ma200+'</div></div>';
    h += '<div class="summary-card"><h4>Dist MA200</h4><div class="value '+vc(w.dist)+'">'+fmt(w.dist)+'%</div></div>';
    h += '<div class="summary-card"><h4>Mom 10w</h4><div class="value '+vc(w.mom)+'">'+fmt(w.mom)+'%</div></div>';
    h += '<div class="summary-card"><h4>VIX</h4><div class="value">'+w.vix+'</div></div>';
    h += '</div>';

    h += '<table class="indicator-table"><tr><th>Indicador</th><th>Valor</th><th>Score</th><th>Rango</th></tr>';
    for (let i = 0; i < 5; i++) {{
        h += '<tr><td class="left"><b>'+labels[i]+'</b></td><td>'+vals[i]+'</td>';
        h += '<td class="'+vc(w.scores[i])+'"><b>'+fmt(w.scores[i])+'</b></td><td>'+ranges[i]+'</td></tr>';
    }}
    h += '<tr style="background:#e3f2fd;"><td class="left"><b>TOTAL</b></td><td></td>';
    h += '<td><b style="font-size:16px;">'+fmt(w.total)+'</b></td><td>-12.5 a +8.0</td></tr></table>';

    h += '<div class="note"><b>Subsectores:</b> '+w.n_sub+' | <b>DD sanos (&gt;-10%):</b> '+w.pct_dd_h+'% | <b>DD profundos (&lt;-20%):</b> '+w.pct_dd_d+'% | <b>RSI &gt;55:</b> '+w.pct_rsi+'%</div>';

    // Events
    if (w.events.length > 0) {{
        h += '<h2>2. Eventos Activos ('+w.events.length+')</h2><table style="width:auto;min-width:400px;"><tr><th>Evento</th><th>Intensidad</th></tr>';
        w.events.forEach(e => {{
            const bw = Math.round(e.intensity / 2.0 * 100);
            const ec = e.intensity >= 1.5 ? '#2e7d32' : e.intensity >= 1.0 ? '#1565c0' : '#ff9800';
            h += '<tr><td class="left">'+e.name+'</td><td><div style="background:#eee;border-radius:3px;height:14px;width:120px;display:inline-block;vertical-align:middle;"><div style="background:'+ec+';height:14px;border-radius:3px;width:'+bw+'px;"></div></div> <b>'+e.intensity+'</b></td></tr>';
        }});
        h += '</table>';
    }}

    // Subsectors
    h += '<h2>3. Fair Value Subsectores ('+w.subs.length+')</h2>';
    h += '<table id="subTable"><tr><th>#</th><th>&#x25B2;&#x25BC;</th><th>Subsector</th><th>Tickers</th><th>FV Raw</th><th>FV Ajustado</th><th>DD 52w</th><th>RSI 14w</th><th>ATR%</th><th onclick="sortSubTable()" style="cursor:pointer;">Ret %  &#x25B2;&#x25BC;</th></tr>';
    w.subs.forEach((s, i) => {{
        const rc2 = s.fva >= 5.5 ? ' class="fv-high"' : s.fva < 4.5 ? ' class="fv-low"' : '';
        const retStr = s.ret !== null ? '<span class="'+vc(s.ret)+'">'+fmt(s.ret, 2)+'%</span>' : '<span class="neutral">-</span>';
        let chgStr = '<span class="chg-same">-</span>';
        if (s.chg !== null && s.chg !== undefined) {{
            if (s.chg > 0) chgStr = '<span class="chg-up">&#x25B2;'+s.chg+'</span>';
            else if (s.chg < 0) chgStr = '<span class="chg-down">&#x25BC;'+Math.abs(s.chg)+'</span>';
            else chgStr = '<span class="chg-same">=</span>';
        }}
        h += '<tr'+rc2+' data-ret="'+(s.ret !== null ? s.ret : -999)+'"><td>'+(i+1)+'</td><td>'+chgStr+'</td><td class="left"><b>'+s.name+'</b></td><td>'+s.n+'</td>';
        h += '<td>'+s.fv.toFixed(2)+'</td><td>'+fvBar(s.fva)+'</td>';
        h += '<td class="'+vc(-Math.abs(s.dd))+'">'+fmt(s.dd)+'%</td><td>'+s.rsi+'</td><td>'+s.atr+'%</td>';
        h += '<td>'+retStr+'</td></tr>';
    }});
    h += '</table>';

    // Picks
    h += '<h2>4. Picks: '+w.n_long+'L + '+w.n_short+'S ('+w.regime+')</h2>';

    if (w.longs_sel.length > 0) {{
        h += '<h3>LONGS Seleccionados ('+w.longs_sel.length+')</h3>';
        h += '<table><tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV</th><th>Mom 12w</th><th>RSI</th><th>DD 52w</th><th>Score</th></tr>';
        w.longs_sel.forEach((c, i) => {{
            h += '<tr class="pick-long"><td>'+(i+1)+'</td><td><b>'+c.t+'</b></td><td>$'+c.p+'</td>';
            h += '<td class="left">'+c.sub+'</td><td><b>'+c.fv+'</b></td>';
            h += '<td class="'+vc(c.mom)+'">'+fmt(c.mom)+'%</td><td>'+c.rsi+'</td>';
            h += '<td class="'+vc(c.dd)+'">'+fmt(c.dd)+'%</td><td><b>'+c.sc+'</b></td></tr>';
        }});
        h += '</table>';
    }}

    if (w.longs.length > 0) {{
        h += '<details><summary style="cursor:pointer;color:#1565c0;font-weight:bold;margin:8px 0;">Top 20 candidatos LONG</summary><table>';
        h += '<tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV</th><th>Mom</th><th>RSI</th><th>DD</th><th>Score</th></tr>';
        w.longs.forEach((c, i) => {{
            const isSel = w.longs_sel.some(s => s.t === c.t);
            h += '<tr'+(isSel ? ' class="pick-long"' : '')+'><td>'+(i+1)+'</td><td><b>'+c.t+'</b></td><td>$'+c.p+'</td>';
            h += '<td class="left">'+c.sub+'</td><td>'+c.fv+'</td>';
            h += '<td class="'+vc(c.mom)+'">'+fmt(c.mom)+'%</td><td>'+c.rsi+'</td>';
            h += '<td class="'+vc(c.dd)+'">'+fmt(c.dd)+'%</td><td>'+c.sc+'</td></tr>';
        }});
        h += '</table></details>';
    }}

    if (w.shorts_sel.length > 0) {{
        h += '<h3>SHORTS Seleccionados ('+w.shorts_sel.length+')</h3>';
        h += '<table><tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV</th><th>Mom</th><th>RSI</th><th>DD</th><th>ATR%</th><th>Score</th></tr>';
        w.shorts_sel.forEach((c, i) => {{
            h += '<tr class="pick-short"><td>'+(i+1)+'</td><td><b>'+c.t+'</b></td><td>$'+c.p+'</td>';
            h += '<td class="left">'+c.sub+'</td><td><b>'+c.fv+'</b></td>';
            h += '<td class="'+vc(c.mom)+'">'+fmt(c.mom)+'%</td><td>'+c.rsi+'</td>';
            h += '<td class="'+vc(c.dd)+'">'+fmt(c.dd)+'%</td><td>'+c.atr+'%</td><td><b>'+c.sc+'</b></td></tr>';
        }});
        h += '</table>';
    }}

    if (w.shorts.length > 0) {{
        h += '<details><summary style="cursor:pointer;color:#c62828;font-weight:bold;margin:8px 0;">Top 20 candidatos SHORT</summary><table>';
        h += '<tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV</th><th>Mom</th><th>RSI</th><th>DD</th><th>ATR%</th><th>Score</th></tr>';
        w.shorts.forEach((c, i) => {{
            const isSel = w.shorts_sel.some(s => s.t === c.t);
            h += '<tr'+(isSel ? ' class="pick-short"' : '')+'><td>'+(i+1)+'</td><td><b>'+c.t+'</b></td><td>$'+c.p+'</td>';
            h += '<td class="left">'+c.sub+'</td><td>'+c.fv+'</td>';
            h += '<td class="'+vc(c.mom)+'">'+fmt(c.mom)+'%</td><td>'+c.rsi+'</td>';
            h += '<td class="'+vc(c.dd)+'">'+fmt(c.dd)+'%</td><td>'+(c.atr||'-')+'%</td><td>'+c.sc+'</td></tr>';
        }});
        h += '</table></details>';
    }}

    if (w.longs_sel.length === 0 && w.shorts_sel.length === 0) {{
        h += '<div class="note">NO HAY CANDIDATOS</div>';
    }}

    // ===== 5. Estadisticas por posicion y regimen =====
    h += '<h2>5. Rendimiento Historico por Posicion FV</h2>';
    const regStats = RANK_STATS[w.regime];
    if (regStats) {{
        h += '<div class="note"><b>Regimen actual: '+w.regime+'</b> - Retorno medio historico de los subsectores segun su posicion en el ranking FV ajustado, para todas las semanas clasificadas como '+w.regime+'.</div>';
        h += '<table class="rank-stats-table"><tr><th>Pos</th><th>N semanas</th><th>Ret Medio</th><th>Win Rate</th><th>Ret Acumulado</th></tr>';
        const ranks = Object.keys(regStats).sort((a,b) => parseInt(a) - parseInt(b));
        ranks.forEach(r => {{
            const st = regStats[r];
            const wrCls = st.wr >= 55 ? 'pos' : st.wr < 48 ? 'neg' : 'neutral';
            h += '<tr><td><b>'+r+'</b></td><td>'+st.n+'</td>';
            h += '<td class="'+vc(st.avg)+'">'+fmt(st.avg, 3)+'%</td>';
            h += '<td class="'+wrCls+'">'+st.wr.toFixed(1)+'%</td>';
            h += '<td class="'+vc(st.total)+'">'+fmt(st.total, 1)+'%</td></tr>';
        }});
        h += '</table>';
    }}

    // Comparativa top 10 posiciones en todos los regimenes
    h += '<details><summary style="cursor:pointer;color:#1565c0;font-weight:bold;margin:8px 0;">Comparativa Top 10 posiciones - Todos los regimenes</summary>';
    const allRegs = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION'];
    const activeRegs = allRegs.filter(r => RANK_STATS[r]);
    h += '<table class="rank-stats-table"><tr><th>Pos</th>';
    activeRegs.forEach(r => {{
        h += '<th style="background:'+(REGIME_COLORS[r]||'#666')+';font-size:10px;">'+r+'</th>';
    }});
    h += '</tr>';
    for (let pos = 1; pos <= 10; pos++) {{
        h += '<tr><td><b>'+pos+'</b></td>';
        activeRegs.forEach(r => {{
            const st = RANK_STATS[r][pos];
            if (st) {{
                h += '<td class="'+vc(st.avg)+'" title="N='+st.n+' WR='+st.wr+'%">'+fmt(st.avg, 2)+'%</td>';
            }} else {{
                h += '<td class="neutral">-</td>';
            }}
        }});
        h += '</tr>';
    }}
    h += '</table>';
    h += '<p style="font-size:11px;color:#999;margin-top:5px;">Hover sobre cada celda para ver N semanas y Win Rate</p>';
    h += '</details>';

    document.getElementById('content').innerHTML = h;
}}

let subSortAsc = false;
function sortSubTable() {{
    const table = document.getElementById('subTable');
    if (!table) return;
    const rows = Array.from(table.querySelectorAll('tr[data-ret]'));
    subSortAsc = !subSortAsc;
    rows.sort((a, b) => {{
        const va = parseFloat(a.dataset.ret), vb = parseFloat(b.dataset.ret);
        return subSortAsc ? va - vb : vb - va;
    }});
    rows.forEach((r, i) => {{
        r.cells[0].textContent = i + 1;
        r.cells[1].innerHTML = '<span class="chg-same">-</span>';
        table.appendChild(r);
    }});
}}
</script>
</body></html>"""

with open('senal_navegable.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"OK -> senal_navegable.html ({len(all_weeks_data)} semanas)")
