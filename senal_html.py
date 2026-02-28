"""
Genera HTML de verificacion de senal semanal
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0
ATR_MIN = 1.5

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
print("Cargando datos...")
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY symbol, date
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

df_weekly = df_weekly.groupby('symbol', group_keys=False).apply(calc_stock_metrics)

sub_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
sub_weekly = sub_weekly.sort_values(['symbol', 'date'])
sub_agg = sub_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean')).reset_index()
sub_agg = sub_agg.sort_values(['subsector', 'date'])

sub_weekly_ret = df_weekly.groupby(['subsector', 'date']).agg(
    avg_atr=('atr_pct', 'mean')).reset_index()
sub_agg = sub_agg.merge(sub_weekly_ret, on=['subsector', 'date'], how='left')

date_counts = sub_agg.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_agg = sub_agg[sub_agg['date'].isin(valid_dates)]

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

sub_agg = sub_agg.groupby('subsector', group_keys=False).apply(calc_sub_metrics)
dd_wide = sub_agg.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_agg.pivot(index='date', columns='subsector', values='rsi_14w')
atr_wide = sub_agg.pivot(index='date', columns='subsector', values='avg_atr')

spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

vix_df = pd.read_sql("""
    SELECT date, close as vix FROM price_history_vix
    WHERE symbol='^VIX' ORDER BY date
""", engine)
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.set_index('date')

weekly_events = build_weekly_events('2000-01-01', '2026-02-28')

# ================================================================
# SENAL - Regimen desde CSV (fuente de verdad: regimenes_historico.py)
# ================================================================
target_date = pd.Timestamp('2026-02-26')
all_dates = dd_wide.index[dd_wide.index <= target_date]
signal_date = all_dates[-1]

# Leer regimen del CSV (calculado con 66 subsectores S&P 500 reales)
reg_df = pd.read_csv('data/regimenes_historico.csv')
reg_df['fecha_senal'] = pd.to_datetime(reg_df['fecha_senal'])
# Buscar la fila mas cercana a signal_date
reg_row = reg_df.iloc[(reg_df['fecha_senal'] - signal_date).abs().argsort().iloc[0]]

regime = reg_row['regime']
total = float(reg_row['total'])
s1 = float(reg_row['s_bdd'])
s2 = float(reg_row['s_brsi'])
s3 = float(reg_row['s_ddp'])
s4 = float(reg_row['s_spy'])
s5 = float(reg_row['s_mom'])
pct_dd_healthy = float(reg_row['pct_dd_h'])
pct_dd_deep = float(reg_row['pct_dd_d'])
pct_rsi_above55 = float(reg_row['pct_rsi'])
n_total = int(reg_row['n_sub'])
spy_close = float(reg_row['spy_close'])
spy_ma200 = float(reg_row['spy_ma200'])
spy_dist = float(reg_row['spy_dist'])
spy_mom_10w = float(reg_row['spy_mom'])
spy_above_ma200 = 1 if spy_dist > 0 else 0

vix_dates = vix_df.index[vix_df.index <= target_date]
vix_val = float(reg_row['vix'])
vix_prev = float(vix_df.loc[vix_dates[-2], 'vix']) if len(vix_dates) >= 2 else vix_val
vix_date = vix_dates[-1].strftime('%Y-%m-%d')

# DD/RSI del event_map subsectors (para fair value, no para regimen)
dd_row_reg = dd_wide.loc[signal_date] if signal_date in dd_wide.index else None
rsi_row_reg = rsi_wide.loc[signal_date] if signal_date in rsi_wide.index else None

# Fair Value
if signal_date in weekly_events.index:
    evt_date = signal_date
else:
    nearest_idx = weekly_events.index.get_indexer([signal_date], method='nearest')[0]
    evt_date = weekly_events.index[nearest_idx]
events_row = weekly_events.loc[evt_date]
active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
scores_evt = score_fair(active)

prev_dates = dd_wide.index[dd_wide.index < signal_date]
dd_row = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
rsi_row = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
atr_row = atr_wide.loc[signal_date] if signal_date in atr_wide.index else None
scores_v3 = adjust_score_by_price(scores_evt, dd_row, rsi_row)

# Subsector data
sub_data_list = []
for sub_id in sorted(SUBSECTORS.keys()):
    fv_raw = scores_evt.get(sub_id, 5.0)
    fv_adj = scores_v3.get(sub_id, 5.0)
    dd = dd_row.get(sub_id, 0) if dd_row is not None else 0
    rsi = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
    atr = atr_row.get(sub_id, 0) if atr_row is not None else 0
    dd_cur = dd_row_reg.get(sub_id, 0) if dd_row_reg is not None else 0
    rsi_cur = rsi_row_reg.get(sub_id, 50) if rsi_row_reg is not None else 50
    if not pd.notna(dd): dd = 0
    if not pd.notna(rsi): rsi = 50
    if not pd.notna(atr): atr = 0
    if not pd.notna(dd_cur): dd_cur = 0
    if not pd.notna(rsi_cur): rsi_cur = 50
    sub_data_list.append({
        'id': sub_id, 'name': SUBSECTORS[sub_id]['label'],
        'etf': SUBSECTORS[sub_id].get('etf', ''),
        'n_tickers': len(SUBSECTORS[sub_id]['tickers']),
        'fv_raw': fv_raw, 'fv_adj': fv_adj,
        'dd': dd_cur, 'rsi': rsi_cur, 'atr': atr,
    })
sub_data_list.sort(key=lambda x: -x['fv_adj'])

# Picks estrategia A
top3_a = []
if regime in ('ALCISTA', 'GOLDILOCKS', 'CAPITULACION', 'RECOVERY'):
    top3_a = sorted([(s['id'], s['fv_adj'], s['name'], s['etf']) for s in sub_data_list if s['fv_adj'] > 5.5],
                  key=lambda x: -x[1])[:3]

# Picks estrategia B
n_long, n_short = REGIME_ALLOC.get(regime, (5,5))
stock_data = df_weekly[df_weekly['date'] <= signal_date].sort_values(['symbol', 'date'])
latest_stocks = stock_data.groupby('symbol').last().reset_index()

long_candidates = []
if n_long > 0:
    good_subs = {s['id']: s['fv_adj'] for s in sub_data_list if s['fv_adj'] > 5.5}
    for _, stock in latest_stocks.iterrows():
        ticker = stock['symbol']
        sub = ticker_to_sub.get(ticker)
        if sub not in good_subs: continue
        fv = good_subs[sub]
        mom = stock.get('mom_12w', 0)
        rsi = stock.get('rsi_14w', 50)
        dd = stock.get('dd_52w', 0)
        price = stock.get('close', 0)
        if not pd.notna(mom): mom = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(dd): dd = 0
        composite = (np.clip((fv - 5.0) / 4.0, 0, 1) * 3.0 +
                     np.clip((mom + 20) / 60, 0, 1) * 3.0 +
                     np.clip((rsi - 30) / 50, 0, 1) * 2.0 +
                     np.clip((dd + 30) / 30, 0, 1) * 2.0)
        long_candidates.append({
            'ticker': ticker, 'sub': sub, 'sub_name': SUBSECTORS[sub]['label'],
            'fv': fv, 'mom': mom, 'rsi': rsi, 'dd': dd, 'price': price, 'score': composite,
        })

short_candidates = []
if n_short > 0:
    bad_subs = {s['id']: s['fv_adj'] for s in sub_data_list if s['fv_adj'] < 4.5}
    for _, stock in latest_stocks.iterrows():
        ticker = stock['symbol']
        sub = ticker_to_sub.get(ticker)
        if sub not in bad_subs: continue
        fv = bad_subs[sub]
        mom = stock.get('mom_12w', 0)
        rsi = stock.get('rsi_14w', 50)
        dd = stock.get('dd_52w', 0)
        atr = stock.get('atr_pct', 0)
        price = stock.get('close', 0)
        if not pd.notna(mom): mom = 0
        if not pd.notna(rsi): rsi = 50
        if not pd.notna(dd): dd = 0
        if not pd.notna(atr): atr = 0
        if dd < -40 or rsi < 15 or atr < 1.5: continue
        composite = (np.clip((5.0 - fv) / 4.0, 0, 1) * 3.0 +
                     np.clip((-mom + 20) / 60, 0, 1) * 3.0 +
                     np.clip((70 - rsi) / 50, 0, 1) * 2.0 +
                     np.clip((atr - 1.5) / 5.0, 0, 1) * 2.0)
        short_candidates.append({
            'ticker': ticker, 'sub': sub, 'sub_name': SUBSECTORS[sub]['label'],
            'fv': fv, 'mom': mom, 'rsi': rsi, 'dd': dd, 'price': price, 'atr': atr, 'score': composite,
        })

long_candidates.sort(key=lambda x: -x['score'])
short_candidates.sort(key=lambda x: -x['score'])

def diversified_pick(candidates, n, max_per_sub=2):
    picked = []
    sub_count = {}
    for c in candidates:
        s = c['sub']
        if sub_count.get(s, 0) >= max_per_sub: continue
        picked.append(c)
        sub_count[s] = sub_count.get(s, 0) + 1
        if len(picked) >= n: break
    return picked

longs_sel = diversified_pick(long_candidates, n_long)
shorts_sel = diversified_pick(short_candidates, n_short)

# Semana ISO
week_num = signal_date.isocalendar()[1]
trade_open = '27/02/2026'
trade_close = '06/03/2026'

print(f"Generando HTML... Regimen: {regime} ({total:+.1f})")

# ================================================================
# HTML
# ================================================================
def val_class(v):
    if v > 0: return 'pos'
    if v < 0: return 'neg'
    return 'neutral'

def fv_bar(v):
    pct = v / 10 * 100
    if v >= 7: color = '#2e7d32'
    elif v >= 5.5: color = '#1565c0'
    elif v >= 4.5: color = '#ff9800'
    else: color = '#c62828'
    return f'<div style="background:#eee;border-radius:3px;height:14px;width:100px;display:inline-block;vertical-align:middle;"><div style="background:{color};height:14px;border-radius:3px;width:{pct:.0f}px;"></div></div> <b>{v:.2f}</b>'

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Senal Semana {week_num} - 2026</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #fff; color: #222; margin: 20px; }}
h1 {{ color: #1565c0; text-align: center; margin-bottom: 5px; }}
h2 {{ color: #333; margin-top: 30px; margin-bottom: 10px; border-bottom: 2px solid #1565c0; padding-bottom: 5px; }}
h3 {{ color: #555; margin-top: 20px; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 25px; font-size: 14px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }}
th {{ background: #1565c0; color: #fff; padding: 8px 6px; text-align: center; border: 1px solid #ccc; position: sticky; top: 0; z-index: 1; }}
td {{ padding: 6px; text-align: center; border: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #f5f7fa; }}
tr:nth-child(odd) {{ background: #fff; }}
tr:hover {{ background: #e3f2fd; }}
.pos {{ color: #2e7d32; font-weight: bold; }}
.neg {{ color: #c62828; font-weight: bold; }}
.neutral {{ color: #999; }}
td.left {{ text-align: left; }}
.summary-box {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }}
.summary-card {{ background: #f5f7fa; border: 1px solid #ddd; border-radius: 8px; padding: 15px; flex: 1; min-width: 140px; text-align: center; }}
.summary-card h4 {{ margin: 0 0 8px 0; color: #1565c0; font-size: 12px; }}
.summary-card .value {{ font-size: 22px; font-weight: bold; }}
.regime-badge {{ display: inline-block; padding: 5px 18px; border-radius: 5px; font-size: 18px; font-weight: bold; color: #fff; }}
.note {{ background: #fffde7; padding: 10px 15px; border-radius: 8px; border-left: 4px solid #ffd600; margin-bottom: 20px; font-size: 13px; }}
.indicator-table td {{ padding: 8px 12px; }}
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
</style>
</head>
<body>

<h1>Senal Semanal - Semana {week_num} (2026)</h1>
<p class="subtitle">Senal: jueves {signal_date.strftime('%d/%m/%Y')} cierre | Trading: viernes {trade_open} open &rarr; viernes {trade_close} open | Datos FMP hasta {signal_date.strftime('%d/%m/%Y')}</p>

<!-- REGIMEN -->
<h2>1. Regimen de Mercado</h2>
<div style="text-align:center; margin: 20px 0;">
    <span class="regime-badge" style="background:{REGIME_COLORS.get(regime, '#666')};">{regime}</span>
    <span style="font-size:20px; margin-left:15px; font-weight:bold;">Score: {total:+.1f}</span>
</div>

<div class="summary-box">
<div class="summary-card"><h4>SPY</h4><div class="value">${spy_close:.2f}</div></div>
<div class="summary-card"><h4>MA200</h4><div class="value">${spy_ma200:.2f}</div></div>
<div class="summary-card"><h4>Dist MA200</h4><div class="value {val_class(spy_dist)}">{spy_dist:+.1f}%</div></div>
<div class="summary-card"><h4>Mom 10w</h4><div class="value {val_class(spy_mom_10w)}">{spy_mom_10w:+.1f}%</div></div>
<div class="summary-card"><h4>VIX ({vix_date})</h4><div class="value">{vix_val:.1f}</div></div>
</div>

<table class="indicator-table" style="width:auto; margin: 0 auto 20px auto; min-width:700px;">
<tr><th>Indicador</th><th>Valor</th><th>Score</th><th>Rango</th><th>Umbral</th></tr>
<tr><td class="left"><b>BDD</b> - Breadth Drawdown</td><td>{pct_dd_healthy:.1f}%</td><td class="{val_class(s1)}"><b>{s1:+.1f}</b></td><td>-3.0 a +2.0</td><td>&ge;75% &rarr; +2.0</td></tr>
<tr><td class="left"><b>BRSI</b> - Breadth RSI</td><td>{pct_rsi_above55:.1f}%</td><td class="{val_class(s2)}"><b>{s2:+.1f}</b></td><td>-3.0 a +2.0</td><td>&ge;75% &rarr; +2.0</td></tr>
<tr><td class="left"><b>DDP</b> - Deep Drawdown %</td><td>{pct_dd_deep:.1f}%</td><td class="{val_class(s3)}"><b>{s3:+.1f}</b></td><td>-2.5 a +1.5</td><td>&le;5% &rarr; +1.5</td></tr>
<tr><td class="left"><b>SPY</b> - vs MA200</td><td>{spy_dist:+.1f}%</td><td class="{val_class(s4)}"><b>{s4:+.1f}</b></td><td>-2.5 a +1.5</td><td>&gt;MA200 &amp; &gt;5% &rarr; +1.5</td></tr>
<tr><td class="left"><b>MOM</b> - Momentum 10w</td><td>{spy_mom_10w:+.1f}%</td><td class="{val_class(s5)}"><b>{s5:+.1f}</b></td><td>-1.5 a +1.0</td><td>&gt;5% &rarr; +1.0</td></tr>
<tr style="background:#e3f2fd;"><td class="left"><b>TOTAL</b></td><td></td><td><b style="font-size:16px;">{total:+.1f}</b></td><td>-12.5 a +8.0</td><td>&ge;4.0 &rarr; ALCISTA</td></tr>
</table>

<div class="note">
<b>VIX:</b> {vix_val:.1f} (anterior: {vix_prev:.1f}) | <b>Veto VIX:</b> {'SI - rebajado' if vix_val >= 30 else 'NO'} |
<b>Subsectores:</b> {n_total} analizados | <b>DD sanos (&gt;-10%):</b> {pct_dd_healthy:.0f}% | <b>DD profundos (&lt;-20%):</b> {pct_dd_deep:.0f}% | <b>RSI &gt;55:</b> {pct_rsi_above55:.0f}%
</div>

<!-- EVENTOS -->
<h2>2. Eventos Activos ({len(active)})</h2>
<table style="width:auto; min-width:400px;">
<tr><th>Evento</th><th>Intensidad</th></tr>
"""

for evt, intensity in sorted(active.items(), key=lambda x: -x[1]):
    bar_w = int(intensity / 2.0 * 100)
    color = '#2e7d32' if intensity >= 1.5 else '#1565c0' if intensity >= 1.0 else '#ff9800'
    html += f'<tr><td class="left">{evt}</td><td><div style="background:#eee;border-radius:3px;height:14px;width:120px;display:inline-block;vertical-align:middle;"><div style="background:{color};height:14px;border-radius:3px;width:{bar_w}px;"></div></div> <b>{intensity:.1f}</b></td></tr>\n'

html += "</table>\n"

# SUBSECTORES
html += f"""
<h2>3. Fair Value Subsectores (todos los {len(sub_data_list)})</h2>
<table>
<tr><th>#</th><th>Subsector</th><th>Tickers</th><th>FV Raw</th><th>FV Ajustado</th><th>DD 52w</th><th>RSI 14w</th><th>ATR%</th></tr>
"""
for i, s in enumerate(sub_data_list, 1):
    row_class = ''
    if s['fv_adj'] >= 5.5: row_class = ' class="fv-high"'
    elif s['fv_adj'] < 4.5: row_class = ' class="fv-low"'
    html += f'<tr{row_class}><td>{i}</td><td class="left"><b>{s["name"]}</b></td><td>{s["n_tickers"]}</td>'
    html += f'<td>{s["fv_raw"]:.2f}</td><td>{fv_bar(s["fv_adj"])}</td>'
    html += f'<td class="{val_class(-abs(s["dd"]))}">{s["dd"]:+.1f}%</td>'
    html += f'<td>{s["rsi"]:.0f}</td><td>{s["atr"]:.1f}%</td></tr>\n'

html += "</table>\n"

# ESTRATEGIA A
html += f"""
<h2>4. Estrategia A: Subsectores ({regime})</h2>
"""
if top3_a:
    html += '<table style="width:auto; min-width:600px;">\n<tr><th>#</th><th>Direccion</th><th>Subsector</th><th>ETF</th><th>FV</th><th>Tickers</th></tr>\n'
    for i, (sub_id, score, name, etf) in enumerate(top3_a, 1):
        tickers = ', '.join(SUBSECTORS[sub_id]['tickers'])
        html += f'<tr class="pick-long"><td>{i}</td><td><span class="badge badge-long">LONG</span></td><td class="left"><b>{name}</b></td><td>{etf}</td><td><b>{score:.2f}</b></td><td class="left">{tickers}</td></tr>\n'
    html += '</table>\n'
else:
    html += '<div class="note">NO HAY CANDIDATOS - NO OPERAR</div>\n'

# ESTRATEGIA B
html += f"""
<h2>5. Estrategia B: 10 Acciones ({n_long}L + {n_short}S)</h2>
"""
if longs_sel:
    html += f'<h3>LONGS ({len(longs_sel)})</h3>\n'
    html += '<table>\n<tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV Sub</th><th>Mom 12w</th><th>RSI 14w</th><th>DD 52w</th><th>Score</th></tr>\n'
    for i, c in enumerate(longs_sel, 1):
        html += f'<tr class="pick-long"><td>{i}</td><td><b>{c["ticker"]}</b></td><td>${c["price"]:.2f}</td>'
        html += f'<td class="left">{c["sub_name"]}</td><td><b>{c["fv"]:.1f}</b></td>'
        html += f'<td class="{val_class(c["mom"])}">{c["mom"]:+.1f}%</td>'
        html += f'<td>{c["rsi"]:.0f}</td><td class="{val_class(c["dd"])}">{c["dd"]:+.1f}%</td>'
        html += f'<td><b>{c["score"]:.1f}</b></td></tr>\n'
    html += '</table>\n'

    # Top 20 long candidates
    html += f'<details><summary style="cursor:pointer;color:#1565c0;font-weight:bold;margin:10px 0;">Ver top 20 candidatos LONG completos</summary>\n'
    html += '<table>\n<tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV</th><th>Mom 12w</th><th>RSI 14w</th><th>DD 52w</th><th>Score</th></tr>\n'
    for i, c in enumerate(long_candidates[:20], 1):
        sel = ' class="pick-long"' if c in longs_sel else ''
        html += f'<tr{sel}><td>{i}</td><td><b>{c["ticker"]}</b></td><td>${c["price"]:.2f}</td>'
        html += f'<td class="left">{c["sub_name"]}</td><td>{c["fv"]:.1f}</td>'
        html += f'<td class="{val_class(c["mom"])}">{c["mom"]:+.1f}%</td>'
        html += f'<td>{c["rsi"]:.0f}</td><td class="{val_class(c["dd"])}">{c["dd"]:+.1f}%</td>'
        html += f'<td>{c["score"]:.1f}</td></tr>\n'
    html += '</table></details>\n'

if shorts_sel:
    html += f'<h3>SHORTS ({len(shorts_sel)})</h3>\n'
    html += '<table>\n<tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV Sub</th><th>Mom 12w</th><th>RSI 14w</th><th>DD 52w</th><th>ATR%</th><th>Score</th></tr>\n'
    for i, c in enumerate(shorts_sel, 1):
        html += f'<tr class="pick-short"><td>{i}</td><td><b>{c["ticker"]}</b></td><td>${c["price"]:.2f}</td>'
        html += f'<td class="left">{c["sub_name"]}</td><td><b>{c["fv"]:.1f}</b></td>'
        html += f'<td class="{val_class(c["mom"])}">{c["mom"]:+.1f}%</td>'
        html += f'<td>{c["rsi"]:.0f}</td><td class="{val_class(c["dd"])}">{c["dd"]:+.1f}%</td>'
        html += f'<td>{c["atr"]:.1f}%</td><td><b>{c["score"]:.1f}</b></td></tr>\n'
    html += '</table>\n'

    html += f'<details><summary style="cursor:pointer;color:#c62828;font-weight:bold;margin:10px 0;">Ver top 20 candidatos SHORT completos</summary>\n'
    html += '<table>\n<tr><th>#</th><th>Ticker</th><th>Precio</th><th>Subsector</th><th>FV</th><th>Mom 12w</th><th>RSI 14w</th><th>DD 52w</th><th>ATR%</th><th>Score</th></tr>\n'
    for i, c in enumerate(short_candidates[:20], 1):
        sel = ' class="pick-short"' if c in shorts_sel else ''
        html += f'<tr{sel}><td>{i}</td><td><b>{c["ticker"]}</b></td><td>${c["price"]:.2f}</td>'
        html += f'<td class="left">{c["sub_name"]}</td><td>{c["fv"]:.1f}</td>'
        html += f'<td class="{val_class(c["mom"])}">{c["mom"]:+.1f}%</td>'
        html += f'<td>{c["rsi"]:.0f}</td><td class="{val_class(c["dd"])}">{c["dd"]:+.1f}%</td>'
        html += f'<td>{c["atr"]:.1f}%</td><td>{c["score"]:.1f}</td></tr>\n'
    html += '</table></details>\n'

if not longs_sel and not shorts_sel:
    html += '<div class="note">NO HAY CANDIDATOS - NO OPERAR</div>\n'

html += "\n</body>\n</html>"

with open('senal_semanal.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"OK -> senal_semanal.html")
