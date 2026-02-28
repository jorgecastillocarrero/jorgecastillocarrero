"""
Analisis predictivo: dado el regimen del viernes, que probabilidad hay
de cada regimen la semana siguiente? Matriz de transicion Markov.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

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

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_price_metrics(g):
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

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

spy_daily = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100
spy_w['ret_spy'] = spy_w['close'].pct_change()

# SPY lunes open para retornos reales de trading (lun open -> lun open)
spy_mon_open = spy_daily['open'].resample('W-MON').first().dropna()
spy_mon_ret = (spy_mon_open.shift(-1) / spy_mon_open - 1) * 100  # en %
spy_mon_ret = spy_mon_ret.to_frame('spy_ret_mon')

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
    skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

def classify_regime(date):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: return 'NEUTRAL', 0.0
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]; rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL', 0.0
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
    if pct_dd_healthy >= 75: s1 = 2.0
    elif pct_dd_healthy >= 60: s1 = 1.0
    elif pct_dd_healthy >= 45: s1 = 0.0
    elif pct_dd_healthy >= 30: s1 = -1.0
    elif pct_dd_healthy >= 15: s1 = -2.0
    else: s1 = -3.0
    if pct_rsi_above55 >= 75: s2 = 2.0
    elif pct_rsi_above55 >= 60: s2 = 1.0
    elif pct_rsi_above55 >= 45: s2 = 0.0
    elif pct_rsi_above55 >= 30: s2 = -1.0
    elif pct_rsi_above55 >= 15: s2 = -2.0
    else: s2 = -3.0
    if pct_dd_deep <= 5: s3 = 1.5
    elif pct_dd_deep <= 15: s3 = 0.5
    elif pct_dd_deep <= 30: s3 = -0.5
    elif pct_dd_deep <= 50: s3 = -1.5
    else: s3 = -2.5
    if spy_above_ma200 and spy_dist > 5: s4 = 1.5
    elif spy_above_ma200: s4 = 0.5
    elif spy_dist > -5: s4 = -0.5
    elif spy_dist > -15: s4 = -1.5
    else: s4 = -2.5
    if spy_mom_10w > 5: s5 = 1.0
    elif spy_mom_10w > 0: s5 = 0.5
    elif spy_mom_10w > -5: s5 = -0.5
    elif spy_mom_10w > -15: s5 = -1.0
    else: s5 = -1.5
    total = s1 + s2 + s3 + s4 + s5
    is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'): regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL': regime = 'CAUTIOUS'
    # CAPITULACION: PANICO + VIX bajando vs semana anterior
    if regime == 'PANICO':
        vix_all = vix_df.index[vix_df.index <= date]
        if len(vix_all) >= 2:
            prev_vix = vix_df.loc[vix_all[-2], 'vix']
            if pd.notna(prev_vix) and vix_val < prev_vix:
                regime = 'CAPITULACION'
    return regime, total

print("Clasificando regimenes...")
all_weeks = []
for date in dd_wide.index:
    if date.year < 2001: continue
    regime, score = classify_regime(date)

    # 1) Senal: Fri close -> Fri close (para clasificacion de regimen)
    spy_dates_after = spy_w.index[spy_w.index > date]
    spy_ret_fri = 0
    if len(spy_dates_after) >= 1:
        next_fri = spy_dates_after[0]
        spy_now = spy_w.loc[spy_w.index[spy_w.index <= date][-1], 'close']
        spy_next = spy_w.loc[next_fri, 'close']
        spy_ret_fri = (spy_next / spy_now - 1) * 100

    # 2) Rentabilidad: Mon open -> Mon open (retorno real de trading)
    target_mon = date + pd.Timedelta(days=3)
    mon_dates = spy_mon_ret.index
    diffs = abs(mon_dates - target_mon)
    closest_idx = diffs.argmin()
    closest_mon = mon_dates[closest_idx]
    if abs((closest_mon - target_mon).days) <= 3 and closest_mon in spy_mon_ret.index:
        spy_ret_trade = spy_mon_ret.loc[closest_mon, 'spy_ret_mon']
    else:
        spy_ret_trade = 0

    all_weeks.append({'date': date, 'regime': regime, 'score': score,
                      'spy_ret_fri': spy_ret_fri, 'spy_ret_mon': spy_ret_trade})

df = pd.DataFrame(all_weeks).sort_values('date').reset_index(drop=True)
df['regime_next'] = df['regime'].shift(-1)
df['score_next'] = df['score'].shift(-1)
df = df.dropna(subset=['regime_next'])

REGIME_ORDER = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO', 'CAPITULACION']
REGIME_COLOR = {
    'BURBUJA': '#ff6600', 'GOLDILOCKS': '#00aa00', 'ALCISTA': '#33cc33',
    'NEUTRAL': '#888888', 'CAUTIOUS': '#cc9900', 'BEARISH': '#cc3333',
    'CRISIS': '#990000', 'PANICO': '#660066', 'CAPITULACION': '#0066cc'
}
REGIME_BG = {
    'BURBUJA': '#fff3e0', 'GOLDILOCKS': '#e8f5e9', 'ALCISTA': '#f1f8e9',
    'NEUTRAL': '#f5f5f5', 'CAUTIOUS': '#fff8e1', 'BEARISH': '#ffebee',
    'CRISIS': '#fce4ec', 'PANICO': '#f3e5f5', 'CAPITULACION': '#e3f2fd'
}

# ================================================================
# CONSTRUIR MATRIZ DE TRANSICION
# ================================================================
print("Construyendo matrices de transicion...")

# Matriz principal: P(next | current)
trans_count = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER)
for _, row in df.iterrows():
    curr = row['regime']
    nxt = row['regime_next']
    if curr in REGIME_ORDER and nxt in REGIME_ORDER:
        trans_count.loc[curr, nxt] += 1

trans_pct = trans_count.div(trans_count.sum(axis=1), axis=0) * 100

# Retorno SPY promedio por transicion (ambos metodos)
spy_ret_by_trans_fri = {}
spy_ret_by_trans_mon = {}
for curr in REGIME_ORDER:
    for nxt in REGIME_ORDER:
        mask = (df['regime'] == curr) & (df['regime_next'] == nxt)
        if mask.sum() > 0:
            spy_ret_by_trans_fri[(curr, nxt)] = df.loc[mask, 'spy_ret_fri'].mean()
            spy_ret_by_trans_mon[(curr, nxt)] = df.loc[mask, 'spy_ret_mon'].mean()

# Persistencia: cuantas semanas seguidas dura cada regimen
episodes = []
curr_reg = None
curr_start = None
curr_count = 0
for _, row in df.iterrows():
    if row['regime'] != curr_reg:
        if curr_reg is not None:
            episodes.append({'regime': curr_reg, 'start': curr_start, 'weeks': curr_count})
        curr_reg = row['regime']
        curr_start = row['date']
        curr_count = 1
    else:
        curr_count += 1
if curr_reg is not None:
    episodes.append({'regime': curr_reg, 'start': curr_start, 'weeks': curr_count})
df_ep = pd.DataFrame(episodes)

# Analisis por zona
df['zone'] = df['regime'].map(lambda r: 'ALCISTA' if r in ('BURBUJA','GOLDILOCKS','ALCISTA')
                               else 'NEUTRAL' if r == 'NEUTRAL'
                               else 'BAJISTA')
df['zone_next'] = df['regime_next'].map(lambda r: 'ALCISTA' if r in ('BURBUJA','GOLDILOCKS','ALCISTA')
                               else 'NEUTRAL' if r == 'NEUTRAL'
                               else 'BAJISTA')

zone_order = ['ALCISTA', 'NEUTRAL', 'BAJISTA']
zone_count = pd.DataFrame(0, index=zone_order, columns=zone_order)
for _, row in df.iterrows():
    zone_count.loc[row['zone'], row['zone_next']] += 1
zone_pct = zone_count.div(zone_count.sum(axis=1), axis=0) * 100

# ================================================================
# GENERAR HTML
# ================================================================
print("Generando HTML...")

html = []
html.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Analisis Predictivo - Transiciones de Regimen</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #fafafa; max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { color: #333; border-bottom: 3px solid #333; padding-bottom: 10px; }
h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
h3 { color: #666; }
.intro { background: white; padding: 15px; border-radius: 8px; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
table { border-collapse: collapse; margin: 15px 0; }
th { background: #333; color: white; padding: 10px 14px; text-align: center; font-size: 13px; }
td { padding: 8px 12px; text-align: center; border: 1px solid #e0e0e0; font-size: 13px; }
.matrix-table td { min-width: 70px; }
.pct-high { background: #1b5e20; color: white; font-weight: bold; }
.pct-med { background: #4caf50; color: white; }
.pct-low { background: #c8e6c9; }
.pct-zero { background: #f5f5f5; color: #ccc; }
.pct-self { border: 3px solid #333; }
.stat-card { display: inline-block; background: white; padding: 15px 25px; margin: 8px; border-radius: 8px;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 120px; }
.stat-value { font-size: 28px; font-weight: bold; }
.stat-label { font-size: 12px; color: #888; margin-top: 4px; }
.regime-section { background: white; padding: 20px; margin: 15px 0; border-radius: 8px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.bar-container { display: flex; align-items: center; margin: 4px 0; }
.bar-label { width: 90px; font-size: 12px; font-weight: bold; text-align: right; padding-right: 8px; }
.bar { height: 24px; border-radius: 3px; display: flex; align-items: center; padding-left: 6px;
       font-size: 11px; color: white; font-weight: bold; min-width: 2px; }
.bar-value { font-size: 12px; margin-left: 6px; color: #666; }
.persistence-table td { text-align: center; }
.zone-badge { display: inline-block; padding: 4px 12px; border-radius: 15px; font-weight: bold;
              font-size: 13px; color: white; margin: 2px; }
.insight { background: #e3f2fd; padding: 12px 16px; border-radius: 6px; margin: 10px 0;
           border-left: 4px solid #1976d2; font-size: 14px; }
.warning { background: #fff3e0; border-left-color: #f57c00; }
</style></head><body>

<h1>Analisis Predictivo: Transiciones de Regimen de Mercado</h1>
<div class="intro">
<p><b>Pregunta:</b> Si el viernes al cierre el regimen es X, que probabilidad hay de que la semana siguiente sea cada regimen?</p>
<p>Basado en <b>""" + str(len(df)) + """ semanas</b> de datos (2001-2026). Cada celda muestra P(siguiente | actual).</p>
</div>
""")

# ================================================================
# 1. MATRIZ DE TRANSICION SIMPLIFICADA (ZONAS)
# ================================================================
html.append('<h2>1. Vista Simplificada: Transiciones por Zona</h2>')
ZONE_COLOR = {'ALCISTA': '#33cc33', 'NEUTRAL': '#888888', 'BAJISTA': '#cc3333'}

html.append('<table class="matrix-table"><tr><th></th><th colspan="3">Semana Siguiente</th></tr>')
html.append('<tr><th>Viernes actual</th>')
for z in zone_order:
    html.append(f'<th style="background:{ZONE_COLOR[z]}">{z}</th>')
html.append('<th>N</th></tr>')
for z_from in zone_order:
    html.append(f'<tr><td style="background:{ZONE_COLOR[z_from]};color:white;font-weight:bold">{z_from}</td>')
    n_total = zone_count.loc[z_from].sum()
    for z_to in zone_order:
        pct = zone_pct.loc[z_from, z_to]
        n = zone_count.loc[z_from, z_to]
        if pct >= 70: cls = 'pct-high'
        elif pct >= 30: cls = 'pct-med'
        elif pct > 0: cls = 'pct-low'
        else: cls = 'pct-zero'
        if z_from == z_to: cls += ' pct-self'
        html.append(f'<td class="{cls}">{pct:.1f}%<br><small>({n})</small></td>')
    html.append(f'<td><b>{n_total}</b></td></tr>')
html.append('</table>')

# Insights zonas
for z in zone_order:
    persist = zone_pct.loc[z, z]
    html.append(f'<div class="insight">Si estamos en zona <b>{z}</b>, hay un <b>{persist:.0f}%</b> '
                f'de probabilidad de seguir en la misma zona la semana siguiente.</div>')

# ================================================================
# 2. MATRIZ DE TRANSICION COMPLETA (8x8)
# ================================================================
html.append('<h2>2. Matriz Completa: 8 Regimenes</h2>')
html.append('<p>Cada fila muestra: dado el regimen actual (viernes), probabilidad de cada regimen la semana siguiente.</p>')
html.append('<table class="matrix-table"><tr><th rowspan="2">Viernes<br>actual</th>'
            '<th colspan="8">Regimen semana siguiente</th><th rowspan="2">N</th></tr><tr>')
for r in REGIME_ORDER:
    html.append(f'<th style="background:{REGIME_COLOR[r]}">{r[:4]}</th>')
html.append('</tr>')

for r_from in REGIME_ORDER:
    n_total = trans_count.loc[r_from].sum()
    if n_total == 0: continue
    html.append(f'<tr><td style="background:{REGIME_COLOR[r_from]};color:white;font-weight:bold">{r_from}</td>')
    for r_to in REGIME_ORDER:
        pct = trans_pct.loc[r_from, r_to]
        n = trans_count.loc[r_from, r_to]
        if pct >= 50: cls = 'pct-high'
        elif pct >= 20: cls = 'pct-med'
        elif pct > 0: cls = 'pct-low'
        else: cls = 'pct-zero'
        if r_from == r_to: cls += ' pct-self'
        cell = f'{pct:.0f}%' if pct >= 1 else '-'
        html.append(f'<td class="{cls}">{cell}<br><small>({n})</small></td>')
    html.append(f'<td><b>{n_total}</b></td></tr>')
html.append('</table>')

# ================================================================
# 3. DETALLE POR REGIMEN (barras horizontales)
# ================================================================
html.append('<h2>3. Detalle por Regimen: Probabilidades de Transicion</h2>')

for r_from in REGIME_ORDER:
    n_total = trans_count.loc[r_from].sum()
    if n_total == 0: continue

    persist_pct = trans_pct.loc[r_from, r_from]

    # Calcular zona probable
    pct_pos = sum(trans_pct.loc[r_from, r] for r in ['BURBUJA','GOLDILOCKS','ALCISTA'])
    pct_neu = trans_pct.loc[r_from, 'NEUTRAL']
    pct_neg = sum(trans_pct.loc[r_from, r] for r in ['CAUTIOUS','BEARISH','CRISIS','PANICO'])

    # SPY ret promedio cuando estamos en este regimen (Mon->Mon = trading real)
    mask_from = df['regime'] == r_from
    spy_avg = df.loc[mask_from, 'spy_ret_mon'].mean()
    spy_med = df.loc[mask_from, 'spy_ret_mon'].median()

    # Duracion promedio del episodio
    ep_data = df_ep[df_ep['regime'] == r_from]
    avg_dur = ep_data['weeks'].mean() if len(ep_data) > 0 else 0
    med_dur = ep_data['weeks'].median() if len(ep_data) > 0 else 0
    max_dur = ep_data['weeks'].max() if len(ep_data) > 0 else 0

    html.append(f'<div class="regime-section" style="border-left: 5px solid {REGIME_COLOR[r_from]}">')
    html.append(f'<h3 style="color:{REGIME_COLOR[r_from]};margin-top:0">{r_from} ({n_total} semanas)</h3>')

    # Stats cards
    html.append('<div>')
    html.append(f'<div class="stat-card"><div class="stat-value" style="color:{REGIME_COLOR[r_from]}">'
                f'{persist_pct:.0f}%</div><div class="stat-label">Persistencia</div></div>')
    html.append(f'<div class="stat-card"><div class="stat-value" style="color:#33cc33">'
                f'{pct_pos:.0f}%</div><div class="stat-label">Prob. Alcista</div></div>')
    html.append(f'<div class="stat-card"><div class="stat-value" style="color:#888">'
                f'{pct_neu:.0f}%</div><div class="stat-label">Prob. Neutral</div></div>')
    html.append(f'<div class="stat-card"><div class="stat-value" style="color:#cc3333">'
                f'{pct_neg:.0f}%</div><div class="stat-label">Prob. Bajista</div></div>')
    spy_color = '#33cc33' if spy_avg >= 0 else '#cc3333'
    html.append(f'<div class="stat-card"><div class="stat-value" style="color:{spy_color}">'
                f'{spy_avg:+.2f}%</div><div class="stat-label">SPY avg (L-L)</div></div>')
    html.append(f'<div class="stat-card"><div class="stat-value">'
                f'{avg_dur:.1f}</div><div class="stat-label">Duracion media (sem)</div></div>')
    html.append('</div>')

    # Barras de probabilidad
    html.append('<div style="margin-top:15px">')
    for r_to in REGIME_ORDER:
        pct = trans_pct.loc[r_from, r_to]
        n = trans_count.loc[r_from, r_to]
        if n == 0: continue
        bar_width = max(pct * 4, 2)
        spy_t = spy_ret_by_trans_mon.get((r_from, r_to), 0)
        spy_str = f'SPY {spy_t:+.2f}%' if spy_t != 0 else ''
        html.append(f'<div class="bar-container">'
                    f'<div class="bar-label" style="color:{REGIME_COLOR[r_to]}">{r_to}</div>'
                    f'<div class="bar" style="width:{bar_width}px;background:{REGIME_COLOR[r_to]}">'
                    f'{pct:.0f}%</div>'
                    f'<div class="bar-value">({n} sem) {spy_str}</div></div>')
    html.append('</div>')

    # Insights por regimen
    # Top transicion de cambio (no persistencia)
    changes = [(r, trans_pct.loc[r_from, r]) for r in REGIME_ORDER if r != r_from and trans_count.loc[r_from, r] > 0]
    changes.sort(key=lambda x: -x[1])
    if changes:
        top = changes[0]
        html.append(f'<div class="insight">Si cambia, lo mas probable es pasar a '
                    f'<b style="color:{REGIME_COLOR[top[0]]}">{top[0]}</b> ({top[1]:.0f}% de las veces).</div>')

    # Duracion
    if len(ep_data) > 0:
        html.append(f'<div class="insight">Duracion de episodios: mediana {med_dur:.0f} sem, '
                    f'media {avg_dur:.1f} sem, max {max_dur} sem ({len(ep_data)} episodios).</div>')

    html.append('</div>')

# ================================================================
# 4. PERSISTENCIA Y DURACION
# ================================================================
html.append('<h2>4. Persistencia y Duracion de Episodios</h2>')
html.append('<table class="persistence-table"><tr><th>Regimen</th><th>Persistencia</th>'
            '<th>Episodios</th><th>Duracion media</th><th>Duracion mediana</th>'
            '<th>Max</th><th>1 sem</th><th>2-4 sem</th><th>5-10 sem</th><th>10+ sem</th></tr>')

for r in REGIME_ORDER:
    ep = df_ep[df_ep['regime'] == r]
    if len(ep) == 0: continue
    persist = trans_pct.loc[r, r]
    n_ep = len(ep)
    avg_d = ep['weeks'].mean()
    med_d = ep['weeks'].median()
    max_d = ep['weeks'].max()
    n_1 = (ep['weeks'] == 1).sum()
    n_24 = ((ep['weeks'] >= 2) & (ep['weeks'] <= 4)).sum()
    n_510 = ((ep['weeks'] >= 5) & (ep['weeks'] <= 10)).sum()
    n_10p = (ep['weeks'] > 10).sum()
    html.append(f'<tr><td style="background:{REGIME_COLOR[r]};color:white;font-weight:bold">{r}</td>'
                f'<td><b>{persist:.0f}%</b></td><td>{n_ep}</td>'
                f'<td>{avg_d:.1f}</td><td>{med_d:.0f}</td><td>{max_d}</td>'
                f'<td>{n_1}</td><td>{n_24}</td><td>{n_510}</td><td>{n_10p}</td></tr>')
html.append('</table>')

# ================================================================
# 5. PROBABILIDAD ACUMULADA A 2, 3 Y 4 SEMANAS
# ================================================================
html.append('<h2>5. Probabilidad a 2, 3 y 4 Semanas Vista</h2>')
html.append('<p>Si hoy es viernes y el regimen es X, cual es la distribucion de probabilidad en 2/3/4 semanas?</p>')

# Matriz de transicion como numpy para potencias
trans_matrix = trans_pct.values / 100.0
# Asegurar que sume 1 por fila
for i in range(len(trans_matrix)):
    s = trans_matrix[i].sum()
    if s > 0:
        trans_matrix[i] /= s

for n_weeks in [2, 3, 4]:
    matrix_n = np.linalg.matrix_power(trans_matrix, n_weeks)
    html.append(f'<h3>En {n_weeks} semanas:</h3>')
    html.append('<table class="matrix-table"><tr><th>Desde</th>')
    for r in REGIME_ORDER:
        html.append(f'<th style="background:{REGIME_COLOR[r]}">{r[:4]}</th>')
    html.append('</tr>')
    for i, r_from in enumerate(REGIME_ORDER):
        if trans_count.loc[r_from].sum() == 0: continue
        html.append(f'<tr><td style="background:{REGIME_COLOR[r_from]};color:white;font-weight:bold">{r_from}</td>')
        for j, r_to in enumerate(REGIME_ORDER):
            pct = matrix_n[i, j] * 100
            if pct >= 50: cls = 'pct-high'
            elif pct >= 20: cls = 'pct-med'
            elif pct > 0.5: cls = 'pct-low'
            else: cls = 'pct-zero'
            if r_from == r_to: cls += ' pct-self'
            cell = f'{pct:.0f}%' if pct >= 1 else '-'
            html.append(f'<td class="{cls}">{cell}</td>')
        html.append('</tr>')
    html.append('</table>')

# ================================================================
# 6. TABLA RENTABILIDAD: Lunes open -> Lunes open
# ================================================================
html.append('<h2>6. Retorno SPY por Regimen (Lunes open &rarr; Lunes open)</h2>')
html.append('<p>Retorno real de trading: senal viernes &rarr; compra lunes open &rarr; venta lunes open (+1 semana). Incluye gap fin de semana.</p>')
html.append('<table><tr><th>Regimen actual</th><th>N</th><th>SPY avg %</th><th>SPY med %</th>'
            '<th>SPY std %</th><th>% semanas SPY+</th><th>Mejor</th><th>Peor</th></tr>')

for r in REGIME_ORDER:
    mask = df['regime'] == r
    if mask.sum() == 0: continue
    sub = df.loc[mask, 'spy_ret_mon'].dropna()
    if len(sub) == 0: continue
    avg = sub.mean()
    med = sub.median()
    std = sub.std()
    wr = (sub > 0).mean() * 100
    best = sub.max()
    worst = sub.min()
    avg_color = '#33cc33' if avg >= 0 else '#cc3333'
    html.append(f'<tr><td style="background:{REGIME_COLOR[r]};color:white;font-weight:bold">{r}</td>'
                f'<td>{len(sub)}</td>'
                f'<td style="color:{avg_color};font-weight:bold">{avg:+.2f}%</td>'
                f'<td>{med:+.2f}%</td><td>{std:.2f}%</td>'
                f'<td>{wr:.0f}%</td><td>{best:+.1f}%</td><td>{worst:+.1f}%</td></tr>')
html.append('</table>')

html.append('<div class="insight warning">Nota: Retorno real operativo Mon open &rarr; Mon open. '
            'Incluye el gap del fin de semana entre la senal (viernes) y la ejecucion (lunes).</div>')

html.append('</body></html>')

output_path = 'C:/Users/usuario/financial-data-project/analisis_transiciones.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))
print(f"HTML generado: {output_path}")
