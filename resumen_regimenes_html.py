"""
Resumen ejecutivo ano a ano de los regimenes de mercado - Version HTML
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

# Clasificar regimen
def classify_regime(date):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: return 'NEUTRAL', 0.0, {}
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]; rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL', 0.0, {}
    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50
    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) > 0:
        spy_last = spy_w.loc[spy_dates[-1]]
        spy_above_ma200 = spy_last.get('above_ma200', 0.5)
        spy_mom_10w = spy_last.get('mom_10w', 0)
        spy_dist = spy_last.get('dist_ma200', 0)
        spy_close = spy_last.get('close', 0)
    else:
        spy_above_ma200 = 0.5; spy_mom_10w = 0; spy_dist = 0; spy_close = 0
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
    return regime, total, {'score': total, 'vix': vix_val, 'spy_close': spy_close,
        'spy_dist': spy_dist, 'spy_mom': spy_mom_10w,
        'dd_healthy': pct_dd_healthy, 'dd_deep': pct_dd_deep, 'rsi_broad': pct_rsi_above55}

print("Clasificando regimenes...")
all_weeks = []
prev_vix_val = None
for date in dd_wide.index:
    if date.year < 2001:
        _, _, det = classify_regime(date)
        prev_vix_val = det.get('vix', 20)
        continue
    regime, score, details = classify_regime(date)
    vix_val = details.get('vix', 20)
    # CAPITULACION / RECOVERY: VIX bajando vs semana anterior
    if prev_vix_val is not None and vix_val < prev_vix_val:
        if regime == 'PANICO':
            regime = 'CAPITULACION'
        elif regime == 'BEARISH':
            regime = 'RECOVERY'
    prev_vix_val = vix_val
    spy_ret = 0
    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) >= 2:
        spy_ret = (spy_w.loc[spy_dates[-1], 'close'] / spy_w.loc[spy_dates[-2], 'close'] - 1) * 100
    all_weeks.append({'date': date, 'year': date.year, 'month': date.month,
        'week_num': date.isocalendar()[1], 'regime': regime, 'score': score,
        'spy_ret': spy_ret, **details})

df = pd.DataFrame(all_weeks)

# ================================================================
# GENERAR HTML
# ================================================================
REGIME_ORDER = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'RECOVERY', 'CRISIS', 'PANICO', 'CAPITULACION']
REGIME_COLOR = {
    'BURBUJA': '#ff6600', 'GOLDILOCKS': '#00aa00', 'ALCISTA': '#33cc33',
    'NEUTRAL': '#888888', 'CAUTIOUS': '#cc9900', 'BEARISH': '#cc3333',
    'RECOVERY': '#0088cc', 'CRISIS': '#990000', 'PANICO': '#660066',
    'CAPITULACION': '#0066cc'
}
REGIME_BG = {
    'BURBUJA': '#fff3e0', 'GOLDILOCKS': '#e8f5e9', 'ALCISTA': '#f1f8e9',
    'NEUTRAL': '#f5f5f5', 'CAUTIOUS': '#fff8e1', 'BEARISH': '#ffebee',
    'RECOVERY': '#e3f2fd', 'CRISIS': '#fce4ec', 'PANICO': '#f3e5f5',
    'CAPITULACION': '#e1f5fe'
}

html = []
html.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Regimenes de Mercado 2001-2026</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #fafafa; }
h1 { color: #333; border-bottom: 3px solid #333; padding-bottom: 10px; }
h2 { color: #555; margin-top: 30px; }
.year-card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 5px solid #333; }
.year-title { font-size: 24px; font-weight: bold; color: #333; }
.spy-info { font-size: 16px; color: #666; margin: 5px 0; }
.spy-pos { color: #00aa00; font-weight: bold; }
.spy-neg { color: #cc3333; font-weight: bold; }
.timeline { display: flex; gap: 2px; margin: 10px 0; }
.month-cell { width: 60px; height: 40px; display: flex; align-items: center; justify-content: center;
              border-radius: 4px; font-weight: bold; font-size: 12px; color: white; }
.month-labels { display: flex; gap: 2px; margin-bottom: 2px; }
.month-label { width: 60px; text-align: center; font-size: 11px; color: #999; }
.regime-counts { margin: 8px 0; font-size: 14px; }
.regime-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; margin: 2px;
                font-size: 12px; font-weight: bold; color: white; }
.quarter { margin: 5px 0; padding: 8px 12px; background: #f8f8f8; border-radius: 4px; font-size: 13px; }
.quarter-name { font-weight: bold; color: #555; }
.transitions { margin: 8px 0; font-size: 13px; color: #666; }
.trans-arrow { color: #333; font-weight: bold; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th { background: #333; color: white; padding: 8px 12px; text-align: center; font-size: 13px; }
td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; font-size: 13px; }
tr:hover { background: #f0f0f0; }
.legend { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 13px; }
.legend-dot { width: 16px; height: 16px; border-radius: 50%; }
.score-info { font-size: 13px; color: #888; }
.section-neg { border-left-color: #cc3333; }
.section-pos { border-left-color: #00aa00; }
.section-mix { border-left-color: #cc9900; }
.week-strip { display: flex; gap: 1px; margin: 8px 0; flex-wrap: wrap; }
.week-dot { width: 14px; height: 14px; border-radius: 2px; }
</style></head><body>
<h1>Regimenes de Mercado - Resumen Ejecutivo (2001-2026)</h1>
<p style="color:#666">Sistema de clasificacion basado en 5 indicadores: Breadth DD, Breadth RSI,
DD Deep%, SPY vs MA200, SPY Momentum 10 semanas. Rango de score: [-12.5, +8.0]</p>
<div class="legend">""")

for r in REGIME_ORDER:
    html.append(f'<div class="legend-item"><div class="legend-dot" style="background:{REGIME_COLOR[r]}"></div>{r}</div>')
html.append('</div>')

MONTHS = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year].sort_values('date')
    if len(yr) == 0: continue

    spy_yr = spy_w[(spy_w.index.year == year) & spy_w['ret_spy'].notna()]
    spy_annual = ((1 + spy_yr['ret_spy']).prod() - 1) * 100 if len(spy_yr) > 0 else 0
    spy_start = spy_yr.iloc[0]['close'] if len(spy_yr) > 0 else 0
    spy_end = spy_yr.iloc[-1]['close'] if len(spy_yr) > 0 else 0
    rc = yr['regime'].value_counts()
    avg_score = yr['score'].mean()
    avg_vix = yr['vix'].mean()

    n_pos = sum(rc.get(r, 0) for r in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'RECOVERY', 'CAPITULACION'])
    n_neg = sum(rc.get(r, 0) for r in ['CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO']
)
    if n_neg > n_pos * 1.5: section_class = 'section-neg'
    elif n_pos > n_neg * 1.5: section_class = 'section-pos'
    else: section_class = 'section-mix'

    spy_class = 'spy-pos' if spy_annual >= 0 else 'spy-neg'

    html.append(f'<div class="year-card {section_class}">')
    html.append(f'<div class="year-title">{year}</div>')
    html.append(f'<div class="spy-info">S&P 500: <span class="{spy_class}">{spy_annual:+.1f}%</span> '
                f'({spy_start:.0f} &rarr; {spy_end:.0f}) &nbsp;|&nbsp; '
                f'Score avg: {avg_score:+.1f} &nbsp;|&nbsp; VIX avg: {avg_vix:.0f} &nbsp;|&nbsp; '
                f'{len(yr)} semanas</div>')

    # Timeline mensual
    html.append('<div class="month-labels">')
    for m in MONTHS:
        html.append(f'<div class="month-label">{m}</div>')
    html.append('</div><div class="timeline">')
    for month in range(1, 13):
        m_data = yr[yr['month'] == month]
        if len(m_data) == 0:
            html.append('<div class="month-cell" style="background:#ddd">&nbsp;</div>')
        else:
            dom = m_data['regime'].value_counts().index[0]
            html.append(f'<div class="month-cell" style="background:{REGIME_COLOR[dom]}">{dom[:3]}</div>')
    html.append('</div>')

    # Tira semanal (cada semana un cuadradito)
    html.append('<div class="week-strip">')
    for _, row in yr.iterrows():
        r = row['regime']
        title = f"Sem {row['week_num']}: {r} (score {row['score']:+.1f}, VIX {row['vix']:.0f})"
        html.append(f'<div class="week-dot" style="background:{REGIME_COLOR[r]}" title="{title}"></div>')
    html.append('</div>')

    # Badges por regimen
    html.append('<div class="regime-counts">')
    for r in REGIME_ORDER:
        n = rc.get(r, 0)
        if n > 0:
            html.append(f'<span class="regime-badge" style="background:{REGIME_COLOR[r]}">{r}: {n}</span>')
    html.append('</div>')

    # Transiciones significativas
    transitions = []
    prev_reg = None
    for _, row in yr.iterrows():
        if row['regime'] != prev_reg:
            if prev_reg is not None:
                transitions.append((row['date'].strftime('%d/%m'), prev_reg, row['regime']))
            prev_reg = row['regime']

    positive = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'RECOVERY', 'CAPITULACION'}
    neutral = {'NEUTRAL'}
    negative = {'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO'}
    def zone(r):
        if r in positive: return 'POS'
        if r in neutral: return 'NEU'
        return 'NEG'

    sig_trans = []
    for date_str, fr, to in transitions:
        if zone(fr) != zone(to) or (fr in negative and to in negative and fr != to):
            sig_trans.append(f'{date_str}: <span style="color:{REGIME_COLOR[fr]}">{fr}</span> '
                           f'<span class="trans-arrow">&rarr;</span> '
                           f'<span style="color:{REGIME_COLOR[to]}">{to}</span>')
    if sig_trans:
        html.append(f'<div class="transitions"><b>Transiciones:</b> {" &nbsp;|&nbsp; ".join(sig_trans)}</div>')

    # Trimestres
    for q in range(1, 5):
        q_data = yr[yr['month'].between((q-1)*3+1, q*3)]
        if len(q_data) == 0: continue
        q_rc = q_data['regime'].value_counts()
        q_spy = q_data['spy_ret'].sum()
        q_vix = q_data['vix'].mean()
        q_name = ['Q1 (Ene-Mar)', 'Q2 (Abr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dic)'][q-1]
        spy_q_class = 'spy-pos' if q_spy >= 0 else 'spy-neg'
        badges = ''
        for r in REGIME_ORDER:
            n = q_rc.get(r, 0)
            if n > 0:
                badges += f'<span class="regime-badge" style="background:{REGIME_COLOR[r]};font-size:11px">{r[:3]}:{n}</span>'
        html.append(f'<div class="quarter"><span class="quarter-name">{q_name}:</span> '
                    f'SPY <span class="{spy_q_class}">{q_spy:+.1f}%</span> &nbsp; VIX {q_vix:.0f} &nbsp; {badges}</div>')

    html.append('</div>')

# Tabla resumen
html.append('<h2>Tabla Resumen: Semanas por Regimen y Ano</h2>')
html.append('<table><tr><th>Ano</th><th>SPY%</th>')
for r in REGIME_ORDER:
    html.append(f'<th style="background:{REGIME_COLOR[r]}">{r[:4]}</th>')
html.append('<th>TOTAL</th><th>Zona+</th><th>Neutral</th><th>Zona-</th></tr>')

totals = {r: 0 for r in REGIME_ORDER}
for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year]
    rc = yr['regime'].value_counts()
    spy_yr = spy_w[(spy_w.index.year == year) & spy_w['ret_spy'].notna()]
    spy_annual = ((1 + spy_yr['ret_spy']).prod() - 1) * 100 if len(spy_yr) > 0 else 0
    spy_class = 'spy-pos' if spy_annual >= 0 else 'spy-neg'

    html.append(f'<tr><td><b>{year}</b></td><td class="{spy_class}">{spy_annual:+.1f}%</td>')
    for r in REGIME_ORDER:
        n = rc.get(r, 0)
        totals[r] += n
        if n > 0:
            html.append(f'<td style="background:{REGIME_BG[r]};font-weight:bold">{n}</td>')
        else:
            html.append('<td style="color:#ddd">-</td>')
    n_total = len(yr)
    n_pos = sum(rc.get(r, 0) for r in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'RECOVERY', 'CAPITULACION'])
    n_neu = rc.get('NEUTRAL', 0)
    n_neg = sum(rc.get(r, 0) for r in ['CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO']
)
    html.append(f'<td><b>{n_total}</b></td><td style="color:#00aa00">{n_pos}</td>'
                f'<td style="color:#888">{n_neu}</td><td style="color:#cc3333">{n_neg}</td></tr>')

# Fila totales
n_all = sum(totals.values())
html.append('<tr style="font-weight:bold;background:#eee"><td>TOTAL</td><td></td>')
for r in REGIME_ORDER:
    pct = totals[r] / n_all * 100
    html.append(f'<td>{totals[r]}<br><small>{pct:.1f}%</small></td>')
n_pos = sum(totals[r] for r in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'RECOVERY', 'CAPITULACION'])
n_neu = totals['NEUTRAL']
n_neg = sum(totals[r] for r in ['CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO']
)
html.append(f'<td>{n_all}</td><td style="color:#00aa00">{n_pos}<br><small>{n_pos/n_all*100:.1f}%</small></td>'
            f'<td style="color:#888">{n_neu}<br><small>{n_neu/n_all*100:.1f}%</small></td>'
            f'<td style="color:#cc3333">{n_neg}<br><small>{n_neg/n_all*100:.1f}%</small></td></tr>')
html.append('</table>')

# Detalle semanal por ano
html.append('<h2>Detalle Semanal por Ano</h2>')
for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year].sort_values('date')
    if len(yr) == 0: continue
    html.append(f'<h3>{year}</h3>')
    html.append('<table><tr><th>Sem</th><th>Fecha</th><th>Regimen</th><th>Score</th>'
                '<th>VIX</th><th>SPY</th><th>SPY dist MA200</th><th>SPY mom 10w</th>'
                '<th>DD Healthy%</th><th>DD Deep%</th><th>RSI Broad%</th></tr>')
    for _, row in yr.iterrows():
        r = row['regime']
        html.append(f'<tr style="background:{REGIME_BG[r]}">'
                    f'<td>{row["week_num"]}</td>'
                    f'<td>{row["date"].strftime("%d/%m/%Y")}</td>'
                    f'<td style="color:{REGIME_COLOR[r]};font-weight:bold">{r}</td>'
                    f'<td>{row["score"]:+.1f}</td>'
                    f'<td>{row["vix"]:.0f}</td>'
                    f'<td>{row["spy_close"]:.0f}</td>'
                    f'<td>{row["spy_dist"]:+.1f}%</td>'
                    f'<td>{row["spy_mom"]:+.1f}%</td>'
                    f'<td>{row["dd_healthy"]:.0f}%</td>'
                    f'<td>{row["dd_deep"]:.0f}%</td>'
                    f'<td>{row["rsi_broad"]:.0f}%</td></tr>')
    html.append('</table>')

html.append('</body></html>')

output_path = 'C:/Users/usuario/financial-data-project/resumen_regimenes.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))
print(f"HTML generado: {output_path}")
