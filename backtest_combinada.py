#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtest Combinada: E2 Semanal (regimen) + MIX Mensual (momentum 12M+3M)
Con calculo de drawdowns completos
Capital: $400K E2 + $400K MIX = $800K combinado
"""
import re, json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
COST = 20000   # $ por accion
SLIP = 0.003   # 0.3% = $60/trade
CAP_E2 = 400000   # 20 acciones × $20K
CAP_MIX = 400000  # 20 acciones × $20K
CAP_COMB = CAP_E2 + CAP_MIX

REGIME_COLORS = {
    'BURBUJA': '#ff6b6b', 'GOLDILOCKS': '#ffd93d', 'ALCISTA': '#6bcb77',
    'NEUTRAL': '#4d96ff', 'CAUTIOUS': '#ff922b', 'BEARISH': '#c084fc',
    'CRISIS': '#ff6b6b', 'PANICO': '#dc2626', 'RECOVERY': '#22d3ee',
    'CAPITULACION': '#f472b6'
}
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS',
                'BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']

# ── E2 Strategy ──────────────────────────────────────────────────────
STRAT_E2 = {
    'BURBUJA':      [(0, 10, 'long'),  (-20, -10, 'short')],
    'GOLDILOCKS':   [(0, 10, 'long'),  (-20, -10, 'short')],
    'ALCISTA':      [(0, 10, 'long'),  (-10, None, 'short')],
    'NEUTRAL':      [(10, 20, 'long'), (-20, -10, 'short')],
    'CAUTIOUS':     [(-10, None, 'short'), (-20, -10, 'short')],
    'BEARISH':      [(-10, None, 'short'), (-20, -10, 'short')],
    'CRISIS':       [(0, 10, 'short'), (10, 20, 'short')],
    'PANICO':       [(0, 10, 'short'), (10, 20, 'short')],
    'RECOVERY':     [(0, 10, 'long'),  (-20, -10, 'long')],
    'CAPITULACION': [(10, 20, 'long'), (20, 30, 'long')],
}

# ── Load data ────────────────────────────────────────────────────────
print("Loading acciones_navegable.html (E2)...")
with open(BASE / 'acciones_navegable.html', 'r', encoding='utf-8') as f:
    html_text = f.read()
m = re.search(r'const T\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
T = json.loads(m.group(1))
m2 = re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
WEEKS = json.loads(m2.group(1))
del html_text

print("Loading momentum_mensual_mix.html (MIX)...")
with open(BASE / 'momentum_mensual_mix.html', 'r', encoding='utf-8') as f:
    html_text = f.read()
m = re.search(r'const D\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
MIX_MONTHS = json.loads(m.group(1))
m2 = re.search(r'const YS\s*=\s*(\{.+?\});\s*\n', html_text, re.DOTALL)
MIX_YS = json.loads(m2.group(1))
del html_text
print(f"  E2: {len(WEEKS)} weeks, MIX: {len(MIX_MONTHS)} months")

# ── E2: Run weekly backtest ──────────────────────────────────────────
print("\nRunning E2 weekly...")
e2_weekly = []
for w in WEEKS:
    date, year, sem, regime = w['d'], w['y'], w['w'], w['r']
    spy_ret = w.get('sr') or 0
    stocks = w['s']
    strat = STRAT_E2.get(regime, [])
    pnl = 0
    n_pos = 0
    if strat:
        for gi, (start, end, direction) in enumerate(strat):
            selected = stocks[start:end] if end is not None else stocks[start:]
            for s in selected:
                ret_val = s[8]
                if ret_val is None:
                    continue
                ret_val = max(-50, min(50, ret_val))
                if direction == 'long':
                    pnl += COST * (ret_val / 100 - SLIP)
                else:
                    pnl += COST * (-ret_val / 100 - SLIP)
                n_pos += 1
    month_key = date[:7]
    e2_weekly.append({
        'date': date, 'year': year, 'sem': sem, 'regime': regime,
        'spy_ret': spy_ret, 'pnl': round(pnl, 2), 'n_pos': n_pos,
        'month': month_key
    })

# ── MIX: Calculate monthly PnL (SIN clipping, con slippage) ─────────
print("Calculating MIX monthly...")
mix_monthly = {}
for d in MIX_MONTHS:
    month_key = d['m']
    pnl = 0
    n_pos = 0
    for s in d['top'] + d['bot']:
        ret = s[3]
        if ret is not None:
            pnl += COST * (ret / 100 - SLIP)  # SIN clip
            n_pos += 1
    mix_monthly[month_key] = {
        'pnl': round(pnl, 2), 'n_pos': n_pos, 'year': d['y'],
        'regime': d.get('reg', '')
    }

# Verify vs HTML YS
print("  Verificando vs HTML:")
for yr_str, ys in sorted(MIX_YS.items()):
    yr = int(yr_str)
    my_pnl = sum(v['pnl'] for v in mix_monthly.values() if v['year'] == yr)
    html_pnl = ys['cmb_pnl']
    diff = abs(my_pnl - html_pnl)
    ok = 'OK' if diff < 100 else f'DIFF={diff:.0f}'
    print(f"    {yr}: mine=${my_pnl:>10,.0f}  html=${html_pnl:>10,.0f}  {ok}")

# ── Aggregate E2 to monthly ─────────────────────────────────────────
e2_monthly = defaultdict(lambda: {'pnl': 0, 'n_weeks': 0, 'regimes': set()})
for w in e2_weekly:
    mk = w['month']
    e2_monthly[mk]['pnl'] += w['pnl']
    e2_monthly[mk]['n_weeks'] += 1
    e2_monthly[mk]['regimes'].add(w['regime'])

# ── Combine monthly ─────────────────────────────────────────────────
all_months = sorted(set(list(e2_monthly.keys()) + list(mix_monthly.keys())))
combined = []
for mk in all_months:
    e2 = e2_monthly.get(mk, {'pnl': 0, 'n_weeks': 0, 'regimes': set()})
    mx = mix_monthly.get(mk, {'pnl': 0, 'n_pos': 0})
    year = int(mk[:4])
    month = int(mk[5:7])
    combined.append({
        'm': mk, 'y': year, 'mn': month,
        'e2': round(e2['pnl'], 2), 'mix': round(mx['pnl'], 2),
        'total': round(e2['pnl'] + mx['pnl'], 2),
        'regs': list(e2.get('regimes', set()))
    })

# ── Year stats (capital fijo: E2=$400K, MIX=$400K, COMB=$800K) ──────
years = sorted(set(c['y'] for c in combined))
year_stats = {}
for y in years:
    ms = [c for c in combined if c['y'] == y]
    e2_y = sum(c['e2'] for c in ms)
    mix_y = sum(c['mix'] for c in ms)
    tot_y = e2_y + mix_y
    n = len(ms)
    e2_win = sum(1 for c in ms if c['e2'] > 0)
    mix_win = sum(1 for c in ms if c['mix'] > 0)
    tot_win = sum(1 for c in ms if c['total'] > 0)
    ret_e2 = e2_y / CAP_E2 * 100
    ret_mix = mix_y / CAP_MIX * 100
    ret_tot = tot_y / CAP_COMB * 100
    reg_dist = {}
    for c in ms:
        for r in c['regs']:
            reg_dist[r] = reg_dist.get(r, 0) + 1
    year_stats[y] = {
        'n': n, 'e2': round(e2_y), 'mix': round(mix_y), 'total': round(tot_y),
        'e2_ret': round(ret_e2, 1), 'mix_ret': round(ret_mix, 1), 'tot_ret': round(ret_tot, 1),
        'e2_wr': round(e2_win / n * 100, 1), 'mix_wr': round(mix_win / n * 100, 1),
        'tot_wr': round(tot_win / n * 100, 1),
        'dist': reg_dist
    }

# ── Drawdowns ────────────────────────────────────────────────────────
def calc_drawdowns(pnl_series):
    equity = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    dd_pct = np.where(peak > 0, drawdown / peak * 100, 0)
    max_dd = float(drawdown.min())
    # Drawdown periods
    dd_periods = []
    in_dd = False
    dd_start = 0
    for i in range(len(drawdown)):
        if drawdown[i] < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif drawdown[i] >= 0 and in_dd:
            in_dd = False
            dd_periods.append((dd_start, i - 1, float(drawdown[dd_start:i].min())))
    if in_dd:
        dd_periods.append((dd_start, len(drawdown) - 1, float(drawdown[dd_start:].min())))
    dd_periods.sort(key=lambda x: x[2])
    return {
        'eq': [round(float(v)) for v in equity],
        'dd': [round(float(v)) for v in drawdown],
        'max_dd': round(max_dd),
        'top5': [(int(s), int(e), round(d)) for s, e, d in dd_periods[:5]],
        'n_dd': int(np.sum(drawdown < 0)),
        'pct_dd': round(float(np.sum(drawdown < 0) / len(drawdown) * 100), 1)
    }

e2_pnl = np.array([c['e2'] for c in combined])
mix_pnl = np.array([c['mix'] for c in combined])
total_pnl = np.array([c['total'] for c in combined])

dd_e2 = calc_drawdowns(e2_pnl)
dd_mix = calc_drawdowns(mix_pnl)
dd_total = calc_drawdowns(total_pnl)

# ── Print results ────────────────────────────────────────────────────
g_e2 = round(sum(c['e2'] for c in combined))
g_mix = round(sum(c['mix'] for c in combined))
g_total = g_e2 + g_mix
n_months = len(combined)

print(f"\n{'='*95}")
print(f"{'ESTRATEGIA COMBINADA: E2 + MIX':^95}")
print(f"{'='*95}")
print(f"{'':>16} {'E2 ($400K)':>16} {'MIX ($400K)':>16} {'COMB ($800K)':>16}")
print(f"{'-'*95}")
print(f"{'PnL Total':>16} ${g_e2:>14,} ${g_mix:>14,} ${g_total:>14,}")
print(f"{'Ret Total':>16} {g_e2/CAP_E2*100:>14.1f}% {g_mix/CAP_MIX*100:>14.1f}% {g_total/CAP_COMB*100:>14.1f}%")
print(f"{'Avg Anual':>16} {g_e2/CAP_E2*100/len(years):>14.1f}% {g_mix/CAP_MIX*100/len(years):>14.1f}% {g_total/CAP_COMB*100/len(years):>14.1f}%")
print(f"{'Max Drawdown':>16} ${dd_e2['max_dd']:>14,} ${dd_mix['max_dd']:>14,} ${dd_total['max_dd']:>14,}")
print(f"{'Max DD %':>16} {dd_e2['max_dd']/CAP_E2*100:>14.1f}% {dd_mix['max_dd']/CAP_MIX*100:>14.1f}% {dd_total['max_dd']/CAP_COMB*100:>14.1f}%")
print(f"{'Meses en DD':>16} {dd_e2['pct_dd']:>14.1f}% {dd_mix['pct_dd']:>14.1f}% {dd_total['pct_dd']:>14.1f}%")

print(f"\n{'RETORNOS ANUALES':^95}")
print(f"{'Ano':>6} {'E2 PnL':>10} {'E2%':>8} {'MIX PnL':>10} {'MIX%':>8} {'COMB PnL':>10} {'COMB%':>8} {'WR':>5} {'Resultado':>10}")
print(f"{'-'*95}")
wins, losses = 0, 0
for y in years:
    s = year_stats[y]
    res = 'GANANCIA' if s['total'] > 0 else 'PERDIDA'
    if s['total'] > 0: wins += 1
    else: losses += 1
    print(f"{y:>6} ${s['e2']:>8,} {s['e2_ret']:>7.1f}% ${s['mix']:>8,} {s['mix_ret']:>7.1f}% "
          f"${s['total']:>8,} {s['tot_ret']:>7.1f}% {s['tot_wr']:>4.0f}% {res:>10}")
print(f"{'-'*95}")
print(f"{'TOTAL':>6} ${g_e2:>8,} {g_e2/CAP_E2*100:>7.1f}% ${g_mix:>8,} {g_mix/CAP_MIX*100:>7.1f}% "
      f"${g_total:>8,} {g_total/CAP_COMB*100:>7.1f}%")
print(f"\nAnos ganadores: {wins} | Perdedores: {losses} | Ratio: {wins}/{wins+losses}")

print(f"\n{'DRAWDOWNS TOP 5 (COMBINADA)':^95}")
for i, (s, e, d) in enumerate(dd_total['top5']):
    ms_s, ms_e = combined[s]['m'], combined[e]['m']
    dur = e - s + 1
    dd_pct = d / CAP_COMB * 100
    print(f"  {i+1}. {ms_s} -> {ms_e}  ({dur} meses)  ${d:>10,}  ({dd_pct:.1f}%)")

# ── Generate HTML ────────────────────────────────────────────────────
print("\nGenerating HTML...")

comb_json = json.dumps([{
    'm': c['m'], 'y': c['y'], 'e2': round(c['e2']), 'mix': round(c['mix']),
    'tot': round(c['total']), 'regs': c['regs']
} for c in combined], separators=(',',':'))

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Combinada: E2 + MIX</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:14px}}
h1{{text-align:center;font-size:1.4em;margin-bottom:4px;color:#f8fafc}}
.sub{{text-align:center;color:#94a3b8;margin-bottom:14px;font-size:0.82em}}
.card{{background:#1e293b;border-radius:10px;padding:16px;margin-bottom:12px;border:1px solid #334155}}
.card h2{{font-size:1.05em;color:#f1f5f9;margin-bottom:8px;border-bottom:1px solid #334155;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:0.8em}}
th{{background:#334155;color:#f1f5f9;padding:6px 4px;text-align:left;position:sticky;top:0}}
td{{padding:5px 4px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#334155}}
.pos{{color:#22c55e}}.neg{{color:#ef4444}}
.badge{{display:inline-block;padding:2px 7px;border-radius:9px;font-size:0.72em;font-weight:600;color:#000}}
.badge-sm{{padding:1px 4px;font-size:0.67em;border-radius:7px}}
.side3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}}
@media(max-width:900px){{.side3{{grid-template-columns:1fr}}}}
.gv-row{{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:8px}}
.gv{{background:#0f172a;border-radius:7px;padding:8px 12px;text-align:center;min-width:110px}}
.gv .val{{font-size:1.15em;font-weight:700}}.gv .lbl{{font-size:0.73em;color:#94a3b8;margin-top:2px}}
.chart-wrap{{position:relative;height:250px;background:#0f172a;border-radius:7px;overflow:hidden;margin-bottom:8px}}
.chart-wrap canvas{{width:100%!important;height:100%!important}}
.legend{{text-align:center;font-size:0.73em;color:#94a3b8;margin-top:4px}}
.e2c{{color:#60a5fa}}.mixc{{color:#a78bfa}}.totc{{color:#fbbf24}}.ddc{{color:#ef4444}}
.win-row td{{background:rgba(34,197,94,0.08)}}.loss-row td{{background:rgba(239,68,68,0.08)}}
</style>
</head>
<body>
<h1>Estrategia Combinada: E2 Semanal + MIX Mensual</h1>
<div class="sub">E2: Regimen semanal ($400K) + MIX: Momentum 12M+3M mensual ($400K) = $800K capital total | $60/trade slippage</div>

<div class="card">
<h2>1. Resumen</h2>
<div class="side3">
<div style="border-left:3px solid #60a5fa;padding-left:10px">
<h3 class="e2c" style="font-size:0.9em;margin-bottom:6px">E2 Semanal ($400K)</h3>
<div class="gv-row" id="gs_e2"></div>
</div>
<div style="border-left:3px solid #a78bfa;padding-left:10px">
<h3 class="mixc" style="font-size:0.9em;margin-bottom:6px">MIX Mensual ($400K)</h3>
<div class="gv-row" id="gs_mix"></div>
</div>
<div style="border-left:3px solid #fbbf24;padding-left:10px">
<h3 class="totc" style="font-size:0.9em;margin-bottom:6px">COMBINADA ($800K)</h3>
<div class="gv-row" id="gs_tot"></div>
</div>
</div>
</div>

<div class="card">
<h2>2. PnL Acumulado</h2>
<div class="chart-wrap"><canvas id="eqChart"></canvas></div>
<div class="legend"><span class="e2c">&#9644; E2</span> &nbsp; <span class="mixc">&#9644; MIX</span> &nbsp; <span class="totc">&#9644; COMBINADA</span></div>
</div>

<div class="card">
<h2>3. Drawdown ($)</h2>
<div class="chart-wrap"><canvas id="ddChart"></canvas></div>
<div class="legend"><span class="e2c">&#9644; E2</span> &nbsp; <span class="mixc">&#9644; MIX</span> &nbsp; <span class="totc">&#9644; COMBINADA</span></div>
<div style="margin-top:10px" id="dd-stats"></div>
</div>

<div class="card">
<h2>4. Retornos Anuales</h2>
<table>
<thead><tr><th>Ano</th><th>E2 PnL</th><th>E2 Ret%</th><th>MIX PnL</th><th>MIX Ret%</th><th>Comb PnL</th><th>Comb Ret%</th><th>WR Mes</th><th>Resultado</th></tr></thead>
<tbody id="yr-body"></tbody>
<tfoot id="yr-foot"></tfoot>
</table>
</div>

<div class="card">
<h2>5. Top 5 Peores Drawdowns (Combinada)</h2>
<table>
<thead><tr><th>#</th><th>Inicio</th><th>Fin</th><th>Meses</th><th>Max DD ($)</th><th>Max DD (%)</th><th>Recuperacion</th></tr></thead>
<tbody id="dd-body"></tbody>
</table>
</div>

<div class="card">
<h2>6. Detalle Mensual</h2>
<div style="max-height:500px;overflow-y:auto">
<table>
<thead><tr><th>Mes</th><th>E2</th><th>MIX</th><th>Combinado</th><th>E2 Acum</th><th>MIX Acum</th><th>Comb Acum</th><th>DD Comb</th></tr></thead>
<tbody id="mn-body"></tbody>
</table>
</div>
</div>

<script>
const RC={json.dumps(REGIME_COLORS)};
const RO={json.dumps(REGIME_ORDER)};
const C={comb_json};
const DE2={json.dumps(dd_e2)};
const DMX={json.dumps(dd_mix)};
const DT={json.dumps(dd_total)};
const YS={json.dumps(year_stats)};
const CAP_E2={CAP_E2},CAP_MIX={CAP_MIX},CAP_COMB={CAP_COMB};
const fmt=v=>{{let s=v<0?'-$':'$';return s+Math.abs(Math.round(v)).toLocaleString('en-US')}};
const cls=v=>v>=0?'pos':'neg';

// ── 1. Summary ──
function gvs(id,pnl,cap,maxdd,pctdd,n_months){{
  let ret=(pnl/cap*100).toFixed(1),avg=(pnl/cap*100/26).toFixed(1);
  let h='';
  [['PnL Total',fmt(pnl),cls(pnl)],['Ret Total',ret+'%',cls(pnl)],
   ['Avg Anual',avg+'%/yr',cls(pnl)],['Max DD',fmt(maxdd)+' ('+(maxdd/cap*100).toFixed(1)+'%)','neg'],
   ['Meses DD',pctdd+'%','neg']].forEach(([l,v,c])=>{{
    h+=`<div class="gv"><div class="val ${{c}}">${{v}}</div><div class="lbl">${{l}}</div></div>`}});
  document.getElementById(id).innerHTML=h;
}}
gvs('gs_e2',{g_e2},CAP_E2,{dd_e2['max_dd']},{dd_e2['pct_dd']},{n_months});
gvs('gs_mix',{g_mix},CAP_MIX,{dd_mix['max_dd']},{dd_mix['pct_dd']},{n_months});
gvs('gs_tot',{g_total},CAP_COMB,{dd_total['max_dd']},{dd_total['pct_dd']},{n_months});

// ── 2+3. Charts ──
function drawChart(canvasId,datasets,isDD){{
  const canvas=document.getElementById(canvasId);
  const ctx=canvas.getContext('2d');
  function draw(){{
    const W=canvas.parentElement.clientWidth,H=canvas.parentElement.clientHeight;
    canvas.width=W*2;canvas.height=H*2;ctx.scale(2,2);
    let allV=[];datasets.forEach(d=>allV.push(...d.data));
    if(!isDD)allV.push(0);
    const mn=Math.min(...allV),mx=Math.max(...allV,0);
    const pad={{t:16,b:26,l:70,r:16}};
    const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
    const range=mx-mn||1;
    const n=datasets[0].data.length;
    const xS=cw/n,yS=ch/range;
    ctx.clearRect(0,0,W,H);
    ctx.strokeStyle='#334155';ctx.lineWidth=0.5;
    for(let i=0;i<=4;i++){{
      let yv=mn+(range/4)*i,yy=pad.t+ch-(yv-mn)*yS;
      ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();
      ctx.fillStyle='#94a3b8';ctx.font='10px sans-serif';ctx.textAlign='right';
      ctx.fillText(fmt(yv),pad.l-4,yy+3);
    }}
    if(!isDD){{let y0=pad.t+ch-(0-mn)*yS;ctx.strokeStyle='#64748b';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(pad.l,y0);ctx.lineTo(W-pad.r,y0);ctx.stroke()}}
    datasets.forEach(d=>{{
      ctx.strokeStyle=d.color;ctx.lineWidth=d.width||1.5;ctx.beginPath();
      d.data.forEach((v,i)=>{{let x=pad.l+i*xS,y=pad.t+ch-(v-mn)*yS;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}});
      ctx.stroke();
    }});
    ctx.fillStyle='#64748b';ctx.font='9px sans-serif';ctx.textAlign='center';
    let lastY='';
    C.forEach((c,i)=>{{if(String(c.y)!==lastY){{lastY=String(c.y);ctx.fillText(c.y,pad.l+i*xS,H-pad.b+12)}}}});
  }}
  draw();window.addEventListener('resize',draw);
}}
drawChart('eqChart',[
  {{data:DE2.eq,color:'rgb(96,165,250)',width:1.2}},
  {{data:DMX.eq,color:'rgb(167,139,250)',width:1.2}},
  {{data:DT.eq,color:'rgb(251,191,36)',width:2}}
],false);
drawChart('ddChart',[
  {{data:DE2.dd,color:'rgb(96,165,250)',width:1}},
  {{data:DMX.dd,color:'rgb(167,139,250)',width:1}},
  {{data:DT.dd,color:'rgb(251,191,36)',width:1.5}}
],true);

// ── DD Stats ──
(function(){{
  let h='<table><tr><th></th><th>Max DD ($)</th><th>Max DD (%)</th><th>Meses en DD</th></tr>';
  [['E2',DE2,CAP_E2,'e2c'],['MIX',DMX,CAP_MIX,'mixc'],['COMBINADA',DT,CAP_COMB,'totc']].forEach(([l,d,cap,c])=>{{
    let ndd=d.dd.filter(v=>v<0).length;
    h+=`<tr><td class="${{c}}"><strong>${{l}}</strong></td>
    <td class="neg">${{fmt(d.max_dd)}}</td>
    <td class="neg">${{(d.max_dd/cap*100).toFixed(1)}}%</td>
    <td>${{ndd}}/${{d.dd.length}} (${{(ndd/d.dd.length*100).toFixed(1)}}%)</td></tr>`;
  }});
  h+='</table>';
  document.getElementById('dd-stats').innerHTML=h;
}})();

// ── 4. Year Table ──
(function(){{
  let yrs=Object.keys(YS).sort(),b='',te2=0,tmx=0,tt=0,w=0,l=0;
  yrs.forEach(y=>{{
    let s=YS[y];te2+=s.e2;tmx+=s.mix;tt+=s.total;
    let isW=s.total>0;if(isW)w++;else l++;
    let pills='';
    RO.forEach(r=>{{if(s.dist&&s.dist[r])pills+=`<span class="badge badge-sm" style="background:${{RC[r]}}">${{r[0]}}${{s.dist[r]}}</span> `}});
    b+=`<tr class="${{isW?'win-row':'loss-row'}}"><td>${{y}}</td>
    <td class="${{cls(s.e2)}}">${{fmt(s.e2)}}</td><td class="${{cls(s.e2_ret)}}">${{s.e2_ret>=0?'+':''}}${{s.e2_ret}}%</td>
    <td class="${{cls(s.mix)}}">${{fmt(s.mix)}}</td><td class="${{cls(s.mix_ret)}}">${{s.mix_ret>=0?'+':''}}${{s.mix_ret}}%</td>
    <td class="${{cls(s.total)}}"><strong>${{fmt(s.total)}}</strong></td>
    <td class="${{cls(s.tot_ret)}}"><strong>${{s.tot_ret>=0?'+':''}}${{s.tot_ret}}%</strong></td>
    <td>${{s.tot_wr}}%</td>
    <td style="font-weight:700;color:${{isW?'#22c55e':'#ef4444'}}">${{isW?'GANANCIA':'PERDIDA'}}</td></tr>`;
  }});
  document.getElementById('yr-body').innerHTML=b;
  document.getElementById('yr-foot').innerHTML=
    `<tr style="font-weight:700;border-top:2px solid #475569"><td>TOTAL</td>
    <td class="${{cls(te2)}}">${{fmt(te2)}}</td><td>${{(te2/CAP_E2*100).toFixed(1)}}%</td>
    <td class="${{cls(tmx)}}">${{fmt(tmx)}}</td><td>${{(tmx/CAP_MIX*100).toFixed(1)}}%</td>
    <td class="${{cls(tt)}}"><strong>${{fmt(tt)}}</strong></td><td>${{(tt/CAP_COMB*100).toFixed(1)}}%</td>
    <td></td><td>${{w}}W / ${{l}}L</td></tr>`;
}})();

// ── 5. DD Table ──
(function(){{
  let top5=DT.top5||{json.dumps(dd_total['top5'])};
  let months=C.map(c=>c.m);
  let b='';
  top5.forEach((dd,i)=>{{
    let [s,e,d]=dd;
    let dur=e-s+1;
    let eq=DT.eq;
    let peakVal=Math.max(...eq.slice(0,s+1));
    let recIdx=eq.findIndex((v,j)=>j>e&&v>=peakVal);
    let recStr=recIdx>=0?months[recIdx]:'No recuperado';
    let recDur=recIdx>=0?(recIdx-e)+' meses':'';
    b+=`<tr><td>${{i+1}}</td><td>${{months[s]}}</td><td>${{months[e]}}</td>
    <td>${{dur}} meses</td><td class="neg">${{fmt(d)}}</td>
    <td class="neg">${{(d/CAP_COMB*100).toFixed(1)}}%</td>
    <td>${{recStr}} ${{recDur}}</td></tr>`;
  }});
  document.getElementById('dd-body').innerHTML=b;
}})();

// ── 6. Monthly ──
(function(){{
  let b='',ce2=0,cmx=0,ct=0;
  C.forEach((c,i)=>{{
    ce2+=c.e2;cmx+=c.mix;ct+=c.tot;
    let dd=DT.dd[i];
    b+=`<tr><td>${{c.m}}</td>
    <td class="${{cls(c.e2)}}">${{fmt(c.e2)}}</td>
    <td class="${{cls(c.mix)}}">${{fmt(c.mix)}}</td>
    <td class="${{cls(c.tot)}}"><strong>${{fmt(c.tot)}}</strong></td>
    <td class="${{cls(ce2)}}">${{fmt(ce2)}}</td>
    <td class="${{cls(cmx)}}">${{fmt(cmx)}}</td>
    <td class="${{cls(ct)}}">${{fmt(ct)}}</td>
    <td class="${{dd<0?'neg':''}}">${{dd<0?fmt(dd):''}}</td></tr>`;
  }});
  document.getElementById('mn-body').innerHTML=b;
}})();
</script>
</body>
</html>"""

out_path = BASE / 'backtest_combinada.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML saved to: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
