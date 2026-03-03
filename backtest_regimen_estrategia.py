#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtest: Estrategia por Regimen de Mercado - Acciones Individuales S&P 500
Datos: acciones_navegable.html (rankings FVA + retornos Fri close -> Fri close)
"""
import re, json, sys
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

# ── 1. Strategy definition ──────────────────────────────────────────
# (start_idx, end_idx, direction) sobre ranking FVA (0=best)
STRATEGY = {
    'BURBUJA':      [(0, 10, 'long'),  (-20, -10, 'short')],
    'GOLDILOCKS':   [(0, 10, 'long'),  (-20, -10, 'short')],
    'ALCISTA':      [(0, 10, 'long'),  (-10, None, 'short')],
    'NEUTRAL':      [(10, 20, 'long'), (-20, -10, 'short')],
    'CAUTIOUS':     [(-10, None, 'short'), (-20, -10, 'short')],
    'BEARISH':      [(-10, None, 'short'), (-20, -10, 'short')],
    'CRISIS':       [],
    'PANICO':       [],
    'RECOVERY':     [(0, 10, 'long'),  (-20, -10, 'long')],
    'CAPITULACION': [(10, 20, 'long'), (20, 30, 'long')],
}
STRATEGY_LABELS = {
    'BURBUJA':      ['Top 10 Long', 'Bot 11-20 Short'],
    'GOLDILOCKS':   ['Top 10 Long', 'Bot 11-20 Short'],
    'ALCISTA':      ['Top 10 Long', 'Bot 10 Short'],
    'NEUTRAL':      ['Top 11-20 Long', 'Bot 11-20 Short'],
    'CAUTIOUS':     ['Bot 10 Short', 'Bot 11-20 Short'],
    'BEARISH':      ['Bot 10 Short', 'Bot 11-20 Short'],
    'CRISIS':       ['FUERA', ''],
    'PANICO':       ['FUERA', ''],
    'RECOVERY':     ['Top 10 Long', 'Bot 11-20 Long'],
    'CAPITULACION': ['Top 11-20 Long', 'Top 21-30 Long'],
}
COST = 20000   # $ por accion
SLIP = 0.003   # 0.3% slippage+comision = $60/accion

REGIME_COLORS = {
    'BURBUJA': '#ff6b6b', 'GOLDILOCKS': '#ffd93d', 'ALCISTA': '#6bcb77',
    'NEUTRAL': '#4d96ff', 'CAUTIOUS': '#ff922b', 'BEARISH': '#c084fc',
    'CRISIS': '#ff6b6b', 'PANICO': '#dc2626', 'RECOVERY': '#22d3ee',
    'CAPITULACION': '#f472b6'
}
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS',
                'BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']

# ── 2. Load data from acciones_navegable.html ───────────────────────
print("Loading acciones_navegable.html...")
with open(BASE / 'acciones_navegable.html', 'r', encoding='utf-8') as f:
    html_text = f.read()

m = re.search(r'const T\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
T = json.loads(m.group(1))  # ticker lookup: [{t: "GEV", s: "Ind Power"}, ...]

m2 = re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
W = json.loads(m2.group(1))  # weeks with stock rankings
del html_text
print(f"  {len(T)} tickers, {len(W)} weeks")

# Stock entry format: [ticker_idx, FV, FVA, DD, RSI, ?, price, next_price, ret_pct]
# Sorted by FVA descending (position 0 = best)

# ── 3. Run backtest ─────────────────────────────────────────────────
print("\nRunning backtest...")
weekly_results = []
CLIP = 50  # clip returns at +-50%

for w in W:
    date = w['d']
    year = w['y']
    sem = w['w']
    regime = w['r']
    spy_ret = w.get('sr') or 0
    stocks = w['s']

    strat = STRATEGY.get(regime, [])
    if not strat:
        weekly_results.append({
            'date': date, 'year': year, 'sem': sem, 'regime': regime,
            'spy_ret': spy_ret, 'pnl': 0, 'g1': 0, 'g2': 0,
            'n_pos': 0, 'n_win': 0, 'capital': 0,
            'positions': [], 'active': False
        })
        continue

    positions = []
    pnl_g1 = 0
    pnl_g2 = 0

    for gi, (start, end, direction) in enumerate(strat):
        selected = stocks[start:end] if end is not None else stocks[start:]

        for s in selected:
            ret_val = s[8]  # weekly return Fri close -> Fri close
            if ret_val is None:
                continue

            ticker_idx = s[0]
            ticker = T[ticker_idx]['t']
            sector = T[ticker_idx]['s']
            price = s[6]
            fva = s[2]

            ret_val = max(-CLIP, min(CLIP, ret_val))

            if direction == 'long':
                pnl = COST * (ret_val / 100 - SLIP)
            else:
                pnl = COST * (-ret_val / 100 - SLIP)

            positions.append({
                'tk': ticker, 'sec': sector, 'fva': fva,
                'dir': direction, 'ret': round(ret_val, 2),
                'pnl': round(pnl, 2), 'g': gi + 1, 'px': price
            })

            if gi == 0:
                pnl_g1 += pnl
            else:
                pnl_g2 += pnl

    total_pnl = pnl_g1 + pnl_g2
    n_win = sum(1 for p in positions if p['pnl'] > 0)
    capital = len(positions) * COST

    weekly_results.append({
        'date': date, 'year': year, 'sem': sem, 'regime': regime,
        'spy_ret': spy_ret, 'pnl': round(total_pnl, 2),
        'g1': round(pnl_g1, 2), 'g2': round(pnl_g2, 2),
        'n_pos': len(positions), 'n_win': n_win, 'capital': capital,
        'positions': positions, 'active': len(positions) > 0
    })

print(f"  {len(weekly_results)} weeks processed")

# ── 4. Aggregate stats ──────────────────────────────────────────────
regime_stats = {}
for reg in REGIME_ORDER:
    wks = [w for w in weekly_results if w['regime'] == reg]
    active = [w for w in wks if w['active']]
    if not wks:
        continue
    total_pnl = sum(w['pnl'] for w in wks)
    total_g1 = sum(w['g1'] for w in wks)
    total_g2 = sum(w['g2'] for w in wks)
    n_win_wk = sum(1 for w in active if w['pnl'] > 0)
    avg_pnl = total_pnl / len(active) if active else 0
    wr = n_win_wk / len(active) * 100 if active else 0
    max_wk = max((w['pnl'] for w in active), default=0)
    min_wk = min((w['pnl'] for w in active), default=0)
    avg_pos = np.mean([w['n_pos'] for w in active]) if active else 0
    avg_cap = np.mean([w['capital'] for w in active]) if active else 0
    # Slippage total
    slip_total = sum(w['n_pos'] for w in active) * COST * SLIP
    regime_stats[reg] = {
        'n': len(wks), 'active': len(active),
        'total': round(total_pnl), 'g1': round(total_g1), 'g2': round(total_g2),
        'avg': round(avg_pnl), 'wr': round(wr, 1),
        'best': round(max_wk), 'worst': round(min_wk),
        'avg_pos': round(avg_pos, 1), 'avg_cap': round(avg_cap),
        'slip': round(slip_total)
    }

years = sorted(set(w['year'] for w in weekly_results))
year_data = {}
for y in years:
    wks = [w for w in weekly_results if w['year'] == y]
    active = [w for w in wks if w['active']]
    total_pnl = sum(w['pnl'] for w in wks)
    n_win_wk = sum(1 for w in active if w['pnl'] > 0)
    wr = n_win_wk / len(active) * 100 if active else 0
    reg_dist = {}
    for w in wks:
        reg_dist[w['regime']] = reg_dist.get(w['regime'], 0) + 1
    avg_cap = np.mean([w['capital'] for w in active]) if active else 0
    # Retorno anual = PnL / capital medio desplegado
    ret_pct = (total_pnl / avg_cap * 100) if avg_cap > 0 else 0
    year_data[y] = {
        'n': len(wks), 'active': len(active), 'total': round(total_pnl),
        'avg': round(total_pnl / len(active)) if active else 0,
        'wr': round(wr, 1), 'dist': reg_dist,
        'cap': round(avg_cap), 'ret': round(ret_pct, 1)
    }

all_active = [w for w in weekly_results if w['active']]
g_pnl = sum(w['pnl'] for w in weekly_results)
g_win = sum(1 for w in all_active if w['pnl'] > 0)
g_wr = g_win / len(all_active) * 100 if all_active else 0
g_slip = sum(w['n_pos'] for w in all_active) * COST * SLIP
g_avg_pos = np.mean([w['n_pos'] for w in all_active])
g_avg_cap = np.mean([w['capital'] for w in all_active])

print(f"\n{'='*70}")
print(f"GLOBAL: PnL=${g_pnl:,.0f}  Active={len(all_active)}/{len(weekly_results)}  "
      f"WR={g_wr:.1f}%  Slip=${g_slip:,.0f}")
print(f"Avg positions/week: {g_avg_pos:.0f}  Avg capital/week: ${g_avg_cap:,.0f}")
print(f"{'='*70}")
for reg in REGIME_ORDER:
    if reg in regime_stats:
        s = regime_stats[reg]
        print(f"  {reg:12s}: N={s['n']:3d} Act={s['active']:3d} "
              f"PnL=${s['total']:>10,} G1=${s['g1']:>10,} G2=${s['g2']:>10,} "
              f"WR={s['wr']:5.1f}% Pos={s['avg_pos']:.0f} Slip=${s['slip']:>8,}")

print(f"\n{'='*70}")
print(f"{'RETORNOS ANUALES':^70}")
print(f"{'='*70}")
print(f"{'Ano':>6} {'Sem':>4} {'Act':>4} {'PnL':>12} {'AvgCap':>10} {'Ret%':>8} {'WR%':>6}")
print(f"{'-'*70}")
for y in years:
    d = year_data[y]
    print(f"{y:>6} {d['n']:>4} {d['active']:>4} ${d['total']:>10,} "
          f"${d['cap']:>9,} {d['ret']:>7.1f}% {d['wr']:>5.1f}%")
cum_ret = g_pnl / g_avg_cap * 100 if g_avg_cap > 0 else 0
n_years = len(years)
avg_annual = cum_ret / n_years
print(f"{'-'*70}")
print(f"{'TOTAL':>6} {'':<4} {'':<4} ${g_pnl:>10,.0f} ${g_avg_cap:>9,.0f} {cum_ret:>7.1f}% {g_wr:>5.1f}%")
print(f"{'Avg/yr':>6} {'':<4} {'':<4} ${g_pnl/n_years:>10,.0f} {'':<10} {avg_annual:>7.1f}%")

# ── 5. Prepare JSON for HTML ────────────────────────────────────────
weeks_json = []
cum = 0
for w in weekly_results:
    cum += w['pnl']
    wj = {
        'd': w['date'], 'y': w['year'], 's': w['sem'], 'r': w['regime'],
        'sr': w['spy_ret'], 'p': round(w['pnl']), 'g1': round(w['g1']),
        'g2': round(w['g2']), 'np': w['n_pos'], 'nw': w['n_win'],
        'a': w['active'], 'cum': round(cum), 'cap': w['capital'],
        'pos': [{'t': p['tk'], 'sc': p['sec'], 'fva': p['fva'],
                 'di': p['dir'], 'r': p['ret'], 'p': round(p['pnl']),
                 'g': p['g'], 'px': p['px']}
                for p in w['positions']]
    }
    weeks_json.append(wj)

# ── 6. Generate HTML ────────────────────────────────────────────────
print("\nGenerating HTML...")

rc_json = json.dumps(REGIME_COLORS)
ro_json = json.dumps(REGIME_ORDER)
sl_json = json.dumps(STRATEGY_LABELS)
rs_json = json.dumps(regime_stats)
yd_json = json.dumps(year_data)
wk_json = json.dumps(weeks_json, separators=(',',':'))

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Estrategia por Regimen - Acciones Individuales</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}
h1{{text-align:center;font-size:1.5em;margin-bottom:6px;color:#f8fafc}}
.sub{{text-align:center;color:#94a3b8;margin-bottom:18px;font-size:0.85em}}
.card{{background:#1e293b;border-radius:12px;padding:18px;margin-bottom:14px;border:1px solid #334155}}
.card h2{{font-size:1.1em;color:#f1f5f9;margin-bottom:10px;border-bottom:1px solid #334155;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:0.82em}}
th{{background:#334155;color:#f1f5f9;padding:7px 5px;text-align:left;position:sticky;top:0}}
td{{padding:5px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#334155}}
.pos{{color:#22c55e}}.neg{{color:#ef4444}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.73em;font-weight:600;color:#000}}
.badge-sm{{padding:1px 5px;font-size:0.68em;border-radius:8px}}
.glob{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}}
.gv{{background:#0f172a;border-radius:8px;padding:10px;text-align:center}}
.gv .val{{font-size:1.4em;font-weight:700}}.gv .lbl{{font-size:0.78em;color:#94a3b8;margin-top:3px}}
.rb{{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;justify-content:center}}
.rb button{{padding:3px 10px;border-radius:14px;border:2px solid transparent;cursor:pointer;font-size:0.78em;font-weight:600;color:#000;opacity:0.7;transition:all 0.2s}}
.rb button:hover,.rb button.on{{opacity:1;transform:scale(1.05);border-color:#fff}}
.rb button.rst{{background:#475569;color:#e2e8f0}}
.nav{{display:flex;gap:8px;align-items:center;justify-content:center;margin-bottom:10px;flex-wrap:wrap}}
.nav button{{background:#334155;color:#e2e8f0;border:none;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:0.85em}}
.nav button:hover{{background:#475569}}
.nav select{{background:#334155;color:#e2e8f0;border:1px solid #475569;padding:5px;border-radius:6px;font-size:0.82em;max-width:350px}}
.nav span{{color:#94a3b8;font-size:0.82em}}
.wk-detail{{background:#0f172a;border-radius:8px;padding:14px;margin-top:10px}}
.wk-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px}}
.pos-tbl{{margin-top:6px}}
.pos-tbl th{{background:#1e293b}}
.fuera{{text-align:center;color:#94a3b8;padding:20px;font-style:italic}}
.chart-wrap{{position:relative;height:280px;background:#0f172a;border-radius:8px;overflow:hidden}}
.chart-wrap canvas{{width:100%!important;height:100%!important}}
.yr-btn{{background:none;border:1px solid #475569;color:#94a3b8;padding:2px 7px;border-radius:4px;cursor:pointer;font-size:0.73em}}
.yr-btn:hover,.yr-btn.on{{background:#475569;color:#f1f5f9}}
.g-tbl{{max-height:500px;overflow-y:auto}}
</style>
</head>
<body>
<h1>Backtest: Estrategia por Regimen - Acciones Individuales</h1>
<div class="sub">S&amp;P 500 acciones | $20,000/accion | $60 slippage+comision/trade (0.3%) | Fri close &rarr; Fri close | 2001-2026</div>

<div class="card">
<h2>1. Resumen Global</h2>
<div class="glob" id="gsum"></div>
</div>

<div class="card">
<h2>2. Estrategia por Regimen</h2>
<div class="g-tbl">
<table id="strat-tbl">
<thead><tr><th>Regimen</th><th>Grupo 1</th><th>Grupo 2</th><th>Sem</th><th>Pos/Sem</th><th>PnL Total</th><th>PnL G1</th><th>PnL G2</th><th>Slip</th><th>Avg/Sem</th><th>WR%</th></tr></thead>
<tbody id="strat-body"></tbody>
<tfoot id="strat-foot"></tfoot>
</table>
</div>
</div>

<div class="card">
<h2>3. Resultados Anuales</h2>
<div class="rb" id="yr-btns"></div>
<div class="g-tbl">
<table id="yr-tbl">
<thead><tr><th>Ano</th><th>Sem</th><th>Act</th><th>Capital Medio</th><th>PnL</th><th>Ret%</th><th>Avg/Sem</th><th>WR%</th><th>Regimenes</th></tr></thead>
<tbody id="yr-body"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>4. PnL Acumulado</h2>
<div class="chart-wrap"><canvas id="cumChart"></canvas></div>
</div>

<div class="card">
<h2>5. Detalle Semanal</h2>
<div class="rb" id="reg-btns"></div>
<div class="nav" id="wk-nav"></div>
<div id="wk-detail"></div>
</div>

<script>
const RC={rc_json};
const RO={ro_json};
const SL={sl_json};
const RS={rs_json};
const YD={yd_json};
const WK={wk_json};

const fmt=v=>{{let s=v<0?'-$':'$';return s+Math.abs(Math.round(v)).toLocaleString('en-US')}};
const cls=v=>v>=0?'pos':'neg';

// ── 1. Global Summary ──
(function(){{
let ta=WK.filter(w=>w.a),tp=WK.reduce((s,w)=>s+w.p,0);
let nw=ta.filter(w=>w.p>0).length,wr=ta.length?nw/ta.length*100:0;
let avg=ta.length?tp/ta.length:0;
let maxW=Math.max(...ta.map(w=>w.p)),minW=Math.min(...ta.map(w=>w.p));
let totPos=ta.reduce((s,w)=>s+w.np,0);
let totSlip=totPos*{COST}*{SLIP};
let avgCap=ta.length?ta.reduce((s,w)=>s+w.cap,0)/ta.length:0;
let h='';
[['PnL Total',fmt(tp),cls(tp)],['Semanas Activas',ta.length+'/'+WK.length,''],
 ['Win Rate',wr.toFixed(1)+'%',wr>=50?'pos':'neg'],['Avg PnL/Sem',fmt(avg),cls(avg)],
 ['Mejor Semana',fmt(maxW),'pos'],['Peor Semana',fmt(minW),'neg'],
 ['Avg Capital/Sem','$'+(avgCap/1000).toFixed(0)+'K',''],
 ['Slippage Total',fmt(totSlip),'neg'],
 ['Total Trades',totPos.toLocaleString(),'']].forEach(([l,v,c])=>{{
  h+=`<div class="gv"><div class="val ${{c}}">${{v}}</div><div class="lbl">${{l}}</div></div>`;
}});
document.getElementById('gsum').innerHTML=h;
}})();

// ── 2. Strategy Table ──
(function(){{
let b='',totP=0,totG1=0,totG2=0,totS=0;
RO.forEach(r=>{{
  let s=RS[r];if(!s)return;
  let sl=SL[r]||['',''];
  totP+=s.total;totG1+=s.g1;totG2+=s.g2;totS+=s.slip;
  b+=`<tr><td><span class="badge" style="background:${{RC[r]}}">${{r}}</span></td>
  <td>${{sl[0]}}</td><td>${{sl[1]}}</td><td>${{s.active}}</td>
  <td>${{s.avg_pos}}</td>
  <td class="${{cls(s.total)}}">${{fmt(s.total)}}</td>
  <td class="${{cls(s.g1)}}">${{fmt(s.g1)}}</td>
  <td class="${{cls(s.g2)}}">${{fmt(s.g2)}}</td>
  <td class="neg">${{fmt(s.slip)}}</td>
  <td class="${{cls(s.avg)}}">${{fmt(s.avg)}}</td>
  <td>${{s.wr}}%</td></tr>`;
}});
document.getElementById('strat-body').innerHTML=b;
document.getElementById('strat-foot').innerHTML=
  `<tr style="font-weight:700;border-top:2px solid #475569"><td colspan=4>TOTAL</td><td></td>
  <td class="${{cls(totP)}}">${{fmt(totP)}}</td><td class="${{cls(totG1)}}">${{fmt(totG1)}}</td>
  <td class="${{cls(totG2)}}">${{fmt(totG2)}}</td><td class="neg">${{fmt(totS)}}</td>
  <td colspan=2></td></tr>`;
}})();

// ── 3. Year Table ──
(function(){{
let yrs=Object.keys(YD).sort();
let b='';
yrs.forEach(y=>{{
  let d=YD[y];
  let pills='';
  RO.forEach(r=>{{if(d.dist[r])pills+=`<span class="badge badge-sm" style="background:${{RC[r]}}">${{r[0]}}${{d.dist[r]}}</span> `}});
  let retC=d.ret>=0?'pos':'neg';
  b+=`<tr><td>${{y}}</td><td>${{d.n}}</td><td>${{d.active}}</td>
  <td>${{d.cap?(d.cap/1000).toFixed(0)+'K':''}}</td>
  <td class="${{cls(d.total)}}">${{fmt(d.total)}}</td>
  <td class="${{retC}}">${{d.ret>=0?'+':''}}${{d.ret}}%</td>
  <td class="${{cls(d.avg)}}">${{fmt(d.avg)}}</td><td>${{d.wr}}%</td><td>${{pills}}</td></tr>`;
}});
document.getElementById('yr-body').innerHTML=b;
}})();

// ── 4. Cumulative PnL Chart ──
(function(){{
const canvas=document.getElementById('cumChart');
const ctx=canvas.getContext('2d');
function draw(){{
  const WW=canvas.parentElement.clientWidth,H=canvas.parentElement.clientHeight;
  canvas.width=WW*2;canvas.height=H*2;ctx.scale(2,2);
  const vals=WK.map(w=>w.cum);
  const mn=Math.min(...vals,0),mx=Math.max(...vals,0);
  const pad={{t:20,b:30,l:70,r:20}};
  const cw=WW-pad.l-pad.r,ch=H-pad.t-pad.b;
  const range=mx-mn||1;
  const xS=cw/vals.length,yS=ch/range;
  ctx.clearRect(0,0,WW,H);
  ctx.strokeStyle='#334155';ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){{
    let yv=mn+(range/4)*i,yy=pad.t+ch-(yv-mn)*yS;
    ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(WW-pad.r,yy);ctx.stroke();
    ctx.fillStyle='#94a3b8';ctx.font='10px sans-serif';ctx.textAlign='right';
    ctx.fillText(fmt(yv),pad.l-4,yy+3);
  }}
  let y0=pad.t+ch-(0-mn)*yS;
  ctx.strokeStyle='#64748b';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,y0);ctx.lineTo(WW-pad.r,y0);ctx.stroke();
  // Color line segments by regime
  for(let i=1;i<vals.length;i++){{
    let x1=pad.l+(i-1)*xS,y1=pad.t+ch-(vals[i-1]-mn)*yS;
    let x2=pad.l+i*xS,y2=pad.t+ch-(vals[i]-mn)*yS;
    ctx.strokeStyle=RC[WK[i].r]||'#60a5fa';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  }}
  ctx.fillStyle='#64748b';ctx.font='9px sans-serif';ctx.textAlign='center';
  let lastY='';
  WK.forEach((w,i)=>{{if(String(w.y)!==lastY){{lastY=String(w.y);ctx.fillText(w.y,pad.l+i*xS,H-pad.b+12)}}}});
}}
draw();window.addEventListener('resize',draw);
}})();

// ── 5. Weekly Navigator ──
(function(){{
let fIdx=WK.map((_,i)=>i);
let cur=0,fReg=null,fYear=null;

function applyFilter(){{
  fIdx=[];
  WK.forEach((w,i)=>{{
    if(fReg&&w.r!==fReg)return;
    if(fYear&&w.y!==fYear)return;
    fIdx.push(i);
  }});
  cur=fIdx.length-1;
  renderNav();renderWeek();
}}

let rbH='<button class="rst" onclick="wkR()">Todos</button>';
RO.forEach(r=>{{
  let n=WK.filter(w=>w.r===r).length;
  rbH+=`<button style="background:${{RC[r]}}" onclick="wkFR('${{r}}')" id="rb_${{r}}">${{r}} (${{n}})</button>`;
}});
document.getElementById('reg-btns').innerHTML=rbH;

window.wkFR=function(r){{
  fReg=fReg===r?null:r;fYear=null;
  document.querySelectorAll('#reg-btns button').forEach(b=>b.classList.remove('on'));
  if(fReg)document.getElementById('rb_'+fReg).classList.add('on');
  applyFilter();
}};
window.wkR=function(){{fReg=null;fYear=null;
  document.querySelectorAll('#reg-btns button').forEach(b=>b.classList.remove('on'));
  applyFilter();
}};

function renderNav(){{
  let yrs=[...new Set(fIdx.map(i=>WK[i].y))].sort();
  let h=`<button onclick="wkP()">&laquo;</button>`;
  h+=`<select onchange="wkG(this.value)">`;
  fIdx.forEach((wi,fi)=>{{
    let w=WK[wi];
    h+=`<option value="${{fi}}" ${{fi===cur?'selected':''}}>${{w.d}} S${{w.s}} ${{w.r}} ${{w.np}}pos</option>`;
  }});
  h+=`</select>`;
  h+=`<button onclick="wkN()">&#187;</button>`;
  h+=`<span>${{cur+1}}/${{fIdx.length}}</span>`;
  h+='<span style="margin-left:12px">';
  yrs.forEach(y=>{{h+=`<button class="yr-btn${{fYear===y?' on':''}}" onclick="wkFY(${{y}})">${{y}}</button> `}});
  h+='</span>';
  document.getElementById('wk-nav').innerHTML=h;
}}

window.wkP=function(){{if(cur>0){{cur--;renderNav();renderWeek()}}}};
window.wkN=function(){{if(cur<fIdx.length-1){{cur++;renderNav();renderWeek()}}}};
window.wkG=function(v){{cur=parseInt(v);renderNav();renderWeek()}};
window.wkFY=function(y){{
  fYear=fYear===y?null:y;fReg=null;
  document.querySelectorAll('#reg-btns button').forEach(b=>b.classList.remove('on'));
  applyFilter();
}};

function renderWeek(){{
  if(!fIdx.length){{document.getElementById('wk-detail').innerHTML='<div class="fuera">No hay semanas</div>';return}}
  let w=WK[fIdx[cur]];
  let sl=SL[w.r]||['',''];
  let h=`<div class="wk-detail">`;
  h+=`<div class="wk-head">
    <div><strong>${{w.d}}</strong> &mdash; Semana ${{w.s}}, ${{w.y}}
    <span class="badge" style="background:${{RC[w.r]}}">${{w.r}}</span>
    <span style="color:#94a3b8;font-size:0.8em;margin-left:8px">${{sl[0]}} + ${{sl[1]}}</span></div>
    <div>SPY: <span class="${{cls(w.sr)}}">${{w.sr>=0?'+':''}}${{(w.sr||0).toFixed(2)}}%</span>
    &nbsp; PnL: <span class="${{cls(w.p)}}"><strong>${{fmt(w.p)}}</strong></span>
    &nbsp; Acum: <span class="${{cls(w.cum)}}">${{fmt(w.cum)}}</span>
    &nbsp; Capital: ${{(w.cap/1000).toFixed(0)}}K (${{w.np}} pos)</div></div>`;

  if(!w.a){{
    h+=`<div class="fuera">FUERA DE MERCADO &mdash; Sin posiciones (${{w.r}})</div>`;
  }} else {{
    h+=`<div style="display:flex;gap:16px;margin-bottom:6px;flex-wrap:wrap;font-size:0.85em">
    <div>G1 (${{sl[0]}}): <span class="${{cls(w.g1)}}">${{fmt(w.g1)}}</span></div>
    <div>G2 (${{sl[1]}}): <span class="${{cls(w.g2)}}">${{fmt(w.g2)}}</span></div>
    <div>Ganadoras: ${{w.nw}}/${{w.np}} (${{(w.np?w.nw/w.np*100:0).toFixed(0)}}%)</div>
    <div>Slip: <span class="neg">${{fmt(w.np*{COST}*{SLIP})}}</span></div></div>`;

    // Group 1
    let g1=w.pos.filter(p=>p.g===1), g2=w.pos.filter(p=>p.g===2);
    [['G1: '+sl[0],g1],['G2: '+sl[1],g2]].forEach(([label,grp])=>{{
      if(!grp.length)return;
      h+=`<div style="margin-top:8px"><strong style="font-size:0.85em">${{label}} (${{grp.length}} acciones)</strong>`;
      h+=`<table class="pos-tbl"><thead><tr><th>#</th><th>Ticker</th><th>Sector</th><th>FVA</th><th>Precio</th><th>Dir</th><th>Ret%</th><th>PnL</th></tr></thead><tbody>`;
      grp.forEach((p,i)=>{{
        let dc=p.di==='long'?'#22c55e':'#ef4444';
        let arrow=p.di==='long'?'&#9650;':'&#9660;';
        h+=`<tr><td>${{i+1}}</td><td><strong>${{p.t}}</strong></td><td style="color:#94a3b8;font-size:0.8em">${{p.sc}}</td>
        <td>${{p.fva}}</td><td>$${{p.px}}</td>
        <td style="color:${{dc}}">${{arrow}} ${{p.di.toUpperCase()}}</td>
        <td class="${{cls(p.r)}}">${{p.r>=0?'+':''}}${{p.r}}%</td>
        <td class="${{cls(p.p)}}">${{fmt(p.p)}}</td></tr>`;
      }});
      h+=`</tbody></table></div>`;
    }});
  }}
  h+=`</div>`;
  document.getElementById('wk-detail').innerHTML=h;
}}

applyFilter();
}})();
</script>
</body>
</html>"""

out_path = BASE / 'backtest_regimen_estrategia.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nHTML saved to: {out_path}")
print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
