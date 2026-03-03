#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtest Comparativo: Estrategia 1 (CRISIS/PANICO fuera) vs Estrategia 2 (CRISIS/PANICO short)
"""
import re, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

# ── Strategies ───────────────────────────────────────────────────────
STRAT1 = {
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
STRAT2 = {
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
LABELS1 = {
    'BURBUJA': ['Top 10 Long','Bot 11-20 Short'], 'GOLDILOCKS': ['Top 10 Long','Bot 11-20 Short'],
    'ALCISTA': ['Top 10 Long','Bot 10 Short'], 'NEUTRAL': ['Top 11-20 Long','Bot 11-20 Short'],
    'CAUTIOUS': ['Bot 10 Short','Bot 11-20 Short'], 'BEARISH': ['Bot 10 Short','Bot 11-20 Short'],
    'CRISIS': ['FUERA',''], 'PANICO': ['FUERA',''],
    'RECOVERY': ['Top 10 Long','Bot 11-20 Long'], 'CAPITULACION': ['Top 11-20 Long','Top 21-30 Long'],
}
LABELS2 = dict(LABELS1)
LABELS2['CRISIS'] = ['Top 10 Short', 'Top 11-20 Short']
LABELS2['PANICO'] = ['Top 10 Short', 'Top 11-20 Short']

COST = 20000
SLIP = 0.003
CLIP = 50

REGIME_COLORS = {
    'BURBUJA': '#ff6b6b', 'GOLDILOCKS': '#ffd93d', 'ALCISTA': '#6bcb77',
    'NEUTRAL': '#4d96ff', 'CAUTIOUS': '#ff922b', 'BEARISH': '#c084fc',
    'CRISIS': '#ff6b6b', 'PANICO': '#dc2626', 'RECOVERY': '#22d3ee',
    'CAPITULACION': '#f472b6'
}
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS',
                'BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']

# ── Load data ────────────────────────────────────────────────────────
print("Loading acciones_navegable.html...")
with open(BASE / 'acciones_navegable.html', 'r', encoding='utf-8') as f:
    html_text = f.read()
m = re.search(r'const T\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
T = json.loads(m.group(1))
m2 = re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
WEEKS = json.loads(m2.group(1))
del html_text
print(f"  {len(T)} tickers, {len(WEEKS)} weeks")

# ── Backtest function ────────────────────────────────────────────────
def run_backtest(strategy, label):
    print(f"\nRunning {label}...")
    results = []
    for w in WEEKS:
        date, year, sem, regime = w['d'], w['y'], w['w'], w['r']
        spy_ret = w.get('sr') or 0
        stocks = w['s']
        strat = strategy.get(regime, [])
        if not strat:
            results.append({'date': date, 'year': year, 'sem': sem, 'regime': regime,
                            'spy_ret': spy_ret, 'pnl': 0, 'g1': 0, 'g2': 0,
                            'n_pos': 0, 'n_win': 0, 'capital': 0, 'pos': [], 'active': False})
            continue
        positions, pnl_g1, pnl_g2 = [], 0, 0
        for gi, (start, end, direction) in enumerate(strat):
            selected = stocks[start:end] if end is not None else stocks[start:]
            for s in selected:
                ret_val = s[8]
                if ret_val is None:
                    continue
                ticker = T[s[0]]['t']
                sector = T[s[0]]['s']
                ret_val = max(-CLIP, min(CLIP, ret_val))
                pnl = COST * (ret_val / 100 - SLIP) if direction == 'long' else COST * (-ret_val / 100 - SLIP)
                positions.append({'t': ticker, 'sc': sector, 'fva': s[2], 'di': direction,
                                  'r': round(ret_val, 2), 'p': round(pnl, 2), 'g': gi + 1, 'px': s[6]})
                if gi == 0: pnl_g1 += pnl
                else: pnl_g2 += pnl
        total_pnl = pnl_g1 + pnl_g2
        n_win = sum(1 for p in positions if p['p'] > 0)
        results.append({'date': date, 'year': year, 'sem': sem, 'regime': regime,
                        'spy_ret': spy_ret, 'pnl': round(total_pnl, 2),
                        'g1': round(pnl_g1, 2), 'g2': round(pnl_g2, 2),
                        'n_pos': len(positions), 'n_win': n_win,
                        'capital': len(positions) * COST, 'pos': positions,
                        'active': len(positions) > 0})
    return results

def aggregate(results):
    rs = {}
    for reg in REGIME_ORDER:
        wks = [w for w in results if w['regime'] == reg]
        act = [w for w in wks if w['active']]
        if not wks: continue
        tp = sum(w['pnl'] for w in wks)
        tg1 = sum(w['g1'] for w in wks)
        tg2 = sum(w['g2'] for w in wks)
        nw = sum(1 for w in act if w['pnl'] > 0)
        rs[reg] = {
            'n': len(wks), 'act': len(act), 'total': round(tp), 'g1': round(tg1), 'g2': round(tg2),
            'avg': round(tp / len(act)) if act else 0, 'wr': round(nw / len(act) * 100, 1) if act else 0,
            'slip': round(sum(w['n_pos'] for w in act) * COST * SLIP),
            'avg_pos': round(np.mean([w['n_pos'] for w in act]), 1) if act else 0
        }
    years = sorted(set(w['year'] for w in results))
    yd = {}
    for y in years:
        wks = [w for w in results if w['year'] == y]
        act = [w for w in wks if w['active']]
        tp = sum(w['pnl'] for w in wks)
        nw = sum(1 for w in act if w['pnl'] > 0)
        ac = np.mean([w['capital'] for w in act]) if act else 0
        ret = (tp / ac * 100) if ac > 0 else 0
        dist = {}
        for w in wks: dist[w['regime']] = dist.get(w['regime'], 0) + 1
        yd[y] = {'n': len(wks), 'act': len(act), 'total': round(tp),
                 'avg': round(tp / len(act)) if act else 0,
                 'wr': round(nw / len(act) * 100, 1) if act else 0,
                 'cap': round(ac), 'ret': round(ret, 1), 'dist': dist}
    aa = [w for w in results if w['active']]
    gp = sum(w['pnl'] for w in results)
    gw = sum(1 for w in aa if w['pnl'] > 0)
    gwr = gw / len(aa) * 100 if aa else 0
    gac = np.mean([w['capital'] for w in aa]) if aa else 0
    glob = {'pnl': round(gp), 'active': len(aa), 'total': len(results),
            'wr': round(gwr, 1), 'avg_cap': round(gac),
            'slip': round(sum(w['n_pos'] for w in aa) * COST * SLIP),
            'avg': round(gp / len(aa)) if aa else 0,
            'ret': round(gp / gac * 100, 1) if gac else 0,
            'avg_annual': round(gp / gac * 100 / len(years), 1) if gac else 0}
    return rs, yd, glob, years

# ── Run both ─────────────────────────────────────────────────────────
r1 = run_backtest(STRAT1, "Estrategia 1 (CRISIS/PANICO fuera)")
r2 = run_backtest(STRAT2, "Estrategia 2 (CRISIS/PANICO short)")

rs1, yd1, g1, years = aggregate(r1)
rs2, yd2, g2, _ = aggregate(r2)

# Print comparison
print(f"\n{'='*80}")
print(f"{'COMPARATIVA':^80}")
print(f"{'='*80}")
print(f"{'':>14} {'EST 1 (fuera)':>20} {'EST 2 (short)':>20} {'DELTA':>15}")
print(f"{'-'*80}")
print(f"{'PnL Total':>14} ${g1['pnl']:>18,} ${g2['pnl']:>18,} ${g2['pnl']-g1['pnl']:>13,}")
print(f"{'WR':>14} {g1['wr']:>19.1f}% {g2['wr']:>19.1f}%")
print(f"{'Avg/Sem':>14} ${g1['avg']:>18,} ${g2['avg']:>18,}")
print(f"{'Ret Total':>14} {g1['ret']:>19.1f}% {g2['ret']:>19.1f}%")
print(f"{'Avg Anual':>14} {g1['avg_annual']:>19.1f}% {g2['avg_annual']:>19.1f}%")
print(f"{'Slippage':>14} ${g1['slip']:>18,} ${g2['slip']:>18,}")
print(f"{'Active':>14} {g1['active']:>15}/{g1['total']} {g2['active']:>15}/{g2['total']}")

print(f"\n{'POR REGIMEN':^80}")
print(f"{'Regimen':>14} {'E1 PnL':>12} {'E2 PnL':>12} {'Delta':>12} {'E1 WR':>8} {'E2 WR':>8}")
print(f"{'-'*80}")
for reg in REGIME_ORDER:
    s1 = rs1.get(reg, {'total': 0, 'wr': 0})
    s2 = rs2.get(reg, {'total': 0, 'wr': 0})
    d = s2['total'] - s1['total']
    mark = ' ***' if d != 0 and s1['total'] == 0 else ''
    print(f"  {reg:>12} ${s1['total']:>10,} ${s2['total']:>10,} ${d:>10,} {s1['wr']:>7.1f}% {s2['wr']:>7.1f}%{mark}")

print(f"\n{'RETORNOS ANUALES':^80}")
print(f"{'Ano':>6} {'E1 PnL':>12} {'E1 Ret%':>8} {'E2 PnL':>12} {'E2 Ret%':>8} {'Delta':>12}")
print(f"{'-'*80}")
for y in years:
    d1, d2 = yd1[y], yd2[y]
    delta = d2['total'] - d1['total']
    print(f"{y:>6} ${d1['total']:>10,} {d1['ret']:>7.1f}% ${d2['total']:>10,} {d2['ret']:>7.1f}% ${delta:>10,}")
print(f"{'-'*80}")
print(f"{'TOTAL':>6} ${g1['pnl']:>10,} {g1['ret']:>7.1f}% ${g2['pnl']:>10,} {g2['ret']:>7.1f}% ${g2['pnl']-g1['pnl']:>10,}")

# ── Prepare JSON for HTML ────────────────────────────────────────────
def make_weeks_json(results):
    wj, cum = [], 0
    for w in results:
        cum += w['pnl']
        wj.append({
            'd': w['date'], 'y': w['year'], 's': w['sem'], 'r': w['regime'],
            'sr': w['spy_ret'], 'p': round(w['pnl']), 'g1': round(w['g1']),
            'g2': round(w['g2']), 'np': w['n_pos'], 'nw': w['n_win'],
            'a': w['active'], 'cum': round(cum), 'cap': w['capital'],
            'pos': [{'t': p['t'], 'sc': p['sc'], 'fva': p['fva'], 'di': p['di'],
                     'r': p['r'], 'p': round(p['p']), 'g': p['g'], 'px': p['px']}
                    for p in w['pos']]
        })
    return wj

wj1 = make_weeks_json(r1)
wj2 = make_weeks_json(r2)

# ── Generate HTML ────────────────────────────────────────────────────
print("\nGenerating HTML...")

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Comparativa: Estrategia 1 vs 2</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:14px}}
h1{{text-align:center;font-size:1.4em;margin-bottom:6px;color:#f8fafc}}
.sub{{text-align:center;color:#94a3b8;margin-bottom:16px;font-size:0.82em}}
.card{{background:#1e293b;border-radius:10px;padding:16px;margin-bottom:12px;border:1px solid #334155}}
.card h2{{font-size:1.05em;color:#f1f5f9;margin-bottom:8px;border-bottom:1px solid #334155;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:0.8em}}
th{{background:#334155;color:#f1f5f9;padding:6px 4px;text-align:left;position:sticky;top:0}}
td{{padding:5px 4px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#334155}}
.pos{{color:#22c55e}}.neg{{color:#ef4444}}
.badge{{display:inline-block;padding:2px 7px;border-radius:9px;font-size:0.72em;font-weight:600;color:#000}}
.badge-sm{{padding:1px 4px;font-size:0.67em;border-radius:7px}}
.side{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:900px){{.side{{grid-template-columns:1fr}}}}
.gv-row{{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-bottom:10px}}
.gv{{background:#0f172a;border-radius:7px;padding:8px 14px;text-align:center;min-width:130px}}
.gv .val{{font-size:1.3em;font-weight:700}}.gv .lbl{{font-size:0.75em;color:#94a3b8;margin-top:2px}}
.rb{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;justify-content:center}}
.rb button{{padding:3px 9px;border-radius:12px;border:2px solid transparent;cursor:pointer;font-size:0.75em;font-weight:600;color:#000;opacity:0.7;transition:all 0.2s}}
.rb button:hover,.rb button.on{{opacity:1;border-color:#fff}}
.rb button.rst{{background:#475569;color:#e2e8f0}}
.nav{{display:flex;gap:6px;align-items:center;justify-content:center;margin-bottom:8px;flex-wrap:wrap}}
.nav button{{background:#334155;color:#e2e8f0;border:none;padding:4px 10px;border-radius:5px;cursor:pointer;font-size:0.82em}}
.nav button:hover{{background:#475569}}
.nav select{{background:#334155;color:#e2e8f0;border:1px solid #475569;padding:4px;border-radius:5px;font-size:0.8em;max-width:300px}}
.nav span{{color:#94a3b8;font-size:0.8em}}
.wk-d{{background:#0f172a;border-radius:7px;padding:12px;margin-top:8px}}
.wk-h{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:4px}}
.pt th{{background:#1e293b}}
.fuera{{text-align:center;color:#94a3b8;padding:16px;font-style:italic}}
.chart-wrap{{position:relative;height:260px;background:#0f172a;border-radius:7px;overflow:hidden}}
.chart-wrap canvas{{width:100%!important;height:100%!important}}
.yr-btn{{background:none;border:1px solid #475569;color:#94a3b8;padding:1px 6px;border-radius:3px;cursor:pointer;font-size:0.72em}}
.yr-btn:hover,.yr-btn.on{{background:#475569;color:#f1f5f9}}
.diff{{color:#fbbf24;font-weight:600}}
.e1{{border-left:3px solid #60a5fa}}.e2{{border-left:3px solid #f472b6}}
</style>
</head>
<body>
<h1>Comparativa: Estrategia 1 vs Estrategia 2</h1>
<div class="sub">E1: CRISIS/PANICO fuera | E2: CRISIS/PANICO Top10+Top11-20 Short | $20K/accion | 0.3% slip | Fri&rarr;Fri</div>

<div class="card">
<h2>1. Resumen Comparativo</h2>
<div class="side">
<div class="e1" style="padding-left:10px">
<h3 style="color:#60a5fa;font-size:0.95em;margin-bottom:6px">Estrategia 1 &mdash; CRISIS/PANICO fuera</h3>
<div class="gv-row" id="gs1"></div>
</div>
<div class="e2" style="padding-left:10px">
<h3 style="color:#f472b6;font-size:0.95em;margin-bottom:6px">Estrategia 2 &mdash; CRISIS/PANICO short</h3>
<div class="gv-row" id="gs2"></div>
</div>
</div>
</div>

<div class="card">
<h2>2. Por Regimen (diferencias en amarillo)</h2>
<table>
<thead><tr><th>Regimen</th><th>Grupo 1</th><th>Grupo 2</th><th>Sem</th><th>E1 PnL</th><th>E2 PnL</th><th>Delta</th><th>E1 WR</th><th>E2 WR</th></tr></thead>
<tbody id="reg-body"></tbody>
<tfoot id="reg-foot"></tfoot>
</table>
</div>

<div class="card">
<h2>3. Retornos Anuales</h2>
<table>
<thead><tr><th>Ano</th><th>E1 PnL</th><th>E1 Ret%</th><th>E2 PnL</th><th>E2 Ret%</th><th>Delta</th><th>Regimenes</th></tr></thead>
<tbody id="yr-body"></tbody>
<tfoot id="yr-foot"></tfoot>
</table>
</div>

<div class="card">
<h2>4. PnL Acumulado</h2>
<div class="chart-wrap"><canvas id="cumChart"></canvas></div>
<div style="text-align:center;margin-top:4px;font-size:0.75em;color:#94a3b8">
<span style="color:#60a5fa">&#9644; E1</span> &nbsp; <span style="color:#f472b6">&#9644; E2</span></div>
</div>

<div class="card">
<h2>5. Detalle Semanal</h2>
<div class="rb" id="reg-btns"></div>
<div class="nav" id="wk-nav"></div>
<div class="side" id="wk-detail"></div>
</div>

<script>
const RC={json.dumps(REGIME_COLORS)};
const RO={json.dumps(REGIME_ORDER)};
const SL1={json.dumps(LABELS1)};
const SL2={json.dumps(LABELS2)};
const RS1={json.dumps(rs1)};
const RS2={json.dumps(rs2)};
const YD1={json.dumps(yd1)};
const YD2={json.dumps(yd2)};
const G1={json.dumps(g1)};
const G2={json.dumps(g2)};
const W1={json.dumps(wj1, separators=(',',':'))};
const W2={json.dumps(wj2, separators=(',',':'))};

const fmt=v=>{{let s=v<0?'-$':'$';return s+Math.abs(Math.round(v)).toLocaleString('en-US')}};
const cls=v=>v>=0?'pos':'neg';

// ── 1. Global Summary ──
function fillGS(id,g){{
  let h='';
  [['PnL Total',fmt(g.pnl),cls(g.pnl)],['WR',g.wr+'%',g.wr>=50?'pos':'neg'],
   ['Avg/Sem',fmt(g.avg),cls(g.avg)],['Ret Total',g.ret+'%',cls(g.ret)],
   ['Avg Anual',g.avg_annual+'%',cls(g.avg_annual)],['Activas',g.active+'/'+g.total,''],
   ['Slippage',fmt(g.slip),'neg']].forEach(([l,v,c])=>{{
    h+=`<div class="gv"><div class="val ${{c}}">${{v}}</div><div class="lbl">${{l}}</div></div>`;
  }});
  document.getElementById(id).innerHTML=h;
}}
fillGS('gs1',G1);fillGS('gs2',G2);

// ── 2. Regime Table ──
(function(){{
  let b='',t1=0,t2=0;
  RO.forEach(r=>{{
    let s1=RS1[r]||{{total:0,wr:0,act:0}}, s2=RS2[r]||{{total:0,wr:0,act:0}};
    let d=s2.total-s1.total, isDiff=d!==0;
    let sl=SL2[r]||['',''];
    t1+=s1.total;t2+=s2.total;
    b+=`<tr${{isDiff?' style="background:#1a1a2e"':''}}>
    <td><span class="badge" style="background:${{RC[r]}}">${{r}}</span></td>
    <td>${{sl[0]}}</td><td>${{sl[1]}}</td><td>${{s2.act||s1.act}}</td>
    <td class="${{cls(s1.total)}}">${{fmt(s1.total)}}</td>
    <td class="${{cls(s2.total)}}">${{fmt(s2.total)}}</td>
    <td class="${{isDiff?'diff':cls(d)}}">${{d>=0?'+':''}}${{fmt(d).replace('-$','-$')}}</td>
    <td>${{s1.wr}}%</td><td>${{s2.wr}}%</td></tr>`;
  }});
  document.getElementById('reg-body').innerHTML=b;
  let dt=t2-t1;
  document.getElementById('reg-foot').innerHTML=
    `<tr style="font-weight:700;border-top:2px solid #475569"><td colspan=4>TOTAL</td>
    <td class="${{cls(t1)}}">${{fmt(t1)}}</td><td class="${{cls(t2)}}">${{fmt(t2)}}</td>
    <td class="diff">${{dt>=0?'+':''}}${{fmt(dt).replace('-$','-$')}}</td><td colspan=2></td></tr>`;
}})();

// ── 3. Year Table ──
(function(){{
  let yrs=Object.keys(YD1).sort(), b='', t1=0, t2=0;
  yrs.forEach(y=>{{
    let d1=YD1[y], d2=YD2[y];
    let delta=d2.total-d1.total;
    t1+=d1.total;t2+=d2.total;
    let pills='';
    RO.forEach(r=>{{if(d1.dist[r])pills+=`<span class="badge badge-sm" style="background:${{RC[r]}}">${{r[0]}}${{d1.dist[r]}}</span> `}});
    b+=`<tr><td>${{y}}</td>
    <td class="${{cls(d1.total)}}">${{fmt(d1.total)}}</td><td class="${{cls(d1.ret)}}">${{d1.ret>=0?'+':''}}${{d1.ret}}%</td>
    <td class="${{cls(d2.total)}}">${{fmt(d2.total)}}</td><td class="${{cls(d2.ret)}}">${{d2.ret>=0?'+':''}}${{d2.ret}}%</td>
    <td class="${{delta!==0?'diff':cls(delta)}}">${{delta>=0?'+':''}}${{fmt(delta).replace('-$','-$')}}</td>
    <td>${{pills}}</td></tr>`;
  }});
  document.getElementById('yr-body').innerHTML=b;
  let dt=t2-t1;
  document.getElementById('yr-foot').innerHTML=
    `<tr style="font-weight:700;border-top:2px solid #475569"><td>TOTAL</td>
    <td class="${{cls(t1)}}">${{fmt(t1)}}</td><td>${{G1.ret}}%</td>
    <td class="${{cls(t2)}}">${{fmt(t2)}}</td><td>${{G2.ret}}%</td>
    <td class="diff">${{dt>=0?'+':''}}${{fmt(dt).replace('-$','-$')}}</td><td></td></tr>`;
}})();

// ── 4. Chart ──
(function(){{
const canvas=document.getElementById('cumChart');
const ctx=canvas.getContext('2d');
function draw(){{
  const WW=canvas.parentElement.clientWidth,H=canvas.parentElement.clientHeight;
  canvas.width=WW*2;canvas.height=H*2;ctx.scale(2,2);
  const v1=W1.map(w=>w.cum),v2=W2.map(w=>w.cum);
  const allV=[...v1,...v2];
  const mn=Math.min(...allV,0),mx=Math.max(...allV,0);
  const pad={{t:20,b:28,l:70,r:20}};
  const cw=WW-pad.l-pad.r,ch=H-pad.t-pad.b;
  const range=mx-mn||1;
  const xS=cw/v1.length,yS=ch/range;
  ctx.clearRect(0,0,WW,H);
  // Grid
  ctx.strokeStyle='#334155';ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){{
    let yv=mn+(range/4)*i,yy=pad.t+ch-(yv-mn)*yS;
    ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(WW-pad.r,yy);ctx.stroke();
    ctx.fillStyle='#94a3b8';ctx.font='10px sans-serif';ctx.textAlign='right';
    ctx.fillText(fmt(yv),pad.l-4,yy+3);
  }}
  // Zero
  let y0=pad.t+ch-(0-mn)*yS;
  ctx.strokeStyle='#64748b';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,y0);ctx.lineTo(WW-pad.r,y0);ctx.stroke();
  // E1 line
  ctx.strokeStyle='#60a5fa';ctx.lineWidth=1.5;ctx.beginPath();
  v1.forEach((v,i)=>{{let x=pad.l+i*xS,y=pad.t+ch-(v-mn)*yS;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}});
  ctx.stroke();
  // E2 line
  ctx.strokeStyle='#f472b6';ctx.lineWidth=1.5;ctx.beginPath();
  v2.forEach((v,i)=>{{let x=pad.l+i*xS,y=pad.t+ch-(v-mn)*yS;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}});
  ctx.stroke();
  // Year labels
  ctx.fillStyle='#64748b';ctx.font='9px sans-serif';ctx.textAlign='center';
  let lastY='';
  W1.forEach((w,i)=>{{if(String(w.y)!==lastY){{lastY=String(w.y);ctx.fillText(w.y,pad.l+i*xS,H-pad.b+12)}}}});
}}
draw();window.addEventListener('resize',draw);
}})();

// ── 5. Weekly Navigator ──
(function(){{
let fIdx=W1.map((_,i)=>i);
let cur=0,fReg=null,fYear=null;

function applyFilter(){{
  fIdx=[];
  W1.forEach((w,i)=>{{
    if(fReg&&w.r!==fReg)return;
    if(fYear&&w.y!==fYear)return;
    fIdx.push(i);
  }});
  cur=fIdx.length-1;
  renderNav();renderWeek();
}}

let rbH='<button class="rst" onclick="wkR()">Todos</button>';
RO.forEach(r=>{{
  let n=W1.filter(w=>w.r===r).length;
  rbH+=`<button style="background:${{RC[r]}}" onclick="wkFR('${{r}}')" id="rb_${{r}}">${{r}} (${{n}})</button>`;
}});
document.getElementById('reg-btns').innerHTML=rbH;

window.wkFR=function(r){{fReg=fReg===r?null:r;fYear=null;
  document.querySelectorAll('#reg-btns button').forEach(b=>b.classList.remove('on'));
  if(fReg)document.getElementById('rb_'+fReg).classList.add('on');
  applyFilter();}};
window.wkR=function(){{fReg=null;fYear=null;
  document.querySelectorAll('#reg-btns button').forEach(b=>b.classList.remove('on'));
  applyFilter();}};

function renderNav(){{
  let yrs=[...new Set(fIdx.map(i=>W1[i].y))].sort();
  let h=`<button onclick="wkP()">&laquo;</button>`;
  h+=`<select onchange="wkG(this.value)">`;
  fIdx.forEach((wi,fi)=>{{let w=W1[wi];
    h+=`<option value="${{fi}}" ${{fi===cur?'selected':''}}>${{w.d}} S${{w.s}} ${{w.r}}</option>`}});
  h+=`</select><button onclick="wkN()">&#187;</button><span>${{cur+1}}/${{fIdx.length}}</span>`;
  h+='<span style="margin-left:8px">';
  yrs.forEach(y=>{{h+=`<button class="yr-btn${{fYear===y?' on':''}}" onclick="wkFY(${{y}})">${{y}}</button> `}});
  h+='</span>';
  document.getElementById('wk-nav').innerHTML=h;
}}
window.wkP=function(){{if(cur>0){{cur--;renderNav();renderWeek()}}}};
window.wkN=function(){{if(cur<fIdx.length-1){{cur++;renderNav();renderWeek()}}}};
window.wkG=function(v){{cur=parseInt(v);renderNav();renderWeek()}};
window.wkFY=function(y){{fYear=fYear===y?null:y;fReg=null;
  document.querySelectorAll('#reg-btns button').forEach(b=>b.classList.remove('on'));
  applyFilter();}};

function renderWkSide(w, sl, label, cssClass){{
  let h=`<div class="wk-d ${{cssClass}}">`;
  h+=`<div class="wk-h"><div><strong>${{label}}</strong>
    PnL: <span class="${{cls(w.p)}}">${{fmt(w.p)}}</span>
    Acum: <span class="${{cls(w.cum)}}">${{fmt(w.cum)}}</span></div>
    <div>${{w.np}} pos | Cap ${{(w.cap/1000).toFixed(0)}}K</div></div>`;
  if(!w.a){{h+=`<div class="fuera">FUERA</div>`}}
  else{{
    h+=`<div style="font-size:0.8em;margin-bottom:4px">G1(${{sl[0]}}): <span class="${{cls(w.g1)}}">${{fmt(w.g1)}}</span>
    &nbsp; G2(${{sl[1]}}): <span class="${{cls(w.g2)}}">${{fmt(w.g2)}}</span>
    &nbsp; Win: ${{w.nw}}/${{w.np}}</div>`;
    h+=`<table class="pt"><thead><tr><th>G</th><th>Ticker</th><th>Sector</th><th>Dir</th><th>Ret%</th><th>PnL</th></tr></thead><tbody>`;
    w.pos.forEach(p=>{{
      let dc=p.di==='long'?'#22c55e':'#ef4444';
      let ar=p.di==='long'?'&#9650;':'&#9660;';
      h+=`<tr><td>${{p.g}}</td><td><strong>${{p.t}}</strong></td><td style="color:#94a3b8;font-size:0.78em">${{p.sc}}</td>
      <td style="color:${{dc}}">${{ar}}</td><td class="${{cls(p.r)}}">${{p.r>=0?'+':''}}${{p.r}}%</td>
      <td class="${{cls(p.p)}}">${{fmt(p.p)}}</td></tr>`}});
    h+=`</tbody></table>`;
  }}
  h+=`</div>`;
  return h;
}}

function renderWeek(){{
  if(!fIdx.length){{document.getElementById('wk-detail').innerHTML='<div class="fuera">No hay semanas</div>';return}}
  let i=fIdx[cur];
  let w1=W1[i],w2=W2[i];
  let sl1=SL1[w1.r]||['',''],sl2=SL2[w2.r]||['',''];
  let hdr=`<div style="grid-column:1/-1;margin-bottom:4px">
    <strong>${{w1.d}}</strong> &mdash; S${{w1.s}} ${{w1.y}}
    <span class="badge" style="background:${{RC[w1.r]}}">${{w1.r}}</span>
    &nbsp; SPY: <span class="${{cls(w1.sr)}}">${{(w1.sr||0)>=0?'+':''}}${{(w1.sr||0).toFixed(2)}}%</span>
    &nbsp; Delta PnL: <span class="diff">${{fmt(w2.p-w1.p)}}</span></div>`;
  document.getElementById('wk-detail').innerHTML=hdr+
    renderWkSide(w1,sl1,'E1','e1')+renderWkSide(w2,sl2,'E2','e2');
}}
applyFilter();
}})();
</script>
</body>
</html>"""

out_path = BASE / 'backtest_comparativa.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML saved to: {out_path}")
print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
