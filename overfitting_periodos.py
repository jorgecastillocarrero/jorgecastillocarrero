"""
HTML: Overfitting por periodo - E2 por regimen x 3 modos x periodos temporales
"""
import re, json, csv, bisect, sys, io
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

COST_E2, SLIP = 20000, 0.003
START_YEAR = 2005

STRAT_E2 = {
    'BURBUJA':      [(0,10,'long'),(-20,-10,'short')],
    'GOLDILOCKS':   [(0,10,'long'),(-20,-10,'short')],
    'ALCISTA':      [(0,10,'long'),(-10,None,'short')],
    'NEUTRAL':      [(10,20,'long'),(-20,-10,'short')],
    'CAUTIOUS':     [(-10,None,'short'),(-20,-10,'short')],
    'BEARISH':      [(-10,None,'short'),(-20,-10,'short')],
    'RECOVERY':     [(0,10,'long'),(-20,-10,'long')],
    'CAPITULACION': [(10,20,'long'),(20,30,'long')],
    'CRISIS':       [(-10,None,'short'),(-20,-10,'short')],
    'PANICO':       [(10,20,'long'),(20,30,'long')],
}

REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
                'RECOVERY','CRISIS','PANICO','CAPITULACION']
REGIME_COLORS = {
    'BURBUJA':'#e91e63','GOLDILOCKS':'#4caf50','ALCISTA':'#2196f3',
    'NEUTRAL':'#ff9800','CAUTIOUS':'#ff5722','BEARISH':'#795548',
    'CRISIS':'#9c27b0','PANICO':'#f44336','CAPITULACION':'#00bcd4','RECOVERY':'#8bc34a',
}

MODE_LABELS = ['Original', 'Hybrid', 'MinDD']
MODE_CSVS = ['data/regimenes_historico.csv', 'data/regimenes_hybrid.csv', 'data/regimenes_mindd.csv']
MODE_COLORS = ['#1e293b', '#16a34a', '#dc2626']

PERIODS = [(2005,2009,'2005-09'), (2010,2014,'2010-14'), (2015,2019,'2015-19'), (2020,2026,'2020-26')]

# Load SPY returns by date
spy_by_date = {}
with open('data/regimenes_historico.csv') as f:
    for row in csv.DictReader(f):
        if 'spy_ret_pct' in row and row['spy_ret_pct']:
            spy_by_date[row['fecha_senal'][:10]] = float(row['spy_ret_pct'])

# Load E2
print("Loading E2...")
with open('acciones_navegable.html', 'r', encoding='utf-8') as f:
    html = f.read()
WEEKS = json.loads(re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))
print(f"  {len(WEEKS)} weeks")

# ═══════════════════════════════════════════════════════════════
# Compute E2 per mode x regime x period
# ═══════════════════════════════════════════════════════════════
# Structure: data[mode][regime][period] = {n, pnl_long, pnl_short, pnl_total, wr, spy_avg, weekly_pnls}
# Also: data[mode][regime]['ALL'] for totals
# Also: data[mode]['ALL'][period] for regime totals per period

ALL_DATA = {}

for mi, (label, csv_path) in enumerate(zip(MODE_LABELS, MODE_CSVS)):
    print(f"Processing {label}...")
    entries = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            entries.append((str(row['fecha_senal'])[:10], row['regime']))
    entries.sort()
    rd = [e[0] for e in entries]
    rv = [e[1] for e in entries]
    def get_r(d):
        idx = bisect.bisect_right(rd, str(d)[:10]) - 1
        return rv[idx] if idx >= 0 else 'UNKNOWN'

    mode_data = {}
    for reg in REGIME_ORDER + ['ALL']:
        mode_data[reg] = {}
        for _, _, plbl in PERIODS:
            mode_data[reg][plbl] = {'n':0, 'pnl_l':0, 'pnl_s':0, 'wr_w':0, 'wr_t':0, 'spy':[], 'wk_pnls':[]}
        mode_data[reg]['TOTAL'] = {'n':0, 'pnl_l':0, 'pnl_s':0, 'wr_w':0, 'wr_t':0, 'spy':[], 'wk_pnls':[]}

    for w in WEEKS:
        if w['y'] < START_YEAR: continue
        reg = get_r(w['d'])
        # Find period
        plbl = None
        for y1, y2, pl in PERIODS:
            if y1 <= w['y'] <= y2:
                plbl = pl
                break
        if not plbl: continue

        strat = STRAT_E2.get(reg, [])
        wk_l = 0
        wk_s = 0
        for start, end, direction in strat:
            sel = w['s'][start:end] if end else w['s'][start:]
            for s in sel:
                rv2 = s[8]
                if rv2 is None: continue
                rv2 = max(-50, min(50, rv2))
                if direction == 'long':
                    wk_l += COST_E2 * (rv2/100 - SLIP)
                else:
                    wk_s += COST_E2 * (-rv2/100 - SLIP)

        wk_total = wk_l + wk_s
        spy_ret = spy_by_date.get(w['d'][:10], None)

        for bucket in [reg, 'ALL']:
            for period_key in [plbl, 'TOTAL']:
                d = mode_data[bucket][period_key]
                d['n'] += 1
                d['pnl_l'] += wk_l
                d['pnl_s'] += wk_s
                d['wk_pnls'].append(wk_total)
                if wk_total > 0: d['wr_w'] += 1
                d['wr_t'] += 1
                if spy_ret is not None: d['spy'].append(spy_ret)

    ALL_DATA[mi] = mode_data

# ═══════════════════════════════════════════════════════════════
# Compute overfitting metrics
# ═══════════════════════════════════════════════════════════════
print("Computing overfitting metrics...")

def compute_stats(d):
    """Compute stats from a bucket dict"""
    n = d['n']
    if n == 0:
        return {'n':0, 'pnl':0, 'pnl_l':0, 'pnl_s':0, 'per_wk':0, 'wr':0, 'spy_avg':0, 'std':0, 'sharpe':0}
    pnl = d['pnl_l'] + d['pnl_s']
    wr = d['wr_w'] / d['wr_t'] * 100 if d['wr_t'] > 0 else 0
    spy_avg = np.mean(d['spy']) if d['spy'] else 0
    wk = np.array(d['wk_pnls'])
    std = float(np.std(wk)) if len(wk) > 1 else 0
    sharpe = float(np.mean(wk) / np.std(wk) * np.sqrt(52)) if std > 0 else 0
    return {
        'n': n, 'pnl': round(pnl), 'pnl_l': round(d['pnl_l']), 'pnl_s': round(d['pnl_s']),
        'per_wk': round(pnl / n), 'wr': round(wr, 1), 'spy_avg': round(spy_avg, 3),
        'std': round(std), 'sharpe': round(sharpe, 2)
    }

# Build JSON for HTML
# stats_json[mode][regime][period] = {n, pnl, pnl_l, pnl_s, per_wk, wr, spy_avg, std, sharpe}
stats_json = {}
for mi in range(3):
    stats_json[mi] = {}
    for reg in REGIME_ORDER + ['ALL']:
        stats_json[mi][reg] = {}
        for _, _, plbl in PERIODS:
            stats_json[mi][reg][plbl] = compute_stats(ALL_DATA[mi][reg][plbl])
        stats_json[mi][reg]['TOTAL'] = compute_stats(ALL_DATA[mi][reg]['TOTAL'])

# Overfitting score per regime: CV of per_wk across periods, and max_diff across modes
overfitting = {}
for reg in REGIME_ORDER:
    # Cross-mode variance (for TOTAL period)
    mode_pnls = [stats_json[mi][reg]['TOTAL']['pnl'] for mi in range(3)]
    max_diff_modes = max(mode_pnls) - min(mode_pnls)

    # Cross-period variance per mode
    period_vars = []
    for mi in range(3):
        per_wks = [stats_json[mi][reg][pl]['per_wk'] for _, _, pl in PERIODS if stats_json[mi][reg][pl]['n'] > 0]
        if len(per_wks) >= 2:
            period_vars.append(float(np.std(per_wks)))
        else:
            period_vars.append(0)

    # SPY discrimination consistency
    spy_avgs = []
    for _, _, pl in PERIODS:
        spys = [stats_json[mi][reg][pl]['spy_avg'] for mi in range(3)]
        spy_avgs.append(np.mean(spys))

    spy_cv = float(np.std(spy_avgs) / abs(np.mean(spy_avgs)) * 100) if np.mean(spy_avgs) != 0 else 999

    overfitting[reg] = {
        'max_diff_modes': max_diff_modes,
        'period_var': [round(v) for v in period_vars],
        'avg_period_var': round(np.mean(period_vars)),
        'spy_cv': round(spy_cv, 1),
        'score': round(max_diff_modes / 1000 + np.mean(period_vars) / 100, 1)
    }

# ═══════════════════════════════════════════════════════════════
# Generate HTML
# ═══════════════════════════════════════════════════════════════
print("Generating HTML...")

period_labels = [pl for _, _, pl in PERIODS] + ['TOTAL']

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Overfitting por Periodo - E2 x 3 Modos</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#fff;color:#1e293b;padding:14px;font-size:13px}}
h1{{text-align:center;font-size:1.3em;margin-bottom:4px}}
.sub{{text-align:center;color:#64748b;margin-bottom:14px;font-size:0.8em}}
.card{{background:#f8fafc;border-radius:10px;padding:16px;margin-bottom:14px;border:1px solid #e2e8f0}}
.card h2{{font-size:1em;color:#0f172a;margin-bottom:10px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:0.78em}}
th{{background:#e2e8f0;color:#1e293b;padding:5px 4px;text-align:center;position:sticky;top:0;font-size:0.72em;white-space:nowrap}}
td{{padding:4px 3px;border-bottom:1px solid #e2e8f0;white-space:nowrap;text-align:center}}
tr:hover td{{background:#f1f5f9}}
.pos{{color:#16a34a}}.neg{{color:#dc2626}}
.rb{{display:inline-block;padding:2px 6px;border-radius:3px;font-size:0.7em;font-weight:bold;color:#fff}}
.of-low{{background:#dcfce7;color:#166534}}.of-mid{{background:#fef9c3;color:#854d0e}}.of-high{{background:#fee2e2;color:#991b1b}}
.mode-hdr{{font-weight:700;font-size:0.8em;padding:6px 4px}}
.sep-row td{{border-top:2px solid #475569;font-weight:700}}
.chart-wrap{{position:relative;height:320px;background:#fff;border:1px solid #e2e8f0;border-radius:7px;overflow:hidden;margin-bottom:4px}}
.chart-wrap canvas{{width:100%!important;height:100%!important}}
.tabs{{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}}
.tab{{padding:5px 12px;border-radius:5px;cursor:pointer;font-size:0.78em;border:1px solid #cbd5e1;background:#fff}}
.tab.active{{background:#1e293b;color:#fff;border-color:#1e293b}}
.legend{{text-align:center;font-size:0.72em;color:#64748b;margin:4px 0;line-height:2}}
</style></head>
<body>
<h1>Analisis de Overfitting por Periodo</h1>
<div class="sub">E2 PnL por regimen x 3 modos x 4 periodos | {START_YEAR}-2026 | Detectar inestabilidad temporal y entre modos</div>

<!-- 1. Overfitting Summary -->
<div class="card"><h2>1. Score de Overfitting por Regimen</h2>
<p style="font-size:0.75em;color:#64748b;margin-bottom:8px">
MaxDiff = diferencia maxima de PnL entre los 3 modos. Var Periodo = variabilidad del EUR/semana entre periodos.
SPY CV = coeficiente de variacion del retorno SPY entre periodos (estabilidad de la clasificacion).
Overfitting alto = resultados dependen del modo o del periodo, no de una ventaja real.</p>
<table>
<thead><tr><th>Regimen</th><th>Sem</th><th>MaxDiff Modos</th><th>Var Orig</th><th>Var Hyb</th><th>Var MinDD</th><th>SPY CV%</th><th>Overfitting</th></tr></thead>
<tbody id="of-body"></tbody></table></div>

<!-- 2. Tabla completa por regimen -->
<div class="card"><h2>2. E2 PnL por Regimen y Periodo</h2>
<div class="tabs" id="reg-tabs"></div>
<div id="reg-content"></div></div>

<!-- 3. Heatmap: EUR/semana por regimen x periodo x modo -->
<div class="card"><h2>3. Heatmap: EUR/semana por Regimen x Periodo</h2>
<div class="tabs" id="hm-tabs"></div>
<div id="hm-content"></div></div>

<!-- 4. Retorno SPY por regimen x periodo -->
<div class="card"><h2>4. Retorno SPY medio por Regimen x Periodo (validacion de clasificacion)</h2>
<table>
<thead><tr><th>Regimen</th>"""

for _, _, pl in PERIODS:
    html += f"<th>{pl}<br>SPY%</th>"
html += "<th>TOTAL<br>SPY%</th><th>Rango</th><th>Estable?</th></tr></thead><tbody id='spy-body'></tbody></table></div>"

# 5. Bar chart per period
html += """
<!-- 5. PnL por periodo -->
<div class="card"><h2>5. PnL Total E2 por Periodo</h2>
<div class="chart-wrap" style="height:260px"><canvas id="barChart"></canvas></div>
<div class="legend">
<span style="color:#1e293b">&#9632; Original</span> &nbsp;
<span style="color:#16a34a">&#9632; Hybrid</span> &nbsp;
<span style="color:#dc2626">&#9632; MinDD</span>
</div></div>

<!-- 6. Conclusion -->
<div class="card"><h2>6. Diagnostico</h2>
<div id="diag"></div></div>
"""

html += f"""
<script>
const S={json.dumps(stats_json, separators=(',',':'))};
const OF={json.dumps(overfitting, separators=(',',':'))};
const RO={json.dumps(REGIME_ORDER)};
const RC={json.dumps(REGIME_COLORS, separators=(',',':'))};
const ML={json.dumps(MODE_LABELS)};
const PL={json.dumps(period_labels)};
const MC={json.dumps(MODE_COLORS)};

const fmt=v=>{{let s=v<0?'-':'';return s+'EUR '+Math.abs(v).toLocaleString('en-US')}};
const cls=v=>v>=0?'pos':'neg';
const fc=(v,d)=>{{d=d||2;return(v>=0?'+':'')+v.toFixed(d)+'%'}};

// 1. Overfitting table
(function(){{
  let h='';
  RO.forEach(r=>{{
    const o=OF[r];
    const n=S[0][r]['TOTAL'].n;
    let level='of-low',label='BAJO';
    if(o.max_diff_modes>100000||o.avg_period_var>3000){{level='of-high';label='ALTO'}}
    else if(o.max_diff_modes>50000||o.avg_period_var>1500){{level='of-mid';label='MEDIO'}}
    h+=`<tr><td><span class="rb" style="background:${{RC[r]}}">${{r}}</span></td>
    <td>${{n}}</td>
    <td class="${{o.max_diff_modes>100000?'neg':o.max_diff_modes>50000?'neg':''}}">${{fmt(o.max_diff_modes)}}</td>
    <td>${{fmt(o.period_var[0])}}</td><td>${{fmt(o.period_var[1])}}</td><td>${{fmt(o.period_var[2])}}</td>
    <td>${{o.spy_cv>200?'<span class="neg">'+o.spy_cv+'%</span>':o.spy_cv+'%'}}</td>
    <td><span class="${{level}}" style="padding:2px 8px;border-radius:3px;font-weight:700">${{label}}</span></td></tr>`;
  }});
  document.getElementById('of-body').innerHTML=h;
}})();

// 2. Regime detail tables
(function(){{
  const tabs=document.getElementById('reg-tabs');
  const content=document.getElementById('reg-content');
  let tabH='';
  RO.concat(['ALL']).forEach((r,i)=>{{
    tabH+=`<div class="tab ${{i===0?'active':''}}" onclick="showReg('${{r}}',this)">${{r==='ALL'?'TOTAL':r}}</div>`;
  }});
  tabs.innerHTML=tabH;

  window.showReg=function(reg,el){{
    tabs.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    el.classList.add('active');
    let h='<table><thead><tr><th>Modo</th><th>Periodo</th><th>Sem</th><th>PnL Long</th><th>PnL Short</th><th>PnL Total</th><th>EUR/sem</th><th>WR%</th><th>Sharpe</th><th>SPY%</th></tr></thead><tbody>';
    for(let mi=0;mi<3;mi++){{
      h+=`<tr><td colspan="10" class="mode-hdr" style="text-align:left;color:${{MC[mi]}}">${{ML[mi]}}</td></tr>`;
      PL.forEach(pl=>{{
        const s=S[mi][reg][pl];
        const isTot=pl==='TOTAL';
        h+=`<tr class="${{isTot?'sep-row':''}}">
        <td></td><td>${{pl}}</td><td>${{s.n}}</td>
        <td class="${{cls(s.pnl_l)}}">${{fmt(s.pnl_l)}}</td>
        <td class="${{cls(s.pnl_s)}}">${{fmt(s.pnl_s)}}</td>
        <td class="${{cls(s.pnl)}}" style="font-weight:${{isTot?700:400}}">${{fmt(s.pnl)}}</td>
        <td class="${{cls(s.per_wk)}}">${{fmt(s.per_wk)}}</td>
        <td class="${{s.wr>=50?'pos':'neg'}}">${{s.wr}}%</td>
        <td class="${{s.sharpe>=0?'pos':'neg'}}">${{s.sharpe.toFixed(2)}}</td>
        <td class="${{cls(s.spy_avg)}}">${{fc(s.spy_avg,3)}}</td></tr>`;
      }});
    }}
    // Add cross-mode comparison row
    h+='<tr><td colspan="10" class="mode-hdr" style="text-align:left;color:#64748b">Diferencia entre modos</td></tr>';
    PL.forEach(pl=>{{
      const pnls=[0,1,2].map(mi=>S[mi][reg][pl].pnl);
      const mx=Math.max(...pnls),mn=Math.min(...pnls),diff=mx-mn;
      const perWks=[0,1,2].map(mi=>S[mi][reg][pl].per_wk);
      const mxW=Math.max(...perWks),mnW=Math.min(...perWks);
      h+=`<tr style="background:#fef3c7"><td></td><td>${{pl}}</td><td></td><td></td><td></td>
      <td class="${{diff>100000?'neg':''}}">${{fmt(diff)}}</td>
      <td class="${{(mxW-mnW)>2000?'neg':''}}">${{fmt(mxW-mnW)}}</td>
      <td></td><td></td><td></td></tr>`;
    }});
    h+='</tbody></table>';
    content.innerHTML=h;
  }};
  showReg('BURBUJA',tabs.querySelector('.tab'));
}})();

// 3. Heatmap
(function(){{
  const tabs=document.getElementById('hm-tabs');
  const content=document.getElementById('hm-content');
  let tabH='';
  ML.forEach((m,i)=>{{
    tabH+=`<div class="tab ${{i===0?'active':''}}" onclick="showHM(${{i}},this)">${{m}}</div>`;
  }});
  tabs.innerHTML=tabH;

  window.showHM=function(mi,el){{
    tabs.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    el.classList.add('active');
    let h='<table><thead><tr><th>Regimen</th>';
    PL.forEach(pl=>h+=`<th>${{pl}}</th>`);
    h+='</tr></thead><tbody>';
    // Find max abs for color scale
    let maxAbs=0;
    RO.forEach(r=>PL.forEach(pl=>{{maxAbs=Math.max(maxAbs,Math.abs(S[mi][r][pl].per_wk))}}));
    RO.forEach(r=>{{
      h+=`<tr><td><span class="rb" style="background:${{RC[r]}}">${{r}}</span></td>`;
      PL.forEach(pl=>{{
        const v=S[mi][r][pl].per_wk;
        const n=S[mi][r][pl].n;
        const intensity=Math.min(Math.abs(v)/maxAbs,1);
        let bg;
        if(v>0)bg=`rgba(34,197,94,${{intensity*0.5}})`;
        else if(v<0)bg=`rgba(239,68,68,${{intensity*0.5}})`;
        else bg='transparent';
        h+=`<td style="background:${{bg}};font-weight:700" title="N=${{n}}">${{n>0?fmt(v):'-'}}</td>`;
      }});
      h+='</tr>';
    }});
    // Total row
    h+='<tr class="sep-row"><td><b>TOTAL</b></td>';
    PL.forEach(pl=>{{
      const v=S[mi]['ALL'][pl].per_wk;
      h+=`<td class="${{cls(v)}}" style="font-weight:700">${{fmt(v)}}</td>`;
    }});
    h+='</tr></tbody></table>';
    content.innerHTML=h;
  }};
  showHM(0,tabs.querySelector('.tab'));
}})();

// 4. SPY table
(function(){{
  let h='';
  RO.forEach(r=>{{
    h+=`<tr><td><span class="rb" style="background:${{RC[r]}}">${{r}}</span></td>`;
    let vals=[];
    PL.forEach(pl=>{{
      // Average SPY across modes for this regime+period
      const spys=[0,1,2].map(mi=>S[mi][r][pl]);
      const avgN=spys.reduce((a,s)=>a+s.n,0)/3;
      const avgSpy=spys.reduce((a,s)=>a+s.spy_avg,0)/3;
      vals.push(avgSpy);
      const c=avgSpy>=0?'pos':'neg';
      h+=`<td class="${{c}}">${{avgN>0?fc(avgSpy,3):'-'}}</td>`;
    }});
    const nonZero=vals.filter(v=>v!==0);
    const rng=nonZero.length>1?Math.max(...nonZero)-Math.min(...nonZero):0;
    const stable=rng<0.3;
    h+=`<td>${{rng.toFixed(3)}}%</td>`;
    h+=`<td><span class="${{stable?'of-low':'of-high'}}" style="padding:2px 6px;border-radius:3px;font-weight:700">${{stable?'SI':'NO'}}</span></td></tr>`;
  }});
  document.getElementById('spy-body').innerHTML=h;
}})();

// 5. Bar chart
function drawBar(){{
  const cv=document.getElementById('barChart'),cx=cv.getContext('2d');
  const W=cv.parentElement.clientWidth,H=cv.parentElement.clientHeight;
  cv.width=W*2;cv.height=H*2;cx.scale(2,2);
  const periods=PL.slice(0,-1); // exclude TOTAL
  const n=periods.length;
  const pad={{t:14,b:24,l:60,r:14}},w=W-pad.l-pad.r,h=H-pad.t-pad.b;
  let all=[];
  periods.forEach(pl=>{{for(let mi=0;mi<3;mi++)all.push(S[mi]['ALL'][pl].pnl)}});
  all.push(0);
  const mn=Math.min(...all),mx=Math.max(...all),range=mx-mn||1,yS=h/range;
  cx.clearRect(0,0,W,H);
  let y0=pad.t+h-(0-mn)*yS;
  cx.strokeStyle='#cbd5e1';cx.lineWidth=1;
  cx.beginPath();cx.moveTo(pad.l,y0);cx.lineTo(W-pad.r,y0);cx.stroke();
  // Grid
  cx.strokeStyle='#e2e8f0';cx.lineWidth=0.5;cx.fillStyle='#64748b';cx.font='9px sans-serif';cx.textAlign='right';
  for(let i=0;i<=5;i++){{
    const v=mn+(range/5)*i;const yy=pad.t+h-(v-mn)*yS;
    cx.beginPath();cx.moveTo(pad.l,yy);cx.lineTo(W-pad.r,yy);cx.stroke();
    cx.fillText((v/1000).toFixed(0)+'K',pad.l-4,yy+3);
  }}
  const grpW=w/n,bW=Math.min(grpW/4,30);
  periods.forEach((pl,i)=>{{
    const cx2=pad.l+i*grpW+grpW/2;
    for(let mi=0;mi<3;mi++){{
      const v=S[mi]['ALL'][pl].pnl;
      const bx=cx2+(mi-1)*bW*1.2;
      const by=pad.t+h-(v-mn)*yS;
      const bh=Math.abs(v)*yS;
      cx.fillStyle=MC[mi];
      if(v>=0)cx.fillRect(bx-bW/2,by,bW,bh);
      else cx.fillRect(bx-bW/2,y0,bW,bh);
    }}
    cx.fillStyle='#1e293b';cx.font='11px sans-serif';cx.textAlign='center';
    cx.fillText(pl,cx2,H-pad.b+14);
  }});
}}
drawBar();window.addEventListener('resize',drawBar);

// 6. Diagnostic
(function(){{
  let h='<div style="font-size:0.82em;line-height:1.7">';
  h+='<p style="font-weight:700;margin-bottom:6px">Regimenes con ALTO overfitting (resultados no fiables):</p><ul>';
  RO.forEach(r=>{{
    const o=OF[r];
    if(o.max_diff_modes>100000||o.avg_period_var>3000){{
      h+=`<li><span class="rb" style="background:${{RC[r]}}">${{r}}</span> MaxDiff modos: ${{fmt(o.max_diff_modes)}} | Var periodo: ${{fmt(o.avg_period_var)}}</li>`;
    }}
  }});
  h+='</ul><p style="font-weight:700;margin:10px 0 6px">Regimenes ESTABLES (fiables para produccion):</p><ul>';
  RO.forEach(r=>{{
    const o=OF[r];
    if(o.max_diff_modes<=50000&&o.avg_period_var<=1500){{
      h+=`<li><span class="rb" style="background:${{RC[r]}}">${{r}}</span> MaxDiff: ${{fmt(o.max_diff_modes)}} | Var: ${{fmt(o.avg_period_var)}}</li>`;
    }}
  }});
  h+='</ul><p style="font-weight:700;margin:10px 0 6px">Resumen:</p><ul>';
  const stable=RO.filter(r=>OF[r].max_diff_modes<=50000&&OF[r].avg_period_var<=1500);
  const unstable=RO.filter(r=>OF[r].max_diff_modes>100000||OF[r].avg_period_var>3000);
  h+=`<li>Estables: ${{stable.join(', ')}} (${{stable.length}}/10)</li>`;
  h+=`<li>Inestables: ${{unstable.join(', ')}} (${{unstable.length}}/10)</li>`;
  h+=`<li>Conclusion: los regimenes de crisis/transicion son sensibles al modo de clasificacion y al periodo temporal.</li>`;
  h+='</ul></div>';
  document.getElementById('diag').innerHTML=h;
}})();
</script></body></html>"""

out_path = Path('.') / 'overfitting_periodos.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML: {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
