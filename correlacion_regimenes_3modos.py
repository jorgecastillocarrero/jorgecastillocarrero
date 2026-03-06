"""
Estadisticas y correlaciones de los 3 modos (Original, Hybrid, MinDD)
por regimen y totales, con SPY/QQQ/Gold
"""
import re, json, csv, bisect, sys, io
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import groupby
from sqlalchemy import create_engine

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
BASE = Path('.')
engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

COST_3DH, COST_E2, SLIP = 25000, 20000, 0.003
CAP_3DH, CAP_E2 = 200000, 400000
CAP_TOTAL = CAP_3DH + CAP_E2
MAX_3DH, START_YEAR = 30, 2005

H3_ACTIVE = {'CAUTIOUS', 'RECOVERY', 'CRISIS', 'PANICO'}
E2_ACTIVE = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'BEARISH', 'CAPITULACION', 'RECOVERY'}

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

MODE_LABELS = ['Original', 'Hybrid', 'Min DD']
MODE_CSVS = ['data/regimenes_historico.csv', 'data/regimenes_hybrid.csv', 'data/regimenes_mindd.csv']
MODE_COLORS = ['#1e293b', '#16a34a', '#dc2626']

# ═══════════════════════════════════════════════════════════════
# 1. LOAD ASSETS
# ═══════════════════════════════════════════════════════════════
print("Loading assets...")
asset_prices = {}
for sym in ['SPY', 'QQQ', 'GLD']:
    df = pd.read_sql(f"SELECT date, close FROM fmp_price_history WHERE symbol='{sym}' AND date >= '2004-01-01' ORDER BY date", engine)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    wk = df['close'].resample('W-FRI').last().dropna()
    asset_prices[sym] = wk.pct_change() * 100
    print(f"  {sym}: {len(wk)} weeks")

# ═══════════════════════════════════════════════════════════════
# 2. LOAD 3DH
# ═══════════════════════════════════════════════════════════════
print("Loading 3DH...")
with open('data/3dh_opt_4d_trades.json') as f:
    h3_raw = json.load(f)
h3_all = [{'sym':t['sym'],'sig':t['sig'],'entry':t['entry'],'exit':t['exit'],
    'ret':t['ret'],'pnl':t['pnl'],'contrast':t.get('d5',0)-t.get('d50',0)} for t in h3_raw]
h3_all.sort(key=lambda t: t['entry'])
h3_trades = []
open_by_exit = defaultdict(int)
for ed, grp in groupby(h3_all, key=lambda t: t['entry']):
    dt = list(grp)
    exp = [ex for ex in open_by_exit if ex <= ed]
    for ex in exp: del open_by_exit[ex]
    avail = max(0, MAX_3DH - sum(open_by_exit.values()))
    if avail <= 0: continue
    if len(dt) <= avail:
        for t in dt: h3_trades.append(t); open_by_exit[t['exit']] += 1
    else:
        dt.sort(key=lambda t: -t['contrast'])
        for i in range(avail): h3_trades.append(dt[i]); open_by_exit[dt[i]['exit']] += 1
print(f"  {len(h3_trades)} trades")

# ═══════════════════════════════════════════════════════════════
# 3. LOAD E2
# ═══════════════════════════════════════════════════════════════
print("Loading E2...")
with open('acciones_navegable.html', 'r', encoding='utf-8') as f:
    html = f.read()
WEEKS = json.loads(re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))
print(f"  {len(WEEKS)} weeks")

# ═══════════════════════════════════════════════════════════════
# 4. COMPUTE WEEKLY PNL FOR EACH MODE
# ═══════════════════════════════════════════════════════════════
mode_data = []  # list of [{date, regime, e2, h3, dual, dual_ret}, ...]

for mi, (label, csv_path) in enumerate(zip(MODE_LABELS, MODE_CSVS)):
    print(f"\nProcessing {label}...")
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

    # 3DH by signal date - ALL regimes (no filter)
    h3_by_sig = defaultdict(float)
    for t in h3_trades:
        h3_by_sig[t['sig']] += t['pnl']

    # E2 + 3DH weekly
    weeks_data = []
    for w in WEEKS:
        if w['y'] < START_YEAR: continue
        reg = get_r(w['d'])

        # E2 - ALL regimes (no filter)
        e2_pnl = 0
        strat = STRAT_E2.get(reg, [])
        if strat:
            for start, end, direction in strat:
                sel = w['s'][start:end] if end else w['s'][start:]
                for s in sel:
                    rv2 = s[8]
                    if rv2 is None: continue
                    rv2 = max(-50, min(50, rv2))
                    if direction == 'long': e2_pnl += COST_E2 * (rv2/100 - SLIP)
                    else: e2_pnl += COST_E2 * (-rv2/100 - SLIP)

        # 3DH for this week
        dt = pd.Timestamp(w['d'])
        h3_pnl = 0
        for delta in range(0, 7):
            check = (dt - pd.Timedelta(days=delta)).strftime('%Y-%m-%d')
            if check in h3_by_sig: h3_pnl += h3_by_sig[check]

        dual = e2_pnl + h3_pnl
        weeks_data.append({
            'date': w['d'], 'regime': reg,
            'e2': round(e2_pnl, 2), 'h3': round(h3_pnl, 2),
            'dual': round(dual, 2), 'dual_ret': round(dual / CAP_TOTAL * 100, 4),
        })

    mode_data.append(weeks_data)
    total = sum(w['dual'] for w in weeks_data)
    print(f"  {label}: {len(weeks_data)} weeks, PnL EUR {total:,.0f}")

# ═══════════════════════════════════════════════════════════════
# 5. BUILD ALIGNED DATAFRAME
# ═══════════════════════════════════════════════════════════════
print("\nBuilding aligned dataframe...")
rows = []
for wi in range(len(mode_data[0])):
    w0 = mode_data[0][wi]
    dt = pd.Timestamp(w0['date'])
    fri = dt
    while fri.weekday() != 4: fri += pd.Timedelta(days=1)

    row = {'date': w0['date']}
    for mi in range(3):
        w = mode_data[mi][wi]
        pfx = ['orig', 'hyb', 'mindd'][mi]
        row[f'{pfx}_regime'] = w['regime']
        row[f'{pfx}_e2'] = w['e2']
        row[f'{pfx}_h3'] = w['h3']
        row[f'{pfx}_dual'] = w['dual']
        row[f'{pfx}_ret'] = w['dual_ret']

    for sym in ['SPY', 'QQQ', 'GLD']:
        ret_s = asset_prices[sym]
        cands = ret_s.index[(ret_s.index >= fri - pd.Timedelta(days=3)) & (ret_s.index <= fri + pd.Timedelta(days=3))]
        row[sym.lower()] = round(float(ret_s[cands[0]]), 3) if len(cands) > 0 else None

    rows.append(row)

df = pd.DataFrame(rows).dropna(subset=['spy'])
print(f"  {len(df)} aligned weeks")

# ═══════════════════════════════════════════════════════════════
# 6. STATS PER REGIME PER MODE (3DH + E2 + DUAL)
# ═══════════════════════════════════════════════════════════════
print("\nComputing per-regime stats...")

def regime_stats(df_sub, ret_col):
    r = df_sub[ret_col].values
    n = len(r)
    if n == 0:
        return {'n':0,'pnl':0,'mean':0,'wr':0,'pf':0,'std':0}
    mean = float(np.mean(r))
    wins = int(np.sum(r > 0))
    wr = wins / n * 100
    gw = float(np.sum(r[r > 0]))
    gl = abs(float(np.sum(r[r < 0])))
    pf = gw / gl if gl > 0 else 99.9
    return {'n': n, 'pnl': round(float(np.sum(r))), 'mean': round(mean),
            'wr': round(wr, 1), 'pf': round(min(pf, 99.9), 2), 'std': round(float(np.std(r)))}

# Per-regime stats for each mode
regime_stats_all = {}  # {mode_idx: {regime: {dual, e2, h3, spy stats}}}
for mi, pfx in enumerate(['orig', 'hyb', 'mindd']):
    regime_stats_all[mi] = {}
    for reg in REGIME_ORDER:
        sub = df[df[f'{pfx}_regime'] == reg]
        regime_stats_all[mi][reg] = {
            'dual': regime_stats(sub, f'{pfx}_dual'),
            'e2': regime_stats(sub, f'{pfx}_e2'),
            'h3': regime_stats(sub, f'{pfx}_h3'),
            'spy': regime_stats(sub, 'spy') if len(sub) > 0 else {'n':0},
        }

# ═══════════════════════════════════════════════════════════════
# 7. CORRELATIONS
# ═══════════════════════════════════════════════════════════════
print("Computing correlations...")

# Total weekly correlations (all 3 modes + assets)
corr_data = {
    'Original': df['orig_ret'].values,
    'Hybrid': df['hyb_ret'].values,
    'MinDD': df['mindd_ret'].values,
    'SPY': df['spy'].values,
    'QQQ': df['qqq'].values,
    'Gold': df['gld'].values,
}
weekly_corr = pd.DataFrame(corr_data).corr().round(4)
print("\n  Weekly correlations:")
print(weekly_corr.to_string())

# Monthly
df['month'] = df['date'].str[:7]
def cg(s):
    v = s.dropna()
    return round((np.prod(1 + v/100) - 1) * 100, 2) if len(v) > 0 else 0
monthly = df.groupby('month').agg(
    orig=('orig_ret', 'sum'), hyb=('hyb_ret', 'sum'), mindd=('mindd_ret', 'sum'),
    spy=('spy', cg), qqq=('qqq', cg), gld=('gld', cg),
).reset_index()
monthly_corr = monthly[['orig','hyb','mindd','spy','qqq','gld']].corr().round(4)
monthly_corr.columns = ['Original','Hybrid','MinDD','SPY','QQQ','Gold']
monthly_corr.index = ['Original','Hybrid','MinDD','SPY','QQQ','Gold']
print("\n  Monthly correlations:")
print(monthly_corr.to_string())

# Per-regime correlations (orig_dual vs spy, hyb_dual vs spy, etc.)
print("\n  Per-regime correlation with SPY:")
for reg in REGIME_ORDER:
    sub = df[df['orig_regime'] == reg]
    if len(sub) < 10: continue
    c_orig = np.corrcoef(sub['orig_ret'], sub['spy'])[0,1]
    sub_h = df[df['hyb_regime'] == reg]
    c_hyb = np.corrcoef(sub_h['hyb_ret'], sub_h['spy'])[0,1] if len(sub_h) >= 10 else 0
    sub_m = df[df['mindd_regime'] == reg]
    c_mindd = np.corrcoef(sub_m['mindd_ret'], sub_m['spy'])[0,1] if len(sub_m) >= 10 else 0
    print(f"    {reg:15s}: Orig {c_orig:+.4f} | Hyb {c_hyb:+.4f} | MinDD {c_mindd:+.4f}")

# ═══════════════════════════════════════════════════════════════
# 8. TOTAL STATS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("  ESTADISTICAS TOTALES")
print("="*100)

def total_stats(rets, label):
    r = np.array(rets, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        return {'label': label, 'n': 0, 'total': 0, 'cagr': 0, 'max_dd': 0,
                'sharpe': 0, 'sortino': 0, 'wr': 0, 'mean': 0, 'std': 0}
    cum = np.cumprod(1 + r/100)
    total = round((cum[-1]-1)*100, 2)
    pk = np.maximum.accumulate(cum)
    dd = (cum - pk) / pk * 100
    max_dd = round(float(dd.min()), 2)
    cagr = round((cum[-1]**(1/(n/52)) - 1)*100, 2)
    sharpe = round(float(np.mean(r)/np.std(r)*np.sqrt(52)), 2)
    down = r[r < 0]
    sortino = round(float(np.mean(r)/np.std(down)*np.sqrt(52)), 2) if len(down) > 0 and np.std(down) > 0 else 0
    wr = round(np.sum(r > 0)/n*100, 1)
    return {'label': label, 'n': n, 'total': total, 'cagr': cagr, 'max_dd': max_dd,
            'sharpe': sharpe, 'sortino': sortino, 'wr': wr,
            'mean': round(float(np.mean(r)), 3), 'std': round(float(np.std(r)), 3)}

all_stats = [
    total_stats(df['orig_ret'], 'Original'),
    total_stats(df['hyb_ret'], 'Hybrid'),
    total_stats(df['mindd_ret'], 'MinDD'),
    total_stats(df['spy'], 'SPY'),
    total_stats(df['qqq'], 'QQQ'),
    total_stats(df['gld'], 'Gold'),
]

print(f"\n  {'':20s} {'Ret Total':>10s} {'CAGR':>7s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>8s} {'WR':>6s} {'Media':>7s} {'Std':>7s}")
print(f"  {'-'*20} {'-'*10} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*7}")
for s in all_stats:
    print(f"  {s['label']:20s} {s['total']:+9.1f}% {s['cagr']:+6.1f}% {s['sharpe']:+6.2f} {s['sortino']:+7.2f} {s['max_dd']:+7.1f}% {s['wr']:5.1f}% {s['mean']:+6.3f}% {s['std']:6.3f}%")

# ═══════════════════════════════════════════════════════════════
# 9. HTML GENERATION
# ═══════════════════════════════════════════════════════════════
print("\nGenerating HTML...")

# Prepare JSON data
W_CORR = weekly_corr.values.tolist()
M_CORR = monthly_corr.values.tolist()
CORR_LABELS = ['Original','Hybrid','MinDD','SPY','QQQ','Gold']

# Per-regime stats JSON: {mode_idx: {regime: {dual:{n,pnl,wr,pf}, e2:{...}, h3:{...}, spy:{...}}}}
RS_JSON = json.dumps(regime_stats_all, separators=(',',':'))

# Annual data
df['year'] = df['date'].str[:4].astype(int)
annual = df.groupby('year').agg(
    orig=('orig_dual', 'sum'), hyb=('hyb_dual', 'sum'), mindd=('mindd_dual', 'sum'),
    spy=('spy', cg), qqq=('qqq', cg), gld=('gld', cg), n=('date', 'count'),
).reset_index()
A_JSON = [[int(r['year']), round(r['orig']/CAP_TOTAL*100,2), round(r['hyb']/CAP_TOTAL*100,2),
           round(r['mindd']/CAP_TOTAL*100,2), round(r['spy'],2), round(r['qqq'],2), round(r['gld'],2),
           int(r['orig']), int(r['hyb']), int(r['mindd']), int(r['n'])] for _, r in annual.iterrows()]

# Equity curves
def ceq(rets):
    r = np.array(rets, dtype=float)
    r = np.where(np.isnan(r), 0, r)
    return [round(v,2) for v in ((np.cumprod(1+r/100)-1)*100).tolist()]
EQ = [ceq(df['orig_ret']), ceq(df['hyb_ret']), ceq(df['mindd_ret']),
      ceq(df['spy']), ceq(df['qqq']), ceq(df['gld'])]

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>3 Modos: Original vs Hybrid vs MinDD + Activos</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#fff;color:#1e293b;padding:14px}}
h1{{text-align:center;font-size:1.4em;margin-bottom:4px}}
.sub{{text-align:center;color:#64748b;margin-bottom:14px;font-size:0.8em;line-height:1.5}}
.card{{background:#f8fafc;border-radius:10px;padding:16px;margin-bottom:12px;border:1px solid #e2e8f0}}
.card h2{{font-size:1.05em;color:#0f172a;margin-bottom:10px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:0.78em}}
th{{background:#e2e8f0;color:#1e293b;padding:6px 4px;text-align:left;position:sticky;top:0;font-size:0.73em;white-space:nowrap;z-index:1}}
td{{padding:5px 4px;border-bottom:1px solid #e2e8f0;white-space:nowrap}}
tr:hover td{{background:#f1f5f9}}
.pos{{color:#16a34a}}.neg{{color:#dc2626}}
.side2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.side3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}}
@media(max-width:900px){{.side2,.side3{{grid-template-columns:1fr}}}}
.chart-wrap{{position:relative;height:320px;background:#fff;border:1px solid #e2e8f0;border-radius:7px;overflow:hidden;margin-bottom:4px}}
.chart-wrap canvas{{width:100%!important;height:100%!important}}
.legend{{text-align:center;font-size:0.72em;color:#64748b;margin-top:2px;line-height:2}}
.corr-tbl td,.corr-tbl th{{text-align:center;padding:8px 5px;font-size:0.8em}}
.corr-tbl td{{font-weight:700;min-width:65px}}
.rb{{display:inline-block;padding:2px 8px;border-radius:3px;font-size:0.7em;font-weight:bold;color:#fff}}
.scroll-t{{max-height:600px;overflow-y:auto}}
.best{{font-weight:700;text-decoration:underline}}
</style></head>
<body>
<h1>Comparativa 3 Modos + Activos</h1>
<div class="sub">
Original vs Hybrid (avg score) vs Min DD | SPY, QQQ, Gold | {START_YEAR}-2026 | {len(df)} semanas
</div>

<!-- 1. Stats totales -->
<div class="card"><h2>1. Estadisticas Totales</h2>
<table>
<thead><tr><th>Activo</th><th>Ret Total</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th>
<th>Max DD</th><th>WR</th><th>Media Sem</th><th>Std Sem</th></tr></thead>
<tbody id="st-body"></tbody></table></div>

<!-- 2. Equity -->
<div class="card"><h2>2. Equity Curves (retorno compuesto %)</h2>
<div class="chart-wrap"><canvas id="eqChart"></canvas></div>
<div class="legend">
<span style="color:#1e293b">&#9644; Original</span> &nbsp;
<span style="color:#16a34a">&#9644; Hybrid</span> &nbsp;
<span style="color:#dc2626">&#9644; Min DD</span> &nbsp;
<span style="color:#2563eb">&#9644; SPY</span> &nbsp;
<span style="color:#7c3aed">&#9644; QQQ</span> &nbsp;
<span style="color:#d97706">&#9644; Gold</span>
</div></div>

<!-- 3. Correlaciones -->
<div class="card"><h2>3. Matrices de Correlacion</h2>
<div class="side2">
<div><h3 style="text-align:center;font-size:0.9em;margin-bottom:8px">Semanal</h3>
<table class="corr-tbl" id="wcorr"></table></div>
<div><h3 style="text-align:center;font-size:0.9em;margin-bottom:8px">Mensual</h3>
<table class="corr-tbl" id="mcorr"></table></div>
</div></div>

<!-- 4. Annual -->
<div class="card"><h2>4. Retornos Anuales (%)</h2>
<div class="chart-wrap" style="height:280px"><canvas id="barChart"></canvas></div>
<div class="legend">
<span style="color:#1e293b">&#9632; Original</span> &nbsp;
<span style="color:#16a34a">&#9632; Hybrid</span> &nbsp;
<span style="color:#dc2626">&#9632; MinDD</span> &nbsp;
<span style="color:#2563eb">&#9632; SPY</span> &nbsp;
<span style="color:#7c3aed">&#9632; QQQ</span> &nbsp;
<span style="color:#d97706">&#9632; Gold</span>
</div>
<div class="scroll-t" style="max-height:500px;margin-top:10px"><table>
<thead><tr><th>Ano</th><th>Orig %</th><th>Hyb %</th><th>MinDD %</th><th>SPY %</th><th>QQQ %</th><th>Gold %</th>
<th>Orig EUR</th><th>Hyb EUR</th><th>MinDD EUR</th><th>Sem</th></tr></thead>
<tbody id="yr-body"></tbody><tfoot id="yr-foot"></tfoot></table></div></div>

<!-- 5. Stats por regimen -->
<div class="card"><h2>5. DUAL PnL por Regimen - Los 3 Modos</h2>
<div class="side3" id="reg-tables"></div></div>

<!-- 6. E2 por regimen -->
<div class="card"><h2>6. E2 PnL por Regimen - Los 3 Modos</h2>
<div class="side3" id="e2-tables"></div></div>

<!-- 7. 3DH por regimen -->
<div class="card"><h2>7. 3DH PnL por Regimen - Los 3 Modos</h2>
<div class="side3" id="h3-tables"></div></div>

<script>
const STATS={json.dumps(all_stats, separators=(',',':'))};
const EQ={json.dumps(EQ, separators=(',',':'))};
const WCORR={json.dumps(W_CORR, separators=(',',':'))};
const MCORR={json.dumps(M_CORR, separators=(',',':'))};
const CL={json.dumps(CORR_LABELS)};
const A={json.dumps(A_JSON, separators=(',',':'))};
const RS={RS_JSON};
const RO={json.dumps(REGIME_ORDER)};
const RC={json.dumps(REGIME_COLORS, separators=(',',':'))};
const ML={json.dumps(MODE_LABELS)};
const COLORS=['#1e293b','#16a34a','#dc2626','#2563eb','#7c3aed','#d97706'];
const W_DATES={json.dumps(df['date'].tolist(), separators=(',',':'))};

const fmt=v=>{{let s=v<0?'-':'';return s+'EUR '+Math.abs(Math.round(v)).toLocaleString('en-US')}};
const cls=v=>v>=0?'pos':'neg';
const fc=(v,d)=>{{d=d||2;return(v>=0?'+':'')+v.toFixed(d)+'%'}};

// Stats table
(function(){{
  let h='';
  STATS.forEach(s=>{{
    h+=`<tr><td><b>${{s.label}}</b></td>
    <td class="${{cls(s.total)}}">${{fc(s.total,1)}}</td>
    <td class="${{cls(s.cagr)}}">${{fc(s.cagr,1)}}</td>
    <td class="${{s.sharpe>=0.5?'pos':'neg'}}">${{s.sharpe.toFixed(2)}}</td>
    <td class="${{s.sortino>=0.8?'pos':'neg'}}">${{s.sortino.toFixed(2)}}</td>
    <td class="neg">${{fc(s.max_dd,1)}}</td>
    <td class="${{s.wr>=50?'pos':'neg'}}">${{s.wr.toFixed(1)}}%</td>
    <td class="${{cls(s.mean)}}">${{fc(s.mean,3)}}</td>
    <td>${{s.std.toFixed(3)}}%</td></tr>`}});
  document.getElementById('st-body').innerHTML=h
}})();

// Equity chart
function drawEQ(){{
  const cv=document.getElementById('eqChart'),cx=cv.getContext('2d');
  const W=cv.parentElement.clientWidth,H=cv.parentElement.clientHeight;
  cv.width=W*2;cv.height=H*2;cx.scale(2,2);
  let all=[];EQ.forEach(e=>all.push(...e));all.push(0);
  const mn=Math.min(...all),mx=Math.max(...all);
  const pad={{t:16,b:26,l:60,r:16}},w=W-pad.l-pad.r,h=H-pad.t-pad.b;
  const range=mx-mn||1,n=EQ[0].length,xS=w/n,yS=h/range;
  cx.clearRect(0,0,W,H);
  cx.strokeStyle='#e2e8f0';cx.lineWidth=0.5;
  for(let i=0;i<=5;i++){{let yv=mn+(range/5)*i,yy=pad.t+h-(yv-mn)*yS;
    cx.beginPath();cx.moveTo(pad.l,yy);cx.lineTo(W-pad.r,yy);cx.stroke();
    cx.fillStyle='#64748b';cx.font='10px sans-serif';cx.textAlign='right';
    cx.fillText(yv.toFixed(0)+'%',pad.l-4,yy+3)}}
  let y0=pad.t+h-(0-mn)*yS;cx.strokeStyle='#94a3b8';cx.lineWidth=1;
  cx.beginPath();cx.moveTo(pad.l,y0);cx.lineTo(W-pad.r,y0);cx.stroke();
  const widths=[2.5,2.5,2,1.5,1.5,1.5];
  EQ.forEach((eq,ei)=>{{
    cx.strokeStyle=COLORS[ei];cx.lineWidth=widths[ei];
    if(ei>=3)cx.setLineDash([4,3]);else cx.setLineDash([]);
    cx.beginPath();eq.forEach((v,i)=>{{let x=pad.l+i*xS,y=pad.t+h-(v-mn)*yS;i===0?cx.moveTo(x,y):cx.lineTo(x,y)}});
    cx.stroke();cx.setLineDash([])}});
  cx.fillStyle='#94a3b8';cx.font='9px sans-serif';cx.textAlign='center';
  let lastY='';W_DATES.forEach((d,i)=>{{const yr=d.substring(0,4);if(yr!==lastY){{lastY=yr;cx.fillText(yr,pad.l+i*xS,H-pad.b+12)}}}})
}}
drawEQ();window.addEventListener('resize',()=>{{drawEQ();drawBar()}});

// Bar chart
function drawBar(){{
  const cv=document.getElementById('barChart'),cx=cv.getContext('2d');
  const W=cv.parentElement.clientWidth,H=cv.parentElement.clientHeight;
  cv.width=W*2;cv.height=H*2;cx.scale(2,2);
  const n=A.length;
  const pad={{t:14,b:22,l:50,r:14}},w=W-pad.l-pad.r,h=H-pad.t-pad.b;
  let all=[];A.forEach(a=>{{for(let j=1;j<=6;j++)all.push(a[j])}});all.push(0);
  const mn=Math.min(...all),mx=Math.max(...all),range=mx-mn||1,yS=h/range;
  cx.clearRect(0,0,W,H);
  let y0=pad.t+h-(0-mn)*yS;cx.strokeStyle='#cbd5e1';cx.lineWidth=1;
  cx.beginPath();cx.moveTo(pad.l,y0);cx.lineTo(W-pad.r,y0);cx.stroke();
  const grpW=w/n,bW=Math.min(grpW/7,8);
  A.forEach((a,i)=>{{
    const cx2=pad.l+i*grpW+grpW/2;
    for(let j=0;j<6;j++){{
      const v=a[j+1],bx=cx2+(j-2.5)*bW*1.1;
      const by=pad.t+h-(v-mn)*yS,bh=Math.abs(v)*yS;
      cx.fillStyle=COLORS[j];
      v>=0?cx.fillRect(bx,by,bW,bh):cx.fillRect(bx,y0,bW,bh)
    }};
    cx.fillStyle='#94a3b8';cx.font='8px sans-serif';cx.textAlign='center';
    cx.fillText(String(a[0]).slice(2),cx2,H-pad.b+12)
  }})
}}
drawBar();

// Correlation matrix
function drawCorr(id,data){{
  const tbl=document.getElementById(id);
  let h='<thead><tr><th></th>';
  CL.forEach(l=>h+=`<th style="font-size:0.7em">${{l}}</th>`);
  h+='</tr></thead><tbody>';
  data.forEach((row,i)=>{{
    h+=`<tr><th style="text-align:left;font-size:0.7em">${{CL[i]}}</th>`;
    row.forEach((v,j)=>{{
      let bg='transparent';
      if(i!==j){{
        if(v>0.5)bg='rgba(34,197,94,'+Math.abs(v)*0.4+')';
        else if(v>0.2)bg='rgba(34,197,94,'+Math.abs(v)*0.25+')';
        else if(v<-0.2)bg='rgba(239,68,68,'+Math.abs(v)*0.35+')';
        else if(v<-0.05)bg='rgba(239,68,68,'+Math.abs(v)*0.2+')';
        else bg='rgba(148,163,184,0.1)';
      }}else bg='rgba(59,130,246,0.08)';
      h+=`<td style="background:${{bg}}">${{i===j?'1.00':v.toFixed(4)}}</td>`}});
    h+='</tr>'}});
  h+='</tbody>';tbl.innerHTML=h}}
drawCorr('wcorr',WCORR);drawCorr('mcorr',MCORR);

// Annual table
(function(){{
  let h='';
  let tots=[0,0,0,0,0,0];
  A.forEach(a=>{{
    const vals=[a[1],a[2],a[3],a[4],a[5],a[6]];
    const best=Math.max(...vals);
    for(let j=0;j<6;j++)tots[j]+=vals[j];
    h+=`<tr><td><b>${{a[0]}}</b></td>`;
    vals.forEach((v,j)=>{{h+=`<td class="${{cls(v)}} ${{v===best?'best':''}}">${{fc(v,1)}}</td>`}});
    h+=`<td class="${{cls(a[7])}}">${{fmt(a[7])}}</td><td class="${{cls(a[8])}}">${{fmt(a[8])}}</td>
    <td class="${{cls(a[9])}}">${{fmt(a[9])}}</td><td>${{a[10]}}</td></tr>`}});
  document.getElementById('yr-body').innerHTML=h;
  const ny=A.length;
  let f='<tr style="font-weight:700;border-top:2px solid #475569"><td>TOTAL</td>';
  tots.forEach(t=>{{f+=`<td class="${{cls(t)}}">${{fc(t,1)}}</td>`}});
  f+='<td></td><td></td><td></td><td></td></tr>';
  f+='<tr style="font-weight:700"><td>ANUAL</td>';
  tots.forEach(t=>{{f+=`<td class="${{cls(t/ny)}}">${{fc(t/ny,1)}}</td>`}});
  f+='<td></td><td></td><td></td><td></td></tr>';
  document.getElementById('yr-foot').innerHTML=f
}})();

// Regime stats tables
function buildRegTables(containerId, subKey, title){{
  const ctn=document.getElementById(containerId);
  let h='';
  for(let mi=0;mi<3;mi++){{
    h+=`<div><h4 style="font-size:0.85em;margin-bottom:6px;color:${{COLORS[mi]}}">${{ML[mi]}}</h4>`;
    h+='<table><thead><tr><th>Regimen</th><th>N</th><th>PnL</th><th>WR</th><th>PF</th><th>PnL/sem</th></tr></thead><tbody>';
    let totN=0,totP=0;
    RO.forEach(r=>{{
      const s=RS[mi][r][subKey];
      if(!s||!s.n)return;
      totN+=s.n;totP+=s.pnl;
      const rc=RC[r]||'#666';
      h+=`<tr><td><span class="rb" style="background:${{rc}}">${{r}}</span></td>
      <td>${{s.n}}</td><td class="${{cls(s.pnl)}}">${{fmt(s.pnl)}}</td>
      <td class="${{s.wr>50?'pos':'neg'}}">${{s.wr.toFixed(1)}}%</td>
      <td>${{s.pf>=99?'inf':s.pf.toFixed(2)}}</td>
      <td class="${{cls(s.mean)}}">${{fmt(s.mean)}}</td></tr>`}});
    h+=`<tr style="font-weight:700;border-top:2px solid #475569"><td>TOTAL</td><td>${{totN}}</td>
    <td class="${{cls(totP)}}">${{fmt(totP)}}</td><td></td><td></td><td class="${{cls(totP/totN)}}">${{fmt(Math.round(totP/totN))}}</td></tr>`;
    h+='</tbody></table></div>'
  }};
  ctn.innerHTML=h
}}
buildRegTables('reg-tables','dual');
buildRegTables('e2-tables','e2');
buildRegTables('h3-tables','h3');
</script></body></html>"""

out_path = BASE / 'correlacion_3modos.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML: {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
