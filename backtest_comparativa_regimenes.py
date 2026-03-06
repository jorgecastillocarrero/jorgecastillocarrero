"""
Backtest Comparativa Regimenes: Original vs Min DD
Capital: EUR 600,000 = 200K 3DH + 400K E2
RECOVERY opera en ambas patas (3DH + E2)
"""
import re, json, sys, io, csv, bisect
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import groupby

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
BASE = Path(__file__).parent

COST_3DH = 25000
COST_E2 = 20000
CAP_3DH = 200000
CAP_E2 = 400000
CAP_TOTAL = CAP_3DH + CAP_E2
SLIP = 0.003
START_YEAR = 2005
MAX_3DH = 30
NM = 2  # number of modes

MODE_LABELS = ['Original', 'Min DD']
MODE_CSVS = ['data/regimenes_historico.csv', 'data/regimenes_mindd.csv']
E2_SKIP = [('CRISIS', 'PANICO'), ()]

# Regimenes seleccionados - RECOVERY en ambas patas
H3_ACTIVE = {'CAUTIOUS', 'RECOVERY', 'CRISIS', 'PANICO'}
E2_ACTIVE = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'BEARISH', 'CAPITULACION', 'RECOVERY'}

REGIME_COLORS = {
    'BURBUJA': '#e91e63', 'GOLDILOCKS': '#4caf50', 'ALCISTA': '#2196f3',
    'NEUTRAL': '#ff9800', 'CAUTIOUS': '#ff5722', 'BEARISH': '#795548',
    'CRISIS': '#9c27b0', 'PANICO': '#f44336', 'CAPITULACION': '#00bcd4',
    'RECOVERY': '#8bc34a',
}
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH',
                'RECOVERY','CRISIS','PANICO','CAPITULACION']

STRAT_E2_ALL = {
    'BURBUJA':      [(0, 10, 'long'),  (-20, -10, 'short')],
    'GOLDILOCKS':   [(0, 10, 'long'),  (-20, -10, 'short')],
    'ALCISTA':      [(0, 10, 'long'),  (-10, None, 'short')],
    'NEUTRAL':      [(10, 20, 'long'), (-20, -10, 'short')],
    'CAUTIOUS':     [(-10, None, 'short'), (-20, -10, 'short')],
    'BEARISH':      [(-10, None, 'short'), (-20, -10, 'short')],
    'RECOVERY':     [(0, 10, 'long'),  (-20, -10, 'long')],
    'CAPITULACION': [(10, 20, 'long'), (20, 30, 'long')],
    'CRISIS':       [(-10, None, 'short'), (-20, -10, 'short')],
    'PANICO':       [(10, 20, 'long'), (20, 30, 'long')],
}

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD 3DH + CAP30
# ═══════════════════════════════════════════════════════════════════════
print("Loading 3DH OPT from cache (cap 25/day by contrast)...")
with open(BASE / 'data' / '3dh_opt_4d_trades.json', 'r') as f:
    h3_raw = json.load(f)
h3_all = []
for t in h3_raw:
    contrast = t.get('d5', 0) - t.get('d50', 0)
    h3_all.append({'sym': t['sym'], 'sig': t['sig'], 'entry': t['entry'], 'exit': t['exit'],
        'ret': t['ret'], 'pnl': t['pnl'], 'month': t['sig'][:7], 'year': int(t['sig'][:4]),
        'contrast': contrast})
h3_all.sort(key=lambda t: t['entry'])

# ── Apply MAX_3DH concurrent limit (rank by contrast) ──
print(f"  Applying max {MAX_3DH} concurrent limit...")
h3_trades = []
open_by_exit = defaultdict(int)
n_capped_days = 0
n_capped_signals = 0
for entry_date, group in groupby(h3_all, key=lambda t: t['entry']):
    day_trades = list(group)
    expired = [ex for ex in open_by_exit if ex <= entry_date]
    for ex in expired:
        del open_by_exit[ex]
    currently_open = sum(open_by_exit.values())
    available = max(0, MAX_3DH - currently_open)
    n_signals = len(day_trades)
    if available <= 0:
        n_capped_days += 1
        n_capped_signals += n_signals
        continue
    if n_signals <= available:
        for t in day_trades:
            t['capped'] = False
            t['n_sigs'] = n_signals
            h3_trades.append(t)
            open_by_exit[t['exit']] += 1
    else:
        n_capped_days += 1
        n_capped_signals += n_signals - available
        day_trades.sort(key=lambda t: -t['contrast'])
        for i in range(available):
            tpl = day_trades[i]
            h3_trades.append({**tpl, 'capped': True, 'n_sigs': n_signals})
            open_by_exit[tpl['exit']] += 1

print(f"  Original: {len(h3_all)} trades | Cap {MAX_3DH}: {len(h3_trades)} trades")
print(f"  Dias cap: {n_capped_days}, signals descartadas/avg: {n_capped_signals}")

# 3DH monthly (mode-independent)
h3_monthly_base = defaultdict(lambda: {'pnl': 0, 'n': 0})
for t in h3_trades:
    h3_monthly_base[t['month']]['pnl'] += t['pnl']
    h3_monthly_base[t['month']]['n'] += 1

# 3DH concurrent peaks (mode-independent)
events = []
for t in h3_trades:
    events.append((t['entry'], 1))
    events.append((t['exit'], -1))
events.sort(key=lambda x: (x[0], x[1]))
concurrent = 0
h3_peak_month = defaultdict(int)
for ds, delta in events:
    concurrent += delta
    mo = ds[:7]
    h3_peak_month[mo] = max(h3_peak_month[mo], concurrent)

# ═══════════════════════════════════════════════════════════════════════
# 2. LOAD E2 STOCK DATA (raw, no regime assignment yet)
# ═══════════════════════════════════════════════════════════════════════
print("\nLoading E2 semanal from acciones_navegable.html...")
with open(BASE / 'acciones_navegable.html', 'r', encoding='utf-8') as f:
    html_text = f.read()
m2 = re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
WEEKS = json.loads(m2.group(1))
del html_text
print(f"  E2: {len(WEEKS)} weeks loaded")

# ═══════════════════════════════════════════════════════════════════════
# 3. REGIME LOOKUP BUILDER
# ═══════════════════════════════════════════════════════════════════════
def build_regime_lookup(csv_path, mode_idx):
    entries = []
    with open(BASE / csv_path) as f:
        for row in csv.DictReader(f):
            d = str(row['fecha_senal'])[:10]
            r = row['regime']
            s = True  # Todos los regimenes operan al 100%, sin filtro shortable
            entries.append((d, r, s))
    entries.sort()
    dates = [e[0] for e in entries]
    regimes = [e[1] for e in entries]
    shortable = [e[2] for e in entries]
    def get(sig_date):
        idx = bisect.bisect_right(dates, str(sig_date)[:10]) - 1
        return (regimes[idx], shortable[idx]) if idx >= 0 else ('UNKNOWN', True)
    return get, len(entries)

# ═══════════════════════════════════════════════════════════════════════
# 4. PROCESS EACH MODE
# ═══════════════════════════════════════════════════════════════════════
all_modes = [{} for _ in range(NM)]

for mi in range(NM):
    label = MODE_LABELS[mi]
    skip_set = E2_SKIP[mi]
    print(f"\n{'='*70}")
    print(f"  MODE {mi}: {label} ({MODE_CSVS[mi]})")
    print(f"{'='*70}")

    # (a) Build regime lookup
    get_regime, n_weeks = build_regime_lookup(MODE_CSVS[mi], mi)
    print(f"  Loaded {n_weeks} weeks from CSV")

    # (b) Assign regime to 3DH trades (copy)
    h3_mode = []
    for t in h3_trades:
        regime, _ = get_regime(t['sig'])
        h3_mode.append({**t, 'regime': regime})

    # (c) 3DH regime stats (RS)
    h3_by_regime = defaultdict(list)
    for t in h3_mode:
        h3_by_regime[t['regime']].append(t)
    rs = {}
    for reg in REGIME_ORDER:
        trades = h3_by_regime.get(reg, [])
        if not trades:
            rs[reg] = {'n': 0, 'wr': 0, 'avg': 0, 'med': 0, 'pf': 0, 'pnl': 0, 'ppt': 0,
                       'wins': 0, 'gw': 0, 'gl': 0}
            continue
        rets = [t['ret'] for t in trades]
        wins = sum(1 for r in rets if r > 0)
        gw = sum(r for r in rets if r > 0)
        gl = abs(sum(r for r in rets if r < 0))
        pf = gw / gl if gl > 0 else 99.9
        tp = sum(t['pnl'] for t in trades)
        gw_eur = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gl_eur = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        rs[reg] = {'n': len(trades), 'wr': round(wins/len(trades)*100, 1),
            'avg': round(float(np.mean(rets)), 2), 'med': round(float(np.median(rets)), 2),
            'pf': round(min(pf, 99.9), 2), 'pnl': round(tp), 'ppt': round(tp/len(trades)),
            'wins': wins, 'gw': round(gw_eur), 'gl': round(gl_eur)}

    # (d) H3MR: {month: {regime: [pnl, n_trades]}}
    h3_mr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for t in h3_mode:
        h3_mr[t['month']][t['regime']][0] += t['pnl']
        h3_mr[t['month']][t['regime']][1] += 1
    h3mr = {m: {r: [round(v[0]), v[1]] for r, v in regs.items()} for m, regs in h3_mr.items()}

    # (e) H3C: [[entry, exit, regime], ...]
    h3c = [[t['entry'], t['exit'], t['regime']] for t in h3_mode]

    # (f) HT: {year: [[sym, sig, entry, exit, ret, pnl, cap_flag, n_sigs, regime], ...]}
    ht = defaultdict(list)
    for t in h3_mode:
        cap_flag = 1 if t.get('capped') else 0
        n_sigs = t.get('n_sigs', 0)
        ht[t['year']].append([t['sym'], t['sig'], t['entry'], t['exit'],
                              t['ret'], t['pnl'], cap_flag, n_sigs, t['regime']])

    # (g) E2 weekly PnL
    e2_weekly = []
    e2_skipped = 0
    for w in WEEKS:
        date, year = w['d'], w['y']
        if year < START_YEAR:
            continue
        stocks = w['s']
        regime, shortable = get_regime(date)
        skipped = regime in skip_set
        if skipped:
            e2_skipped += 1

        strat = STRAT_E2_ALL.get(regime, [])
        pnl = 0
        n_pos = 0
        if strat:
            for start, end, direction in strat:
                if direction == 'short' and not shortable:
                    continue
                selected = stocks[start:end] if end is not None else stocks[start:]
                for s in selected:
                    rv = s[8]
                    if rv is None:
                        continue
                    rv = max(-50, min(50, rv))
                    if direction == 'long':
                        pnl += COST_E2 * (rv / 100 - SLIP)
                    else:
                        pnl += COST_E2 * (-rv / 100 - SLIP)
                    n_pos += 1
        e2_weekly.append({'date': date, 'year': year, 'month': date[:7],
            'regime': regime, 'pnl': round(pnl, 2), 'n_pos': n_pos, 'skipped': skipped})

    # (h) E2 regime stats (ES2)
    e2_by_regime = defaultdict(list)
    for w in e2_weekly:
        e2_by_regime[w['regime']].append(w)
    es2 = {}
    for reg in REGIME_ORDER:
        weeks = e2_by_regime.get(reg, [])
        if not weeks:
            es2[reg] = {'n': 0, 'pnl': 0, 'wins': 0, 'gw': 0, 'gl': 0,
                        'wr': 0, 'pf': 0, 'avg': 0, 'active': reg not in skip_set}
            continue
        pnls = [w['pnl'] for w in weeks]
        wins = sum(1 for p in pnls if p > 0)
        gw_eur = sum(p for p in pnls if p > 0)
        gl_eur = abs(sum(p for p in pnls if p < 0))
        pf = gw_eur / gl_eur if gl_eur > 0 else 99.9
        es2[reg] = {'n': len(weeks), 'pnl': round(sum(pnls)),
            'wins': wins, 'gw': round(gw_eur), 'gl': round(gl_eur),
            'wr': round(wins/len(weeks)*100, 1), 'pf': round(min(pf, 99.9), 2),
            'avg': round(float(np.mean(pnls))),
            'active': reg not in skip_set}

    # (i) E2C: [[month, regime, n_pos, pnl], ...]
    e2c = [[w['month'], w['regime'], w['n_pos'], round(w['pnl'], 2)] for w in e2_weekly]

    # (j) ET: {year: [[date, regime, n_pos, pnl, skipped], ...]}
    et = defaultdict(list)
    for w in e2_weekly:
        et[w['year']].append([w['date'], w['regime'], w['n_pos'],
                              round(w['pnl']), 1 if w['skipped'] else 0])

    # E2 monthly (for combined)
    e2_monthly = defaultdict(lambda: {'pnl': 0, 'n_weeks': 0, 'n_skip': 0})
    for w in e2_weekly:
        e2_monthly[w['month']]['pnl'] += (w['pnl'] if not w['skipped'] else 0)
        e2_monthly[w['month']]['n_weeks'] += 1
        if w['skipped']:
            e2_monthly[w['month']]['n_skip'] += 1

    e2_cap_month = defaultdict(int)
    for w in e2_weekly:
        if not w['skipped']:
            cap = w['n_pos'] * COST_E2
            e2_cap_month[w['month']] = max(e2_cap_month[w['month']], cap)

    # (k) Summary stats - filtered by H3_ACTIVE / E2_ACTIVE regimes
    h3_monthly_filt = defaultdict(lambda: {'pnl': 0, 'n': 0})
    for t in h3_mode:
        if t['regime'] in H3_ACTIVE:
            h3_monthly_filt[t['month']]['pnl'] += t['pnl']
            h3_monthly_filt[t['month']]['n'] += 1

    e2_monthly_filt = defaultdict(lambda: {'pnl': 0})
    for w in e2_weekly:
        if w['regime'] in E2_ACTIVE and not w['skipped']:
            e2_monthly_filt[w['month']]['pnl'] += w['pnl']

    all_months_mode = sorted(mk for mk in set(
        list(h3_monthly_filt.keys()) + list(e2_monthly_filt.keys()) +
        list(h3_monthly_base.keys()) + list(e2_monthly.keys())
    ) if int(mk[:4]) >= START_YEAR)

    monthly_pnl = []
    for mk in all_months_mode:
        h3_f = h3_monthly_filt.get(mk, {'pnl': 0})
        e2_f = e2_monthly_filt.get(mk, {'pnl': 0})
        monthly_pnl.append(h3_f['pnl'] + e2_f['pnl'])

    eq = np.cumsum(monthly_pnl)
    pk = np.maximum.accumulate(eq)
    dd = eq - pk
    max_dd = round(float(dd.min())) if len(dd) > 0 else 0

    total_pnl = round(sum(monthly_pnl))
    years_set = sorted(set(int(mk[:4]) for mk in all_months_mode))
    year_pnl = {}
    for y in years_set:
        yp = sum(p for mk, p in zip(all_months_mode, monthly_pnl) if int(mk[:4]) == y)
        year_pnl[y] = round(yp)
    wins = sum(1 for y in years_set if year_pnl[y] > 0)
    losses = len(years_set) - wins

    g_e2_pnl_filt = round(sum(v['pnl'] for v in e2_monthly_filt.values()))
    g_h3_pnl_filt = round(sum(v['pnl'] for v in h3_monthly_filt.values()))
    g_e2_pnl_all = round(sum(v['pnl'] for v in e2_monthly.values()))
    g_h3_pnl_all = round(sum(h3_monthly_base[mk]['pnl'] for mk in h3_monthly_base))

    summary = {
        'total': total_pnl, 'max_dd': max_dd,
        'wins': wins, 'losses': losses,
        'h3': g_h3_pnl_filt, 'e2': g_e2_pnl_filt,
        'h3_all': g_h3_pnl_all, 'e2_all': g_e2_pnl_all,
        'e2_skipped': e2_skipped,
    }

    # (l) Weekly stats for DUAL (filtered)
    # Aggregate 3DH by E2 week dates
    e2_dates = sorted(set(w['date'] for w in e2_weekly))
    h3_by_week = defaultdict(float)
    for t in h3_mode:
        if t['regime'] in H3_ACTIVE:
            sig = t['sig']
            idx = bisect.bisect_right(e2_dates, sig) - 1
            if idx >= 0:
                h3_by_week[e2_dates[idx]] += t['pnl']

    e2_by_week = {}
    for w in e2_weekly:
        if w['regime'] in E2_ACTIVE and not w['skipped']:
            e2_by_week[w['date']] = e2_by_week.get(w['date'], 0) + w['pnl']

    weekly_pnl_list = []
    for d in e2_dates:
        if int(d[:4]) < START_YEAR:
            continue
        weekly_pnl_list.append(h3_by_week.get(d, 0) + e2_by_week.get(d, 0))

    wa = np.array(weekly_pnl_list)
    w_mean = round(float(np.mean(wa)))
    w_med = round(float(np.median(wa)))
    w_std = round(float(np.std(wa)))
    w_wins = int(np.sum(wa > 0))
    w_losses = int(np.sum(wa < 0))
    w_wr = round(w_wins / len(wa) * 100, 1) if len(wa) > 0 else 0
    w_sharpe = round(float(np.mean(wa) / np.std(wa) * np.sqrt(52)), 2) if np.std(wa) > 0 else 0
    w_down = wa[wa < 0]
    w_sortino = round(float(np.mean(wa) / np.std(w_down) * np.sqrt(52)), 2) if len(w_down) > 0 and np.std(w_down) > 0 else 0
    w_best = round(float(np.max(wa)))
    w_worst = round(float(np.min(wa)))

    wstats = {
        'n': len(wa), 'mean': w_mean, 'med': w_med, 'std': w_std,
        'wr': w_wr, 'sharpe': w_sharpe, 'sortino': w_sortino,
        'best': w_best, 'worst': w_worst,
        'wins': w_wins, 'losses': w_losses
    }
    summary['wstats'] = wstats

    print(f"  E2 PnL (filtrado): EUR {g_e2_pnl_filt:,} | 3DH PnL (filtrado): EUR {g_h3_pnl_filt:,}")
    print(f"  Dual filtrado: EUR {total_pnl:,} | MaxDD: EUR {max_dd:,} | {wins}W/{losses}L")
    print(f"  Weekly: Sharpe {w_sharpe}, Sortino {w_sortino}, WR {w_wr}%")

    # Store everything
    all_modes[mi] = {
        'rs': rs, 'es2': es2, 'h3mr': h3mr, 'h3c': h3c,
        'e2c': e2c, 'ht': dict(ht), 'et': dict(et),
        'e2_weekly': e2_weekly, 'e2_monthly': dict(e2_monthly),
        'e2_cap_month': dict(e2_cap_month),
        'summary': summary, 'monthly_pnl': monthly_pnl,
    }

# ═══════════════════════════════════════════════════════════════════════
# 5. BUILD C (combined monthly)
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding combined monthly array...")
all_months = sorted(mk for mk in set(
    list(h3_monthly_base.keys())
) if int(mk[:4]) >= START_YEAR)
for mi in range(NM):
    for mk in all_modes[mi]['e2_monthly']:
        if int(mk[:4]) >= START_YEAR and mk not in all_months:
            all_months.append(mk)
all_months = sorted(set(all_months))

C = []
for mk in all_months:
    year = int(mk[:4])
    C.append({'m': mk, 'y': year})
print(f"  Combined months: {len(C)}")

# ═══════════════════════════════════════════════════════════════════════
# 6. PRE-COMPUTE COMPARISON EQUITY/DD CURVES
# ═══════════════════════════════════════════════════════════════════════
print("\nComputing comparison equity/DD curves...")
N = len(C)
EQ_CMP = []
DD_CMP = []

for mi in range(NM):
    mode = all_modes[mi]
    h3mr_mode = mode['h3mr']
    e2_weekly_mode = mode['e2_weekly']

    h3_filt_m = defaultdict(float)
    for mk, regs in h3mr_mode.items():
        for reg, (pnl, n) in regs.items():
            if reg in H3_ACTIVE:
                h3_filt_m[mk] += pnl

    e2_filt_m = defaultdict(float)
    for w in e2_weekly_mode:
        if w['regime'] in E2_ACTIVE and not w['skipped']:
            e2_filt_m[w['month']] += w['pnl']

    monthly_total = []
    for c in C:
        mk = c['m']
        monthly_total.append(h3_filt_m.get(mk, 0) + e2_filt_m.get(mk, 0))

    eq_np = np.cumsum(monthly_total)
    eq = [int(round(float(v))) for v in eq_np]
    pk_arr = [int(round(float(v))) for v in np.maximum.accumulate(eq_np)]
    dd = [eq[i] - pk_arr[i] for i in range(len(eq))]
    EQ_CMP.append(eq)
    DD_CMP.append(dd)

# ═══════════════════════════════════════════════════════════════════════
# 7. JSON PREPARATION
# ═══════════════════════════════════════════════════════════════════════
print("\nPreparing JSON...")

comb_json = json.dumps([{'m': c['m'], 'y': c['y']} for c in C], separators=(',', ':'))

# Per-mode JSON
mode_json = []
for mi in range(NM):
    mode = all_modes[mi]
    mj = {
        'rs': json.dumps(mode['rs'], separators=(',', ':')),
        'es2': json.dumps(mode['es2'], separators=(',', ':')),
        'h3mr': json.dumps(mode['h3mr'], separators=(',', ':')),
        'h3c': json.dumps(mode['h3c'], separators=(',', ':')),
        'e2c': json.dumps(mode['e2c'], separators=(',', ':')),
        'ht': json.dumps(mode['ht'], separators=(',', ':')),
        'et': json.dumps(mode['et'], separators=(',', ':')),
        'summary': mode['summary'],
    }
    mode_json.append(mj)

eq_cmp_json = json.dumps(EQ_CMP, separators=(',', ':'))
dd_cmp_json = json.dumps(DD_CMP, separators=(',', ':'))

h3_monthly_json = json.dumps({mk: {'pnl': round(v['pnl']), 'n': v['n']}
                              for mk, v in h3_monthly_base.items()}, separators=(',', ':'))
h3_peak_json = json.dumps(dict(h3_peak_month), separators=(',', ':'))

# ═══════════════════════════════════════════════════════════════════════
# 8. HTML GENERATION
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating HTML...")

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Comparativa Regimenes: 3DH + E2 | Original vs Min DD</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#fff;color:#1e293b;padding:14px}}
h1{{text-align:center;font-size:1.5em;margin-bottom:4px;color:#0f172a}}
.sub{{text-align:center;color:#64748b;margin-bottom:14px;font-size:0.82em;line-height:1.5}}
.card{{background:#f8fafc;border-radius:10px;padding:16px;margin-bottom:12px;border:1px solid #e2e8f0}}
.card h2{{font-size:1.05em;color:#0f172a;margin-bottom:10px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:0.78em}}
th{{background:#e2e8f0;color:#1e293b;padding:6px 4px;text-align:left;position:sticky;top:0;font-size:0.75em;white-space:nowrap}}
td{{padding:5px 4px;border-bottom:1px solid #e2e8f0}}
tr:hover td{{background:#f1f5f9}}
.pos{{color:#16a34a}}.neg{{color:#dc2626}}
.h3c{{color:#ea580c}}.totc{{color:#b45309}}.e2c{{color:#2563eb}}
.gv-row{{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:8px}}
.gv{{background:#fff;border:1px solid #e2e8f0;border-radius:7px;padding:10px 14px;text-align:center;min-width:110px}}
.gv .val{{font-size:1.15em;font-weight:700}}.gv .lbl{{font-size:0.72em;color:#64748b;margin-top:2px}}
.side3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}}
.side2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:1000px){{.side3{{grid-template-columns:1fr}}.side2{{grid-template-columns:1fr}}}}
.chart-wrap{{position:relative;height:280px;background:#fff;border:1px solid #e2e8f0;border-radius:7px;overflow:hidden;margin-bottom:4px}}
.chart-wrap.half{{height:210px}}
.chart-wrap canvas{{width:100%!important;height:100%!important}}
.legend{{text-align:center;font-size:0.72em;color:#64748b;margin-top:2px}}
.win-row td{{background:rgba(34,197,94,0.08)}}.loss-row td{{background:rgba(239,68,68,0.08)}}
.tab-bar{{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap}}
.tab-btn{{padding:6px 16px;border-radius:6px;border:1px solid #cbd5e1;background:transparent;color:#64748b;cursor:pointer;font-size:0.8em}}
.tab-btn:hover{{background:#f1f5f9}}.tab-btn.act{{background:#3b82f6;color:#fff;border-color:#3b82f6}}
.scroll-t{{max-height:600px;overflow-y:auto}}.scroll-t th{{position:sticky;top:0;z-index:1}}
.rb{{display:inline-block;padding:2px 8px;border-radius:3px;font-size:0.7em;font-weight:bold;color:#fff}}
.skip-row td{{opacity:0.45}}
.cap-warn{{background:rgba(239,68,68,0.1)!important;font-weight:700}}
.cap-bar{{display:flex;gap:20px;justify-content:center;margin-top:8px;font-size:0.8em;flex-wrap:wrap}}
.mode-bar{{display:flex;gap:6px;justify-content:center;margin-bottom:16px}}
.mode-btn{{padding:10px 24px;border-radius:8px;border:2px solid #cbd5e1;background:transparent;color:#64748b;cursor:pointer;font-size:0.9em;font-weight:700;transition:all 0.15s}}
.mode-btn:hover{{background:#f1f5f9}}
.mode-btn.act{{background:#3b82f6;color:#fff;border-color:#3b82f6}}
.cmp-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}
@media(max-width:800px){{.cmp-grid{{grid-template-columns:1fr}}}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:10px 0}}
.stat-box{{background:#fff;border:1px solid #e2e8f0;border-radius:7px;padding:10px;text-align:center}}
.stat-box .sv{{font-size:1.1em;font-weight:700}}.stat-box .sl{{font-size:0.7em;color:#64748b;margin-top:2px}}
</style></head>
<body>
<h1>Comparativa Regimenes: 3DH + E2 | EUR 600K</h1>
<div class="sub">
<b class="h3c">3DH Short 4d</b>: CAUTIOUS, RECOVERY, CRISIS, PANICO | Max {MAX_3DH} concurrent | EUR {CAP_3DH//1000}K<br>
<b class="e2c">E2 Semanal</b>: BURBUJA, GOLDILOCKS, ALCISTA, NEUTRAL, BEARISH, CAPITULACION, RECOVERY | EUR {CAP_E2//1000}K<br>
Slip 0.3% | {START_YEAR}-2026 | Original vs Min DD | RECOVERY opera en ambas patas
</div>

<!-- Section 0: Comparativa -->
<div class="card"><h2>0. Comparativa de Modos</h2>
<div class="mode-bar" id="mode-bar"></div>
<table style="margin-bottom:12px">
<thead><tr><th>Modo</th><th>PnL Total</th><th>Ret %</th><th>Max DD</th><th>DD %</th><th>W/L</th><th>3DH</th><th>E2</th><th>E2 Skip</th></tr></thead>
<tbody id="cmp-body"></tbody></table>
<div class="cmp-grid">
<div><h3 style="text-align:center;font-size:0.9em;margin-bottom:6px">Equity Curves</h3>
<div class="chart-wrap half"><canvas id="cmpEqChart"></canvas></div>
<div class="legend"><span style="color:#1e293b">&#9644; Original</span> &nbsp;<span style="color:#dc2626">&#9644; Min DD</span></div></div>
<div><h3 style="text-align:center;font-size:0.9em;margin-bottom:6px">Drawdown Curves</h3>
<div class="chart-wrap half"><canvas id="cmpDDChart"></canvas></div>
<div class="legend"><span style="color:#1e293b">&#9644; Original</span> &nbsp;<span style="color:#dc2626">&#9644; Min DD</span></div></div>
</div></div>

<!-- Section 1: Summary -->
<div class="card"><h2>1. Resumen Global</h2>
<div class="side3">
<div style="border-left:3px solid #f97316;padding-left:10px">
<h3 class="h3c" style="font-size:0.9em;margin-bottom:8px">3DH Short (cap30)</h3>
<div class="gv-row" id="gs_h3"></div></div>
<div style="border-left:3px solid #60a5fa;padding-left:10px">
<h3 class="e2c" style="font-size:0.9em;margin-bottom:8px">E2 Semanal</h3>
<div class="gv-row" id="gs_e2"></div></div>
<div style="border-left:3px solid #fbbf24;padding-left:10px">
<h3 class="totc" style="font-size:0.9em;margin-bottom:8px">DUAL</h3>
<div class="gv-row" id="gs_tot"></div></div>
</div>
<div class="cap-bar" id="cap-bar"></div>
</div>

<!-- 2+3. Equity & Drawdown -->
<div class="side2">
<div class="card"><h2>2. PnL Acumulado</h2>
<div class="chart-wrap half"><canvas id="eqChart"></canvas></div>
<div class="legend"><span class="h3c">&#9644; 3DH</span> &nbsp;<span class="e2c">&#9644; E2</span> &nbsp;<span class="totc">&#9644; DUAL</span></div></div>
<div class="card"><h2>3. Drawdown</h2>
<div class="chart-wrap half"><canvas id="ddChart"></canvas></div>
<div class="legend"><span class="h3c">&#9644; 3DH</span> &nbsp;<span class="e2c">&#9644; E2</span> &nbsp;<span class="totc">&#9644; DUAL</span></div></div>
</div>

<!-- 4+5. Capital & Bar -->
<div class="side2">
<div class="card"><h2>4. Capital Empleado - Linea roja = EUR {CAP_TOTAL//1000}K</h2>
<div class="chart-wrap half"><canvas id="capChart"></canvas></div>
<div class="legend"><span class="h3c">&#9644; 3DH</span> &nbsp;<span class="e2c">&#9644; E2</span> &nbsp;<span class="totc">&#9644; TOT</span> &nbsp;<span style="color:#dc2626">-- {CAP_TOTAL//1000}K</span></div></div>
<div class="card"><h2>5. PnL Anual</h2>
<div class="chart-wrap half"><canvas id="barChart"></canvas></div>
<div class="legend"><span class="h3c">&#9632; 3DH</span> &nbsp;<span class="e2c">&#9632; E2</span> &nbsp;<span class="totc">&#9679; DUAL</span></div></div>
</div>

<!-- 6. Yearly -->
<div class="card"><h2>6. Retornos Anuales</h2>
<div class="scroll-t"><table><thead><tr>
<th>Ano</th>
<th>3DH</th><th>3DH%</th><th>3DH#</th>
<th>E2</th><th>E2%</th><th>E2sk</th>
<th>Dual</th><th>Ret%</th><th>WR</th><th>CapPk</th><th>3DHpk</th><th>Res</th>
</tr></thead><tbody id="yr-body"></tbody><tfoot id="yr-foot"></tfoot></table></div></div>

<!-- 7. Estadisticas Semanales -->
<div class="card"><h2>7. Estadisticas Semanales (DUAL)</h2>
<div class="stat-grid" id="wk-stats"></div>
</div>

<!-- 8. Selector Interactivo + Stats por Regimen -->
<div class="card"><h2>8. Selector Interactivo de Regimenes</h2>
<p style="font-size:0.8em;color:#94a3b8;margin-bottom:14px">
<b>Seleccion base:</b> 3DH &rarr; CAUTIOUS, RECOVERY, CRISIS, PANICO |
E2 &rarr; BURBUJA, GOLDILOCKS, ALCISTA, NEUTRAL, BEARISH, CAPITULACION, RECOVERY<br>
Activa/desactiva regimenes para ver el impacto en cada estrategia y el Dual total.</p>
<div class="side2">
<div style="border-left:3px solid #f97316;padding-left:12px">
<h3 class="h3c" style="font-size:0.95em;margin-bottom:8px">3DH Short (cap30)</h3>
<div id="h3tg" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px"></div>
<div id="h3sm" class="gv-row"></div></div>
<div style="border-left:3px solid #60a5fa;padding-left:12px">
<h3 class="e2c" style="font-size:0.95em;margin-bottom:8px">E2 Semanal</h3>
<div id="e2tg" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px"></div>
<div id="e2sm" class="gv-row"></div></div>
</div>
<div style="margin-top:14px;padding:12px;background:#fff;border:1px solid #e2e8f0;border-radius:8px;text-align:center">
<h3 class="totc" style="font-size:1em;margin-bottom:6px">DUAL = 3DH seleccionados + E2 seleccionados</h3>
<div id="trsm" class="gv-row" style="justify-content:center"></div></div>
<h4 class="h3c" style="font-size:0.95em;margin:18px 0 8px;border-bottom:2px solid #f97316;padding-bottom:4px">3DH Short por Regimen: Original vs Min DD</h4>
<div class="side2">
<div><h4 style="font-size:0.82em;margin-bottom:6px;color:#1e293b">Original</h4>
<table><thead><tr><th>Reg</th><th>N</th><th>WR</th><th>Avg%</th><th>PF</th><th>PnL</th><th>PnL/Tr</th></tr></thead><tbody id="h3rf0"></tbody></table></div>
<div><h4 style="font-size:0.82em;margin-bottom:6px;color:#dc2626">Min DD</h4>
<table><thead><tr><th>Reg</th><th>N</th><th>WR</th><th>Avg%</th><th>PF</th><th>PnL</th><th>PnL/Tr</th></tr></thead><tbody id="h3rf1"></tbody></table></div>
</div>
<h4 class="e2c" style="font-size:0.95em;margin:18px 0 8px;border-bottom:2px solid #60a5fa;padding-bottom:4px">E2 Semanal por Regimen: Original vs Min DD</h4>
<div class="side2">
<div><h4 style="font-size:0.82em;margin-bottom:6px;color:#1e293b">Original</h4>
<table><thead><tr><th>Reg</th><th>N sem</th><th>WR</th><th>PF</th><th>PnL</th><th>Base</th></tr></thead><tbody id="e2rf0"></tbody></table></div>
<div><h4 style="font-size:0.82em;margin-bottom:6px;color:#dc2626">Min DD</h4>
<table><thead><tr><th>Reg</th><th>N sem</th><th>WR</th><th>PF</th><th>PnL</th><th>Base</th></tr></thead><tbody id="e2rf1"></tbody></table></div>
</div></div>

<!-- 9. DD -->
<div class="card"><h2>9. Top 5 Drawdowns (Dual)</h2>
<table><thead><tr><th>#</th><th>Inicio</th><th>Fin</th><th>Meses</th><th>Max DD</th><th>Recuperacion</th></tr></thead>
<tbody id="dd-body"></tbody></table></div>

<!-- 10. Monthly -->
<div class="card"><h2>10. Detalle Mensual</h2>
<div class="scroll-t" style="max-height:500px"><table><thead><tr>
<th>Mes</th>
<th>3DH</th><th>3DH#</th><th>E2</th><th>Dual</th><th>Ac</th><th>DD</th>
<th style="border-left:2px solid #f97316">3DHpk</th><th>Cap3DH</th><th>CapE2</th><th>CapTot</th>
</tr></thead><tbody id="mn-body"></tbody></table></div></div>

<!-- 11. Trades -->
<div class="card"><h2>11. Trades por Ano</h2>
<div class="tab-bar" id="yr-tabs"></div><div id="trade-content"></div></div>

<script>
// ═══════════════════════════════════════════════════════════════
// DATA CONSTANTS
// ═══════════════════════════════════════════════════════════════
const C={comb_json};
const H3M_BASE={h3_monthly_json};
const H3P_BASE={h3_peak_json};
const E2C_A=[{mode_json[0]['e2c']},{mode_json[1]['e2c']}];
const H3MR_A=[{mode_json[0]['h3mr']},{mode_json[1]['h3mr']}];
const H3C_A=[{mode_json[0]['h3c']},{mode_json[1]['h3c']}];
const RS_A=[{mode_json[0]['rs']},{mode_json[1]['rs']}];
const ES2_A=[{mode_json[0]['es2']},{mode_json[1]['es2']}];
const HT_A=[{mode_json[0]['ht']},{mode_json[1]['ht']}];
const ET_A=[{mode_json[0]['et']},{mode_json[1]['et']}];
const SUMM={json.dumps([mode_json[i]['summary'] for i in range(NM)])};
const EQ_CMP={eq_cmp_json};
const DD_CMP={dd_cmp_json};
const ML=['Original','Min DD'];
const MC=['#1e293b','#dc2626'];
const RC={json.dumps(REGIME_COLORS)};
const RO={json.dumps(REGIME_ORDER)};
const CAP_H3={CAP_3DH},CAP_E2={CAP_E2},CAP_TOT={CAP_TOTAL};
const CST_H3={COST_3DH},CST_E2={COST_E2};
const N_C={N};
const H3A=new Set({json.dumps(sorted(H3_ACTIVE))});
const E2A=new Set({json.dumps(sorted(E2_ACTIVE))});

// ── Active mode pointers ──
let mode=0;
let E2C=E2C_A[0],H3MR=H3MR_A[0],H3C=H3C_A[0],RS=RS_A[0],ES2=ES2_A[0];
let HT_Y=HT_A[0],ET_Y=ET_A[0];
let E2MR={{}};
const h3S={{}},e2S={{}};

// ── Helpers ──
const fmt=v=>{{let s=v<0?'-':'';return s+'EUR '+Math.abs(Math.round(v)).toLocaleString('en-US')}};
const cls=v=>v>=0?'pos':'neg';
const fc=(v,d)=>{{d=d||2;return(v>=0?'+':'')+v.toFixed(d)+'%'}};
function gvs(id,items){{let h='';items.forEach(([l,v,c])=>{{h+=`<div class="gv"><div class="val ${{c}}">${{v}}</div><div class="lbl">${{l}}</div></div>`}});document.getElementById(id).innerHTML=h}}
function sb(id,items){{let h='';items.forEach(([l,v,c])=>{{h+=`<div class="stat-box"><div class="sv ${{c}}">${{v}}</div><div class="sl">${{l}}</div></div>`}});document.getElementById(id).innerHTML=h}}

const _CG={{}};let _resizeBound=false;let _barData=null;
const NY=SUMM[0].wins+SUMM[0].losses;

// ═══════════════════════════════════════════════════════════════
// LINE CHART
// ═══════════════════════════════════════════════════════════════
function drawLC(cid,datasets,isDD,refLine){{
  _CG[cid]={{ds:datasets,isDD:isDD,ref:refLine}};
  const canvas=document.getElementById(cid),ctx=canvas.getContext('2d');
  _drawLC_inner(canvas,ctx,datasets,isDD,refLine);
  if(!_resizeBound){{
    _resizeBound=true;
    window.addEventListener('resize',()=>{{
      Object.keys(_CG).forEach(k=>{{
        const cv=document.getElementById(k);if(!cv)return;
        const cx=cv.getContext('2d');const dd2=_CG[k];if(!dd2)return;
        _drawLC_inner(cv,cx,dd2.ds,dd2.isDD,dd2.ref)
      }});
      if(_barData)drawBar(_barData)
    }})
  }}
}}

function _drawLC_inner(canvas,ctx,datasets,isDD,refLine){{
  const W=canvas.parentElement.clientWidth,H=canvas.parentElement.clientHeight;
  canvas.width=W*2;canvas.height=H*2;ctx.scale(2,2);
  let allV=[];datasets.forEach(d=>allV.push(...d.data));
  if(!isDD)allV.push(0);if(refLine)allV.push(refLine*1.05);
  const mn=Math.min(...allV),mx=Math.max(...allV,0);
  const pad={{t:16,b:26,l:72,r:16}},cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  const range=mx-mn||1,n=datasets[0].data.length,xS=cw/n,yS=ch/range;
  ctx.clearRect(0,0,W,H);ctx.strokeStyle='#e2e8f0';ctx.lineWidth=0.5;
  for(let i=0;i<=5;i++){{let yv=mn+(range/5)*i,yy=pad.t+ch-(yv-mn)*yS;
    ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();
    ctx.fillStyle='#64748b';ctx.font='10px sans-serif';ctx.textAlign='right';
    ctx.fillText(fmt(yv),pad.l-4,yy+3)}}
  if(!isDD&&!refLine){{let y0=pad.t+ch-(0-mn)*yS;ctx.strokeStyle='#94a3b8';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.l,y0);ctx.lineTo(W-pad.r,y0);ctx.stroke()}}
  if(refLine){{let ry=pad.t+ch-(refLine-mn)*yS;
    ctx.strokeStyle='#dc2626';ctx.lineWidth=2;ctx.setLineDash([8,4]);
    ctx.beginPath();ctx.moveTo(pad.l,ry);ctx.lineTo(W-pad.r,ry);ctx.stroke();ctx.setLineDash([])}}
  datasets.forEach(d=>{{
    ctx.strokeStyle=d.color;ctx.lineWidth=d.width||1.5;ctx.setLineDash(d.dash||[]);
    ctx.beginPath();d.data.forEach((v,i)=>{{let x=pad.l+i*xS,y=pad.t+ch-(v-mn)*yS;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}});
    ctx.stroke();ctx.setLineDash([])}});
  ctx.fillStyle='#94a3b8';ctx.font='9px sans-serif';ctx.textAlign='center';
  let lastY='';C.forEach((c,i)=>{{if(String(c.y)!==lastY){{lastY=String(c.y);ctx.fillText(c.y,pad.l+i*xS,H-pad.b+12)}}}})
}}

// ═══════════════════════════════════════════════════════════════
// BAR CHART
// ═══════════════════════════════════════════════════════════════
function drawBar(ys){{
  _barData=ys;
  const canvas=document.getElementById('barChart'),ctx=canvas.getContext('2d');
  const W=canvas.parentElement.clientWidth,H=canvas.parentElement.clientHeight;
  canvas.width=W*2;canvas.height=H*2;ctx.scale(2,2);
  const yrs=Object.keys(ys).map(Number).sort((a,b)=>a-b);
  const pad={{t:14,b:22,l:72,r:14}},cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  let allV=[];yrs.forEach(y=>{{let s=ys[y];allV.push(s.h3,s.e2,s.total)}});allV.push(0);
  const mn=Math.min(...allV),mx=Math.max(...allV),range=mx-mn||1,yS=ch/range;
  const grpW=cw/yrs.length,barW=Math.min(grpW*0.28,13);
  ctx.clearRect(0,0,W,H);
  let y0=pad.t+ch-(0-mn)*yS;ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,y0);ctx.lineTo(W-pad.r,y0);ctx.stroke();
  const colors=['rgb(234,88,12)','rgb(37,99,235)'];
  yrs.forEach((yr,i)=>{{let s=ys[yr],cx2=pad.l+i*grpW+grpW/2;
    [s.h3,s.e2].forEach((v,j)=>{{let bx=cx2+(j-0.5)*barW,by=pad.t+ch-(v-mn)*yS,bh=Math.abs(v)*yS;
      ctx.fillStyle=colors[j];v>=0?ctx.fillRect(bx,by,barW,bh):ctx.fillRect(bx,y0,barW,bh)}});
    let ty=pad.t+ch-(s.total-mn)*yS;ctx.fillStyle='rgb(180,83,9)';ctx.beginPath();ctx.arc(cx2,ty,3.5,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#94a3b8';ctx.font='8px sans-serif';ctx.textAlign='center';ctx.fillText(String(yr).slice(2),cx2,H-pad.b+12)}})
}}

// ═══════════════════════════════════════════════════════════════
// COMPARISON CHARTS (overlay 2 modes)
// ═══════════════════════════════════════════════════════════════
function drawCompCharts(){{
  drawLC('cmpEqChart',[
    {{data:EQ_CMP[0],color:MC[0],width:2}},
    {{data:EQ_CMP[1],color:MC[1],width:2}}
  ],false);
  drawLC('cmpDDChart',[
    {{data:DD_CMP[0],color:MC[0],width:2}},
    {{data:DD_CMP[1],color:MC[1],width:2}}
  ],true);
}}

// ═══════════════════════════════════════════════════════════════
// DRAWDOWN CALCULATOR
// ═══════════════════════════════════════════════════════════════
function calcDD(pnlArr){{
  const n=pnlArr.length,eq=new Array(n),dd=new Array(n);
  let cum=0,pk=0;
  for(let i=0;i<n;i++){{cum+=pnlArr[i];eq[i]=Math.round(cum);if(cum>pk)pk=cum;dd[i]=Math.round(cum-pk)}}
  let mdd=0,periods=[],inDD=false,ds=0;
  for(let i=0;i<n;i++){{
    if(dd[i]<0&&!inDD){{inDD=true;ds=i}}
    else if(dd[i]>=0&&inDD){{inDD=false;let md=0;for(let j=ds;j<i;j++)if(dd[j]<md)md=dd[j];periods.push([ds,i-1,md])}}
  }}
  if(inDD){{let md=0;for(let j=ds;j<n;j++)if(dd[j]<md)md=dd[j];periods.push([ds,n-1,md])}}
  periods.sort((a,b)=>a[2]-b[2]);
  mdd=periods.length>0?periods[0][2]:0;
  return{{eq:eq,dd:dd,max_dd:mdd,top5:periods.slice(0,5)}}
}}

// ═══════════════════════════════════════════════════════════════
// BUILD E2MR from E2C
// ═══════════════════════════════════════════════════════════════
function buildE2MR(){{
  E2MR={{}};
  E2C.forEach(w=>{{
    const mo=w[0],reg=w[1],npos=w[2],pnl=w[3];
    if(!E2MR[mo])E2MR[mo]={{}};
    if(!E2MR[mo][reg])E2MR[mo][reg]={{p:0,c:0,w:0}};
    E2MR[mo][reg].p+=pnl;
    E2MR[mo][reg].c=Math.max(E2MR[mo][reg].c,npos*CST_E2);
    E2MR[mo][reg].w+=1
  }})
}}

// ═══════════════════════════════════════════════════════════════
// TOGGLE SETUP
// ═══════════════════════════════════════════════════════════════
const h3Btns=[],e2Btns=[];
function mkTog(ctn,st,btnsArr,cb){{
  RO.forEach(r=>{{
    const b=document.createElement('button');
    b.style.cssText='padding:5px 12px;border-radius:5px;font-size:0.78em;font-weight:bold;cursor:pointer;transition:all 0.15s;border:2px solid '+(RC[r]||'#666');
    b.textContent=r;b._reg=r;
    function us(){{if(st[r]){{b.style.background=RC[r]||'#666';b.style.color='#fff';b.style.opacity='1'}}else{{b.style.background='transparent';b.style.color='#64748b';b.style.opacity='0.6'}}}}
    us();b.onclick=()=>{{st[r]=!st[r];us();cb()}};
    b._update=us;btnsArr.push(b);ctn.appendChild(b)
  }})
}}

// ═══════════════════════════════════════════════════════════════
// REFERENCE TABLES (update on mode change)
// ═══════════════════════════════════════════════════════════════
function buildRefTables(){{
  // Build 3DH and E2 tables for BOTH modes (always show both)
  for(let mi=0;mi<2;mi++){{
    const rsM=RS_A[mi],es2M=ES2_A[mi];
    let h3b='',e2b='';
    RO.forEach(r=>{{const s=rsM[r]||{{n:0}};if(!s.n)return;
      const pS=s.pf>=99?'inf':s.pf.toFixed(2);
      h3b+=`<tr><td><span class="rb" style="background:${{RC[r]||'#666'}}">${{r}}</span></td>
      <td>${{s.n}}</td><td class="${{s.wr>50?'pos':'neg'}}">${{s.wr.toFixed(1)}}%</td>
      <td class="${{cls(s.avg)}}">${{s.avg>=0?'+':''}}${{s.avg.toFixed(2)}}%</td>
      <td>${{pS}}</td><td class="${{cls(s.pnl)}}">${{fmt(s.pnl)}}</td>
      <td class="${{cls(s.pnl/s.n)}}">${{fmt(s.pnl/s.n)}}</td></tr>`}});
    document.getElementById('h3rf'+mi).innerHTML=h3b;
    RO.forEach(r=>{{const s=es2M[r]||{{n:0}};if(!s.n)return;
      const pS=s.pf>=99?'inf':s.pf.toFixed(2);
      const df=s.active?'<span style="color:#22c55e">SI</span>':'<span style="color:#ef4444">NO</span>';
      e2b+=`<tr><td><span class="rb" style="background:${{RC[r]||'#666'}}">${{r}}</span></td>
      <td>${{s.n}}</td><td class="${{s.wr>50?'pos':'neg'}}">${{s.wr.toFixed(1)}}%</td>
      <td>${{pS}}</td><td class="${{cls(s.pnl)}}">${{fmt(s.pnl)}}</td><td>${{df}}</td></tr>`}});
    document.getElementById('e2rf'+mi).innerHTML=e2b
  }}
}}

// ═══════════════════════════════════════════════════════════════
// recalcAll
// ═══════════════════════════════════════════════════════════════
function recalcAll(){{
  const N=C.length,months=C.map(c=>c.m);

  const mH3=new Array(N).fill(0),mH3n=new Array(N).fill(0);
  C.forEach((c,i)=>{{
    const mr=H3MR[c.m];if(!mr)return;
    RO.forEach(r=>{{if(!h3S[r])return;if(!mr[r])return;mH3[i]+=mr[r][0];mH3n[i]+=mr[r][1]}})
  }});

  const mE2=new Array(N).fill(0),mE2w=new Array(N).fill(0),mE2sk=new Array(N).fill(0);
  C.forEach((c,i)=>{{
    const mr=E2MR[c.m];if(!mr)return;
    RO.forEach(r=>{{
      if(!mr[r])return;
      if(e2S[r]){{mE2[i]+=mr[r].p;mE2w[i]+=mr[r].w}}
      else{{mE2sk[i]+=mr[r].w}}
    }})
  }});

  const h3pk_m={{}};
  const evts=[];
  H3C.forEach(t=>{{if(!h3S[t[2]])return;evts.push([t[0],1]);evts.push([t[1],-1])}});
  evts.sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:a[1]-b[1]);
  let conc=0;
  evts.forEach(e=>{{conc+=e[1];const mo=e[0].substring(0,7);h3pk_m[mo]=Math.max(h3pk_m[mo]||0,conc)}});

  const e2cap_m={{}};
  C.forEach(c=>{{
    const mr=E2MR[c.m];if(!mr)return;
    let mx=0;RO.forEach(r=>{{if(!e2S[r]||!mr[r])return;mx=Math.max(mx,mr[r].c)}});
    e2cap_m[c.m]=mx
  }});

  const mTot=new Array(N),mCh=new Array(N),mCe=new Array(N),mCt=new Array(N),mH3pk=new Array(N);
  C.forEach((c,i)=>{{
    mTot[i]=mH3[i]+mE2[i];
    mCh[i]=(h3pk_m[c.m]||0)*CST_H3;
    mCe[i]=e2cap_m[c.m]||0;
    mCt[i]=mCh[i]+mCe[i];
    mH3pk[i]=h3pk_m[c.m]||0
  }});

  const ddH3=calcDD(mH3);
  const ddE2=calcDD(mE2);
  const ddTot=calcDD(mTot);

  const ys={{}};
  const yearSet=new Set(C.map(c=>c.y));
  yearSet.forEach(y=>{{
    let h3=0,h3n=0,e2=0,e2sk=0,tot=0,nm=0,tw=0,cpk=0,h3pkY=0;
    C.forEach((c,i)=>{{if(c.y!==y)return;
      h3+=mH3[i];h3n+=mH3n[i];e2+=mE2[i];e2sk+=mE2sk[i];
      tot+=mTot[i];nm++;if(mTot[i]>0)tw++;cpk=Math.max(cpk,mCt[i]);h3pkY=Math.max(h3pkY,mH3pk[i])
    }});
    ys[y]={{n:nm,
      h3:Math.round(h3),h3n:h3n,e2:Math.round(e2),e2sk:e2sk,total:Math.round(tot),
      h3_ret:+(h3/CAP_H3*100).toFixed(1),
      e2_ret:+(e2/CAP_E2*100).toFixed(1),tot_ret:+(tot/CAP_TOT*100).toFixed(1),
      tot_wr:nm>0?+(tw/nm*100).toFixed(1):0,cap_max:cpk,h3pk:h3pkY}}
  }});

  const gH3=Math.round(mH3.reduce((a,b)=>a+b,0));
  const gE2=Math.round(mE2.reduce((a,b)=>a+b,0));
  const gTot=gH3+gE2;
  const avgCap=Math.round(mCt.reduce((a,b)=>a+b,0)/N);
  const maxCap=Math.max(...mCt);
  const overM=mCt.filter(v=>v>CAP_TOT).length;

  let h3n_t=0,h3w_t=0,h3gw_t=0,h3gl_t=0,h3p_t=0;
  RO.forEach(r=>{{if(!h3S[r])return;const s=RS[r];if(!s||!s.n)return;
    h3n_t+=s.n;h3w_t+=s.wins;h3gw_t+=s.gw;h3gl_t+=s.gl;h3p_t+=s.pnl}});
  const h3wr_t=h3n_t>0?(h3w_t/h3n_t*100).toFixed(1):'0.0';
  const h3pf_t=h3gl_t>0?(h3gw_t/h3gl_t).toFixed(2):'inf';

  let e2n_t=0,e2w_t=0,e2gw_t=0,e2gl_t=0,e2p_t=0;
  RO.forEach(r=>{{if(!e2S[r])return;const s=ES2[r];if(!s||!s.n)return;
    e2n_t+=s.n;e2w_t+=s.wins;e2gw_t+=s.gw;e2gl_t+=s.gl;e2p_t+=s.pnl}});
  const e2wr_t=e2n_t>0?(e2w_t/e2n_t*100).toFixed(1):'0.0';
  const e2pf_t=e2gl_t>0?(e2gw_t/e2gl_t).toFixed(2):'inf';

  // Section 1: Summary
  gvs('gs_h3',[['PnL',fmt(gH3),cls(gH3)],[h3n_t.toLocaleString()+' trades',fmt(gH3/NY)+'/yr',cls(gH3)],['Max DD',fmt(ddH3.max_dd),'neg']]);
  gvs('gs_e2',[['PnL',fmt(gE2),cls(gE2)],[(gE2/CAP_E2*100/NY).toFixed(1)+'%/yr',fmt(gE2/NY)+'/yr',cls(gE2)],['Max DD',fmt(ddE2.max_dd),'neg']]);
  gvs('gs_tot',[['PnL',fmt(gTot),cls(gTot)],[(gTot/CAP_TOT*100).toFixed(1)+'% ('+(gTot/CAP_TOT*100/NY).toFixed(1)+'%/yr)',fmt(gTot/NY)+'/yr',cls(gTot)],['Max DD',fmt(ddTot.max_dd),'neg']]);
  document.getElementById('cap-bar').innerHTML=
    `<div>Capital medio: <b>${{fmt(avgCap)}}</b></div>`+
    `<div>Capital pico: <b class="neg">${{fmt(maxCap)}}</b></div>`+
    `<div>Meses &gt; {CAP_TOTAL//1000}K: <b class="${{overM>0?'neg':''}}">${{overM}}/${{N}}</b></div>`;

  // Section 2-5: Charts
  drawLC('eqChart',[
    {{data:ddH3.eq,color:'rgb(234,88,12)',width:1}},
    {{data:ddE2.eq,color:'rgb(37,99,235)',width:1}},
    {{data:ddTot.eq,color:'rgb(180,83,9)',width:2.5}}
  ],false);
  drawLC('ddChart',[
    {{data:ddH3.dd,color:'rgb(234,88,12)',width:1}},
    {{data:ddE2.dd,color:'rgb(37,99,235)',width:1}},
    {{data:ddTot.dd,color:'rgb(180,83,9)',width:2}}
  ],true);
  drawLC('capChart',[
    {{data:mCh,color:'rgb(234,88,12)',width:1.5}},
    {{data:mCe,color:'rgb(37,99,235)',width:1}},
    {{data:mCt,color:'rgb(180,83,9)',width:2.5}}
  ],false,CAP_TOT);
  drawBar(ys);

  // Section 7: Weekly stats
  const ws=SUMM[mode].wstats;
  sb('wk-stats',[
    ['Semanas',ws.n,''],
    ['Media',fmt(ws.mean),cls(ws.mean)],
    ['Mediana',fmt(ws.med),cls(ws.med)],
    ['Std Dev',fmt(ws.std),''],
    ['Win Rate',ws.wr+'%',ws.wr>=50?'pos':'neg'],
    ['W/L',ws.wins+'/'+ws.losses,''],
    ['Sharpe',ws.sharpe.toFixed(2),ws.sharpe>=0.5?'pos':'neg'],
    ['Sortino',ws.sortino.toFixed(2),ws.sortino>=1?'pos':'neg'],
    ['Mejor',fmt(ws.best),'pos'],
    ['Peor',fmt(ws.worst),'neg']
  ]);

  // Section 8: Toggle summary
  gvs('h3sm',[['Trades',h3n_t.toLocaleString(),''],['WR',h3wr_t+'%',parseFloat(h3wr_t)>=50?'pos':'neg'],
    ['PF',h3pf_t,parseFloat(h3pf_t)>=1?'pos':'neg'],['PnL',fmt(h3p_t),cls(h3p_t)]]);
  gvs('e2sm',[['Semanas',e2n_t.toLocaleString(),''],['WR',e2wr_t+'%',parseFloat(e2wr_t)>=50?'pos':'neg'],
    ['PF',e2pf_t,parseFloat(e2pf_t)>=1?'pos':'neg'],['PnL',fmt(e2p_t),cls(e2p_t)]]);
  gvs('trsm',[['3DH (sel)',fmt(gH3),cls(gH3)],
    ['E2 (sel)',fmt(gE2),cls(gE2)],['DUAL',fmt(gTot),cls(gTot)]]);

  // Section 6: Yearly Table
  const yrs=Object.keys(ys).sort();
  let yb='',th3=0,te2=0,tt=0,tn=0,tsk=0;
  yrs.forEach(y=>{{let s=ys[y];th3+=s.h3;te2+=s.e2;tt+=s.total;tn+=s.h3n;tsk+=s.e2sk;
    let isW=s.total>0;
    yb+=`<tr class="${{isW?'win-row':'loss-row'}}">
    <td>${{y}}</td>
    <td class="${{cls(s.h3)}}">${{fmt(s.h3)}}</td><td class="${{cls(s.h3_ret)}}">${{fc(s.h3_ret,1)}}</td>
    <td>${{s.h3n||'-'}}</td>
    <td class="${{cls(s.e2)}}">${{fmt(s.e2)}}</td><td class="${{cls(s.e2_ret)}}">${{fc(s.e2_ret,1)}}</td>
    <td style="color:#94a3b8">${{s.e2sk||''}}</td>
    <td class="${{cls(s.total)}}"><strong>${{fmt(s.total)}}</strong></td>
    <td class="${{cls(s.tot_ret)}}">${{fc(s.tot_ret,1)}}</td><td>${{s.tot_wr}}%</td>
    <td class="${{s.cap_max>CAP_TOT?'neg':''}}">${{fmt(s.cap_max)}}</td>
    <td>${{s.h3pk}}</td>
    <td style="font-weight:700;color:${{isW?'#22c55e':'#ef4444'}}">${{isW?'WIN':'LOSS'}}</td></tr>`}});
  document.getElementById('yr-body').innerHTML=yb;
  document.getElementById('yr-foot').innerHTML=
    `<tr style="font-weight:700;border-top:2px solid #475569"><td>TOTAL</td>
    <td class="${{cls(th3)}}">${{fmt(th3)}}</td><td>${{(th3/CAP_H3*100).toFixed(1)}}%</td><td>${{tn}}</td>
    <td class="${{cls(te2)}}">${{fmt(te2)}}</td><td>${{(te2/CAP_E2*100).toFixed(1)}}%</td><td>${{tsk}}</td>
    <td class="${{cls(tt)}}"><strong>${{fmt(tt)}}</strong></td><td>${{(tt/CAP_TOT*100).toFixed(1)}}%</td>
    <td></td><td></td><td></td><td></td></tr>`;

  // Section 9: DD Table
  const top5=ddTot.top5;
  let db='';
  top5.forEach((dd2,i)=>{{let [s,e,d]=dd2,dur=e-s+1;
    let peakVal=Math.max(...ddTot.eq.slice(0,s+1));
    let recIdx=ddTot.eq.findIndex((v,j)=>j>e&&v>=peakVal);
    let recStr=recIdx>=0?months[recIdx]+' ('+(recIdx-e)+' m)':'No recuperado';
    db+=`<tr><td>${{i+1}}</td><td>${{months[s]}}</td><td>${{months[e]}}</td>
    <td>${{dur}} m</td><td class="neg">${{fmt(d)}}</td><td>${{recStr}}</td></tr>`}});
  document.getElementById('dd-body').innerHTML=db;

  // Section 10: Monthly Table
  let mb2='',ct=0;
  C.forEach((c,i)=>{{ct+=mTot[i];let dd3=ddTot.dd[i];
    let oc=mCt[i]>CAP_TOT;
    mb2+=`<tr${{oc?' class="cap-warn"':''}}>
    <td>${{c.m}}</td>
    <td class="${{cls(mH3[i])}}">${{fmt(mH3[i])}}</td><td style="color:#94a3b8">${{mH3n[i]||''}}</td>
    <td class="${{cls(mE2[i])}}">${{fmt(mE2[i])}}</td>
    <td class="${{cls(mTot[i])}}"><strong>${{fmt(mTot[i])}}</strong></td>
    <td class="${{cls(ct)}}">${{fmt(ct)}}</td>
    <td class="${{dd3<0?'neg':''}}">${{dd3<0?fmt(dd3):''}}</td>
    <td style="border-left:2px solid #f97316">${{mH3pk[i]||''}}</td>
    <td>${{mCh[i]?fmt(mCh[i]):''}}</td><td>${{mCe[i]?fmt(mCe[i]):''}}</td>
    <td class="${{oc?'neg':''}}">${{fmt(mCt[i])}}</td></tr>`}});
  document.getElementById('mn-body').innerHTML=mb2
}}

// ═══════════════════════════════════════════════════════════════
// SWITCH MODE
// ═══════════════════════════════════════════════════════════════
function switchMode(m){{
  mode=m;
  E2C=E2C_A[m];H3MR=H3MR_A[m];H3C=H3C_A[m];RS=RS_A[m];ES2=ES2_A[m];
  HT_Y=HT_A[m];ET_Y=ET_A[m];
  buildE2MR();
  RO.forEach(r=>{{h3S[r]=H3A.has(r);e2S[r]=E2A.has(r)}});
  h3Btns.forEach(b=>{{b._update()}});
  e2Btns.forEach(b=>{{b._update()}});
  document.querySelectorAll('.mode-btn').forEach((b,i)=>{{
    b.classList.toggle('act',i===m)
  }});
  buildRefTables();
  recalcAll();
  const activeTab=document.querySelector('#yr-tabs .tab-btn.act');
  if(activeTab)showYear(parseInt(activeTab.textContent))
}}

// ═══════════════════════════════════════════════════════════════
// TRADE TABS
// ═══════════════════════════════════════════════════════════════
let _currentTradeYear=null;
function showYear(y){{
  _currentTradeYear=y;
  const tabs=document.getElementById('yr-tabs'),content=document.getElementById('trade-content');
  tabs.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('act',parseInt(b.textContent)===y));
  let h='';
  const hT=HT_Y[y]||[];
  if(hT.length>0){{
    h+=`<h3 class="h3c" style="margin:14px 0 6px;font-size:0.95em">3DH Short - ${{y}} (${{hT.length}} trades, cap {MAX_3DH}) [Modo: ${{ML[mode]}}]</h3>`;
    h+='<div class="scroll-t" style="max-height:350px"><table>';
    h+='<tr><th>#</th><th>Symbol</th><th>Signal</th><th>Entry</th><th>Exit</th><th>Reg</th><th>Ret%</th><th>PnL</th><th>Cap</th></tr>';
    hT.forEach((t,i)=>{{
      const regC=RC[t[8]]||'#666';
      const capNote=t[6]?`<span style="color:#ef4444;font-size:0.7em">avg${{t[7]}}</span>`:'';
      h+=`<tr><td>${{i+1}}</td><td><b>${{t[0]}}</b></td><td>${{t[1]}}</td><td>${{t[2]}}</td><td>${{t[3]}}</td>
      <td><span class="rb" style="background:${{regC}};font-size:0.6em">${{t[8]}}</span></td>
      <td class="${{cls(t[4])}}">${{fc(t[4],2)}}</td><td class="${{cls(t[5])}}">${{fmt(t[5])}}</td>
      <td>${{capNote}}</td></tr>`}});
    let tp=hT.reduce((s,t)=>s+t[5],0),tw=hT.filter(t=>t[4]>0).length;
    h+=`<tr style="font-weight:700;border-top:2px solid #475569"><td colspan="6">${{hT.length}} trades | WR ${{(tw/hT.length*100).toFixed(1)}}%</td><td></td><td class="${{cls(tp)}}">${{fmt(tp)}}</td><td></td></tr></table></div>`
  }}
  const eT=ET_Y[y]||[];
  if(eT.length>0){{
    h+=`<h3 class="e2c" style="margin:14px 0 6px;font-size:0.95em">E2 Semanal - ${{y}} (${{eT.length}} sem) [Modo: ${{ML[mode]}}]</h3>`;
    h+='<div class="scroll-t" style="max-height:350px"><table>';
    h+='<tr><th>#</th><th>Fecha</th><th>Regimen</th><th>Pos</th><th>PnL</th></tr>';
    eT.forEach((t,i)=>{{const regC=RC[t[1]]||'#666',skC=t[4]?'skip-row':'';
      h+=`<tr class="${{skC}}"><td>${{i+1}}</td><td>${{t[0]}}</td>
      <td><span class="rb" style="background:${{regC}}">${{t[1]}}</span></td>
      <td>${{t[4]?'FUERA':t[2]}}</td><td class="${{cls(t[3])}}">${{t[4]?'-':fmt(t[3])}}</td></tr>`}});
    let tp=eT.reduce((s,t)=>s+t[3],0),tw=eT.filter(t=>t[3]>0&&!t[4]).length,ta=eT.filter(t=>!t[4]).length;
    h+=`<tr style="font-weight:700;border-top:2px solid #475569"><td colspan="3">${{ta}} act / ${{eT.length-ta}} skip | WR ${{ta>0?(tw/ta*100).toFixed(1):'0'}}%</td><td></td><td class="${{cls(tp)}}">${{fmt(tp)}}</td></tr></table></div>`
  }}
  if(!h)h='<p style="color:#94a3b8;text-align:center;padding:20px">Sin trades</p>';
  content.innerHTML=h
}}

// ═══════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════
(function(){{
  const mbar=document.getElementById('mode-bar');
  ML.forEach((label,i)=>{{
    const btn=document.createElement('button');
    btn.className='mode-btn'+(i===0?' act':'');
    btn.textContent=label;
    btn.style.borderColor=MC[i];
    btn.onclick=()=>switchMode(i);
    mbar.appendChild(btn)
  }});

  let cb='';
  SUMM.forEach((s,i)=>{{
    const isMax=Math.max(...SUMM.map(x=>x.total))===s.total;
    cb+=`<tr${{i===0?' style="font-weight:600"':''}}>
    <td><span style="color:${{MC[i]}};font-weight:700">${{ML[i]}}</span></td>
    <td class="${{cls(s.total)}}" style="${{isMax?'font-weight:700':''}}">${{fmt(s.total)}}</td>
    <td class="${{cls(s.total)}}">${{(s.total/CAP_TOT*100).toFixed(1)}}%</td>
    <td class="neg">${{fmt(s.max_dd)}}</td>
    <td class="neg">${{(s.max_dd/CAP_TOT*100).toFixed(1)}}%</td>
    <td>${{s.wins}}W/${{s.losses}}L</td>
    <td class="${{cls(s.h3)}}">${{fmt(s.h3)}}</td>
    <td class="${{cls(s.e2)}}">${{fmt(s.e2)}}</td>
    <td style="color:#94a3b8">${{s.e2_skipped}}</td></tr>`
  }});
  document.getElementById('cmp-body').innerHTML=cb;

  drawCompCharts();

  mkTog(document.getElementById('h3tg'),h3S,h3Btns,recalcAll);
  mkTog(document.getElementById('e2tg'),e2S,e2Btns,recalcAll);

  buildE2MR();
  RO.forEach(r=>{{h3S[r]=H3A.has(r);e2S[r]=E2A.has(r)}});
  buildRefTables();
  recalcAll();

  const allYears=new Set();
  C.forEach(c=>allYears.add(c.y));
  const yrs=[...allYears].sort((a,b)=>a-b);
  const tabs=document.getElementById('yr-tabs');
  yrs.forEach(y=>{{
    const btn=document.createElement('button');
    btn.className='tab-btn';btn.textContent=y;
    btn.onclick=()=>showYear(y);tabs.appendChild(btn)
  }});
  if(yrs.length>0)showYear(yrs[yrs.length-1])
}})();
</script></body></html>"""

out_path = BASE / 'backtest_comparativa_regimenes.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML saved to: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
print(f"\nData processing complete.")
print(f"Months: {len(C)}")
for mi in range(NM):
    s = all_modes[mi]['summary']
    print(f"  {MODE_LABELS[mi]}: PnL EUR {s['total']:,} | MaxDD EUR {s['max_dd']:,} | {s['wins']}W/{s['losses']}L")
    ws = s['wstats']
    print(f"    Weekly: Sharpe {ws['sharpe']}, Sortino {ws['sortino']}, WR {ws['wr']}%")
