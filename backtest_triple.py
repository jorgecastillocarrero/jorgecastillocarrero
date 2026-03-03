#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtest Triple: E2 Semanal + MIX Mensual + RELAX5 Short 15d
Capital: EUR 1,000,000 = 400K E2 + 400K MIX + 200K RELAX5
"""
import re, json, os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sqlalchemy import create_engine

BASE = Path(__file__).parent
COST_E2 = 20000      # EUR/accion E2
COST_MIX = 20000     # EUR/accion MIX
COST_R5 = 25000      # EUR/trade RELAX5
SLIP = 0.003         # 0.3%
CAP_E2 = 400000
CAP_MIX = 400000
CAP_R5 = 200000
CAP_TOTAL = 1000000
HOLD_R5 = 15         # dias

REGIME_COLORS = {
    'BURBUJA': '#ff6b6b', 'GOLDILOCKS': '#ffd93d', 'ALCISTA': '#6bcb77',
    'NEUTRAL': '#4d96ff', 'CAUTIOUS': '#ff922b', 'BEARISH': '#c084fc',
    'CRISIS': '#ff6b6b', 'PANICO': '#dc2626', 'RECOVERY': '#22d3ee',
    'CAPITULACION': '#f472b6'
}

# ── E2 Strategy rules ──
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

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD E2 SEMANAL
# ═══════════════════════════════════════════════════════════════════════
print("Loading E2 semanal from acciones_navegable.html...")
with open(BASE / 'acciones_navegable.html', 'r', encoding='utf-8') as f:
    html_text = f.read()
m = re.search(r'const T\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
T = json.loads(m.group(1))
m2 = re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
WEEKS = json.loads(m2.group(1))
del html_text

print(f"  E2: {len(WEEKS)} weeks")

e2_weekly = []
for w in WEEKS:
    date, year, sem, regime = w['d'], w['y'], w['w'], w['r']
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
                    pnl += COST_E2 * (ret_val / 100 - SLIP)
                else:
                    pnl += COST_E2 * (-ret_val / 100 - SLIP)
                n_pos += 1
    month_key = date[:7]
    e2_weekly.append({
        'date': date, 'year': year, 'month': month_key,
        'regime': regime, 'pnl': round(pnl, 2)
    })

# ═══════════════════════════════════════════════════════════════════════
# 2. LOAD MIX MENSUAL
# ═══════════════════════════════════════════════════════════════════════
print("Loading MIX mensual from momentum_mensual_mix.html...")
with open(BASE / 'momentum_mensual_mix.html', 'r', encoding='utf-8') as f:
    html_text = f.read()
m = re.search(r'const D\s*=\s*(\[.+?\]);\s*\n', html_text, re.DOTALL)
MIX_MONTHS = json.loads(m.group(1))
del html_text

print(f"  MIX: {len(MIX_MONTHS)} months")

mix_monthly = {}
for d in MIX_MONTHS:
    month_key = d['m']
    pnl = 0
    n_pos = 0
    for s in d['top'] + d['bot']:
        ret = s[3]
        if ret is not None:
            pnl += COST_MIX * (ret / 100 - SLIP)
            n_pos += 1
    mix_monthly[month_key] = {
        'pnl': round(pnl, 2), 'n_pos': n_pos, 'year': d['y']
    }

# ═══════════════════════════════════════════════════════════════════════
# 3. RELAX5 SHORT 15d
# ═══════════════════════════════════════════════════════════════════════
cache_path = BASE / 'data' / 'relax5_15d_trades.json'

if cache_path.exists():
    print(f"Loading RELAX5 from cache: {cache_path}")
    with open(cache_path, 'r') as f:
        r5_trades = json.load(f)
    print(f"  RELAX5: {len(r5_trades)} trades loaded from cache")
else:
    print("Computing RELAX5 from FMP database...")
    engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

    with open(BASE / 'data' / 'sp500_constituents.json', 'r') as f:
        sp500 = json.load(f)
    tickers = sorted(set(s['symbol'] for s in sp500))

    # Earnings
    print('  Loading earnings...')
    ec = []
    for i in range(0, len(tickers), 50):
        b = tickers[i:i+50]
        ec.append(pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated
           FROM fmp_earnings WHERE symbol = ANY(%(t)s) AND eps_actual IS NOT NULL
           ORDER BY symbol, date""", engine, params={'t': b}))
    earn = pd.concat(ec, ignore_index=True)
    earn['date'] = pd.to_datetime(earn['date'])
    earn['eps_actual'] = earn['eps_actual'].astype(float)
    earn['eps_estimated'] = earn['eps_estimated'].astype(float)

    def compute_ef(grp):
        grp = grp.sort_values('date').reset_index(drop=True)
        grp['eps_ttm'] = grp['eps_actual'].rolling(4, min_periods=4).sum()
        grp['eps_ttm_prev'] = grp['eps_ttm'].shift(4)
        grp['eps_growth_yoy'] = np.where(
            grp['eps_ttm_prev'].abs() > 0.01,
            (grp['eps_ttm'] / grp['eps_ttm_prev'] - 1) * 100, np.nan)
        return grp[['symbol', 'date', 'eps_growth_yoy']]

    ef = earn.groupby('symbol', group_keys=False).apply(compute_ef)
    earn_lk = {}
    for sym, grp in ef.groupby('symbol'):
        grp = grp.sort_values('date')
        earn_lk[sym] = list(zip(grp['date'].values, grp['eps_growth_yoy'].values))

    # Prices
    print('  Loading prices...')
    prc = []
    for i in range(0, len(tickers), 25):
        b = tickers[i:i+25]
        prc.append(pd.read_sql("""SELECT symbol, date, open, close
           FROM fmp_price_history WHERE symbol = ANY(%(t)s) AND date >= '2005-01-01'
           ORDER BY symbol, date""", engine, params={'t': b}))
        if (i // 25) % 5 == 0:
            print(f'    Batch {i//25+1}/{(len(tickers)+24)//25}...')
    df = pd.concat(prc, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])

    # Precompute
    sym_data = {}
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('date').reset_index(drop=True)
        n = len(g)
        if n < 210:
            continue
        c = g['close'].values.astype(float)
        o = g['open'].values.astype(float)
        d = g['date'].values
        s50 = pd.Series(c).rolling(50).mean().values
        s100 = pd.Series(c).rolling(100).mean().values
        s200 = pd.Series(c).rolling(200).mean().values
        sym_data[sym] = (c, o, d, s50, s100, s200, n)
    del df

    def get_eg(ll, sig):
        eg = None
        for dd, g in ll:
            if dd < sig:
                eg = g
            else:
                break
        return eg

    # RELAX5: d50>8, d100>4, d200+-5, epsg<0
    print('  Computing RELAX5 signals...')
    r5_trades = []
    for sym, (c, o, dates, s50, s100, s200, n) in sym_data.items():
        ell = earn_lk.get(sym, [])
        busy = -1
        for i in range(200, n - 1):
            if np.isnan(s50[i]) or np.isnan(s100[i]) or np.isnan(s200[i]):
                continue
            if s50[i] <= 0 or s100[i] <= 0 or s200[i] <= 0:
                continue
            dd50 = (c[i] / s50[i] - 1) * 100
            dd100 = (c[i] / s100[i] - 1) * 100
            dd200 = (c[i] / s200[i] - 1) * 100
            if dd50 < 8 or dd100 < 4:
                continue
            if dd200 < -5 or dd200 > 5:
                continue
            sig = dates[i]
            eg = get_eg(ell, sig)
            if eg is None or np.isnan(eg) or eg >= 0:
                continue
            bi = i + 1
            if bi >= n or bi <= busy:
                continue
            bp = o[bi]
            if bp <= 0:
                continue
            si = bi + HOLD_R5
            if si >= n:
                continue
            cv = o[si]
            ret = round((bp / cv - 1) * 100 - 0.3, 2)
            pnl = round(COST_R5 * ret / 100, 2)
            busy = si
            sig_date = str(pd.Timestamp(sig).date())
            entry_date = str(pd.Timestamp(dates[bi]).date())
            exit_date = str(pd.Timestamp(dates[si]).date())
            r5_trades.append({
                'sym': sym, 'sig': sig_date, 'entry': entry_date, 'exit': exit_date,
                'year': pd.Timestamp(sig).year,
                'month': sig_date[:7],
                'ret': ret, 'pnl': pnl
            })

    # Save cache
    os.makedirs(BASE / 'data', exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(r5_trades, f, separators=(',', ':'))
    print(f"  RELAX5: {len(r5_trades)} trades saved to {cache_path}")

print(f"  RELAX5 total PnL: ${sum(t['pnl'] for t in r5_trades):,.0f}")

# ═══════════════════════════════════════════════════════════════════════
# 4. COMBINE MONTHLY
# ═══════════════════════════════════════════════════════════════════════
print("\nCombining strategies...")

# E2 monthly
e2_monthly = defaultdict(lambda: {'pnl': 0, 'n_weeks': 0})
for w in e2_weekly:
    e2_monthly[w['month']]['pnl'] += w['pnl']
    e2_monthly[w['month']]['n_weeks'] += 1

# R5 monthly
r5_monthly = defaultdict(lambda: {'pnl': 0, 'n_trades': 0})
for t in r5_trades:
    r5_monthly[t['month']]['pnl'] += t['pnl']
    r5_monthly[t['month']]['n_trades'] += 1

# All months
all_months = sorted(set(
    list(e2_monthly.keys()) + list(mix_monthly.keys()) + list(r5_monthly.keys())
))

combined = []
for mk in all_months:
    e2 = e2_monthly.get(mk, {'pnl': 0})
    mx = mix_monthly.get(mk, {'pnl': 0})
    r5 = r5_monthly.get(mk, {'pnl': 0, 'n_trades': 0})
    year = int(mk[:4])
    combined.append({
        'm': mk, 'y': year,
        'e2': round(e2['pnl'], 2),
        'mix': round(mx['pnl'], 2),
        'r5': round(r5['pnl'], 2),
        'r5n': r5.get('n_trades', 0),
        'total': round(e2['pnl'] + mx['pnl'] + r5['pnl'], 2)
    })

# ═══════════════════════════════════════════════════════════════════════
# 5. YEAR STATS
# ═══════════════════════════════════════════════════════════════════════
years = sorted(set(c['y'] for c in combined))
year_stats = {}
for y in years:
    ms = [c for c in combined if c['y'] == y]
    e2_y = sum(c['e2'] for c in ms)
    mix_y = sum(c['mix'] for c in ms)
    r5_y = sum(c['r5'] for c in ms)
    r5_n = sum(c['r5n'] for c in ms)
    tot_y = e2_y + mix_y + r5_y
    n = len(ms)
    tot_win = sum(1 for c in ms if c['total'] > 0)
    year_stats[y] = {
        'n': n,
        'e2': round(e2_y), 'mix': round(mix_y), 'r5': round(r5_y), 'total': round(tot_y),
        'e2_ret': round(e2_y / CAP_E2 * 100, 1),
        'mix_ret': round(mix_y / CAP_MIX * 100, 1),
        'r5_ret': round(r5_y / CAP_R5 * 100, 1),
        'tot_ret': round(tot_y / CAP_TOTAL * 100, 1),
        'r5n': r5_n,
        'tot_wr': round(tot_win / n * 100, 1) if n > 0 else 0
    }

# ═══════════════════════════════════════════════════════════════════════
# 6. DRAWDOWNS
# ═══════════════════════════════════════════════════════════════════════
def calc_drawdowns(pnl_series):
    equity = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0
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
        'pct_dd': round(float(np.sum(drawdown < 0) / max(len(drawdown), 1) * 100), 1)
    }

e2_pnl = np.array([c['e2'] for c in combined])
mix_pnl = np.array([c['mix'] for c in combined])
r5_pnl = np.array([c['r5'] for c in combined])
total_pnl = np.array([c['total'] for c in combined])

dd_e2 = calc_drawdowns(e2_pnl)
dd_mix = calc_drawdowns(mix_pnl)
dd_r5 = calc_drawdowns(r5_pnl)
dd_total = calc_drawdowns(total_pnl)

# Also compute E2+MIX without R5 for comparison
em_pnl = np.array([c['e2'] + c['mix'] for c in combined])
dd_em = calc_drawdowns(em_pnl)

# ═══════════════════════════════════════════════════════════════════════
# 7. PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════
g_e2 = round(sum(c['e2'] for c in combined))
g_mix = round(sum(c['mix'] for c in combined))
g_r5 = round(sum(c['r5'] for c in combined))
g_em = g_e2 + g_mix
g_total = g_e2 + g_mix + g_r5
n_months = len(combined)
n_years = len(years)

print(f"\n{'='*130}")
print(f"{'ESTRATEGIA TRIPLE: E2 + MIX + RELAX5 = EUR 1,000,000':^130}")
print(f"{'='*130}")
print(f"{'':>16} {'E2 (400K)':>14} {'MIX (400K)':>14} {'R5 (200K)':>14} {'E2+MIX':>14} {'TRIPLE':>14}")
print(f"{'-'*130}")
print(f"{'PnL Total':>16} E{g_e2:>12,} E{g_mix:>12,} E{g_r5:>12,} E{g_em:>12,} E{g_total:>12,}")
print(f"{'Ret Total':>16} {g_e2/CAP_E2*100:>12.1f}% {g_mix/CAP_MIX*100:>12.1f}% {g_r5/CAP_R5*100:>12.1f}% {g_em/(CAP_E2+CAP_MIX)*100:>12.1f}% {g_total/CAP_TOTAL*100:>12.1f}%")
print(f"{'Avg Anual':>16} E{g_e2//n_years:>12,} E{g_mix//n_years:>12,} E{g_r5//n_years:>12,} E{g_em//n_years:>12,} E{g_total//n_years:>12,}")
print(f"{'Avg %/Ano':>16} {g_e2/CAP_E2*100/n_years:>12.1f}% {g_mix/CAP_MIX*100/n_years:>12.1f}% {g_r5/CAP_R5*100/n_years:>12.1f}% {g_em/(CAP_E2+CAP_MIX)*100/n_years:>12.1f}% {g_total/CAP_TOTAL*100/n_years:>12.1f}%")
print(f"{'Max Drawdown':>16} E{dd_e2['max_dd']:>12,} E{dd_mix['max_dd']:>12,} E{dd_r5['max_dd']:>12,} E{dd_em['max_dd']:>12,} E{dd_total['max_dd']:>12,}")

print(f"\n{'RETORNOS ANUALES':^130}")
print(f"{'Ano':>6} {'E2 PnL':>10} {'E2%':>7} {'MIX PnL':>10} {'MIX%':>7} {'R5 PnL':>10} {'R5%':>7} {'R5#':>4} {'TRIPLE':>10} {'Ret%':>7} {'WR':>5} {'Result':>8}")
print(f"{'-'*130}")
wins, losses = 0, 0
for y in years:
    s = year_stats[y]
    res = 'WIN' if s['total'] > 0 else 'LOSS'
    if s['total'] > 0: wins += 1
    else: losses += 1
    print(f"{y:>6} E{s['e2']:>8,} {s['e2_ret']:>+6.1f}% E{s['mix']:>8,} {s['mix_ret']:>+6.1f}% "
          f"E{s['r5']:>8,} {s['r5_ret']:>+6.1f}% {s['r5n']:>3} E{s['total']:>8,} {s['tot_ret']:>+6.1f}% {s['tot_wr']:>4.0f}% {res:>8}")
print(f"{'-'*130}")
print(f"{'TOTAL':>6} E{g_e2:>8,} {g_e2/CAP_E2*100:>+6.1f}% E{g_mix:>8,} {g_mix/CAP_MIX*100:>+6.1f}% "
      f"E{g_r5:>8,} {g_r5/CAP_R5*100:>+6.1f}% {sum(s['r5n'] for s in year_stats.values()):>3} E{g_total:>8,} {g_total/CAP_TOTAL*100:>+6.1f}%")
print(f"\nCapital: EUR 1,000,000 (E2:400K + MIX:400K + R5:200K)")
print(f"Anos ganadores: {wins} | Perdedores: {losses} | Ratio: {wins/(wins+losses)*100:.0f}%")
print(f"Aporte RELAX5: E{g_r5:+,} ({g_r5/max(g_em,1)*100:+.1f}% sobre E2+MIX)")

# ═══════════════════════════════════════════════════════════════════════
# 8. GENERATE HTML
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating HTML...")

comb_json = json.dumps([{
    'm': c['m'], 'y': c['y'], 'e2': round(c['e2']), 'mix': round(c['mix']),
    'r5': round(c['r5']), 'r5n': c['r5n'], 'tot': round(c['total'])
} for c in combined], separators=(',', ':'))

# R5 yearly for chart
r5_yearly_json = json.dumps([{
    'y': y, 'pnl': year_stats[y]['r5'], 'n': year_stats[y]['r5n']
} for y in years], separators=(',', ':'))

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Triple: E2+MIX+RELAX5 EUR 1M</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:14px}}
h1{{text-align:center;font-size:1.4em;margin-bottom:4px;color:#f8fafc}}
.sub{{text-align:center;color:#94a3b8;margin-bottom:14px;font-size:0.82em}}
.card{{background:#1e293b;border-radius:10px;padding:16px;margin-bottom:12px;border:1px solid #334155}}
.card h2{{font-size:1.05em;color:#f1f5f9;margin-bottom:8px;border-bottom:1px solid #334155;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:0.8em}}
th{{background:#334155;color:#f1f5f9;padding:6px 4px;text-align:left;position:sticky;top:0;cursor:pointer}}
td{{padding:5px 4px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#334155}}
.pos{{color:#22c55e}}.neg{{color:#ef4444}}
.r5c{{color:#f97316}}.e2c{{color:#60a5fa}}.mixc{{color:#a78bfa}}.totc{{color:#fbbf24}}.emc{{color:#94a3b8}}
.gv-row{{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:8px}}
.gv{{background:#0f172a;border-radius:7px;padding:8px 12px;text-align:center;min-width:100px}}
.gv .val{{font-size:1.12em;font-weight:700}}.gv .lbl{{font-size:0.72em;color:#94a3b8;margin-top:2px}}
.side4{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px}}
@media(max-width:1000px){{.side4{{grid-template-columns:1fr 1fr}}}}
.chart-wrap{{position:relative;height:270px;background:#0f172a;border-radius:7px;overflow:hidden;margin-bottom:8px}}
.chart-wrap canvas{{width:100%!important;height:100%!important}}
.legend{{text-align:center;font-size:0.73em;color:#94a3b8;margin-top:4px}}
.win-row td{{background:rgba(34,197,94,0.06)}}.loss-row td{{background:rgba(239,68,68,0.06)}}
.bar-chart{{display:flex;align-items:flex-end;gap:2px;height:180px;margin:8px 0}}
.bar-col{{flex:1;display:flex;flex-direction:column;align-items:center;gap:1px}}
.bar{{width:100%;border-radius:3px 3px 0 0;min-height:1px;position:relative}}
.bar-lbl{{font-size:0.6em;color:#94a3b8;writing-mode:vertical-rl;transform:rotate(180deg);margin-top:4px}}
.bar:hover::after{{content:attr(data-tip);position:absolute;bottom:100%;left:50%;transform:translateX(-50%);
  background:#1e293b;color:#e2e8f0;padding:3px 6px;border-radius:4px;font-size:0.7em;white-space:nowrap;z-index:10;border:1px solid #475569}}
</style>
</head>
<body>
<h1>Estrategia Triple: EUR 1,000,000</h1>
<div class="sub">
E2: Regimen semanal (EUR 400K, 20 acciones x EUR 20K) |
MIX: Momentum 12M+3M mensual (EUR 400K, 20 acciones x EUR 20K) |
RELAX5: Short 15d (EUR 200K, trades a EUR 25K) |
Slippage 0.3%
</div>

<div class="card">
<h2>1. Resumen Global</h2>
<div class="side4">
<div style="border-left:3px solid #60a5fa;padding-left:10px">
<h3 class="e2c" style="font-size:0.85em;margin-bottom:6px">E2 Semanal</h3>
<div class="gv-row" id="gs_e2"></div>
</div>
<div style="border-left:3px solid #a78bfa;padding-left:10px">
<h3 class="mixc" style="font-size:0.85em;margin-bottom:6px">MIX Mensual</h3>
<div class="gv-row" id="gs_mix"></div>
</div>
<div style="border-left:3px solid #f97316;padding-left:10px">
<h3 class="r5c" style="font-size:0.85em;margin-bottom:6px">RELAX5 Short 15d</h3>
<div class="gv-row" id="gs_r5"></div>
</div>
<div style="border-left:3px solid #fbbf24;padding-left:10px">
<h3 class="totc" style="font-size:0.85em;margin-bottom:6px">TRIPLE COMBINADA</h3>
<div class="gv-row" id="gs_tot"></div>
</div>
</div>
</div>

<div class="card">
<h2>2. PnL Acumulado</h2>
<div class="chart-wrap"><canvas id="eqChart"></canvas></div>
<div class="legend">
<span class="e2c">&#9644; E2</span> &nbsp;
<span class="mixc">&#9644; MIX</span> &nbsp;
<span class="emc">&#9644; E2+MIX</span> &nbsp;
<span class="r5c">&#9644; RELAX5</span> &nbsp;
<span class="totc">&#9644; TRIPLE</span>
</div>
</div>

<div class="card">
<h2>3. Drawdown (EUR)</h2>
<div class="chart-wrap"><canvas id="ddChart"></canvas></div>
<div class="legend">
<span class="emc">&#9644; E2+MIX</span> &nbsp;
<span class="r5c">&#9644; RELAX5</span> &nbsp;
<span class="totc">&#9644; TRIPLE</span>
</div>
</div>

<div class="card">
<h2>4. PnL Anual por Estrategia</h2>
<div class="chart-wrap" style="height:220px"><canvas id="barChart"></canvas></div>
<div class="legend">
<span class="e2c">&#9632; E2</span> &nbsp;
<span class="mixc">&#9632; MIX</span> &nbsp;
<span class="r5c">&#9632; RELAX5</span> &nbsp;
<span class="totc">&#9644; TRIPLE</span>
</div>
</div>

<div class="card">
<h2>5. Retornos Anuales</h2>
<table>
<thead><tr>
<th>Ano</th>
<th>E2 PnL</th><th>E2 %</th>
<th>MIX PnL</th><th>MIX %</th>
<th>R5 PnL</th><th>R5 %</th><th>R5 #</th>
<th>Triple PnL</th><th>WR Mes</th><th>Resultado</th>
</tr></thead>
<tbody id="yr-body"></tbody>
<tfoot id="yr-foot"></tfoot>
</table>
</div>

<div class="card">
<h2>6. Top 5 Peores Drawdowns (Triple)</h2>
<table>
<thead><tr><th>#</th><th>Inicio</th><th>Fin</th><th>Meses</th><th>Max DD (EUR)</th><th>Recuperacion</th></tr></thead>
<tbody id="dd-body"></tbody>
</table>
</div>

<div class="card">
<h2>7. Detalle Mensual</h2>
<div style="max-height:500px;overflow-y:auto">
<table>
<thead><tr><th>Mes</th><th>E2</th><th>MIX</th><th>R5</th><th>R5#</th><th>Triple</th><th>E2 Acum</th><th>MIX Acum</th><th>R5 Acum</th><th>Triple Acum</th><th>DD</th></tr></thead>
<tbody id="mn-body"></tbody>
</table>
</div>
</div>

<script>
const C={comb_json};
const DE2={json.dumps(dd_e2)};
const DMX={json.dumps(dd_mix)};
const DR5={json.dumps(dd_r5)};
const DT={json.dumps(dd_total)};
const DEM={json.dumps(dd_em)};
const YS={json.dumps(year_stats)};
const CAP_E2={CAP_E2},CAP_MIX={CAP_MIX},CAP_R5={CAP_R5},CAP_TOT={CAP_TOTAL};
const fmt=v=>{{let s=v<0?'-':'';return s+'EUR '+Math.abs(Math.round(v)).toLocaleString('en-US')}};
const pct=(v,c)=>(v/c*100).toFixed(1)+'%';
const cls=v=>v>=0?'pos':'neg';
const N_YEARS={n_years};

// ── 1. Summary ──
function gvs(id,items){{
  let h='';
  items.forEach(([l,v,c])=>{{h+=`<div class="gv"><div class="val ${{c}}">${{v}}</div><div class="lbl">${{l}}</div></div>`}});
  document.getElementById(id).innerHTML=h;
}}
gvs('gs_e2',[['PnL',fmt({g_e2}),cls({g_e2})],['{g_e2/CAP_E2*100:.1f}% total',fmt({g_e2//n_years})+'/yr',cls({g_e2})],['Max DD',fmt({dd_e2['max_dd']}),'neg']]);
gvs('gs_mix',[['PnL',fmt({g_mix}),cls({g_mix})],['{g_mix/CAP_MIX*100:.1f}% total',fmt({g_mix//n_years})+'/yr',cls({g_mix})],['Max DD',fmt({dd_mix['max_dd']}),'neg']]);
gvs('gs_r5',[['PnL',fmt({g_r5}),cls({g_r5})],['{sum(s["r5n"] for s in year_stats.values())} trades','{g_r5/CAP_R5*100:.1f}% sobre 200K',''],['Max DD',fmt({dd_r5['max_dd']}),'neg']]);
gvs('gs_tot',[['PnL',fmt({g_total}),cls({g_total})],['{g_total/CAP_TOTAL*100:.1f}% total',fmt({g_total//n_years})+'/yr ({g_total/CAP_TOTAL*100/n_years:.1f}%)',cls({g_total})],['Max DD',fmt({dd_total['max_dd']}),'neg']]);

// ── 2+3. Line Charts ──
function drawLC(cid,datasets,isDD){{
  const canvas=document.getElementById(cid);
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
      ctx.strokeStyle=d.color;ctx.lineWidth=d.width||1.5;
      if(d.dash)ctx.setLineDash(d.dash);else ctx.setLineDash([]);
      ctx.beginPath();
      d.data.forEach((v,i)=>{{let x=pad.l+i*xS,y=pad.t+ch-(v-mn)*yS;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}});
      ctx.stroke();ctx.setLineDash([]);
    }});
    ctx.fillStyle='#64748b';ctx.font='9px sans-serif';ctx.textAlign='center';
    let lastY='';
    C.forEach((c,i)=>{{if(String(c.y)!==lastY){{lastY=String(c.y);ctx.fillText(c.y,pad.l+i*xS,H-pad.b+12)}}}});
  }}
  draw();window.addEventListener('resize',draw);
}}
drawLC('eqChart',[
  {{data:DE2.eq,color:'rgb(96,165,250)',width:1}},
  {{data:DMX.eq,color:'rgb(167,139,250)',width:1}},
  {{data:DEM.eq,color:'rgb(148,163,184)',width:1,dash:[4,3]}},
  {{data:DR5.eq,color:'rgb(249,115,22)',width:1.3}},
  {{data:DT.eq,color:'rgb(251,191,36)',width:2}}
],false);
drawLC('ddChart',[
  {{data:DEM.dd,color:'rgb(148,163,184)',width:1,dash:[4,3]}},
  {{data:DR5.dd,color:'rgb(249,115,22)',width:1}},
  {{data:DT.dd,color:'rgb(251,191,36)',width:1.5}}
],true);

// ── 4. Bar Chart ──
(function(){{
  const canvas=document.getElementById('barChart');
  const ctx=canvas.getContext('2d');
  function draw(){{
    const W=canvas.parentElement.clientWidth,H=canvas.parentElement.clientHeight;
    canvas.width=W*2;canvas.height=H*2;ctx.scale(2,2);
    const yrs=Object.keys(YS).map(Number).sort((a,b)=>a-b);
    const pad={{t:14,b:22,l:70,r:14}};
    const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
    let allV=[];yrs.forEach(y=>{{let s=YS[y];allV.push(s.e2,s.mix,s.r5,s.total)}});
    allV.push(0);
    const mn=Math.min(...allV),mx=Math.max(...allV);
    const range=mx-mn||1;
    const yS=ch/range;
    const grpW=cw/yrs.length;
    const barW=Math.min(grpW*0.22,12);
    ctx.clearRect(0,0,W,H);
    let y0=pad.t+ch-(0-mn)*yS;
    ctx.strokeStyle='#475569';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,y0);ctx.lineTo(W-pad.r,y0);ctx.stroke();
    ctx.strokeStyle='#334155';ctx.lineWidth=0.5;
    for(let i=0;i<=4;i++){{
      let yv=mn+(range/4)*i,yy=pad.t+ch-(yv-mn)*yS;
      ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();
      ctx.fillStyle='#94a3b8';ctx.font='9px sans-serif';ctx.textAlign='right';
      ctx.fillText(fmt(yv),pad.l-4,yy+3);
    }}
    const colors=['rgb(96,165,250)','rgb(167,139,250)','rgb(249,115,22)'];
    yrs.forEach((yr,i)=>{{
      let s=YS[yr];
      let cx=pad.l+i*grpW+grpW/2;
      let vals=[s.e2,s.mix,s.r5];
      vals.forEach((v,j)=>{{
        let bx=cx+(j-1)*barW-barW/2;
        let by=pad.t+ch-(v-mn)*yS;
        let bh=Math.abs(v)*yS;
        ctx.fillStyle=colors[j];
        if(v>=0)ctx.fillRect(bx,by,barW,bh);
        else ctx.fillRect(bx,y0,barW,bh);
      }});
      // Triple line dot
      let ty=pad.t+ch-(s.total-mn)*yS;
      ctx.fillStyle='rgb(251,191,36)';ctx.beginPath();ctx.arc(cx,ty,3,0,Math.PI*2);ctx.fill();
      // Year label
      ctx.fillStyle='#64748b';ctx.font='8px sans-serif';ctx.textAlign='center';
      ctx.fillText(String(yr).slice(2),cx,H-pad.b+12);
    }});
  }}
  draw();window.addEventListener('resize',draw);
}})();

// ── 5. Year Table ──
(function(){{
  let yrs=Object.keys(YS).sort(),b='',te2=0,tmx=0,tr5=0,tt=0,w=0,l=0,tn=0;
  yrs.forEach(y=>{{
    let s=YS[y];te2+=s.e2;tmx+=s.mix;tr5+=s.r5;tt+=s.total;tn+=s.r5n;
    let isW=s.total>0;if(isW)w++;else l++;
    let r5r=s.r5_ret!==undefined?((s.r5_ret>=0?'+':'')+s.r5_ret+'%'):'-';
    let tr=s.tot_ret!==undefined?((s.tot_ret>=0?'+':'')+s.tot_ret+'%'):'-';
    b+=`<tr class="${{isW?'win-row':'loss-row'}}"><td>${{y}}</td>
    <td class="${{cls(s.e2)}}">${{fmt(s.e2)}}</td><td class="${{cls(s.e2_ret)}}">${{s.e2_ret>=0?'+':''}}${{s.e2_ret}}%</td>
    <td class="${{cls(s.mix)}}">${{fmt(s.mix)}}</td><td class="${{cls(s.mix_ret)}}">${{s.mix_ret>=0?'+':''}}${{s.mix_ret}}%</td>
    <td class="${{cls(s.r5)}}">${{fmt(s.r5)}}</td><td class="${{cls(s.r5_ret||0)}}">${{r5r}}</td><td>${{s.r5n||'-'}}</td>
    <td class="${{cls(s.total)}}"><strong>${{fmt(s.total)}}</strong></td>
    <td>${{s.tot_wr}}%</td>
    <td style="font-weight:700;color:${{isW?'#22c55e':'#ef4444'}}">${{isW?'WIN':'LOSS'}}</td></tr>`;
  }});
  document.getElementById('yr-body').innerHTML=b;
  document.getElementById('yr-foot').innerHTML=
    `<tr style="font-weight:700;border-top:2px solid #475569"><td>TOTAL</td>
    <td class="${{cls(te2)}}">${{fmt(te2)}}</td><td>${{(te2/CAP_E2*100).toFixed(1)}}%</td>
    <td class="${{cls(tmx)}}">${{fmt(tmx)}}</td><td>${{(tmx/CAP_MIX*100).toFixed(1)}}%</td>
    <td class="${{cls(tr5)}}">${{fmt(tr5)}}</td><td>${{(tr5/CAP_R5*100).toFixed(1)}}%</td><td>${{tn}}</td>
    <td class="${{cls(tt)}}"><strong>${{fmt(tt)}}</strong></td>
    <td></td><td>${{w}}W / ${{l}}L</td></tr>`;
}})();

// ── 6. DD Table ──
(function(){{
  let top5=DT.top5;
  let months=C.map(c=>c.m);
  let b='';
  top5.forEach((dd,i)=>{{
    let [s,e,d]=dd;
    let dur=e-s+1;
    let eq=DT.eq;
    let peakVal=Math.max(...eq.slice(0,s+1));
    let recIdx=eq.findIndex((v,j)=>j>e&&v>=peakVal);
    let recStr=recIdx>=0?months[recIdx]+' ('+(recIdx-e)+' m)':'No recuperado';
    b+=`<tr><td>${{i+1}}</td><td>${{months[s]}}</td><td>${{months[e]}}</td>
    <td>${{dur}} m</td><td class="neg">${{fmt(d)}}</td>
    <td>${{recStr}}</td></tr>`;
  }});
  document.getElementById('dd-body').innerHTML=b;
}})();

// ── 7. Monthly ──
(function(){{
  let b='',ce2=0,cmx=0,cr5=0,ct=0;
  C.forEach((c,i)=>{{
    ce2+=c.e2;cmx+=c.mix;cr5+=c.r5;ct+=c.tot;
    let dd=DT.dd[i];
    b+=`<tr><td>${{c.m}}</td>
    <td class="${{cls(c.e2)}}">${{fmt(c.e2)}}</td>
    <td class="${{cls(c.mix)}}">${{fmt(c.mix)}}</td>
    <td class="${{cls(c.r5)}}">${{fmt(c.r5)}}</td>
    <td>${{c.r5n||''}}</td>
    <td class="${{cls(c.tot)}}"><strong>${{fmt(c.tot)}}</strong></td>
    <td class="${{cls(ce2)}}">${{fmt(ce2)}}</td>
    <td class="${{cls(cmx)}}">${{fmt(cmx)}}</td>
    <td class="${{cls(cr5)}}">${{fmt(cr5)}}</td>
    <td class="${{cls(ct)}}"><strong>${{fmt(ct)}}</strong></td>
    <td class="${{dd<0?'neg':''}}">${{dd<0?fmt(dd):''}}</td></tr>`;
  }});
  document.getElementById('mn-body').innerHTML=b;
}})();
</script>
</body>
</html>"""

out_path = BASE / 'backtest_triple.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nHTML saved to: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
