#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Panel de Control: posiciones abiertas + screener historico
Triple: E2 Semanal + MIX Mensual + RELAX5 Short 15d
"""
import re, json
from pathlib import Path
import pandas as pd
import yfinance as yf

BASE = Path(__file__).parent

STRAT_E2 = {
    'BURBUJA':      [(0, 10, 'L'),  (-20, -10, 'S')],
    'GOLDILOCKS':   [(0, 10, 'L'),  (-20, -10, 'S')],
    'ALCISTA':      [(0, 10, 'L'),  (-10, None, 'S')],
    'NEUTRAL':      [(10, 20, 'L'), (-20, -10, 'S')],
    'CAUTIOUS':     [(-10, None, 'S'), (-20, -10, 'S')],
    'BEARISH':      [(-10, None, 'S'), (-20, -10, 'S')],
    'CRISIS':       [(0, 10, 'S'), (10, 20, 'S')],
    'PANICO':       [(0, 10, 'S'), (10, 20, 'S')],
    'RECOVERY':     [(0, 10, 'L'),  (-20, -10, 'L')],
    'CAPITULACION': [(10, 20, 'L'), (20, 30, 'L')],
}

# ── Load E2 ──
print("Loading E2...")
with open(BASE / 'acciones_navegable.html', 'r', encoding='utf-8') as f:
    html = f.read()
T = json.loads(re.search(r'const T\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))
WEEKS = json.loads(re.search(r'const W\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))
del html

# Current E2 positions (last week)
last_w = WEEKS[-1]
e2_current = []
strat = STRAT_E2.get(last_w['r'], [])
if strat:
    for start, end, dirn in strat:
        selected = last_w['s'][start:end] if end is not None else last_w['s'][start:]
        for s in selected:
            ticker = T[s[0]]['t'] if s[0] < len(T) else f'?{s[0]}'
            e2_current.append({
                't': ticker, 'd': 'LONG' if dirn == 'L' else 'SHORT',
                'fva': round(s[2], 1) if s[2] else 0, 'fv': round(s[1], 1) if s[1] else 0,
                'dd': round(s[3]) if s[3] else 0, 'rsi': round(s[4]) if s[4] else 0,
                'p': round(s[6], 2) if s[6] else 0
            })

# E2 historical trades (all weeks)
print("  Building E2 trade history...")
e2_trades = []
for w in WEEKS:
    strat = STRAT_E2.get(w['r'], [])
    if not strat:
        continue
    for start, end, dirn in strat:
        selected = w['s'][start:end] if end is not None else w['s'][start:]
        for s in selected:
            ret = s[8]
            if ret is None:
                continue
            ret = max(-50, min(50, ret))
            ticker = T[s[0]]['t'] if s[0] < len(T) else '?'
            actual_ret = ret if dirn == 'L' else -ret
            e2_trades.append([
                w['d'], ticker, 'L' if dirn == 'L' else 'S',
                round(actual_ret - 0.3, 2), w['r'],
                round(s[2], 1) if s[2] else 0  # FVA
            ])

print(f"  E2: {len(e2_trades)} trades, current: {len(e2_current)} positions")

# ── Load MIX ──
print("Loading MIX...")
with open(BASE / 'momentum_mensual_mix.html', 'r', encoding='utf-8') as f:
    html = f.read()
MIX_MONTHS = json.loads(re.search(r'const D\s*=\s*(\[.+?\]);\s*\n', html, re.DOTALL).group(1))
del html

# Current MIX positions (last month)
last_m = MIX_MONTHS[-1]
mix_current_top = []
mix_current_bot = []
for s in last_m['top']:
    mix_current_top.append({'t': s[0], 'sec': s[1], 'mom': round(s[2], 1), 'p': round(s[4], 2) if s[4] else 0})
for s in last_m['bot']:
    mix_current_bot.append({'t': s[0], 'sec': s[1], 'mom': round(s[2], 1), 'p': round(s[4], 2) if s[4] else 0})

# MIX historical trades
print("  Building MIX trade history...")
mix_trades = []
for d in MIX_MONTHS:
    for s in d['top']:
        ret = s[3]
        if ret is None:
            continue
        mix_trades.append([d['m'], s[0], 'L', round(ret - 0.3, 2), 'T12M', round(s[2], 1)])
    for s in d['bot']:
        ret = s[3]
        if ret is None:
            continue
        mix_trades.append([d['m'], s[0], 'L', round(ret - 0.3, 2), 'B3M', round(s[2], 1)])

print(f"  MIX: {len(mix_trades)} trades, current: {len(mix_current_top)+len(mix_current_bot)} positions")

# ── Load RELAX5 ──
print("Loading RELAX5...")
with open(BASE / 'data' / 'relax5_15d_trades.json', 'r') as f:
    r5_all = json.load(f)

today = '2026-03-03'
r5_open = [t for t in r5_all if t['exit'] >= today]
r5_recent = [t for t in r5_all if t['sig'] >= '2025-01-01']

# R5 trades for screener: [date, ticker, dir, ret, info, signal_value]
r5_trades = []
for t in r5_all:
    r5_trades.append([t['sig'], t['sym'], 'S', t['ret'], f"{t['entry']}->{t['exit']}", 0])

print(f"  RELAX5: {len(r5_all)} trades, open: {len(r5_open)}, recent: {len(r5_recent)}")

# ── Compute live returns via yfinance ──
print("\nComputing live returns...")
SLIP_PCT = 0.30

e2_tickers = [p['t'] for p in e2_current]
mix_tickers_all = [s['t'] for s in mix_current_top + mix_current_bot]
r5_tickers = [t['sym'] for t in r5_open]
all_open = sorted(set(e2_tickers + mix_tickers_all + r5_tickers))

e2_entry_date_str = ''
mix_entry_date_str = ''
latest_date_str = ''
e2_total_pnl = 0
mix_total_pnl = 0
r5_live_pnl = 0

if all_open:
    e2_signal = last_w['d']
    mix_start = last_m['m'] + '-01'
    dl_start = min(e2_signal, mix_start)
    dl_end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"  Downloading {len(all_open)} tickers from yfinance ({dl_start} -> {dl_end})...")
    raw = yf.download(all_open, start=dl_start, end=dl_end, progress=False)

    # Normalize: {ticker: DataFrame(Open, Close)}
    tk_data = {}
    if len(all_open) == 1:
        tk = all_open[0]
        tk_data[tk] = raw[['Open', 'Close']].dropna()
    else:
        for tk in all_open:
            try:
                sub = pd.DataFrame({'Open': raw['Open'][tk], 'Close': raw['Close'][tk]}).dropna()
                if len(sub):
                    tk_data[tk] = sub
            except Exception:
                pass

    # Latest date
    all_dates = set()
    for sub in tk_data.values():
        all_dates.update(sub.index)
    if all_dates:
        latest_date_str = str(max(all_dates))[:10]

    print(f"  Got data for {len(tk_data)}/{len(all_open)} tickers, latest: {latest_date_str}")

    # E2: entry = open on first trading day AFTER signal
    e2_sig_ts = pd.Timestamp(e2_signal)
    for p in e2_current:
        sub = tk_data.get(p['t'])
        if sub is None or len(sub) == 0:
            continue
        after_sig = sub[sub.index > e2_sig_ts]
        if len(after_sig) == 0:
            continue
        ep = float(after_sig.iloc[0]['Open'])
        cp = float(sub.iloc[-1]['Close'])
        if not e2_entry_date_str:
            e2_entry_date_str = str(after_sig.index[0])[:10]
        if ep > 0 and cp > 0:
            if p['d'] == 'LONG':
                ret = (cp / ep - 1) * 100 - SLIP_PCT
            else:
                ret = (ep / cp - 1) * 100 - SLIP_PCT
            p['entry_p'] = round(ep, 2)
            p['curr_p'] = round(cp, 2)
            p['ret'] = round(ret, 2)
            p['pnl'] = round(ret / 100 * 20000, 0)

    # MIX: entry = open on first trading day >= month start
    mix_ts = pd.Timestamp(mix_start)
    for lst in [mix_current_top, mix_current_bot]:
        for s in lst:
            sub = tk_data.get(s['t'])
            if sub is None or len(sub) == 0:
                continue
            after_start = sub[sub.index >= mix_ts]
            if len(after_start) == 0:
                continue
            ep = float(after_start.iloc[0]['Open'])
            cp = float(sub.iloc[-1]['Close'])
            if not mix_entry_date_str:
                mix_entry_date_str = str(after_start.index[0])[:10]
            if ep > 0 and cp > 0:
                ret = (cp / ep - 1) * 100 - SLIP_PCT
                s['entry_p'] = round(ep, 2)
                s['curr_p'] = round(cp, 2)
                s['ret'] = round(ret, 2)
                s['pnl'] = round(ret / 100 * 20000, 0)

    # R5 open shorts
    for t in r5_open:
        sub = tk_data.get(t['sym'])
        if sub is None or len(sub) == 0:
            continue
        entry_ts = pd.Timestamp(t['entry'])
        entry_rows = sub[sub.index == entry_ts]
        if len(entry_rows) == 0:
            continue
        ep = float(entry_rows.iloc[0]['Open'])
        cp = float(sub.iloc[-1]['Close'])
        if ep > 0 and cp > 0:
            ret = (ep / cp - 1) * 100 - SLIP_PCT
            t['live_ret'] = round(ret, 2)
            t['live_pnl'] = round(ret / 100 * 25000, 0)
            t['entry_p'] = round(ep, 2)
            t['curr_p'] = round(cp, 2)

e2_total_pnl = sum(p.get('pnl', 0) for p in e2_current)
mix_total_pnl = sum(s.get('pnl', 0) for s in mix_current_top + mix_current_bot)
r5_live_pnl = sum(t.get('live_pnl', 0) for t in r5_open)
total_unrealized = e2_total_pnl + mix_total_pnl + r5_live_pnl

print(f"  E2 unrealized: EUR {e2_total_pnl:+,.0f} (entry: {e2_entry_date_str})")
print(f"  MIX unrealized: EUR {mix_total_pnl:+,.0f} (entry: {mix_entry_date_str})")
print(f"  R5 unrealized: EUR {r5_live_pnl:+,.0f}")
print(f"  TOTAL unrealized: EUR {total_unrealized:+,.0f} (prices: {latest_date_str})")

# ── Generate HTML ──
print("\nGenerating panel HTML...")

# Compact trade data for screener
# Format: [date, ticker, direction, net_return, strategy_info, score]
all_trades_json = json.dumps({
    'e2': e2_trades,
    'mix': mix_trades,
    'r5': r5_trades
}, separators=(',', ':'))

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Panel de Control - Triple Strategy</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:12px}}
h1{{text-align:center;font-size:1.3em;margin-bottom:4px;color:#f8fafc}}
.sub{{text-align:center;color:#94a3b8;margin-bottom:12px;font-size:0.8em}}
.tabs{{display:flex;gap:4px;margin-bottom:12px;justify-content:center}}
.tab{{padding:8px 20px;border-radius:8px 8px 0 0;cursor:pointer;font-size:0.85em;font-weight:600;
  background:#1e293b;color:#94a3b8;border:1px solid #334155;border-bottom:none;transition:all 0.2s}}
.tab.active{{background:#334155;color:#f8fafc;border-color:#475569}}
.tab:hover{{color:#e2e8f0}}
.panel{{display:none;background:#1e293b;border-radius:0 10px 10px 10px;padding:16px;border:1px solid #334155}}
.panel.active{{display:block}}
.card{{background:#0f172a;border-radius:8px;padding:12px;margin-bottom:10px;border:1px solid #1e293b}}
.card h3{{font-size:0.95em;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #334155}}
table{{width:100%;border-collapse:collapse;font-size:0.78em}}
th{{background:#334155;color:#f1f5f9;padding:5px 4px;text-align:left;position:sticky;top:0;cursor:pointer;user-select:none}}
th:hover{{background:#475569}}
td{{padding:4px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:rgba(100,116,139,0.15)}}
.pos{{color:#22c55e}}.neg{{color:#ef4444}}
.long{{background:#166534;color:#86efac;padding:1px 6px;border-radius:4px;font-size:0.75em;font-weight:600}}
.short{{background:#991b1b;color:#fca5a5;padding:1px 6px;border-radius:4px;font-size:0.75em;font-weight:600}}
.badge{{display:inline-block;padding:2px 7px;border-radius:9px;font-size:0.7em;font-weight:600}}
.strat-badge{{padding:2px 6px;border-radius:4px;font-size:0.7em;font-weight:600}}
.e2b{{background:#1e3a5f;color:#60a5fa}}.mixb{{background:#3b1f6e;color:#a78bfa}}.r5b{{background:#7c2d12;color:#fb923c}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
@media(max-width:900px){{.grid3,.grid2{{grid-template-columns:1fr}}}}
.summary-row{{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:12px}}
.stat-box{{background:#0f172a;border-radius:7px;padding:8px 14px;text-align:center;min-width:90px}}
.stat-box .val{{font-size:1.1em;font-weight:700}}.stat-box .lbl{{font-size:0.7em;color:#94a3b8}}
input,select{{background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:6px 10px;font-size:0.82em}}
input:focus,select:focus{{outline:none;border-color:#60a5fa}}
.filters{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;align-items:center}}
.filters label{{font-size:0.75em;color:#94a3b8}}
.cnt{{font-size:0.75em;color:#94a3b8;margin-left:8px}}
#screener-table{{max-height:600px;overflow-y:auto;display:block}}
.no-r5{{color:#64748b;font-style:italic;text-align:center;padding:20px}}
</style>
</head>
<body>
<h1>Panel de Control - Triple Strategy</h1>
<div class="sub">E2 Semanal + MIX Mensual + RELAX5 Short 15d | Capital: EUR 1,000,000 | Precios al: {latest_date_str} | 0.30% slippage</div>

<div class="summary-row">
<div class="stat-box"><div class="val" style="color:#60a5fa">{len(e2_current)}</div><div class="lbl">E2 Pos (S{last_w['w']})</div></div>
<div class="stat-box"><div class="val {'pos' if e2_total_pnl >= 0 else 'neg'}">EUR {e2_total_pnl:+,.0f}</div><div class="lbl">E2 PnL</div></div>
<div class="stat-box"><div class="val" style="color:#a78bfa">{len(mix_current_top)+len(mix_current_bot)}</div><div class="lbl">MIX Pos ({last_m['m']})</div></div>
<div class="stat-box"><div class="val {'pos' if mix_total_pnl >= 0 else 'neg'}">EUR {mix_total_pnl:+,.0f}</div><div class="lbl">MIX PnL</div></div>
<div class="stat-box"><div class="val" style="color:#fb923c">{len(r5_open)}</div><div class="lbl">R5 Abiertas</div></div>
<div class="stat-box"><div class="val {'pos' if r5_live_pnl >= 0 else 'neg'}">EUR {r5_live_pnl:+,.0f}</div><div class="lbl">R5 PnL</div></div>
<div class="stat-box" style="border:1px solid #475569"><div class="val {'pos' if total_unrealized >= 0 else 'neg'}" style="font-size:1.3em">EUR {total_unrealized:+,.0f}</div><div class="lbl">TOTAL PnL</div></div>
<div class="stat-box"><div class="val">{last_w['r']}</div><div class="lbl">Regimen S{last_w['w']}</div></div>
</div>

<div class="tabs">
<div class="tab active" onclick="showTab(0)">Posiciones Abiertas</div>
<div class="tab" onclick="showTab(1)">Screener Historico</div>
<div class="tab" onclick="showTab(2)">Por Accion</div>
</div>

<!-- TAB 0: POSICIONES ABIERTAS -->
<div class="panel active" id="p0">
<div class="grid3">

<div class="card">
<h3 style="color:#60a5fa">E2 Semanal - S{last_w['w']} {last_w['d']} <span class="badge" style="background:#4d96ff;color:#000">{last_w['r']}</span></h3>
<table>
<thead><tr><th>Ticker</th><th>Dir</th><th>Entry</th><th>Actual</th><th>Ret%</th><th>PnL</th><th>FVA</th></tr></thead>
<tbody>"""

for p in e2_current:
    dc = 'long' if p['d'] == 'LONG' else 'short'
    ret = p.get('ret')
    pnl = p.get('pnl', 0)
    rc = 'pos' if ret is not None and ret >= 0 else 'neg'
    html += f"<tr><td><strong>{p['t']}</strong></td><td><span class='{dc}'>{p['d']}</span></td>"
    if ret is not None:
        html += f"<td>{p['entry_p']:,.2f}</td><td>{p['curr_p']:,.2f}</td>"
        html += f"<td class='{rc}'>{ret:+.2f}%</td><td class='{rc}'>{pnl:+,.0f}</td>"
    else:
        html += f"<td>-</td><td>-</td><td>-</td><td>-</td>"
    html += f"<td>{p['fva']}</td></tr>"

e2_long_pnl = sum(p.get('pnl', 0) for p in e2_current if p['d'] == 'LONG')
e2_short_pnl = sum(p.get('pnl', 0) for p in e2_current if p['d'] == 'SHORT')
e2c = 'pos' if e2_total_pnl >= 0 else 'neg'
html += f"<tr style='font-weight:700;border-top:2px solid #475569'>"
html += f"<td colspan='2'>TOTAL</td><td colspan='2' style='font-size:0.72em'>L: {e2_long_pnl:+,.0f} | S: {e2_short_pnl:+,.0f}</td>"
html += f"<td class='{e2c}'>{e2_total_pnl/400000*100:+.2f}%</td><td class='{e2c}'>{e2_total_pnl:+,.0f}</td><td></td></tr>"

html += f"""</tbody>
</table>
<div style="margin-top:6px;font-size:0.72em;color:#94a3b8">
Entry: {e2_entry_date_str} open | EUR 20K/accion x 20 = EUR 400K
</div>
</div>

<div class="card">
<h3 style="color:#a78bfa">MIX Mensual - {last_m['m']} <span class="badge" style="background:#a78bfa;color:#000">ALL LONG</span></h3>
<div style="font-size:0.78em;color:#86efac;margin-bottom:4px;font-weight:600">Top 10 - Momentum 12M</div>
<table>
<thead><tr><th>Ticker</th><th>Sector</th><th>Entry</th><th>Actual</th><th>Ret%</th><th>PnL</th></tr></thead>
<tbody>"""

for s in mix_current_top:
    ret = s.get('ret')
    pnl = s.get('pnl', 0)
    rc = 'pos' if ret is not None and ret >= 0 else 'neg'
    html += f"<tr><td><strong>{s['t']}</strong></td><td style='font-size:0.72em'>{s['sec']}</td>"
    if ret is not None:
        html += f"<td>{s['entry_p']:,.2f}</td><td>{s['curr_p']:,.2f}</td>"
        html += f"<td class='{rc}'>{ret:+.2f}%</td><td class='{rc}'>{pnl:+,.0f}</td>"
    else:
        html += f"<td>-</td><td>-</td><td>-</td><td>-</td>"
    html += "</tr>"

mix_top_pnl = sum(s.get('pnl', 0) for s in mix_current_top)
mtc = 'pos' if mix_top_pnl >= 0 else 'neg'
html += f"<tr style='font-weight:700;border-top:2px solid #475569'><td colspan='4'>Subtotal Top10</td>"
html += f"<td class='{mtc}'>{mix_top_pnl/200000*100:+.2f}%</td><td class='{mtc}'>{mix_top_pnl:+,.0f}</td></tr>"

html += """</tbody></table>
<div style="font-size:0.78em;color:#fcd34d;margin:6px 0 4px;font-weight:600">Bot 10 - Mean Reversion 3M</div>
<table>
<thead><tr><th>Ticker</th><th>Sector</th><th>Entry</th><th>Actual</th><th>Ret%</th><th>PnL</th></tr></thead>
<tbody>"""

for s in mix_current_bot:
    ret = s.get('ret')
    pnl = s.get('pnl', 0)
    rc = 'pos' if ret is not None and ret >= 0 else 'neg'
    html += f"<tr><td><strong>{s['t']}</strong></td><td style='font-size:0.72em'>{s['sec']}</td>"
    if ret is not None:
        html += f"<td>{s['entry_p']:,.2f}</td><td>{s['curr_p']:,.2f}</td>"
        html += f"<td class='{rc}'>{ret:+.2f}%</td><td class='{rc}'>{pnl:+,.0f}</td>"
    else:
        html += f"<td>-</td><td>-</td><td>-</td><td>-</td>"
    html += "</tr>"

mix_bot_pnl = sum(s.get('pnl', 0) for s in mix_current_bot)
mbc = 'pos' if mix_bot_pnl >= 0 else 'neg'
html += f"<tr style='font-weight:700;border-top:2px solid #475569'><td colspan='4'>Subtotal Bot10</td>"
html += f"<td class='{mbc}'>{mix_bot_pnl/200000*100:+.2f}%</td><td class='{mbc}'>{mix_bot_pnl:+,.0f}</td></tr>"

mixc = 'pos' if mix_total_pnl >= 0 else 'neg'
html += f"<tr style='font-weight:700;background:#1e293b'><td colspan='4'>TOTAL MIX</td>"
html += f"<td class='{mixc}'>{mix_total_pnl/400000*100:+.2f}%</td><td class='{mixc}'>{mix_total_pnl:+,.0f}</td></tr>"

html += f"""</tbody></table>
<div style="margin-top:6px;font-size:0.72em;color:#94a3b8">
Entry: {mix_entry_date_str} open | EUR 20K/accion x 20 = EUR 400K
</div>
</div>

<div class="card">
<h3 style="color:#fb923c">RELAX5 Short 15d <span class="badge" style="background:#f97316;color:#000">{len(r5_open)} OPEN</span></h3>"""

if r5_open:
    html += """<table>
<thead><tr><th>Ticker</th><th>Entry$</th><th>Actual$</th><th>Exit Date</th><th>Ret%</th><th>PnL</th></tr></thead>
<tbody>"""
    for t in sorted(r5_open, key=lambda x: x['exit']):
        lr = t.get('live_ret', t['ret'])
        lp = t.get('live_pnl', t['pnl'])
        c = 'pos' if lr > 0 else 'neg'
        html += f"<tr><td><strong>{t['sym']}</strong></td>"
        html += f"<td>{t.get('entry_p', '-')}</td><td>{t.get('curr_p', '-')}</td>"
        html += f"<td>{t['exit']}</td>"
        html += f"<td class='{c}'>{lr:+.2f}%</td><td class='{c}'>{lp:+,.0f}</td></tr>"
    r5c = 'pos' if r5_live_pnl >= 0 else 'neg'
    html += f"<tr style='font-weight:700;border-top:2px solid #475569'><td colspan='4'>TOTAL</td>"
    html += f"<td class='{r5c}'>{r5_live_pnl/200000*100:+.2f}%</td><td class='{r5c}'>{r5_live_pnl:+,.0f}</td></tr>"
    html += "</tbody></table>"
else:
    html += '<div class="no-r5">Sin posiciones abiertas actualmente</div>'

# Show recent R5 trades
html += f"""<div style="margin-top:10px;font-size:0.78em;color:#94a3b8;font-weight:600">Ultimos trades 2025-2026</div>
<div style="max-height:300px;overflow-y:auto">
<table>
<thead><tr><th>Ticker</th><th>Senal</th><th>Entry</th><th>Exit</th><th>Ret%</th><th>PnL</th></tr></thead>
<tbody>"""

for t in sorted(r5_recent, key=lambda x: x['sig'], reverse=True):
    c = 'pos' if t['ret'] > 0 else 'neg'
    status = ' style="opacity:0.5"' if t['exit'] < today else ''
    html += f"<tr{status}><td><strong>{t['sym']}</strong></td><td>{t['sig']}</td><td>{t['entry']}</td>"
    html += f"<td>{t['exit']}</td><td class='{c}'>{t['ret']:+.2f}%</td><td class='{c}'>€{t['pnl']:+,.0f}</td></tr>"

html += f"""</tbody></table></div>
<div style="margin-top:6px;font-size:0.72em;color:#94a3b8">
Filtros: d50>8% d100>4% d200+-5% epsg&lt;0 | EUR 25K/trade | 15d hold
</div>
</div>

</div>
</div>

<!-- TAB 1: SCREENER -->
<div class="panel" id="p1">
<div class="filters">
<label>Ticker: <input type="text" id="fTk" placeholder="AAPL, TSLA..." oninput="applyF()"></label>
<label>Estrategia: <select id="fSt" onchange="applyF()">
<option value="">Todas</option><option value="e2">E2 Semanal</option>
<option value="mix">MIX Mensual</option><option value="r5">RELAX5 Short</option>
</select></label>
<label>Direccion: <select id="fDr" onchange="applyF()">
<option value="">Todas</option><option value="L">Long</option><option value="S">Short</option>
</select></label>
<label>Desde: <input type="text" id="fFrom" placeholder="2020-01" oninput="applyF()"></label>
<label>Hasta: <input type="text" id="fTo" placeholder="2026-12" oninput="applyF()"></label>
<label>Ret min: <input type="number" id="fRMin" placeholder="-50" style="width:60px" oninput="applyF()"></label>
<label>Ret max: <input type="number" id="fRMax" placeholder="50" style="width:60px" oninput="applyF()"></label>
<span class="cnt" id="fCnt"></span>
</div>
<div id="screener-table">
<table>
<thead><tr>
<th onclick="sortSc(0)">Fecha</th><th onclick="sortSc(1)">Ticker</th><th onclick="sortSc(2)">Dir</th>
<th onclick="sortSc(3)">Ret%</th><th onclick="sortSc(4)">PnL</th>
<th onclick="sortSc(5)">Estrategia</th><th onclick="sortSc(6)">Info</th>
</tr></thead>
<tbody id="sc-body"></tbody>
</table>
</div>
</div>

<!-- TAB 2: POR ACCION -->
<div class="panel" id="p2">
<div class="filters">
<label>Buscar Ticker: <input type="text" id="tkSearch" placeholder="AAPL" oninput="searchTk()"></label>
<span class="cnt" id="tkCnt"></span>
</div>
<div id="tk-detail"></div>
<div style="margin-top:10px">
<h3 style="font-size:0.9em;margin-bottom:8px">Top 20 acciones mas operadas</h3>
<div id="tk-top"></div>
</div>
</div>

<script>
const AT={all_trades_json};
// Flatten all trades: [date, ticker, dir, ret, strat, info]
let FL=[];
AT.e2.forEach(t=>FL.push([t[0],t[1],t[2],t[3],'E2',t[4]+'|'+t[5]]));
AT.mix.forEach(t=>FL.push([t[0],t[1],t[2],t[3],'MIX',t[4]+'|'+t[5]]));
AT.r5.forEach(t=>FL.push([t[0],t[1],t[2],t[3],'R5',t[4]]));
FL.sort((a,b)=>b[0].localeCompare(a[0]));

const CAP={{'E2':20000,'MIX':20000,'R5':25000}};
let filteredFL=FL;
let sortCol=-1,sortAsc=true;

function showTab(n){{
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===n));
  document.querySelectorAll('.panel').forEach((p,i)=>p.classList.toggle('active',i===n));
  if(n===1&&!document.getElementById('sc-body').innerHTML)applyF();
  if(n===2)initTkTop();
}}

function applyF(){{
  let tk=document.getElementById('fTk').value.toUpperCase().trim();
  let st=document.getElementById('fSt').value;
  let dr=document.getElementById('fDr').value;
  let fr=document.getElementById('fFrom').value;
  let to=document.getElementById('fTo').value;
  let rMin=parseFloat(document.getElementById('fRMin').value);
  let rMax=parseFloat(document.getElementById('fRMax').value);
  let tks=tk?tk.split(',').map(s=>s.trim()).filter(s=>s):[];
  filteredFL=FL.filter(t=>{{
    if(tks.length&&!tks.some(s=>t[1].includes(s)))return false;
    if(st==='e2'&&t[4]!=='E2')return false;
    if(st==='mix'&&t[4]!=='MIX')return false;
    if(st==='r5'&&t[4]!=='R5')return false;
    if(dr&&t[2]!==dr)return false;
    if(fr&&t[0]<fr)return false;
    if(to&&t[0]>to+'~')return false;
    if(!isNaN(rMin)&&t[3]<rMin)return false;
    if(!isNaN(rMax)&&t[3]>rMax)return false;
    return true;
  }});
  renderSc();
}}

function sortSc(col){{
  if(sortCol===col)sortAsc=!sortAsc;else{{sortCol=col;sortAsc=col===3||col===4?false:true}}
  filteredFL.sort((a,b)=>{{
    let va=col===3||col===4?a[3]:String(a[col]);
    let vb=col===3||col===4?b[3]:String(b[col]);
    if(typeof va==='number')return sortAsc?va-vb:vb-va;
    return sortAsc?va.localeCompare(vb):vb.localeCompare(va);
  }});
  renderSc();
}}

function renderSc(){{
  let show=filteredFL.slice(0,500);
  let totalRet=filteredFL.reduce((s,t)=>s+t[3],0);
  let wins=filteredFL.filter(t=>t[3]>0).length;
  let wr=filteredFL.length?((wins/filteredFL.length)*100).toFixed(1):0;
  document.getElementById('fCnt').innerHTML=
    `${{filteredFL.length}} trades | WR ${{wr}}% | Sum ${{totalRet>=0?'+':''}}${{totalRet.toFixed(1)}}%`+
    (filteredFL.length>500?' (mostrando 500)':'');
  let h='';
  show.forEach(t=>{{
    let c=t[3]>=0?'pos':'neg';
    let dc=t[2]==='L'?'long':'short';
    let sc=t[4]==='E2'?'e2b':t[4]==='MIX'?'mixb':'r5b';
    let pnl=t[3]/100*CAP[t[4]];
    h+=`<tr><td>${{t[0]}}</td><td><strong>${{t[1]}}</strong></td>
    <td><span class="${{dc}}">${{t[2]==='L'?'LONG':'SHORT'}}</span></td>
    <td class="${{c}}">${{t[3]>=0?'+':''}}${{t[3].toFixed(2)}}%</td>
    <td class="${{c}}">€${{Math.round(pnl).toLocaleString()}}</td>
    <td><span class="strat-badge ${{sc}}">${{t[4]}}</span></td>
    <td style="font-size:0.7em;color:#64748b">${{t[5]}}</td></tr>`;
  }});
  document.getElementById('sc-body').innerHTML=h;
}}

// ── Tab 3: Per Ticker ──
function searchTk(){{
  let q=document.getElementById('tkSearch').value.toUpperCase().trim();
  if(q.length<1){{document.getElementById('tk-detail').innerHTML='';document.getElementById('tkCnt').innerHTML='';return}}
  let trades=FL.filter(t=>t[1]===q);
  if(!trades.length){{
    document.getElementById('tk-detail').innerHTML=`<div class="no-r5">No trades para ${{q}}</div>`;
    document.getElementById('tkCnt').innerHTML='0 trades';
    return;
  }}
  trades.sort((a,b)=>b[0].localeCompare(a[0]));
  let e2t=trades.filter(t=>t[4]==='E2'),mixt=trades.filter(t=>t[4]==='MIX'),r5t=trades.filter(t=>t[4]==='R5');
  let stats=(arr,cap)=>{{
    if(!arr.length)return{{n:0,wr:0,avg:0,sum:0,pnl:0}};
    let w=arr.filter(t=>t[3]>0).length;
    let s=arr.reduce((a,t)=>a+t[3],0);
    return{{n:arr.length,wr:(w/arr.length*100).toFixed(1),avg:(s/arr.length).toFixed(2),sum:s.toFixed(1),pnl:Math.round(s/100*cap)}};
  }};
  let se2=stats(e2t,20000),smix=stats(mixt,20000),sr5=stats(r5t,25000);
  let stot={{n:trades.length,pnl:se2.pnl+smix.pnl+sr5.pnl}};
  document.getElementById('tkCnt').innerHTML=`${{trades.length}} trades para ${{q}}`;
  let h=`<div class="summary-row">
  <div class="stat-box"><div class="val">${{stot.n}}</div><div class="lbl">Total trades</div></div>
  <div class="stat-box"><div class="val ${{stot.pnl>=0?'pos':'neg'}}">€${{stot.pnl.toLocaleString()}}</div><div class="lbl">PnL Total</div></div>
  <div class="stat-box"><div class="val" style="color:#60a5fa">${{se2.n}}</div><div class="lbl">E2 (${{se2.wr}}% WR)</div></div>
  <div class="stat-box"><div class="val" style="color:#a78bfa">${{smix.n}}</div><div class="lbl">MIX (${{smix.wr}}% WR)</div></div>
  <div class="stat-box"><div class="val" style="color:#fb923c">${{sr5.n}}</div><div class="lbl">R5 (${{sr5.wr}}% WR)</div></div>
  </div>`;
  h+=`<div style="max-height:400px;overflow-y:auto"><table>
  <thead><tr><th>Fecha</th><th>Dir</th><th>Ret%</th><th>PnL</th><th>Estrategia</th><th>Info</th></tr></thead><tbody>`;
  trades.forEach(t=>{{
    let c=t[3]>=0?'pos':'neg';
    let dc=t[2]==='L'?'long':'short';
    let sc=t[4]==='E2'?'e2b':t[4]==='MIX'?'mixb':'r5b';
    let pnl=t[3]/100*CAP[t[4]];
    h+=`<tr><td>${{t[0]}}</td><td><span class="${{dc}}">${{t[2]==='L'?'LONG':'SHORT'}}</span></td>
    <td class="${{c}}">${{t[3]>=0?'+':''}}${{t[3].toFixed(2)}}%</td>
    <td class="${{c}}">€${{Math.round(pnl).toLocaleString()}}</td>
    <td><span class="strat-badge ${{sc}}">${{t[4]}}</span></td>
    <td style="font-size:0.7em;color:#64748b">${{t[5]}}</td></tr>`;
  }});
  h+='</tbody></table></div>';
  document.getElementById('tk-detail').innerHTML=h;
}}

let tkTopDone=false;
function initTkTop(){{
  if(tkTopDone)return;tkTopDone=true;
  let byTk={{}};
  FL.forEach(t=>{{
    if(!byTk[t[1]])byTk[t[1]]={{n:0,pnl:0,wins:0,e2:0,mix:0,r5:0}};
    byTk[t[1]].n++;
    byTk[t[1]].pnl+=t[3]/100*CAP[t[4]];
    if(t[3]>0)byTk[t[1]].wins++;
    byTk[t[1]][t[4].toLowerCase()]++;
  }});
  let arr=Object.entries(byTk).map(([k,v])=>({{tk:k,...v,wr:(v.wins/v.n*100).toFixed(1)}}));
  arr.sort((a,b)=>b.n-a.n);
  let top=arr.slice(0,30);
  let h=`<table><thead><tr><th>Ticker</th><th>Trades</th><th>WR</th><th>PnL</th><th>E2</th><th>MIX</th><th>R5</th></tr></thead><tbody>`;
  top.forEach(t=>{{
    let c=t.pnl>=0?'pos':'neg';
    h+=`<tr style="cursor:pointer" onclick="document.getElementById('tkSearch').value='${{t.tk}}';searchTk();showTab(2)">
    <td><strong>${{t.tk}}</strong></td><td>${{t.n}}</td><td>${{t.wr}}%</td>
    <td class="${{c}}">€${{Math.round(t.pnl).toLocaleString()}}</td>
    <td>${{t.e2||''}}</td><td>${{t.mix||''}}</td><td>${{t.r5||''}}</td></tr>`;
  }});
  h+='</tbody></table>';
  // Also most profitable
  arr.sort((a,b)=>b.pnl-a.pnl);
  let topP=arr.slice(0,20);
  let botP=arr.slice(-20).reverse();
  h+=`<h3 style="font-size:0.9em;margin:12px 0 8px">Top 20 mas rentables</h3><table>
  <thead><tr><th>Ticker</th><th>Trades</th><th>WR</th><th>PnL</th></tr></thead><tbody>`;
  topP.forEach(t=>{{h+=`<tr style="cursor:pointer" onclick="document.getElementById('tkSearch').value='${{t.tk}}';searchTk();showTab(2)">
  <td><strong>${{t.tk}}</strong></td><td>${{t.n}}</td><td>${{t.wr}}%</td><td class="pos">€${{Math.round(t.pnl).toLocaleString()}}</td></tr>`}});
  h+=`</tbody></table>`;
  h+=`<h3 style="font-size:0.9em;margin:12px 0 8px">Top 20 menos rentables</h3><table>
  <thead><tr><th>Ticker</th><th>Trades</th><th>WR</th><th>PnL</th></tr></thead><tbody>`;
  botP.forEach(t=>{{h+=`<tr style="cursor:pointer" onclick="document.getElementById('tkSearch').value='${{t.tk}}';searchTk();showTab(2)">
  <td><strong>${{t.tk}}</strong></td><td>${{t.n}}</td><td>${{t.wr}}%</td><td class="neg">€${{Math.round(t.pnl).toLocaleString()}}</td></tr>`}});
  h+=`</tbody></table>`;
  document.getElementById('tk-top').innerHTML=h;
}}
</script>
</body>
</html>"""

out_path = BASE / 'panel_control.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"HTML saved to: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
