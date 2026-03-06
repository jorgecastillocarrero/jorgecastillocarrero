"""Verificar manualmente el regimen S10 con datos del jueves 05/03/2026"""
import psycopg2, json, numpy as np, sys, io
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
cur = conn.cursor()

# SPY datos para MA200 y momentum
cur.execute('''SELECT date, close, open FROM fmp_price_history
               WHERE symbol='SPY' AND date <= '2026-03-05' ORDER BY date DESC LIMIT 210''')
spy_rows = cur.fetchall()
spy_rows.reverse()
closes = [float(r[1]) for r in spy_rows]
dates = [r[0] for r in spy_rows]

spy_close = closes[-1]
ma200 = np.mean(closes[-200:]) if len(closes) >= 200 else np.mean(closes)
spy_dist = (spy_close / ma200 - 1) * 100
above_ma200 = spy_close > ma200

# Momentum 10 semanas (~50 dias trading)
cur.execute('''SELECT date, close FROM fmp_price_history
               WHERE symbol='SPY' AND EXTRACT(DOW FROM date) = 4
               AND date <= '2026-03-05' ORDER BY date DESC LIMIT 12''')
thu_rows = cur.fetchall()
thu_rows.reverse()
if len(thu_rows) >= 11:
    spy_10w_ago = float(thu_rows[-11][1])
    spy_mom = (spy_close / spy_10w_ago - 1) * 100
    print(f'Momentum 10w: SPY {dates[-1]} ({spy_close:.2f}) vs {thu_rows[-11][0]} ({spy_10w_ago:.2f}) = {spy_mom:+.2f}%')
else:
    spy_mom = 0
    print('No hay suficientes jueves para momentum')

# VIX
cur.execute('''SELECT date, close FROM price_history_vix WHERE symbol='^VIX' AND date <= '2026-03-05' ORDER BY date DESC LIMIT 1''')
vix_row = cur.fetchone()
vix_val = float(vix_row[1])

cur.execute('''SELECT date, close FROM price_history_vix WHERE symbol='^VIX' AND date <= '2026-02-27' ORDER BY date DESC LIMIT 1''')
vix_prev = cur.fetchone()

print(f'\n=== DATOS BASICOS S10 (Jue 05/03/2026) ===')
print(f'SPY close:  {spy_close:.2f}')
print(f'SPY MA200:  {ma200:.2f}')
print(f'SPY dist:   {spy_dist:+.2f}%')
print(f'SPY > MA200: {above_ma200}')
print(f'SPY mom10w: {spy_mom:+.2f}%')
print(f'VIX:        {vix_val:.2f} (fecha: {vix_row[0]})')
print(f'VIX prev:   {float(vix_prev[1]):.2f} (fecha: {vix_prev[0]})')
print(f'VIX delta:  {vix_val - float(vix_prev[1]):+.2f}')

# Subsectores
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
symbols = [s['symbol'] for s in sp500]

cur.execute('SELECT symbol, industry FROM fmp_profiles WHERE symbol = ANY(%s)', (symbols,))
profiles = {r[0]: r[1] for r in cur.fetchall()}

# Weekly Thursday data for subsectors (last 60 weeks)
cur.execute('''
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol = ANY(%s) AND EXTRACT(DOW FROM date) = 4
    AND date BETWEEN '2025-01-01' AND '2026-03-05'
    ORDER BY date
''', (symbols,))
price_data = cur.fetchall()

sub_prices = defaultdict(lambda: defaultdict(list))
for sym, dt, close in price_data:
    sub = profiles.get(sym)
    if sub:
        sub_prices[sub][dt].append(float(close))

valid_subs = {sub for sub, dates in sub_prices.items()
              if max(len(stocks) for stocks in dates.values()) >= 3}

all_thursdays = sorted(set(dt for sub in sub_prices for dt in sub_prices[sub]))

print(f'\nSubsectores validos: {len(valid_subs)}')
print(f'Ultimo jueves con datos: {all_thursdays[-1]}')
print(f'Total jueves: {len(all_thursdays)}')

n_total = 0
n_dd_h = 0
n_dd_d = 0
n_rsi_total = 0
n_rsi_55 = 0

for sub in sorted(valid_subs):
    dates_dict = sub_prices[sub]
    series = []
    for thu in all_thursdays:
        if thu in dates_dict and len(dates_dict[thu]) >= 2:
            series.append((thu, np.mean(dates_dict[thu])))

    if len(series) < 14:
        continue

    prices_arr = np.array([s[1] for s in series])

    window = min(52, len(prices_arr))
    high_52 = np.max(prices_arr[-window:])
    current = prices_arr[-1]
    dd = (current / high_52 - 1) * 100

    if len(prices_arr) >= 15:
        deltas = np.diff(prices_arr[-15:])
        gains = np.mean([d for d in deltas if d > 0]) if any(d > 0 for d in deltas) else 0
        losses = np.mean([-d for d in deltas if d < 0]) if any(d < 0 for d in deltas) else 0.001
        rs = gains / losses if losses > 0 else 100
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50

    n_total += 1
    if dd > -10:
        n_dd_h += 1
    if dd < -20:
        n_dd_d += 1
    n_rsi_total += 1
    if rsi > 55:
        n_rsi_55 += 1

pct_dd_h = n_dd_h / n_total * 100
pct_dd_d = n_dd_d / n_total * 100
pct_rsi = n_rsi_55 / n_rsi_total * 100

print(f'\n=== BREADTH (Jueves 05/03) ===')
print(f'Subsectores: {n_total}')
print(f'DD > -10% (healthy): {n_dd_h}/{n_total} = {pct_dd_h:.1f}%')
print(f'DD < -20% (deep):    {n_dd_d}/{n_total} = {pct_dd_d:.1f}%')
print(f'RSI > 55:            {n_rsi_55}/{n_rsi_total} = {pct_rsi:.1f}%')

# Scores
if pct_dd_h >= 75: s_bdd = 2.0
elif pct_dd_h >= 60: s_bdd = 1.0
elif pct_dd_h >= 45: s_bdd = 0.0
elif pct_dd_h >= 30: s_bdd = -1.0
elif pct_dd_h >= 15: s_bdd = -2.0
else: s_bdd = -3.0

if pct_rsi >= 75: s_brsi = 2.0
elif pct_rsi >= 60: s_brsi = 1.0
elif pct_rsi >= 45: s_brsi = 0.0
elif pct_rsi >= 30: s_brsi = -1.0
elif pct_rsi >= 15: s_brsi = -2.0
else: s_brsi = -3.0

if pct_dd_d <= 5: s_ddp = 1.5
elif pct_dd_d <= 15: s_ddp = 0.5
elif pct_dd_d <= 30: s_ddp = -0.5
elif pct_dd_d <= 50: s_ddp = -1.5
else: s_ddp = -2.5

if above_ma200 and spy_dist > 5: s_spy = 1.5
elif above_ma200: s_spy = 0.5
elif spy_dist > -5: s_spy = -0.5
elif spy_dist > -15: s_spy = -1.5
else: s_spy = -2.5

if spy_mom > 5: s_mom = 1.0
elif spy_mom > 0: s_mom = 0.5
elif spy_mom > -5: s_mom = -0.5
elif spy_mom > -15: s_mom = -1.0
else: s_mom = -1.5

total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

print(f'\n=== SCORES ===')
print(f'BDD  (breadth DD):  {s_bdd:+.1f}  (healthy {pct_dd_h:.1f}%)')
print(f'BRSI (breadth RSI): {s_brsi:+.1f}  (RSI>55 {pct_rsi:.1f}%)')
print(f'DDP  (deep DD):     {s_ddp:+.1f}  (deep {pct_dd_d:.1f}%)')
print(f'SPY  (vs MA200):    {s_spy:+.1f}  (dist {spy_dist:+.1f}%, above={above_ma200})')
print(f'MOM  (10w):         {s_mom:+.1f}  (mom {spy_mom:+.1f}%)')
print(f'TOTAL:              {total:+.1f}')

# Regime
if total >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90: regime = 'BURBUJA'
elif total >= 7.0: regime = 'GOLDILOCKS'
elif total >= 4.0: regime = 'ALCISTA'
elif total >= 0.5: regime = 'NEUTRAL'
elif total >= -2.0: regime = 'CAUTIOUS'
elif total >= -5.0: regime = 'BEARISH'
elif total >= -9.0: regime = 'CRISIS'
else: regime = 'PANICO'

# VIX veto
vix_veto = ''
if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
    vix_veto = f'{regime}->NEUTRAL'
    regime = 'NEUTRAL'
elif vix_val >= 35 and regime == 'NEUTRAL':
    vix_veto = 'NEUTRAL->CAUTIOUS'
    regime = 'CAUTIOUS'

# RECOVERY/CAPITULACION
vix_delta = vix_val - float(vix_prev[1])
if regime == 'PANICO' and vix_delta < 0:
    regime = 'CAPITULACION'
elif regime == 'BEARISH' and vix_delta < 0:
    regime = 'RECOVERY'

print(f'\n{"="*50}')
print(f'  REGIMEN S10: {regime}  (score {total:+.1f})')
print(f'  VIX veto: {vix_veto if vix_veto else "ninguno"}')
print(f'  VIX delta: {vix_delta:+.2f} ({float(vix_prev[1]):.2f} -> {vix_val:.2f})')
print(f'{"="*50}')

# Comparar con lo que dice el CSV
print(f'\n=== COMPARACION CON CSV ===')
import csv
with open('data/regimenes_historico.csv') as f:
    rows = list(csv.DictReader(f))
    last = rows[-1]
    print(f'CSV fecha_senal: {last["fecha_senal"]}')
    print(f'CSV regime:      {last["regime"]}')
    print(f'CSV score:       {last["total"]}')
    print(f'CSV spy_close:   {last["spy_close"]}')
    print(f'CSV vix:         {last["vix"]}')
    print(f'CSV spy_dist:    {last["spy_dist"]}')
    print(f'CSV spy_mom:     {last["spy_mom"]}')

cur.close()
conn.close()
