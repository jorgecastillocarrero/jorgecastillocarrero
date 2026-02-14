import csv
from datetime import datetime, timedelta

trades = []

with open(r'C:\Users\usuario\Downloads\U17236599_20260101_20260213.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 15 and row[0] == 'Operaciones' and row[1] == 'Data' and row[2] == 'Order' and row[3] == 'Futuros':
            symbol = row[5]
            fecha_hora = row[6]
            qty = float(row[7].replace(',', ''))
            precio = float(row[8])
            pnl = float(row[13]) if row[13] else 0
            codigo = row[15] if len(row) > 15 else ''

            dt = datetime.strptime(fecha_hora.strip(), '%Y-%m-%d, %H:%M:%S')

            trades.append({
                'dt': dt,
                'fecha': dt.strftime('%d/%m/%Y'),
                'hora': dt.strftime('%H:%M'),
                'symbol': symbol,
                'qty': int(qty),
                'precio': precio,
                'pnl': pnl,
                'codigo': codigo,
                'es_open': 'O' in codigo,
                'es_close': 'C' in codigo
            })

# Separar opens y closes por simbolo
opens_by_symbol = {}
closes_by_symbol = {}

for t in trades:
    s = t['symbol']
    if t['es_open']:
        if s not in opens_by_symbol:
            opens_by_symbol[s] = []
        opens_by_symbol[s].append(t)
    if t['es_close']:
        if s not in closes_by_symbol:
            closes_by_symbol[s] = []
        closes_by_symbol[s].append(t)

# Emparejar usando FIFO y calcular duracion
matched_trades = []

for symbol in closes_by_symbol:
    opens = sorted(opens_by_symbol.get(symbol, []), key=lambda x: x['dt'])
    closes = sorted(closes_by_symbol.get(symbol, []), key=lambda x: x['dt'])

    open_queue = []
    for o in opens:
        for _ in range(abs(o['qty'])):
            open_queue.append(o)

    for c in closes:
        close_qty = abs(c['qty'])
        pnl_per_contract = c['pnl'] / close_qty if close_qty > 0 else 0

        for _ in range(close_qty):
            if open_queue:
                o = open_queue.pop(0)
                duration = c['dt'] - o['dt']
                duration_mins = duration.total_seconds() / 60

                matched_trades.append({
                    'symbol': symbol,
                    'open_dt': o['dt'],
                    'close_dt': c['dt'],
                    'open_fecha': o['fecha'],
                    'close_fecha': c['fecha'],
                    'open_hora': o['hora'],
                    'close_hora': c['hora'],
                    'duration_mins': duration_mins,
                    'pnl': pnl_per_contract
                })

# Categorizar por duracion
def categorize_duration(mins):
    if mins <= 5:
        return '0-5 min'
    elif mins <= 15:
        return '5-15 min'
    elif mins <= 30:
        return '15-30 min'
    elif mins <= 60:
        return '30-60 min'
    elif mins <= 120:
        return '1-2 horas'
    elif mins <= 360:
        return '2-6 horas'
    elif mins <= 1440:
        return '6-24 horas'
    else:
        return '+24 horas'

def duration_order(cat):
    order = ['0-5 min', '5-15 min', '15-30 min', '30-60 min', '1-2 horas', '2-6 horas', '6-24 horas', '+24 horas']
    return order.index(cat) if cat in order else 99

for t in matched_trades:
    t['duration_cat'] = categorize_duration(t['duration_mins'])

# Agrupar por categoria de duracion
by_duration = {}
for t in matched_trades:
    cat = t['duration_cat']
    if cat not in by_duration:
        by_duration[cat] = {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0, 'total_mins': 0}
    by_duration[cat]['trades'] += 1
    by_duration[cat]['pnl'] += t['pnl']
    by_duration[cat]['total_mins'] += t['duration_mins']
    if t['pnl'] > 0:
        by_duration[cat]['wins'] += 1
    elif t['pnl'] < 0:
        by_duration[cat]['losses'] += 1

print('FUTUROS POR DURACION DEL TRADE (01/01 - 13/02/2026)')
print('=' * 90)
print(f"{'Duracion':<14} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'Win%':>8} {'Dur.Prom':>12} {'P&L USD':>18}")
print('-' * 90)

total_pnl = 0
sorted_durations = sorted(by_duration.items(), key=lambda x: duration_order(x[0]))

for cat, data in sorted_durations:
    total_pnl += data['pnl']
    win_rate = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
    avg_mins = data['total_mins'] / data['trades'] if data['trades'] > 0 else 0

    if avg_mins < 60:
        dur_str = f"{avg_mins:.0f} min"
    elif avg_mins < 1440:
        dur_str = f"{avg_mins/60:.1f} hrs"
    else:
        dur_str = f"{avg_mins/1440:.1f} dias"

    sign = '+' if data['pnl'] >= 0 else ''
    print(f"{cat:<14} {data['trades']:>8} {data['wins']:>6} {data['losses']:>8} {win_rate:>7.0f}% {dur_str:>12} {sign}{data['pnl']:>17,.2f}")

print('-' * 90)
total_wins = sum(d['wins'] for d in by_duration.values())
total_losses = sum(d['losses'] for d in by_duration.values())
total_trades = sum(d['trades'] for d in by_duration.values())
win_rate_total = total_wins / total_trades * 100 if total_trades > 0 else 0
sign = '+' if total_pnl >= 0 else ''
print(f"{'TOTAL':<14} {total_trades:>8} {total_wins:>6} {total_losses:>8} {win_rate_total:>7.0f}% {'':<12} {sign}{total_pnl:>17,.2f}")

# Ranking
print('')
print('RANKING POR P&L:')
print('-' * 90)
sorted_by_pnl = sorted(by_duration.items(), key=lambda x: x[1]['pnl'], reverse=True)
for i, (cat, data) in enumerate(sorted_by_pnl, 1):
    sign = '+' if data['pnl'] >= 0 else ''
    wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
    avg_pnl = data['pnl'] / data['trades'] if data['trades'] > 0 else 0
    sign_avg = '+' if avg_pnl >= 0 else ''
    print(f"  {i}. {cat:<14} {sign}{data['pnl']:>12,.2f} USD   Win: {wr:.0f}%   Prom: {sign_avg}{avg_pnl:>8,.2f} USD/trade")

# Detalle de mejores y peores trades por duracion
print('')
print('TOP 5 MEJORES TRADES:')
print('-' * 90)
best = sorted(matched_trades, key=lambda x: x['pnl'], reverse=True)[:5]
for t in best:
    mins = t['duration_mins']
    if mins < 60:
        dur_str = f"{mins:.0f} min"
    elif mins < 1440:
        dur_str = f"{mins/60:.1f} hrs"
    else:
        dur_str = f"{mins/1440:.1f} dias"
    print(f"  {t['symbol']:<8} {t['open_fecha']} {t['open_hora']} -> {t['close_fecha']} {t['close_hora']}  ({dur_str:>10})  +{t['pnl']:>10,.2f} USD")

print('')
print('TOP 5 PEORES TRADES:')
print('-' * 90)
worst = sorted(matched_trades, key=lambda x: x['pnl'])[:5]
for t in worst:
    mins = t['duration_mins']
    if mins < 60:
        dur_str = f"{mins:.0f} min"
    elif mins < 1440:
        dur_str = f"{mins/60:.1f} hrs"
    else:
        dur_str = f"{mins/1440:.1f} dias"
    print(f"  {t['symbol']:<8} {t['open_fecha']} {t['open_hora']} -> {t['close_fecha']} {t['close_hora']}  ({dur_str:>10})  {t['pnl']:>11,.2f} USD")
