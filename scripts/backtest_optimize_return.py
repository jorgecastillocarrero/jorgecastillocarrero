"""
Optimización: Maximizar retorno por trade
- 50 posiciones
- Comparar max 2, 3, 4, 5 entradas/salidas por semana
- Ranking por score que maximiza retorno
"""
import psycopg2
from datetime import timedelta

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
cur = conn.cursor()

COMMISSION_RATE = 0.003
POSITION_SIZE = 20000
MAX_POSITIONS = 50

cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]
print(f'Semanas: {len(weeks)}')

def get_ranked_stocks(week_ending):
    """Obtiene stocks rankeados por score optimizado para retorno"""
    cur.execute('''
        SELECT m.symbol, m.market_cap/1e9 as mcap, b.beat_streak,
               p.peg_ratio, e.eps_growth_yoy, r.revenue_growth_yoy
        FROM market_cap_weekly m
        JOIN beat_streak_weekly b ON m.symbol = b.symbol AND m.week_ending = b.week_ending
        JOIN peg_weekly p ON m.symbol = p.symbol AND m.week_ending = p.week_ending
        JOIN eps_ttm_weekly e ON m.symbol = e.symbol AND m.week_ending = e.week_ending
        JOIN revenue_ttm_weekly r ON m.symbol = r.symbol AND m.week_ending = r.week_ending
        WHERE m.week_ending = %s
        AND m.market_cap >= 1000000000
        AND b.beat_streak >= 4
        AND p.peg_ratio > 0 AND p.peg_ratio <= 1.5
        AND e.eps_growth_yoy > 20
        AND r.revenue_growth_yoy > 12
        AND m.symbol NOT LIKE '%%.%%'
        AND m.symbol NOT LIKE '%%-%%'
        AND LENGTH(m.symbol) <= 5
        AND m.symbol !~ '[0-9]'
        AND RIGHT(m.symbol, 1) NOT IN ('F', 'Y')
    ''', (week_ending,))

    stocks = []
    for row in cur.fetchall():
        symbol, mcap, beat, peg, eps_g, rev_g = row
        peg = float(peg) if peg else 1.5
        beat = int(beat) if beat else 0
        eps_g = float(eps_g) if eps_g else 0
        rev_g = float(rev_g) if rev_g else 0

        # Score optimizado para retorno:
        # - Menor PEG = más barato (peso alto)
        # - Mayor EPS growth = más momentum
        # - Mayor Rev growth = crecimiento real
        # - Beat streak = consistencia
        score = (1.5 - peg) * 50 + min(eps_g, 150) * 0.4 + min(rev_g, 100) * 0.6 + min(beat, 15) * 2

        stocks.append({
            'symbol': symbol,
            'score': score,
            'peg': peg,
            'beat': beat,
            'eps_g': eps_g,
            'rev_g': rev_g
        })

    stocks.sort(key=lambda x: x['score'], reverse=True)
    return stocks

def get_monday_price(symbol, friday, price_type='close'):
    monday = friday + timedelta(days=3)
    cur.execute('''
        SELECT date, open, close FROM fmp_price_history
        WHERE symbol = %s AND date >= %s AND date <= %s
        ORDER BY date LIMIT 1
    ''', (symbol, monday, monday + timedelta(days=5)))
    row = cur.fetchone()
    if row:
        return row[1] if price_type == 'open' else row[2], row[0]
    cur.execute('''
        SELECT date, open, close FROM fmp_price_history
        WHERE symbol = %s AND date <= %s ORDER BY date DESC LIMIT 1
    ''', (symbol, friday + timedelta(days=7)))
    row = cur.fetchone()
    return (row[1] if price_type == 'open' else row[2], row[0]) if row else (None, None)

def run_backtest(max_changes, label):
    """Backtest con 50 posiciones y limite de cambios"""
    positions = {}
    closed_trades = []
    total_commissions = 0

    for i, week in enumerate(weeks[:-1]):
        ranked = get_ranked_stocks(week)
        top50_symbols = set(s['symbol'] for s in ranked[:MAX_POSITIONS])
        ranked_dict = {s['symbol']: s['score'] for s in ranked}

        current_symbols = set(positions.keys())
        is_first_week = (i == 0)

        # Determinar ventas y compras necesarias
        to_sell = current_symbols - top50_symbols
        to_buy = top50_symbols - current_symbols

        # Ordenar ventas por peor score
        sells_list = [(s, ranked_dict.get(s, -9999)) for s in to_sell]
        sells_list.sort(key=lambda x: x[1])

        # Ordenar compras por mejor score
        buys_list = [(s, ranked_dict.get(s, 0)) for s in to_buy]
        buys_list.sort(key=lambda x: x[1], reverse=True)

        sells_done = 0
        buys_done = 0

        # Vender
        max_sells = len(sells_list) if is_first_week else max_changes
        for symbol, _ in sells_list:
            if sells_done >= max_sells:
                break
            if symbol in positions:
                pos = positions[symbol]
                price, date = get_monday_price(symbol, week, 'close')
                if price:
                    gross = price * pos['shares']
                    comm = gross * COMMISSION_RATE
                    total_commissions += comm
                    pnl = gross - comm - pos['cost_basis']
                    pct_return = (pnl / pos['cost_basis']) * 100
                    closed_trades.append({
                        'symbol': symbol,
                        'pnl': pnl,
                        'pct_return': pct_return,
                        'entry_date': pos['entry_date'],
                        'exit_date': date
                    })
                del positions[symbol]
                sells_done += 1

        # Comprar
        max_buys = MAX_POSITIONS if is_first_week else max_changes
        for symbol, _ in buys_list:
            if buys_done >= max_buys:
                break
            if len(positions) >= MAX_POSITIONS:
                break
            price, date = get_monday_price(symbol, week, 'close')
            if price and price > 0:
                shares = round(POSITION_SIZE / price)
                if shares > 0:
                    gross = price * shares
                    comm = gross * COMMISSION_RATE
                    total_commissions += comm
                    positions[symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date,
                        'cost_basis': gross + comm
                    }
                    buys_done += 1

    # Cerrar posiciones restantes
    for symbol, pos in list(positions.items()):
        price, date = get_monday_price(symbol, weeks[-2], 'close')
        if price:
            gross = price * pos['shares']
            comm = gross * COMMISSION_RATE
            total_commissions += comm
            pnl = gross - comm - pos['cost_basis']
            pct_return = (pnl / pos['cost_basis']) * 100
            closed_trades.append({
                'symbol': symbol,
                'pnl': pnl,
                'pct_return': pct_return,
                'entry_date': pos['entry_date'],
                'exit_date': date
            })

    total_pnl = sum(t['pnl'] for t in closed_trades)
    winners = [t for t in closed_trades if t['pnl'] > 0]
    avg_return = sum(t['pct_return'] for t in closed_trades) / len(closed_trades) if closed_trades else 0
    avg_winner = sum(t['pct_return'] for t in winners) / len(winners) if winners else 0
    losers = [t for t in closed_trades if t['pnl'] <= 0]
    avg_loser = sum(t['pct_return'] for t in losers) / len(losers) if losers else 0

    return {
        'label': label,
        'max_changes': max_changes,
        'trades': len(closed_trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(closed_trades) * 100 if closed_trades else 0,
        'pnl': total_pnl,
        'pnl_per_trade': total_pnl / len(closed_trades) if closed_trades else 0,
        'avg_return_pct': avg_return,
        'avg_winner_pct': avg_winner,
        'avg_loser_pct': avg_loser,
        'commissions': total_commissions
    }

print('\n' + '='*90)
print('OPTIMIZACIÓN: MAXIMIZAR RETORNO POR TRADE')
print('50 posiciones, comparando límites de 2, 3, 4, 5 cambios/semana')
print('='*90)

results = []
for max_changes in [2, 3, 4, 5]:
    print(f'\nEjecutando backtest con max {max_changes} entradas/salidas...')
    r = run_backtest(max_changes, f'MAX_{max_changes}')
    results.append(r)
    print(f'  Trades: {r["trades"]}, PnL/trade: ${r["pnl_per_trade"]:,.0f}, Win rate: {r["win_rate"]:.1f}%')

print('\n' + '='*90)
print('COMPARATIVA')
print('='*90)

print(f'\n{"Metrica":<25} {"MAX 2":<15} {"MAX 3":<15} {"MAX 4":<15} {"MAX 5":<15}')
print('-'*85)

print(f'{"Trades totales":<25}', end='')
for r in results:
    print(f'{r["trades"]:>10}     ', end='')
print()

print(f'{"Ganadores":<25}', end='')
for r in results:
    print(f'{r["winners"]:>10}     ', end='')
print()

print(f'{"Perdedores":<25}', end='')
for r in results:
    print(f'{r["losers"]:>10}     ', end='')
print()

print(f'{"Win rate":<25}', end='')
for r in results:
    print(f'{r["win_rate"]:>9.1f}%    ', end='')
print()

print(f'{"PnL total":<25}', end='')
for r in results:
    print(f'${r["pnl"]/1000:>8.0f}K    ', end='')
print()

print(f'{"PnL por trade":<25}', end='')
for r in results:
    print(f'${r["pnl_per_trade"]:>9,.0f}    ', end='')
print()

print(f'{"Retorno medio %":<25}', end='')
for r in results:
    print(f'{r["avg_return_pct"]:>9.1f}%    ', end='')
print()

print(f'{"Ganador medio %":<25}', end='')
for r in results:
    print(f'{r["avg_winner_pct"]:>9.1f}%    ', end='')
print()

print(f'{"Perdedor medio %":<25}', end='')
for r in results:
    print(f'{r["avg_loser_pct"]:>9.1f}%    ', end='')
print()

print(f'{"Comisiones":<25}', end='')
for r in results:
    print(f'${r["commissions"]/1000:>8.0f}K    ', end='')
print()

# Calcular CAGR
print(f'\n{"CAGR (3 años)":<25}', end='')
capital = MAX_POSITIONS * POSITION_SIZE
for r in results:
    cagr = ((1 + r['pnl']/capital) ** (1/3) - 1) * 100
    print(f'{cagr:>9.1f}%    ', end='')
print()

# Mejor resultado
best = max(results, key=lambda x: x['pnl_per_trade'])
print(f'\n*** MEJOR PnL POR TRADE: {best["label"]} con ${best["pnl_per_trade"]:,.0f}/trade ***')

# Mostrar top 50 acciones actuales
print('\n' + '='*90)
print('TOP 50 ACCIONES ACTUALES (última semana disponible)')
print('='*90)

last_week = weeks[-2]
top50 = get_ranked_stocks(last_week)[:50]

print(f'\nSemana: {last_week}')
print(f'\n{"#":<4} {"Symbol":<8} {"Score":<10} {"PEG":<8} {"Beat":<6} {"EPS%":<10} {"Rev%":<10}')
print('-'*60)

for i, s in enumerate(top50):
    print(f'{i+1:<4} {s["symbol"]:<8} {s["score"]:>8.1f} {s["peg"]:>7.2f} {s["beat"]:>5} {s["eps_g"]:>9.1f} {s["rev_g"]:>9.1f}')

cur.close()
conn.close()
