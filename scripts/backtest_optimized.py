"""
Backtesting optimizado:
- 50 posiciones máximo
- Máximo 2 rebalanceos por semana
- Ranking por score compuesto
"""
import psycopg2
from datetime import timedelta

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
cur = conn.cursor()

COMMISSION_RATE = 0.003
POSITION_SIZE = 20000
MAX_POSITIONS = 50
MAX_ENTRIES = 2   # máximo 2 compras por semana
MAX_EXITS = 2     # máximo 2 ventas por semana

# Obtener semanas desde 2023
cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]
print(f'Semanas: {len(weeks)}')

def get_ranked_stocks(week_ending, limit=100):
    """Obtiene stocks que pasan filtro, rankeados por score"""
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
        # Convertir a float
        peg = float(peg) if peg else 0
        beat = int(beat) if beat else 0
        eps_g = float(eps_g) if eps_g else 0
        rev_g = float(rev_g) if rev_g else 0
        # Score: menor PEG mejor, mayor beat mejor, mayor growth mejor
        score = (1.5 - peg) * 30 + min(beat, 20) * 2 + min(eps_g, 100) * 0.3 + min(rev_g, 50) * 0.5
        stocks.append((symbol, score, peg, beat, eps_g, rev_g))

    stocks.sort(key=lambda x: x[1], reverse=True)
    return stocks[:limit]

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
        WHERE symbol = %s AND date <= %s
        ORDER BY date DESC LIMIT 1
    ''', (symbol, friday + timedelta(days=7)))
    row = cur.fetchone()
    return (row[1] if price_type == 'open' else row[2], row[0]) if row else (None, None)

def run_optimized_backtest(price_type='close'):
    positions = {}
    closed_trades = []
    total_commissions = 0
    weekly_stats = []

    for i, week in enumerate(weeks[:-1]):
        ranked = get_ranked_stocks(week)
        ranked_symbols = set(s[0] for s in ranked[:MAX_POSITIONS])
        ranked_dict = {s[0]: s[1] for s in ranked}

        # Primera semana: llenar hasta 50 posiciones sin limite
        is_first_week = (i == 0)

        current_symbols = set(positions.keys())

        to_sell = current_symbols - ranked_symbols
        to_buy = ranked_symbols - current_symbols

        # Ordenar ventas por peor score (vender primero los peores)
        sells_needed = [(s, ranked_dict.get(s, -999)) for s in to_sell]
        sells_needed.sort(key=lambda x: x[1])  # peor score primero
        sells_needed = [s[0] for s in sells_needed]

        # Ordenar compras por mejor score
        buys_needed = [(s, ranked_dict.get(s, 0)) for s in to_buy]
        buys_needed.sort(key=lambda x: x[1], reverse=True)
        buys_needed = [s[0] for s in buys_needed]

        sells_done = 0
        buys_done = 0

        # Vender hasta MAX_EXITS
        for symbol in sells_needed:
            if sells_done >= MAX_EXITS:
                break
            if symbol in positions:
                pos = positions[symbol]
                price, date = get_monday_price(symbol, week, price_type)
                if price:
                    gross_proceeds = price * pos['shares']
                    sell_commission = gross_proceeds * COMMISSION_RATE
                    total_commissions += sell_commission
                    pnl = gross_proceeds - sell_commission - pos['cost_basis']
                    closed_trades.append({
                        'symbol': symbol, 'pnl': pnl,
                        'entry_date': pos['entry_date'], 'exit_date': date
                    })
                del positions[symbol]
                sells_done += 1

        # Comprar hasta MAX_ENTRIES (solo si hay espacio)
        # Primera semana: sin limite para llenar cartera inicial
        max_buys = MAX_POSITIONS if is_first_week else MAX_ENTRIES
        for symbol in buys_needed:
            if buys_done >= max_buys:
                break
            if len(positions) >= MAX_POSITIONS:
                break
            price, date = get_monday_price(symbol, week, price_type)
            if price and price > 0:
                shares = round(POSITION_SIZE / price)
                if shares > 0:
                    gross_cost = price * shares
                    buy_commission = gross_cost * COMMISSION_RATE
                    total_commissions += buy_commission
                    positions[symbol] = {
                        'shares': shares, 'entry_price': price, 'entry_date': date,
                        'cost_basis': gross_cost + buy_commission, 'buy_commission': buy_commission
                    }
                    buys_done += 1

        weekly_stats.append({
            'week': week, 'positions': len(positions),
            'sells': sells_done, 'buys': buys_done
        })

        if (i + 1) % 30 == 0:
            print(f'Semana {i+1}/{len(weeks)}: {len(positions)} pos, {len(closed_trades)} trades')

    # Cerrar posiciones restantes
    for symbol, pos in list(positions.items()):
        price, date = get_monday_price(symbol, weeks[-2], price_type)
        if price:
            gross_proceeds = price * pos['shares']
            sell_commission = gross_proceeds * COMMISSION_RATE
            total_commissions += sell_commission
            pnl = gross_proceeds - sell_commission - pos['cost_basis']
            closed_trades.append({'symbol': symbol, 'pnl': pnl, 'entry_date': pos['entry_date'], 'exit_date': date})

    total_pnl = sum(t['pnl'] for t in closed_trades)
    winners = len([t for t in closed_trades if t['pnl'] > 0])

    return {
        'total_trades': len(closed_trades),
        'winners': winners,
        'losers': len(closed_trades) - winners,
        'win_rate': winners / len(closed_trades) * 100 if closed_trades else 0,
        'total_pnl': total_pnl,
        'total_commissions': total_commissions,
        'weekly_stats': weekly_stats
    }

print('\n=== BACKTEST OPTIMIZADO: 50 posiciones, max 2 rebalanceos/semana ===')
result = run_optimized_backtest('close')
print(f'\nResultados:')
print(f'  Trades totales: {result["total_trades"]}')
print(f'  Ganadores: {result["winners"]} | Perdedores: {result["losers"]}')
print(f'  Win rate: {result["win_rate"]:.1f}%')
print(f'  PnL total: ${result["total_pnl"]:,.0f}')
print(f'  Comisiones: ${result["total_commissions"]:,.0f}')

stats = result['weekly_stats']
avg_pos = sum(s['positions'] for s in stats) / len(stats)
avg_trades = sum(s['sells'] + s['buys'] for s in stats) / len(stats)
weeks_with_trades = len([s for s in stats if s['sells'] + s['buys'] > 0])

print(f'\nEstadisticas semanales:')
print(f'  Posiciones promedio: {avg_pos:.1f}')
print(f'  Operaciones promedio/semana: {avg_trades:.2f}')
print(f'  Semanas con operaciones: {weeks_with_trades}/{len(stats)}')

# Mostrar evolucion de posiciones
print(f'\nEvolucion de posiciones (cada 10 semanas):')
for i, s in enumerate(stats):
    if i % 10 == 0:
        print(f'  Sem {i+1:>3} ({s["week"]}): {s["positions"]:>2} pos, +{s["buys"]} -{s["sells"]}')

cur.close()
conn.close()
