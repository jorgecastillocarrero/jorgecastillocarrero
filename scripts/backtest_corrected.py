"""
Backtesting corregido con:
- Último precio disponible para acciones que desaparecen
- Comisiones y slippage del 0.3% sobre el total de cada operación
"""
import psycopg2
from datetime import datetime, timedelta
import sys

try:
    conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data', connect_timeout=5)
    cur = conn.cursor()
except Exception as e:
    print(f"ERROR: No se pudo conectar a FMP database: {e}")
    print("Asegúrate de que Docker esté corriendo: docker-compose -f docker-compose-fmp.yml up -d")
    sys.exit(1)

COMMISSION_RATE = 0.003  # 0.3%

# Obtener todas las semanas disponibles
cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]
print(f'Semanas disponibles: {len(weeks)} (desde {weeks[0]} hasta {weeks[-1]})')

# Función para obtener acciones que pasan el filtro en una semana
def get_filtered_stocks(week_ending):
    cur.execute('''
        SELECT DISTINCT m.symbol
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
    return set(row[0] for row in cur.fetchall())

# Función para obtener precio del lunes (o siguiente día hábil)
def get_monday_price(symbol, friday, price_type='close'):
    monday = friday + timedelta(days=3)
    cur.execute('''
        SELECT date, open, close
        FROM fmp_price_history
        WHERE symbol = %s AND date >= %s AND date <= %s
        ORDER BY date LIMIT 1
    ''', (symbol, monday, monday + timedelta(days=5)))
    row = cur.fetchone()
    if row:
        return row[1] if price_type == 'open' else row[2], row[0]
    return None, None

# Función para obtener ÚLTIMO precio disponible (para ventas de acciones que desaparecen)
def get_last_available_price(symbol, before_date, price_type='close'):
    cur.execute('''
        SELECT date, open, close
        FROM fmp_price_history
        WHERE symbol = %s AND date <= %s
        ORDER BY date DESC LIMIT 1
    ''', (symbol, before_date))
    row = cur.fetchone()
    if row:
        return row[1] if price_type == 'open' else row[2], row[0]
    return None, None

# Ejecutar backtesting
def run_backtest(price_type='close'):
    positions = {}  # symbol -> {shares, entry_price, entry_date, cost_basis}
    closed_trades = []
    position_size = 20000

    total_commissions = 0

    for i, week in enumerate(weeks):
        current_stocks = get_filtered_stocks(week)

        # Cerrar posiciones que ya no están en el filtro
        symbols_to_close = [s for s in positions if s not in current_stocks]
        for symbol in symbols_to_close:
            pos = positions[symbol]
            # Intentar precio del lunes
            price, date = get_monday_price(symbol, week, price_type)
            if price is None:
                # Si no hay precio el lunes, usar último disponible
                price, date = get_last_available_price(symbol, week + timedelta(days=7), price_type)

            if price:
                gross_proceeds = price * pos['shares']
                sell_commission = gross_proceeds * COMMISSION_RATE
                net_proceeds = gross_proceeds - sell_commission
                total_commissions += sell_commission

                pnl = net_proceeds - pos['cost_basis']
                closed_trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'shares': pos['shares'],
                    'cost_basis': pos['cost_basis'],
                    'gross_proceeds': gross_proceeds,
                    'commission': pos['buy_commission'] + sell_commission,
                    'pnl': pnl
                })
            del positions[symbol]

        # Abrir posiciones nuevas
        new_stocks = [s for s in current_stocks if s not in positions]
        for symbol in new_stocks:
            price, date = get_monday_price(symbol, week, price_type)
            if price and price > 0:
                # Calcular acciones: redondear al más cercano a $20k
                shares_floor = int(position_size / price)
                shares_ceil = shares_floor + 1
                diff_floor = abs(shares_floor * price - position_size)
                diff_ceil = abs(shares_ceil * price - position_size)
                shares = shares_floor if diff_floor <= diff_ceil else shares_ceil

                if shares > 0:
                    gross_cost = price * shares
                    buy_commission = gross_cost * COMMISSION_RATE
                    total_commissions += buy_commission

                    positions[symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date,
                        'cost_basis': gross_cost + buy_commission,
                        'buy_commission': buy_commission
                    }

        if (i + 1) % 50 == 0:
            print(f'Semana {i+1}/{len(weeks)}: {len(positions)} posiciones, {len(closed_trades)} trades cerrados')

    # Cerrar posiciones restantes al último precio disponible
    still_open_count = len(positions)
    for symbol, pos in list(positions.items()):
        price, date = get_last_available_price(symbol, weeks[-1] + timedelta(days=7), price_type)
        if price:
            gross_proceeds = price * pos['shares']
            sell_commission = gross_proceeds * COMMISSION_RATE
            net_proceeds = gross_proceeds - sell_commission
            total_commissions += sell_commission

            pnl = net_proceeds - pos['cost_basis']
            closed_trades.append({
                'symbol': symbol,
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'entry_price': pos['entry_price'],
                'exit_price': price,
                'shares': pos['shares'],
                'cost_basis': pos['cost_basis'],
                'gross_proceeds': gross_proceeds,
                'commission': pos['buy_commission'] + sell_commission,
                'pnl': pnl,
                'still_open': True
            })

    total_pnl = sum(t['pnl'] for t in closed_trades)
    winners = [t for t in closed_trades if t['pnl'] > 0]
    losers = [t for t in closed_trades if t['pnl'] <= 0]

    return {
        'price_type': price_type,
        'total_trades': len(closed_trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(closed_trades) * 100 if closed_trades else 0,
        'total_pnl': total_pnl,
        'total_commissions': total_commissions,
        'avg_pnl_per_trade': total_pnl / len(closed_trades) if closed_trades else 0,
        'still_open': still_open_count
    }

print('\n=== BACKTESTING LUNES APERTURA (con 0.3% comisiones) ===')
result_open = run_backtest('open')
print(f"Trades totales: {result_open['total_trades']}")
print(f"Ganadores: {result_open['winners']} | Perdedores: {result_open['losers']}")
print(f"Win rate: {result_open['win_rate']:.1f}%")
print(f"PnL total: ${result_open['total_pnl']:,.0f}")
print(f"Comisiones totales: ${result_open['total_commissions']:,.0f}")
print(f"Posiciones abiertas al final: {result_open['still_open']}")

print('\n=== BACKTESTING LUNES CIERRE (con 0.3% comisiones) ===')
result_close = run_backtest('close')
print(f"Trades totales: {result_close['total_trades']}")
print(f"Ganadores: {result_close['winners']} | Perdedores: {result_close['losers']}")
print(f"Win rate: {result_close['win_rate']:.1f}%")
print(f"PnL total: ${result_close['total_pnl']:,.0f}")
print(f"Comisiones totales: ${result_close['total_commissions']:,.0f}")
print(f"Posiciones abiertas al final: {result_close['still_open']}")

print('\n=== COMPARATIVA ===')
print(f"Diferencia PnL (Cierre - Apertura): ${result_close['total_pnl'] - result_open['total_pnl']:,.0f}")
mejor = 'CIERRE' if result_close['total_pnl'] > result_open['total_pnl'] else 'APERTURA'
print(f"Mejor estrategia: LUNES {mejor}")

cur.close()
conn.close()
