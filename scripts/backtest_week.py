"""
Backtesting semana a semana
Uso: py -3 scripts/backtest_week.py [numero_semana]
"""
import psycopg2
from datetime import timedelta
import sys

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
cur = conn.cursor()

COMMISSION_RATE = 0.003
POSITION_SIZE = 20000

# Obtener semana especificada (default 1)
week_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1

cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
all_weeks = [row[0] for row in cur.fetchall()]

if week_num < 1 or week_num > len(all_weeks):
    print(f"Error: semana debe estar entre 1 y {len(all_weeks)}")
    sys.exit(1)

week = all_weeks[week_num - 1]
prev_week = all_weeks[week_num - 2] if week_num > 1 else None

print(f"SEMANA {week_num}: {week} (viernes)")
print("=" * 90)

# Funcion para obtener stocks filtrados
def get_filtered_stocks(w):
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
    ''', (w,))
    return set(row[0] for row in cur.fetchall())

current_stocks = get_filtered_stocks(week)
prev_stocks = get_filtered_stocks(prev_week) if prev_week else set()

new_stocks = current_stocks - prev_stocks
exit_stocks = prev_stocks - current_stocks
hold_stocks = current_stocks & prev_stocks

print(f"Acciones en filtro: {len(current_stocks)}")
print(f"  - Nuevas (COMPRAR): {len(new_stocks)}")
print(f"  - Mantener: {len(hold_stocks)}")
print(f"  - Salen (VENDER): {len(exit_stocks)}")
print()

# Obtener precio del lunes
def get_monday_price(symbol, friday):
    monday = friday + timedelta(days=3)
    cur.execute('''
        SELECT date, close FROM fmp_price_history
        WHERE symbol = %s AND date >= %s AND date <= %s
        ORDER BY date LIMIT 1
    ''', (symbol, monday, monday + timedelta(days=5)))
    row = cur.fetchone()
    if row:
        return row[1], row[0]
    # Ultimo precio disponible
    cur.execute('''
        SELECT date, close FROM fmp_price_history
        WHERE symbol = %s AND date <= %s
        ORDER BY date DESC LIMIT 1
    ''', (symbol, friday + timedelta(days=7)))
    row = cur.fetchone()
    return (row[1], row[0]) if row else (None, None)

monday = week + timedelta(days=3)
print(f"Ejecucion: Lunes {monday}")
print()

# COMPRAS
if new_stocks:
    print(">>> COMPRAS")
    print(f"{'Symbol':<8} {'Precio':>10} {'Acciones':>10} {'Inversion':>12} {'Comision':>10}")
    print("-" * 55)

    total_buy = 0
    total_comm_buy = 0
    for symbol in sorted(new_stocks):
        price, date = get_monday_price(symbol, week)
        if price and price > 0:
            shares_floor = int(POSITION_SIZE / price)
            shares_ceil = shares_floor + 1
            diff_floor = abs(shares_floor * price - POSITION_SIZE)
            diff_ceil = abs(shares_ceil * price - POSITION_SIZE)
            shares = shares_floor if diff_floor <= diff_ceil else shares_ceil

            if shares > 0:
                investment = price * shares
                commission = investment * COMMISSION_RATE
                total_buy += investment
                total_comm_buy += commission
                print(f"{symbol:<8} {price:>10.2f} {shares:>10} {investment:>12,.0f} {commission:>10,.0f}")

    print("-" * 55)
    print(f"{'TOTAL':<8} {'':>10} {'':>10} {total_buy:>12,.0f} {total_comm_buy:>10,.0f}")
    print()

# VENTAS
if exit_stocks:
    print(">>> VENTAS")
    print(f"{'Symbol':<8} {'Precio':>10} {'Fecha Venta':>15}")
    print("-" * 40)

    for symbol in sorted(exit_stocks):
        price, date = get_monday_price(symbol, week)
        if price:
            print(f"{symbol:<8} {price:>10.2f} {str(date):>15}")
    print()

# MANTIENE
if hold_stocks:
    print(f">>> MANTIENE ({len(hold_stocks)} posiciones)")
    print(", ".join(sorted(hold_stocks)[:20]))
    if len(hold_stocks) > 20:
        print(f"... y {len(hold_stocks) - 20} mas")
    print()

cur.close()
conn.close()
