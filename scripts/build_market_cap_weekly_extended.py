"""
Construye market_cap_weekly extendido usando:
- Precios de fmp_price_history
- Shares outstanding de fmp_income_statements
"""
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from collections import defaultdict

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')

def get_fridays(start_date, end_date):
    """Genera todos los viernes entre start y end."""
    fridays = []
    current = start_date
    while current <= end_date:
        if current.weekday() == 4:  # Viernes
            fridays.append(current)
        current += timedelta(days=1)
    return fridays

def load_shares_data():
    """Carga shares_outstanding de income statements."""
    print('Cargando shares_outstanding...', flush=True)
    cur = conn.cursor()

    cur.execute('''
        SELECT symbol, date, shares_outstanding
        FROM fmp_income_statements
        WHERE shares_outstanding IS NOT NULL AND shares_outstanding > 0
        ORDER BY symbol, date
    ''')

    shares_data = defaultdict(list)
    for row in cur.fetchall():
        symbol, date, shares = row
        shares_data[symbol].append({'date': date, 'shares': int(shares)})

    cur.close()
    print(f'  Cargados {len(shares_data):,} símbolos con shares', flush=True)
    return shares_data

def get_shares_as_of(shares_list, as_of_date):
    """Obtiene shares_outstanding más reciente antes de as_of_date."""
    valid = [s for s in shares_list if s['date'] <= as_of_date]
    if not valid:
        return None
    return max(valid, key=lambda x: x['date'])['shares']

def load_price_data(start_date, end_date):
    """Carga precios de cierre."""
    print(f'Cargando precios desde {start_date} hasta {end_date}...', flush=True)
    cur = conn.cursor()

    cur.execute('''
        SELECT symbol, date, close
        FROM fmp_price_history
        WHERE date >= %s AND date <= %s
        ORDER BY symbol, date
    ''', (start_date, end_date))

    prices = defaultdict(dict)
    for row in cur.fetchall():
        symbol, date, close = row
        prices[symbol][date] = float(close)

    cur.close()
    print(f'  Cargados precios para {len(prices):,} símbolos', flush=True)
    return prices

def get_friday_price(prices_dict, friday):
    """Obtiene precio del viernes o último día disponible de la semana."""
    for days_back in range(5):
        check_date = friday - timedelta(days=days_back)
        if check_date in prices_dict:
            return prices_dict[check_date]
    return None

def get_existing_weeks():
    """Obtiene semanas que ya existen en market_cap_weekly."""
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT week_ending FROM market_cap_weekly ORDER BY week_ending')
    weeks = set(row[0] for row in cur.fetchall())
    cur.close()
    return weeks

def main():
    print('='*70, flush=True)
    print('CONSTRUCCION MARKET_CAP_WEEKLY EXTENDIDO', flush=True)
    print('='*70, flush=True)

    # Verificar rangos
    cur = conn.cursor()
    cur.execute('SELECT MIN(date), MAX(date) FROM fmp_price_history')
    price_range = cur.fetchone()
    print(f'Precios disponibles: {price_range[0]} a {price_range[1]}', flush=True)

    cur.execute('SELECT MIN(date), MAX(date) FROM fmp_income_statements WHERE shares_outstanding > 0')
    shares_range = cur.fetchone()
    print(f'Shares disponibles: {shares_range[0]} a {shares_range[1]}', flush=True)
    cur.close()

    # Determinar rango
    start_date = max(price_range[0], shares_range[0])
    end_date = min(price_range[1], shares_range[1])
    print(f'Rango a procesar: {start_date} a {end_date}', flush=True)

    # Semanas existentes
    existing_weeks = get_existing_weeks()
    print(f'Semanas existentes: {len(existing_weeks)}', flush=True)

    # Viernes a procesar
    all_fridays = get_fridays(start_date, end_date)
    fridays_to_process = [f for f in all_fridays if f not in existing_weeks]
    print(f'Viernes a procesar: {len(fridays_to_process)}', flush=True)

    if not fridays_to_process:
        print('No hay nuevas semanas para procesar.')
        return

    # Cargar datos
    shares_data = load_shares_data()
    prices = load_price_data(start_date, end_date)

    # Procesar
    cur = conn.cursor()
    total_inserted = 0

    for i, friday in enumerate(fridays_to_process):
        rows = []

        for symbol in shares_data.keys():
            if symbol not in prices:
                continue

            # Precio
            price = get_friday_price(prices[symbol], friday)
            if price is None or price <= 0:
                continue

            # Shares
            shares = get_shares_as_of(shares_data[symbol], friday)
            if shares is None or shares <= 0:
                continue

            # Market cap
            market_cap = price * shares
            if market_cap < 1000000:  # Mínimo $1M
                continue

            rows.append((symbol, friday, price, shares, market_cap))

        if rows:
            execute_values(cur, '''
                INSERT INTO market_cap_weekly (symbol, week_ending, close_price, shares_outstanding, market_cap)
                VALUES %s
                ON CONFLICT (symbol, week_ending) DO NOTHING
            ''', rows)
            conn.commit()
            total_inserted += len(rows)

        if (i + 1) % 10 == 0:
            pct = (i + 1) / len(fridays_to_process) * 100
            print(f'[{pct:5.1f}%] {i+1}/{len(fridays_to_process)} semanas | Insertados: {total_inserted:,}', flush=True)

    cur.close()
    print(f'\nFinalizado. Total insertados: {total_inserted:,}', flush=True)

if __name__ == '__main__':
    main()
    conn.close()
