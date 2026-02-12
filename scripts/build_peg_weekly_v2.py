"""
Construye peg_weekly extendido - VERSION OPTIMIZADA
Procesa semana por semana para evitar cargar 150M de precios en memoria.
"""
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from collections import defaultdict

DB_URL = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'

def get_fridays(start_date, end_date):
    """Genera todos los viernes entre start y end."""
    fridays = []
    current = start_date
    while current <= end_date:
        if current.weekday() == 4:
            fridays.append(current)
        current += timedelta(days=1)
    return fridays

def load_eps_data(conn):
    """Carga EPS trimestrales de income statements."""
    print('Cargando EPS de income statements...', flush=True)
    cur = conn.cursor()

    cur.execute('''
        SELECT symbol, date, period, eps
        FROM fmp_income_statements
        WHERE eps IS NOT NULL
        ORDER BY symbol, date
    ''')

    eps_data = defaultdict(list)
    for row in cur.fetchall():
        symbol, date, period, eps = row
        eps_data[symbol].append({'date': date, 'period': period, 'eps': float(eps)})

    cur.close()
    print(f'  Cargados {len(eps_data):,} simbolos con EPS', flush=True)
    return eps_data

def calculate_eps_ttm(eps_list, as_of_date):
    """Calcula EPS TTM (suma de ultimos 4 trimestres antes de as_of_date)."""
    valid_quarters = [e for e in eps_list if e['date'] <= as_of_date]
    if len(valid_quarters) < 4:
        return None
    last_4 = sorted(valid_quarters, key=lambda x: x['date'], reverse=True)[:4]
    return sum(e['eps'] for e in last_4)

def calculate_eps_growth(eps_list, as_of_date):
    """Calcula crecimiento YoY del EPS TTM."""
    eps_ttm_now = calculate_eps_ttm(eps_list, as_of_date)
    if eps_ttm_now is None:
        return None

    one_year_ago = as_of_date - timedelta(days=365)
    eps_ttm_prev = calculate_eps_ttm(eps_list, one_year_ago)
    if eps_ttm_prev is None or eps_ttm_prev <= 0:
        return None

    return (eps_ttm_now - eps_ttm_prev) / abs(eps_ttm_prev) * 100

def get_weekly_prices(conn, friday):
    """Obtiene precios de la semana (lunes a viernes)."""
    monday = friday - timedelta(days=4)
    cur = conn.cursor()

    cur.execute('''
        SELECT symbol, date, close
        FROM fmp_price_history
        WHERE date >= %s AND date <= %s
        ORDER BY symbol, date DESC
    ''', (monday, friday))

    prices = {}
    for row in cur.fetchall():
        symbol, date, close = row
        if symbol not in prices:
            prices[symbol] = float(close)

    cur.close()
    return prices

def get_existing_weeks(conn):
    """Obtiene semanas que ya existen en peg_weekly."""
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT week_ending FROM peg_weekly ORDER BY week_ending')
    weeks = set(row[0] for row in cur.fetchall())
    cur.close()
    return weeks

def main():
    conn = psycopg2.connect(DB_URL)

    print('='*70, flush=True)
    print('PEG_WEEKLY - VERSION OPTIMIZADA', flush=True)
    print('='*70, flush=True)

    cur = conn.cursor()
    cur.execute('SELECT MIN(date), MAX(date) FROM fmp_price_history')
    price_range = cur.fetchone()
    print(f'Precios disponibles: {price_range[0]} a {price_range[1]}', flush=True)

    cur.execute('SELECT MIN(date), MAX(date) FROM fmp_income_statements WHERE eps IS NOT NULL')
    eps_range = cur.fetchone()
    print(f'EPS disponibles: {eps_range[0]} a {eps_range[1]}', flush=True)
    cur.close()

    start_date = max(price_range[0], eps_range[0])
    end_date = min(price_range[1], eps_range[1])
    print(f'Rango a procesar: {start_date} a {end_date}', flush=True)

    existing_weeks = get_existing_weeks(conn)
    print(f'Semanas existentes: {len(existing_weeks)}', flush=True)

    all_fridays = get_fridays(start_date, end_date)
    fridays_to_process = [f for f in all_fridays if f not in existing_weeks]
    print(f'Viernes a procesar: {len(fridays_to_process)}', flush=True)

    if not fridays_to_process:
        print('No hay nuevas semanas para procesar.')
        conn.close()
        return

    # Cargar EPS UNA vez (esto es manejable)
    eps_data = load_eps_data(conn)

    # Procesar semana por semana
    cur = conn.cursor()
    total_inserted = 0

    for i, friday in enumerate(fridays_to_process):
        # Cargar solo precios de esta semana
        prices = get_weekly_prices(conn, friday)

        rows = []
        for symbol in eps_data.keys():
            if symbol not in prices:
                continue

            price = prices[symbol]
            if price is None or price <= 0:
                continue

            eps_ttm = calculate_eps_ttm(eps_data[symbol], friday)
            if eps_ttm is None or eps_ttm <= 0 or eps_ttm > 9e9:
                continue

            pe_ratio = price / eps_ttm
            if pe_ratio <= 0 or pe_ratio > 1000 or pe_ratio > 9e9:
                continue

            eps_growth = calculate_eps_growth(eps_data[symbol], friday)
            if eps_growth is None or eps_growth <= 0 or eps_growth > 1e9:
                continue

            peg_ratio = pe_ratio / eps_growth
            if peg_ratio <= 0 or peg_ratio > 10 or peg_ratio > 1e15:
                continue

            # Validar que close_price no exceda limites
            if price > 1e7:
                continue

            rows.append((symbol, friday, price, eps_ttm, pe_ratio, eps_growth, peg_ratio))

        if rows:
            execute_values(cur, '''
                INSERT INTO peg_weekly (symbol, week_ending, close_price, eps_ttm, pe_ratio, eps_growth_yoy, peg_ratio)
                VALUES %s
                ON CONFLICT (symbol, week_ending) DO NOTHING
            ''', rows)
            conn.commit()
            total_inserted += len(rows)

        if (i + 1) % 50 == 0 or (i + 1) == len(fridays_to_process):
            pct = (i + 1) / len(fridays_to_process) * 100
            print(f'[{pct:5.1f}%] {i+1}/{len(fridays_to_process)} semanas | Simbolos esta semana: {len(rows):,} | Total: {total_inserted:,}', flush=True)

    cur.close()
    conn.close()
    print(f'\nFinalizado. Total insertados: {total_inserted:,}', flush=True)

if __name__ == '__main__':
    main()
