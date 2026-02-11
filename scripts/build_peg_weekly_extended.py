"""
Construye peg_weekly extendido usando:
- Precios de fmp_price_history
- EPS de fmp_income_statements (calculando TTM)
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

def load_eps_data():
    """Carga EPS trimestrales de income statements."""
    print('Cargando EPS de income statements...', flush=True)
    cur = conn.cursor()

    # Obtener EPS por símbolo y fecha (trimestral)
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
    print(f'  Cargados {len(eps_data):,} símbolos con EPS', flush=True)
    return eps_data

def calculate_eps_ttm(eps_list, as_of_date):
    """Calcula EPS TTM (suma de últimos 4 trimestres antes de as_of_date)."""
    # Filtrar trimestres anteriores a la fecha
    valid_quarters = [e for e in eps_list if e['date'] <= as_of_date]
    if len(valid_quarters) < 4:
        return None

    # Tomar últimos 4
    last_4 = sorted(valid_quarters, key=lambda x: x['date'], reverse=True)[:4]
    eps_ttm = sum(e['eps'] for e in last_4)
    return eps_ttm

def calculate_eps_growth(eps_list, as_of_date):
    """Calcula crecimiento YoY del EPS TTM."""
    eps_ttm_now = calculate_eps_ttm(eps_list, as_of_date)
    if eps_ttm_now is None:
        return None

    # EPS TTM de hace 1 año
    one_year_ago = as_of_date - timedelta(days=365)
    eps_ttm_prev = calculate_eps_ttm(eps_list, one_year_ago)
    if eps_ttm_prev is None or eps_ttm_prev <= 0:
        return None

    growth = (eps_ttm_now - eps_ttm_prev) / abs(eps_ttm_prev) * 100
    return growth

def load_price_data(start_date, end_date):
    """Carga precios de cierre del viernes (o último día de la semana)."""
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
    """Obtiene semanas que ya existen en peg_weekly."""
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT week_ending FROM peg_weekly ORDER BY week_ending')
    weeks = set(row[0] for row in cur.fetchall())
    cur.close()
    return weeks

def main():
    print('='*70, flush=True)
    print('CONSTRUCCION PEG_WEEKLY EXTENDIDO', flush=True)
    print('='*70, flush=True)

    # Verificar rango de precios disponibles
    cur = conn.cursor()
    cur.execute('SELECT MIN(date), MAX(date) FROM fmp_price_history')
    price_range = cur.fetchone()
    print(f'Precios disponibles: {price_range[0]} a {price_range[1]}', flush=True)

    # Verificar rango de EPS disponibles
    cur.execute('SELECT MIN(date), MAX(date) FROM fmp_income_statements WHERE eps IS NOT NULL')
    eps_range = cur.fetchone()
    print(f'EPS disponibles: {eps_range[0]} a {eps_range[1]}', flush=True)
    cur.close()

    # Determinar rango a procesar
    start_date = max(price_range[0], eps_range[0])
    end_date = min(price_range[1], eps_range[1])
    print(f'Rango a procesar: {start_date} a {end_date}', flush=True)

    # Obtener semanas existentes
    existing_weeks = get_existing_weeks()
    print(f'Semanas existentes en peg_weekly: {len(existing_weeks)}', flush=True)

    # Generar viernes a procesar
    all_fridays = get_fridays(start_date, end_date)
    fridays_to_process = [f for f in all_fridays if f not in existing_weeks]
    print(f'Viernes a procesar: {len(fridays_to_process)}', flush=True)

    if not fridays_to_process:
        print('No hay nuevas semanas para procesar.')
        return

    # Cargar datos
    eps_data = load_eps_data()
    prices = load_price_data(start_date, end_date)

    # Procesar cada viernes
    cur = conn.cursor()
    total_inserted = 0

    for i, friday in enumerate(fridays_to_process):
        rows = []

        for symbol in eps_data.keys():
            if symbol not in prices:
                continue

            # Obtener precio del viernes
            price = get_friday_price(prices[symbol], friday)
            if price is None or price <= 0:
                continue

            # Calcular EPS TTM
            eps_ttm = calculate_eps_ttm(eps_data[symbol], friday)
            if eps_ttm is None or eps_ttm <= 0:
                continue

            # Calcular PE ratio
            pe_ratio = price / eps_ttm
            if pe_ratio <= 0 or pe_ratio > 1000:
                continue

            # Calcular EPS growth
            eps_growth = calculate_eps_growth(eps_data[symbol], friday)
            if eps_growth is None or eps_growth <= 0:
                continue

            # Calcular PEG ratio
            peg_ratio = pe_ratio / eps_growth
            if peg_ratio <= 0 or peg_ratio > 10:
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

        if (i + 1) % 10 == 0:
            pct = (i + 1) / len(fridays_to_process) * 100
            print(f'[{pct:5.1f}%] {i+1}/{len(fridays_to_process)} semanas | Insertados: {total_inserted:,}', flush=True)

    cur.close()
    print(f'\nFinalizado. Total insertados: {total_inserted:,}', flush=True)

if __name__ == '__main__':
    main()
    conn.close()
