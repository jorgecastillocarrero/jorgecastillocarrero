"""Verificar estado de earnings"""
import psycopg2

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
cur = conn.cursor()

# Estado final de tablas
tables = ['fmp_symbols', 'fmp_profiles', 'fmp_price_history', 'fmp_income_statements',
          'fmp_balance_sheets', 'fmp_cash_flow', 'fmp_dividends', 'fmp_splits',
          'fmp_commodities', 'fmp_forex', 'fmp_crypto', 'fmp_earnings']

print('='*55)
print('ESTADO FINAL DE LA BASE DE DATOS FMP')
print('='*55)
total = 0
for t in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM {t}')
        count = cur.fetchone()[0]
        total += count
        print(f'{t:25} {count:>15,}')
    except:
        pass
print('-'*55)
print(f'{"TOTAL":25} {total:>15,}')
print('='*55)

# AAPL earnings
print()
print('EJEMPLO EARNINGS AAPL:')
print('-'*55)
cur.execute('''
    SELECT date, eps_actual, eps_estimated
    FROM fmp_earnings
    WHERE symbol = 'AAPL'
    ORDER BY date ASC
    LIMIT 3
''')
print('Mas antiguos:')
for row in cur.fetchall():
    if row[1] and row[2] and row[2] != 0:
        surprise = round(((row[1] - row[2]) / abs(row[2])) * 100, 2)
        surprise_str = f'{surprise}%'
    else:
        surprise_str = 'N/A'
    print(f'  {row[0]} | EPS: {row[1]} | Est: {row[2]} | Surprise: {surprise_str}')

cur.execute('''
    SELECT date, eps_actual, eps_estimated
    FROM fmp_earnings
    WHERE symbol = 'AAPL'
    ORDER BY date DESC
    LIMIT 3
''')
print('Mas recientes:')
for row in cur.fetchall():
    if row[1] and row[2] and row[2] != 0:
        surprise = round(((row[1] - row[2]) / abs(row[2])) * 100, 2)
        surprise_str = f'{surprise}%'
    else:
        surprise_str = 'N/A'
    print(f'  {row[0]} | EPS: {row[1]} | Est: {row[2]} | Surprise: {surprise_str}')

cur.execute('SELECT COUNT(*) FROM fmp_earnings WHERE symbol = %s', ('AAPL',))
print()
print(f'Total trimestres AAPL: {cur.fetchone()[0]}')

cur.execute('SELECT COUNT(DISTINCT symbol) FROM fmp_earnings')
print(f'Total simbolos con earnings: {cur.fetchone()[0]:,}')

cur.close()
conn.close()
