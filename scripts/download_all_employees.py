"""Download employee data for all tickers from Yahoo Finance."""
import sqlite3
import yfinance as yf
from datetime import date
import time

DB_PATH = r'C:\Users\usuario\financial-data-project\data\financial_data.db'

def main():
    conn = sqlite3.connect(DB_PATH)

    # Get all symbols
    cur = conn.execute('SELECT id, code FROM symbols WHERE is_active = 1 ORDER BY code')
    symbols = cur.fetchall()
    total = len(symbols)
    print(f'Total símbolos: {total}')

    today = date.today().isoformat()
    updated = 0
    errors = 0
    no_data = 0

    for i, (symbol_id, code) in enumerate(symbols):
        try:
            ticker = yf.Ticker(code)
            info = ticker.info
            employees = info.get('fullTimeEmployees')

            if employees:
                conn.execute('''
                    INSERT OR REPLACE INTO trabajadores (symbol_id, fecha, employees)
                    VALUES (?, ?, ?)
                ''', (symbol_id, today, employees))
                updated += 1
                if updated % 100 == 0:
                    conn.commit()
                    print(f'[{i+1}/{total}] {code}: {employees:,} (total: {updated})')
            else:
                no_data += 1

        except Exception as e:
            errors += 1

        # Progress every 500
        if (i + 1) % 500 == 0:
            print(f'Progreso: {i+1}/{total} - Actualizados: {updated}, Sin datos: {no_data}, Errores: {errors}')
            conn.commit()

    conn.commit()
    conn.close()

    print(f'\n=== COMPLETADO ===')
    print(f'Total procesados: {total}')
    print(f'Actualizados: {updated}')
    print(f'Sin datos: {no_data}')
    print(f'Errores: {errors}')

if __name__ == '__main__':
    main()
