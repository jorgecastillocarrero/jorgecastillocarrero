"""
Descarga precios históricos de FMP para extender market_cap_weekly.
Descarga desde 2015 para tener ~10 años de datos.
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, date
import time

API_KEY = 'PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf'
BASE_URL = 'https://financialmodelingprep.com/api/v3'
START_DATE = '2015-01-01'
END_DATE = '2021-02-06'  # Hasta donde ya tenemos datos

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')

def get_symbols_to_download():
    """Obtener símbolos que necesitamos (los que están en el backtest)."""
    cur = conn.cursor()

    # Símbolos que ya usamos en el backtest (US stocks)
    cur.execute('''
        SELECT DISTINCT symbol FROM market_cap_weekly
        WHERE symbol NOT LIKE '%%.%%'
        AND symbol NOT LIKE '%%-%%'
        AND LENGTH(symbol) <= 5
        ORDER BY symbol
    ''')
    symbols = [row[0] for row in cur.fetchall()]
    cur.close()

    print(f'Símbolos a descargar: {len(symbols)}')
    return symbols

def get_existing_dates():
    """Ver qué fechas ya tenemos por símbolo."""
    cur = conn.cursor()
    cur.execute('''
        SELECT symbol, MIN(date), MAX(date)
        FROM fmp_price_history
        GROUP BY symbol
    ''')
    existing = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    cur.close()
    return existing

async def fetch_historical_prices(session, symbol, semaphore):
    """Descarga precios históricos para un símbolo."""
    url = f'{BASE_URL}/historical-price-full/{symbol}'
    params = {
        'apikey': API_KEY,
        'from': START_DATE,
        'to': END_DATE
    }

    async with semaphore:
        try:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'historical' in data:
                        return symbol, data['historical']
                elif resp.status == 429:
                    await asyncio.sleep(1)
                    return symbol, None
                return symbol, []
        except Exception as e:
            return symbol, None

def save_prices(symbol, prices):
    """Guarda precios en la base de datos."""
    if not prices:
        return 0

    cur = conn.cursor()

    # Preparar datos
    rows = []
    for p in prices:
        try:
            rows.append((
                symbol,
                p['date'],
                float(p.get('open', 0)),
                float(p.get('high', 0)),
                float(p.get('low', 0)),
                float(p.get('close', 0)),
                float(p.get('adjClose', p.get('close', 0))),
                int(p.get('volume', 0)),
                int(p.get('unadjustedVolume', p.get('volume', 0))),
                float(p.get('change', 0)),
                float(p.get('changePercent', 0)),
                float(p.get('vwap', 0)),
                p.get('label', ''),
                float(p.get('changeOverTime', 0))
            ))
        except (ValueError, TypeError):
            continue

    if not rows:
        cur.close()
        return 0

    # Insertar con ON CONFLICT
    execute_values(cur, '''
        INSERT INTO fmp_price_history
        (symbol, date, open, high, low, close, adj_close, volume,
         unadjusted_volume, change, change_percent, vwap, label, change_over_time)
        VALUES %s
        ON CONFLICT (symbol, date) DO NOTHING
    ''', rows)

    inserted = cur.rowcount
    conn.commit()
    cur.close()
    return inserted

async def main():
    symbols = get_symbols_to_download()
    existing = get_existing_dates()

    # Filtrar símbolos que ya tienen datos desde 2015
    symbols_needed = []
    for s in symbols:
        if s not in existing:
            symbols_needed.append(s)
        elif existing[s][0] > date(2015, 1, 1):
            symbols_needed.append(s)

    print(f'Símbolos que necesitan datos históricos: {len(symbols_needed)}')

    if not symbols_needed:
        print('Todos los símbolos ya tienen datos históricos.')
        return

    semaphore = asyncio.Semaphore(5)  # Limitar concurrencia
    total_inserted = 0

    async with aiohttp.ClientSession() as session:
        batch_size = 100
        for i in range(0, len(symbols_needed), batch_size):
            batch = symbols_needed[i:i+batch_size]

            tasks = [fetch_historical_prices(session, s, semaphore) for s in batch]
            results = await asyncio.gather(*tasks)

            for symbol, prices in results:
                if prices:
                    inserted = save_prices(symbol, prices)
                    total_inserted += inserted

            print(f'Progreso: {min(i+batch_size, len(symbols_needed))}/{len(symbols_needed)} | Insertados: {total_inserted:,}')
            await asyncio.sleep(0.5)  # Rate limiting

    print(f'\nTotal insertados: {total_inserted:,}')

if __name__ == '__main__':
    print('='*60)
    print('DESCARGA DE PRECIOS HISTORICOS FMP')
    print(f'Rango: {START_DATE} a {END_DATE}')
    print('='*60)

    asyncio.run(main())
    conn.close()
