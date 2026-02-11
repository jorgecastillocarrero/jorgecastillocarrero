"""
Descarga precios históricos de FMP 2009-2014 usando el endpoint stable.
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datetime import date

API_KEY = 'PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf'
BASE_URL = 'https://financialmodelingprep.com/stable'
START_DATE = '2009-01-01'
END_DATE = '2014-12-31'

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')

def get_symbols_to_download():
    cur = conn.cursor()
    # Símbolos que ya tienen datos en 2015+ pero no en 2009-2014
    cur.execute('''
        SELECT DISTINCT symbol FROM fmp_price_history
        WHERE date >= '2015-01-01'
        AND symbol NOT LIKE '%%.%%'
        AND symbol NOT LIKE '%%-%%'
        AND LENGTH(symbol) <= 5
        AND symbol NOT IN (
            SELECT DISTINCT symbol FROM fmp_price_history WHERE date < '2015-01-01'
        )
        ORDER BY symbol
    ''')
    symbols = [row[0] for row in cur.fetchall()]
    cur.close()
    return symbols

async def fetch_prices(session, symbol, semaphore):
    url = f'{BASE_URL}/historical-price-eod/full'
    params = {'symbol': symbol, 'apikey': API_KEY, 'from': START_DATE, 'to': END_DATE}

    async with semaphore:
        try:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list):
                        return symbol, data
                elif resp.status == 429:
                    await asyncio.sleep(2)
                return symbol, []
        except:
            return symbol, []

def save_prices(symbol, prices):
    if not prices:
        return 0
    cur = conn.cursor()
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
                float(p.get('close', 0)),
                int(p.get('volume', 0)),
                int(p.get('volume', 0)),
                float(p.get('change', 0)),
                float(p.get('changePercent', 0)),
                float(p.get('vwap', 0))
            ))
        except:
            continue

    if rows:
        execute_values(cur, '''
            INSERT INTO fmp_price_history
            (symbol, date, open, high, low, close, adj_close, volume,
             unadjusted_volume, change, change_percent, vwap)
            VALUES %s ON CONFLICT (symbol, date) DO NOTHING
        ''', rows)
        conn.commit()
    cur.close()
    return len(rows)

async def main():
    symbols = get_symbols_to_download()
    total = len(symbols)
    print(f'Simbolos a descargar: {total}', flush=True)

    if not symbols:
        print('Nada que descargar.')
        return

    semaphore = asyncio.Semaphore(5)
    total_inserted = 0
    symbols_with_data = 0

    async with aiohttp.ClientSession() as session:
        batch_size = 25
        for i in range(0, total, batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [fetch_prices(session, s, semaphore) for s in batch]
            results = await asyncio.gather(*tasks)

            batch_inserted = 0
            for symbol, prices in results:
                if prices:
                    inserted = save_prices(symbol, prices)
                    batch_inserted += inserted
                    total_inserted += inserted
                    if inserted > 0:
                        symbols_with_data += 1

            pct = (i + len(batch)) / total * 100
            print(f'[{pct:5.1f}%] {i+len(batch):,}/{total:,} | Batch: {batch_inserted:,} | Total: {total_inserted:,} | Simbolos: {symbols_with_data}', flush=True)

            await asyncio.sleep(0.5)

    print(f'\nFinalizado. Total insertados: {total_inserted:,} de {symbols_with_data} simbolos')

if __name__ == '__main__':
    print('='*70, flush=True)
    print(f'DESCARGA PRECIOS HISTORICOS: {START_DATE} a {END_DATE}', flush=True)
    print('='*70, flush=True)
    asyncio.run(main())
    conn.close()
