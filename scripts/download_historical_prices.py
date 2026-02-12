"""
Descarga precios históricos en periodos de 6 años.
Ejecutar después de que termine 2003-2008.
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

API_KEY = 'PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf'
BASE_URL = 'https://financialmodelingprep.com/stable'

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')

def get_symbols_to_download(start_date, end_date):
    """Símbolos que tienen datos después pero no en este periodo."""
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT symbol FROM fmp_price_history
        WHERE date >= %s
        AND symbol NOT LIKE '%%.%%'
        AND symbol NOT LIKE '%%-%%'
        AND LENGTH(symbol) <= 5
        AND symbol NOT IN (
            SELECT DISTINCT symbol FROM fmp_price_history
            WHERE date >= %s AND date <= %s
        )
        ORDER BY symbol
    ''', (end_date, start_date, end_date))
    symbols = [row[0] for row in cur.fetchall()]
    cur.close()
    return symbols

async def fetch_prices(session, symbol, start_date, end_date, semaphore):
    url = f'{BASE_URL}/historical-price-eod/full'
    params = {'symbol': symbol, 'apikey': API_KEY, 'from': start_date, 'to': end_date}

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
                symbol, p['date'],
                float(p.get('open', 0)), float(p.get('high', 0)),
                float(p.get('low', 0)), float(p.get('close', 0)),
                float(p.get('close', 0)), int(p.get('volume', 0)),
                int(p.get('volume', 0)), float(p.get('change', 0)),
                float(p.get('changePercent', 0)), float(p.get('vwap', 0))
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

async def download_period(start_date, end_date, log_file):
    symbols = get_symbols_to_download(start_date, end_date)
    total = len(symbols)

    with open(log_file, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"DESCARGA PRECIOS: {start_date} a {end_date}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Símbolos a descargar: {total}\n")

    print(f"\n{'='*70}")
    print(f"DESCARGA: {start_date} a {end_date}")
    print(f"Símbolos: {total:,}")
    print(f"{'='*70}")

    if not symbols:
        print("Nada que descargar.")
        return

    semaphore = asyncio.Semaphore(5)
    total_inserted = 0
    symbols_with_data = 0

    async with aiohttp.ClientSession() as session:
        batch_size = 25
        for i in range(0, total, batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [fetch_prices(session, s, start_date, end_date, semaphore) for s in batch]
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
            msg = f'[{pct:5.1f}%] {i+len(batch):,}/{total:,} | Batch: {batch_inserted:,} | Total: {total_inserted:,} | Símbolos: {symbols_with_data}'
            print(msg, flush=True)

            with open(log_file, 'a') as f:
                f.write(msg + '\n')

            await asyncio.sleep(0.5)

    print(f'\nFinalizado {start_date}-{end_date}: {total_inserted:,} registros de {symbols_with_data} símbolos')
    return total_inserted

async def main():
    periods = [
        ('1997-01-01', '2002-12-31', 'logs/fmp_prices_1997_2002.log'),
        ('1991-01-01', '1996-12-31', 'logs/fmp_prices_1991_1996.log'),
    ]

    for start, end, log in periods:
        await download_period(start, end, log)

    print("\n" + "="*70)
    print("TODAS LAS DESCARGAS COMPLETADAS")
    print("="*70)
    conn.close()

if __name__ == '__main__':
    asyncio.run(main())
