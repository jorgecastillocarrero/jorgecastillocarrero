"""
Descarga precios FMP para:
1. RELLENAR GAP: SPY, TIP, IEF para 2015-01-01 a 2021-02-06
2. ACTUALIZAR: Todos los SP500 + SPY/TIP/IEF desde 2026-02-07 hasta hoy

Usa el endpoint stable de FMP.
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, date
import json
import time

API_KEY = 'PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf'
BASE_URL = 'https://financialmodelingprep.com/stable'

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')

async def fetch_prices(session, symbol, from_date, to_date, semaphore):
    url = f'{BASE_URL}/historical-price-eod/full'
    params = {'symbol': symbol, 'apikey': API_KEY, 'from': from_date, 'to': to_date}
    async with semaphore:
        try:
            async with session.get(url, params=params, timeout=60) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list):
                        return symbol, data
                    elif isinstance(data, dict) and 'historical' in data:
                        return symbol, data['historical']
                elif resp.status == 429:
                    print(f'  RATE LIMIT for {symbol}, waiting...', flush=True)
                    await asyncio.sleep(5)
                    async with session.get(url, params=params, timeout=60) as resp2:
                        if resp2.status == 200:
                            data = await resp2.json()
                            if isinstance(data, list):
                                return symbol, data
                else:
                    print(f'  {symbol}: HTTP {resp.status}', flush=True)
                return symbol, []
        except Exception as e:
            print(f'  ERROR {symbol}: {e}', flush=True)
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
                float(p.get('close', 0)),  # adj_close
                int(p.get('volume', 0)),
                int(p.get('volume', 0)),   # unadjusted_volume
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

async def download_batch(session, symbols, from_date, to_date, semaphore, label):
    total = len(symbols)
    total_inserted = 0
    symbols_with_data = 0
    batch_size = 20

    for i in range(0, total, batch_size):
        batch = symbols[i:i+batch_size]
        tasks = [fetch_prices(session, s, from_date, to_date, semaphore) for s in batch]
        results = await asyncio.gather(*tasks)

        batch_inserted = 0
        for symbol, prices in results:
            if prices:
                inserted = save_prices(symbol, prices)
                batch_inserted += inserted
                total_inserted += inserted
                if inserted > 0:
                    symbols_with_data += 1

        pct = min(100, (i + len(batch)) / total * 100)
        print(f'  [{label}] {pct:5.1f}% | {i+len(batch):,}/{total:,} | '
              f'Batch: {batch_inserted:,} | Total: {total_inserted:,} | '
              f'Sym con datos: {symbols_with_data}', flush=True)

        await asyncio.sleep(0.3)

    return total_inserted, symbols_with_data

async def main():
    t0 = time.time()
    semaphore = asyncio.Semaphore(5)

    # Load SP500 symbols
    with open('data/sp500_constituents.json') as f:
        current_members = json.load(f)
    with open('data/sp500_historical_changes.json') as f:
        all_changes = json.load(f)
    current_set = {d['symbol'] for d in current_members}
    all_sp500 = set(current_set)
    for ch in all_changes:
        if ch.get('date', '') >= '2004-01-01':
            if ch.get('removedTicker'): all_sp500.add(ch['removedTicker'])
            if ch.get('symbol'): all_sp500.add(ch['symbol'])

    # Key ETFs needed for scoring
    key_etfs = ['SPY', 'TIP', 'IEF']

    async with aiohttp.ClientSession() as session:
        # ============================================================
        # PART 1: FILL GAP - SPY, TIP, IEF for 2015-01-01 to 2021-02-06
        # ============================================================
        print('=' * 80, flush=True)
        print('PART 1: RELLENAR GAP (SPY, TIP, IEF) 2015-01-01 a 2021-02-06', flush=True)
        print('=' * 80, flush=True)

        gap_inserted, gap_syms = await download_batch(
            session, key_etfs, '2015-01-01', '2021-02-06', semaphore, 'GAP'
        )
        print(f'\n  Gap: {gap_inserted:,} registros insertados de {gap_syms} simbolos', flush=True)

        # ============================================================
        # PART 2: UPDATE - All SP500 + ETFs from 2026-02-07 to today
        # ============================================================
        print(f'\n{"=" * 80}', flush=True)
        print(f'PART 2: ACTUALIZAR TODOS ({len(all_sp500)} SP500 + ETFs) 2026-02-07 a 2026-02-24', flush=True)
        print(f'{"=" * 80}', flush=True)

        update_symbols = sorted(list(all_sp500 | set(key_etfs)))
        update_inserted, update_syms = await download_batch(
            session, update_symbols, '2026-02-07', '2026-02-24', semaphore, 'UPDATE'
        )
        print(f'\n  Update: {update_inserted:,} registros insertados de {update_syms} simbolos', flush=True)

    # ============================================================
    # VERIFY
    # ============================================================
    print(f'\n{"=" * 80}', flush=True)
    print('VERIFICACION', flush=True)
    print(f'{"=" * 80}', flush=True)

    cur = conn.cursor()
    for sym in key_etfs:
        cur.execute(f'''
            SELECT EXTRACT(YEAR FROM date) as year, COUNT(*) as days
            FROM fmp_price_history
            WHERE symbol = %s AND date >= '2014-01-01'
            GROUP BY 1 ORDER BY 1
        ''', (sym,))
        rows = cur.fetchall()
        print(f'\n  {sym}:')
        for year, days in rows:
            print(f'    {int(year)}: {days} dias')

    cur.execute("SELECT MAX(date) FROM fmp_price_history WHERE symbol = 'SPY'")
    latest = cur.fetchone()[0]
    print(f'\n  SPY ultimo dato: {latest}')

    cur.execute("SELECT MAX(date) FROM fmp_price_history WHERE symbol = 'AAPL'")
    latest = cur.fetchone()[0]
    print(f'  AAPL ultimo dato: {latest}')

    cur.close()

    print(f'\n  Tiempo total: {time.time()-t0:.0f}s')
    print(f'\n{"=" * 80}')
    print('DESCARGA COMPLETADA')
    print(f'{"=" * 80}')

if __name__ == '__main__':
    asyncio.run(main())
    conn.close()
