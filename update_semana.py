"""
Actualizar precios FMP + recalcular regimenes para la semana actual
1. Descarga precios SP500 + VIX desde ultima fecha en DB hasta hoy
2. Regenera regimenes_historico.csv, regimenes_hybrid.csv, regimenes_mindd.csv
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, date
import json
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = 'PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf'
BASE_URL = 'https://financialmodelingprep.com/stable'
DB_URL = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'

conn = psycopg2.connect(DB_URL)

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
                    await asyncio.sleep(5)
                    async with session.get(url, params=params, timeout=60) as resp2:
                        if resp2.status == 200:
                            data = await resp2.json()
                            if isinstance(data, list):
                                return symbol, data
                return symbol, []
        except Exception as e:
            print(f'  ERROR {symbol}: {e}', flush=True)
            return symbol, []

def save_prices(symbol, prices, table='fmp_price_history'):
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
                float(p.get('close', 0)),  # adj_close
                int(p.get('volume', 0)), int(p.get('volume', 0)),
                float(p.get('change', 0)), float(p.get('changePercent', 0)),
                float(p.get('vwap', 0))
            ))
        except:
            continue
    if rows:
        execute_values(cur, f'''
            INSERT INTO {table}
            (symbol, date, open, high, low, close, adj_close, volume,
             unadjusted_volume, change, change_percent, vwap)
            VALUES %s ON CONFLICT (symbol, date) DO NOTHING
        ''', rows)
        conn.commit()
    cur.close()
    return len(rows)

def save_vix(prices):
    if not prices:
        return 0
    cur = conn.cursor()
    rows = []
    for p in prices:
        try:
            rows.append((
                '^VIX', p['date'],
                float(p.get('open', 0)), float(p.get('high', 0)),
                float(p.get('low', 0)), float(p.get('close', 0)),
                float(p.get('close', 0)), int(p.get('volume', 0))
            ))
        except:
            continue
    if rows:
        execute_values(cur, '''
            INSERT INTO price_history_vix
            (symbol, date, open, high, low, close, adj_close, volume)
            VALUES %s ON CONFLICT (symbol, date) DO NOTHING
        ''', rows)
        conn.commit()
    cur.close()
    return len(rows)

async def main():
    t0 = time.time()
    today = date.today().strftime('%Y-%m-%d')

    # Find last date in DB
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM fmp_price_history WHERE symbol='SPY'")
    last_date = cur.fetchone()[0]
    cur.execute("SELECT MAX(date) FROM price_history_vix WHERE symbol='^VIX'")
    last_vix = cur.fetchone()[0]
    cur.close()

    from_date = str(last_date) if last_date else '2026-02-27'
    from_vix = str(last_vix) if last_vix else '2026-02-27'

    print(f"Ultima fecha SPY en DB: {last_date}")
    print(f"Ultima fecha VIX en DB: {last_vix}")
    print(f"Descargando hasta: {today}")

    if str(last_date) >= today:
        print("\nDatos ya actualizados! No hay nada que descargar.")
        return

    # Load SP500 symbols
    with open('data/sp500_constituents.json') as f:
        current_members = json.load(f)
    symbols = [d['symbol'] for d in current_members]

    # Add historical members
    with open('data/sp500_historical_changes.json') as f:
        all_changes = json.load(f)
    all_sp500 = set(symbols)
    for ch in all_changes:
        if ch.get('date', '') >= '2004-01-01':
            if ch.get('removedTicker'): all_sp500.add(ch['removedTicker'])
            if ch.get('symbol'): all_sp500.add(ch['symbol'])
    all_symbols = sorted(all_sp500)

    semaphore = asyncio.Semaphore(5)

    async with aiohttp.ClientSession() as session:
        # 1. VIX
        print(f"\n--- VIX ({from_vix} -> {today}) ---")
        # FMP VIX symbol
        _, vix_data = await fetch_prices(session, '^VIX', from_vix, today, semaphore)
        if not vix_data:
            _, vix_data = await fetch_prices(session, 'VIXM', from_vix, today, semaphore)
        if vix_data:
            n = save_vix(vix_data)
            print(f"  VIX: {n} registros insertados")
        else:
            print("  VIX: sin datos nuevos de API, probando CBOE...")
            # Try alternative: download from Yahoo-style endpoint
            url = f'{BASE_URL}/historical-price-eod/full'
            params = {'symbol': '^VIX', 'apikey': API_KEY, 'from': from_vix, 'to': today}
            async with session.get(url, params=params, timeout=60) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"  Response type: {type(data)}, len: {len(data) if isinstance(data, list) else 'dict'}")

        # 2. SP500 stocks
        print(f"\n--- SP500 ({from_date} -> {today}) - {len(all_symbols)} symbols ---")
        total_inserted = 0
        batch_size = 20
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i+batch_size]
            tasks = [fetch_prices(session, s, from_date, today, semaphore) for s in batch]
            results = await asyncio.gather(*tasks)

            batch_ins = 0
            for symbol, prices in results:
                if prices:
                    n = save_prices(symbol, prices)
                    batch_ins += n
                    total_inserted += n

            pct = min(100, (i + len(batch)) / len(all_symbols) * 100)
            print(f'  {pct:5.1f}% | {i+len(batch):,}/{len(all_symbols):,} | Batch: {batch_ins:,} | Total: {total_inserted:,}', flush=True)
            await asyncio.sleep(0.3)

        print(f"\nTotal insertados: {total_inserted:,}")

    # Verify
    cur = conn.cursor()
    cur.execute("SELECT MAX(date), close FROM fmp_price_history WHERE symbol='SPY' GROUP BY date ORDER BY date DESC LIMIT 3")
    for row in cur.fetchall():
        print(f"  SPY {row[0]}: {row[1]}")
    cur.execute("SELECT MAX(date), close FROM price_history_vix WHERE symbol='^VIX' GROUP BY date ORDER BY date DESC LIMIT 3")
    for row in cur.fetchall():
        print(f"  VIX {row[0]}: {row[1]}")
    cur.close()

    elapsed = time.time() - t0
    print(f"\nCompletado en {elapsed:.0f}s")

if __name__ == '__main__':
    asyncio.run(main())
