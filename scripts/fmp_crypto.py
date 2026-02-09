"""
FMP - Descarga de datos de Crypto
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values

API_KEY = "PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf"
BASE_URL = "https://financialmodelingprep.com"
DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"

def clean_date(value):
    if value is None or value == '' or value == 'None':
        return None
    return value

class FMPCryptoDownloader:
    def __init__(self):
        self.conn = psycopg2.connect(DB_URL)
        self.session = None
        self.stats = {'requests': 0, 'success': 0, 'errors': 0, 'records': 0}
        self.semaphore = asyncio.Semaphore(30)

    async def fetch(self, url):
        async with self.semaphore:
            try:
                self.stats['requests'] += 1
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        self.stats['success'] += 1
                        return await resp.json()
                    else:
                        self.stats['errors'] += 1
                        return None
            except Exception as e:
                self.stats['errors'] += 1
                return None

    def batch_insert(self, table, columns, values):
        if not values:
            return
        try:
            cur = self.conn.cursor()
            cols = ', '.join(columns)
            query = f"INSERT INTO {table} ({cols}) VALUES %s ON CONFLICT DO NOTHING"
            execute_values(cur, query, values)
            self.conn.commit()
            self.stats['records'] += len(values)
            cur.close()
        except Exception as e:
            print(f"Error inserting into {table}: {e}")
            self.conn.rollback()

    async def download_crypto(self):
        """Descargar historial de precios de crypto"""
        print("[CRYPTO] Descargando lista de cryptocurrencies...")

        # Obtener lista de cryptos
        url = f"{BASE_URL}/stable/cryptocurrency-list?apikey={API_KEY}"
        cryptos = await self.fetch(url)

        if not cryptos:
            print("  No se pudieron obtener cryptos")
            return 0

        symbols = [c.get('symbol') for c in cryptos if c.get('symbol')]
        print(f"  {len(symbols):,} cryptos encontrados")

        total = 0
        batch_size = 20

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/historical-price-eod/full?symbol={s}&apikey={API_KEY}")
                for s in batch
            ]

            results = await asyncio.gather(*tasks)

            values = []
            for j, result in enumerate(results):
                if result and isinstance(result, list):
                    for d in result:
                        values.append((
                            batch[j],
                            clean_date(d.get('date')),
                            d.get('open'),
                            d.get('high'),
                            d.get('low'),
                            d.get('close'),
                            d.get('adjClose'),
                            d.get('volume')
                        ))

            if values:
                self.batch_insert(
                    'fmp_crypto',
                    ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'],
                    values
                )
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} - {total:,} registros crypto", end='\r')

            await asyncio.sleep(0.05)

        print(f"\n[OK] {total:,} registros crypto guardados")
        return total

    async def run(self):
        print("="*60)
        print("[START] FMP - DESCARGA DE CRYPTO")
        print("="*60)

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.download_crypto()

        print("\n" + "="*60)
        print("[STATS] ESTADISTICAS FINALES")
        print("="*60)
        print(f"  Requests totales: {self.stats['requests']:,}")
        print(f"  Exitosas: {self.stats['success']:,}")
        print(f"  Errores: {self.stats['errors']:,}")
        print(f"  Registros insertados: {self.stats['records']:,}")
        print("="*60)

        self.conn.close()

if __name__ == "__main__":
    downloader = FMPCryptoDownloader()
    asyncio.run(downloader.run())
