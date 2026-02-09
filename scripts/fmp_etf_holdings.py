"""
FMP - Descarga de ETF Holdings
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
    # Handle datetime strings like "2026-02-09 08:02:23"
    if isinstance(value, str) and ' ' in value:
        return value.split(' ')[0]
    return value

class FMPETFHoldingsDownloader:
    def __init__(self):
        self.conn = psycopg2.connect(DB_URL)
        self.session = None
        self.stats = {'requests': 0, 'success': 0, 'errors': 0, 'records': 0}
        self.semaphore = asyncio.Semaphore(10)

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

    def batch_insert(self, values):
        if not values:
            return
        try:
            cur = self.conn.cursor()
            query = """
                INSERT INTO fmp_etf_holdings
                (etf_symbol, holding_symbol, name, weight_percentage, shares, market_value, updated_at)
                VALUES %s ON CONFLICT DO NOTHING
            """
            execute_values(cur, query, values)
            self.conn.commit()
            self.stats['records'] += len(values)
            cur.close()
        except Exception as e:
            print(f"Error inserting: {e}")
            self.conn.rollback()

    def get_etf_symbols(self):
        cur = self.conn.cursor()
        cur.execute("SELECT symbol FROM fmp_profiles WHERE is_etf = true")
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        return symbols

    async def download_etf_holdings(self, symbols):
        """Descargar holdings de ETFs"""
        print(f"\n[ETF HOLDINGS] Descargando para {len(symbols):,} ETFs...")

        batch_size = 10
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/etf/holdings?symbol={s}&apikey={API_KEY}")
                for s in batch
            ]

            results = await asyncio.gather(*tasks)

            values = []
            for j, result in enumerate(results):
                if result and isinstance(result, list):
                    for h in result:
                        values.append((
                            batch[j],
                            h.get('asset'),
                            h.get('name'),
                            h.get('weightPercentage'),
                            h.get('sharesNumber'),
                            h.get('marketValue'),
                            clean_date(h.get('updatedAt'))
                        ))

            if values:
                self.batch_insert(values)
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} - {total:,} holdings", end='\r')

            await asyncio.sleep(0.1)

        print(f"\n[OK] {total:,} holdings guardados")
        return total

    async def run(self):
        print("="*60)
        print("[START] FMP - DESCARGA DE ETF HOLDINGS")
        print("="*60)

        symbols = self.get_etf_symbols()
        print(f"[INFO] {len(symbols):,} ETFs en base de datos")

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.download_etf_holdings(symbols)

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
    downloader = FMPETFHoldingsDownloader()
    asyncio.run(downloader.run())
