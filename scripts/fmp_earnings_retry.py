"""
FMP - Reintentar descarga de Earnings para simbolos faltantes
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

class FMPEarningsRetry:
    def __init__(self):
        self.conn = psycopg2.connect(DB_URL)
        self.session = None
        self.stats = {'requests': 0, 'success': 0, 'errors': 0, 'records': 0}
        self.semaphore = asyncio.Semaphore(10)  # Menos concurrencia para evitar rate limiting

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
                INSERT INTO fmp_earnings
                (symbol, date, eps_actual, eps_estimated, revenue_actual, revenue_estimated, last_updated)
                VALUES %s ON CONFLICT (symbol, date) DO NOTHING
            """
            execute_values(cur, query, values)
            self.conn.commit()
            self.stats['records'] += len(values)
            cur.close()
        except Exception as e:
            print(f"Error inserting: {e}")
            self.conn.rollback()

    def get_missing_symbols(self):
        """Obtener simbolos que no tienen earnings"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT s.symbol
            FROM fmp_symbols s
            LEFT JOIN fmp_earnings e ON s.symbol = e.symbol
            WHERE e.symbol IS NULL
            ORDER BY s.symbol
        """)
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        return symbols

    async def download_earnings(self, symbols):
        """Descargar earnings para simbolos faltantes"""
        print(f"\n[RETRY] Descargando earnings para {len(symbols):,} simbolos faltantes...")

        batch_size = 10  # Batches mas pequenos
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/earnings?symbol={s}&apikey={API_KEY}")
                for s in batch
            ]

            results = await asyncio.gather(*tasks)

            values = []
            for j, result in enumerate(results):
                if result and isinstance(result, list):
                    for e in result:
                        if e.get('epsActual') is not None or e.get('epsEstimated') is not None:
                            values.append((
                                batch[j],
                                clean_date(e.get('date')),
                                e.get('epsActual'),
                                e.get('epsEstimated'),
                                e.get('revenueActual'),
                                e.get('revenueEstimated'),
                                clean_date(e.get('lastUpdated'))
                            ))

            if values:
                self.batch_insert(values)
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} - {total:,} earnings", end='\r')

            await asyncio.sleep(0.1)  # Mas delay entre batches

        print(f"\n[OK] {total:,} earnings guardados")
        return total

    async def run(self):
        print("="*60)
        print("[START] FMP - RETRY EARNINGS FALTANTES")
        print("="*60)

        symbols = self.get_missing_symbols()
        print(f"[INFO] {len(symbols):,} simbolos sin earnings")

        if not symbols:
            print("[OK] Todos los simbolos ya tienen earnings")
            return

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.download_earnings(symbols)

        print("\n" + "="*60)
        print("[STATS] ESTADISTICAS")
        print("="*60)
        print(f"  Requests totales: {self.stats['requests']:,}")
        print(f"  Exitosas: {self.stats['success']:,}")
        print(f"  Errores: {self.stats['errors']:,}")
        print(f"  Registros insertados: {self.stats['records']:,}")
        print("="*60)

        self.conn.close()

if __name__ == "__main__":
    downloader = FMPEarningsRetry()
    asyncio.run(downloader.run())
