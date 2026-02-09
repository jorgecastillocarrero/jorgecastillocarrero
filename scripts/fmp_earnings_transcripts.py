"""
FMP - Descarga de Earnings Call Transcripts
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

API_KEY = "PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf"
BASE_URL = "https://financialmodelingprep.com"
DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"

def clean_date(value):
    if value is None or value == '' or value == 'None':
        return None
    return value

class FMPTranscriptsDownloader:
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
                INSERT INTO fmp_earnings_transcripts
                (symbol, year, quarter, date, content)
                VALUES %s ON CONFLICT (symbol, year, quarter) DO NOTHING
            """
            execute_values(cur, query, values)
            self.conn.commit()
            self.stats['records'] += len(values)
            cur.close()
        except Exception as e:
            print(f"Error inserting: {e}")
            self.conn.rollback()

    def get_symbols_with_earnings(self):
        """Obtener simbolos que tienen datos de earnings"""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM fmp_earnings ORDER BY symbol")
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        return symbols

    async def download_transcripts(self, symbols):
        """Descargar transcripts de earnings calls"""
        print(f"\n[TRANSCRIPTS] Descargando para {len(symbols):,} simbolos...")

        # Años a descargar (últimos 10 años)
        current_year = datetime.now().year
        years = list(range(current_year, current_year - 10, -1))
        quarters = [1, 2, 3, 4]

        batch_size = 10
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]

            # Para cada simbolo, intentar los ultimos años y trimestres
            tasks = []
            task_info = []

            for s in batch:
                for year in years[:3]:  # Solo últimos 3 años para no sobrecargar
                    for q in quarters:
                        url = f"{BASE_URL}/stable/earning-call-transcript?symbol={s}&year={year}&quarter={q}&apikey={API_KEY}"
                        tasks.append(self.fetch(url))
                        task_info.append((s, year, q))

            results = await asyncio.gather(*tasks)

            values = []
            for idx, result in enumerate(results):
                if result and isinstance(result, list) and len(result) > 0:
                    r = result[0]
                    s, year, q = task_info[idx]
                    content = r.get('content', '')
                    if content and len(content) > 100:  # Solo guardar si hay contenido real
                        values.append((
                            s,
                            year,
                            f"Q{q}",
                            clean_date(r.get('date')),
                            content
                        ))

            if values:
                self.batch_insert(values)
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} simbolos - {total:,} transcripts", end='\r')

            await asyncio.sleep(0.1)

        print(f"\n[OK] {total:,} transcripts guardados")
        return total

    async def run(self):
        print("="*60)
        print("[START] FMP - DESCARGA DE EARNINGS TRANSCRIPTS")
        print("="*60)

        symbols = self.get_symbols_with_earnings()
        print(f"[INFO] {len(symbols):,} simbolos con earnings")

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.download_transcripts(symbols)

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
    downloader = FMPTranscriptsDownloader()
    asyncio.run(downloader.run())
