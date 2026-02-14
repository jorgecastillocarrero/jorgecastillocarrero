"""
FMP - Descarga de Technical Indicators
Indicadores: SMA, EMA, RSI, MACD, Williams %R, ADX, Standard Deviation
"""
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

API_KEY = "PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf"
BASE_URL = "https://financialmodelingprep.com/stable/technical-indicators"
DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"

# Indicadores a descargar con sus periodos
INDICATORS = {
    'sma': [20, 50, 200],
    'ema': [12, 26, 50],
    'rsi': [14],
    'williams': [14],
    'adx': [14],
    'standarddeviation': [20],
}

def create_table(conn):
    """Crea la tabla si no existe."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fmp_technical_indicators (
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            indicator VARCHAR(30) NOT NULL,
            period INTEGER NOT NULL,
            value DOUBLE PRECISION,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            PRIMARY KEY (symbol, date, indicator, period)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_tech_ind_symbol ON fmp_technical_indicators(symbol)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_tech_ind_date ON fmp_technical_indicators(date)
    """)
    conn.commit()
    cur.close()
    print("[OK] Tabla fmp_technical_indicators creada/verificada")


class FMPTechnicalDownloader:
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
                INSERT INTO fmp_technical_indicators
                (symbol, date, indicator, period, value, open, high, low, close, volume)
                VALUES %s ON CONFLICT (symbol, date, indicator, period) DO NOTHING
            """
            execute_values(cur, query, values, page_size=1000)
            self.conn.commit()
            self.stats['records'] += len(values)
            cur.close()
        except Exception as e:
            print(f"Error inserting: {e}")
            self.conn.rollback()

    def get_symbols(self, limit=None):
        """Obtiene simbolos con suficientes datos de precios."""
        cur = self.conn.cursor()
        query = """
            SELECT symbol FROM fmp_price_history
            GROUP BY symbol
            HAVING COUNT(*) >= 200
            ORDER BY symbol
        """
        if limit:
            query = query.replace("ORDER BY symbol", f"ORDER BY symbol LIMIT {limit}")
        cur.execute(query)
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        return symbols

    def get_processed_symbols(self, indicator, period):
        """Simbolos ya procesados para un indicador/periodo."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT DISTINCT symbol FROM fmp_technical_indicators
            WHERE indicator = %s AND period = %s
        """, (indicator, period))
        processed = set(row[0] for row in cur.fetchall())
        cur.close()
        return processed

    async def download_indicator(self, symbols, indicator, period):
        """Descargar un indicador especifico."""
        # Filtrar simbolos ya procesados
        processed = self.get_processed_symbols(indicator, period)
        pending = [s for s in symbols if s not in processed]

        if not pending:
            print(f"  [{indicator.upper()}-{period}] Ya completado")
            return 0

        print(f"  [{indicator.upper()}-{period}] {len(pending):,} simbolos pendientes...")

        batch_size = 10
        total = 0

        for i in range(0, len(pending), batch_size):
            batch = pending[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/{indicator}?symbol={s}&periodLength={period}&timeframe=1day&apikey={API_KEY}")
                for s in batch
            ]

            results = await asyncio.gather(*tasks)

            values = []
            for j, result in enumerate(results):
                if result and isinstance(result, list):
                    for r in result:
                        date_str = r.get('date', '')[:10] if r.get('date') else None
                        if date_str:
                            values.append((
                                batch[j],
                                date_str,
                                indicator,
                                period,
                                r.get(indicator) or r.get('value'),
                                r.get('open'),
                                r.get('high'),
                                r.get('low'),
                                r.get('close'),
                                r.get('volume')
                            ))

            if values:
                self.batch_insert(values)
                total += len(values)

            progress = min(i + batch_size, len(pending))
            if progress % 100 == 0 or progress == len(pending):
                print(f"    {progress:,}/{len(pending):,} - {total:,} registros", end='\r')

            await asyncio.sleep(0.1)

        print(f"  [{indicator.upper()}-{period}] {total:,} registros guardados")
        return total

    async def run(self, limit=None):
        print("="*60)
        print(f"[START] FMP - TECHNICAL INDICATORS - {datetime.now()}")
        print("="*60)

        create_table(self.conn)

        symbols = self.get_symbols(limit)
        print(f"[INFO] {len(symbols):,} simbolos con +200 dias de datos")

        total_records = 0

        async with aiohttp.ClientSession() as session:
            self.session = session

            for indicator, periods in INDICATORS.items():
                print(f"\n[{indicator.upper()}]")
                for period in periods:
                    records = await self.download_indicator(symbols, indicator, period)
                    total_records += records

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
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    downloader = FMPTechnicalDownloader()
    asyncio.run(downloader.run(limit))
