"""
FMP Intraday 15min Data Downloader
Descarga datos de 15 minutos para S&P 500 (actuales + historicos) + ETFs + VIX
Periodo: Enero 2019 - Presente
"""

import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
import time
import json
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple, Optional

# Configuracion
API_KEY = "PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf"
DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
BASE_URL = "https://financialmodelingprep.com"

# Rate limiting
MAX_CONCURRENT = 40
REQUESTS_PER_SECOND = 40

# Periodo
START_DATE = date(2019, 1, 1)
END_DATE = date.today()

# ETFs y VIX
EXTRA_SYMBOLS = [
    'SPY', 'QQQ', '^NDX',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'XLY', 'XLP', 'XLU', 'XLRE', 'XLC', 'XLB',
    '^VIX', 'VIXY'
]

DATA_DIR = "C:/Users/usuario/financial-data-project/data"


def load_symbols() -> List[Dict]:
    """Cargar todos los simbolos: S&P 500 actual + removidos desde 2019 + ETFs + VIX"""

    with open(f"{DATA_DIR}/sp500_constituents.json") as f:
        current = json.load(f)
    with open(f"{DATA_DIR}/sp500_historical_changes.json") as f:
        changes = json.load(f)

    # Tickers actuales: descargar rango completo
    symbols = []
    current_tickers = set()
    for s in current:
        ticker = s['symbol']
        current_tickers.add(ticker)
        symbols.append({
            'symbol': ticker,
            'from': START_DATE,
            'to': END_DATE,
            'group': 'SP500_CURRENT'
        })

    # Tickers removidos desde 2019: descargar hasta fecha de salida + 1 mes
    changes_2019 = [x for x in changes if x.get('date', '') >= '2019-01-01']
    for change in changes_2019:
        ticker = change.get('removedTicker')
        if ticker and ticker not in current_tickers:
            removed_date = datetime.strptime(change['date'], '%Y-%m-%d').date()
            end = min(removed_date + relativedelta(months=1), END_DATE)
            symbols.append({
                'symbol': ticker,
                'from': START_DATE,
                'to': end,
                'group': 'SP500_REMOVED'
            })
            current_tickers.add(ticker)  # evitar duplicados

    # ETFs y VIX: rango completo
    for ticker in EXTRA_SYMBOLS:
        if ticker not in current_tickers:
            symbols.append({
                'symbol': ticker,
                'from': START_DATE,
                'to': END_DATE,
                'group': 'ETF_INDEX'
            })

    return symbols


def generate_monthly_ranges(from_date: date, to_date: date) -> List[Tuple[str, str]]:
    """Generar rangos mensuales para paginar (max ~800 records por llamada)"""
    ranges = []
    current = from_date.replace(day=1)
    while current <= to_date:
        month_start = max(current, from_date)
        month_end = min(current + relativedelta(months=1, days=-1), to_date)
        ranges.append((month_start.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')))
        current += relativedelta(months=1)
    return ranges


class IntradayDownloader:
    def __init__(self):
        self.conn = psycopg2.connect(DB_URL)
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.request_times = []
        self.stats = {
            'requests': 0,
            'success': 0,
            'errors': 0,
            'records_inserted': 0,
            'symbols_done': 0,
            'symbols_total': 0
        }
        self.start_time = None

    def create_table(self):
        """Crear tabla para datos intraday 15min"""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fmp_price_history_15min (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                datetime TIMESTAMP NOT NULL,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                UNIQUE(symbol, datetime)
            );
            CREATE INDEX IF NOT EXISTS idx_15min_symbol ON fmp_price_history_15min(symbol);
            CREATE INDEX IF NOT EXISTS idx_15min_datetime ON fmp_price_history_15min(datetime);
            CREATE INDEX IF NOT EXISTS idx_15min_symbol_datetime ON fmp_price_history_15min(symbol, datetime);
        """)
        self.conn.commit()
        cur.close()
        print("[OK] Tabla fmp_price_history_15min creada/verificada")

    async def rate_limit(self):
        """Control de rate limiting"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1]
        if len(self.request_times) >= REQUESTS_PER_SECOND:
            await asyncio.sleep(1.0 / REQUESTS_PER_SECOND)
        self.request_times.append(time.time())

    async def fetch(self, url: str) -> Optional[List]:
        """Fetch con rate limiting y reintentos"""
        async with self.semaphore:
            await self.rate_limit()
            self.stats['requests'] += 1

            for attempt in range(3):
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, list):
                                self.stats['success'] += 1
                                return data
                            return None
                        elif response.status == 429:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            if attempt == 2:
                                self.stats['errors'] += 1
                                return None
                except Exception:
                    if attempt == 2:
                        self.stats['errors'] += 1
                        return None
                    await asyncio.sleep(1)
        return None

    def batch_insert(self, values: List[tuple]) -> int:
        """Insert masivo con ON CONFLICT DO NOTHING"""
        if not values:
            return 0

        cur = self.conn.cursor()
        try:
            execute_values(
                cur,
                """INSERT INTO fmp_price_history_15min
                   (symbol, datetime, open, high, low, close, volume)
                   VALUES %s ON CONFLICT (symbol, datetime) DO NOTHING""",
                values,
                page_size=2000
            )
            self.conn.commit()
            inserted = cur.rowcount
            self.stats['records_inserted'] += inserted
            cur.close()
            return inserted
        except Exception as e:
            self.conn.rollback()
            cur.close()
            print(f"\n[ERROR] Insert: {e}")
            return 0

    def get_existing_ranges(self, symbol: str) -> set:
        """Obtener meses ya descargados para un simbolo (para reanudar)"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT DISTINCT TO_CHAR(datetime, 'YYYY-MM')
            FROM fmp_price_history_15min
            WHERE symbol = %s
        """, (symbol,))
        months = {row[0] for row in cur.fetchall()}
        cur.close()
        return months

    def print_progress(self):
        """Mostrar progreso"""
        s = self.stats
        elapsed = time.time() - self.start_time
        rate = s['requests'] / elapsed if elapsed > 0 else 0
        pct = 100 * s['symbols_done'] / s['symbols_total'] if s['symbols_total'] > 0 else 0
        eta_sec = (elapsed / s['symbols_done']) * (s['symbols_total'] - s['symbols_done']) if s['symbols_done'] > 0 else 0
        eta_min = eta_sec / 60
        print(f"  [{s['symbols_done']}/{s['symbols_total']}] {pct:.1f}% | "
              f"{s['records_inserted']:,} records | "
              f"{s['requests']:,} reqs ({rate:.1f}/s) | "
              f"Errors: {s['errors']} | "
              f"ETA: {eta_min:.1f}min", end='\r')

    async def download_symbol(self, sym_info: Dict):
        """Descargar todos los meses de un simbolo"""
        symbol = sym_info['symbol']
        ranges = generate_monthly_ranges(sym_info['from'], sym_info['to'])

        # Verificar meses ya descargados
        existing = self.get_existing_ranges(symbol)
        pending_ranges = [
            (f, t) for f, t in ranges
            if f[:7] not in existing
        ]

        if not pending_ranges:
            self.stats['symbols_done'] += 1
            return

        # Descargar meses pendientes en batches
        batch_size = 6  # 6 meses en paralelo por simbolo
        all_values = []

        for i in range(0, len(pending_ranges), batch_size):
            batch = pending_ranges[i:i+batch_size]
            tasks = []
            for from_date, to_date in batch:
                url = (f"{BASE_URL}/stable/historical-chart/15min"
                       f"?symbol={symbol}&from={from_date}&to={to_date}"
                       f"&apikey={API_KEY}")
                tasks.append(self.fetch(url))

            results = await asyncio.gather(*tasks)

            for result in results:
                if result:
                    for bar in result:
                        all_values.append((
                            symbol,
                            bar.get('date'),
                            bar.get('open'),
                            bar.get('high'),
                            bar.get('low'),
                            bar.get('close'),
                            bar.get('volume')
                        ))

        # Insert all at once per symbol
        if all_values:
            self.batch_insert(all_values)

        self.stats['symbols_done'] += 1
        self.print_progress()

    async def download_all(self, symbols: List[Dict]):
        """Descargar todos los simbolos"""
        self.stats['symbols_total'] = len(symbols)
        self.start_time = time.time()

        print(f"\n[DOWNLOAD] {len(symbols)} simbolos, periodo {START_DATE} - {END_DATE}")

        # Procesar en batches de simbolos
        batch_size = 10  # 10 simbolos en paralelo
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [self.download_symbol(s) for s in batch]
            await asyncio.gather(*tasks)

        elapsed = time.time() - self.start_time
        s = self.stats
        print(f"\n\n[DONE] Completado en {elapsed/60:.1f} minutos")
        print(f"  Simbolos: {s['symbols_done']}")
        print(f"  Registros insertados: {s['records_inserted']:,}")
        print(f"  Requests: {s['requests']:,} (success: {s['success']:,}, errors: {s['errors']})")

    async def run(self):
        """Ejecutar descarga completa"""
        print("=" * 60)
        print("FMP Intraday 15min Downloader")
        print("=" * 60)

        # Crear tabla
        self.create_table()

        # Cargar simbolos
        symbols = load_symbols()
        groups = {}
        for s in symbols:
            groups.setdefault(s['group'], []).append(s)

        print(f"\nSimbolos a descargar:")
        for group, items in groups.items():
            print(f"  {group}: {len(items)}")
        print(f"  TOTAL: {len(symbols)}")

        # Estimar calls
        total_months = sum(
            len(generate_monthly_ranges(s['from'], s['to']))
            for s in symbols
        )
        print(f"\nLlamadas API estimadas: {total_months:,}")

        # Verificar datos existentes
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fmp_price_history_15min")
        existing = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT symbol) FROM fmp_price_history_15min")
        existing_symbols = cur.fetchone()[0]
        cur.close()

        if existing > 0:
            print(f"\nDatos existentes: {existing:,} registros ({existing_symbols} simbolos)")
            print("Se reanudarara la descarga (meses ya descargados se saltan)")

        print(f"\nIniciando descarga...")

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            await self.download_all(symbols)

        # Estadisticas finales
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fmp_price_history_15min")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT symbol) FROM fmp_price_history_15min")
        total_symbols = cur.fetchone()[0]
        cur.execute("""
            SELECT MIN(datetime), MAX(datetime)
            FROM fmp_price_history_15min
        """)
        min_dt, max_dt = cur.fetchone()
        cur.close()

        print(f"\n{'=' * 60}")
        print(f"ESTADISTICAS FINALES")
        print(f"{'=' * 60}")
        print(f"  Total registros: {total:,}")
        print(f"  Simbolos: {total_symbols}")
        print(f"  Rango: {min_dt} - {max_dt}")

        self.conn.close()


async def main():
    downloader = IntradayDownloader()
    await downloader.run()


if __name__ == "__main__":
    asyncio.run(main())
