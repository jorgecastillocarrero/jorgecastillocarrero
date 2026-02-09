"""
FMP - Descarga de Ratios Financieros
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

class FMPRatiosDownloader:
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
                INSERT INTO fmp_ratios
                (symbol, date, period, current_ratio, quick_ratio, cash_ratio,
                 days_of_sales_outstanding, days_of_inventory_outstanding, days_of_payables_outstanding,
                 operating_cycle, cash_conversion_cycle, gross_profit_margin, operating_profit_margin,
                 pretax_profit_margin, net_profit_margin, effective_tax_rate, return_on_assets,
                 return_on_equity, return_on_capital_employed, debt_ratio, debt_equity_ratio,
                 long_term_debt_to_capitalization, interest_coverage, cash_flow_to_debt_ratio,
                 pe_ratio, price_to_sales_ratio, price_to_book_ratio, price_to_free_cash_flow_ratio,
                 price_earnings_to_growth_ratio, ev_to_sales, ev_to_ebitda, ev_to_operating_cash_flow,
                 ev_to_free_cash_flow, dividend_yield, payout_ratio)
                VALUES %s ON CONFLICT (symbol, date, period) DO NOTHING
            """
            execute_values(cur, query, values)
            self.conn.commit()
            self.stats['records'] += len(values)
            cur.close()
        except Exception as e:
            print(f"Error inserting: {e}")
            self.conn.rollback()

    def get_symbols(self):
        cur = self.conn.cursor()
        cur.execute("SELECT symbol FROM fmp_symbols")
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        return symbols

    async def download_ratios(self, symbols):
        """Descargar ratios financieros"""
        print(f"\n[RATIOS] Descargando para {len(symbols):,} simbolos...")

        batch_size = 10
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/ratios?symbol={s}&apikey={API_KEY}")
                for s in batch
            ]

            results = await asyncio.gather(*tasks)

            values = []
            for j, result in enumerate(results):
                if result and isinstance(result, list):
                    for r in result:
                        values.append((
                            batch[j],
                            clean_date(r.get('date')),
                            r.get('period'),
                            r.get('currentRatio'),
                            r.get('quickRatio'),
                            r.get('cashRatio'),
                            r.get('daysOfSalesOutstanding'),
                            r.get('daysOfInventoryOutstanding'),
                            r.get('daysOfPayablesOutstanding'),
                            r.get('operatingCycle'),
                            r.get('cashConversionCycle'),
                            r.get('grossProfitMargin'),
                            r.get('operatingProfitMargin'),
                            r.get('pretaxProfitMargin'),
                            r.get('netProfitMargin'),
                            r.get('effectiveTaxRate'),
                            r.get('returnOnAssets'),
                            r.get('returnOnEquity'),
                            r.get('returnOnCapitalEmployed'),
                            r.get('debtToAssetsRatio'),
                            r.get('debtToEquityRatio'),
                            r.get('longTermDebtToCapitalRatio'),
                            r.get('interestCoverageRatio'),
                            r.get('cashFlowToDebtRatio'),
                            r.get('priceToEarningsRatio'),
                            r.get('priceToSalesRatio'),
                            r.get('priceToBookRatio'),
                            r.get('priceToFreeCashFlowRatio'),
                            r.get('priceToEarningsGrowthRatio'),
                            r.get('evToSales'),
                            r.get('evToEBITDA'),
                            r.get('evToOperatingCashFlow'),
                            r.get('evToFreeCashFlow'),
                            r.get('dividendYield'),
                            r.get('dividendPayoutRatio')
                        ))

            if values:
                self.batch_insert(values)
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} - {total:,} ratios", end='\r')

            await asyncio.sleep(0.1)

        print(f"\n[OK] {total:,} ratios guardados")
        return total

    async def run(self):
        print("="*60)
        print("[START] FMP - DESCARGA DE RATIOS")
        print("="*60)

        symbols = self.get_symbols()
        print(f"[INFO] {len(symbols):,} simbolos en base de datos")

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.download_ratios(symbols)

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
    downloader = FMPRatiosDownloader()
    asyncio.run(downloader.run())
