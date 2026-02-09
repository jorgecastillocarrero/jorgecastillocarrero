"""
FMP - Descarga de Key Metrics
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

class FMPKeyMetricsDownloader:
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
                INSERT INTO fmp_key_metrics
                (symbol, date, period, revenue_per_share, net_income_per_share,
                 operating_cash_flow_per_share, free_cash_flow_per_share, cash_per_share,
                 book_value_per_share, tangible_book_value_per_share, shareholders_equity_per_share,
                 interest_debt_per_share, market_cap, enterprise_value, pe_ratio,
                 price_to_sales_ratio, pocf_ratio, pfcf_ratio, pb_ratio, ptb_ratio,
                 ev_to_sales, ev_to_ebitda, ev_to_operating_cash_flow, ev_to_free_cash_flow,
                 earnings_yield, free_cash_flow_yield, debt_to_equity, debt_to_assets,
                 net_debt_to_ebitda, current_ratio, interest_coverage, income_quality,
                 dividend_yield, payout_ratio, sga_to_revenue, rd_to_revenue,
                 intangibles_to_total_assets, capex_to_operating_cash_flow, capex_to_revenue,
                 capex_to_depreciation, stock_based_compensation_to_revenue, graham_number,
                 roic, roe, roa)
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

    async def download_key_metrics(self, symbols):
        """Descargar key metrics"""
        print(f"\n[KEY METRICS] Descargando para {len(symbols):,} simbolos...")

        batch_size = 10
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/key-metrics?symbol={s}&apikey={API_KEY}")
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
                            r.get('revenuePerShare'),
                            r.get('netIncomePerShare'),
                            r.get('operatingCashFlowPerShare'),
                            r.get('freeCashFlowPerShare'),
                            r.get('cashPerShare'),
                            r.get('bookValuePerShare'),
                            r.get('tangibleBookValuePerShare'),
                            r.get('shareholdersEquityPerShare'),
                            r.get('interestDebtPerShare'),
                            r.get('marketCap'),
                            r.get('enterpriseValue'),
                            r.get('priceToEarningsRatio'),
                            r.get('priceToSalesRatio'),
                            r.get('priceToOperatingCashFlowRatio'),
                            r.get('priceToFreeCashFlowRatio'),
                            r.get('priceToBookRatio'),
                            r.get('priceTangibleBookRatio'),
                            r.get('evToSales'),
                            r.get('evToEBITDA'),
                            r.get('evToOperatingCashFlow'),
                            r.get('evToFreeCashFlow'),
                            r.get('earningsYield'),
                            r.get('freeCashFlowYield'),
                            r.get('debtToEquityRatio'),
                            r.get('debtToAssetsRatio'),
                            r.get('netDebtToEBITDA'),
                            r.get('currentRatio'),
                            r.get('interestCoverageRatio'),
                            r.get('incomeQuality'),
                            r.get('dividendYield'),
                            r.get('dividendPayoutRatio'),
                            r.get('salesGeneralAndAdministrativeToRevenue'),
                            r.get('researchAndDevelopementToRevenue'),
                            r.get('intangiblesToTotalAssets'),
                            r.get('capexToOperatingCashFlow'),
                            r.get('capexToRevenue'),
                            r.get('capexToDepreciation'),
                            r.get('stockBasedCompensationToRevenue'),
                            r.get('grahamNumber'),
                            r.get('returnOnInvestedCapital'),
                            r.get('returnOnEquity'),
                            r.get('returnOnAssets')
                        ))

            if values:
                self.batch_insert(values)
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} - {total:,} key metrics", end='\r')

            await asyncio.sleep(0.1)

        print(f"\n[OK] {total:,} key metrics guardados")
        return total

    async def run(self):
        print("="*60)
        print("[START] FMP - DESCARGA DE KEY METRICS")
        print("="*60)

        symbols = self.get_symbols()
        print(f"[INFO] {len(symbols):,} simbolos en base de datos")

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.download_key_metrics(symbols)

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
    downloader = FMPKeyMetricsDownloader()
    asyncio.run(downloader.run())
