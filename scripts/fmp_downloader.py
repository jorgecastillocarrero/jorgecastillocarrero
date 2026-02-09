"""
FMP Complete Data Downloader
Descarga TODOS los datos de Financial Modeling Prep a PostgreSQL local
"""

import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
import time
from datetime import datetime
from typing import List, Dict, Any
import json

# Configuración
API_KEY = "PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf"
DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
BASE_URL = "https://financialmodelingprep.com"

# Rate limiting
MAX_CONCURRENT = 50  # requests simultáneas
REQUESTS_PER_SECOND = 45  # ligeramente bajo el límite

def clean_date(value):
    """Convertir fecha vacía a None para PostgreSQL"""
    if value is None or value == '' or value == 'None':
        return None
    return value

def clean_value(value):
    """Limpiar valores vacíos"""
    if value == '' or value == 'None':
        return None
    return value

class FMPDownloader:
    def __init__(self):
        self.conn = psycopg2.connect(DB_URL)
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.request_times = []
        self.stats = {
            'requests': 0,
            'success': 0,
            'errors': 0,
            'records_inserted': 0
        }

    def create_tables(self):
        """Crear todas las tablas necesarias"""
        cur = self.conn.cursor()

        tables_sql = """
        -- Símbolos/Stocks
        CREATE TABLE IF NOT EXISTS fmp_symbols (
            symbol VARCHAR(20) PRIMARY KEY,
            name VARCHAR(500),
            exchange VARCHAR(50),
            exchange_short VARCHAR(20),
            type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_symbols_exchange ON fmp_symbols(exchange);
        CREATE INDEX IF NOT EXISTS idx_fmp_symbols_type ON fmp_symbols(type);

        -- Perfiles de empresas
        CREATE TABLE IF NOT EXISTS fmp_profiles (
            symbol VARCHAR(20) PRIMARY KEY,
            company_name VARCHAR(500),
            currency VARCHAR(10),
            cik VARCHAR(20),
            isin VARCHAR(20),
            cusip VARCHAR(20),
            exchange VARCHAR(50),
            exchange_short VARCHAR(20),
            industry VARCHAR(200),
            sector VARCHAR(200),
            country VARCHAR(100),
            description TEXT,
            ceo VARCHAR(200),
            employees INTEGER,
            phone VARCHAR(50),
            address VARCHAR(500),
            city VARCHAR(100),
            state VARCHAR(50),
            zip VARCHAR(20),
            website VARCHAR(500),
            ipo_date DATE,
            beta FLOAT,
            vol_avg BIGINT,
            mkt_cap BIGINT,
            last_div FLOAT,
            price FLOAT,
            is_etf BOOLEAN,
            is_actively_trading BOOLEAN,
            is_adr BOOLEAN,
            is_fund BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_profiles_sector ON fmp_profiles(sector);
        CREATE INDEX IF NOT EXISTS idx_fmp_profiles_industry ON fmp_profiles(industry);
        CREATE INDEX IF NOT EXISTS idx_fmp_profiles_country ON fmp_profiles(country);

        -- Precios históricos diarios
        CREATE TABLE IF NOT EXISTS fmp_price_history (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            adj_close FLOAT,
            volume BIGINT,
            unadjusted_volume BIGINT,
            change FLOAT,
            change_percent FLOAT,
            vwap FLOAT,
            UNIQUE(symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_price_symbol ON fmp_price_history(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_price_date ON fmp_price_history(date);
        CREATE INDEX IF NOT EXISTS idx_fmp_price_symbol_date ON fmp_price_history(symbol, date);

        -- Income Statements
        CREATE TABLE IF NOT EXISTS fmp_income_statements (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            period VARCHAR(10),
            fiscal_year INTEGER,
            reported_currency VARCHAR(10),
            revenue BIGINT,
            cost_of_revenue BIGINT,
            gross_profit BIGINT,
            gross_profit_ratio FLOAT,
            rd_expenses BIGINT,
            sga_expenses BIGINT,
            operating_expenses BIGINT,
            operating_income BIGINT,
            operating_income_ratio FLOAT,
            interest_income BIGINT,
            interest_expense BIGINT,
            ebitda BIGINT,
            ebitda_ratio FLOAT,
            income_before_tax BIGINT,
            income_tax_expense BIGINT,
            net_income BIGINT,
            net_income_ratio FLOAT,
            eps FLOAT,
            eps_diluted FLOAT,
            shares_outstanding BIGINT,
            shares_outstanding_diluted BIGINT,
            UNIQUE(symbol, date, period)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_income_symbol ON fmp_income_statements(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_income_date ON fmp_income_statements(date);

        -- Balance Sheets
        CREATE TABLE IF NOT EXISTS fmp_balance_sheets (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            period VARCHAR(10),
            fiscal_year INTEGER,
            reported_currency VARCHAR(10),
            cash_and_equivalents BIGINT,
            short_term_investments BIGINT,
            cash_and_short_term BIGINT,
            net_receivables BIGINT,
            inventory BIGINT,
            other_current_assets BIGINT,
            total_current_assets BIGINT,
            ppe_net BIGINT,
            goodwill BIGINT,
            intangible_assets BIGINT,
            long_term_investments BIGINT,
            other_non_current_assets BIGINT,
            total_non_current_assets BIGINT,
            total_assets BIGINT,
            accounts_payable BIGINT,
            short_term_debt BIGINT,
            deferred_revenue BIGINT,
            other_current_liabilities BIGINT,
            total_current_liabilities BIGINT,
            long_term_debt BIGINT,
            other_non_current_liabilities BIGINT,
            total_non_current_liabilities BIGINT,
            total_liabilities BIGINT,
            common_stock BIGINT,
            retained_earnings BIGINT,
            total_stockholders_equity BIGINT,
            total_equity BIGINT,
            total_liabilities_and_equity BIGINT,
            minority_interest BIGINT,
            total_investments BIGINT,
            total_debt BIGINT,
            net_debt BIGINT,
            UNIQUE(symbol, date, period)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_balance_symbol ON fmp_balance_sheets(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_balance_date ON fmp_balance_sheets(date);

        -- Cash Flow Statements
        CREATE TABLE IF NOT EXISTS fmp_cash_flow (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            period VARCHAR(10),
            fiscal_year INTEGER,
            reported_currency VARCHAR(10),
            net_income BIGINT,
            depreciation BIGINT,
            deferred_income_tax BIGINT,
            stock_based_compensation BIGINT,
            change_in_working_capital BIGINT,
            accounts_receivables BIGINT,
            inventory BIGINT,
            accounts_payables BIGINT,
            other_working_capital BIGINT,
            other_non_cash_items BIGINT,
            operating_cash_flow BIGINT,
            capex BIGINT,
            acquisitions BIGINT,
            purchases_of_investments BIGINT,
            sales_of_investments BIGINT,
            other_investing BIGINT,
            investing_cash_flow BIGINT,
            debt_repayment BIGINT,
            common_stock_issued BIGINT,
            common_stock_repurchased BIGINT,
            dividends_paid BIGINT,
            other_financing BIGINT,
            financing_cash_flow BIGINT,
            net_change_in_cash BIGINT,
            cash_at_beginning BIGINT,
            cash_at_end BIGINT,
            free_cash_flow BIGINT,
            UNIQUE(symbol, date, period)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_cashflow_symbol ON fmp_cash_flow(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_cashflow_date ON fmp_cash_flow(date);

        -- Ratios financieros
        CREATE TABLE IF NOT EXISTS fmp_ratios (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            period VARCHAR(10),
            current_ratio FLOAT,
            quick_ratio FLOAT,
            cash_ratio FLOAT,
            days_of_sales_outstanding FLOAT,
            days_of_inventory_outstanding FLOAT,
            days_of_payables_outstanding FLOAT,
            operating_cycle FLOAT,
            cash_conversion_cycle FLOAT,
            gross_profit_margin FLOAT,
            operating_profit_margin FLOAT,
            pretax_profit_margin FLOAT,
            net_profit_margin FLOAT,
            effective_tax_rate FLOAT,
            return_on_assets FLOAT,
            return_on_equity FLOAT,
            return_on_capital_employed FLOAT,
            debt_ratio FLOAT,
            debt_equity_ratio FLOAT,
            long_term_debt_to_capitalization FLOAT,
            interest_coverage FLOAT,
            cash_flow_to_debt_ratio FLOAT,
            pe_ratio FLOAT,
            price_to_sales_ratio FLOAT,
            price_to_book_ratio FLOAT,
            price_to_free_cash_flow_ratio FLOAT,
            price_earnings_to_growth_ratio FLOAT,
            ev_to_sales FLOAT,
            ev_to_ebitda FLOAT,
            ev_to_operating_cash_flow FLOAT,
            ev_to_free_cash_flow FLOAT,
            dividend_yield FLOAT,
            payout_ratio FLOAT,
            UNIQUE(symbol, date, period)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_ratios_symbol ON fmp_ratios(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_ratios_date ON fmp_ratios(date);

        -- Key Metrics
        CREATE TABLE IF NOT EXISTS fmp_key_metrics (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            period VARCHAR(10),
            revenue_per_share FLOAT,
            net_income_per_share FLOAT,
            operating_cash_flow_per_share FLOAT,
            free_cash_flow_per_share FLOAT,
            cash_per_share FLOAT,
            book_value_per_share FLOAT,
            tangible_book_value_per_share FLOAT,
            shareholders_equity_per_share FLOAT,
            interest_debt_per_share FLOAT,
            market_cap BIGINT,
            enterprise_value BIGINT,
            pe_ratio FLOAT,
            price_to_sales_ratio FLOAT,
            pocf_ratio FLOAT,
            pfcf_ratio FLOAT,
            pb_ratio FLOAT,
            ptb_ratio FLOAT,
            ev_to_sales FLOAT,
            ev_to_ebitda FLOAT,
            ev_to_operating_cash_flow FLOAT,
            ev_to_free_cash_flow FLOAT,
            earnings_yield FLOAT,
            free_cash_flow_yield FLOAT,
            debt_to_equity FLOAT,
            debt_to_assets FLOAT,
            net_debt_to_ebitda FLOAT,
            current_ratio FLOAT,
            interest_coverage FLOAT,
            income_quality FLOAT,
            dividend_yield FLOAT,
            payout_ratio FLOAT,
            sga_to_revenue FLOAT,
            rd_to_revenue FLOAT,
            intangibles_to_total_assets FLOAT,
            capex_to_operating_cash_flow FLOAT,
            capex_to_revenue FLOAT,
            capex_to_depreciation FLOAT,
            stock_based_compensation_to_revenue FLOAT,
            graham_number FLOAT,
            roic FLOAT,
            roe FLOAT,
            roa FLOAT,
            UNIQUE(symbol, date, period)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_metrics_symbol ON fmp_key_metrics(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_metrics_date ON fmp_key_metrics(date);

        -- Dividendos históricos
        CREATE TABLE IF NOT EXISTS fmp_dividends (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            label VARCHAR(50),
            adj_dividend FLOAT,
            dividend FLOAT,
            record_date DATE,
            payment_date DATE,
            declaration_date DATE,
            UNIQUE(symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_dividends_symbol ON fmp_dividends(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_dividends_date ON fmp_dividends(date);

        -- Stock Splits
        CREATE TABLE IF NOT EXISTS fmp_splits (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            label VARCHAR(50),
            numerator FLOAT,
            denominator FLOAT,
            UNIQUE(symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_splits_symbol ON fmp_splits(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_splits_date ON fmp_splits(date);

        -- ETF Holdings
        CREATE TABLE IF NOT EXISTS fmp_etf_holdings (
            id SERIAL PRIMARY KEY,
            etf_symbol VARCHAR(20) NOT NULL,
            holding_symbol VARCHAR(20),
            name VARCHAR(500),
            weight_percentage FLOAT,
            shares BIGINT,
            market_value BIGINT,
            updated_at DATE,
            UNIQUE(etf_symbol, holding_symbol, updated_at)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_etf_holdings_etf ON fmp_etf_holdings(etf_symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_etf_holdings_holding ON fmp_etf_holdings(holding_symbol);

        -- Commodities
        CREATE TABLE IF NOT EXISTS fmp_commodities (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            adj_close FLOAT,
            volume BIGINT,
            UNIQUE(symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_commodities_symbol ON fmp_commodities(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_commodities_date ON fmp_commodities(date);

        -- Forex
        CREATE TABLE IF NOT EXISTS fmp_forex (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            adj_close FLOAT,
            UNIQUE(symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_forex_symbol ON fmp_forex(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_forex_date ON fmp_forex(date);

        -- Crypto
        CREATE TABLE IF NOT EXISTS fmp_crypto (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            adj_close FLOAT,
            volume BIGINT,
            UNIQUE(symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_crypto_symbol ON fmp_crypto(symbol);
        CREATE INDEX IF NOT EXISTS idx_fmp_crypto_date ON fmp_crypto(date);

        -- Download log para tracking
        CREATE TABLE IF NOT EXISTS fmp_download_log (
            id SERIAL PRIMARY KEY,
            table_name VARCHAR(50),
            symbol VARCHAR(20),
            records_inserted INTEGER,
            status VARCHAR(20),
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_fmp_log_table ON fmp_download_log(table_name);
        CREATE INDEX IF NOT EXISTS idx_fmp_log_symbol ON fmp_download_log(symbol);
        """

        cur.execute(tables_sql)
        self.conn.commit()
        cur.close()
        print("[OK] Todas las tablas creadas")

    async def rate_limit(self):
        """Control de rate limiting"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1]
        if len(self.request_times) >= REQUESTS_PER_SECOND:
            await asyncio.sleep(1.0 / REQUESTS_PER_SECOND)
        self.request_times.append(time.time())

    async def fetch(self, url: str) -> Dict:
        """Fetch con rate limiting y reintentos"""
        async with self.semaphore:
            await self.rate_limit()
            self.stats['requests'] += 1

            for attempt in range(3):
                try:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            self.stats['success'] += 1
                            return await response.json()
                        elif response.status == 429:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            text = await response.text()
                            if attempt == 2:
                                self.stats['errors'] += 1
                                return None
                except Exception as e:
                    if attempt == 2:
                        self.stats['errors'] += 1
                        return None
                    await asyncio.sleep(1)
        return None

    def batch_insert(self, table: str, columns: List[str], values: List[tuple]):
        """Insert masivo eficiente"""
        if not values:
            return 0

        cur = self.conn.cursor()
        cols = ', '.join(columns)

        try:
            execute_values(
                cur,
                f"INSERT INTO {table} ({cols}) VALUES %s ON CONFLICT DO NOTHING",
                values,
                page_size=1000
            )
            self.conn.commit()
            inserted = cur.rowcount
            self.stats['records_inserted'] += len(values)
            cur.close()
            return inserted
        except Exception as e:
            self.conn.rollback()
            cur.close()
            print(f"Error inserting into {table}: {e}")
            return 0

    async def download_symbols(self):
        """Descargar lista completa de símbolos"""
        print("\n[LIST] Descargando lista de símbolos...")

        url = f"{BASE_URL}/stable/stock-list?apikey={API_KEY}"
        data = await self.fetch(url)

        if data:
            values = [(
                d.get('symbol'),
                d.get('name'),
                d.get('exchange'),
                d.get('exchangeShortName'),
                d.get('type')
            ) for d in data]

            inserted = self.batch_insert(
                'fmp_symbols',
                ['symbol', 'name', 'exchange', 'exchange_short', 'type'],
                values
            )
            print(f"[OK] {len(data):,} símbolos guardados")
            return [d.get('symbol') for d in data]
        return []

    async def download_profiles(self, symbols: List[str]):
        """Descargar perfiles de todas las empresas"""
        print(f"\n[COMPANY] Descargando perfiles ({len(symbols):,} empresas)...")

        batch_size = 100
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = []

            for symbol in batch:
                url = f"{BASE_URL}/stable/profile?symbol={symbol}&apikey={API_KEY}"
                tasks.append(self.fetch(url))

            results = await asyncio.gather(*tasks)

            values = []
            for result in results:
                if result and len(result) > 0:
                    d = result[0] if isinstance(result, list) else result
                    values.append((
                        d.get('symbol'),
                        clean_value(d.get('companyName')),
                        clean_value(d.get('currency')),
                        clean_value(d.get('cik')),
                        clean_value(d.get('isin')),
                        clean_value(d.get('cusip')),
                        clean_value(d.get('exchange')),
                        clean_value(d.get('exchangeShortName')),
                        clean_value(d.get('industry')),
                        clean_value(d.get('sector')),
                        clean_value(d.get('country')),
                        clean_value(d.get('description')),
                        clean_value(d.get('ceo')),
                        d.get('fullTimeEmployees'),
                        clean_value(d.get('phone')),
                        clean_value(d.get('address')),
                        clean_value(d.get('city')),
                        clean_value(d.get('state')),
                        clean_value(d.get('zip')),
                        clean_value(d.get('website')),
                        clean_date(d.get('ipoDate')),
                        d.get('beta'),
                        d.get('volAvg'),
                        d.get('mktCap'),
                        d.get('lastDiv'),
                        d.get('price'),
                        d.get('isEtf'),
                        d.get('isActivelyTrading'),
                        d.get('isAdr'),
                        d.get('isFund')
                    ))

            if values:
                self.batch_insert(
                    'fmp_profiles',
                    ['symbol', 'company_name', 'currency', 'cik', 'isin', 'cusip',
                     'exchange', 'exchange_short', 'industry', 'sector', 'country',
                     'description', 'ceo', 'employees', 'phone', 'address', 'city',
                     'state', 'zip', 'website', 'ipo_date', 'beta', 'vol_avg',
                     'mkt_cap', 'last_div', 'price', 'is_etf', 'is_actively_trading',
                     'is_adr', 'is_fund'],
                    values
                )
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  Progreso: {progress:,}/{len(symbols):,} ({100*progress/len(symbols):.1f}%) - {total:,} perfiles guardados", end='\r')

        print(f"\n[OK] {total:,} perfiles guardados")

    async def download_price_history(self, symbols: List[str]):
        """Descargar precios históricos de todos los símbolos"""
        print(f"\n[PRICES] Descargando precios históricos ({len(symbols):,} símbolos)...")

        batch_size = 50
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = []

            for symbol in batch:
                url = f"{BASE_URL}/stable/historical-price-eod/full?symbol={symbol}&apikey={API_KEY}"
                tasks.append(self.fetch(url))

            results = await asyncio.gather(*tasks)

            values = []
            for j, result in enumerate(results):
                if result and isinstance(result, list):
                    for d in result:
                        values.append((
                            batch[j],
                            d.get('date'),
                            d.get('open'),
                            d.get('high'),
                            d.get('low'),
                            d.get('close'),
                            d.get('adjClose'),
                            d.get('volume'),
                            d.get('unadjustedVolume'),
                            d.get('change'),
                            d.get('changePercent'),
                            d.get('vwap')
                        ))

            if values:
                self.batch_insert(
                    'fmp_price_history',
                    ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close',
                     'volume', 'unadjusted_volume', 'change', 'change_percent', 'vwap'],
                    values
                )
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  Progreso: {progress:,}/{len(symbols):,} ({100*progress/len(symbols):.1f}%) - {total:,} registros", end='\r')

        print(f"\n[OK] {total:,} registros de precios guardados")

    async def download_financial_statements(self, symbols: List[str]):
        """Descargar estados financieros"""
        print(f"\n[STATS] Descargando estados financieros...")

        # Income Statements
        await self._download_statements(symbols, 'income-statement', 'fmp_income_statements')

        # Balance Sheets
        await self._download_statements(symbols, 'balance-sheet-statement', 'fmp_balance_sheets')

        # Cash Flow
        await self._download_statements(symbols, 'cash-flow-statement', 'fmp_cash_flow')

    async def _download_statements(self, symbols: List[str], endpoint: str, table: str):
        """Helper para descargar statements"""
        print(f"  Descargando {endpoint}...")

        batch_size = 50
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = []

            for symbol in batch:
                url = f"{BASE_URL}/stable/{endpoint}?symbol={symbol}&period=annual&apikey={API_KEY}"
                tasks.append(self.fetch(url))
                url_q = f"{BASE_URL}/stable/{endpoint}?symbol={symbol}&period=quarter&apikey={API_KEY}"
                tasks.append(self.fetch(url_q))

            results = await asyncio.gather(*tasks)

            # Procesar según el tipo de statement
            values = []
            for result in results:
                if result and isinstance(result, list):
                    for d in result:
                        if 'income' in endpoint:
                            values.append(self._parse_income(d))
                        elif 'balance' in endpoint:
                            values.append(self._parse_balance(d))
                        elif 'cash-flow' in endpoint:
                            values.append(self._parse_cashflow(d))

            if values:
                cols = self._get_statement_columns(endpoint)
                self.batch_insert(table, cols, values)
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"    {progress:,}/{len(symbols):,} ({100*progress/len(symbols):.1f}%) - {total:,} registros", end='\r')

        print(f"\n  [OK] {total:,} registros de {endpoint}")

    def _parse_income(self, d):
        return (
            d.get('symbol'), d.get('date'), d.get('period'), d.get('calendarYear'),
            d.get('reportedCurrency'), d.get('revenue'), d.get('costOfRevenue'),
            d.get('grossProfit'), d.get('grossProfitRatio'), d.get('researchAndDevelopmentExpenses'),
            d.get('sellingGeneralAndAdministrativeExpenses'), d.get('operatingExpenses'),
            d.get('operatingIncome'), d.get('operatingIncomeRatio'), d.get('interestIncome'),
            d.get('interestExpense'), d.get('ebitda'), d.get('ebitdaratio'),
            d.get('incomeBeforeTax'), d.get('incomeTaxExpense'), d.get('netIncome'),
            d.get('netIncomeRatio'), d.get('eps'), d.get('epsdiluted'),
            d.get('weightedAverageShsOut'), d.get('weightedAverageShsOutDil')
        )

    def _parse_balance(self, d):
        return (
            d.get('symbol'), d.get('date'), d.get('period'), d.get('calendarYear'),
            d.get('reportedCurrency'), d.get('cashAndCashEquivalents'),
            d.get('shortTermInvestments'), d.get('cashAndShortTermInvestments'),
            d.get('netReceivables'), d.get('inventory'), d.get('otherCurrentAssets'),
            d.get('totalCurrentAssets'), d.get('propertyPlantEquipmentNet'),
            d.get('goodwill'), d.get('intangibleAssets'), d.get('longTermInvestments'),
            d.get('otherNonCurrentAssets'), d.get('totalNonCurrentAssets'),
            d.get('totalAssets'), d.get('accountPayables'), d.get('shortTermDebt'),
            d.get('deferredRevenue'), d.get('otherCurrentLiabilities'),
            d.get('totalCurrentLiabilities'), d.get('longTermDebt'),
            d.get('otherNonCurrentLiabilities'), d.get('totalNonCurrentLiabilities'),
            d.get('totalLiabilities'), d.get('commonStock'), d.get('retainedEarnings'),
            d.get('totalStockholdersEquity'), d.get('totalEquity'),
            d.get('totalLiabilitiesAndStockholdersEquity'), d.get('minorityInterest'),
            d.get('totalInvestments'), d.get('totalDebt'), d.get('netDebt')
        )

    def _parse_cashflow(self, d):
        return (
            d.get('symbol'), d.get('date'), d.get('period'), d.get('calendarYear'),
            d.get('reportedCurrency'), d.get('netIncome'),
            d.get('depreciationAndAmortization'), d.get('deferredIncomeTax'),
            d.get('stockBasedCompensation'), d.get('changeInWorkingCapital'),
            d.get('accountsReceivables'), d.get('inventory'), d.get('accountsPayables'),
            d.get('otherWorkingCapital'), d.get('otherNonCashItems'),
            d.get('netCashProvidedByOperatingActivities'), d.get('capitalExpenditure'),
            d.get('acquisitionsNet'), d.get('purchasesOfInvestments'),
            d.get('salesMaturitiesOfInvestments'), d.get('otherInvestingActivites'),
            d.get('netCashUsedForInvestingActivites'), d.get('debtRepayment'),
            d.get('commonStockIssued'), d.get('commonStockRepurchased'),
            d.get('dividendsPaid'), d.get('otherFinancingActivites'),
            d.get('netCashUsedProvidedByFinancingActivities'),
            d.get('netChangeInCash'), d.get('cashAtBeginningOfPeriod'),
            d.get('cashAtEndOfPeriod'), d.get('freeCashFlow')
        )

    def _get_statement_columns(self, endpoint):
        if 'income' in endpoint:
            return ['symbol', 'date', 'period', 'fiscal_year', 'reported_currency',
                    'revenue', 'cost_of_revenue', 'gross_profit', 'gross_profit_ratio',
                    'rd_expenses', 'sga_expenses', 'operating_expenses', 'operating_income',
                    'operating_income_ratio', 'interest_income', 'interest_expense',
                    'ebitda', 'ebitda_ratio', 'income_before_tax', 'income_tax_expense',
                    'net_income', 'net_income_ratio', 'eps', 'eps_diluted',
                    'shares_outstanding', 'shares_outstanding_diluted']
        elif 'balance' in endpoint:
            return ['symbol', 'date', 'period', 'fiscal_year', 'reported_currency',
                    'cash_and_equivalents', 'short_term_investments', 'cash_and_short_term',
                    'net_receivables', 'inventory', 'other_current_assets',
                    'total_current_assets', 'ppe_net', 'goodwill', 'intangible_assets',
                    'long_term_investments', 'other_non_current_assets',
                    'total_non_current_assets', 'total_assets', 'accounts_payable',
                    'short_term_debt', 'deferred_revenue', 'other_current_liabilities',
                    'total_current_liabilities', 'long_term_debt',
                    'other_non_current_liabilities', 'total_non_current_liabilities',
                    'total_liabilities', 'common_stock', 'retained_earnings',
                    'total_stockholders_equity', 'total_equity',
                    'total_liabilities_and_equity', 'minority_interest',
                    'total_investments', 'total_debt', 'net_debt']
        else:  # cashflow
            return ['symbol', 'date', 'period', 'fiscal_year', 'reported_currency',
                    'net_income', 'depreciation', 'deferred_income_tax',
                    'stock_based_compensation', 'change_in_working_capital',
                    'accounts_receivables', 'inventory', 'accounts_payables',
                    'other_working_capital', 'other_non_cash_items', 'operating_cash_flow',
                    'capex', 'acquisitions', 'purchases_of_investments',
                    'sales_of_investments', 'other_investing', 'investing_cash_flow',
                    'debt_repayment', 'common_stock_issued', 'common_stock_repurchased',
                    'dividends_paid', 'other_financing', 'financing_cash_flow',
                    'net_change_in_cash', 'cash_at_beginning', 'cash_at_end',
                    'free_cash_flow']

    async def download_ratios(self, symbols: List[str]):
        """Descargar ratios financieros"""
        print(f"\n[RATIOS] Descargando ratios financieros...")
        # Similar pattern, implementado de forma concisa
        pass  # Se implementará igual que statements

    async def download_dividends(self, symbols: List[str]):
        """Descargar historial de dividendos"""
        print(f"\n[DIVIDENDS] Descargando dividendos históricos...")

        batch_size = 50
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/historical-price-eod/dividend?symbol={s}&apikey={API_KEY}")
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
                            d.get('label'),
                            d.get('adjDividend'),
                            d.get('dividend'),
                            clean_date(d.get('recordDate')),
                            clean_date(d.get('paymentDate')),
                            clean_date(d.get('declarationDate'))
                        ))

            if values:
                self.batch_insert(
                    'fmp_dividends',
                    ['symbol', 'date', 'label', 'adj_dividend', 'dividend',
                     'record_date', 'payment_date', 'declaration_date'],
                    values
                )
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} - {total:,} dividendos", end='\r')

        print(f"\n[OK] {total:,} dividendos guardados")

    async def download_splits(self, symbols: List[str]):
        """Descargar historial de splits"""
        print(f"\n[SPLITS] Descargando splits históricos...")

        batch_size = 50
        total = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/historical-price-eod/stock-split?symbol={s}&apikey={API_KEY}")
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
                            d.get('label'),
                            d.get('numerator'),
                            d.get('denominator')
                        ))

            if values:
                self.batch_insert(
                    'fmp_splits',
                    ['symbol', 'date', 'label', 'numerator', 'denominator'],
                    values
                )
                total += len(values)

            progress = min(i + batch_size, len(symbols))
            print(f"  {progress:,}/{len(symbols):,} - {total:,} splits", end='\r')

        print(f"\n[OK] {total:,} splits guardados")

    async def download_etf_list_and_holdings(self):
        """Descargar lista de ETFs y sus holdings"""
        print(f"\n[ETF] Descargando ETFs y holdings...")

        # Obtener lista de ETFs
        url = f"{BASE_URL}/stable/etf-list?apikey={API_KEY}"
        etfs = await self.fetch(url)

        if not etfs:
            print("  No se pudieron obtener ETFs")
            return

        etf_symbols = [e.get('symbol') for e in etfs if e.get('symbol')]
        print(f"  {len(etf_symbols):,} ETFs encontrados")

        # Descargar holdings
        batch_size = 20
        total = 0

        for i in range(0, len(etf_symbols), batch_size):
            batch = etf_symbols[i:i+batch_size]
            tasks = [
                self.fetch(f"{BASE_URL}/stable/etf-holdings?symbol={s}&apikey={API_KEY}")
                for s in batch
            ]

            results = await asyncio.gather(*tasks)

            values = []
            for j, result in enumerate(results):
                if result and isinstance(result, list):
                    for d in result:
                        values.append((
                            batch[j],
                            d.get('asset'),
                            clean_value(d.get('name')),
                            d.get('weightPercentage'),
                            d.get('sharesNumber'),
                            d.get('marketValue'),
                            clean_date(d.get('updated'))
                        ))

            if values:
                self.batch_insert(
                    'fmp_etf_holdings',
                    ['etf_symbol', 'holding_symbol', 'name', 'weight_percentage',
                     'shares', 'market_value', 'updated_at'],
                    values
                )
                total += len(values)

            progress = min(i + batch_size, len(etf_symbols))
            print(f"  {progress:,}/{len(etf_symbols):,} ETFs - {total:,} holdings", end='\r')

        print(f"\n[OK] {total:,} ETF holdings guardados")

    async def download_commodities(self):
        """Descargar datos de commodities"""
        print(f"\n[COMMODITIES] Descargando commodities...")

        url = f"{BASE_URL}/stable/commodities-list?apikey={API_KEY}"
        commodities = await self.fetch(url)

        if not commodities:
            print("  No se pudieron obtener commodities")
            return

        symbols = [c.get('symbol') for c in commodities if c.get('symbol')]
        total = 0

        for symbol in symbols:
            url = f"{BASE_URL}/stable/historical-price-eod/full?symbol={symbol}&apikey={API_KEY}"
            data = await self.fetch(url)

            if data and isinstance(data, list):
                values = [(
                    symbol,
                    d.get('date'),
                    d.get('open'),
                    d.get('high'),
                    d.get('low'),
                    d.get('close'),
                    d.get('adjClose'),
                    d.get('volume')
                ) for d in data]

                self.batch_insert(
                    'fmp_commodities',
                    ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'],
                    values
                )
                total += len(values)

        print(f"[OK] {total:,} registros de commodities guardados")

    async def download_forex(self):
        """Descargar datos de forex"""
        print(f"\n[FOREX] Descargando forex...")

        url = f"{BASE_URL}/stable/forex-list?apikey={API_KEY}"
        pairs = await self.fetch(url)

        if not pairs:
            print("  No se pudieron obtener pares forex")
            return

        symbols = [p.get('symbol') for p in pairs if p.get('symbol')]
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
                            d.get('date'),
                            d.get('open'),
                            d.get('high'),
                            d.get('low'),
                            d.get('close'),
                            d.get('adjClose')
                        ))

            if values:
                self.batch_insert(
                    'fmp_forex',
                    ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close'],
                    values
                )
                total += len(values)

        print(f"[OK] {total:,} registros de forex guardados")

    async def download_crypto(self):
        """Descargar datos de crypto"""
        print(f"\n[CRYPTO] Descargando crypto...")

        url = f"{BASE_URL}/stable/crypto-list?apikey={API_KEY}"
        cryptos = await self.fetch(url)

        if not cryptos:
            print("  No se pudieron obtener cryptos")
            return

        symbols = [c.get('symbol') for c in cryptos if c.get('symbol')]
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
                            d.get('date'),
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

        print(f"[OK] {total:,} registros de crypto guardados")

    def print_stats(self):
        """Mostrar estadísticas finales"""
        print("\n" + "="*60)
        print("[STATS] ESTADÍSTICAS FINALES")
        print("="*60)
        print(f"  Requests totales: {self.stats['requests']:,}")
        print(f"  Exitosas: {self.stats['success']:,}")
        print(f"  Errores: {self.stats['errors']:,}")
        print(f"  Registros insertados: {self.stats['records_inserted']:,}")
        print("="*60)

    async def run(self):
        """Ejecutar descarga completa"""
        print("="*60)
        print("[START] FMP COMPLETE DATA DOWNLOADER")
        print("="*60)
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Crear tablas
        self.create_tables()

        # Iniciar sesión HTTP
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session

            # 1. Descargar símbolos
            symbols = await self.download_symbols()

            if not symbols:
                print("Error: No se pudieron obtener símbolos")
                return

            # 2. Filtrar solo stocks activos (opcional, para ir más rápido)
            # symbols = symbols[:1000]  # Descomentar para test

            # 3. Descargar todo en paralelo por categoría
            await self.download_profiles(symbols)
            await self.download_price_history(symbols)
            await self.download_financial_statements(symbols)
            await self.download_dividends(symbols)
            await self.download_splits(symbols)
            await self.download_etf_list_and_holdings()
            await self.download_commodities()
            await self.download_forex()
            await self.download_crypto()

        # Estadísticas finales
        self.print_stats()
        print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.conn.close()


if __name__ == "__main__":
    downloader = FMPDownloader()
    asyncio.run(downloader.run())
