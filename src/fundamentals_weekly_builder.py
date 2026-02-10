"""
Fundamentals Weekly Builder - Construye tablas semanales de fundamentales para backtesting

Crea:
- eps_ttm_weekly: EPS TTM y growth YoY
- revenue_ttm_weekly: Revenue TTM y growth YoY
- peg_weekly: PEG ratio semanal

Usage:
    python -m src.fundamentals_weekly_builder --eps
    python -m src.fundamentals_weekly_builder --revenue
    python -m src.fundamentals_weekly_builder --peg
    python -m src.fundamentals_weekly_builder --all
"""

import logging
import argparse
import psycopg2
from psycopg2.extras import execute_values
from datetime import date, timedelta
from typing import List, Optional
from decimal import Decimal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class FundamentalsWeeklyBuilder:
    """Construye tablas semanales de fundamentales."""

    def __init__(self):
        self.conn = None

    def connect(self):
        if not self.conn:
            self.conn = psycopg2.connect(FMP_DATABASE_URL)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()

    def ensure_tables(self):
        """Crea tablas si no existen."""
        conn = self.connect()
        cur = conn.cursor()

        # EPS TTM Weekly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eps_ttm_weekly (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                week_ending DATE NOT NULL,
                eps_ttm NUMERIC(12,4),
                eps_ttm_prev_year NUMERIC(12,4),
                eps_growth_yoy NUMERIC(10,2),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, week_ending)
            );
            CREATE INDEX IF NOT EXISTS idx_eps_symbol ON eps_ttm_weekly(symbol);
            CREATE INDEX IF NOT EXISTS idx_eps_week ON eps_ttm_weekly(week_ending);
            CREATE INDEX IF NOT EXISTS idx_eps_growth ON eps_ttm_weekly(eps_growth_yoy);
        """)

        # Revenue TTM Weekly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS revenue_ttm_weekly (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                week_ending DATE NOT NULL,
                revenue_ttm NUMERIC(20,2),
                revenue_ttm_prev_year NUMERIC(20,2),
                revenue_growth_yoy NUMERIC(10,2),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, week_ending)
            );
            CREATE INDEX IF NOT EXISTS idx_rev_symbol ON revenue_ttm_weekly(symbol);
            CREATE INDEX IF NOT EXISTS idx_rev_week ON revenue_ttm_weekly(week_ending);
            CREATE INDEX IF NOT EXISTS idx_rev_growth ON revenue_ttm_weekly(revenue_growth_yoy);
        """)

        # PEG Weekly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS peg_weekly (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                week_ending DATE NOT NULL,
                pe_ratio NUMERIC(12,4),
                peg_ratio NUMERIC(12,4),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, week_ending)
            );
            CREATE INDEX IF NOT EXISTS idx_peg_symbol ON peg_weekly(symbol);
            CREATE INDEX IF NOT EXISTS idx_peg_week ON peg_weekly(week_ending);
            CREATE INDEX IF NOT EXISTS idx_peg_ratio ON peg_weekly(peg_ratio);
        """)

        conn.commit()
        cur.close()

    def get_us_symbols(self) -> List[str]:
        """Obtiene símbolos US limpios."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT symbol FROM fmp_price_history
            WHERE symbol NOT LIKE '%%.%%'
            AND symbol NOT LIKE '0P%%'
            AND (LENGTH(symbol) <= 4 OR (LENGTH(symbol) = 5
                 AND symbol NOT LIKE '%%F' AND symbol NOT LIKE '%%X'
                 AND symbol NOT LIKE '%%W' AND symbol NOT LIKE '%%U'))
            ORDER BY symbol
        """)
        symbols = [r[0] for r in cur.fetchall()]
        cur.close()
        return symbols

    def get_weeks_from_prices(self, symbol: str) -> List[date]:
        """Obtiene semanas disponibles desde precios."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT (date_trunc('week', date) + interval '4 days')::date as week_ending
            FROM fmp_price_history
            WHERE symbol = %s
            ORDER BY week_ending
        """, (symbol,))
        weeks = [r[0] for r in cur.fetchall()]
        cur.close()
        return weeks

    def build_eps_ttm(self, symbol: str) -> int:
        """Construye EPS TTM semanal para un símbolo."""
        conn = self.connect()
        cur = conn.cursor()

        # Obtener datos FY ordenados
        cur.execute("""
            SELECT date, eps
            FROM fmp_income_statements
            WHERE symbol = %s AND period = 'FY' AND eps IS NOT NULL
            ORDER BY date
        """, (symbol,))
        fy_data = cur.fetchall()

        if len(fy_data) < 2:
            return 0

        # Obtener semanas desde precios
        weeks = self.get_weeks_from_prices(symbol)
        if not weeks:
            return 0

        records = []
        for week in weeks:
            # Encontrar FY más reciente antes de esta semana
            current_fy = None
            prev_fy = None
            for i, (fy_date, eps) in enumerate(fy_data):
                if fy_date <= week:
                    current_fy = (fy_date, float(eps) if eps else None)
                    if i > 0:
                        prev_fy = (fy_data[i-1][0], float(fy_data[i-1][1]) if fy_data[i-1][1] else None)

            if current_fy and current_fy[1] is not None:
                eps_ttm = current_fy[1]
                eps_prev = prev_fy[1] if prev_fy else None

                growth = None
                if eps_prev and eps_prev > 0:
                    growth = round((eps_ttm - eps_prev) / eps_prev * 100, 2)

                records.append((symbol, week, eps_ttm, eps_prev, growth))

        if records:
            execute_values(cur, """
                INSERT INTO eps_ttm_weekly (symbol, week_ending, eps_ttm, eps_ttm_prev_year, eps_growth_yoy)
                VALUES %s
                ON CONFLICT (symbol, week_ending) DO UPDATE SET
                    eps_ttm = EXCLUDED.eps_ttm,
                    eps_ttm_prev_year = EXCLUDED.eps_ttm_prev_year,
                    eps_growth_yoy = EXCLUDED.eps_growth_yoy
            """, records)
            conn.commit()

        cur.close()
        return len(records)

    def build_revenue_ttm(self, symbol: str) -> int:
        """Construye Revenue TTM semanal para un símbolo."""
        conn = self.connect()
        cur = conn.cursor()

        # Obtener datos FY ordenados
        cur.execute("""
            SELECT date, revenue
            FROM fmp_income_statements
            WHERE symbol = %s AND period = 'FY' AND revenue IS NOT NULL
            ORDER BY date
        """, (symbol,))
        fy_data = cur.fetchall()

        if len(fy_data) < 2:
            return 0

        weeks = self.get_weeks_from_prices(symbol)
        if not weeks:
            return 0

        records = []
        for week in weeks:
            current_fy = None
            prev_fy = None
            for i, (fy_date, revenue) in enumerate(fy_data):
                if fy_date <= week:
                    current_fy = (fy_date, float(revenue) if revenue else None)
                    if i > 0:
                        prev_fy = (fy_data[i-1][0], float(fy_data[i-1][1]) if fy_data[i-1][1] else None)

            if current_fy and current_fy[1] is not None:
                rev_ttm = current_fy[1]
                rev_prev = prev_fy[1] if prev_fy else None

                growth = None
                if rev_prev and rev_prev > 0:
                    growth = round((rev_ttm - rev_prev) / rev_prev * 100, 2)

                records.append((symbol, week, rev_ttm, rev_prev, growth))

        if records:
            execute_values(cur, """
                INSERT INTO revenue_ttm_weekly (symbol, week_ending, revenue_ttm, revenue_ttm_prev_year, revenue_growth_yoy)
                VALUES %s
                ON CONFLICT (symbol, week_ending) DO UPDATE SET
                    revenue_ttm = EXCLUDED.revenue_ttm,
                    revenue_ttm_prev_year = EXCLUDED.revenue_ttm_prev_year,
                    revenue_growth_yoy = EXCLUDED.revenue_growth_yoy
            """, records)
            conn.commit()

        cur.close()
        return len(records)

    def build_peg(self, symbol: str) -> int:
        """Construye PEG ratio semanal para un símbolo."""
        conn = self.connect()
        cur = conn.cursor()

        # Obtener datos de PEG de fmp_ratios
        cur.execute("""
            SELECT date, pe_ratio, price_earnings_to_growth_ratio
            FROM fmp_ratios
            WHERE symbol = %s
            AND price_earnings_to_growth_ratio IS NOT NULL
            ORDER BY date
        """, (symbol,))
        peg_data = cur.fetchall()

        if not peg_data:
            return 0

        weeks = self.get_weeks_from_prices(symbol)
        if not weeks:
            return 0

        records = []
        for week in weeks:
            # Encontrar PEG más reciente antes de esta semana
            current_peg = None
            for peg_date, pe, peg in peg_data:
                if peg_date <= week:
                    current_peg = (float(pe) if pe else None, float(peg) if peg else None)

            if current_peg and current_peg[1] is not None:
                records.append((symbol, week, current_peg[0], current_peg[1]))

        if records:
            execute_values(cur, """
                INSERT INTO peg_weekly (symbol, week_ending, pe_ratio, peg_ratio)
                VALUES %s
                ON CONFLICT (symbol, week_ending) DO UPDATE SET
                    pe_ratio = EXCLUDED.pe_ratio,
                    peg_ratio = EXCLUDED.peg_ratio
            """, records)
            conn.commit()

        cur.close()
        return len(records)

    def build_all_eps(self):
        """Construye EPS TTM para todos los símbolos US."""
        self.ensure_tables()
        symbols = self.get_us_symbols()
        logger.info(f"Procesando EPS TTM para {len(symbols)} símbolos...")

        total = 0
        success = 0
        for i, sym in enumerate(symbols):
            try:
                count = self.build_eps_ttm(sym)
                if count > 0:
                    success += 1
                    total += count
            except Exception as e:
                pass

            if (i + 1) % 500 == 0:
                logger.info(f"Progreso: {i+1}/{len(symbols)} - {success} OK, {total:,} registros")

        logger.info(f"EPS TTM completado: {success} símbolos, {total:,} registros")
        return total

    def build_all_revenue(self):
        """Construye Revenue TTM para todos los símbolos US."""
        self.ensure_tables()
        symbols = self.get_us_symbols()
        logger.info(f"Procesando Revenue TTM para {len(symbols)} símbolos...")

        total = 0
        success = 0
        for i, sym in enumerate(symbols):
            try:
                count = self.build_revenue_ttm(sym)
                if count > 0:
                    success += 1
                    total += count
            except Exception as e:
                pass

            if (i + 1) % 500 == 0:
                logger.info(f"Progreso: {i+1}/{len(symbols)} - {success} OK, {total:,} registros")

        logger.info(f"Revenue TTM completado: {success} símbolos, {total:,} registros")
        return total

    def build_all_peg(self):
        """Construye PEG para todos los símbolos US."""
        self.ensure_tables()
        symbols = self.get_us_symbols()
        logger.info(f"Procesando PEG para {len(symbols)} símbolos...")

        total = 0
        success = 0
        for i, sym in enumerate(symbols):
            try:
                count = self.build_peg(sym)
                if count > 0:
                    success += 1
                    total += count
            except Exception as e:
                pass

            if (i + 1) % 500 == 0:
                logger.info(f"Progreso: {i+1}/{len(symbols)} - {success} OK, {total:,} registros")

        logger.info(f"PEG completado: {success} símbolos, {total:,} registros")
        return total


def main():
    parser = argparse.ArgumentParser(description='Construye tablas semanales de fundamentales')
    parser.add_argument('--eps', action='store_true', help='Construir EPS TTM semanal')
    parser.add_argument('--revenue', action='store_true', help='Construir Revenue TTM semanal')
    parser.add_argument('--peg', action='store_true', help='Construir PEG semanal')
    parser.add_argument('--all', action='store_true', help='Construir todas las tablas')

    args = parser.parse_args()
    builder = FundamentalsWeeklyBuilder()

    try:
        if args.all:
            builder.build_all_eps()
            builder.build_all_revenue()
            builder.build_all_peg()
        elif args.eps:
            builder.build_all_eps()
        elif args.revenue:
            builder.build_all_revenue()
        elif args.peg:
            builder.build_all_peg()
        else:
            parser.print_help()
    finally:
        builder.close()


if __name__ == '__main__':
    main()
