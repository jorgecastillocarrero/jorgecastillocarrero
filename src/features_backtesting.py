"""
Features Backtesting Builder - Construye features semanales históricos para backtesting

Calcula para cada lunes histórico:
- market_cap: capitalización de mercado
- beat_streak: trimestres consecutivos batiendo EPS estimates
- peg_ratio: Price/Earnings to Growth
- eps_growth_yoy: crecimiento EPS año sobre año
- rev_growth_yoy: crecimiento Revenue año sobre año

Usage:
    python -m src.features_backtesting --build
    python -m src.features_backtesting --stats
    python -m src.features_backtesting --query "2024-01-01"
"""

import logging
import argparse
import psycopg2
from psycopg2.extras import execute_values
from datetime import date, timedelta
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class FeaturesBacktestingBuilder:
    """Construye features históricos para backtesting."""

    def __init__(self):
        self.conn = None

    def connect(self):
        if not self.conn:
            self.conn = psycopg2.connect(FMP_DATABASE_URL)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()

    def ensure_table(self):
        """Crea tabla features_backtesting si no existe."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features_backtesting (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                week_date DATE NOT NULL,
                market_cap NUMERIC(20,2),
                beat_streak INTEGER,
                peg_ratio NUMERIC(10,4),
                eps_growth_yoy NUMERIC(10,2),
                rev_growth_yoy NUMERIC(10,2),
                pe_ratio NUMERIC(10,2),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, week_date)
            );

            CREATE INDEX IF NOT EXISTS idx_fb_symbol ON features_backtesting(symbol);
            CREATE INDEX IF NOT EXISTS idx_fb_week ON features_backtesting(week_date);
            CREATE INDEX IF NOT EXISTS idx_fb_mcap ON features_backtesting(market_cap);
            CREATE INDEX IF NOT EXISTS idx_fb_beat ON features_backtesting(beat_streak);
            CREATE INDEX IF NOT EXISTS idx_fb_peg ON features_backtesting(peg_ratio);
        """)
        conn.commit()
        cur.close()

    def get_mondays(self, start_date: date, end_date: date) -> List[date]:
        """Genera lista de lunes entre dos fechas."""
        mondays = []
        current = start_date
        # Ajustar al primer lunes
        while current.weekday() != 0:
            current += timedelta(days=1)

        while current <= end_date:
            mondays.append(current)
            current += timedelta(days=7)

        return mondays

    def get_clean_us_symbols(self) -> List[str]:
        """Obtiene símbolos US limpios."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT symbol FROM fmp_price_history
            WHERE symbol NOT LIKE '%%.%%'
            AND symbol NOT LIKE '0P%%'
            AND (
                LENGTH(symbol) <= 4
                OR (
                    LENGTH(symbol) = 5
                    AND symbol NOT LIKE '%%F'
                    AND symbol NOT LIKE '%%X'
                    AND symbol NOT LIKE '%%W'
                    AND symbol NOT LIKE '%%U'
                )
            )
            ORDER BY symbol
        """)
        symbols = [r[0] for r in cur.fetchall()]
        cur.close()
        return symbols

    def get_market_cap_for_week(self, symbol: str, week_date: date) -> Optional[float]:
        """Obtiene market cap más cercano a la fecha."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT market_cap FROM market_cap_weekly
            WHERE symbol = %s AND week_ending <= %s
            ORDER BY week_ending DESC LIMIT 1
        """, (symbol, week_date))
        row = cur.fetchone()
        cur.close()
        return float(row[0]) if row else None

    def get_beat_streak_for_date(self, symbol: str, as_of_date: date) -> int:
        """Calcula beat streak hasta una fecha dada."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT eps_actual, eps_estimated
            FROM fmp_earnings
            WHERE symbol = %s
            AND date <= %s
            AND eps_actual IS NOT NULL
            AND eps_estimated IS NOT NULL
            ORDER BY date DESC
            LIMIT 8
        """, (symbol, as_of_date))

        streak = 0
        for row in cur.fetchall():
            if row[0] > row[1]:  # eps_actual > eps_estimated
                streak += 1
            else:
                break

        cur.close()
        return streak

    def get_peg_for_date(self, symbol: str, as_of_date: date) -> Optional[float]:
        """Obtiene PEG ratio más cercano a la fecha."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT price_earnings_to_growth_ratio, pe_ratio
            FROM fmp_ratios
            WHERE symbol = %s AND date <= %s
            AND price_earnings_to_growth_ratio IS NOT NULL
            ORDER BY date DESC LIMIT 1
        """, (symbol, as_of_date))
        row = cur.fetchone()
        cur.close()
        if row:
            return float(row[0]), float(row[1]) if row[1] else None
        return None, None

    def get_growth_for_date(self, symbol: str, as_of_date: date) -> tuple:
        """Calcula EPS y Revenue growth YoY hasta una fecha dada."""
        conn = self.connect()
        cur = conn.cursor()

        # Obtener los dos últimos FY antes de as_of_date
        cur.execute("""
            SELECT date, revenue, eps
            FROM fmp_income_statements
            WHERE symbol = %s AND period = 'FY' AND date <= %s
            ORDER BY date DESC
            LIMIT 2
        """, (symbol, as_of_date))

        rows = cur.fetchall()
        cur.close()

        if len(rows) < 2:
            return None, None

        current = rows[0]
        previous = rows[1]

        eps_growth = None
        rev_growth = None

        if previous[1] and float(previous[1]) > 0 and current[1]:
            rev_growth = (float(current[1]) - float(previous[1])) / float(previous[1]) * 100

        if previous[2] and float(previous[2]) > 0 and current[2]:
            eps_growth = (float(current[2]) - float(previous[2])) / float(previous[2]) * 100

        return eps_growth, rev_growth

    def build_week(self, week_date: date, symbols: List[str]) -> int:
        """Construye features para una semana."""
        conn = self.connect()
        cur = conn.cursor()

        records = []
        for symbol in symbols:
            market_cap = self.get_market_cap_for_week(symbol, week_date)
            if not market_cap or market_cap < 1e9:  # Solo > 1B
                continue

            beat_streak = self.get_beat_streak_for_date(symbol, week_date)
            peg, pe = self.get_peg_for_date(symbol, week_date)
            eps_growth, rev_growth = self.get_growth_for_date(symbol, week_date)

            records.append((
                symbol,
                week_date,
                market_cap,
                beat_streak,
                peg,
                eps_growth,
                rev_growth,
                pe
            ))

        if records:
            execute_values(cur, """
                INSERT INTO features_backtesting
                (symbol, week_date, market_cap, beat_streak, peg_ratio, eps_growth_yoy, rev_growth_yoy, pe_ratio)
                VALUES %s
                ON CONFLICT (symbol, week_date) DO UPDATE SET
                    market_cap = EXCLUDED.market_cap,
                    beat_streak = EXCLUDED.beat_streak,
                    peg_ratio = EXCLUDED.peg_ratio,
                    eps_growth_yoy = EXCLUDED.eps_growth_yoy,
                    rev_growth_yoy = EXCLUDED.rev_growth_yoy,
                    pe_ratio = EXCLUDED.pe_ratio
            """, records)
            conn.commit()

        cur.close()
        return len(records)

    def build_all(self, start_date: Optional[date] = None, end_date: Optional[date] = None):
        """Construye features para todo el rango histórico."""
        self.ensure_table()

        # Obtener rango de market_cap_weekly
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT MIN(week_ending), MAX(week_ending)
            FROM market_cap_weekly
            WHERE symbol NOT LIKE '%%.%%'
        """)
        row = cur.fetchone()

        if not start_date:
            start_date = row[0]
        if not end_date:
            end_date = row[1]

        logger.info(f"Rango: {start_date} - {end_date}")

        # Obtener símbolos
        symbols = self.get_clean_us_symbols()
        logger.info(f"Símbolos US: {len(symbols)}")

        # Generar lunes
        mondays = self.get_mondays(start_date, end_date)
        logger.info(f"Semanas a procesar: {len(mondays)}")

        total_records = 0
        for i, monday in enumerate(mondays):
            count = self.build_week(monday, symbols)
            total_records += count

            if (i + 1) % 10 == 0:
                logger.info(f"Progreso: {i+1}/{len(mondays)} semanas - {total_records:,} registros")

        logger.info(f"Completado: {total_records:,} registros")
        return total_records

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la tabla."""
        conn = self.connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*), COUNT(DISTINCT symbol), COUNT(DISTINCT week_date),
                   MIN(week_date), MAX(week_date)
            FROM features_backtesting
        """)
        row = cur.fetchone()
        cur.close()

        return {
            'total_records': row[0],
            'symbols': row[1],
            'weeks': row[2],
            'min_date': row[3],
            'max_date': row[4]
        }

    def query_week(self, week_date: date, min_mcap: float = 1e9, min_beat: int = 4,
                   max_peg: float = 1.5, min_eps_growth: float = 20, min_rev_growth: float = 12):
        """Consulta acciones que cumplen criterios en una semana."""
        conn = self.connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT symbol, market_cap, beat_streak, peg_ratio, eps_growth_yoy, rev_growth_yoy
            FROM features_backtesting
            WHERE week_date = %s
            AND market_cap >= %s
            AND beat_streak >= %s
            AND peg_ratio > 0 AND peg_ratio <= %s
            AND eps_growth_yoy >= %s
            AND rev_growth_yoy >= %s
            ORDER BY market_cap DESC
        """, (week_date, min_mcap, min_beat, max_peg, min_eps_growth, min_rev_growth))

        results = cur.fetchall()
        cur.close()
        return results


def main():
    parser = argparse.ArgumentParser(description='Construye features para backtesting')
    parser.add_argument('--build', action='store_true', help='Construir features históricos')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas')
    parser.add_argument('--query', type=str, help='Consultar fecha específica (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='Fecha fin (YYYY-MM-DD)')

    args = parser.parse_args()
    builder = FeaturesBacktestingBuilder()

    try:
        if args.stats:
            stats = builder.get_stats()
            print(f"Registros: {stats['total_records']:,}")
            print(f"Símbolos: {stats['symbols']:,}")
            print(f"Semanas: {stats['weeks']}")
            print(f"Rango: {stats['min_date']} - {stats['max_date']}")

        elif args.build:
            start = date.fromisoformat(args.start) if args.start else None
            end = date.fromisoformat(args.end) if args.end else None
            builder.build_all(start, end)

        elif args.query:
            query_date = date.fromisoformat(args.query)
            results = builder.query_week(query_date)
            print(f"Acciones que cumplen criterios en {query_date}: {len(results)}")
            for r in results:
                print(f"  {r[0]:8} | MCap: {r[1]/1e9:>7.1f}B | Beat: {r[2]} | PEG: {r[3]:.2f} | EPS: {r[4]:+.1f}% | Rev: {r[5]:+.1f}%")
        else:
            parser.print_help()
    finally:
        builder.close()


if __name__ == '__main__':
    main()
