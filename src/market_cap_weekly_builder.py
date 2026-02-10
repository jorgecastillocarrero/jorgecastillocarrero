"""
Market Cap Weekly Builder - Construye tabla de market cap semanal para backtesting

Calcula: market_cap = close_price × shares_outstanding (del último trimestre disponible)

Usage:
    python -m src.market_cap_weekly_builder --all
    python -m src.market_cap_weekly_builder --symbol AAPL
    python -m src.market_cap_weekly_builder --update  # Solo últimas semanas
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


class MarketCapWeeklyBuilder:
    """Construye tabla de market cap semanal desde datos FMP."""

    def __init__(self):
        self.conn = None

    def connect(self):
        if not self.conn:
            self.conn = psycopg2.connect(FMP_DATABASE_URL)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()

    def get_symbols_with_data(self) -> List[str]:
        """Obtiene símbolos que tienen precios y shares_outstanding."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT p.symbol
            FROM fmp_price_history p
            INNER JOIN fmp_income_statements i ON p.symbol = i.symbol
            WHERE i.shares_outstanding IS NOT NULL AND i.shares_outstanding > 0
        """)
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        return symbols

    def get_shares_outstanding_history(self, symbol: str) -> List[Dict]:
        """Obtiene historial de shares_outstanding con fechas de vigencia."""
        conn = self.connect()
        cur = conn.cursor()
        # Ordenar por fecha DESC para tener el más reciente primero
        cur.execute("""
            SELECT date, shares_outstanding
            FROM fmp_income_statements
            WHERE symbol = %s AND shares_outstanding IS NOT NULL AND shares_outstanding > 0
            ORDER BY date DESC
        """, (symbol,))

        result = []
        for row in cur.fetchall():
            result.append({
                'date': row[0],
                'shares': int(row[1])
            })
        cur.close()
        return result

    def get_weekly_prices(self, symbol: str) -> List[Dict]:
        """Obtiene precios semanales (último día de cada semana - viernes o día anterior)."""
        conn = self.connect()
        cur = conn.cursor()

        # Obtener el último precio de cada semana (usando date_trunc)
        cur.execute("""
            SELECT
                (date_trunc('week', date) + interval '4 days')::date as week_ending,
                MAX(date) as actual_date,
                (array_agg(close ORDER BY date DESC))[1] as close_price
            FROM fmp_price_history
            WHERE symbol = %s
            GROUP BY date_trunc('week', date)
            ORDER BY week_ending
        """, (symbol,))

        result = []
        for row in cur.fetchall():
            result.append({
                'week_ending': row[0],
                'actual_date': row[1],
                'close': float(row[2]) if row[2] else None
            })
        cur.close()
        return result

    def find_shares_for_date(self, target_date: date, shares_history: List[Dict]) -> Optional[int]:
        """Encuentra shares_outstanding vigente para una fecha dada.

        Usa el shares_outstanding del trimestre más reciente anterior a la fecha.
        """
        if not shares_history:
            return None

        # shares_history está ordenado DESC por fecha
        for sh in shares_history:
            if sh['date'] <= target_date:
                return sh['shares']

        # Si no hay datos anteriores, usar el más antiguo disponible
        return shares_history[-1]['shares'] if shares_history else None

    def build_symbol(self, symbol: str) -> int:
        """Construye market cap semanal para un símbolo. Retorna cantidad de registros."""
        conn = self.connect()
        cur = conn.cursor()

        # Obtener datos
        shares_history = self.get_shares_outstanding_history(symbol)
        if not shares_history:
            logger.debug(f"{symbol}: Sin shares_outstanding")
            return 0

        weekly_prices = self.get_weekly_prices(symbol)
        if not weekly_prices:
            logger.debug(f"{symbol}: Sin precios")
            return 0

        # Construir registros
        records = []
        for wp in weekly_prices:
            if not wp['close']:
                continue

            shares = self.find_shares_for_date(wp['actual_date'], shares_history)
            if not shares:
                continue

            market_cap = wp['close'] * shares
            records.append((
                symbol,
                wp['week_ending'],
                wp['close'],
                shares,
                market_cap
            ))

        if not records:
            return 0

        # Insertar con ON CONFLICT UPDATE
        execute_values(cur, """
            INSERT INTO market_cap_weekly (symbol, week_ending, close_price, shares_outstanding, market_cap)
            VALUES %s
            ON CONFLICT (symbol, week_ending)
            DO UPDATE SET
                close_price = EXCLUDED.close_price,
                shares_outstanding = EXCLUDED.shares_outstanding,
                market_cap = EXCLUDED.market_cap
        """, records)

        conn.commit()
        cur.close()
        return len(records)

    def build_all(self, limit: Optional[int] = None) -> Dict:
        """Construye market cap semanal para todos los símbolos."""
        symbols = self.get_symbols_with_data()
        if limit:
            symbols = symbols[:limit]

        logger.info(f"Procesando {len(symbols)} símbolos...")

        total_records = 0
        processed = 0
        failed = 0

        for i, sym in enumerate(symbols):
            try:
                count = self.build_symbol(sym)
                total_records += count
                processed += 1
            except Exception as e:
                logger.error(f"{sym}: Error - {e}")
                failed += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Progreso: {i+1}/{len(symbols)} - {total_records:,} registros")

        logger.info(f"Completado: {processed} símbolos, {total_records:,} registros, {failed} errores")
        return {'symbols': processed, 'records': total_records, 'failed': failed}

    def update_recent(self, weeks: int = 4) -> Dict:
        """Actualiza solo las últimas N semanas."""
        conn = self.connect()
        cur = conn.cursor()

        # Eliminar registros de las últimas semanas
        cutoff = date.today() - timedelta(weeks=weeks)
        cur.execute("DELETE FROM market_cap_weekly WHERE week_ending >= %s", (cutoff,))
        deleted = cur.rowcount
        conn.commit()

        logger.info(f"Eliminados {deleted} registros desde {cutoff}")

        # Reconstruir
        return self.build_all()

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la tabla."""
        conn = self.connect()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(week_ending), MAX(week_ending) FROM market_cap_weekly")
        row = cur.fetchone()

        return {
            'total_records': row[0],
            'symbols': row[1],
            'min_date': row[2],
            'max_date': row[3]
        }


def main():
    parser = argparse.ArgumentParser(description='Construye tabla de market cap semanal')
    parser.add_argument('--symbol', type=str, help='Procesar un símbolo')
    parser.add_argument('--all', action='store_true', help='Procesar todos los símbolos')
    parser.add_argument('--update', action='store_true', help='Actualizar últimas 4 semanas')
    parser.add_argument('--limit', type=int, help='Limitar cantidad de símbolos')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas')

    args = parser.parse_args()
    builder = MarketCapWeeklyBuilder()

    try:
        if args.stats:
            stats = builder.get_stats()
            print(f"Registros: {stats['total_records']:,}")
            print(f"Símbolos: {stats['symbols']:,}")
            print(f"Rango: {stats['min_date']} - {stats['max_date']}")
        elif args.symbol:
            count = builder.build_symbol(args.symbol)
            print(f"{args.symbol}: {count} registros")
        elif args.update:
            result = builder.update_recent()
            print(f"Actualizado: {result['records']:,} registros")
        elif args.all:
            result = builder.build_all(limit=args.limit)
            print(f"Total: {result['records']:,} registros de {result['symbols']} símbolos")
        else:
            parser.print_help()
    finally:
        builder.close()


if __name__ == '__main__':
    main()
