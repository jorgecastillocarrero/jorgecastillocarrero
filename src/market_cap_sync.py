"""
Market Cap Sync - Sincroniza market cap desde FMP local a Railway

Calcula: market_cap = precio_actual × shares_outstanding

Usage:
    python -m src.market_cap_sync --all
    python -m src.market_cap_sync --symbol AAPL
    python -m src.market_cap_sync --missing  # Solo símbolos sin market cap
"""

import logging
import argparse
import psycopg2
from datetime import date
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conexiones
FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class MarketCapSync:
    """Sincroniza market cap desde FMP a Railway."""

    def __init__(self):
        self.fmp_conn = None
        self.railway_conn = None

    def connect_fmp(self):
        """Conecta a FMP local."""
        if not self.fmp_conn:
            self.fmp_conn = psycopg2.connect(FMP_DATABASE_URL)
        return self.fmp_conn

    def connect_railway(self):
        """Conecta a Railway."""
        if not self.railway_conn:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self.railway_conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        return self.railway_conn

    def close(self):
        """Cierra conexiones."""
        if self.fmp_conn:
            self.fmp_conn.close()
        if self.railway_conn:
            self.railway_conn.close()

    def get_shares_outstanding(self, symbol: str) -> Optional[int]:
        """Obtiene shares outstanding desde FMP."""
        conn = self.connect_fmp()
        cur = conn.cursor()
        cur.execute("""
            SELECT shares_outstanding
            FROM fmp_income_statements
            WHERE symbol = %s AND shares_outstanding IS NOT NULL
            ORDER BY date DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
        return int(row[0]) if row else None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Obtiene último precio desde FMP."""
        conn = self.connect_fmp()
        cur = conn.cursor()
        cur.execute("""
            SELECT close
            FROM fmp_price_history
            WHERE symbol = %s
            ORDER BY date DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
        return float(row[0]) if row else None

    def get_fmp_profile(self, symbol: str) -> Dict:
        """Obtiene perfil desde FMP."""
        conn = self.connect_fmp()
        cur = conn.cursor()
        cur.execute("""
            SELECT sector, industry, country, employees
            FROM fmp_profiles
            WHERE symbol = %s
        """, (symbol,))
        row = cur.fetchone()
        if row:
            return {
                'sector': row[0],
                'industry': row[1],
                'country': row[2],
                'employees': row[3]
            }
        return {}

    def calculate_market_cap(self, symbol: str) -> Optional[int]:
        """Calcula market cap = precio × shares."""
        shares = self.get_shares_outstanding(symbol)
        price = self.get_latest_price(symbol)

        if shares and price:
            return int(price * shares)
        return None

    def get_railway_symbols_missing_mcap(self) -> List[str]:
        """Obtiene símbolos de Railway sin market cap."""
        conn = self.connect_railway()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT s.code
            FROM symbols s
            LEFT JOIN fundamentals f ON s.id = f.symbol_id AND f.market_cap IS NOT NULL AND f.market_cap > 0
            WHERE f.id IS NULL
            AND s.code NOT LIKE '%%.US'
        """)
        return [row[0] for row in cur.fetchall()]

    def get_railway_all_symbols(self) -> List[str]:
        """Obtiene todos los símbolos de Railway."""
        conn = self.connect_railway()
        cur = conn.cursor()
        cur.execute("SELECT code FROM symbols WHERE code NOT LIKE '%%.US'")
        return [row[0] for row in cur.fetchall()]

    def update_railway_fundamentals(self, symbol: str, market_cap: int, profile: Dict) -> bool:
        """Actualiza fundamentals en Railway."""
        conn = self.connect_railway()
        cur = conn.cursor()

        try:
            # Buscar symbol_id
            cur.execute("SELECT id FROM symbols WHERE code = %s", (symbol,))
            row = cur.fetchone()
            if not row:
                logger.warning(f"{symbol}: No existe en Railway")
                return False

            symbol_id = row[0]
            today = date.today()

            # Verificar si existe registro de hoy
            cur.execute("""
                SELECT id FROM fundamentals
                WHERE symbol_id = %s AND data_date = %s
            """, (symbol_id, today))
            existing = cur.fetchone()

            if existing:
                # Actualizar
                cur.execute("""
                    UPDATE fundamentals
                    SET market_cap = %s, sector = %s, industry = %s, updated_at = NOW()
                    WHERE id = %s
                """, (market_cap, profile.get('sector'), profile.get('industry'), existing[0]))
            else:
                # Insertar
                cur.execute("""
                    INSERT INTO fundamentals (symbol_id, data_date, market_cap, sector, industry, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                """, (symbol_id, today, market_cap, profile.get('sector'), profile.get('industry')))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"{symbol}: Error actualizando Railway - {e}")
            conn.rollback()
            return False

    def sync_symbol(self, symbol: str) -> bool:
        """Sincroniza un símbolo."""
        # Calcular market cap
        market_cap = self.calculate_market_cap(symbol)
        if not market_cap:
            # Intentar desde fmp_key_metrics
            conn = self.connect_fmp()
            cur = conn.cursor()
            cur.execute("""
                SELECT market_cap FROM fmp_key_metrics
                WHERE symbol = %s AND market_cap IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """, (symbol,))
            row = cur.fetchone()
            if row:
                market_cap = int(row[0])

        if not market_cap:
            logger.warning(f"{symbol}: No se pudo calcular market cap")
            return False

        # Obtener perfil
        profile = self.get_fmp_profile(symbol)

        # Actualizar Railway
        success = self.update_railway_fundamentals(symbol, market_cap, profile)
        if success:
            mcap_str = f"{market_cap/1e9:.1f}B" if market_cap > 1e9 else f"{market_cap/1e6:.0f}M"
            logger.info(f"{symbol}: {mcap_str} sincronizado")

        return success

    def sync_missing(self) -> Dict:
        """Sincroniza símbolos sin market cap."""
        symbols = self.get_railway_symbols_missing_mcap()
        logger.info(f"Sincronizando {len(symbols)} símbolos sin market cap...")

        success = 0
        failed = 0

        for sym in symbols:
            if self.sync_symbol(sym):
                success += 1
            else:
                failed += 1

        logger.info(f"Completado: {success} OK, {failed} fallidos")
        return {'total': len(symbols), 'success': success, 'failed': failed}

    def sync_all(self, limit: Optional[int] = None) -> Dict:
        """Sincroniza todos los símbolos."""
        symbols = self.get_railway_all_symbols()
        if limit:
            symbols = symbols[:limit]

        logger.info(f"Sincronizando {len(symbols)} símbolos...")

        success = 0
        failed = 0

        for i, sym in enumerate(symbols):
            if self.sync_symbol(sym):
                success += 1
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Progreso: {i+1}/{len(symbols)}")

        logger.info(f"Completado: {success} OK, {failed} fallidos")
        return {'total': len(symbols), 'success': success, 'failed': failed}


def main():
    parser = argparse.ArgumentParser(description='Sincroniza market cap desde FMP a Railway')
    parser.add_argument('--symbol', type=str, help='Sincronizar un símbolo')
    parser.add_argument('--missing', action='store_true', help='Solo símbolos sin market cap')
    parser.add_argument('--all', action='store_true', help='Todos los símbolos')
    parser.add_argument('--limit', type=int, help='Limitar cantidad')

    args = parser.parse_args()
    sync = MarketCapSync()

    try:
        if args.symbol:
            sync.sync_symbol(args.symbol)
        elif args.missing:
            sync.sync_missing()
        elif args.all:
            sync.sync_all(limit=args.limit)
        else:
            parser.print_help()
    finally:
        sync.close()


if __name__ == '__main__':
    main()
