"""
RSI Builder - Calcula RSI 14 diario para backtesting

Fórmula estándar Wilder:
- RSI = 100 - (100 / (1 + RS))
- RS = Average Gain / Average Loss
- Smoothed averages: (prev_avg * 13 + current) / 14

Usage:
    python -m src.rsi_builder --symbol AAPL
    python -m src.rsi_builder --sp500
    python -m src.rsi_builder --all --limit 100
    python -m src.rsi_builder --stats
"""

import logging
import argparse
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class RSIBuilder:
    """Calcula y almacena RSI 14 diario."""

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
        """Crea tabla rsi_daily si no existe."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rsi_daily (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                close NUMERIC(18,4),
                rsi_14 NUMERIC(8,2),
                rsi_zone VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, date)
            );

            CREATE INDEX IF NOT EXISTS idx_rsi_symbol ON rsi_daily(symbol);
            CREATE INDEX IF NOT EXISTS idx_rsi_date ON rsi_daily(date);
            CREATE INDEX IF NOT EXISTS idx_rsi_value ON rsi_daily(rsi_14);
            CREATE INDEX IF NOT EXISTS idx_rsi_zone ON rsi_daily(rsi_zone);
        """)
        conn.commit()
        cur.close()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI usando método Wilder (smoothed)."""
        delta = prices.diff()

        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Primer promedio: SMA
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()

        # Convertir a arrays para cálculo más rápido
        avg_gain = avg_gain.values
        avg_loss = avg_loss.values
        gains = gains.values
        losses = losses.values

        # Smoothed average (Wilder's method)
        for i in range(period, len(prices)):
            if not np.isnan(avg_gain[i-1]):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period

        # Evitar división por cero
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Manejar casos especiales
        rsi = np.where(avg_loss == 0, 100, rsi)  # Sin pérdidas = RSI 100
        rsi = np.where(avg_gain == 0, 0, rsi)    # Sin ganancias = RSI 0

        return pd.Series(rsi, index=prices.index)

    def get_rsi_zone(self, rsi: float) -> str:
        """Determina zona del RSI."""
        if pd.isna(rsi):
            return None
        if rsi >= 70:
            return 'overbought'
        elif rsi <= 30:
            return 'oversold'
        else:
            return 'neutral'

    def build_symbol(self, symbol: str) -> int:
        """Calcula RSI para un símbolo. Retorna cantidad de registros."""
        conn = self.connect()
        cur = conn.cursor()

        # Obtener precios
        cur.execute("""
            SELECT date, close
            FROM fmp_price_history
            WHERE symbol = %s
            ORDER BY date
        """, (symbol,))
        rows = cur.fetchall()

        if len(rows) < 20:  # Mínimo 20 días para RSI 14
            return 0

        df = pd.DataFrame(rows, columns=['date', 'close'])
        df['close'] = df['close'].astype(float)

        # Calcular RSI
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['rsi_zone'] = df['rsi_14'].apply(self.get_rsi_zone)

        # Filtrar solo registros con RSI válido
        df = df[df['rsi_14'].notna()]

        if df.empty:
            return 0

        # Preparar registros
        records = [
            (symbol, row['date'], float(row['close']),
             float(row['rsi_14']), row['rsi_zone'])
            for _, row in df.iterrows()
        ]

        # Insertar con ON CONFLICT UPDATE
        execute_values(cur, """
            INSERT INTO rsi_daily (symbol, date, close, rsi_14, rsi_zone)
            VALUES %s
            ON CONFLICT (symbol, date)
            DO UPDATE SET
                close = EXCLUDED.close,
                rsi_14 = EXCLUDED.rsi_14,
                rsi_zone = EXCLUDED.rsi_zone
        """, records)

        conn.commit()
        cur.close()
        return len(records)

    def get_sp500_symbols(self) -> List[str]:
        """Obtiene símbolos actuales del S&P 500."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol FROM index_composition_daily
            WHERE index_name = 'SP500'
            AND date = (SELECT MAX(date) FROM index_composition_daily WHERE index_name = 'SP500')
            ORDER BY symbol
        """)
        symbols = [r[0] for r in cur.fetchall()]
        cur.close()
        return symbols

    def get_all_symbols(self) -> List[str]:
        """Obtiene todos los símbolos con precios."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM fmp_price_history ORDER BY symbol")
        symbols = [r[0] for r in cur.fetchall()]
        cur.close()
        return symbols

    def build_sp500(self) -> Dict:
        """Construye RSI para S&P 500."""
        symbols = self.get_sp500_symbols()
        return self._build_symbols(symbols, "S&P 500")

    def build_all(self, limit: Optional[int] = None) -> Dict:
        """Construye RSI para todos los símbolos."""
        symbols = self.get_all_symbols()
        if limit:
            symbols = symbols[:limit]
        return self._build_symbols(symbols, "todos")

    def _build_symbols(self, symbols: List[str], desc: str) -> Dict:
        """Procesa lista de símbolos."""
        logger.info(f"Procesando {len(symbols)} símbolos ({desc})...")

        self.ensure_table()

        total_records = 0
        processed = 0
        failed = 0

        for i, sym in enumerate(symbols):
            try:
                count = self.build_symbol(sym)
                if count > 0:
                    processed += 1
                    total_records += count
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"{sym}: Error - {e}")
                failed += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Progreso: {i+1}/{len(symbols)} - {processed} OK, {total_records:,} registros")

        logger.info(f"Completado: {processed} símbolos, {total_records:,} registros, {failed} fallidos")
        return {'symbols': processed, 'records': total_records, 'failed': failed}

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la tabla."""
        conn = self.connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date)
            FROM rsi_daily
        """)
        row = cur.fetchone()

        # Distribución por zona
        cur.execute("""
            SELECT rsi_zone, COUNT(*)
            FROM rsi_daily
            WHERE rsi_zone IS NOT NULL
            GROUP BY rsi_zone
        """)
        zones = {r[0]: r[1] for r in cur.fetchall()}

        cur.close()
        return {
            'total_records': row[0],
            'symbols': row[1],
            'min_date': row[2],
            'max_date': row[3],
            'zones': zones
        }


def main():
    parser = argparse.ArgumentParser(description='Construye tabla RSI 14 diario')
    parser.add_argument('--symbol', type=str, help='Procesar un símbolo')
    parser.add_argument('--sp500', action='store_true', help='Procesar S&P 500')
    parser.add_argument('--all', action='store_true', help='Procesar todos los símbolos')
    parser.add_argument('--limit', type=int, help='Limitar cantidad de símbolos')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas')

    args = parser.parse_args()
    builder = RSIBuilder()

    try:
        builder.ensure_table()

        if args.stats:
            stats = builder.get_stats()
            print(f"Registros: {stats['total_records']:,}")
            print(f"Símbolos: {stats['symbols']:,}")
            print(f"Rango: {stats['min_date']} - {stats['max_date']}")
            print(f"Zonas: {stats['zones']}")
        elif args.symbol:
            count = builder.build_symbol(args.symbol)
            print(f"{args.symbol}: {count} registros")
        elif args.sp500:
            result = builder.build_sp500()
            print(f"S&P 500: {result['records']:,} registros de {result['symbols']} símbolos")
        elif args.all:
            result = builder.build_all(limit=args.limit)
            print(f"Total: {result['records']:,} registros de {result['symbols']} símbolos")
        else:
            parser.print_help()
    finally:
        builder.close()


if __name__ == '__main__':
    main()
