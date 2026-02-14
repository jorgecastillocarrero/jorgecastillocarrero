"""
Calculo de momentum features con MULTIPROCESSING.
Usa multiples workers para procesar simbolos en paralelo.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
import sys

DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
RISK_FREE_RATE = 0.05
NUM_WORKERS = min(8, cpu_count())  # Maximo 8 workers

def get_connection():
    return psycopg2.connect(DB_URL)

def get_market_returns_cached():
    """Carga returns del S&P 500 (se llama una vez)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT date, close FROM price_history_index WHERE symbol = '^GSPC' ORDER BY date")
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=['date', 'close'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df['close'].pct_change()

def get_missing_symbols():
    """Simbolos que faltan por procesar."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        WITH symbols_with_data AS (
            SELECT symbol FROM fmp_price_history
            GROUP BY symbol HAVING COUNT(*) >= 300
        )
        SELECT s.symbol FROM symbols_with_data s
        WHERE EXISTS (SELECT 1 FROM fmp_income_statements i WHERE i.symbol = s.symbol)
        AND NOT EXISTS (SELECT 1 FROM features_momentum f WHERE f.symbol = s.symbol)
        ORDER BY s.symbol
    """)
    symbols = [row[0] for row in cur.fetchall()]
    conn.close()
    return symbols

def calculate_features(close, returns, market_returns):
    """Calcula todas las features."""
    f = pd.DataFrame(index=close.index)

    # Returns
    f['ret_1d'] = close.pct_change(1)
    f['ret_5d'] = close.pct_change(5)
    f['ret_20d'] = close.pct_change(20)
    f['ret_60d'] = close.pct_change(60)
    f['ret_252d'] = close.pct_change(252)

    # Forward returns
    f['ret_1d_fwd'] = close.pct_change(1).shift(-1)
    f['ret_5d_fwd'] = close.pct_change(5).shift(-5)
    f['ret_20d_fwd'] = close.pct_change(20).shift(-20)

    # Volatility
    f['vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
    f['vol_60d'] = returns.rolling(60).std() * np.sqrt(252)

    # Sharpe
    rf = RISK_FREE_RATE / 252
    f['sharpe_20d'] = (returns.rolling(20).mean() - rf) / returns.rolling(20).std() * np.sqrt(252)
    f['sharpe_60d'] = (returns.rolling(60).mean() - rf) / returns.rolling(60).std() * np.sqrt(252)

    # Sortino
    neg_ret = returns.where(returns < 0, 0)
    f['sortino_20d'] = (returns.rolling(20).mean() - rf) * np.sqrt(252) / (neg_ret.rolling(20).std() * np.sqrt(252))
    f['sortino_60d'] = (returns.rolling(60).mean() - rf) * np.sqrt(252) / (neg_ret.rolling(60).std() * np.sqrt(252))

    # Max Drawdown
    f['max_dd_20d'] = (close - close.rolling(20).max()) / close.rolling(20).max()
    f['max_dd_60d'] = (close - close.rolling(60).max()) / close.rolling(60).max()

    # Momentum score
    f['momentum_score'] = (
        (f['ret_20d'] / f['vol_20d'].replace(0, np.nan)) * 0.4 +
        (f['ret_60d'] / f['vol_20d'].replace(0, np.nan)) * 0.3 +
        (f['ret_252d'] / f['vol_20d'].replace(0, np.nan)) * 0.3
    )

    # Betas vs S&P 500
    aligned = pd.DataFrame({'stock': returns, 'market': market_returns}).dropna()

    if len(aligned) >= 252:
        for window, col in [(20, 'beta_20d'), (60, 'beta_60d'), (120, 'beta_120d'), (252, 'beta_252d')]:
            cov = aligned['stock'].rolling(window).cov(aligned['market'])
            var = aligned['market'].rolling(window).var()
            f[col] = (cov / var).reindex(close.index)

        beta = f['beta_60d']
        conditions = [beta < 0.5, (beta >= 0.5) & (beta < 0.8), (beta >= 0.8) & (beta < 1.2),
                      (beta >= 1.2) & (beta < 1.5), beta >= 1.5]
        choices = ['very_low', 'low', 'market', 'high', 'very_high']
        f['beta_zone'] = pd.Series(np.select(conditions, choices, default=None), index=f.index)
        f['beta_upside_60d'] = None
        f['beta_downside_60d'] = None
    else:
        f['beta_20d'] = None
        f['beta_60d'] = None
        f['beta_120d'] = None
        f['beta_252d'] = None
        f['beta_zone'] = None
        f['beta_upside_60d'] = None
        f['beta_downside_60d'] = None

    return f

def process_symbol(args):
    """Procesa un simbolo (funcion para worker)."""
    symbol, market_returns_dict = args

    # Reconstruir market_returns como Series
    market_returns = pd.Series(market_returns_dict)
    market_returns.index = pd.to_datetime(market_returns.index)

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Obtener precios
        cur.execute("SELECT date, close FROM fmp_price_history WHERE symbol = %s ORDER BY date", (symbol,))
        rows = cur.fetchall()

        if len(rows) < 300:
            conn.close()
            return (symbol, False, 0)

        df = pd.DataFrame(rows, columns=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        close = df['close']
        returns = close.pct_change()

        # Calcular features
        features = calculate_features(close, returns, market_returns)
        features = features.dropna(thresh=len(features.columns) * 0.3)

        if features.empty:
            conn.close()
            return (symbol, False, 0)

        # Preparar datos para insercion
        cols = ['symbol', 'date', 'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d', 'ret_252d',
                'ret_1d_fwd', 'ret_5d_fwd', 'ret_20d_fwd', 'vol_20d', 'vol_60d',
                'sharpe_20d', 'sharpe_60d', 'sortino_20d', 'sortino_60d',
                'max_dd_20d', 'max_dd_60d', 'momentum_score',
                'beta_20d', 'beta_60d', 'beta_120d', 'beta_252d', 'beta_zone',
                'beta_upside_60d', 'beta_downside_60d']

        rows_to_insert = []
        for date, row in features.iterrows():
            record = [symbol, date.date() if hasattr(date, 'date') else date]
            for col in cols[2:]:
                val = row.get(col)
                if pd.isna(val) if not isinstance(val, str) else False:
                    record.append(None)
                elif isinstance(val, (np.floating, float)):
                    if np.isinf(val) or np.isnan(val):
                        record.append(None)
                    else:
                        record.append(float(val))
                else:
                    record.append(val)
            rows_to_insert.append(tuple(record))

        if rows_to_insert:
            query = f"INSERT INTO features_momentum ({', '.join(cols)}) VALUES %s"
            execute_values(cur, query, rows_to_insert, page_size=1000)
            conn.commit()

        conn.close()
        return (symbol, True, len(rows_to_insert))

    except Exception as e:
        return (symbol, False, 0)

def main():
    print(f"{'='*70}", flush=True)
    print(f"MOMENTUM FEATURES - MULTIPROCESSING ({NUM_WORKERS} workers)", flush=True)
    print(f"{'='*70}", flush=True)

    # Cargar market returns
    print("Cargando S&P 500...", flush=True)
    market_returns = get_market_returns_cached()
    print(f"  {len(market_returns):,} dias de mercado", flush=True)

    # Convertir a dict para pasar a workers
    market_returns_dict = market_returns.to_dict()

    # Simbolos faltantes
    print("Obteniendo simbolos faltantes...", flush=True)
    symbols = get_missing_symbols()
    total = len(symbols)
    print(f"  {total:,} simbolos a procesar", flush=True)

    if total == 0:
        print("No hay simbolos faltantes. Completado!", flush=True)
        return

    # Preparar argumentos para workers
    args_list = [(s, market_returns_dict) for s in symbols]

    success, failed, total_records = 0, 0, 0
    start_time = datetime.now()

    # Procesar en paralelo
    print(f"\nIniciando procesamiento con {NUM_WORKERS} workers...", flush=True)

    with Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_symbol, args_list), 1):
            symbol, ok, count = result
            if ok:
                success += 1
                total_records += count
            else:
                failed += 1

            if i % 100 == 0 or i == total:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                eta_sec = (total - i) / rate if rate > 0 else 0
                eta_min = eta_sec / 60
                pct = i / total * 100
                print(f"[{pct:5.1f}%] {i:,}/{total:,} | OK: {success:,} | Fail: {failed:,} | "
                      f"Rec: {total_records:,} | {rate:.1f}/s | ETA: {eta_min:.0f}m", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"COMPLETADO: {success:,} simbolos, {total_records:,} registros", flush=True)
    print(f"Fallidos: {failed:,}", flush=True)
    print(f"Tiempo: {(datetime.now() - start_time).total_seconds()/60:.1f} min", flush=True)

if __name__ == '__main__':
    main()
