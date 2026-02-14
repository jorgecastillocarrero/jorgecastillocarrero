"""
Continua el calculo de features de momentum para simbolos faltantes.
Optimizado: usa execute_values, suprime warnings, mejor logging.
"""
import warnings
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import sys

DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
RISK_FREE_RATE = 0.05

def get_market_returns(conn):
    """Carga returns del S&P 500."""
    cur = conn.cursor()
    cur.execute("SELECT date, close FROM price_history_index WHERE symbol = '^GSPC' ORDER BY date")
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['date', 'close'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df['close'].pct_change()

def get_missing_symbols(conn):
    """Simbolos que faltan por procesar."""
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
    return [row[0] for row in cur.fetchall()]

def get_price_data(conn, symbol):
    """Obtiene precios de un simbolo usando cursor directo."""
    cur = conn.cursor()
    cur.execute("SELECT date, close FROM fmp_price_history WHERE symbol = %s ORDER BY date", (symbol,))
    rows = cur.fetchall()
    if len(rows) < 300:
        return None
    df = pd.DataFrame(rows, columns=['date', 'close'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

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

def save_features_fast(conn, symbol, features):
    """Guarda features usando execute_values (mucho mas rapido)."""
    cur = conn.cursor()

    cols = ['symbol', 'date', 'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d', 'ret_252d',
            'ret_1d_fwd', 'ret_5d_fwd', 'ret_20d_fwd', 'vol_20d', 'vol_60d',
            'sharpe_20d', 'sharpe_60d', 'sortino_20d', 'sortino_60d',
            'max_dd_20d', 'max_dd_60d', 'momentum_score',
            'beta_20d', 'beta_60d', 'beta_120d', 'beta_252d', 'beta_zone',
            'beta_upside_60d', 'beta_downside_60d']

    rows = []
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
        rows.append(tuple(record))

    if rows:
        query = f"INSERT INTO features_momentum ({', '.join(cols)}) VALUES %s"
        execute_values(cur, query, rows, page_size=1000)
        conn.commit()

    return len(rows)

def main():
    conn = psycopg2.connect(DB_URL)

    print(f"{'='*70}", flush=True)
    print(f"MOMENTUM FEATURES - CONTINUACION - {datetime.now()}", flush=True)
    print(f"{'='*70}", flush=True)

    # Market returns
    print("Cargando S&P 500...", flush=True)
    market_returns = get_market_returns(conn)
    print(f"  {len(market_returns):,} dias de mercado", flush=True)

    # Missing symbols
    print("Obteniendo simbolos faltantes...", flush=True)
    symbols = get_missing_symbols(conn)
    total = len(symbols)
    print(f"  {total:,} simbolos a procesar", flush=True)

    if total == 0:
        print("No hay simbolos faltantes. Completado!", flush=True)
        conn.close()
        return

    success, failed, total_records = 0, 0, 0
    start_time = datetime.now()

    for i, symbol in enumerate(symbols, 1):
        try:
            df = get_price_data(conn, symbol)
            if df is None:
                failed += 1
                continue

            close = df['close']
            returns = close.pct_change()

            features = calculate_features(close, returns, market_returns)
            features = features.dropna(thresh=len(features.columns) * 0.3)

            if features.empty:
                failed += 1
                continue

            count = save_features_fast(conn, symbol, features)
            success += 1
            total_records += count

        except Exception as e:
            failed += 1
            if i % 500 == 0 or failed < 5:
                print(f"  Error {symbol}: {str(e)[:60]}", flush=True)

        if i % 200 == 0:
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

    conn.close()

if __name__ == '__main__':
    main()
