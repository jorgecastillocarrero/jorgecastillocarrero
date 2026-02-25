import psycopg2
import yfinance as yf
from datetime import datetime

conn = psycopg2.connect('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')
cur = conn.cursor()

# Get ALL symbols that had prices on 13/02 but missing on 17/02
cur.execute("""
    SELECT DISTINCT s.code
    FROM symbols s
    JOIN price_history p ON s.id = p.symbol_id
    WHERE DATE(p.date) = '2026-02-13'
    AND NOT EXISTS (
        SELECT 1 FROM price_history p2
        WHERE p2.symbol_id = s.id AND DATE(p2.date) = '2026-02-17'
    )
    ORDER BY s.code
""")
missing = [r[0] for r in cur.fetchall()]
print(f"Downloading {len(missing)} missing symbols for 17/02...")

# Get symbol IDs
cur.execute("SELECT id, code FROM symbols WHERE code = ANY(%s)", (missing,))
symbol_map = {r[1]: r[0] for r in cur.fetchall()}
print(f"Found {len(symbol_map)} symbols in database")

# Download from Yahoo
if missing:
    data = yf.download(missing, start='2026-02-17', end='2026-02-18', progress=True, threads=True)

    if not data.empty:
        inserted = 0
        for symbol in missing:
            if symbol not in symbol_map:
                print(f"  {symbol}: NOT IN SYMBOLS TABLE")
                continue

            symbol_id = symbol_map[symbol]

            try:
                if len(missing) == 1:
                    row = data.iloc[0]
                else:
                    if symbol not in data.columns.get_level_values(1):
                        print(f"  {symbol}: NO DATA")
                        continue
                    row = data.xs(symbol, axis=1, level=1).iloc[0]

                if row.isna().all():
                    print(f"  {symbol}: ALL NaN")
                    continue

                cur.execute("""
                    INSERT INTO price_history (symbol_id, date, open, high, low, close, adjusted_close, volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol_id, date) DO UPDATE SET
                        close = EXCLUDED.close,
                        adjusted_close = EXCLUDED.adjusted_close
                """, (
                    symbol_id,
                    '2026-02-17',
                    float(row['Open']) if not row.isna()['Open'] else None,
                    float(row['High']) if not row.isna()['High'] else None,
                    float(row['Low']) if not row.isna()['Low'] else None,
                    float(row['Close']) if not row.isna()['Close'] else None,
                    float(row['Adj Close']) if not row.isna()['Adj Close'] else None,
                    int(row['Volume']) if not row.isna()['Volume'] else None,
                    datetime.now()
                ))
                inserted += 1
                print(f"  {symbol}: OK (${row['Close']:.2f})")
            except Exception as e:
                print(f"  {symbol}: ERROR - {e}")

        conn.commit()
        print(f"\nInserted {inserted} records")
    else:
        print("No data returned from Yahoo")

conn.close()
