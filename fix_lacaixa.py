import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Holding LACAIXA del 12/02 (antes de la venta de BABA)
    print("=== HOLDING LACAIXA 12/02 ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT symbol, shares, precio_entrada, currency, asset_type "
        "FROM holding_diario WHERE account_code = 'LACAIXA' AND fecha::date = '2026-02-12' "
        "ORDER BY symbol"
    ))
    for row in r.fetchall():
        print(f"  {row[0]:10s} | {row[1]} | {row[2]} {row[3]} | {row[4]}")

    # Holding LACAIXA del 13/02 (despues de la venta de BABA)
    print("\n=== HOLDING LACAIXA 13/02 ===")
    r2 = conn.execute(sqlalchemy.text(
        "SELECT symbol, shares, precio_entrada, currency, asset_type "
        "FROM holding_diario WHERE account_code = 'LACAIXA' AND fecha::date = '2026-02-13' "
        "ORDER BY symbol"
    ))
    for row in r2.fetchall():
        print(f"  {row[0]:10s} | {row[1]} | {row[2]} {row[3]} | {row[4]}")

    # Holding LACAIXA 31/12/2025
    print("\n=== HOLDING LACAIXA 31/12/2025 ===")
    r3 = conn.execute(sqlalchemy.text(
        "SELECT symbol, shares, precio_entrada, currency, asset_type "
        "FROM holding_diario WHERE account_code = 'LACAIXA' AND fecha::date = '2025-12-31' "
        "ORDER BY symbol"
    ))
    for row in r3.fetchall():
        print(f"  {row[0]:10s} | {row[1]} | {row[2]} {row[3]} | {row[4]}")

    # Fechas donde NESN.SW aparece
    print("\n=== NESN.SW en holding_diario (todas las fechas) ===")
    r4 = conn.execute(sqlalchemy.text(
        "SELECT fecha::date, shares FROM holding_diario "
        "WHERE account_code = 'LACAIXA' AND symbol = 'NESN.SW' ORDER BY fecha"
    ))
    for row in r4.fetchall():
        print(f"  {row[0]} | {row[1]}")

    # Fechas donde BABA aparece
    print("\n=== BABA en holding_diario (todas las fechas) ===")
    r5 = conn.execute(sqlalchemy.text(
        "SELECT fecha::date, shares FROM holding_diario "
        "WHERE account_code = 'LACAIXA' AND symbol = 'BABA' ORDER BY fecha"
    ))
    for row in r5.fetchall():
        print(f"  {row[0]} | {row[1]}")

    # Todas las fechas LACAIXA con conteo de posiciones
    print("\n=== Resumen por fecha LACAIXA ===")
    r6 = conn.execute(sqlalchemy.text(
        "SELECT fecha::date, COUNT(*) as n, STRING_AGG(symbol, ', ' ORDER BY symbol) as syms "
        "FROM holding_diario WHERE account_code = 'LACAIXA' "
        "GROUP BY fecha::date ORDER BY fecha"
    ))
    for row in r6.fetchall():
        print(f"  {row[0]} | {row[1]} pos | {row[2]}")
