import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Compras del 23/02/2026
    print("=== COMPRAS 23/02/2026 ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT id, fecha, account_code, symbol, shares, precio, currency, importe_total, notas "
        "FROM compras WHERE fecha::date = '2026-02-23' ORDER BY symbol"
    ))
    rows = r.fetchall()
    if rows:
        for row in rows:
            print(f"  {row[3]:8s} | {row[4]:>10} uds @ {row[5]:>10} {row[6]} | Total: {row[7]} | Cuenta: {row[2]} | {row[8]}")
    else:
        print("  No hay compras con fecha 23/02")

    # Buscar compras recientes con OR en symbol
    print("\n=== COMPRAS recientes con 'OR' en symbol (desde 19/02) ===")
    r2 = conn.execute(sqlalchemy.text(
        "SELECT id, fecha, account_code, symbol, shares, precio, currency, importe_total "
        "FROM compras WHERE UPPER(symbol) LIKE '%OR%' AND fecha::date >= '2026-02-19' ORDER BY fecha DESC"
    ))
    for row in r2.fetchall():
        print(f"  {row[1]} | {row[3]:8s} | {row[4]} uds @ {row[5]} {row[6]} | {row[2]}")

    # También buscar en stock_trades
    print("\n=== STOCK_TRADES 23/02/2026 ===")
    r2b = conn.execute(sqlalchemy.text(
        "SELECT id, trade_date, account_code, symbol, trade_type, shares, price, currency, amount_local, notes "
        "FROM stock_trades WHERE trade_date = '2026-02-23' ORDER BY symbol"
    ))
    rows2b = r2b.fetchall()
    if rows2b:
        for row in rows2b:
            print(f"  {row[4]:5s} | {row[3]:8s} | {row[5]:>10} uds @ {row[6]:>10} {row[7]} | {row[8]} | {row[2]} | {row[9]}")
    else:
        print("  No hay trades en stock_trades para 23/02")

    # Holding diario 23/02
    print("\n=== HOLDING_DIARIO 23/02/2026 ===")
    r3 = conn.execute(sqlalchemy.text(
        "SELECT account_code, COUNT(*) as posiciones "
        "FROM holding_diario WHERE fecha::date = '2026-02-23' "
        "GROUP BY account_code ORDER BY account_code"
    ))
    rows3 = r3.fetchall()
    if rows3:
        for row in rows3:
            print(f"  {row[0]}: {row[1]} posiciones")
    else:
        print("  No hay holding_diario para 23/02")
        # Buscar última fecha disponible
        r3b = conn.execute(sqlalchemy.text(
            "SELECT MAX(fecha::date) FROM holding_diario"
        ))
        print(f"  Ultima fecha disponible: {r3b.fetchone()[0]}")

    # Holding diario 20/02 para comparar
    print("\n=== HOLDING_DIARIO 20/02/2026 (comparación) ===")
    r4 = conn.execute(sqlalchemy.text(
        "SELECT account_code, COUNT(*) as posiciones "
        "FROM holding_diario WHERE fecha::date = '2026-02-20' "
        "GROUP BY account_code ORDER BY account_code"
    ))
    for row in r4.fetchall():
        print(f"  {row[0]}: {row[1]} posiciones")

    # Cash diario 23/02
    print("\n=== CASH_DIARIO 23/02/2026 ===")
    r5 = conn.execute(sqlalchemy.text(
        "SELECT account_code, currency, saldo FROM cash_diario "
        "WHERE fecha::date = '2026-02-23' ORDER BY account_code, currency"
    ))
    rows5 = r5.fetchall()
    if rows5:
        for row in rows5:
            print(f"  {row[0]} | {row[1]} | {row[2]:,.2f}")
    else:
        print("  No hay cash_diario para 23/02")
        r5b = conn.execute(sqlalchemy.text(
            "SELECT MAX(fecha::date) FROM cash_diario"
        ))
        print(f"  Ultima fecha disponible: {r5b.fetchone()[0]}")

    # Posicion 23/02
    print("\n=== POSICION 23/02/2026 ===")
    r6 = conn.execute(sqlalchemy.text(
        "SELECT * FROM posicion WHERE fecha::date = '2026-02-23' ORDER BY account_code"
    ))
    cols6 = r6.keys()
    rows6 = r6.fetchall()
    if rows6:
        for row in rows6:
            d = dict(zip(cols6, row))
            print(f"  {d}")
    else:
        print("  No hay posicion para 23/02")
        r6b = conn.execute(sqlalchemy.text(
            "SELECT MAX(fecha::date) FROM posicion"
        ))
        print(f"  Ultima fecha disponible: {r6b.fetchone()[0]}")
