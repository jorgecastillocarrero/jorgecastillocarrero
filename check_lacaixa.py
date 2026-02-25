import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Historico holding_diario LACAIXA
    print("=== HOLDING_DIARIO LACAIXA (ultimas fechas) ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT fecha, symbol, shares, precio_entrada, currency, asset_type "
        "FROM holding_diario WHERE account_code = 'LACAIXA' "
        "ORDER BY fecha DESC, symbol LIMIT 30"
    ))
    fecha_actual = None
    for row in r.fetchall():
        if str(row[0]) != fecha_actual:
            fecha_actual = str(row[0])
            print(f"\n  Fecha: {fecha_actual}")
        print(f"    {row[1]:10s} | {row[2]:>10} uds | entrada: {row[3]} {row[4]} | {row[5]}")

    # Compras LACAIXA
    print("\n=== COMPRAS LACAIXA (todas) ===")
    r2 = conn.execute(sqlalchemy.text(
        "SELECT fecha, symbol, shares, precio, currency, importe_total, asset_type "
        "FROM compras WHERE account_code = 'LACAIXA' ORDER BY fecha DESC"
    ))
    for row in r2.fetchall():
        print(f"  {row[0]} | {row[1]:10s} | {row[2]} @ {row[3]} {row[4]} | {row[5]} | {row[6]}")

    # Ventas LACAIXA
    print("\n=== VENTAS LACAIXA (todas) ===")
    r3 = conn.execute(sqlalchemy.text(
        "SELECT * FROM ventas WHERE account_code = 'LACAIXA' ORDER BY fecha DESC"
    ))
    cols = r3.keys()
    rows = r3.fetchall()
    if rows:
        for row in rows:
            print(f"  {dict(zip(cols, row))}")
    else:
        print("  No hay ventas")

    # Stock trades LACAIXA
    print("\n=== STOCK_TRADES LACAIXA ===")
    r4 = conn.execute(sqlalchemy.text(
        "SELECT trade_date, symbol, trade_type, shares, price, currency "
        "FROM stock_trades WHERE account_code = 'LACAIXA' ORDER BY trade_date DESC"
    ))
    rows4 = r4.fetchall()
    if rows4:
        for row in rows4:
            print(f"  {row[0]} | {row[1]:10s} | {row[2]} | {row[3]} @ {row[4]} {row[5]}")
    else:
        print("  No hay stock_trades")
