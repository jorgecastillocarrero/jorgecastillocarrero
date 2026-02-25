import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Historico cash IB
    print("=== CASH_DIARIO IB (historico reciente) ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT fecha, currency, saldo FROM cash_diario "
        "WHERE account_code = 'IB' ORDER BY fecha DESC, currency LIMIT 20"
    ))
    for row in r.fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2]:>14,.2f}")

    # Trades IB recientes
    print("\n=== STOCK_TRADES IB (recientes) ===")
    r2 = conn.execute(sqlalchemy.text(
        "SELECT trade_date, symbol, trade_type, shares, price, currency, amount_local "
        "FROM stock_trades WHERE account_code = 'IB' AND trade_date >= '2026-02-12' "
        "ORDER BY trade_date DESC"
    ))
    rows = r2.fetchall()
    if rows:
        for row in rows:
            print(f"  {row[0]} | {row[1]:8s} | {row[2]:6s} | {row[3]} @ {row[4]} {row[5]} | {row[6]}")
    else:
        print("  No hay trades IB recientes")

    # Compras IB recientes
    print("\n=== COMPRAS IB (recientes) ===")
    r3 = conn.execute(sqlalchemy.text(
        "SELECT fecha, symbol, shares, precio, currency, importe_total "
        "FROM compras WHERE account_code = 'IB' AND fecha::date >= '2026-02-12' "
        "ORDER BY fecha DESC"
    ))
    rows3 = r3.fetchall()
    if rows3:
        for row in rows3:
            print(f"  {row[0]} | {row[1]:8s} | {row[2]} @ {row[3]} {row[4]} | {row[5]}")
    else:
        print("  No hay compras IB recientes")

    # Holding IB
    print("\n=== HOLDING_DIARIO IB 20/02 ===")
    r4 = conn.execute(sqlalchemy.text(
        "SELECT symbol, shares, precio_entrada, currency, asset_type "
        "FROM holding_diario WHERE account_code = 'IB' AND fecha::date = '2026-02-20' "
        "ORDER BY symbol"
    ))
    for row in r4.fetchall():
        print(f"  {row[0]:8s} | {row[1]} | {row[2]} {row[3]} | {row[4]}")
