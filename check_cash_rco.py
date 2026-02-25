import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Cash diario RCO951 últimos días
    print("=== CASH_DIARIO RCO951 (últimos registros) ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT fecha, account_code, currency, saldo FROM cash_diario "
        "WHERE account_code = 'RCO951' ORDER BY fecha DESC, currency LIMIT 10"
    ))
    for row in r.fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]:,.2f}")

    # Cash diario de TODAS las cuentas el 20/02
    print("\n=== CASH_DIARIO TODAS LAS CUENTAS 20/02 ===")
    r2 = conn.execute(sqlalchemy.text(
        "SELECT fecha, account_code, currency, saldo FROM cash_diario "
        "WHERE fecha::date = '2026-02-20' ORDER BY account_code, currency"
    ))
    for row in r2.fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]:,.2f}")

    # Total compras del 23/02 en USD
    print("\n=== TOTAL COMPRAS 23/02 ===")
    r3 = conn.execute(sqlalchemy.text(
        "SELECT account_code, currency, SUM(importe_total) as total, COUNT(*) as n "
        "FROM compras WHERE fecha::date = '2026-02-23' "
        "GROUP BY account_code, currency"
    ))
    for row in r3.fetchall():
        print(f"  Cuenta: {row[0]} | {row[1]} | Total: {row[2]:,.2f} | {row[3]} compras")
