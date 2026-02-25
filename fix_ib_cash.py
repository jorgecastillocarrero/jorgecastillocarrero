import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.begin() as conn:
    # Corregir IB USD del 20/02
    conn.execute(sqlalchemy.text(
        "UPDATE cash_diario SET saldo = 23995.25 "
        "WHERE account_code = 'IB' AND currency = 'USD' AND fecha::date = '2026-02-20'"
    ))
    print("Corregido: IB USD 20/02 -> 23,995.25")

    # Corregir IB USD del 23/02
    conn.execute(sqlalchemy.text(
        "UPDATE cash_diario SET saldo = 23995.25 "
        "WHERE account_code = 'IB' AND currency = 'USD' AND fecha::date = '2026-02-23'"
    ))
    print("Corregido: IB USD 23/02 -> 23,995.25")

# Verificar
with engine.connect() as conn:
    print("\n=== Verificacion IB USD ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT fecha, currency, saldo FROM cash_diario "
        "WHERE account_code = 'IB' AND currency = 'USD' AND fecha::date >= '2026-02-18' "
        "ORDER BY fecha"
    ))
    for row in r.fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2]:>14,.2f}")
