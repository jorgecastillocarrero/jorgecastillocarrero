import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

symbols = ['AEM.TO', 'ATZ.TO', 'IAG.MC', 'JD', 'NESN.SW']
fechas = ['2026-02-13', '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20']

with engine.begin() as conn:
    total = 0
    for sym in symbols:
        for fecha in fechas:
            # Obtener precio cierre
            r = conn.execute(sqlalchemy.text(
                "SELECT ph.close FROM price_history ph "
                "JOIN symbols s ON s.id = ph.symbol_id "
                "WHERE s.code = :sym AND ph.date = :fecha"
            ), {"sym": sym, "fecha": fecha})
            row = r.fetchone()
            if row:
                precio = row[0]
                r2 = conn.execute(sqlalchemy.text(
                    "UPDATE holding_diario SET precio_entrada = :precio "
                    "WHERE account_code = 'LACAIXA' AND symbol = :sym "
                    "AND fecha::date = :fecha AND precio_entrada IS NULL"
                ), {"precio": precio, "sym": sym, "fecha": fecha})
                if r2.rowcount > 0:
                    total += 1
                    print(f"  {fecha} | {sym:10s} | {precio:.4f}")

    print(f"\nTotal corregidos: {total}")

# Verificar
with engine.connect() as conn:
    print("\n=== Verificacion LACAIXA 13/02-20/02 ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT fecha::date, symbol, shares, precio_entrada, currency "
        "FROM holding_diario WHERE account_code = 'LACAIXA' "
        "AND fecha::date >= '2026-02-13' ORDER BY fecha, symbol"
    ))
    fecha_act = None
    for row in r.fetchall():
        f = str(row[0])
        if f != fecha_act:
            fecha_act = f
            print(f"\n  {f}:")
        print(f"    {row[1]:10s} | {row[2]:>8} uds | entrada: {row[3]:>10.4f} {row[4]}")
