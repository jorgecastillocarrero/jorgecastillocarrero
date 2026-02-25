import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Registros por fecha reciente en price_history
    r = conn.execute(sqlalchemy.text(
        "SELECT date, COUNT(*) as total FROM price_history "
        "WHERE date >= '2026-02-19' GROUP BY date ORDER BY date"
    ))
    rows = r.fetchall()
    print("=== Precios por fecha (price_history) ===")
    ref = 0
    feb23 = 0
    for row in rows:
        d = str(row[0])
        print(f"  {d}  ->  {row[1]} simbolos")
        if d == '2026-02-20':
            ref = row[1]
        if d == '2026-02-23':
            feb23 = row[1]

    if ref > 0 and feb23 > 0:
        pct = round(feb23 / ref * 100, 1)
        print(f"\nCobertura 23/02 vs 20/02: {feb23}/{ref} = {pct}%")
    elif feb23 == 0:
        print(f"\nNO hay datos para el 23/02/2026")
    else:
        print(f"\n23/02: {feb23} simbolos (sin referencia del 20/02)")

    # Ultimos logs de descarga
    print("\n=== Ultimos logs de descarga ===")
    r2 = conn.execute(sqlalchemy.text(
        "SELECT id, download_date, symbols_downloaded, symbols_failed, status, duration_seconds "
        "FROM download_logs ORDER BY id DESC LIMIT 5"
    ))
    for row in r2.fetchall():
        print(f"  {row[1]} | OK: {row[2]} | Fail: {row[3]} | {row[4]} | {row[5]}s")

    # Total simbolos activos
    r3 = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM symbols"))
    total_symbols = r3.fetchone()[0]
    print(f"\nTotal simbolos en tabla symbols: {total_symbols}")
    if feb23 > 0:
        pct_total = round(feb23 / total_symbols * 100, 1)
        print(f"Cobertura 23/02 vs total symbols: {feb23}/{total_symbols} = {pct_total}%")
