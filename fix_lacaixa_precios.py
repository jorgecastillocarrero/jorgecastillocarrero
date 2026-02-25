import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

symbols = ['AEM.TO', 'ATZ.TO', 'IAG.MC', 'JD', 'NESN.SW']
fechas = ['2026-02-13', '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20']

with engine.connect() as conn:
    # Primero verificar monedas en symbols table
    print("=== Monedas de cada activo ===")
    for sym in symbols:
        r = conn.execute(sqlalchemy.text(
            "SELECT code, currency FROM symbols WHERE code = :sym"
        ), {"sym": sym})
        row = r.fetchone()
        print(f"  {sym:10s} -> {row[1] if row else 'NO ENCONTRADO'}")

    # Obtener precios de cierre para cada símbolo y fecha
    print("\n=== Precios de cierre por fecha ===")
    for sym in symbols:
        print(f"\n  {sym}:")
        for fecha in fechas:
            r = conn.execute(sqlalchemy.text(
                "SELECT ph.close, s.currency FROM price_history ph "
                "JOIN symbols s ON s.id = ph.symbol_id "
                "WHERE s.code = :sym AND ph.date = :fecha"
            ), {"sym": sym, "fecha": fecha})
            row = r.fetchone()
            if row:
                print(f"    {fecha} -> {row[0]:.4f} {row[1]}")
            else:
                print(f"    {fecha} -> SIN PRECIO")
