import psycopg2

conn = psycopg2.connect('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')
cur = conn.cursor()

print("=== POSICION ===")
cur.execute("""
    SELECT DATE(fecha), COUNT(*), ROUND(SUM(total_eur)::numeric, 0)
    FROM posicion
    WHERE fecha >= '2026-02-10'
    GROUP BY DATE(fecha)
    ORDER BY DATE(fecha) DESC
""")
for r in cur.fetchall():
    print(f"{r[0]}: {r[1]} cuentas, {r[2]:,.0f} EUR")

print("\n=== HOLDING_DIARIO ===")
cur.execute("""
    SELECT DATE(fecha), COUNT(DISTINCT account_code), COUNT(*)
    FROM holding_diario
    WHERE fecha >= '2026-02-10'
    GROUP BY DATE(fecha)
    ORDER BY DATE(fecha) DESC
""")
for r in cur.fetchall():
    print(f"{r[0]}: {r[1]} cuentas, {r[2]} holdings")

conn.close()
