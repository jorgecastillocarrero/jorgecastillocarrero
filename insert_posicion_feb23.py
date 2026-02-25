import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Ver columnas de posicion
    r = conn.execute(sqlalchemy.text(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'posicion' ORDER BY ordinal_position"
    ))
    print("=== Columnas posicion ===")
    for row in r.fetchall():
        print(f"  {row[0]} ({row[1]})")

    # Ver ultimas posiciones
    print("\n=== Ultimas posiciones (20/02) ===")
    r2 = conn.execute(sqlalchemy.text(
        "SELECT * FROM posicion WHERE fecha::date = '2026-02-20' ORDER BY account_code"
    ))
    cols = r2.keys()
    for row in r2.fetchall():
        print(f"  {dict(zip(cols, row))}")

    # Verificar como calcula el sistema - ver portfolio_data o posicion_calculator
    print("\n=== Tablas relacionadas ===")
    for t in ['portfolio_snapshots', 'daily_metrics', 'portfolio_holdings']:
        r3 = conn.execute(sqlalchemy.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :t ORDER BY ordinal_position"
        ), {"t": t})
        cols3 = [row[0] for row in r3.fetchall()]
        print(f"  {t}: {cols3}")
