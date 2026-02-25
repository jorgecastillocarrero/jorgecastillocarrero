import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Ver desde cuando existe LACAIXA en holding_diario
    print("=== FECHAS LACAIXA en holding_diario ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT DISTINCT fecha::date as f FROM holding_diario "
        "WHERE account_code = 'LACAIXA' ORDER BY f"
    ))
    fechas = [row[0] for row in r.fetchall()]
    print(f"  Desde: {fechas[0]} hasta: {fechas[-1]} ({len(fechas)} fechas)")
    for f in fechas:
        print(f"    {f}")

    # Verificar tipos de cambio disponibles
    print("\n=== Tipos de cambio disponibles ===")
    for pair in ['EURUSD=X', 'CADEUR=X', 'CADUSD=X', 'USDCAD=X', 'CHFUSD=X', 'CHFEUR=X', 'USDCHF=X', 'CAD=X', 'CHF=X', 'CADCHF=X']:
        r2 = conn.execute(sqlalchemy.text(
            "SELECT s.code, ph.date, ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :pair AND ph.date = '2026-02-23' LIMIT 1"
        ), {"pair": pair})
        row = r2.fetchone()
        if row:
            print(f"  {pair}: {row[2]}")

    # Buscar pares de divisas en symbols
    print("\n=== Simbolos de forex disponibles ===")
    r3 = conn.execute(sqlalchemy.text(
        "SELECT code FROM symbols WHERE code LIKE '%CAD%' OR code LIKE '%CHF%' "
        "ORDER BY code LIMIT 20"
    ))
    for row in r3.fetchall():
        print(f"  {row[0]}")

    # Verificar precios de los activos LACAIXA para el 23/02
    print("\n=== Precios LACAIXA activos 23/02 ===")
    for sym in ['AEM.TO', 'ATZ.TO', 'IAG.MC', 'JD', 'NESN.SW']:
        r4 = conn.execute(sqlalchemy.text(
            "SELECT s.code, s.currency, ph.date, ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym AND ph.date = '2026-02-23' LIMIT 1"
        ), {"sym": sym})
        row = r4.fetchone()
        if row:
            print(f"  {row[0]:10s} | moneda: {row[1]} | close: {row[3]}")
        else:
            print(f"  {sym:10s} | SIN PRECIO 23/02")

    # Posicion tabla - ver como se calculo antes
    print("\n=== POSICION tabla - LACAIXA historico ===")
    r5 = conn.execute(sqlalchemy.text(
        "SELECT * FROM posicion WHERE account_code = 'LACAIXA' ORDER BY fecha DESC LIMIT 5"
    ))
    cols = r5.keys()
    for row in r5.fetchall():
        print(f"  {dict(zip(cols, row))}")
