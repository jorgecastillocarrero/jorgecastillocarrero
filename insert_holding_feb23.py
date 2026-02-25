import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.begin() as conn:
    # 1. Copiar holding del 20/02 para todas las cuentas
    # Pero con precio_entrada = cierre del 23/02
    print("=== Insertando holding_diario 23/02 ===\n")

    # Obtener todas las posiciones del 20/02
    r = conn.execute(sqlalchemy.text(
        "SELECT account_code, symbol, shares, currency, asset_type, fecha_compra "
        "FROM holding_diario WHERE fecha::date = '2026-02-20' "
        "ORDER BY account_code, symbol"
    ))
    posiciones_20 = r.fetchall()

    # Añadir las 4 compras nuevas del 23/02 (RCO951)
    compras_nuevas = conn.execute(sqlalchemy.text(
        "SELECT 'RCO951', symbol, shares, currency, asset_type, fecha "
        "FROM compras WHERE fecha::date = '2026-02-23'"
    )).fetchall()

    # Juntar todo
    todas = list(posiciones_20) + list(compras_nuevas)

    inserted = 0
    for pos in todas:
        account, symbol, shares, currency, asset_type, fecha_compra = pos

        # Obtener precio cierre 23/02
        r2 = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym AND ph.date = '2026-02-23'"
        ), {"sym": symbol})
        row = r2.fetchone()
        precio = row[0] if row else None

        conn.execute(sqlalchemy.text(
            "INSERT INTO holding_diario (fecha, account_code, symbol, shares, precio_entrada, currency, asset_type, fecha_compra) "
            "VALUES ('2026-02-23', :account, :symbol, :shares, :precio, :currency, :asset_type, :fecha_compra)"
        ), {
            "account": account,
            "symbol": symbol,
            "shares": shares,
            "precio": precio,
            "currency": currency,
            "asset_type": asset_type,
            "fecha_compra": fecha_compra,
        })
        status = f"{precio:.4f}" if precio else "SIN PRECIO"
        inserted += 1
        print(f"  {account:8s} | {symbol:10s} | {shares:>10} | {status:>12} {currency} | {asset_type}")

    print(f"\nTotal insertados: {inserted}")

# Verificar
with engine.connect() as conn:
    print("\n=== Verificacion holding_diario 23/02 ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT account_code, COUNT(*) as n "
        "FROM holding_diario WHERE fecha::date = '2026-02-23' "
        "GROUP BY account_code ORDER BY account_code"
    ))
    for row in r.fetchall():
        print(f"  {row[0]}: {row[1]} posiciones")
