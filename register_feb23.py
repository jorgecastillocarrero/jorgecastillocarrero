import sqlalchemy
from datetime import datetime

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

# 4 compras del 23/02/2026 - cuenta RCO951
# CIM = Quant, OR/EGO/PAAS = Oro/Mineras
compras = [
    {"symbol": "CIM",  "shares": 1495, "precio": 13.379, "comision": 13.00, "importe_total": 20014.61, "currency": "USD", "asset_type": "Quant"},
    {"symbol": "OR",   "shares": 445,  "precio": 44.943, "comision": 13.01, "importe_total": 20012.65, "currency": "USD", "asset_type": "Oro/Mineras"},
    {"symbol": "EGO",  "shares": 454,  "precio": 44.137, "comision": 13.00, "importe_total": 20051.20, "currency": "USD", "asset_type": "Oro/Mineras"},
    {"symbol": "PAAS", "shares": 309,  "precio": 64.737, "comision": 13.00, "importe_total": 20016.73, "currency": "USD", "asset_type": "Oro/Mineras"},
]

fecha = datetime(2026, 2, 23)

with engine.begin() as conn:
    for c in compras:
        conn.execute(sqlalchemy.text(
            "INSERT INTO compras (fecha, account_code, symbol, shares, precio, comision, currency, importe_total, asset_type) "
            "VALUES (:fecha, :account, :symbol, :shares, :precio, :comision, :currency, :importe, :asset_type)"
        ), {
            "fecha": fecha,
            "account": "RCO951",
            "symbol": c["symbol"],
            "shares": c["shares"],
            "precio": c["precio"],
            "comision": c["comision"],
            "currency": c["currency"],
            "importe": c["importe_total"],
            "asset_type": c["asset_type"],
        })
        print(f"  INSERTADO: {c['symbol']:6s} | {c['shares']} uds @ {c['precio']} USD | {c['asset_type']}")

    print("\n4 compras registradas correctamente.")

# Verificar
with engine.connect() as conn:
    print("\n=== Verificacion: compras 23/02/2026 ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT symbol, shares, precio, currency, importe_total, asset_type "
        "FROM compras WHERE fecha::date = '2026-02-23' ORDER BY symbol"
    ))
    for row in r.fetchall():
        print(f"  {row[0]:6s} | {row[1]} @ {row[2]} {row[3]} | total: {row[4]} | {row[5]}")
