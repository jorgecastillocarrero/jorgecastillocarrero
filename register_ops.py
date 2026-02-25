"""Register operations for RCO951 on 17/02/2026"""
import psycopg2
from datetime import datetime

conn = psycopg2.connect('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')
cur = conn.cursor()

account = 'RCO951'
fecha = '2026-02-17'

# Operations from the Excel file
operations = [
    # (tipo, symbol, shares, price_usd, total_usd, commission)
    ('VENTA', 'COIN', 60, 168.288, 10084.28, 13),
    ('COMPRA', 'GM', 311, 80.495, 25046.95, 13),
]

print(f"=== Registrando operaciones {account} {fecha} ===\n")

for op_type, symbol, shares, price, total_usd, commission in operations:
    if op_type == 'COMPRA':
        # Insert into compras
        cur.execute("""
            INSERT INTO compras (account_code, symbol, fecha, shares, price_usd, total_usd, commission_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (account, symbol, fecha, shares, price, total_usd, commission))
        print(f"COMPRA: {shares} {symbol} @ ${price:.3f} = ${total_usd:,.2f}")
    else:
        # Insert into ventas
        cur.execute("""
            INSERT INTO ventas (account_code, symbol, fecha, shares, price_usd, total_usd, commission_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (account, symbol, fecha, shares, price, total_usd, commission))
        print(f"VENTA: {shares} {symbol} @ ${price:.3f} = ${total_usd:,.2f}")

conn.commit()

# Calculate net cash movement (ventas - compras)
net_usd = 10084.28 - 25046.95  # COIN sale - GM purchase
print(f"\nMovimiento neto cash: ${net_usd:,.2f} USD")

# Get current cash for RCO951
cur.execute("""
    SELECT currency, amount FROM cash_diario
    WHERE account_code = %s AND DATE(fecha) = %s
""", (account, fecha))
current_cash = cur.fetchall()
print(f"\nCash actual {fecha}: {current_cash}")

# Update cash_diario - add the net movement
cur.execute("""
    UPDATE cash_diario
    SET amount = amount + %s
    WHERE account_code = %s AND currency = 'USD' AND DATE(fecha) = %s
""", (net_usd, account, fecha))
print(f"Cash USD actualizado con {net_usd:,.2f}")

conn.commit()

# Verify
cur.execute("""
    SELECT currency, amount FROM cash_diario
    WHERE account_code = %s AND DATE(fecha) = %s
""", (account, fecha))
new_cash = cur.fetchall()
print(f"Cash nuevo {fecha}: {new_cash}")

conn.close()
print("\n=== Operaciones registradas ===")
