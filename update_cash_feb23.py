import sqlalchemy
from datetime import datetime

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

fecha = datetime(2026, 2, 23)

# Cash RCO951: EUR sin cambios, USD = -82907.68 - 80095.19 = -163002.87
# Resto de cuentas: copiar del 20/02 (sin cambios)
cash_entries = [
    {"account": "RCO951",  "currency": "EUR", "saldo": 226607.84},
    {"account": "RCO951",  "currency": "USD", "saldo": -163002.87},
    {"account": "CO3365",  "currency": "EUR", "saldo": 664.27},
    {"account": "CO3365",  "currency": "USD", "saldo": 5161.12},
    {"account": "IB",      "currency": "EUR", "saldo": 690595.70},
    {"account": "IB",      "currency": "USD", "saldo": 1960180.25},
    {"account": "LACAIXA", "currency": "EUR", "saldo": 0.00},
    {"account": "LACAIXA", "currency": "USD", "saldo": 156460.00},
]

with engine.begin() as conn:
    for c in cash_entries:
        conn.execute(sqlalchemy.text(
            "INSERT INTO cash_diario (fecha, account_code, currency, saldo) "
            "VALUES (:fecha, :account, :currency, :saldo)"
        ), {"fecha": fecha, "account": c["account"], "currency": c["currency"], "saldo": c["saldo"]})
        print(f"  {c['account']:8s} | {c['currency']} | {c['saldo']:>14,.2f}")

    print("\nCash diario 23/02 insertado.")

# Verificar
with engine.connect() as conn:
    print("\n=== Verificacion cash_diario 23/02 ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT account_code, currency, saldo FROM cash_diario "
        "WHERE fecha::date = '2026-02-23' ORDER BY account_code, currency"
    ))
    for row in r.fetchall():
        print(f"  {row[0]:8s} | {row[1]} | {row[2]:>14,.2f}")
