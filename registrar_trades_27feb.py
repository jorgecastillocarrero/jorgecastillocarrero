import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy
from datetime import date

engine = sqlalchemy.create_engine(
    'postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway',
    pool_pre_ping=True, pool_recycle=60
)

FECHA = '2026-02-27'

# ============================================================
# CO3365 (Mensual) - VENTAS
# ============================================================
co3365_ventas = [
    ('HEI', 73, 320.91, 13.00, 23413.43),
    ('HEI', 72, 320.911, 13.00, 23092.59),
    ('AVGO', 73, 317.638, 13.00, 23174.57),
    ('AVGO', 72, 317.492, 13.00, 22846.42),
    ('NET', 135, 169.236, 13.00, 22833.86),
    ('NET', 134, 169.236, 13.00, 22664.62),
    ('IDXX', 35, 656.671, 13.00, 22970.49),
    ('IDXX', 36, 656.64, 13.00, 23626.04),
    ('REGN', 32, 785.441, 13.00, 25121.11),
    ('REGN', 32, 785.441, 13.00, 25121.11),
    ('SYK', 65, 386.401, 13.00, 25103.07),
    ('SYK', 66, 386.402, 13.00, 25489.53),
    ('TJX', 161, 160.177, 13.00, 25775.50),
    ('TJX', 160, 160.177, 13.00, 25615.32),
    ('STLD', 135, 191.315, 13.00, 25814.53),
    ('STLD', 134, 191.321, 13.00, 25624.01),
    ('GE', 81, 339.229, 13.00, 27464.55),
    ('GE', 80, 339.22, 13.00, 27124.60),
    ('PAA', 1265, 20.855, 13.00, 26368.58),
    ('PAA', 1250, 20.851, 13.00, 26050.75),
]

# CO3365 (Mensual) - COMPRAS
co3365_compras = [
    ('OHI', 517, 48.399, 13.01, 25035.29),
    ('OHI', 516, 48.399, 13.00, 24986.88),
    ('ED', 148, 112.826, 13.00, 16711.25),
    ('ED', 73, 112.826, 13.00, 8249.30),
    ('ED', 221, 112.822, 13.00, 24946.66),
    ('SCCO', 115, 218.468, 13.00, 25136.82),
    ('SCCO', 114, 218.468, 13.00, 24918.35),
    ('CHD', 238, 105.018, 13.00, 25007.28),
    ('CHD', 238, 105.018, 13.00, 25007.28),
    ('STZ', 157, 158.31, 13.00, 24867.67),
    ('STZ', 157, 158.31, 13.00, 24867.67),
    ('UPS', 214, 116.427, 13.00, 24928.38),
    ('UPS', 214, 116.427, 13.00, 24928.38),
    ('CVX', 134, 186.235, 13.00, 24968.49),
    ('CVX', 134, 186.233, 13.00, 24968.22),
    ('DG', 160, 156.387, 13.00, 25034.92),
    ('DG', 160, 156.387, 13.00, 25034.92),
    ('VLO', 123, 204.301, 13.00, 25142.02),
    ('VLO', 122, 204.325, 13.00, 24940.65),
    ('GOOGL', 81, 308.498, 13.00, 25001.34),
    ('GOOGL', 81, 308.398, 13.00, 24993.24),
]

# ============================================================
# RCO951 (Quant) - VENTAS
# ============================================================
rco951_ventas = [
    ('SAP', 96, 201.314, 13.00, 19313.14),
    ('LTH', 744, 26.902, 13.00, 20002.09),
    ('GLDD', 1328, 16.9101, 13.00, 22443.61),
    ('GMED', 236, 94.804, 13.00, 22360.74),
    ('GIL', 328, 68.005, 13.00, 22292.64),
    ('EME', 29, 718.434, 13.00, 20821.59),
    ('HALO', 295, 69.682, 13.00, 20543.19),
    ('HRMY', 623, 28.568, 13.01, 17784.85),
    ('FSLR', 88, 196.97, 13.00, 17320.36),
    ('CECO', 380, 59.582, 13.00, 22628.16),
]

# RCO951 (Quant) - COMPRAS
rco951_compras = [
    ('HCI', 114, 174.798, 13.00, 19939.97),
    ('AUPH', 1426, 14.04, 13.00, 20034.04),
    ('TMDX', 140, 142.279, 13.00, 19932.06),
    ('RRC', 487, 41.078, 13.01, 20018.00),
    ('LINC', 554, 36.2, 13.00, 20067.80),
    ('CVNA', 60, 331.98, 13.00, 19931.80),
    ('ARGX', 26, 773.575, 13.00, 20125.95),
    ('AROC', 566, 35.318, 13.00, 20002.99),
    ('AS', 528, 37.7799, 13.00, 19960.79),
]

with engine.begin() as conn:
    inserted = {'compras': 0, 'ventas': 0}

    # ========== INSERT VENTAS ==========
    for account, ventas_list in [('CO3365', co3365_ventas), ('RCO951', rco951_ventas)]:
        asset_type = 'Mensual' if account == 'CO3365' else 'Quant'
        for sym, shares, precio, comision, importe in ventas_list:
            conn.execute(sqlalchemy.text("""
                INSERT INTO ventas (fecha, account_code, symbol, shares, precio, comision, currency, importe_total)
                VALUES (:fecha, :acc, :sym, :shares, :precio, :com, 'USD', :importe)
            """), {
                'fecha': FECHA, 'acc': account, 'sym': sym,
                'shares': shares, 'precio': precio, 'com': comision, 'importe': importe
            })
            inserted['ventas'] += 1

    # ========== INSERT COMPRAS ==========
    for account, compras_list in [('CO3365', co3365_compras), ('RCO951', rco951_compras)]:
        asset_type = 'Mensual' if account == 'CO3365' else 'Quant'
        for sym, shares, precio, comision, importe in compras_list:
            conn.execute(sqlalchemy.text("""
                INSERT INTO compras (fecha, account_code, symbol, shares, precio, comision, currency, importe_total, asset_type)
                VALUES (:fecha, :acc, :sym, :shares, :precio, :com, 'USD', :importe, :at)
            """), {
                'fecha': FECHA, 'acc': account, 'sym': sym,
                'shares': shares, 'precio': precio, 'com': comision,
                'importe': importe, 'at': asset_type
            })
            inserted['compras'] += 1

    print(f"Insertadas: {inserted['compras']} compras, {inserted['ventas']} ventas")

    # ========== UPDATE HOLDING_DIARIO CO3365 ==========
    # Delete old CO3365 Mensual holdings for 27/02
    r = conn.execute(sqlalchemy.text("""
        DELETE FROM holding_diario
        WHERE account_code='CO3365' AND fecha=:fecha AND asset_type='Mensual'
    """), {'fecha': FECHA})
    print(f"\nCO3365: eliminados {r.rowcount} holdings Mensual del 27/02")

    # New CO3365 portfolio (aggregated from compras)
    co3365_new_holdings = {}
    for sym, shares, precio, comision, importe in co3365_compras:
        if sym not in co3365_new_holdings:
            co3365_new_holdings[sym] = {'shares': 0, 'total_cost': 0}
        co3365_new_holdings[sym]['shares'] += shares
        co3365_new_holdings[sym]['total_cost'] += shares * precio

    for sym, data in co3365_new_holdings.items():
        avg_price = data['total_cost'] / data['shares']
        conn.execute(sqlalchemy.text("""
            INSERT INTO holding_diario (fecha, account_code, symbol, shares, precio_entrada, currency, asset_type)
            VALUES (:fecha, 'CO3365', :sym, :shares, :precio, 'USD', 'Mensual')
        """), {'fecha': FECHA, 'sym': sym, 'shares': data['shares'], 'precio': round(avg_price, 6)})

    print(f"CO3365: insertados {len(co3365_new_holdings)} nuevos holdings Mensual")
    for sym, data in sorted(co3365_new_holdings.items()):
        avg = data['total_cost'] / data['shares']
        print(f"  {sym}: {data['shares']} shares @ ${avg:.3f}")

    # ========== UPDATE HOLDING_DIARIO RCO951 ==========
    # Remove sold symbols
    sold_quant = [sym for sym, *_ in rco951_ventas]
    for sym in sold_quant:
        r = conn.execute(sqlalchemy.text("""
            DELETE FROM holding_diario
            WHERE account_code='RCO951' AND fecha=:fecha AND symbol=:sym AND asset_type='Quant'
        """), {'fecha': FECHA, 'sym': sym})
        print(f"RCO951: eliminado {sym} ({r.rowcount} rows)")

    # Add bought symbols
    for sym, shares, precio, comision, importe in rco951_compras:
        conn.execute(sqlalchemy.text("""
            INSERT INTO holding_diario (fecha, account_code, symbol, shares, precio_entrada, currency, asset_type)
            VALUES (:fecha, 'RCO951', :sym, :shares, :precio, 'USD', 'Quant')
        """), {'fecha': FECHA, 'sym': sym, 'shares': shares, 'precio': precio})
        print(f"RCO951: insertado {sym} {shares} shares @ ${precio}")

    # ========== UPDATE CASH_DIARIO ==========
    # Calculate cash changes
    co3365_ventas_total = sum(imp for _, _, _, _, imp in co3365_ventas)
    co3365_compras_total = sum(imp for _, _, _, _, imp in co3365_compras)
    co3365_cash_change = co3365_ventas_total - co3365_compras_total

    rco951_ventas_total = sum(imp for _, _, _, _, imp in rco951_ventas)
    rco951_compras_total = sum(imp for _, _, _, _, imp in rco951_compras)
    rco951_cash_change = rco951_ventas_total - rco951_compras_total

    print(f"\n=== Cash Changes ===")
    print(f"CO3365: ventas={co3365_ventas_total:.2f} - compras={co3365_compras_total:.2f} = {co3365_cash_change:+.2f}")
    print(f"RCO951: ventas={rco951_ventas_total:.2f} - compras={rco951_compras_total:.2f} = {rco951_cash_change:+.2f}")

    # Current cash
    r_cash = conn.execute(sqlalchemy.text("""
        SELECT saldo FROM cash_diario
        WHERE account_code='CO3365' AND currency='USD' AND fecha=:fecha
    """), {'fecha': FECHA})
    co3365_usd_old = float(r_cash.fetchone()[0])

    r_cash2 = conn.execute(sqlalchemy.text("""
        SELECT saldo FROM cash_diario
        WHERE account_code='RCO951' AND currency='USD' AND fecha=:fecha
    """), {'fecha': FECHA})
    rco951_usd_old = float(r_cash2.fetchone()[0])

    co3365_usd_new = co3365_usd_old + co3365_cash_change
    rco951_usd_new = rco951_usd_old + rco951_cash_change

    print(f"\nCO3365 USD: {co3365_usd_old:.2f} → {co3365_usd_new:.2f}")
    print(f"RCO951 USD: {rco951_usd_old:.2f} → {rco951_usd_new:.2f}")

    # Update cash
    conn.execute(sqlalchemy.text("""
        UPDATE cash_diario SET saldo=:saldo
        WHERE account_code='CO3365' AND currency='USD' AND fecha=:fecha
    """), {'saldo': co3365_usd_new, 'fecha': FECHA})

    conn.execute(sqlalchemy.text("""
        UPDATE cash_diario SET saldo=:saldo
        WHERE account_code='RCO951' AND currency='USD' AND fecha=:fecha
    """), {'saldo': rco951_usd_new, 'fecha': FECHA})

    print("\nCash actualizado OK")

print("\n=== REGISTRO COMPLETADO ===")
