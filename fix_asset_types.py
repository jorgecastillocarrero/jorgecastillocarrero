import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

oro_mineras = ['CGAU', 'HL']
quant = ['ADI', 'CTRE', 'DXCM', 'EQT', 'NVMI', 'STC', 'TAL', 'TRGP', 'VICR', 'VTR']

with engine.begin() as conn:
    for sym in oro_mineras:
        r = conn.execute(sqlalchemy.text(
            "UPDATE holding_diario SET asset_type = 'Oro/Mineras' "
            "WHERE account_code = 'RCO951' AND symbol = :sym AND (asset_type IS NULL OR asset_type = 'None')"
        ), {"sym": sym})
        print(f"  {sym:6s} -> Oro/Mineras ({r.rowcount} registros)")

    for sym in quant:
        r = conn.execute(sqlalchemy.text(
            "UPDATE holding_diario SET asset_type = 'Quant' "
            "WHERE account_code = 'RCO951' AND symbol = :sym AND (asset_type IS NULL OR asset_type = 'None')"
        ), {"sym": sym})
        print(f"  {sym:6s} -> Quant ({r.rowcount} registros)")

    # También actualizar en compras
    print("\n=== Actualizando compras ===")
    for sym in oro_mineras:
        r = conn.execute(sqlalchemy.text(
            "UPDATE compras SET asset_type = 'Oro/Mineras' "
            "WHERE account_code = 'RCO951' AND symbol = :sym AND (asset_type IS NULL OR asset_type = 'None')"
        ), {"sym": sym})
        if r.rowcount > 0:
            print(f"  {sym:6s} -> Oro/Mineras ({r.rowcount} compras)")

    for sym in quant:
        r = conn.execute(sqlalchemy.text(
            "UPDATE compras SET asset_type = 'Quant' "
            "WHERE account_code = 'RCO951' AND symbol = :sym AND (asset_type IS NULL OR asset_type = 'None')"
        ), {"sym": sym})
        if r.rowcount > 0:
            print(f"  {sym:6s} -> Quant ({r.rowcount} compras)")

# Verificar
with engine.connect() as conn:
    print("\n=== Verificacion: posiciones con asset_type None/NULL ===")
    r = conn.execute(sqlalchemy.text(
        "SELECT symbol, asset_type, COUNT(*) FROM holding_diario "
        "WHERE account_code = 'RCO951' AND (asset_type IS NULL OR asset_type = 'None') "
        "GROUP BY symbol, asset_type ORDER BY symbol"
    ))
    rows = r.fetchall()
    if rows:
        for row in rows:
            print(f"  {row[0]:8s} | {row[1]} | {row[2]} registros")
    else:
        print("  Ninguna - todo clasificado correctamente")
