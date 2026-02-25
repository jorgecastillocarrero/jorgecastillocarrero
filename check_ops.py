"""Verificar operaciones de una fecha específica"""
from src.database import DatabaseManager
from sqlalchemy import text
import sys

fecha = sys.argv[1] if len(sys.argv) > 1 else '2026-02-17'

db = DatabaseManager()
with db.get_session() as session:
    result = session.execute(text(f"""
        SELECT cuenta, tipo, symbol, cantidad, precio, total_eur, fecha
        FROM operaciones
        WHERE fecha::date = '{fecha}'
        ORDER BY cuenta, tipo, symbol
    """))
    ops = result.fetchall()

    if ops:
        print(f"=== OPERACIONES {fecha} ===\n")
        current_cuenta = None
        compras = 0
        ventas = 0
        for op in ops:
            if op[0] != current_cuenta:
                current_cuenta = op[0]
                print(f"--- {current_cuenta} ---")
            tipo = op[1]
            if tipo.lower() == 'compra':
                compras += 1
            else:
                ventas += 1
            print(f"{tipo:6} | {op[2]:8} | {op[3]:>10.2f} acc | ${op[4]:>10.4f} | {op[5]:>12.2f} EUR")
        print(f"\nResumen: {compras} compras, {ventas} ventas")
        print(f"Total operaciones: {len(ops)}")
    else:
        print(f"No hay operaciones registradas para {fecha}")
