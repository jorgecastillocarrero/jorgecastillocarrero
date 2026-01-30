"""
Script para sincronizar datos diarios de IB (TLT + Cash) en holding_diario, cash_diario y posicion.
Ejecutar diariamente o cuando falten datos de IB.

Uso:
    python scripts/sync_ib_daily.py              # Sincroniza todas las fechas faltantes
    python scripts/sync_ib_daily.py 2026-01-29   # Sincroniza una fecha específica
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta
from sqlalchemy import text
from src.database import DatabaseManager

# Configuración IB
IB_TLT_SHARES = 8042  # Shares de TLT en IB
IB_ACCOUNT = 'IB'

def get_db():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'financial_data.db')
    return DatabaseManager(f'sqlite:///{db_path}')

def get_tlt_price(session, fecha):
    """Obtiene el precio de TLT para una fecha desde price_history."""
    result = session.execute(text("""
        SELECT ph.close
        FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'TLT' AND DATE(ph.date) = :fecha
        ORDER BY ph.date DESC
        LIMIT 1
    """), {'fecha': fecha})
    row = result.fetchone()
    return row[0] if row else None

def get_eur_usd_rate(session, fecha):
    """Obtiene el tipo de cambio EUR/USD para una fecha."""
    result = session.execute(text("""
        SELECT ph.close
        FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X' AND DATE(ph.date) <= :fecha
        ORDER BY ph.date DESC
        LIMIT 1
    """), {'fecha': fecha})
    row = result.fetchone()
    return row[0] if row else 1.195  # Default

def get_ib_cash_for_date(session, fecha):
    """Obtiene el cash de IB para una fecha, o el más reciente anterior."""
    result = session.execute(text("""
        SELECT saldo FROM cash_diario
        WHERE account_code = 'IB' AND currency = 'EUR' AND fecha <= :fecha
        ORDER BY fecha DESC LIMIT 1
    """), {'fecha': fecha})
    row = result.fetchone()
    return row[0] if row else 123785.94  # Default del último extracto

def get_all_dates_with_data(session):
    """Obtiene todas las fechas que tienen datos en posicion (excluyendo IB)."""
    result = session.execute(text("""
        SELECT DISTINCT fecha FROM posicion
        WHERE account_code != 'IB'
        ORDER BY fecha
    """))
    return [row[0] for row in result.fetchall()]

def sync_ib_for_date(session, fecha_str):
    """Sincroniza datos de IB para una fecha específica."""
    fecha = fecha_str if isinstance(fecha_str, str) else fecha_str.isoformat()

    # 1. Verificar si ya existe TLT en holding_diario
    result = session.execute(text("""
        SELECT id FROM holding_diario
        WHERE fecha = :fecha AND account_code = 'IB' AND symbol = 'TLT'
    """), {'fecha': fecha})

    tlt_exists = result.fetchone() is not None

    # Obtener precio de TLT
    tlt_price = get_tlt_price(session, fecha)
    if not tlt_price:
        # Usar precio del día anterior
        result = session.execute(text("""
            SELECT precio_entrada FROM holding_diario
            WHERE account_code = 'IB' AND symbol = 'TLT' AND fecha < :fecha
            ORDER BY fecha DESC LIMIT 1
        """), {'fecha': fecha})
        row = result.fetchone()
        tlt_price = row[0] if row else 87.60  # Default

    # Insertar/actualizar TLT en holding_diario
    if not tlt_exists:
        session.execute(text("""
            INSERT INTO holding_diario (fecha, account_code, symbol, shares, precio_entrada, currency, created_at)
            VALUES (:fecha, 'IB', 'TLT', :shares, :price, 'USD', datetime('now'))
        """), {'fecha': fecha, 'shares': IB_TLT_SHARES, 'price': tlt_price})
        print(f"  [holding_diario] Insertado TLT: {IB_TLT_SHARES} @ ${tlt_price:.2f}")
    else:
        session.execute(text("""
            UPDATE holding_diario SET shares = :shares, precio_entrada = :price
            WHERE fecha = :fecha AND account_code = 'IB' AND symbol = 'TLT'
        """), {'fecha': fecha, 'shares': IB_TLT_SHARES, 'price': tlt_price})
        print(f"  [holding_diario] Actualizado TLT: {IB_TLT_SHARES} @ ${tlt_price:.2f}")

    # 2. Cash IB
    cash_eur = get_ib_cash_for_date(session, fecha)

    result = session.execute(text("""
        SELECT id FROM cash_diario
        WHERE fecha = :fecha AND account_code = 'IB' AND currency = 'EUR'
    """), {'fecha': fecha})
    cash_exists = result.fetchone() is not None

    if not cash_exists:
        session.execute(text("""
            INSERT INTO cash_diario (fecha, account_code, currency, saldo, created_at)
            VALUES (:fecha, 'IB', 'EUR', :saldo, datetime('now'))
        """), {'fecha': fecha, 'saldo': cash_eur})
        print(f"  [cash_diario] Insertado cash: {cash_eur:,.2f} EUR")
    else:
        print(f"  [cash_diario] Cash ya existe: {cash_eur:,.2f} EUR")

    # 3. Posicion IB
    eur_usd = get_eur_usd_rate(session, fecha)
    tlt_value_eur = IB_TLT_SHARES * tlt_price / eur_usd
    total_eur = tlt_value_eur + cash_eur

    result = session.execute(text("""
        SELECT id FROM posicion WHERE fecha = :fecha AND account_code = 'IB'
    """), {'fecha': fecha})
    pos_exists = result.fetchone() is not None

    if not pos_exists:
        session.execute(text("""
            INSERT INTO posicion (fecha, account_code, holding_eur, cash_eur, total_eur)
            VALUES (:fecha, 'IB', :holding, :cash, :total)
        """), {'fecha': fecha, 'holding': tlt_value_eur, 'cash': cash_eur, 'total': total_eur})
        print(f"  [posicion] Insertado: holding={tlt_value_eur:,.0f}, cash={cash_eur:,.0f}, total={total_eur:,.0f} EUR")
    else:
        session.execute(text("""
            UPDATE posicion SET holding_eur = :holding, cash_eur = :cash, total_eur = :total
            WHERE fecha = :fecha AND account_code = 'IB'
        """), {'fecha': fecha, 'holding': tlt_value_eur, 'cash': cash_eur, 'total': total_eur})
        print(f"  [posicion] Actualizado: holding={tlt_value_eur:,.0f}, cash={cash_eur:,.0f}, total={total_eur:,.0f} EUR")

    return True

def sync_all_missing_dates(db):
    """Sincroniza IB para todas las fechas que tienen datos de otras cuentas pero no de IB."""
    with db.get_session() as session:
        # Obtener fechas con datos
        all_dates = get_all_dates_with_data(session)

        # Verificar cuáles faltan IB
        result = session.execute(text("""
            SELECT DISTINCT fecha FROM posicion WHERE account_code = 'IB'
        """))
        ib_dates = set(row[0] for row in result.fetchall())

        missing_dates = [d for d in all_dates if d not in ib_dates]

        if not missing_dates:
            print("Todas las fechas tienen datos de IB.")
            return

        print(f"Sincronizando {len(missing_dates)} fechas faltantes para IB...")

        for fecha in missing_dates:
            print(f"\n{fecha}:")
            sync_ib_for_date(session, fecha)

        session.commit()
        print(f"\nOK - Sincronizadas {len(missing_dates)} fechas")

def sync_specific_date(db, fecha_str):
    """Sincroniza IB para una fecha específica."""
    with db.get_session() as session:
        print(f"Sincronizando IB para {fecha_str}:")
        sync_ib_for_date(session, fecha_str)
        session.commit()
        print(f"\n✓ Sincronizado {fecha_str}")

def main():
    db = get_db()

    if len(sys.argv) > 1:
        # Fecha específica
        fecha = sys.argv[1]
        sync_specific_date(db, fecha)
    else:
        # Todas las fechas faltantes
        sync_all_missing_dates(db)

    # Mostrar resumen
    print("\n=== Resumen posicion por fecha (últimas 5) ===")
    with db.get_session() as session:
        result = session.execute(text("""
            SELECT fecha, SUM(total_eur) as total
            FROM posicion
            GROUP BY fecha
            ORDER BY fecha DESC
            LIMIT 5
        """))
        for row in result.fetchall():
            print(f"{row[0]}: {row[1]:,.0f} EUR")

if __name__ == '__main__':
    main()
