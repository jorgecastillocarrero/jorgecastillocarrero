"""
Script para corregir los asset_type de holding_diario usando la tabla asset_types.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from sqlalchemy import text
from src.database import get_db_manager

def fix_asset_types(target_date=None):
    """Actualiza asset_type desde la tabla asset_types para una fecha específica."""
    if target_date is None:
        target_date = date.today()

    db = get_db_manager()

    with db.get_session() as session:
        # Mostrar estado antes
        result = session.execute(text("""
            SELECT asset_type, COUNT(*) as count
            FROM holding_diario
            WHERE fecha = :fecha
            GROUP BY asset_type
        """), {'fecha': target_date})

        print(f"\n=== ESTADO ANTES ({target_date}) ===")
        for row in result.fetchall():
            print(f"  {row[0] or 'NULL':20} : {row[1]} registros")

        # Ejecutar UPDATE - busca primero CUENTA:SYMBOL, luego SYMBOL generico
        result = session.execute(text("""
            UPDATE holding_diario
            SET asset_type = COALESCE(
                (SELECT at.asset_type FROM asset_types at
                 WHERE at.symbol = holding_diario.account_code || ':' || holding_diario.symbol),
                (SELECT at.asset_type FROM asset_types at
                 WHERE at.symbol = holding_diario.symbol),
                asset_type
            )
            WHERE fecha = :fecha
            AND (asset_type IS NULL OR asset_type = 'Otros')
            AND EXISTS (
                SELECT 1 FROM asset_types
                WHERE symbol = holding_diario.account_code || ':' || holding_diario.symbol
                   OR symbol = holding_diario.symbol
            )
        """), {'fecha': target_date})

        updated = result.rowcount
        print(f"\n=== ACTUALIZADOS: {updated} registros ===")

        # Mostrar estado después
        result = session.execute(text("""
            SELECT asset_type, COUNT(*) as count
            FROM holding_diario
            WHERE fecha = :fecha
            GROUP BY asset_type
        """), {'fecha': target_date})

        print(f"\n=== ESTADO DESPUÉS ({target_date}) ===")
        for row in result.fetchall():
            print(f"  {row[0] or 'NULL':20} : {row[1]} registros")

        # Commit
        session.commit()
        print("\n✓ Cambios guardados")

if __name__ == "__main__":
    fix_asset_types()
