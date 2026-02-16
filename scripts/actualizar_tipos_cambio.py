"""Actualizar tipos de cambio en price_history"""
import sys
sys.path.insert(0, '.')
import yfinance as yf
from src.database import get_db_manager
from sqlalchemy import text
from datetime import date
import pandas as pd

db = get_db_manager()

# Configuración de pares y fechas
pares_config = [
    ('EURUSD=X', '2020-01-01'),
    ('CADEUR=X', '2025-01-01'),
    ('CHFEUR=X', '2025-01-01'),
    ('GBPEUR=X', '2025-01-01'),
]

for par, fecha_inicio in pares_config:
    print(f'Descargando {par} desde {fecha_inicio}...')
    try:
        data = yf.download(par, start=fecha_inicio, end='2026-02-15', progress=False)
        if data.empty:
            print(f'  Sin datos para {par}')
            continue

        print(f'  {len(data)} registros descargados')

        with db.get_session() as session:
            # Obtener o crear symbol_id
            symbol_result = session.execute(text('SELECT id FROM symbols WHERE code = :code'), {'code': par}).fetchone()
            if not symbol_result:
                # Insertar sin especificar id
                session.execute(text('INSERT INTO symbols (code, name) VALUES (:code, :name) ON CONFLICT (code) DO NOTHING'), {'code': par, 'name': par})
                session.commit()
                symbol_result = session.execute(text('SELECT id FROM symbols WHERE code = :code'), {'code': par}).fetchone()

            symbol_id = symbol_result[0]
            inserted = 0

            for idx, row in data.iterrows():
                fecha = idx.date() if hasattr(idx, 'date') else idx
                # Manejar MultiIndex de yfinance
                if hasattr(row, 'index') and isinstance(row.index, pd.MultiIndex):
                    close = float(row['Close'].iloc[0])
                else:
                    close = float(row['Close'])

                # Insertar en price_history (usando UPSERT)
                session.execute(text('''
                    INSERT INTO price_history (symbol_id, date, open, high, low, close, volume)
                    VALUES (:sid, :d, :c, :c, :c, :c, 0)
                    ON CONFLICT (symbol_id, date) DO UPDATE SET close = :c
                '''), {'sid': symbol_id, 'd': fecha, 'c': close})
                inserted += 1

            session.commit()
            print(f'  {inserted} registros insertados/actualizados')

    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()

print()
print('Tipos de cambio actualizados.')
