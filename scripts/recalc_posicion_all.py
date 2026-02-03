"""
Script para recalcular la tabla posicion para todas las cuentas.
Lee holding_diario y cash_diario, obtiene precios actuales de price_history,
y actualiza/inserta en posicion.

Uso:
    python scripts/recalc_posicion_all.py              # Recalcula fechas faltantes
    python scripts/recalc_posicion_all.py 2026-01-30   # Recalcula una fecha específica
    python scripts/recalc_posicion_all.py --all        # Recalcula TODAS las fechas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timedelta
from sqlalchemy import text
from src.database import get_db_manager

# Cuentas a procesar
ACCOUNTS = ['CO3365', 'RCO951', 'LACAIXA', 'IB']


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
    return row[0] if row else 1.04  # Default


def get_cad_eur_rate(session, fecha):
    """Obtiene el tipo de cambio CAD/EUR para una fecha."""
    result = session.execute(text("""
        SELECT ph.close
        FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CADEUR=X' AND DATE(ph.date) <= :fecha
        ORDER BY ph.date DESC
        LIMIT 1
    """), {'fecha': fecha})
    row = result.fetchone()
    return row[0] if row else 0.67  # Default


def get_chf_eur_rate(session, fecha):
    """Obtiene el tipo de cambio CHF/EUR para una fecha."""
    result = session.execute(text("""
        SELECT ph.close
        FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CHFEUR=X' AND DATE(ph.date) <= :fecha
        ORDER BY ph.date DESC
        LIMIT 1
    """), {'fecha': fecha})
    row = result.fetchone()
    return row[0] if row else 1.05  # Default


def get_symbol_price(session, symbol, fecha):
    """Obtiene el precio de cierre de un símbolo para una fecha."""
    # Intentar fecha exacta primero, luego buscar hacia atrás hasta 5 días
    for days_back in range(6):
        check_date = (datetime.strptime(fecha, '%Y-%m-%d') - timedelta(days=days_back)).strftime('%Y-%m-%d')
        result = session.execute(text("""
            SELECT ph.close
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE s.code = :symbol AND DATE(ph.date) = :fecha
            LIMIT 1
        """), {'symbol': symbol, 'fecha': check_date})
        row = result.fetchone()
        if row:
            return row[0]
    return None


def get_source_currency(symbol):
    """Determina la moneda de origen basándose en el símbolo."""
    if '.TO' in symbol:
        return 'CAD'
    elif '.MC' in symbol or '.MI' in symbol:
        return 'EUR'
    elif '.SW' in symbol:
        return 'CHF'
    else:
        return 'USD'


def calculate_position_value_eur(session, symbol, shares, fecha, eur_usd, cad_eur, chf_eur):
    """Calcula el valor de una posición en EUR."""
    price = get_symbol_price(session, symbol, fecha)
    if price is None:
        return None

    value_local = price * shares
    source_currency = get_source_currency(symbol)

    if source_currency == 'EUR':
        return value_local
    elif source_currency == 'USD':
        return value_local / eur_usd
    elif source_currency == 'CAD':
        return value_local * cad_eur
    elif source_currency == 'CHF':
        return value_local * chf_eur

    return value_local


def get_holdings_for_account(session, account, fecha):
    """Obtiene todos los holdings de una cuenta para una fecha."""
    result = session.execute(text("""
        SELECT symbol, shares, currency
        FROM holding_diario
        WHERE account_code = :account AND fecha = :fecha
    """), {'account': account, 'fecha': fecha})
    return result.fetchall()


def get_cash_for_account(session, account, fecha):
    """Obtiene el cash de una cuenta para una fecha, retorna dict {currency: amount}."""
    result = session.execute(text("""
        SELECT currency, saldo
        FROM cash_diario
        WHERE account_code = :account AND fecha = :fecha
    """), {'account': account, 'fecha': fecha})
    return {row[0]: row[1] for row in result.fetchall()}


def recalc_posicion_for_date(session, fecha, verbose=True):
    """Recalcula posicion para todas las cuentas en una fecha."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Procesando fecha: {fecha}")
        print('='*60)

    # Obtener tipos de cambio
    eur_usd = get_eur_usd_rate(session, fecha)
    cad_eur = get_cad_eur_rate(session, fecha)
    chf_eur = get_chf_eur_rate(session, fecha)

    if verbose:
        print(f"Tipos de cambio: EUR/USD={eur_usd:.4f}, CAD/EUR={cad_eur:.4f}, CHF/EUR={chf_eur:.4f}")

    total_general = 0

    for account in ACCOUNTS:
        # Calcular valor de holdings
        holdings = get_holdings_for_account(session, account, fecha)
        holding_eur = 0
        missing_prices = []

        for symbol, shares, currency in holdings:
            value = calculate_position_value_eur(
                session, symbol, shares, fecha, eur_usd, cad_eur, chf_eur
            )
            if value:
                holding_eur += value
            else:
                missing_prices.append(symbol)

        # Calcular cash en EUR
        cash = get_cash_for_account(session, account, fecha)
        cash_eur = cash.get('EUR', 0)
        cash_usd = cash.get('USD', 0)
        cash_eur += cash_usd / eur_usd  # Convertir USD a EUR

        # Total
        total_eur = holding_eur + cash_eur
        total_general += total_eur

        if verbose:
            print(f"\n{account}:")
            print(f"  Holdings: {len(holdings)} posiciones = {holding_eur:,.0f} EUR")
            if missing_prices:
                print(f"  [!] Sin precio: {', '.join(missing_prices[:5])}")
            print(f"  Cash: {cash_eur:,.0f} EUR")
            print(f"  Total: {total_eur:,.0f} EUR")

        # Insertar/actualizar en posicion
        result = session.execute(text("""
            SELECT id FROM posicion WHERE fecha = :fecha AND account_code = :account
        """), {'fecha': fecha, 'account': account})
        exists = result.fetchone()

        if exists:
            session.execute(text("""
                UPDATE posicion
                SET holding_eur = :holding, cash_eur = :cash, total_eur = :total
                WHERE fecha = :fecha AND account_code = :account
            """), {
                'fecha': fecha, 'account': account,
                'holding': holding_eur, 'cash': cash_eur, 'total': total_eur
            })
        else:
            session.execute(text("""
                INSERT INTO posicion (fecha, account_code, holding_eur, cash_eur, total_eur)
                VALUES (:fecha, :account, :holding, :cash, :total)
            """), {
                'fecha': fecha, 'account': account,
                'holding': holding_eur, 'cash': cash_eur, 'total': total_eur
            })

    if verbose:
        print(f"\n{'-'*40}")
        print(f"TOTAL GENERAL: {total_general:,.0f} EUR")

    return total_general


def get_dates_in_holding_diario(session):
    """Obtiene todas las fechas distintas en holding_diario."""
    result = session.execute(text("""
        SELECT DISTINCT fecha FROM holding_diario ORDER BY fecha
    """))
    return [row[0] for row in result.fetchall()]


def get_dates_in_posicion(session):
    """Obtiene todas las fechas distintas en posicion."""
    result = session.execute(text("""
        SELECT DISTINCT fecha FROM posicion ORDER BY fecha
    """))
    return set(row[0] for row in result.fetchall())


def recalc_missing_dates(db):
    """Recalcula posicion para fechas que están en holding_diario pero no en posicion."""
    with db.get_session() as session:
        holding_dates = get_dates_in_holding_diario(session)
        posicion_dates = get_dates_in_posicion(session)

        missing = [d for d in holding_dates if d not in posicion_dates]

        if not missing:
            print("[OK] Todas las fechas de holding_diario ya tienen datos en posicion.")
            return

        print(f"Encontradas {len(missing)} fechas sin datos en posicion:")
        for d in missing:
            print(f"  - {d}")

        for fecha in missing:
            recalc_posicion_for_date(session, fecha)

        session.commit()
        print(f"\n[OK] Actualizadas {len(missing)} fechas")


def recalc_all_dates(db):
    """Recalcula posicion para TODAS las fechas en holding_diario."""
    with db.get_session() as session:
        dates = get_dates_in_holding_diario(session)

        print(f"Recalculando {len(dates)} fechas...")

        for fecha in dates:
            recalc_posicion_for_date(session, fecha, verbose=False)
            print(f"  [OK] {fecha}")

        session.commit()
        print(f"\n[OK] Actualizadas {len(dates)} fechas")


def recalc_specific_date(db, fecha_str):
    """Recalcula posicion para una fecha específica."""
    with db.get_session() as session:
        recalc_posicion_for_date(session, fecha_str)
        session.commit()
        print(f"\n[OK] Actualizada fecha {fecha_str}")


def show_summary(db):
    """Muestra resumen de las últimas fechas en posicion."""
    print("\n=== Resumen posicion (últimas 5 fechas) ===")
    with db.get_session() as session:
        result = session.execute(text("""
            SELECT fecha, SUM(total_eur) as total
            FROM posicion
            GROUP BY fecha
            ORDER BY fecha DESC
            LIMIT 5
        """))
        for row in result.fetchall():
            print(f"  {row[0]}: {row[1]:,.0f} EUR")


def main():
    db = get_db_manager()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--all':
            recalc_all_dates(db)
        elif arg == '--help':
            print(__doc__)
            return
        else:
            # Fecha específica
            recalc_specific_date(db, arg)
    else:
        # Fechas faltantes
        recalc_missing_dates(db)

    show_summary(db)


if __name__ == '__main__':
    main()
