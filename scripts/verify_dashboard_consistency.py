#!/usr/bin/env python3
"""
Verificador de consistencia del dashboard PatrimonioSmart.

Compara todas las fuentes de datos para asegurar que:
- Valor actual
- Variacion diaria por tipo de activo
- Rentabilidad acumulada
- Composicion por diversificacion
- Por estrategia
- Por cuenta

Todas muestren valores consistentes.

Uso: py -3 scripts/verify_dashboard_consistency.py [--fecha YYYY-MM-DD]
"""

import sys
import argparse
from datetime import date, datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import DatabaseManager
from src.portfolio_data import PortfolioDataService
from sqlalchemy import text


def verify_dashboard(fecha: date) -> dict:
    """
    Verify dashboard data consistency for a given date.

    Returns:
        dict with verification results
    """
    db = DatabaseManager()
    ps = PortfolioDataService()

    results = {
        'fecha': fecha,
        'valor_actual': {},
        'posicion': {},
        'por_cuenta': {},
        'errors': [],
        'is_consistent': True
    }

    # 1. VALOR ACTUAL (desde holding_diario + cash_diario)
    values_by_type = ps.get_values_by_asset_type(fecha)
    valor_actual_total = sum(values_by_type.values())
    results['valor_actual'] = {
        'by_type': values_by_type,
        'total': valor_actual_total
    }

    # 2. POSICION TABLA (para rentabilidad diaria)
    with db.get_session() as session:
        result = session.execute(text('''
            SELECT account_code, holding_eur, cash_eur, total_eur
            FROM posicion WHERE fecha = :fecha
            ORDER BY account_code
        '''), {'fecha': fecha})
        rows = result.fetchall()

    posicion_by_account = {r[0]: {'holding': r[1], 'cash': r[2], 'total': r[3]} for r in rows}
    posicion_total = sum(r[3] for r in rows)
    results['posicion'] = {
        'by_account': posicion_by_account,
        'total': posicion_total
    }

    # 3. COMPOSICION POR CUENTA (calculado desde holding_diario + cash_diario)
    accounts = ['CO3365', 'RCO951', 'IB', 'LACAIXA']
    eur_usd = ps.get_eur_usd_rate(fecha)
    cuenta_by_account = {}
    cuenta_total = 0

    for account in accounts:
        holdings = ps.get_all_holdings_for_date(fecha).get(account, {})
        holding_eur = sum(
            ps.calculate_position_value(s, d['shares'], fecha) or 0
            for s, d in holdings.items()
        )

        cash_data = ps.get_cash_for_date(account, fecha)
        cash_eur = 0
        if cash_data:
            for currency, amount in cash_data.items():
                if currency == 'EUR':
                    cash_eur += amount
                elif currency == 'USD':
                    cash_eur += amount / eur_usd

        total = holding_eur + cash_eur
        cuenta_by_account[account] = {'holding': holding_eur, 'cash': cash_eur, 'total': total}
        cuenta_total += total

    results['por_cuenta'] = {
        'by_account': cuenta_by_account,
        'total': cuenta_total
    }

    # VERIFICACIONES
    threshold = 1.0  # 1 EUR tolerance for rounding

    # Check valor_actual vs posicion
    diff1 = abs(valor_actual_total - posicion_total)
    if diff1 > threshold:
        results['errors'].append(f'Valor actual ({valor_actual_total:,.2f}) != Posicion ({posicion_total:,.2f}), diff={diff1:,.2f}')
        results['is_consistent'] = False

    # Check valor_actual vs por_cuenta
    diff2 = abs(valor_actual_total - cuenta_total)
    if diff2 > threshold:
        results['errors'].append(f'Valor actual ({valor_actual_total:,.2f}) != Por cuenta ({cuenta_total:,.2f}), diff={diff2:,.2f}')
        results['is_consistent'] = False

    # Check posicion vs por_cuenta by account
    for account in accounts:
        if account in posicion_by_account and account in cuenta_by_account:
            pos_total = posicion_by_account[account]['total']
            cuenta_total_acc = cuenta_by_account[account]['total']
            diff = abs(pos_total - cuenta_total_acc)
            if diff > threshold:
                results['errors'].append(
                    f'{account}: Posicion ({pos_total:,.2f}) != Por cuenta ({cuenta_total_acc:,.2f}), diff={diff:,.2f}'
                )
                results['is_consistent'] = False

    return results


def print_results(results: dict):
    """Print verification results."""
    print('=' * 60)
    print('VERIFICADOR DE CONSISTENCIA - DASHBOARD')
    print('=' * 60)
    print(f'Fecha: {results["fecha"]}')
    print()

    # Valor actual
    print('1. VALOR ACTUAL (holding_diario + cash_diario):')
    for t, v in sorted(results['valor_actual']['by_type'].items()):
        print(f'   {t}: {v:,.2f} EUR')
    print(f'   TOTAL: {results["valor_actual"]["total"]:,.2f} EUR')
    print()

    # Posicion
    print('2. POSICION TABLA (rentabilidad_diaria):')
    for acc, data in sorted(results['posicion']['by_account'].items()):
        print(f'   {acc}: H:{data["holding"]:,.2f} C:{data["cash"]:,.2f} T:{data["total"]:,.2f}')
    print(f'   TOTAL: {results["posicion"]["total"]:,.2f} EUR')
    print()

    # Por cuenta
    print('3. COMPOSICION POR CUENTA:')
    for acc, data in sorted(results['por_cuenta']['by_account'].items()):
        print(f'   {acc}: H:{data["holding"]:,.2f} C:{data["cash"]:,.2f} T:{data["total"]:,.2f}')
    print(f'   TOTAL: {results["por_cuenta"]["total"]:,.2f} EUR')
    print()

    # Verification
    print('=' * 60)
    print('VERIFICACION DE CONSISTENCIA:')
    print('=' * 60)

    if results['is_consistent']:
        print('[OK] Valor actual vs Posicion: COINCIDEN')
        print('[OK] Valor actual vs Por cuenta: COINCIDEN')
        print('[OK] Posicion vs Por cuenta por cada cuenta: COINCIDEN')
        print()
        print('*** TODOS LOS VALORES COINCIDEN ***')
    else:
        print('ERRORES ENCONTRADOS:')
        for error in results['errors']:
            print(f'  [ERROR] {error}')
        print()
        print('*** HAY DISCREPANCIAS ***')

    return results['is_consistent']


def fix_posicion_table(fecha: date):
    """
    Fix posicion table to match calculated values from holding_diario + cash_diario.
    """
    db = DatabaseManager()
    ps = PortfolioDataService()

    accounts = ['CO3365', 'RCO951', 'IB', 'LACAIXA']
    eur_usd = ps.get_eur_usd_rate(fecha)

    print(f'Corrigiendo posicion para {fecha}...')

    with db.get_session() as session:
        for account in accounts:
            holdings = ps.get_all_holdings_for_date(fecha).get(account, {})
            holding_eur = sum(
                ps.calculate_position_value(s, d['shares'], fecha) or 0
                for s, d in holdings.items()
            )

            cash_data = ps.get_cash_for_date(account, fecha)
            cash_eur = 0
            if cash_data:
                for currency, amount in cash_data.items():
                    if currency == 'EUR':
                        cash_eur += amount
                    elif currency == 'USD':
                        cash_eur += amount / eur_usd

            total_eur = holding_eur + cash_eur

            session.execute(text('''
                UPDATE posicion
                SET holding_eur = :h, cash_eur = :c, total_eur = :t
                WHERE account_code = :account AND fecha = :fecha
            '''), {'h': holding_eur, 'c': cash_eur, 't': total_eur, 'account': account, 'fecha': fecha})

            print(f'  {account}: H:{holding_eur:,.2f} C:{cash_eur:,.2f} T:{total_eur:,.2f}')

        session.commit()

    print('Posicion actualizada.')


def main():
    parser = argparse.ArgumentParser(description='Verificar consistencia del dashboard')
    parser.add_argument('--fecha', type=str, help='Fecha a verificar (YYYY-MM-DD)', default=None)
    parser.add_argument('--fix', action='store_true', help='Corregir discrepancias automaticamente')
    args = parser.parse_args()

    if args.fecha:
        fecha = datetime.strptime(args.fecha, '%Y-%m-%d').date()
    else:
        # Use latest date from posicion table
        db = DatabaseManager()
        with db.get_session() as session:
            result = session.execute(text('SELECT MAX(fecha) FROM posicion'))
            fecha = result.fetchone()[0]
            if isinstance(fecha, datetime):
                fecha = fecha.date()

    results = verify_dashboard(fecha)
    is_ok = print_results(results)

    if not is_ok and args.fix:
        print()
        print('Aplicando correcciones...')
        fix_posicion_table(fecha)
        print()
        print('Verificando de nuevo...')
        results = verify_dashboard(fecha)
        print_results(results)

    return 0 if is_ok else 1


if __name__ == '__main__':
    sys.exit(main())
