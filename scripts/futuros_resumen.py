import sys
sys.path.insert(0, '.')
from src.database import get_db_manager
from sqlalchemy import text
from datetime import date, datetime

db = get_db_manager()

with db.get_session() as session:
    # Get futures from holding_diario (symbols ending with futures pattern)
    result = session.execute(text("""
        SELECT DISTINCT ON (symbol, account_code)
            symbol, account_code, shares, precio_entrada, currency, fecha_compra, asset_type
        FROM holding_diario
        WHERE fecha = (SELECT MAX(fecha) FROM holding_diario)
        AND symbol SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        AND shares != 0
        ORDER BY symbol, account_code, fecha DESC
    """)).fetchall()

    print('=' * 100)
    print('POSICIONES FUTUROS ABIERTAS')
    print('=' * 100)
    print(f"{'Symbol':<12} {'Cuenta':<10} {'Contratos':>10} {'P.Entrada':>12} {'Moneda':<8} {'F.Compra':<12} {'Tipo':<15}")
    print('-' * 100)

    for row in result:
        symbol, cuenta, shares, precio, currency, fecha, tipo = row
        if isinstance(fecha, datetime):
            fecha = fecha.date()
        fecha_str = fecha.strftime('%d/%m/%Y') if fecha else '-'
        print(f"{symbol:<12} {cuenta:<10} {int(shares):>10} {precio:>12.2f} {currency:<8} {fecha_str:<12} {tipo or '-':<15}")

print()
print('=' * 100)
print('OPERACIONES CERRADAS FUTUROS')
print('=' * 100)

with db.get_session() as session:
    # Get closed futures from ventas
    result = session.execute(text("""
        SELECT fecha, symbol, account_code, shares, precio, importe_total, currency, pnl
        FROM ventas
        WHERE symbol SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        ORDER BY fecha DESC
    """)).fetchall()

    print(f"{'Fecha':<12} {'Symbol':<12} {'Cuenta':<10} {'Contratos':>10} {'Precio':>12} {'Importe':>14} {'P&L':>12}")
    print('-' * 100)

    total_pnl = 0
    for row in result:
        fecha, symbol, cuenta, shares, precio, importe, currency, pnl = row
        if isinstance(fecha, datetime):
            fecha = fecha.date()
        fecha_str = fecha.strftime('%d/%m/%Y') if fecha else '-'
        pnl_val = pnl or 0
        total_pnl += pnl_val
        importe_str = f"{importe:,.2f}".replace(',', '.')
        pnl_str = f"{pnl_val:+,.2f}".replace(',', '.')
        print(f"{fecha_str:<12} {symbol:<12} {cuenta:<10} {int(shares):>10} {precio:>12.2f} {importe_str:>14} {pnl_str:>12}")

    print('-' * 100)
    total_str = f"{total_pnl:+,.2f}".replace(',', '.')
    print(f"{'TOTAL P&L':>70} {total_str:>12}")

# Also check ib_futures_trades table
print()
print('=' * 100)
print('TRADES IB FUTUROS (ib_futures_trades)')
print('=' * 100)

with db.get_session() as session:
    result = session.execute(text("""
        SELECT trade_date, symbol, quantity, price, commission, realized_pnl, account_id
        FROM ib_futures_trades
        ORDER BY trade_date DESC
        LIMIT 50
    """)).fetchall()

    if result:
        print(f"{'Fecha':<12} {'Symbol':<12} {'Cantidad':>10} {'Precio':>12} {'Comision':>10} {'P&L':>12} {'Cuenta':<10}")
        print('-' * 100)

        for row in result:
            trade_date, symbol, qty, price, commission, pnl, account = row
            if isinstance(trade_date, datetime):
                trade_date = trade_date.date()
            fecha_str = trade_date.strftime('%d/%m/%Y') if trade_date else '-'
            pnl_val = pnl or 0
            comm_val = commission or 0
            print(f"{fecha_str:<12} {symbol:<12} {int(qty):>10} {price:>12.2f} {comm_val:>10.2f} {pnl_val:>+12.2f} {account or '-':<10}")
    else:
        print("No hay trades de futuros en ib_futures_trades")
