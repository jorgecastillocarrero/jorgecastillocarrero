"""Rentabilidad por Market Cap - PERIODO (desde 31/12/2025)"""
import sys
sys.path.insert(0, '.')
from src.portfolio_data import PortfolioDataService
from src.database import get_db_manager
from sqlalchemy import text
from datetime import date, datetime

db = get_db_manager()
portfolio_service = PortfolioDataService()

eur_usd_current = portfolio_service.get_eur_usd_rate(date.today())
eur_usd_31dic = portfolio_service.get_exchange_rate('EURUSD=X', date(2025, 12, 31)) or 1.1747
fecha_corte = date(2025, 12, 31)

with db.get_session() as session:
    latest_date = session.execute(text('SELECT MAX(fecha) FROM posicion')).scalar()
    if hasattr(latest_date, 'date'):
        latest_date = latest_date.date()

    # Market cap
    mcap_result = session.execute(text("""
        SELECT s.code, f.market_cap
        FROM fundamentals f
        JOIN symbols s ON f.symbol_id = s.id
        WHERE f.market_cap IS NOT NULL AND f.market_cap > 0
    """))
    mcap_data = {row[0]: row[1] for row in mcap_result.fetchall()}

def get_cat(mcap):
    if mcap is None: return 'Sin datos'
    m = mcap / 1_000_000
    if m < 5000: return '<5.000M'
    elif m < 10000: return '5.000-10.000M'
    elif m < 50000: return '10.000-50.000M'
    else: return '>50.000M'

def fmt_mcap(mcap):
    if mcap is None: return '-'
    m = mcap / 1_000_000_000
    return f'{m:.1f}B'

EXCHANGE_TO_CURRENCY = {'US': 'USD', 'TO': 'CAD', 'MC': 'EUR', 'SW': 'CHF', 'L': 'GBP', 'DE': 'EUR'}
positions = []

with db.get_session() as session:
    # ============ POSICIONES ABIERTAS ============
    hld = session.execute(text("""
        SELECT h.symbol, h.shares, c.fecha as fecha_compra, c.precio as precio_compra
        FROM holding_diario h
        LEFT JOIN (
            SELECT account_code, symbol, MIN(fecha) as fecha,
                   (SELECT precio FROM compras c2 WHERE c2.account_code = c1.account_code AND c2.symbol = c1.symbol ORDER BY fecha LIMIT 1) as precio
            FROM compras c1 GROUP BY account_code, symbol
        ) c ON h.account_code = c.account_code AND h.symbol = c.symbol
        WHERE h.fecha = :f AND (h.asset_type IS NULL OR h.asset_type NOT IN ('ETF', 'ETFs', 'Future', 'Futures'))
    """), {'f': latest_date})

    for ticker, shares, fecha_compra, precio_compra in hld.fetchall():
        if not precio_compra: continue
        parts = ticker.split('.')
        exchange = parts[1] if len(parts) > 1 else 'US'
        currency = EXCHANGE_TO_CURRENCY.get(exchange, 'USD')

        # Convertir fecha_compra
        if isinstance(fecha_compra, datetime):
            fecha_compra = fecha_compra.date()

        # Precio actual (último día mercado)
        precio_actual = portfolio_service.get_symbol_price(ticker, latest_date)
        if not precio_actual:
            precio_actual = portfolio_service.get_symbol_price(parts[0], latest_date)
        if not precio_actual: continue

        # Precio 31/12/2025
        precio_31dic = portfolio_service.get_symbol_price(ticker, fecha_corte)
        if not precio_31dic:
            precio_31dic = portfolio_service.get_symbol_price(parts[0], fecha_corte)

        # Determinar precio base para periodo
        if fecha_compra and fecha_compra > fecha_corte:
            # Comprada después del 31/12: usar precio compra
            precio_base = precio_compra
        else:
            # Existía a 31/12: usar precio 31/12
            precio_base = precio_31dic if precio_31dic else precio_compra

        # Calcular rentabilidad periodo
        rent_periodo = ((precio_actual / precio_base) - 1) * 100 if precio_base > 0 else 0

        # P&L EUR
        fx = eur_usd_current if currency == 'USD' else 1.0
        valor_actual_eur = (shares * precio_actual) / fx if fx > 1 else shares * precio_actual
        valor_base_eur = (shares * precio_base) / fx if fx > 1 else shares * precio_base
        pnl_eur = valor_actual_eur - valor_base_eur

        mcap = mcap_data.get(parts[0]) or mcap_data.get(ticker)
        positions.append({
            'ticker': parts[0],
            'rent_pct': rent_periodo,
            'pnl_eur': pnl_eur,
            'mcap': mcap,
            'estado': 'A',
            'base': 'compra' if (fecha_compra and fecha_compra > fecha_corte) else '31/12'
        })

    # ============ POSICIONES CERRADAS ============
    vtas = session.execute(text("""
        SELECT v.symbol, SUM(v.shares) as shares,
               SUM(v.importe_total)/SUM(v.shares) as precio_venta,
               v.currency,
               AVG(v.precio_31_12) as precio_31_12
        FROM ventas v
        WHERE v.symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
        AND v.symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        GROUP BY v.symbol, v.currency
    """))

    compras = session.execute(text('SELECT symbol, AVG(precio), MIN(fecha) FROM compras GROUP BY symbol'))
    compras_info = {r[0]: {'precio': r[1], 'fecha': r[2]} for r in compras.fetchall()}

    for symbol, shares, precio_venta, currency, precio_31_12_db in vtas.fetchall():
        compra = compras_info.get(symbol, {})
        precio_compra = compra.get('precio')
        fecha_compra = compra.get('fecha')
        if not precio_compra: continue

        if isinstance(fecha_compra, datetime):
            fecha_compra = fecha_compra.date()

        # Determinar precio base para periodo
        if fecha_compra and fecha_compra > fecha_corte:
            # Comprada después del 31/12: usar precio compra
            precio_base = precio_compra
        else:
            # Existía a 31/12: usar precio 31/12
            precio_base = precio_31_12_db if precio_31_12_db else precio_compra

        # Rentabilidad periodo
        rent_periodo = ((precio_venta / precio_base) - 1) * 100 if precio_base > 0 else 0

        fx = eur_usd_current if currency == 'USD' else 1.0
        pnl = (precio_venta - precio_base) * abs(shares)
        pnl_eur = pnl / fx if fx > 1 else pnl

        ticker = symbol.split('.')[0]
        mcap = mcap_data.get(ticker) or mcap_data.get(symbol)
        positions.append({
            'ticker': ticker,
            'rent_pct': rent_periodo,
            'pnl_eur': pnl_eur,
            'mcap': mcap,
            'estado': 'C',
            'base': 'compra' if (fecha_compra and fecha_compra > fecha_corte) else '31/12'
        })

# Agrupar y mostrar
stats = {'<5.000M': [], '5.000-10.000M': [], '10.000-50.000M': [], '>50.000M': [], 'Sin datos': []}
for p in positions:
    cat = get_cat(p['mcap'])
    stats[cat].append(p)

print(f"RENTABILIDAD POR MARKET CAP - PERIODO (desde 31/12/2025)")
print(f"Fecha inicio: 31/12/2025 | Fecha fin: {latest_date}")
print()

for cat in ['<5.000M', '5.000-10.000M', '10.000-50.000M', '>50.000M', 'Sin datos']:
    lst = stats[cat]
    if not lst: continue
    lst_sorted = sorted(lst, key=lambda x: -x['rent_pct'])
    print()
    print('=' * 70)
    print(f'{cat} ({len(lst)} operaciones)')
    print('=' * 70)
    print(f"{'Ticker':<10} {'Estado':<6} {'Base':<8} {'M.Cap':>10} {'Rent.%':>10} {'P&L EUR':>12}")
    print('-' * 70)
    for p in lst_sorted:
        print(f"{p['ticker']:<10} {p['estado']:<6} {p['base']:<8} {fmt_mcap(p['mcap']):>10} {p['rent_pct']:>+9.1f}% {p['pnl_eur']:>+12,.0f}".replace(',', '.'))

    # Totales
    avg = sum(p['rent_pct'] for p in lst) / len(lst)
    total_pnl = sum(p['pnl_eur'] for p in lst)
    print('-' * 70)
    print(f"{'TOTAL':<10} {len(lst):<6} {'':<8} {'':<10} {avg:>+9.1f}% {total_pnl:>+12,.0f}".replace(',', '.'))

# Resumen final
print()
print('=' * 70)
print('RESUMEN POR MARKET CAP')
print('=' * 70)
print(f"{'Market Cap':<18} {'Ops':>5} {'Rent.Media':>12} {'P&L EUR':>15}")
print('-' * 70)
total_ops = 0
total_pnl_all = 0
for cat in ['<5.000M', '5.000-10.000M', '10.000-50.000M', '>50.000M', 'Sin datos']:
    lst = stats[cat]
    if lst:
        avg = sum(p['rent_pct'] for p in lst) / len(lst)
        total_pnl = sum(p['pnl_eur'] for p in lst)
        print(f"{cat:<18} {len(lst):>5} {avg:>+11.1f}% {total_pnl:>+15,.0f}".replace(',', '.'))
        total_ops += len(lst)
        total_pnl_all += total_pnl
print('-' * 70)
print(f"{'TOTAL':<18} {total_ops:>5} {'':<12} {total_pnl_all:>+15,.0f}".replace(',', '.'))
