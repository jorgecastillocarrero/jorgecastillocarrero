"""Rentabilidad por Market Cap - PERIODO 2026 (con tipos de cambio correctos)"""
import sys
sys.path.insert(0, '.')
from src.portfolio_data import PortfolioDataService
from src.database import get_db_manager
from sqlalchemy import text
from datetime import date

db = get_db_manager()
ps = PortfolioDataService()
fecha_corte = date(2025, 12, 31)

with db.get_session() as session:
    latest = session.execute(text('SELECT MAX(fecha) FROM posicion')).scalar()
    if hasattr(latest, 'date'):
        latest = latest.date()

    # Tipos de cambio actuales
    eur_usd_current = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 1.1871

    cad_eur_current = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CADEUR=X' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 0.619

    chf_eur_current = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CHFEUR=X' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 1.06

    # Tipos de cambio 31/12/2025
    eur_usd_31dic = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X' AND ph.date = '2025-12-31'
    ''')).scalar() or 1.1747

    cad_eur_31dic = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CADEUR=X' AND ph.date = '2025-12-31'
    ''')).scalar() or 0.6215

    chf_eur_31dic = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CHFEUR=X' AND ph.date = '2025-12-31'
    ''')).scalar() or 1.06

    # Market cap data
    mcap_result = session.execute(text('''
        SELECT s.code, f.market_cap FROM fundamentals f
        JOIN symbols s ON f.symbol_id = s.id
        WHERE f.market_cap IS NOT NULL AND f.market_cap > 0
    '''))
    mcap_data = {row[0]: row[1] for row in mcap_result.fetchall()}

print(f'Tipos de cambio 31/12/2025: EUR/USD={eur_usd_31dic:.4f}, CAD/EUR={cad_eur_31dic:.4f}, CHF/EUR={chf_eur_31dic:.4f}')
print(f'Tipos de cambio actuales:   EUR/USD={eur_usd_current:.4f}, CAD/EUR={cad_eur_current:.4f}, CHF/EUR={chf_eur_current:.4f}')
print()

def get_cat(mcap):
    if mcap is None: return 'Sin datos'
    m = mcap / 1_000_000
    if m < 5000: return '<5.000M'
    elif m < 10000: return '5.000-10.000M'
    elif m < 50000: return '10.000-50.000M'
    else: return '>50.000M'

EXCHANGE_TO_CURRENCY = {'US': 'USD', 'TO': 'CAD', 'MC': 'EUR', 'SW': 'CHF', 'L': 'GBP', 'DE': 'EUR', 'MI': 'EUR'}
positions = []

# ABIERTAS
with db.get_session() as session:
    holdings = session.execute(text('''
        SELECT h.symbol, h.shares, c.precio as p_compra, c.fecha as f_compra
        FROM holding_diario h
        LEFT JOIN (
            SELECT account_code, symbol, MIN(fecha) as fecha,
                   (SELECT precio FROM compras c2 WHERE c2.account_code = c1.account_code AND c2.symbol = c1.symbol ORDER BY fecha LIMIT 1) as precio
            FROM compras c1 GROUP BY account_code, symbol
        ) c ON h.account_code = c.account_code AND h.symbol = c.symbol
        WHERE h.fecha = :f
        AND (h.asset_type IS NULL OR h.asset_type NOT IN ('ETF', 'ETFs', 'Future', 'Futures'))
        AND h.symbol NOT LIKE '%SGLE%'
    '''), {'f': latest}).fetchall()

    for ticker, shares, p_compra, f_compra in holdings:
        if not p_compra: continue
        parts = ticker.split('.')
        exchange = parts[1] if len(parts) > 1 else 'US'
        currency = EXCHANGE_TO_CURRENCY.get(exchange, 'USD')

        if hasattr(f_compra, 'date'):
            f_compra = f_compra.date()

        p_actual = ps.get_symbol_price(ticker, latest)
        if not p_actual:
            p_actual = ps.get_symbol_price(parts[0], latest)
        if not p_actual: continue

        p_31dic = ps.get_symbol_price(ticker, fecha_corte)
        if not p_31dic:
            p_31dic = ps.get_symbol_price(parts[0], fecha_corte)

        # Determinar precio base
        if f_compra and f_compra > fecha_corte:
            precio_base = p_compra
            base_tipo = 'Compra'
        else:
            precio_base = p_31dic if p_31dic else p_compra
            base_tipo = '31/12'

        # Calcular valor EUR usando tipos de cambio correctos
        if currency == 'USD':
            if base_tipo == '31/12':
                valor_base_eur = (shares * precio_base) / eur_usd_31dic
            else:
                valor_base_eur = (shares * precio_base) / eur_usd_current  # aproximación
            valor_actual_eur = (shares * p_actual) / eur_usd_current
        elif currency == 'CAD':
            if base_tipo == '31/12':
                valor_base_eur = (shares * precio_base) * cad_eur_31dic
            else:
                valor_base_eur = (shares * precio_base) * cad_eur_current
            valor_actual_eur = (shares * p_actual) * cad_eur_current
        elif currency == 'CHF':
            if base_tipo == '31/12':
                valor_base_eur = (shares * precio_base) * chf_eur_31dic
            else:
                valor_base_eur = (shares * precio_base) * chf_eur_current
            valor_actual_eur = (shares * p_actual) * chf_eur_current
        else:  # EUR
            valor_base_eur = shares * precio_base
            valor_actual_eur = shares * p_actual

        pnl_eur = valor_actual_eur - valor_base_eur
        rent_periodo = ((valor_actual_eur / valor_base_eur) - 1) * 100 if valor_base_eur > 0 else 0

        mcap = mcap_data.get(parts[0]) or mcap_data.get(ticker)
        positions.append({
            'ticker': parts[0],
            'estado': 'Abierta',
            'base_tipo': base_tipo,
            'p_base': precio_base,
            'p_final': p_actual,
            'rent_pct': rent_periodo,
            'pnl_eur': pnl_eur,
            'mcap': mcap,
            'currency': currency
        })

# CERRADAS
with db.get_session() as session:
    ventas = session.execute(text('''
        SELECT v.symbol, SUM(v.shares) as shares,
               SUM(v.importe_total)/SUM(v.shares) as precio_venta, v.currency,
               AVG(v.precio_31_12) as p31_12,
               (SELECT AVG(c.precio) FROM compras c WHERE c.symbol = v.symbol) as p_compra,
               (SELECT MIN(c.fecha) FROM compras c WHERE c.symbol = v.symbol) as f_compra
        FROM ventas v
        WHERE v.symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
        AND v.symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        AND v.symbol NOT LIKE '%SGLE%'
        GROUP BY v.symbol, v.currency
    ''')).fetchall()

    for symbol, shares, precio_venta, currency, p31_12, p_compra, f_compra in ventas:
        if hasattr(f_compra, 'date'):
            f_compra = f_compra.date()

        if f_compra and f_compra > fecha_corte:
            precio_base = p_compra if p_compra else precio_venta
            base_tipo = 'Compra'
        else:
            precio_base = p31_12 if p31_12 else (p_compra if p_compra else precio_venta)
            base_tipo = '31/12' if p31_12 else 'Compra'

        # Calcular valor EUR
        if currency == 'USD':
            if base_tipo == '31/12':
                valor_base_eur = (abs(shares) * precio_base) / eur_usd_31dic
            else:
                valor_base_eur = (abs(shares) * precio_base) / eur_usd_current
            valor_venta_eur = (abs(shares) * precio_venta) / eur_usd_current
        elif currency == 'CAD':
            if base_tipo == '31/12':
                valor_base_eur = (abs(shares) * precio_base) * cad_eur_31dic
            else:
                valor_base_eur = (abs(shares) * precio_base) * cad_eur_current
            valor_venta_eur = (abs(shares) * precio_venta) * cad_eur_current
        else:  # EUR
            valor_base_eur = abs(shares) * precio_base
            valor_venta_eur = abs(shares) * precio_venta

        pnl_eur = valor_venta_eur - valor_base_eur
        rent_periodo = ((valor_venta_eur / valor_base_eur) - 1) * 100 if valor_base_eur > 0 else 0

        ticker = symbol.split('.')[0]
        mcap = mcap_data.get(ticker) or mcap_data.get(symbol)
        positions.append({
            'ticker': ticker,
            'estado': 'Cerrada',
            'base_tipo': base_tipo,
            'p_base': precio_base,
            'p_final': precio_venta,
            'rent_pct': rent_periodo,
            'pnl_eur': pnl_eur,
            'mcap': mcap,
            'currency': currency if currency else 'USD'
        })

# Ordenar por ticker
positions.sort(key=lambda x: x['ticker'])

print(f'DESGLOSE RENTABILIDAD POR ACCION - PERIODO 2026 (CORREGIDO)')
print(f'Fecha base: 31/12/2025 | Fecha fin: {latest}')
print()
print(f"{'Ticker':<10} {'Estado':<8} {'Base':<7} {'Mon':<4} {'P.Base':>10} {'P.Final':>10} {'Rent.%':>8} {'P&L EUR':>12}")
print('-' * 80)

total_pnl = 0
for p in positions:
    print(f"{p['ticker']:<10} {p['estado']:<8} {p['base_tipo']:<7} {p['currency']:<4} {p['p_base']:>10.2f} {p['p_final']:>10.2f} {p['rent_pct']:>+7.1f}% {p['pnl_eur']:>+12,.0f}".replace(',', '.'))
    total_pnl += p['pnl_eur']

print('-' * 80)
print(f"{'TOTAL':<10} {len(positions):<8} {'':<7} {'':<4} {'':<10} {'':<10} {'':<8} {total_pnl:>+12,.0f}".replace(',', '.'))

# Resumen por Market Cap
print()
print('=' * 60)
print('RESUMEN POR MARKET CAP')
print('=' * 60)
stats = {'<5.000M': [], '5.000-10.000M': [], '10.000-50.000M': [], '>50.000M': [], 'Sin datos': []}
for p in positions:
    cat = get_cat(p['mcap'])
    stats[cat].append(p)

print(f"{'Market Cap':<18} {'Ops':>5} {'Rent.Media':>12} {'P&L EUR':>15}")
print('-' * 55)
for cat in ['<5.000M', '5.000-10.000M', '10.000-50.000M', '>50.000M', 'Sin datos']:
    lst = stats[cat]
    if lst:
        avg = sum(p['rent_pct'] for p in lst) / len(lst)
        pnl = sum(p['pnl_eur'] for p in lst)
        print(f"{cat:<18} {len(lst):>5} {avg:>+11.1f}% {pnl:>+15,.0f}".replace(',', '.'))
print('-' * 55)
print(f"{'TOTAL':<18} {len(positions):>5} {'':<12} {total_pnl:>+15,.0f}".replace(',', '.'))
