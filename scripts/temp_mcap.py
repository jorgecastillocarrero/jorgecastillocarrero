"""Rentabilidad por Market Cap"""
import sys
sys.path.insert(0, '.')
from src.portfolio_data import PortfolioDataService
from src.database import get_db_manager
from sqlalchemy import text
from datetime import date

db = get_db_manager()
portfolio_service = PortfolioDataService()
eur_usd_current = portfolio_service.get_eur_usd_rate(date.today())

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

EXCHANGE_TO_CURRENCY = {'US': 'USD', 'TO': 'CAD', 'MC': 'EUR', 'SW': 'CHF', 'L': 'GBP', 'DE': 'EUR'}

positions = []

with db.get_session() as session:
    # Abiertas
    hld = session.execute(text("""
        SELECT h.symbol, h.shares, c.precio
        FROM holding_diario h
        LEFT JOIN (
            SELECT account_code, symbol, MIN(fecha) as fecha,
                   (SELECT precio FROM compras c2
                    WHERE c2.account_code = c1.account_code AND c2.symbol = c1.symbol
                    ORDER BY fecha LIMIT 1) as precio
            FROM compras c1
            GROUP BY account_code, symbol
        ) c ON h.account_code = c.account_code AND h.symbol = c.symbol
        WHERE h.fecha = :f
        AND (h.asset_type IS NULL OR h.asset_type NOT IN ('ETF', 'ETFs', 'Future', 'Futures'))
    """), {'f': latest_date})

    for ticker, shares, precio_compra in hld.fetchall():
        if not precio_compra: continue
        parts = ticker.split('.')
        precio_actual = portfolio_service.get_symbol_price(ticker, latest_date)
        if not precio_actual:
            precio_actual = portfolio_service.get_symbol_price(parts[0], latest_date)
        if not precio_actual: continue

        exchange = parts[1] if len(parts) > 1 else 'US'
        currency = EXCHANGE_TO_CURRENCY.get(exchange, 'USD')
        fx = eur_usd_current if currency == 'USD' else 1.0

        pnl_eur = ((shares * precio_actual) - (shares * precio_compra)) / fx if fx > 1 else (shares * precio_actual) - (shares * precio_compra)
        rent_pct = ((precio_actual / precio_compra) - 1) * 100
        mcap = mcap_data.get(parts[0]) or mcap_data.get(ticker)

        positions.append({'ticker': parts[0], 'rent_pct': rent_pct, 'pnl_eur': pnl_eur, 'mcap': mcap})

    # Cerradas
    vtas = session.execute(text("""
        SELECT symbol, SUM(shares), SUM(importe_total)/SUM(shares), currency
        FROM ventas
        WHERE symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
        AND symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        GROUP BY symbol, currency
    """))
    compras = session.execute(text('SELECT symbol, AVG(precio) FROM compras GROUP BY symbol'))
    precios_compra = {r[0]: r[1] for r in compras.fetchall()}

    for symbol, shares, precio_venta, currency in vtas.fetchall():
        precio_compra = precios_compra.get(symbol)
        if not precio_compra: continue
        fx = eur_usd_current if currency == 'USD' else 1.0
        pnl = (precio_venta - precio_compra) * abs(shares)
        pnl_eur = pnl / fx if fx > 1 else pnl
        rent_pct = ((precio_venta / precio_compra) - 1) * 100
        ticker = symbol.split('.')[0]
        mcap = mcap_data.get(ticker) or mcap_data.get(symbol)
        positions.append({'ticker': ticker, 'rent_pct': rent_pct, 'pnl_eur': pnl_eur, 'mcap': mcap})

# Agrupar por market cap
stats = {'<5.000M': [], '5.000-10.000M': [], '10.000-50.000M': [], '>50.000M': [], 'Sin datos': []}
for p in positions:
    cat = get_cat(p['mcap'])
    stats[cat].append(p)

print('RENTABILIDAD POR MARKET CAP')
print('=' * 85)
print(f"{'Market Cap':<18} {'Ops':>5} {'Rent.Media':>12} {'Max':>10} {'Min':>10} {'P&L EUR':>14}")
print('-' * 85)
for cat in ['<5.000M', '5.000-10.000M', '10.000-50.000M', '>50.000M']:
    lst = stats[cat]
    if lst:
        avg = sum(p['rent_pct'] for p in lst) / len(lst)
        mx = max(p['rent_pct'] for p in lst)
        mn = min(p['rent_pct'] for p in lst)
        pnl = sum(p['pnl_eur'] for p in lst)
        print(f"{cat:<18} {len(lst):>5} {avg:>+11.2f}% {mx:>+9.1f}% {mn:>+9.1f}% {pnl:>+14,.0f}".replace(',', '.'))

if stats['Sin datos']:
    print()
    tickers = set(p['ticker'] for p in stats['Sin datos'])
    print(f"Sin datos market cap: {len(stats['Sin datos'])} ops - {', '.join(list(tickers)[:10])}...")
