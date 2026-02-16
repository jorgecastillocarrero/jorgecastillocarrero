"""Rentabilidad por Sector vs Benchmark ETF - PERIODO 2026"""
import sys
sys.path.insert(0, '.')
from src.portfolio_data import PortfolioDataService
from src.database import get_db_manager
from sqlalchemy import text
from datetime import date

db = get_db_manager()
ps = PortfolioDataService()
fecha_corte = date(2025, 12, 31)

# ETFs benchmark por sector (SPDR Select Sector)
SECTOR_ETF = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial Services': 'XLF',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Basic Materials': 'XLB',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
}

with db.get_session() as session:
    latest = session.execute(text('SELECT MAX(fecha) FROM posicion')).scalar()
    if hasattr(latest, 'date'):
        latest = latest.date()

    # Tipos de cambio
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

    # Sector data
    sector_result = session.execute(text('''
        SELECT s.code, f.sector FROM fundamentals f
        JOIN symbols s ON f.symbol_id = s.id
        WHERE f.sector IS NOT NULL AND f.sector != ''
    '''))
    sector_data = {row[0]: row[1] for row in sector_result.fetchall()}

# Calcular rentabilidad de cada ETF benchmark
print(f'RENTABILIDAD ETF BENCHMARK - Periodo 2026')
print(f'Fecha base: 31/12/2025 | Fecha fin: {latest}')
print()
print(f"{'ETF':<6} {'Sector':<25} {'P.31/12':>10} {'P.Actual':>10} {'Rent.%':>8}")
print('-' * 65)

etf_returns = {}
for sector, etf in SECTOR_ETF.items():
    p_31dic = ps.get_symbol_price(etf, fecha_corte)
    p_actual = ps.get_symbol_price(etf, latest)

    if p_31dic and p_actual:
        rent = ((p_actual / p_31dic) - 1) * 100
        etf_returns[sector] = rent
        print(f"{etf:<6} {sector:<25} {p_31dic:>10.2f} {p_actual:>10.2f} {rent:>+7.1f}%")
    else:
        etf_returns[sector] = None
        print(f"{etf:<6} {sector:<25} {'N/A':>10} {'N/A':>10} {'N/A':>8}")

print()

# Ahora calcular rentabilidad de la cartera por sector (mismo código anterior)
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

        if f_compra and f_compra > fecha_corte:
            precio_base = p_compra
            base_tipo = 'Compra'
        else:
            precio_base = p_31dic if p_31dic else p_compra
            base_tipo = '31/12'

        if currency == 'USD':
            if base_tipo == '31/12':
                valor_base_eur = (shares * precio_base) / eur_usd_31dic
            else:
                valor_base_eur = (shares * precio_base) / eur_usd_current
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
        else:
            valor_base_eur = shares * precio_base
            valor_actual_eur = shares * p_actual

        pnl_eur = valor_actual_eur - valor_base_eur
        rent_periodo = ((valor_actual_eur / valor_base_eur) - 1) * 100 if valor_base_eur > 0 else 0

        sector = sector_data.get(parts[0]) or sector_data.get(ticker)
        positions.append({
            'ticker': parts[0],
            'rent_pct': rent_periodo,
            'pnl_eur': pnl_eur,
            'sector': sector,
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
        else:
            valor_base_eur = abs(shares) * precio_base
            valor_venta_eur = abs(shares) * precio_venta

        pnl_eur = valor_venta_eur - valor_base_eur
        rent_periodo = ((valor_venta_eur / valor_base_eur) - 1) * 100 if valor_base_eur > 0 else 0

        ticker = symbol.split('.')[0]
        sector = sector_data.get(ticker) or sector_data.get(symbol)
        positions.append({
            'ticker': ticker,
            'rent_pct': rent_periodo,
            'pnl_eur': pnl_eur,
            'sector': sector,
        })

# Agrupar por sector
stats = {}
for p in positions:
    sector = p['sector'] or 'Sin datos'
    if sector not in stats:
        stats[sector] = []
    stats[sector].append(p)

# Comparativa
print('=' * 85)
print('COMPARATIVA: CARTERA vs BENCHMARK ETF')
print('=' * 85)
print(f"{'Sector':<25} {'Ops':>4} {'Cartera':>10} {'Benchmark':>10} {'Alpha':>10} {'P&L EUR':>15}")
print('-' * 85)

total_pnl = 0
total_alpha_ponderado = 0
total_peso = 0

sorted_sectors = sorted(stats.items(), key=lambda x: sum(p['pnl_eur'] for p in x[1]), reverse=True)

for sector, lst in sorted_sectors:
    avg_cartera = sum(p['rent_pct'] for p in lst) / len(lst)
    pnl = sum(p['pnl_eur'] for p in lst)
    total_pnl += pnl

    benchmark = etf_returns.get(sector)
    if benchmark is not None:
        alpha = avg_cartera - benchmark
        benchmark_str = f"{benchmark:>+9.1f}%"
        alpha_str = f"{alpha:>+9.1f}%"
        # Ponderar alpha por número de operaciones
        total_alpha_ponderado += alpha * len(lst)
        total_peso += len(lst)
    else:
        benchmark_str = "N/A"
        alpha_str = "N/A"

    print(f"{sector:<25} {len(lst):>4} {avg_cartera:>+9.1f}% {benchmark_str:>10} {alpha_str:>10} {pnl:>+15,.0f}".replace(',', '.'))

print('-' * 85)
alpha_medio = total_alpha_ponderado / total_peso if total_peso > 0 else 0
print(f"{'TOTAL':<25} {len(positions):>4} {'':<10} {'':<10} {alpha_medio:>+9.1f}% {total_pnl:>+15,.0f}".replace(',', '.'))

print()
print('Nota: Alpha = Rentabilidad Cartera - Rentabilidad Benchmark ETF')
print('      Benchmark ETFs: SPDR Select Sector (XLK, XLV, XLF, XLY, XLP, XLI, XLB, XLE, XLU, XLRE, XLC)')
