"""
Análisis diario por tipo de activo.
Cash en tránsito: 190,000 EUR (fijo para todos los días)
Días confirmados:
  - 02/01/2026: CORRECTO
  - 05/01/2026: CORRECTO
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_db_manager, Symbol, AccountHolding, AccountCash, PortfolioSnapshot
from src.portfolio_data import get_portfolio_service, ASSET_TYPE_MAP
from datetime import date

CASH_EN_TRANSITO = 190000  # EUR fijo

db = get_db_manager()
portfolio_service = get_portfolio_service(db)

# Días a analizar (cambiar según se necesite)
trading_dates = [
    date(2025, 12, 31),
    date(2026, 1, 2),
    date(2026, 1, 5),
    date(2026, 1, 6),
    date(2026, 1, 7),
    date(2026, 1, 8),
    date(2026, 1, 9),
    date(2026, 1, 12),
    date(2026, 1, 13),
    date(2026, 1, 14),
    date(2026, 1, 15),
    date(2026, 1, 16),
    date(2026, 1, 20),
    date(2026, 1, 21),
    date(2026, 1, 22),
    date(2026, 1, 23),
    date(2026, 1, 26),
    date(2026, 1, 27),
]

asset_types_order = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/Monetario']

with db.get_session() as session:
    holdings = session.query(AccountHolding).all()
    cash_entries = session.query(AccountCash).all()

    all_data = {}

    for td in trading_dates:
        eur_usd = portfolio_service.get_symbol_price('EURUSD=X', td) or 1.035
        cad_eur = portfolio_service.get_symbol_price('CADEUR=X', td)
        chf_eur = portfolio_service.get_symbol_price('CHFEUR=X', td)

        type_values = {t: 0.0 for t in asset_types_order}

        for holding in holdings:
            symbol = holding.symbol
            shares = holding.shares
            asset_type = ASSET_TYPE_MAP.get(symbol, 'Otros')
            if asset_type not in type_values:
                type_values[asset_type] = 0.0

            price = portfolio_service.get_symbol_price(symbol, td)
            if price is None:
                continue

            value = price * shares

            symbol_obj = session.query(Symbol).filter(Symbol.code == symbol).first()
            if symbol_obj:
                if symbol_obj.currency == 'USD':
                    value = value / eur_usd
                elif symbol_obj.currency == 'CAD' and cad_eur:
                    value = value * cad_eur
                elif symbol_obj.currency == 'CHF' and chf_eur:
                    value = value * chf_eur

            type_values[asset_type] += value

        # Cash de cuentas
        for c in cash_entries:
            if c.currency == 'EUR':
                type_values['Cash/Monetario'] += c.amount
            elif c.currency == 'USD':
                type_values['Cash/Monetario'] += c.amount / eur_usd

        # Cash en tránsito
        type_values['Cash/Monetario'] += CASH_EN_TRANSITO

        all_data[td] = type_values

    # Snapshots para comparar
    snapshots = {}
    for td in trading_dates:
        snap = session.query(PortfolioSnapshot).filter(PortfolioSnapshot.snapshot_date == td).all()
        snapshots[td] = sum(s.total_value for s in snap)

# Imprimir tabla
date_labels = [d.strftime('%d/%m') for d in trading_dates]
header = f"{'Tipo':<18}" + "".join(f"{dl:>12}" for dl in date_labels)
print(header)
print("=" * len(header))

for asset_type in asset_types_order:
    row = f"{asset_type:<18}"
    for td in trading_dates:
        val = all_data[td].get(asset_type, 0)
        row += f"{val:>12,.0f}"
    print(row)

print("-" * len(header))

row_total = f"{'TOTAL':<18}"
for td in trading_dates:
    total = sum(all_data[td].values())
    row_total += f"{total:>12,.0f}"
print(row_total)

row_snap = f"{'Snapshot':<18}"
for td in trading_dates:
    row_snap += f"{snapshots[td]:>12,.0f}"
print(row_snap)

row_diff = f"{'Diferencia':<18}"
for td in trading_dates:
    total = sum(all_data[td].values())
    diff = total - snapshots[td]
    row_diff += f"{diff:>+12,.0f}"
print(row_diff)
