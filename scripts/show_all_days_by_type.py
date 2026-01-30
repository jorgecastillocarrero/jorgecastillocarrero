import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_db_manager, Symbol, AccountHolding, AccountCash, PortfolioSnapshot
from src.portfolio_data import get_portfolio_service, ASSET_TYPE_MAP
from datetime import date

db = get_db_manager()
portfolio_service = get_portfolio_service(db)

trading_dates = [
    date(2025, 12, 31), date(2026, 1, 2),
]

asset_types_order = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/Monetario']

with db.get_session() as session:
    holdings = session.query(AccountHolding).all()
    cash_entries = session.query(AccountCash).all()

    # For each date, calculate value per asset type
    all_data = {}  # {date: {type: value}}

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

            # Currency conversion
            symbol_obj = session.query(Symbol).filter(Symbol.code == symbol).first()
            if symbol_obj:
                if symbol_obj.currency == 'USD':
                    value = value / eur_usd
                elif symbol_obj.currency == 'CAD' and cad_eur:
                    value = value * cad_eur
                elif symbol_obj.currency == 'CHF' and chf_eur:
                    value = value * chf_eur

            type_values[asset_type] += value

        # Add cash
        cash_total = 0
        for c in cash_entries:
            if c.currency == 'EUR':
                cash_total += c.amount
            elif c.currency == 'USD':
                cash_total += c.amount / eur_usd
        type_values['Cash/Monetario'] += cash_total

        # Sin ajuste - valores reales calculados

        all_data[td] = type_values

# Print table
# Header
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

# Snapshot row for verification
row_snap = f"{'Snapshot':<18}"
with db.get_session() as session:
    for td in trading_dates:
        snap = session.query(PortfolioSnapshot).filter(PortfolioSnapshot.snapshot_date == td).all()
        st = sum(s.total_value for s in snap)
        row_snap += f"{st:>12,.0f}"
print(row_snap)
