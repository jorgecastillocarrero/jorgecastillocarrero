"""
Tabla completa de valoración por tipo de activo - todos los días de trading.
Método: precio del día, si no existe usa el día anterior (mercado cerrado).
Sin snapshots. Sin cash en tránsito.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_db_manager, Symbol, PriceHistory, AccountHolding, AccountCash
from src.portfolio_data import get_portfolio_service, ASSET_TYPE_MAP
from datetime import date, timedelta

db = get_db_manager()
portfolio_service = get_portfolio_service(db)
portfolio_service._cache.clear()

trading_dates = [
    date(2025, 12, 31), date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6),
    date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 12),
    date(2026, 1, 13), date(2026, 1, 14), date(2026, 1, 15), date(2026, 1, 16),
    date(2026, 1, 20), date(2026, 1, 21), date(2026, 1, 22), date(2026, 1, 23),
    date(2026, 1, 26), date(2026, 1, 27),
]

asset_types_order = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/Monetario']

def get_price_or_previous(session, sym_id, target_date, max_lookback=5):
    for i in range(max_lookback + 1):
        d = target_date - timedelta(days=i)
        p = session.query(PriceHistory).filter(
            PriceHistory.symbol_id == sym_id,
            PriceHistory.date == d
        ).first()
        if p:
            return p.close, d
    return None, None

with db.get_session() as session:
    holdings = session.query(AccountHolding).all()
    cash_entries = session.query(AccountCash).all()

    # Cache symbol objects
    sym_cache = {}
    for h in holdings:
        if h.symbol not in sym_cache:
            sym_cache[h.symbol] = session.query(Symbol).filter(Symbol.code == h.symbol).first()

    # FX symbol IDs
    eurusd_sym = session.query(Symbol).filter(Symbol.code == 'EURUSD=X').first()
    cadeur_sym = session.query(Symbol).filter(Symbol.code == 'CADEUR=X').first()
    chfeur_sym = session.query(Symbol).filter(Symbol.code == 'CHFEUR=X').first()

    all_data = {}

    for td in trading_dates:
        eur_usd, _ = get_price_or_previous(session, eurusd_sym.id, td)
        cad_eur, _ = get_price_or_previous(session, cadeur_sym.id, td)
        chf_eur, _ = get_price_or_previous(session, chfeur_sym.id, td)

        type_values = {t: 0.0 for t in asset_types_order}

        for holding in holdings:
            symbol = holding.symbol
            shares = holding.shares
            asset_type = ASSET_TYPE_MAP.get(symbol, 'Otros')
            if asset_type not in type_values:
                type_values[asset_type] = 0.0

            sym = sym_cache.get(symbol)
            if not sym:
                continue

            price, _ = get_price_or_previous(session, sym.id, td)
            if price is None:
                continue

            value = price * shares
            value_eur = value
            if sym.currency == 'USD':
                value_eur = value / eur_usd
            elif sym.currency == 'CAD':
                value_eur = value * cad_eur
            elif sym.currency == 'CHF':
                value_eur = value * chf_eur

            type_values[asset_type] += value_eur

        # Cash
        cash_total = 0
        for c in cash_entries:
            if c.currency == 'EUR':
                cash_total += c.amount
            elif c.currency == 'USD':
                cash_total += c.amount / eur_usd
        type_values['Cash/Monetario'] += cash_total

        all_data[td] = type_values

# Print table
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

# Rentabilidad acumulada vs 31/12
initial = sum(all_data[date(2025, 12, 31)].values())
row_rent = f"{'Rent. %':<18}"
for td in trading_dates:
    total = sum(all_data[td].values())
    rent = ((total / initial) - 1) * 100
    row_rent += f"{rent:>+12.2f}"
print(row_rent)
