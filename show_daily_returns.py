import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_db_manager, Symbol
from src.portfolio_data import get_portfolio_service
from datetime import date, datetime
import pandas as pd

db = get_db_manager()
portfolio_service = get_portfolio_service(db)

# Get SPY prices
with db.get_session() as session:
    spy_symbol = session.query(Symbol).filter(Symbol.code == 'SPY').first()
    qqq_symbol = session.query(Symbol).filter(Symbol.code == 'QQQ').first()

    spy_prices = pd.DataFrame()
    qqq_prices = pd.DataFrame()

    if spy_symbol:
        spy_prices = db.get_price_history(session, spy_symbol.id, start_date=date(2025, 12, 1))
        today = date(2026, 1, 28)
        spy_prices = spy_prices[
            (spy_prices.index.date >= date(2025, 12, 31)) &
            (spy_prices.index.date < today) &
            (spy_prices.index.dayofweek < 5)
        ]

    if qqq_symbol:
        qqq_prices = db.get_price_history(session, qqq_symbol.id, start_date=date(2025, 12, 1))
        qqq_prices = qqq_prices[
            (qqq_prices.index.date >= date(2025, 12, 31)) &
            (qqq_prices.index.date < today) &
            (qqq_prices.index.dayofweek < 5)
        ]

print(f"SPY prices: {len(spy_prices)} rows")
print(f"QQQ prices: {len(qqq_prices)} rows")

# Get initial value
initial_value = portfolio_service.get_snapshot_total(date(2025, 12, 31))
print(f"Initial portfolio value (31/12/2025): {initial_value:,.2f} EUR")

# EUR/USD rates (placeholder)
eur_usd_31dic = 1.0350
eur_usd_current = 1.0490

# Build daily returns data
daily_returns_data = []

if not spy_prices.empty and len(spy_prices) >= 2:
    spy_base = spy_prices['close'].iloc[0]
    qqq_base = qqq_prices['close'].iloc[0] if not qqq_prices.empty else None

    for i, date_idx in enumerate(spy_prices.index):
        d = date_idx.date() if hasattr(date_idx, 'date') else date_idx

        spy_ret = ((spy_prices['close'].iloc[i] - spy_base) / spy_base) * 100
        qqq_ret = ((qqq_prices['close'].iloc[i] - qqq_base) / qqq_base) * 100 if qqq_base and i < len(qqq_prices) else 0

        snapshot_val = portfolio_service.get_snapshot_total(d)
        if snapshot_val > 0 and initial_value > 0:
            port_eur_ret = ((snapshot_val / initial_value) - 1) * 100
            # 31/12 es referencia, USD = 0%. Otros días incluyen conversión
            if d == date(2025, 12, 31):
                port_usd_ret = 0.0
            else:
                port_usd_ret = port_eur_ret + ((eur_usd_current / eur_usd_31dic) - 1) * 100
        else:
            port_eur_ret = None
            port_usd_ret = None

        row = {
            'Fecha': date_idx.strftime('%d/%m/%Y'),
            'Cartera EUR': f"{port_eur_ret:+.2f}%" if port_eur_ret is not None else "-",
            'Cartera USD': f"{port_usd_ret:+.2f}%" if port_usd_ret is not None else "-",
            'SPY': f"{spy_ret:+.2f}%",
            'QQQ': f"{qqq_ret:+.2f}%",
        }
        daily_returns_data.append(row)

print(f"\nTotal rows generated: {len(daily_returns_data)}")
print("\n" + "="*80)
print("TABLA RENTABILIDAD DIARIA (desde 31/12/2025)")
print("="*80)

# Show in reverse order (most recent first)
df = pd.DataFrame(daily_returns_data[::-1])
print(df.to_string(index=False))
