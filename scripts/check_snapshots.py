import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_db_manager, Symbol
from src.portfolio_data import get_portfolio_service
from datetime import date

db = get_db_manager()
portfolio_service = get_portfolio_service(db)

# Test dates
test_dates = [
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

print("Checking snapshots for all trading dates:")
initial_value = portfolio_service.get_snapshot_total(date(2025, 12, 31))
print(f"Initial value (31/12/2025): {initial_value:,.2f}")

for d in test_dates:
    val = portfolio_service.get_snapshot_total(d)
    if initial_value > 0 and val > 0:
        ret = ((val / initial_value) - 1) * 100
        print(f"  {d}: {val:,.2f} EUR ({ret:+.2f}%)")
    else:
        print(f"  {d}: {val} (no snapshot)")
