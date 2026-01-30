"""
Script to set up initial holdings for 31/12/2025.
Run this once to populate the daily_holdings table with the starting positions.

IMPORTANT: This script should be updated with the ACTUAL holdings as of 31/12/2025.
Exclude any positions that were bought after 31/12.
"""

import sys
sys.path.insert(0, 'C:/Users/usuario/financial-data-project')

from datetime import date
from src.daily_tracking import get_tracking_service
from src.database import get_db_manager

# =============================================================================
# INITIAL HOLDINGS - 31/12/2025
# =============================================================================
# NOTE: Update this list to reflect ACTUAL positions on 31/12/2025
# Exclude positions bought after this date (e.g., NEM bought 15/01)

INITIAL_HOLDINGS_31DIC = [
    # CO3365 Account - 10 stocks (USD)
    {'account_code': 'CO3365', 'symbol': 'AKAM', 'shares': 555, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'VRTX', 'shares': 113, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'PCAR', 'shares': 441, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'BDX', 'shares': 251, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'AMZN', 'shares': 211, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'MCO', 'shares': 95, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'HCA', 'shares': 102, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'MA', 'shares': 85, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'WST', 'shares': 177, 'currency': 'USD'},
    {'account_code': 'CO3365', 'symbol': 'CRM', 'shares': 185, 'currency': 'USD'},

    # La Caixa Account - 6 stocks (mixed currencies)
    {'account_code': 'LACAIXA', 'symbol': 'JD', 'shares': 3500, 'currency': 'USD'},
    {'account_code': 'LACAIXA', 'symbol': 'BABA', 'shares': 1000, 'currency': 'USD'},
    {'account_code': 'LACAIXA', 'symbol': 'ATZ.TO', 'shares': 330, 'currency': 'CAD'},
    {'account_code': 'LACAIXA', 'symbol': 'IAG.MC', 'shares': 30000, 'currency': 'EUR'},
    {'account_code': 'LACAIXA', 'symbol': 'AEM.TO', 'shares': 1000, 'currency': 'CAD'},
    {'account_code': 'LACAIXA', 'symbol': 'NESN.SW', 'shares': 212, 'currency': 'CHF'},

    # IB Account - TLT
    {'account_code': 'IB', 'symbol': 'TLT', 'shares': 8042, 'currency': 'USD'},

    # RCO951 Account - Positions that existed on 31/12
    # TODO: Remove positions that were bought after 31/12 (e.g., NEM)
    {'account_code': 'RCO951', 'symbol': 'B', 'shares': 3000, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'TFPM', 'shares': 2900, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SSRM', 'shares': 1973, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'MU', 'shares': 132, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'RGLD', 'shares': 160, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'KGC', 'shares': 1175, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'TTMI', 'shares': 440, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'CDE', 'shares': 1531, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'UNFI', 'shares': 1053, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'EZPW', 'shares': 1775, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'KRYS', 'shares': 130, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'LRCX', 'shares': 158, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'STRL', 'shares': 95, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'STX', 'shares': 92, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'BVN', 'shares': 792, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'MFC', 'shares': 825, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'VISN', 'shares': 1680, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'WPM', 'shares': 204, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'TIGO', 'shares': 480, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'EXEL', 'shares': 655, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'ENVA', 'shares': 179, 'currency': 'USD'},
    # {'account_code': 'RCO951', 'symbol': 'NEM', 'shares': 220, 'currency': 'USD'},  # BOUGHT 15/01 - EXCLUDED
    {'account_code': 'RCO951', 'symbol': 'ESLT', 'shares': 39, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'INCY', 'shares': 265, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'WLDN', 'shares': 210, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'FIX', 'shares': 24, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'GOOG', 'shares': 81, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'CECO', 'shares': 380, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'AGX', 'shares': 75, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PEN', 'shares': 73, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SN', 'shares': 208, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'AGI', 'shares': 600, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'LLY', 'shares': 24, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'NMR', 'shares': 2817, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'APH', 'shares': 162, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PFSI', 'shares': 165, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'NIC', 'shares': 172, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'KLAC', 'shares': 16, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'CLS', 'shares': 80, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PRIM', 'shares': 163, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'VRT', 'shares': 135, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'TSM', 'shares': 73, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'W', 'shares': 220, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'EAT', 'shares': 152, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'MPWR', 'shares': 22, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'HRMY', 'shares': 623, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'CPRX', 'shares': 965, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'WING', 'shares': 82, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'YOU', 'shares': 639, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'VIRT', 'shares': 595, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PLMR', 'shares': 175, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'LTH', 'shares': 744, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'GMED', 'shares': 236, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'ONON', 'shares': 470, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'GIL', 'shares': 328, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SBCF', 'shares': 632, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PIPR', 'shares': 59, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'DLO', 'shares': 1420, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SEI', 'shares': 402, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'UI', 'shares': 38, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PAHC', 'shares': 525, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'HALO', 'shares': 295, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'EME', 'shares': 29, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'TGS', 'shares': 640, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'EVR', 'shares': 56, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SKYW', 'shares': 201, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'NVDA', 'shares': 108, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'GLDD', 'shares': 1342, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PARR', 'shares': 570, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'STC', 'shares': 291, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'USAC', 'shares': 793, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PJT', 'shares': 110, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'NMRK', 'shares': 1129, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'AVGO', 'shares': 60, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SHAK', 'shares': 210, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'RCL', 'shares': 64, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'FUTU', 'shares': 114, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SFM', 'shares': 255, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'HQY', 'shares': 206, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'UBER', 'shares': 214, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'HLI', 'shares': 96, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'BIRK', 'shares': 439, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'APP', 'shares': 32, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'HCI', 'shares': 107, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'MNDY', 'shares': 126, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'GWRE', 'shares': 98, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'BZ', 'shares': 825, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'VITL', 'shares': 540, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'PSIX', 'shares': 177, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'COIN', 'shares': 60, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'DOCS', 'shares': 296, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'DUOL', 'shares': 75, 'currency': 'USD'},
    {'account_code': 'RCO951', 'symbol': 'SGLE.MI', 'shares': 1500, 'currency': 'EUR'},
]

# =============================================================================
# INITIAL CASH - 31/12/2025
# =============================================================================

INITIAL_CASH_31DIC = [
    {'account_code': 'CO3365', 'currency': 'EUR', 'amount': 664.27},
    {'account_code': 'CO3365', 'currency': 'USD', 'amount': 215.65},
    {'account_code': 'RCO951', 'currency': 'EUR', 'amount': 722.79},
    {'account_code': 'RCO951', 'currency': 'USD', 'amount': 261970.61},
    {'account_code': 'IB', 'currency': 'EUR', 'amount': 119895.04},
]


def main():
    print("=" * 60)
    print("SETUP INITIAL HOLDINGS - 31/12/2025")
    print("=" * 60)

    tracker = get_tracking_service()

    # Set initial holdings
    print("\nCargando holdings iniciales...")
    count = tracker.set_holdings_for_date(date(2025, 12, 31), INITIAL_HOLDINGS_31DIC)
    print(f"  Holdings cargados: {count}")

    # Set initial cash
    print("\nCargando cash inicial...")
    cash_count = tracker.set_cash_for_date(date(2025, 12, 31), INITIAL_CASH_31DIC)
    print(f"  Cash entries: {cash_count}")

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICACIÓN")
    print("=" * 60)

    result = tracker.get_portfolio_value(date(2025, 12, 31))

    print(f"\nFecha: {result['date']}")
    print(f"EUR/USD: {result['rates']['EUR_USD']:.4f}")
    print(f"CAD/EUR: {result['rates']['CAD_EUR']:.6f}")
    print(f"CHF/EUR: {result['rates']['CHF_EUR']:.6f}")

    print("\nPor cuenta:")
    for acc, data in result['accounts'].items():
        print(f"  {acc}: Holdings {data['holdings_total']:,.0f} + Cash {data['cash']:,.0f} = {data['total']:,.0f} EUR")

    print(f"\nTOTAL HOLDINGS: {result['total_holdings']:,.0f} EUR")
    print(f"TOTAL CASH: {result['total_cash']:,.0f} EUR")
    print(f"TOTAL CARTERA: {result['total']:,.0f} EUR")

    print("\n" + "=" * 60)
    print("NOTA: El valor oficial de extractos es 3,930,154 EUR")
    print("Si hay diferencia, revisar qué posiciones NO existían a 31/12")
    print("=" * 60)


if __name__ == "__main__":
    main()
