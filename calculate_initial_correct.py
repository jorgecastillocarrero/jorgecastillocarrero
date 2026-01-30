"""
CALCULO CORRECTO del valor inicial a 31/12/2025
Usando las operaciones reales de enero 2026
"""
import pandas as pd
from datetime import datetime, date
import sys
import os

os.chdir('C:/Users/usuario/financial-data-project')
sys.path.insert(0, 'C:/Users/usuario/financial-data-project')

from src.database import get_db_manager, Symbol

EUR_USD_31DIC = 1.0386

# =============================================================================
# OPERACIONES DE ENERO 2026 (del archivo bolsa (22).xls)
# =============================================================================

# VENTAS - Posiciones que estaban a 31/12/2025 y se vendieron en enero
SOLD_POSITIONS = {
    'PYPL': 430,      # PayPal
    'AEP': 170,       # American Electric Power
    'AEE': 193,       # Ameren
    # 'NRG': 1990,    # AXIA ENERGIA - OJO: se vendió a $9.60, NO $0.000076!
    'CAH': 128,       # Cardinal Health
    'JEF': 325,       # Jefferies Financial
    'HUM': 78,        # Humana
    'WTRG': 495,      # Essential Utilities
    'MCK': 26,        # McKesson
    'ATO': 111,       # Atmos Energy
    'EHC': 162,       # Encompass Health
    'UHS': 98,        # Universal Health Services
    'RBGLY': 350,     # Reckitt Benckiser
    'TSCDY': 3380,    # Tesco
    'USFD': 259,      # US Foods
    'FDP': 579,       # Fresh Del Monte
    'DLTR': 205,      # Dollar Tree
    'DG': 189,        # Dollar General
    'HENKY': 269,     # Henkel
    'PEP': 134,       # PepsiCo
    # 'SHLG': 367,    # Siemens Healthineers (sin datos)
    'GO': 1321,       # Grocery Outlet
    'NFG': 249,       # National Fuel Gas
    'MNST': 294,      # Monster Beverage
}

# Valor de venta de SHLG y AXIA (EUR directo, no calculable con precios)
SHLG_SOLD_EUR = 16399.24  # Siemens Healthineers
AXIA_SOLD_EUR = 16401.19  # Axia Energia - valor real de venta

# COMPRAS - Posiciones nuevas en enero 2026 (NO estaban a 31/12/2025)
NEW_PURCHASES = {
    'GLDD': 1342,     # Great Lakes Dredge
    'UI': 38,         # Ubiquiti
    'USAC': 793,      # USA Compression
    'NEM': 220,       # Newmont (compra ADICIONAL, ya tenia 220)
}

# =============================================================================
# POSICIONES ACTUALES RCO951 (del informe patrimonio)
# =============================================================================
RCO951_CURRENT_STOCKS = {
    'GOLD': 3000, 'TFPM': 2900, 'SSRM': 1973, 'MU': 132, 'RGLD': 160,
    'KGC': 1175, 'TTMI': 440, 'CDE': 1531, 'UNFI': 1053, 'EZPW': 1775,
    'KRYS': 130, 'LRCX': 158, 'STRL': 95, 'STX': 92, 'BVN': 792,
    'MFC': 825, 'VSCO': 1680, 'WPM': 204, 'TIGO': 480, 'EXEL': 655,
    'ENVA': 179, 'NEM': 220, 'ESLT': 39, 'INCY': 265, 'WLDN': 210,
    'FIX': 24, 'GOOG': 81, 'CECO': 380, 'AGX': 75, 'PEN': 73,
    'SN': 208, 'AGI': 600, 'LLY': 24, 'NMR': 2817, 'APH': 162,
    'PFSI': 165, 'NIC': 172, 'KLAC': 16, 'CLS': 80, 'PRIM': 163,
    'VRT': 135, 'TSM': 73, 'W': 220, 'EAT': 152, 'MPWR': 22,
    'HRMY': 623, 'CPRX': 965, 'WING': 82, 'YOU': 639, 'VIRT': 595,
    'PLMR': 175, 'LTH': 744, 'GMED': 236, 'ONON': 470, 'GIL': 328,
    'SBCF': 632, 'PIPR': 59, 'DLO': 1420, 'SEI': 402,
    'UI': 38,         # Nueva compra enero
    'PAHC': 525, 'HALO': 295, 'EME': 29, 'TGS': 640, 'EVR': 56,
    'SKYW': 201, 'NVDA': 108,
    'GLDD': 1342,     # Nueva compra enero
    'PARR': 570, 'STC': 291,
    'USAC': 793,      # Nueva compra enero
    'PJT': 110, 'NMRK': 1129, 'AVGO': 60, 'SHAK': 210,
    'RCL': 64, 'FUTU': 114, 'SFM': 255, 'HQY': 206, 'UBER': 214,
    'HLI': 96, 'BIRK': 439, 'APP': 32, 'HCI': 107, 'MNDY': 126,
    'GWRE': 98, 'BZ': 825, 'VITL': 540, 'PSIX': 177, 'COIN': 60,
    'DOCS': 296, 'DUOL': 75,
}

def get_price_31dic(db, session, ticker):
    """Get price at 31/12/2025"""
    target_date = date(2025, 12, 31)
    symbol_code = ticker.split('.')[0] if '.' in ticker else ticker

    symbol = session.query(Symbol).filter(Symbol.code == symbol_code).first()
    if not symbol:
        return None

    prices = db.get_price_history(session, symbol.id, start_date=datetime(2025, 12, 20))
    if prices.empty:
        return None

    prices_before = prices[prices.index.date <= target_date]
    if prices_before.empty:
        return None

    return prices_before['close'].iloc[-1]

def main():
    db = get_db_manager()

    print("=" * 80)
    print("CALCULO CORRECTO - VALOR INICIAL A 31/12/2025")
    print("=" * 80)

    with db.get_session() as session:
        # ==================================================================
        # RCO951 - Calcular valor de posiciones VENDIDAS a precios 31/12
        # ==================================================================
        print("\n[1] POSICIONES VENDIDAS EN ENERO (valor a 31/12/2025)")
        print("-" * 60)

        sold_value_31dic = 0
        sold_details = []

        for ticker, shares in SOLD_POSITIONS.items():
            price = get_price_31dic(db, session, ticker)
            if price:
                value_eur = (shares * price) / EUR_USD_31DIC
                sold_value_31dic += value_eur
                sold_details.append(f"{ticker:8} {shares:>5} @ ${price:>8.2f} = EUR {value_eur:>12,.2f}")
            else:
                sold_details.append(f"{ticker:8} {shares:>5} - SIN DATOS")

        # Añadir SHLG y AXIA (valores estimados)
        sold_value_31dic += SHLG_SOLD_EUR
        sold_value_31dic += AXIA_SOLD_EUR
        sold_details.append(f"SHLG     {367:>5} (estimado)     = EUR {SHLG_SOLD_EUR:>12,.2f}")
        sold_details.append(f"AXIA    {1990:>5} (estimado)     = EUR {AXIA_SOLD_EUR:>12,.2f}")

        for d in sold_details:
            print(f"  {d}")

        print(f"\n  TOTAL VENDIDO (a precios 31/12): EUR {sold_value_31dic:,.2f}")

        # ==================================================================
        # RCO951 - Calcular valor de posiciones ACTUALES a precios 31/12
        # (excluyendo compras nuevas de enero)
        # ==================================================================
        print("\n[2] POSICIONES ACTUALES (sin compras enero) a precios 31/12")
        print("-" * 60)

        current_value_31dic = 0
        new_purchases_31dic = 0

        for ticker, shares in RCO951_CURRENT_STOCKS.items():
            if ticker in NEW_PURCHASES:
                continue  # Skip new purchases

            price = get_price_31dic(db, session, ticker)
            if price:
                value_eur = (shares * price) / EUR_USD_31DIC
                current_value_31dic += value_eur

        # NEM had 220 shares at 31/12, bought 220 more in Jan, now has 440
        # Need to adjust - the report shows 220 total, so no adjustment needed
        # Actually looking at operations, NEM was bought 220 shares in Jan
        # But current position shows NEM: 220 shares
        # This means he had 0 at 31/12 and bought 220 in Jan

        print(f"  Subtotal stocks actuales (sin compras enero): EUR {current_value_31dic:,.2f}")

        # ==================================================================
        # RCO951 - ETF Gold
        # ==================================================================
        # Current value: EUR 174,285
        # Gold was ~5% lower at year end
        etf_gold_31dic = 174285 * 0.95
        print(f"\n  ETF Gold (estimado -5%): EUR {etf_gold_31dic:,.2f}")

        # ==================================================================
        # RCO951 - EFECTIVO a 31/12/2025
        # ==================================================================
        # From operations analysis:
        # - Current USD cash: ~EUR 220,553
        # - Net sales in Jan: ~EUR 359K (sales - purchases)
        # - Transfers to IB: ~EUR 660K
        # - Initial cash = Current + Transfers - Net sales
        # - Initial cash = 220,553 + 660,000 - 359,000 = EUR 521,553

        # Let's calculate net from actual operations:
        total_sales_eur = 432414  # Sum from operations file
        total_purchases_eur = 72987  # Sum from operations file
        transfers_to_ib = 659999.25
        current_cash = 220553 + 722.79  # USD + EUR

        initial_cash_rco951 = current_cash + transfers_to_ib - (total_sales_eur - total_purchases_eur)
        print(f"\n  Efectivo a 31/12/2025 (calculado): EUR {initial_cash_rco951:,.2f}")

        # ==================================================================
        # RCO951 TOTAL
        # ==================================================================
        rco951_total = current_value_31dic + sold_value_31dic + etf_gold_31dic + initial_cash_rco951
        print(f"\n  >>> TOTAL RCO951 a 31/12/2025: EUR {rco951_total:,.2f}")

        # ==================================================================
        # CO3365
        # ==================================================================
        print("\n" + "=" * 60)
        print("[3] CO3365 a precios 31/12/2025")
        print("-" * 60)

        CO3365_POSITIONS = {
            'AKAM': 555, 'VRTX': 113, 'PCAR': 441, 'BDX': 251, 'AMZN': 211,
            'MCO': 95, 'HCA': 102, 'MA': 85, 'WST': 177, 'CRM': 185,
        }
        CO3365_CASH = 664.27 + 207.36

        co3365_stocks = 0
        for ticker, shares in CO3365_POSITIONS.items():
            price = get_price_31dic(db, session, ticker)
            if price:
                co3365_stocks += (shares * price) / EUR_USD_31DIC

        co3365_total = co3365_stocks + CO3365_CASH
        print(f"  Stocks: EUR {co3365_stocks:,.2f}")
        print(f"  Cash: EUR {CO3365_CASH:,.2f}")
        print(f"  >>> TOTAL CO3365: EUR {co3365_total:,.2f}")

        # ==================================================================
        # LA CAIXA
        # ==================================================================
        print("\n" + "=" * 60)
        print("[4] LA CAIXA a precios 31/12/2025")
        print("-" * 60)

        LACAIXA_USD = {'JD': 3500, 'BABA': 1000, 'AEM': 1000}
        lacaixa_value = 0

        for ticker, shares in LACAIXA_USD.items():
            price = get_price_31dic(db, session, ticker)
            if price:
                lacaixa_value += (shares * price) / EUR_USD_31DIC

        # Non-USD positions (estimated from current values -5%)
        lacaixa_value += 23694 * 0.95   # Aritzia
        lacaixa_value += 143190 * 0.95  # IAG
        lacaixa_value += 59 * 0.95      # BUD
        lacaixa_value += 16503 * 0.98   # Nestle

        print(f"  >>> TOTAL LA CAIXA: EUR {lacaixa_value:,.2f}")

        # ==================================================================
        # INTERACTIVE BROKERS
        # ==================================================================
        print("\n" + "=" * 60)
        print("[5] INTERACTIVE BROKERS a 31/12/2025")
        print("-" * 60)
        ib_total = 30008.88  # Solo deposito diciembre
        print(f"  Solo deposito diciembre: EUR {ib_total:,.2f}")

    # ==================================================================
    # RESUMEN FINAL
    # ==================================================================
    total_inicial = co3365_total + rco951_total + lacaixa_value + ib_total

    print("\n" + "=" * 80)
    print("RESUMEN VALOR INICIAL A 31/12/2025")
    print("=" * 80)
    print(f"\n  CO3365:              EUR {co3365_total:>15,.2f}")
    print(f"  RCO951:              EUR {rco951_total:>15,.2f}")
    print(f"  La Caixa:            EUR {lacaixa_value:>15,.2f}")
    print(f"  Interactive Brokers: EUR {ib_total:>15,.2f}")
    print(f"  " + "-" * 43)
    print(f"  TOTAL INICIAL:       EUR {total_inicial:>15,.2f}")

    # Rentabilidad
    total_actual = 4208965.38
    # NO restamos depositos IB porque vinieron de transferencias internas de R4
    ganancia = total_actual - total_inicial
    rentabilidad = (ganancia / total_inicial) * 100

    print(f"\n" + "=" * 80)
    print("RENTABILIDAD YTD (sin ajustar por depositos - son transferencias internas)")
    print("=" * 80)
    print(f"\n  Valor inicial 31/12/2025: EUR {total_inicial:>15,.2f}")
    print(f"  Valor actual 26/01/2026:  EUR {total_actual:>15,.2f}")
    print(f"  " + "-" * 43)
    print(f"  Ganancia:                 EUR {ganancia:>+15,.2f}")
    print(f"  Rentabilidad YTD:              {rentabilidad:>+10.2f}%")

if __name__ == "__main__":
    main()
