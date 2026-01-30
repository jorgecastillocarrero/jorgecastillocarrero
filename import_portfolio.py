"""
Import portfolio data from Excel file.
Maps company names to tickers and organizes by sector.
"""
import os
import sys

# Set working directory to project root for correct database path
os.chdir('C:/Users/usuario/financial-data-project')
sys.path.insert(0, 'C:/Users/usuario/financial-data-project')

import pandas as pd
from datetime import date
from src.database import get_db_manager, Symbol, Portfolio, PortfolioHolding, PriceHistory

# Mapping: Company Name -> (Ticker, Sector)
COMPANY_MAPPING = {
    # CO3365 - Acciones Mensual
    "AKAMAI TECHNOLOGIES INC": ("AKAM", "Technology"),
    "AMAZON.COM INC": ("AMZN", "Consumer Discretionary"),
    "BECTON DICKINSON AND CO": ("BDX", "Healthcare"),
    "HCA HEALTHCARE INC": ("HCA", "Healthcare"),
    "MASTERCARD INCORPORATED": ("MA", "Financial"),
    "MOODYS CORP": ("MCO", "Financial"),
    "PACCAR INC": ("PCAR", "Industrial"),
    "SALESFORCE INC": ("CRM", "Technology"),
    "VERTEX PHARMACEUTICALS INC": ("VRTX", "Healthcare"),
    "WEST PHARMACEUTICAL SERVICES INC": ("WST", "Healthcare"),

    # La Caixa
    "ARITZIA INC": ("ATZ", "Consumer Discretionary"),
    "ALIBABA GROUP HOLDING LTD": ("BABA", "Consumer Discretionary"),
    "IAG INT.CONS.AIRLINES GRP *MC*": ("IAG.MC", "Industrial"),
    "JD.COM INC *NASDAQ*": ("JD", "Consumer Discretionary"),
    "NESTLE LTD. *ZRH*": ("NESN.SW", "Consumer Staples"),
    "AGNICO-EAGLE MINES LTD **TOR**": ("AEM", "Materials"),
    "ANHEUSER BUSCH INBEV SA N": ("BUD", "Consumer Staples"),

    # RCO951 - Growth
    "ALPHABET INC CL C": ("GOOG", "Technology"),
    "AMPHENOL CORP": ("APH", "Technology"),
    "APPLOVIN CORP": ("APP", "Technology"),
    "ARGAN INC": ("AGX", "Industrial"),
    "BIRKENSTOCK HOLDING PLC": ("BIRK", "Consumer Discretionary"),
    "BRINKER INTERNATIONAL INC": ("EAT", "Consumer Discretionary"),
    "BROADCOM INC": ("AVGO", "Technology"),
    "CATALYST PHARMACEUTICAL PARTNERS": ("CPRX", "Healthcare"),
    "CECO ENVIRONMENTAL CORP": ("CECO", "Industrial"),
    "CELESTICA INC": ("CLS", "Technology"),
    "CLEAR SECURE INC": ("YOU", "Technology"),
    "COINBASE GLOBAL INC": ("COIN", "Financial"),
    "COMFORT SYSTEM USA INC": ("FIX", "Industrial"),
    "DLOCAL LTD/URUGUAY": ("DLO", "Technology"),
    "DOXIMITY INC": ("DOCS", "Healthcare"),
    "DUOLINGO INC": ("DUOL", "Technology"),
    "ELBIT SYSTEMS LTD": ("ESLT", "Industrial"),
    "ELI LILLY AND COMPANY": ("LLY", "Healthcare"),
    "EMCOR GROUP INC": ("EME", "Industrial"),
    "ENOVA INTERNATIONAL INC": ("ENVA", "Financial"),
    "EVERCORE INC": ("EVR", "Financial"),
    "EXELIXIS INC": ("EXEL", "Healthcare"),
    "EZCORP INC": ("EZPW", "Financial"),
    "FUTU HOLDINGS LTD": ("FUTU", "Financial"),
    "GILDAN ACTIVEWEAR INC": ("GIL", "Consumer Discretionary"),
    "GLOBUS MEDICAL INC": ("GMED", "Healthcare"),
    "GUIDEWIRE SOFTWARE INC": ("GWRE", "Technology"),
    "HALO ZYME THERAPEUTICS": ("HALO", "Healthcare"),
    "HARMONY BIOSCIENCES HOLDINGS I": ("HRMY", "Healthcare"),
    "HCI GROUP INC": ("HCI", "Financial"),
    "HEALTHEQUITY INC": ("HQY", "Healthcare"),
    "HOULIHAN LOKEY INC": ("HLI", "Financial"),
    "INCYTE GENOMICS INC": ("INCY", "Healthcare"),
    "KANZHUN LTD": ("BZ", "Technology"),
    "KLA CORP": ("KLAC", "Technology"),
    "KRYSTAL BIOTECH INC": ("KRYS", "Healthcare"),
    "LAM RESEARCH CORP": ("LRCX", "Technology"),
    "LIFE TIME GROUP HOLDINGS INC": ("LTH", "Consumer Discretionary"),
    "MANULIFE FINANCIAL CORP": ("MFC", "Financial"),
    "MICRON TECHNOLOGY INC": ("MU", "Technology"),
    "MILLICOM INTERNATIONAL CELLULAR SA": ("TIGO", "Communication Services"),
    "MONDAY.COM LTD": ("MNDY", "Technology"),
    "MONOLITHIC POWER SYSTEMS INC": ("MPWR", "Technology"),
    "NEWMARK GROUP INC": ("NMRK", "Real Estate"),
    "NICOLET BANKSHARES INC": ("NIC", "Financial"),
    "NOMURA HOLDINGS INC": ("NMR", "Financial"),
    "NVIDIA CORPORATION": ("NVDA", "Technology"),
    "ON HOLDING AG": ("ONON", "Consumer Discretionary"),
    "PALOMAR HOLDINGS INC": ("PLMR", "Financial"),
    "PAR PACIFIC HOLDINGS INC": ("PARR", "Energy"),
    "PAYPAL HOLDINGS INC": ("PYPL", "Financial"),
    "PENNYMAC FINANCIAL SERVICES IN": ("PFSI", "Financial"),
    "PENUMBRA INC": ("PEN", "Healthcare"),
    "PHIBRO ANIMAL HEALTH CORP-A": ("PAHC", "Healthcare"),
    "PIPER SANDLER COS": ("PIPR", "Financial"),
    "PJT PARTNERS INC- A": ("PJT", "Financial"),
    "POWER SOLUTIONS INTERNATIONAL": ("PSIX", "Industrial"),
    "PRIMORIS SERVICES": ("PRIM", "Industrial"),
    "ROYAL CARIBBEAN CRUISES LTD": ("RCL", "Consumer Discretionary"),
    "SEACOAST BANKING CORP": ("SBCF", "Financial"),
    "SEAGATE TECHNOLOGY HOLDINGS PL": ("STX", "Technology"),
    "SHAKE SHACK INC -CLASS A": ("SHAK", "Consumer Discretionary"),
    "SHARKNINJA INC": ("SN", "Consumer Discretionary"),
    "SKYWEST INC": ("SKYW", "Industrial"),
    "SOLARIS ENRGY RG-A": ("SEI", "Energy"),
    "SPROUTS FARMERS MARKET INC": ("SFM", "Consumer Staples"),
    "STERLING CONSTRUCTION CO": ("STRL", "Industrial"),
    "STEWART INFORMATION SERVICES CORP": ("STC", "Financial"),
    "TAIWAN SEMICONDUCTOR MANFT LTD ADR": ("TSM", "Technology"),
    "TRANSPORTADORA DE GAS DEL SUR SA": ("TGS", "Energy"),
    "TTM TECH INC": ("TTMI", "Technology"),
    "UBER TECHNOLOGIES INC": ("UBER", "Technology"),
    "UNITED NATURAL FOODS INC": ("UNFI", "Consumer Staples"),
    "VERTIV HOLDINGS CO": ("VRT", "Industrial"),
    "VIRTU FINANCIAL INC CLASS A": ("VIRT", "Financial"),
    "VISTANCE NETWORK RG": ("VSEC", "Industrial"),
    "VITAL FARMS INC": ("VITL", "Consumer Staples"),
    "WAYFAIR INC-CLASS A": ("W", "Consumer Discretionary"),
    "WILLDAN GROUP INC": ("WLDN", "Industrial"),
    "WINGSTOP INC": ("WING", "Consumer Discretionary"),

    # RCO951 - Oro/Mineras
    "ALAMOS GOLD INC": ("AGI", "Materials"),
    "BARRICK MINING CORP": ("GOLD", "Materials"),
    "COEUR MINING INC": ("CDE", "Materials"),
    "COMPANIA DE MINAS BUENAVENTURA SA ADS": ("BVN", "Materials"),
    "INVESCO PHYSICAL GOLD ETC EUR HEDGED": ("SGLD", "Materials"),
    "KINROSS GOLD CORP": ("KGC", "Materials"),
    "ROYAL GOLD INC": ("RGLD", "Materials"),
    "SSR MINING INC": ("SSRM", "Materials"),
    "TRIPLE FLAG PRECIOUS METALS CO": ("TFPM", "Materials"),
    "WHEATON PRECIOUS METALS CORP": ("WPM", "Materials"),
}

def import_portfolio():
    """Import portfolio from Excel file."""
    df = pd.read_excel(r'C:\Users\usuario\Downloads\dashboard_activos_ranking_dashboard_v3.xlsx')

    # Filter only 'En cartera' positions
    en_cartera = df[df['Estado'] == 'En cartera'].copy()

    db = get_db_manager()
    entry_date = date(2025, 12, 31)

    # Create portfolios for each account
    portfolios_to_create = [
        ("CO3365 - Mensual", "CO3365", "Acciones - Mensual"),
        ("La Caixa - Growth", "La Caixa", "Acciones - Growth"),
        ("La Caixa - Value", "La Caixa", "Acciones - Value"),
        ("La Caixa - Oro", "La Caixa", "Oro/mineras"),
        ("RCO951 - Growth", "RCO951", "Acciones - Growth"),
        ("RCO951 - Oro", "RCO951", "Oro/mineras"),
    ]

    with db.get_session() as session:
        # Create main portfolio for December 2025
        main_portfolio = Portfolio(
            name="Cartera Completa 31/12/2025",
            month=12,
            year=2025,
            initial_capital=sum(en_cartera['Inicio_EUR']),
            description="Portfolio completo con todas las posiciones a 31/12/2025"
        )
        session.add(main_portfolio)
        session.flush()

        # Track stats
        added = 0
        skipped = 0
        missing_symbols = []

        # Organize by sector
        by_sector = {}

        for idx, row in en_cartera.iterrows():
            company_name = row['VALOR']
            value_eur = row['Inicio_EUR']

            if company_name in COMPANY_MAPPING:
                ticker, sector = COMPANY_MAPPING[company_name]

                # Find symbol in database
                symbol = session.query(Symbol).filter(Symbol.code == ticker).first()

                if symbol:
                    # Get entry price from database (last price of 2025)
                    last_price = session.query(PriceHistory).filter(
                        PriceHistory.symbol_id == symbol.id,
                        PriceHistory.date <= entry_date
                    ).order_by(PriceHistory.date.desc()).first()

                    entry_price = last_price.close if last_price else None
                    shares = value_eur / entry_price if entry_price else 0

                    # Add to portfolio
                    holding = PortfolioHolding(
                        portfolio_id=main_portfolio.id,
                        symbol_id=symbol.id,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        shares=shares
                    )
                    session.add(holding)
                    added += 1

                    # Track by sector
                    if sector not in by_sector:
                        by_sector[sector] = []
                    by_sector[sector].append({
                        'ticker': ticker,
                        'name': company_name[:30],
                        'value': value_eur,
                        'price': entry_price,
                        'shares': shares
                    })
                else:
                    missing_symbols.append(ticker)
                    skipped += 1
            else:
                print(f"No mapping for: {company_name}")
                skipped += 1

        session.commit()

        # Print summary by sector
        print("\n" + "="*80)
        print("PORTFOLIO ORGANIZADO POR SECTOR - 31/12/2025")
        print("="*80)

        total_value = 0
        for sector in sorted(by_sector.keys()):
            holdings = by_sector[sector]
            sector_value = sum(h['value'] for h in holdings)
            total_value += sector_value

            print(f"\n{'='*60}")
            print(f"  {sector.upper()} - {len(holdings)} posiciones - EUR {sector_value:,.2f}")
            print(f"{'='*60}")

            for h in sorted(holdings, key=lambda x: x['value'], reverse=True):
                price_str = f"${h['price']:.2f}" if h['price'] else "N/A"
                print(f"  {h['ticker']:8} | {h['name']:30} | EUR {h['value']:>12,.2f} | {price_str:>10}")

        print(f"\n{'='*80}")
        print(f"TOTAL: EUR {total_value:,.2f}")
        print(f"Posiciones agregadas: {added}")
        print(f"Posiciones omitidas: {skipped}")
        if missing_symbols:
            print(f"Simbolos no encontrados en DB: {missing_symbols}")
        print("="*80)

if __name__ == "__main__":
    import_portfolio()
