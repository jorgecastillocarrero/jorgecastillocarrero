"""Check composition with actual operations from 01/01/2026 to 27/01/2026"""
import pandas as pd
from datetime import datetime, date
from src.database import get_db_manager, Symbol, IBTrade

PORTFOLIO_EXCEL_PATH = r"C:\Users\usuario\Downloads\dashboard_activos_ranking_dashboard_v3 (2).xlsx"

# Positions SOLD in 2026 - exclude from portfolio at 26/01
SOLD_POSITIONS = {
    # 26/01/2026
    "PAYPAL HOLDINGS INC": date(2026, 1, 26),
    # 15/01/2026
    "AMERICAN ELECTRIC POWER CO INC": date(2026, 1, 15),
    "AMEREN CORP": date(2026, 1, 15),
    "AXIA ENERGIA": date(2026, 1, 15),
    "CARDINAL HEALTH INC": date(2026, 1, 15),
    "JEFFERIES FINANCIAL GROUP INC": date(2026, 1, 15),
    "HUMANA INC": date(2026, 1, 15),
    "ESSENTIAL UTILITIES INC": date(2026, 1, 15),
    "MCKESSON HBOC INC": date(2026, 1, 15),
    "ATMOS ENERGY CORP": date(2026, 1, 15),
    "ENCOMPASS HEALTH CORP": date(2026, 1, 15),
    "UNIVERSAL HEALTH SERVICES INC": date(2026, 1, 15),
    # 12/01/2026
    "RECKITT BENCKISER GROUP PLC": date(2026, 1, 12),
    # 05/01/2026
    "TESCO PLC": date(2026, 1, 5),
    "US FOODS HOLDING CORP": date(2026, 1, 5),
    "FRESH DEL MONTE PRODUCE INC": date(2026, 1, 5),
    "DOLLAR TREE INC": date(2026, 1, 5),
    "DOLLAR GENERAL CORP": date(2026, 1, 5),
    "HENKEL ORD": date(2026, 1, 5),
    "PEPSICO INC": date(2026, 1, 5),
    "SIEMENS HEALTHINEERS AG": date(2026, 1, 5),
    "GROCERY OUTLET HOLDING CORP": date(2026, 1, 5),
    "NATIONAL FUEL GAS CO": date(2026, 1, 5),
    "MONSTER BEVERAGE CORP": date(2026, 1, 5),
}

# New purchases in 2026
NEW_PURCHASES = [
    # (ticker, company, buy_date, shares, price_usd, cost_eur)
    ("GLDD", "GREAT LAKES DREDGE & DOCK CO", date(2026, 1, 26), 1342, 14.9094, 16906.13),
    ("UI", "UBIQUITI INC", date(2026, 1, 21), 38, 530.829, 17328.80),
    ("USAC", "USA COMPRESSION PARTNERS LP", date(2026, 1, 21), 793, 25.229, 17187.24),
    ("NEM", "NEWMONT CORP", date(2026, 1, 15), 220, 113.371, 21565.28),
]

# Positions that appear in Excel but should be excluded (replaced by new purchase)
EXCLUDE_FROM_EXCEL = {"NEWMONT CORP"}

# Full mapping
COMPANY_TO_TICKER = {
    "AGNICO-EAGLE MINES LTD **TOR**": "AEM",
    "ALIBABA GROUP HOLDING LTD": "BABA",
    "BARRICK MINING CORP": "GOLD",
    "TTM TECH INC": "TTMI",
    "TRIPLE FLAG PRECIOUS METALS CO": "TFPM",
    "MICRON TECHNOLOGY INC": "MU",
    "LAM RESEARCH CORP": "LRCX",
    "COEUR MINING INC": "CDE",
    "INVESCO PHYSICAL GOLD ETC EUR HEDGED": "SGLD.L",
    "ROYAL GOLD INC": "RGLD",
    "KINROSS GOLD CORP": "KGC",
    "WILLDAN GROUP INC": "WLDN",
    "ELBIT SYSTEMS LTD": "ESLT",
    "KLA CORP": "KLAC",
    "PACCAR INC": "PCAR",
    "COMPANIA DE MINAS BUENAVENTURA SA ADS": "BVN",
    "ARGAN INC": "AGX",
    "KRYSTAL BIOTECH INC": "KRYS",
    "SEAGATE TECHNOLOGY HOLDINGS PL": "STX",
    "PENNYMAC FINANCIAL SERVICES IN": "PFSI",
    "COMFORT SYSTEM USA INC": "FIX",
    "SOLARIS ENRGY RG-A": "SEI",
    "STERLING CONSTRUCTION CO": "STRL",
    "PRIMORIS SERVICES": "PRIM",
    "AKAMAI TECHNOLOGIES INC": "AKAM",
    "EZCORP INC": "EZPW",
    "SHAKE SHACK INC -CLASS A": "SHAK",
    "WHEATON PRECIOUS METALS CORP": "WPM",
    "BECTON DICKINSON AND CO": "BDX",
    "WAYFAIR INC-CLASS A": "W",
    "POWER SOLUTIONS INTERNATIONAL": "PSIX",
    "SHARKNINJA INC": "SN",
    "AMPHENOL CORP": "APH",
    "WINGSTOP INC": "WING",
    "MOODYS CORP": "MCO",
    "PENUMBRA INC": "PEN",
    "TAIWAN SEMICONDUCTOR MANFT LTD ADR": "TSM",
    "MONOLITHIC POWER SYSTEMS INC": "MPWR",
    "CECO ENVIRONMENTAL CORP": "CECO",
    "NOMURA HOLDINGS INC": "NMR",
    "EMCOR GROUP INC": "EME",
    "PJT PARTNERS INC- A": "PJT",
    "VIRTU FINANCIAL INC CLASS A": "VIRT",
    "SSR MINING INC": "SSRM",
    "BRINKER INTERNATIONAL INC": "EAT",
    "PIPER SANDLER COS": "PIPR",
    "EVERCORE INC": "EVR",
    "AMAZON.COM INC": "AMZN",
    "INCYTE GENOMICS INC": "INCY",
    "VERTIV HOLDINGS CO": "VRT",
    "PHIBRO ANIMAL HEALTH CORP-A": "PAHC",
    "HUMANA INC": "HUM",
    "VISTANCE NETWORK RG": "VSCO",
    "NICOLET BANKSHARES INC": "NIC",
    "MILLICOM INTERNATIONAL CELLULAR SA": "TIGO",
    "UNITED NATURAL FOODS INC": "UNFI",
    "ARITZIA INC": "ATZ.TO",
    "SEACOAST BANKING CORP": "SBCF",
    "CELESTICA INC": "CLS",
    "HOULIHAN LOKEY INC": "HLI",
    "GLOBUS MEDICAL INC": "GMED",
    "JD.COM INC *NASDAQ*": "JD",
    "ALPHABET INC CL C": "GOOG",
    "CARDINAL HEALTH INC": "CAH",
    "HALO ZYME THERAPEUTICS": "HALO",
    "MANULIFE FINANCIAL CORP": "MFC",
    "PAR PACIFIC HOLDINGS INC": "PARR",
    "FUTU HOLDINGS LTD": "FUTU",
    "COINBASE GLOBAL INC": "COIN",
    "AXIA ENERGIA": "NRG",
    "RECKITT BENCKISER GROUP PLC": "RBGLY",
    "DOLLAR TREE INC": "DLTR",
    "UBER TECHNOLOGIES INC": "UBER",
    "ESSENTIAL UTILITIES INC": "WTRG",
    "HCA HEALTHCARE INC": "HCA",
    "DOLLAR GENERAL CORP": "DG",
    "AMERICAN ELECTRIC POWER CO INC": "AEP",
    "JEFFERIES FINANCIAL GROUP INC": "JEF",
    "AMEREN CORP": "AEE",
    "LIFE TIME GROUP HOLDINGS INC": "LTH",
    "BROADCOM INC": "AVGO",
    "MCKESSON HBOC INC": "MCK",
    "GILDAN ACTIVEWEAR INC": "GIL",
    "ALAMOS GOLD INC": "AGI",
    "EXELIXIS INC": "EXEL",
    "ENOVA INTERNATIONAL INC": "ENVA",
    "NEWMARK GROUP INC": "NMRK",
    "NATIONAL FUEL GAS CO": "NFG",
    "ATMOS ENERGY CORP": "ATO",
    "NVIDIA CORPORATION": "NVDA",
    "CLEAR SECURE INC": "YOU",
    "DLOCAL LTD/URUGUAY": "DLO",
    "SKYWEST INC": "SKYW",
    "NEWMONT CORP": "NEM",
    "HENKEL ORD": "HENKY",
    "US FOODS HOLDING CORP": "USFD",
    "ROYAL CARIBBEAN CRUISES LTD": "RCL",
    "TESCO PLC": "TSCDY",
    "ANHEUSER BUSCH INBEV SA N": "BUD",
    "SPROUTS FARMERS MARKET INC": "SFM",
    "SIEMENS HEALTHINEERS AG": "SHL.DE",
    "IAG INT.CONS.AIRLINES GRP *MC*": "IAG.MC",
    "PAYPAL HOLDINGS INC": "PYPL",
    "CATALYST PHARMACEUTICAL PARTNERS": "CPRX",
    "ENCOMPASS HEALTH CORP": "EHC",
    "ON HOLDING AG": "ONON",
    "MONSTER BEVERAGE CORP": "MNST",
    "HARMONY BIOSCIENCES HOLDINGS I": "HRMY",
    "PEPSICO INC": "PEP",
    "GROCERY OUTLET HOLDING CORP": "GO",
    "ELI LILLY AND COMPANY": "LLY",
    "FRESH DEL MONTE PRODUCE INC": "FDP",
    "VERTEX PHARMACEUTICALS INC": "VRTX",
    "KANZHUN LTD": "BZ",
    "DOXIMITY INC": "DOCS",
    "PALOMAR HOLDINGS INC": "PLMR",
    "BIRKENSTOCK HOLDING PLC": "BIRK",
    "NESTLE LTD. *ZRH*": "NESN.SW",
    "STEWART INFORMATION SERVICES CORP": "STC",
    "HEALTHEQUITY INC": "HQY",
    "TRANSPORTADORA DE GAS DEL SUR SA": "TGS",
    "VITAL FARMS INC": "VITL",
    "UNIVERSAL HEALTH SERVICES INC": "UHS",
    "HCI GROUP INC": "HCI",
    "DUOLINGO INC": "DUOL",
    "MASTERCARD INCORPORATED": "MA",
    "WEST PHARMACEUTICAL SERVICES INC": "WST",
    "MONDAY.COM LTD": "MNDY",
    "APPLOVIN CORP": "APP",
    "GUIDEWIRE SOFTWARE INC": "GWRE",
    "SALESFORCE INC": "CRM",
}

xl = pd.ExcelFile(PORTFOLIO_EXCEL_PATH)
df = pd.read_excel(xl, sheet_name='Consolidado_por_activo')

db = get_db_manager()
eur_usd = 1.04
start_date = datetime(2025, 12, 31)
today = datetime.now().date()

open_positions = []  # Positions still open at 26/01
sold_positions = []  # Positions sold in 2026

with db.get_session() as session:
    for _, row in df.iterrows():
        company = row['VALOR']
        inicio_eur = row['Inicio_EUR'] if pd.notna(row['Inicio_EUR']) else 0

        if inicio_eur <= 0:
            continue

        # Skip positions that are replaced by new purchases
        if company in EXCLUDE_FROM_EXCEL:
            continue

        ticker = COMPANY_TO_TICKER.get(company)
        if not ticker:
            continue

        # Check if position was sold
        if company in SOLD_POSITIONS:
            sale_date = SOLD_POSITIONS[company]
            # Get price at sale date
            symbol = session.query(Symbol).filter(Symbol.code == ticker).first()
            if not symbol:
                symbol_code = ticker.split('.')[0] if '.' in ticker else ticker
                symbol = session.query(Symbol).filter(Symbol.code == symbol_code).first()

            if symbol:
                prices = db.get_price_history(session, symbol.id, start_date=start_date)
                if not prices.empty:
                    # Get day0 price and sale price
                    day0_price = prices['close'].iloc[0]
                    sale_prices = prices[prices.index.date <= sale_date]
                    if not sale_prices.empty:
                        sale_price = sale_prices['close'].iloc[-1]
                        shares = (inicio_eur * eur_usd) / day0_price
                        sale_value_usd = shares * sale_price
                        sale_value_eur = sale_value_usd / eur_usd
                        return_pct = ((sale_price - day0_price) / day0_price) * 100
                        sold_positions.append({
                            'ticker': ticker,
                            'company': company[:30],
                            'inicio_eur': inicio_eur,
                            'venta_eur': sale_value_eur,
                            'return_pct': return_pct,
                            'sale_date': sale_date
                        })
            continue

        # Position still open - get current value
        symbol = session.query(Symbol).filter(Symbol.code == ticker).first()
        if not symbol:
            symbol_code = ticker.split('.')[0] if '.' in ticker else ticker
            symbol = session.query(Symbol).filter(Symbol.code == symbol_code).first()

        if not symbol:
            print(f"NOT FOUND: {ticker} - {company}")
            continue

        prices = db.get_price_history(session, symbol.id, start_date=start_date)
        if prices.empty:
            print(f"NO PRICES: {ticker}")
            continue

        prices = prices[prices.index.date < today]
        if prices.empty:
            continue

        day0_price = prices['close'].iloc[0]
        current_price = prices['close'].iloc[-1]

        shares = (inicio_eur * eur_usd) / day0_price
        value_usd = shares * current_price
        value_eur = value_usd / eur_usd
        return_pct = ((current_price - day0_price) / day0_price) * 100

        open_positions.append({
            'ticker': ticker,
            'inicio_eur': inicio_eur,
            'actual_eur': value_eur,
            'return_pct': return_pct
        })

    # Add new purchases
    for ticker, company, buy_date, shares, price_usd, cost_eur in NEW_PURCHASES:
        symbol = session.query(Symbol).filter(Symbol.code == ticker).first()
        if symbol:
            prices = db.get_price_history(session, symbol.id, start_date=datetime(2026, 1, 1))
            if not prices.empty:
                prices = prices[prices.index.date < today]
                if not prices.empty:
                    current_price = prices['close'].iloc[-1]
                    value_usd = shares * current_price
                    value_eur = value_usd / eur_usd
                    return_pct = ((current_price - price_usd) / price_usd) * 100
                    open_positions.append({
                        'ticker': ticker + " (NEW)",
                        'inicio_eur': cost_eur,
                        'actual_eur': value_eur,
                        'return_pct': return_pct
                    })
                    print(f"Added new: {ticker} - {shares} shares @ ${price_usd:.2f} -> ${current_price:.2f}")
        else:
            print(f"NEW NOT FOUND: {ticker}")

    # TLT (Cash IB)
    tlt_trades = session.query(IBTrade).filter(IBTrade.symbol == 'TLT', IBTrade.quantity > 0).all()
    if tlt_trades:
        tlt_shares = sum(t.quantity for t in tlt_trades)
        tlt_cost = sum(t.total_cost for t in tlt_trades)
        tlt_symbol = session.query(Symbol).filter(Symbol.code == 'TLT').first()
        if tlt_symbol:
            tlt_prices = db.get_price_history(session, tlt_symbol.id, start_date=start_date)
            if not tlt_prices.empty:
                tlt_prices = tlt_prices[tlt_prices.index.date < today]
                if not tlt_prices.empty:
                    tlt_day0 = tlt_prices['close'].iloc[0]
                    tlt_current = tlt_prices['close'].iloc[-1]
                    tlt_value_eur = (tlt_shares * tlt_current) / eur_usd
                    tlt_return = ((tlt_current - tlt_day0) / tlt_day0) * 100
                    open_positions.append({
                        'ticker': 'TLT (Cash IB)',
                        'inicio_eur': tlt_cost / eur_usd,
                        'actual_eur': tlt_value_eur,
                        'return_pct': tlt_return
                    })

# Summary
print("=" * 80)
print("POSICIONES ABIERTAS A 26/01/2026")
print("=" * 80)
sorted_open = sorted(open_positions, key=lambda x: x['actual_eur'], reverse=True)
for i, r in enumerate(sorted_open, 1):
    print(f"{i:3}. {r['ticker']:22} Inicio: {r['inicio_eur']:>10,.0f}  Actual: {r['actual_eur']:>10,.0f}  Rent: {r['return_pct']:>+6.1f}%")

total_open_inicio = sum(r['inicio_eur'] for r in open_positions)
total_open_actual = sum(r['actual_eur'] for r in open_positions)

print(f"\n{'=' * 80}")
print("POSICIONES VENDIDAS EN 2026 (rentabilidad realizada)")
print("=" * 80)
sorted_sold = sorted(sold_positions, key=lambda x: x['venta_eur'], reverse=True)
for r in sorted_sold:
    print(f"{r['ticker']:8} {r['sale_date']}  Inicio: {r['inicio_eur']:>10,.0f}  Venta: {r['venta_eur']:>10,.0f}  Rent: {r['return_pct']:>+6.1f}%")

total_sold_inicio = sum(r['inicio_eur'] for r in sold_positions)
total_sold_venta = sum(r['venta_eur'] for r in sold_positions)
realized_pnl = total_sold_venta - total_sold_inicio

# Cash calculation
futures_pnl_usd = 19760.72 - 4054.94
futures_pnl_eur = futures_pnl_usd / eur_usd
fixed_cash = 303279.92 + futures_pnl_eur + realized_pnl  # Add realized P&L to cash

print(f"\n{'=' * 80}")
print("RESUMEN TOTAL")
print("=" * 80)
print(f"Posiciones abiertas ({len(open_positions)}):")
print(f"  Inicio:              EUR {total_open_inicio:>12,.0f}")
print(f"  Actual:              EUR {total_open_actual:>12,.0f}")
print(f"  Rentabilidad:        EUR {total_open_actual - total_open_inicio:>+12,.0f}")
print(f"\nPosiciones vendidas ({len(sold_positions)}):")
print(f"  Inicio:              EUR {total_sold_inicio:>12,.0f}")
print(f"  Venta:               EUR {total_sold_venta:>12,.0f}")
print(f"  P&L Realizado:       EUR {realized_pnl:>+12,.0f}")
print(f"\nCash:")
print(f"  Renta4 + CO3365:     EUR {303279.92:>12,.0f}")
print(f"  Futuros P&L:         EUR {futures_pnl_eur:>+12,.0f}")
print(f"  Ventas realizadas:   EUR {realized_pnl:>+12,.0f}")
print(f"  Total Cash:          EUR {fixed_cash:>12,.0f}")
print(f"\n{'=' * 80}")
print(f"CARTERA TOTAL 26/01/2026")
print(f"  Acciones:            EUR {total_open_actual:>12,.0f}")
print(f"  Cash:                EUR {fixed_cash:>12,.0f}")
print(f"  TOTAL:               EUR {total_open_actual + fixed_cash:>12,.0f}")
print("=" * 80)
