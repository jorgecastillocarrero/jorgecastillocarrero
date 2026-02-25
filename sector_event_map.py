"""
Mapa de Eventos → Impacto Sub-Sectorial (Granular)
====================================================
Mapeo a nivel de sub-sector, no sector.
Cada sub-sector tiene sus propios drivers.

Score: +2 fuerte positivo, +1 positivo, -1 negativo, -2 fuerte negativo
"""

# ── SUB-SECTORES DEL S&P 500 ──────────────────────────
# Agrupados por sector ETF, con tickers representativos

SUBSECTORS = {
    # ── XLK - Technology ────────────────────────────
    "semiconductors": {
        "etf": "XLK", "label": "Semiconductors",
        "tickers": ["NVDA", "AVGO", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX",
                     "KLAC", "MCHP", "ADI", "NXPI", "ON", "MPWR", "SWKS", "QRVO", "MRVL", "GFS"],
    },
    "software_app": {
        "etf": "XLK", "label": "Software Application",
        "tickers": ["CRM", "ADSK", "CDNS", "SNPS", "FICO", "DDOG", "APP", "INTU",
                     "ANSS", "PLTR", "TTWO", "EA", "ORCL", "NOW"],
    },
    "software_infra": {
        "etf": "XLK", "label": "Software Infrastructure",
        "tickers": ["MSFT", "ADBE", "CRWD", "PANW", "FTNT", "GEN", "FFIV", "AKAM",
                     "CPAY", "ROP", "TOST", "GDDY", "PTC", "MANH", "NICE"],
    },
    "hardware": {
        "etf": "XLK", "label": "Hardware & Equipment",
        "tickers": ["AAPL", "DELL", "HPQ", "ANET", "NTAP", "STX", "WDC", "SMCI",
                     "APH", "GLW", "JBL", "KEYS", "GRMN", "TDY", "ZBRA", "FTV", "CSCO", "MSI"],
    },
    "it_services": {
        "etf": "XLK", "label": "IT Services",
        "tickers": ["ACN", "IBM", "CDW", "CTSH", "FIS", "FISV", "BR", "EPAM", "LDOS", "IT", "HPE"],
    },

    # ── XLV - Healthcare ───────────────────────────
    "pharma_major": {
        "etf": "XLV", "label": "Big Pharma",
        "tickers": ["JNJ", "LLY", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "BIIB"],
    },
    "biotech": {
        "etf": "XLV", "label": "Biotech",
        "tickers": ["MRNA", "REGN", "VRTX", "INCY", "TECH"],
    },
    "medical_devices": {
        "etf": "XLV", "label": "Medical Devices & Instruments",
        "tickers": ["ABT", "MDT", "BSX", "SYK", "ISRG", "EW", "DXCM", "ALGN",
                     "BDX", "BAX", "COO", "HOLX", "RMD", "ZBH"],
    },
    "health_services": {
        "etf": "XLV", "label": "Health Services & Plans",
        "tickers": ["UNH", "CI", "ELV", "CNC", "HUM", "MOH", "CVS", "HCA", "DVA",
                     "UHS", "GEHC", "CAH", "COR", "MCK", "HSIC"],
    },
    "diagnostics_research": {
        "etf": "XLV", "label": "Diagnostics & Research",
        "tickers": ["DHR", "TMO", "A", "IQV", "DGX", "IDXX", "CRL", "MTD", "WAT", "RVTY", "LH"],
    },

    # ── XLF - Financial Services ───────────────────
    "banks_major": {
        "etf": "XLF", "label": "Major Banks",
        "tickers": ["JPM", "BAC", "WFC", "C"],
    },
    "banks_regional": {
        "etf": "XLF", "label": "Regional Banks",
        "tickers": ["USB", "PNC", "TFC", "CFG", "MTB", "FITB", "HBAN", "KEY", "RF"],
    },
    "insurance": {
        "etf": "XLF", "label": "Insurance",
        "tickers": ["BRK-B", "PGR", "CB", "ALL", "MET", "AIG", "AFL", "TRV", "HIG",
                     "PRU", "GL", "CINF", "L", "EG", "AIZ", "ERIE",
                     "AON", "AJG", "WTW", "BRO", "MRSH", "PFG"],
    },
    "capital_markets": {
        "etf": "XLF", "label": "Capital Markets & Exchanges",
        "tickers": ["GS", "MS", "SCHW", "RJF", "HOOD", "BLK", "BX", "KKR", "APO",
                     "ARES", "BK", "BEN", "TROW", "IVZ", "NTRS",
                     "CME", "ICE", "CBOE", "NDAQ", "MCO", "SPGI", "MSCI", "FDS", "COIN", "IBKR"],
    },
    "credit_payments": {
        "etf": "XLF", "label": "Credit & Payments",
        "tickers": ["V", "MA", "AXP", "PYPL", "COF", "SYF"],
    },

    # ── XLE - Energy ───────────────────────────────
    "oil_integrated": {
        "etf": "XLE", "label": "Oil & Gas Integrated",
        "tickers": ["XOM", "CVX"],
    },
    "oil_exploration": {
        "etf": "XLE", "label": "Oil & Gas E&P",
        "tickers": ["COP", "EOG", "PXD", "DVN", "FANG", "MRO", "APA", "CTRA", "EQT", "OVV"],
    },
    "oil_midstream": {
        "etf": "XLE", "label": "Oil & Gas Midstream",
        "tickers": ["WMB", "KMI", "OKE", "TRGP"],
    },
    "oil_refining": {
        "etf": "XLE", "label": "Refining & Marketing",
        "tickers": ["MPC", "VLO", "PSX"],
    },
    "oil_services": {
        "etf": "XLE", "label": "Oilfield Services",
        "tickers": ["SLB", "HAL", "BKR"],
    },
    "solar": {
        "etf": "XLE", "label": "Solar Energy",
        "tickers": ["FSLR"],
    },

    # ── XLI - Industrials ──────────────────────────
    "aerospace_defense": {
        "etf": "XLI", "label": "Aerospace & Defense",
        "tickers": ["RTX", "LMT", "BA", "GD", "NOC", "LHX", "GE", "HII",
                     "TDG", "HWM", "LDOS", "AXON"],
    },
    "airlines": {
        "etf": "XLI", "label": "Airlines",
        "tickers": ["DAL", "UAL", "LUV"],
    },
    "machinery": {
        "etf": "XLI", "label": "Industrial Machinery",
        "tickers": ["CAT", "DE", "CMI", "EMR", "ETN", "ROK", "DOV", "ITW",
                     "AME", "IR", "PH", "OTIS", "XYL", "NDSN", "IEX", "PNR", "AOS", "PCAR"],
    },
    "construction": {
        "etf": "XLI", "label": "Construction & Building",
        "tickers": ["JCI", "CARR", "LII", "TT", "BLDR", "MAS"],
    },
    "freight_logistics": {
        "etf": "XLI", "label": "Freight & Logistics",
        "tickers": ["UPS", "FDX", "JBHT", "CHRW", "EXPD", "ODFL", "CSX", "NSC", "UNP", "WAB"],
    },
    "professional_services": {
        "etf": "XLI", "label": "Professional & Business Services",
        "tickers": ["ADP", "PAYX", "CTAS", "WM", "RSG", "URI", "GPN", "VRSK", "EFX",
                     "SNA", "SWK", "FAST", "GWW", "POOL", "ALLE"],
    },
    "engineering": {
        "etf": "XLI", "label": "Engineering & Construction",
        "tickers": ["PWR", "EME", "FIX", "J"],
    },

    # ── XLY - Consumer Discretionary ───────────────
    "ecommerce_retail": {
        "etf": "XLY", "label": "E-Commerce & Specialty Retail",
        "tickers": ["AMZN", "EBAY", "ORLY", "AZO", "BBY", "ULTA", "GPC", "TSCO", "KMX"],
    },
    "home_improvement": {
        "etf": "XLY", "label": "Home Improvement",
        "tickers": ["HD", "LOW"],
    },
    "restaurants": {
        "etf": "XLY", "label": "Restaurants & QSR",
        "tickers": ["MCD", "SBUX", "CMG", "YUM", "DPZ", "DRI"],
    },
    "travel_leisure": {
        "etf": "XLY", "label": "Travel, Hotels & Leisure",
        "tickers": ["BKNG", "ABNB", "EXPE", "MAR", "HLT", "RCL", "NCLH", "CCL",
                     "LVS", "MGM", "WYNN", "HAS"],
    },
    "auto": {
        "etf": "XLY", "label": "Autos & EV",
        "tickers": ["TSLA", "GM", "F", "APTV", "CPRT", "CVNA"],
    },
    "apparel": {
        "etf": "XLY", "label": "Apparel & Fashion",
        "tickers": ["NKE", "LULU", "ROST", "TJX", "RL", "DECK", "TPR"],
    },
    "homebuilders": {
        "etf": "XLY", "label": "Homebuilders",
        "tickers": ["DHI", "LEN", "NVR", "PHM"],
    },

    # ── XLP - Consumer Staples ─────────────────────
    "food_beverage": {
        "etf": "XLP", "label": "Food & Beverage",
        "tickers": ["KO", "PEP", "MNST", "KDP", "STZ", "BF-B", "TAP",
                     "GIS", "CPB", "HRL", "SJM", "CAG", "KHC", "MDLZ", "HSY", "MKC", "LW"],
    },
    "household_personal": {
        "etf": "XLP", "label": "Household & Personal Care",
        "tickers": ["PG", "CL", "KMB", "CLX", "CHD", "EL", "KVUE"],
    },
    "discount_grocery": {
        "etf": "XLP", "label": "Discount Stores & Grocery",
        "tickers": ["WMT", "COST", "TGT", "DG", "DLTR", "KR", "SYY"],
    },
    "tobacco": {
        "etf": "XLP", "label": "Tobacco",
        "tickers": ["PM", "MO"],
    },
    "agriculture": {
        "etf": "XLP", "label": "Agriculture & Farm Products",
        "tickers": ["ADM", "BG", "TSN"],
    },

    # ── XLU - Utilities ────────────────────────────
    "electric_regulated": {
        "etf": "XLU", "label": "Regulated Electric Utilities",
        "tickers": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "WEC",
                     "ES", "ED", "DTE", "XEL", "PEG", "EIX", "ETR", "AEE",
                     "CMS", "PPL", "FE", "EVRG", "CNP", "AES", "PNW", "LNT", "NI"],
    },
    "power_producers": {
        "etf": "XLU", "label": "Independent Power & Renewables",
        "tickers": ["NRG", "VST", "CEG", "GEV"],
    },
    "water_gas": {
        "etf": "XLU", "label": "Water & Gas Utilities",
        "tickers": ["AWK", "ATO"],
    },

    # ── XLB - Materials ────────────────────────────
    "chemicals": {
        "etf": "XLB", "label": "Chemicals & Specialty Chemicals",
        "tickers": ["LIN", "SHW", "APD", "ECL", "DD", "DOW", "PPG", "ALB", "LYB", "IFF"],
    },
    "gold": {
        "etf": "XLB", "label": "Gold Mining",
        "tickers": ["NEM"],
    },
    "copper": {
        "etf": "XLB", "label": "Copper Mining",
        "tickers": ["FCX"],
    },
    "steel": {
        "etf": "XLB", "label": "Steel",
        "tickers": ["NUE", "STLD"],
    },
    "construction_materials": {
        "etf": "XLB", "label": "Construction Materials",
        "tickers": ["VMC", "MLM", "CRH"],
    },
    "ag_inputs": {
        "etf": "XLB", "label": "Agricultural Inputs (Fertilizers)",
        "tickers": ["CF", "MOS", "CTVA"],
    },
}

# ── MAPA DE EVENTOS → IMPACTO SUB-SECTORIAL ───────────
# Ahora a nivel de sub-sector, no sector

EVENT_SUBSECTOR_MAP = {

    # ═══════════════════════════════════════════════════
    # GEOPOLITICA Y CONFLICTOS
    # ═══════════════════════════════════════════════════

    "guerra_medio_oriente": {
        "keywords": ["war", "iran", "iraq", "syria", "middle east", "israel", "gaza",
                     "military strike", "bombing", "missile attack", "hezbollah", "houthi",
                     "strait of hormuz", "red sea"],
        "impacto": {
            "oil_exploration": +2, "oil_integrated": +2, "oil_services": +1,  # Petroleo sube
            "gold": +2,                          # Refugio seguro
            "aerospace_defense": +2,             # Gasto militar
            "airlines": -2,                      # Rutas afectadas, fuel caro
            "travel_leisure": -2,                 # Turismo cae
            "ecommerce_retail": -1,              # Incertidumbre
            "freight_logistics": -1,             # Rutas marítimas (Red Sea)
        }
    },

    "guerra_rusia_ucrania": {
        "keywords": ["russia", "ukraine", "putin", "nato", "sanctions russia",
                     "nord stream", "russian invasion", "crimea", "donbass"],
        "impacto": {
            "oil_exploration": +2, "oil_integrated": +2,  # Gas/petróleo europeo
            "ag_inputs": +2,                     # Fertilizantes (Rusia gran productor)
            "gold": +2,                          # Safe haven
            "aerospace_defense": +2,             # Gasto OTAN sube
            "steel": +1,                         # Demanda militar
            "airlines": -1,                      # Rutas cerradas
            "banks_major": -1,                   # Exposición Rusia
            "chemicals": -1,                     # Gas europeo = input
        }
    },

    "tension_china_taiwan": {
        "keywords": ["china", "taiwan", "tsmc", "chip ban", "trade war china",
                     "tariff china", "huawei", "xi jinping", "south china sea",
                     "semiconductor export control"],
        "impacto": {
            "semiconductors": -2,                # Cadena de suministro chips
            "hardware": -1,                      # Componentes de China
            "gold": +1,                          # Incertidumbre
            "copper": -1,                        # China mayor consumidor, demanda cae
            "aerospace_defense": +1,             # Tensión militar
            "machinery": -1,                     # Exportaciones a China
            "ecommerce_retail": -1,              # Supply chain
            "apparel": -1,                       # Producción en China
        }
    },

    "sanciones_comerciales": {
        "keywords": ["sanctions", "embargo", "trade ban", "tariff", "import duty",
                     "export control", "trade restriction", "protectionism"],
        "impacto": {
            "steel": +2,                         # Proteccionismo = producción local
            "machinery": +1,                     # Reshoring
            "construction_materials": +1,
            "semiconductors": -1,                # Restricciones chips
            "apparel": -1,                       # Producción offshore afectada
            "auto": -1,                          # Aranceles auto
        }
    },

    "terrorismo": {
        "keywords": ["terrorist", "terrorism", "attack civilians", "homeland security",
                     "security threat", "mass shooting"],
        "impacto": {
            "aerospace_defense": +2,             # Seguridad
            "gold": +1,                          # Safe haven
            "travel_leisure": -2,                # Pánico
            "airlines": -2,
            "ecommerce_retail": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # ECONOMÍA Y POLÍTICA MONETARIA
    # ═══════════════════════════════════════════════════

    "subida_tipos_interes": {
        "keywords": ["rate hike", "interest rate increase", "fed tightening",
                     "hawkish", "monetary tightening", "higher rates", "fed funds rate"],
        "impacto": {
            "banks_major": +2, "banks_regional": +2,  # Mayor margen
            "insurance": +1,                     # Inversiones rinden mas
            "capital_markets": -1,               # Menos deals
            "homebuilders": -2,                  # Hipotecas caras
            "home_improvement": -1,
            "electric_regulated": -2,            # Yield competition
            "water_gas": -1,                     # Yield competition
            "power_producers": -1,               # Capital intensivo
            "solar": -2,                         # Growth + capital intensivo
            "software_app": -1,                  # Growth penalizado
            "biotech": -1,                       # Growth penalizado
            "medical_devices": -1,               # Growth names (ISRG)
            "gold": -1,                          # Tipos reales altos = oro baja
        }
    },

    "bajada_tipos_interes": {
        "keywords": ["rate cut", "interest rate decrease", "fed easing",
                     "dovish", "monetary easing", "lower rates"],
        "impacto": {
            "homebuilders": +2,                  # Hipotecas baratas
            "home_improvement": +1,
            "electric_regulated": +2,            # Yield attractive
            "water_gas": +1,                     # Yield attractive
            "power_producers": +1,               # Capex mas barato
            "solar": +1,                         # Growth + financiacion
            "software_app": +1,                  # Growth favorecido
            "biotech": +1,
            "medical_devices": +1,               # Growth favorecido
            "capital_markets": +1,               # Mas IPOs, deals
            "gold": +1,                          # Tipos reales bajos = oro sube
            "travel_leisure": +1,                # Gasto discrecional sube
            "airlines": +1,                      # Financiacion flota
            "banks_major": -1, "banks_regional": -2,  # Menor margen
        }
    },

    "inflacion_alta": {
        "keywords": ["inflation", "cpi increase", "price surge", "cost of living",
                     "hyperinflation", "consumer prices rising", "ppi surge"],
        "impacto": {
            "oil_exploration": +1, "oil_integrated": +1,  # Commodities suben
            "gold": +2,                          # Cobertura inflacion
            "ag_inputs": +1,                     # Precios agricolas
            "food_beverage": +1,                 # Pricing power
            "tobacco": +1,                       # Pricing power inelastico
            "discount_grocery": +1,              # Consumidores buscan descuento
            "copper": +1,                        # Commodities suben con inflacion
            "steel": +1,                         # Commodities suben
            "travel_leisure": -2,                # Discrecional cae
            "restaurants": -1,                   # Costes suben
            "apparel": -1,                       # Discrecional
            "auto": -1,                          # Financiacion mas cara
            "household_personal": -1,            # Input costs suben (PG, CL)
            "professional_services": -1,         # Wage inflation = coste
            "airlines": -1,                      # Fuel + costes operativos
        }
    },

    "recesion": {
        "keywords": ["recession", "economic downturn", "gdp contraction",
                     "unemployment surge", "layoffs", "economic crisis", "depression",
                     "nonfarm payrolls miss"],
        "impacto": {
            # Defensivos (relativamente mejor, no inmunes)
            "gold": +2,                          # Refugio
            "food_beverage": +1,                 # Defensivo pero no inmune (-17% en Q4'08)
            "household_personal": +1,            # Esencial pero no inmune (-17% en Q4'08)
            "discount_grocery": +2,              # Trade down (WMT aguanto bien en 2008)
            "tobacco": +1,                       # Inelastico
            "pharma_major": +1,                  # Defensivo (solo -1.6% en Q4'08)
            "electric_regulated": +1,            # Esencial + yield
            "health_services": +1,               # Esencial
            "water_gas": +1,                     # Defensivo esencial
            # Ciclicos (caen fuerte)
            "travel_leisure": -2,                # Primer recorte
            "restaurants": -2,
            "apparel": -2,
            "auto": -2,
            "homebuilders": -2,
            "home_improvement": -2,              # Ligado a housing (-9% Q4'08)
            "ecommerce_retail": -1,              # Gasto cae
            "banks_regional": -2,                # Morosidad
            "banks_major": -1,                   # Defaults suben
            "capital_markets": -1,
            "credit_payments": -1,               # Volumen transacciones cae
            "insurance": -1,                     # Perdidas inversion (-21% Q4'08)
            "machinery": -1,
            "freight_logistics": -1,
            "construction": -2,                  # Obra se para (-38% Q4'08)
            "engineering": -1,                   # Proyectos cancelados
            "professional_services": -1,         # Gasto empresarial cae
            "steel": -1,
            "copper": -2,                        # Demanda industrial cae (-54% Q4'08)
            "chemicals": -1,
            "construction_materials": -1,        # Demanda cae (-25% Q1'09)
            "ag_inputs": -1,                     # Demanda cae (-47% Q4'08)
            # Tecnologia (capex cuts)
            "semiconductors": -1,                # Capex cuts (-30% Q4'08)
            "software_app": -1,                  # Gasto enterprise cae
            "hardware": -1,                      # Capex cae (-33% Q4'08)
            "it_services": -1,                   # Consulting cae
            # Energia (demanda cae)
            "oil_exploration": -1,               # Demanda cae
            "oil_services": -1,                  # Drilling cuts (-43% Q4'08)
            "oil_midstream": -1,                 # Volumenes caen
            "solar": -1,                         # Capex cae
            "power_producers": -1,               # Demanda industrial cae
            # Healthcare growth names
            "medical_devices": -1,               # Procedimientos electivos se retrasan
            "diagnostics_research": -1,          # R&D budgets cut
        }
    },

    "crecimiento_economico": {
        "keywords": ["gdp growth", "economic boom", "expansion", "job creation",
                     "consumer spending strong", "economic recovery", "soft landing",
                     "nonfarm payrolls beat"],
        "impacto": {
            # Ciclicos (se benefician directamente)
            "travel_leisure": +2,                # Gasto discrecional
            "restaurants": +1,
            "apparel": +1,                       # Gasto consumo
            "auto": +1,
            "homebuilders": +1,
            "home_improvement": +1,              # Housing activity
            "ecommerce_retail": +1,              # Retail spending
            # Financieros (mas actividad)
            "banks_major": +1,
            "banks_regional": +1,                # Loan growth
            "capital_markets": +1,               # IPOs, M&A
            "credit_payments": +1,               # Volumen transacciones
            "insurance": +1,                     # Mas polizas
            # Industriales (capex + demanda)
            "machinery": +2,                     # Capex corporativo
            "freight_logistics": +1,
            "construction": +1,                  # Mas obra
            "engineering": +1,                   # Proyectos
            "construction_materials": +1,        # Demanda
            "professional_services": +1,         # Servicios empresariales
            "airlines": +1,                      # Viajes negocios + leisure
            # Materiales (demanda industrial)
            "copper": +2,                        # Demanda industrial
            "steel": +1,
            "chemicals": +1,
            "ag_inputs": +1,                     # Gasto agricola
            # Tecnologia (presupuestos IT suben)
            "semiconductors": +1,                # Capex cycle
            "hardware": +1,                      # Capex empresarial
            "it_services": +1,                   # Consulting demand
            "software_app": +1,                  # Enterprise spending
            # Energia (mas demanda)
            "oil_services": +1,                  # Drilling activity
            "oil_midstream": +1,                 # Volumes
            "oil_integrated": +1,                # Demanda crudo
            "power_producers": +1,               # Demanda industrial
            # Healthcare growth
            "medical_devices": +1,               # Mas procedimientos
            "diagnostics_research": +1,          # R&D spending
            # Rotacion de defensivos (sale dinero)
            "food_beverage": -1,                 # Rotacion de defensivo
            "electric_regulated": -1,
            "gold": -1,                          # Menos necesidad refugio
            "tobacco": -1,                       # Rotacion a growth
            "discount_grocery": -1,              # Rotacion a premium
        }
    },

    "crisis_bancaria": {
        "keywords": ["bank failure", "bank run", "banking crisis", "bank collapse",
                     "silicon valley bank", "credit suisse", "lehman", "bank bailout",
                     "financial crisis", "credit crunch", "deposit flight"],
        "impacto": {
            "banks_regional": -2,                # Epicentro
            "banks_major": -2,
            "capital_markets": -2,               # IPOs se paran, AUM cae
            "credit_payments": -2,               # Defaults, credit freeze
            "insurance": -2,                     # AIG en 2008
            "homebuilders": -1,                  # Credit freeze -> hipotecas
            "auto": -1,                          # Financiacion auto se seca
            "copper": -1,                        # Demanda industrial cae
            "gold": +2,                          # Flight to safety
            "electric_regulated": +1,            # Defensivo
            "food_beverage": +1,
            "pharma_major": +1,
            "discount_grocery": +1,              # Defensivo
        }
    },

    "estimulo_fiscal": {
        "keywords": ["stimulus", "fiscal spending", "infrastructure bill",
                     "government spending", "relief package", "quantitative easing",
                     "build back better", "chips act"],
        "impacto": {
            "construction_materials": +2,        # Infraestructura
            "steel": +2,
            "machinery": +2,
            "engineering": +2,
            "construction": +2,
            "semiconductors": +1,                # CHIPS Act
            "copper": +1,
            "solar": +1,                         # Subsidios energia limpia
            "ecommerce_retail": +1,              # Estimulo al consumo
            "auto": +1,                          # Subsidios EV / cash for clunkers
            "airlines": +1,                      # Bailouts (2020)
            "travel_leisure": +1,                # Stimulus checks -> viajes
            "discount_grocery": +1,              # Stimulus checks
        }
    },

    # ═══════════════════════════════════════════════════
    # ENERGÍA Y COMMODITIES
    # ═══════════════════════════════════════════════════

    "precio_petroleo_sube": {
        "keywords": ["oil price surge", "crude oil rally", "opec cut", "oil supply cut",
                     "brent surge", "wti surge", "oil shortage", "refinery shutdown",
                     "oil production cut"],
        "impacto": {
            "oil_exploration": +2, "oil_integrated": +2, "oil_services": +2,
            "oil_midstream": +1, "oil_refining": +1,
            "gold": +1,                          # Inflacion / geopolitica correlacionada
            "airlines": -2,                      # Fuel cost
            "freight_logistics": -1,             # Transporte caro
            "chemicals": -1,                     # Petroquimica input
            "auto": -1,                          # Gasolina cara -> menos ventas
            "agriculture": -1,                   # Costes transporte/input suben
        }
    },

    "precio_petroleo_baja": {
        "keywords": ["oil price crash", "crude oil drop", "opec increase", "oil glut",
                     "oil demand collapse", "oil oversupply", "shale bust"],
        "impacto": {
            "oil_exploration": -2, "oil_integrated": -1, "oil_services": -2,
            "oil_midstream": -1,                 # Volumenes y precios caen
            "oil_refining": -1,                  # Crack spreads se comprimen
            "airlines": +2,                      # Fuel barato
            "freight_logistics": +1,
            "auto": +1,                          # Gasolina barata
            "chemicals": +1,                     # Input barato
            "discount_grocery": +1,              # Transporte mas barato
            "agriculture": +1,                   # Costes input bajan
        }
    },

    "transicion_energetica": {
        "keywords": ["green energy", "renewable", "solar subsidy", "wind power",
                     "electric vehicle mandate", "carbon tax", "climate policy",
                     "paris agreement", "clean energy", "hydrogen", "ev mandate"],
        "impacto": {
            "solar": +2,
            "power_producers": +2,               # Renovables
            "auto": +1,                          # EV (Tesla)
            "copper": +2,                        # Cableado EV/renovables (driver clave)
            "construction": +1,                  # Instalaciones energia
            "engineering": +1,                   # Proyectos renovables
            "oil_exploration": -2,               # Fosiles penalizados
            "oil_services": -1,
            "oil_midstream": -1,                 # Menor volumen fosil a futuro
        }
    },

    "desastre_natural": {
        "keywords": ["hurricane", "earthquake", "flood", "wildfire", "drought",
                     "natural disaster", "tornado", "tsunami", "extreme weather"],
        "impacto": {
            "construction_materials": +2,        # Reconstrucción
            "construction": +2,
            "homebuilders": +1,                  # Reemplazo viviendas
            "insurance": -2,                     # Pagos masivos
            "electric_regulated": -1,            # Infraestructura dañada
            "ag_inputs": +1,                     # Precios agrícolas suben
            "agriculture": -1,                   # Cosechas dañadas
        }
    },

    # ═══════════════════════════════════════════════════
    # SALUD Y PANDEMIAS
    # ═══════════════════════════════════════════════════

    "pandemia": {
        "keywords": ["pandemic", "virus outbreak", "covid", "coronavirus",
                     "lockdown", "quarantine", "epidemic", "who emergency",
                     "variant", "bird flu", "h5n1"],
        "impacto": {
            "pharma_major": +2,                  # Vacunas, tratamientos
            "biotech": +2,                       # I+D vacunas
            "diagnostics_research": +2,          # Testing
            "software_infra": +2,                # Trabajo remoto
            "software_app": +1,                  # Cloud, SaaS
            "ecommerce_retail": +2,              # Compra online
            "discount_grocery": +1,              # Esencial
            "food_beverage": +1,
            "household_personal": +1,            # Productos esenciales (PG, CL, KMB)
            "travel_leisure": -2,                # Cerrado
            "airlines": -2,
            "restaurants": -2,
            "apparel": -1,
            "oil_exploration": -2,               # Demanda colapsa
            "auto": -1,
            "medical_devices": -1,               # Cirugia electiva se cancela
            "copper": -1,                        # Demanda industrial para
            "construction": -1,                  # Obras paradas
            "machinery": -1,                     # Fabricas paradas
        }
    },

    "fda_aprobacion": {
        "keywords": ["fda approves", "breakthrough therapy", "drug approval",
                     "clinical trial success", "vaccine approved", "fda clearance"],
        "impacto": {
            "pharma_major": +1,
            "biotech": +2,
            "medical_devices": +1,
            "diagnostics_research": +1,          # Mas testing/research
            "health_services": +1,               # Nuevos tratamientos
        }
    },

    "regulacion_pharma": {
        "keywords": ["drug pricing", "medicare negotiate", "pharmaceutical regulation",
                     "drug price cap", "insulin price", "pharmacy benefit"],
        "impacto": {
            "pharma_major": -2,                  # Presión precios
            "biotech": -1,
            "health_services": -1,               # PBMs afectados
            "discount_grocery": +1,              # Farmacia discount
        }
    },

    # ═══════════════════════════════════════════════════
    # TECNOLOGÍA
    # ═══════════════════════════════════════════════════

    "innovacion_ai": {
        "keywords": ["artificial intelligence", "ai breakthrough", "chatgpt", "openai",
                     "machine learning", "generative ai", "large language model",
                     "nvidia ai", "ai chip", "data center"],
        "impacto": {
            "semiconductors": +2,                # GPUs, AI chips, HBM
            "software_infra": +2,                # Cloud, cybersecurity
            "hardware": +1,                      # Servidores, memorias RAM/SSD suben
            "it_services": +1,                   # Consulting AI
            "power_producers": +2,               # Data centers = mucha mas energia
            "copper": +1,                        # Cableado data centers
            "electric_regulated": +1,            # Demanda electrica data centers
            "diagnostics_research": +1,          # AI en laboratorios
            "software_app": -1,                  # IA disruptea SaaS tradicional
        }
    },

    "regulacion_tech": {
        "keywords": ["antitrust tech", "tech regulation", "data privacy", "gdpr",
                     "big tech breakup", "section 230", "social media ban",
                     "tech monopoly", "doj lawsuit"],
        "impacto": {
            "software_app": -1,
            "software_infra": -1,
            "ecommerce_retail": -1,              # Amazon, etc
            "it_services": -1,
        }
    },

    "ciberseguridad": {
        "keywords": ["cyber attack", "data breach", "ransomware", "hacking",
                     "cybersecurity", "cyber warfare", "critical infrastructure attack"],
        "impacto": {
            "software_infra": +2,                # CrowdStrike, Palo Alto
            "it_services": +1,                   # Consulting seguridad
            "banks_major": -1,                   # Target frecuente
        }
    },

    # ═══════════════════════════════════════════════════
    # CONSUMO
    # ═══════════════════════════════════════════════════

    "confianza_consumidor_alta": {
        "keywords": ["consumer confidence surge", "retail sales beat",
                     "holiday shopping record", "consumer spending strong", "wage growth",
                     "black friday record"],
        "impacto": {
            "travel_leisure": +2,
            "restaurants": +1,
            "ecommerce_retail": +2,
            "apparel": +1,
            "home_improvement": +1,
            "auto": +1,
            "airlines": +1,                      # Mas viajes leisure
            "credit_payments": +1,               # Mas transacciones
            "professional_services": +1,         # Mas contratacion
            "food_beverage": -1,                 # Rotacion de defensivo
            "household_personal": -1,            # Rotacion a discretionary
            "tobacco": -1,                       # Rotacion a growth
            "gold": -1,                          # Risk-on
        }
    },

    "crisis_consumo": {
        "keywords": ["consumer confidence drop", "retail sales miss",
                     "credit card default", "consumer debt crisis", "foreclosure",
                     "student loan", "delinquency rate"],
        "impacto": {
            "discount_grocery": +2,              # Trade down
            "food_beverage": +1,
            "tobacco": +1,                       # Inelastico
            "household_personal": +1,            # Productos esenciales
            "pharma_major": +1,                  # Defensivo
            "travel_leisure": -2,
            "restaurants": -1,
            "apparel": -2,
            "auto": -1,
            "homebuilders": -1,
            "ecommerce_retail": -1,              # Gasto cae
            "home_improvement": -1,              # Discretionary cuts
            "credit_payments": -1,               # Morosidad
            "banks_regional": -1,
            "professional_services": -1,         # Menos contratacion
        }
    },

    # ═══════════════════════════════════════════════════
    # INMOBILIARIO
    # ═══════════════════════════════════════════════════

    "boom_inmobiliario": {
        "keywords": ["housing boom", "home prices record", "construction boom",
                     "housing starts surge", "mortgage demand surge"],
        "impacto": {
            "homebuilders": +2,
            "home_improvement": +2,
            "construction_materials": +2,
            "construction": +2,
            "engineering": +1,                   # Proyectos
            "steel": +1,
            "copper": +1,
            "banks_major": +1,                   # Hipotecas
            "banks_regional": +1,                # Mortgage origination
            "insurance": +1,                     # Mas polizas
        }
    },

    "crisis_inmobiliaria": {
        "keywords": ["housing crash", "mortgage crisis", "subprime",
                     "foreclosure wave", "real estate collapse", "housing bubble"],
        "impacto": {
            "homebuilders": -2,
            "home_improvement": -2,
            "construction": -2,                  # Obra se para
            "construction_materials": -2,        # Demanda materiales cae
            "engineering": -1,                   # Proyectos cancelados
            "steel": -1,                         # Menos construccion
            "copper": -1,                        # Menos cableado
            "banks_major": -2,                   # Hipotecas toxicas
            "banks_regional": -2,
            "insurance": -1,
            "capital_markets": -1,               # MBS, CDOs toxicos
            "credit_payments": -1,               # Defaults suben
            "gold": +1,                          # Refugio
        }
    },

    "infraestructura": {
        "keywords": ["infrastructure plan", "bridge repair", "road construction",
                     "transportation bill", "public works", "rail expansion",
                     "broadband expansion"],
        "impacto": {
            "construction_materials": +2,
            "construction": +2,
            "steel": +2,
            "engineering": +2,
            "machinery": +1,
            "copper": +1,
        }
    },

    # ═══════════════════════════════════════════════════
    # CHINA Y MATERIAS PRIMAS
    # ═══════════════════════════════════════════════════

    "demanda_china_fuerte": {
        "keywords": ["china demand", "china growth", "china stimulus",
                     "china construction", "china imports surge", "china pmi beat",
                     "china manufacturing expansion"],
        "impacto": {
            "copper": +2,                        # China = 50% demanda global
            "steel": +2,                         # China = mayor consumidor
            "construction_materials": +1,        # Infraestructura china
            "chemicals": +1,
            "oil_exploration": +1,               # Mas demanda crudo
            "oil_integrated": +1,                # Mas demanda crudo
            "machinery": +1,                     # Exportaciones
            "ag_inputs": +1,                     # Demanda agricola
            "agriculture": +1,                   # Importaciones alimentos
            "semiconductors": +1,                # Componentes
            "auto": +1,                          # Mercado auto China
            "gold": -1,                          # Risk-on
        }
    },

    "china_desaceleracion": {
        "keywords": ["china slowdown", "china recession", "china property crisis",
                     "evergrande", "china deflation", "china exports drop",
                     "china pmi miss", "china manufacturing contraction"],
        "impacto": {
            "copper": -2,                        # Demanda colapsa (China = 50%)
            "steel": -2,                         # China = mayor consumidor
            "construction_materials": -1,        # Infraestructura para
            "chemicals": -1,
            "oil_exploration": -1,
            "oil_integrated": -1,                # Menos demanda crudo
            "machinery": -1,                     # Exportaciones caen
            "ag_inputs": -1,                     # Demanda fertilizantes cae
            "agriculture": -1,                   # Importaciones caen
            "semiconductors": -1,                # Menos componentes
            "auto": -1,                          # Mercado auto China
            "gold": +1,                          # Incertidumbre global
        }
    },

    "supply_chain_crisis": {
        "keywords": ["supply chain disruption", "shipping delay", "container shortage",
                     "port congestion", "chip shortage", "semiconductor shortage",
                     "suez canal", "panama canal"],
        "impacto": {
            "freight_logistics": +1,             # Precios flete suben
            "semiconductors": -1,                # Escasez
            "auto": -2,                          # Sin chips
            "hardware": -1,
            "ecommerce_retail": -1,              # Entregas retrasadas
            "ag_inputs": +1,                     # Escasez = precios suben
        }
    },

    # ═══════════════════════════════════════════════════
    # AVIACION
    # ═══════════════════════════════════════════════════

    "accidente_aviacion": {
        "keywords": ["plane crash", "aviation accident", "aircraft grounded",
                     "faa grounding", "737 max", "airline disaster", "fatal crash",
                     "ntsb investigation", "airworthiness directive"],
        "impacto": {
            "airlines": -2,
            "aerospace_defense": -1,
            "travel_leisure": -1,
            "insurance": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # AUTOMOTRIZ
    # ═══════════════════════════════════════════════════

    "recall_auto_masivo": {
        "keywords": ["vehicle recall", "auto recall", "nhtsa recall",
                     "defective airbag", "takata airbag", "ignition switch",
                     "dieselgate", "emissions cheating", "autopilot crash"],
        "impacto": {
            "auto": -2,
            "insurance": -1,
        }
    },

    "revolucion_ev": {
        "keywords": ["electric vehicle", "ev adoption", "ev sales record",
                     "battery breakthrough", "ev mandate", "charging network",
                     "solid state battery", "ev tax credit"],
        "impacto": {
            "auto": +2,
            "copper": +1,
            "semiconductors": +1,
            "oil_exploration": -1,
            "oil_refining": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # ALIMENTOS: RECALLS Y CONTAMINACION
    # ═══════════════════════════════════════════════════

    "recall_alimentario": {
        "keywords": ["food recall", "contamination", "e. coli outbreak",
                     "salmonella", "listeria", "food safety", "fda recall",
                     "food poisoning outbreak"],
        "impacto": {
            "food_beverage": -1,
            "discount_grocery": -1,
            "restaurants": -1,
            "agriculture": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # PHARMA: ESCANDALOS
    # ═══════════════════════════════════════════════════

    "escandalo_pharma": {
        "keywords": ["opioid", "drug price gouging", "pharma fraud",
                     "clinical trial fraud", "drug withdrawal", "vioxx",
                     "oxycontin", "drug scandal", "pharma scandal"],
        "impacto": {
            "pharma_major": -2,
            "health_services": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # FRAUDE CONTABLE
    # ═══════════════════════════════════════════════════

    "fraude_contable": {
        "keywords": ["accounting fraud", "earnings manipulation", "sec fraud",
                     "ponzi scheme", "financial fraud", "restatement",
                     "enron", "worldcom", "wirecard"],
        "impacto": {
            "capital_markets": -1,
            "banks_major": -1,
            "it_services": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # ESCANDALO BANCARIO ESPECIFICO
    # ═══════════════════════════════════════════════════

    "escandalo_bancario": {
        "keywords": ["fake accounts", "money laundering", "libor manipulation",
                     "forex rigging", "wells fargo scandal", "bank fine",
                     "rogue trader", "bank misconduct"],
        "impacto": {
            "banks_major": -2,
            "banks_regional": -1,
            "capital_markets": -1,
            "credit_payments": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # DATOS Y PRIVACIDAD
    # ═══════════════════════════════════════════════════

    "escandalo_datos_privacidad": {
        "keywords": ["data breach scandal", "cambridge analytica", "privacy scandal",
                     "user data misuse", "facebook scandal", "ftc fine",
                     "theranos", "tech fraud"],
        "impacto": {
            "software_app": -2,
            "software_infra": -1,
            "ecommerce_retail": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # CRYPTO
    # ═══════════════════════════════════════════════════

    "crypto_colapso": {
        "keywords": ["bitcoin crash", "crypto collapse", "ftx", "exchange collapse",
                     "stablecoin depeg", "crypto regulation", "crypto winter",
                     "defi hack"],
        "impacto": {
            "capital_markets": -1,
            "software_app": -1,
            "banks_major": +1,
        }
    },

    # ═══════════════════════════════════════════════════
    # HUELGAS SECTORIALES
    # ═══════════════════════════════════════════════════

    "huelga_sector": {
        "keywords": ["strike", "labor dispute", "uaw strike", "union walkout",
                     "port strike", "rail strike", "work stoppage",
                     "collective bargaining"],
        "impacto": {
            "auto": -1,
            "freight_logistics": -1,
            "aerospace_defense": -1,
            "airlines": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # ACCIDENTE INDUSTRIAL / ENERGIA
    # ═══════════════════════════════════════════════════

    "accidente_industrial_energia": {
        "keywords": ["refinery explosion", "oil spill", "pipeline rupture",
                     "chemical plant explosion", "mining accident",
                     "tailings dam", "platform fire"],
        "impacto": {
            "oil_refining": -2,
            "oil_exploration": -1,
            "oil_integrated": -1,
            "oil_services": -1,
            "oil_midstream": -1,                 # Pipeline risk
            "chemicals": -1,
            "copper": -1,                        # Mining accident
            "insurance": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # DISRUPCION MARITIMA
    # ═══════════════════════════════════════════════════

    "disrupcion_maritima": {
        "keywords": ["port strike", "port congestion", "shipping disruption",
                     "canal drought", "panama canal", "shipping backlog",
                     "freight rate surge"],
        "impacto": {
            "freight_logistics": +1,
            "ecommerce_retail": -1,
            "auto": -1,
            "food_beverage": -1,
            "chemicals": -1,
        }
    },

    # ═══════════════════════════════════════════════════
    # CONTRATOS DEFENSA MAYORES
    # ═══════════════════════════════════════════════════

    "contrato_defensa_mayor": {
        "keywords": ["defense contract", "pentagon contract", "military procurement",
                     "nato spending", "defense budget", "weapons program",
                     "f-35", "aircraft carrier", "missile defense"],
        "impacto": {
            "aerospace_defense": +2,
            "engineering": +1,
            "steel": +1,
            "semiconductors": +1,
        }
    },
}

# ── PRINT SUMMARY ──────────────────────────────────────
def print_subsector_summary():
    """Print what events affect each sub-sector."""

    # Aggregate impacts per sub-sector
    subsector_impacts = {}
    for event_name, event_data in EVENT_SUBSECTOR_MAP.items():
        for subsec, score in event_data['impacto'].items():
            subsector_impacts.setdefault(subsec, []).append((event_name, score))

    # Print by ETF sector
    current_etf = None
    for subsec_id, subsec_data in SUBSECTORS.items():
        etf = subsec_data['etf']
        if etf != current_etf:
            print(f"\n{'=' * 70}")
            print(f"{etf}")
            print(f"{'=' * 70}")
            current_etf = etf

        impacts = subsector_impacts.get(subsec_id, [])
        pos = [(n, s) for n, s in impacts if s > 0]
        neg = [(n, s) for n, s in impacts if s < 0]

        print(f"\n  {subsec_data['label']} ({len(subsec_data['tickers'])} stocks)")
        if pos:
            pos_str = ", ".join(f"{n}({'++' if s>=2 else '+'})" for n, s in sorted(pos, key=lambda x: -x[1]))
            print(f"    + {pos_str}")
        if neg:
            neg_str = ", ".join(f"{n}({'--' if s<=-2 else '-'})" for n, s in sorted(neg, key=lambda x: x[1]))
            print(f"    - {neg_str}")
        if not pos and not neg:
            print(f"    (sin eventos mapeados)")


if __name__ == '__main__':
    print_subsector_summary()

    # Stats
    total_events = len(EVENT_SUBSECTOR_MAP)
    total_keywords = sum(len(e['keywords']) for e in EVENT_SUBSECTOR_MAP.values())
    total_subsectors = len(SUBSECTORS)
    total_impacts = sum(len(e['impacto']) for e in EVENT_SUBSECTOR_MAP.values())
    print(f"\n\nTOTAL: {total_events} eventos, {total_keywords} keywords, "
          f"{total_subsectors} sub-sectores, {total_impacts} impactos mapeados")
