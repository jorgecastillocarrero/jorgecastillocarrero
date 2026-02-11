# Arquitectura Macroeconómica - PatrimonioSmart

## 1. Visión General

Sistema de datos macroeconómicos que alimenta:
- Detección de régimen de mercado
- Ajuste de ratios diana adaptativos
- Señales de contexto para selección de acciones

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUENTES DE DATOS                              │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│    FRED     │   Yahoo     │    FMP      │   Quandl    │  Custom │
│ (Macro US)  │  (VIX,FX)   │ (Earnings)  │  (Alt data) │ (Calc)  │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴────┬────┘
       │             │             │             │           │
       ▼             ▼             ▼             ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   macro_indicators (Raw Data)                    │
│  - 200+ series macroeconómicas                                  │
│  - Frecuencia: diaria, semanal, mensual, trimestral             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 macro_features (Calculated)                      │
│  - Transformaciones: YoY, MoM, Z-score, percentiles             │
│  - Derivados: yield curve slope, real rates, spreads            │
│  - Scores por dimensión: growth, inflation, stress, liquidity   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 macro_regime (Classification)                    │
│  - Régimen actual (7 regímenes naturales)                       │
│  - Probabilidades por régimen                                   │
│  - Señales de transición                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          features_master (Enlace con Acciones)                   │
│  - macro_regime_current                                         │
│  - ratios_diana ajustados por régimen                           │
│  - regime_alignment_score por acción                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Las 12 Categorías Macroeconómicas

### Catálogo de Categorías

| # | Categoría | Descripción | Rol en Ciclo |
|---|-----------|-------------|--------------|
| 1 | **INFLATION** | IPC, PCE, PPI, salarios, expectativas | Lagging/Coincident |
| 2 | **GROWTH** | PIB, producción industrial, consumo, PMIs | Coincident |
| 3 | **LABOR** | Desempleo, payrolls, claims, vacantes | Lagging |
| 4 | **POLICY** | Tipos oficiales, tipos reales, balance Fed | Leading |
| 5 | **RATES** | Curva de tipos, niveles y pendientes | Leading |
| 6 | **LIQUIDITY** | M2, crédito bancario, condiciones financieras | Leading |
| 7 | **CREDIT** | Spreads IG/HY, defaults, deuda pública/privada | Leading |
| 8 | **STRESS** | VIX, volatilidad realizada, índices de estrés | Leading |
| 9 | **FX** | Dólar, EUR/USD, balanza comercial | Coincident |
| 10 | **COMMODITIES** | Energía, metales, agricultura, transporte | Leading |
| 11 | **PRECIOUS** | Oro, plata, real rates vs oro | Coincident |
| 12 | **SENTIMENT** | EPU, confianza consumidor/empresarial, geopolítico | Leading |

---

## 3. Tipos de Indicadores Macro (Clasificaciones)

### 3.1 Por Rol en el Ciclo

| Tipo | Descripción | Ejemplos |
|------|-------------|----------|
| **LEADING** | Anticipan el ciclo (3-12 meses) | PMIs, curva de tipos, crédito, condiciones financieras, claims |
| **COINCIDENT** | Describen el presente | Producción industrial, empleo, ventas retail |
| **LAGGING** | Confirman tarde | Inflación core, desempleo (a veces), inventarios |

### 3.2 Por Naturaleza Estadística

| Tipo | Descripción | Ejemplos |
|------|-------------|----------|
| **LEVEL** | Valor absoluto | CPI index, GDP level |
| **RATE** | Tasa de cambio | YoY, MoM, QoQ |
| **STOCK** | Acumulado | Deuda total, M2 |
| **FLOW** | Flujo por periodo | Déficit, ventas, producción |
| **DIFFUSION** | % de componentes | PMI, breadth |
| **REAL** | Ajustado por inflación | GDP real, tipos reales |
| **NOMINAL** | Sin ajustar | GDP nominal, tipos nominales |

### 3.3 Por Comportamiento en Datos

| Tipo | Descripción | Implicaciones |
|------|-------------|---------------|
| **REVISABLE** | Sujeto a revisiones posteriores | GDP, payrolls → usar con cuidado en backtest |
| **NO_REVISABLE** | Dato final | VIX, precios → más confiable |
| **HIGH_VOLATILITY** | Alta variabilidad | Claims semanales → suavizar |
| **STABLE** | Baja variabilidad | Desempleo → cambios significativos |
| **REAL_TIME** | Disponible inmediatamente | VIX, precios |
| **DELAYED** | Publicación con rezago | GDP (1 mes+), CPI (2 semanas) |

### 3.4 Por Rol en el Modelo

| Tipo | Descripción | Ejemplos |
|------|-------------|----------|
| **CORE_INPUT** | Definen régimen directamente | VIX, curva, PMI, spreads crédito |
| **CONTEXT_INPUT** | Explican pero no deciden | Oro, dólar, sentiment surveys |
| **DERIVED** | Calculados a partir de otros | Z-scores, momentum, tendencias |
| **OUTPUT** | Resultado del modelo | Probabilidad régimen, alertas |

---

## 4. Tabla: macro_catalog

```sql
CREATE TABLE macro_catalog (
    indicator_id SERIAL PRIMARY KEY,

    -- Identificación
    ticker VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,

    -- Clasificación por categoría
    category VARCHAR(30) NOT NULL,            -- 12 categorías principales
    subcategory VARCHAR(50),                  -- headline, core, etc.

    -- Clasificación por tipo
    cycle_role VARCHAR(20),                   -- leading, coincident, lagging
    stat_nature VARCHAR(20),                  -- level, rate, stock, flow, diffusion
    data_behavior VARCHAR(20),                -- revisable, stable, volatile, real_time
    model_role VARCHAR(20),                   -- core_input, context_input, derived

    -- Fuente y metadata
    source VARCHAR(50) NOT NULL,
    source_ticker VARCHAR(50),
    frequency VARCHAR(20) NOT NULL,
    units VARCHAR(50),
    seasonal_adj BOOLEAN DEFAULT TRUE,
    publication_lag_days INTEGER,             -- Días de retraso en publicación

    -- Transformación por defecto
    default_transform VARCHAR(20),            -- level, yoy, mom, diff, zscore

    -- Peso en scores
    growth_weight FLOAT DEFAULT 0,
    inflation_weight FLOAT DEFAULT 0,
    stress_weight FLOAT DEFAULT 0,
    liquidity_weight FLOAT DEFAULT 0,

    -- Estado
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_macro_cat_category ON macro_catalog(category);
CREATE INDEX idx_macro_cat_cycle_role ON macro_catalog(cycle_role);
CREATE INDEX idx_macro_cat_model_role ON macro_catalog(model_role);
```

---

## 5. Indicadores por Categoría

### 5.1 INFLATION (Inflación y Precios)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| CPIAUCSL | CPI All Items | headline | Monthly | lagging |
| CPILFESL | CPI Core (ex Food/Energy) | core | Monthly | lagging |
| PCEPI | PCE Price Index | headline | Monthly | lagging |
| PCEPILFE | PCE Core | core | Monthly | lagging |
| PPIACO | PPI All Commodities | production | Monthly | leading |
| AHETPI | Avg Hourly Earnings | wages | Monthly | coincident |
| MICH | Michigan Inflation Expect | expectations | Monthly | leading |
| T5YIE | 5Y Breakeven Inflation | market | Daily | leading |
| T10YIE | 10Y Breakeven Inflation | market | Daily | leading |

### 5.2 GROWTH (Crecimiento y Actividad Real)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| GDP | Gross Domestic Product | gdp | Quarterly | coincident |
| GDPC1 | Real GDP (Chained) | gdp_real | Quarterly | coincident |
| INDPRO | Industrial Production | production | Monthly | coincident |
| RSAFS | Retail Sales | consumption | Monthly | coincident |
| PCE | Personal Consumption Exp | consumption | Monthly | coincident |
| NEWORDER | New Orders Durable Goods | orders | Monthly | leading |
| DGORDER | Durable Goods Orders | orders | Monthly | leading |
| UMTMNO | Manufacturers New Orders | orders | Monthly | leading |
| BUSINV | Business Inventories | inventories | Monthly | lagging |
| ISRATIO | Inventory/Sales Ratio | inventories | Monthly | lagging |

### 5.3 LABOR (Mercado Laboral)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| UNRATE | Unemployment Rate | unemployment | Monthly | lagging |
| PAYEMS | Nonfarm Payrolls | employment | Monthly | coincident |
| ICSA | Initial Jobless Claims | claims | Weekly | leading |
| CCSA | Continued Claims | claims | Weekly | leading |
| JTSJOL | Job Openings (JOLTS) | openings | Monthly | leading |
| JTSQUR | Quits Rate | turnover | Monthly | leading |
| CIVPART | Labor Force Participation | participation | Monthly | lagging |
| U6RATE | U6 Unemployment | underemployment | Monthly | lagging |

### 5.4 POLICY (Política Monetaria)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| FEDFUNDS | Fed Funds Rate | official | Monthly | leading |
| DFF | Fed Funds Daily | official | Daily | leading |
| DFEDTARU | Fed Funds Target Upper | target | Daily | leading |
| REAINTRATREARAT10Y | 10Y Real Interest Rate | real_rate | Monthly | leading |
| WALCL | Fed Total Assets | balance_sheet | Weekly | leading |
| WTREGEN | Fed Treasury Holdings | balance_sheet | Weekly | leading |
| RRPONTSYD | Reverse Repo | liquidity | Daily | coincident |

### 5.5 RATES (Curva de Tipos)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| DTB3 | 3-Month T-Bill | short | Daily | coincident |
| DTB6 | 6-Month T-Bill | short | Daily | coincident |
| DGS1 | 1-Year Treasury | short | Daily | coincident |
| DGS2 | 2-Year Treasury | medium | Daily | coincident |
| DGS5 | 5-Year Treasury | medium | Daily | coincident |
| DGS10 | 10-Year Treasury | long | Daily | coincident |
| DGS30 | 30-Year Treasury | long | Daily | coincident |
| T10Y2Y | 10Y-2Y Spread | slope | Daily | leading |
| T10Y3M | 10Y-3M Spread | slope | Daily | leading |
| T10YFF | 10Y-Fed Funds Spread | slope | Daily | leading |

### 5.6 LIQUIDITY (Liquidez y Condiciones Financieras)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| M2SL | M2 Money Supply | monetary | Monthly | leading |
| M2V | Velocity of M2 | velocity | Quarterly | lagging |
| TOTLL | Total Loans & Leases | credit | Weekly | coincident |
| DRTSCILM | Bank Lending Standards | survey | Quarterly | leading |
| NFCI | Chicago Fed NFCI | conditions | Weekly | leading |
| ANFCI | Adjusted NFCI | conditions | Weekly | leading |
| STLFSI4 | St Louis Fin Stress | stress | Weekly | leading |

### 5.7 CREDIT (Crédito y Spreads)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| BAMLC0A0CM | IG Corporate Spread | ig_spread | Daily | leading |
| BAMLH0A0HYM2 | HY Corporate Spread | hy_spread | Daily | leading |
| BAMLC0A4CBBB | BBB Corporate Spread | bbb_spread | Daily | leading |
| TEDRATE | TED Spread | interbank | Daily | leading |
| GFDEBTN | Federal Debt Total | govt_debt | Quarterly | context |
| GFDEGDQ188S | Federal Debt/GDP | govt_debt | Quarterly | context |
| TDSP | Household Debt Service | private_debt | Quarterly | lagging |

### 5.8 STRESS (Riesgo y Volatilidad)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| VIXCLS | VIX Index | implied_vol | Daily | leading |
| -- | VIX3M (via Yahoo) | term_structure | Daily | leading |
| -- | MOVE Index (via Yahoo) | bond_vol | Daily | leading |
| NFCI | Chicago Fed NFCI | stress_index | Weekly | leading |
| STLFSI4 | St Louis Stress | stress_index | Weekly | leading |
| CPALTT01USM657N | Correl (implied) | systemic | Monthly | leading |

### 5.9 FX (Divisas y Sector Exterior)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| DTWEXBGS | Trade Weighted USD Broad | dollar_index | Daily | coincident |
| DTWEXAFEGS | Trade Weighted USD AFE | dollar_index | Daily | coincident |
| DEXUSEU | EUR/USD | major_pair | Daily | coincident |
| DEXJPUS | USD/JPY | major_pair | Daily | coincident |
| DEXCHUS | USD/CNY | major_pair | Daily | coincident |
| BOPGSTB | Trade Balance | external | Monthly | lagging |
| NETEXP | Net Exports | external | Quarterly | lagging |

### 5.10 COMMODITIES (Energía y Materias Primas)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| DCOILWTICO | WTI Crude Oil | energy | Daily | leading |
| DCOILBRENTEU | Brent Crude | energy | Daily | leading |
| DHHNGSP | Natural Gas Henry Hub | energy | Daily | leading |
| PPIENG | PPI Energy | energy_index | Monthly | leading |
| -- | Copper (via Yahoo) | metals | Daily | leading |
| -- | Bloomberg Commodity Index | broad | Daily | leading |
| -- | Baltic Dry Index | shipping | Daily | leading |

### 5.11 PRECIOUS (Metales Preciosos)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| GOLDAMGBD228NLBM | Gold London PM | gold | Daily | coincident |
| -- | Silver (via Yahoo) | silver | Daily | coincident |
| -- | Gold/Silver Ratio | ratio | Daily | context |
| -- | Gold vs Real Rates | relationship | Daily | context |

### 5.12 SENTIMENT (Confianza e Incertidumbre)

| Ticker FRED | Nombre | Subcategoría | Frecuencia | Rol |
|-------------|--------|--------------|------------|-----|
| UMCSENT | Michigan Consumer Sent | consumer | Monthly | leading |
| CSCICP03USM665S | Consumer Confidence | consumer | Monthly | leading |
| BSCICP03USM665S | Business Confidence | business | Monthly | leading |
| USEPUINDXD | Economic Policy Uncertainty | uncertainty | Daily | leading |
| -- | Geopolitical Risk Index | geopolitical | Monthly | context |
| AAII | AAII Bull/Bear | investor | Weekly | contrarian |

---

## 6. Los 7 Regímenes Naturales de Mercado

### 6.1 Definición de Regímenes

| Régimen | Descripción | Firma de Scores |
|---------|-------------|-----------------|
| **GOLDILOCKS** | Expansión estable, baja inflación, condiciones óptimas | Growth↑, Inflation→, Liquidity→, Stress↓, Vol↓ |
| **REFLATION** | Crecimiento fuerte, inflación subiendo, commodities fuertes | Growth↑, Inflation↑, Liquidity→, Commodities↑ |
| **TIGHTENING** | Inflación alta, Fed hawkish, liquidez endurece | Inflation↑↑, Liquidity↓, Stress↑, Tech↓ |
| **SLOWDOWN** | Crecimiento cae, inflación se enfría, mercado selectivo | Growth↓, Inflation↓, Vol→/↑, Selectivo |
| **STAGFLATION** | Crecimiento débil + inflación alta, shock de oferta | Growth↓, Inflation↑, Stress↑, Commodities↑ |
| **CREDIT_CRISIS** | Estrés crédito alto, deleveraging, liquidez mala | Stress↑↑, Liquidity↓↓, Spreads↑↑, Vol↑↑ |
| **RECOVERY** | Rebote desde crisis, growth recuperando, vol aún alta | Growth↑, Stress↓, Vol↑→, Early Cyclicals↑ |

### 6.2 Matriz de Clasificación

```python
REGIME_SIGNATURES = {
    'goldilocks': {
        'growth_score': (0.3, 1.0),      # Alto
        'inflation_score': (-0.3, 0.3),   # Contenida
        'stress_score': (0, 0.3),         # Bajo
        'liquidity_score': (-0.2, 0.5),   # Cómoda
        'description': 'Expansión estable, inflación contenida'
    },
    'reflation': {
        'growth_score': (0.3, 1.0),       # Alto
        'inflation_score': (0.2, 0.7),    # Subiendo
        'stress_score': (0, 0.4),         # Bajo-medio
        'commodity_score': (0.3, 1.0),    # Fuerte
        'description': 'Crecimiento fuerte, inflación subiendo'
    },
    'tightening': {
        'inflation_score': (0.5, 1.0),    # Alta/persistente
        'liquidity_score': (-1.0, -0.3),  # Endureciendo
        'stress_score': (0.3, 0.7),       # Subiendo
        'rate_change': (0.2, 1.0),        # Fed subiendo
        'description': 'Inflación alta, Fed hawkish'
    },
    'slowdown': {
        'growth_score': (-0.5, 0.1),      # Cayendo
        'inflation_score': (-0.5, 0.2),   # Enfriándose
        'stress_score': (0.2, 0.5),       # Medio
        'description': 'Crecimiento cae, inflación se enfría'
    },
    'stagflation': {
        'growth_score': (-0.7, -0.1),     # Débil
        'inflation_score': (0.4, 1.0),    # Alta
        'stress_score': (0.4, 0.8),       # Medio-alto
        'description': 'Crecimiento débil + inflación alta'
    },
    'credit_crisis': {
        'stress_score': (0.7, 1.0),       # Muy alto
        'liquidity_score': (-1.0, -0.5),  # Muy mala
        'credit_spread_zscore': (2.0, 10),# Spreads disparados
        'vol_zscore': (1.5, 10),          # Vol muy alta
        'description': 'Crisis de crédito, deleveraging'
    },
    'recovery': {
        'growth_score': (0.1, 0.6),       # Rebotando
        'growth_momentum': (0.3, 1.0),    # Mejorando
        'stress_score': (0.3, 0.6),       # Aún elevado pero bajando
        'stress_momentum': (-1.0, -0.2),  # Mejorando
        'description': 'Rebote desde crisis'
    },
}
```

### 6.3 Estrategia por Régimen

| Régimen | Sectores Favorecidos | Beta Objetivo | Cash % | Style |
|---------|---------------------|---------------|--------|-------|
| **GOLDILOCKS** | Tech, Consumer Disc, Growth | 1.0-1.3 | 5% | Growth, Momentum |
| **REFLATION** | Energy, Materials, Industrials | 1.1-1.4 | 5% | Value, Cyclicals |
| **TIGHTENING** | Utilities, Healthcare, Staples | 0.6-0.9 | 15% | Defensive, Quality |
| **SLOWDOWN** | Mixed, selectivo | 0.7-1.0 | 20% | Quality, Low Vol |
| **STAGFLATION** | Energy, Commodities, TIPS | 0.5-0.8 | 25% | Real Assets |
| **CREDIT_CRISIS** | Cash, Treasuries, Gold | 0.3-0.5 | 40% | Capital Preservation |
| **RECOVERY** | Financials, Cons Disc, Small Cap | 1.2-1.5 | 5% | Early Cyclicals |

---

## 7. Tablas de Datos

### 7.1 Tabla: macro_indicators (Raw Data)

```sql
CREATE TABLE macro_indicators (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value FLOAT,

    -- Metadata
    source VARCHAR(50),
    vintage_date DATE,
    is_preliminary BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(ticker, date)
);

CREATE INDEX idx_macro_ind_ticker_date ON macro_indicators(ticker, date);
CREATE INDEX idx_macro_ind_date ON macro_indicators(date);
```

### 7.2 Tabla: macro_features (Transformed & Scores)

```sql
CREATE TABLE macro_features (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,

    -- ═══════════════════════════════════════════════════════════════
    -- INFLACIÓN
    -- ═══════════════════════════════════════════════════════════════
    cpi_yoy FLOAT,
    cpi_mom FLOAT,
    cpi_core_yoy FLOAT,
    pce_yoy FLOAT,
    pce_core_yoy FLOAT,
    ppi_yoy FLOAT,
    wages_yoy FLOAT,
    inflation_expect_1y FLOAT,
    breakeven_5y FLOAT,
    breakeven_10y FLOAT,
    inflation_trend VARCHAR(20),          -- rising, falling, stable
    inflation_zscore FLOAT,

    -- ═══════════════════════════════════════════════════════════════
    -- CRECIMIENTO
    -- ═══════════════════════════════════════════════════════════════
    gdp_yoy FLOAT,
    gdp_qoq_saar FLOAT,
    industrial_prod_yoy FLOAT,
    industrial_prod_mom FLOAT,
    retail_sales_yoy FLOAT,
    retail_sales_mom FLOAT,
    new_orders_yoy FLOAT,
    pmi_manufacturing FLOAT,
    pmi_services FLOAT,
    inventories_yoy FLOAT,
    inv_sales_ratio FLOAT,
    growth_trend VARCHAR(20),
    growth_zscore FLOAT,
    growth_momentum FLOAT,                 -- Cambio en z-score

    -- ═══════════════════════════════════════════════════════════════
    -- LABORAL
    -- ═══════════════════════════════════════════════════════════════
    unemployment_rate FLOAT,
    unemployment_change_3m FLOAT,
    payrolls_change FLOAT,
    payrolls_3m_avg FLOAT,
    claims_initial FLOAT,
    claims_initial_4wk_avg FLOAT,
    claims_continued FLOAT,
    job_openings FLOAT,
    quits_rate FLOAT,
    participation_rate FLOAT,
    labor_trend VARCHAR(20),
    labor_zscore FLOAT,

    -- ═══════════════════════════════════════════════════════════════
    -- POLÍTICA MONETARIA
    -- ═══════════════════════════════════════════════════════════════
    fed_funds FLOAT,
    fed_funds_change_3m FLOAT,
    fed_funds_change_12m FLOAT,
    real_rate_10y FLOAT,
    fed_balance_sheet FLOAT,
    fed_balance_yoy FLOAT,
    policy_stance VARCHAR(20),             -- dovish, neutral, hawkish
    policy_momentum VARCHAR(20),           -- easing, stable, tightening

    -- ═══════════════════════════════════════════════════════════════
    -- CURVA DE TIPOS
    -- ═══════════════════════════════════════════════════════════════
    yield_3m FLOAT,
    yield_2y FLOAT,
    yield_5y FLOAT,
    yield_10y FLOAT,
    yield_30y FLOAT,
    curve_2s10s FLOAT,
    curve_3m10y FLOAT,
    curve_2s10s_change_3m FLOAT,
    curve_inverted BOOLEAN,
    curve_inversion_days INTEGER,          -- Días consecutivos invertida
    curve_zscore FLOAT,
    term_premium_10y FLOAT,

    -- ═══════════════════════════════════════════════════════════════
    -- LIQUIDEZ
    -- ═══════════════════════════════════════════════════════════════
    m2_yoy FLOAT,
    m2_mom FLOAT,
    credit_growth_yoy FLOAT,
    lending_standards FLOAT,               -- Survey, + = tightening
    nfci FLOAT,
    nfci_credit FLOAT,
    nfci_leverage FLOAT,
    liquidity_trend VARCHAR(20),
    liquidity_zscore FLOAT,

    -- ═══════════════════════════════════════════════════════════════
    -- CRÉDITO Y SPREADS
    -- ═══════════════════════════════════════════════════════════════
    spread_ig FLOAT,
    spread_hy FLOAT,
    spread_bbb FLOAT,
    spread_ig_change_1m FLOAT,
    spread_hy_change_1m FLOAT,
    spread_ig_zscore FLOAT,
    spread_hy_zscore FLOAT,
    ted_spread FLOAT,
    credit_stress VARCHAR(20),             -- low, elevated, high, extreme

    -- ═══════════════════════════════════════════════════════════════
    -- STRESS Y VOLATILIDAD
    -- ═══════════════════════════════════════════════════════════════
    vix FLOAT,
    vix_zscore FLOAT,
    vix_percentile FLOAT,                  -- Percentil histórico
    vix_term_structure FLOAT,              -- VIX/VIX3M ratio
    vix_change_1w FLOAT,
    move_index FLOAT,                      -- Bond volatility
    st_louis_stress FLOAT,
    stress_trend VARCHAR(20),
    vol_regime VARCHAR(20),                -- low, normal, elevated, extreme

    -- ═══════════════════════════════════════════════════════════════
    -- FX
    -- ═══════════════════════════════════════════════════════════════
    dxy FLOAT,
    dxy_change_1m FLOAT,
    dxy_zscore FLOAT,
    eurusd FLOAT,
    usdjpy FLOAT,
    dollar_trend VARCHAR(20),              -- strong, neutral, weak

    -- ═══════════════════════════════════════════════════════════════
    -- COMMODITIES
    -- ═══════════════════════════════════════════════════════════════
    oil_wti FLOAT,
    oil_brent FLOAT,
    oil_change_1m FLOAT,
    oil_yoy FLOAT,
    nat_gas FLOAT,
    copper FLOAT,
    copper_yoy FLOAT,
    commodity_index FLOAT,
    commodity_yoy FLOAT,
    commodity_trend VARCHAR(20),

    -- ═══════════════════════════════════════════════════════════════
    -- METALES PRECIOSOS
    -- ═══════════════════════════════════════════════════════════════
    gold FLOAT,
    gold_change_1m FLOAT,
    gold_yoy FLOAT,
    silver FLOAT,
    gold_silver_ratio FLOAT,
    gold_real_rate_spread FLOAT,           -- Gold return - real rate

    -- ═══════════════════════════════════════════════════════════════
    -- SENTIMIENTO
    -- ═══════════════════════════════════════════════════════════════
    consumer_sentiment FLOAT,
    consumer_sentiment_zscore FLOAT,
    business_confidence FLOAT,
    epu_index FLOAT,                       -- Economic Policy Uncertainty
    epu_zscore FLOAT,
    geopolitical_risk FLOAT,
    aaii_bull_bear FLOAT,                  -- Bull% - Bear%
    sentiment_trend VARCHAR(20),

    -- ═══════════════════════════════════════════════════════════════
    -- SCORES COMPUESTOS (Calculados)
    -- ═══════════════════════════════════════════════════════════════
    growth_score FLOAT,                    -- -1 a 1
    inflation_score FLOAT,                 -- -1 a 1
    stress_score FLOAT,                    -- 0 a 1
    liquidity_score FLOAT,                 -- -1 a 1
    commodity_score FLOAT,                 -- -1 a 1

    -- Momentum de scores (cambio 1 mes)
    growth_score_mom FLOAT,
    inflation_score_mom FLOAT,
    stress_score_mom FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_macro_features_date ON macro_features(date);
```

### 7.3 Tabla: macro_regime (Classification)

```sql
CREATE TABLE macro_regime (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,

    -- Régimen principal (7 regímenes naturales)
    regime VARCHAR(30) NOT NULL,
    regime_confidence FLOAT,

    -- Probabilidades por régimen
    prob_goldilocks FLOAT,
    prob_reflation FLOAT,
    prob_tightening FLOAT,
    prob_slowdown FLOAT,
    prob_stagflation FLOAT,
    prob_credit_crisis FLOAT,
    prob_recovery FLOAT,

    -- Régimen secundario (segundo más probable)
    regime_secondary VARCHAR(30),
    regime_secondary_prob FLOAT,

    -- Transiciones
    regime_change_signal BOOLEAN DEFAULT FALSE,
    days_in_current_regime INTEGER,
    regime_stability FLOAT,                -- 0-1, qué tan estable es

    -- Sub-clasificaciones
    cycle_phase VARCHAR(20),               -- early, mid, late
    inflation_regime VARCHAR(20),          -- disinflation, stable, rising, high
    vol_regime VARCHAR(20),                -- low, normal, high, extreme

    -- Ratios diana sugeridos (output del modelo)
    target_pe_max FLOAT,
    target_beta_min FLOAT,
    target_beta_max FLOAT,
    target_cash_pct FLOAT,
    target_defensive_pct FLOAT,
    target_growth_pct FLOAT,
    target_value_pct FLOAT,
    target_cyclical_pct FLOAT,

    -- Sectores sugeridos
    sectors_overweight TEXT,               -- JSON array
    sectors_underweight TEXT,              -- JSON array

    -- Metadata
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_macro_regime_date ON macro_regime(date);
CREATE INDEX idx_macro_regime_regime ON macro_regime(regime);
```

---

## 8. Ratios Diana por Régimen

```python
REGIME_TARGETS = {
    'goldilocks': {
        'pe_max': 28,
        'beta_range': (1.0, 1.3),
        'cash_pct': 5,
        'defensive_pct': 15,
        'growth_pct': 40,
        'value_pct': 25,
        'cyclical_pct': 20,
        'sharpe_min': 0.8,
        'max_drawdown': 18,
        'sectors_ow': ['Technology', 'Consumer Discretionary', 'Communication Services'],
        'sectors_uw': ['Utilities', 'Consumer Staples'],
    },
    'reflation': {
        'pe_max': 22,
        'beta_range': (1.1, 1.4),
        'cash_pct': 5,
        'defensive_pct': 10,
        'growth_pct': 25,
        'value_pct': 35,
        'cyclical_pct': 30,
        'sharpe_min': 0.7,
        'max_drawdown': 20,
        'sectors_ow': ['Energy', 'Materials', 'Industrials', 'Financials'],
        'sectors_uw': ['Technology', 'Utilities'],
    },
    'tightening': {
        'pe_max': 18,
        'beta_range': (0.6, 0.9),
        'cash_pct': 15,
        'defensive_pct': 40,
        'growth_pct': 15,
        'value_pct': 30,
        'cyclical_pct': 15,
        'sharpe_min': 1.0,
        'max_drawdown': 12,
        'sectors_ow': ['Healthcare', 'Consumer Staples', 'Utilities'],
        'sectors_uw': ['Technology', 'Consumer Discretionary', 'Real Estate'],
    },
    'slowdown': {
        'pe_max': 20,
        'beta_range': (0.7, 1.0),
        'cash_pct': 20,
        'defensive_pct': 35,
        'growth_pct': 20,
        'value_pct': 25,
        'cyclical_pct': 20,
        'sharpe_min': 1.0,
        'max_drawdown': 12,
        'sectors_ow': ['Healthcare', 'Consumer Staples', 'Quality'],
        'sectors_uw': ['Industrials', 'Materials'],
    },
    'stagflation': {
        'pe_max': 15,
        'beta_range': (0.5, 0.8),
        'cash_pct': 25,
        'defensive_pct': 30,
        'growth_pct': 10,
        'value_pct': 25,
        'cyclical_pct': 35,  # Commodities
        'sharpe_min': 1.2,
        'max_drawdown': 10,
        'sectors_ow': ['Energy', 'Materials', 'Utilities'],
        'sectors_uw': ['Technology', 'Consumer Discretionary', 'Financials'],
    },
    'credit_crisis': {
        'pe_max': 12,
        'beta_range': (0.3, 0.5),
        'cash_pct': 40,
        'defensive_pct': 45,
        'growth_pct': 5,
        'value_pct': 10,
        'cyclical_pct': 5,
        'sharpe_min': 2.0,
        'max_drawdown': 8,
        'sectors_ow': ['Consumer Staples', 'Utilities', 'Healthcare'],
        'sectors_uw': ['Financials', 'Real Estate', 'Consumer Discretionary'],
    },
    'recovery': {
        'pe_max': 25,
        'beta_range': (1.2, 1.5),
        'cash_pct': 5,
        'defensive_pct': 10,
        'growth_pct': 30,
        'value_pct': 25,
        'cyclical_pct': 35,
        'sharpe_min': 0.6,
        'max_drawdown': 22,
        'sectors_ow': ['Financials', 'Consumer Discretionary', 'Industrials', 'Small Cap'],
        'sectors_uw': ['Utilities', 'Consumer Staples'],
    },
}
```

---

## 9. Scripts a Implementar

| Script | Función | Prioridad |
|--------|---------|-----------|
| `src/macro/config.py` | Configuración y API keys | 1 |
| `src/macro/fred_client.py` | Cliente FRED API | 1 |
| `src/macro/yahoo_macro.py` | Descarga VIX, commodities de Yahoo | 1 |
| `src/macro/macro_downloader.py` | Orquestador de descargas | 1 |
| `src/macro/macro_catalog_seeder.py` | Poblar macro_catalog | 2 |
| `src/macro/macro_features_calc.py` | Calcular transformaciones y scores | 2 |
| `src/macro/regime_detector.py` | Clasificar régimen | 3 |
| `src/macro/target_ratios_calc.py` | Calcular ratios diana | 3 |

---

## 10. Próximos Pasos

1. [ ] Crear tablas en PostgreSQL (fmp_data)
2. [ ] Obtener FRED API Key (gratuita)
3. [ ] Implementar fred_client.py
4. [ ] Poblar macro_catalog con indicadores
5. [ ] Descargar histórico (10+ años)
6. [ ] Implementar macro_features_calc.py
7. [ ] Implementar regime_detector.py
8. [ ] Integrar con features_master
9. [ ] Testing con datos reales
