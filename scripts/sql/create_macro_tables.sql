-- ============================================================================
-- MACRO TABLES - PatrimonioSmart
-- Sistema de datos macroeconómicos para detección de régimen
-- ============================================================================

-- ============================================================================
-- 1. CATÁLOGO DE INDICADORES
-- ============================================================================
CREATE TABLE IF NOT EXISTS macro_catalog (
    indicator_id SERIAL PRIMARY KEY,

    -- Identificación
    ticker VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,

    -- Clasificación por categoría (12 categorías)
    category VARCHAR(30) NOT NULL,            -- inflation, growth, labor, policy, rates, liquidity, credit, stress, fx, commodities, precious, sentiment
    subcategory VARCHAR(50),                  -- headline, core, etc.

    -- Clasificación por tipo
    cycle_role VARCHAR(20),                   -- leading, coincident, lagging
    stat_nature VARCHAR(20),                  -- level, rate, stock, flow, diffusion
    data_behavior VARCHAR(20),                -- revisable, stable, volatile, real_time
    model_role VARCHAR(20),                   -- core_input, context_input, derived

    -- Fuente y metadata
    source VARCHAR(50) NOT NULL,              -- FRED, Yahoo, FMP, calculated
    source_ticker VARCHAR(50),
    frequency VARCHAR(20) NOT NULL,           -- daily, weekly, monthly, quarterly
    units VARCHAR(50),
    seasonal_adj BOOLEAN DEFAULT TRUE,
    publication_lag_days INTEGER,

    -- Transformación por defecto
    default_transform VARCHAR(20),            -- level, yoy, mom, diff, zscore

    -- Peso en scores (0-1)
    growth_weight FLOAT DEFAULT 0,
    inflation_weight FLOAT DEFAULT 0,
    stress_weight FLOAT DEFAULT 0,
    liquidity_weight FLOAT DEFAULT 0,

    -- Estado
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_macro_cat_category ON macro_catalog(category);
CREATE INDEX IF NOT EXISTS idx_macro_cat_cycle_role ON macro_catalog(cycle_role);
CREATE INDEX IF NOT EXISTS idx_macro_cat_model_role ON macro_catalog(model_role);
CREATE INDEX IF NOT EXISTS idx_macro_cat_source ON macro_catalog(source);

-- ============================================================================
-- 2. INDICADORES RAW (Datos descargados)
-- ============================================================================
CREATE TABLE IF NOT EXISTS macro_indicators (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    value FLOAT,

    -- Metadata
    source VARCHAR(50),
    vintage_date DATE,                        -- Para datos revisados
    is_preliminary BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_macro_ind_ticker_date ON macro_indicators(ticker, date);
CREATE INDEX IF NOT EXISTS idx_macro_ind_date ON macro_indicators(date);
CREATE INDEX IF NOT EXISTS idx_macro_ind_ticker ON macro_indicators(ticker);

-- ============================================================================
-- 3. FEATURES MACRO (Transformaciones y Scores)
-- ============================================================================
CREATE TABLE IF NOT EXISTS macro_features (
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
    inflation_trend VARCHAR(20),              -- rising, falling, stable
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
    growth_momentum FLOAT,

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
    policy_stance VARCHAR(20),                -- dovish, neutral, hawkish
    policy_momentum VARCHAR(20),              -- easing, stable, tightening

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
    curve_inversion_days INTEGER,
    curve_zscore FLOAT,
    term_premium_10y FLOAT,

    -- ═══════════════════════════════════════════════════════════════
    -- LIQUIDEZ
    -- ═══════════════════════════════════════════════════════════════
    m2_yoy FLOAT,
    m2_mom FLOAT,
    credit_growth_yoy FLOAT,
    lending_standards FLOAT,
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
    credit_stress VARCHAR(20),                -- low, elevated, high, extreme

    -- ═══════════════════════════════════════════════════════════════
    -- STRESS Y VOLATILIDAD
    -- ═══════════════════════════════════════════════════════════════
    vix FLOAT,
    vix_zscore FLOAT,
    vix_percentile FLOAT,
    vix_term_structure FLOAT,
    vix_change_1w FLOAT,
    move_index FLOAT,
    st_louis_stress FLOAT,
    stress_trend VARCHAR(20),
    vol_regime VARCHAR(20),                   -- low, normal, elevated, extreme

    -- ═══════════════════════════════════════════════════════════════
    -- FX
    -- ═══════════════════════════════════════════════════════════════
    dxy FLOAT,
    dxy_change_1m FLOAT,
    dxy_zscore FLOAT,
    eurusd FLOAT,
    usdjpy FLOAT,
    dollar_trend VARCHAR(20),                 -- strong, neutral, weak

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
    gold_real_rate_spread FLOAT,

    -- ═══════════════════════════════════════════════════════════════
    -- SENTIMIENTO
    -- ═══════════════════════════════════════════════════════════════
    consumer_sentiment FLOAT,
    consumer_sentiment_zscore FLOAT,
    business_confidence FLOAT,
    epu_index FLOAT,
    epu_zscore FLOAT,
    geopolitical_risk FLOAT,
    aaii_bull_bear FLOAT,
    sentiment_trend VARCHAR(20),

    -- ═══════════════════════════════════════════════════════════════
    -- SCORES COMPUESTOS
    -- ═══════════════════════════════════════════════════════════════
    growth_score FLOAT,                       -- -1 a 1
    inflation_score FLOAT,                    -- -1 a 1
    stress_score FLOAT,                       -- 0 a 1
    liquidity_score FLOAT,                    -- -1 a 1
    commodity_score FLOAT,                    -- -1 a 1

    -- Momentum de scores
    growth_score_mom FLOAT,
    inflation_score_mom FLOAT,
    stress_score_mom FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_macro_features_date ON macro_features(date);

-- ============================================================================
-- 4. RÉGIMEN DE MERCADO
-- ============================================================================
CREATE TABLE IF NOT EXISTS macro_regime (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,

    -- Régimen principal (7 regímenes naturales)
    regime VARCHAR(30) NOT NULL,              -- goldilocks, reflation, tightening, slowdown, stagflation, credit_crisis, recovery
    regime_confidence FLOAT,

    -- Probabilidades por régimen
    prob_goldilocks FLOAT,
    prob_reflation FLOAT,
    prob_tightening FLOAT,
    prob_slowdown FLOAT,
    prob_stagflation FLOAT,
    prob_credit_crisis FLOAT,
    prob_recovery FLOAT,

    -- Régimen secundario
    regime_secondary VARCHAR(30),
    regime_secondary_prob FLOAT,

    -- Transiciones
    regime_change_signal BOOLEAN DEFAULT FALSE,
    days_in_current_regime INTEGER,
    regime_stability FLOAT,

    -- Sub-clasificaciones
    cycle_phase VARCHAR(20),                  -- early, mid, late
    inflation_regime VARCHAR(20),             -- disinflation, stable, rising, high
    vol_regime VARCHAR(20),                   -- low, normal, high, extreme

    -- Ratios diana sugeridos
    target_pe_max FLOAT,
    target_beta_min FLOAT,
    target_beta_max FLOAT,
    target_cash_pct FLOAT,
    target_defensive_pct FLOAT,
    target_growth_pct FLOAT,
    target_value_pct FLOAT,
    target_cyclical_pct FLOAT,

    -- Sectores sugeridos (JSON arrays)
    sectors_overweight TEXT,
    sectors_underweight TEXT,

    -- Metadata
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_macro_regime_date ON macro_regime(date);
CREATE INDEX IF NOT EXISTS idx_macro_regime_regime ON macro_regime(regime);

-- ============================================================================
-- 5. HISTÓRICO DE CAMBIOS DE RÉGIMEN
-- ============================================================================
CREATE TABLE IF NOT EXISTS macro_regime_transitions (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,

    regime_from VARCHAR(30) NOT NULL,
    regime_to VARCHAR(30) NOT NULL,

    confidence FLOAT,
    days_in_previous_regime INTEGER,

    -- Scores al momento de la transición
    growth_score FLOAT,
    inflation_score FLOAT,
    stress_score FLOAT,
    liquidity_score FLOAT,

    -- Triggers principales
    primary_trigger TEXT,
    secondary_triggers TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_macro_trans_date ON macro_regime_transitions(date);
CREATE INDEX IF NOT EXISTS idx_macro_trans_regime ON macro_regime_transitions(regime_to);

-- ============================================================================
-- 6. LOG DE DESCARGAS
-- ============================================================================
CREATE TABLE IF NOT EXISTS macro_download_log (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,

    download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    records_downloaded INTEGER,
    date_from DATE,
    date_to DATE,

    status VARCHAR(20),                       -- success, error, partial
    error_message TEXT,

    execution_time_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_macro_dl_ticker ON macro_download_log(ticker);
CREATE INDEX IF NOT EXISTS idx_macro_dl_date ON macro_download_log(download_date);

-- ============================================================================
-- CONFIRMACIÓN
-- ============================================================================
SELECT 'Tablas macro creadas correctamente' as status;
SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'macro%' ORDER BY table_name;
