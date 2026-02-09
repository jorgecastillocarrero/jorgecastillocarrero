# Infraestructura ML/AI Trading System

## 1. Arquitectura de Datos

### 1.1 Bases de Datos

| Base | Host | Puerto | Uso |
|------|------|--------|-----|
| FMP (fmp_data) | localhost | 5433 | Features, precios historicos, fundamentales |
| Railway (caring) | railway | 5432 | Dashboard produccion, cartera |

### 1.2 Tablas de Features

```
fmp_data/
├── fmp_price_history      -- OHLCV historico (source)
├── fmp_key_metrics        -- Metricas fundamentales (source)
├── fmp_earnings           -- Earnings reports (source)
├── fmp_ratios             -- Ratios financieros (source)
│
├── features_technical     -- [COMPLETADA] 61 columnas
├── features_momentum      -- [PENDIENTE] Retornos y riesgo
├── features_fundamental   -- [PENDIENTE] Market cap, P/E, sector
├── features_earnings      -- [PENDIENTE] Earnings surprise
└── features_master        -- [PENDIENTE] Vista unificada
```

---

## 2. Esquema de Tablas

### 2.1 features_technical (61 columnas) - COMPLETADA
```sql
-- Base
symbol VARCHAR(20), date DATE, open, high, low, close, volume

-- Tendencia (10)
sma_20, sma_50, sma_200, ema_12, ema_26
above_sma_20, above_sma_50, above_sma_200 (BOOLEAN)
trend_strength (ADX), trend_direction (+1/-1/0)

-- Momentum (8)
rsi, rsi_2, macd_hist, stoch, williams, cci, mfi, ultimate_osc

-- Zonas (8)
rsi_zone, stoch_zone, cci_zone, mfi_zone, bb_zone
trend_zone, volatility_zone, volume_zone

-- Volatilidad (6)
atr, atr_pct, bb_width, bb_position, volatility_20d, keltner_position

-- Volumen (4)
volume_ratio, obv_slope, cmf, adl_slope

-- Senales (8)
psar_signal, macd_signal, aroon_signal, vortex_signal
ichimoku_signal, squeeze_on, elder_signal, donchian_breakout

-- Price Action (9)
daily_return, gap, crash, spike, drawdown_20d
dist_52w_high, dist_52w_low, new_high, new_low
```

### 2.2 features_momentum (25 cols) - COMPLETADA
```sql
CREATE TABLE features_momentum (
    symbol VARCHAR(20),
    date DATE,

    -- Retornos historicos
    ret_1d FLOAT,           -- Retorno 1 dia
    ret_5d FLOAT,           -- Retorno 5 dias
    ret_20d FLOAT,          -- Retorno 20 dias (1 mes)
    ret_60d FLOAT,          -- Retorno 60 dias (3 meses)
    ret_252d FLOAT,         -- Retorno 252 dias (1 ano)

    -- Retornos FUTUROS (TARGET para ML)
    ret_1d_fwd FLOAT,       -- Retorno futuro 1 dia
    ret_5d_fwd FLOAT,       -- Retorno futuro 5 dias
    ret_20d_fwd FLOAT,      -- Retorno futuro 20 dias

    -- Volatilidad
    vol_20d FLOAT,          -- Volatilidad anualizada 20d
    vol_60d FLOAT,          -- Volatilidad anualizada 60d

    -- Riesgo
    sharpe_20d FLOAT,
    sharpe_60d FLOAT,
    sortino_20d FLOAT,
    sortino_60d FLOAT,
    max_dd_20d FLOAT,
    max_dd_60d FLOAT,
    momentum_score FLOAT,   -- Score compuesto

    -- Betas (rolling vs SPY)
    beta_20d FLOAT,         -- Beta 20 dias (swing trading)
    beta_60d FLOAT,         -- Beta 60 dias (position trading)
    beta_120d FLOAT,        -- Beta 120 dias (6 meses)
    beta_252d FLOAT,        -- Beta 252 dias (1 ano)
    beta_zone VARCHAR(20),  -- very_low/low/market/high/very_high

    PRIMARY KEY (symbol, date)
);
```

### 2.3 features_fundamental (34 cols) - COMPLETADA
```sql
CREATE TABLE features_fundamental (
    symbol VARCHAR(20),
    date DATE,

    -- Clasificacion (estaticos)
    sector VARCHAR(50),
    industry VARCHAR(100),
    country VARCHAR(50),
    exchange VARCHAR(20),
    employees INTEGER,

    -- Tamano (diario)
    market_cap BIGINT,
    market_cap_cat VARCHAR(20),  -- micro/small/mid/large/mega

    -- Valoracion (diaria)
    pe_ratio FLOAT,
    pe_fwd FLOAT,
    pe_zone VARCHAR(20),         -- cheap/fair/expensive
    pb_ratio FLOAT,
    ps_ratio FLOAT,

    -- Revenue (TTM y crecimiento)
    revenue_ttm BIGINT,
    revenue_fwd BIGINT,
    revenue_growth_3y FLOAT,     -- CAGR 3 anos
    revenue_growth_5y FLOAT,     -- CAGR 5 anos

    -- EPS (TTM y crecimiento)
    eps_ttm FLOAT,
    eps_fwd FLOAT,
    eps_growth_3y FLOAT,
    eps_growth_5y FLOAT,

    -- Margenes
    gross_margin FLOAT,
    operating_margin FLOAT,
    profit_margin FLOAT,

    -- Rentabilidad
    roe FLOAT,
    roa FLOAT,

    -- Salud financiera
    total_debt BIGINT,
    total_equity BIGINT,
    debt_to_equity FLOAT,
    current_ratio FLOAT,

    -- Indices
    sp500_member BOOLEAN,
    nasdaq100_member BOOLEAN,

    PRIMARY KEY (symbol, date)
);
```

### 2.4 features_earnings (10 cols) - COMPLETADA
```sql
CREATE TABLE features_earnings (
    symbol VARCHAR(20),
    date DATE,

    -- Proximos earnings
    earnings_date_next DATE,
    days_to_earnings INTEGER,

    -- Sorpresas
    earnings_surprise_last FLOAT,      -- Ultimo trimestre
    earnings_surprise_avg_4q FLOAT,    -- Media 4 trimestres
    revenue_surprise_last FLOAT,
    revenue_surprise_avg_4q FLOAT,

    -- Historial
    beat_streak INTEGER,               -- Trimestres seguidos batiendo

    PRIMARY KEY (symbol, date)
);
```

---

## 3. Ejemplos de Queries (Casos de Uso)

### 3.1 Backtest con Stop Loss
**Caso:** "Acciones con mas probabilidad de subir 5 dias, market cap > 10B, stop loss 2%"

**Columnas necesarias:**
- `open` - Precio entrada (apertura dia T+1)
- `low` - Verificar stop loss durante 5 dias
- `close` - Senal y precio salida
- `market_cap` - Filtro capitalizacion

**Query:**
```sql
WITH signals AS (
    SELECT t.symbol, t.date as signal_date, t.close as signal_close
    FROM features_technical t
    JOIN features_fundamental f ON t.symbol = f.symbol AND t.date = f.date
    WHERE t.rsi_zone = 'oversold'
    AND t.trend_direction = 1
    AND f.market_cap > 10000000000
),
trades AS (
    SELECT
        s.symbol,
        s.signal_date,
        t1.open as entry_price,
        t1.date as entry_date,
        LEAST(t1.low, t2.low, t3.low, t4.low, t5.low) as min_low_5d,
        t5.close as exit_price,
        t5.date as exit_date
    FROM signals s
    JOIN features_technical t1 ON s.symbol = t1.symbol
        AND t1.date = (SELECT MIN(date) FROM features_technical WHERE symbol = s.symbol AND date > s.signal_date)
    JOIN features_technical t2 ON s.symbol = t2.symbol AND t2.date = t1.date + 1
    JOIN features_technical t3 ON s.symbol = t3.symbol AND t3.date = t1.date + 2
    JOIN features_technical t4 ON s.symbol = t4.symbol AND t4.date = t1.date + 3
    JOIN features_technical t5 ON s.symbol = t5.symbol AND t5.date = t1.date + 4
)
SELECT
    symbol,
    signal_date,
    entry_date,
    entry_price,
    exit_price,
    min_low_5d,
    CASE WHEN min_low_5d < entry_price * 0.98 THEN 'STOP_HIT' ELSE 'HELD' END as status,
    CASE
        WHEN min_low_5d < entry_price * 0.98 THEN -0.02
        ELSE (exit_price - entry_price) / entry_price
    END as return_pct
FROM trades;
```

### 3.2 Screener: Oversold + Earnings Positivos
**Caso:** "Top 5 S&P500 oversold con earnings surprise positivo ultimos 4 trimestres"

```sql
SELECT
    t.symbol, t.date, t.rsi, t.close,
    f.market_cap, f.sector,
    e.earnings_surprise_avg_4q
FROM features_technical t
JOIN features_fundamental f ON t.symbol = f.symbol AND t.date = f.date
JOIN features_earnings e ON t.symbol = e.symbol AND t.date = e.date
WHERE t.rsi_zone = 'oversold'
AND f.sp500_member = true
AND e.earnings_surprise_avg_4q > 0
ORDER BY t.rsi ASC
LIMIT 5;
```

### 3.3 Squeeze Breakout
**Caso:** "Acciones saliendo de squeeze con momentum positivo"

```sql
SELECT
    t1.symbol, t1.date, t1.close,
    t1.macd_hist, t1.volume_ratio
FROM features_technical t1
JOIN features_technical t0 ON t1.symbol = t0.symbol
    AND t0.date = t1.date - INTERVAL '1 day'
WHERE t1.squeeze_on = false      -- Hoy: fuera de squeeze
AND t0.squeeze_on = true         -- Ayer: en squeeze
AND t1.macd_hist > 0             -- Momentum positivo
AND t1.volume_ratio > 1.5        -- Volumen elevado
ORDER BY t1.volume_ratio DESC;
```

### 3.4 Mean Reversion (Larry Connors RSI2)
**Caso:** "RSI(2) extremo con tendencia alcista largo plazo"

```sql
SELECT symbol, date, close, rsi_2, rsi
FROM features_technical
WHERE rsi_2 < 10                 -- RSI(2) extremadamente oversold
AND above_sma_200 = true         -- Por encima de media 200
AND trend_strength > 20          -- Tendencia presente
ORDER BY rsi_2 ASC;
```

### 3.5 New Highs con Volumen
**Caso:** "Nuevos maximos 52 semanas con volumen 3x"

```sql
SELECT symbol, date, close, dist_52w_high, volume_ratio
FROM features_technical
WHERE new_high = true
AND volume_ratio > 3
AND trend_direction = 1
ORDER BY volume_ratio DESC;
```

### 3.6 Evitar Earnings en Trade
**Caso:** "Filtrar trades que caen en periodo de earnings"

```sql
SELECT t.*, e.days_to_earnings
FROM features_technical t
JOIN features_earnings e ON t.symbol = e.symbol AND t.date = e.date
WHERE t.rsi_zone = 'oversold'
AND (e.days_to_earnings IS NULL OR e.days_to_earnings > 5)  -- No earnings en proximos 5 dias
```

---

## 4. Reglas de Backtesting

### 4.1 Point-in-Time (Evitar Look-Ahead Bias)
- **Senal:** Usar datos disponibles al CIERRE del dia T
- **Entrada:** Precio OPEN del dia T+1
- **Salida:** Precio CLOSE del dia T+N o cuando se active stop

### 4.2 Stop Loss Verification
```
entry_price = open[T+1]
stop_price = entry_price * (1 - stop_pct)

Para cada dia i desde T+1 hasta T+N:
    if low[i] < stop_price:
        exit_price = stop_price
        break
```

### 4.3 Columnas Criticas

| Columna | Uso | Tabla |
|---------|-----|-------|
| `open` | Precio entrada | features_technical |
| `high` | Trailing stop, targets | features_technical |
| `low` | Stop loss check | features_technical |
| `close` | Senal, salida | features_technical |
| `ret_Xd_fwd` | Target ML | features_momentum |
| `market_cap` | Filtro liquidez | features_fundamental |
| `days_to_earnings` | Evitar eventos | features_earnings |

---

## 5. Scripts

| Script | Funcion | Estado |
|--------|---------|--------|
| `src/features_calculator.py` | Calcular features_technical | COMPLETADO |
| `src/momentum_calculator.py` | Calcular features_momentum | COMPLETADO |
| `src/fundamental_calculator.py` | Calcular features_fundamental | COMPLETADO |
| `src/earnings_calculator.py` | Calcular features_earnings | COMPLETADO |

### Comandos
```bash
# Procesar un simbolo
py -3 -m src.features_calculator --symbol AAPL
py -3 -m src.momentum_calculator --symbol AAPL
py -3 -m src.fundamental_calculator --symbol AAPL
py -3 -m src.earnings_calculator --symbol AAPL

# Procesar todos
py -3 -m src.features_calculator --all
py -3 -m src.momentum_calculator --all
py -3 -m src.fundamental_calculator --all
py -3 -m src.earnings_calculator --all

# Procesar con limite
py -3 -m src.features_calculator --all --limit 100
```

---

## 6. Fases del Proyecto

| Fase | Descripcion | Estado |
|------|-------------|--------|
| 1.1 | Tabla features_technical (61 cols) | COMPLETADA |
| 1.2 | Script features_calculator | COMPLETADO |
| 2.1 | Tabla features_momentum (25 cols + betas) | COMPLETADA |
| 2.2 | Script momentum_calculator | COMPLETADO |
| 2.3 | Tabla features_fundamental (34 cols) | COMPLETADA |
| 2.4 | Script fundamental_calculator | COMPLETADO |
| 2.5 | Tabla features_earnings (10 cols) | COMPLETADA |
| 2.6 | Script earnings_calculator | COMPLETADO |
| 2.7 | Vista features_master (120 cols) | COMPLETADA |
| 3.x | Modelos ML | PENDIENTE |
| 4.x | Backtesting engine | PENDIENTE |
| 5.x | RAG + Chat | PENDIENTE |

---

## 7. Notas Tecnicas

### 7.1 Indices Importantes
```sql
-- features_technical
CREATE INDEX idx_ft_date ON features_technical(date);
CREATE INDEX idx_ft_rsi_zone ON features_technical(rsi_zone);
CREATE INDEX idx_ft_trend_zone ON features_technical(trend_zone);
CREATE INDEX idx_ft_squeeze ON features_technical(squeeze_on);

-- features_fundamental (futuro)
CREATE INDEX idx_ff_market_cap ON features_fundamental(market_cap);
CREATE INDEX idx_ff_sector ON features_fundamental(sector);
CREATE INDEX idx_ff_sp500 ON features_fundamental(sp500_member);
```

### 7.2 Tamanos Estimados
- features_technical: ~120M filas (5000 symbols x 20 years x 252 days)
- features_momentum: ~120M filas
- features_fundamental: ~120M filas (aunque muchos campos static)
- features_earnings: ~20M filas (4 earnings/year)
