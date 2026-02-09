# Features Requirements - ML/AI Trading System

## Ejemplos de Queries Complejas (Requisitos)

### Ejemplo 1: Backtest con Stop Loss
**Query:** "Acciones con mas posibilidades de subir en los proximos 5 dias, cap > 10B, stop loss 2%"

**Datos necesarios:**
- `open` - Precio entrada (apertura dia T+1)
- `low` - Para verificar stop loss en los 5 dias
- `high` - Para trailing stops
- `close` - Precio salida y senal
- `market_cap` - Filtro capitalizacion (de fundamentales)

**Logica backtest:**
1. Senal en cierre dia T
2. Compra en apertura dia T+1
3. Verificar LOW de T+1 a T+5 no < entry * 0.98 (stop 2%)
4. Venta en cierre T+5 o cuando toca stop

```sql
-- Stop loss tocado si:
min_low_5d < entry_price * 0.98
```

---

### Ejemplo 2: Screener Oversold + Earnings
**Query:** "Top 5 S&P500 oversold con earnings surprise positivo ultimos 4 trimestres"

**Datos necesarios:**
- `rsi_zone` = 'oversold'
- `earnings_surprise_q1, q2, q3, q4` > 0 (de features_earnings)
- `sp500_member` = true (de features_fundamental)

---

### Ejemplo 3: Momentum + Volumen
**Query:** "Acciones rompiendo maximo 52 semanas con volumen 3x normal"

**Datos necesarios:**
- `new_high` = true
- `volume_ratio` > 3
- `trend_direction` = 1

---

### Ejemplo 4: Mean Reversion
**Query:** "RSI(2) < 10 con tendencia alcista largo plazo"

**Datos necesarios:**
- `rsi_2` < 10
- `above_sma_200` = true
- `trend_strength` > 25

---

### Ejemplo 5: Squeeze Breakout
**Query:** "Acciones saliendo de squeeze con momentum positivo"

**Datos necesarios:**
- `squeeze_on` = false (hoy)
- `squeeze_on` = true (ayer) -- necesita LAG
- `macd_hist` > 0

---

## Tablas del Sistema

### 1. features_technical (COMPLETADA - 61 columnas)
```
Base: symbol, date, open, high, low, close, volume
Tendencia: sma_20/50/200, ema_12/26, above_sma_*, trend_strength, trend_direction
Momentum: rsi, rsi_2, macd_hist, stoch, williams, cci, mfi, ultimate_osc
Zonas: rsi_zone, stoch_zone, cci_zone, mfi_zone, bb_zone, trend_zone, volatility_zone, volume_zone
Volatilidad: atr, atr_pct, bb_width, bb_position, volatility_20d, keltner_position
Volumen: volume_ratio, obv_slope, cmf, adl_slope
Senales: psar_signal, macd_signal, aroon_signal, vortex_signal, ichimoku_signal, squeeze_on, elder_signal, donchian_breakout
Price Action: daily_return, gap, crash, spike, drawdown_20d, dist_52w_high/low, new_high/low
```

### 2. features_momentum (PENDIENTE)
```
symbol, date
ret_1d, ret_5d, ret_20d, ret_60d, ret_252d  -- Retornos
ret_1d_fwd, ret_5d_fwd, ret_20d_fwd         -- Retornos FUTUROS (target para ML)
sharpe_20d, sharpe_60d                       -- Sharpe ratio
sortino_20d, sortino_60d                     -- Sortino ratio
max_dd_20d, max_dd_60d                       -- Max drawdown
```

### 3. features_fundamental (PENDIENTE)
```
symbol, date
market_cap, market_cap_category (micro/small/mid/large/mega)
pe_ratio, pe_zone (cheap/fair/expensive)
pb_ratio, ps_ratio
roe, roa, profit_margin, operating_margin
debt_to_equity, current_ratio
dividend_yield
sector, industry
sp500_member, nasdaq100_member, russell2000_member
```

### 4. features_earnings (PENDIENTE)
```
symbol, date
earnings_date_next, days_to_earnings
earnings_surprise_last, earnings_surprise_avg_4q
revenue_surprise_last, revenue_surprise_avg_4q
eps_growth_yoy, revenue_growth_yoy
beat_estimates_streak
```

### 5. features_master (VISTA UNIFICADA)
```sql
CREATE VIEW features_master AS
SELECT
    t.*,
    m.ret_5d_fwd, m.sharpe_20d,
    f.market_cap, f.pe_ratio, f.sector, f.sp500_member,
    e.days_to_earnings, e.earnings_surprise_avg_4q
FROM features_technical t
LEFT JOIN features_momentum m ON t.symbol = m.symbol AND t.date = m.date
LEFT JOIN features_fundamental f ON t.symbol = f.symbol AND t.date = f.date
LEFT JOIN features_earnings e ON t.symbol = e.symbol AND t.date = e.date
```

---

## Columnas Criticas para Backtesting

| Columna | Tabla | Uso |
|---------|-------|-----|
| `open` | technical | Precio entrada (compra en apertura) |
| `high` | technical | Trailing stop, targets |
| `low` | technical | Stop loss verification |
| `close` | technical | Senales, precio salida |
| `ret_Xd_fwd` | momentum | Target para ML (retorno futuro) |
| `market_cap` | fundamental | Filtro liquidez |
| `days_to_earnings` | earnings | Evitar earnings durante trade |

---

## Notas Importantes

1. **Point-in-time**: Nunca usar datos futuros para generar senales
2. **Retornos futuros**: Solo en features_momentum, claramente marcados como `_fwd`
3. **Stop loss**: Requiere verificar LOW de multiples dias
4. **Entry price**: Siempre OPEN del dia siguiente a la senal
5. **Slippage**: Considerar en backtests reales (no en features)

---

## Queries SQL de Referencia

### Backtest basico con stop loss
```sql
WITH signals AS (
    SELECT symbol, date as signal_date, close
    FROM features_technical
    WHERE rsi_zone = 'oversold' AND trend_direction = 1
),
trades AS (
    SELECT
        s.symbol, s.signal_date,
        t1.open as entry_price,
        LEAST(t1.low, t2.low, t3.low, t4.low, t5.low) as min_low_5d,
        t5.close as exit_price
    FROM signals s
    JOIN features_technical t1 ON s.symbol = t1.symbol AND t1.date = s.signal_date + INTERVAL '1 day'
    JOIN features_technical t2 ON s.symbol = t2.symbol AND t2.date = s.signal_date + INTERVAL '2 days'
    -- ... etc
)
SELECT *,
    CASE WHEN min_low_5d < entry_price * 0.98 THEN -0.02
         ELSE (exit_price - entry_price) / entry_price
    END as return_pct
FROM trades;
```

### Filtro por market cap
```sql
SELECT t.*, f.market_cap
FROM features_technical t
JOIN features_fundamental f ON t.symbol = f.symbol AND t.date = f.date
WHERE f.market_cap > 10000000000  -- > 10B
AND t.rsi_zone = 'oversold'
```
