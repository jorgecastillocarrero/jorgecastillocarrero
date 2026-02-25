# Sistema Fair V3 - Score Eventos + Ajuste por Precio
## Trading Semanal por Subsectores del S&P 500

---

## 1. Arquitectura del Sistema

El sistema opera semanalmente sobre 49 subsectores del S&P 500.
Cada semana calcula un score para cada subsector y decide posiciones long/short.

```
Eventos macro activos
        |
        v
[Score Eventos 0-10]  +  [Estado del Precio: DD, RSI]
        |                         |
        v                         v
   Score Fair              Ajuste por precio
   (promedio)              (oversold/overbought)
        |                         |
        +----------+--------------+
                   |
                   v
            Score Final (0-10)
                   |
                   v
        [Asignacion NL x NS]
         (bear_ratio conteo)
                   |
                   v
        [Seleccion + Pesos]
         (score-weighted)
                   |
                   v
           Posiciones semanales
```

---

## 2. Score de Eventos (Fair, 0-10)

### Problema original
- **Score RAW (suma)**: favorecia subsectores con muchos eventos (Gold +13, Water +4)
- **Score normalizado (rango)**: favorecia subsectores con pocos eventos (Water 10.0, Gold 5.2)

### Solucion: promedio de contribuciones

Para cada subsector, calcular el PROMEDIO de las contribuciones de todos los eventos activos:

```
contribucion = intensidad_evento * impacto_subsector

score = 5.0 + (promedio_contribuciones / MAX_CONTRIBUTION) * 5.0
score = clamp(score, 0.0, 10.0)
```

- `MAX_CONTRIBUTION = 4.0` (impacto max +-2 * intensidad max 2.0)
- Subsector sin eventos activos = 5.0 (neutral)
- Score 0.0 = max bearish (todos los eventos negativos al maximo)
- Score 10.0 = max bullish
- Score 5.0 = neutral o eventos que se cancelan

### Ejemplo: Regional Banks en enero 2008
- bajada_tipos_interes: 2.0 * (-2) = -4.0
- crisis_inmobiliaria: 2.0 * (-2) = -4.0
- recesion: 2.0 * (-2) = -4.0
- Promedio = (-4.0 + -4.0 + -4.0) / 3 = -4.0
- Score = 5.0 + (-4.0 / 4.0) * 5.0 = **0.0** (max bearish)

### Ejemplo: Gold Mining en enero 2008
- bajada_tipos: +2.0, crisis_inmobiliaria: +2.0, inflacion: +3.0, petroleo: +2.0, recesion: +4.0
- Promedio = (+2.0 +2.0 +3.0 +2.0 +4.0) / 5 = +2.6
- Score = 5.0 + (2.6 / 4.0) * 5.0 = **8.25** (bullish)

---

## 3. Ajuste por Estado del Precio

### Problema
El score de eventos no considera si un subsector ya esta sobrevendido/sobrecomprado.
Shortear un sector que ya ha caido -40% tiene alto riesgo de short squeeze.

### Solucion: ajustar el score segun DD y RSI

#### Para shorts (score < 5.0) - Oversold adjustment
```
dd_factor  = clamp((|DD_52w| - 15) / 30, 0, 1)
             -> 0 si DD > -15%
             -> 0.5 si DD = -30%
             -> 1.0 si DD <= -45%

rsi_factor = clamp((35 - RSI_14w) / 20, 0, 1)
             -> 0 si RSI > 35
             -> 0.5 si RSI = 25
             -> 1.0 si RSI <= 15

oversold = max(dd_factor, rsi_factor)

ajuste = (5.0 - score) * oversold * 0.5
score_final = score + ajuste
```

El ajuste mueve el score HACIA 5.0 (neutral), reduciendo la conviction del short.
Maximo ajuste = 50% de la distancia a neutral.

#### Para longs (score > 5.0) - Overbought adjustment
```
rsi_factor = clamp((RSI_14w - 70) / 15, 0, 1)
             -> 0 si RSI < 70
             -> 0.67 si RSI = 80
             -> 1.0 si RSI >= 85

ajuste = (score - 5.0) * rsi_factor * 0.5
score_final = score - ajuste
```

### Efecto practico: rotacion automatica de shorts

La rotacion ocurre naturalmente cada semana porque el sistema recalcula todo:
1. Subsector A cae -30%, RSI baja a 25 -> score sube de 0.0 a 1.25 (menos conviction)
2. Subsector B aun no ha caido mucho (DD -10%, RSI 45) -> score se mantiene en 0.5
3. El sistema prefiere shortear B (score 0.5) sobre A (score 1.25)
4. Cuando A rebota y RSI vuelve a 40, su ajuste desaparece y vuelve al pool

### Ejemplo: enero 2008
| Subsector | Score evt | DD | RSI | Ajuste | Score final |
|-----------|-----------|-----|-----|--------|-------------|
| Regional Banks | 0.0 | -30% | 29 | +1.25 | 1.25 |
| Construction | 0.0 | -16% | 44 | +0.07 | 0.07 |
| Constr Materials | 1.2 | -28% | 43 | +0.81 | 2.01 |
| Major Banks | 1.7 | -41% | 22 | +1.44 | 3.14 |

Construction (DD -16%) queda como short fuerte.
Regional Banks (DD -30%) se penaliza, menos peso.
Major Banks (DD -41%) casi sale del pool de shorts.

---

## 4. Sistema de Asignacion (bear_ratio)

### Problema del promedio
El promedio de scores da 4.23 en crisis porque subsectores defensivos (Gold 8.2, Utilities 8.8)
suben el promedio falsamente. Con promedio 4.23 el sistema da 3L+3S en plena crisis.

### Solucion: contar subsectores bearish vs bullish

```
bear_count = subsectores con score_final < 3.5
bull_count = subsectores con score_final > 6.5

bear_ratio = bear_count / (bear_count + bull_count)
```

### Tabla de asignacion

| bear_ratio | Config | Descripcion |
|------------|--------|-------------|
| >= 0.70 | 0L+3S | Crisis extrema, solo shorts |
| >= 0.60 | 1L+3S | Crisis, 1 refugio + shorts |
| >= 0.55 | 2L+3S | Bearish |
| >= 0.45 | 3L+3S | Equilibrado, leve bear |
| >= 0.40 | 3L+2S | Ligeramente bull |
| >= 0.30 | 3L+1S | Bullish |
| < 0.30 | 3L+0S | Bull fuerte, solo longs |

### Ejemplo: enero 2008
- 24 subsectores con score < 3.5 (bearish)
- 11 subsectores con score > 6.5 (bullish)
- bear_ratio = 24/35 = 0.69 -> **1L+3S** (crisis)

---

## 5. Seleccion y Pesos (score-weighted)

### Pool de candidatos
- **Longs pool**: subsectores con score_final > 6.5, ordenados de mayor a menor
- **Shorts pool**: subsectores con score_final < 3.5, ordenados de menor a mayor

### Filtro ATR
- Shorts requieren ATR semanal >= 1.5% (volatilidad minima para que el short sea rentable)
- ATR lagged (semana anterior) para evitar lookahead bias

### Pesos por distancia al neutral
```
long_weight  = score_final - 5.0  (cuanto mas bullish, mas peso)
short_weight = 5.0 - score_final  (cuanto mas bearish, mas peso)

peso_normalizado = weight / suma_total_weights
capital_asignado = CAPITAL * peso_normalizado
```

### Ejemplo: 1L+3S en enero 2008
| Posicion | Score final | Distancia a 5.0 | Peso |
|----------|------------|-----------------|------|
| LONG Utilities | 8.8 | 3.8 | 24% |
| SHORT Construction | 0.07 | 4.93 | 31% |
| SHORT Reg Banks | 1.25 | 3.75 | 24% |
| SHORT Restaurants | 1.6 | 3.4 | 21% |

---

## 6. Fixes tecnicos

### Bug KKR (resampling semanal)
KKR tiene su ultimo dato semanal en jueves cuando el resto cae en viernes.
Esto crea 2 fechas por ISO week (jueves con 1/49 datos + viernes con 49/49).
El ATR lagged apuntaba al jueves (1 dato) causando NaN en todos los shorts.

**Fix**: filtrar fechas con menos de 40 subsectores antes de calcular el lag ATR.

### RSI semanal
RSI de 14 semanas calculado con media movil simple de ganancias/perdidas.
min_periods=7 para tener datos desde el inicio del historico.

### Drawdown 52 semanas
DD = (close_actual / max_high_52w - 1) * 100
Usa el maximo de highs semanales, no closes, para capturar el verdadero pico.

---

## 7. Resultados 2008

### V3 (eventos + precio) vs V2 (solo eventos)

| Mes | V3 | V2 | Mejora | SPY |
|-----|-----|-----|--------|-----|
| Enero | +$20,467 | +$8,851 | +$11,616 | -5.2% |
| Febrero | +$9,578 | +$2,888 | +$6,690 | -4.1% |
| Marzo | +$1,923 | -$8,968 | +$10,891 | -1.1% |
| Abril | +$1,251 | -$22,402 | +$23,654 | +1.2% |
| Mayo | +$21,186 | +$19,622 | +$1,564 | -0.5% |
| Junio | +$40,922 | +$71,643 | -$30,721 | -7.9% |
| Julio | +$8,149 | -$13,661 | +$21,809 | -1.2% |
| Agosto | +$17,026 | -$17,804 | +$34,829 | +2.1% |
| Septiembre | -$14,736 | -$62,279 | +$47,543 | -9.4% |
| Octubre | +$61,143 | +$19,334 | +$41,809 | -16.6% |
| Noviembre | +$57,157 | +$72,612 | -$15,455 | -7.2% |
| Diciembre | +$21,882 | +$4,156 | +$17,726 | +9.9% |
| **TOTAL** | **+$245,948** | +$73,991 | **+$171,956** | -37.7% |

- V3: **+49.2%** en 2008 (SPY -37.7%)
- V2: +14.8%
- Mejora: 3.3x mejor con ajuste por precio
- Capital base: $500,000 por semana

### Meses clave
- **Septiembre (Lehman)**: V3 -$15K vs V2 -$62K (el ajuste saco shorts sobrevendidos)
- **Octubre (crash)**: V3 +$61K vs V2 +$19K (rotacion evito short squeeze en rebotes)
- **Junio**: V3 pierde vs V2 porque el ajuste saco bancos sobrevendidos que siguieron cayendo

---

## 8. Parametros del sistema

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| CAPITAL | $500,000 | Capital asignado por semana |
| MAX_CONTRIBUTION | 4.0 | Max contribucion por evento (impacto 2 * intensidad 2) |
| ATR_MIN | 1.5% | Volatilidad minima para shorts |
| max_pos | 3 | Maximo posiciones por lado (longs o shorts) |
| Long threshold | > 6.5 | Score minimo para entrar en pool de longs |
| Short threshold | < 3.5 | Score maximo para entrar en pool de shorts |
| DD oversold start | -15% | Drawdown donde empieza el ajuste |
| DD oversold full | -45% | Drawdown donde ajuste es maximo |
| RSI oversold start | 35 | RSI donde empieza ajuste bearish |
| RSI oversold full | 15 | RSI donde ajuste bearish es maximo |
| RSI overbought start | 70 | RSI donde empieza ajuste bullish |
| RSI overbought full | 85 | RSI donde ajuste bullish es maximo |
| Max correction | 50% | Maximo ajuste hacia neutral |

---

## 9. Pendiente

- [ ] Verificar en año superalcista (2021) - logica overbought para longs
- [ ] Backtest completo 2003-2025 con Fair V3
- [ ] Comparar con V3 original (616%) y sistema 3-Tier (P1+P2)
- [ ] Definir si el ajuste por precio mejora o empeora en años normales
- [ ] Optimizar parametros (DD thresholds, RSI thresholds, max correction)
