# CLAUDE.md - PatrimonioSmart Financial Project

## Que es este proyecto

PatrimonioSmart es una plataforma de gestion patrimonial que:
- Descarga y almacena datos financieros diarios (precios, fundamentales, noticias)
- Calcula posiciones, valoraciones y metricas de cartera
- Muestra un dashboard interactivo (Streamlit) en patrimoniosmart.club
- Incluye un asistente IA con RAG (LangChain + ChromaDB)
- Scheduler automatico: precios 22:30 (lun-vie), posiciones 02:00 (mar-sab)

---

## Stack tecnologico

| Componente | Tecnologia |
|------------|-----------|
| Backend/Data | Python 3.12, Pandas, SQLAlchemy, yfinance |
| Base de datos | PostgreSQL (Railway, servicio "caring") |
| Dashboard | Streamlit (web/app.py) |
| IA/RAG | Anthropic Claude, LangChain, ChromaDB, HuggingFace embeddings |
| Scheduler | APScheduler (precios 22:30 lun-vie, posiciones 02:00 mar-sab) |
| Deploy web | Railway (servicio "enthusiastic" en patrimoniosmart.club) |
| Deploy scheduler | Railway (servicio "awake-light", SERVICE_MODE=scheduler) |
| Repo | GitHub: jorgecastillocarrero/jorgecastillocarrero |

---

## Estructura principal

```
financial-data-project/
  src/
    config.py          - Configuracion (DATABASE_URL, API keys)
    database.py         - ORM SQLAlchemy, modelos, conexion PostgreSQL
    yahoo_downloader.py - Descarga de precios desde Yahoo Finance
    portfolio_data.py   - PROTEGIDO - Logica de calculo de cartera
    posicion_calculator.py - Calculo de posiciones diarias
    technical.py        - Metricas tecnicas (RSI, MACD, etc.)
    scheduler.py        - Scheduler automatico (APScheduler)
    news_manager.py     - Gestion de noticias (GDELT, NewsAPI)
    ai_assistant.py     - Asistente IA con RAG
    portfolio_rag.py    - RAG: LangChain + ChromaDB
  web/
    app.py              - PROTEGIDO - Dashboard Streamlit (3600+ lineas)
  scripts/
    continue_migration.py     - Migracion incremental SQLite -> PostgreSQL
    migrate_exchange_rates.py - Migracion tipos de cambio
    fmp_key_metrics.py        - Descarga Key Metrics desde FMP API
    fmp_ratios.py             - Descarga Ratios financieros desde FMP API
    fmp_etf_holdings.py       - Descarga ETF Holdings desde FMP API
    fmp_earnings_transcripts.py - Descarga Earnings Call Transcripts desde FMP API
  data/                 - Datos locales (excluidos de deploy)
  Dockerfile            - Build para Railway (soporta web y scheduler)
  railway.toml          - Config Railway
  requirements.txt      - Dependencias Python
```

---

## Credenciales y conexiones

- PostgreSQL (Railway): `DATABASE_URL` en .env
- PostgreSQL FMP (Docker local): `postgresql://fmp:fmp123@localhost:5433/fmp_data`
- FMP API Key: `PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf` (suscripcion Plus)
- Anthropic API: `ANTHROPIC_API_KEY` en .env
- Dashboard login: password en `DASHBOARD_PASSWORD` en .env
- Dominio: patrimoniosmart.club (servicio "enthusiastic" en Railway)

---

## PROTEGIDO - NO MODIFICAR SIN AUTORIZACION

1. **src/portfolio_data.py** - Funciones de calculo y obtencion de datos
2. **web/app.py - Pagina Posicion** - Codigo de la pagina Posicion
3. **web/app.py - Pagina Composicion** - Codigo de la pagina Composicion
4. **web/app.py - Pagina Acciones** - Codigo de la pagina Acciones
5. **web/app.py - Pagina Futuros y ETF** - Codigo de la pagina Futuros y ETF

Para desbloquear: "Autorizo modificar [X]" o "Desbloquea [X]"

## PERMITIDO - TODO LO DEMAS

- Insertar/actualizar datos en tablas
- Registrar compras, ventas, cash
- Actualizar precios y holdings
- Anadir nuevos simbolos a ASSET_TYPE_MAP
- Modificar otras paginas del dashboard
- Crear nuevas funcionalidades
- Configuracion de deploy y scheduler

---

## Estado actual (Febrero 2026)

### Hecho
- Migracion completa SQLite -> PostgreSQL (23.8M registros, 5813 symbols)
- Dashboard funcionando en patrimoniosmart.club (servicio "enthusiastic")
- RAG implementado: LangChain + ChromaDB con embeddings multilingue
- Asistente IA integrado con RAG (124 documentos indexados)
- NewsManager migrado a PostgreSQL (7592 articulos)
- Dockerfile preparado con SERVICE_MODE (web/scheduler)
- APScheduler en requirements.txt
- Scheduler "awake-light" configurado y activo en Railway (05/02/2026)
  - Variables: SERVICE_MODE=scheduler, DATABASE_URL, ANTHROPIC_API_KEY
  - Conectado al repo, deploy automatico en cada push a main
  - Fix aplicado: job.next_run_time -> getattr() para APScheduler 3.11
- docker-compose.yml creado (web en puerto 8502, scheduler opcional con profile)
- .dockerignore corregido: excluye data/ para evitar contexto de 3GB+
- Acceso directo Desktop renombrado: PatrimonioSmart.bat
- Secuencia download_logs_id_seq corregida (estaba desincronizada tras migracion)

### Actualizacion 13/02/2026
- Registradas 3 compras RCO951 del 04/02 que faltaban:
  - TAL: 1,707 acciones @ 11.649 USD = 19,897.84 USD
  - PLTR: 145 acciones @ 137.777 USD = 19,990.67 USD
  - GLDD: 1,328 acciones @ 14.877 USD = 19,769.66 USD
- Corregido cash_diario RCO951 desde 04/02 al 12/02:
  - 04-11/02: USD -156,146.31 (refleja las 3 compras del 04/02)
  - 12/02: USD -261,609.75 (tras operaciones del 12/02)
- Corregido holding_diario: TAL, PLTR, GLDD añadidos al 10, 11, 12/02
- Recalculadas posiciones RCO951 con cash y holding correctos
- Posicion global verificada dia a dia (02-12/02/2026)
- Total cartera 12/02/2026: 4,149,533.53 EUR

### Actualizacion 12/02/2026
- Precios actualizados hasta 11/02/2026 (5,729 simbolos con cobertura 98.6%)
- Config.py: anadido `extra="ignore"` en Settings para ignorar vars NLP mientras modulo pausado
- Holding_diario IB actualizado con datos reales del extracto:
  - TLT: 17.424 shares (vendieron 6,555 el 02/02)
  - ESH6: -1 contrato futuro (short S&P 500)
  - Cash: 690,596 EUR + 17,703 USD
  - Total IB: ~707,000 EUR
- Registradas operaciones IB 02-11/02: 19 compras + 18 ventas (ETFs y Futuros)
- Registradas operaciones RCO951 12/02: 15 compras + 11 ventas (Quant)
- Posiciones Quant nuevas en RCO951: LC, SPOT, NBN, HOOD, PEGA, MC, LYFT, HUBS, HFWA, GE, DIOD, CURB, ROAD, AMD, AEIS

### Descarga masiva FMP (09/02/2026)
- Base de datos FMP local en Docker (localhost:5433) con 101.4M registros, 25 GB
- Descargados Key Metrics: 200,897 registros
- Descargados Ratios: 199,607 registros
- Descargados ETF Holdings: 565,514 registros (1,753 ETFs con datos)
- Descargados Earnings Transcripts: 42,244 transcripts
- Scripts creados: fmp_key_metrics.py, fmp_ratios.py, fmp_etf_holdings.py, fmp_earnings_transcripts.py
- 13F Institutional Holdings: endpoint no devuelve datos (pendiente investigar)

### Bugs PostgreSQL corregidos (05/02/2026)
- exchange_rate_service.py: eliminadas queries DEBUG que hacian spam en logs
- parse_db_date(): PostgreSQL devuelve datetime en vez de date, ahora convierte correctamente
- calculate_all_trading_days(): misma correccion datetime->date (1 enero aparecia, SPY/QQQ faltaban)
- Resumen cartera: colores dinamicos rojo/verde segun valor positivo/negativo (antes hardcoded verde)
- Pagina Acciones: fecha_compra datetime vs date causaba TypeError (crash silencioso)
- Pagina Acciones: comparacion fecha_compra > '2025-12-31' (string) cambiado a date()
- Pagina Futuros y ETF: GROUP BY sin agregar precio_entrada (PostgreSQL estricto, SQLite lo permitia)
- Pagina Futuros y ETF: fecha IB ahora usa MAX de holding_diario IB (no posicion global)

### Pendiente / En progreso
- Dashboard local lento por latencia a Railway PostgreSQL (cada query ~200ms vs <1ms en produccion)
- Metricas tecnicas no se calculan (requieren 200+ registros, PostgreSQL tiene ~191 por simbolo)
- Scheduler Railway actualizado: precios 22:30 (mismo dia), posiciones 02:00 (dia siguiente)
- Actualizar holding_diario para fechas intermedias 02-10/02 con datos correctos de IB

### Servicios Railway
| Servicio | Funcion | Estado |
|----------|---------|--------|
| enthusiastic | Web (Streamlit) en patrimoniosmart.club | Activo |
| caring | PostgreSQL | Activo |
| awake-light | Scheduler (precios 22:30 lun-vie, posiciones 02:00 mar-sab, fundamentales dom 01:00) | Activo |

---

## Docker local

### Entorno multi-proyecto
| Proyecto | Puerto | URL |
|----------|--------|-----|
| ERP La Carihuela | 8501 | localhost:8501 |
| PatrimonioSmart | 8502 | localhost:8502 |

### docker-compose.yml
- Servicio web: Streamlit en puerto 8502, SERVICE_MODE=web
- Servicio scheduler: SERVICE_MODE=scheduler (opcional, activar con `--profile scheduler`)
- Variables de entorno desde .env
- .dockerignore excluye data/ para builds rapidos

### Acceso directo Desktop
- Archivo: `C:\Users\usuario\Desktop\PatrimonioSmart.bat`

### Despues de reiniciar el PC
1. Abrir Docker Desktop (esperar icono verde "Docker is running")
2. Verificar en terminal: `docker --version`
3. Ir a la carpeta del proyecto: `cd C:\Users\usuario\financial-data-project`
4. Levantar: `docker-compose up --build`
5. Acceder al dashboard en localhost:8502

---

## Base de datos FMP (Financial Modeling Prep)

Base de datos local en Docker con datos financieros descargados desde la API de FMP.

### Conexion
- Host: localhost:5433
- Database: fmp_data
- User: fmp
- Password: fmp123
- Conexion: `postgresql://fmp:fmp123@localhost:5433/fmp_data`

### Estado actual (09/02/2026)
- **Tamano total**: 25 GB
- **Total registros**: 101.4 millones

### Tablas FMP
| Tabla | Registros | Descripcion |
|-------|-----------|-------------|
| fmp_price_history | 87,272,323 | Precios historicos diarios |
| fmp_crypto | 5,784,510 | Precios criptomonedas |
| fmp_earnings | 2,819,775 | Datos de earnings |
| fmp_forex | 1,942,728 | Tipos de cambio forex |
| fmp_dividends | 801,556 | Historial de dividendos |
| fmp_balance_sheets | 656,133 | Balance sheets |
| fmp_income_statements | 653,929 | Income statements |
| fmp_cash_flow | 648,223 | Cash flow statements |
| fmp_key_metrics | 200,897 | Metricas clave (P/E, EV/EBITDA, ROE, etc.) |
| fmp_ratios | 199,607 | Ratios financieros |
| fmp_etf_holdings | 565,514 | Holdings de ETFs |
| fmp_profiles | 92,036 | Perfiles de empresas |
| fmp_symbols | 89,103 | Lista de simbolos |
| fmp_commodities | 51,452 | Precios commodities |
| fmp_earnings_transcripts | 42,244 | Transcripts de earnings calls |
| fmp_splits | 27,232 | Stock splits |

### Scripts de descarga
Los scripts en `scripts/fmp_*.py` usan async/aiohttp con semaforo de 10 conexiones concurrentes.
Ejecutar con: `py -3 scripts/fmp_[nombre].py`

### Endpoints FMP API (Stable)
- Key Metrics: `/stable/key-metrics?symbol=X`
- Ratios: `/stable/ratios?symbol=X`
- ETF Holdings: `/stable/etf/holdings?symbol=X`
- Earnings Transcripts: `/stable/earning-call-transcript?symbol=X&year=Y&quarter=Q`

---

## Modulo NLP / Sentiment Analysis (11/02/2026)

Modulo de analisis de sentimiento con transformers, disenado para escalar de 35 GB a 1 TB.

### Estado: INFRAESTRUCTURA CREADA - INTEGRACION PAUSADA

El modulo esta creado pero la integracion con el resto del sistema esta pausada hasta que la infraestructura de datos este completa (objetivo: 700 GB - 1.75 TB).

### Estructura creada (36 archivos, ~4,200 lineas)

```
src/nlp/
├── __init__.py, config.py           # Configuracion Pydantic (NLP_* env vars)
├── models/                          # Modelos de Sentiment
│   ├── base.py                      # BaseSentimentModel + SentimentResult
│   ├── finbert.py                   # ProsusAI/finbert (financiero)
│   ├── roberta.py                   # cardiffnlp/roberta-sentiment
│   └── ensemble.py                  # Combinacion 60% FinBERT + 40% RoBERTa
├── processors/                      # Preprocesamiento
│   ├── text_cleaner.py              # Limpieza HTML, URLs, whitespace
│   ├── chunker.py                   # Segmentacion textos largos
│   └── entity_extractor.py          # Extraccion tickers, empresas, montos
├── analyzers/                       # Analizadores especializados
│   ├── news_analyzer.py             # Analisis titulo + contenido
│   └── transcript_analyzer.py       # Analisis por seccion, Q&A delta
├── calculators/                     # Calculadores (patron existente)
│   ├── news_sentiment_calc.py       # news_history -> nlp_sentiment_news
│   ├── transcript_sentiment_calc.py # transcripts -> nlp_sentiment_transcript
│   └── aggregate_sentiment_calc.py  # Features diarias agregadas
├── services/                        # Servicios singleton
│   ├── sentiment_service.py         # Servicio principal con fallback
│   └── embedding_service.py         # Generacion de embeddings
├── storage/                         # Capa de almacenamiento escalable
│   ├── base.py                      # Interfaz abstracta
│   ├── postgres.py                  # PostgreSQL + pgvector
│   ├── vector_store.py              # ChromaDB / pgvector
│   └── cache.py                     # Redis / memoria
└── pipelines/                       # Pipelines de procesamiento
    ├── batch_processor.py           # Procesamiento masivo
    └── incremental_processor.py     # Actualizaciones tiempo real

scripts/nlp/
├── create_tables.sql                # Schema SQL (tablas particionadas)
├── run_batch_processing.py          # CLI batch processing
└── benchmark_models.py              # Benchmark de rendimiento

tests/nlp/
├── test_models.py                   # Tests unitarios
└── test_integration.py              # Tests de integracion
```

### Tablas SQL preparadas (sin ejecutar)

| Tabla | Descripcion |
|-------|-------------|
| nlp_sentiment_news | Sentiment por articulo (particionada por ano) |
| nlp_sentiment_transcript | Sentiment de earnings calls con Q&A delta |
| features_sentiment_daily | Features agregadas por simbolo/dia |
| nlp_embeddings | Embeddings con pgvector (768 dims) |

### Dependencias anadidas a requirements.txt

```
transformers>=4.36.0
torch>=2.1.0
accelerate>=0.25.0
pgvector>=0.2.0
redis>=5.0.0
```

### Fases de implementacion NLP

| Fase | Estado | Descripcion |
|------|--------|-------------|
| Fase 1: Infraestructura | ✅ Completa | Estructura, config, SQL, BaseSentimentModel |
| Fase 2: Modelos | ✅ Completa | FinBERT, RoBERTa, Ensemble, SentimentService |
| Fase 3: Procesadores | ✅ Completa | TextCleaner, Chunker, Analyzers |
| Fase 4: Calculadores | ✅ Completa | News, Transcript, Aggregate calculators |
| Fase 5: Integracion | ⏸️ PAUSADA | Integrar con ai_assistant, portfolio_rag, scheduler |

### Cuando activar (futuro)

1. Ejecutar: `psql -U fmp -d fmp_data -f scripts/nlp/create_tables.sql`
2. Instalar: `pip install transformers torch accelerate`
3. Procesar historico: `python -m scripts.nlp.run_batch_processing`
4. Integrar con ai_assistant.py, portfolio_rag.py, scheduler.py

### Hoja de ruta escalabilidad

```
FASE 1-2 (ACTUAL):  35 GB   - SQLite + PostgreSQL FMP
FASE 3 (+NLP):      50-100 GB - Activar modulo NLP, pgvector
FASE 4 (+ML):       100-300 GB - Embeddings, features ML, tick data
FASE 5 (+DL):       300-500 GB - TimescaleDB, modelos DL
FASE 6 (Enterprise): 500 GB-1 TB - Cluster PostgreSQL distribuido
```

---

## TAREAS INMEDIATAS AL INICIAR SESION (hacer sin preguntar)

1. Ejecutar `docker-compose up --build` (levantar dashboard local en localhost:8502)
2. Verificar que el scheduler "awake-light" esta corriendo en Railway (check logs)
3. Comprobar que los datos estan actualizados (ultimo dia de mercado)

---

## Como continuar en la proxima sesion

Si el usuario dice "Continuar con PatrimonioSmart":
1. Ejecutar las TAREAS INMEDIATAS de arriba (sin preguntar)
2. Si hay tareas pendientes arriba, continuarlas

Comandos utiles:
- Scheduler manual: `py -3 -m src.scheduler --run-now`
- Dashboard local sin Docker: `py -3 -m streamlit run web/app.py`
- Dashboard con Docker: `docker-compose up --build`
- Railway CLI: `C:\Users\usuario\financial-data-project\railway.exe` (requiere login)

---

## Sistema Trading Semanal S&P 500

### Concepto

Sistema que analiza el mercado cada viernes, clasifica el regimen de mercado y opera
la semana siguiente. Se basa en subsectores del S&P 500 (industrias con >= 3 acciones).

### Flujo temporal (CRITICO - no cambiar)

```
Semana W:
  Vie W-1 close ──► Vie W close    = SENAL (datos para calcular regimen)
                                      Con los datos del Vie W se calcula:
                                      - Indicadores breadth (DD, RSI de subsectores)
                                      - SPY vs MA200, momentum, distancia
                                      - VIX
                                      → Score total → Regimen de mercado

  Vie W close ──► Lun W+1 open     = GAP fin de semana (no se opera, se asume)

  Lun W+1 open ──► Lun W+2 open    = TRADING (rentabilidad real del trade)
                                      Entrada: apertura lunes siguiente a la senal
                                      Salida: apertura lunes una semana despues
```

Ejemplo semana 1 de 2026:
- Senal: Vie 26/12/2025 close (690.31) → Vie 02/01/2026 close (683.17) = -1.03%
- Regimen calculado con datos del 02/01/2026 → NEUTRAL (score +3.5)
- Trading: Lun 05/01/2026 open (686.54) → Lun 12/01/2026 open (690.68) = +0.60%

IMPORTANTE: El retorno de la senal (Vie→Vie) NO es el retorno del trade.
El retorno real es Lun open → Lun open, con el gap del fin de semana incluido.

### Clasificacion de regimen de mercado

5 indicadores → score individual → suma total → regimen:

#### 1. BDD - Breadth Drawdown (rango: -3.0 a +2.0)
Que mide: % de subsectores con drawdown 52w > -10% (saludables)
```
>= 75% → +2.0    >= 60% → +1.0    >= 45% → 0.0
>= 30% → -1.0    >= 15% → -2.0    < 15%  → -3.0
```

#### 2. BRSI - Breadth RSI (rango: -3.0 a +2.0)
Que mide: % de subsectores con RSI 14 semanas > 55
```
>= 75% → +2.0    >= 60% → +1.0    >= 45% → 0.0
>= 30% → -1.0    >= 15% → -2.0    < 15%  → -3.0
```

#### 3. DDP - Deep Drawdown Percentage (rango: -2.5 a +1.5)
Que mide: % de subsectores con drawdown 52w < -20% (profundos)
```
<= 5%  → +1.5    <= 15% → +0.5    <= 30% → -0.5
<= 50% → -1.5    > 50%  → -2.5
```

#### 4. SPY - SPY vs MA200 (rango: -2.5 a +1.5)
Que mide: posicion del SPY respecto a su media movil de 200 dias
```
> MA200 y dist > 5%   → +1.5    > MA200 y dist <= 5%  → +0.5
< MA200 y dist > -5%  → -0.5    < MA200 y dist > -15% → -1.5
< MA200 y dist <= -15% → -2.5
```

#### 5. MOM - Momentum SPY 10 semanas (rango: -1.5 a +1.0)
Que mide: cambio % del SPY en las ultimas 10 semanas
```
> 5%   → +1.0    > 0%   → +0.5    > -5%  → -0.5
> -15%  → -1.0    <= -15% → -1.5
```

#### Score total y regimenes
Rango posible: -12.5 a +8.0
```
BURBUJA:    >= 8.0 (Y ademas DD_H >= 85% Y RSI>55 >= 90%)
GOLDILOCKS: >= 7.0
ALCISTA:    >= 4.0
NEUTRAL:    >= 0.5
CAUTIOUS:   >= -2.0
BEARISH:    >= -5.0
CRISIS:     >= -9.0
PANICO:     < -9.0
```

#### VIX Veto (filtro de seguridad)
- VIX >= 30: rebaja BURBUJA/GOLDILOCKS/ALCISTA → NEUTRAL
- VIX >= 35: rebaja NEUTRAL → CAUTIOUS
El VIX protege contra falsos positivos en mercados alcistas con alta volatilidad.

### Rentabilidad SPY por regimen (2001-2026, Mon→Mon, 1311 semanas)

| Regimen    |   N  |  %Sem  | Avg%   |  WR%  | Total% |
|------------|------|--------|--------|-------|--------|
| BURBUJA    |   21 |  1.6%  | +0.33% | 71.4% |   +7.0 |
| GOLDILOCKS |  230 | 17.5%  | +0.47% | 68.3% | +107.9 |
| ALCISTA    |  364 | 27.8%  | +0.32% | 62.9% | +116.7 |
| NEUTRAL    |  256 | 19.5%  | +0.14% | 56.2% |  +36.9 |
| CAUTIOUS   |  134 | 10.2%  | -0.19% | 47.8% |  -25.5 |
| BEARISH    |  113 |  8.6%  | +0.04% | 52.2% |   +4.5 |
| CRISIS     |  119 |  9.1%  | -0.42% | 49.6% |  -49.7 |
| PANICO     |   74 |  5.6%  | +0.06% | 47.3% |   +4.5 |

Los regimenes discriminan correctamente: GOLDILOCKS/ALCISTA positivos,
CRISIS/CAUTIOUS negativos. PANICO es neutro (incluye rebotes violentos).

### Mapeo de retornos en report_compound.py (CRITICO)

```python
# En df_monday: return_mon = open(this_mon) / open(prev_mon) - 1
# Es backward-looking: almacenado en lunes W, representa retorno semana W-1→W

# Para senal viernes W:
#   Trading = Lun W+1 open → Lun W+2 open
#   return_mon del Lun W+2 = open(W+2) / open(W+1) - 1 = retorno correcto
#   fri + 10 dias = Lun W+2 (donde esta almacenado el retorno)

target = fri + pd.Timedelta(days=10)  # CORRECTO
# target = fri + pd.Timedelta(days=3)  # INCORRECTO - daba retorno ya conocido
```

### FV Scores - datos disponibles

```python
# Para calcular scores/rankings de subsectores se usa:
prev_dates = dd_wide.index[dd_wide.index <= date]  # CORRECTO: incluye viernes senal
# prev_dates = dd_wide.index[dd_wide.index < date]  # INCORRECTO: excluia datos del dia
```

El viernes de la senal sus datos ya son conocidos (close ya ocurrio),
por lo tanto se deben incluir con <= date.

### Datos de subsectores
- Fuente: acciones S&P 500 agrupadas por `industry` (campo de fmp_profiles)
- Filtro: solo subsectores con >= 3 acciones
- Total: 66 subsectores validos
- Precio subsector: media del close de todas las acciones del subsector (semanal viernes)
- Metricas: drawdown 52w, RSI 14w, calculadas sobre el precio medio del subsector

### Archivos del sistema

| Archivo | Descripcion |
|---------|-------------|
| `report_compound.py` | Backtest principal: calcula regimenes, selecciona subsectores, simula trades |
| `analisis_transiciones.py` | Analisis de transiciones entre regimenes, genera HTML |
| `analisis_transiciones.html` | Visualizacion de transiciones (rentabilidades Mon→Mon) |
| `regimenes_historico.py` | Calcula regimenes + SPY ret para todas las semanas desde 2001 |
| `regimenes_2026_tabla.py` | Tabla detallada de regimenes 2026 con todos los indicadores |
| `semana1_completo.py` | Ejemplo completo del proceso para semana 1 de 2026 |
| `senal_semana9_dual.py` | Genera senal semanal (picks para estrategia A y B) |
| `data/regimenes_historico.csv` | CSV con todos los regimenes e indicadores (2001-2026) |
| `data/vix_weekly.csv` | Datos VIX semanales |
| `data/sp500_constituents.json` | Constituyentes actuales S&P 500 |
| `data/sp500_historical_changes.json` | Cambios historicos S&P 500 |

### Errores corregidos (27/02/2026)

1. **Mapeo retornos fri+3 → fri+10**: El backtest usaba fri+3 que mapeaba al lunes siguiente
   donde return_mon = retorno de la semana YA PASADA (dato conocido). Corregido a fri+10
   que mapea al lunes de 2 semanas despues donde esta el retorno real del trade.
   Impacto: resultados pasaron de $500K→$119M (falso) a $500K→$42K (real).

2. **FV scores < date → <= date**: Los scores excluian los datos del viernes de senal.
   Corregido a <= date porque el close del viernes ya es conocido al generar la senal.

3. **analisis_transiciones.py Fri→Fri → Mon→Mon**: La tabla de retornos SPY por regimen
   usaba Fri close → Fri close. Corregido a Mon open → Mon open (retorno real de trading).
   PANICO paso de +0.98% (falso) a +0.06% (real).
