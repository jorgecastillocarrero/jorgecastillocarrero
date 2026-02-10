# CLAUDE.md - PatrimonioSmart Financial Project

## Que es este proyecto

PatrimonioSmart es una plataforma de gestion patrimonial que:
- Descarga y almacena datos financieros diarios (precios, fundamentales, noticias)
- Calcula posiciones, valoraciones y metricas de cartera
- Muestra un dashboard interactivo (Streamlit) en patrimoniosmart.club
- Incluye un asistente IA con RAG (LangChain + ChromaDB)
- Scheduler automatico para actualizaciones diarias a las 00:01 ET

---

## Stack tecnologico

| Componente | Tecnologia |
|------------|-----------|
| Backend/Data | Python 3.12, Pandas, SQLAlchemy, yfinance |
| Base de datos | PostgreSQL (Railway, servicio "caring") |
| Dashboard | Streamlit (web/app.py) |
| IA/RAG | Anthropic Claude, LangChain, ChromaDB, HuggingFace embeddings |
| Scheduler | APScheduler (cron diario 00:01 ET) |
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
- Datos actualizados hasta 04/02/2026 (5877 registros, posicion: 4,111,180 EUR)

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
- Holding_diario de IB solo hasta 30/01/2026 (falta actualizar)
- Metricas tecnicas no se calculan (requieren 200+ registros, PostgreSQL tiene ~191 por simbolo)
- Verificar que el scheduler de Railway ejecuta correctamente a las 00:01 ET diario

### Servicios Railway
| Servicio | Funcion | Estado |
|----------|---------|--------|
| enthusiastic | Web (Streamlit) en patrimoniosmart.club | Activo |
| caring | PostgreSQL | Activo |
| awake-light | Scheduler (00:01 ET diario, dom 01:00 fundamentales) | Activo |

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
