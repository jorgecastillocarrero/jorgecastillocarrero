# Financial Data Automation Project

Sistema de automatización para descargar, almacenar y visualizar datos financieros desde EODHD, con capacidades de análisis técnico e IA.

## Características

- **Cliente EODHD completo**: Soporte para datos EOD, tiempo real, fundamentales y bulk
- **Base de datos SQLAlchemy**: SQLite por defecto, compatible con PostgreSQL
- **Scheduler automatizado**: Descargas programadas con APScheduler
- **Análisis técnico**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- **Detección de patrones**: Tendencias, soportes/resistencias, cruces de medias
- **Dashboard Streamlit**: Visualización interactiva con Plotly
- **Integración OpenAI** (opcional): Resúmenes generados por IA

## Estructura del Proyecto

```
financial-data-project/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuración y variables de entorno
│   ├── eodhd_client.py        # Cliente API para EODHD
│   ├── database.py            # Modelos SQLAlchemy
│   ├── scheduler.py           # Automatización de descargas
│   └── analysis/
│       ├── __init__.py
│       └── ai_analyzer.py     # Análisis técnico e IA
├── web/
│   └── app.py                 # Dashboard Streamlit
├── data/                      # Base de datos SQLite
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Instalación

1. **Clonar/Crear el proyecto**
   ```bash
   cd financial-data-project
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
   ```bash
   cp .env.example .env
   # Editar .env con tu API key de EODHD
   ```

## Uso

### Dashboard Web

```bash
streamlit run web/app.py
```

Abre http://localhost:8501 en tu navegador.

### Cliente EODHD (CLI)

```bash
# Probar conexión y descargar datos de prueba
python -m src.eodhd_client
```

### Scheduler

```bash
# Probar el scheduler
python -m src.scheduler
```

### Análisis

```bash
# Probar el analizador con datos de ejemplo
python -m src.analysis.ai_analyzer
```

## API Endpoints Soportados

| Endpoint | Descripción |
|----------|-------------|
| `/eod/{symbol}` | Datos históricos End-of-Day |
| `/real-time/{symbol}` | Cotizaciones en tiempo real |
| `/fundamentals/{symbol}` | Datos fundamentales |
| `/bulk-fundamentals/{exchange}` | Descarga masiva de fundamentales |
| `/exchanges-list` | Lista de exchanges disponibles |
| `/exchange-symbol-list/{exchange}` | Símbolos por exchange |
| `/div/{symbol}` | Datos de dividendos |
| `/splits/{symbol}` | Datos de splits |

## Indicadores Técnicos

- **SMA**: Media móvil simple (20, 50, 200 períodos)
- **EMA**: Media móvil exponencial (12, 26 períodos)
- **RSI**: Índice de fuerza relativa (14 períodos)
- **MACD**: Convergencia/divergencia de medias móviles
- **Bollinger Bands**: Bandas de volatilidad
- **ATR**: Rango verdadero promedio

## Base de Datos

### Modelos

- `Exchange`: Información de mercados/bolsas
- `Symbol`: Instrumentos financieros
- `PriceHistory`: Datos históricos OHLCV
- `Fundamental`: Datos fundamentales
- `DownloadLog`: Registro de descargas

### Migración a PostgreSQL

Simplemente cambia `DATABASE_URL` en `.env`:

```
DATABASE_URL=postgresql://user:password@localhost:5432/financial_data
```

## Configuración del Scheduler

El scheduler puede configurarse para ejecutar descargas automáticas:

```python
from src.scheduler import SchedulerManager

scheduler = SchedulerManager()
scheduler.add_eod_job(
    symbols=["AAPL.US", "MSFT.US"],
    hour=18,  # 6 PM
    minute=0
)
scheduler.start()
```

## Integración con OpenAI

Para habilitar resúmenes generados por IA:

1. Configura `OPENAI_API_KEY` en `.env`
2. El dashboard mostrará automáticamente resúmenes de análisis

## Requisitos

- Python 3.11+
- API Key de EODHD (https://eodhd.com)
- OpenAI API Key (opcional)

## Licencia

MIT License
