# Plan: Enterprise-Ready Code Cleanup v2

## Objetivo
Corregir issues críticos de calidad, seguridad y mantenibilidad identificados en el análisis.

---

## FASE 1: Corregir Excepciones Silenciosas (CRÍTICO)

### 1.1 web/app.py - Reemplazar `except:` con manejo específico

| Línea | Contexto | Cambio |
|-------|----------|--------|
| 1327 | Conversión moneda | `except ValueError as e:` + logging |
| 1479 | Parsing P&L | `except ValueError as e:` + logging |
| 2050 | Query tabla | `except Exception as e:` + mostrar error |

**Antes:**
```python
except:
    pass
```

**Después:**
```python
except (ValueError, TypeError) as e:
    logger.warning(f"Error parsing value: {e}")
```

### 1.2 src/yahoo_downloader.py:165

**Antes:**
```python
except:
    info = {}
    name = clean_symbol
```

**Después:**
```python
except Exception as e:
    logger.warning(f"Could not fetch info for {clean_symbol}: {e}")
    info = {}
    name = clean_symbol
```

### 1.3 import_ib_data.py:133, 168

**Antes:**
```python
except:
    continue
```

**Después:**
```python
except ValueError as e:
    print(f"  Skipping trade with invalid datetime: {t.get('trade_datetime')} - {e}")
    continue
```

---

## FASE 2: Configuración Segura

### 2.1 src/config.py - Eliminar path absoluto hardcodeado

**Línea 25 - Antes:**
```python
database_url: str = "sqlite:///C:/Users/usuario/financial-data-project/data/financial_data.db"
```

**Después:**
```python
database_url: str = ""  # Will be set from DATA_DIR

@property
def effective_database_url(self) -> str:
    """Get database URL, defaulting to data/financial_data.db in project root."""
    if self.database_url:
        return self.database_url
    return f"sqlite:///{self.data_dir}/financial_data.db"
```

### 2.2 Crear src/constants.py - Centralizar valores por defecto

```python
"""Default constants for the application."""

# Exchange rate fallbacks (used when market data unavailable)
DEFAULT_EXCHANGE_RATES = {
    "EUR_USD": 1.04,
    "USD_CAD": 1.44,
    "USD_CHF": 0.90,
    "GBP_EUR": 1.18,
    "CAD_EUR": 0.68,
    "CHF_EUR": 1.05,
}

# Rate lookup settings
PRICE_LOOKBACK_DAYS = 6  # Days to look back for prices
```

---

## FASE 3: Refactorizar Exchange Rates (Eliminar Duplicados)

### 3.1 Crear src/exchange_rate_service.py

```python
"""Centralized exchange rate service."""

from datetime import date, timedelta
from typing import Optional
import logging

from .constants import DEFAULT_EXCHANGE_RATES, PRICE_LOOKBACK_DAYS
from .database import get_db_manager, Symbol, PriceHistory

logger = logging.getLogger(__name__)


class ExchangeRateService:
    """Service for fetching and caching exchange rates."""

    def __init__(self, db_manager=None):
        self.db = db_manager or get_db_manager()
        self._cache = {}

    def get_rate(self, pair: str, target_date: date) -> Optional[float]:
        """Get exchange rate for a currency pair on a specific date."""
        cache_key = f"{pair}_{target_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        with self.db.get_session() as session:
            symbol = session.query(Symbol).filter(Symbol.code == pair).first()
            if not symbol:
                return None

            for i in range(PRICE_LOOKBACK_DAYS):
                check_date = target_date - timedelta(days=i)
                price = session.query(PriceHistory).filter(
                    PriceHistory.symbol_id == symbol.id,
                    PriceHistory.date == check_date
                ).first()
                if price and price.close:
                    self._cache[cache_key] = price.close
                    return price.close

        return None

    def get_eur_usd(self, target_date: date) -> float:
        """Get EUR/USD rate with fallback."""
        rate = self.get_rate('EURUSD=X', target_date)
        return rate or DEFAULT_EXCHANGE_RATES["EUR_USD"]

    def get_cad_eur(self, target_date: date) -> float:
        """Get CAD/EUR rate with fallback."""
        rate = self.get_rate('CADEUR=X', target_date)
        if rate:
            return rate
        eur_usd = self.get_eur_usd(target_date)
        usd_cad = self.get_rate('USDCAD=X', target_date) or DEFAULT_EXCHANGE_RATES["USD_CAD"]
        return 1 / (usd_cad * eur_usd)

    def get_chf_eur(self, target_date: date) -> float:
        """Get CHF/EUR rate with fallback."""
        rate = self.get_rate('CHFEUR=X', target_date)
        if rate:
            return rate
        eur_usd = self.get_eur_usd(target_date)
        usd_chf = self.get_rate('USDCHF=X', target_date) or DEFAULT_EXCHANGE_RATES["USD_CHF"]
        return 1 / (usd_chf * eur_usd)

    def convert_to_eur(self, amount: float, currency: str, target_date: date) -> float:
        """Convert amount from any currency to EUR."""
        if currency == 'EUR':
            return amount
        elif currency == 'USD':
            return amount / self.get_eur_usd(target_date)
        elif currency == 'CAD':
            return amount * self.get_cad_eur(target_date)
        elif currency == 'CHF':
            return amount * self.get_chf_eur(target_date)
        elif currency == 'GBP':
            return amount * DEFAULT_EXCHANGE_RATES["GBP_EUR"]
        else:
            logger.warning(f"Unknown currency {currency}, returning original amount")
            return amount


# Singleton
_rate_service: Optional[ExchangeRateService] = None

def get_exchange_rate_service(db_manager=None) -> ExchangeRateService:
    global _rate_service
    if _rate_service is None:
        _rate_service = ExchangeRateService(db_manager)
    return _rate_service
```

### 3.2 Actualizar archivos que usan exchange rates

**src/portfolio_data.py** - Usar nuevo servicio:
```python
from .exchange_rate_service import get_exchange_rate_service

# En __init__:
self.rate_service = get_exchange_rate_service(db_manager)

# Reemplazar métodos get_eur_usd_rate, get_cad_eur_rate, get_chf_eur_rate
# con llamadas a self.rate_service
```

**src/daily_tracking.py:433-448** - Eliminar función `get_rate()` nested, usar servicio.

**web/app.py:78-88** - Eliminar `_get_price_or_previous()`, usar servicio.

---

## FASE 4: Validación de Inputs

### 4.1 Crear src/validators.py

```python
"""Input validation utilities."""

import re
from typing import Optional


def validate_symbol(symbol: str) -> tuple[str, str]:
    """
    Validate and parse a symbol string.

    Args:
        symbol: Symbol in format "CODE" or "CODE.EXCHANGE"

    Returns:
        Tuple of (symbol_code, exchange_code)

    Raises:
        ValueError: If symbol format is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Invalid symbol: {symbol}")

    symbol = symbol.strip().upper()

    if not re.match(r'^[A-Z0-9\-\.=]+$', symbol):
        raise ValueError(f"Symbol contains invalid characters: {symbol}")

    parts = symbol.split(".")
    if len(parts) == 1:
        return parts[0], "US"
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise ValueError(f"Invalid symbol format: {symbol}")


def validate_positive_int(value: int, name: str, max_value: Optional[int] = None) -> int:
    """Validate a positive integer parameter."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got: {value}")
    if max_value and value > max_value:
        raise ValueError(f"{name} exceeds maximum value {max_value}: {value}")
    return value
```

### 4.2 Aplicar validación en web/app.py

**Línea ~215:**
```python
def get_price_data(symbol: str, days: int = 365) -> pd.DataFrame:
    from src.validators import validate_symbol, validate_positive_int

    symbol_code, exchange_code = validate_symbol(symbol)
    days = validate_positive_int(days, "days", max_value=3650)
    # ... resto del código
```

---

## FASE 5: Logging Estructurado

### 5.1 Añadir logger a web/app.py

**Al inicio del archivo:**
```python
import logging

logger = logging.getLogger(__name__)
```

### 5.2 Reemplazar `print()` con `logger` en scripts

Archivos afectados:
- `import_ib_data.py`
- `batch_download.py`
- `download_full_history.py`

---

## FASE 6: Tests Adicionales

### 6.1 tests/test_exchange_rate_service.py (NUEVO)

```python
"""Tests for exchange rate service."""

import pytest
from datetime import date
from unittest.mock import patch, MagicMock

from src.exchange_rate_service import ExchangeRateService, get_exchange_rate_service
from src.constants import DEFAULT_EXCHANGE_RATES


class TestExchangeRateService:
    def test_get_eur_usd_fallback(self):
        """Test EUR/USD returns fallback when no data."""
        service = ExchangeRateService(db_manager=MagicMock())
        with patch.object(service, 'get_rate', return_value=None):
            rate = service.get_eur_usd(date(2024, 1, 15))
            assert rate == DEFAULT_EXCHANGE_RATES["EUR_USD"]

    def test_convert_to_eur_usd(self):
        """Test USD to EUR conversion."""
        service = ExchangeRateService(db_manager=MagicMock())
        with patch.object(service, 'get_eur_usd', return_value=1.10):
            result = service.convert_to_eur(110.0, 'USD', date(2024, 1, 15))
            assert result == 100.0

    def test_convert_to_eur_passthrough(self):
        """Test EUR stays as EUR."""
        service = ExchangeRateService(db_manager=MagicMock())
        result = service.convert_to_eur(100.0, 'EUR', date(2024, 1, 15))
        assert result == 100.0
```

### 6.2 tests/test_validators.py (NUEVO)

```python
"""Tests for input validators."""

import pytest
from src.validators import validate_symbol, validate_positive_int


class TestValidateSymbol:
    def test_valid_us_symbol(self):
        code, exchange = validate_symbol("AAPL")
        assert code == "AAPL"
        assert exchange == "US"

    def test_valid_with_exchange(self):
        code, exchange = validate_symbol("IAG.MC")
        assert code == "IAG"
        assert exchange == "MC"

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            validate_symbol("")

    def test_invalid_characters(self):
        with pytest.raises(ValueError):
            validate_symbol("AAPL;DROP TABLE")


class TestValidatePositiveInt:
    def test_valid(self):
        assert validate_positive_int(100, "days") == 100

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            validate_positive_int(0, "days")

    def test_exceeds_max(self):
        with pytest.raises(ValueError):
            validate_positive_int(5000, "days", max_value=3650)
```

---

## Resumen de Archivos

### Archivos a CREAR (4):
| Archivo | Propósito |
|---------|-----------|
| `src/constants.py` | Constantes centralizadas |
| `src/exchange_rate_service.py` | Servicio de tipos de cambio |
| `src/validators.py` | Validación de inputs |
| `tests/test_validators.py` | Tests de validación |

### Archivos a MODIFICAR (6):
| Archivo | Cambios |
|---------|---------|
| `web/app.py` | Excepciones específicas, usar servicios, logging |
| `src/config.py` | Eliminar path hardcodeado |
| `src/portfolio_data.py` | Usar ExchangeRateService |
| `src/daily_tracking.py` | Usar ExchangeRateService |
| `src/yahoo_downloader.py` | Excepción específica |
| `import_ib_data.py` | Excepciones específicas |

---

## Restricciones (según CLAUDE.md)

**REQUIERE AUTORIZACIÓN para modificar:**
- `web/app.py` - Páginas Posición, Composición, Acciones/ETF, Futuros
- `src/portfolio_data.py` - Funciones de cálculo

---

## Orden de Ejecución

1. **FASE 1**: Excepciones (no requiere autorización en la mayoría)
2. **FASE 2**: Configuración (crear nuevos archivos)
3. **FASE 3**: Exchange Rate Service (requiere autorización para portfolio_data.py)
4. **FASE 4**: Validadores (crear nuevos archivos)
5. **FASE 5**: Logging (cambios menores)
6. **FASE 6**: Tests (crear nuevos archivos)

---

## Resultado Esperado

- ✅ 0 excepciones silenciosas
- ✅ 0 valores hardcodeados en código
- ✅ Código DRY (sin duplicados)
- ✅ Inputs validados
- ✅ Logging estructurado
- ✅ Tests para nuevos módulos
