"""
AI Assistant Module for Financial Data Project
All data queries use LOCAL DATABASE ONLY - no external API calls

Architecture:
- symbols + fundamentals + price_history = all market data
- db_analysis_tools = screening, technical indicators
- backtest_engine = backtesting strategies
- ml_predictor = price/trend predictions
- risk_analyzer = risk metrics, portfolio optimization
- data_validator = data quality checks
- external_data = economic indicators, news
- AI backends (Claude/Gemini) = natural language responses
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# AI MODEL BACKENDS
# =============================================================================

class AIBackend:
    """Base class for AI backends"""

    def __init__(self):
        self.name = "base"
        self.available = False

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError


class GeminiBackend(AIBackend):
    """Google Gemini backend"""

    def __init__(self):
        super().__init__()
        self.name = "gemini"
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.available = True
            except Exception:
                self.available = False

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if not self.available:
            raise RuntimeError("Gemini not available")

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt
        )
        return response.text


class GroqBackend(AIBackend):
    """Groq backend (fast inference)"""

    def __init__(self):
        super().__init__()
        self.name = "groq"
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                self.available = True
            except Exception:
                self.available = False

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if not self.available:
            raise RuntimeError("Groq not available")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content


class AnthropicBackend(AIBackend):
    """Anthropic Claude backend"""

    def __init__(self):
        super().__init__()
        self.name = "claude"
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.client = None

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.available = True
            except Exception:
                self.available = False

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if not self.available:
            raise RuntimeError("Claude not available")

        kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


# =============================================================================
# FINANCIAL ASSISTANT
# =============================================================================

class FinancialAssistant:
    """
    AI Assistant with full database analysis capabilities.
    ALL data comes from local database - no external API calls.
    """

    SYSTEM_PROMPT = """Eres un asistente financiero experto con acceso a una base de datos completa de mercados.

DATOS DISPONIBLES (todo desde base de datos local):
- 5,813 simbolos con datos fundamentales
- 23.8 millones de registros de precios historicos
- Market cap, PE, PEG, margenes, crecimiento para cada accion
- 40 indicadores tecnicos calculables (RSI, MACD, Bollinger, etc.)

PORTFOLIO DEL USUARIO:
- Cuentas: CO3365, RCO951, LACAIXA, IB
- Posiciones largas y cortas
- Efectivo en EUR y USD

CAPACIDADES:
1. Screening: filtrar acciones por market cap, RSI, PE, sector, etc.
2. Analisis tecnico: RSI, MACD, Bollinger, Stochastic, ADX, ATR, CCI, etc.
3. Fundamentales: market cap, PE ratio, margenes, crecimiento
4. Portfolio: posiciones, cash, operaciones
5. Backtesting: simular estrategias (SMA crossover, RSI, MACD, Bollinger, custom)
6. Predicciones ML: Prophet, ARIMA, XGBoost, LightGBM para predecir precios/tendencias
7. Analisis de riesgo: VaR, CVaR, Sharpe, Sortino, Max Drawdown, correlaciones
8. Optimizacion de portfolio: pesos optimos para maximizar Sharpe o minimizar volatilidad
9. Validacion de datos: detectar outliers, datos faltantes, problemas de consistencia
10. Datos externos: indicadores economicos (Fed), noticias, eventos historicos de mercado

ESTRATEGIAS DE BACKTEST DISPONIBLES:
- sma_crossover / sma_20_50 / sma_50_200: Cruce de medias moviles
- rsi_oversold: Compra en sobreventa, venta en sobrecompra
- macd_signal: Cruce de MACD con signal
- bollinger_breakout: Compra en banda inferior, venta en superior
- combined: Combinacion de SMA + RSI + MACD
- buy_hold: Buy and hold (benchmark)

MODELOS DE PREDICCION:
- prophet: Series temporales (mejor para largo plazo)
- arima: ARIMA estadistico
- xgboost/lightgbm: Clasificacion de tendencia (alcista/bajista)
- linear: Regresion lineal simple

IMPORTANTE:
- RSI < 30 = sobrevendido, RSI > 70 = sobrecomprado
- Valores negativos en shares = posicion corta (short)
- VaR 95% indica la perdida maxima esperada 1 de cada 20 dias
- Sharpe > 1 es bueno, > 2 es excelente
- Responde siempre en espanol, de forma clara y concisa
- Usa tablas cuando sea apropiado
- Incluye siempre los valores numericos exactos
"""

    def __init__(self, preferred_backend: str = None):
        # Initialize database analyzer
        self.db = None
        try:
            from src.db_analysis_tools import DatabaseAnalyzer
            self.db = DatabaseAnalyzer()
        except ImportError:
            pass

        # Initialize new modules
        self.backtest = None
        self.ml = None
        self.risk = None
        self.validator = None
        self.external = None

        try:
            from src.backtest_engine import BacktestEngine
            self.backtest = BacktestEngine()
        except ImportError:
            pass

        try:
            from src.ml_predictor import MLPredictor
            self.ml = MLPredictor()
        except ImportError:
            pass

        try:
            from src.risk_analyzer import RiskAnalyzer
            self.risk = RiskAnalyzer()
        except ImportError:
            pass

        try:
            from src.data_validator import DataValidator
            self.validator = DataValidator()
        except ImportError:
            pass

        try:
            from src.external_data import ExternalDataProvider
            self.external = ExternalDataProvider()
        except ImportError:
            pass

        # News manager
        self.news = None
        try:
            from src.news_manager import NewsManager
            self.news = NewsManager()
        except ImportError:
            pass

        # Initialize AI backends (Claude > Gemini > Groq)
        self.backends = {}
        self.active_backend = None

        for BackendClass in [AnthropicBackend, GeminiBackend, GroqBackend]:
            backend = BackendClass()
            self.backends[backend.name] = backend

        if preferred_backend and preferred_backend in self.backends:
            if self.backends[preferred_backend].available:
                self.active_backend = self.backends[preferred_backend]

        if not self.active_backend:
            for backend in self.backends.values():
                if backend.available:
                    self.active_backend = backend
                    break

    def _parse_query(self, query: str) -> Dict:
        """Parse query to extract parameters"""
        q = query.lower()
        params = {}

        # RSI threshold
        rsi_match = re.search(r'rsi[^\d]*(\d+)', q)
        if rsi_match:
            params['rsi_value'] = int(rsi_match.group(1))
            if any(w in q for w in ['bajo', 'below', 'menor', 'debajo', '<']):
                params['rsi_below'] = params['rsi_value']
            elif any(w in q for w in ['alto', 'above', 'mayor', 'encima', '>']):
                params['rsi_above'] = params['rsi_value']

        # Market cap
        cap_match = re.search(r'([\d.]+)\s*(billion|b\b|mil millones|billones)', q)
        if cap_match:
            params['min_market_cap_b'] = float(cap_match.group(1))
        else:
            cap_match2 = re.search(r'capitaliz[^\d]*([\d.]+)', q)
            if cap_match2:
                params['min_market_cap_b'] = float(cap_match2.group(1))

        # Sector
        sectors = ['technology', 'healthcare', 'financial', 'energy', 'consumer',
                   'industrial', 'utilities', 'materials', 'real estate', 'communication']
        for sector in sectors:
            if sector in q:
                params['sector'] = sector
                break

        # Specific symbol
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', query)
        if symbol_match and symbol_match.group(1) not in ['RSI', 'MACD', 'PE', 'ETF', 'USD', 'EUR', 'VAR', 'GDP', 'VIX']:
            params['symbol'] = symbol_match.group(1)

        # PERFORMANCE / RETURN queries
        if any(w in q for w in ['subieron', 'subido', 'ganaron', 'ganado', 'subida', 'mejor rendimiento',
                                 'top gainers', 'gainers', 'mejores', 'ganadoras', 'alcistas']):
            params['wants_performance'] = True
            params['performance_type'] = 'gainers'
        elif any(w in q for w in ['bajaron', 'bajado', 'perdieron', 'perdido', 'bajada', 'peor rendimiento',
                                   'top losers', 'losers', 'peores', 'perdedoras', 'bajistas']):
            params['wants_performance'] = True
            params['performance_type'] = 'losers'
        elif any(w in q for w in ['rendimiento', 'return', 'performance', 'rentabilidad']):
            params['wants_performance'] = True
            params['performance_type'] = 'all'

        # Year extraction
        year_match = re.search(r'20(2[0-9])', q)
        if year_match:
            params['year'] = int('20' + year_match.group(1))

        # Period in days
        period_match = re.search(r'ultimos?\s*(\d+)\s*(dias?|days?)', q)
        if period_match:
            params['period_days'] = int(period_match.group(1))

        # NEW: Detect query types for new modules
        # Backtest
        if any(w in q for w in ['backtest', 'simula', 'estrategia', 'strategy', 'sma crossover']):
            params['wants_backtest'] = True
            # Detect strategy
            if 'sma' in q and ('50' in q and '200' in q):
                params['strategy'] = 'sma_50_200'
            elif 'sma' in q:
                params['strategy'] = 'sma_20_50'
            elif 'rsi' in q:
                params['strategy'] = 'rsi_oversold'
            elif 'macd' in q:
                params['strategy'] = 'macd_signal'
            elif 'bollinger' in q:
                params['strategy'] = 'bollinger_breakout'
            elif 'combin' in q:
                params['strategy'] = 'combined'
            else:
                params['strategy'] = 'sma_20_50'

        # Prediction
        if any(w in q for w in ['predice', 'prediccion', 'forecast', 'pronostico', 'predecir']):
            params['wants_prediction'] = True
            if 'tendencia' in q or 'trend' in q:
                params['prediction_type'] = 'trend'
            else:
                params['prediction_type'] = 'price'
            # Detect model
            if 'prophet' in q:
                params['model'] = 'prophet'
            elif 'arima' in q:
                params['model'] = 'arima'
            elif 'xgboost' in q:
                params['model'] = 'xgboost'
            elif 'lightgbm' in q:
                params['model'] = 'lightgbm'
            # Detect days
            days_match = re.search(r'(\d+)\s*(dias|days)', q)
            if days_match:
                params['prediction_days'] = int(days_match.group(1))

        # Risk analysis
        if any(w in q for w in ['riesgo', 'risk', 'var', 'sharpe', 'sortino', 'drawdown', 'volatil']):
            params['wants_risk'] = True

        # Correlation
        if any(w in q for w in ['correlacion', 'correlation', 'diversific']):
            params['wants_correlation'] = True

        # Portfolio optimization
        if any(w in q for w in ['optimiza', 'optim', 'peso', 'weight', 'asignacion', 'allocation']):
            params['wants_optimization'] = True

        # Stress test
        if any(w in q for w in ['stress', 'escenario', 'crash', 'crisis']):
            params['wants_stress_test'] = True

        # Data validation
        if any(w in q for w in ['valida', 'verifica', 'calidad', 'outlier', 'faltante', 'missing']):
            params['wants_validation'] = True

        # External data
        if any(w in q for w in ['fed', 'inflacion', 'desempleo', 'gdp', 'treasury', 'economico', 'economic']):
            params['wants_economic'] = True

        if any(w in q for w in ['noticia', 'news', 'evento', 'event']):
            params['wants_news'] = True

        if any(w in q for w in ['vix', 'miedo', 'fear', 'greed']):
            params['wants_vix'] = True

        return params

    def _run_analysis(self, query: str) -> str:
        """Run database analysis based on query"""
        if not self.db:
            return "Error: Database analyzer not available"

        context_parts = []
        q = query.lower()
        params = self._parse_query(query)

        # SCREENING by RSI and/or market cap
        if 'rsi' in q and ('rsi_below' in params or 'rsi_above' in params):
            min_cap = params.get('min_market_cap_b', 1)
            rsi_below = params.get('rsi_below')
            rsi_above = params.get('rsi_above')

            context_parts.append(f"\n=== SCREENING DESDE BASE DE DATOS LOCAL ===")
            context_parts.append(f"Criterios: RSI {'<' + str(rsi_below) if rsi_below else '>' + str(rsi_above)}, Market Cap > ${min_cap}B")

            results = self.db.screen_stocks(
                min_market_cap_b=min_cap,
                rsi_below=rsi_below,
                rsi_above=rsi_above,
                sector=params.get('sector'),
                limit=50
            )

            context_parts.append(f"Acciones encontradas: {len(results)}")
            if results:
                for r in results:
                    context_parts.append(
                        f"  {r['symbol']}: RSI={r['rsi']}, Cap=${r['market_cap_b']}B, "
                        f"PE={r.get('pe_ratio', 'N/A')}, Sector={r.get('sector', 'N/A')}"
                    )
            else:
                context_parts.append("  (Ninguna accion cumple los criterios)")

        # TECHNICAL ANALYSIS for specific symbol
        if params.get('symbol') and any(w in q for w in ['tecnico', 'technical', 'indicador', 'rsi', 'macd']) and not params.get('wants_backtest'):
            symbol = params['symbol']
            context_parts.append(f"\n=== ANALISIS TECNICO {symbol} ===")

            tech = self.db.get_technical_indicators(symbol)
            if tech and 'error' not in tech:
                for k, v in tech.items():
                    if v is not None:
                        context_parts.append(f"  {k}: {v}")
            else:
                context_parts.append(f"  No hay datos para {symbol}")

        # FUNDAMENTALS for specific symbol
        if params.get('symbol') and any(w in q for w in ['fundamental', 'pe', 'market cap', 'empresa', 'compania']):
            symbol = params['symbol']
            context_parts.append(f"\n=== FUNDAMENTALES {symbol} ===")

            info = self.db.get_symbol_info(symbol)
            if info:
                context_parts.append(f"  Nombre: {info.get('name')}")
                context_parts.append(f"  Sector: {info.get('sector')}")
                context_parts.append(f"  Industry: {info.get('industry')}")
                context_parts.append(f"  Market Cap: ${info.get('market_cap_b')}B")
                context_parts.append(f"  PE Ratio: {info.get('pe_ratio')}")
                context_parts.append(f"  Forward PE: {info.get('forward_pe')}")
                context_parts.append(f"  Dividend Yield: {info.get('dividend_yield_pct')}%")
                context_parts.append(f"  Profit Margin: {info.get('profit_margin_pct')}%")
                context_parts.append(f"  Revenue Growth: {info.get('revenue_growth_pct')}%")
            else:
                context_parts.append(f"  No hay datos para {symbol}")

        # PORTFOLIO queries
        if any(w in q for w in ['portfolio', 'cartera', 'posicion', 'cuenta', 'holding']):
            summary = self.db.get_portfolio_summary()
            context_parts.append(f"\n=== PORTFOLIO ({summary['fecha']}) ===")
            context_parts.append(f"Total: EUR {summary['total_portfolio_eur']:,.2f}")
            context_parts.append(f"Holdings: EUR {summary['total_holdings_eur']:,.2f}")
            context_parts.append(f"Cash: EUR {summary['total_cash_eur']:,.2f}")
            context_parts.append("\nPor cuenta:")
            for acc in summary['accounts']:
                context_parts.append(
                    f"  {acc['account']}: Holdings EUR {acc['holdings_eur']:,.2f}, "
                    f"Cash EUR {acc['cash_eur']:,.2f}, Total EUR {acc['total_eur']:,.2f}"
                )

        # CASH positions
        if any(w in q for w in ['cash', 'efectivo', 'liquidez', 'dinero']):
            cash = self.db.get_cash_positions()
            context_parts.append(f"\n=== EFECTIVO ({cash['fecha']}) ===")
            for c in cash['cash']:
                context_parts.append(f"  {c['account']}: {c['balance']:,.2f} {c['currency']}")

        # MARKET STATS
        if any(w in q for w in ['mercado', 'estadisticas', 'cuantos', 'total acciones']):
            stats = self.db.get_market_stats()
            context_parts.append(f"\n=== ESTADISTICAS DEL MERCADO ===")
            context_parts.append(f"Total simbolos: {stats['total_symbols']}")
            context_parts.append(f"Con fundamentales: {stats['symbols_with_fundamentals']}")
            context_parts.append(f"Con precios: {stats['symbols_with_prices']}")

        # PERFORMANCE SCREENING (top gainers/losers)
        if params.get('wants_performance'):
            min_cap = params.get('min_market_cap_b', 5)
            year = params.get('year')
            period_days = params.get('period_days')
            perf_type = params.get('performance_type', 'all')

            period_label = f"Ano {year}" if year else f"Ultimos {period_days or 365} dias"
            context_parts.append(f"\n=== RENDIMIENTO DE ACCIONES ({period_label}) ===")
            context_parts.append(f"Market Cap minimo: ${min_cap}B")

            try:
                if perf_type == 'gainers':
                    results = self.db.get_top_gainers(year=year, period_days=period_days, min_market_cap_b=min_cap, limit=30)
                    context_parts.append(f"\nTOP GANADORAS (que mas subieron):")
                elif perf_type == 'losers':
                    results = self.db.get_top_losers(year=year, period_days=period_days, min_market_cap_b=min_cap, limit=30)
                    context_parts.append(f"\nTOP PERDEDORAS (que mas bajaron):")
                else:
                    results = self.db.screen_by_performance(
                        min_market_cap_b=min_cap,
                        year=year,
                        period_days=period_days,
                        sort_by='return_desc',
                        limit=30
                    )
                    context_parts.append(f"\nRENDIMIENTO (ordenado por retorno):")

                if results:
                    context_parts.append(f"Encontradas: {len(results)} acciones\n")
                    context_parts.append(f"{'Symbol':<8} {'Nombre':<30} {'Sector':<20} {'Cap($B)':<10} {'Return':<10}")
                    context_parts.append("-" * 80)
                    for r in results:
                        name = (r['name'] or '')[:28]
                        sector = (r['sector'] or 'N/A')[:18]
                        ret_str = f"{r['return_pct']:+.1f}%"
                        context_parts.append(
                            f"{r['symbol']:<8} {name:<30} {sector:<20} {r['market_cap_b']:<10.1f} {ret_str:<10}"
                        )
                else:
                    context_parts.append("  No se encontraron resultados con los criterios especificados")
            except Exception as e:
                context_parts.append(f"  Error: {str(e)}")

        # =====================================================================
        # NEW MODULES INTEGRATION
        # =====================================================================

        # BACKTESTING
        if params.get('wants_backtest') and self.backtest and params.get('symbol'):
            symbol = params['symbol']
            strategy = params.get('strategy', 'sma_20_50')
            try:
                result = self.backtest.run_strategy(symbol, strategy)
                context_parts.append(f"\n{result.summary()}")
            except Exception as e:
                context_parts.append(f"\n=== BACKTEST ERROR ===\n{str(e)}")

        # Compare strategies
        if 'comparar' in q or 'compare' in q or 'todas las estrategias' in q:
            if self.backtest and params.get('symbol'):
                symbol = params['symbol']
                try:
                    comparison = self.backtest.compare_strategies(symbol)
                    context_parts.append(f"\n{self.backtest.comparison_summary(comparison)}")
                except Exception as e:
                    context_parts.append(f"\n=== COMPARISON ERROR ===\n{str(e)}")

        # PREDICTIONS
        if params.get('wants_prediction') and self.ml and params.get('symbol'):
            symbol = params['symbol']
            pred_type = params.get('prediction_type', 'price')
            days = params.get('prediction_days', 30)

            try:
                if pred_type == 'trend':
                    model = params.get('model', 'xgboost')
                    if model not in ['xgboost', 'lightgbm']:
                        model = 'xgboost'
                    result = self.ml.predict_trend(symbol, model=model)
                else:
                    model = params.get('model', 'linear')  # Use linear as default (faster, no deps)
                    if model not in ['prophet', 'arima', 'linear']:
                        model = 'linear'
                    result = self.ml.predict_price(symbol, days=days, model=model)

                context_parts.append(f"\n{result.summary()}")
            except Exception as e:
                context_parts.append(f"\n=== PREDICTION ERROR ===\n{str(e)}")

        # RISK ANALYSIS
        if params.get('wants_risk') and self.risk and params.get('symbol'):
            symbol = params['symbol']
            try:
                metrics = self.risk.get_risk_metrics(symbol)
                context_parts.append(f"\n{metrics.summary()}")
            except Exception as e:
                context_parts.append(f"\n=== RISK ERROR ===\n{str(e)}")

        # CORRELATION
        if params.get('wants_correlation') and self.risk:
            # Extract multiple symbols from query
            symbols = re.findall(r'\b([A-Z]{1,5})\b', query)
            symbols = [s for s in symbols if s not in ['RSI', 'MACD', 'PE', 'ETF', 'USD', 'EUR', 'VAR']]
            if len(symbols) >= 2:
                try:
                    summary = self.risk.correlation_summary(symbols)
                    context_parts.append(f"\n{summary}")
                except Exception as e:
                    context_parts.append(f"\n=== CORRELATION ERROR ===\n{str(e)}")

        # PORTFOLIO OPTIMIZATION
        if params.get('wants_optimization') and self.risk:
            symbols = re.findall(r'\b([A-Z]{1,5})\b', query)
            symbols = [s for s in symbols if s not in ['RSI', 'MACD', 'PE', 'ETF', 'USD', 'EUR', 'VAR']]
            if len(symbols) >= 2:
                try:
                    result = self.risk.optimize_portfolio(symbols)
                    context_parts.append(f"\n{result.summary()}")
                except Exception as e:
                    context_parts.append(f"\n=== OPTIMIZATION ERROR ===\n{str(e)}")

        # STRESS TEST
        if params.get('wants_stress_test') and self.risk and params.get('symbol'):
            symbol = params['symbol']
            try:
                summary = self.risk.stress_test_summary(symbol)
                context_parts.append(f"\n{summary}")
            except Exception as e:
                context_parts.append(f"\n=== STRESS TEST ERROR ===\n{str(e)}")

        # DATA VALIDATION
        if params.get('wants_validation') and self.validator and params.get('symbol'):
            symbol = params['symbol']
            try:
                report = self.validator.check_data_quality(symbol)
                context_parts.append(f"\n{report.summary()}")
            except Exception as e:
                context_parts.append(f"\n=== VALIDATION ERROR ===\n{str(e)}")

        # ECONOMIC DATA
        if params.get('wants_economic') and self.external:
            try:
                summary = self.external.economic_summary()
                context_parts.append(f"\n{summary}")
            except Exception as e:
                context_parts.append(f"\n=== ECONOMIC DATA ERROR ===\n{str(e)}")

        # VIX / FEAR
        if params.get('wants_vix') and self.external:
            try:
                vix = self.external.get_vix_status()
                if vix:
                    context_parts.append(f"\n=== VIX / FEAR INDEX ===")
                    context_parts.append(f"VIX: {vix['vix']} - {vix['status']}")
                    context_parts.append(f"Promedio 30d: {vix['avg_30d']}")
                    context_parts.append(f"Fecha: {vix['date']}")
            except Exception as e:
                context_parts.append(f"\n=== VIX ERROR ===\n{str(e)}")

        # NEWS / EVENTS
        if params.get('wants_news') and self.external:
            try:
                # Check for historical events
                if any(w in q for w in ['historico', 'historical', 'crash', 'crisis', '2008', '2020']):
                    event_type = 'crash' if 'crash' in q or 'crisis' in q else None
                    summary = self.external.events_summary(event_type)
                    context_parts.append(f"\n{summary}")
                else:
                    # Recent news
                    symbol = params.get('symbol')
                    summary = self.external.news_summary(symbol=symbol)
                    context_parts.append(f"\n{summary}")
            except Exception as e:
                context_parts.append(f"\n=== NEWS ERROR ===\n{str(e)}")

        return "\n".join(context_parts)

    def ask(self, query: str) -> str:
        """Process a user query"""
        if not self.active_backend:
            return "Error: No hay backends de IA disponibles. Configura una API key en .env"

        try:
            # Run database analysis
            analysis = self._run_analysis(query)

            # Build prompt
            full_prompt = f"""DATOS DE ANALISIS:
{analysis if analysis else "No se requirio analisis especifico."}

PREGUNTA DEL USUARIO:
{query}

Responde de forma clara y concisa basandote en los datos proporcionados."""

            # Generate response
            response = self.active_backend.generate(full_prompt, self.SYSTEM_PROMPT)
            return response

        except Exception as e:
            return f"Error al procesar la consulta: {str(e)}"

    def get_quick_summary(self) -> str:
        """Get quick portfolio summary"""
        if not self.db:
            return "Database not available"

        summary = self.db.get_portfolio_summary()
        lines = [
            f"Portfolio ({summary['fecha']})",
            "",
        ]
        for acc in summary['accounts']:
            lines.append(f"  {acc['account']}: EUR {acc['total_eur']:,.2f}")
        lines.append("")
        lines.append(f"Holdings: EUR {summary['total_holdings_eur']:,.2f}")
        lines.append(f"Cash: EUR {summary['total_cash_eur']:,.2f}")
        lines.append(f"TOTAL: EUR {summary['total_portfolio_eur']:,.2f}")

        return "\n".join(lines)

    def switch_backend(self, backend_name: str) -> bool:
        """Switch AI backend"""
        if backend_name in self.backends and self.backends[backend_name].available:
            self.active_backend = self.backends[backend_name]
            return True
        return False

    def list_backends(self) -> Dict[str, bool]:
        """List available backends"""
        return {name: backend.available for name, backend in self.backends.items()}

    # =========================================================================
    # NEW MODULE SHORTCUTS
    # =========================================================================

    def run_backtest(self, symbol: str, strategy: str = 'sma_20_50') -> str:
        """Run backtest for a symbol"""
        if not self.backtest:
            return "Backtest module not available"
        try:
            result = self.backtest.run_strategy(symbol, strategy)
            return result.summary()
        except Exception as e:
            return f"Error: {str(e)}"

    def predict_price(self, symbol: str, days: int = 30, model: str = 'linear') -> str:
        """Predict future prices"""
        if not self.ml:
            return "ML module not available"
        try:
            result = self.ml.predict_price(symbol, days, model)
            return result.summary()
        except Exception as e:
            return f"Error: {str(e)}"

    def predict_trend(self, symbol: str, model: str = 'xgboost') -> str:
        """Predict price trend"""
        if not self.ml:
            return "ML module not available"
        try:
            result = self.ml.predict_trend(symbol, model)
            return result.summary()
        except Exception as e:
            return f"Error: {str(e)}"

    def get_risk_analysis(self, symbol: str) -> str:
        """Get risk metrics for a symbol"""
        if not self.risk:
            return "Risk module not available"
        try:
            result = self.risk.get_risk_metrics(symbol)
            return result.summary()
        except Exception as e:
            return f"Error: {str(e)}"

    def optimize_portfolio(self, symbols: List[str]) -> str:
        """Optimize portfolio allocation"""
        if not self.risk:
            return "Risk module not available"
        try:
            result = self.risk.optimize_portfolio(symbols)
            return result.summary()
        except Exception as e:
            return f"Error: {str(e)}"

    def check_data_quality(self, symbol: str) -> str:
        """Check data quality for a symbol"""
        if not self.validator:
            return "Validator module not available"
        try:
            result = self.validator.check_data_quality(symbol)
            return result.summary()
        except Exception as e:
            return f"Error: {str(e)}"

    def get_economic_data(self) -> str:
        """Get economic indicators"""
        if not self.external:
            return "External data module not available"
        try:
            return self.external.economic_summary()
        except Exception as e:
            return f"Error: {str(e)}"

    def get_market_events(self, event_type: str = None) -> str:
        """Get historical market events"""
        if not self.external:
            return "External data module not available"
        try:
            return self.external.events_summary(event_type)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_available_capabilities(self) -> Dict[str, bool]:
        """List available capabilities"""
        return {
            'database': self.db is not None,
            'backtest': self.backtest is not None,
            'ml_predictor': self.ml is not None,
            'risk_analyzer': self.risk is not None,
            'data_validator': self.validator is not None,
            'external_data': self.external is not None,
            'news_manager': self.news is not None,
            'ai_backend': self.active_backend is not None
        }

    def get_news(self, symbol: str = None, category: str = None, days: int = 30) -> str:
        """Get news from database"""
        if not self.news:
            return "News module not available"
        try:
            return self.news.get_news_summary(symbol=symbol, days=days)
        except Exception as e:
            return f"Error: {str(e)}"

    def search_news(self, query: str) -> str:
        """Search news by keyword"""
        if not self.news:
            return "News module not available"
        try:
            df = self.news.search_news(query, limit=20)
            if df.empty:
                return f"No se encontraron noticias para: {query}"

            lines = [f"=== BUSQUEDA: {query} ===", f"Resultados: {len(df)}\n"]
            for _, row in df.iterrows():
                title = str(row['title'])[:65]
                source = row['source'] or 'N/A'
                lines.append(f"[{source}] {title}...")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_news_stats(self) -> str:
        """Get news database statistics"""
        if not self.news:
            return "News module not available"
        try:
            return self.news.stats_summary()
        except Exception as e:
            return f"Error: {str(e)}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== Financial AI Assistant ===\n")

    assistant = FinancialAssistant()
    print(f"Backend: {assistant.active_backend.name if assistant.active_backend else 'None'}")

    print("\nCapabilities:")
    for cap, available in assistant.get_available_capabilities().items():
        status = "OK" if available else "Not available"
        print(f"  {cap}: {status}")

    print("\n" + assistant.get_quick_summary())

    print("\n--- Test: RSI Screening ---")
    response = assistant.ask("cuantas acciones hay con RSI bajo 20 y market cap mayor a 9.87 billion")
    print(response[:500] + "..." if len(response) > 500 else response)

    print("\n--- Test: Backtest ---")
    print(assistant.run_backtest("AAPL", "sma_20_50"))

    print("\n--- Test: Risk Analysis ---")
    print(assistant.get_risk_analysis("MSFT"))

    print("\n--- Test: Market Events ---")
    print(assistant.get_market_events("crash"))
