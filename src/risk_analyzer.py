"""
Risk Analyzer Module for Financial AI Assistant
Integrates: quantstats, scipy

Provides risk metrics, portfolio optimization, and stress testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    symbol: str
    period_days: int
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration_days: int
    beta: float = None
    alpha: float = None
    correlation_market: float = None

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'period_days': self.period_days,
            'var_95': f"{self.var_95:.2f}%",
            'var_99': f"{self.var_99:.2f}%",
            'cvar_95': f"{self.cvar_95:.2f}%",
            'cvar_99': f"{self.cvar_99:.2f}%",
            'volatility_annual': f"{self.volatility_annual:.2f}%",
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': f"{self.max_drawdown:.2f}%",
            'avg_drawdown': f"{self.avg_drawdown:.2f}%",
            'drawdown_duration_days': self.drawdown_duration_days,
            'beta': round(self.beta, 2) if self.beta else None,
            'alpha': f"{self.alpha:.2f}%" if self.alpha else None
        }

    def summary(self) -> str:
        """Text summary for AI assistant"""
        lines = [
            f"=== ANALISIS DE RIESGO: {self.symbol} ===",
            f"Periodo: {self.period_days} dias",
            "",
            "VALUE AT RISK (VaR):",
            f"  VaR 95%: {self.var_95:.2f}% (perdida maxima 1 de cada 20 dias)",
            f"  VaR 99%: {self.var_99:.2f}% (perdida maxima 1 de cada 100 dias)",
            f"  CVaR 95%: {self.cvar_95:.2f}% (expected shortfall)",
            f"  CVaR 99%: {self.cvar_99:.2f}%",
            "",
            "VOLATILIDAD:",
            f"  Volatilidad anualizada: {self.volatility_annual:.2f}%",
            "",
            "RATIOS AJUSTADOS POR RIESGO:",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {self.sortino_ratio:.2f}",
            "",
            "DRAWDOWN:",
            f"  Max Drawdown: {self.max_drawdown:.2f}%",
            f"  Avg Drawdown: {self.avg_drawdown:.2f}%",
            f"  Mayor duracion: {self.drawdown_duration_days} dias"
        ]

        if self.beta is not None:
            lines.extend([
                "",
                "VS MERCADO (SPY):",
                f"  Beta: {self.beta:.2f}",
                f"  Alpha: {self.alpha:.2f}%" if self.alpha else ""
            ])

        return "\n".join(lines)


@dataclass
class PortfolioOptimization:
    """Container for portfolio optimization results"""
    symbols: List[str]
    target: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float

    def to_dict(self) -> Dict:
        return {
            'symbols': self.symbols,
            'optimization_target': self.target,
            'weights': {k: f"{v:.1%}" for k, v in self.weights.items()},
            'expected_return': f"{self.expected_return:.2f}%",
            'volatility': f"{self.volatility:.2f}%",
            'sharpe_ratio': round(self.sharpe_ratio, 2)
        }

    def summary(self) -> str:
        """Text summary for AI assistant"""
        lines = [
            f"=== OPTIMIZACION DE PORTFOLIO ===",
            f"Objetivo: {self.target}",
            f"Activos: {', '.join(self.symbols)}",
            "",
            "PESOS OPTIMOS:"
        ]

        for symbol, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            if weight > 0.001:  # Only show meaningful allocations
                lines.append(f"  {symbol}: {weight:.1%}")

        lines.extend([
            "",
            "METRICAS ESPERADAS:",
            f"  Retorno esperado: {self.expected_return:.2f}%",
            f"  Volatilidad: {self.volatility:.2f}%",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}"
        ])

        return "\n".join(lines)


class RiskAnalyzer:
    """
    Risk analysis and portfolio optimization.
    Uses data from local database via DatabaseAnalyzer.
    """

    def __init__(self, db_path: str = "data/financial_data.db"):
        self.db_path = db_path
        self._db = None
        self._qs = None

    @property
    def db(self):
        """Lazy load database analyzer"""
        if self._db is None:
            from src.db_analysis_tools import DatabaseAnalyzer
            self._db = DatabaseAnalyzer(self.db_path)
        return self._db

    @property
    def qs(self):
        """Lazy load quantstats"""
        if self._qs is None:
            try:
                import quantstats as qs
                self._qs = qs
            except ImportError:
                self._qs = None
        return self._qs

    def get_returns(self, symbol: str, days: int = 252) -> pd.Series:
        """Get daily returns for a symbol"""
        df = self.db.get_price_history(symbol, days=days)
        if df.empty:
            raise ValueError(f"No price data for {symbol}")

        returns = df.set_index('date')['close'].pct_change().dropna()
        returns.name = symbol
        return returns

    def get_prices(self, symbol: str, days: int = 252) -> pd.Series:
        """Get prices for a symbol"""
        df = self.db.get_price_history(symbol, days=days)
        if df.empty:
            raise ValueError(f"No price data for {symbol}")

        prices = df.set_index('date')['close']
        prices.name = symbol
        return prices

    # =========================================================================
    # VAR / CVAR CALCULATIONS
    # =========================================================================

    def calculate_var(self, symbol: str, confidence: float = 0.95,
                      days: int = 252, method: str = 'historical') -> float:
        """
        Calculate Value at Risk.

        Args:
            symbol: Stock symbol
            confidence: Confidence level (0.95 = 95%)
            days: Days of data to use
            method: 'historical' or 'parametric'

        Returns:
            VaR as percentage (negative value)
        """
        returns = self.get_returns(symbol, days)

        if method == 'historical':
            var = np.percentile(returns, (1 - confidence) * 100)
        else:  # parametric
            mean = returns.mean()
            std = returns.std()
            from scipy import stats
            z = stats.norm.ppf(1 - confidence)
            var = mean + z * std

        return var * 100

    def calculate_cvar(self, symbol: str, confidence: float = 0.95,
                       days: int = 252) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            symbol: Stock symbol
            confidence: Confidence level
            days: Days of data

        Returns:
            CVaR as percentage
        """
        returns = self.get_returns(symbol, days)
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        return cvar * 100

    # =========================================================================
    # DRAWDOWN ANALYSIS
    # =========================================================================

    def calculate_drawdown(self, symbol: str, days: int = 252) -> Dict:
        """
        Calculate drawdown statistics.

        Returns:
            Dict with max_drawdown, avg_drawdown, current_drawdown, max_duration
        """
        prices = self.get_prices(symbol, days)
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max * 100

        # Calculate drawdown durations
        in_drawdown = drawdown < 0
        dd_start = None
        max_duration = 0
        current_duration = 0

        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd:
                if dd_start is None:
                    dd_start = i
                current_duration = i - dd_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                dd_start = None
                current_duration = 0

        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'current_drawdown': drawdown.iloc[-1],
            'max_duration_days': max_duration,
            'drawdown_series': drawdown
        }

    # =========================================================================
    # SHARPE / SORTINO
    # =========================================================================

    def calculate_sharpe(self, symbol: str, risk_free_rate: float = 0.0,
                         days: int = 252) -> float:
        """Calculate annualized Sharpe ratio"""
        returns = self.get_returns(symbol, days)
        excess = returns - (risk_free_rate / 252)
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess.mean() / returns.std()

    def calculate_sortino(self, symbol: str, risk_free_rate: float = 0.0,
                          days: int = 252) -> float:
        """Calculate annualized Sortino ratio"""
        returns = self.get_returns(symbol, days)
        excess = returns - (risk_free_rate / 252)
        downside = returns[returns < 0]

        if len(downside) == 0 or downside.std() == 0:
            return 0

        return np.sqrt(252) * excess.mean() / downside.std()

    # =========================================================================
    # FULL RISK METRICS
    # =========================================================================

    def get_risk_metrics(self, symbol: str, days: int = 252,
                         benchmark: str = 'SPY') -> RiskMetrics:
        """
        Get comprehensive risk metrics for a symbol.

        Args:
            symbol: Stock symbol
            days: Days of data
            benchmark: Benchmark symbol for beta/alpha

        Returns:
            RiskMetrics object
        """
        returns = self.get_returns(symbol, days)
        dd = self.calculate_drawdown(symbol, days)

        # VaR/CVaR
        var_95 = self.calculate_var(symbol, 0.95, days)
        var_99 = self.calculate_var(symbol, 0.99, days)
        cvar_95 = self.calculate_cvar(symbol, 0.95, days)
        cvar_99 = self.calculate_cvar(symbol, 0.99, days)

        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100

        # Risk-adjusted
        sharpe = self.calculate_sharpe(symbol, days=days)
        sortino = self.calculate_sortino(symbol, days=days)

        # Beta/Alpha vs benchmark
        beta = None
        alpha = None
        correlation = None

        if benchmark and benchmark.upper() != symbol.upper():
            try:
                bench_returns = self.get_returns(benchmark, days)

                # Align dates
                aligned = pd.concat([returns, bench_returns], axis=1).dropna()
                if len(aligned) > 20:
                    stock_ret = aligned.iloc[:, 0]
                    bench_ret = aligned.iloc[:, 1]

                    # Beta = Cov(stock, market) / Var(market)
                    covariance = np.cov(stock_ret, bench_ret)[0][1]
                    variance = np.var(bench_ret)
                    beta = covariance / variance if variance > 0 else 1

                    # Alpha = stock_return - beta * market_return (annualized)
                    alpha = (stock_ret.mean() - beta * bench_ret.mean()) * 252 * 100

                    # Correlation
                    correlation = np.corrcoef(stock_ret, bench_ret)[0][1]
            except:
                pass

        return RiskMetrics(
            symbol=symbol,
            period_days=len(returns),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility_annual=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=dd['max_drawdown'],
            avg_drawdown=dd['avg_drawdown'],
            drawdown_duration_days=dd['max_duration_days'],
            beta=beta,
            alpha=alpha,
            correlation_market=correlation
        )

    # =========================================================================
    # CORRELATION MATRIX
    # =========================================================================

    def correlation_matrix(self, symbols: List[str], days: int = 252) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple symbols.

        Args:
            symbols: List of stock symbols
            days: Days of data

        Returns:
            Correlation matrix DataFrame
        """
        returns_dict = {}
        for symbol in symbols:
            try:
                returns_dict[symbol] = self.get_returns(symbol, days)
            except:
                continue

        if not returns_dict:
            raise ValueError("No valid data for any symbol")

        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()

    def correlation_summary(self, symbols: List[str], days: int = 252) -> str:
        """Generate text summary of correlations"""
        corr = self.correlation_matrix(symbols, days)

        lines = [
            f"=== MATRIZ DE CORRELACION ({len(corr)} activos) ===",
            f"Periodo: {days} dias\n"
        ]

        # Header
        header = "         " + "  ".join(f"{s:>7}" for s in corr.columns)
        lines.append(header)
        lines.append("-" * len(header))

        # Matrix
        for symbol in corr.index:
            row = f"{symbol:>8} "
            for other in corr.columns:
                val = corr.loc[symbol, other]
                row += f"{val:>7.2f}  "
            lines.append(row)

        # Find high correlations
        lines.append("\nCORRELACIONES ALTAS (>0.7):")
        found = False
        for i, s1 in enumerate(corr.columns):
            for s2 in corr.columns[i+1:]:
                val = corr.loc[s1, s2]
                if abs(val) > 0.7:
                    lines.append(f"  {s1} - {s2}: {val:.2f}")
                    found = True

        if not found:
            lines.append("  (ninguna)")

        # Find low correlations (good for diversification)
        lines.append("\nCORRELACIONES BAJAS (<0.3) - buenos para diversificacion:")
        found = False
        for i, s1 in enumerate(corr.columns):
            for s2 in corr.columns[i+1:]:
                val = corr.loc[s1, s2]
                if abs(val) < 0.3:
                    lines.append(f"  {s1} - {s2}: {val:.2f}")
                    found = True

        if not found:
            lines.append("  (ninguna)")

        return "\n".join(lines)

    # =========================================================================
    # PORTFOLIO OPTIMIZATION
    # =========================================================================

    def optimize_portfolio(self, symbols: List[str], days: int = 252,
                          target: str = 'sharpe',
                          risk_free_rate: float = 0.0) -> PortfolioOptimization:
        """
        Optimize portfolio weights.

        Args:
            symbols: List of stock symbols
            days: Days of data
            target: 'sharpe' (max Sharpe), 'min_volatility', 'max_return'
            risk_free_rate: Annual risk-free rate

        Returns:
            PortfolioOptimization object
        """
        from scipy.optimize import minimize

        # Get returns for all symbols
        returns_dict = {}
        for symbol in symbols:
            try:
                returns_dict[symbol] = self.get_returns(symbol, days)
            except:
                continue

        if len(returns_dict) < 2:
            raise ValueError("Need at least 2 symbols with valid data")

        returns_df = pd.DataFrame(returns_dict).dropna()
        n_assets = len(returns_df.columns)

        # Expected returns and covariance
        mean_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized

        def portfolio_return(weights):
            return np.sum(weights * mean_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def portfolio_sharpe(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - risk_free_rate) / vol if vol > 0 else 0

        def neg_return(weights):
            return -portfolio_return(weights)

        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)

        # Optimize based on target
        if target == 'sharpe':
            result = minimize(portfolio_sharpe, init_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        elif target == 'min_volatility':
            result = minimize(portfolio_volatility, init_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        elif target == 'max_return':
            result = minimize(neg_return, init_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            raise ValueError(f"Unknown target: {target}")

        optimal_weights = result.x
        weights_dict = {symbol: weight for symbol, weight in
                       zip(returns_df.columns, optimal_weights)}

        return PortfolioOptimization(
            symbols=list(returns_df.columns),
            target=target,
            weights=weights_dict,
            expected_return=portfolio_return(optimal_weights) * 100,
            volatility=portfolio_volatility(optimal_weights) * 100,
            sharpe_ratio=(portfolio_return(optimal_weights) - risk_free_rate) /
                        portfolio_volatility(optimal_weights) if portfolio_volatility(optimal_weights) > 0 else 0
        )

    # =========================================================================
    # STRESS TESTING
    # =========================================================================

    def stress_test(self, symbol: str, scenarios: Dict[str, float] = None,
                    position_value: float = 10000) -> Dict:
        """
        Run stress test on a position.

        Args:
            symbol: Stock symbol
            scenarios: Dict of scenario_name -> price_change_pct
            position_value: Current position value

        Returns:
            Dict of scenario results
        """
        if scenarios is None:
            scenarios = {
                'Flash Crash (-10%)': -10,
                'Correccion (-20%)': -20,
                'Bear Market (-30%)': -30,
                'Crisis (-40%)': -40,
                'Black Swan (-50%)': -50,
                'Rally (+20%)': 20,
                'Bull Market (+50%)': 50
            }

        results = {
            'symbol': symbol,
            'position_value': position_value,
            'scenarios': {}
        }

        for name, change_pct in scenarios.items():
            new_value = position_value * (1 + change_pct / 100)
            pnl = new_value - position_value
            results['scenarios'][name] = {
                'price_change': f"{change_pct:+.1f}%",
                'new_value': round(new_value, 2),
                'pnl': round(pnl, 2)
            }

        return results

    def stress_test_summary(self, symbol: str, position_value: float = 10000) -> str:
        """Generate text summary of stress test"""
        results = self.stress_test(symbol, position_value=position_value)

        lines = [
            f"=== STRESS TEST: {symbol} ===",
            f"Valor de posicion: ${position_value:,.2f}\n",
            f"{'Escenario':<25} {'Cambio':>10} {'Nuevo Valor':>15} {'P&L':>12}",
            "-" * 65
        ]

        for name, data in results['scenarios'].items():
            lines.append(
                f"{name:<25} {data['price_change']:>10} "
                f"${data['new_value']:>14,.2f} ${data['pnl']:>11,.2f}"
            )

        return "\n".join(lines)

    # =========================================================================
    # PORTFOLIO RISK
    # =========================================================================

    def portfolio_risk_summary(self, holdings: Dict[str, float], days: int = 252) -> str:
        """
        Calculate risk metrics for a portfolio.

        Args:
            holdings: Dict of symbol -> position_value
            days: Days of data

        Returns:
            Text summary
        """
        total_value = sum(holdings.values())
        weights = {s: v/total_value for s, v in holdings.items()}

        # Get returns for all holdings
        returns_dict = {}
        valid_holdings = {}
        for symbol, value in holdings.items():
            try:
                returns_dict[symbol] = self.get_returns(symbol, days)
                valid_holdings[symbol] = value
            except:
                continue

        if not returns_dict:
            return "Error: No valid data for portfolio holdings"

        # Calculate portfolio returns
        returns_df = pd.DataFrame(returns_dict).dropna()
        weights_array = np.array([weights.get(s, 0) for s in returns_df.columns])
        weights_array = weights_array / weights_array.sum()  # Renormalize

        portfolio_returns = returns_df.dot(weights_array)

        # Calculate metrics
        var_95 = np.percentile(portfolio_returns, 5) * 100
        var_99 = np.percentile(portfolio_returns, 1) * 100
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0

        # Max drawdown
        equity = (1 + portfolio_returns).cumprod()
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        lines = [
            f"=== RIESGO DE PORTFOLIO ===",
            f"Valor total: ${total_value:,.2f}",
            f"Activos: {len(valid_holdings)}",
            f"Periodo: {len(portfolio_returns)} dias\n",
            "COMPOSICION:",
        ]

        for symbol, value in sorted(valid_holdings.items(), key=lambda x: -x[1]):
            pct = value / total_value * 100
            lines.append(f"  {symbol}: ${value:,.2f} ({pct:.1f}%)")

        lines.extend([
            "",
            "METRICAS DE RIESGO:",
            f"  VaR 95%: {var_95:.2f}% (${total_value * abs(var_95)/100:,.2f})",
            f"  VaR 99%: {var_99:.2f}% (${total_value * abs(var_99)/100:,.2f})",
            f"  CVaR 95%: {cvar_95:.2f}%",
            f"  Volatilidad anual: {volatility:.2f}%",
            f"  Sharpe Ratio: {sharpe:.2f}",
            f"  Max Drawdown: {max_dd:.2f}%"
        ])

        # Concentration risk
        max_position_pct = max(v/total_value*100 for v in valid_holdings.values())
        lines.extend([
            "",
            "RIESGO DE CONCENTRACION:",
            f"  Mayor posicion: {max_position_pct:.1f}%"
        ])

        if max_position_pct > 25:
            lines.append("  ADVERTENCIA: Alta concentracion (>25% en un activo)")

        return "\n".join(lines)


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Risk Analyzer Test ===\n")

    analyzer = RiskAnalyzer()

    print("--- Risk Metrics for AAPL ---")
    try:
        metrics = analyzer.get_risk_metrics("AAPL")
        print(metrics.summary())
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Correlation Matrix ---")
    try:
        summary = analyzer.correlation_summary(["AAPL", "MSFT", "GOOGL", "AMZN"])
        print(summary)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Portfolio Optimization ---")
    try:
        opt = analyzer.optimize_portfolio(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
        print(opt.summary())
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Stress Test ---")
    try:
        print(analyzer.stress_test_summary("AAPL", 50000))
    except Exception as e:
        print(f"Error: {e}")
