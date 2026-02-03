"""
Backtest Engine Module for Financial AI Assistant
Integrates: backtesting.py, vectorbt, quantstats

Provides backtesting capabilities with predefined and custom strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """Container for backtest results"""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    profit_factor: float
    avg_trade_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    equity_curve: pd.Series = None
    trades: pd.DataFrame = None

    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy_name,
            'symbol': self.symbol,
            'period': f"{self.start_date} to {self.end_date}",
            'initial_capital': self.initial_capital,
            'final_value': round(self.final_value, 2),
            'total_return': f"{self.total_return_pct:.2f}%",
            'annual_return': f"{self.annual_return_pct:.2f}%",
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': f"{self.max_drawdown_pct:.2f}%",
            'win_rate': f"{self.win_rate_pct:.1f}%",
            'total_trades': self.total_trades,
            'profit_factor': round(self.profit_factor, 2),
            'avg_trade': f"{self.avg_trade_pct:.2f}%",
            'best_trade': f"{self.best_trade_pct:.2f}%",
            'worst_trade': f"{self.worst_trade_pct:.2f}%"
        }

    def summary(self) -> str:
        """Text summary for AI assistant"""
        lines = [
            f"=== BACKTEST: {self.strategy_name} en {self.symbol} ===",
            f"Periodo: {self.start_date} a {self.end_date}",
            f"Capital inicial: ${self.initial_capital:,.2f}",
            f"Valor final: ${self.final_value:,.2f}",
            f"Retorno total: {self.total_return_pct:.2f}%",
            f"Retorno anualizado: {self.annual_return_pct:.2f}%",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Sortino Ratio: {self.sortino_ratio:.2f}",
            f"Max Drawdown: {self.max_drawdown_pct:.2f}%",
            f"Win Rate: {self.win_rate_pct:.1f}%",
            f"Trades: {self.total_trades}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Avg Trade: {self.avg_trade_pct:.2f}%",
            f"Mejor Trade: {self.best_trade_pct:.2f}%",
            f"Peor Trade: {self.worst_trade_pct:.2f}%"
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    Uses data from local database via DatabaseAnalyzer.
    """

    STRATEGIES = {
        'sma_crossover': 'SMA Crossover (default 20/50)',
        'sma_20_50': 'SMA Crossover 20/50',
        'sma_50_200': 'SMA Crossover 50/200',
        'rsi_oversold': 'RSI Oversold/Overbought',
        'macd_signal': 'MACD Signal Crossover',
        'bollinger_breakout': 'Bollinger Bands Breakout',
        'combined': 'Combined (SMA + RSI + MACD)',
        'buy_hold': 'Buy and Hold (benchmark)'
    }

    def __init__(self, db_path: str = "data/financial_data.db"):
        self.db_path = db_path
        self._db = None

    @property
    def db(self):
        """Lazy load database analyzer"""
        if self._db is None:
            from src.db_analysis_tools import DatabaseAnalyzer
            self._db = DatabaseAnalyzer(self.db_path)
        return self._db

    def get_price_data(self, symbol: str, days: int = 500) -> pd.DataFrame:
        """Get OHLCV data for backtesting"""
        df = self.db.get_price_history(symbol, days=days)
        if df.empty:
            raise ValueError(f"No price data for {symbol}")
        df.set_index('date', inplace=True)
        return df

    # =========================================================================
    # STRATEGY SIGNALS
    # =========================================================================

    def _sma_crossover_signals(self, df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
        """SMA crossover strategy: buy when fast crosses above slow"""
        sma_fast = df['close'].rolling(fast).mean()
        sma_slow = df['close'].rolling(slow).mean()
        signal = pd.Series(0, index=df.index)
        signal[sma_fast > sma_slow] = 1
        signal[sma_fast <= sma_slow] = -1
        return signal

    def _rsi_signals(self, df: pd.DataFrame, period: int = 14,
                     oversold: int = 30, overbought: int = 70) -> pd.Series:
        """RSI strategy: buy when oversold, sell when overbought"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(len(df)):
            if rsi.iloc[i] < oversold and not in_position:
                signal.iloc[i] = 1  # Buy signal
                in_position = True
            elif rsi.iloc[i] > overbought and in_position:
                signal.iloc[i] = -1  # Sell signal
                in_position = False
            elif in_position:
                signal.iloc[i] = 1  # Hold

        return signal

    def _macd_signals(self, df: pd.DataFrame) -> pd.Series:
        """MACD strategy: buy when MACD crosses above signal"""
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()

        signal = pd.Series(0, index=df.index)
        signal[macd > signal_line] = 1
        signal[macd <= signal_line] = -1
        return signal

    def _bollinger_signals(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.Series:
        """Bollinger Bands: buy at lower band, sell at upper band"""
        sma = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        signal = pd.Series(0, index=df.index)
        in_position = False

        for i in range(len(df)):
            if df['close'].iloc[i] < lower.iloc[i] and not in_position:
                signal.iloc[i] = 1  # Buy
                in_position = True
            elif df['close'].iloc[i] > upper.iloc[i] and in_position:
                signal.iloc[i] = -1  # Sell
                in_position = False
            elif in_position:
                signal.iloc[i] = 1  # Hold

        return signal

    def _combined_signals(self, df: pd.DataFrame) -> pd.Series:
        """Combined strategy: require agreement of SMA, RSI, and MACD"""
        sma_sig = self._sma_crossover_signals(df, 20, 50)

        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        signal = pd.Series(0, index=df.index)

        # Buy: SMA bullish + RSI not overbought + MACD bullish
        buy_condition = (sma_sig == 1) & (rsi < 70) & (macd > macd_signal)
        # Sell: SMA bearish OR RSI overbought OR MACD bearish
        sell_condition = (sma_sig == -1) | (rsi > 75) | (macd < macd_signal)

        in_position = False
        for i in range(len(df)):
            if buy_condition.iloc[i] and not in_position:
                signal.iloc[i] = 1
                in_position = True
            elif sell_condition.iloc[i] and in_position:
                signal.iloc[i] = -1
                in_position = False
            elif in_position:
                signal.iloc[i] = 1

        return signal

    def _buy_hold_signals(self, df: pd.DataFrame) -> pd.Series:
        """Buy and hold: always in the market"""
        signal = pd.Series(1, index=df.index)
        return signal

    # =========================================================================
    # BACKTEST EXECUTION
    # =========================================================================

    def _calculate_returns(self, df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
        """Calculate strategy returns based on signals"""
        df = df.copy()
        df['signal'] = signal
        df['position'] = df['signal'].shift(1).fillna(0)  # Next day execution
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']
        df['equity'] = (1 + df['strategy_returns']).cumprod()
        return df

    def _calculate_metrics(self, returns: pd.Series, equity: pd.Series,
                           initial_capital: float, days: int) -> Dict:
        """Calculate performance metrics from returns series"""
        # Total and annual return
        final_value = initial_capital * equity.iloc[-1]
        total_return = (final_value / initial_capital - 1) * 100
        years = days / 252
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else total_return

        # Risk metrics
        excess_returns = returns - 0  # Assuming 0 risk-free rate
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else returns.std()
        sortino = np.sqrt(252) * returns.mean() / downside_std if downside_std > 0 else 0

        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd
        }

    def _analyze_trades(self, df: pd.DataFrame) -> Dict:
        """Analyze individual trades"""
        # Find trade entry/exit points
        position = df['position']
        entries = position.diff() == 1
        exits = position.diff() == -1

        if entries.sum() == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        # Calculate trade returns
        trade_returns = []
        entry_price = None

        for i in range(len(df)):
            if entries.iloc[i]:
                entry_price = df['close'].iloc[i]
            elif exits.iloc[i] and entry_price is not None:
                exit_price = df['close'].iloc[i]
                trade_return = (exit_price / entry_price - 1) * 100
                trade_returns.append(trade_return)
                entry_price = None

        if not trade_returns:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        return {
            'total_trades': len(trade_returns),
            'win_rate': len(wins) / len(trade_returns) * 100 if trade_returns else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'avg_trade': np.mean(trade_returns) if trade_returns else 0,
            'best_trade': max(trade_returns) if trade_returns else 0,
            'worst_trade': min(trade_returns) if trade_returns else 0
        }

    def run_strategy(self, symbol: str, strategy: str = 'sma_crossover',
                     start_date: str = None, end_date: str = None,
                     initial_capital: float = 10000,
                     **strategy_params) -> BacktestResult:
        """
        Run a backtest for a specific strategy on a symbol.

        Args:
            symbol: Stock symbol
            strategy: Strategy name (see STRATEGIES)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            **strategy_params: Additional strategy parameters

        Returns:
            BacktestResult with all metrics
        """
        # Get price data
        days = 1000  # Get plenty of data
        df = self.get_price_data(symbol, days)

        # Filter by date if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        if len(df) < 50:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} days")

        # Get strategy signals
        strategy_map = {
            'sma_crossover': lambda: self._sma_crossover_signals(df,
                strategy_params.get('fast', 20), strategy_params.get('slow', 50)),
            'sma_20_50': lambda: self._sma_crossover_signals(df, 20, 50),
            'sma_50_200': lambda: self._sma_crossover_signals(df, 50, 200),
            'rsi_oversold': lambda: self._rsi_signals(df,
                strategy_params.get('period', 14),
                strategy_params.get('oversold', 30),
                strategy_params.get('overbought', 70)),
            'macd_signal': lambda: self._macd_signals(df),
            'bollinger_breakout': lambda: self._bollinger_signals(df,
                strategy_params.get('period', 20),
                strategy_params.get('std', 2)),
            'combined': lambda: self._combined_signals(df),
            'buy_hold': lambda: self._buy_hold_signals(df)
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategy_map.keys())}")

        signals = strategy_map[strategy]()

        # Calculate returns
        results_df = self._calculate_returns(df, signals)
        strategy_returns = results_df['strategy_returns'].dropna()
        equity = results_df['equity'].dropna()

        # Calculate metrics
        metrics = self._calculate_metrics(
            strategy_returns, equity, initial_capital, len(df)
        )

        # Analyze trades
        trades = self._analyze_trades(results_df)

        # Build result
        return BacktestResult(
            strategy_name=self.STRATEGIES.get(strategy, strategy),
            symbol=symbol,
            start_date=df.index[0].strftime('%Y-%m-%d'),
            end_date=df.index[-1].strftime('%Y-%m-%d'),
            initial_capital=initial_capital,
            final_value=metrics['final_value'],
            total_return_pct=metrics['total_return'],
            annual_return_pct=metrics['annual_return'],
            sharpe_ratio=metrics['sharpe'],
            sortino_ratio=metrics['sortino'],
            max_drawdown_pct=metrics['max_drawdown'],
            win_rate_pct=trades['win_rate'],
            total_trades=trades['total_trades'],
            profit_factor=trades['profit_factor'] if trades['profit_factor'] != float('inf') else 99.99,
            avg_trade_pct=trades['avg_trade'],
            best_trade_pct=trades['best_trade'],
            worst_trade_pct=trades['worst_trade'],
            equity_curve=equity * initial_capital,
            trades=results_df
        )

    def compare_strategies(self, symbol: str, strategies: List[str] = None,
                          start_date: str = None, end_date: str = None,
                          initial_capital: float = 10000) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies on the same symbol.

        Args:
            symbol: Stock symbol
            strategies: List of strategy names (default: all)
            start_date: Start date
            end_date: End date
            initial_capital: Starting capital

        Returns:
            Dict of strategy_name -> BacktestResult
        """
        if strategies is None:
            strategies = list(self.STRATEGIES.keys())

        results = {}
        for strategy in strategies:
            try:
                result = self.run_strategy(
                    symbol=symbol,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
                results[strategy] = result
            except Exception as e:
                print(f"Error with {strategy}: {e}")

        return results

    def comparison_summary(self, results: Dict[str, BacktestResult]) -> str:
        """Generate text summary comparing strategies"""
        if not results:
            return "No results to compare"

        lines = ["=== COMPARACION DE ESTRATEGIAS ===\n"]
        lines.append(f"{'Estrategia':<25} {'Retorno':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>7} {'WinRate':>8}")
        lines.append("-" * 75)

        # Sort by total return
        sorted_results = sorted(results.items(),
                               key=lambda x: x[1].total_return_pct, reverse=True)

        for name, r in sorted_results:
            lines.append(
                f"{r.strategy_name[:24]:<25} "
                f"{r.total_return_pct:>9.1f}% "
                f"{r.sharpe_ratio:>8.2f} "
                f"{r.max_drawdown_pct:>9.1f}% "
                f"{r.total_trades:>7} "
                f"{r.win_rate_pct:>7.1f}%"
            )

        lines.append("-" * 75)
        best = sorted_results[0][1]
        lines.append(f"\nMejor estrategia: {best.strategy_name} ({best.total_return_pct:.1f}% retorno)")

        return "\n".join(lines)

    def get_available_strategies(self) -> Dict[str, str]:
        """Return available strategies"""
        return self.STRATEGIES.copy()


# =============================================================================
# QUANTSTATS INTEGRATION
# =============================================================================

class PerformanceAnalyzer:
    """
    Performance analysis using quantstats.
    Provides detailed metrics and reports.
    """

    def __init__(self):
        self._qs = None

    @property
    def qs(self):
        """Lazy load quantstats"""
        if self._qs is None:
            try:
                import quantstats as qs
                self._qs = qs
            except ImportError:
                raise ImportError("quantstats not installed. Run: pip install quantstats")
        return self._qs

    def get_metrics(self, returns: pd.Series, benchmark: pd.Series = None) -> Dict:
        """Get comprehensive performance metrics"""
        qs = self.qs

        metrics = {
            'total_return': qs.stats.comp(returns) * 100,
            'cagr': qs.stats.cagr(returns) * 100,
            'sharpe': qs.stats.sharpe(returns),
            'sortino': qs.stats.sortino(returns),
            'max_drawdown': qs.stats.max_drawdown(returns) * 100,
            'avg_drawdown': qs.stats.avg_drawdown(returns) * 100,
            'volatility': qs.stats.volatility(returns) * 100,
            'calmar': qs.stats.calmar(returns),
            'skew': qs.stats.skew(returns),
            'kurtosis': qs.stats.kurtosis(returns),
            'var_95': qs.stats.var(returns) * 100,
            'cvar_95': qs.stats.cvar(returns) * 100,
            'best_day': qs.stats.best(returns) * 100,
            'worst_day': qs.stats.worst(returns) * 100,
            'win_rate': qs.stats.win_rate(returns) * 100,
            'profit_factor': qs.stats.profit_factor(returns),
            'payoff_ratio': qs.stats.payoff_ratio(returns),
            'avg_win': qs.stats.avg_win(returns) * 100,
            'avg_loss': qs.stats.avg_loss(returns) * 100
        }

        # Add benchmark comparison if provided
        if benchmark is not None:
            try:
                metrics['alpha'] = qs.stats.greeks(returns, benchmark)['alpha'] * 100
                metrics['beta'] = qs.stats.greeks(returns, benchmark)['beta']
                metrics['information_ratio'] = qs.stats.information_ratio(returns, benchmark)
            except:
                pass

        return metrics

    def generate_report(self, returns: pd.Series, benchmark: pd.Series = None,
                       title: str = "Strategy Performance") -> str:
        """Generate text performance report"""
        metrics = self.get_metrics(returns, benchmark)

        lines = [
            f"=== {title.upper()} ===\n",
            "RETURNS:",
            f"  Total Return: {metrics['total_return']:.2f}%",
            f"  CAGR: {metrics['cagr']:.2f}%",
            f"  Best Day: {metrics['best_day']:.2f}%",
            f"  Worst Day: {metrics['worst_day']:.2f}%",
            "",
            "RISK:",
            f"  Volatility (ann.): {metrics['volatility']:.2f}%",
            f"  Max Drawdown: {metrics['max_drawdown']:.2f}%",
            f"  Avg Drawdown: {metrics['avg_drawdown']:.2f}%",
            f"  VaR (95%): {metrics['var_95']:.2f}%",
            f"  CVaR (95%): {metrics['cvar_95']:.2f}%",
            "",
            "RISK-ADJUSTED:",
            f"  Sharpe Ratio: {metrics['sharpe']:.2f}",
            f"  Sortino Ratio: {metrics['sortino']:.2f}",
            f"  Calmar Ratio: {metrics['calmar']:.2f}",
            "",
            "TRADE STATS:",
            f"  Win Rate: {metrics['win_rate']:.1f}%",
            f"  Profit Factor: {metrics['profit_factor']:.2f}",
            f"  Payoff Ratio: {metrics['payoff_ratio']:.2f}",
            f"  Avg Win: {metrics['avg_win']:.2f}%",
            f"  Avg Loss: {metrics['avg_loss']:.2f}%"
        ]

        if 'alpha' in metrics:
            lines.extend([
                "",
                "VS BENCHMARK:",
                f"  Alpha: {metrics['alpha']:.2f}%",
                f"  Beta: {metrics['beta']:.2f}",
                f"  Information Ratio: {metrics['information_ratio']:.2f}"
            ])

        return "\n".join(lines)


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Backtest Engine Test ===\n")

    engine = BacktestEngine()

    print("Available strategies:")
    for k, v in engine.get_available_strategies().items():
        print(f"  {k}: {v}")

    print("\n--- Running backtest: SMA 20/50 on AAPL ---")
    try:
        result = engine.run_strategy("AAPL", "sma_20_50")
        print(result.summary())
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Comparing strategies on MSFT ---")
    try:
        comparison = engine.compare_strategies(
            "MSFT",
            strategies=['buy_hold', 'sma_20_50', 'rsi_oversold', 'macd_signal']
        )
        print(engine.comparison_summary(comparison))
    except Exception as e:
        print(f"Error: {e}")
