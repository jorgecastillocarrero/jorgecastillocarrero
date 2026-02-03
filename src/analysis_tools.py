"""
Analysis Tools for the Financial AI Assistant
Provides technical analysis, screening, predictions, and fundamental data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class TechnicalAnalysisTools:
    """Technical analysis calculations using ta and pandas-ta"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all major technical indicators"""
        import ta

        if len(df) < 20:
            return df

        # Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # OBV (On Balance Volume)
        if 'Volume' in df.columns:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

        # CCI (Commodity Channel Index)
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

        return df

    @staticmethod
    def get_current_indicators(symbol: str, period: str = "3mo") -> Dict:
        """Get current indicator values for a symbol"""
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if len(df) < 20:
                return {"error": f"Insufficient data for {symbol}"}

            df = TechnicalAnalysisTools.calculate_indicators(df)
            latest = df.iloc[-1]

            return {
                "symbol": symbol,
                "price": round(latest['Close'], 2),
                "rsi": round(latest['rsi'], 2) if pd.notna(latest['rsi']) else None,
                "macd": round(latest['macd'], 4) if pd.notna(latest['macd']) else None,
                "macd_signal": round(latest['macd_signal'], 4) if pd.notna(latest['macd_signal']) else None,
                "sma_20": round(latest['sma_20'], 2) if pd.notna(latest['sma_20']) else None,
                "sma_50": round(latest['sma_50'], 2) if pd.notna(latest['sma_50']) else None,
                "bb_upper": round(latest['bb_upper'], 2) if pd.notna(latest['bb_upper']) else None,
                "bb_lower": round(latest['bb_lower'], 2) if pd.notna(latest['bb_lower']) else None,
                "stoch_k": round(latest['stoch_k'], 2) if pd.notna(latest['stoch_k']) else None,
                "atr": round(latest['atr'], 2) if pd.notna(latest['atr']) else None,
                "adx": round(latest['adx'], 2) if pd.notna(latest['adx']) else None,
                "cci": round(latest['cci'], 2) if pd.notna(latest['cci']) else None,
                "williams_r": round(latest['williams_r'], 2) if pd.notna(latest['williams_r']) else None,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_signals(symbol: str) -> Dict:
        """Get buy/sell signals based on indicators"""
        indicators = TechnicalAnalysisTools.get_current_indicators(symbol)

        if "error" in indicators:
            return indicators

        signals = {"symbol": symbol, "signals": [], "overall": "NEUTRAL"}
        buy_signals = 0
        sell_signals = 0

        # RSI signals
        rsi = indicators.get('rsi')
        if rsi:
            if rsi < 30:
                signals["signals"].append({"indicator": "RSI", "signal": "BUY", "value": rsi, "reason": "Oversold"})
                buy_signals += 1
            elif rsi > 70:
                signals["signals"].append({"indicator": "RSI", "signal": "SELL", "value": rsi, "reason": "Overbought"})
                sell_signals += 1

        # MACD signals
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd and macd_signal:
            if macd > macd_signal:
                signals["signals"].append({"indicator": "MACD", "signal": "BUY", "reason": "MACD above signal"})
                buy_signals += 1
            else:
                signals["signals"].append({"indicator": "MACD", "signal": "SELL", "reason": "MACD below signal"})
                sell_signals += 1

        # Price vs SMA signals
        price = indicators.get('price')
        sma_50 = indicators.get('sma_50')
        if price and sma_50:
            if price > sma_50:
                signals["signals"].append({"indicator": "SMA50", "signal": "BUY", "reason": "Price above SMA50"})
                buy_signals += 1
            else:
                signals["signals"].append({"indicator": "SMA50", "signal": "SELL", "reason": "Price below SMA50"})
                sell_signals += 1

        # Stochastic signals
        stoch = indicators.get('stoch_k')
        if stoch:
            if stoch < 20:
                signals["signals"].append({"indicator": "Stochastic", "signal": "BUY", "value": stoch, "reason": "Oversold"})
                buy_signals += 1
            elif stoch > 80:
                signals["signals"].append({"indicator": "Stochastic", "signal": "SELL", "value": stoch, "reason": "Overbought"})
                sell_signals += 1

        # Overall signal
        if buy_signals > sell_signals + 1:
            signals["overall"] = "BUY"
        elif sell_signals > buy_signals + 1:
            signals["overall"] = "SELL"
        else:
            signals["overall"] = "NEUTRAL"

        signals["buy_signals"] = buy_signals
        signals["sell_signals"] = sell_signals

        return signals


class ScreenerTools:
    """Stock screening tools"""

    @staticmethod
    def screen_stocks(symbols: List[str], filters: Dict) -> List[Dict]:
        """
        Screen stocks based on filters

        Filters can include:
        - rsi_below: float
        - rsi_above: float
        - market_cap_min: float (in billions)
        - market_cap_max: float (in billions)
        - pe_below: float
        - pe_above: float
        - price_above_sma50: bool
        - price_below_sma50: bool
        """
        import yfinance as yf
        import ta

        results = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")

                if len(hist) < 20:
                    continue

                info = ticker.info
                current_price = hist['Close'].iloc[-1]

                # Calculate RSI
                rsi = ta.momentum.RSIIndicator(hist['Close'], window=14).rsi().iloc[-1]

                # Calculate SMA50
                sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None

                # Get fundamentals
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE')

                # Apply filters
                passes = True

                if 'rsi_below' in filters and rsi >= filters['rsi_below']:
                    passes = False
                if 'rsi_above' in filters and rsi <= filters['rsi_above']:
                    passes = False
                if 'market_cap_min' in filters and market_cap < filters['market_cap_min'] * 1e9:
                    passes = False
                if 'market_cap_max' in filters and market_cap > filters['market_cap_max'] * 1e9:
                    passes = False
                if 'pe_below' in filters and (pe_ratio is None or pe_ratio >= filters['pe_below']):
                    passes = False
                if 'pe_above' in filters and (pe_ratio is None or pe_ratio <= filters['pe_above']):
                    passes = False
                if 'price_above_sma50' in filters and filters['price_above_sma50']:
                    if sma_50 is None or current_price <= sma_50:
                        passes = False
                if 'price_below_sma50' in filters and filters['price_below_sma50']:
                    if sma_50 is None or current_price >= sma_50:
                        passes = False

                if passes:
                    results.append({
                        "symbol": symbol,
                        "price": round(current_price, 2),
                        "rsi": round(rsi, 2) if pd.notna(rsi) else None,
                        "sma_50": round(sma_50, 2) if sma_50 and pd.notna(sma_50) else None,
                        "market_cap_b": round(market_cap / 1e9, 2) if market_cap else None,
                        "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                    })

            except Exception as e:
                continue

        return results

    @staticmethod
    def find_oversold(symbols: List[str], rsi_threshold: float = 30) -> List[Dict]:
        """Find oversold stocks (RSI below threshold)"""
        return ScreenerTools.screen_stocks(symbols, {"rsi_below": rsi_threshold})

    @staticmethod
    def find_overbought(symbols: List[str], rsi_threshold: float = 70) -> List[Dict]:
        """Find overbought stocks (RSI above threshold)"""
        return ScreenerTools.screen_stocks(symbols, {"rsi_above": rsi_threshold})


class PredictionTools:
    """Price prediction tools using Prophet and ML"""

    @staticmethod
    def predict_prophet(symbol: str, days: int = 30) -> Dict:
        """Predict future prices using Prophet"""
        try:
            from prophet import Prophet
            import yfinance as yf

            # Get historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y")

            if len(df) < 100:
                return {"error": "Insufficient historical data"}

            # Prepare data for Prophet
            prophet_df = df.reset_index()[['Date', 'Close']].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

            # Train model
            model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
            model.fit(prophet_df)

            # Make future predictions
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            # Get predictions
            last_price = df['Close'].iloc[-1]
            predictions = forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            return {
                "symbol": symbol,
                "current_price": round(last_price, 2),
                "predictions": [
                    {
                        "date": row['ds'].strftime('%Y-%m-%d'),
                        "predicted": round(row['yhat'], 2),
                        "lower": round(row['yhat_lower'], 2),
                        "upper": round(row['yhat_upper'], 2),
                    }
                    for _, row in predictions.iterrows()
                ],
                "price_in_30_days": round(forecast['yhat'].iloc[-1], 2),
                "change_percent": round((forecast['yhat'].iloc[-1] - last_price) / last_price * 100, 2),
            }

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def predict_trend(symbol: str) -> Dict:
        """Predict short-term trend using simple ML"""
        try:
            from sklearn.linear_model import LinearRegression
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")

            if len(df) < 30:
                return {"error": "Insufficient data"}

            # Prepare features
            df['returns'] = df['Close'].pct_change()
            df['ma_5'] = df['Close'].rolling(5).mean()
            df['ma_20'] = df['Close'].rolling(20).mean()
            df = df.dropna()

            # Simple linear regression on recent prices
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['Close'].values

            model = LinearRegression()
            model.fit(X[-30:], y[-30:])  # Fit on last 30 days

            # Predict next 5 days
            future_X = np.arange(len(df), len(df) + 5).reshape(-1, 1)
            predictions = model.predict(future_X)

            current_price = df['Close'].iloc[-1]
            trend = "UP" if model.coef_[0] > 0 else "DOWN"
            strength = abs(model.coef_[0]) / current_price * 100

            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "trend": trend,
                "trend_strength": round(strength, 4),
                "prediction_5_days": round(predictions[-1], 2),
                "change_percent": round((predictions[-1] - current_price) / current_price * 100, 2),
            }

        except Exception as e:
            return {"error": str(e)}


class FundamentalTools:
    """Fundamental analysis tools"""

    @staticmethod
    def get_fundamentals(symbol: str) -> Dict:
        """Get fundamental data for a symbol"""
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "name": info.get('longName', symbol),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "market_cap": info.get('marketCap'),
                "market_cap_b": round(info.get('marketCap', 0) / 1e9, 2) if info.get('marketCap') else None,
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "peg_ratio": info.get('pegRatio'),
                "price_to_book": info.get('priceToBook'),
                "dividend_yield": info.get('dividendYield'),
                "profit_margin": info.get('profitMargins'),
                "revenue_growth": info.get('revenueGrowth'),
                "earnings_growth": info.get('earningsGrowth'),
                "debt_to_equity": info.get('debtToEquity'),
                "current_ratio": info.get('currentRatio'),
                "roe": info.get('returnOnEquity'),
                "roa": info.get('returnOnAssets'),
                "52_week_high": info.get('fiftyTwoWeekHigh'),
                "52_week_low": info.get('fiftyTwoWeekLow'),
                "50_day_avg": info.get('fiftyDayAverage'),
                "200_day_avg": info.get('twoHundredDayAverage'),
                "avg_volume": info.get('averageVolume'),
                "beta": info.get('beta'),
            }

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def compare_fundamentals(symbols: List[str]) -> List[Dict]:
        """Compare fundamentals of multiple symbols"""
        results = []
        for symbol in symbols:
            fund = FundamentalTools.get_fundamentals(symbol)
            if "error" not in fund:
                results.append(fund)
        return results


class PortfolioAnalysisTools:
    """Portfolio-level analysis tools"""

    @staticmethod
    def analyze_portfolio_risk(holdings: List[Dict]) -> Dict:
        """Analyze portfolio risk metrics"""
        import yfinance as yf

        symbols = [h['symbol'] for h in holdings if h.get('symbol')]
        weights = {}
        total_value = sum(abs(h.get('value', 0)) for h in holdings)

        if total_value == 0:
            return {"error": "No portfolio value"}

        for h in holdings:
            if h.get('symbol'):
                weights[h['symbol']] = abs(h.get('value', 0)) / total_value

        # Get historical returns
        returns_data = {}
        for symbol in symbols[:20]:  # Limit for speed
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if len(hist) > 20:
                    returns_data[symbol] = hist['Close'].pct_change().dropna()
            except:
                continue

        if len(returns_data) < 2:
            return {"error": "Insufficient data for risk analysis"}

        # Calculate portfolio metrics
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()

        # Individual stock volatilities
        volatilities = returns_df.std() * np.sqrt(252)  # Annualized

        # Portfolio volatility (simplified)
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].mean()

        return {
            "num_positions": len(symbols),
            "avg_correlation": round(avg_correlation, 3),
            "diversification": "HIGH" if avg_correlation < 0.3 else "MEDIUM" if avg_correlation < 0.6 else "LOW",
            "top_volatilities": {
                symbol: round(vol, 4) for symbol, vol in volatilities.nlargest(5).items()
            },
            "lowest_volatilities": {
                symbol: round(vol, 4) for symbol, vol in volatilities.nsmallest(5).items()
            },
        }


# Main analysis function that combines all tools
def analyze_query(query: str, portfolio_symbols: List[str] = None) -> Dict:
    """
    Analyze a query and return relevant data
    This function interprets the query and calls appropriate tools
    """
    query_lower = query.lower()
    results = {}

    # Check for RSI queries
    if 'rsi' in query_lower:
        if any(word in query_lower for word in ['bajo', 'below', 'menor', 'oversold', 'sobrevendid']):
            threshold = 30
            # Try to extract threshold from query
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                threshold = int(numbers[0])
            results['screener'] = ScreenerTools.find_oversold(portfolio_symbols or [], threshold)

        elif any(word in query_lower for word in ['alto', 'above', 'mayor', 'overbought', 'sobrecomprad']):
            threshold = 70
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                threshold = int(numbers[0])
            results['screener'] = ScreenerTools.find_overbought(portfolio_symbols or [], threshold)

    # Check for prediction queries
    if any(word in query_lower for word in ['prediccion', 'prediction', 'futuro', 'forecast', 'predecir']):
        # Extract symbol if mentioned
        for symbol in (portfolio_symbols or []):
            if symbol.lower() in query_lower:
                results['prediction'] = PredictionTools.predict_prophet(symbol)
                break

    # Check for technical analysis queries
    if any(word in query_lower for word in ['tecnico', 'technical', 'indicador', 'macd', 'bollinger', 'signal']):
        for symbol in (portfolio_symbols or []):
            if symbol.lower() in query_lower:
                results['technical'] = TechnicalAnalysisTools.get_current_indicators(symbol)
                results['signals'] = TechnicalAnalysisTools.get_signals(symbol)
                break

    # Check for fundamental queries
    if any(word in query_lower for word in ['fundamental', 'pe', 'ratio', 'market cap', 'capitalizacion', 'dividend']):
        for symbol in (portfolio_symbols or []):
            if symbol.lower() in query_lower:
                results['fundamentals'] = FundamentalTools.get_fundamentals(symbol)
                break

    return results
