"""
ML Predictor Module for Financial AI Assistant
Integrates: scikit-learn, xgboost, lightgbm, prophet, statsmodels

Provides price predictions, trend classification, and pattern detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PricePrediction:
    """Container for price predictions"""
    symbol: str
    model: str
    current_price: float
    predictions: pd.DataFrame  # date, predicted, lower, upper
    predicted_change_pct: float
    confidence: str

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'model': self.model,
            'current_price': round(self.current_price, 2),
            'predicted_change': f"{self.predicted_change_pct:+.2f}%",
            'confidence': self.confidence,
            'forecast_days': len(self.predictions),
            'predictions': self.predictions.to_dict('records')
        }

    def summary(self) -> str:
        """Text summary for AI assistant"""
        last_pred = self.predictions.iloc[-1]
        lines = [
            f"=== PREDICCION: {self.symbol} ({self.model}) ===",
            f"Precio actual: ${self.current_price:.2f}",
            f"Horizonte: {len(self.predictions)} dias",
            f"Confianza: {self.confidence}",
            "",
            "PREDICCION:",
            f"  Precio final esperado: ${last_pred['predicted']:.2f}",
            f"  Cambio esperado: {self.predicted_change_pct:+.2f}%",
            f"  Rango (95%): ${last_pred.get('lower', 0):.2f} - ${last_pred.get('upper', 0):.2f}",
            "",
            "PRIMEROS 5 DIAS:"
        ]

        for _, row in self.predictions.head().iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            lines.append(f"  {date_str}: ${row['predicted']:.2f}")

        return "\n".join(lines)


@dataclass
class TrendPrediction:
    """Container for trend classification"""
    symbol: str
    model: str
    current_trend: str  # 'bullish', 'bearish', 'neutral'
    trend_probability: float
    trend_strength: float  # 0-1
    signals: Dict[str, str]  # indicator -> signal

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'model': self.model,
            'current_trend': self.current_trend,
            'probability': f"{self.trend_probability:.1%}",
            'strength': f"{self.trend_strength:.1%}",
            'signals': self.signals
        }

    def summary(self) -> str:
        """Text summary for AI assistant"""
        trend_emoji = {'bullish': 'ALCISTA', 'bearish': 'BAJISTA', 'neutral': 'NEUTRAL'}
        lines = [
            f"=== PREDICCION DE TENDENCIA: {self.symbol} ===",
            f"Modelo: {self.model}",
            "",
            f"TENDENCIA: {trend_emoji.get(self.current_trend, self.current_trend)}",
            f"Probabilidad: {self.trend_probability:.1%}",
            f"Fuerza: {self.trend_strength:.1%}",
            "",
            "SENALES DE INDICADORES:"
        ]

        for indicator, signal in self.signals.items():
            lines.append(f"  {indicator}: {signal}")

        return "\n".join(lines)


class MLPredictor:
    """
    Machine Learning predictions for financial data.
    Uses data from local database via DatabaseAnalyzer.
    """

    MODELS = {
        'prophet': 'Facebook Prophet (series temporales)',
        'arima': 'ARIMA (statsmodels)',
        'xgboost': 'XGBoost (gradient boosting)',
        'lightgbm': 'LightGBM (gradient boosting)',
        'linear': 'Linear Regression (tendencia simple)',
        'random_forest': 'Random Forest (feature importance)'
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
        """Get OHLCV data for predictions"""
        df = self.db.get_price_history(symbol, days=days)
        if df.empty:
            raise ValueError(f"No price data for {symbol}")
        return df

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for ML models"""
        df = df.copy()

        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_sma + 2 * bb_std
        df['bb_lower'] = bb_sma - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volatility
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)

        # Volume features
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Day of week, month (for seasonality)
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date']).dt.month

        return df.dropna()

    def _create_target(self, df: pd.DataFrame, horizon: int = 5,
                       target_type: str = 'direction') -> pd.DataFrame:
        """Create target variable for prediction"""
        df = df.copy()

        if target_type == 'direction':
            # Binary: 1 if price up in horizon days, 0 otherwise
            df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
        elif target_type == 'returns':
            # Future returns
            df['target'] = df['close'].shift(-horizon) / df['close'] - 1
        elif target_type == 'price':
            # Future price
            df['target'] = df['close'].shift(-horizon)

        return df.dropna()

    # =========================================================================
    # PROPHET PREDICTIONS
    # =========================================================================

    def _predict_prophet(self, symbol: str, days: int = 30) -> PricePrediction:
        """Price prediction using Facebook Prophet"""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet not installed. Run: pip install prophet")

        # Get data
        df = self.get_price_data(symbol, days=500)
        current_price = df['close'].iloc[-1]

        # Prepare for Prophet
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df['close']
        })

        # Train model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)

        # Make predictions
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        # Get future predictions only
        predictions = forecast[forecast['ds'] > prophet_df['ds'].max()].copy()
        predictions = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'ds': 'date', 'yhat': 'predicted', 'yhat_lower': 'lower', 'yhat_upper': 'upper'}
        )

        # Calculate expected change
        final_price = predictions['predicted'].iloc[-1]
        change_pct = (final_price / current_price - 1) * 100

        # Confidence based on prediction interval width
        avg_interval = (predictions['upper'] - predictions['lower']).mean()
        confidence_ratio = avg_interval / current_price
        if confidence_ratio < 0.1:
            confidence = "Alta"
        elif confidence_ratio < 0.2:
            confidence = "Media"
        else:
            confidence = "Baja"

        return PricePrediction(
            symbol=symbol,
            model="Prophet",
            current_price=current_price,
            predictions=predictions,
            predicted_change_pct=change_pct,
            confidence=confidence
        )

    # =========================================================================
    # ARIMA PREDICTIONS
    # =========================================================================

    def _predict_arima(self, symbol: str, days: int = 30) -> PricePrediction:
        """Price prediction using ARIMA"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("statsmodels not installed. Run: pip install statsmodels")

        # Get data
        df = self.get_price_data(symbol, days=500)
        current_price = df['close'].iloc[-1]
        prices = df.set_index('date')['close']

        # Fit ARIMA model
        model = ARIMA(prices, order=(5, 1, 2))
        fitted = model.fit()

        # Forecast
        forecast = fitted.forecast(steps=days)
        conf_int = fitted.get_forecast(steps=days).conf_int()

        # Create predictions DataFrame
        last_date = pd.to_datetime(df['date'].iloc[-1])
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')

        predictions = pd.DataFrame({
            'date': future_dates,
            'predicted': forecast.values,
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values
        })

        # Calculate expected change
        final_price = predictions['predicted'].iloc[-1]
        change_pct = (final_price / current_price - 1) * 100

        return PricePrediction(
            symbol=symbol,
            model="ARIMA(5,1,2)",
            current_price=current_price,
            predictions=predictions,
            predicted_change_pct=change_pct,
            confidence="Media"
        )

    # =========================================================================
    # XGBOOST / LIGHTGBM TREND PREDICTION
    # =========================================================================

    def _predict_trend_xgboost(self, symbol: str, horizon: int = 5) -> TrendPrediction:
        """Trend prediction using XGBoost"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        # Get and prepare data
        df = self.get_price_data(symbol, days=500)
        df = self._create_features(df)
        df = self._create_target(df, horizon=horizon, target_type='direction')

        # Define features
        feature_cols = [c for c in df.columns if c not in
                       ['date', 'open', 'high', 'low', 'close', 'volume', 'target']]

        X = df[feature_cols].values
        y = df['target'].values

        # Train/test split (use last 20% for test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Predict on last observation (current)
        current_features = X[-1].reshape(1, -1)
        prob = model.predict_proba(current_features)[0]

        # Get feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

        # Determine trend
        bullish_prob = prob[1] if len(prob) > 1 else prob[0]
        if bullish_prob > 0.6:
            trend = 'bullish'
        elif bullish_prob < 0.4:
            trend = 'bearish'
        else:
            trend = 'neutral'

        # Get indicator signals
        last_row = df.iloc[-1]
        signals = {
            'RSI': 'Sobreventa' if last_row.get('rsi', 50) < 30 else 'Sobrecompra' if last_row.get('rsi', 50) > 70 else 'Neutral',
            'MACD': 'Alcista' if last_row.get('macd_hist', 0) > 0 else 'Bajista',
            'SMA20': 'Por encima' if last_row.get('sma_ratio_20', 1) > 1 else 'Por debajo',
            'Bollinger': 'Alto' if last_row.get('bb_position', 0.5) > 0.8 else 'Bajo' if last_row.get('bb_position', 0.5) < 0.2 else 'Medio'
        }

        return TrendPrediction(
            symbol=symbol,
            model="XGBoost",
            current_trend=trend,
            trend_probability=bullish_prob,
            trend_strength=abs(bullish_prob - 0.5) * 2,
            signals=signals
        )

    def _predict_trend_lightgbm(self, symbol: str, horizon: int = 5) -> TrendPrediction:
        """Trend prediction using LightGBM"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        # Get and prepare data
        df = self.get_price_data(symbol, days=500)
        df = self._create_features(df)
        df = self._create_target(df, horizon=horizon, target_type='direction')

        # Define features
        feature_cols = [c for c in df.columns if c not in
                       ['date', 'open', 'high', 'low', 'close', 'volume', 'target']]

        X = df[feature_cols].values
        y = df['target'].values

        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

        # Predict on current
        current_features = X[-1].reshape(1, -1)
        prob = model.predict_proba(current_features)[0]

        bullish_prob = prob[1] if len(prob) > 1 else prob[0]
        if bullish_prob > 0.6:
            trend = 'bullish'
        elif bullish_prob < 0.4:
            trend = 'bearish'
        else:
            trend = 'neutral'

        # Get indicator signals
        last_row = df.iloc[-1]
        signals = {
            'RSI': 'Sobreventa' if last_row.get('rsi', 50) < 30 else 'Sobrecompra' if last_row.get('rsi', 50) > 70 else 'Neutral',
            'MACD': 'Alcista' if last_row.get('macd_hist', 0) > 0 else 'Bajista',
            'SMA20': 'Por encima' if last_row.get('sma_ratio_20', 1) > 1 else 'Por debajo',
            'Momentum': 'Positivo' if last_row.get('momentum_10', 0) > 0 else 'Negativo'
        }

        return TrendPrediction(
            symbol=symbol,
            model="LightGBM",
            current_trend=trend,
            trend_probability=bullish_prob,
            trend_strength=abs(bullish_prob - 0.5) * 2,
            signals=signals
        )

    # =========================================================================
    # LINEAR REGRESSION
    # =========================================================================

    def _predict_linear(self, symbol: str, days: int = 30) -> PricePrediction:
        """Simple linear regression prediction"""
        from sklearn.linear_model import LinearRegression

        # Get data
        df = self.get_price_data(symbol, days=200)
        current_price = df['close'].iloc[-1]

        # Prepare data
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['close'].values

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Predict future
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        future_prices = model.predict(future_X)

        # Calculate confidence interval (simple std-based)
        residuals = y - model.predict(X)
        std_error = np.std(residuals)

        last_date = pd.to_datetime(df['date'].iloc[-1])
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')

        predictions = pd.DataFrame({
            'date': future_dates,
            'predicted': future_prices,
            'lower': future_prices - 1.96 * std_error,
            'upper': future_prices + 1.96 * std_error
        })

        final_price = predictions['predicted'].iloc[-1]
        change_pct = (final_price / current_price - 1) * 100

        return PricePrediction(
            symbol=symbol,
            model="Linear Regression",
            current_price=current_price,
            predictions=predictions,
            predicted_change_pct=change_pct,
            confidence="Baja"  # Linear is too simple for stock prices
        )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Get feature importance using Random Forest"""
        from sklearn.ensemble import RandomForestClassifier

        # Get and prepare data
        df = self.get_price_data(symbol, days=500)
        df = self._create_features(df)
        df = self._create_target(df, horizon=5, target_type='direction')

        feature_cols = [c for c in df.columns if c not in
                       ['date', 'open', 'high', 'low', 'close', 'volume', 'target']]

        X = df[feature_cols].values
        y = df['target'].values

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Get importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def feature_importance_summary(self, symbol: str) -> str:
        """Generate text summary of feature importance"""
        importance = self.get_feature_importance(symbol)

        lines = [
            f"=== IMPORTANCIA DE FEATURES: {symbol} ===",
            f"Modelo: Random Forest\n",
            "TOP 10 FEATURES MAS IMPORTANTES:"
        ]

        for i, (feature, imp) in enumerate(list(importance.items())[:10], 1):
            bar = "=" * int(imp * 50)
            lines.append(f"  {i:2}. {feature:<20} {imp:.3f} {bar}")

        return "\n".join(lines)

    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================

    def predict_price(self, symbol: str, days: int = 30,
                      model: str = 'prophet') -> PricePrediction:
        """
        Predict future prices.

        Args:
            symbol: Stock symbol
            days: Days to predict
            model: 'prophet', 'arima', or 'linear'

        Returns:
            PricePrediction object
        """
        model_map = {
            'prophet': self._predict_prophet,
            'arima': self._predict_arima,
            'linear': self._predict_linear
        }

        if model not in model_map:
            raise ValueError(f"Unknown model: {model}. Available: {list(model_map.keys())}")

        return model_map[model](symbol, days)

    def predict_trend(self, symbol: str, model: str = 'xgboost',
                      horizon: int = 5) -> TrendPrediction:
        """
        Predict price trend direction.

        Args:
            symbol: Stock symbol
            model: 'xgboost' or 'lightgbm'
            horizon: Days ahead to predict

        Returns:
            TrendPrediction object
        """
        model_map = {
            'xgboost': self._predict_trend_xgboost,
            'lightgbm': self._predict_trend_lightgbm
        }

        if model not in model_map:
            raise ValueError(f"Unknown model: {model}. Available: {list(model_map.keys())}")

        return model_map[model](symbol, horizon)

    def get_available_models(self) -> Dict[str, str]:
        """Return available models"""
        return self.MODELS.copy()


# =============================================================================
# PATTERN DETECTION
# =============================================================================

class PatternDetector:
    """
    Detects chart patterns in price data.
    """

    def __init__(self, db_path: str = "data/financial_data.db"):
        self.db_path = db_path
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from src.db_analysis_tools import DatabaseAnalyzer
            self._db = DatabaseAnalyzer(self.db_path)
        return self._db

    def detect_patterns(self, symbol: str, days: int = 100) -> List[Dict]:
        """Detect common chart patterns"""
        df = self.db.get_price_history(symbol, days=days)
        if len(df) < 20:
            return []

        patterns = []

        # Double Top/Bottom detection (simplified)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Find local maxima/minima
        for i in range(10, len(df) - 5):
            # Double Top
            if highs[i] > highs[i-5:i].max() and highs[i] > highs[i+1:i+5].max():
                # Look for another peak nearby
                for j in range(i+10, min(i+30, len(df)-5)):
                    if highs[j] > highs[j-5:j].max() and highs[j] > highs[j+1:j+5].max():
                        if abs(highs[i] - highs[j]) / highs[i] < 0.03:  # Within 3%
                            patterns.append({
                                'pattern': 'Double Top',
                                'date': df['date'].iloc[j],
                                'signal': 'Bearish',
                                'strength': 'Medium'
                            })
                            break

            # Double Bottom
            if lows[i] < lows[i-5:i].min() and lows[i] < lows[i+1:i+5].min():
                for j in range(i+10, min(i+30, len(df)-5)):
                    if lows[j] < lows[j-5:j].min() and lows[j] < lows[j+1:j+5].min():
                        if abs(lows[i] - lows[j]) / lows[i] < 0.03:
                            patterns.append({
                                'pattern': 'Double Bottom',
                                'date': df['date'].iloc[j],
                                'signal': 'Bullish',
                                'strength': 'Medium'
                            })
                            break

        # Trend detection
        sma_20 = pd.Series(closes).rolling(20).mean()
        sma_50 = pd.Series(closes).rolling(50).mean()

        if len(sma_50.dropna()) > 0:
            if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-10] < sma_50.iloc[-10]:
                patterns.append({
                    'pattern': 'Golden Cross',
                    'date': df['date'].iloc[-1],
                    'signal': 'Bullish',
                    'strength': 'Strong'
                })
            elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-10] > sma_50.iloc[-10]:
                patterns.append({
                    'pattern': 'Death Cross',
                    'date': df['date'].iloc[-1],
                    'signal': 'Bearish',
                    'strength': 'Strong'
                })

        return patterns

    def patterns_summary(self, symbol: str) -> str:
        """Generate text summary of detected patterns"""
        patterns = self.detect_patterns(symbol)

        if not patterns:
            return f"No se detectaron patrones significativos en {symbol}"

        lines = [
            f"=== PATRONES DETECTADOS: {symbol} ===\n"
        ]

        for p in patterns:
            date_str = p['date'].strftime('%Y-%m-%d') if hasattr(p['date'], 'strftime') else str(p['date'])[:10]
            lines.append(
                f"{p['pattern']} ({date_str}): "
                f"Senal {p['signal']}, Fuerza {p['strength']}"
            )

        return "\n".join(lines)


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    print("=== ML Predictor Test ===\n")

    predictor = MLPredictor()

    print("Available models:")
    for k, v in predictor.get_available_models().items():
        print(f"  {k}: {v}")

    print("\n--- Price Prediction (Linear) ---")
    try:
        pred = predictor.predict_price("AAPL", days=30, model='linear')
        print(pred.summary())
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Trend Prediction (XGBoost) ---")
    try:
        trend = predictor.predict_trend("AAPL", model='xgboost')
        print(trend.summary())
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Feature Importance ---")
    try:
        print(predictor.feature_importance_summary("AAPL"))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Pattern Detection ---")
    try:
        detector = PatternDetector()
        print(detector.patterns_summary("AAPL"))
    except Exception as e:
        print(f"Error: {e}")
