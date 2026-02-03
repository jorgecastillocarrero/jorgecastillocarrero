"""
External Data Module for Financial AI Assistant
Integrates: fredapi, finnhub, newsapi

Provides access to economic indicators, market data, and news.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()


@dataclass
class EconomicIndicator:
    """Container for economic indicator data"""
    name: str
    value: float
    date: str
    change: float = None
    unit: str = None

    def __str__(self):
        change_str = f" ({self.change:+.2f}%)" if self.change else ""
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{self.name}: {self.value:.2f}{unit_str}{change_str} ({self.date})"


@dataclass
class NewsItem:
    """Container for news item"""
    title: str
    source: str
    date: str
    url: str
    sentiment: str = None
    summary: str = None

    def __str__(self):
        return f"[{self.date}] {self.source}: {self.title}"


@dataclass
class MarketEvent:
    """Container for market event (crashes, rallies, etc)"""
    date: str
    event_type: str
    description: str
    market_impact: str
    related_symbols: List[str] = None


class ExternalDataProvider:
    """
    Provider for external data: Fed data, market quotes, and news.
    Uses optional API keys from environment.
    """

    # Historical market events database
    MARKET_EVENTS = [
        MarketEvent("2020-03-23", "crash", "COVID-19 Market Bottom", "S&P 500 down 34% from peak",
                    ["SPY", "VIX", "XLF"]),
        MarketEvent("2020-03-12", "crash", "Black Thursday COVID", "S&P 500 -9.5% single day",
                    ["SPY", "DIA", "QQQ"]),
        MarketEvent("2022-01-03", "correction", "Tech Selloff Begins", "NASDAQ starts 33% decline",
                    ["QQQ", "ARKK", "TSLA"]),
        MarketEvent("2022-06-16", "bottom", "2022 Bear Market Bottom", "S&P 500 reaches -24%",
                    ["SPY", "IWM"]),
        MarketEvent("2018-12-24", "correction", "Christmas Eve Massacre", "S&P 500 -2.7%, near bear market",
                    ["SPY", "XLF"]),
        MarketEvent("2015-08-24", "flash_crash", "China-Induced Flash Crash", "Dow drops 1000 points",
                    ["DIA", "FXI", "EEM"]),
        MarketEvent("2011-08-08", "crash", "US Credit Downgrade Selloff", "S&P 500 -6.7%",
                    ["SPY", "TLT"]),
        MarketEvent("2010-05-06", "flash_crash", "Flash Crash 2010", "Dow drops 1000 points in minutes",
                    ["DIA", "SPY"]),
        MarketEvent("2008-10-10", "crash", "Financial Crisis Bottom Week", "Extreme volatility",
                    ["SPY", "XLF", "C", "BAC"]),
        MarketEvent("2008-09-29", "crash", "Lehman Aftermath", "Dow drops 777 points",
                    ["DIA", "XLF", "AIG"]),
        MarketEvent("2008-09-15", "crash", "Lehman Brothers Bankruptcy", "Financial system crisis",
                    ["XLF", "C", "JPM", "GS"]),
        MarketEvent("2007-10-09", "peak", "Pre-Crisis Market Peak", "S&P 500 all-time high before crisis",
                    ["SPY"]),
        MarketEvent("2000-03-10", "peak", "Dot-com Bubble Peak", "NASDAQ reaches 5048",
                    ["QQQ", "MSFT", "CSCO"]),
        MarketEvent("2002-10-09", "bottom", "Dot-com Crash Bottom", "S&P 500 down 49% from peak",
                    ["SPY", "QQQ"]),
        MarketEvent("1987-10-19", "crash", "Black Monday", "Dow drops 22.6% in single day",
                    ["DIA", "SPY"]),
    ]

    def __init__(self):
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")

        self._fred = None
        self._finnhub = None

    # =========================================================================
    # FRED (Federal Reserve Economic Data)
    # =========================================================================

    @property
    def fred(self):
        """Lazy load FRED client"""
        if self._fred is None and self.fred_api_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.fred_api_key)
            except ImportError:
                pass
        return self._fred

    def get_fed_rate(self) -> Optional[EconomicIndicator]:
        """Get current Federal Funds Rate"""
        if not self.fred:
            return None

        try:
            data = self.fred.get_series('FEDFUNDS', observation_start='2020-01-01')
            if not data.empty:
                current = data.iloc[-1]
                previous = data.iloc[-2] if len(data) > 1 else current
                return EconomicIndicator(
                    name="Fed Funds Rate",
                    value=current,
                    date=data.index[-1].strftime('%Y-%m-%d'),
                    change=current - previous,
                    unit="%"
                )
        except Exception as e:
            print(f"Error getting Fed Rate: {e}")
        return None

    def get_inflation(self) -> Optional[EconomicIndicator]:
        """Get current CPI inflation rate"""
        if not self.fred:
            return None

        try:
            data = self.fred.get_series('CPIAUCSL', observation_start='2020-01-01')
            if not data.empty:
                # Calculate YoY change
                current = data.iloc[-1]
                year_ago = data.iloc[-12] if len(data) >= 12 else data.iloc[0]
                yoy_change = (current / year_ago - 1) * 100
                return EconomicIndicator(
                    name="CPI Inflation (YoY)",
                    value=yoy_change,
                    date=data.index[-1].strftime('%Y-%m-%d'),
                    unit="%"
                )
        except Exception:
            pass
        return None

    def get_unemployment(self) -> Optional[EconomicIndicator]:
        """Get current unemployment rate"""
        if not self.fred:
            return None

        try:
            data = self.fred.get_series('UNRATE', observation_start='2020-01-01')
            if not data.empty:
                current = data.iloc[-1]
                previous = data.iloc[-2] if len(data) > 1 else current
                return EconomicIndicator(
                    name="Unemployment Rate",
                    value=current,
                    date=data.index[-1].strftime('%Y-%m-%d'),
                    change=current - previous,
                    unit="%"
                )
        except Exception:
            pass
        return None

    def get_gdp_growth(self) -> Optional[EconomicIndicator]:
        """Get GDP growth rate"""
        if not self.fred:
            return None

        try:
            data = self.fred.get_series('A191RL1Q225SBEA', observation_start='2020-01-01')
            if not data.empty:
                current = data.iloc[-1]
                return EconomicIndicator(
                    name="GDP Growth (QoQ)",
                    value=current,
                    date=data.index[-1].strftime('%Y-%m-%d'),
                    unit="%"
                )
        except Exception:
            pass
        return None

    def get_treasury_yields(self) -> Dict[str, EconomicIndicator]:
        """Get Treasury yields (2Y, 10Y, 30Y)"""
        if not self.fred:
            return {}

        yields = {}
        series_map = {
            '2Y Treasury': 'DGS2',
            '10Y Treasury': 'DGS10',
            '30Y Treasury': 'DGS30'
        }

        for name, series_id in series_map.items():
            try:
                data = self.fred.get_series(series_id, observation_start='2023-01-01')
                if not data.empty:
                    current = data.dropna().iloc[-1]
                    yields[name] = EconomicIndicator(
                        name=name,
                        value=current,
                        date=data.index[-1].strftime('%Y-%m-%d'),
                        unit="%"
                    )
            except Exception:
                continue

        return yields

    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get all key economic indicators"""
        indicators = {}

        fed_rate = self.get_fed_rate()
        if fed_rate:
            indicators['fed_rate'] = fed_rate

        inflation = self.get_inflation()
        if inflation:
            indicators['inflation'] = inflation

        unemployment = self.get_unemployment()
        if unemployment:
            indicators['unemployment'] = unemployment

        gdp = self.get_gdp_growth()
        if gdp:
            indicators['gdp_growth'] = gdp

        yields = self.get_treasury_yields()
        indicators.update(yields)

        return indicators

    def economic_summary(self) -> str:
        """Generate text summary of economic indicators"""
        indicators = self.get_economic_indicators()

        if not indicators:
            return "No se pudo obtener datos economicos. Verifica FRED_API_KEY."

        lines = ["=== INDICADORES ECONOMICOS (FRED) ===\n"]

        for name, ind in indicators.items():
            lines.append(str(ind))

        # Add yield curve analysis
        if '2Y Treasury' in indicators and '10Y Treasury' in indicators:
            spread = indicators['10Y Treasury'].value - indicators['2Y Treasury'].value
            status = "NORMAL" if spread > 0 else "INVERTIDA (recesion warning)"
            lines.append(f"\nCurva de rendimiento: {spread:.2f}% - {status}")

        return "\n".join(lines)

    # =========================================================================
    # FINNHUB (Real-time Market Data)
    # =========================================================================

    @property
    def finnhub(self):
        """Lazy load Finnhub client"""
        if self._finnhub is None and self.finnhub_api_key:
            try:
                import finnhub
                self._finnhub = finnhub.Client(api_key=self.finnhub_api_key)
            except ImportError:
                pass
        return self._finnhub

    def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Finnhub"""
        if not self.finnhub:
            return None

        try:
            quote = self.finnhub.quote(symbol.upper())
            if quote and quote.get('c'):
                return {
                    'symbol': symbol.upper(),
                    'current_price': quote['c'],
                    'change': quote['d'],
                    'change_pct': quote['dp'],
                    'high': quote['h'],
                    'low': quote['l'],
                    'open': quote['o'],
                    'prev_close': quote['pc'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            print(f"Error getting quote: {e}")
        return None

    def get_market_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get market sentiment from Finnhub"""
        if not self.finnhub:
            return None

        try:
            # Get recommendation trends
            recs = self.finnhub.recommendation_trends(symbol.upper())
            if recs:
                latest = recs[0]
                total = latest['buy'] + latest['hold'] + latest['sell'] + latest['strongBuy'] + latest['strongSell']
                if total > 0:
                    bullish = (latest['strongBuy'] + latest['buy']) / total
                    bearish = (latest['strongSell'] + latest['sell']) / total

                    if bullish > 0.6:
                        sentiment = "Bullish"
                    elif bearish > 0.4:
                        sentiment = "Bearish"
                    else:
                        sentiment = "Neutral"

                    return {
                        'symbol': symbol.upper(),
                        'sentiment': sentiment,
                        'strong_buy': latest['strongBuy'],
                        'buy': latest['buy'],
                        'hold': latest['hold'],
                        'sell': latest['sell'],
                        'strong_sell': latest['strongSell'],
                        'period': latest['period']
                    }
        except Exception:
            pass
        return None

    # =========================================================================
    # NEWS API
    # =========================================================================

    def get_news(self, query: str = None, symbol: str = None,
                 days: int = 7) -> List[NewsItem]:
        """
        Get recent news articles.

        Args:
            query: Search query (e.g., "market crash", "fed rate")
            symbol: Stock symbol to search
            days: Days back to search (max 30 for free tier)

        Returns:
            List of NewsItem
        """
        if not self.news_api_key:
            return []

        from_date = (datetime.now() - timedelta(days=min(days, 30))).strftime('%Y-%m-%d')

        if symbol:
            query = f"{symbol} stock"
        elif not query:
            query = "stock market"

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.news_api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get('status') == 'ok':
                articles = []
                for article in data.get('articles', [])[:10]:
                    articles.append(NewsItem(
                        title=article.get('title', ''),
                        source=article.get('source', {}).get('name', 'Unknown'),
                        date=article.get('publishedAt', '')[:10],
                        url=article.get('url', ''),
                        summary=article.get('description', '')
                    ))
                return articles
        except Exception as e:
            print(f"Error getting news: {e}")

        return []

    def news_summary(self, query: str = None, symbol: str = None) -> str:
        """Generate text summary of news"""
        news = self.get_news(query=query, symbol=symbol)

        if not news:
            return "No se encontraron noticias. Verifica NEWS_API_KEY."

        title = f"NOTICIAS: {symbol or query or 'Mercado'}"
        lines = [f"=== {title.upper()} ===\n"]

        for item in news:
            lines.append(f"[{item.date}] {item.source}")
            lines.append(f"  {item.title}")
            if item.summary:
                lines.append(f"  {item.summary[:100]}...")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # HISTORICAL MARKET EVENTS
    # =========================================================================

    def get_market_events(self, event_type: str = None,
                          year_from: int = None,
                          year_to: int = None) -> List[MarketEvent]:
        """
        Get historical market events (crashes, corrections, etc).

        Args:
            event_type: 'crash', 'correction', 'flash_crash', 'peak', 'bottom'
            year_from: Start year filter
            year_to: End year filter

        Returns:
            List of MarketEvent
        """
        events = self.MARKET_EVENTS.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if year_from:
            events = [e for e in events if int(e.date[:4]) >= year_from]

        if year_to:
            events = [e for e in events if int(e.date[:4]) <= year_to]

        return events

    def search_events(self, query: str) -> List[MarketEvent]:
        """
        Search market events by keyword.

        Args:
            query: Search term (e.g., "crash", "COVID", "Lehman")

        Returns:
            Matching events
        """
        query = query.lower()
        return [e for e in self.MARKET_EVENTS
                if query in e.description.lower() or query in e.event_type.lower()]

    def events_summary(self, event_type: str = None) -> str:
        """Generate text summary of market events"""
        events = self.get_market_events(event_type=event_type)

        type_name = event_type.upper() if event_type else "TODOS"
        lines = [f"=== EVENTOS DE MERCADO HISTORICOS ({type_name}) ===\n"]

        for event in events:
            lines.append(f"{event.date} - {event.event_type.upper()}")
            lines.append(f"  {event.description}")
            lines.append(f"  Impacto: {event.market_impact}")
            if event.related_symbols:
                lines.append(f"  Simbolos: {', '.join(event.related_symbols)}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # VIX / FEAR & GREED
    # =========================================================================

    def get_vix_status(self) -> Optional[Dict]:
        """Get VIX status (fear indicator) from database"""
        try:
            from src.db_analysis_tools import DatabaseAnalyzer
            db = DatabaseAnalyzer()
            df = db.get_price_history("VIX", days=30)

            if not df.empty:
                current = df['close'].iloc[-1]
                avg_30d = df['close'].mean()

                if current < 15:
                    status = "Extreme Greed"
                elif current < 20:
                    status = "Low Fear"
                elif current < 25:
                    status = "Neutral"
                elif current < 30:
                    status = "Fear"
                else:
                    status = "Extreme Fear"

                return {
                    'vix': round(current, 2),
                    'avg_30d': round(avg_30d, 2),
                    'status': status,
                    'date': df['date'].iloc[-1].strftime('%Y-%m-%d')
                }
        except Exception:
            pass
        return None

    # =========================================================================
    # FULL SUMMARY
    # =========================================================================

    def get_market_overview(self) -> str:
        """Get comprehensive market overview"""
        lines = ["=== OVERVIEW DEL MERCADO ===\n"]

        # VIX
        vix = self.get_vix_status()
        if vix:
            lines.append(f"VIX: {vix['vix']} - {vix['status']}")
            lines.append(f"  Promedio 30d: {vix['avg_30d']}\n")

        # Economic indicators
        indicators = self.get_economic_indicators()
        if indicators:
            lines.append("INDICADORES ECONOMICOS:")
            for name, ind in list(indicators.items())[:5]:
                lines.append(f"  {ind}")
            lines.append("")

        # Recent events
        recent_events = [e for e in self.MARKET_EVENTS if int(e.date[:4]) >= 2020]
        if recent_events:
            lines.append("EVENTOS RECIENTES (2020+):")
            for event in recent_events[:3]:
                lines.append(f"  {event.date}: {event.description}")

        return "\n".join(lines)


# =============================================================================
# GDELT EVENTS (Historical News/Events Database)
# =============================================================================

class GDELTProvider:
    """
    Provider for GDELT (Global Database of Events, Language, and Tone).
    Free access to historical events and news.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2"

    def search_events(self, query: str, mode: str = 'ArtList',
                      max_records: int = 10) -> List[Dict]:
        """
        Search GDELT for events/articles.

        Args:
            query: Search query
            mode: 'ArtList' for articles, 'TimelineVol' for volume
            max_records: Maximum records to return

        Returns:
            List of articles/events
        """
        url = f"{self.BASE_URL}/doc/doc"
        params = {
            'query': query,
            'mode': mode,
            'maxrecords': max_records,
            'format': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'source': article.get('domain', ''),
                    'date': article.get('seendate', '')[:10],
                    'language': article.get('language', 'en')
                })

            return articles
        except Exception as e:
            print(f"GDELT error: {e}")
            return []

    def search_market_crash_news(self, start_date: str = None,
                                  end_date: str = None) -> List[Dict]:
        """Search for market crash related news"""
        query = '"market crash" OR "stock market crash" OR "financial crisis"'
        return self.search_events(query, max_records=20)

    def search_symbol_news(self, symbol: str, company_name: str = None) -> List[Dict]:
        """Search for news about a specific stock"""
        if company_name:
            query = f'"{company_name}" OR "{symbol} stock"'
        else:
            query = f'"{symbol} stock"'
        return self.search_events(query, max_records=15)


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    print("=== External Data Provider Test ===\n")

    provider = ExternalDataProvider()

    print("--- Economic Summary ---")
    print(provider.economic_summary())

    print("\n--- VIX Status ---")
    vix = provider.get_vix_status()
    if vix:
        print(f"VIX: {vix['vix']} ({vix['status']})")

    print("\n--- Historical Market Events ---")
    print(provider.events_summary("crash"))

    print("\n--- Search Events ---")
    events = provider.search_events("COVID")
    for e in events:
        print(f"  {e.date}: {e.description}")

    print("\n--- GDELT Test ---")
    gdelt = GDELTProvider()
    articles = gdelt.search_events("stock market", max_records=5)
    for a in articles:
        print(f"  [{a['date']}] {a['title'][:60]}...")
