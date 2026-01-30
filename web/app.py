"""
Streamlit Dashboard for Financial Data Visualization.
Run with: streamlit run web/app.py
"""

import sys
import logging
from pathlib import Path
import importlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force reload of portfolio_data to pick up changes
import src.portfolio_data
importlib.reload(src.portfolio_data)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date

from src.config import get_settings
from src.database import (
    get_db_manager, Symbol, Exchange, PriceHistory, DownloadLog,
    Fundamental, Portfolio, PortfolioHolding, DailyMetrics,
    IBAccount, IBTrade, IBFuturesTrade, AccountHolding, AccountCash
)
from src.yahoo_client import YahooFinanceClient, format_number, format_percent
from src.yahoo_downloader import YahooDownloader, DEFAULT_SYMBOLS
from src.analysis.ai_analyzer import AIAnalyzer, TechnicalAnalyzer
from src.technical import MetricsCalculator
from src.portfolio_data import (
    get_portfolio_service,
    ASSET_TYPE_MAP, CURRENCY_MAP, CURRENCY_SYMBOL_MAP,
)

# Page configuration
st.set_page_config(
    page_title="Financial Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize components
settings = get_settings()
db = get_db_manager()
analyzer = AIAnalyzer()
technical = TechnicalAnalyzer()
yahoo_client = YahooFinanceClient()
metrics_calc = MetricsCalculator()
portfolio_service = get_portfolio_service(db)

# =============================================================================
# PORTFOLIO VALUES - All values now come from database at runtime
# =============================================================================

# IB futures values now come from ib_futures_trades table via portfolio_service


# =============================================================================
# RUNTIME CALCULATIONS - These values are calculated from database prices
# =============================================================================

# Trading dates
TRADING_DATES = [
    date(2025, 12, 31), date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6),
    date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 12),
    date(2026, 1, 13), date(2026, 1, 14), date(2026, 1, 15), date(2026, 1, 16),
    date(2026, 1, 20), date(2026, 1, 21), date(2026, 1, 22), date(2026, 1, 23),
    date(2026, 1, 26), date(2026, 1, 27),
]

ASSET_TYPES_ORDER = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/Monetario/Bonos/Futuros']


def _get_price_or_previous(session, sym_id, target_date, max_lookback=5):
    """Get price for a date, or the previous available date if market was closed."""
    for i in range(max_lookback + 1):
        d = target_date - timedelta(days=i)
        p = session.query(PriceHistory).filter(
            PriceHistory.symbol_id == sym_id,
            PriceHistory.date == d
        ).first()
        if p:
            return p.close, d
    return None, None


def calculate_portfolio_by_type(target_date):
    """
    Calculate portfolio value by asset type for a given date.
    Uses holding_diario and posicion tables for accurate daily data.
    Returns dict: {asset_type: value_eur}
    """
    # Use the centralized service that reads from holding_diario and posicion
    return portfolio_service.get_values_by_asset_type(target_date)


def calculate_portfolio_total(target_date):
    """Calculate total portfolio value in EUR for a given date."""
    return sum(calculate_portfolio_by_type(target_date).values())


def calculate_all_trading_days():
    """
    Get portfolio totals for all trading days from posicion table.
    Excludes current date (incomplete data).
    Returns dict: {date: total_eur}
    """
    from sqlalchemy import text
    today = date.today()
    result = {}
    with db.get_session() as session:
        rows = session.execute(text("""
            SELECT fecha, SUM(total_eur) as total
            FROM posicion
            WHERE fecha < :today
            GROUP BY fecha
            ORDER BY fecha
        """), {'today': today.isoformat()})
        for row in rows.fetchall():
            # Convert string date to date object if needed
            fecha = row[0]
            if isinstance(fecha, str):
                fecha = datetime.strptime(fecha, '%Y-%m-%d').date()
            result[fecha] = row[1]
    return result


def get_account_totals_from_db(target_date: date):
    """Calculate account totals dynamically from holding_diario + prices."""
    all_holdings = portfolio_service.get_all_holdings_for_date(target_date)
    eur_usd = portfolio_service.get_eur_usd_rate(target_date)

    accounts = ['CO3365', 'RCO951', 'LACAIXA', 'IB']
    result = {}

    for account in accounts:
        holdings = all_holdings.get(account, {})
        holding_value = 0

        for symbol, data in holdings.items():
            shares = data['shares']
            value = portfolio_service.calculate_position_value(symbol, shares, target_date)
            if value:
                holding_value += value

        # Get cash
        cash_data = portfolio_service.get_cash_for_date(account, target_date)
        cash_eur = 0
        if cash_data:
            cash_eur += cash_data.get('EUR', 0)
            cash_eur += cash_data.get('USD', 0) / eur_usd

        result[account] = holding_value + cash_eur

    result['TOTAL'] = sum(result.values())
    return result


def get_rco951_breakdown_from_db(target_date: date):
    """Get RCO951 breakdown: stocks vs ETF gold."""
    eur_usd = portfolio_service.get_eur_usd_rate(target_date)

    # Get holdings from database
    rco951_holdings = portfolio_service.get_holdings_for_date('RCO951', target_date)

    # ETF Gold (SGLE.MI) - in EUR
    sgle_price = portfolio_service.get_symbol_price('SGLE.MI', target_date)
    sgle_shares = rco951_holdings.get('SGLE.MI', {}).get('shares', 0)
    etf_gold_eur = (sgle_price * sgle_shares) if sgle_price else 0

    # Stocks (all other positions)
    stocks_total = 0
    for symbol, data in rco951_holdings.items():
        if symbol == 'SGLE.MI':
            continue
        shares = data.get('shares', 0)
        val = portfolio_service.calculate_position_value(symbol, shares, target_date)
        if val:
            stocks_total += val

    return {
        'stocks': stocks_total,
        'etf_gold': etf_gold_eur,
    }


def get_ib_tlt_value_from_db(target_date: date):
    """Get TLT value in EUR from database."""
    eur_usd = portfolio_service.get_eur_usd_rate(target_date)
    tlt_price = portfolio_service.get_symbol_price('TLT', target_date)
    # Get TLT shares from holding_diario
    ib_holdings = portfolio_service.get_holdings_for_date('IB', target_date)
    tlt_shares = ib_holdings.get('TLT', {}).get('shares', 0)
    if tlt_price and eur_usd and tlt_shares:
        return (tlt_price * tlt_shares) / eur_usd
    return 0


def get_available_symbols() -> list[str]:
    """Get list of symbols with data in database."""
    with db.get_session() as session:
        symbols = (
            session.query(Symbol)
            .join(Exchange)
            .filter(Symbol.prices.any())
            .all()
        )
        return [f"{s.code}.{s.exchange.code}" for s in symbols]


def get_price_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Get price data for a symbol."""
    parts = symbol.split(".")
    symbol_code = parts[0]
    exchange_code = parts[1] if len(parts) > 1 else "US"

    with db.get_session() as session:
        db_symbol = db.get_symbol_by_code(session, symbol_code, exchange_code)
        if not db_symbol:
            return pd.DataFrame()

        start_date = datetime.now() - timedelta(days=days)
        return db.get_price_history(session, db_symbol.id, start_date=start_date)


def create_candlestick_chart(df: pd.DataFrame, symbol: str, show_volume: bool = True) -> go.Figure:
    """Create a candlestick chart with optional volume."""
    if show_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )
    else:
        fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1 if show_volume else None,
        col=1 if show_volume else None,
    )

    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20", line=dict(color="blue", width=1)),
            row=1 if show_volume else None,
            col=1 if show_volume else None,
        )
    if "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50", line=dict(color="orange", width=1)),
            row=1 if show_volume else None,
            col=1 if show_volume else None,
        )

    if show_volume and "volume" in df.columns:
        colors = ["red" if df["close"].iloc[i] < df["open"].iloc[i] else "green" for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
    )

    return fig


def create_indicator_chart(df: pd.DataFrame, indicator: str) -> go.Figure:
    """Create a chart for a specific indicator."""
    fig = go.Figure()

    if indicator == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="purple")))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title="RSI (14)", yaxis_range=[0, 100])

    elif indicator == "MACD":
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="orange")))
        fig.add_trace(
            go.Bar(x=df.index, y=df["macd_histogram"], name="Histogram", marker_color="gray")
        )
        fig.update_layout(title="MACD")

    elif indicator == "Bollinger":
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="white")))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="Upper Band", line=dict(color="red", dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_middle"], name="Middle Band", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="Lower Band", line=dict(color="green", dash="dash")))
        fig.update_layout(title="Bollinger Bands")

    fig.update_layout(height=300, template="plotly_dark")
    return fig


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("Financial Data Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("**Data Source:** Yahoo Finance")

page = st.sidebar.radio(
    "Navigation",
    ["Posición", "Composición", "Acciones/ETF", "Futuros", "Estrategia Mensual", "Screener", "Portfolios", "Symbol Analysis", "Data Management", "Download Status"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Programación**")
page_prog = st.sidebar.radio(
    "Técnico",
    ["", "BBDD", "Pantalla"],
    label_visibility="collapsed"
)
if page_prog:
    page = page_prog

# =============================================================================
# Main Content
# =============================================================================

if page == "Posición":
    st.title("CARTERA LA CARIHUELA")

    # =========================================================================
    # OBTENER VALORES DESDE TABLA POSICION (BD)
    # =========================================================================
    # Fecha inicial: 31/12/2025
    # Fecha actual: último día con datos (día anterior al actual)
    # =========================================================================

    with db.get_session() as session:
        from sqlalchemy import text

        # Obtener última fecha disponible en posicion (día anterior = último con datos)
        result = session.execute(text("""
            SELECT MAX(fecha) FROM posicion WHERE fecha < date('now')
        """))
        latest_date_str = result.fetchone()[0]
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date() if latest_date_str else date(2026, 1, 28)

        # Obtener fecha anterior para comparación diaria
        result = session.execute(text("""
            SELECT MAX(fecha) FROM posicion WHERE fecha < :latest
        """), {'latest': latest_date_str})
        prev_date_str = result.fetchone()[0]
        prev_date = datetime.strptime(prev_date_str, '%Y-%m-%d').date() if prev_date_str else latest_date

        # Valor inicial 31/12/2025 desde tabla posicion
        result = session.execute(text("""
            SELECT SUM(total_eur) FROM posicion WHERE fecha = '2025-12-31'
        """))
        initial_value = result.fetchone()[0] or 0

    # Valor actual calculado dinámicamente desde holding_diario + precios
    current_value = calculate_portfolio_total(latest_date)

    # Calcular rentabilidad
    return_eur = current_value - initial_value
    return_pct = (return_eur / initial_value) * 100 if initial_value > 0 else 0

    # EUR/USD rates from service
    eur_usd_31dic = portfolio_service.get_exchange_rate('EURUSD=X', date(2025, 12, 31)) or 1.1747
    eur_usd_current = portfolio_service.get_eur_usd_rate(latest_date)

    # Get benchmark returns from database
    start_date = datetime(2025, 12, 31)
    today = datetime.now().date()

    latest_data_date = latest_date.strftime('%d/%m') if latest_date else "28/01"

    with db.get_session() as session:
        spy_symbol = session.query(Symbol).filter(Symbol.code == 'SPY').first()
        qqq_symbol = session.query(Symbol).filter(Symbol.code == 'QQQ').first()

        spy_return = 0
        qqq_return = 0

        if spy_symbol:
            spy_data = db.get_price_history(session, spy_symbol.id, start_date=start_date)
            if not spy_data.empty:
                spy_data = spy_data[spy_data.index.date < today]
                if len(spy_data) >= 2:
                    spy_return = ((spy_data['close'].iloc[-1] - spy_data['close'].iloc[0]) / spy_data['close'].iloc[0]) * 100

        if qqq_symbol:
            qqq_data = db.get_price_history(session, qqq_symbol.id, start_date=start_date)
            if not qqq_data.empty:
                qqq_data = qqq_data[qqq_data.index.date < today]
                if len(qqq_data) >= 2:
                    qqq_return = ((qqq_data['close'].iloc[-1] - qqq_data['close'].iloc[0]) / qqq_data['close'].iloc[0]) * 100

    # Helper function for Spanish number format
    def format_eur(value, show_sign=False):
        """Format number in Spanish style with € symbol"""
        if show_sign:
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:,.0f} €".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{value:,.0f} €".replace(",", "X").replace(".", ",").replace("X", ".")

    def format_pct(value):
        """Format percentage with sign"""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"

    # Metrics display
    st.subheader("Resumen de Cartera")

    def format_usd(value, show_sign=False):
        """Format number in Spanish style with $ symbol"""
        if show_sign:
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:,.0f} $".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{value:,.0f} $".replace(",", "X").replace(".", ",").replace("X", ".")

    # Calculate USD values with respective exchange rates
    initial_usd = initial_value * eur_usd_31dic
    current_usd = current_value * eur_usd_current
    return_usd = current_usd - initial_usd
    return_pct_usd = (return_usd / initial_usd) * 100 if initial_usd > 0 else 0

    # Compact layout using HTML table
    st.markdown(f"""
    <style>
        .portfolio-table {{ width: 100%; border-collapse: collapse; }}
        .portfolio-table th {{ text-align: left; padding: 5px 10px; font-weight: bold; font-size: 14px; }}
        .portfolio-table td {{ padding: 2px 10px; vertical-align: top; }}
        .eur-value {{ font-size: 24px; font-weight: bold; display: block; }}
        .usd-value {{ font-size: 20px; display: block; }}
        .fx-rate {{ font-size: 11px; color: #808080; }}
        .green {{ color: #00cc00; }}
        .benchmark {{ font-size: 22px; font-weight: bold; color: #00cc00; }}
    </style>
    <table class="portfolio-table">
        <tr>
            <th>Valor Inicial 31/12</th>
            <th>Valor Actual {latest_data_date}</th>
            <th>Ganancia Acum. {date.today().year}</th>
            <th>Rent. Acum. {date.today().year}</th>
            <th>SPY Acum. {date.today().year}</th>
            <th>QQQ Acum. {date.today().year}</th>
        </tr>
        <tr>
            <td>
                <span class="eur-value">{format_eur(initial_value)}</span>
                <span class="usd-value">{format_usd(initial_usd)}</span>
                <span class="fx-rate">(EUR/USD {eur_usd_31dic})</span>
            </td>
            <td>
                <span class="eur-value">{format_eur(current_value)}</span>
                <span class="usd-value">{format_usd(current_usd)}</span>
                <span class="fx-rate">(EUR/USD {eur_usd_current})</span>
            </td>
            <td>
                <span class="eur-value green">{format_eur(return_eur, show_sign=True)}</span>
                <span class="usd-value green">{format_usd(return_usd, show_sign=True)}</span>
            </td>
            <td>
                <span class="eur-value green">{format_pct(return_pct)}</span>
                <span class="usd-value green">{format_pct(return_pct_usd)}</span>
            </td>
            <td><span class="benchmark">{format_pct(spy_return)}</span></td>
            <td><span class="benchmark">{format_pct(qqq_return)}</span></td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Daily Comparison by Asset Type (using centralized service)
    st.subheader("Variación Diaria por Tipo de Activo")

    # Use dates from service
    day_prev = prev_date
    day_last = latest_date

    if day_prev and day_last:
        # Get values by asset type for both dates (direct calculation)
        prev_totals = calculate_portfolio_by_type(day_prev)
        last_totals = calculate_portfolio_by_type(day_last)

        # Build comparison table (Futuros ya incluido en Cash/Monetario/Bonos/Futuros)
        comparison_data = []
        tipos = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/Monetario/Bonos/Futuros']
        total_prev = 0
        total_last = 0

        for tipo in tipos:
            v_prev = prev_totals.get(tipo, 0)
            v_last = last_totals.get(tipo, 0)
            diff = v_last - v_prev
            diff_pct = (diff / v_prev * 100) if v_prev > 0 else 0
            total_prev += v_prev
            total_last += v_last
            comparison_data.append({
                'Tipo': tipo,
                f'{day_prev.strftime("%d/%m")}': f"{v_prev:,.0f} €",
                f'{day_last.strftime("%d/%m")}': f"{v_last:,.0f} €",
                'Diferencia': f"{diff:+,.0f} €",
                'Var %': f"{diff_pct:+.2f}%"
            })

        # Add total row
        total_diff = total_last - total_prev
        total_diff_pct = (total_diff / total_prev * 100) if total_prev > 0 else 0
        comparison_data.append({
            'Tipo': 'TOTAL',
            f'{day_prev.strftime("%d/%m")}': f"{total_prev:,.0f} €",
            f'{day_last.strftime("%d/%m")}': f"{total_last:,.0f} €",
            'Diferencia': f"{total_diff:+,.0f} €",
            'Var %': f"{total_diff_pct:+.2f}%"
        })

        comp_df = pd.DataFrame(comparison_data)

        # Display table with styling
        def color_diff(val):
            if isinstance(val, str) and ('€' in val or '%' in val):
                if val.startswith('+'):
                    return 'color: #00cc00'
                elif val.startswith('-'):
                    return 'color: #ff4444'
            return ''

        def highlight_total(row):
            if row['Tipo'] == 'TOTAL':
                return ['background-color: #5a5a5a; font-weight: bold; color: white'] * len(row)
            return [''] * len(row)

        styled_comp = comp_df.style.map(color_diff, subset=['Diferencia', 'Var %']).apply(highlight_total, axis=1)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(styled_comp, use_container_width=True, hide_index=True)

        with col2:
            # Bar chart for differences
            chart_data = [d for d in comparison_data if d['Tipo'] != 'TOTAL']
            tipos_chart = [d['Tipo'] for d in chart_data]
            diffs = [float(d['Diferencia'].replace(' €', '').replace(',', '').replace('+', '')) for d in chart_data]
            colors = ['#00cc00' if d >= 0 else '#ff4444' for d in diffs]

            fig_diff = go.Figure(data=[
                go.Bar(x=tipos_chart, y=diffs, marker_color=colors, text=[f"{d:+,.0f}€" for d in diffs], textposition='outside')
            ])
            fig_diff.update_layout(
                title=f'Variación {day_prev.strftime("%d/%m")} → {day_last.strftime("%d/%m")}',
                height=350,
                template='plotly_dark',
                yaxis_title='EUR',
                showlegend=False
            )
            st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown("---")

    # Performance Chart vs Benchmark
    st.subheader(f"Rentabilidad Acumulada {date.today().year} vs Benchmark (desde 31/12/{date.today().year - 1})")

    # Get benchmark daily data (start from 30/12 to ensure 31/12 is included)
    query_start = datetime(2025, 12, 30)

    with db.get_session() as session:
        spy_symbol = session.query(Symbol).filter(Symbol.code == 'SPY').first()
        qqq_symbol = session.query(Symbol).filter(Symbol.code == 'QQQ').first()
        eurusd_symbol = session.query(Symbol).filter(Symbol.code == 'EURUSD=X').first()

        spy_prices = pd.DataFrame()
        qqq_prices = pd.DataFrame()
        eurusd_prices = pd.DataFrame()

        if spy_symbol:
            spy_prices = db.get_price_history(session, spy_symbol.id, start_date=query_start)
            if not spy_prices.empty:
                # Filter: from 31/12/2025 to before today, weekdays only (no weekends)
                spy_prices = spy_prices[
                    (spy_prices.index.date >= date(2025, 12, 31)) &
                    (spy_prices.index.date < today) &
                    (spy_prices.index.dayofweek < 5)  # 0=Mon, 4=Fri
                ]

        if qqq_symbol:
            qqq_prices = db.get_price_history(session, qqq_symbol.id, start_date=query_start)
            if not qqq_prices.empty:
                # Filter: weekdays only
                qqq_prices = qqq_prices[
                    (qqq_prices.index.date >= date(2025, 12, 31)) &
                    (qqq_prices.index.date < today) &
                    (qqq_prices.index.dayofweek < 5)
                ]

        if eurusd_symbol:
            eurusd_prices = db.get_price_history(session, eurusd_symbol.id, start_date=query_start)
            if not eurusd_prices.empty:
                eurusd_prices = eurusd_prices[(eurusd_prices.index.date >= date(2025, 12, 31)) & (eurusd_prices.index.date < today)]

    # Calculate portfolio returns from database prices (from posicion table)
    all_day_totals = calculate_all_trading_days()
    initial_value = all_day_totals.get(date(2025, 12, 31), portfolio_service.get_initial_total())

    # Get sorted list of trading dates from posicion table
    trading_dates_from_db = sorted(all_day_totals.keys())

    portfolio_dates = []
    portfolio_eur_values = []
    portfolio_usd_values = []

    # Fechas a excluir del cuadro y gráfica
    excluded_dates = {date(2026, 1, 1), date(2026, 1, 19)}

    for td in trading_dates_from_db:
        # Skip weekends and excluded dates
        if td.weekday() >= 5 or td in excluded_dates:
            continue
        total_val = all_day_totals.get(td, 0)
        if total_val > 0 and initial_value > 0:
            ret_eur = ((total_val / initial_value) - 1) * 100
            # USD return = EUR return + FX effect
            eurusd_td = portfolio_service.get_exchange_rate('EURUSD=X', td) or eur_usd_31dic
            ret_usd = ret_eur + ((eurusd_td / eur_usd_31dic) - 1) * 100
            portfolio_dates.append(datetime(td.year, td.month, td.day))
            portfolio_eur_values.append(ret_eur)
            portfolio_usd_values.append(ret_usd)

    if portfolio_dates:
        portfolio_returns_eur = pd.Series(portfolio_eur_values, index=portfolio_dates)
        portfolio_returns_usd = pd.Series(portfolio_usd_values, index=portfolio_dates)
    else:
        portfolio_returns_eur = pd.Series([0, return_pct], index=[datetime(2025, 12, 31), datetime.now()])
        portfolio_returns_usd = pd.Series([0, return_pct_usd], index=[datetime(2025, 12, 31), datetime.now()])

    # Combined chart with 4 lines: EUR, USD, SPY, QQQ
    fig = go.Figure()

    # 1. Cartera EUR - daily data
    fig.add_trace(go.Scatter(
        x=portfolio_returns_eur.index,
        y=portfolio_returns_eur.values,
        mode='lines+markers',
        name=f'Cartera EUR ({return_pct:+.2f}%)',
        line=dict(color='#00cc00', width=3),
        marker=dict(size=6)
    ))

    # 2. Cartera USD - daily data
    if isinstance(portfolio_returns_usd, pd.Series):
        fig.add_trace(go.Scatter(
            x=portfolio_returns_usd.index,
            y=portfolio_returns_usd.values,
            mode='lines+markers',
            name=f'Cartera USD ({return_pct_usd:+.2f}%)',
            line=dict(color='#90EE90', width=3, dash='dash'),
            marker=dict(size=6)
        ))

    # 3. SPY benchmark
    if not spy_prices.empty and len(spy_prices) >= 2:
        spy_base = spy_prices['close'].iloc[0]
        spy_returns_series = ((spy_prices['close'] - spy_base) / spy_base) * 100
        fig.add_trace(go.Scatter(
            x=spy_prices.index,
            y=spy_returns_series,
            mode='lines',
            name=f'SPY ({spy_return:+.2f}%)',
            line=dict(color='#636EFA', width=2)
        ))

    # 4. QQQ benchmark
    if not qqq_prices.empty and len(qqq_prices) >= 2:
        qqq_base = qqq_prices['close'].iloc[0]
        qqq_returns_series = ((qqq_prices['close'] - qqq_base) / qqq_base) * 100
        fig.add_trace(go.Scatter(
            x=qqq_prices.index,
            y=qqq_returns_series,
            mode='lines',
            name=f'QQQ ({qqq_return:+.2f}%)',
            line=dict(color='#EF553B', width=2)
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f'Rentabilidad Acumulada {date.today().year}: Cartera vs Benchmark',
        height=450,
        template='plotly_dark',
        yaxis_title='Rentabilidad %',
        xaxis_title='',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily returns table
    st.subheader(f"Rentabilidad Diaria (desde 31/12/{date.today().year - 1})")

    # Use totals from posicion table
    initial_val_eur = all_day_totals.get(date(2025, 12, 31), portfolio_service.get_initial_total())
    initial_val_usd = initial_val_eur * eur_usd_31dic  # Valor inicial en USD
    daily_returns_data = []

    for d in trading_dates_from_db:
        # Skip weekends and excluded dates (1/1, 19/1)
        if d.weekday() >= 5 or d in excluded_dates:
            continue

        total_val_eur = all_day_totals.get(d, 0)
        eurusd_td = portfolio_service.get_exchange_rate('EURUSD=X', d) or eur_usd_31dic
        total_val_usd = total_val_eur * eurusd_td  # Valor del día en USD

        if total_val_eur > 0 and initial_val_eur > 0:
            port_eur_ret = ((total_val_eur / initial_val_eur) - 1) * 100
            port_usd_ret = ((total_val_usd / initial_val_usd) - 1) * 100
        else:
            port_eur_ret = 0.0
            port_usd_ret = 0.0

        # Get SPY/QQQ returns for this date
        spy_ret = 0.0
        qqq_ret = 0.0
        if not spy_prices.empty:
            spy_base = spy_prices['close'].iloc[0]
            matching = spy_prices[spy_prices.index.date == d]
            if not matching.empty:
                spy_ret = ((matching['close'].iloc[0] - spy_base) / spy_base) * 100
        if not qqq_prices.empty:
            qqq_base = qqq_prices['close'].iloc[0]
            matching = qqq_prices[qqq_prices.index.date == d]
            if not matching.empty:
                qqq_ret = ((matching['close'].iloc[0] - qqq_base) / qqq_base) * 100

        row = {
            'Fecha': d.strftime('%d/%m/%Y'),
            'Valor EUR': f"{total_val_eur:,.0f}",
            'Valor USD': f"{total_val_usd:,.0f}",
            'Cartera EUR': port_eur_ret,
            'Cartera USD': port_usd_ret,
            'SPY': spy_ret,
            'QQQ': qqq_ret,
        }
        daily_returns_data.append(row)

    # Create dataframe (most recent first)
    daily_returns_df = pd.DataFrame(daily_returns_data[::-1])

    # Style function for coloring percentages
    def color_pct(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'background-color: #2E7D32; color: white'  # Green (softer)
            elif val < 0:
                return 'background-color: #C62828; color: white'  # Red (softer)
        return ''

    # Format percentages for display
    pct_cols = ['Cartera EUR', 'Cartera USD', 'SPY', 'QQQ']
    styled_df = daily_returns_df.style.map(color_pct, subset=pct_cols).format({
        'Cartera EUR': '{:+.2f}%',
        'Cartera USD': '{:+.2f}%',
        'SPY': '{:+.2f}%',
        'QQQ': '{:+.2f}%',
    })

    # Display table
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=700)


elif page == "Composición":
    st.title("COMPOSICIÓN DE CARTERA")

    # Get latest date (mismo método que Posición)
    with db.get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("""
            SELECT MAX(fecha) FROM posicion WHERE fecha < date('now')
        """))
        latest_date_str = result.fetchone()[0]
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date() if latest_date_str else date.today()

    # Get account totals calculated dynamically from holding_diario + prices
    account_totals = get_account_totals_from_db(latest_date)
    CO3365_TOTAL = account_totals.get('CO3365', 0)
    RCO951_TOTAL = account_totals.get('RCO951', 0)
    LACAIXA_TOTAL = account_totals.get('LACAIXA', 0)
    IB_TOTAL = account_totals.get('IB', 0)
    total_cartera = account_totals.get('TOTAL', 0)

    # Get values by strategy (asset type) - mismo método que Variación Diaria
    strategy_values = calculate_portfolio_by_type(latest_date)

    st.markdown(f"**Fecha:** {latest_date.strftime('%d/%m/%Y')} | **Total Cartera:** EUR {total_cartera:,.0f}")
    st.markdown("---")

    # =========================================================================
    # 1. COMPOSICIÓN POR DIVERSIFICACIÓN
    # =========================================================================
    st.subheader("Composición por Diversificación")

    col1, col2 = st.columns(2)

    # Diversificación: Bolsa, Oro, Liquidez (usando strategy_values)
    bolsa = (strategy_values.get('Quant', 0) +
             strategy_values.get('Value', 0) +
             strategy_values.get('Alpha Picks', 0) +
             strategy_values.get('Mensual', 0))
    oro = strategy_values.get('Oro/Mineras', 0)
    liquidez = strategy_values.get('Cash/Monetario/Bonos/Futuros', 0)

    diversificacion_data = {
        'Bolsa': bolsa,
        'Oro': oro,
        'Liquidez': liquidez,
    }

    with col1:
        fig_div = go.Figure(data=[go.Pie(
            labels=list(diversificacion_data.keys()),
            values=list(diversificacion_data.values()),
            hole=0.4,
            textposition='inside',
            textinfo='label+percent',
            textfont_size=14,
            marker_colors=['#636EFA', '#FFA15A', '#00CC96']
        )])
        fig_div.update_layout(
            title='Diversificación por Clase de Activo',
            height=400,
            template='plotly_dark',
            showlegend=False
        )
        st.plotly_chart(fig_div, use_container_width=True)

    with col2:
        div_table = []
        for tipo, value in diversificacion_data.items():
            pct = (value / total_cartera) * 100 if total_cartera > 0 else 0
            div_table.append({
                'Clase': tipo,
                'Valor EUR': f"{value:,.0f}",
                '% Peso': f"{pct:.1f}%"
            })
        div_table.append({
            'Clase': 'TOTAL',
            'Valor EUR': f"{sum(diversificacion_data.values()):,.0f}",
            '% Peso': "100.0%"
        })
        st.dataframe(pd.DataFrame(div_table), use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================================================================
    # 2. COMPOSICIÓN POR ESTRATEGIA
    # =========================================================================
    st.subheader("Composición por Estrategia")

    col1, col2 = st.columns(2)

    # Usar valores del servicio (ya agrupados por tipo de activo/estrategia)
    estrategia_data = {k: v for k, v in strategy_values.items() if v > 0}

    with col1:
        fig_est = go.Figure(data=[go.Pie(
            labels=list(estrategia_data.keys()),
            values=list(estrategia_data.values()),
            hole=0.4,
            textposition='inside',
            textinfo='label+percent',
            textfont_size=11
        )])
        fig_est.update_layout(
            title='Distribución por Estrategia',
            height=400,
            template='plotly_dark',
            showlegend=False
        )
        st.plotly_chart(fig_est, use_container_width=True)

    with col2:
        est_table = []
        total_estrategias = sum(estrategia_data.values())
        for estrategia, value in sorted(estrategia_data.items(), key=lambda x: -x[1]):
            pct = (value / total_estrategias) * 100 if total_estrategias > 0 else 0
            est_table.append({
                'Estrategia': estrategia,
                'Valor EUR': f"{value:,.0f}",
                '% Peso': f"{pct:.1f}%"
            })
        est_table.append({
            'Estrategia': 'TOTAL',
            'Valor EUR': f"{total_estrategias:,.0f}",
            '% Peso': "100.0%"
        })
        st.dataframe(pd.DataFrame(est_table), use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================================================================
    # 3. COMPOSICIÓN POR CUENTA
    # =========================================================================
    st.subheader("Composición por Cuenta")

    col1, col2 = st.columns(2)

    cuenta_data = {
        'CO3365': CO3365_TOTAL,
        'RCO951': RCO951_TOTAL,
        'La Caixa': LACAIXA_TOTAL,
        'Interactive Brokers': IB_TOTAL,
    }

    with col1:
        fig_cuenta = go.Figure(data=[go.Pie(
            labels=list(cuenta_data.keys()),
            values=list(cuenta_data.values()),
            hole=0.4,
            textposition='inside',
            textinfo='label+percent',
            textfont_size=12
        )])
        fig_cuenta.update_layout(
            title='Distribución por Cuenta',
            height=400,
            template='plotly_dark',
            showlegend=False
        )
        st.plotly_chart(fig_cuenta, use_container_width=True)

    with col2:
        cuenta_table = []
        for cuenta, value in cuenta_data.items():
            pct = (value / total_cartera) * 100 if total_cartera > 0 else 0
            cuenta_table.append({
                'Cuenta': cuenta,
                'Valor EUR': f"{value:,.0f}",
                '% Peso': f"{pct:.1f}%"
            })
        cuenta_table.append({
            'Cuenta': 'TOTAL',
            'Valor EUR': f"{total_cartera:,.0f}",
            '% Peso': "100.0%"
        })
        st.dataframe(pd.DataFrame(cuenta_table), use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================================================================
    # 4. DETALLE POR CUENTA
    # =========================================================================
    st.subheader("Detalle por Cuenta")

    tab1, tab2, tab3, tab4 = st.tabs(["CO3365", "RCO951", "La Caixa", "Interactive Brokers"])

    with tab1:
        st.markdown("**CO3365 - Acciones Mensual**")
        co3365_table = []
        co3365_holdings_total = 0
        co3365_holdings = portfolio_service.get_holdings_for_date('CO3365', latest_date)
        co3365_cash = portfolio_service.get_cash_for_date('CO3365', latest_date)
        for ticker, data in co3365_holdings.items():
            shares = data.get('shares', 0)
            val_eur = portfolio_service.calculate_position_value(ticker, shares, latest_date) or 0
            co3365_holdings_total += val_eur
            co3365_table.append({
                'Ticker': ticker,
                'Titulos': int(shares),
                'Valor EUR': f"{val_eur:,.2f}"
            })
        co3365_cash_eur = co3365_cash.get('EUR', 0)
        co3365_cash_usd = co3365_cash.get('USD', 0)
        co3365_cash_usd_eur = co3365_cash_usd / portfolio_service.get_eur_usd_rate(latest_date) if co3365_cash_usd else 0
        co3365_table.append({'Ticker': 'Cash EUR', 'Titulos': '-', 'Valor EUR': f"{co3365_cash_eur:,.2f}"})
        co3365_table.append({'Ticker': 'Cash USD', 'Titulos': '-', 'Valor EUR': f"{co3365_cash_usd_eur:,.2f}"})
        co3365_table.append({'Ticker': 'TOTAL', 'Titulos': '-', 'Valor EUR': f"{CO3365_TOTAL:,.2f}"})
        st.dataframe(pd.DataFrame(co3365_table), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**RCO951 - Growth + Oro**")
        rco951_holdings = portfolio_service.get_holdings_for_date('RCO951', latest_date)
        rco951_cash = portfolio_service.get_cash_for_date('RCO951', latest_date)
        rco951_cash_eur = rco951_cash.get('EUR', 0)
        rco951_cash_usd = rco951_cash.get('USD', 0)
        rco951_cash_usd_eur = rco951_cash_usd / portfolio_service.get_eur_usd_rate(latest_date) if rco951_cash_usd else 0

        # Calcular ETF Gold (SGLE.MI)
        sgle_shares = rco951_holdings.get('SGLE.MI', {}).get('shares', 0)
        sgle_price = portfolio_service.get_symbol_price('SGLE.MI', latest_date) or 0
        rco951_etf_gold = sgle_price * sgle_shares

        # Calcular stocks (todo menos SGLE.MI)
        rco951_stocks = 0
        rco951_num_stocks = 0
        for sym, data in rco951_holdings.items():
            if sym != 'SGLE.MI':
                val = portfolio_service.calculate_position_value(sym, data.get('shares', 0), latest_date) or 0
                rco951_stocks += val
                rco951_num_stocks += 1

        st.markdown(f"- Acciones ({rco951_num_stocks} posiciones): EUR {rco951_stocks:,.2f}")
        st.markdown(f"- ETF Oro (SGLE.MI, {int(sgle_shares):,} titulos): EUR {rco951_etf_gold:,.2f}")
        st.markdown(f"- Cash EUR: EUR {rco951_cash_eur:,.2f}")
        st.markdown(f"- Cash USD: EUR {rco951_cash_usd_eur:,.2f}")
        st.markdown(f"**TOTAL: EUR {RCO951_TOTAL:,.2f}**")

    with tab3:
        st.markdown("**La Caixa**")
        lacaixa_table = []
        lacaixa_holdings_total = 0
        lacaixa_holdings = portfolio_service.get_holdings_for_date('LACAIXA', latest_date)
        for ticker, data in lacaixa_holdings.items():
            shares = data.get('shares', 0)
            val_eur = portfolio_service.calculate_position_value(ticker, shares, latest_date) or 0
            lacaixa_holdings_total += val_eur
            lacaixa_table.append({
                'Ticker': ticker,
                'Titulos': int(shares),
                'Valor EUR': f"{val_eur:,.2f}"
            })
        lacaixa_table.append({'Ticker': 'TOTAL', 'Titulos': '-', 'Valor EUR': f"{LACAIXA_TOTAL:,.2f}"})
        st.dataframe(pd.DataFrame(lacaixa_table), use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("**Interactive Brokers (U17236599)**")
        tlt_price = portfolio_service.get_symbol_price('TLT', latest_date) or 88.35
        ib_cash = portfolio_service.get_cash_for_date('IB', latest_date)
        ib_cash_net_eur = ib_cash.get('EUR', 0)
        ib_holdings = portfolio_service.get_holdings_for_date('IB', latest_date)
        ib_tlt_shares = int(ib_holdings.get('TLT', {}).get('shares', 0))
        ib_tlt_value_eur = (tlt_price * ib_tlt_shares / portfolio_service.get_eur_usd_rate(latest_date)) if ib_tlt_shares else 0
        st.markdown(f"- TLT ({ib_tlt_shares:,} shares @ ${tlt_price:.2f}): EUR {ib_tlt_value_eur:,.2f}")
        st.markdown(f"- Cash Neto: EUR {ib_cash_net_eur:,.2f}")
        st.markdown(f"**TOTAL: EUR {IB_TOTAL:,.2f}**")

        st.markdown("---")
        st.markdown("**Operaciones de Futuros (Day Trading)**")
        eur_usd_rate = portfolio_service.get_eur_usd_rate(latest_date)
        futures_summary = portfolio_service.get_futures_summary()
        futures_by_contract = futures_summary['by_contract']
        futures_data = []
        for contract, data in futures_by_contract.items():
            pnl_usd = data['realized_usd']
            pnl_eur = pnl_usd / eur_usd_rate
            sign = '+' if pnl_usd >= 0 else ''
            futures_data.append({
                'Contrato': f"{contract} (Gold)",
                'P&L USD': f"{sign}${pnl_usd:,.2f}",
                'P&L EUR': f"{sign}EUR {pnl_eur:,.2f}"
            })
        # Add total row
        total_usd = futures_summary['total_realized_usd']
        total_eur = futures_summary['total_realized_eur']
        sign = '+' if total_usd >= 0 else ''
        futures_data.append({
            'Contrato': 'TOTAL',
            'P&L USD': f"{sign}${total_usd:,.2f}",
            'P&L EUR': f"{sign}EUR {total_eur:,.2f}"
        })
        st.dataframe(pd.DataFrame(futures_data), use_container_width=True, hide_index=True)


elif page == "Acciones/ETF":
    st.title("ACCIONES / ETF")

    # Using centralized ASSET_TYPE_MAP and CURRENCY_SYMBOL_MAP from portfolio_data

    # Date range
    today = date.today()
    start_date = datetime(2025, 12, 30)

    # Get latest trading date (mismo método que Posición)
    with db.get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("""
            SELECT MAX(fecha) FROM posicion WHERE fecha < date('now')
        """))
        latest_date_str = result.fetchone()[0]
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date() if latest_date_str else today

    # EUR/USD exchange rates from service
    eur_usd_31dic = portfolio_service.get_exchange_rate('EURUSD=X', date(2025, 12, 31)) or 1.1747
    eur_usd_current = portfolio_service.get_eur_usd_rate(latest_date)

    # CAD and CHF exchange rates from service
    cad_eur_31dic = portfolio_service.get_cad_eur_rate(date(2025, 12, 31))
    cad_eur_current = portfolio_service.get_cad_eur_rate(latest_date)
    chf_eur_31dic = portfolio_service.get_chf_eur_rate(date(2025, 12, 31))
    chf_eur_current = portfolio_service.get_chf_eur_rate(latest_date)

    # Exchange code to currency symbol mapping
    EXCHANGE_TO_CURRENCY = {'US': '$', 'TO': 'C$', 'MC': '€', 'SW': 'CHF', 'L': '£', 'DE': '€', 'F': '€', 'MI': '€'}

    with db.get_session() as session:
        # Collect all holdings from all accounts (from database)
        holdings_data = []

        # Get all holdings from database for the latest date
        all_holdings = portfolio_service.get_all_holdings_for_date(latest_date)

        # CO3365 (all USD)
        co3365_holdings = all_holdings.get('CO3365', {})
        for ticker, data in co3365_holdings.items():
            shares = data.get('shares', 0)
            holdings_data.append({
                'ticker': ticker,
                'ticker_full': ticker,
                'cuenta': 'CO3365',
                'shares': shares,
                'valor_actual': 0,  # Will be calculated from prices
                'currency': '$',
                'tipo': ASSET_TYPE_MAP.get(ticker, 'Otro')
            })

        # La Caixa (mixed currencies)
        lacaixa_holdings = all_holdings.get('LACAIXA', {})
        for ticker, data in lacaixa_holdings.items():
            shares = data.get('shares', 0)
            parts = ticker.split('.')
            base_ticker = parts[0]
            exchange = parts[1] if len(parts) > 1 else 'US'
            currency = EXCHANGE_TO_CURRENCY.get(exchange, '$')

            holdings_data.append({
                'ticker': base_ticker,
                'ticker_full': ticker,
                'cuenta': 'La Caixa',
                'shares': shares,
                'valor_actual': 0,  # Will be calculated from prices
                'currency': currency,
                'tipo': ASSET_TYPE_MAP.get(ticker, ASSET_TYPE_MAP.get(base_ticker, 'Otro'))
            })

        # RCO951 (mostly USD, some EUR)
        rco951_holdings = all_holdings.get('RCO951', {})
        for ticker, data in rco951_holdings.items():
            shares = data.get('shares', 0)
            parts = ticker.split('.')
            base_ticker = parts[0]
            exchange = parts[1] if len(parts) > 1 else 'US'
            currency = EXCHANGE_TO_CURRENCY.get(exchange, '$')

            holdings_data.append({
                'ticker': base_ticker,
                'ticker_full': ticker,
                'cuenta': 'RCO951',
                'shares': shares,
                'valor_actual': 0,
                'currency': currency,
                'tipo': ASSET_TYPE_MAP.get(ticker, ASSET_TYPE_MAP.get(base_ticker, 'Otro'))
            })

        # Calculate returns for each holding
        asset_returns = []
        for holding in holdings_data:
            ticker = holding['ticker']
            ticker_full = holding['ticker_full']
            # For non-US stocks (like IAG.MC), look for the full ticker first
            if '.' in ticker_full:
                symbol = session.query(Symbol).filter(Symbol.code == ticker_full).first()
                if not symbol:
                    symbol = session.query(Symbol).filter(Symbol.code == ticker).first()
            else:
                symbol = session.query(Symbol).filter(Symbol.code == ticker).first()

            if symbol:
                prices = db.get_price_history(session, symbol.id, start_date=start_date)
                if not prices.empty and len(prices) >= 2:
                    prices = prices[prices.index.date < today]
                    if len(prices) >= 2:
                        precio_inicial = prices['close'].iloc[0]
                        precio_actual = prices['close'].iloc[-1]
                        rentabilidad_local = ((precio_actual - precio_inicial) / precio_inicial) * 100
                        currency = holding['currency']
                        shares = holding['shares']

                        # Calculate EUR values and return
                        if currency == '$':  # USD
                            valor_inicial_eur = (shares * precio_inicial) / eur_usd_31dic
                            valor_actual_eur = (shares * precio_actual) / eur_usd_current
                        elif currency == 'C$':  # CAD - use database rate
                            valor_inicial_eur = (shares * precio_inicial) * cad_eur_31dic
                            valor_actual_eur = (shares * precio_actual) * cad_eur_current
                        elif currency == 'CHF':  # Swiss Franc - use database rate
                            valor_inicial_eur = (shares * precio_inicial) * chf_eur_31dic
                            valor_actual_eur = (shares * precio_actual) * chf_eur_current
                        else:  # EUR or other
                            valor_inicial_eur = shares * precio_inicial
                            valor_actual_eur = shares * precio_actual

                        rentabilidad_eur_pct = ((valor_actual_eur - valor_inicial_eur) / valor_inicial_eur) * 100
                        rentabilidad_eur_abs = valor_actual_eur - valor_inicial_eur  # Absolute EUR gain

                        asset_returns.append({
                            'Ticker': holding['ticker_full'],
                            'Tipo': holding['tipo'],
                            'Cuenta': holding['cuenta'],
                            'Titulos': holding['shares'],
                            'Precio 31/12': f"{currency}{precio_inicial:.2f}",
                            'Precio Actual': f"{currency}{precio_actual:.2f}",
                            'Rent. %': rentabilidad_local,
                            'Rent. EUR': rentabilidad_eur_abs,
                            'Rent. EUR %': rentabilidad_eur_pct,
                            'Valor EUR': valor_actual_eur
                        })

        # Sort by return descending (only stocks/ETFs, excluding Cash/TLT)
        asset_returns_df = pd.DataFrame(asset_returns)
        if not asset_returns_df.empty:
            asset_returns_df = asset_returns_df.sort_values('Rent. %', ascending=False)

            # Calcular valor inicial para estadísticas
            asset_returns_df['Valor_Inicial_EUR'] = asset_returns_df['Valor EUR'] / (1 + asset_returns_df['Rent. EUR %'] / 100)

            # Calculate totals for open positions
            total_valor_eur = asset_returns_df['Valor EUR'].sum()
            total_rent_eur = asset_returns_df['Rent. EUR'].sum()
            total_inicial_eur = asset_returns_df['Valor_Inicial_EUR'].sum()
            total_rent_pct = ((total_valor_eur / total_inicial_eur) - 1) * 100 if total_inicial_eur > 0 else 0

            # =====================================================
            # CALCULAR POSICIONES CERRADAS PARA ESTADÍSTICAS
            # =====================================================
            with db.get_session() as session:
                ventas_result = session.execute(text("""
                    SELECT fecha, account_code, symbol, shares, precio, importe_total, currency, pnl
                    FROM ventas
                    ORDER BY fecha DESC
                """))
                ventas_rows = ventas_result.fetchall()

            closed_positions = []
            total_closed_pnl_eur = 0
            total_closed_valor_eur = 0

            for venta in ventas_rows:
                fecha_venta, cuenta, symbol, shares, precio_venta, importe_venta, currency, pnl_db = venta
                precio_compra = portfolio_service.get_symbol_price(symbol, date(2025, 12, 31))
                if precio_compra is None:
                    precio_compra = precio_venta * 0.95

                pnl_local = (precio_venta - precio_compra) * shares
                rent_pct = ((precio_venta - precio_compra) / precio_compra) * 100 if precio_compra > 0 else 0

                if currency == 'USD':
                    pnl_eur = pnl_local / eur_usd_current
                    valor_venta_eur = importe_venta / eur_usd_current
                    currency_symbol = '$'
                elif currency == 'GBP':
                    pnl_eur = pnl_local * 1.18
                    valor_venta_eur = importe_venta * 1.18
                    currency_symbol = '£'
                elif currency == 'EUR':
                    pnl_eur = pnl_local
                    valor_venta_eur = importe_venta
                    currency_symbol = '€'
                else:
                    pnl_eur = pnl_local
                    valor_venta_eur = importe_venta
                    currency_symbol = currency

                total_closed_pnl_eur += pnl_eur
                total_closed_valor_eur += valor_venta_eur
                closed_positions.append({
                    'Fecha Venta': fecha_venta,
                    'Ticker': symbol,
                    'Cuenta': cuenta,
                    'Titulos': int(shares),
                    'Precio Compra': f"{currency_symbol}{precio_compra:.2f}",
                    'Precio Venta': f"{currency_symbol}{precio_venta:.2f}",
                    'Rent. %': rent_pct,
                    'Rent. EUR': pnl_eur,
                })

            # Combinar estadísticas: abiertas + cerradas
            import numpy as np
            all_returns_pct = list(asset_returns_df['Rent. %'].values) + [c['Rent. %'] for c in closed_positions]
            all_returns_eur = list(asset_returns_df['Rent. EUR'].values) + [c['Rent. EUR'] for c in closed_positions]

            # Recalcular totales para asegurar consistencia
            total_rent_eur_abiertas = asset_returns_df['Rent. EUR'].sum()
            total_rent_eur_cerradas = total_closed_pnl_eur
            combined_rent_eur = total_rent_eur_abiertas + total_rent_eur_cerradas
            total_count = len(all_returns_pct)
            positive_count = sum(1 for r in all_returns_pct if r > 0)
            negative_count = sum(1 for r in all_returns_pct if r < 0)
            positive_pct = (positive_count / total_count * 100) if total_count > 0 else 0
            max_return = max(all_returns_pct) if all_returns_pct else 0
            min_return = min(all_returns_pct) if all_returns_pct else 0
            total_gains = sum(r for r in all_returns_eur if r > 0)
            total_losses = abs(sum(r for r in all_returns_eur if r < 0))
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

            # Calcular rentabilidad % combinada (abiertas + cerradas)
            # Valor inicial cerradas = suma de (precio_compra * shares) para cada venta
            closed_inicial_eur = 0
            for pos in closed_positions:
                # Extraer precio compra numérico
                precio_str = pos['Precio Compra'].replace('$', '').replace('£', '').replace('€', '').replace('C', '')
                try:
                    precio_compra_num = float(precio_str)
                    shares = pos['Titulos']
                    # Convertir a EUR según moneda
                    if '$' in pos['Precio Compra']:
                        closed_inicial_eur += (precio_compra_num * shares) / eur_usd_current
                    elif '£' in pos['Precio Compra']:
                        gbp_eur = portfolio_service.get_exchange_rate('EURGBP=X', date.today())
                        gbp_eur_rate = 1 / gbp_eur if gbp_eur else 1.18
                        closed_inicial_eur += (precio_compra_num * shares) * gbp_eur_rate
                    else:
                        closed_inicial_eur += precio_compra_num * shares
                except (ValueError, TypeError, KeyError) as e:
                    logging.warning(f"Error parsing closed position: {e}")

            combined_inicial_eur = total_inicial_eur + closed_inicial_eur
            combined_rent_pct = (combined_rent_eur / combined_inicial_eur * 100) if combined_inicial_eur > 0 else 0

            # =====================================================
            # ESTADÍSTICAS (abiertas + cerradas)
            # =====================================================
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rentabilidad Acumulada Año", f"{combined_rent_eur:+,.0f} €".replace(",", "."))
            col2.metric("Mejor", f"{max_return:+.2f}%")
            col3.metric("Peor", f"{min_return:+.2f}%")
            col4.metric("Profit Factor", f"{profit_factor:.2f}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Positivos", f"{positive_count}")
            col6.metric("Negativos", f"{negative_count}")
            col7.metric("% Positivos", f"{positive_pct:.2f}%")
            col8.metric("Rentabilidad %", f"{combined_rent_pct:+.2f}%")

            st.markdown("---")

            # =====================================================
            # DETAILED TABLE WITH ALL STOCKS
            # =====================================================
            st.subheader("Posiciones Abiertas")

            # Calculate totals for the detailed table
            total_valor_eur = asset_returns_df['Valor EUR'].sum()
            total_rent_eur = asset_returns_df['Rent. EUR'].sum()
            total_inicial_eur = asset_returns_df['Valor_Inicial_EUR'].sum()
            total_rent_pct = ((total_valor_eur / total_inicial_eur) - 1) * 100 if total_inicial_eur > 0 else 0

            # Format for display
            display_df = asset_returns_df.copy()
            display_df['Titulos'] = display_df['Titulos'].astype(str)  # Convert to string to handle '-'
            display_df['Rent. %'] = display_df['Rent. %'].apply(lambda x: f"{x:+.2f}%")
            display_df['Rent. EUR'] = display_df['Rent. EUR'].apply(lambda x: f"{x:+,.0f} €".replace(",", "."))
            display_df['Rent. EUR %'] = display_df['Rent. EUR %'].apply(lambda x: f"{x:+.2f}%")
            display_df['Valor EUR'] = display_df['Valor EUR'].apply(lambda x: f"{x:,.0f} €".replace(",", "."))

            # Drop the helper column
            display_df = display_df.drop(columns=['Valor_Inicial_EUR'])

            # Reorder columns
            display_df = display_df[['Ticker', 'Tipo', 'Cuenta', 'Titulos', 'Precio 31/12', 'Precio Actual', 'Rent. %', 'Rent. EUR', 'Rent. EUR %', 'Valor EUR']]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Tipo': st.column_config.TextColumn('Tipo', width='small'),
                    'Cuenta': st.column_config.TextColumn('Cuenta', width='small'),
                    'Titulos': st.column_config.TextColumn('Títulos', width='small'),
                    'Precio 31/12': st.column_config.TextColumn('Precio 31/12', width='small'),
                    'Precio Actual': st.column_config.TextColumn('Precio Actual', width='small'),
                    'Rent. %': st.column_config.TextColumn('Rent. %', width='small'),
                    'Rent. EUR': st.column_config.TextColumn('Rent. EUR', width='small'),
                    'Rent. EUR %': st.column_config.TextColumn('Rent. EUR %', width='small'),
                    'Valor EUR': st.column_config.TextColumn('Valor EUR', width='medium'),
                }
            )

            # Total row with black background
            st.markdown(
                f"""
                <div style="background-color: #1a1a1a; color: white; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between; font-weight: bold;">
                    <span>TOTAL ABIERTAS</span>
                    <span>Valor EUR: {total_valor_eur:,.0f} €</span>
                    <span>Rent. EUR: {total_rent_eur:+,.0f} €</span>
                </div>
                """.replace(",", "."),
                unsafe_allow_html=True
            )

            # =====================================================
            # POSICIONES CERRADAS (ventas)
            # =====================================================
            st.markdown("---")
            st.subheader("Posiciones Cerradas")

            if closed_positions:
                closed_df = pd.DataFrame(closed_positions)

                # Format for display
                display_closed = closed_df.copy()
                display_closed['Rent. %'] = display_closed['Rent. %'].apply(lambda x: f"{x:+.2f}%")
                display_closed['Rent. EUR'] = display_closed['Rent. EUR'].apply(lambda x: f"{x:+,.0f} €".replace(",", "."))

                st.dataframe(
                    display_closed,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )

                # Total row for closed positions
                st.markdown(
                    f"""
                    <div style="background-color: #1a1a1a; color: white; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between; font-weight: bold;">
                        <span>TOTAL CERRADAS</span>
                        <span>Valor EUR: {total_closed_valor_eur:,.0f} €</span>
                        <span>Rent. EUR: {total_closed_pnl_eur:+,.0f} €</span>
                    </div>
                    """.replace(",", "."),
                    unsafe_allow_html=True
                )
            else:
                st.info("No hay posiciones cerradas")


elif page == "Futuros":
    st.title("FUTUROS")

    # Obtener datos de futuros desde la base de datos
    futures_summary = portfolio_service.get_futures_summary()
    futures_open_position = futures_summary['open_position']
    futures_total_usd = futures_summary['total_realized_usd']
    futures_total_eur = futures_summary['total_realized_eur']

    all_trades = portfolio_service.get_futures_trades_df()
    # Ordenar por fecha más reciente primero
    all_trades = all_trades.iloc[::-1].reset_index(drop=True)

    # Calcular estadísticas de operaciones cerradas
    trades_cerradas = all_trades[all_trades['Estado'] == 'Cerrada'] if not all_trades.empty else all_trades
    total_ops = len(trades_cerradas)

    # Contar ganadoras/perdedoras y calcular P&L numérico
    ops_ganadoras = 0
    ops_perdedoras = 0
    pnl_values = []
    total_gains = 0
    total_losses = 0

    for _, row in trades_cerradas.iterrows():
        pnl_str = row['P&G']
        if isinstance(pnl_str, str) and pnl_str != '-':
            pnl_clean = pnl_str.replace('$', '').replace(',', '').replace('+', '')
            try:
                pnl_val = float(pnl_clean)
                pnl_values.append(pnl_val)
                if pnl_val >= 0:
                    ops_ganadoras += 1
                    total_gains += pnl_val
                else:
                    ops_perdedoras += 1
                    total_losses += abs(pnl_val)
            except (ValueError, TypeError) as e:
                logging.debug(f"Could not parse P&L value '{pnl_clean}': {e}")

    pct_ganadoras = (ops_ganadoras / total_ops * 100) if total_ops > 0 else 0
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
    ganancia_media_usd = sum(pnl_values) / len(pnl_values) if pnl_values else 0
    eur_usd_rate = portfolio_service.get_eur_usd_rate(date.today())
    ganancia_media_eur = ganancia_media_usd / eur_usd_rate

    import numpy as np
    if len(pnl_values) > 1:
        avg_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values)
        sqn = (np.sqrt(len(pnl_values)) * avg_pnl / std_pnl) if std_pnl > 0 else 0
    else:
        sqn = 0

    # Estadísticas (debajo del título)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ops. Positivas", f"{ops_ganadoras} ({pct_ganadoras:.1f}%)")
    col2.metric("Ops. Negativas", f"{ops_perdedoras} ({100-pct_ganadoras:.1f}%)")
    col3.metric("Total Operaciones", f"{total_ops}")
    col4.metric("Profit Factor", f"{profit_factor:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Ganancia Media USD", f"${ganancia_media_usd:,.2f}".replace(",", "."))
    col6.metric("Ganancia Media EUR", f"{ganancia_media_eur:,.2f} €".replace(",", "."))
    col7.metric("SQN", f"{sqn:.2f}")
    col8.metric("Posición Abierta", f"{futures_open_position['contracts']} contratos")

    st.markdown("---")

    # Historial de Operaciones
    st.subheader("Historial de Operaciones")

    def color_trade_row(row):
        styles = [''] * len(row)
        pnl_idx = row.index.get_loc('P&G')
        val = row['P&G']
        if isinstance(val, str):
            if val.startswith('+'):
                styles[pnl_idx] = 'color: #00cc00; font-weight: bold'
            elif val.startswith('-') and val != '-':
                styles[pnl_idx] = 'color: #ff4444; font-weight: bold'
        estado_idx = row.index.get_loc('Estado')
        if row['Estado'] == 'ABIERTA':
            styles[estado_idx] = 'background-color: #cccccc; color: #333333; font-weight: bold'
        return styles

    styled_trades = all_trades.style.apply(color_trade_row, axis=1)
    st.dataframe(styled_trades, use_container_width=True, hide_index=True, height=500)

    # Total row with black background
    pnl_sign = '+' if futures_total_usd >= 0 else ''
    st.markdown(
        f"""
        <div style="background-color: #1a1a1a; color: white; padding: 15px; border-radius: 5px; display: flex; justify-content: space-between; font-weight: bold; align-items: center;">
            <span style="font-size: 1.2em;">TOTAL</span>
            <span style="font-size: 1.4em;">P&G USD: {pnl_sign}${futures_total_usd:,.2f}</span>
            <span style="font-size: 1.4em;">P&G EUR: {pnl_sign}{futures_total_eur:,.2f} €</span>
        </div>
        """.replace(",", "."),
        unsafe_allow_html=True
    )


elif page == "Estrategia Mensual":
    st.title("Seleccion Mensual")
    st.markdown("Day 0 = Ultimo dia de negociacion del mes anterior (End-of-Day prices)")

    month_names = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
                  5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
                  9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}

    monthly_selections = {
        "Febrero 2026": "AVGO, SYK, NET, STLD, TJX, PAA, REGN, GE, IDXX, HEI",
        "Enero 2026": "AMZN, MCO, MA, VRTX, WST, BDX, HCA, PCAR, CRM, AKAM",
        "Diciembre 2025": "FE, AVGO, KLAC, ANET, HPE, EL, PEP, AMGN, ABBV, ALL",
        "Noviembre 2025": "IBM, MSFT, MNST, CBRE, AME, AXP, SCHW, CVX, GM, ULTA",
        "Octubre 2025": "EVRG, D, LDOS, BTG, STLD, JPM, BAC, WFC, E, NFLX",
        "Septiembre 2025": "CEG, INTC, RIO, PWR, AIZ, FTI, ORLY",
        "Agosto 2025": "XEL, MSI, SYY, IRM, PAYC, CTAS, PODD, CBOE, BRK-B, TJX",
    }

    col1, col2 = st.columns(2)
    with col1:
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Generar opciones: mes siguiente + mes actual + 12 meses anteriores
        month_options = []
        # Incluir mes siguiente (para preparar seleccion)
        next_month = current_month + 1 if current_month < 12 else 1
        next_year = current_year if current_month < 12 else current_year + 1
        month_options.append(f"{month_names[next_month]} {next_year}")
        # Mes actual hacia atras
        m, y = current_month, current_year
        for _ in range(13):  # 13 meses hacia atras
            month_options.append(f"{month_names[m]} {y}")
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        selected_month = st.selectbox("Seleccion", month_options, index=0)
    with col2:
        show_returns = st.checkbox("Show returns %", value=True)

    parts = selected_month.split()
    month_name = parts[0]
    year = int(parts[1])
    month = [k for k, v in month_names.items() if v == month_name][0]

    default_symbols = monthly_selections.get(selected_month, "")

    symbols_input = st.text_input(
        "Symbols",
        value=default_symbols,
        help="Stock symbols for this month's selection"
    )

    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if symbols and st.button("Load Prices", type="primary"):
        st.markdown("---")

        if month == 1:
            prev_month, prev_year = 12, year - 1
        else:
            prev_month, prev_year = month - 1, year

        first_day_current = datetime(year, month, 1)
        today = datetime.now().date()

        with db.get_session() as session:
            all_data = {}
            start_date = datetime(prev_year, prev_month, 1)

            for symbol_code in symbols:
                db_symbol = session.query(Symbol).filter(Symbol.code == symbol_code).first()
                if db_symbol:
                    prices = db.get_price_history(session, db_symbol.id, start_date=start_date)
                    if not prices.empty:
                        prices = prices[prices.index.date < today]
                        prev_month_prices = prices[prices.index < first_day_current]
                        current_month_prices = prices[prices.index >= first_day_current]

                        if not prev_month_prices.empty:
                            day_0_price = prev_month_prices['close'].iloc[-1]
                            day_0_date = prev_month_prices.index[-1]
                            combined = pd.concat([
                                pd.Series([day_0_price], index=[day_0_date], name='close'),
                                current_month_prices['close']
                            ])
                            all_data[symbol_code] = combined
                        elif not current_month_prices.empty:
                            all_data[symbol_code] = current_month_prices['close']

        if all_data:
            price_df = pd.DataFrame(all_data)
            day_labels = ["0"] + [str(i) for i in range(1, len(price_df))]
            date_labels = price_df.index.strftime("%d/%m").tolist()
            column_labels = [f"{d} ({dt})" for d, dt in zip(day_labels, date_labels)]
            price_df.index = column_labels
            price_table = price_df.T
            price_table.index.name = "Symbol"

            st.dataframe(price_table.style.format("${:.2f}"), use_container_width=True)

            if show_returns:
                st.subheader("Returns % (from Day 0)")
                returns_from_day0 = pd.DataFrame(all_data)
                for col in returns_from_day0.columns:
                    day0_price = returns_from_day0[col].iloc[0]
                    returns_from_day0[col] = ((returns_from_day0[col] - day0_price) / day0_price) * 100

                returns_from_day0.index = column_labels
                returns_table = returns_from_day0.T

                def style_returns(val):
                    if pd.isna(val):
                        return ""
                    color = "green" if val > 0 else "red" if val < 0 else "gray"
                    return f"color: {color}"

                st.dataframe(
                    returns_table.style.format("{:+.2f}%").map(style_returns),
                    use_container_width=True
                )

            st.subheader("Month-to-Date Summary")
            summary_data = []
            for symbol_code in symbols:
                if symbol_code in all_data:
                    prices = all_data[symbol_code]
                    if len(prices) >= 2:
                        day0_price = prices.iloc[0]
                        current_price = prices.iloc[-1]
                        mtd_return = ((current_price - day0_price) / day0_price) * 100
                        summary_data.append({
                            "Symbol": symbol_code,
                            "Day 0 Price": f"${day0_price:.2f}",
                            "Current Price": f"${current_price:.2f}",
                            "MTD Change": f"${current_price - day0_price:+.2f}",
                            "MTD Return %": f"{mtd_return:+.2f}%",
                        })

            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            st.markdown("---")
            csv = price_table.to_csv()
            st.download_button(
                "Export to CSV",
                csv,
                f"daily_prices_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
            )
        else:
            st.warning("No price data found. Make sure symbols are downloaded.")


elif page == "Symbol Analysis":
    st.title("Symbol Analysis")

    col1, col2 = st.columns([3, 1])
    with col1:
        symbol_input = st.text_input("Enter symbol (e.g., AAPL.US)", value="AAPL.US")
    with col2:
        days = st.selectbox("Time period", [30, 60, 90, 180, 365], index=2)

    if st.button("Analyze") or symbol_input:
        df = get_price_data(symbol_input, days=days)

        if df.empty:
            with st.spinner("Downloading data from Yahoo Finance..."):
                try:
                    downloader = YahooDownloader()
                    downloader.download_historical_prices(symbol_input)
                    downloader.download_fundamentals(symbol_input)
                    df = get_price_data(symbol_input, days=days)
                except Exception as e:
                    st.error(f"Error downloading data: {e}")

        if not df.empty:
            df = technical.calculate_sma(df, periods=[20, 50, 200])
            df = technical.calculate_ema(df, periods=[12, 26])
            df = technical.calculate_rsi(df)
            df = technical.calculate_macd(df)
            df = technical.calculate_bollinger_bands(df)

            analysis = analyzer.analyze_symbol(df, symbol_input)

            st.subheader("Price Chart")
            fig = create_candlestick_chart(df, symbol_input)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Technical Indicators")
                indicators = analysis["technical_indicators"]
                ind_df = pd.DataFrame([
                    {"Indicator": "SMA 20", "Value": f"{indicators['sma_20']:.2f}" if indicators['sma_20'] else "N/A"},
                    {"Indicator": "SMA 50", "Value": f"{indicators['sma_50']:.2f}" if indicators['sma_50'] else "N/A"},
                    {"Indicator": "SMA 200", "Value": f"{indicators['sma_200']:.2f}" if indicators['sma_200'] else "N/A"},
                    {"Indicator": "RSI (14)", "Value": f"{indicators['rsi']:.2f}" if indicators['rsi'] else "N/A"},
                    {"Indicator": "MACD", "Value": f"{indicators['macd']:.4f}" if indicators['macd'] else "N/A"},
                    {"Indicator": "ATR", "Value": f"{indicators['atr']:.2f}" if indicators['atr'] else "N/A"},
                ])
                st.table(ind_df)

            with col2:
                st.subheader("Signals & Trend")
                st.markdown(f"**Current Trend:** {analysis['trend'].upper()}")
                st.markdown(f"**Support:** ${analysis['support_resistance']['support']:.2f}")
                st.markdown(f"**Resistance:** ${analysis['support_resistance']['resistance']:.2f}")
                st.markdown("**Signals:**")
                for signal in analysis["signals"]:
                    emoji = "🟢" if signal["action"] in ["uptrend", "potential_buy", "hold_buy"] else "🔴" if signal["action"] in ["downtrend", "potential_sell", "hold_sell"] else "⚪"
                    st.markdown(f"{emoji} {signal['indicator']}: {signal['signal']}")

            st.subheader("Indicator Charts")
            tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])
            with tab1:
                st.plotly_chart(create_indicator_chart(df, "RSI"), use_container_width=True)
            with tab2:
                st.plotly_chart(create_indicator_chart(df, "MACD"), use_container_width=True)
            with tab3:
                st.plotly_chart(create_indicator_chart(df, "Bollinger"), use_container_width=True)
        else:
            st.warning("No data available for this symbol.")


elif page == "Data Management":
    st.title("Data Management")
    st.info("Data source: Yahoo Finance (free, unlimited)")

    st.subheader("Download Data")

    symbols_input = st.text_area(
        "Symbols to download (one per line)",
        value="\n".join(DEFAULT_SYMBOLS[:5]),
    )
    symbols = [s.strip() for s in symbols_input.split("\n") if s.strip()]

    col1, col2, col3 = st.columns(3)
    with col1:
        include_fundamentals = st.checkbox("Include fundamentals", value=True)
    with col2:
        incremental = st.checkbox("Incremental (only new data)", value=True)
    with col3:
        period = st.selectbox("History period", ["max", "10y", "5y", "2y", "1y"], index=0)

    if st.button("Download Selected", type="primary"):
        downloader = YahooDownloader()
        progress = st.progress(0)
        status = st.empty()

        total_records = 0
        for i, symbol in enumerate(symbols):
            status.text(f"Downloading {symbol}...")
            try:
                count = downloader.download_historical_prices(
                    symbol, period=period, incremental=incremental
                )
                total_records += count
                if include_fundamentals:
                    downloader.download_fundamentals(symbol)
                status.text(f"{symbol}: {count} records")
            except Exception as e:
                st.error(f"Error downloading {symbol}: {e}")

            progress.progress((i + 1) / len(symbols))

        st.success(f"Download complete! {total_records:,} total records")
        st.rerun()

    st.markdown("---")
    st.subheader("Quick Actions")

    if st.button("Download All Default Symbols"):
        with st.spinner("Downloading all default symbols..."):
            downloader = YahooDownloader()
            results = downloader.download_all(DEFAULT_SYMBOLS)
            st.success(
                f"Done! Prices: {results['prices_success']}/{results['total_symbols']}, "
                f"Fundamentals: {results['fundamentals_success']}/{results['total_symbols']}, "
                f"Records: {results['total_price_records']:,}"
            )


elif page == "Screener":
    st.title("Stock Screener")
    st.markdown("Filter stocks based on technical and fundamental metrics")

    with db.get_session() as session:
        metrics_count = session.query(DailyMetrics).count()

    if metrics_count == 0:
        st.warning("No metrics calculated yet. Run: `py -m src.technical --recent` to calculate metrics.")
    else:
        st.info(f"Metrics available for {metrics_count} records. Use filters below.")


elif page == "Portfolios":
    st.title("Portfolio Management")
    st.info("Ver Dashboard para el resumen completo de la cartera.")


elif page == "Download Status":
    st.title("Download Status")

    with db.get_session() as session:
        logs = (
            session.query(DownloadLog)
            .order_by(DownloadLog.started_at.desc())
            .limit(50)
            .all()
        )

        if logs:
            log_data = []
            for log in logs:
                log_data.append({
                    "Time": log.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Operation": log.operation,
                    "Symbol": log.symbol or "-",
                    "Records": log.records_downloaded,
                    "Status": log.status,
                    "Duration": f"{log.duration_seconds:.2f}s" if log.duration_seconds else "-",
                })

            df = pd.DataFrame(log_data)
            st.dataframe(df, use_container_width=True)

            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            success_count = sum(1 for l in logs if l.status == "success")
            error_count = sum(1 for l in logs if l.status == "error")
            total_records = sum(l.records_downloaded for l in logs)
            col1.metric("Successful Downloads", success_count)
            col2.metric("Failed Downloads", error_count)
            col3.metric("Total Records Downloaded", f"{total_records:,}")
        else:
            st.info("No download logs available.")


elif page == "BBDD":
    st.title("BBDD - Documentación Técnica")

    st.markdown("""
    Esta sección documenta la estructura de la base de datos y las tablas utilizadas
    para el seguimiento de la cartera financiera.
    """)

    # Get table info from database
    with db.get_session() as session:
        from sqlalchemy import text

        # Get all tables
        result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))
        tables = [row[0] for row in result.fetchall()]

    st.markdown("---")

    # =====================================================
    # TABLAS PRINCIPALES
    # =====================================================
    st.header("1. Tablas Principales de Posición")

    st.markdown("""
    Estas son las tablas clave que se utilizan para calcular y mostrar la posición diaria de la cartera.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        | Tabla | Registros |
        |-------|----------:|
        | **posicion** | ~120 |
        | **holding_diario** | ~3,500 |
        | **cash_diario** | ~150 |
        """)

    with col2:
        st.markdown("""
        **posicion** - Posición diaria consolidada por cuenta
        ```
        fecha | account_code | holding_eur | cash_eur | total_eur
        ```

        **holding_diario** - Holdings diarios con precio de mercado
        ```
        fecha | account_code | symbol | shares | precio_entrada | currency
        ```

        **cash_diario** - Saldo de efectivo diario
        ```
        fecha | account_code | currency | saldo
        ```
        """)

    st.markdown("---")

    # =====================================================
    # FLUJO DE DATOS
    # =====================================================
    st.header("2. Flujo de Datos")

    st.markdown("""
    ```
    ┌─────────────────┐     ┌─────────────────┐
    │ holding_diario  │     │   cash_diario   │
    │  (detalle por   │     │  (saldo por     │
    │    símbolo)     │     │    moneda)      │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             │   ┌───────────────┐   │
             └──►│   posicion    │◄──┘
                 │ (consolidado) │
                 └───────────────┘
    ```
    """)

    st.markdown("---")

    # =====================================================
    # TABLAS DE PRECIOS
    # =====================================================
    st.header("3. Tablas de Precios y Símbolos")

    st.markdown("""
    | Tabla | Registros | Descripción |
    |-------|----------:|-------------|
    | **symbols** | ~5,800 | Catálogo de símbolos (acciones, ETFs, FX, índices) |
    | **price_history** | ~24M | Histórico de precios OHLCV - fuente principal para valoración |
    | **fundamentals** | ~5,900 | Datos fundamentales (PE, market cap, revenue, etc.) |
    | **asset_types** | ~115 | Clasificación de activos (Mensual, Quant, Oro/Mineras, etc.) |
    """)

    st.markdown("""
    **Estructura price_history:**
    ```sql
    symbol_id | date | open | high | low | close | adjusted_close | volume
    ```
    """)

    st.markdown("---")

    # =====================================================
    # TABLAS DE TRANSACCIONES
    # =====================================================
    st.header("4. Tablas de Transacciones")

    st.markdown("""
    | Tabla | Registros | Descripción |
    |-------|----------:|-------------|
    | **compras** | ~4 | Compras de acciones/ETFs |
    | **ventas** | ~24 | Ventas de acciones/ETFs |
    | **ib_futures_trades** | ~32 | Trades de futuros en IB (incluye P&L) |
    | **movimientos_cash** | ~6 | Transferencias entre cuentas |
    """)

    st.markdown("---")

    # =====================================================
    # FÓRMULAS DE CÁLCULO
    # =====================================================
    st.header("5. Fórmulas de Cálculo")

    st.subheader("Cuenta IB (Interactive Brokers)")
    st.code("""
Cash EUR = 30,000 + Transfers - (TLT_Cost × FX) + (P&L_Futures × FX)
TLT EUR  = Shares × Precio_Mercado × FX
Total    = Cash + TLT
    """, language="text")

    st.subheader("Otras cuentas (CO3365, RCO951, LACAIXA)")
    st.code("""
Holding EUR = Σ(shares × precio_mercado × FX_rate)
Cash EUR    = Σ(saldo × FX_rate)
Total       = Holding + Cash
    """, language="text")

    st.subheader("Tipos de Cambio (de price_history)")
    st.code("""
USD→EUR = 1 / EURUSD=X
GBP→EUR = GBPUSD=X / EURUSD=X
CAD→EUR = CADEUR=X
CHF→EUR = CHFEUR=X
    """, language="text")

    st.markdown("---")

    # =====================================================
    # ESTRUCTURA COMPLETA
    # =====================================================
    st.header("6. Estructura Completa de la Base de Datos")

    # Show all tables with counts
    with db.get_session() as session:
        from sqlalchemy import text

        table_info = []
        for table in tables:
            if table == 'sqlite_sequence':
                continue
            try:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.fetchone()[0]
                table_info.append({'Tabla': table, 'Registros': f"{count:,}"})
            except Exception as e:
                logging.warning(f"Error counting table {table}: {e}")
                table_info.append({'Tabla': table, 'Registros': 'Error'})

        table_df = pd.DataFrame(table_info)

    # Split into two columns
    mid = len(table_df) // 2
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(table_df.iloc[:mid], use_container_width=True, hide_index=True)

    with col2:
        st.dataframe(table_df.iloc[mid:], use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================================================
    # CUENTAS
    # =====================================================
    st.header("7. Cuentas del Sistema")

    st.markdown("""
    | Código | Descripción | Broker | Moneda Base |
    |--------|-------------|--------|-------------|
    | **CO3365** | Cartera Mensual | Clicktrade | EUR/USD |
    | **RCO951** | Cartera Quant + Oro | Clicktrade | EUR/USD |
    | **LACAIXA** | Cartera Value | CaixaBank | EUR |
    | **IB** | Interactive Brokers | IBKR Ireland | EUR/USD |
    """)

    st.markdown("---")

    # =====================================================
    # TIPOS DE ACTIVO
    # =====================================================
    st.header("8. Tipos de Activo")

    st.markdown("""
    | Tipo | Descripción | Cuenta Principal |
    |------|-------------|------------------|
    | **Mensual** | Selección mensual de acciones US | CO3365 |
    | **Quant** | Estrategia cuantitativa | RCO951 |
    | **Value** | Acciones value/growth | LACAIXA |
    | **Alpha Picks** | Selecciones alpha | RCO951 |
    | **Oro/Mineras** | ETF oro + mineras | RCO951 |
    | **Cash/Monetario/Bonos/Futuros** | Efectivo + TLT | Todas |
    """)

    st.markdown("---")

    # =====================================================
    # SCRIPTS DE MANTENIMIENTO
    # =====================================================
    st.header("9. Scripts de Mantenimiento")

    st.markdown("""
    Los siguientes scripts se utilizan para actualizar los datos:

    | Script | Función |
    |--------|---------|
    | `recalc_cash.py` | Recalcula saldos de cash diarios |
    | `setup_ib_daily.py` | Configura posición IB (TLT + Cash + Futuros) |
    | `recalc_posicion_all.py` | Recalcula tabla posicion para todas las cuentas |
    | `update_portfolio_cash.py` | Actualiza cash en portfolio |
    | `populate_portfolio.py` | Pobla tabla portfolio desde holding_diario |

    **Ubicación:** `scratchpad/` (directorio temporal de trabajo)
    """)


elif page == "Pantalla":
    st.title("PANTALLA - Documentación Técnica")

    st.markdown("""
    Esta sección documenta la estructura y configuración de las pantallas del dashboard.
    """)

    st.markdown("---")

    # =====================================================
    # PÁGINA POSICIÓN - RESUMEN DE CARTERA
    # =====================================================
    st.header("1. Página POSICIÓN - Resumen de Cartera")

    st.markdown("""
    ### Obtención de Datos

    Los valores del resumen de cartera se obtienen directamente de la **tabla `posicion`** en la base de datos:

    | Campo | Fuente | Query |
    |-------|--------|-------|
    | **Valor Inicial (31/12)** | `posicion` | `SELECT SUM(total_eur) FROM posicion WHERE fecha = '2025-12-31'` |
    | **Valor Actual** | `posicion` | `SELECT SUM(total_eur) FROM posicion WHERE fecha = [última fecha]` |
    | **Fecha Actual** | `posicion` | `SELECT MAX(fecha) FROM posicion WHERE fecha < date('now')` |

    ### Lógica de Fechas

    - **Fecha actual**: Siempre es el **día anterior** al día de hoy (último día con datos disponibles)
    - **Fecha inicial**: Fija en **31/12/2025** (inicio del año fiscal)

    ### Cálculos

    ```python
    # Rentabilidad EUR
    return_eur = current_value - initial_value
    return_pct = (return_eur / initial_value) * 100

    # Conversión a USD
    initial_usd = initial_value * eur_usd_31dic
    current_usd = current_value * eur_usd_current
    return_usd = current_usd - initial_usd
    ```

    ### Tipos de Cambio

    - **EUR/USD 31/12**: Se obtiene de `price_history` para el símbolo `EURUSD=X`
    - **EUR/USD Actual**: Se obtiene de `price_history` para la fecha actual
    """)

    st.markdown("---")

    # =====================================================
    # VARIACIÓN DIARIA POR TIPO DE ACTIVO
    # =====================================================
    st.header("2. Variación Diaria por Tipo de Activo")

    st.markdown("""
    ### Ubicación
    Página **Posición** → Sección "Variación Diaria por Tipo de Activo"

    ### Obtención de Datos

    La variación diaria se calcula usando la función `calculate_portfolio_by_type(fecha)` que internamente
    llama a `portfolio_service.get_values_by_asset_type(fecha)`.

    | Tabla | Uso |
    |-------|-----|
    | `holding_diario` | Posiciones (symbol, shares) para cada cuenta en la fecha |
    | `posicion` | Cash total en EUR por cuenta |
    | `price_history` | Precios de cierre y tipos de cambio |

    ### Query de Holdings
    ```sql
    SELECT account_code, symbol, shares, currency
    FROM holding_diario
    WHERE fecha = :fecha
    ```

    ### Query de Cash
    ```sql
    SELECT account_code, cash_eur
    FROM posicion
    WHERE fecha = :fecha
    ```

    ### Tipos de Activo (ASSET_TYPE_MAP)

    | Tipo | Descripción | Cuenta Principal |
    |------|-------------|------------------|
    | **Mensual** | 10 acciones CO3365 | CO3365 |
    | **Quant** | ~65 acciones growth | RCO951 |
    | **Value** | JD, BABA, IAG.MC, NESN.SW | LACAIXA |
    | **Alpha Picks** | 12 acciones seleccionadas | RCO951 |
    | **Oro/Mineras** | B, TFPM, SSRM, RGLD, KGC, etc. + SGLE.MI | RCO951, LACAIXA |
    | **Cash/Monetario/Bonos/Futuros** | TLT + Cash de todas las cuentas | IB + todos |

    ### Cálculo de Valor por Tipo

    ```python
    def get_values_by_asset_type(fecha):
        values = {}
        all_holdings = get_all_holdings_for_date(fecha)  # desde holding_diario

        for account, holdings in all_holdings.items():
            for symbol, data in holdings.items():
                shares = data['shares']
                asset_type = ASSET_TYPE_MAP.get(symbol, 'Otros')

                # Calcular valor en EUR
                price = get_symbol_price(symbol, fecha)
                value_eur = convert_to_eur(price * shares, symbol_currency)

                values[asset_type] += value_eur

        # Añadir cash desde tabla posicion
        posicion = get_posicion_for_date(fecha)
        total_cash = sum(p['cash_eur'] for p in posicion.values())
        values['Cash/Monetario/Bonos/Futuros'] += total_cash

        return values
    ```

    ### Conversión de Divisas

    | Divisa | Símbolo FX | Cálculo |
    |--------|------------|---------|
    | USD → EUR | EURUSD=X | `value_usd / eurusd` |
    | CAD → EUR | CADEUR=X | `value_cad * cadeur` |
    | CHF → EUR | CHFEUR=X | `value_chf * chfeur` |
    | EUR → EUR | - | Sin conversión |

    ### Fechas Utilizadas

    - **Día Anterior (day_prev)**: `SELECT MAX(fecha) FROM posicion WHERE fecha < [día_actual]`
    - **Día Actual (day_last)**: `SELECT MAX(fecha) FROM posicion WHERE fecha < date('now')`

    ### Verificación

    El total calculado debe coincidir con la tabla `posicion`:
    ```sql
    SELECT SUM(total_eur) FROM posicion WHERE fecha = :fecha
    ```
    """)

    st.markdown("---")

    # =====================================================
    # RENTABILIDAD DIARIA
    # =====================================================
    st.header("3. Rentabilidad Diaria")

    st.markdown("""
    ### Ubicación
    Página **Posición** → Sección "Rentabilidad Diaria"

    ### Obtención de Datos

    Los datos se obtienen de la tabla `posicion` excluyendo el día actual (datos incompletos):

    ```sql
    SELECT fecha, SUM(total_eur) as total
    FROM posicion
    WHERE fecha < date('now')
    GROUP BY fecha
    ORDER BY fecha
    ```

    ### Filtros Aplicados

    - **Excluye día actual**: Solo muestra hasta el día anterior (último día con datos completos)
    - **Excluye fines de semana**: `d.weekday() >= 5` se omite
    - **Excluye fechas específicas**: 1/1/2026 y 19/1/2026 (festivos sin datos)

    ```python
    excluded_dates = {date(2026, 1, 1), date(2026, 1, 19)}
    ```

    ### Gráfica de Rentabilidad

    - **Datos**: Usa los mismos datos filtrados que la tabla (sin fines de semana ni fechas excluidas)
    - **Sin saltos**: La gráfica no tiene rangebreaks, las líneas conectan directamente los puntos
    - **Series**: Cartera EUR, Cartera USD, SPY, QQQ

    ### Cálculo de Valores

    | Campo | Fórmula |
    |-------|---------|
    | **Valor EUR** | Directo de `posicion.total_eur` |
    | **Valor USD** | `valor_eur × EUR/USD del día` |
    | **Rent. EUR %** | `(valor_eur / valor_inicial_eur - 1) × 100` |
    | **Rent. USD %** | `(valor_usd / valor_inicial_usd - 1) × 100` |

    ### Valores Iniciales (31/12)

    ```python
    initial_val_eur = posicion['2025-12-31'].total_eur
    initial_val_usd = initial_val_eur × EUR/USD_31dic
    ```

    ### Tipos de Cambio

    - Se obtienen de `price_history` para el símbolo `EURUSD=X`
    - Cada día usa su propio tipo de cambio
    - El valor USD refleja tanto la variación de la cartera como el efecto FX

    ### Formato Visual

    | Condición | Estilo |
    |-----------|--------|
    | Porcentaje > 0 | Fondo verde (#2E7D32), texto blanco |
    | Porcentaje < 0 | Fondo rojo (#C62828), texto blanco |
    | Porcentaje = 0 | Sin color especial |

    ### Columnas de la Tabla

    | Columna | Descripción |
    |---------|-------------|
    | Fecha | Formato dd/mm/yyyy |
    | Valor EUR | Valor total cartera en EUR (sin decimales) |
    | Valor USD | Valor total cartera en USD (sin decimales) |
    | Cartera EUR | Rentabilidad acumulada en EUR vs 31/12 |
    | Cartera USD | Rentabilidad acumulada en USD vs 31/12 |
    | SPY | Rentabilidad SPY vs 31/12 (benchmark) |
    | QQQ | Rentabilidad QQQ vs 31/12 (benchmark) |

    ### Ejemplo de Cálculo

    ```
    31/12/2025: EUR 3,930,529 × 1.1747 = USD 4,617,308
    28/01/2026: EUR 4,235,788 × 1.1965 = USD 5,067,944

    Rent. EUR: (4,235,788 / 3,930,529 - 1) × 100 = +7.77%
    Rent. USD: (5,067,944 / 4,617,308 - 1) × 100 = +9.76%
    ```
    """)

    st.markdown("---")

    # =====================================================
    # COMPOSICIÓN DE CARTERA
    # =====================================================
    st.header("4. Composición de Cartera")

    st.markdown("""
    ### Ubicación
    Página **Composición** (pestaña independiente)

    ### Secciones

    #### 4.1 Composición por Diversificación

    Agrupa la cartera por clase de activo:

    | Clase | Descripción |
    |-------|-------------|
    | **Acciones** | CO3365 + RCO951 (stocks) + La Caixa |
    | **ETF Oro** | RCO951 ETF Gold (GLD, IAU, etc.) |
    | **Bonos (TLT)** | Interactive Brokers TLT |
    | **Cash** | Efectivo en CO3365 + RCO951 + IB |

    ```python
    diversificacion_data = {
        'Acciones': co3365_stocks + rco951_stocks + lacaixa,
        'ETF Oro': rco951_etf_gold,
        'Bonos (TLT)': ib_tlt_value,
        'Cash': total_cash_eur,
    }
    ```

    #### 4.2 Composición por Estrategia

    Agrupa por tipo de estrategia según `ASSET_TYPE_MAP`:

    | Estrategia | Descripción |
    |------------|-------------|
    | **Mensual** | Selecciones mensuales (AKAM, VRTX, PCAR, etc.) |
    | **Quant** | Estrategia cuantitativa |
    | **Value** | Inversión en valor |
    | **Alpha Picks** | Selecciones alpha |
    | **Oro/Mineras** | ETFs de oro y mineras |
    | **Cash/Monetario/Bonos/Futuros** | Efectivo y monetarios |

    ```python
    strategy_values = portfolio_service.get_values_by_asset_type(latest_date)
    ```

    #### 4.3 Composición por Cuenta

    Agrupa por cuenta/broker:

    | Cuenta | Descripción |
    |--------|-------------|
    | **CO3365** | Charles Schwab - cuenta principal |
    | **RCO951** | Charles Schwab - cuenta secundaria |
    | **La Caixa** | CaixaBank - acciones españolas/HK |
    | **Interactive Brokers** | IB - TLT y futuros |

    ### Obtención de Datos

    ```python
    account_totals = get_account_totals_from_db(latest_date)
    rco951_breakdown = get_rco951_breakdown_from_db(latest_date)
    ib_tlt_value = get_ib_tlt_value_from_db(latest_date)
    strategy_values = portfolio_service.get_values_by_asset_type(latest_date)
    ```

    ### Formato Visual

    - Cada sección tiene un **gráfico de tarta** (pie chart) a la izquierda
    - Tabla con **Valor EUR** y **% Peso** a la derecha
    - Template: `plotly_dark`
    - Gráficos con `hole=0.4` (donut chart)
    """)

    st.markdown("---")

    # =====================================================
    # ESTRUCTURA DEL DASHBOARD
    # =====================================================
    st.header("5. Estructura del Dashboard")

    st.markdown("""
    | Página | Descripción | Archivo |
    |--------|-------------|---------|
    | **Posición** | Resumen de cartera, variación diaria, gráficos | `web/app.py` |
    | **Acciones/ETF** | Rentabilidad por tipo de activo y detalle | `web/app.py` |
    | **Futuros** | Operaciones de futuros IB, P&L | `web/app.py` |
    | **Daily Tracking** | Seguimiento mensual de selecciones | `web/app.py` |
    | **Screener** | Filtrado de acciones por métricas | `web/app.py` |
    | **Symbol Analysis** | Análisis técnico de símbolos | `web/app.py` |
    | **Data Management** | Descarga de datos de Yahoo Finance | `web/app.py` |
    | **Download Status** | Log de descargas | `web/app.py` |
    """)

    st.markdown("---")

    # =====================================================
    # ARCHIVOS PRINCIPALES
    # =====================================================
    st.header("4. Archivos Principales")

    st.markdown("""
    ```
    financial-data-project/
    ├── web/
    │   └── app.py              # Dashboard principal (Streamlit)
    ├── src/
    │   ├── config.py           # Configuración general
    │   ├── database.py         # Modelos SQLAlchemy y conexión BD
    │   ├── portfolio_data.py   # Servicio centralizado de datos
    │   ├── yahoo_client.py     # Cliente Yahoo Finance
    │   ├── yahoo_downloader.py # Descargador de datos
    │   ├── technical.py        # Cálculos de métricas técnicas
    │   └── analysis/
    │       └── ai_analyzer.py  # Análisis técnico
    └── data/
        └── financial_data.db   # Base de datos SQLite
    ```
    """)

    st.markdown("---")

    # =====================================================
    # SERVICIO DE PORTFOLIO
    # =====================================================
    st.header("5. Servicio de Portfolio (portfolio_data.py)")

    st.markdown("""
    El archivo `src/portfolio_data.py` centraliza la configuración y acceso a datos de la cartera:

    **Datos desde Base de Datos:**
    - Holdings: tabla `holding_diario` (posiciones diarias por cuenta)
    - Cash: tabla `cash_diario` (saldos de efectivo diarios)
    - Posición: tabla `posicion` (resumen diario por cuenta)

    **Constantes de Configuración:**
    - `ASSET_TYPE_MAP` - Clasificación de activos por tipo
    - `CURRENCY_MAP`, `CURRENCY_SYMBOL_MAP` - Mapeos de divisas
    - `FUTURES_TRADES`, `FUTURES_PNL` - Datos de futuros (sin PnL en BD)

    **Métodos principales:**
    ```python
    # Datos desde BD
    portfolio_service.get_holdings_for_date(account, date)   # Holdings desde holding_diario
    portfolio_service.get_cash_for_date(account, date)       # Cash desde cash_diario
    portfolio_service.get_posicion_for_date(date)            # Posición desde posicion
    portfolio_service.get_initial_total()                    # Total inicial 31/12/2025

    # Precios y FX
    portfolio_service.get_eur_usd_rate(date)                 # Tipo de cambio EUR/USD
    portfolio_service.get_symbol_price(symbol, date)         # Precio de símbolo

    # Cálculos
    portfolio_service.calculate_position_value(symbol, shares, date)  # Valor posición
    portfolio_service.get_values_by_asset_type(date)         # Valores por tipo de activo
    ```
    """)

    st.markdown("---")

    # =====================================================
    # FUNCIONES DE CÁLCULO
    # =====================================================
    st.header("6. Funciones de Cálculo del Dashboard")

    st.markdown("""
    **Funciones principales en app.py:**

    | Función | Descripción |
    |---------|-------------|
    | `calculate_portfolio_by_type(date)` | Calcula valor por tipo de activo |
    | `calculate_portfolio_total(date)` | Calcula valor total de cartera |
    | `get_account_totals_from_db(date)` | Obtiene totales por cuenta desde BD |
    | `get_rco951_breakdown_from_db(date)` | Desglose RCO951 (stocks vs ETF) |
    | `get_ib_tlt_value_from_db(date)` | Valor TLT en EUR |
    | `get_price_data(symbol, days)` | Obtiene precios históricos |
    | `create_candlestick_chart(df, symbol)` | Crea gráfico de velas |
    """)

    st.markdown("---")

    # =====================================================
    # CONFIGURACIÓN STREAMLIT
    # =====================================================
    st.header("7. Configuración Streamlit")

    st.markdown("""
    **Ejecutar dashboard:**
    ```bash
    streamlit run web/app.py --server.headless=true
    ```

    **Configuración en app.py:**
    ```python
    st.set_page_config(
        page_title="Financial Data Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ```

    **URL por defecto:** http://localhost:8501
    """)

    st.markdown("---")

    # =====================================================
    # DEPENDENCIAS
    # =====================================================
    st.header("8. Dependencias Principales")

    st.markdown("""
    | Librería | Uso |
    |----------|-----|
    | `streamlit` | Framework del dashboard |
    | `pandas` | Manejo de datos |
    | `plotly` | Gráficos interactivos |
    | `sqlalchemy` | ORM para base de datos |
    | `yfinance` | Datos de Yahoo Finance |
    """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Financial Data Project v3.0")
st.sidebar.markdown("*Data: 28/01/2026*")
