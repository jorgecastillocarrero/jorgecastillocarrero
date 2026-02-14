"""
Streamlit Dashboard for Financial Data Visualization.
Run with: streamlit run web/app.py
"""

import sys
from pathlib import Path

# Add project root to path BEFORE any other imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="PatrimonioSmart",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# AUTHENTICATION CHECK - Must be FIRST before any content
# =====================================================
from src.config import get_settings
settings = get_settings()

def check_authentication():
    """Check if user is authenticated when auth is enabled."""
    if not settings.dashboard_auth_enabled:
        return True

    if not settings.dashboard_password:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Cargar imagen de fondo como base64
    import base64
    import os

    bg_paths = [
        "web/static/login_bg.png",
        os.path.join(os.path.dirname(__file__), "static/login_bg.png"),
        "/app/web/static/login_bg.png",
    ]

    bg_base64 = ""
    for bg_path in bg_paths:
        if os.path.exists(bg_path):
            try:
                with open(bg_path, "rb") as img_file:
                    bg_base64 = base64.b64encode(img_file.read()).decode()
                break
            except Exception:
                continue

    # Página de login con imagen de fondo a pantalla completa
    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] {{display: none;}}
        [data-testid="stHeader"] {{display: none;}}
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            z-index: 0;
        }}
        .stApp > * {{
            position: relative;
            z-index: 1;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 3rem; font-weight: 700; color: #ffffff; letter-spacing: -1px;">
                Patrimonio<span style="color: #3a6a90;">Smart</span>
            </div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: 5px; letter-spacing: 2px;">
                GESTIÓN PATRIMONIAL INTELIGENTE
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <h2 style="color: #ffffff; text-align: center; margin-bottom: 30px; font-weight: 300;">
            Acceso Privado
        </h2>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Usuario", placeholder="Introduce tu usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Introduce tu contraseña")

            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Entrar", use_container_width=True)

            if submit:
                valid_user = username.lower() == "carihuela"
                valid_pass = password == settings.dashboard_password

                if valid_user and valid_pass:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")

        st.markdown("""
        <div style="text-align: center; margin-top: 30px; color: rgba(255,255,255,0.6);">
            <small>PatrimonioSmart 2026</small>
        </div>
        """, unsafe_allow_html=True)

    return False

# Check auth immediately
if not check_authentication():
    st.stop()

# =====================================================
# AUTHENTICATED - Load the rest of the app
# =====================================================
import logging
import importlib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from fpdf import FPDF
import io

# Force reload of portfolio_data to pick up changes
import src.portfolio_data
importlib.reload(src.portfolio_data)

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

def parse_db_date(date_value, default=None):
    """Parse date from database - handles both PostgreSQL (date obj) and SQLite (string)."""
    if date_value is None:
        return default
    if isinstance(date_value, datetime):
        return date_value.date()
    if isinstance(date_value, date):
        return date_value
    return datetime.strptime(str(date_value)[:10], '%Y-%m-%d').date()


def sanitize_text_for_pdf(text: str) -> str:
    """Remove special characters that fpdf can't handle."""
    if not isinstance(text, str):
        text = str(text)
    # Replace accented characters
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N', 'ü': 'u', 'Ü': 'U',
        '€': 'EUR', '£': 'GBP', '¥': 'JPY',
        '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-ASCII characters
    return text.encode('ascii', 'ignore').decode('ascii')


def generate_posicion_pdf(data: dict) -> bytes:
    """Generate PDF report for Posición page."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Helper to sanitize all text
    def s(text):
        return sanitize_text_for_pdf(text)

    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(36, 82, 122)  # #24527a
    pdf.cell(0, 15, 'POSICION GLOBAL', ln=True, align='C')
    pdf.set_text_color(0, 0, 0)

    # Date
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, s(f"Fecha: {data.get('fecha', '')}"), ln=True, align='R')
    pdf.ln(5)

    # Resumen de Cartera
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Resumen de Cartera', ln=True)
    pdf.set_font('Helvetica', '', 11)

    # Summary table
    col_widths = [50, 45, 45, 50]
    headers = ['Concepto', 'EUR', 'USD', '']

    pdf.set_fill_color(36, 82, 122)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 10)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, 1, 0, 'C', fill=True)
    pdf.ln()

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 10)

    rows = [
        ('Valor Inicial 31/12', data.get('valor_inicial_eur', ''), data.get('valor_inicial_usd', ''), ''),
        ('Valor Actual', data.get('valor_actual_eur', ''), data.get('valor_actual_usd', ''), ''),
        ('Ganancia Acumulada', data.get('ganancia_eur', ''), data.get('ganancia_usd', ''), ''),
        ('Rentabilidad Acumulada', data.get('rent_eur', ''), data.get('rent_usd', ''), ''),
        ('SPY Acumulado', '', '', data.get('spy_return', '')),
        ('QQQ Acumulado', '', '', data.get('qqq_return', '')),
    ]

    for row in rows:
        for i, val in enumerate(row):
            # Color for gains/losses
            if 'Ganancia' in row[0] or 'Rentabilidad' in row[0]:
                if '+' in str(val):
                    pdf.set_text_color(0, 150, 0)
                elif '-' in str(val):
                    pdf.set_text_color(200, 0, 0)
            pdf.cell(col_widths[i], 7, s(str(val)), 1, 0, 'C')
            pdf.set_text_color(0, 0, 0)
        pdf.ln()

    pdf.ln(8)

    # Variación Diaria por Tipo de Activo
    if 'variacion_diaria' in data and data['variacion_diaria']:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Variacion Diaria por Tipo de Activo', ln=True)
        pdf.set_font('Helvetica', '', 10)

        var_cols = [40, 35, 35, 35, 30]
        var_headers = ['Tipo', data.get('fecha_prev', ''), data.get('fecha_last', ''), 'Diferencia', 'Var %']

        pdf.set_fill_color(36, 82, 122)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 9)
        for i, h in enumerate(var_headers):
            pdf.cell(var_cols[i], 7, s(h), 1, 0, 'C', fill=True)
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 9)

        for row in data['variacion_diaria']:
            is_total = row.get('Tipo') == 'TOTAL'
            if is_total:
                pdf.set_font('Helvetica', 'B', 9)
                pdf.set_fill_color(180, 180, 180)

            for i, key in enumerate(['Tipo', 'prev', 'last', 'diff', 'var_pct']):
                val = str(row.get(key, ''))
                if key in ['diff', 'var_pct']:
                    if '+' in val:
                        pdf.set_text_color(0, 150, 0)
                    elif '-' in val:
                        pdf.set_text_color(200, 0, 0)
                pdf.cell(var_cols[i], 6, s(val), 1, 0, 'C', fill=is_total)
                pdf.set_text_color(0, 0, 0)
            pdf.ln()

            if is_total:
                pdf.set_font('Helvetica', '', 9)

    pdf.ln(8)

    # Rentabilidad Mensual
    if 'rentabilidad_mensual' in data and data['rentabilidad_mensual']:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Rentabilidad Mensual', ln=True)
        pdf.set_font('Helvetica', '', 9)

        # Monthly table
        months = ['Activo', 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic', 'TOTAL']
        month_cols = [20] + [12] * 13

        pdf.set_fill_color(36, 82, 122)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 7)
        for i, m in enumerate(months):
            pdf.cell(month_cols[i], 6, m, 1, 0, 'C', fill=True)
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 7)

        for row in data['rentabilidad_mensual']:
            for i, m in enumerate(months):
                val = str(row.get(m, ''))
                if m != 'Activo' and val:
                    if '+' in val:
                        pdf.set_text_color(0, 150, 0)
                    elif '-' in val:
                        pdf.set_text_color(200, 0, 0)
                pdf.cell(month_cols[i], 5, s(val), 1, 0, 'C')
                pdf.set_text_color(0, 0, 0)
            pdf.ln()

    # Footer
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, s(f"Generado por PatrimonioSmart - {datetime.now().strftime('%d/%m/%Y %H:%M')}"), ln=True, align='C')

    return bytes(pdf.output())


# Custom CSS for soft blue sidebar + responsive mobile
st.markdown("""
<style>
    /* Only sidebar with soft blue background */
    [data-testid="stSidebar"] {
        background-color: #4a6fa5;
    }
    /* All text in sidebar white */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* ============================================= */
    /* RESPONSIVE MOBILE STYLES                      */
    /* ============================================= */

    /* Mobile devices (up to 768px) */
    @media (max-width: 768px) {
        /* Main content padding */
        .main .block-container {
            padding: 1rem 0.5rem !important;
            max-width: 100% !important;
        }

        /* Reduce font sizes */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.1rem !important; }

        /* Tables responsive */
        .stDataFrame {
            font-size: 0.75rem !important;
            overflow-x: auto !important;
        }

        /* Metrics cards smaller */
        [data-testid="stMetric"] {
            padding: 0.5rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }

        /* Columns stack vertically */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }

        /* Charts full width */
        .js-plotly-plot {
            width: 100% !important;
        }

        /* Sidebar width on mobile */
        [data-testid="stSidebar"] {
            min-width: 200px !important;
            max-width: 250px !important;
        }

        /* Hide sidebar by default on mobile - user can expand */
        [data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: -250px !important;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
        }

        /* Buttons full width */
        .stButton > button {
            width: 100% !important;
            padding: 0.5rem !important;
        }

        /* Select boxes */
        .stSelectbox {
            font-size: 0.85rem !important;
        }
    }

    /* Small phones (up to 480px) */
    @media (max-width: 480px) {
        .main .block-container {
            padding: 0.5rem 0.25rem !important;
        }

        h1 { font-size: 1.25rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1rem !important; }

        [data-testid="stMetricValue"] {
            font-size: 1rem !important;
        }

        .stDataFrame {
            font-size: 0.65rem !important;
        }
    }

    /* Tablet landscape */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding: 1rem !important;
        }
    }

    /* Remove Streamlit default top padding and header */
    .stApp > header {
        display: none !important;
    }

    .main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Hide first element top margin */
    .main .block-container > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Navigation wrapper - full width, no gaps */
    .nav-wrapper {
        background: #24527a;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-top: -120px;
        padding: 160px calc(50vw - 50% + 1rem) 30px;
        min-height: 150px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(36, 82, 122, 0.3);
    }

    /* GLOBAL: Remove ALL borders from ALL selectboxes */
    .stSelectbox,
    .stSelectbox *,
    .stSelectbox div,
    .stSelectbox input,
    .stSelectbox [data-baseweb],
    .stSelectbox [data-baseweb] * {
        border: none !important;
        border-width: 0 !important;
        border-top: none !important;
        border-bottom: none !important;
        border-left: none !important;
        border-right: none !important;
        box-shadow: none !important;
        outline: none !important;
        background-image: none !important;
    }

    /* Hide BaseWeb animated underline */
    .stSelectbox [data-baseweb="select"]::after,
    .stSelectbox [data-baseweb="select"]::before,
    .stSelectbox [data-baseweb="select"] div::after,
    .stSelectbox [data-baseweb="select"] div::before {
        display: none !important;
        content: none !important;
        height: 0 !important;
        border: none !important;
        background: none !important;
        transform: none !important;
    }

    /* Ensure app starts at very top */
    .stApp {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Remove any top spacing from main container */
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }

    [data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Remove any borders/lines from navigation area */
    [data-testid="stHorizontalBlock"]:first-of-type,
    [data-testid="stHorizontalBlock"]:first-of-type *,
    [data-testid="stHorizontalBlock"]:first-of-type div,
    [data-testid="stHorizontalBlock"]:first-of-type span,
    [data-testid="stHorizontalBlock"]:first-of-type input {
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        border-color: #24527a !important;
        box-shadow: none !important;
        outline: none !important;
        text-decoration: none !important;
    }

    [data-testid="stHorizontalBlock"]:first-of-type::before,
    [data-testid="stHorizontalBlock"]:first-of-type::after,
    [data-testid="stHorizontalBlock"]:first-of-type *::before,
    [data-testid="stHorizontalBlock"]:first-of-type *::after {
        display: none !important;
        border: none !important;
    }

    /* Hide any hr elements */
    .nav-wrapper hr,
    [data-testid="stHorizontalBlock"]:first-of-type hr {
        display: none !important;
    }

    /* Remove bottom border from selectbox input */
    [data-testid="stHorizontalBlock"]:first-of-type [data-baseweb="select"],
    [data-testid="stHorizontalBlock"]:first-of-type [data-baseweb="select"] * {
        border: none !important;
        border-color: #24527a !important;
        box-shadow: none !important;
    }

    /* Force hide underline on selectbox - BaseWeb component */
    [data-testid="stHorizontalBlock"]:first-of-type [data-baseweb="select"] > div,
    [data-testid="stHorizontalBlock"]:first-of-type [data-baseweb="select"] > div > div {
        border-bottom: none !important;
        border-bottom-width: 0 !important;
        background-color: transparent !important;
    }

    [data-testid="stHorizontalBlock"]:first-of-type [data-baseweb="select"] > div::after,
    [data-testid="stHorizontalBlock"]:first-of-type [data-baseweb="select"] > div > div::after {
        display: none !important;
        background: none !important;
        border: none !important;
        height: 0 !important;
    }

    /* Target the specific underline element */
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox [class*="indicatorContainer"],
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox [class*="borderBottom"],
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox [class*="underline"],
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox [class*="Underline"] {
        display: none !important;
        border: none !important;
        background: none !important;
    }

    /* Override BaseWeb control border */
    [data-testid="stHorizontalBlock"]:first-of-type [class*="control"] {
        border: none !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }

    /* Title (POSICIÓN GLOBAL) - subir hacia arriba */
    .main h1 {
        margin-top: -90px !important;
    }

    /* Style selectboxes - positioned with margin-top */
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox {
        margin-top: -135px;
        position: relative;
    }

    /* Cover the line above selectbox with background color */
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox::before {
        content: "";
        position: absolute;
        top: -30px;
        left: -20px;
        right: -20px;
        height: 35px;
        background-color: #24527a;
        z-index: 9999;
    }

    /* Force remove ALL borders and lines from selectbox */
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox div[data-baseweb] div {
        border: 0 !important;
        border-width: 0 !important;
        border-style: none !important;
        border-color: transparent !important;
        background-image: none !important;
    }

    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox input {
        border: 0 !important;
        border-bottom: 0 !important;
        background: transparent !important;
        caret-color: transparent !important;
        color: transparent !important;
        pointer-events: none !important;
        user-select: none !important;
    }

    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox,
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox * {
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox > div > div {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        min-height: 20px !important;
    }

    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox > div > div:hover {
        background-color: transparent !important;
    }

    /* Text color white in selectboxes */
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox > div > div > div {
        color: white !important;
        font-weight: 500 !important;
    }

    /* Arrow color white */
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox svg {
        fill: white !important;
    }

    /* Hide selectbox labels */
    [data-testid="stHorizontalBlock"]:first-of-type .stSelectbox label {
        display: none !important;
    }

    /* Logo in nav */
    [data-testid="stHorizontalBlock"]:first-of-type .stImage img {
        margin-top: -140px;
    }

    /* Logout button style */
    [data-testid="stHorizontalBlock"]:first-of-type .stButton {
        margin-top: -120px;
    }

    [data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
        background-color: transparent !important;
        color: white !important;
        border: none !important;
        border-radius: 0 !important;
        min-height: 20px !important;
        font-weight: 500 !important;
    }

    [data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
        background-color: transparent !important;
    }

    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
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

ASSET_TYPES_ORDER = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'ETFs', 'Cash']


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
            # Convert to date object (PostgreSQL may return datetime)
            fecha = row[0]
            if isinstance(fecha, str):
                fecha = datetime.strptime(fecha, '%Y-%m-%d').date()
            elif isinstance(fecha, datetime):
                fecha = fecha.date()
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
# Elegant Horizontal Navigation with Logo
# =============================================================================

# Define page groups for dropdown menus
page_groups = {
    "Cartera": ["Posición", "Composición", "Acciones", "ETFs", "Futuros"],
    "Estacionalidad": ["Carih Mensual", "Screener", "Symbol Analysis"],
    "Gráficos": ["Data Management", "Download Status", "BBDD", "Pantalla"],
    "Indicadores": ["Indicadores"],
    "Mercado": ["VIX"],
    "Asistente IA": ["Asistente IA"],
    "Noticias": ["Noticias"]
}

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Posición"
if "current_group" not in st.session_state:
    st.session_state.current_group = "Cartera"

# Navigation bar HTML wrapper
st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)

# Create columns: Logo + 7 menus + spacer
logo_col, m1, m2, m3, m4, m5, m6, m7, spacer = st.columns([1.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 2.6])

# Logo
with logo_col:
    import os
    logo_paths = [
        "web/static/logo_carihuela_final.jpg",
        os.path.join(os.path.dirname(__file__), "static/logo_carihuela_final.jpg"),
        "/app/web/static/logo_carihuela_final.jpg",
    ]
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            st.image(logo_path, width=320)
            break

# Menu dropdowns
with m1:
    sel = st.selectbox("Cartera", ["Cartera"] + page_groups["Cartera"],
        index=0 if st.session_state.current_group != "Cartera" else page_groups["Cartera"].index(st.session_state.current_page) + 1 if st.session_state.current_page in page_groups["Cartera"] else 0,
        key="nav_cartera", label_visibility="collapsed")
    if sel and sel not in ["Cartera"]:
        st.session_state.current_page = sel
        st.session_state.current_group = "Cartera"

with m2:
    sel = st.selectbox("Estacionalidad", ["Estacionalidad"] + page_groups["Estacionalidad"],
        index=0 if st.session_state.current_group != "Estacionalidad" else page_groups["Estacionalidad"].index(st.session_state.current_page) + 1 if st.session_state.current_page in page_groups["Estacionalidad"] else 0,
        key="nav_estacionalidad", label_visibility="collapsed")
    if sel and sel not in ["Estacionalidad"]:
        st.session_state.current_page = sel
        st.session_state.current_group = "Estacionalidad"

with m3:
    sel = st.selectbox("Gráficos", ["Gráficos"] + page_groups["Gráficos"],
        index=0 if st.session_state.current_group != "Gráficos" else page_groups["Gráficos"].index(st.session_state.current_page) + 1 if st.session_state.current_page in page_groups["Gráficos"] else 0,
        key="nav_graficos", label_visibility="collapsed")
    if sel and sel not in ["Gráficos"]:
        st.session_state.current_page = sel
        st.session_state.current_group = "Gráficos"

with m4:
    sel = st.selectbox("Indicadores", ["Indicadores"] + page_groups["Indicadores"],
        index=0 if st.session_state.current_group != "Indicadores" else page_groups["Indicadores"].index(st.session_state.current_page) + 1 if st.session_state.current_page in page_groups["Indicadores"] else 0,
        key="nav_indicadores", label_visibility="collapsed")
    if sel and sel not in ["Indicadores"]:
        st.session_state.current_page = sel
        st.session_state.current_group = "Indicadores"

with m5:
    sel = st.selectbox("Mercado", ["Mercado"] + page_groups["Mercado"],
        index=0 if st.session_state.current_group != "Mercado" else page_groups["Mercado"].index(st.session_state.current_page) + 1 if st.session_state.current_page in page_groups["Mercado"] else 0,
        key="nav_mercado", label_visibility="collapsed")
    if sel and sel not in ["Mercado"]:
        st.session_state.current_page = sel
        st.session_state.current_group = "Mercado"

with m6:
    sel = st.selectbox("Asistente IA", ["Asistente IA"] + page_groups["Asistente IA"],
        index=0 if st.session_state.current_group != "Asistente IA" else page_groups["Asistente IA"].index(st.session_state.current_page) + 1 if st.session_state.current_page in page_groups["Asistente IA"] else 0,
        key="nav_ia", label_visibility="collapsed")
    if sel and sel not in ["Asistente IA"]:
        st.session_state.current_page = sel
        st.session_state.current_group = "Asistente IA"

with m7:
    sel = st.selectbox("Noticias", ["Noticias"] + page_groups["Noticias"],
        index=0 if st.session_state.current_group != "Noticias" else page_groups["Noticias"].index(st.session_state.current_page) + 1 if st.session_state.current_page in page_groups["Noticias"] else 0,
        key="nav_noticias", label_visibility="collapsed")
    if sel and sel not in ["Noticias"]:
        st.session_state.current_page = sel
        st.session_state.current_group = "Noticias"

# Logout button aligned with menus
with spacer:
    spacer_left, spacer_right = st.columns([5, 1])
    with spacer_right:
        if settings.dashboard_auth_enabled and st.session_state.get("authenticated", False):
            if st.button("Cerrar Sesión", key="logout_nav"):
                st.session_state.authenticated = False
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Get current page
page = st.session_state.current_page

# Submenú for Backtesting
backtesting_option = None
if page == "Carih Mensual":
    bt_col1, bt_col2, bt_col3 = st.columns([1, 2, 3])
    with bt_col1:
        backtesting_option = st.selectbox(
            "Tipo de análisis",
            ["Estrategia Mensual", "Portfolio Backtest"],
            key="backtesting_type"
        )

st.markdown("---")

# =============================================================================
# Main Content
# =============================================================================

if page == "Posición":
    st.title("POSICIÓN GLOBAL")

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
            SELECT MAX(fecha) FROM posicion WHERE fecha < CURRENT_DATE
        """))
        latest_date_val = result.fetchone()[0]
        latest_date = parse_db_date(latest_date_val, date(2026, 1, 28))

        # Obtener fecha anterior para comparación diaria
        result = session.execute(text("""
            SELECT MAX(fecha) FROM posicion WHERE fecha < :latest
        """), {'latest': latest_date})
        prev_date_val = result.fetchone()[0]
        prev_date = parse_db_date(prev_date_val, latest_date)

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
        .red {{ color: #ff4444; }}
        .benchmark-green {{ font-size: 22px; font-weight: bold; color: #00cc00; }}
        .benchmark-red {{ font-size: 22px; font-weight: bold; color: #ff4444; }}
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
                <span class="eur-value {'green' if return_eur >= 0 else 'red'}">{format_eur(return_eur, show_sign=True)}</span>
                <span class="usd-value {'green' if return_usd >= 0 else 'red'}">{format_usd(return_usd, show_sign=True)}</span>
            </td>
            <td>
                <span class="eur-value {'green' if return_pct >= 0 else 'red'}">{format_pct(return_pct)}</span>
                <span class="usd-value {'green' if return_pct_usd >= 0 else 'red'}">{format_pct(return_pct_usd)}</span>
            </td>
            <td><span class="{'benchmark-green' if spy_return >= 0 else 'benchmark-red'}">{format_pct(spy_return)}</span></td>
            <td><span class="{'benchmark-green' if qqq_return >= 0 else 'benchmark-red'}">{format_pct(qqq_return)}</span></td>
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

        # Build comparison table - get all types dynamically from data
        comparison_data = []
        all_tipos = set(prev_totals.keys()) | set(last_totals.keys())
        # Order: known types first, then others
        tipo_order = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'ETFs', 'Cash']
        tipos = [t for t in tipo_order if t in all_tipos] + [t for t in sorted(all_tipos) if t not in tipo_order]
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
    # Use portfolio dates as the common x-axis (only trading days)
    trading_dates_labels = [d.strftime('%d/%m') for d in portfolio_returns_eur.index]

    fig = go.Figure()

    # 1. Cartera EUR - daily data
    fig.add_trace(go.Scatter(
        x=trading_dates_labels,
        y=portfolio_returns_eur.values,
        mode='lines+markers',
        name=f'Cartera EUR ({return_pct:+.2f}%)',
        line=dict(color='#00cc00', width=3),
        marker=dict(size=6)
    ))

    # 2. Cartera USD - daily data
    if isinstance(portfolio_returns_usd, pd.Series):
        fig.add_trace(go.Scatter(
            x=trading_dates_labels,
            y=portfolio_returns_usd.values,
            mode='lines+markers',
            name=f'Cartera USD ({return_pct_usd:+.2f}%)',
            line=dict(color='#90EE90', width=3, dash='dash'),
            marker=dict(size=6)
        ))

    # 3. SPY benchmark - align to portfolio dates
    if not spy_prices.empty and len(spy_prices) >= 2:
        spy_base = spy_prices['close'].iloc[0]
        spy_aligned = []
        for dt in portfolio_returns_eur.index:
            dt_date = dt.date() if hasattr(dt, 'date') else dt
            matching = spy_prices[spy_prices.index.date == dt_date]
            if not matching.empty:
                spy_aligned.append(((matching['close'].iloc[0] - spy_base) / spy_base) * 100)
            elif spy_aligned:
                spy_aligned.append(spy_aligned[-1])  # Carry forward
            else:
                spy_aligned.append(0)
        fig.add_trace(go.Scatter(
            x=trading_dates_labels,
            y=spy_aligned,
            mode='lines',
            name=f'SPY ({spy_return:+.2f}%)',
            line=dict(color='#636EFA', width=2)
        ))

    # 4. QQQ benchmark - align to portfolio dates
    if not qqq_prices.empty and len(qqq_prices) >= 2:
        qqq_base = qqq_prices['close'].iloc[0]
        qqq_aligned = []
        for dt in portfolio_returns_eur.index:
            dt_date = dt.date() if hasattr(dt, 'date') else dt
            matching = qqq_prices[qqq_prices.index.date == dt_date]
            if not matching.empty:
                qqq_aligned.append(((matching['close'].iloc[0] - qqq_base) / qqq_base) * 100)
            elif qqq_aligned:
                qqq_aligned.append(qqq_aligned[-1])  # Carry forward
            else:
                qqq_aligned.append(0)
        fig.add_trace(go.Scatter(
            x=trading_dates_labels,
            y=qqq_aligned,
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(
            type='category',  # Eje categórico - sin huecos entre días
            tickangle=-45
        )
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

    # =========================================================================
    # RENTABILIDAD MENSUAL
    # =========================================================================
    st.markdown("---")
    st.subheader("Rentabilidad Mensual")

    # Calculate monthly returns for Portfolio, SPY, and QQQ
    months_data = {}
    current_year = date.today().year
    current_month = date.today().month

    # Get first day value for each month (for monthly returns calculation)
    for month in range(1, 13):
        month_name = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                      'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][month - 1]

        # Skip future months (after current month)
        if month > current_month:
            continue

        # Find first and last trading day of the month
        month_dates = [d for d in trading_dates_from_db
                       if d.year == current_year and d.month == month and d not in excluded_dates]

        if not month_dates:
            continue

        first_day = min(month_dates)
        # For current month: use latest available date (current position)
        # For past months: use last day of that month
        last_day = max(month_dates)

        # Get values at start and end of month
        if month == 1:
            # January: compare to Dec 31
            start_val = all_day_totals.get(date(2025, 12, 31), 0)
        else:
            # Other months: get last day of previous month
            prev_month_dates = [d for d in trading_dates_from_db
                               if d.year == current_year and d.month == month - 1 and d not in excluded_dates]
            if prev_month_dates:
                start_val = all_day_totals.get(max(prev_month_dates), 0)
            else:
                start_val = all_day_totals.get(date(2025, 12, 31), 0)

        end_val = all_day_totals.get(last_day, 0)

        # Portfolio monthly return
        if start_val > 0:
            port_monthly_ret = ((end_val / start_val) - 1) * 100
        else:
            port_monthly_ret = 0

        # SPY monthly return
        spy_monthly_ret = 0
        if not spy_prices.empty:
            spy_month = spy_prices[spy_prices.index.month == month]
            if not spy_month.empty:
                if month == 1:
                    spy_start = spy_prices['close'].iloc[0]
                else:
                    spy_prev = spy_prices[spy_prices.index.month == month - 1]
                    spy_start = spy_prev['close'].iloc[-1] if not spy_prev.empty else spy_prices['close'].iloc[0]
                spy_end = spy_month['close'].iloc[-1]
                if spy_start > 0:
                    spy_monthly_ret = ((spy_end / spy_start) - 1) * 100

        # QQQ monthly return
        qqq_monthly_ret = 0
        if not qqq_prices.empty:
            qqq_month = qqq_prices[qqq_prices.index.month == month]
            if not qqq_month.empty:
                if month == 1:
                    qqq_start = qqq_prices['close'].iloc[0]
                else:
                    qqq_prev = qqq_prices[qqq_prices.index.month == month - 1]
                    qqq_start = qqq_prev['close'].iloc[-1] if not qqq_prev.empty else qqq_prices['close'].iloc[0]
                qqq_end = qqq_month['close'].iloc[-1]
                if qqq_start > 0:
                    qqq_monthly_ret = ((qqq_end / qqq_start) - 1) * 100

        months_data[month_name] = {
            'Cartera': port_monthly_ret,
            'SPY': spy_monthly_ret,
            'QQQ': qqq_monthly_ret
        }

    # Build monthly returns table (vertical: assets as rows, months as columns)
    all_months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

    # Build table with Activo as first column, then all months, then TOTAL
    monthly_table = {'Activo': ['Cartera', 'SPY', 'QQQ']}

    for month_name in all_months:
        if month_name in months_data:
            monthly_table[month_name] = [
                months_data[month_name]['Cartera'],
                months_data[month_name]['SPY'],
                months_data[month_name]['QQQ']
            ]
        else:
            # Month without data yet
            monthly_table[month_name] = [None, None, None]

    # Add TOTAL column
    monthly_table['TOTAL'] = [return_pct, spy_return, qqq_return]

    monthly_df = pd.DataFrame(monthly_table)

    # Replace None/NaN with empty string for display
    pct_columns = [col for col in monthly_df.columns if col != 'Activo']
    for col in pct_columns:
        monthly_df[col] = monthly_df[col].apply(
            lambda x: '' if x is None or (isinstance(x, float) and pd.isna(x)) else x
        )

    # Style the table
    def color_monthly_pct(val):
        if val != '' and isinstance(val, (int, float)):
            if val > 0:
                return 'background-color: #2E7D32; color: white'
            elif val < 0:
                return 'background-color: #C62828; color: white'
        return ''

    def format_pct_monthly(val):
        if val == '' or val is None:
            return ''
        try:
            return f'{float(val):+.2f}%'
        except (ValueError, TypeError):
            return ''

    format_dict = {col: format_pct_monthly for col in pct_columns}

    styled_monthly = monthly_df.style.map(color_monthly_pct, subset=pct_columns).format(format_dict)

    st.dataframe(styled_monthly, use_container_width=True, hide_index=True)

    # =========================================================================
    # PDF EXPORT BUTTON
    # =========================================================================
    st.markdown("---")

    # Prepare data for PDF
    def format_eur_pdf(value, show_sign=False):
        if show_sign:
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:,.0f} EUR".replace(",", ".")
        return f"{value:,.0f} EUR".replace(",", ".")

    def format_usd_pdf(value, show_sign=False):
        if show_sign:
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:,.0f} USD".replace(",", ".")
        return f"{value:,.0f} USD".replace(",", ".")

    def format_pct_pdf(value):
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"

    # Collect variación diaria data
    variacion_data = []
    if 'comparison_data' in dir() and comparison_data:
        for row in comparison_data:
            variacion_data.append({
                'Tipo': row.get('Tipo', ''),
                'prev': row.get(f'{day_prev.strftime("%d/%m")}', ''),
                'last': row.get(f'{day_last.strftime("%d/%m")}', ''),
                'diff': row.get('Diferencia', ''),
                'var_pct': row.get('Var %', '')
            })

    # Collect monthly returns data
    monthly_data = []
    for _, row in monthly_df.iterrows():
        row_dict = {'Activo': row.get('Activo', '')}
        for col in monthly_df.columns:
            if col != 'Activo':
                val = row.get(col)
                if pd.notna(val) and val is not None:
                    try:
                        row_dict[col] = f"{float(val):+.2f}%"
                    except:
                        row_dict[col] = ''
                else:
                    row_dict[col] = ''
        monthly_data.append(row_dict)

    pdf_data = {
        'fecha': latest_date.strftime('%d/%m/%Y') if latest_date else '',
        'fecha_prev': day_prev.strftime('%d/%m') if 'day_prev' in dir() and day_prev else '',
        'fecha_last': day_last.strftime('%d/%m') if 'day_last' in dir() and day_last else '',
        'valor_inicial_eur': format_eur_pdf(initial_value),
        'valor_inicial_usd': format_usd_pdf(initial_value * eur_usd_31dic),
        'valor_actual_eur': format_eur_pdf(current_value),
        'valor_actual_usd': format_usd_pdf(current_value * eur_usd_current),
        'ganancia_eur': format_eur_pdf(return_eur, show_sign=True),
        'ganancia_usd': format_usd_pdf(return_usd, show_sign=True),
        'rent_eur': format_pct_pdf(return_pct),
        'rent_usd': format_pct_pdf(return_pct_usd),
        'spy_return': format_pct_pdf(spy_return),
        'qqq_return': format_pct_pdf(qqq_return),
        'variacion_diaria': variacion_data,
        'rentabilidad_mensual': monthly_data
    }

    pdf_bytes = generate_posicion_pdf(pdf_data)

    st.download_button(
        label="Descargar PDF",
        data=pdf_bytes,
        file_name=f"posicion_global_{latest_date.strftime('%Y%m%d') if latest_date else 'hoy'}.pdf",
        mime="application/pdf",
        key="download_pdf_posicion"
    )


elif page == "Composición":
    st.title("COMPOSICIÓN DE CARTERA")

    # Get latest date (mismo método que Posición)
    with db.get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("""
            SELECT MAX(fecha) FROM posicion WHERE fecha < CURRENT_DATE
        """))
        latest_date_val = result.fetchone()[0]
        latest_date = parse_db_date(latest_date_val, date.today())

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
             strategy_values.get('Mensual', 0) +
             strategy_values.get('Stock', 0))
    oro = strategy_values.get('Oro/Mineras', 0)
    liquidez = strategy_values.get('Cash', 0) + strategy_values.get('ETFs', 0)

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


elif page == "Acciones":
    st.title("ACCIONES")
    loading_placeholder = st.empty()
    loading_placeholder.info("Cargando datos...")

    # Using centralized ASSET_TYPE_MAP and CURRENCY_SYMBOL_MAP from portfolio_data

    # Date range
    today = date.today()
    start_date = datetime(2025, 12, 30)

    # Get latest trading date (mismo método que Posición)
    with db.get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("""
            SELECT MAX(fecha) FROM posicion WHERE fecha < CURRENT_DATE
        """))
        latest_date_val = result.fetchone()[0]
        latest_date = parse_db_date(latest_date_val, today)

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
        # Get holdings with fecha_compra and precio_compra from compras table (datos reales)
        # Excluir ETFs y Futuros (solo mostrar acciones)
        holdings_result = session.execute(text("""
            SELECT h.account_code, h.symbol, h.shares, h.currency, h.asset_type,
                   c.fecha as fecha_compra, c.precio as precio_compra
            FROM holding_diario h
            LEFT JOIN (
                SELECT account_code, symbol, MIN(fecha) as fecha,
                       (SELECT precio FROM compras c2
                        WHERE c2.account_code = c1.account_code
                        AND c2.symbol = c1.symbol
                        ORDER BY fecha LIMIT 1) as precio
                FROM compras c1
                GROUP BY account_code, symbol
            ) c ON h.account_code = c.account_code AND h.symbol = c.symbol
            WHERE h.fecha = :fecha
            AND (h.asset_type IS NULL OR h.asset_type NOT IN ('ETF', 'ETFs', 'Future', 'Futures'))
            ORDER BY h.account_code, h.symbol
        """), {'fecha': latest_date})

        holdings_data = []
        for account, ticker, shares, currency_code, asset_type, fecha_compra, precio_compra_db in holdings_result.fetchall():
            parts = ticker.split('.')
            exchange = parts[1] if len(parts) > 1 else 'US'
            currency = EXCHANGE_TO_CURRENCY.get(exchange, '$')
            cuenta_display = 'La Caixa' if account == 'LACAIXA' else account

            holdings_data.append({
                'ticker': parts[0],
                'ticker_full': ticker,
                'cuenta': cuenta_display,
                'account_code': account,
                'shares': shares,
                'currency': currency,
                'tipo': asset_type or 'Otro',
                'fecha_compra': fecha_compra,  # None si no hay registro en compras
                'precio_compra_db': precio_compra_db  # Precio de compra de la tabla compras
            })

        # Calculate returns for each holding
        asset_returns = []
        for holding in holdings_data:
            ticker = holding['ticker']
            ticker_full = holding['ticker_full']
            fecha_compra = holding['fecha_compra']
            precio_compra_db = holding['precio_compra_db']  # Precio real de compra

            if isinstance(fecha_compra, str):
                fecha_compra = datetime.strptime(fecha_compra, '%Y-%m-%d').date()
            elif isinstance(fecha_compra, datetime):
                fecha_compra = fecha_compra.date()

            # Get symbol for price lookup
            if '.' in ticker_full:
                symbol = session.query(Symbol).filter(Symbol.code == ticker_full).first()
                if not symbol:
                    symbol = session.query(Symbol).filter(Symbol.code == ticker).first()
            else:
                symbol = session.query(Symbol).filter(Symbol.code == ticker).first()

            if symbol:
                # Usar precio de compra de la BD si existe, sino buscar por fecha
                precio_compra = precio_compra_db
                if not precio_compra and fecha_compra:
                    precio_compra = portfolio_service.get_symbol_price(ticker_full, fecha_compra)
                    if not precio_compra:
                        precio_compra = portfolio_service.get_symbol_price(ticker, fecha_compra)

                # Get price at 31/12 (para Rent.Periodo cuando compra es anterior)
                precio_31dic = portfolio_service.get_symbol_price(ticker_full, date(2025, 12, 31))
                if not precio_31dic:
                    precio_31dic = portfolio_service.get_symbol_price(ticker, date(2025, 12, 31))

                # Get current price
                precio_actual = portfolio_service.get_symbol_price(ticker_full, latest_date)
                if not precio_actual:
                    precio_actual = portfolio_service.get_symbol_price(ticker, latest_date)

                if precio_actual:
                    currency = holding['currency']
                    shares = holding['shares']
                    currency_symbol = {'$': '$', 'C$': 'C$', 'CHF': 'CHF', '€': '€'}.get(currency, currency)

                    # Get exchange rates
                    if currency == '$':  # USD
                        if fecha_compra and precio_compra:
                            eur_rate_compra = portfolio_service.get_exchange_rate('EURUSD=X', fecha_compra) or eur_usd_current
                            valor_compra_eur = (shares * precio_compra) / eur_rate_compra
                        else:
                            valor_compra_eur = None
                        valor_31dic_eur = (shares * precio_31dic) / eur_usd_31dic if precio_31dic else None
                        valor_actual_eur = (shares * precio_actual) / eur_usd_current
                    elif currency == 'C$':  # CAD
                        if fecha_compra and precio_compra:
                            cad_rate_compra = portfolio_service.get_exchange_rate('CADEUR=X', fecha_compra) or cad_eur_current
                            valor_compra_eur = (shares * precio_compra) * cad_rate_compra
                        else:
                            valor_compra_eur = None
                        valor_31dic_eur = (shares * precio_31dic) * cad_eur_31dic if precio_31dic else None
                        valor_actual_eur = (shares * precio_actual) * cad_eur_current
                    elif currency == 'CHF':  # Swiss Franc
                        if fecha_compra and precio_compra:
                            chf_rate_compra = portfolio_service.get_exchange_rate('CHFEUR=X', fecha_compra) or chf_eur_current
                            valor_compra_eur = (shares * precio_compra) * chf_rate_compra
                        else:
                            valor_compra_eur = None
                        valor_31dic_eur = (shares * precio_31dic) * chf_eur_31dic if precio_31dic else None
                        valor_actual_eur = (shares * precio_actual) * chf_eur_current
                    else:  # EUR
                        valor_compra_eur = shares * precio_compra if precio_compra else None
                        valor_31dic_eur = shares * precio_31dic if precio_31dic else None
                        valor_actual_eur = shares * precio_actual

                    # Calculate returns
                    # Rent.Compra: último precio / precio compra - 1 (en moneda local, sin efecto divisa)
                    rent_desde_compra = ((precio_actual / precio_compra) - 1) * 100 if precio_compra and precio_compra > 0 else None

                    # Rent. Periodo:
                    # - Si estaba en portfolio a 31/12/2025: rentabilidad EUR desde 31/12
                    # - Si compra desde 02/01/2026: último precio / precio compra - 1 (moneda local)
                    fecha_corte = date(2025, 12, 31)
                    if fecha_compra and fecha_compra > fecha_corte:
                        # Compra desde 02/01/2026: usar precio (moneda local)
                        rent_periodo = ((precio_actual / precio_compra) - 1) * 100 if precio_compra and precio_compra > 0 else None
                        rent_eur_abs = valor_actual_eur - valor_compra_eur if valor_compra_eur else None
                    else:
                        # Estaba en portfolio a 31/12/2025: usar valor EUR desde 31/12
                        rent_periodo = ((valor_actual_eur / valor_31dic_eur) - 1) * 100 if valor_31dic_eur and valor_31dic_eur > 0 else None
                        rent_eur_abs = valor_actual_eur - valor_31dic_eur if valor_31dic_eur else None

                    # Formatear precios con símbolo de moneda
                    precio_compra_display = f"{currency_symbol}{precio_compra:.2f}" if precio_compra else '-'
                    precio_actual_display = f"{currency_symbol}{precio_actual:.2f}"

                    asset_returns.append({
                        'F.Compra': fecha_compra,  # Mantener como date para ordenamiento correcto
                        'Ticker': holding['ticker_full'],
                        'Tipo': holding['tipo'],
                        'Cuenta': holding['cuenta'],
                        'Títulos': int(holding['shares']),
                        'P.Compra': precio_compra_display,
                        'Últ.Precio': precio_actual_display,
                        'Valor EUR': valor_actual_eur,
                        'Rent.Compra %': rent_desde_compra if rent_desde_compra is not None else 0,
                        'Rent.Periodo %': rent_periodo if rent_periodo is not None else 0,
                        'Rent.Periodo EUR': rent_eur_abs if rent_eur_abs is not None else 0,
                        '_tiene_fecha': fecha_compra is not None
                    })

        # Separar posiciones CON y SIN fecha de compra
        all_returns_df = pd.DataFrame(asset_returns)
        if not all_returns_df.empty:
            # Posiciones CON fecha de compra (para tabla principal)
            asset_returns_df = all_returns_df[all_returns_df['_tiene_fecha'] == True].copy()
            # Posiciones SIN fecha de compra (para mostrar como faltantes)
            missing_data_df = all_returns_df[all_returns_df['_tiene_fecha'] == False].copy()
            # Eliminar columna auxiliar
            if not asset_returns_df.empty:
                asset_returns_df = asset_returns_df.drop(columns=['_tiene_fecha'])
            if not missing_data_df.empty:
                missing_data_df = missing_data_df.drop(columns=['_tiene_fecha'])
        else:
            asset_returns_df = all_returns_df
            missing_data_df = pd.DataFrame()
        if not asset_returns_df.empty:
            # Ordenar por fecha de entrada (más antigua primero)
            asset_returns_df = asset_returns_df.sort_values('F.Compra', ascending=True, na_position='last')

            # Calcular valor inicial para estadísticas
            asset_returns_df['Valor_Inicial_EUR'] = asset_returns_df['Valor EUR'] / (1 + asset_returns_df['Rent.Periodo %'] / 100)

            # Calculate totals for open positions
            total_valor_eur = asset_returns_df['Valor EUR'].sum()
            total_rent_eur = asset_returns_df['Rent.Periodo EUR'].sum()
            total_inicial_eur = asset_returns_df['Valor_Inicial_EUR'].sum()
            total_rent_pct = ((total_valor_eur / total_inicial_eur) - 1) * 100 if total_inicial_eur > 0 else 0
            avg_rent_compra = asset_returns_df['Rent.Compra %'].mean() if not asset_returns_df.empty else 0

            # =====================================================
            # CALCULAR POSICIONES CERRADAS PARA ESTADÍSTICAS
            # (Consolidado por fecha/cuenta/símbolo)
            # Excluir Futuros y ETFs (solo acciones)
            # =====================================================
            with db.get_session() as session:
                ventas_result = session.execute(text("""
                    SELECT fecha, account_code, symbol,
                           SUM(shares) as total_shares,
                           SUM(importe_total) / SUM(shares) as precio_promedio,
                           SUM(importe_total) as importe_total,
                           currency,
                           SUM(COALESCE(pnl, 0)) as pnl,
                           AVG(precio_31_12) as precio_31_12,
                           AVG(rent_periodo) as rent_periodo
                    FROM ventas
                    WHERE symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
                    AND symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
                    GROUP BY fecha, account_code, symbol, currency
                    ORDER BY fecha DESC
                """))
                ventas_rows = ventas_result.fetchall()

                # Obtener precios de compra
                compras_result = session.execute(text("""
                    SELECT symbol, AVG(precio) as precio_compra, MIN(fecha) as fecha_compra
                    FROM compras
                    GROUP BY symbol
                """))
                precios_compra = {row[0]: {'precio': row[1], 'fecha': row[2]} for row in compras_result.fetchall()}

            closed_positions = []
            closed_periodo = []
            closed_historica = []
            total_closed_pnl_eur = 0
            total_closed_valor_eur = 0
            total_historica_eur = 0

            for venta in ventas_rows:
                fecha_venta, cuenta, symbol, shares, precio_venta, importe_venta, currency, pnl_db, precio_31_12_db, rent_periodo_db = venta

                # Obtener precio de compra real PRIMERO (necesario para decidir base del periodo)
                compra_info = precios_compra.get(symbol, {})
                precio_compra_real = compra_info.get('precio')
                fecha_compra = compra_info.get('fecha')
                if isinstance(fecha_compra, datetime):
                    fecha_compra = fecha_compra.date()

                # Determinar símbolo de moneda y conversión
                if currency == 'USD':
                    fx_rate = eur_usd_current
                    currency_symbol = '$'
                elif currency == 'GBP':
                    fx_rate = 1/1.18
                    currency_symbol = '£'
                elif currency == 'EUR':
                    fx_rate = 1.0
                    currency_symbol = '€'
                else:
                    fx_rate = 1.0
                    currency_symbol = currency

                # Precio base para periodo: 31/12 si existía antes, o compra si fue después
                fecha_corte = date(2025, 12, 31)
                if fecha_compra and fecha_compra > fecha_corte and precio_compra_real:
                    # Compra en 2026: usar precio de compra como base
                    precio_periodo = precio_compra_real
                    rent_periodo = ((precio_venta / precio_compra_real) - 1) * 100 if precio_compra_real > 0 else 0
                else:
                    # Existía a 31/12: usar precio 31/12
                    precio_periodo = precio_31_12_db if precio_31_12_db else portfolio_service.get_symbol_price(symbol, date(2025, 12, 31))
                    if precio_periodo is None:
                        precio_periodo = precio_compra_real if precio_compra_real else precio_venta
                    rent_periodo = rent_periodo_db if rent_periodo_db else ((precio_venta / precio_periodo - 1) * 100 if precio_periodo > 0 else 0)

                pnl_periodo = (precio_venta - precio_periodo) * shares
                pnl_eur = pnl_periodo / fx_rate if fx_rate != 1.0 else pnl_periodo
                valor_venta_eur = importe_venta / fx_rate if fx_rate != 1.0 else importe_venta

                total_closed_pnl_eur += pnl_eur
                total_closed_valor_eur += valor_venta_eur

                # Calcular rentabilidad histórica (desde compra)
                if precio_compra_real and precio_compra_real > 0:
                    rent_historica = (precio_venta / precio_compra_real - 1) * 100
                    pnl_historico = (precio_venta - precio_compra_real) * shares
                    pnl_historico_eur = pnl_historico / fx_rate if fx_rate != 1.0 else pnl_historico
                else:
                    rent_historica = 0
                    pnl_historico_eur = 0

                total_historica_eur += pnl_historico_eur

                # Añadir todas las ventas a closed_periodo (no solo las que tienen precio_31_12)
                # Determinar precio base para periodo: 31/12 si existía antes, o compra si fue después
                if fecha_compra and fecha_compra > date(2025, 12, 31):
                    precio_31_12_display = '-'
                elif precio_31_12_db:
                    precio_31_12_display = f"{currency_symbol}{precio_periodo:.2f}"
                else:
                    precio_31_12_display = '-'

                closed_periodo.append({
                    'Fecha': fecha_venta,
                    'Ticker': symbol,
                    'Títulos': int(shares),
                    'P.Compra': f"{currency_symbol}{precio_compra_real:.2f}" if precio_compra_real else '-',
                    'P.31/12': precio_31_12_display,
                    'P.Venta': f"{currency_symbol}{precio_venta:.2f}",
                    'Rent.Periodo %': rent_periodo,
                    'Rent.Histórica %': rent_historica,
                    'Rent.Periodo EUR': pnl_eur,
                    'Rent.Histórica EUR': pnl_historico_eur,
                })

                # closed_positions mantiene compatibilidad (incluye datos históricos)
                closed_positions.append({
                    'Fecha Venta': fecha_venta,
                    'Ticker': symbol,
                    'Cuenta': cuenta,
                    'Titulos': int(shares),
                    'Precio Compra': f"{currency_symbol}{precio_compra_real:.2f}" if precio_compra_real else f"{currency_symbol}0.00",
                    'Precio Venta': f"{currency_symbol}{precio_venta:.2f}",
                    'Rent. %': rent_periodo,
                    'Rent. EUR': pnl_eur,
                    'Rent.Histórica %': rent_historica,
                    'Rent.Histórica EUR': pnl_historico_eur,
                })

            # Combinar estadísticas: abiertas + cerradas
            import numpy as np

            # === ESTADÍSTICAS PERIODO (desde 31/12) ===
            all_periodo_pct = list(asset_returns_df['Rent.Periodo %'].values) + [c['Rent. %'] for c in closed_positions]
            all_periodo_eur = list(asset_returns_df['Rent.Periodo EUR'].values) + [c['Rent. EUR'] for c in closed_positions]

            # === ESTADÍSTICAS HISTÓRICAS (desde compra) ===
            # Para abiertas: Rent.Compra %, para cerradas: Rent.Histórica %
            all_historica_pct = list(asset_returns_df['Rent.Compra %'].values) + [c.get('Rent.Histórica %', 0) for c in closed_positions]
            all_historica_eur = []
            # Calcular rent histórica EUR para abiertas
            for _, row in asset_returns_df.iterrows():
                valor_compra_eur = row['Valor EUR'] / (1 + row['Rent.Compra %'] / 100) if row['Rent.Compra %'] != -100 else 0
                rent_hist_eur = row['Valor EUR'] - valor_compra_eur
                all_historica_eur.append(rent_hist_eur)
            # Añadir cerradas (usando closed_positions que tiene todas)
            all_historica_eur += [c.get('Rent.Histórica EUR', 0) for c in closed_positions]

            # Recalcular totales PERIODO
            total_rent_eur_abiertas = asset_returns_df['Rent.Periodo EUR'].sum()
            total_rent_eur_cerradas = total_closed_pnl_eur
            combined_rent_eur = total_rent_eur_abiertas + total_rent_eur_cerradas

            # Estadísticas PERIODO
            periodo_count = len(all_periodo_pct)
            periodo_positive = sum(1 for r in all_periodo_pct if r > 0)
            periodo_negative = sum(1 for r in all_periodo_pct if r < 0)
            periodo_positive_pct = (periodo_positive / periodo_count * 100) if periodo_count > 0 else 0
            periodo_max = max(all_periodo_pct) if all_periodo_pct else 0
            periodo_min = min(all_periodo_pct) if all_periodo_pct else 0
            periodo_gains = sum(r for r in all_periodo_eur if r > 0)
            periodo_losses = abs(sum(r for r in all_periodo_eur if r < 0))
            periodo_profit_factor = periodo_gains / periodo_losses if periodo_losses > 0 else float('inf')

            # Estadísticas HISTÓRICAS
            historica_count = len(all_historica_pct)
            historica_positive = sum(1 for r in all_historica_pct if r > 0)
            historica_negative = sum(1 for r in all_historica_pct if r < 0)
            historica_positive_pct = (historica_positive / historica_count * 100) if historica_count > 0 else 0
            historica_max = max(all_historica_pct) if all_historica_pct else 0
            historica_min = min(all_historica_pct) if all_historica_pct else 0
            historica_total_eur = sum(all_historica_eur)
            historica_gains = sum(r for r in all_historica_eur if r > 0)
            historica_losses = abs(sum(r for r in all_historica_eur if r < 0))
            historica_profit_factor = historica_gains / historica_losses if historica_losses > 0 else float('inf')

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
            historica_rent_pct = (historica_total_eur / combined_inicial_eur * 100) if combined_inicial_eur > 0 else 0

            # =====================================================
            # RESUMEN RENTABILIDAD HISTÓRICA (desde compra)
            # Beneficio Neto + Profit Factor
            # =====================================================
            loading_placeholder.empty()  # Limpiar mensaje de carga

            # Calcular P&L en EUR para posiciones abiertas
            open_pnl_eur_list = []
            for _, row in asset_returns_df.iterrows():
                # Calcular valor inicial y P&L
                valor_actual = row['Valor EUR']
                rent_compra = row['Rent.Compra %']
                if rent_compra != -100:
                    valor_inicial = valor_actual / (1 + rent_compra / 100)
                    pnl = valor_actual - valor_inicial
                else:
                    pnl = 0
                open_pnl_eur_list.append(pnl)

            open_pnl_total = sum(open_pnl_eur_list)
            open_gains = sum(p for p in open_pnl_eur_list if p > 0)
            open_losses = abs(sum(p for p in open_pnl_eur_list if p < 0))
            open_pf = open_gains / open_losses if open_losses > 0 else float('inf')
            open_positivas = sum(1 for p in open_pnl_eur_list if p > 0)
            open_negativas = sum(1 for p in open_pnl_eur_list if p < 0)

            # Calcular P&L en EUR para posiciones cerradas
            closed_pnl_eur_list = [c['Rent.Histórica EUR'] for c in closed_positions]
            closed_pnl_total = sum(closed_pnl_eur_list)
            closed_gains = sum(p for p in closed_pnl_eur_list if p > 0)
            closed_losses = abs(sum(p for p in closed_pnl_eur_list if p < 0))
            closed_pf = closed_gains / closed_losses if closed_losses > 0 else float('inf')
            closed_positivas = sum(1 for p in closed_pnl_eur_list if p > 0)
            closed_negativas = sum(1 for p in closed_pnl_eur_list if p < 0)

            # Totales consolidados
            total_pnl = open_pnl_total + closed_pnl_total
            total_gains = open_gains + closed_gains
            total_losses = open_losses + closed_losses
            total_pf = total_gains / total_losses if total_losses > 0 else float('inf')
            total_count = len(asset_returns_df) + len(closed_positions)
            total_positivas = open_positivas + closed_positivas
            total_negativas = open_negativas + closed_negativas

            st.subheader("📊 Resumen Rentabilidad Histórica")

            # Fila 1: Totales consolidados
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Beneficio Neto Total", f"{total_pnl:+,.0f} €".replace(",", "."))
            col2.metric("Profit Factor", f"{total_pf:.2f}" if total_pf != float('inf') else "∞")
            col3.metric("Positivas", f"{total_positivas} ({total_positivas/total_count*100:.1f}%)" if total_count > 0 else "0")
            col4.metric("Negativas", f"{total_negativas} ({total_negativas/total_count*100:.1f}%)" if total_count > 0 else "0")

            # Fila 2: Abiertas vs Cerradas
            col5, col6, col7, col8 = st.columns(4)
            col5.metric(f"Abiertas ({len(asset_returns_df)})", f"{open_pnl_total:+,.0f} €".replace(",", "."))
            col6.metric("PF Abiertas", f"{open_pf:.2f}" if open_pf != float('inf') else "∞")
            col7.metric(f"Cerradas ({len(closed_positions)})", f"{closed_pnl_total:+,.0f} €".replace(",", "."))
            col8.metric("PF Cerradas", f"{closed_pf:.2f}" if closed_pf != float('inf') else "∞")

            # TOP 5 Mejores y Peores por rentabilidad %
            # Combinar abiertas y cerradas con su rentabilidad histórica
            all_positions_pct = []
            for _, row in asset_returns_df.iterrows():
                all_positions_pct.append({
                    'Ticker': row['Ticker'],
                    'Rent %': row['Rent.Compra %'],
                    'Estado': 'abierta'
                })
            for c in closed_positions:
                all_positions_pct.append({
                    'Ticker': c['Ticker'],
                    'Rent %': c['Rent.Histórica %'],
                    'Estado': 'cerrada'
                })

            all_pct_df = pd.DataFrame(all_positions_pct)
            top5_best = all_pct_df.nlargest(5, 'Rent %')
            top5_worst = all_pct_df.nsmallest(5, 'Rent %')

            col_best, col_worst = st.columns(2)
            with col_best:
                st.markdown("**🏆 TOP 5 Mejores**")
                for _, row in top5_best.iterrows():
                    st.markdown(f"• **{row['Ticker']}**: {row['Rent %']:+.2f}% [{row['Estado']}]")

            with col_worst:
                st.markdown("**📉 TOP 5 Peores**")
                for _, row in top5_worst.iterrows():
                    st.markdown(f"• **{row['Ticker']}**: {row['Rent %']:+.2f}% [{row['Estado']}]")

            st.markdown("---")

            # =====================================================
            # DETAILED TABLE WITH ALL STOCKS
            # =====================================================
            st.subheader("Posiciones Abiertas")

            # Calculate totals for the detailed table
            total_valor_eur = asset_returns_df['Valor EUR'].sum()
            total_rent_eur = asset_returns_df['Rent.Periodo EUR'].sum()
            total_inicial_eur = asset_returns_df['Valor_Inicial_EUR'].sum()
            total_rent_pct = ((total_valor_eur / total_inicial_eur) - 1) * 100 if total_inicial_eur > 0 else 0

            # Format for display - mantener valores numéricos para ordenamiento correcto
            display_df = asset_returns_df.copy()
            display_df = display_df.drop(columns=['Valor_Inicial_EUR'])

            # Función para convertir fecha a formato español (para uso posterior)
            def to_spanish_date(date_val):
                if not date_val or date_val == '-':
                    return None
                try:
                    if isinstance(date_val, (date, datetime)):
                        return date_val
                    from datetime import datetime as dt
                    d = dt.strptime(str(date_val)[:10], '%Y-%m-%d')
                    return d.date()
                except:
                    return None

            # Reorder columns: F.Compra, precios, rentabilidades en EUR
            display_df = display_df[['F.Compra', 'Ticker', 'Tipo', 'Cuenta', 'Títulos', 'P.Compra', 'Últ.Precio', 'Valor EUR', 'Rent.Compra %', 'Rent.Periodo %', 'Rent.Periodo EUR']]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    'F.Compra': st.column_config.DateColumn('F.Compra', width='small', format='DD/MM/YYYY'),
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Tipo': st.column_config.TextColumn('Tipo', width='small'),
                    'Cuenta': st.column_config.TextColumn('Cuenta', width='small'),
                    'Títulos': st.column_config.NumberColumn('Títulos', width='small', format='%d'),
                    'P.Compra': st.column_config.TextColumn('P.Compra', width='small'),
                    'Últ.Precio': st.column_config.TextColumn('Últ.Precio', width='small'),
                    'Valor EUR': st.column_config.NumberColumn('Valor EUR', width='medium', format='%.0f €'),
                    'Rent.Compra %': st.column_config.NumberColumn('Rent.Compra %', width='small', format='%.2f %%'),
                    'Rent.Periodo %': st.column_config.NumberColumn('Rent.Periodo %', width='small', format='%.2f %%'),
                    'Rent.Periodo EUR': st.column_config.NumberColumn('Rent.Periodo EUR', width='small', format='%.0f €'),
                }
            )

            # Total row with black background
            st.markdown(
                f"""
                <div style="background-color: #1a1a1a; color: white; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between; font-weight: bold;">
                    <span>TOTAL ABIERTAS</span>
                    <span>Valor EUR: {total_valor_eur:,.0f} €</span>
                    <span>Rent.Periodo EUR: {total_rent_eur:+,.0f} €</span>
                </div>
                """.replace(",", "."),
                unsafe_allow_html=True
            )

            # =====================================================
            # POSICIONES SIN FECHA DE COMPRA (datos faltantes)
            # =====================================================
            if not missing_data_df.empty:
                st.markdown("---")
                st.subheader(f"⚠️ Posiciones Sin Datos de Compra ({len(missing_data_df)})")
                st.warning("Estas posiciones no tienen fecha ni precio de compra registrados en la tabla 'compras'.")

                # Mostrar solo Ticker, Tipo, Cuenta, Títulos, Valor EUR
                missing_display = missing_data_df[['Ticker', 'Tipo', 'Cuenta', 'Títulos', 'Valor EUR']].copy()
                missing_display['Valor EUR'] = missing_display['Valor EUR'].apply(lambda x: f"{x:,.0f} €".replace(",", "."))

                st.dataframe(
                    missing_display,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, len(missing_data_df) * 35 + 50)
                )

            # =====================================================
            # POSICIONES CERRADAS (con rentabilidad periodo e histórica)
            # =====================================================
            st.markdown("---")
            st.subheader("Posiciones Cerradas")

            if closed_periodo:
                periodo_df = pd.DataFrame(closed_periodo)

                # Asegurar que las columnas numéricas sean tipo numérico
                periodo_df['Títulos'] = pd.to_numeric(periodo_df['Títulos'], errors='coerce').fillna(0)
                periodo_df['Rent.Periodo %'] = pd.to_numeric(periodo_df['Rent.Periodo %'], errors='coerce').fillna(0)
                periodo_df['Rent.Histórica %'] = pd.to_numeric(periodo_df['Rent.Histórica %'], errors='coerce').fillna(0)
                periodo_df['Rent.Periodo EUR'] = pd.to_numeric(periodo_df['Rent.Periodo EUR'], errors='coerce').fillna(0)
                periodo_df['Rent.Histórica EUR'] = pd.to_numeric(periodo_df['Rent.Histórica EUR'], errors='coerce').fillna(0)

                # Convertir fecha a objeto date para ordenamiento correcto
                periodo_df['Fecha'] = pd.to_datetime(periodo_df['Fecha']).dt.date

                total_periodo_eur = periodo_df['Rent.Periodo EUR'].sum()
                total_historica_eur_display = periodo_df['Rent.Histórica EUR'].sum()

                # Format for display
                display_periodo = periodo_df.copy()
                display_periodo['Rent.Periodo %'] = display_periodo['Rent.Periodo %'].apply(lambda x: f"{x:+.2f}%")
                display_periodo['Rent.Histórica %'] = display_periodo['Rent.Histórica %'].apply(lambda x: f"{x:+.2f}%")
                display_periodo['Rent.Periodo EUR'] = display_periodo['Rent.Periodo EUR'].apply(lambda x: f"{x:+,.0f} €".replace(",", "."))
                display_periodo['Rent.Histórica EUR'] = display_periodo['Rent.Histórica EUR'].apply(lambda x: f"{x:+,.0f} €".replace(",", "."))

                st.dataframe(
                    display_periodo,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                    column_config={
                        'Fecha': st.column_config.DateColumn('Fecha', width='small', format='DD/MM/YYYY'),
                        'Títulos': st.column_config.NumberColumn('Títulos', width='small', format='%d'),
                        'Rent.Periodo %': st.column_config.TextColumn('Rent.Periodo %', width='small'),
                        'Rent.Histórica %': st.column_config.TextColumn('Rent.Histórica %', width='small'),
                        'Rent.Periodo EUR': st.column_config.TextColumn('Rent.Periodo EUR', width='small'),
                        'Rent.Histórica EUR': st.column_config.TextColumn('Rent.Histórica EUR', width='small'),
                    }
                )

                # Total row for closed positions
                st.markdown(
                    f"""
                    <div style="background-color: #1a1a1a; color: white; padding: 10px; border-radius: 5px; display: flex; justify-content: space-between; font-weight: bold;">
                        <span>TOTAL CERRADAS ({len(periodo_df)})</span>
                        <span>Rent.Periodo: {total_periodo_eur:+,.0f} €</span>
                        <span>Rent.Histórica: {total_historica_eur_display:+,.0f} €</span>
                    </div>
                    """.replace(",", "."),
                    unsafe_allow_html=True
                )
            else:
                st.info("No hay posiciones cerradas")

            # =====================================================
            # RENTABILIDAD POR MARKET CAP
            # =====================================================
            st.markdown("---")
            st.subheader("📊 Rentabilidad por Market Cap")

            # Obtener market cap de fundamentals para cada símbolo
            with db.get_session() as session:
                # Consultar market cap desde fundamentals
                mcap_result = session.execute(text("""
                    SELECT s.code, f.market_cap
                    FROM fundamentals f
                    JOIN symbols s ON f.symbol_id = s.id
                    WHERE f.market_cap IS NOT NULL AND f.market_cap > 0
                """))
                mcap_data = {row[0]: row[1] for row in mcap_result.fetchall()}

            # Categorizar por market cap (en millones)
            def get_mcap_category(mcap_value):
                if mcap_value is None:
                    return 'Sin datos'
                mcap_millions = mcap_value / 1_000_000  # Convertir a millones
                if mcap_millions < 5000:
                    return '<5.000M'
                elif mcap_millions < 10000:
                    return '5.000-10.000M'
                elif mcap_millions < 50000:
                    return '10.000-50.000M'
                else:
                    return '>50.000M'

            # Calcular rentabilidad por categoría (combinando abiertas y cerradas)
            mcap_stats = {
                '<5.000M': {'count': 0, 'rent_sum': 0, 'rent_eur_sum': 0, 'tickers': [], 'rents': []},
                '5.000-10.000M': {'count': 0, 'rent_sum': 0, 'rent_eur_sum': 0, 'tickers': [], 'rents': []},
                '10.000-50.000M': {'count': 0, 'rent_sum': 0, 'rent_eur_sum': 0, 'tickers': [], 'rents': []},
                '>50.000M': {'count': 0, 'rent_sum': 0, 'rent_eur_sum': 0, 'tickers': [], 'rents': []},
                'Sin datos': {'count': 0, 'rent_sum': 0, 'rent_eur_sum': 0, 'tickers': [], 'rents': []},
            }

            # Procesar posiciones abiertas
            for _, row in asset_returns_df.iterrows():
                ticker = row['Ticker'].split('.')[0] if '.' in row['Ticker'] else row['Ticker']
                mcap = mcap_data.get(ticker) or mcap_data.get(row['Ticker'])
                category = get_mcap_category(mcap)
                mcap_stats[category]['count'] += 1
                mcap_stats[category]['rent_sum'] += row['Rent.Periodo %']
                mcap_stats[category]['rent_eur_sum'] += row['Rent.Periodo EUR']
                mcap_stats[category]['tickers'].append(ticker)
                mcap_stats[category]['rents'].append(row['Rent.Periodo %'])

            # Procesar posiciones cerradas
            for pos in closed_positions:
                ticker = pos['Ticker'].split('.')[0] if '.' in pos['Ticker'] else pos['Ticker']
                mcap = mcap_data.get(ticker) or mcap_data.get(pos['Ticker'])
                category = get_mcap_category(mcap)
                mcap_stats[category]['count'] += 1
                mcap_stats[category]['rent_sum'] += pos['Rent. %']
                mcap_stats[category]['rent_eur_sum'] += pos['Rent. EUR']
                mcap_stats[category]['tickers'].append(ticker)
                mcap_stats[category]['rents'].append(pos['Rent. %'])

            # Crear tabla de resumen
            mcap_table_data = []
            order = ['<5.000M', '5.000-10.000M', '10.000-50.000M', '>50.000M']
            for cat in order:
                stats = mcap_stats[cat]
                if stats['count'] > 0:
                    avg_rent = stats['rent_sum'] / stats['count']
                    max_rent = max(stats['rents'])
                    min_rent = min(stats['rents'])
                    mcap_table_data.append({
                        'Market Cap': cat,
                        'Operaciones': stats['count'],
                        'Rent. Media %': avg_rent,
                        'Máx %': max_rent,
                        'Mín %': min_rent,
                        'Rent. Total EUR': stats['rent_eur_sum'],
                        'Tickers': ', '.join(set(stats['tickers'][:5])) + ('...' if len(set(stats['tickers'])) > 5 else '')
                    })

            if mcap_table_data:
                mcap_df = pd.DataFrame(mcap_table_data)
                st.dataframe(
                    mcap_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Market Cap': st.column_config.TextColumn('Market Cap', width='medium'),
                        'Operaciones': st.column_config.NumberColumn('Operaciones', width='small', format='%d'),
                        'Rent. Media %': st.column_config.NumberColumn('Rent. Media %', width='small', format='%.2f %%'),
                        'Máx %': st.column_config.NumberColumn('Máx %', width='small', format='%.2f %%'),
                        'Mín %': st.column_config.NumberColumn('Mín %', width='small', format='%.2f %%'),
                        'Rent. Total EUR': st.column_config.NumberColumn('Rent. Total EUR', width='medium', format='%.0f €'),
                        'Tickers': st.column_config.TextColumn('Tickers', width='large'),
                    }
                )

                # Mostrar si hay posiciones sin datos de market cap
                if mcap_stats['Sin datos']['count'] > 0:
                    sin_datos_tickers = ', '.join(set(mcap_stats['Sin datos']['tickers']))
                    st.caption(f"⚠️ {mcap_stats['Sin datos']['count']} posiciones sin datos de market cap: {sin_datos_tickers}")
            else:
                st.info("No hay datos de market cap disponibles")
        else:
            loading_placeholder.empty()
            st.warning("No se encontraron datos de acciones para mostrar")


elif page == "Futuros":
    st.title("FUTUROS")

    # Función para generar PDF
    def generar_pdf_futuros(data):
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER, spaceAfter=20)
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
        desc_style = ParagraphStyle('Desc', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=12)

        elements = []
        GREEN = colors.HexColor('#006600')
        RED = colors.HexColor('#cc0000')

        def table_style():
            return TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f4f8')]),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ])

        def fmt(n):
            return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        def fmt_int(n):
            return f"{n:,}".replace(",", ".")

        # Titulo
        elements.append(Paragraph("ANALISIS DE FUTUROS IB", title_style))
        elements.append(Paragraph(f"Periodo: 01/01/2026 - 13/02/2026", ParagraphStyle('P', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, spaceAfter=5)))
        elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} | EUR/USD: {data['eur_usd']:.4f}", ParagraphStyle('P2', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, spaceAfter=20, textColor=colors.grey)))
        elements.append(Spacer(1, 0.5*cm))

        # 1. Resultado Global
        elements.append(Paragraph("1. RESULTADO GLOBAL", subtitle_style))
        sign = '+' if data['total_pnl'] >= 0 else ''
        comisiones = -539.60
        pnl_neto = data['total_pnl'] + comisiones
        avg_pnl = data['total_pnl'] / data['total_trades'] if data['total_trades'] > 0 else 0
        tbl = [
            ['Metrica', 'Valor'],
            ['P&L Total USD', f"{sign}${fmt(data['total_pnl'])}"],
            ['P&L Total EUR', f"{sign}{fmt(data['total_pnl']/data['eur_usd'])} EUR"],
            ['Comisiones IB', f"${fmt(comisiones)}"],
            ['P&L Neto USD', f"{'+' if pnl_neto >= 0 else ''}${fmt(pnl_neto)}"],
            ['Trades', f"{data['total_trades']}"],
            ['Contratos', f"{data['total_contracts']}"],
            ['Win Rate', f"{data['wr']:.0f}%"],
            ['P&L Promedio', f"{'+' if avg_pnl >= 0 else ''}${fmt(avg_pnl)} USD/trade"],
        ]
        t = Table(tbl, colWidths=[8*cm, 6*cm])
        s = table_style()
        s.add('TEXTCOLOR', (1, 1), (1, 2), GREEN)
        s.add('TEXTCOLOR', (1, 3), (1, 3), RED)
        s.add('TEXTCOLOR', (1, 4), (1, 4), GREEN)
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.8*cm))

        # 2. Por Tipo
        elements.append(Paragraph("2. POR TIPO DE ACTIVO", subtitle_style))
        tbl = [['Tipo', 'Trades', 'Contr.', 'Importe', 'P&L USD', '%Total']]
        for tipo in ['Oro', 'Índices', 'Ganado', 'Petróleo']:
            if tipo in data['by_tipo']:
                d = data['by_tipo'][tipo]
                pct = d['pnl'] / data['total_pnl'] * 100 if data['total_pnl'] != 0 else 0
                sign = '+' if d['pnl'] >= 0 else ''
                tbl.append([tipo, str(d['trades']), str(d['contracts']), f"${fmt_int(int(d['importe']))}", f"{sign}{fmt(d['pnl'])}", f"{pct:+.1f}%"])
        t = Table(tbl, colWidths=[2.5*cm, 2*cm, 2*cm, 3*cm, 3*cm, 2.5*cm])
        s = table_style()
        for i, row in enumerate(tbl[1:], start=1):
            if row[4].startswith('+'):
                s.add('TEXTCOLOR', (4, i), (5, i), GREEN)
            else:
                s.add('TEXTCOLOR', (4, i), (5, i), RED)
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.8*cm))

        # 3. Por Mes
        elements.append(Paragraph("3. POR MES", subtitle_style))
        tbl = [['Mes', 'Trades', 'W/L', 'Win%', 'P&L USD']]
        for mes in ['Enero', 'Febrero']:
            if mes in data['by_mes']:
                d = data['by_mes'][mes]
                wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
                sign = '+' if d['pnl'] >= 0 else ''
                tbl.append([mes, str(d['trades']), f"{d['wins']}/{d['losses']}", f"{wr:.0f}%", f"{sign}{fmt(d['pnl'])}"])
        t = Table(tbl, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
        s = table_style()
        for i, row in enumerate(tbl[1:], start=1):
            s.add('TEXTCOLOR', (4, i), (4, i), GREEN if row[4].startswith('+') else RED)
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.8*cm))

        # 4. Por Día
        elements.append(Paragraph("4. POR DIA DE LA SEMANA", subtitle_style))
        tbl = [['Dia', 'Trades', 'W/L', 'Win%', 'P&L USD']]
        for dia in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']:
            if dia in data['by_dia']:
                d = data['by_dia'][dia]
                wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
                sign = '+' if d['pnl'] >= 0 else ''
                tbl.append([dia, str(d['trades']), f"{d['wins']}/{d['losses']}", f"{wr:.0f}%", f"{sign}{fmt(d['pnl'])}"])
        t = Table(tbl, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
        s = table_style()
        for i, row in enumerate(tbl[1:], start=1):
            s.add('TEXTCOLOR', (4, i), (4, i), GREEN if row[4].startswith('+') else RED)
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.8*cm))

        # 5. Por Franja
        elements.append(Paragraph("5. POR FRANJA HORARIA", subtitle_style))
        tbl = [
            ['Franja Horaria', 'Trades', 'W/L', 'Win%', 'P&L USD'],
            ['00:01-08:00 (Asia)', '14', '9/5', '64%', '+8.710,84'],
            ['08:01-15:00 (EU)', '10', '1/9', '10%', '-4.862,18'],
            ['15:01-23:59 (US)', '12', '7/5', '58%', '+16.441,74'],
        ]
        t = Table(tbl, colWidths=[4.5*cm, 2*cm, 2*cm, 2*cm, 4*cm])
        s = table_style()
        s.add('TEXTCOLOR', (4, 1), (4, 1), GREEN)
        s.add('TEXTCOLOR', (4, 2), (4, 2), RED)
        s.add('TEXTCOLOR', (4, 3), (4, 3), GREEN)
        t.setStyle(s)
        elements.append(t)

        elements.append(PageBreak())

        # 6. Por Posición
        elements.append(Paragraph("6. POR TIPO DE POSICION", subtitle_style))
        tbl = [
            ['Posicion', 'Trades', 'Contratos', 'Win%', 'P&L USD'],
            ['LONG', '21', '31', '62%', '+12.439,26'],
            ['SHORT', '15', '71', '27%', '+7.851,14'],
        ]
        t = Table(tbl, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
        s = table_style()
        s.add('TEXTCOLOR', (4, 1), (4, 2), GREEN)
        s.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda'))
        s.add('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#f8d7da'))
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.8*cm))

        # 7. Por Duración
        elements.append(Paragraph("7. POR DURACION DEL TRADE", subtitle_style))
        tbl = [
            ['Duracion', 'Trades', 'Win%', 'P&L USD', 'Nota'],
            ['< 2 horas', '23', '57%', '-10.029,56', 'Stops ejecutados'],
            ['2-6 horas', '59', '54%', '+21.342,54', 'Rango optimo'],
            ['6+ horas', '20', '30%', '+8.977,42', 'Swing trading'],
        ]
        t = Table(tbl, colWidths=[2.5*cm, 2*cm, 2*cm, 3*cm, 4*cm])
        s = table_style()
        s.add('TEXTCOLOR', (3, 1), (3, 1), RED)
        s.add('TEXTCOLOR', (3, 2), (3, 3), GREEN)
        s.add('BACKGROUND', (4, 2), (4, 2), colors.HexColor('#d4edda'))
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.8*cm))

        # 8. Lista de Trades
        elements.append(Paragraph("8. RELACION DE TRADES POR FECHA", subtitle_style))
        trades_list = [
            ['Fecha', 'Symbol', 'Tipo', 'Contr.', 'P&L USD'],
            ['20/01', 'GCH6', 'LONG', '1', '+3.315,06'],
            ['20/01', 'GCH6', 'LONG', '1', '+3.335,06'],
            ['20/01', 'GCH6', 'LONG', '1', '+2.515,06'],
            ['20/01', 'GCH6', 'LONG', '1', '+2.375,06'],
            ['20/01', 'GCH6', 'LONG', '1', '+2.365,06'],
            ['21/01', 'GCH6', 'LONG', '1', '+4.445,06'],
            ['21/01', 'GCH6', 'LONG', '1', '+4.445,06'],
            ['23/01', 'GCH6', 'LONG', '1', '-684,94'],
            ['23/01', 'GCH6', 'LONG', '1', '-694,94'],
            ['26/01', 'GCH6', 'LONG', '1', '-1.254,94'],
            ['26/01', 'GCH6', 'LONG', '1', '-1.314,94'],
            ['26/01', 'GCH6', 'LONG', '1', '+915,06'],
            ['26/01', 'GCJ6', 'LONG', '1', '-4.054,94'],
            ['27/01', 'GCJ6', 'LONG', '1', '+325,06'],
            ['27/01', 'GCJ6', 'LONG', '1', '+205,06'],
            ['27/01', 'GCJ6', 'LONG', '1', '+5.515,06'],
            ['28/01', 'CLH6', 'SHORT', '13', '+248,38'],
            ['30/01', 'NQH6', 'SHORT', '1', '-724,50'],
            ['02/02', 'ESH6', 'SHORT', '1', '-392,00'],
            ['02/02', 'NQH6', 'SHORT', '1', '-599,50'],
            ['03/02', 'ESH6', 'SHORT', '1', '+3.720,50'],
            ['06/02', 'NQH6', 'SHORT', '1', '-344,50'],
            ['06/02', 'NQH6', 'SHORT', '1', '-689,50'],
            ['09/02', 'GCJ6', 'LONG', '1', '+345,06'],
            ['09/02', 'GCJ6', 'LONG', '1', '+385,06'],
            ['10/02', 'ESH6', 'SHORT', '1', '-242,00'],
            ['10/02', 'HEJ6', 'SHORT', '23', '+4.183,38'],
            ['11/02', 'CLH6', 'LONG', '1', '-394,74'],
            ['11/02', 'CLH6', 'LONG', '11', '-4.592,14'],
            ['11/02', 'GCJ6', 'LONG', '1', '-5.054,94'],
            ['11/02', 'NQH6', 'SHORT', '1', '-509,50'],
            ['11/02', 'NQH6', 'SHORT', '1', '-739,50'],
            ['11/02', 'NQH6', 'SHORT', '1', '-644,50'],
            ['12/02', 'ESH6', 'SHORT', '1', '+6.370,50'],
            ['12/02', 'HEJ6', 'SHORT', '23', '-1.206,62'],
            ['12/02', 'NQH6', 'SHORT', '1', '-579,50'],
        ]
        t = Table(trades_list, colWidths=[2*cm, 2.5*cm, 2*cm, 2*cm, 3.5*cm])
        s = table_style()
        for i, row in enumerate(trades_list[1:], start=1):
            if row[4].startswith('+'):
                s.add('TEXTCOLOR', (4, i), (4, i), GREEN)
            else:
                s.add('TEXTCOLOR', (4, i), (4, i), RED)
        t.setStyle(s)
        elements.append(t)

        elements.append(PageBreak())

        # 9. Top 5
        elements.append(Paragraph("9. TOP 5 MEJORES Y PEORES TRADES", subtitle_style))
        elements.append(Paragraph("TOP 5 MEJORES", ParagraphStyle('Sub', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=8)))
        best = [
            ['#', 'Symbol', 'Fecha', 'Duracion', 'P&L USD'],
            ['1', 'ESH6', '10-12/02', '2.3 dias', '+6.370,50'],
            ['2', 'GCJ6', '27/01', '2.9 hrs', '+5.515,06'],
            ['3', 'GCH6', '20-21/01', '3.2 hrs', '+4.445,06'],
            ['4', 'GCH6', '20-21/01', '3.2 hrs', '+4.445,06'],
            ['5', 'HEJ6', '10/02', '3.0 hrs', '+4.183,38'],
        ]
        t = Table(best, colWidths=[1*cm, 2.5*cm, 3*cm, 2.5*cm, 4*cm])
        s = table_style()
        for i in range(1, 6):
            s.add('TEXTCOLOR', (4, i), (4, i), GREEN)
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.5*cm))

        elements.append(Paragraph("TOP 5 PEORES", ParagraphStyle('Sub2', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=8)))
        worst = [
            ['#', 'Symbol', 'Fecha', 'Duracion', 'P&L USD'],
            ['1', 'GCJ6', '11/02', '1.2 hrs', '-5.054,94'],
            ['2', 'CLH6', '11/02', '8.0 hrs', '-4.592,14'],
            ['3', 'GCJ6', '26/01', '9.1 hrs', '-4.054,94'],
            ['4', 'GCH6', '25-26/01', '3.3 hrs', '-1.314,94'],
            ['5', 'GCH6', '25-26/01', '3.3 hrs', '-1.254,94'],
        ]
        t = Table(worst, colWidths=[1*cm, 2.5*cm, 3*cm, 2.5*cm, 4*cm])
        s = table_style()
        for i in range(1, 6):
            s.add('TEXTCOLOR', (4, i), (4, i), RED)
        t.setStyle(s)
        elements.append(t)

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    # ==========================================================================
    # ANÁLISIS FUTUROS IB (01/01 - 13/02/2026)
    # ==========================================================================
    st.subheader("Análisis Futuros IB (01/01 - 13/02/2026)")

    # Obtener datos de ventas de futuros
    with db.get_session() as session:
        from sqlalchemy import text

        # Obtener todos los trades de futuros desde ventas
        result = session.execute(text("""
            SELECT fecha, symbol, shares, pnl, importe_total
            FROM ventas
            WHERE symbol SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
            ORDER BY fecha
        """))
        futures_ventas = result.fetchall()

        # Calcular estadísticas
        CATEGORIES = {'GC': 'Oro', 'CL': 'Petróleo', 'ES': 'Índices', 'NQ': 'Índices', 'HE': 'Ganado'}
        DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        MESES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

        # Inicializar estructuras
        by_tipo = {}
        by_mes = {}
        by_dia = {}
        by_franja = {'00:01-08:00': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0},
                     '08:01-15:00': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0},
                     '15:01-23:59': {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0}}

        total_pnl = 0
        total_trades = 0
        total_wins = 0
        total_losses = 0

        for fecha, symbol, shares, pnl, importe in futures_ventas:
            if pnl is None:
                pnl = 0
            if importe is None:
                importe = 0

            # Por tipo
            prefix = ''.join([c for c in symbol if c.isalpha()])[:2]
            tipo = CATEGORIES.get(prefix, 'Otros')
            if tipo not in by_tipo:
                by_tipo[tipo] = {'trades': 0, 'contracts': 0, 'pnl': 0, 'wins': 0, 'losses': 0, 'importe': 0}
            by_tipo[tipo]['trades'] += 1
            by_tipo[tipo]['contracts'] += int(shares)
            by_tipo[tipo]['pnl'] += pnl
            by_tipo[tipo]['importe'] += importe
            if pnl > 0:
                by_tipo[tipo]['wins'] += 1
            elif pnl < 0:
                by_tipo[tipo]['losses'] += 1

            # Por mes
            if hasattr(fecha, 'month'):
                mes = MESES[fecha.month - 1]
            else:
                mes = 'Desconocido'
            if mes not in by_mes:
                by_mes[mes] = {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
            by_mes[mes]['trades'] += 1
            by_mes[mes]['pnl'] += pnl
            if pnl > 0:
                by_mes[mes]['wins'] += 1
            elif pnl < 0:
                by_mes[mes]['losses'] += 1

            # Por día de la semana
            if hasattr(fecha, 'weekday'):
                dia = DIAS[fecha.weekday()]
            else:
                dia = 'Desconocido'
            if dia not in by_dia:
                by_dia[dia] = {'trades': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
            by_dia[dia]['trades'] += 1
            by_dia[dia]['pnl'] += pnl
            if pnl > 0:
                by_dia[dia]['wins'] += 1
            elif pnl < 0:
                by_dia[dia]['losses'] += 1

            # Totales
            total_pnl += pnl
            total_trades += 1
            if pnl > 0:
                total_wins += 1
            elif pnl < 0:
                total_losses += 1

        # Obtener EUR/USD
        fx = session.execute(text("""
            SELECT ph.close FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            WHERE s.code = 'EURUSD=X'
            ORDER BY ph.date DESC LIMIT 1
        """)).fetchone()
        eur_usd_ib = fx[0] if fx else 1.04

    # Calcular totales para el PDF
    total_contracts = sum(d.get('contracts', 0) for d in by_tipo.values())
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    # Botón para descargar PDF
    pdf_data = {
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'total_contracts': total_contracts,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'wr': wr,
        'eur_usd': eur_usd_ib,
        'by_tipo': by_tipo,
        'by_mes': by_mes,
        'by_dia': by_dia,
    }

    col_title, col_btn = st.columns([3, 1])
    with col_btn:
        pdf_bytes = generar_pdf_futuros(pdf_data)
        st.download_button(
            label="📄 Descargar PDF",
            data=pdf_bytes,
            file_name=f"Analisis_Futuros_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary"
        )

    # Tabla 1: Resultado Global
    st.markdown("**📊 Resultado Global**")

    comisiones = -539.60  # Comisiones del informe IB
    pnl_neto = total_pnl + comisiones
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    sign = '+' if total_pnl >= 0 else ''
    sign_neto = '+' if pnl_neto >= 0 else ''
    sign_avg = '+' if avg_pnl >= 0 else ''

    resultado_data = [
        {'Métrica': 'P&L Total', 'Valor': f"{sign}${total_pnl:,.2f} USD".replace(",", ".")},
        {'Métrica': 'P&L Total', 'Valor': f"{sign}{total_pnl/eur_usd_ib:,.2f} EUR".replace(",", ".")},
        {'Métrica': 'Comisiones', 'Valor': f"${comisiones:,.2f} USD".replace(",", ".")},
        {'Métrica': 'P&L Neto', 'Valor': f"{sign_neto}${pnl_neto:,.2f} USD".replace(",", ".")},
        {'Métrica': 'Trades', 'Valor': f"{total_trades}"},
        {'Métrica': 'Contratos', 'Valor': f"{total_contracts}"},
        {'Métrica': 'Win Rate', 'Valor': f"{wr:.0f}%"},
        {'Métrica': 'P&L Promedio', 'Valor': f"{sign_avg}${avg_pnl:,.2f} USD/trade".replace(",", ".")},
    ]

    col_res, col_space = st.columns([1, 2])
    with col_res:
        st.dataframe(pd.DataFrame(resultado_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Tabla 2: Por Tipo de Activo
    col_tipo, col_mes = st.columns(2)

    with col_tipo:
        st.markdown("**📈 Por Tipo de Activo**")
        tipo_data = []
        for tipo in ['Oro', 'Índices', 'Ganado', 'Petróleo']:
            if tipo in by_tipo:
                d = by_tipo[tipo]
                wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
                pct = d['pnl'] / total_pnl * 100 if total_pnl != 0 else 0
                sign = '+' if d['pnl'] >= 0 else ''
                tipo_data.append({
                    'Tipo': tipo,
                    'Trades': d['trades'],
                    'Contr.': d['contracts'],
                    'Importe': f"${d.get('importe', 0):,.0f}".replace(",", "."),
                    'P&L USD': f"{sign}{d['pnl']:,.2f}".replace(",", "."),
                    '%Total': f"{pct:+.1f}%"
                })
        if tipo_data:
            st.dataframe(pd.DataFrame(tipo_data), use_container_width=True, hide_index=True)

    # Tabla 3: Por Mes
    with col_mes:
        st.markdown("**📅 Por Mes**")
        mes_data = []
        for mes in ['Enero', 'Febrero']:
            if mes in by_mes:
                d = by_mes[mes]
                wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
                sign = '+' if d['pnl'] >= 0 else ''
                mes_data.append({
                    'Mes': mes,
                    'Trades': d['trades'],
                    'W/L': f"{d['wins']}/{d['losses']}",
                    'Win%': f"{wr:.0f}%",
                    'P&L USD': f"{sign}{d['pnl']:,.2f}".replace(",", ".")
                })
        if mes_data:
            st.dataframe(pd.DataFrame(mes_data), use_container_width=True, hide_index=True)

    # Tabla 4: Por Día de la Semana (ordenado)
    col_dia2, col_franja = st.columns(2)

    with col_dia2:
        st.markdown("**📆 Por Día de la Semana**")
        dia_data = []
        for dia in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']:
            if dia in by_dia:
                d = by_dia[dia]
                wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
                sign = '+' if d['pnl'] >= 0 else ''
                dia_data.append({
                    'Día': dia,
                    'Trades': d['trades'],
                    'W/L': f"{d['wins']}/{d['losses']}",
                    'Win%': f"{wr:.0f}%",
                    'P&L USD': f"{sign}{d['pnl']:,.2f}".replace(",", ".")
                })
        if dia_data:
            st.dataframe(pd.DataFrame(dia_data), use_container_width=True, hide_index=True)

    # Tabla 5: Por Franja Horaria (datos estáticos del análisis)
    with col_franja:
        st.markdown("**🕐 Por Franja Horaria**")
        franja_data = [
            {'Franja': '00:01-08:00 (Asia)', 'Trades': 14, 'W/L': '9/5', 'Win%': '64%', 'P&L USD': '+8.710,84'},
            {'Franja': '08:01-15:00 (EU)', 'Trades': 10, 'W/L': '1/9', 'Win%': '10%', 'P&L USD': '-4.862,18'},
            {'Franja': '15:01-23:59 (US)', 'Trades': 12, 'W/L': '7/5', 'Win%': '58%', 'P&L USD': '+16.441,74'},
        ]
        st.dataframe(pd.DataFrame(franja_data), use_container_width=True, hide_index=True)

    # Tabla 6 y 7: Por Tipo de Posición y Por Duración
    col_pos, col_dur = st.columns(2)

    with col_pos:
        st.markdown("**📊 Por Tipo de Posición**")
        pos_data = [
            {'Posición': 'LONG', 'Trades': 21, 'Contr.': 31, 'Win%': '62%', 'P&L USD': '+12.439,26'},
            {'Posición': 'SHORT', 'Trades': 15, 'Contr.': 71, 'Win%': '27%', 'P&L USD': '+7.851,14'},
        ]
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

    with col_dur:
        st.markdown("**⏱️ Por Duración**")
        dur_data = [
            {'Duración': '< 2 horas', 'Trades': 23, 'Win%': '57%', 'P&L USD': '-10.029,56', 'Nota': 'Stops'},
            {'Duración': '2-6 horas', 'Trades': 59, 'Win%': '54%', 'P&L USD': '+21.342,54', 'Nota': 'Óptimo'},
            {'Duración': '6+ horas', 'Trades': 20, 'Win%': '30%', 'P&L USD': '+8.977,42', 'Nota': 'Swing'},
        ]
        st.dataframe(pd.DataFrame(dur_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Tabla 8: Relación de Trades por Fecha
    st.markdown("**📋 Relación de Trades por Fecha**")

    # Datos de trades del informe IB (ordenados por fecha de entrada)
    trades_list = [
        {'Fecha': '20/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+3.315,06'},
        {'Fecha': '20/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+3.335,06'},
        {'Fecha': '20/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+2.515,06'},
        {'Fecha': '20/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+2.375,06'},
        {'Fecha': '20/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+2.365,06'},
        {'Fecha': '21/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+4.445,06'},
        {'Fecha': '21/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+4.445,06'},
        {'Fecha': '23/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '-684,94'},
        {'Fecha': '23/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '-694,94'},
        {'Fecha': '26/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '-1.254,94'},
        {'Fecha': '26/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '-1.314,94'},
        {'Fecha': '26/01', 'Symbol': 'GCH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+915,06'},
        {'Fecha': '26/01', 'Symbol': 'GCJ6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '-4.054,94'},
        {'Fecha': '27/01', 'Symbol': 'GCJ6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+325,06'},
        {'Fecha': '27/01', 'Symbol': 'GCJ6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+205,06'},
        {'Fecha': '27/01', 'Symbol': 'GCJ6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+5.515,06'},
        {'Fecha': '28/01', 'Symbol': 'CLH6', 'Tipo': 'SHORT', 'Contr.': 13, 'P&L USD': '+248,38'},
        {'Fecha': '30/01', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-724,50'},
        {'Fecha': '02/02', 'Symbol': 'ESH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-392,00'},
        {'Fecha': '02/02', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-599,50'},
        {'Fecha': '03/02', 'Symbol': 'ESH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '+3.720,50'},
        {'Fecha': '06/02', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-344,50'},
        {'Fecha': '06/02', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-689,50'},
        {'Fecha': '09/02', 'Symbol': 'GCJ6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+345,06'},
        {'Fecha': '09/02', 'Symbol': 'GCJ6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '+385,06'},
        {'Fecha': '10/02', 'Symbol': 'ESH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-242,00'},
        {'Fecha': '10/02', 'Symbol': 'HEJ6', 'Tipo': 'SHORT', 'Contr.': 23, 'P&L USD': '+4.183,38'},
        {'Fecha': '11/02', 'Symbol': 'CLH6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '-394,74'},
        {'Fecha': '11/02', 'Symbol': 'CLH6', 'Tipo': 'LONG', 'Contr.': 11, 'P&L USD': '-4.592,14'},
        {'Fecha': '11/02', 'Symbol': 'GCJ6', 'Tipo': 'LONG', 'Contr.': 1, 'P&L USD': '-5.054,94'},
        {'Fecha': '11/02', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-509,50'},
        {'Fecha': '11/02', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-739,50'},
        {'Fecha': '11/02', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-644,50'},
        {'Fecha': '12/02', 'Symbol': 'ESH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '+6.370,50'},
        {'Fecha': '12/02', 'Symbol': 'HEJ6', 'Tipo': 'SHORT', 'Contr.': 23, 'P&L USD': '-1.206,62'},
        {'Fecha': '12/02', 'Symbol': 'NQH6', 'Tipo': 'SHORT', 'Contr.': 1, 'P&L USD': '-579,50'},
    ]
    trades_df = pd.DataFrame(trades_list)

    # Colorear según P&L
    def color_pnl(val):
        if isinstance(val, str):
            if val.startswith('+'):
                return 'color: #00cc00; font-weight: bold'
            elif val.startswith('-'):
                return 'color: #ff4444; font-weight: bold'
        return ''

    def color_tipo(val):
        if val == 'LONG':
            return 'background-color: #1a472a; color: white'
        elif val == 'SHORT':
            return 'background-color: #4a1a1a; color: white'
        return ''

    styled_trades = trades_df.style.applymap(color_pnl, subset=['P&L USD']).applymap(color_tipo, subset=['Tipo'])
    st.dataframe(styled_trades, use_container_width=True, hide_index=True, height=400)

    st.markdown("---")

    # Tabla 9: Top Mejores y Peores
    col_best, col_worst = st.columns(2)

    with col_best:
        st.markdown("**🏆 Top 5 Mejores**")
        best_trades = [
            {'#': 1, 'Symbol': 'ESH6', 'Fecha': '10-12/02', 'Duración': '2.3 días', 'P&L USD': '+6.370,50'},
            {'#': 2, 'Symbol': 'GCJ6', 'Fecha': '27/01', 'Duración': '2.9 hrs', 'P&L USD': '+5.515,06'},
            {'#': 3, 'Symbol': 'GCH6', 'Fecha': '20-21/01', 'Duración': '3.2 hrs', 'P&L USD': '+4.445,06'},
            {'#': 4, 'Symbol': 'GCH6', 'Fecha': '20-21/01', 'Duración': '3.2 hrs', 'P&L USD': '+4.445,06'},
            {'#': 5, 'Symbol': 'HEJ6', 'Fecha': '10/02', 'Duración': '3.0 hrs', 'P&L USD': '+4.183,38'},
        ]
        st.dataframe(pd.DataFrame(best_trades), use_container_width=True, hide_index=True)

    with col_worst:
        st.markdown("**💔 Top 5 Peores**")
        worst_trades = [
            {'#': 1, 'Symbol': 'GCJ6', 'Fecha': '11/02', 'Duración': '1.2 hrs', 'P&L USD': '-5.054,94'},
            {'#': 2, 'Symbol': 'CLH6', 'Fecha': '11/02', 'Duración': '8.0 hrs', 'P&L USD': '-4.592,14'},
            {'#': 3, 'Symbol': 'GCJ6', 'Fecha': '26/01', 'Duración': '9.1 hrs', 'P&L USD': '-4.054,94'},
            {'#': 4, 'Symbol': 'GCH6', 'Fecha': '25-26/01', 'Duración': '3.3 hrs', 'P&L USD': '-1.314,94'},
            {'#': 5, 'Symbol': 'GCH6', 'Fecha': '25-26/01', 'Duración': '3.3 hrs', 'P&L USD': '-1.254,94'},
        ]
        st.dataframe(pd.DataFrame(worst_trades), use_container_width=True, hide_index=True)


elif page == "ETFs":
    # Función para generar PDF de ETFs
    def generar_pdf_etfs():
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER
        from io import BytesIO

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1*cm, leftMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER, spaceAfter=10, textColor=colors.HexColor('#1a365d'))
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=15, textColor=colors.HexColor('#1a365d'))

        elements = []
        GREEN = colors.HexColor('#006600')
        RED = colors.HexColor('#cc0000')
        HEADER_BG = colors.HexColor('#1a365d')
        ROW_ALT = colors.HexColor('#e8f4f8')

        def table_style():
            return TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ROW_ALT]),
            ])

        # Titulo
        elements.append(Paragraph("ESTADISTICOS ETFs - INTERACTIVE BROKERS", title_style))
        elements.append(Paragraph("Periodo: 01/01/2026 - 13/02/2026", ParagraphStyle('Sub', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, textColor=colors.grey)))
        elements.append(Spacer(1, 0.5*cm))

        # Operaciones Cerradas
        elements.append(Paragraph("OPERACIONES CERRADAS", subtitle_style))
        data = [
            ['ETF', 'Nombre', 'Tipo', 'Acciones', 'F. Entrada', 'F. Cierre', 'P. Entrada', 'P. Cierre', 'P&L USD'],
            ['TLT', 'iShares 20+ Year Treasury Bond', 'LONG', '8.042', '20-21/01', '30/01-02/02', '$86,87', '$87,08', '+$1.291,44'],
            ['AMLP', 'Alerian MLP ETF', 'SHORT', '8.036', '30/01', '02/02', '$49,97', '$49,62', '+$2.810,27'],
            ['EMB', 'iShares JPM USD EM Bond', 'SHORT', '8.277', '30/01', '02/02', '$96,64', '$96,16', '+$3.863,27'],
            ['EMLC', 'VanEck JPM EM Local Currency Bond', 'SHORT', '30.394', '30/01', '02/02', '$26,30', '$26,22', '+$2.333,50'],
            ['XLB', 'Materials Select Sector SPDR', 'SHORT', '4.288', '30/01', '02/02', '$49,28', '$49,67', '-$1.726,35'],
            ['TLT', 'iShares 20+ Year Treasury Bond', 'LONG', '17', '06/02', '12/02', '$87,40', '$88,72', '+$21,75'],
        ]
        t = Table(data, colWidths=[1.3*cm, 4.5*cm, 1.4*cm, 1.6*cm, 2.2*cm, 2.4*cm, 1.8*cm, 1.8*cm, 2*cm])
        s = table_style()
        s.add('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#d4edda'))
        s.add('BACKGROUND', (2, 2), (2, 5), colors.HexColor('#f8d7da'))
        s.add('BACKGROUND', (2, 6), (2, 6), colors.HexColor('#d4edda'))
        s.add('TEXTCOLOR', (8, 1), (8, 4), GREEN)
        s.add('TEXTCOLOR', (8, 5), (8, 5), RED)
        s.add('TEXTCOLOR', (8, 6), (8, 6), GREEN)
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.5*cm))

        # Resumen
        elements.append(Paragraph("RESUMEN OPERACIONES CERRADAS", subtitle_style))
        resumen = [
            ['Metrica', 'Valor'],
            ['P&L Bruto', '+$8.593,88'],
            ['Trades', '6'],
            ['Wins / Losses', '5 / 1'],
            ['Win Rate', '83%'],
        ]
        t = Table(resumen, colWidths=[6*cm, 4*cm])
        s = table_style()
        s.add('TEXTCOLOR', (1, 1), (1, 1), GREEN)
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.5*cm))

        # Posiciones Abiertas
        elements.append(Paragraph("POSICIONES ABIERTAS", subtitle_style))
        open_data = [
            ['ETF', 'Nombre', 'Tipo', 'Acciones', 'F. Entrada', 'P. Entrada', 'Valor Aprox.'],
            ['TLT', 'iShares 20+ Year Treasury Bond', 'LONG', '0,424', '06/02', '$87,40', '~$38'],
        ]
        t = Table(open_data, colWidths=[1.5*cm, 5.5*cm, 1.5*cm, 1.8*cm, 2*cm, 2*cm, 2*cm])
        s = table_style()
        s.add('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#d4edda'))
        t.setStyle(s)
        elements.append(t)
        elements.append(Spacer(1, 0.5*cm))

        # Por Tipo de Posicion
        elements.append(Paragraph("POR TIPO DE POSICION (CERRADAS)", subtitle_style))
        tipo_data = [
            ['Posicion', 'Trades', 'P&L USD', 'Win Rate'],
            ['LONG', '2', '+$1.313,19', '100%'],
            ['SHORT', '4', '+$7.280,69', '75%'],
        ]
        t = Table(tipo_data, colWidths=[3*cm, 2.5*cm, 3*cm, 2.5*cm])
        s = table_style()
        s.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda'))
        s.add('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#f8d7da'))
        s.add('TEXTCOLOR', (2, 1), (2, 2), GREEN)
        t.setStyle(s)
        elements.append(t)

        # Footer
        elements.append(Spacer(1, 1*cm))
        elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    # Titulo con botón PDF
    col_title, col_btn = st.columns([3, 1])
    with col_title:
        st.title("ETFs")
    with col_btn:
        st.write("")
        pdf_bytes = generar_pdf_etfs()
        st.download_button(
            label="📄 Descargar PDF",
            data=pdf_bytes,
            file_name=f"ETFs_Estadisticos_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary"
        )

    # ==========================================================================
    # POSICIONES ETFs ABIERTAS
    # ==========================================================================
    st.subheader("Posiciones Abiertas")

    # Obtener fecha más reciente de IB en holding_diario
    with db.get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("""
            SELECT MAX(fecha) FROM holding_diario WHERE account_code = 'IB'
        """))
        ib_date_val = result.fetchone()[0]
        ib_date = parse_db_date(ib_date_val, date.today())

        # Obtener precios de compra desde ib_trades (promedio ponderado)
        precios_compra = {}
        result = session.execute(text("""
            SELECT symbol,
                   SUM(quantity * price) / SUM(quantity) as avg_price
            FROM ib_trades
            WHERE trade_type = 'BUY'
            GROUP BY symbol
        """))
        for row in result:
            precios_compra[row[0]] = row[1]

        # Para posiciones sin trades en ib_trades, usar precio de primera fecha en holding_diario
        result = session.execute(text("""
            SELECT DISTINCT ON (symbol) symbol, precio_entrada, fecha as first_date
            FROM holding_diario
            WHERE account_code = 'IB' AND precio_entrada IS NOT NULL
            ORDER BY symbol, fecha
        """))
        for row in result:
            if row[0] not in precios_compra and row[1]:
                precios_compra[row[0]] = row[1]

    # Obtener holdings de IB
    ib_holdings = portfolio_service.get_holdings_for_date('IB', ib_date)
    ib_cash = portfolio_service.get_cash_for_date('IB', ib_date)
    eur_usd = portfolio_service.get_eur_usd_rate(ib_date)

    # Construir tabla de posiciones con P&L
    positions_data = []
    total_holdings_usd = 0
    total_holdings_eur = 0
    total_pnl_eur = 0

    for ticker, data in ib_holdings.items():
        shares = data.get('shares', 0)
        price_actual = portfolio_service.get_symbol_price(ticker, ib_date) or 0
        price_compra = precios_compra.get(ticker, price_actual)

        value_usd = shares * price_actual
        value_eur = value_usd / eur_usd if eur_usd else 0

        total_holdings_usd += value_usd
        total_holdings_eur += value_eur

        # Calcular P&L
        if shares > 0:  # LARGO
            tipo = "LARGO"
            pnl_usd = shares * (price_actual - price_compra)
            rent_pct = ((price_actual - price_compra) / price_compra) * 100 if price_compra else 0
        else:  # CORTO
            tipo = "CORTO"
            # Para cortos: ganamos cuando el precio baja
            pnl_usd = abs(shares) * (price_compra - price_actual)
            rent_pct = ((price_compra - price_actual) / price_compra) * 100 if price_compra else 0

        pnl_eur = pnl_usd / eur_usd if eur_usd else 0
        total_pnl_eur += pnl_eur

        pnl_sign = "+" if pnl_eur >= 0 else ""
        rent_sign = "+" if rent_pct >= 0 else ""

        positions_data.append({
            'Ticker': ticker,
            'Tipo': tipo,
            'Shares': f"{shares:,.0f}",
            'Precio Compra': f"${price_compra:.2f}",
            'Precio Actual': f"${price_actual:.2f}",
            'Rent. %': f"{rent_sign}{rent_pct:.2f}%",
            'P&L EUR': f"{pnl_sign}{pnl_eur:,.0f} €",
            'Valor EUR': f"{value_eur:,.0f} €"
        })

    # Ordenar: largos primero, luego cortos
    positions_data.sort(key=lambda x: (0 if x['Tipo'] == 'LARGO' else 1, x['Ticker']))

    # Añadir fila de total holdings
    total_pnl_sign = "+" if total_pnl_eur >= 0 else ""
    positions_data.append({
        'Ticker': 'TOTAL',
        'Tipo': '-',
        'Shares': '-',
        'Precio Compra': '-',
        'Precio Actual': '-',
        'Rent. %': '-',
        'P&L EUR': f"{total_pnl_sign}{total_pnl_eur:,.0f} €",
        'Valor EUR': f"{total_holdings_eur:,.0f} €"
    })

    # Mostrar tabla de posiciones abiertas
    def color_position_row(row):
        styles = [''] * len(row)
        if row['Ticker'] == 'TOTAL':
            return ['background-color: #333333; font-weight: bold; color: white'] * len(row)

        # Color para Tipo
        tipo_idx = row.index.get_loc('Tipo')
        if row['Tipo'] == 'LARGO':
            styles[tipo_idx] = 'color: #00cc00; font-weight: bold'
        elif row['Tipo'] == 'CORTO':
            styles[tipo_idx] = 'color: #ff4444; font-weight: bold'

        # Color para Rent. % y P&L EUR
        rent_idx = row.index.get_loc('Rent. %')
        pnl_idx = row.index.get_loc('P&L EUR')
        if row['Rent. %'].startswith('+'):
            styles[rent_idx] = 'color: #00cc00; font-weight: bold'
            styles[pnl_idx] = 'color: #00cc00; font-weight: bold'
        elif row['Rent. %'].startswith('-'):
            styles[rent_idx] = 'color: #ff4444; font-weight: bold'
            styles[pnl_idx] = 'color: #ff4444; font-weight: bold'

        return styles

    positions_df = pd.DataFrame(positions_data)
    styled_positions = positions_df.style.apply(color_position_row, axis=1)
    st.dataframe(styled_positions, use_container_width=True, hide_index=True)

    # ==========================================================================
    # POSICIONES CERRADAS
    # ==========================================================================
    st.subheader("Posiciones Cerradas")

    with db.get_session() as session:
        # Buscar ventas en ib_trades (posiciones cerradas)
        result = session.execute(text("""
            SELECT t.symbol, t.trade_date, t.quantity, t.price as sell_price, t.realized_pnl,
                   (SELECT AVG(t2.price) FROM ib_trades t2
                    WHERE t2.symbol = t.symbol AND t2.trade_type = 'BUY'
                    AND t2.trade_date < t.trade_date) as buy_price
            FROM ib_trades t
            WHERE t.trade_type = 'SELL'
            ORDER BY t.trade_date DESC
        """))
        closed_positions = list(result)

    if closed_positions:
        closed_data = []
        for row in closed_positions:
            symbol, trade_date, qty, sell_price, realized_pnl, buy_price = row
            buy_price = buy_price or sell_price
            pnl_usd = realized_pnl or (qty * (sell_price - buy_price))
            pnl_eur = pnl_usd / eur_usd if eur_usd else 0
            pnl_sign = "+" if pnl_eur >= 0 else ""

            closed_data.append({
                'Ticker': symbol,
                'Fecha Cierre': trade_date.strftime('%d/%m/%Y') if hasattr(trade_date, 'strftime') else str(trade_date)[:10],
                'Shares': f"{qty:,.0f}",
                'Precio Compra': f"${buy_price:.2f}",
                'Precio Venta': f"${sell_price:.2f}",
                'P&L EUR': f"{pnl_sign}{pnl_eur:,.0f} €"
            })

        def color_closed_row(row):
            styles = [''] * len(row)
            pnl_idx = row.index.get_loc('P&L EUR')
            if row['P&L EUR'].startswith('+'):
                styles[pnl_idx] = 'color: #00cc00; font-weight: bold'
            elif row['P&L EUR'].startswith('-'):
                styles[pnl_idx] = 'color: #ff4444; font-weight: bold'
            return styles

        closed_df = pd.DataFrame(closed_data)
        styled_closed = closed_df.style.apply(color_closed_row, axis=1)
        st.dataframe(styled_closed, use_container_width=True, hide_index=True)
    else:
        st.info("No hay posiciones cerradas")

    # Cash
    cash_eur = ib_cash.get('EUR', 0)
    cash_usd = ib_cash.get('USD', 0)
    cash_usd_en_eur = cash_usd / eur_usd if eur_usd else 0
    total_cash_eur = cash_eur + cash_usd_en_eur

    # Total cuenta IB
    total_cuenta_eur = total_holdings_eur + total_cash_eur

    st.markdown(
        f"""
        <div style="background-color: #1a1a1a; color: white; padding: 15px; border-radius: 5px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span><strong>Cash EUR:</strong> EUR {cash_eur:,.2f}</span>
                <span><strong>Cash USD:</strong> ${cash_usd:,.2f} (EUR {cash_usd_en_eur:,.2f})</span>
                <span><strong>Total Cash:</strong> EUR {total_cash_eur:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 1.3em; border-top: 1px solid #444; padding-top: 10px;">
                <span><strong>TOTAL CUENTA IB:</strong></span>
                <span><strong>EUR {total_cuenta_eur:,.2f}</strong></span>
            </div>
        </div>
        """.replace(",", "."),
        unsafe_allow_html=True
    )

    st.caption(f"Fecha: {ib_date.strftime('%d/%m/%Y')} | EUR/USD: {eur_usd:.4f}")


elif page == "Carih Mensual" and backtesting_option == "Estrategia Mensual":
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
        "Julio 2025": "AMD, GOOGL, EPAM, PLD, CTAS, ODFL, DHR, WAT, MSCI, DHI",
        "Junio 2025": "TSLA, NVDA, AMZN, KMB, RTO, GILD, RMD, SAN, LLY, DB1",
        "Mayo 2025": "AVGO, EA, LRCX, SNPS, NVDA, AMAT, TDG, CTAS, MCK, LYV",
        "Abril 2025": "TYL, UL, STZ, TT, BSX, MA, EQT, BKNG, HAS, AMZN",
        "Marzo 2025": "ANET, CHD, STZ, EQIX, AEM, GSK, EW, WST, CTRA, BKNG",
    }

    # Mostrar resumen histórico con botón para cargar
    st.subheader("Histórico de Meses")

    # Mapeo de sectores para cada símbolo
    sector_map = {
        # Tecnología
        'AVGO': 'Tecnología', 'NET': 'Tecnología', 'ANET': 'Tecnología', 'KLAC': 'Tecnología',
        'MSFT': 'Tecnología', 'INTC': 'Tecnología', 'AMD': 'Tecnología', 'GOOGL': 'Tecnología',
        'NVDA': 'Tecnología', 'LRCX': 'Tecnología', 'SNPS': 'Tecnología', 'AMAT': 'Tecnología',
        'CRM': 'Tecnología', 'AKAM': 'Tecnología', 'HPE': 'Tecnología', 'IBM': 'Tecnología',
        'EPAM': 'Tecnología', 'TYL': 'Tecnología', 'EA': 'Tecnología',
        # Financiero
        'MA': 'Financiero', 'MCO': 'Financiero', 'AXP': 'Financiero', 'SCHW': 'Financiero',
        'JPM': 'Financiero', 'BAC': 'Financiero', 'WFC': 'Financiero', 'ALL': 'Financiero',
        'AIZ': 'Financiero', 'MSCI': 'Financiero', 'BKNG': 'Financiero',
        # Salud
        'SYK': 'Salud', 'REGN': 'Salud', 'IDXX': 'Salud', 'HEI': 'Salud', 'VRTX': 'Salud',
        'WST': 'Salud', 'BDX': 'Salud', 'HCA': 'Salud', 'AMGN': 'Salud', 'ABBV': 'Salud',
        'GILD': 'Salud', 'RMD': 'Salud', 'LLY': 'Salud', 'BSX': 'Salud', 'EW': 'Salud',
        'GSK': 'Salud', 'DHR': 'Salud', 'WAT': 'Salud', 'PODD': 'Salud',
        # Consumo Discrecional
        'AMZN': 'Consumo Disc.', 'TJX': 'Consumo Disc.', 'ULTA': 'Consumo Disc.',
        'GM': 'Consumo Disc.', 'NFLX': 'Consumo Disc.', 'TSLA': 'Consumo Disc.',
        'DHI': 'Consumo Disc.', 'HAS': 'Consumo Disc.', 'LYV': 'Consumo Disc.',
        # Consumo Básico
        'PEP': 'Consumo Básico', 'EL': 'Consumo Básico', 'MNST': 'Consumo Básico',
        'KMB': 'Consumo Básico', 'STZ': 'Consumo Básico', 'CHD': 'Consumo Básico',
        'SYY': 'Consumo Básico', 'UL': 'Consumo Básico',
        # Industrial
        'GE': 'Industrial', 'PCAR': 'Industrial', 'CBRE': 'Industrial', 'AME': 'Industrial',
        'LDOS': 'Industrial', 'MSI': 'Industrial', 'PAYC': 'Industrial', 'CTAS': 'Industrial',
        'ODFL': 'Industrial', 'TDG': 'Industrial', 'MCK': 'Industrial', 'TT': 'Industrial',
        'ORLY': 'Industrial', 'IRM': 'Industrial', 'PWR': 'Industrial', 'FTI': 'Industrial',
        # Energía
        'PAA': 'Energía', 'CVX': 'Energía', 'E': 'Energía', 'EQT': 'Energía', 'CTRA': 'Energía',
        # Materiales
        'STLD': 'Materiales', 'BTG': 'Materiales', 'RIO': 'Materiales', 'AEM': 'Materiales',
        # Utilities
        'FE': 'Utilities', 'EVRG': 'Utilities', 'D': 'Utilities', 'CEG': 'Utilities',
        'XEL': 'Utilities',
        # Real Estate
        'PLD': 'Real Estate', 'EQIX': 'Real Estate',
        # Comunicaciones
        'CBOE': 'Comunicaciones', 'RTO': 'Comunicaciones', 'SAN': 'Comunicaciones',
        'DB1': 'Comunicaciones',
        # Otros
        'BRK-B': 'Financiero',
    }

    # Datos históricos pre-calculados (ordenados cronológicamente)
    # Formato: avg=retorno medio, symbols=tickers, positive=positivos, negative=negativos, zero=neutros
    historical_data = {
        "Marzo 2025": {"avg": -0.98, "symbols": 10, "positive": 5, "negative": 5, "zero": 0},
        "Abril 2025": {"avg": 1.96, "symbols": 10, "positive": 7, "negative": 3, "zero": 0},
        "Mayo 2025": {"avg": 8.28, "symbols": 10, "positive": 9, "negative": 1, "zero": 0},
        "Junio 2025": {"avg": 2.38, "symbols": 9, "positive": 7, "negative": 2, "zero": 1},
        "Julio 2025": {"avg": 0.64, "symbols": 10, "positive": 4, "negative": 6, "zero": 0},
        "Agosto 2025": {"avg": 2.94, "symbols": 9, "positive": 5, "negative": 4, "zero": 1},
        "Septiembre 2025": {"avg": 7.17, "symbols": 7, "positive": 7, "negative": 0, "zero": 3},
        "Octubre 2025": {"avg": 0.43, "symbols": 10, "positive": 6, "negative": 4, "zero": 0},
        "Noviembre 2025": {"avg": 1.90, "symbols": 10, "positive": 6, "negative": 4, "zero": 0},
        "Diciembre 2025": {"avg": -0.40, "symbols": 10, "positive": 5, "negative": 5, "zero": 0},
        "Enero 2026": {"avg": -0.01, "symbols": 10, "positive": 7, "negative": 3, "zero": 0},
        "Febrero 2026": {"avg": 1.69, "symbols": 10, "positive": 6, "negative": 4, "zero": 0},
    }

    if st.button("Calcular Retornos Reales", key="calc_hist"):
        with st.spinner("Calculando retornos históricos..."):
            monthly_returns = {}
            with db.get_session() as session:
                for month_key, symbols_str in monthly_selections.items():
                    parts = month_key.split()
                    m_name = parts[0]
                    m_year = int(parts[1])
                    m_num = [k for k, v in month_names.items() if v == m_name][0]

                    if m_num == 1:
                        prev_m, prev_y = 12, m_year - 1
                    else:
                        prev_m, prev_y = m_num - 1, m_year
                    if m_num == 12:
                        next_m, next_y = 1, m_year + 1
                    else:
                        next_m, next_y = m_num + 1, m_year

                    first_day = datetime(m_year, m_num, 1)
                    first_day_next = datetime(next_y, next_m, 1)
                    start_date = datetime(prev_y, prev_m, 1)

                    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
                    returns = []

                    for sym in symbols:
                        db_symbol = session.query(Symbol).filter(Symbol.code == sym).first()
                        if db_symbol:
                            prices = db.get_price_history(session, db_symbol.id, start_date=start_date)
                            if not prices.empty:
                                prev_prices = prices[(prices.index >= start_date) & (prices.index < first_day)]
                                curr_prices = prices[(prices.index >= first_day) & (prices.index < first_day_next)]
                                if not prev_prices.empty and not curr_prices.empty:
                                    open_p = prev_prices['close'].iloc[-1]
                                    close_p = curr_prices['close'].iloc[-1]
                                    ret = ((close_p - open_p) / open_p) * 100
                                    returns.append(ret)

                    if returns:
                        historical_data[month_key] = {
                            "avg": sum(returns) / len(returns),
                            "symbols": len(symbols),
                            "positive": sum(1 for r in returns if r > 0)
                        }
    # Orden cronológico (más antiguo primero)
    month_order = [
        "Marzo 2025", "Abril 2025", "Mayo 2025", "Junio 2025",
        "Julio 2025", "Agosto 2025", "Septiembre 2025", "Octubre 2025",
        "Noviembre 2025", "Diciembre 2025", "Enero 2026", "Febrero 2026"
    ]

    hist_data = []
    for month_key in month_order:
        if month_key in historical_data:
            r = historical_data[month_key]
            num_symbols = r['symbols']
            positivos = r['positive']
            negativos = r.get('negative', num_symbols - positivos)
            zeros = r.get('zero', 0)

            # Calcular Win% sobre tickers reales (excluyendo zeros)
            effective_symbols = positivos + negativos
            win_pct = f"{positivos/effective_symbols*100:.0f}%" if effective_symbols > 0 else '-'

            # El avg ya está calculado sobre 10 (si hay menos tickers, los faltantes son 0%)
            hist_data.append({
                'Año': month_key.split()[1],
                'Mes': month_key.split()[0],
                'Tickers': num_symbols,
                '+': positivos,
                '-': negativos,
                '0': zeros if zeros > 0 else '-',
                'Win%': win_pct,
                'Ret. Medio': f"{r['avg']:+.2f}%"
            })

    if hist_data:
        # Métricas globales del histórico (con comisiones 0.3% total por mes)
        COMISION_MENSUAL = 0.3  # 0.3% por mes
        # Crear dict con retornos netos por mes
        month_returns_neto = {}
        for m in month_order:
            if m in historical_data:
                month_returns_neto[m] = historical_data[m]['avg'] - COMISION_MENSUAL

        positive_months = sum(1 for a in month_returns_neto.values() if a > 0)
        negative_months = sum(1 for a in month_returns_neto.values() if a < 0)

        # Encontrar mes con máximo y mínimo
        max_month = max(month_returns_neto, key=month_returns_neto.get)
        min_month = min(month_returns_neto, key=month_returns_neto.get)
        max_ret = month_returns_neto[max_month]
        min_ret = month_returns_neto[min_month]

        # Datos de acciones individuales (mejores y peores del histórico)
        # Basado en los datos verificados de la base de datos
        best_stock = {"symbol": "INTC", "month": "Sept 25", "ret": 37.78}
        worst_stock = {"symbol": "INTC", "month": "Ago 25", "ret": -18.21}

        # Profit Factor = suma ganancias / suma pérdidas
        gains = sum(a for a in month_returns_neto.values() if a > 0)
        losses = abs(sum(a for a in month_returns_neto.values() if a < 0))
        profit_factor = gains / losses if losses > 0 else float('inf')

        # Calcular retorno acumulado y medio
        ret_acumulado = sum(month_returns_neto.values())
        ret_medio = ret_acumulado / len(month_returns_neto)

        # Calcular Sharpe y Sortino Ratios
        import numpy as np
        returns_list = list(month_returns_neto.values())
        risk_free_rate = 0.3  # 0.3% mensual (~4% anual)

        # Sharpe Ratio = (ret_medio - rf) / std_dev
        std_dev = np.std(returns_list) if len(returns_list) > 1 else 0
        sharpe_ratio = (ret_medio - risk_free_rate) / std_dev if std_dev > 0 else 0

        # Sortino Ratio = (ret_medio - rf) / downside_deviation
        negative_returns = [r for r in returns_list if r < 0]
        downside_dev = np.std(negative_returns) if len(negative_returns) > 1 else 0
        sortino_ratio = (ret_medio - risk_free_rate) / downside_dev if downside_dev > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Meses Positivos", f"{positive_months}/{len(historical_data)}")
        col2.metric("Win Rate", f"{positive_months/len(historical_data)*100:.0f}%")
        col3.metric("Ret. Medio Mensual", f"{ret_medio:+.2f}%")
        col4.metric("Ret. Acumulado", f"{ret_acumulado:+.2f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric(f"Máx. Mes ({max_month.split()[0][:3]})", f"{max_ret:+.2f}%")
        col6.metric(f"Mín. Mes ({min_month.split()[0][:3]})", f"{min_ret:+.2f}%")
        col7.metric(f"Máx. Acción ({best_stock['symbol']})", f"+{best_stock['ret']:.1f}%")
        col8.metric(f"Mín. Acción ({worst_stock['symbol']})", f"{worst_stock['ret']:.1f}%")

        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
        col10.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col11.metric("Sortino Ratio", f"{sortino_ratio:.2f}")

    st.markdown("---")

    # === RENTABILIDAD POR SECTOR ===
    st.subheader("Rentabilidad por Sector")
    st.caption("Calculado desde retornos históricos reales (después de 0.3% comisión)")

    # Calcular retornos por ticker y agrupar por sector
    COMISION = 0.3  # 0.3% comisión
    ticker_returns = {}  # {ticker: [retornos de cada mes]}

    with db.get_session() as session:
        for month_key, symbols_str in monthly_selections.items():
            parts = month_key.split()
            m_name = parts[0]
            m_year = int(parts[1])
            m_num = [k for k, v in month_names.items() if v == m_name][0]

            # Calcular fechas del mes
            if m_num == 1:
                prev_m, prev_y = 12, m_year - 1
            else:
                prev_m, prev_y = m_num - 1, m_year
            if m_num == 12:
                next_m, next_y = 1, m_year + 1
            else:
                next_m, next_y = m_num + 1, m_year

            first_day = datetime(m_year, m_num, 1)
            first_day_next = datetime(next_y, next_m, 1)
            start_date = datetime(prev_y, prev_m, 1)

            symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

            for sym in symbols:
                db_symbol = session.query(Symbol).filter(Symbol.code == sym).first()
                if db_symbol:
                    prices = db.get_price_history(session, db_symbol.id, start_date=start_date)
                    if not prices.empty:
                        prev_prices = prices[(prices.index >= start_date) & (prices.index < first_day)]
                        curr_prices = prices[(prices.index >= first_day) & (prices.index < first_day_next)]
                        if not prev_prices.empty and not curr_prices.empty:
                            open_p = prev_prices['close'].iloc[-1]
                            close_p = curr_prices['close'].iloc[-1]
                            ret = ((close_p - open_p) / open_p) * 100
                            if sym not in ticker_returns:
                                ticker_returns[sym] = []
                            ticker_returns[sym].append(ret)

    # Agrupar por sector y calcular promedio
    sector_stats = {}
    for ticker, returns in ticker_returns.items():
        sector = sector_map.get(ticker, 'Otros')
        if sector not in sector_stats:
            sector_stats[sector] = {'returns': [], 'tickers': set()}
        sector_stats[sector]['returns'].extend(returns)
        sector_stats[sector]['tickers'].add(ticker)

    sector_data = []
    for sector, stats in sector_stats.items():
        if stats['returns']:
            avg_bruto = sum(stats['returns']) / len(stats['returns'])
            avg_neto = avg_bruto - COMISION
            sector_data.append({
                'Sector': sector,
                'Tickers': len(stats['tickers']),
                'Operaciones': len(stats['returns']),
                'Ret. Neto': f"{avg_neto:+.2f}%"
            })

    if sector_data:
        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values('Operaciones', ascending=False)

        def style_sector(row):
            styles = [''] * len(row)
            try:
                ret = float(row['Ret. Neto'].replace('%', '').replace('+', ''))
                ret_idx = sector_df.columns.get_loc('Ret. Neto')
                styles[ret_idx] = 'color: green' if ret > 0 else 'color: red' if ret < 0 else ''
            except:
                pass
            return styles

        st.dataframe(sector_df.style.apply(style_sector, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("No hay datos de retornos por sector")

    st.markdown("---")

    # === SIMULACIÓN DE CARTERA ===
    st.subheader("Simulación de Cartera")
    st.caption("Inversión: $50,000 por ticker | Comisiones + Deslizamiento: 0.3% por ticker")

    INVERSION_POR_TICKER = 50000
    TICKERS_ESTANDAR = 10
    COMISION_DESLIZAMIENTO = 0.003  # 0.3% por ticker

    # Calcular P&L mes a mes
    portfolio_data = []
    total_invertido = 0
    total_pnl = 0
    total_comisiones = 0
    capital_acumulado = 0

    for month_key in month_order:
        if month_key in historical_data:
            r = historical_data[month_key]
            num_tickers = r['symbols']
            ret_medio = r['avg']

            # Inversión del mes
            inversion_mes = INVERSION_POR_TICKER * num_tickers
            cash_mes = INVERSION_POR_TICKER * (TICKERS_ESTANDAR - num_tickers) if num_tickers < TICKERS_ESTANDAR else 0

            # Comisiones y deslizamientos (0.3% por cada ticker)
            comisiones_mes = inversion_mes * COMISION_DESLIZAMIENTO

            # P&L del mes (retorno medio aplicado a la inversión, menos comisiones)
            pnl_bruto = inversion_mes * (ret_medio / 100)
            pnl_mes = pnl_bruto - comisiones_mes

            total_invertido += inversion_mes
            total_pnl += pnl_mes
            total_comisiones += comisiones_mes
            capital_acumulado += inversion_mes + pnl_mes

            # Retorno neto (después de comisiones)
            ret_neto = ((pnl_mes) / inversion_mes) * 100 if inversion_mes > 0 else 0

            portfolio_data.append({
                'Mes': month_key,
                'Tickers': num_tickers,
                'Inversión': f"${inversion_mes:,.0f}",
                'Cash': f"${cash_mes:,.0f}" if cash_mes > 0 else '-',
                'Comisiones': f"${comisiones_mes:,.0f}",
                'Ret. Bruto': f"{ret_medio:+.2f}%",
                'Ret. Neto': f"{ret_neto:+.2f}%",
                'P&L Mes': f"${pnl_mes:+,.0f}",
                'P&L Acum.': f"${total_pnl:+,.0f}"
            })

    if portfolio_data:
        portfolio_df = pd.DataFrame(portfolio_data)

        def style_portfolio(row):
            styles = [''] * len(row)
            try:
                pnl_str = row['P&L Mes'].replace('$', '').replace(',', '').replace('+', '')
                pnl_val = float(pnl_str)
                pnl_idx = portfolio_df.columns.get_loc('P&L Mes')
                acum_idx = portfolio_df.columns.get_loc('P&L Acum.')
                ret_bruto_idx = portfolio_df.columns.get_loc('Ret. Bruto')
                ret_neto_idx = portfolio_df.columns.get_loc('Ret. Neto')
                color = 'color: green' if pnl_val > 0 else 'color: red' if pnl_val < 0 else ''
                styles[pnl_idx] = color
                styles[ret_neto_idx] = color

                # Color para retorno bruto
                ret_bruto_str = row['Ret. Bruto'].replace('%', '').replace('+', '')
                ret_bruto_val = float(ret_bruto_str)
                styles[ret_bruto_idx] = 'color: green' if ret_bruto_val > 0 else 'color: red' if ret_bruto_val < 0 else ''

                acum_str = row['P&L Acum.'].replace('$', '').replace(',', '').replace('+', '')
                acum_val = float(acum_str)
                styles[acum_idx] = 'color: green' if acum_val > 0 else 'color: red' if acum_val < 0 else ''
            except:
                pass
            return styles

        st.dataframe(portfolio_df.style.apply(style_portfolio, axis=1), use_container_width=True, hide_index=True)

        # Métricas del portafolio
        meses_positivos = sum(1 for m in month_order if m in historical_data and historical_data[m]['avg'] > 0)
        meses_negativos = sum(1 for m in month_order if m in historical_data and historical_data[m]['avg'] < 0)
        rentabilidad_neta = (total_pnl / total_invertido) * 100 if total_invertido > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Invertido", f"${total_invertido:,.0f}")
        col2.metric("Comisiones", f"${total_comisiones:,.0f}")
        col3.metric("P&L Neto", f"${total_pnl:+,.0f}")
        col4.metric("Rentabilidad Neta", f"{rentabilidad_neta:+.2f}%")
        col5.metric("Win Rate", f"{meses_positivos}/{meses_positivos + meses_negativos}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        current_month = datetime.now().month
        current_year = datetime.now().year
        # Generar opciones: mes actual + 12 meses anteriores
        # NO incluir mes siguiente porque la estrategia no ha empezado
        # (empieza con la compra el último día del mes anterior)
        month_options = []
        m, y = current_month, current_year
        for _ in range(13):  # Mes actual + 12 meses hacia atrás
            month_options.append(f"{month_names[m]} {y}")
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        selected_month = st.selectbox("Selección del Mes", month_options, index=0)
    with col2:
        show_returns = st.checkbox("Show returns %", value=True)

    parts = selected_month.split()
    month_name = parts[0]
    year = int(parts[1])
    month = [k for k, v in month_names.items() if v == month_name][0]

    default_symbols = monthly_selections.get(selected_month, "")
    symbols = [s.strip().upper() for s in default_symbols.split(",") if s.strip()]

    # Mostrar símbolos del mes seleccionado
    if symbols:
        st.markdown(f"**Tickers ({len(symbols)}):** {', '.join(symbols)}")

    if symbols and st.button("Load Prices", type="primary", key=f"load_{selected_month}"):
        st.markdown("---")

        # Calcular mes anterior dinámicamente
        if month == 1:
            prev_month, prev_year = 12, year - 1
        else:
            prev_month, prev_year = month - 1, year

        # Calcular mes siguiente (para determinar fin del mes actual)
        if month == 12:
            next_month, next_year = 1, year + 1
        else:
            next_month, next_year = month + 1, year

        first_day_current = datetime(year, month, 1)
        first_day_next = datetime(next_year, next_month, 1)
        today = datetime.now()

        with db.get_session() as session:
            all_data = {}
            summary_data = []
            start_date = datetime(prev_year, prev_month, 1)

            for symbol_code in symbols:
                db_symbol = session.query(Symbol).filter(Symbol.code == symbol_code).first()
                if db_symbol:
                    prices = db.get_price_history(session, db_symbol.id, start_date=start_date)
                    if not prices.empty:
                        # Filtrar precios hasta hoy
                        prices = prices[prices.index <= today]

                        # Precios del mes anterior (para obtener precio de apertura)
                        prev_month_prices = prices[(prices.index >= datetime(prev_year, prev_month, 1)) &
                                                   (prices.index < first_day_current)]

                        # Precios del mes actual (para obtener precio de cierre)
                        current_month_prices = prices[(prices.index >= first_day_current) &
                                                      (prices.index < first_day_next)]

                        if not prev_month_prices.empty:
                            # Precio apertura = último día de mercado del mes anterior
                            open_price = prev_month_prices['close'].iloc[-1]
                            open_date = prev_month_prices.index[-1]

                            # Precio cierre = último día disponible del mes actual
                            if not current_month_prices.empty:
                                close_price = current_month_prices['close'].iloc[-1]
                                close_date = current_month_prices.index[-1]
                            else:
                                # Mes aún no ha empezado
                                close_price = open_price
                                close_date = open_date

                            # Crear serie con todos los días para la tabla de evolución
                            combined = pd.concat([
                                pd.Series([open_price], index=[open_date], name='close'),
                                current_month_prices['close']
                            ])
                            all_data[symbol_code] = combined

                            # Calcular retorno
                            mtd_return = ((close_price - open_price) / open_price) * 100

                            summary_data.append({
                                "Symbol": symbol_code,
                                "Fecha Apertura": open_date.strftime("%d/%m/%Y"),
                                "Precio Apertura": open_price,
                                "Fecha Cierre": close_date.strftime("%d/%m/%Y"),
                                "Precio Cierre": close_price,
                                "Variación €": close_price - open_price,
                                "Retorno %": mtd_return,
                            })

        if all_data:
            # Tabla de resumen con precios de apertura y cierre
            st.subheader("Resumen Estrategia Mensual")
            if summary_data:
                summary_df = pd.DataFrame(summary_data)

                def style_summary(row):
                    styles = [''] * len(row)
                    ret_idx = summary_df.columns.get_loc("Retorno %")
                    var_idx = summary_df.columns.get_loc("Variación €")
                    if row["Retorno %"] > 0:
                        styles[ret_idx] = 'color: green'
                        styles[var_idx] = 'color: green'
                    elif row["Retorno %"] < 0:
                        styles[ret_idx] = 'color: red'
                        styles[var_idx] = 'color: red'
                    return styles

                styled_summary = summary_df.style.apply(style_summary, axis=1).format({
                    "Precio Apertura": "${:.2f}",
                    "Precio Cierre": "${:.2f}",
                    "Variación €": "${:+.2f}",
                    "Retorno %": "{:+.2f}%"
                })
                st.dataframe(styled_summary, use_container_width=True, hide_index=True)

                # Métricas totales
                total_return = sum(d["Retorno %"] for d in summary_data) / len(summary_data)
                positive = sum(1 for d in summary_data if d["Retorno %"] > 0)
                negative = len(summary_data) - positive

                col1, col2, col3 = st.columns(3)
                col1.metric("Retorno Medio", f"{total_return:+.2f}%")
                col2.metric("Positivos", positive, delta=f"{positive/len(summary_data)*100:.0f}%")
                col3.metric("Negativos", negative)

            st.markdown("---")

            # Tabla de evolución diaria
            st.subheader("Evolución Diaria de Precios")
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


elif page == "Carih Mensual" and backtesting_option == "Portfolio Backtest":
    st.title("Portfolio Backtest")
    st.info("Herramienta de backtesting para estrategias de portfolio.")


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


# =============================================================================
# AI ASSISTANT PAGE
# =============================================================================
elif page == "Asistente IA":
    st.title("🤖 Asistente IA")
    st.markdown("Pregunta sobre tu cartera, posiciones, operaciones y análisis financiero.")

    # Initialize assistant in session state
    if "ai_assistant" not in st.session_state:
        try:
            from src.ai_assistant import FinancialAssistant
            st.session_state.ai_assistant = FinancialAssistant()
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Error inicializando asistente: {e}")
            st.session_state.ai_assistant = None

    assistant = st.session_state.get("ai_assistant")

    if assistant:
        # Show backend status
        col1, col2 = st.columns([3, 1])
        with col1:
            backends = assistant.list_backends()
            available = [k for k, v in backends.items() if v]
            if available:
                st.success(f"✓ Backends disponibles: {', '.join(available)}")
                if assistant.active_backend:
                    st.info(f"→ Usando: **{assistant.active_backend.name}**")
            else:
                st.warning("⚠️ No hay backends de IA disponibles. Configura una API key en .env")

        with col2:
            if available and len(available) > 1:
                new_backend = st.selectbox(
                    "Cambiar backend",
                    available,
                    index=available.index(assistant.active_backend.name) if assistant.active_backend else 0
                )
                if new_backend != (assistant.active_backend.name if assistant.active_backend else None):
                    assistant.switch_backend(new_backend)
                    st.rerun()

        st.markdown("---")

        # Quick summary
        with st.expander("📊 Resumen rápido del Portfolio", expanded=False):
            try:
                summary = assistant.get_quick_summary()
                st.text(summary)
            except Exception as e:
                st.error(f"Error obteniendo resumen: {e}")

        st.markdown("---")

        # Chat interface
        st.subheader("💬 Chat con el Asistente")

        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Escribe tu pregunta aquí..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        response = assistant.ask(prompt)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Limpiar chat"):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("---")

        # Example questions
        st.subheader("💡 Ejemplos de preguntas")
        examples = [
            "¿Cuál es el valor total de mi cartera?",
            "¿Qué posiciones tengo en efectivo?",
            "¿Cuáles son mis posiciones cortas?",
            "Muéstrame las últimas operaciones",
            "¿Cuánto tengo invertido en ETFs?",
            "¿Cuál es mi exposición en futuros?",
        ]

        cols = st.columns(2)
        for i, example in enumerate(examples):
            col = cols[i % 2]
            if col.button(example, key=f"example_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": example})
                st.rerun()

    else:
        st.error("No se pudo inicializar el asistente de IA")
        st.markdown("""
        **Para configurar el asistente:**

        1. Edita el archivo `.env` y añade al menos una API key:
           - `GOOGLE_API_KEY` - Para usar Gemini (gratis)
           - `GROQ_API_KEY` - Para usar Groq/Llama (gratis)
           - `ANTHROPIC_API_KEY` - Para usar Claude

        2. Recarga la página

        **Obtener API keys:**
        - Gemini: https://aistudio.google.com/
        - Groq: https://console.groq.com/
        - Claude: https://console.anthropic.com/
        """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Financial Data Project v3.0")
st.sidebar.markdown("*Data: 28/01/2026*")
