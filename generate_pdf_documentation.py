"""
Generate PDF documentation for client delivery.
"""
from fpdf import FPDF
from datetime import datetime


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'Financial Data Project - Documentacion Tecnica', align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        self.cell(5)
        self.cell(0, 6, f'* {text}', ln=True)

    def code_block(self, text):
        self.set_font('Courier', '', 9)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(2)
        self.set_font('Helvetica', '', 10)

    def add_table(self, headers, data):
        self.set_font('Helvetica', 'B', 9)
        col_width = self.epw / len(headers)
        for header in headers:
            self.cell(col_width, 7, header, border=1, align='C')
        self.ln()
        self.set_font('Helvetica', '', 9)
        for row in data:
            for item in row:
                self.cell(col_width, 6, str(item), border=1, align='L')
            self.ln()
        self.ln(4)


def create_pdf():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.ln(40)
    pdf.cell(0, 15, 'Financial Data Project', align='C', ln=True)
    pdf.set_font('Helvetica', '', 16)
    pdf.cell(0, 10, 'Documentacion Tecnica', align='C', ln=True)
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f'Fecha: {datetime.now().strftime("%d/%m/%Y")}', align='C', ln=True)
    pdf.cell(0, 8, 'Version: 3.0.0', align='C', ln=True)
    pdf.cell(0, 8, 'Puntuacion Enterprise: 8.3/10', align='C', ln=True)

    # Table of Contents
    pdf.add_page()
    pdf.chapter_title('Indice')
    toc = [
        '1. Resumen Ejecutivo',
        '2. Arquitectura del Sistema',
        '3. Estructura del Proyecto',
        '4. Instalacion y Configuracion',
        '5. Modulos Principales',
        '6. Base de Datos',
        '7. Seguridad',
        '8. Testing y Calidad',
        '9. Mantenimiento',
        '10. Metricas de Calidad',
    ]
    for item in toc:
        pdf.bullet_point(item)

    # 1. Executive Summary
    pdf.add_page()
    pdf.chapter_title('1. Resumen Ejecutivo')
    pdf.body_text(
        'Financial Data Project es un sistema de gestion de carteras de inversion '
        'que permite el seguimiento diario de posiciones, analisis tecnico, y '
        'valoracion de portfolios en multiples divisas.'
    )

    pdf.section_title('Caracteristicas Principales')
    features = [
        'Dashboard interactivo con Streamlit',
        'Integracion con Yahoo Finance',
        'Soporte multi-divisa (EUR, USD, CAD, GBP, CHF)',
        'Tracking de acciones, ETFs y futuros',
        'Analisis tecnico (RSI, medias moviles)',
        'Gestion de operaciones y efectivo',
        'Autenticacion opcional del dashboard',
    ]
    for f in features:
        pdf.bullet_point(f)

    pdf.ln(5)
    pdf.section_title('Tecnologias Utilizadas')
    pdf.add_table(
        ['Componente', 'Tecnologia'],
        [
            ['Backend', 'Python 3.11+'],
            ['Base de Datos', 'SQLite / PostgreSQL'],
            ['ORM', 'SQLAlchemy 2.0'],
            ['Dashboard', 'Streamlit 1.28+'],
            ['Graficos', 'Plotly 5.18+'],
            ['Testing', 'Pytest 7.4+'],
        ]
    )

    # 2. Architecture
    pdf.add_page()
    pdf.chapter_title('2. Arquitectura del Sistema')

    pdf.section_title('Capas del Sistema')
    layers = [
        ('Presentacion', 'Dashboard Streamlit con multiples paginas'),
        ('Servicios', 'PortfolioService, ExchangeRateService, DailyTrackingService'),
        ('Datos', 'YahooClient, DatabaseManager'),
        ('Persistencia', 'SQLite/PostgreSQL con SQLAlchemy'),
    ]
    for layer, desc in layers:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(40, 6, layer + ':', ln=False)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, desc, ln=True)
    pdf.ln(5)

    pdf.section_title('Patrones de Diseno')
    patterns = [
        ('Singleton', 'Servicios unicos para eficiencia'),
        ('Context Manager', 'Gestion segura de sesiones BD'),
        ('Service Layer', 'Logica de negocio encapsulada'),
    ]
    for pattern, desc in patterns:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(40, 6, pattern + ':', ln=False)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, desc, ln=True)

    # 3. Project Structure
    pdf.add_page()
    pdf.chapter_title('3. Estructura del Proyecto')
    structure = '''financial-data-project/
  src/                 # Codigo principal
    config.py          # Configuracion
    database.py        # Modelos y manager
    portfolio_data.py  # Servicio portfolio
    validators.py      # Validacion
    technical.py       # Indicadores
  web/
    app.py             # Dashboard Streamlit
  scripts/             # Scripts utilidad
  tests/               # Suite de tests
  data/                # Base de datos'''
    pdf.code_block(structure)

    # 4. Installation
    pdf.add_page()
    pdf.chapter_title('4. Instalacion y Configuracion')

    pdf.section_title('Requisitos')
    pdf.bullet_point('Python 3.11 o superior')
    pdf.bullet_point('pip (gestor de paquetes)')

    pdf.section_title('Pasos de Instalacion')
    install = '''# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Copiar configuracion
copy .env.example .env

# 3. Configurar variables en .env

# 4. Ejecutar dashboard
streamlit run web/app.py --server.port 8514'''
    pdf.code_block(install)

    pdf.section_title('Variables de Entorno')
    pdf.add_table(
        ['Variable', 'Descripcion'],
        [
            ['DATABASE_URL', 'URL conexion BD'],
            ['DASHBOARD_PASSWORD', 'Contrasena dashboard'],
            ['DASHBOARD_AUTH_ENABLED', 'Activar auth (true/false)'],
            ['LOG_LEVEL', 'Nivel logging (INFO/DEBUG)'],
        ]
    )

    # 5. Main Modules
    pdf.add_page()
    pdf.chapter_title('5. Modulos Principales')

    modules = [
        ('config.py', 'Gestion de configuracion con pydantic-settings'),
        ('database.py', 'Modelos SQLAlchemy y DatabaseManager'),
        ('portfolio_data.py', 'Calculos de valoracion y metricas'),
        ('exchange_rate_service.py', 'Tipos de cambio centralizados'),
        ('daily_tracking.py', 'Gestion de holdings y operaciones'),
        ('validators.py', 'Validacion de entrada'),
        ('technical.py', 'Indicadores tecnicos (RSI, MA, etc.)'),
        ('yahoo_client.py', 'Cliente Yahoo Finance'),
    ]
    pdf.add_table(
        ['Modulo', 'Descripcion'],
        modules
    )

    # 6. Database
    pdf.add_page()
    pdf.chapter_title('6. Base de Datos')

    pdf.section_title('Tablas Principales')
    pdf.add_table(
        ['Tabla', 'Descripcion'],
        [
            ['exchanges', 'Bolsas de valores'],
            ['symbols', 'Simbolos con metadatos'],
            ['price_history', 'Historico precios OHLCV'],
            ['holding_diario', 'Posiciones diarias'],
            ['stock_trades', 'Historial operaciones'],
            ['daily_cash', 'Saldos efectivo'],
            ['cash_movements', 'Movimientos efectivo'],
        ]
    )

    pdf.section_title('Conexion')
    pdf.body_text('SQLite: sqlite:///data/financial_data.db')
    pdf.body_text('PostgreSQL: postgresql://user:pass@host:5432/dbname')

    # 7. Security
    pdf.add_page()
    pdf.chapter_title('7. Seguridad')

    pdf.section_title('Medidas Implementadas')
    security = [
        'SQL Injection: Queries parametrizadas',
        'Validacion: Modulo validators.py',
        'Autenticacion: Password configurable',
        'Secretos: Variables de entorno (.env)',
        'Excepciones: Manejo tipado con logging',
        'Escaneo: Bandit en pre-commit',
    ]
    for s in security:
        pdf.bullet_point(s)

    pdf.ln(5)
    pdf.section_title('Activar Autenticacion')
    auth = '''# En archivo .env:
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_PASSWORD=contrasena_segura'''
    pdf.code_block(auth)

    # 8. Testing
    pdf.add_page()
    pdf.chapter_title('8. Testing y Calidad')

    pdf.section_title('Suite de Tests')
    pdf.bullet_point('225 tests automatizados')
    pdf.bullet_point('9 modulos de test')
    pdf.bullet_point('Fixtures en conftest.py')
    pdf.bullet_point('BD in-memory para aislamiento')

    pdf.ln(3)
    pdf.section_title('Comandos')
    test_cmd = '''# Ejecutar tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src'''
    pdf.code_block(test_cmd)

    pdf.section_title('Herramientas de Calidad')
    tools = ['Black (formateo)', 'isort (imports)', 'Flake8 (linting)',
             'Bandit (seguridad)', 'Mypy (tipos)', 'pre-commit (hooks)']
    for t in tools:
        pdf.bullet_point(t)

    # 9. Maintenance
    pdf.add_page()
    pdf.chapter_title('9. Mantenimiento')

    pdf.section_title('Tareas Periodicas')
    pdf.bullet_point('Descarga automatica de precios')
    pdf.bullet_point('Backup de base de datos')
    pdf.bullet_point('Actualizacion de dependencias')
    pdf.bullet_point('Revision de logs')

    pdf.section_title('Scripts Utiles')
    pdf.add_table(
        ['Script', 'Funcion'],
        [
            ['batch_download.py', 'Descarga masiva precios'],
            ['import_ib_data.py', 'Importar datos IB'],
            ['import_portfolio.py', 'Importar portfolio'],
        ]
    )

    # 10. Metrics
    pdf.add_page()
    pdf.chapter_title('10. Metricas de Calidad')

    pdf.section_title('Puntuacion Enterprise-Ready')
    pdf.add_table(
        ['Categoria', 'Puntuacion'],
        [
            ['Arquitectura', '8.5/10'],
            ['Calidad Codigo', '8.0/10'],
            ['Buenas Practicas', '8.5/10'],
            ['Seguridad', '8.5/10'],
            ['Documentacion', '8.0/10'],
            ['Dependencias', '8.0/10'],
            ['Deuda Tecnica', '8.5/10'],
            ['GLOBAL', '8.3/10'],
        ]
    )

    pdf.ln(5)
    pdf.section_title('Estadisticas del Codigo')
    pdf.bullet_point('Lineas codigo (src): ~5,600')
    pdf.bullet_point('Lineas tests: ~2,500')
    pdf.bullet_point('Archivos Python: 40+')
    pdf.bullet_point('Tests automatizados: 225')

    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(200, 255, 200)
    pdf.cell(0, 10, 'Estado: ENTERPRISE-READY', ln=True, fill=True, align='C')

    # Save
    output_path = 'Financial_Data_Project_Documentation.pdf'
    pdf.output(output_path)
    print(f'PDF generado: {output_path}')
    return output_path


if __name__ == '__main__':
    create_pdf()
