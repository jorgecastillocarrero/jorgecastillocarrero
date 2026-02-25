"""
Generador de PDF - Carihuela Inversiones
Informe completo de inversiones
"""
import sys
sys.path.insert(0, '.')

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime, date
from src.database import get_db_manager
from sqlalchemy import text

# =============================================================================
# CONFIGURACION VISUAL (igual que Futuros)
# =============================================================================
GREEN = colors.HexColor('#006600')
RED = colors.HexColor('#cc0000')
HEADER_BG = colors.HexColor('#1a365d')  # Azul oscuro
ROW_ALT = colors.HexColor('#e8f4f8')    # Azul claro alterno

def create_table_style():
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ROW_ALT]),
    ])

def fmt(n):
    """Formato numero con punto como separador de miles"""
    return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_int(n):
    return f"{n:,}".replace(",", ".")

# =============================================================================
# CONEXION A BASE DE DATOS
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

db = get_db_manager()

# =============================================================================
# OBTENER DATOS DINAMICOS (Seccion 2 - Cartera)
# =============================================================================
def to_date(d):
    if isinstance(d, datetime):
        return d.date()
    return d

with db.get_session() as session:
    fecha_actual = to_date(session.execute(text('SELECT MAX(fecha) FROM posicion')).scalar())
    fecha_anterior = to_date(session.execute(text(
        'SELECT MAX(fecha) FROM posicion WHERE fecha < :fecha'
    ), {'fecha': fecha_actual}).scalar())

    fx = session.execute(text('''
        SELECT ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 1.04
    fx_inicial = 1.1747

    valor_inicial = session.execute(text(
        "SELECT SUM(total_eur) FROM posicion WHERE fecha = '2025-12-31'"
    )).scalar() or 3930529
    valor_actual = session.execute(text(
        'SELECT SUM(total_eur) FROM posicion WHERE fecha = :fecha'
    ), {'fecha': fecha_actual}).scalar()
    valor_anterior = session.execute(text(
        'SELECT SUM(total_eur) FROM posicion WHERE fecha = :fecha'
    ), {'fecha': fecha_anterior}).scalar()

    ganancia_eur = valor_actual - valor_inicial
    rent_pct = (valor_actual / valor_inicial - 1) * 100

    spy_ini = session.execute(text('''
        SELECT ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY' AND ph.date = '2025-12-31'
    ''')).scalar() or 681.92
    spy_act = session.execute(text('''
        SELECT ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()
    qqq_ini = session.execute(text('''
        SELECT ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ' AND ph.date = '2025-12-31'
    ''')).scalar() or 614.31
    qqq_act = session.execute(text('''
        SELECT ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()

    spy_rent = (spy_act / spy_ini - 1) * 100
    qqq_rent = (qqq_act / qqq_ini - 1) * 100

    valor_enero = session.execute(text('''
        SELECT SUM(total_eur) FROM posicion
        WHERE fecha = (SELECT MAX(fecha) FROM posicion WHERE fecha <= '2026-01-31')
    ''')).scalar()
    spy_enero = session.execute(text('''
        SELECT ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY' AND ph.date <= '2026-01-31' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()
    qqq_enero = session.execute(text('''
        SELECT ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ' AND ph.date <= '2026-01-31' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()

    # Datos para grafico
    cartera_data_chart = session.execute(text('''
        SELECT fecha, SUM(total_eur) as total FROM posicion
        WHERE fecha >= '2025-12-31' AND fecha != '2026-01-01'
        GROUP BY fecha ORDER BY fecha
    ''')).fetchall()
    spy_data_chart = session.execute(text('''
        SELECT ph.date, ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY' AND ph.date >= '2025-12-31' ORDER BY ph.date
    ''')).fetchall()
    qqq_data_chart = session.execute(text('''
        SELECT ph.date, ph.close FROM price_history ph JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ' AND ph.date >= '2025-12-31' ORDER BY ph.date
    ''')).fetchall()

rent_ene = (valor_enero / valor_inicial - 1) * 100
rent_feb = (valor_actual / valor_enero - 1) * 100
spy_ene = (spy_enero / spy_ini - 1) * 100
spy_feb = (spy_act / spy_enero - 1) * 100
qqq_ene = (qqq_enero / qqq_ini - 1) * 100
qqq_feb = (qqq_act / qqq_enero - 1) * 100

# Generar grafico benchmark
cartera_fechas = [to_date(r[0]) for r in cartera_data_chart]
cartera_valores = [float(r[1]) for r in cartera_data_chart]
cartera_ini_v = cartera_valores[0]
cartera_rent_chart = [(v / cartera_ini_v - 1) * 100 for v in cartera_valores]
spy_fechas = [to_date(r[0]) for r in spy_data_chart]
spy_precios = [float(r[1]) for r in spy_data_chart]
spy_ini_v = spy_precios[0]
spy_rent_chart = [(p / spy_ini_v - 1) * 100 for p in spy_precios]
qqq_fechas = [to_date(r[0]) for r in qqq_data_chart]
qqq_precios = [float(r[1]) for r in qqq_data_chart]
qqq_ini_v = qqq_precios[0]
qqq_rent_chart = [(p / qqq_ini_v - 1) * 100 for p in qqq_precios]

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
ax.set_facecolor('white')
ax.plot(cartera_fechas, cartera_rent_chart, label=f'Cartera (+{cartera_rent_chart[-1]:.2f}%)', color='#1a365d', linewidth=2.5)
ax.plot(spy_fechas, spy_rent_chart, label=f'SPY ({spy_rent_chart[-1]:+.2f}%)', color='#e74c3c', linewidth=2, linestyle='--')
ax.plot(qqq_fechas, qqq_rent_chart, label=f'QQQ ({qqq_rent_chart[-1]:+.2f}%)', color='#f39c12', linewidth=2, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Rentabilidad (%)', fontsize=11)
ax.set_title('Rentabilidad Cartera vs Benchmark (desde 31/12/2025)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
benchmark_chart_path = r'C:\Users\usuario\Downloads\rentabilidad_benchmark.png'
plt.savefig(benchmark_chart_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# Variacion diaria dinamica
from src.variacion_diaria_tipo_activo import VariacionDiariaTipoActivo
variacion_calc = VariacionDiariaTipoActivo()
variacion_results = variacion_calc.calculate_variacion_diaria(str(fecha_actual))

# =============================================================================
# CREAR PDF
# =============================================================================
pdf_path = r'C:\Users\usuario\Downloads\Carihuela_Inversiones.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, alignment=TA_CENTER, spaceAfter=10, textColor=HEADER_BG)
subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=16, spaceAfter=10, spaceBefore=20, textColor=HEADER_BG)
section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=15)
description_style = ParagraphStyle('Description', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=12, spaceBefore=5)
normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, spaceAfter=6)

elements = []

# =============================================================================
# PORTADA
# =============================================================================
elements.append(Spacer(1, 4*cm))
elements.append(Paragraph("CARIHUELA INVERSIONES", title_style))
elements.append(Spacer(1, 1*cm))
elements.append(Paragraph("Informe de Inversiones", ParagraphStyle('Sub', parent=styles['Normal'], fontSize=16, alignment=TA_CENTER, textColor=colors.grey)))
elements.append(Spacer(1, 2*cm))
elements.append(Paragraph(f"Fecha: {datetime.now().strftime('%d de %B de %Y')}", ParagraphStyle('Date', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER)))
elements.append(Spacer(1, 0.5*cm))
elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ParagraphStyle('Gen', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=colors.grey)))

elements.append(PageBreak())

# =============================================================================
# INDICE
# =============================================================================
elements.append(Paragraph("INDICE", subtitle_style))
elements.append(Spacer(1, 0.5*cm))

indice = [
    "1. [Pendiente]",
    "2. Cartera",
    "3. Composicion",
    "4. Acciones",
    "5. [Pendiente]",
    "6. ETFs",
    "7. Futuros",
    "8. [Pendiente]",
]

for item in indice:
    elements.append(Paragraph(item, normal_style))

elements.append(PageBreak())

# =============================================================================
# 2. CARTERA
# =============================================================================
elements.append(Paragraph("2. CARTERA", subtitle_style))
elements.append(Paragraph(
    "Resumen de la posicion global de la cartera y comparativa con benchmark.",
    description_style
))

# 2.1 Resumen de Cartera
elements.append(Paragraph("2.1 Resumen de Cartera", section_style))
cartera_data = [
    ['Metrica', 'EUR', 'USD'],
    ['Valor Inicial 31/12/2025', f'{fmt_int(int(valor_inicial))} EUR', f'${fmt_int(int(valor_inicial * fx_inicial))}'],
    [f'Valor Actual {fecha_actual.strftime("%d/%m/%Y")}', f'{fmt_int(int(valor_actual))} EUR', f'${fmt_int(int(valor_actual * fx))}'],
    ['Ganancia Acumulada 2026', f'+{fmt_int(int(ganancia_eur))} EUR (+{rent_pct:.2f}%)', f'+${fmt_int(int(ganancia_eur * fx))} (+{rent_pct:.2f}%)'],
]
t = Table(cartera_data, colWidths=[5.5*cm, 4.5*cm, 4.5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 3), (2, 3), GREEN if ganancia_eur >= 0 else RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(f"Tipos de cambio: EUR/USD 31/12/2025: {fx_inicial:.4f} | EUR/USD actual: {fx:.4f}", description_style))
elements.append(Spacer(1, 0.5*cm))

# 2.2 Benchmark
elements.append(Paragraph("2.2 Benchmark 2026", section_style))
benchmark_data = [
    ['Indice', 'Rentabilidad'],
    ['SPY', f'{spy_rent:+.2f}%'],
    ['QQQ', f'{qqq_rent:+.2f}%'],
    ['Alpha vs SPY', f'+{rent_pct - spy_rent:.2f}%'],
    ['Alpha vs QQQ', f'+{rent_pct - qqq_rent:.2f}%'],
]
t = Table(benchmark_data, colWidths=[5*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 1), GREEN if spy_rent >= 0 else RED)
s.add('TEXTCOLOR', (1, 2), (1, 2), GREEN if qqq_rent >= 0 else RED)
s.add('TEXTCOLOR', (1, 3), (1, 4), GREEN)
s.add('FONTNAME', (0, 3), (1, 4), 'Helvetica-Bold')
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 2.3 Variacion Diaria por Tipo de Activo
elements.append(Paragraph("2.3 Variacion Diaria por Tipo de Activo", section_style))
elements.append(Paragraph(
    f"Comparativa del valor de la cartera por tipo de activo entre {fecha_anterior.strftime('%d/%m')} y {fecha_actual.strftime('%d/%m')}.",
    description_style
))

by_strat = dict(variacion_results.get('by_strategy', {}))
cash_d = by_strat.pop('Cash', {})
etfs_d = by_strat.pop('ETFs', {})
by_strat['Cash/ETFs'] = {
    'actual': cash_d.get('actual', 0) + etfs_d.get('actual', 0),
    'anterior': cash_d.get('anterior', 0) + etfs_d.get('anterior', 0),
    'diff': cash_d.get('diff', 0) + etfs_d.get('diff', 0),
}
by_strat['Cash/ETFs']['pct'] = (by_strat['Cash/ETFs']['diff'] / by_strat['Cash/ETFs']['anterior'] * 100) if by_strat['Cash/ETFs']['anterior'] > 0 else 0

variacion_data = [['Tipo', fecha_anterior.strftime('%d/%m'), fecha_actual.strftime('%d/%m'), 'Diferencia', 'Var %']]
var_color_rules = []
var_row_idx = 1
for tipo in ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/ETFs']:
    data = by_strat.get(tipo)
    if not data:
        continue
    diff = data['diff']
    pct = data['pct']
    variacion_data.append([
        tipo, fmt_int(int(data['anterior'])), fmt_int(int(data['actual'])),
        f'{diff:+,.0f}'.replace(',', '.'), f'{pct:+.2f}%'.replace('.', ',')
    ])
    var_color_rules.append(('TEXTCOLOR', (3, var_row_idx), (4, var_row_idx), GREEN if diff >= 0 else RED))
    var_row_idx += 1

total_diff = valor_actual - valor_anterior
total_pct_var = (total_diff / valor_anterior * 100) if valor_anterior > 0 else 0
variacion_data.append([
    'TOTAL', fmt_int(int(valor_anterior)), fmt_int(int(valor_actual)),
    f'{total_diff:+,.0f}'.replace(',', '.'), f'{total_pct_var:+.2f}%'.replace('.', ',')
])
var_color_rules.append(('TEXTCOLOR', (3, var_row_idx), (4, var_row_idx), GREEN if total_diff >= 0 else RED))

t = Table(variacion_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm])
s = create_table_style()
for rule in var_color_rules:
    s.add(*rule)
s.add('FONTNAME', (0, var_row_idx), (-1, var_row_idx), 'Helvetica-Bold')
s.add('BACKGROUND', (0, var_row_idx), (-1, var_row_idx), colors.HexColor('#d9e2ec'))
s.add('TEXTCOLOR', (0, var_row_idx), (2, var_row_idx), colors.HexColor('#1a365d'))
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 2.4 Grafica de Rentabilidad vs Benchmark
elements.append(Paragraph("2.4 Rentabilidad vs Benchmark", section_style))
elements.append(Paragraph(
    "Evolucion de la rentabilidad de la cartera comparada con SPY y QQQ desde 31/12/2025.",
    description_style
))
if os.path.exists(benchmark_chart_path):
    chart_img = Image(benchmark_chart_path, width=16*cm, height=9*cm)
    elements.append(chart_img)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    f"La cartera supera significativamente a los indices de referencia con +{rent_pct:.2f}% vs SPY ({spy_rent:+.2f}%) y QQQ ({qqq_rent:+.2f}%).",
    description_style
))
elements.append(Spacer(1, 0.5*cm))

# 2.5 Rentabilidad Mensual vs Benchmark
elements.append(Paragraph("2.5 Rentabilidad Mensual vs Benchmark", section_style))
elements.append(Paragraph(
    "Comparativa de rentabilidad mensual de la cartera frente a los indices SPY y QQQ.",
    description_style
))
rent_mensual_data = [
    ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Total 2026'],
    ['Cartera', f'+{rent_ene:.2f}%', f'+{rent_feb:.2f}%', '-', '-', '-', f'+{rent_pct:.2f}%'],
    ['SPY', f'{spy_ene:+.2f}%', f'{spy_feb:+.2f}%', '-', '-', '-', f'{spy_rent:+.2f}%'],
    ['QQQ', f'{qqq_ene:+.2f}%', f'{qqq_feb:+.2f}%', '-', '-', '-', f'{qqq_rent:+.2f}%'],
]
t = Table(rent_mensual_data, colWidths=[2*cm, 2.2*cm, 2.2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2.5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 1), GREEN if rent_ene >= 0 else RED)
s.add('TEXTCOLOR', (2, 1), (2, 1), GREEN if rent_feb >= 0 else RED)
s.add('TEXTCOLOR', (6, 1), (6, 1), GREEN if rent_pct >= 0 else RED)
s.add('TEXTCOLOR', (1, 2), (1, 2), GREEN if spy_ene >= 0 else RED)
s.add('TEXTCOLOR', (2, 2), (2, 2), GREEN if spy_feb >= 0 else RED)
s.add('TEXTCOLOR', (6, 2), (6, 2), GREEN if spy_rent >= 0 else RED)
s.add('TEXTCOLOR', (1, 3), (1, 3), GREEN if qqq_ene >= 0 else RED)
s.add('TEXTCOLOR', (2, 3), (2, 3), GREEN if qqq_feb >= 0 else RED)
s.add('TEXTCOLOR', (6, 3), (6, 3), GREEN if qqq_rent >= 0 else RED)
s.add('FONTNAME', (6, 0), (6, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (6, 0), (6, 0), colors.HexColor('#333333'))
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    f"Alpha vs SPY: +{rent_pct - spy_rent:.2f}% | Alpha vs QQQ: +{rent_pct - qqq_rent:.2f}%",
    ParagraphStyle('AlphaNote', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=GREEN)
))

elements.append(PageBreak())

# =============================================================================
# 3. COMPOSICION
# =============================================================================
elements.append(Paragraph("3. COMPOSICION", subtitle_style))
elements.append(Paragraph(
    "Distribucion de la cartera por diversificacion, estrategia y cuenta.",
    description_style
))

# Obtener datos de composicion desde portfolio_service
from src.portfolio_data import PortfolioDataService
portfolio_service = PortfolioDataService()

# Fecha actual
comp_fecha = db.get_session().__enter__().execute(text('SELECT MAX(fecha) FROM posicion')).scalar()
if hasattr(comp_fecha, 'date'):
    comp_fecha = comp_fecha.date()

# Valores por tipo de activo (combinar Cash + ETFs)
strategy_values = portfolio_service.get_values_by_asset_type(comp_fecha)
if 'Cash' in strategy_values or 'ETFs' in strategy_values:
    strategy_values['Cash/ETFs'] = strategy_values.pop('Cash', 0) + strategy_values.pop('ETFs', 0)

# Totales por cuenta
def get_account_totals(fecha):
    all_holdings = portfolio_service.get_all_holdings_for_date(fecha)
    eur_usd_rate = portfolio_service.get_eur_usd_rate(fecha)
    result = {}
    for account in ['CO3365', 'RCO951', 'LACAIXA', 'IB']:
        holdings = all_holdings.get(account, {})
        holding_value = 0
        for symbol, data in holdings.items():
            shares = data['shares']
            value = portfolio_service.calculate_position_value(symbol, shares, fecha)
            if value:
                holding_value += value
        cash_data = portfolio_service.get_cash_for_date(account, fecha)
        cash_eur = 0
        if cash_data:
            cash_eur += cash_data.get('EUR', 0)
            cash_eur += cash_data.get('USD', 0) / eur_usd_rate
        result[account] = holding_value + cash_eur
    result['TOTAL'] = sum(result.values())
    return result

account_totals = get_account_totals(comp_fecha)
total_cartera = account_totals['TOTAL']

# 3.1 Diversificacion
elements.append(Paragraph("3.1 Composicion por Diversificacion", section_style))

bolsa = (strategy_values.get('Quant', 0) +
         strategy_values.get('Value', 0) +
         strategy_values.get('Alpha Picks', 0) +
         strategy_values.get('Mensual', 0) +
         strategy_values.get('Stock', 0))
oro = strategy_values.get('Oro/Mineras', 0)
liquidez = strategy_values.get('Cash/ETFs', 0) or (strategy_values.get('Cash', 0) + strategy_values.get('ETFs', 0))

# Grafico de queso - Diversificacion
fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
div_labels = ['Bolsa', 'Oro', 'Liquidez']
div_values = [bolsa, oro, liquidez]
div_colors = ['#636EFA', '#FFA15A', '#00CC96']
wedges, texts, autotexts = ax.pie(div_values, labels=div_labels, autopct='%1.1f%%',
                                   colors=div_colors, wedgeprops=dict(width=0.6))
ax.set_title('Diversificacion por Clase de Activo', fontsize=12, fontweight='bold')
plt.tight_layout()
div_chart_path = r'C:\Users\usuario\Downloads\comp_diversificacion.png'
plt.savefig(div_chart_path, dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

# Tabla y grafico lado a lado usando tabla
div_data = [
    ['Clase', 'Valor EUR', '%'],
    ['Bolsa', fmt(bolsa), f'{bolsa/total_cartera*100:.1f}%'],
    ['Oro', fmt(oro), f'{oro/total_cartera*100:.1f}%'],
    ['Liquidez', fmt(liquidez), f'{liquidez/total_cartera*100:.1f}%'],
    ['TOTAL', fmt(total_cartera), '100%'],
]
t = Table(div_data, colWidths=[3*cm, 3*cm, 2*cm])
s = create_table_style()
s.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, -1), (-1, -1), colors.white)
t.setStyle(s)

# Layout: grafico a la izquierda, tabla a la derecha
div_img = Image(div_chart_path, width=7*cm, height=7*cm)
layout_div = Table([[div_img, t]], colWidths=[8*cm, 9*cm])
layout_div.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
elements.append(layout_div)
elements.append(Spacer(1, 0.5*cm))

# 3.2 Por Estrategia
elements.append(Paragraph("3.2 Composicion por Estrategia", section_style))

# Ordenar por valor
sorted_strategies = sorted([(k, v) for k, v in strategy_values.items() if v > 0], key=lambda x: -x[1])
total_estrategias = sum(v for k, v in sorted_strategies)

# Grafico de queso - Estrategia
fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
est_labels = [s[0] for s in sorted_strategies]
est_values = [s[1] for s in sorted_strategies]
wedges, texts, autotexts = ax.pie(est_values, labels=est_labels, autopct='%1.1f%%',
                                   wedgeprops=dict(width=0.6))
ax.set_title('Distribucion por Estrategia', fontsize=12, fontweight='bold')
plt.tight_layout()
est_chart_path = r'C:\Users\usuario\Downloads\comp_estrategia.png'
plt.savefig(est_chart_path, dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

est_data = [['Estrategia', 'Valor EUR', '%']]
for estrategia, valor in sorted_strategies:
    pct = valor / total_estrategias * 100 if total_estrategias > 0 else 0
    est_data.append([estrategia, fmt(valor), f'{pct:.1f}%'])
est_data.append(['TOTAL', fmt(total_estrategias), '100%'])

t = Table(est_data, colWidths=[3*cm, 3*cm, 2*cm])
s = create_table_style()
s.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, -1), (-1, -1), colors.white)
t.setStyle(s)

est_img = Image(est_chart_path, width=7*cm, height=7*cm)
layout_est = Table([[est_img, t]], colWidths=[8*cm, 9*cm])
layout_est.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
elements.append(layout_est)
elements.append(Spacer(1, 0.5*cm))

# 3.3 Por Cuenta
elements.append(Paragraph("3.3 Composicion por Cuenta", section_style))

# Grafico de queso - Cuenta
fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
cuenta_labels = ['RCO951', 'La Caixa', 'CO3365', 'IB']
cuenta_values = [account_totals.get('RCO951', 0), account_totals.get('LACAIXA', 0),
                 account_totals.get('CO3365', 0), account_totals.get('IB', 0)]
wedges, texts, autotexts = ax.pie(cuenta_values, labels=cuenta_labels, autopct='%1.1f%%',
                                   wedgeprops=dict(width=0.6))
ax.set_title('Distribucion por Cuenta', fontsize=12, fontweight='bold')
plt.tight_layout()
cuenta_chart_path = r'C:\Users\usuario\Downloads\comp_cuenta.png'
plt.savefig(cuenta_chart_path, dpi=120, bbox_inches='tight', facecolor='white')
plt.close()

cuenta_data = [['Cuenta', 'Valor EUR', '%']]
for cuenta, label in [('RCO951', 'RCO951'), ('LACAIXA', 'La Caixa'), ('CO3365', 'CO3365'), ('IB', 'IB')]:
    valor = account_totals.get(cuenta, 0)
    pct = valor / total_cartera * 100 if total_cartera > 0 else 0
    cuenta_data.append([label, fmt(valor), f'{pct:.1f}%'])
cuenta_data.append(['TOTAL', fmt(total_cartera), '100%'])

t = Table(cuenta_data, colWidths=[3*cm, 3*cm, 2*cm])
s = create_table_style()
s.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, -1), (-1, -1), colors.white)
t.setStyle(s)

cuenta_img = Image(cuenta_chart_path, width=7*cm, height=7*cm)
layout_cuenta = Table([[cuenta_img, t]], colWidths=[8*cm, 9*cm])
layout_cuenta.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
elements.append(layout_cuenta)

elements.append(PageBreak())

# =============================================================================
# 4. ACCIONES
# =============================================================================
from src.portfolio_data import PortfolioDataService
acciones_service = PortfolioDataService()

elements.append(Paragraph("4. Acciones", subtitle_style))
elements.append(Paragraph(
    "Posiciones en acciones (excluye ETFs y Futuros). Datos de compra, rentabilidad periodo y rentabilidad historica.",
    description_style
))

# Obtener fecha mas reciente
with db.get_session() as session:
    acc_fecha = session.execute(text('SELECT MAX(fecha) FROM posicion')).scalar()
    if hasattr(acc_fecha, 'date'):
        acc_fecha = acc_fecha.date()

# Tipos de cambio
eur_usd_current = acciones_service.get_eur_usd_rate(acc_fecha)
eur_usd_31dic = acciones_service.get_exchange_rate('EURUSD=X', date(2025, 12, 31)) or 1.1747
cad_eur_current = acciones_service.get_cad_eur_rate(acc_fecha) or 0.619
chf_eur_current = acciones_service.get_chf_eur_rate(acc_fecha) or 1.06

# Mapeo exchange a moneda
EXCHANGE_TO_CURRENCY = {'US': '$', 'TO': 'C$', 'MC': '€', 'SW': 'CHF', 'L': '£', 'DE': '€', 'F': '€', 'MI': '€'}

# 4.1 Posiciones Abiertas
elements.append(Paragraph("4.1 Posiciones Abiertas", section_style))

with db.get_session() as session:
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
    """), {'fecha': acc_fecha})

    acc_data = [['Ticker', 'Cuenta', 'Titulos', 'P.Compra', 'P.Actual', 'Valor EUR', 'Rent.%']]
    total_valor_eur = 0
    total_rent_eur = 0
    posiciones = []

    for account, ticker, shares, currency_code, asset_type, fecha_compra, precio_compra_db in holdings_result.fetchall():
        parts = ticker.split('.')
        exchange = parts[1] if len(parts) > 1 else 'US'
        currency_symbol = EXCHANGE_TO_CURRENCY.get(exchange, '$')
        cuenta_display = 'La Caixa' if account == 'LACAIXA' else account

        # Precios
        precio_compra = precio_compra_db
        precio_actual = acciones_service.get_symbol_price(ticker, acc_fecha)
        if not precio_actual:
            precio_actual = acciones_service.get_symbol_price(parts[0], acc_fecha)

        if precio_actual:
            # Calcular valor EUR según moneda
            if currency_symbol == '$':  # USD
                valor_eur = (shares * precio_actual) / eur_usd_current
            elif currency_symbol == 'C$':  # CAD
                valor_eur = (shares * precio_actual) * cad_eur_current
            elif currency_symbol == 'CHF':  # CHF
                valor_eur = (shares * precio_actual) * chf_eur_current
            else:  # EUR
                valor_eur = shares * precio_actual

            # Rentabilidad
            rent_pct = ((precio_actual / precio_compra) - 1) * 100 if precio_compra and precio_compra > 0 else 0

            total_valor_eur += valor_eur
            if precio_compra:
                if currency_symbol == '$':  # USD
                    valor_compra_eur = (shares * precio_compra) / eur_usd_current
                elif currency_symbol == 'C$':  # CAD
                    valor_compra_eur = (shares * precio_compra) * cad_eur_current
                elif currency_symbol == 'CHF':  # CHF
                    valor_compra_eur = (shares * precio_compra) * chf_eur_current
                else:  # EUR
                    valor_compra_eur = shares * precio_compra
                total_rent_eur += valor_eur - valor_compra_eur

            posiciones.append({
                'ticker': parts[0],
                'cuenta': cuenta_display,
                'shares': int(shares),
                'p_compra': f"{currency_symbol}{precio_compra:.2f}" if precio_compra else '-',
                'p_actual': f"{currency_symbol}{precio_actual:.2f}",
                'valor_eur': valor_eur,
                'rent_pct': rent_pct
            })

# Ordenar por valor EUR descendente y tomar top 15
posiciones.sort(key=lambda x: -x['valor_eur'])
for pos in posiciones[:15]:
    rent_color = '+' if pos['rent_pct'] >= 0 else ''
    acc_data.append([
        pos['ticker'],
        pos['cuenta'],
        fmt_int(pos['shares']),
        pos['p_compra'],
        pos['p_actual'],
        fmt(pos['valor_eur']),
        f"{rent_color}{pos['rent_pct']:.1f}%"
    ])

if len(posiciones) > 15:
    otros_valor = sum(p['valor_eur'] for p in posiciones[15:])
    acc_data.append(['... otros', '', '', '', '', fmt(otros_valor), ''])

acc_data.append(['TOTAL', '', '', '', '', fmt(total_valor_eur), f"{total_rent_eur/total_valor_eur*100:.1f}%" if total_valor_eur > 0 else ''])

t = Table(acc_data, colWidths=[2.2*cm, 2*cm, 1.5*cm, 2*cm, 2*cm, 2.5*cm, 2*cm])
s = create_table_style()
s.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, -1), (-1, -1), colors.white)
# Colorear rentabilidades
for i, pos in enumerate(posiciones[:15], start=1):
    if pos['rent_pct'] >= 0:
        s.add('TEXTCOLOR', (6, i), (6, i), GREEN)
    else:
        s.add('TEXTCOLOR', (6, i), (6, i), RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 4.2 Posiciones Cerradas
elements.append(Paragraph("4.2 Posiciones Cerradas", section_style))

with db.get_session() as session:
    ventas_result = session.execute(text("""
        SELECT fecha, account_code, symbol,
               SUM(shares) as total_shares,
               SUM(importe_total) / SUM(shares) as precio_venta,
               currency,
               AVG(rent_periodo) as rent_periodo
        FROM ventas
        WHERE symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
        AND symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        GROUP BY fecha, account_code, symbol, currency
        ORDER BY fecha DESC
        LIMIT 10
    """))

    # Precios compra
    compras_result = session.execute(text("""
        SELECT symbol, AVG(precio) as precio_compra
        FROM compras
        GROUP BY symbol
    """))
    precios_compra_map = {row[0]: row[1] for row in compras_result.fetchall()}

ventas_data = [['Fecha', 'Ticker', 'Titulos', 'P.Compra', 'P.Venta', 'Rent.%']]
for venta in ventas_result.fetchall():
    fecha_v, cuenta_v, symbol_v, shares_v, precio_venta_v, currency_v, rent_periodo_v = venta
    precio_compra_v = precios_compra_map.get(symbol_v, 0)
    curr_sym = '$' if currency_v == 'USD' else '€'

    if precio_compra_v and precio_compra_v > 0:
        rent_hist = ((precio_venta_v / precio_compra_v) - 1) * 100
    else:
        rent_hist = 0

    rent_sign = '+' if rent_hist >= 0 else ''
    ventas_data.append([
        fecha_v.strftime('%d/%m') if hasattr(fecha_v, 'strftime') else str(fecha_v)[:5],
        symbol_v.split('.')[0],
        fmt_int(int(shares_v)),
        f"{curr_sym}{precio_compra_v:.2f}" if precio_compra_v else '-',
        f"{curr_sym}{precio_venta_v:.2f}",
        f"{rent_sign}{rent_hist:.1f}%"
    ])

t = Table(ventas_data, colWidths=[2*cm, 2.5*cm, 1.5*cm, 2.2*cm, 2.2*cm, 2*cm])
s = create_table_style()
# Colorear rentabilidades
for i in range(1, len(ventas_data)):
    rent_str = ventas_data[i][5]
    if rent_str.startswith('+'):
        s.add('TEXTCOLOR', (5, i), (5, i), GREEN)
    elif rent_str.startswith('-'):
        s.add('TEXTCOLOR', (5, i), (5, i), RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 4.3 Resumen
elements.append(Paragraph("4.3 Resumen Acciones", section_style))

# Calcular estadisticas
total_posiciones = len(posiciones)
positivas = sum(1 for p in posiciones if p['rent_pct'] > 0)
negativas = sum(1 for p in posiciones if p['rent_pct'] < 0)

resumen_acc = [
    ['Metrica', 'Valor'],
    ['Total Posiciones Abiertas', str(total_posiciones)],
    ['Valor Total EUR', fmt(total_valor_eur)],
    ['P&L EUR', f"{'+' if total_rent_eur >= 0 else ''}{fmt(total_rent_eur)}"],
    ['Positivas / Negativas', f"{positivas} / {negativas}"],
    ['% Positivas', f"{positivas/total_posiciones*100:.1f}%" if total_posiciones > 0 else '0%'],
]
t = Table(resumen_acc, colWidths=[5*cm, 4*cm])
s = create_table_style()
if total_rent_eur >= 0:
    s.add('TEXTCOLOR', (1, 3), (1, 3), GREEN)
else:
    s.add('TEXTCOLOR', (1, 3), (1, 3), RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 4.5 Rentabilidad por Market Cap
elements.append(Paragraph("4.5 Rentabilidad por Market Cap - Periodo 2026", section_style))
elements.append(Paragraph(
    "Rentabilidad agrupada por capitalizacion de mercado. Base: 31/12/2025 para acciones existentes, precio de compra para nuevas.",
    description_style
))

# Obtener tipos de cambio
with db.get_session() as session:
    eur_usd_31dic = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X' AND ph.date = '2025-12-31'
    ''')).scalar() or 1.1747

    eur_usd_current = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 1.1871

    cad_eur_31dic = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CADEUR=X' AND ph.date = '2025-12-31'
    ''')).scalar() or 0.6215

    cad_eur_current = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CADEUR=X' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 0.619

    chf_eur_31dic = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CHFEUR=X' AND ph.date = '2025-12-31'
    ''')).scalar() or 1.06

    chf_eur_current = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'CHFEUR=X' ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 1.06

    # Market cap data
    mcap_result = session.execute(text('''
        SELECT s.code, f.market_cap FROM fundamentals f
        JOIN symbols s ON f.symbol_id = s.id
        WHERE f.market_cap IS NOT NULL AND f.market_cap > 0
    '''))
    mcap_data = {row[0]: row[1] for row in mcap_result.fetchall()}

def get_mcap_cat(mcap):
    if mcap is None: return 'Sin datos'
    m = mcap / 1_000_000
    if m < 5000: return '<5.000M'
    elif m < 10000: return '5.000-10.000M'
    elif m < 50000: return '10.000-50.000M'
    else: return '>50.000M'

EXCHANGE_TO_CURRENCY_MCAP = {'US': 'USD', 'TO': 'CAD', 'MC': 'EUR', 'SW': 'CHF', 'L': 'GBP', 'DE': 'EUR', 'MI': 'EUR'}
fecha_corte_mcap = date(2025, 12, 31)
mcap_positions = []

# ABIERTAS
with db.get_session() as session:
    holdings_mcap = session.execute(text('''
        SELECT h.symbol, h.shares, c.precio as p_compra, c.fecha as f_compra
        FROM holding_diario h
        LEFT JOIN (
            SELECT account_code, symbol, MIN(fecha) as fecha,
                   (SELECT precio FROM compras c2 WHERE c2.account_code = c1.account_code AND c2.symbol = c1.symbol ORDER BY fecha LIMIT 1) as precio
            FROM compras c1 GROUP BY account_code, symbol
        ) c ON h.account_code = c.account_code AND h.symbol = c.symbol
        WHERE h.fecha = :f
        AND (h.asset_type IS NULL OR h.asset_type NOT IN ('ETF', 'ETFs', 'Future', 'Futures'))
        AND h.symbol NOT LIKE '%SGLE%'
    '''), {'f': acc_fecha}).fetchall()

    for ticker, shares, p_compra, f_compra in holdings_mcap:
        if not p_compra: continue
        parts = ticker.split('.')
        exchange = parts[1] if len(parts) > 1 else 'US'
        currency = EXCHANGE_TO_CURRENCY_MCAP.get(exchange, 'USD')

        if hasattr(f_compra, 'date'):
            f_compra = f_compra.date()

        p_actual = acciones_service.get_symbol_price(ticker, acc_fecha)
        if not p_actual:
            p_actual = acciones_service.get_symbol_price(parts[0], acc_fecha)
        if not p_actual: continue

        p_31dic = acciones_service.get_symbol_price(ticker, fecha_corte_mcap)
        if not p_31dic:
            p_31dic = acciones_service.get_symbol_price(parts[0], fecha_corte_mcap)

        if f_compra and f_compra > fecha_corte_mcap:
            precio_base = p_compra
        else:
            precio_base = p_31dic if p_31dic else p_compra

        # Calcular valor EUR con tipos de cambio correctos
        if currency == 'USD':
            if f_compra and f_compra > fecha_corte_mcap:
                valor_base_eur = (shares * precio_base) / eur_usd_current
            else:
                valor_base_eur = (shares * precio_base) / eur_usd_31dic
            valor_actual_eur = (shares * p_actual) / eur_usd_current
        elif currency == 'CAD':
            if f_compra and f_compra > fecha_corte_mcap:
                valor_base_eur = (shares * precio_base) * cad_eur_current
            else:
                valor_base_eur = (shares * precio_base) * cad_eur_31dic
            valor_actual_eur = (shares * p_actual) * cad_eur_current
        elif currency == 'CHF':
            if f_compra and f_compra > fecha_corte_mcap:
                valor_base_eur = (shares * precio_base) * chf_eur_current
            else:
                valor_base_eur = (shares * precio_base) * chf_eur_31dic
            valor_actual_eur = (shares * p_actual) * chf_eur_current
        else:
            valor_base_eur = shares * precio_base
            valor_actual_eur = shares * p_actual

        pnl_eur = valor_actual_eur - valor_base_eur
        rent_periodo = ((valor_actual_eur / valor_base_eur) - 1) * 100 if valor_base_eur > 0 else 0
        mcap = mcap_data.get(parts[0]) or mcap_data.get(ticker)
        mcap_positions.append({'ticker': parts[0], 'rent_pct': rent_periodo, 'pnl_eur': pnl_eur, 'mcap': mcap})

# CERRADAS
with db.get_session() as session:
    ventas_mcap = session.execute(text('''
        SELECT v.symbol, SUM(v.shares) as shares,
               SUM(v.importe_total)/SUM(v.shares) as precio_venta, v.currency,
               AVG(v.precio_31_12) as p31_12,
               (SELECT AVG(c.precio) FROM compras c WHERE c.symbol = v.symbol) as p_compra,
               (SELECT MIN(c.fecha) FROM compras c WHERE c.symbol = v.symbol) as f_compra
        FROM ventas v
        WHERE v.symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
        AND v.symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        AND v.symbol NOT LIKE '%SGLE%'
        GROUP BY v.symbol, v.currency
    ''')).fetchall()

    for symbol, shares, precio_venta, currency, p31_12, p_compra, f_compra in ventas_mcap:
        if hasattr(f_compra, 'date'):
            f_compra = f_compra.date()

        if f_compra and f_compra > fecha_corte_mcap:
            precio_base = p_compra if p_compra else precio_venta
        else:
            precio_base = p31_12 if p31_12 else (p_compra if p_compra else precio_venta)

        if currency == 'USD':
            if f_compra and f_compra > fecha_corte_mcap:
                valor_base_eur = (abs(shares) * precio_base) / eur_usd_current
            else:
                valor_base_eur = (abs(shares) * precio_base) / eur_usd_31dic
            valor_venta_eur = (abs(shares) * precio_venta) / eur_usd_current
        elif currency == 'CAD':
            if f_compra and f_compra > fecha_corte_mcap:
                valor_base_eur = (abs(shares) * precio_base) * cad_eur_current
            else:
                valor_base_eur = (abs(shares) * precio_base) * cad_eur_31dic
            valor_venta_eur = (abs(shares) * precio_venta) * cad_eur_current
        else:
            valor_base_eur = abs(shares) * precio_base
            valor_venta_eur = abs(shares) * precio_venta

        pnl_eur = valor_venta_eur - valor_base_eur
        rent_periodo = ((valor_venta_eur / valor_base_eur) - 1) * 100 if valor_base_eur > 0 else 0
        ticker = symbol.split('.')[0]
        mcap = mcap_data.get(ticker) or mcap_data.get(symbol)
        mcap_positions.append({'ticker': ticker, 'rent_pct': rent_periodo, 'pnl_eur': pnl_eur, 'mcap': mcap})

# Agrupar por market cap
mcap_stats = {'<5.000M': [], '5.000-10.000M': [], '10.000-50.000M': [], '>50.000M': [], 'Sin datos': []}
for p in mcap_positions:
    cat = get_mcap_cat(p['mcap'])
    mcap_stats[cat].append(p)

mcap_table_data = [['Market Cap', 'Operaciones', 'Rent. Media', 'P&L EUR']]
total_mcap_pnl = 0
for cat in ['<5.000M', '5.000-10.000M', '10.000-50.000M', '>50.000M']:
    lst = mcap_stats[cat]
    if lst:
        avg = sum(p['rent_pct'] for p in lst) / len(lst)
        pnl = sum(p['pnl_eur'] for p in lst)
        total_mcap_pnl += pnl
        mcap_table_data.append([cat, str(len(lst)), f"{avg:+.1f}%", f"{pnl:+,.0f}".replace(',', '.')])

mcap_table_data.append(['TOTAL', str(len(mcap_positions)), '', f"{total_mcap_pnl:+,.0f}".replace(',', '.')])

t = Table(mcap_table_data, colWidths=[4*cm, 3*cm, 3*cm, 4*cm])
s = create_table_style()
s.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, -1), (-1, -1), colors.white)
# Colorear P&L
for i in range(1, len(mcap_table_data) - 1):
    pnl_str = mcap_table_data[i][3]
    if pnl_str.startswith('+'):
        s.add('TEXTCOLOR', (3, i), (3, i), GREEN)
    elif pnl_str.startswith('-'):
        s.add('TEXTCOLOR', (3, i), (3, i), RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 4.6 Rentabilidad por Sector vs Benchmark
elements.append(Paragraph("4.6 Rentabilidad por Sector vs Benchmark - Periodo 2026", section_style))
elements.append(Paragraph(
    "Comparativa de la rentabilidad de la cartera por sector frente al ETF benchmark correspondiente (SPDR Select Sector).",
    description_style
))

# ETFs benchmark por sector
SECTOR_ETF = {
    'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF',
    'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Industrials': 'XLI',
    'Basic Materials': 'XLB', 'Energy': 'XLE', 'Utilities': 'XLU',
    'Real Estate': 'XLRE', 'Communication Services': 'XLC',
}

# Obtener datos de sector
with db.get_session() as session:
    sector_result = session.execute(text('''
        SELECT s.code, f.sector FROM fundamentals f
        JOIN symbols s ON f.symbol_id = s.id
        WHERE f.sector IS NOT NULL AND f.sector != ''
    '''))
    sector_data = {row[0]: row[1] for row in sector_result.fetchall()}

# Calcular rentabilidad de cada ETF benchmark
etf_returns = {}
for sector, etf in SECTOR_ETF.items():
    p_31dic = acciones_service.get_symbol_price(etf, fecha_corte_mcap)
    p_actual = acciones_service.get_symbol_price(etf, acc_fecha)
    if p_31dic and p_actual:
        etf_returns[sector] = ((p_actual / p_31dic) - 1) * 100
    else:
        etf_returns[sector] = None

# Calcular rentabilidad de la cartera por sector (reutilizar mcap_positions)
sector_positions = []
for p in mcap_positions:
    sector = sector_data.get(p['ticker'])
    sector_positions.append({'ticker': p['ticker'], 'rent_pct': p['rent_pct'], 'pnl_eur': p['pnl_eur'], 'sector': sector})

# Agrupar por sector
sector_stats = {}
for p in sector_positions:
    sector = p['sector'] or 'Sin datos'
    if sector not in sector_stats:
        sector_stats[sector] = []
    sector_stats[sector].append(p)

# Ordenar por P&L
sorted_sectors = sorted(sector_stats.items(), key=lambda x: sum(p['pnl_eur'] for p in x[1]), reverse=True)

sector_table_data = [['Sector', 'Ops', 'Cartera', 'Benchmark', 'Alpha', 'P&L EUR']]
total_sector_pnl = 0
for sector, lst in sorted_sectors:
    avg_cartera = sum(p['rent_pct'] for p in lst) / len(lst)
    pnl = sum(p['pnl_eur'] for p in lst)
    total_sector_pnl += pnl
    benchmark = etf_returns.get(sector)
    if benchmark is not None:
        alpha = avg_cartera - benchmark
        benchmark_str = f"{benchmark:+.1f}%"
        alpha_str = f"{alpha:+.1f}%"
    else:
        benchmark_str = "N/A"
        alpha_str = "N/A"
    sector_table_data.append([
        sector[:20], str(len(lst)), f"{avg_cartera:+.1f}%",
        benchmark_str, alpha_str, f"{pnl:+,.0f}".replace(',', '.')
    ])

sector_table_data.append(['TOTAL', str(len(sector_positions)), '', '', '', f"{total_sector_pnl:+,.0f}".replace(',', '.')])

t = Table(sector_table_data, colWidths=[4*cm, 1.2*cm, 2*cm, 2.2*cm, 2*cm, 2.8*cm])
s = create_table_style()
s.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, -1), (-1, -1), colors.white)
# Colorear Alpha y P&L
for i in range(1, len(sector_table_data) - 1):
    # Alpha
    alpha_str = sector_table_data[i][4]
    if alpha_str != 'N/A':
        if alpha_str.startswith('+'):
            s.add('TEXTCOLOR', (4, i), (4, i), GREEN)
        elif alpha_str.startswith('-'):
            s.add('TEXTCOLOR', (4, i), (4, i), RED)
    # P&L
    pnl_str = sector_table_data[i][5]
    if pnl_str.startswith('+'):
        s.add('TEXTCOLOR', (5, i), (5, i), GREEN)
    elif pnl_str.startswith('-'):
        s.add('TEXTCOLOR', (5, i), (5, i), RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    "Benchmark ETFs: XLK (Tech), XLV (Health), XLF (Financials), XLY (Consumer Cycl), XLP (Consumer Def), XLI (Industrials), XLB (Materials), XLE (Energy), XLU (Utilities), XLRE (Real Estate)",
    ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
))

elements.append(PageBreak())

# =============================================================================
# 6. ETFs
# =============================================================================
elements.append(Paragraph("6. ETFs", subtitle_style))
elements.append(Paragraph(
    f"Operaciones de ETFs en Interactive Brokers. Periodo: 01/01/2026 - {fecha_actual.strftime('%d/%m/%Y')}.",
    description_style
))

# 6.1 Operaciones Cerradas
elements.append(Paragraph("6.1 Operaciones Cerradas", section_style))
etf_data = [
    ['ETF', 'Nombre', 'Tipo', 'Acciones', 'F. Entrada', 'F. Cierre', 'P. Entrada', 'P. Cierre', 'P&L USD'],
    ['TLT', 'iShares 20+ Year\nTreasury Bond', 'LONG', '8.042', '20-21/01', '30/01\n02/02', '$86,87', '$87,08', '+1.291,44'],
    ['AMLP', 'Alerian MLP ETF', 'SHORT', '8.036', '30/01', '02/02', '$49,97', '$49,62', '+2.810,27'],
    ['EMB', 'iShares JPM USD\nEM Bond', 'SHORT', '8.277', '30/01', '02/02', '$96,64', '$96,16', '+3.863,27'],
    ['EMLC', 'VanEck JPM EM\nLocal Currency', 'SHORT', '30.394', '30/01', '02/02', '$26,30', '$26,22', '+2.333,50'],
    ['XLB', 'Materials Select\nSector SPDR', 'SHORT', '4.288', '30/01', '02/02', '$49,28', '$49,67', '-1.726,35'],
    ['TLT', 'iShares 20+ Year\nTreasury Bond', 'LONG', '17', '06/02', '12/02', '$87,40', '$88,72', '+21,75'],
]
t = Table(etf_data, colWidths=[1.2*cm, 3.2*cm, 1.3*cm, 1.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm])
s = create_table_style()
s.add('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#d4edda'))
s.add('BACKGROUND', (2, 2), (2, 5), colors.HexColor('#f8d7da'))
s.add('BACKGROUND', (2, 6), (2, 6), colors.HexColor('#d4edda'))
s.add('TEXTCOLOR', (8, 1), (8, 4), GREEN)
s.add('TEXTCOLOR', (8, 5), (8, 5), RED)
s.add('TEXTCOLOR', (8, 6), (8, 6), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 6.2 Resumen
elements.append(Paragraph("6.2 Resumen", section_style))
etf_resumen = [
    ['Metrica', 'Valor'],
    ['P&L Bruto', '+$8.593,88'],
    ['Trades', '6'],
    ['Wins / Losses', '5 / 1'],
    ['Win Rate', '83%'],
]
t = Table(etf_resumen, colWidths=[6*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 1), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 6.3 Posiciones Abiertas
elements.append(Paragraph("6.3 Posiciones Abiertas", section_style))
etf_open = [
    ['ETF', 'Nombre', 'Tipo', 'Acciones', 'F. Entrada', 'P. Entrada', 'Valor Aprox.'],
    ['TLT', 'iShares 20+ Year Treasury Bond', 'LONG', '0,424', '06/02', '$87,40', '~$38'],
]
t = Table(etf_open, colWidths=[1.5*cm, 5.5*cm, 1.5*cm, 1.8*cm, 2*cm, 2*cm, 2*cm])
s = create_table_style()
s.add('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#d4edda'))
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 6.4 Por Tipo de Posicion
elements.append(Paragraph("6.4 Por Tipo de Posicion", section_style))
etf_tipo = [
    ['Posicion', 'Trades', 'P&L USD', 'Win Rate'],
    ['LONG', '2', '+$1.313,19', '100%'],
    ['SHORT', '4', '+$7.280,69', '75%'],
]
t = Table(etf_tipo, colWidths=[3*cm, 2.5*cm, 3*cm, 2.5*cm])
s = create_table_style()
s.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda'))
s.add('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#f8d7da'))
s.add('TEXTCOLOR', (2, 1), (2, 2), GREEN)
t.setStyle(s)
elements.append(t)

elements.append(PageBreak())

# =============================================================================
# 7. FUTUROS
# =============================================================================
elements.append(Paragraph("7. FUTUROS", subtitle_style))
elements.append(Paragraph(
    f"Analisis de operaciones de futuros en Interactive Brokers. Periodo: 01/01/2026 - {fecha_actual.strftime('%d/%m/%Y')}.",
    description_style
))

# Obtener datos de futuros
with db.get_session() as session:
    # EUR/USD
    fx = session.execute(text("""
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X'
        ORDER BY ph.date DESC LIMIT 1
    """)).fetchone()
    eur_usd = fx[0] if fx else 1.04

    # Trades de futuros
    result = session.execute(text("""
        SELECT fecha, symbol, shares, pnl, importe_total
        FROM ventas
        WHERE symbol SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        ORDER BY fecha
    """))
    futures_ventas = result.fetchall()

# Procesar datos futuros
CATEGORIES = {'GC': 'Oro', 'CL': 'Petroleo', 'ES': 'Indices', 'NQ': 'Indices', 'HE': 'Ganado'}
DIAS = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']

by_tipo = {}
by_mes = {}
by_dia = {}
total_pnl = 0
total_trades = 0
total_wins = 0
total_losses = 0

for fecha, symbol, shares, pnl, importe in futures_ventas:
    if pnl is None:
        pnl = 0
    if importe is None:
        importe = 0

    prefix = ''.join([c for c in symbol if c.isalpha()])[:2]
    tipo = CATEGORIES.get(prefix, 'Otros')
    if tipo not in by_tipo:
        by_tipo[tipo] = {'trades': 0, 'contracts': 0, 'pnl': 0, 'wins': 0, 'losses': 0, 'importe': 0}
    by_tipo[tipo]['trades'] += 1
    by_tipo[tipo]['contracts'] += abs(int(shares))
    by_tipo[tipo]['pnl'] += pnl
    by_tipo[tipo]['importe'] += abs(importe) if importe else 0
    if pnl > 0:
        by_tipo[tipo]['wins'] += 1
    elif pnl < 0:
        by_tipo[tipo]['losses'] += 1

    if hasattr(fecha, 'month'):
        mes = 'Enero' if fecha.month == 1 else 'Febrero' if fecha.month == 2 else str(fecha.month)
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

    total_pnl += pnl
    total_trades += 1
    if pnl > 0:
        total_wins += 1
    elif pnl < 0:
        total_losses += 1

# 7.1 Resultado Global Futuros
elements.append(Paragraph("7.1 Resultado Global", section_style))

comisiones = -539.60
pnl_neto = total_pnl + comisiones
total_contracts = sum(d.get('contracts', 0) for d in by_tipo.values())
wr = total_wins / total_trades * 100 if total_trades > 0 else 0
avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
sign = '+' if total_pnl >= 0 else ''

tbl = [
    ['Metrica', 'Valor'],
    ['P&L Total USD', f'{sign}${fmt(total_pnl)}'],
    ['P&L Total EUR', f'{sign}{fmt(total_pnl/eur_usd)} EUR'],
    ['Comisiones IB', f'${fmt(comisiones)}'],
    ['P&L Neto USD', f"{'+' if pnl_neto >= 0 else ''}${fmt(pnl_neto)}"],
    ['Trades', f'{total_trades}'],
    ['Contratos', f'{total_contracts}'],
    ['Win Rate', f'{wr:.0f}%'],
    ['P&L Promedio', f"{'+' if avg_pnl >= 0 else ''}${fmt(avg_pnl)} USD/trade"],
]
t = Table(tbl, colWidths=[7*cm, 5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 2), GREEN)
s.add('TEXTCOLOR', (1, 3), (1, 3), RED)
s.add('TEXTCOLOR', (1, 4), (1, 4), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 7.2 Por Tipo de Activo
elements.append(Paragraph("7.2 Por Tipo de Activo", section_style))
tbl = [['Tipo', 'Trades', 'Contr.', 'Importe', 'P&L USD', '%Total']]
for tipo in ['Oro', 'Indices', 'Ganado', 'Petroleo']:
    if tipo in by_tipo:
        d = by_tipo[tipo]
        pct = d['pnl'] / total_pnl * 100 if total_pnl != 0 else 0
        sign = '+' if d['pnl'] >= 0 else ''
        tbl.append([tipo, str(d['trades']), str(d['contracts']), f"${fmt_int(int(d['importe']))}", f"{sign}{fmt(d['pnl'])}", f"{pct:+.1f}%"])
t = Table(tbl, colWidths=[2.5*cm, 1.8*cm, 1.8*cm, 2.8*cm, 2.8*cm, 2*cm])
s = create_table_style()
for i, row in enumerate(tbl[1:], start=1):
    color = GREEN if row[4].startswith('+') else RED
    s.add('TEXTCOLOR', (4, i), (5, i), color)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 7.3 Por Mes
elements.append(Paragraph("7.3 Por Mes", section_style))
tbl = [['Mes', 'Trades', 'W/L', 'Win%', 'P&L USD']]
for mes in ['Enero', 'Febrero']:
    if mes in by_mes:
        d = by_mes[mes]
        wr_mes = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        sign = '+' if d['pnl'] >= 0 else ''
        tbl.append([mes, str(d['trades']), f"{d['wins']}/{d['losses']}", f"{wr_mes:.0f}%", f"{sign}{fmt(d['pnl'])}"])
t = Table(tbl, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
s = create_table_style()
for i, row in enumerate(tbl[1:], start=1):
    s.add('TEXTCOLOR', (4, i), (4, i), GREEN if row[4].startswith('+') else RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 7.4 Por Dia
elements.append(Paragraph("7.4 Por Dia de la Semana", section_style))
tbl = [['Dia', 'Trades', 'W/L', 'Win%', 'P&L USD']]
for dia in ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']:
    if dia in by_dia:
        d = by_dia[dia]
        wr_dia = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        sign = '+' if d['pnl'] >= 0 else ''
        tbl.append([dia, str(d['trades']), f"{d['wins']}/{d['losses']}", f"{wr_dia:.0f}%", f"{sign}{fmt(d['pnl'])}"])
t = Table(tbl, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
s = create_table_style()
for i, row in enumerate(tbl[1:], start=1):
    s.add('TEXTCOLOR', (4, i), (4, i), GREEN if row[4].startswith('+') else RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 7.5 Por Franja Horaria
elements.append(Paragraph("7.5 Por Franja Horaria", section_style))
tbl = [
    ['Franja Horaria', 'Trades', 'W/L', 'Win%', 'P&L USD'],
    ['00:01-08:00 (Asia)', '14', '9/5', '64%', '+8.710,84'],
    ['08:01-15:00 (EU)', '10', '1/9', '10%', '-4.862,18'],
    ['15:01-23:59 (US)', '12', '7/5', '58%', '+16.441,74'],
]
t = Table(tbl, colWidths=[4*cm, 2*cm, 2*cm, 2*cm, 3.5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (4, 1), (4, 1), GREEN)
s.add('TEXTCOLOR', (4, 2), (4, 2), RED)
s.add('TEXTCOLOR', (4, 3), (4, 3), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 7.6 Por Posicion
elements.append(Paragraph("7.6 Por Tipo de Posicion", section_style))
tbl = [
    ['Posicion', 'Trades', 'Contratos', 'Win%', 'P&L USD'],
    ['LONG', '21', '31', '62%', '+12.439,26'],
    ['SHORT', '15', '71', '27%', '+7.851,14'],
]
t = Table(tbl, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (4, 1), (4, 2), GREEN)
s.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda'))
s.add('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#f8d7da'))
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 7.7 Por Duracion
elements.append(Paragraph("7.7 Por Duracion del Trade", section_style))
tbl = [
    ['Duracion', 'Trades', 'Win%', 'P&L USD', 'Nota'],
    ['< 2 horas', '23', '57%', '-10.029,56', 'Stops ejecutados'],
    ['2-6 horas', '59', '54%', '+21.342,54', 'Rango optimo'],
    ['6+ horas', '20', '30%', '+8.977,42', 'Swing trading'],
]
t = Table(tbl, colWidths=[2.5*cm, 2*cm, 2*cm, 3*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (3, 1), (3, 1), RED)
s.add('TEXTCOLOR', (3, 2), (3, 3), GREEN)
s.add('BACKGROUND', (4, 2), (4, 2), colors.HexColor('#d4edda'))
t.setStyle(s)
elements.append(t)

elements.append(PageBreak())

# 7.8 Top 5 Mejores y Peores
elements.append(Paragraph("7.8 Top 5 Mejores y Peores Trades", section_style))

elements.append(Paragraph("Top 5 Mejores:", normal_style))
tbl = [
    ['#', 'Symbol', 'Fecha', 'Duracion', 'P&L USD'],
    ['1', 'ESH6', '10-12/02', '2.3 dias', '+6.370,50'],
    ['2', 'GCJ6', '27/01', '2.9 hrs', '+5.515,06'],
    ['3', 'GCH6', '20-21/01', '3.2 hrs', '+4.445,06'],
    ['4', 'GCH6', '20-21/01', '3.2 hrs', '+4.445,06'],
    ['5', 'HEJ6', '10/02', '3.0 hrs', '+4.183,38'],
]
t = Table(tbl, colWidths=[1*cm, 2.5*cm, 3*cm, 2.5*cm, 4*cm])
s = create_table_style()
for i in range(1, 6):
    s.add('TEXTCOLOR', (4, i), (4, i), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.3*cm))

elements.append(Paragraph("Top 5 Peores:", normal_style))
tbl = [
    ['#', 'Symbol', 'Fecha', 'Duracion', 'P&L USD'],
    ['1', 'GCJ6', '11/02', '1.2 hrs', '-5.054,94'],
    ['2', 'CLH6', '11/02', '8.0 hrs', '-4.592,14'],
    ['3', 'GCJ6', '26/01', '9.1 hrs', '-4.054,94'],
    ['4', 'GCH6', '25-26/01', '3.3 hrs', '-1.314,94'],
    ['5', 'GCH6', '25-26/01', '3.3 hrs', '-1.254,94'],
]
t = Table(tbl, colWidths=[1*cm, 2.5*cm, 3*cm, 2.5*cm, 4*cm])
s = create_table_style()
for i in range(1, 6):
    s.add('TEXTCOLOR', (4, i), (4, i), RED)
t.setStyle(s)
elements.append(t)

# =============================================================================
# GENERAR PDF
# =============================================================================
doc.build(elements)
print(f"PDF generado: {pdf_path}")
