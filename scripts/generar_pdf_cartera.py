"""
Generador de PDF - Cartera Resumen
Secciones 2.1 a 2.5 del documento Carihuela
"""
import sys
sys.path.insert(0, '.')

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime, date
from src.database import get_db_manager
from sqlalchemy import text
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# =============================================================================
# CONFIGURACION VISUAL
# =============================================================================
GREEN = colors.HexColor('#006600')
RED = colors.HexColor('#cc0000')
HEADER_BG = colors.HexColor('#1a365d')
ROW_ALT = colors.HexColor('#e8f4f8')

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
    return f"{n:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def to_date(d):
    if isinstance(d, datetime):
        return d.date()
    return d

# =============================================================================
# OBTENER DATOS
# =============================================================================
db = get_db_manager()

with db.get_session() as session:
    # Fecha más reciente
    fecha_actual = to_date(session.execute(text('SELECT MAX(fecha) FROM posicion')).scalar())

    # Fecha anterior
    fecha_anterior = to_date(session.execute(text('''
        SELECT MAX(fecha) FROM posicion WHERE fecha < :fecha
    '''), {'fecha': fecha_actual}).scalar())

    # EUR/USD
    fx = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X'
        ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 1.04

    fx_inicial = 1.1747  # EUR/USD 31/12/2025

    # Valores cartera
    valor_inicial = session.execute(text('''
        SELECT SUM(total_eur) FROM posicion WHERE fecha = '2025-12-31'
    ''')).scalar() or 3930529

    valor_actual = session.execute(text('''
        SELECT SUM(total_eur) FROM posicion WHERE fecha = :fecha
    '''), {'fecha': fecha_actual}).scalar()

    valor_anterior = session.execute(text('''
        SELECT SUM(total_eur) FROM posicion WHERE fecha = :fecha
    '''), {'fecha': fecha_anterior}).scalar()

    ganancia_eur = valor_actual - valor_inicial
    rent_pct = (valor_actual / valor_inicial - 1) * 100

    # Benchmark
    spy_ini = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY' AND ph.date = '2025-12-31'
    ''')).scalar() or 681.92

    spy_act = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY'
        ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()

    qqq_ini = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ' AND ph.date = '2025-12-31'
    ''')).scalar() or 614.31

    qqq_act = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ'
        ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()

    spy_rent = (spy_act / spy_ini - 1) * 100
    qqq_rent = (qqq_act / qqq_ini - 1) * 100

    # Rentabilidad mensual
    valor_enero = session.execute(text('''
        SELECT SUM(total_eur) FROM posicion WHERE fecha = '2026-01-31'
    ''')).scalar()
    if not valor_enero:
        valor_enero = session.execute(text('''
            SELECT SUM(total_eur) FROM posicion
            WHERE fecha = (SELECT MAX(fecha) FROM posicion WHERE fecha <= '2026-01-31')
        ''')).scalar()

    spy_enero = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY' AND ph.date <= '2026-01-31'
        ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()

    qqq_enero = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ' AND ph.date <= '2026-01-31'
        ORDER BY ph.date DESC LIMIT 1
    ''')).scalar()

    # Datos para gráfico (excluyendo 01/01 con datos incorrectos)
    cartera_data = session.execute(text('''
        SELECT fecha, SUM(total_eur) as total
        FROM posicion
        WHERE fecha >= '2025-12-31' AND fecha != '2026-01-01'
        GROUP BY fecha
        ORDER BY fecha
    ''')).fetchall()

    spy_data = session.execute(text('''
        SELECT ph.date, ph.close
        FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'SPY' AND ph.date >= '2025-12-31'
        ORDER BY ph.date
    ''')).fetchall()

    qqq_data = session.execute(text('''
        SELECT ph.date, ph.close
        FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'QQQ' AND ph.date >= '2025-12-31'
        ORDER BY ph.date
    ''')).fetchall()

# Calcular rentabilidades mensuales
rent_ene = (valor_enero / valor_inicial - 1) * 100
rent_feb = (valor_actual / valor_enero - 1) * 100
spy_ene = (spy_enero / spy_ini - 1) * 100
spy_feb = (spy_act / spy_enero - 1) * 100
qqq_ene = (qqq_enero / qqq_ini - 1) * 100
qqq_feb = (qqq_act / qqq_enero - 1) * 100

# Generar gráfico
cartera_fechas = [to_date(r[0]) for r in cartera_data]
cartera_valores = [float(r[1]) for r in cartera_data]
cartera_ini = cartera_valores[0]
cartera_rent_chart = [(v / cartera_ini - 1) * 100 for v in cartera_valores]

spy_fechas = [to_date(r[0]) for r in spy_data]
spy_precios = [float(r[1]) for r in spy_data]
spy_ini_chart = spy_precios[0]
spy_rent_chart = [(p / spy_ini_chart - 1) * 100 for p in spy_precios]

qqq_fechas = [to_date(r[0]) for r in qqq_data]
qqq_precios = [float(r[1]) for r in qqq_data]
qqq_ini_chart = qqq_precios[0]
qqq_rent_chart = [(p / qqq_ini_chart - 1) * 100 for p in qqq_precios]

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

chart_path = r'C:\Users\usuario\Downloads\cartera_benchmark_chart.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# CREAR PDF
# =============================================================================
pdf_path = r'C:\Users\usuario\Downloads\Cartera_Resumen.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, alignment=TA_CENTER, spaceAfter=10, textColor=HEADER_BG)
subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=15, textColor=HEADER_BG)
section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=12, spaceAfter=6, spaceBefore=12)
normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, spaceAfter=6)
description_style = ParagraphStyle('Description', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=8)

elements = []

# Titulo
elements.append(Paragraph("CARTERA - RESUMEN", title_style))
elements.append(Paragraph(f"Periodo: 01/01/2026 - {fecha_actual.strftime('%d/%m/%Y')}", ParagraphStyle('Sub', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, textColor=colors.grey)))
elements.append(Spacer(1, 0.5*cm))

# 2.1 Resumen de Cartera
elements.append(Paragraph("2.1 Resumen de Cartera", subtitle_style))
cartera_tbl = [
    ['Metrica', 'EUR', 'USD'],
    ['Valor Inicial 31/12/2025', f'{fmt(valor_inicial)} EUR', f'${fmt(valor_inicial * fx_inicial)}'],
    ['Valor Actual ' + fecha_actual.strftime('%d/%m/%Y'), f'{fmt(valor_actual)} EUR', f'${fmt(valor_actual * fx)}'],
    ['Ganancia Acumulada 2026', f'+{fmt(ganancia_eur)} EUR (+{rent_pct:.2f}%)', f'+${fmt(ganancia_eur * fx)} (+{rent_pct:.2f}%)'],
]
t = Table(cartera_tbl, colWidths=[5.5*cm, 4.5*cm, 4.5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 3), (2, 3), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(f"Tipos de cambio: EUR/USD 31/12/2025: {fx_inicial:.4f} | EUR/USD actual: {fx:.4f}", description_style))
elements.append(Spacer(1, 0.5*cm))

# 2.2 Benchmark
elements.append(Paragraph("2.2 Benchmark 2026", subtitle_style))
bench_tbl = [
    ['Indice', 'Rentabilidad'],
    ['SPY', f'{spy_rent:+.2f}%'],
    ['QQQ', f'{qqq_rent:+.2f}%'],
    ['Alpha vs SPY', f'+{rent_pct - spy_rent:.2f}%'],
    ['Alpha vs QQQ', f'+{rent_pct - qqq_rent:.2f}%'],
]
t = Table(bench_tbl, colWidths=[5*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 1), RED if spy_rent < 0 else GREEN)
s.add('TEXTCOLOR', (1, 2), (1, 2), RED if qqq_rent < 0 else GREEN)
s.add('TEXTCOLOR', (1, 3), (1, 4), GREEN)
s.add('FONTNAME', (0, 3), (1, 4), 'Helvetica-Bold')
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 2.3 Variación Diaria por Tipo de Activo
elements.append(Paragraph("2.3 Variacion Diaria por Tipo de Activo", subtitle_style))
elements.append(Paragraph(
    f"Comparativa del valor de la cartera por tipo de activo entre {fecha_anterior.strftime('%d/%m')} y {fecha_actual.strftime('%d/%m')}.",
    description_style
))
# Calcular variación dinámica por tipo de activo
from src.variacion_diaria_tipo_activo import VariacionDiariaTipoActivo
variacion_calc = VariacionDiariaTipoActivo()
variacion_results = variacion_calc.calculate_variacion_diaria(str(fecha_actual))

variacion_data = [['Tipo', fecha_anterior.strftime('%d/%m'), fecha_actual.strftime('%d/%m'), 'Diferencia', 'Var %']]
tipo_order = ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/ETFs']

# Combine Cash + ETFs into Cash/ETFs
by_strat = dict(variacion_results.get('by_strategy', {}))
cash_data = by_strat.pop('Cash', {})
etfs_data = by_strat.pop('ETFs', {})
cash_val_ant = cash_data.get('anterior', 0) + etfs_data.get('anterior', 0)
cash_val_act = cash_data.get('actual', 0) + etfs_data.get('actual', 0)
cash_diff = cash_val_act - cash_val_ant
cash_pct = (cash_diff / cash_val_ant * 100) if cash_val_ant > 0 else 0
by_strat['Cash/ETFs'] = {'actual': cash_val_act, 'anterior': cash_val_ant, 'diff': cash_diff, 'pct': cash_pct}

var_color_rules = []
row_idx = 1
for tipo in tipo_order:
    data = by_strat.get(tipo)
    if not data:
        continue
    diff = data['diff']
    pct = data['pct']
    variacion_data.append([
        tipo, fmt(data['anterior']), fmt(data['actual']),
        f'{diff:+,.0f}'.replace(',', '.'), f'{pct:+.2f}%'.replace('.', ',')
    ])
    color = GREEN if diff >= 0 else RED
    var_color_rules.append(('TEXTCOLOR', (3, row_idx), (4, row_idx), color))
    row_idx += 1

# TOTAL row
total_diff = valor_actual - valor_anterior
total_pct = (total_diff / valor_anterior * 100) if valor_anterior > 0 else 0
variacion_data.append([
    'TOTAL', fmt(valor_anterior), fmt(valor_actual),
    f'{total_diff:+,.0f}'.replace(',', '.'), f'{total_pct:+.2f}%'.replace('.', ',')
])
total_color = GREEN if total_diff >= 0 else RED
var_color_rules.append(('TEXTCOLOR', (3, row_idx), (4, row_idx), total_color))

t = Table(variacion_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm])
s = create_table_style()
for rule in var_color_rules:
    s.add(*rule)
s.add('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold')
s.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#d9e2ec'))
s.add('TEXTCOLOR', (0, row_idx), (2, row_idx), colors.HexColor('#1a365d'))
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 2.4 Gráfico
elements.append(Paragraph("2.4 Rentabilidad vs Benchmark", subtitle_style))
elements.append(Paragraph(
    "Evolucion de la rentabilidad de la cartera comparada con SPY y QQQ desde 31/12/2025.",
    description_style
))
if os.path.exists(chart_path):
    chart_img = Image(chart_path, width=16*cm, height=9*cm)
    elements.append(chart_img)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    f"La cartera supera significativamente a los indices de referencia con +{rent_pct:.2f}% vs SPY ({spy_rent:+.2f}%) y QQQ ({qqq_rent:+.2f}%).",
    description_style
))
elements.append(Spacer(1, 0.5*cm))

# 2.5 Rentabilidad Mensual
elements.append(Paragraph("2.5 Rentabilidad Mensual vs Benchmark", subtitle_style))
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
# Cartera (fila 1)
s.add('TEXTCOLOR', (1, 1), (2, 1), GREEN)
s.add('TEXTCOLOR', (6, 1), (6, 1), GREEN)
# SPY (fila 2)
s.add('TEXTCOLOR', (1, 2), (1, 2), GREEN if spy_ene >= 0 else RED)
s.add('TEXTCOLOR', (2, 2), (2, 2), GREEN if spy_feb >= 0 else RED)
s.add('TEXTCOLOR', (6, 2), (6, 2), GREEN if spy_rent >= 0 else RED)
# QQQ (fila 3)
s.add('TEXTCOLOR', (1, 3), (1, 3), GREEN if qqq_ene >= 0 else RED)
s.add('TEXTCOLOR', (2, 3), (2, 3), GREEN if qqq_feb >= 0 else RED)
s.add('TEXTCOLOR', (6, 3), (6, 3), GREEN if qqq_rent >= 0 else RED)
# Columna Total
s.add('FONTNAME', (6, 0), (6, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (6, 0), (6, 0), colors.HexColor('#333333'))
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    f"Alpha vs SPY: +{rent_pct - spy_rent:.2f}% | Alpha vs QQQ: +{rent_pct - qqq_rent:.2f}%",
    ParagraphStyle('AlphaNote', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=GREEN)
))

# Footer
elements.append(Spacer(1, 1*cm))
elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)))

# Generar
doc.build(elements)
print(f"PDF generado: {pdf_path}")
