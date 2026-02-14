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
db = get_db_manager()

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
    "3. [Pendiente]",
    "4. [Pendiente]",
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
    ['Valor Inicial 31/12/2025', '3.930.529 EUR', '$4.617.308'],
    ['Valor Actual 13/02/2026', '4.196.615 EUR', '$4.981.737'],
    ['Ganancia Acumulada 2026', '+266.086 EUR (+6,77%)', '+$364.429 (+7,89%)'],
]
t = Table(cartera_data, colWidths=[5.5*cm, 4.5*cm, 4.5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 3), (2, 3), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph("Tipos de cambio: EUR/USD 31/12/2025: 1,1747 | EUR/USD actual: 1,1871", description_style))
elements.append(Spacer(1, 0.5*cm))

# 2.2 Benchmark
elements.append(Paragraph("2.2 Benchmark 2026", section_style))
benchmark_data = [
    ['Indice', 'Rentabilidad'],
    ['SPY', '-0,02%'],
    ['QQQ', '-2,02%'],
    ['Alpha vs SPY', '+6,79%'],
    ['Alpha vs QQQ', '+8,79%'],
]
t = Table(benchmark_data, colWidths=[5*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 2), RED)
s.add('TEXTCOLOR', (1, 3), (1, 4), GREEN)
s.add('FONTNAME', (0, 3), (1, 4), 'Helvetica-Bold')
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 2.3 Variacion Diaria por Tipo de Activo
elements.append(Paragraph("2.3 Variacion Diaria por Tipo de Activo", section_style))
elements.append(Paragraph(
    "Comparativa del valor de la cartera por tipo de activo entre dos dias consecutivos.",
    description_style
))
variacion_data = [
    ['Tipo', '12/02', '13/02', 'Diferencia', 'Var %'],
    ['Mensual', '415.581', '416.727', '+1.146', '+0,28%'],
    ['Quant', '1.487.173', '1.477.718', '-9.455', '-0,64%'],
    ['Value', '379.754', '247.932', '-131.822', '-34,71%'],
    ['Alpha Picks', '369.132', '369.040', '-92', '-0,03%'],
    ['Oro/Mineras', '783.802', '808.331', '+24.529', '+3,13%'],
    ['Cash/ETFs', '716.667', '876.867', '+160.199', '+22,35%'],
    ['TOTAL', '4.152.110', '4.196.615', '+44.505', '+1,07%'],
]
t = Table(variacion_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm])
s = create_table_style()
# Color para diferencias positivas/negativas
s.add('TEXTCOLOR', (3, 1), (4, 1), GREEN)  # Mensual +
s.add('TEXTCOLOR', (3, 2), (4, 2), RED)    # Quant -
s.add('TEXTCOLOR', (3, 3), (4, 3), RED)    # Value -
s.add('TEXTCOLOR', (3, 4), (4, 4), RED)    # Alpha Picks -
s.add('TEXTCOLOR', (3, 5), (4, 5), GREEN)  # Oro/Mineras +
s.add('TEXTCOLOR', (3, 6), (4, 6), GREEN)  # Cash/ETFs +
s.add('TEXTCOLOR', (3, 7), (4, 7), GREEN)  # TOTAL +
# Fila TOTAL en negrita
s.add('FONTNAME', (0, 7), (-1, 7), 'Helvetica-Bold')
s.add('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, 7), (2, 7), colors.white)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 2.4 Grafica de Rentabilidad vs Benchmark
elements.append(Paragraph("2.4 Rentabilidad vs Benchmark", section_style))
elements.append(Paragraph(
    "Evolucion de la rentabilidad de la cartera comparada con SPY y QQQ desde 31/12/2025.",
    description_style
))
# Insertar imagen del grafico
chart_path = r'C:\Users\usuario\Downloads\rentabilidad_benchmark.png'
try:
    chart_img = Image(chart_path, width=16*cm, height=10*cm)
    elements.append(chart_img)
except:
    elements.append(Paragraph("[Grafico no disponible - ejecutar generador de grafico primero]", normal_style))
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    "La cartera supera significativamente a los indices de referencia con +6,77% vs SPY (-0,02%) y QQQ (-2,02%).",
    description_style
))

elements.append(PageBreak())

# =============================================================================
# 6. ETFs
# =============================================================================
elements.append(Paragraph("6. ETFs", subtitle_style))
elements.append(Paragraph(
    "Operaciones de ETFs en Interactive Brokers. Periodo: 01/01/2026 - 13/02/2026.",
    description_style
))

# 6.1 Operaciones Cerradas
elements.append(Paragraph("6.1 Operaciones Cerradas", section_style))
etf_data = [
    ['ETF', 'Nombre', 'Tipo', 'Acciones', 'F. Entrada', 'F. Cierre', 'P. Entrada', 'P. Cierre', 'P&L USD'],
    ['TLT', 'iShares 20+ Year Treasury Bond', 'LONG', '8.042', '20-21/01', '30/01-02/02', '$86,87', '$87,08', '+1.291,44'],
    ['AMLP', 'Alerian MLP ETF', 'SHORT', '8.036', '30/01', '02/02', '$49,97', '$49,62', '+2.810,27'],
    ['EMB', 'iShares JPM USD EM Bond', 'SHORT', '8.277', '30/01', '02/02', '$96,64', '$96,16', '+3.863,27'],
    ['EMLC', 'VanEck JPM EM Local Currency', 'SHORT', '30.394', '30/01', '02/02', '$26,30', '$26,22', '+2.333,50'],
    ['XLB', 'Materials Select Sector SPDR', 'SHORT', '4.288', '30/01', '02/02', '$49,28', '$49,67', '-1.726,35'],
    ['TLT', 'iShares 20+ Year Treasury Bond', 'LONG', '17', '06/02', '12/02', '$87,40', '$88,72', '+21,75'],
]
t = Table(etf_data, colWidths=[1.2*cm, 4*cm, 1.3*cm, 1.4*cm, 2*cm, 2.2*cm, 1.6*cm, 1.6*cm, 1.8*cm])
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
    "Analisis de operaciones de futuros en Interactive Brokers. Periodo: 01/01/2026 - 13/02/2026.",
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

elements.append(PageBreak())

# =============================================================================
# 8. ETFs
# =============================================================================
elements.append(Paragraph("8. ETFs", subtitle_style))
elements.append(Paragraph(
    "Operaciones de ETFs en Interactive Brokers. Periodo: 01/01/2026 - 13/02/2026. "
    "Incluye posiciones largas y cortas ejecutadas.",
    description_style
))

# 8.1 Resultado Global ETFs
elements.append(Paragraph("8.1 Resultado Global", section_style))
elements.append(Paragraph(
    "Resumen de rendimiento de todas las operaciones de ETFs cerradas en el periodo.",
    description_style
))

etf_pnl_bruto = 8869.76
etf_comisiones = -242.93
etf_pnl_neto = etf_pnl_bruto + etf_comisiones
etf_trades = 5
etf_wins = 4
etf_wr = etf_wins / etf_trades * 100

tbl = [
    ['Metrica', 'Valor'],
    ['P&L Bruto USD', f'+${fmt(etf_pnl_bruto)}'],
    ['P&L Bruto EUR', f'+{fmt(etf_pnl_bruto/eur_usd)} EUR'],
    ['Comisiones IB', f'${fmt(etf_comisiones)}'],
    ['P&L Neto USD', f'+${fmt(etf_pnl_neto)}'],
    ['Trades Cerrados', str(etf_trades)],
    ['Win Rate', f'{etf_wr:.0f}%'],
    ['P&L Promedio', f'+${fmt(etf_pnl_neto/etf_trades)} USD/trade'],
]
t = Table(tbl, colWidths=[7*cm, 5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 2), GREEN)
s.add('TEXTCOLOR', (1, 3), (1, 3), RED)
s.add('TEXTCOLOR', (1, 4), (1, 5), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 8.2 Por ETF
elements.append(Paragraph("8.2 Por ETF", section_style))
elements.append(Paragraph(
    "Desglose de P&L por cada ETF operado. La mayoria fueron posiciones cortas (SHORT) "
    "apostando a la bajada de bonos y mercados emergentes.",
    description_style
))

tbl = [
    ['ETF', 'Tipo', 'Acciones', 'P. Venta', 'P. Compra', 'P&L USD'],
    ['AMLP', 'SHORT', '500', '$48,83', '$43,21', '+2.812,60'],
    ['EMB', 'SHORT', '500', '$83,17', '$75,23', '+3.972,96'],
    ['EMLC', 'SHORT', '1.000', '$21,50', '$19,07', '+2.431,52'],
    ['TLT', 'LONG', '200', '$93,62', '$86,78', '+1.367,88'],
    ['XLB', 'SHORT', '200', '$82,69', '$91,27', '-1.715,20'],
]
t = Table(tbl, colWidths=[1.8*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm, 2.8*cm])
s = create_table_style()
# Color SHORT cells
s.add('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#f8d7da'))
s.add('BACKGROUND', (1, 2), (1, 2), colors.HexColor('#f8d7da'))
s.add('BACKGROUND', (1, 3), (1, 3), colors.HexColor('#f8d7da'))
s.add('BACKGROUND', (1, 4), (1, 4), colors.HexColor('#d4edda'))
s.add('BACKGROUND', (1, 5), (1, 5), colors.HexColor('#f8d7da'))
# Color P&L
s.add('TEXTCOLOR', (5, 1), (5, 4), GREEN)
s.add('TEXTCOLOR', (5, 5), (5, 5), RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 8.3 Por Tipo de Posicion
elements.append(Paragraph("8.3 Por Tipo de Posicion", section_style))
elements.append(Paragraph(
    "Comparativa entre posiciones largas (apuesta alcista) y cortas (apuesta bajista).",
    description_style
))

tbl = [
    ['Posicion', 'Trades', 'Acciones', 'Win%', 'P&L USD'],
    ['LONG', '1', '200', '100%', '+1.367,88'],
    ['SHORT', '4', '2.200', '75%', '+7.501,88'],
]
t = Table(tbl, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (4, 1), (4, 2), GREEN)
s.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda'))
s.add('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#f8d7da'))
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 8.4 Por Sector
elements.append(Paragraph("8.4 Por Sector/Categoria", section_style))
elements.append(Paragraph(
    "Clasificacion de ETFs por tipo de activo subyacente.",
    description_style
))

tbl = [
    ['Sector', 'ETFs', 'P&L USD', 'Comentario'],
    ['Bonos Emergentes', 'EMB, EMLC', '+6.404,48', 'Short en deuda EM - exitoso'],
    ['Bonos USA', 'TLT', '+1.367,88', 'Long en treasuries - exitoso'],
    ['Energia', 'AMLP', '+2.812,60', 'Short MLPs - exitoso'],
    ['Materiales', 'XLB', '-1.715,20', 'Short fallido - loss'],
]
t = Table(tbl, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (2, 1), (2, 3), GREEN)
s.add('TEXTCOLOR', (2, 4), (2, 4), RED)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 8.5 Posiciones Abiertas
elements.append(Paragraph("8.5 Posiciones Abiertas Actuales", section_style))
elements.append(Paragraph(
    "ETFs actualmente en cartera con posiciones abiertas.",
    description_style
))

tbl = [
    ['ETF', 'Tipo', 'Acciones', 'P. Entrada', 'Valor USD', 'P&L Latente'],
    ['AMLP', 'SHORT', '500', '$48,83', '$24.415', 'Pendiente'],
    ['EMB', 'SHORT', '1.500', '$82,50', '$123.750', 'Pendiente'],
    ['EMLC', 'SHORT', '2.000', '$21,20', '$42.400', 'Pendiente'],
    ['TLT', 'LONG', '2.700', '$87,50', '$236.250', 'Pendiente'],
    ['XLB', 'SHORT', '2.000', '$87,00', '$174.000', 'Pendiente'],
]
t = Table(tbl, colWidths=[1.8*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm])
s = create_table_style()
s.add('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#f8d7da'))
s.add('BACKGROUND', (1, 2), (1, 2), colors.HexColor('#f8d7da'))
s.add('BACKGROUND', (1, 3), (1, 3), colors.HexColor('#f8d7da'))
s.add('BACKGROUND', (1, 4), (1, 4), colors.HexColor('#d4edda'))
s.add('BACKGROUND', (1, 5), (1, 5), colors.HexColor('#f8d7da'))
t.setStyle(s)
elements.append(t)

# =============================================================================
# GENERAR PDF
# =============================================================================
doc.build(elements)
print(f"PDF generado: {pdf_path}")
