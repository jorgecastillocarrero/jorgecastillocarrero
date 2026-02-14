"""
Generador de PDF - Analisis de Futuros IB
Datos dinamicos desde la base de datos (tablas 1-4)
Datos estaticos verificados del informe IB (tablas 5-9)
"""
import sys
sys.path.insert(0, '.')

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
from src.database import get_db_manager
from sqlalchemy import text

# =============================================================================
# OBTENER DATOS DE LA BASE DE DATOS
# =============================================================================
db = get_db_manager()

with db.get_session() as session:
    # EUR/USD dinamico
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

# Procesar datos
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

# Calculos globales
comisiones = -539.60
pnl_neto = total_pnl + comisiones
total_contracts = sum(d.get('contracts', 0) for d in by_tipo.values())
wr = total_wins / total_trades * 100 if total_trades > 0 else 0
avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

print(f"EUR/USD: {eur_usd:.4f}")
print(f"P&L Total: ${total_pnl:,.2f} USD = {total_pnl/eur_usd:,.2f} EUR")

# =============================================================================
# CREAR PDF
# =============================================================================
pdf_path = r'C:\Users\usuario\Downloads\Analisis_Futuros_IB_2026_color.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER, spaceAfter=20)
subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
description_style = ParagraphStyle('Description', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=12, spaceBefore=5)

elements = []

# Titulo
elements.append(Paragraph("ANALISIS DE FUTUROS IB", title_style))
elements.append(Paragraph("Periodo: 01/01/2026 - 13/02/2026", ParagraphStyle('Period', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, spaceAfter=5)))
elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} | EUR/USD: {eur_usd:.4f}", ParagraphStyle('Date', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, spaceAfter=20, textColor=colors.grey)))
elements.append(Spacer(1, 0.5*cm))

def create_table_style():
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),  # Azul oscuro
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f4f8')]),  # Azul claro alterno
    ])

GREEN = colors.HexColor('#006600')
RED = colors.HexColor('#cc0000')

def fmt(n):
    return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_int(n):
    return f"{n:,}".replace(",", ".")

# =============================================================================
# 1. RESULTADO GLOBAL
# =============================================================================
elements.append(Paragraph("1. RESULTADO GLOBAL", subtitle_style))
elements.append(Paragraph(
    "Resumen consolidado del rendimiento total en operaciones de futuros durante el periodo. "
    "Incluye P&L bruto, comisiones de Interactive Brokers, P&L neto, numero total de trades y contratos operados, "
    "win rate y P&L promedio por operacion.",
    description_style
))

sign = '+' if total_pnl >= 0 else ''
sign_neto = '+' if pnl_neto >= 0 else ''
sign_avg = '+' if avg_pnl >= 0 else ''

data_global = [
    ['Metrica', 'Valor'],
    ['P&L Total USD', f'{sign}${fmt(total_pnl)}'],
    ['P&L Total EUR', f'{sign}{fmt(total_pnl/eur_usd)} EUR'],
    ['Comisiones IB', f'${fmt(comisiones)}'],
    ['P&L Neto USD', f'{sign_neto}${fmt(pnl_neto)}'],
    ['Trades', f'{total_trades}'],
    ['Contratos', f'{total_contracts}'],
    ['Win Rate', f'{wr:.0f}%'],
    ['P&L Promedio', f'{sign_avg}${fmt(avg_pnl)} USD/trade'],
]
t = Table(data_global, colWidths=[8*cm, 6*cm])
style = create_table_style()
# Colorear P&L positivos en verde
style.add('TEXTCOLOR', (1, 1), (1, 1), GREEN)  # P&L Total USD
style.add('TEXTCOLOR', (1, 2), (1, 2), GREEN)  # P&L Total EUR
style.add('TEXTCOLOR', (1, 3), (1, 3), RED)    # Comisiones
style.add('TEXTCOLOR', (1, 4), (1, 4), GREEN)  # P&L Neto
style.add('TEXTCOLOR', (1, 8), (1, 8), GREEN)  # P&L Promedio
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 0.8*cm))

# =============================================================================
# 2. POR TIPO DE ACTIVO
# =============================================================================
elements.append(Paragraph("2. POR TIPO DE ACTIVO", subtitle_style))
elements.append(Paragraph(
    "Desglose del rendimiento por categoria de activo subyacente. "
    "Muestra trades, contratos, importe operado, P&L en USD y porcentaje del beneficio total.",
    description_style
))

data_tipo = [['Tipo', 'Trades', 'Contr.', 'Importe', 'P&L USD', '%Total']]
for tipo in ['Oro', 'Indices', 'Ganado', 'Petroleo']:
    if tipo in by_tipo:
        d = by_tipo[tipo]
        pct = d['pnl'] / total_pnl * 100 if total_pnl != 0 else 0
        sign = '+' if d['pnl'] >= 0 else ''
        data_tipo.append([tipo, str(d['trades']), str(d['contracts']), f"${fmt_int(int(d['importe']))}", f"{sign}{fmt(d['pnl'])}", f"{pct:+.1f}%"])

t = Table(data_tipo, colWidths=[2.5*cm, 2*cm, 2*cm, 3*cm, 3*cm, 2.5*cm])
style = create_table_style()
for i, row in enumerate(data_tipo[1:], start=1):
    if row[4].startswith('+'):
        style.add('TEXTCOLOR', (4, i), (4, i), GREEN)
        style.add('TEXTCOLOR', (5, i), (5, i), GREEN)
    elif row[4].startswith('-'):
        style.add('TEXTCOLOR', (4, i), (4, i), RED)
        style.add('TEXTCOLOR', (5, i), (5, i), RED)
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 0.8*cm))

# =============================================================================
# 3. POR MES
# =============================================================================
elements.append(Paragraph("3. POR MES", subtitle_style))
elements.append(Paragraph(
    "Comparativa mensual del rendimiento. Permite identificar tendencias y cambios en las condiciones de mercado.",
    description_style
))

data_mes = [['Mes', 'Trades', 'W/L', 'Win%', 'P&L USD']]
for mes in ['Enero', 'Febrero']:
    if mes in by_mes:
        d = by_mes[mes]
        wr_mes = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        sign = '+' if d['pnl'] >= 0 else ''
        data_mes.append([mes, str(d['trades']), f"{d['wins']}/{d['losses']}", f"{wr_mes:.0f}%", f"{sign}{fmt(d['pnl'])}"])

t = Table(data_mes, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
style = create_table_style()
for i, row in enumerate(data_mes[1:], start=1):
    if row[4].startswith('+'):
        style.add('TEXTCOLOR', (4, i), (4, i), GREEN)
    elif row[4].startswith('-'):
        style.add('TEXTCOLOR', (4, i), (4, i), RED)
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 0.8*cm))

# =============================================================================
# 4. POR DIA DE LA SEMANA
# =============================================================================
elements.append(Paragraph("4. POR DIA DE LA SEMANA", subtitle_style))
elements.append(Paragraph(
    "Analisis del rendimiento segun el dia de la semana. Permite identificar los dias mas y menos favorables.",
    description_style
))

data_dia = [['Dia', 'Trades', 'W/L', 'Win%', 'P&L USD']]
for dia in ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']:
    if dia in by_dia:
        d = by_dia[dia]
        wr_dia = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        sign = '+' if d['pnl'] >= 0 else ''
        data_dia.append([dia, str(d['trades']), f"{d['wins']}/{d['losses']}", f"{wr_dia:.0f}%", f"{sign}{fmt(d['pnl'])}"])

t = Table(data_dia, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
style = create_table_style()
for i, row in enumerate(data_dia[1:], start=1):
    if row[4].startswith('+'):
        style.add('TEXTCOLOR', (4, i), (4, i), GREEN)
    elif row[4].startswith('-'):
        style.add('TEXTCOLOR', (4, i), (4, i), RED)
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 0.8*cm))

# =============================================================================
# 5. POR FRANJA HORARIA (ordenado cronologicamente)
# =============================================================================
elements.append(Paragraph("5. POR FRANJA HORARIA", subtitle_style))
elements.append(Paragraph(
    "Rendimiento por sesion de mercado. Asia (madrugada), EU (manana), US (tarde-noche).",
    description_style
))

data_franja = [
    ['Franja Horaria', 'Trades', 'W/L', 'Win%', 'P&L USD'],
    ['00:01-08:00 (Asia)', '14', '9/5', '64%', '+8.710,84'],
    ['08:01-15:00 (EU)', '10', '1/9', '10%', '-4.862,18'],
    ['15:01-23:59 (US)', '12', '7/5', '58%', '+16.441,74'],
]
t = Table(data_franja, colWidths=[4.5*cm, 2*cm, 2*cm, 2*cm, 4*cm])
style = create_table_style()
style.add('TEXTCOLOR', (4, 1), (4, 1), GREEN)  # Asia +
style.add('TEXTCOLOR', (4, 2), (4, 2), RED)    # EU -
style.add('TEXTCOLOR', (4, 3), (4, 3), GREEN)  # US +
t.setStyle(style)
elements.append(t)

elements.append(PageBreak())

# =============================================================================
# 6. POR TIPO DE POSICION
# =============================================================================
elements.append(Paragraph("6. POR TIPO DE POSICION", subtitle_style))
elements.append(Paragraph(
    "Comparativa entre operaciones en largo (compra) y corto (venta).",
    description_style
))

data_pos = [
    ['Posicion', 'Trades', 'Contratos', 'Win%', 'P&L USD'],
    ['LONG', '21', '31', '62%', '+12.439,26'],
    ['SHORT', '15', '71', '27%', '+7.851,14'],
]
t = Table(data_pos, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
style = create_table_style()
style.add('TEXTCOLOR', (4, 1), (4, 1), GREEN)  # LONG +
style.add('TEXTCOLOR', (4, 2), (4, 2), GREEN)  # SHORT +
style.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda'))  # LONG verde claro
style.add('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#f8d7da'))  # SHORT rojo claro
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 0.8*cm))

# =============================================================================
# 7. POR DURACION
# =============================================================================
elements.append(Paragraph("7. POR DURACION DEL TRADE", subtitle_style))
elements.append(Paragraph(
    "Analisis segun el tiempo de mantenimiento. Los trades < 2 horas son principalmente stops ejecutados "
    "(gestion de riesgo correcta). El rango optimo es 2-6 horas. Los trades 6+ horas son swing trading.",
    description_style
))

data_dur = [
    ['Duracion', 'Trades', 'Win%', 'P&L USD', 'Nota'],
    ['< 2 horas', '23', '57%', '-10.029,56', 'Stops ejecutados'],
    ['2-6 horas', '59', '54%', '+21.342,54', 'Rango optimo'],
    ['6+ horas', '20', '30%', '+8.977,42', 'Swing trading'],
]
t = Table(data_dur, colWidths=[2.5*cm, 2*cm, 2*cm, 3*cm, 4*cm])
style = create_table_style()
style.add('TEXTCOLOR', (3, 1), (3, 1), RED)    # < 2h -
style.add('TEXTCOLOR', (3, 2), (3, 2), GREEN)  # 2-6h +
style.add('TEXTCOLOR', (3, 3), (3, 3), GREEN)  # 6+h +
style.add('BACKGROUND', (4, 2), (4, 2), colors.HexColor('#d4edda'))  # Optimo verde
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 0.8*cm))

# =============================================================================
# 8. RELACION DE TRADES
# =============================================================================
elements.append(Paragraph("8. RELACION DE TRADES POR FECHA", subtitle_style))
elements.append(Paragraph(
    "Listado cronologico de todas las operaciones cerradas durante el periodo.",
    description_style
))

data_trades = [
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
t = Table(data_trades, colWidths=[2*cm, 2.5*cm, 2*cm, 2*cm, 3.5*cm])
style = create_table_style()
for i, row in enumerate(data_trades[1:], start=1):
    if row[4].startswith('+'):
        style.add('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#006600'))
    elif row[4].startswith('-'):
        style.add('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#cc0000'))
t.setStyle(style)
elements.append(t)

elements.append(PageBreak())

# =============================================================================
# 9. TOP 5
# =============================================================================
elements.append(Paragraph("9. TOP 5 MEJORES Y PEORES TRADES", subtitle_style))
elements.append(Paragraph(
    "Ranking de las operaciones con mayor y menor rentabilidad del periodo.",
    description_style
))

elements.append(Paragraph("TOP 5 MEJORES", ParagraphStyle('SubHeader', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=8, spaceBefore=10)))

data_best = [
    ['#', 'Symbol', 'Fecha', 'Duracion', 'P&L USD'],
    ['1', 'ESH6', '10-12/02', '2.3 dias', '+6.370,50'],
    ['2', 'GCJ6', '27/01', '2.9 hrs', '+5.515,06'],
    ['3', 'GCH6', '20-21/01', '3.2 hrs', '+4.445,06'],
    ['4', 'GCH6', '20-21/01', '3.2 hrs', '+4.445,06'],
    ['5', 'HEJ6', '10/02', '3.0 hrs', '+4.183,38'],
]
t = Table(data_best, colWidths=[1*cm, 2.5*cm, 3*cm, 2.5*cm, 4*cm])
style = create_table_style()
for i in range(1, len(data_best)):
    style.add('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#006600'))
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

elements.append(Paragraph("TOP 5 PEORES", ParagraphStyle('SubHeader', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold', spaceAfter=8, spaceBefore=10)))

data_worst = [
    ['#', 'Symbol', 'Fecha', 'Duracion', 'P&L USD'],
    ['1', 'GCJ6', '11/02', '1.2 hrs', '-5.054,94'],
    ['2', 'CLH6', '11/02', '8.0 hrs', '-4.592,14'],
    ['3', 'GCJ6', '26/01', '9.1 hrs', '-4.054,94'],
    ['4', 'GCH6', '25-26/01', '3.3 hrs', '-1.314,94'],
    ['5', 'GCH6', '25-26/01', '3.3 hrs', '-1.254,94'],
]
t = Table(data_worst, colWidths=[1*cm, 2.5*cm, 3*cm, 2.5*cm, 4*cm])
style = create_table_style()
for i in range(1, len(data_worst)):
    style.add('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#cc0000'))
t.setStyle(style)
elements.append(t)
elements.append(Spacer(1, 1*cm))

# =============================================================================
# CONCLUSIONES
# =============================================================================
elements.append(Paragraph("CONCLUSIONES Y RECOMENDACIONES", subtitle_style))

mejor_dia = max(by_dia.items(), key=lambda x: x[1]['pnl'])
peor_dia = min(by_dia.items(), key=lambda x: x[1]['pnl'])

conclusions = f"""
<b>Rendimiento General:</b> Resultado positivo de +${fmt(total_pnl)} USD brutos (+${fmt(pnl_neto)} netos),
equivalente a +{fmt(total_pnl/eur_usd)} EUR, con win rate del {wr:.0f}%.

<b>Activos:</b> El Oro es el activo principal con {by_tipo.get('Oro', {}).get('pnl', 0)/total_pnl*100:.0f}% del beneficio total.

<b>Timing:</b> Mejor dia: {mejor_dia[0]} (+${fmt(mejor_dia[1]['pnl'])}).
Peor dia: {peor_dia[0]} (${fmt(peor_dia[1]['pnl'])}). Sesiones US y Asia rentables; evitar EU.

<b>Duracion:</b> Rango optimo 2-6 horas. Trades < 2h son stops (gestion correcta).

<b>Direccionalidad:</b> LONG mejor win rate (62% vs 27%), ambas rentables.
"""

elements.append(Paragraph(conclusions, ParagraphStyle('Conclusions', parent=styles['Normal'], fontSize=9, leading=14)))

# Generar
doc.build(elements)
print(f"PDF generado: {pdf_path}")
