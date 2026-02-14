"""
Generador de PDF - Estadisticos ETFs
Interactive Brokers (01/01 - 13/02/2026)
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

# Configuracion visual
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
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ROW_ALT]),
    ])

def fmt(n):
    return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Crear PDF
pdf_path = r'C:\Users\usuario\Downloads\ETFs_Estadisticos_v2.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=1*cm, leftMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER, spaceAfter=10, textColor=HEADER_BG)
subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=15, textColor=HEADER_BG)
normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, spaceAfter=6)

elements = []

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
s = create_table_style()
# Color LONG/SHORT
s.add('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#d4edda'))  # TLT LONG
s.add('BACKGROUND', (2, 2), (2, 5), colors.HexColor('#f8d7da'))  # SHORT
s.add('BACKGROUND', (2, 6), (2, 6), colors.HexColor('#d4edda'))  # TLT LONG
# Color P&L
s.add('TEXTCOLOR', (8, 1), (8, 4), GREEN)
s.add('TEXTCOLOR', (8, 5), (8, 5), RED)
s.add('TEXTCOLOR', (8, 6), (8, 6), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# Resumen
elements.append(Paragraph("RESUMEN OPERACIONES CERRADAS", subtitle_style))

resumen_data = [
    ['Metrica', 'Valor'],
    ['P&L Bruto', '+$8.593,88'],
    ['Trades', '6'],
    ['Wins / Losses', '5 / 1'],
    ['Win Rate', '83%'],
]
t = Table(resumen_data, colWidths=[6*cm, 4*cm])
s = create_table_style()
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
t = Table(open_data, colWidths=[1.3*cm, 5.5*cm, 1.5*cm, 1.8*cm, 2*cm, 2*cm, 2*cm])
s = create_table_style()
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
s = create_table_style()
s.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#d4edda'))
s.add('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#f8d7da'))
s.add('TEXTCOLOR', (2, 1), (2, 2), GREEN)
t.setStyle(s)
elements.append(t)

# Footer
elements.append(Spacer(1, 1*cm))
elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)))

# Generar
doc.build(elements)
print(f"PDF generado: {pdf_path}")
