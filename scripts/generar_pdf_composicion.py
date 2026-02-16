"""
Generador de PDF - Composicion de Cartera
Punto 3 del documento Carihuela
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
from src.portfolio_data import PortfolioDataService
from sqlalchemy import text
import matplotlib.pyplot as plt

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

# =============================================================================
# OBTENER DATOS
# =============================================================================
db = get_db_manager()
portfolio_service = PortfolioDataService()

with db.get_session() as session:
    # Fecha actual
    comp_fecha = session.execute(text('SELECT MAX(fecha) FROM posicion')).scalar()
    if hasattr(comp_fecha, 'date'):
        comp_fecha = comp_fecha.date()

# Valores por tipo de activo
strategy_values = portfolio_service.get_values_by_asset_type(comp_fecha)

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

# Calcular valores de diversificacion
bolsa = (strategy_values.get('Quant', 0) +
         strategy_values.get('Value', 0) +
         strategy_values.get('Alpha Picks', 0) +
         strategy_values.get('Mensual', 0) +
         strategy_values.get('Stock', 0))
oro = strategy_values.get('Oro/Mineras', 0)
liquidez = strategy_values.get('Cash/ETFs', 0) or (strategy_values.get('Cash', 0) + strategy_values.get('ETFs', 0))

# Estrategias ordenadas
sorted_strategies = sorted([(k, v) for k, v in strategy_values.items() if v > 0], key=lambda x: -x[1])
total_estrategias = sum(v for k, v in sorted_strategies)

# =============================================================================
# GENERAR GRAFICOS
# =============================================================================

# Grafico 1: Diversificacion
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

# Grafico 2: Estrategia
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

# Grafico 3: Cuenta
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

# =============================================================================
# CREAR PDF
# =============================================================================
pdf_path = r'C:\Users\usuario\Downloads\Composicion_Cartera.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, alignment=TA_CENTER, spaceAfter=10, textColor=HEADER_BG)
subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=15, textColor=HEADER_BG)
section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=12, spaceAfter=6, spaceBefore=12)
description_style = ParagraphStyle('Description', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=8)

elements = []

# Titulo
elements.append(Paragraph("COMPOSICION DE CARTERA", title_style))
elements.append(Paragraph(f"Fecha: {comp_fecha.strftime('%d/%m/%Y')} | Total: {fmt(total_cartera)} EUR",
    ParagraphStyle('Sub', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, textColor=colors.grey)))
elements.append(Spacer(1, 0.5*cm))

# 3.1 Diversificacion
elements.append(Paragraph("3.1 Composicion por Diversificacion", subtitle_style))

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

div_img = Image(div_chart_path, width=7*cm, height=7*cm)
layout_div = Table([[div_img, t]], colWidths=[8*cm, 9*cm])
layout_div.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
elements.append(layout_div)
elements.append(Spacer(1, 0.5*cm))

# 3.2 Por Estrategia
elements.append(Paragraph("3.2 Composicion por Estrategia", subtitle_style))

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
elements.append(Paragraph("3.3 Composicion por Cuenta", subtitle_style))

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

# Footer
elements.append(Spacer(1, 1*cm))
elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)))

# Generar
doc.build(elements)
print(f"PDF generado: {pdf_path}")
