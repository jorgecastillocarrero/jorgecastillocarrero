"""
Generador de PDF - Cartera
Resumen de posicion global y benchmark
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

    # EUR/USD
    fx = session.execute(text('''
        SELECT ph.close FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X'
        ORDER BY ph.date DESC LIMIT 1
    ''')).scalar() or 1.04

    # Valores cartera
    valor_inicial = session.execute(text('''
        SELECT SUM(total_eur) FROM posicion WHERE fecha = '2025-12-31'
    ''')).scalar() or 3930529

    valor_actual = session.execute(text('''
        SELECT SUM(total_eur) FROM posicion WHERE fecha = :fecha
    '''), {'fecha': fecha_actual}).scalar()

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

    # Holdings por tipo
    holdings_result = session.execute(text('''
        SELECT h.symbol, h.shares, h.asset_type, h.account_code, h.currency,
               COALESCE(
                   (SELECT ph.close FROM price_history ph
                    JOIN symbols s ON ph.symbol_id = s.id
                    WHERE s.code = h.symbol
                    ORDER BY ph.date DESC LIMIT 1),
                   h.precio_entrada
               ) as precio
        FROM holding_diario h
        WHERE h.fecha = :fecha AND h.shares != 0
    '''), {'fecha': fecha_actual}).fetchall()

    holdings = []
    for symbol, shares, asset_type, account, currency, precio in holdings_result:
        if precio and shares:
            valor_usd = float(shares) * float(precio)
            valor_eur = valor_usd / fx if currency == 'USD' else valor_usd
            holdings.append({
                'symbol': symbol,
                'asset_type': asset_type or 'Sin tipo',
                'account': account,
                'valor_eur': valor_eur
            })

    total_holdings = sum(h['valor_eur'] for h in holdings)

    # Por tipo de activo
    by_type = {}
    for h in holdings:
        t = h['asset_type']
        by_type[t] = by_type.get(t, 0) + h['valor_eur']

    # Por cuenta
    by_account = {}
    for h in holdings:
        a = h['account']
        by_account[a] = by_account.get(a, 0) + h['valor_eur']

    # Top 10 posiciones
    top_positions = sorted(holdings, key=lambda x: -x['valor_eur'])[:10]

    # Datos para gráfico
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

# Generar gráfico
cartera_fechas = [to_date(r[0]) for r in cartera_data]
cartera_valores = [float(r[1]) for r in cartera_data]
cartera_ini = cartera_valores[0]
cartera_rent = [(v / cartera_ini - 1) * 100 for v in cartera_valores]

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
ax.plot(cartera_fechas, cartera_rent, label=f'Cartera (+{cartera_rent[-1]:.2f}%)', color='#1a365d', linewidth=2.5)
ax.plot(spy_fechas, spy_rent_chart, label=f'SPY ({spy_rent_chart[-1]:+.2f}%)', color='#e74c3c', linewidth=2, linestyle='--')
ax.plot(qqq_fechas, qqq_rent_chart, label=f'QQQ ({qqq_rent_chart[-1]:+.2f}%)', color='#f39c12', linewidth=2, linestyle='--')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
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
normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, spaceAfter=6)
description_style = ParagraphStyle('Description', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=8)

elements = []

# Titulo
elements.append(Paragraph("CARTERA - RESUMEN", title_style))
elements.append(Paragraph(f"Fecha: {fecha_actual.strftime('%d/%m/%Y')}", ParagraphStyle('Sub', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, textColor=colors.grey)))
elements.append(Spacer(1, 0.5*cm))

# 1. Resumen de Cartera
elements.append(Paragraph("1. Resumen de Cartera", subtitle_style))
cartera_tbl = [
    ['Metrica', 'EUR', 'USD'],
    ['Valor Inicial 31/12/2025', f'{fmt(valor_inicial)} EUR', f'${fmt(valor_inicial * 1.1747)}'],
    ['Valor Actual', f'{fmt(valor_actual)} EUR', f'${fmt(valor_actual * fx)}'],
    ['Ganancia 2026', f'+{fmt(ganancia_eur)} EUR (+{rent_pct:.2f}%)', f'+${fmt(ganancia_eur * fx)} (+{rent_pct:.2f}%)'],
]
t = Table(cartera_tbl, colWidths=[5.5*cm, 4.5*cm, 4.5*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 3), (2, 3), GREEN)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 2. Benchmark
elements.append(Paragraph("2. Benchmark 2026", subtitle_style))
bench_tbl = [
    ['Indice', 'Rentabilidad'],
    ['Cartera', f'+{rent_pct:.2f}%'],
    ['SPY', f'{spy_rent:+.2f}%'],
    ['QQQ', f'{qqq_rent:+.2f}%'],
    ['Alpha vs SPY', f'+{rent_pct - spy_rent:.2f}%'],
    ['Alpha vs QQQ', f'+{rent_pct - qqq_rent:.2f}%'],
]
t = Table(bench_tbl, colWidths=[5*cm, 4*cm])
s = create_table_style()
s.add('TEXTCOLOR', (1, 1), (1, 1), GREEN)
s.add('TEXTCOLOR', (1, 2), (1, 3), RED if spy_rent < 0 else GREEN)
s.add('TEXTCOLOR', (1, 4), (1, 5), GREEN)
s.add('FONTNAME', (0, 4), (1, 5), 'Helvetica-Bold')
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 3. Gráfico
elements.append(Paragraph("3. Rentabilidad vs Benchmark", subtitle_style))
if os.path.exists(chart_path):
    chart_img = Image(chart_path, width=16*cm, height=9*cm)
    elements.append(chart_img)
elements.append(Spacer(1, 0.5*cm))

# 4. Por Tipo de Activo
elements.append(Paragraph("4. Composicion por Tipo de Activo", subtitle_style))
tipo_tbl = [['Tipo', 'Valor EUR', '%']]
for tipo, valor in sorted(by_type.items(), key=lambda x: -x[1]):
    pct = valor / total_holdings * 100
    tipo_tbl.append([tipo, fmt(valor), f'{pct:.1f}%'])
tipo_tbl.append(['TOTAL', fmt(total_holdings), '100%'])
t = Table(tipo_tbl, colWidths=[4*cm, 4*cm, 2.5*cm])
s = create_table_style()
s.add('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold')
s.add('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#333333'))
s.add('TEXTCOLOR', (0, -1), (-1, -1), colors.white)
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 5. Por Cuenta
elements.append(Paragraph("5. Composicion por Cuenta", subtitle_style))
acc_tbl = [['Cuenta', 'Valor EUR', '%']]
for acc, valor in sorted(by_account.items(), key=lambda x: -x[1]):
    pct = valor / total_holdings * 100
    acc_tbl.append([acc, fmt(valor), f'{pct:.1f}%'])
t = Table(acc_tbl, colWidths=[4*cm, 4*cm, 2.5*cm])
s = create_table_style()
t.setStyle(s)
elements.append(t)
elements.append(Spacer(1, 0.5*cm))

# 6. Top 10 Posiciones
elements.append(Paragraph("6. Top 10 Posiciones (Diversificacion)", subtitle_style))
div_tbl = [['Simbolo', 'Tipo', 'Valor EUR', '%']]
for h in top_positions:
    pct = h['valor_eur'] / total_holdings * 100
    div_tbl.append([h['symbol'], h['asset_type'], fmt(h['valor_eur']), f'{pct:.1f}%'])
t = Table(div_tbl, colWidths=[2.5*cm, 3.5*cm, 3*cm, 2*cm])
s = create_table_style()
t.setStyle(s)
elements.append(t)

# Footer
elements.append(Spacer(1, 1*cm))
elements.append(Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)))

# Generar
doc.build(elements)
print(f"PDF generado: {pdf_path}")
