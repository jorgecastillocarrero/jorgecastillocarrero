import os
import sys
sys.path.insert(0, '.')

from src.database import get_db_manager
from sqlalchemy import text
from datetime import date, datetime
import pandas as pd

db = get_db_manager()

# Get EUR/USD rate
with db.get_session() as session:
    fx = session.execute(text("""
        SELECT ph.close
        FROM price_history ph
        JOIN symbols s ON ph.symbol_id = s.id
        WHERE s.code = 'EURUSD=X'
        ORDER BY ph.date DESC LIMIT 1
    """)).fetchone()
    eur_usd = fx[0] if fx else 1.04

# POSICIONES ABIERTAS
with db.get_session() as session:
    result = session.execute(text("""
        WITH latest_holdings AS (
            SELECT DISTINCT ON (symbol, account_code)
                symbol, account_code, shares, precio_entrada, currency, fecha_compra as fc_holding
            FROM holding_diario
            WHERE fecha = (SELECT MAX(fecha) FROM holding_diario)
            ORDER BY symbol, account_code, fecha DESC
        ),
        compras_info AS (
            SELECT symbol,
                   AVG(precio) as precio_compra,
                   MIN(fecha) as fecha_compra
            FROM compras
            GROUP BY symbol
        ),
        latest_prices AS (
            SELECT DISTINCT ON (s.code)
                s.code as symbol, ph.close as precio_actual
            FROM price_history ph
            JOIN symbols s ON ph.symbol_id = s.id
            ORDER BY s.code, ph.date DESC
        )
        SELECT h.symbol, h.account_code, h.shares,
               COALESCE(p.precio_actual, h.precio_entrada) as precio_actual,
               h.currency,
               COALESCE(c.precio_compra, h.precio_entrada) as precio_compra,
               COALESCE(c.fecha_compra, h.fc_holding) as fecha_compra
        FROM latest_holdings h
        LEFT JOIN compras_info c ON h.symbol = c.symbol
        LEFT JOIN latest_prices p ON h.symbol = p.symbol
        WHERE h.shares > 0
        AND h.symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
        AND h.symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
        ORDER BY fecha_compra
    """))
    open_positions = result.fetchall()

open_data = []
for row in open_positions:
    symbol, cuenta, shares, precio_actual, currency, precio_compra, fecha_compra = row

    if not precio_compra or not fecha_compra or not precio_actual:
        continue

    if isinstance(fecha_compra, datetime):
        fecha_compra = fecha_compra.date()

    dias = (date.today() - fecha_compra).days
    anos = dias / 365.25

    rent_hist = ((precio_actual / precio_compra) - 1) * 100
    pnl_local = (precio_actual - precio_compra) * shares

    if anos > 0.08:
        cagr = ((precio_actual / precio_compra) ** (1/anos) - 1) * 100
    else:
        cagr = None

    if currency == 'USD':
        pnl_eur = pnl_local / eur_usd
    elif currency == 'GBP':
        pnl_eur = pnl_local * 1.18
    elif currency == 'CAD':
        pnl_eur = pnl_local * 0.67
    elif currency == 'CHF':
        pnl_eur = pnl_local * 0.95
    else:
        pnl_eur = pnl_local

    open_data.append({
        'symbol': symbol,
        'fecha_compra': fecha_compra,
        'dias': dias,
        'rent_hist': rent_hist,
        'cagr': cagr,
        'pnl_eur': pnl_eur
    })

df_open = pd.DataFrame(open_data)

# POSICIONES CERRADAS
with db.get_session() as session:
    result = session.execute(text("""
        SELECT v.fecha as fecha_venta, v.symbol, v.account_code, v.shares,
               v.precio_venta, v.importe_total, v.currency,
               c.precio_compra, c.fecha_compra
        FROM (
            SELECT fecha, symbol, account_code,
                   SUM(shares) as shares,
                   SUM(importe_total) / SUM(shares) as precio_venta,
                   SUM(importe_total) as importe_total,
                   currency
            FROM ventas
            WHERE symbol NOT IN ('TLT', 'EMB', 'GLD', 'SLV', 'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VOO')
            AND symbol NOT SIMILAR TO '%[FGHJKMNQUVXZ][0-9]'
            GROUP BY fecha, symbol, account_code, currency
        ) v
        LEFT JOIN (
            SELECT symbol, AVG(precio) as precio_compra, MIN(fecha) as fecha_compra
            FROM compras
            GROUP BY symbol
        ) c ON v.symbol = c.symbol
        ORDER BY v.fecha
    """))
    closed_positions = result.fetchall()

closed_data = []
for row in closed_positions:
    fecha_venta, symbol, cuenta, shares, precio_venta, importe_total, currency, precio_compra, fecha_compra = row

    if not precio_compra or not precio_venta:
        continue

    if isinstance(fecha_compra, datetime):
        fecha_compra = fecha_compra.date()
    if isinstance(fecha_venta, datetime):
        fecha_venta = fecha_venta.date()

    dias = (fecha_venta - fecha_compra).days
    if dias < 1:
        dias = 1
    anos = dias / 365.25

    rent_hist = ((precio_venta / precio_compra) - 1) * 100
    pnl_local = (precio_venta - precio_compra) * shares

    if anos > 0.08:
        cagr = ((precio_venta / precio_compra) ** (1/anos) - 1) * 100
    else:
        cagr = None

    if currency == 'USD':
        pnl_eur = pnl_local / eur_usd
    elif currency == 'GBP':
        pnl_eur = pnl_local * 1.18
    elif currency == 'CAD':
        pnl_eur = pnl_local * 0.67
    else:
        pnl_eur = pnl_local

    closed_data.append({
        'symbol': symbol,
        'fecha_compra': fecha_compra,
        'fecha_venta': fecha_venta,
        'dias': dias,
        'rent_hist': rent_hist,
        'cagr': cagr,
        'pnl_eur': pnl_eur
    })

df_closed = pd.DataFrame(closed_data)

# CALCULOS
open_total = len(df_open)
open_pnl_total = df_open['pnl_eur'].sum()
open_gains = df_open[df_open['pnl_eur'] > 0]['pnl_eur'].sum()
open_losses = abs(df_open[df_open['pnl_eur'] < 0]['pnl_eur'].sum())
open_pf = open_gains / open_losses if open_losses > 0 else float('inf')
open_positivas = (df_open['pnl_eur'] > 0).sum()
open_negativas = (df_open['pnl_eur'] < 0).sum()
df_open_cagr = df_open[df_open['cagr'].notna()]

closed_total = len(df_closed)
closed_pnl_total = df_closed['pnl_eur'].sum()
closed_gains = df_closed[df_closed['pnl_eur'] > 0]['pnl_eur'].sum()
closed_losses = abs(df_closed[df_closed['pnl_eur'] < 0]['pnl_eur'].sum())
closed_pf = closed_gains / closed_losses if closed_losses > 0 else float('inf')
closed_positivas = (df_closed['pnl_eur'] > 0).sum()
closed_negativas = (df_closed['pnl_eur'] < 0).sum()
df_closed_cagr = df_closed[df_closed['cagr'].notna()]

df_all = pd.concat([df_open, df_closed], ignore_index=True)
total_pnl = df_all['pnl_eur'].sum()
total_gains = df_all[df_all['pnl_eur'] > 0]['pnl_eur'].sum()
total_losses = abs(df_all[df_all['pnl_eur'] < 0]['pnl_eur'].sum())
total_pf = total_gains / total_losses if total_losses > 0 else float('inf')

# IMPRIMIR
def fmt(n):
    return f"{n:,.0f}".replace(",", ".")

print('=' * 80)
print('                    RESUMEN RENTABILIDAD ACCIONES')
print('=' * 80)
print()
print(f'POSICIONES ABIERTAS ({open_total} posiciones)')
print('-' * 80)
print(f"{'':30} {'Rent. Historica':>20} {'CAGR (>30d)':>15}")
print('-' * 80)
print(f"{'Media:':<30} {df_open['rent_hist'].mean():>+19.2f}% {df_open_cagr['cagr'].mean():>+14.2f}%")
print(f"{'Mediana:':<30} {df_open['rent_hist'].median():>+19.2f}% {df_open_cagr['cagr'].median():>+14.2f}%")
best_idx = df_open['rent_hist'].idxmax()
worst_idx = df_open['rent_hist'].idxmin()
print(f"{'Mejor:':<30} {df_open['rent_hist'].max():>+19.2f}% ({df_open.loc[best_idx, 'symbol']})")
print(f"{'Peor:':<30} {df_open['rent_hist'].min():>+19.2f}% ({df_open.loc[worst_idx, 'symbol']})")
print(f"{'Positivas:':<30} {open_positivas:>15} ({open_positivas/open_total*100:.1f}%)")
print(f"{'Negativas:':<30} {open_negativas:>15} ({open_negativas/open_total*100:.1f}%)")
print(f"{'Holding promedio:':<30} {df_open['dias'].mean():>12.0f} dias ({df_open['dias'].mean()/365.25:.1f} anos)")
print()
print(f"{'BENEFICIO NETO:':<30} {'+' if open_pnl_total >= 0 else ''}{fmt(open_pnl_total):>17} EUR")
print(f"{'  Ganancias:':<30} +{fmt(open_gains):>16} EUR")
print(f"{'  Perdidas:':<30} -{fmt(open_losses):>16} EUR")
print(f"{'PROFIT FACTOR:':<30} {open_pf:>18.2f}")
print()

print(f'POSICIONES CERRADAS ({closed_total} posiciones)')
print('-' * 80)
print(f"{'':30} {'Rent. Historica':>20} {'CAGR (>30d)':>15}")
print('-' * 80)
print(f"{'Media:':<30} {df_closed['rent_hist'].mean():>+19.2f}% {df_closed_cagr['cagr'].mean():>+14.2f}%")
print(f"{'Mediana:':<30} {df_closed['rent_hist'].median():>+19.2f}% {df_closed_cagr['cagr'].median():>+14.2f}%")
best_idx_c = df_closed['rent_hist'].idxmax()
worst_idx_c = df_closed['rent_hist'].idxmin()
print(f"{'Mejor:':<30} {df_closed['rent_hist'].max():>+19.2f}% ({df_closed.loc[best_idx_c, 'symbol']})")
print(f"{'Peor:':<30} {df_closed['rent_hist'].min():>+19.2f}% ({df_closed.loc[worst_idx_c, 'symbol']})")
print(f"{'Positivas:':<30} {closed_positivas:>15} ({closed_positivas/closed_total*100:.1f}%)")
print(f"{'Negativas:':<30} {closed_negativas:>15} ({closed_negativas/closed_total*100:.1f}%)")
print(f"{'Holding promedio:':<30} {df_closed['dias'].mean():>12.0f} dias ({df_closed['dias'].mean()/365.25:.1f} anos)")
print()
print(f"{'BENEFICIO NETO:':<30} {'+' if closed_pnl_total >= 0 else ''}{fmt(closed_pnl_total):>17} EUR")
print(f"{'  Ganancias:':<30} +{fmt(closed_gains):>16} EUR")
print(f"{'  Perdidas:':<30} -{fmt(closed_losses):>16} EUR")
print(f"{'PROFIT FACTOR:':<30} {closed_pf:>18.2f}")
print()

print(f'CONSOLIDADO ({len(df_all)} posiciones totales)')
print('-' * 80)
print(f"{'Media Rent. Historica:':<30} {df_all['rent_hist'].mean():>+19.2f}%")
all_pos = (df_all['pnl_eur'] > 0).sum()
all_neg = (df_all['pnl_eur'] < 0).sum()
print(f"{'Positivas:':<30} {all_pos:>15} ({all_pos/len(df_all)*100:.1f}%)")
print(f"{'Negativas:':<30} {all_neg:>15} ({all_neg/len(df_all)*100:.1f}%)")
print()
print(f"{'BENEFICIO NETO TOTAL:':<30} {'+' if total_pnl >= 0 else ''}{fmt(total_pnl):>17} EUR")
print(f"{'  Ganancias totales:':<30} +{fmt(total_gains):>16} EUR")
print(f"{'  Perdidas totales:':<30} -{fmt(total_losses):>16} EUR")
print(f"{'PROFIT FACTOR TOTAL:':<30} {total_pf:>18.2f}")
print()

print('TOP 5 MEJORES (P&L EUR):')
top5_best = df_all.nlargest(5, 'pnl_eur')
for i, (_, row) in enumerate(top5_best.iterrows(), 1):
    estado = 'abierta' if row['symbol'] in df_open['symbol'].values else 'cerrada'
    print(f"  {i}. {row['symbol']:<8} +{fmt(row['pnl_eur']):>11} EUR  ({row['rent_hist']:>+.2f}%) [{estado}]")

print()
print('TOP 5 PEORES (P&L EUR):')
top5_worst = df_all.nsmallest(5, 'pnl_eur')
for i, (_, row) in enumerate(top5_worst.iterrows(), 1):
    estado = 'abierta' if row['symbol'] in df_open['symbol'].values else 'cerrada'
    print(f"  {i}. {row['symbol']:<8} {fmt(row['pnl_eur']):>12} EUR  ({row['rent_hist']:>+.2f}%) [{estado}]")

print('=' * 80)
