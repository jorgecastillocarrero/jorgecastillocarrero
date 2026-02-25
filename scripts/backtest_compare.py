"""
Comparativa de 3 estrategias:
1. ORIGINAL: EPS>20%, Rev>12%, sin limites
2. ESTRICTA: EPS>30%, Rev>20%, sin limites
3. OPTIMIZADA: EPS>20%, Rev>12%, 50 posiciones, max 2+2
"""
import psycopg2
import sys
from datetime import timedelta

try:
    conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data', connect_timeout=5)
    cur = conn.cursor()
except Exception as e:
    print(f"ERROR: No se pudo conectar a FMP database: {e}")
    print("Asegúrate de que Docker esté corriendo: docker-compose -f docker-compose-fmp.yml up -d")
    sys.exit(1)

COMMISSION_RATE = 0.003
POSITION_SIZE = 20000

cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]
print(f'Semanas: {len(weeks)}')

def get_filtered_stocks(week, eps_min=20, rev_min=12):
    cur.execute('''
        SELECT DISTINCT m.symbol
        FROM market_cap_weekly m
        JOIN beat_streak_weekly b ON m.symbol = b.symbol AND m.week_ending = b.week_ending
        JOIN peg_weekly p ON m.symbol = p.symbol AND m.week_ending = p.week_ending
        JOIN eps_ttm_weekly e ON m.symbol = e.symbol AND m.week_ending = e.week_ending
        JOIN revenue_ttm_weekly r ON m.symbol = r.symbol AND m.week_ending = r.week_ending
        WHERE m.week_ending = %s
        AND m.market_cap >= 1000000000
        AND b.beat_streak >= 4
        AND p.peg_ratio > 0 AND p.peg_ratio <= 1.5
        AND e.eps_growth_yoy > %s
        AND r.revenue_growth_yoy > %s
        AND m.symbol NOT LIKE '%%.%%'
        AND m.symbol NOT LIKE '%%-%%'
        AND LENGTH(m.symbol) <= 5
        AND m.symbol !~ '[0-9]'
        AND RIGHT(m.symbol, 1) NOT IN ('F', 'Y')
    ''', (week, eps_min, rev_min))
    return set(row[0] for row in cur.fetchall())

def get_monday_price(symbol, friday, price_type='close'):
    monday = friday + timedelta(days=3)
    cur.execute('''
        SELECT date, open, close FROM fmp_price_history
        WHERE symbol = %s AND date >= %s AND date <= %s
        ORDER BY date LIMIT 1
    ''', (symbol, monday, monday + timedelta(days=5)))
    row = cur.fetchone()
    if row:
        return row[1] if price_type == 'open' else row[2], row[0]
    cur.execute('''
        SELECT date, open, close FROM fmp_price_history
        WHERE symbol = %s AND date <= %s ORDER BY date DESC LIMIT 1
    ''', (symbol, friday + timedelta(days=7)))
    row = cur.fetchone()
    return (row[1] if price_type == 'open' else row[2], row[0]) if row else (None, None)

def run_backtest(eps_min, rev_min, label):
    """Backtest sin limites de posiciones ni rebalanceos"""
    positions = {}
    closed_trades = []
    total_commissions = 0
    weekly_positions = []
    weekly_entries = []
    weekly_exits = []

    for i, week in enumerate(weeks[:-1]):
        current_stocks = get_filtered_stocks(week, eps_min, rev_min)
        weekly_positions.append(len(current_stocks))

        entries = 0
        exits = 0

        # Cerrar posiciones que salen del filtro
        for symbol in list(positions.keys()):
            if symbol not in current_stocks:
                pos = positions[symbol]
                price, date = get_monday_price(symbol, week, 'close')
                if price:
                    gross = price * pos['shares']
                    comm = gross * COMMISSION_RATE
                    total_commissions += comm
                    pnl = gross - comm - pos['cost_basis']
                    closed_trades.append({'symbol': symbol, 'pnl': pnl})
                del positions[symbol]
                exits += 1

        # Abrir nuevas posiciones
        for symbol in current_stocks:
            if symbol not in positions:
                price, date = get_monday_price(symbol, week, 'close')
                if price and price > 0:
                    shares = round(POSITION_SIZE / price)
                    if shares > 0:
                        gross = price * shares
                        comm = gross * COMMISSION_RATE
                        total_commissions += comm
                        positions[symbol] = {
                            'shares': shares, 'entry_price': price,
                            'cost_basis': gross + comm, 'entry_date': date
                        }
                        entries += 1

        weekly_entries.append(entries)
        weekly_exits.append(exits)

        if (i + 1) % 50 == 0:
            print(f'  {label}: Semana {i+1}/{len(weeks)}, {len(positions)} posiciones')

    # Cerrar posiciones restantes
    for symbol, pos in positions.items():
        price, _ = get_monday_price(symbol, weeks[-2], 'close')
        if price:
            gross = price * pos['shares']
            comm = gross * COMMISSION_RATE
            total_commissions += comm
            pnl = gross - comm - pos['cost_basis']
            closed_trades.append({'symbol': symbol, 'pnl': pnl})

    total_pnl = sum(t['pnl'] for t in closed_trades)
    winners = len([t for t in closed_trades if t['pnl'] > 0])
    avg_pos = sum(weekly_positions) / len(weekly_positions)
    avg_entries = sum(weekly_entries) / len(weekly_entries)
    avg_exits = sum(weekly_exits) / len(weekly_exits)

    return {
        'label': label,
        'eps_min': eps_min,
        'rev_min': rev_min,
        'trades': len(closed_trades),
        'winners': winners,
        'losers': len(closed_trades) - winners,
        'win_rate': winners / len(closed_trades) * 100 if closed_trades else 0,
        'pnl': total_pnl,
        'commissions': total_commissions,
        'avg_positions': avg_pos,
        'capital': avg_pos * POSITION_SIZE,
        'avg_entries': avg_entries,
        'avg_exits': avg_exits
    }

print('\n' + '='*80)
print('EJECUTANDO BACKTESTS...')
print('='*80)

# 1. Original: EPS>20, Rev>12
print('\n1. Estrategia ORIGINAL (EPS>20%, Rev>12%)...')
r_original = run_backtest(20, 12, 'ORIGINAL')

# 2. Estricta: EPS>30, Rev>20
print('\n2. Estrategia ESTRICTA (EPS>30%, Rev>20%)...')
r_strict = run_backtest(30, 20, 'ESTRICTA')

# Resultados
print('\n' + '='*80)
print('COMPARATIVA DE ESTRATEGIAS (2023-2026)')
print('='*80)

strategies = [r_original, r_strict]

print(f'\n{"Metrica":<25} {"ORIGINAL":<20} {"ESTRICTA":<20}')
print(f'{"":25} {"(EPS>20,Rev>12)":<20} {"(EPS>30,Rev>20)":<20}')
print('-'*65)

print(f'{"Posiciones promedio":<25} {r_original["avg_positions"]:>15.0f} {r_strict["avg_positions"]:>15.0f}')
print(f'{"Capital necesario":<25} ${r_original["capital"]:>14,.0f} ${r_strict["capital"]:>14,.0f}')
print(f'{"Trades totales":<25} {r_original["trades"]:>15} {r_strict["trades"]:>15}')
print(f'{"Ganadores":<25} {r_original["winners"]:>15} {r_strict["winners"]:>15}')
print(f'{"Perdedores":<25} {r_original["losers"]:>15} {r_strict["losers"]:>15}')
print(f'{"Win rate":<25} {r_original["win_rate"]:>14.1f}% {r_strict["win_rate"]:>14.1f}%')
print(f'{"PnL total":<25} ${r_original["pnl"]:>14,.0f} ${r_strict["pnl"]:>14,.0f}')
print(f'{"Comisiones":<25} ${r_original["commissions"]:>14,.0f} ${r_strict["commissions"]:>14,.0f}')
print(f'{"Entradas/semana (avg)":<25} {r_original["avg_entries"]:>15.1f} {r_strict["avg_entries"]:>15.1f}')
print(f'{"Salidas/semana (avg)":<25} {r_original["avg_exits"]:>15.1f} {r_strict["avg_exits"]:>15.1f}')

print('\n' + '-'*65)
print('EFICIENCIA:')
print('-'*65)

for r in strategies:
    roi = (r['pnl'] / r['capital']) * 100 if r['capital'] > 0 else 0
    cagr = ((1 + r['pnl']/r['capital']) ** (1/3) - 1) * 100 if r['capital'] > 0 else 0
    pnl_per_trade = r['pnl'] / r['trades'] if r['trades'] > 0 else 0
    r['roi'] = roi
    r['cagr'] = cagr
    r['pnl_per_trade'] = pnl_per_trade

print(f'{"ROI (3 anos)":<25} {r_original["roi"]:>14.1f}% {r_strict["roi"]:>14.1f}%')
print(f'{"CAGR anualizado":<25} {r_original["cagr"]:>14.1f}% {r_strict["cagr"]:>14.1f}%')
print(f'{"PnL por trade":<25} ${r_original["pnl_per_trade"]:>14,.0f} ${r_strict["pnl_per_trade"]:>14,.0f}')

# Incluir optimizada (datos del backtest anterior)
print('\n' + '='*80)
print('TABLA COMPLETA CON OPTIMIZADA')
print('='*80)
print(f'\n{"Metrica":<22} {"ORIGINAL":<18} {"ESTRICTA":<18} {"OPTIMIZADA":<18}')
print(f'{"":22} {"(EPS>20,Rev>12)":<18} {"(EPS>30,Rev>20)":<18} {"(50pos,2+2max)":<18}')
print('-'*76)
print(f'{"Posiciones":<22} {r_original["avg_positions"]:>13.0f} {r_strict["avg_positions"]:>13.0f} {"50":>13}')
print(f'{"Capital":<22} ${r_original["capital"]/1e6:>12.1f}M ${r_strict["capital"]/1e6:>12.1f}M ${"1.0":>11}M')
print(f'{"Trades":<22} {r_original["trades"]:>13} {r_strict["trades"]:>13} {"333":>13}')
print(f'{"Win rate":<22} {r_original["win_rate"]:>12.1f}% {r_strict["win_rate"]:>12.1f}% {"57.1":>12}%')
print(f'{"PnL total":<22} ${r_original["pnl"]/1e6:>12.2f}M ${r_strict["pnl"]/1e6:>12.2f}M ${"0.70":>11}M')
print(f'{"CAGR":<22} {r_original["cagr"]:>12.1f}% {r_strict["cagr"]:>12.1f}% {"19.4":>12}%')

cur.close()
conn.close()
