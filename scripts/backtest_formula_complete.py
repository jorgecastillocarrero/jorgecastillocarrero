"""
Test completo de fórmulas de scoring incluyendo:
- Fórmulas originales
- MIN_DRAWDOWN: prioriza estabilidad (bajo PEG, alto beat, menor volatilidad implícita)
- SECTOR_DIV: penaliza concentración sectorial
"""
import psycopg2
import sys
from datetime import timedelta
from collections import defaultdict

try:
    conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data', connect_timeout=5)
    cur = conn.cursor()
except Exception as e:
    print(f"ERROR: No se pudo conectar a FMP database: {e}")
    print("Asegúrate de que Docker esté corriendo: docker-compose -f docker-compose-fmp.yml up -d")
    sys.exit(1)

COMMISSION_RATE = 0.003
POSITION_SIZE = 20000
MAX_POSITIONS = 50
MAX_CHANGES = 2

cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]

# Cargar sectores de fmp_profiles
cur.execute('SELECT symbol, sector FROM fmp_profiles WHERE sector IS NOT NULL')
SECTORS = {row[0]: row[1] for row in cur.fetchall()}

FORMULAS = {
    'ORIGINAL': lambda peg, beat, eps, rev, sector, sector_count:
        (1.5 - peg) * 50 + min(eps, 150) * 0.4 + min(rev, 100) * 0.6 + min(beat, 15) * 2,

    'BEAT_HEAVY': lambda peg, beat, eps, rev, sector, sector_count:
        (1.5 - peg) * 30 + min(eps, 150) * 0.3 + min(rev, 100) * 0.4 + min(beat, 30) * 5,

    'QUALITY': lambda peg, beat, eps, rev, sector, sector_count:
        (1.5 - peg) * 30 + min(eps, 100) * 0.3 + min(rev, 80) * 0.3 + min(beat, 25) * 4,

    'LOW_PEG_ONLY': lambda peg, beat, eps, rev, sector, sector_count:
        (1.5 - peg) * 100,

    # MIN_DRAWDOWN: prioriza estabilidad - alto beat (consistencia), bajo PEG (value), growth moderado
    'MIN_DRAWDOWN': lambda peg, beat, eps, rev, sector, sector_count:
        (1.0 - peg) * 60 + min(beat, 30) * 6 + min(eps, 80) * 0.2 + min(rev, 50) * 0.2,

    # SECTOR_DIV: penaliza si ya hay muchas acciones del mismo sector
    'SECTOR_DIV': lambda peg, beat, eps, rev, sector, sector_count:
        (1.5 - peg) * 40 + min(eps, 150) * 0.4 + min(rev, 100) * 0.5 + min(beat, 20) * 3 - sector_count * 5,
}

def get_ranked_stocks(week_ending, formula_func, current_positions=None):
    cur.execute('''
        SELECT m.symbol, m.market_cap/1e9 as mcap, b.beat_streak,
               p.peg_ratio, e.eps_growth_yoy, r.revenue_growth_yoy
        FROM market_cap_weekly m
        JOIN beat_streak_weekly b ON m.symbol = b.symbol AND m.week_ending = b.week_ending
        JOIN peg_weekly p ON m.symbol = p.symbol AND m.week_ending = p.week_ending
        JOIN eps_ttm_weekly e ON m.symbol = e.symbol AND m.week_ending = e.week_ending
        JOIN revenue_ttm_weekly r ON m.symbol = r.symbol AND m.week_ending = r.week_ending
        WHERE m.week_ending = %s
        AND m.market_cap >= 1000000000
        AND b.beat_streak >= 4
        AND p.peg_ratio > 0 AND p.peg_ratio <= 1.5
        AND e.eps_growth_yoy > 20
        AND r.revenue_growth_yoy > 12
        AND m.symbol NOT LIKE '%%.%%'
        AND m.symbol NOT LIKE '%%-%%'
        AND LENGTH(m.symbol) <= 5
        AND m.symbol !~ '[0-9]'
        AND RIGHT(m.symbol, 1) NOT IN ('F', 'Y')
    ''', (week_ending,))

    # Contar sectores actuales
    sector_counts = defaultdict(int)
    if current_positions:
        for sym in current_positions:
            sec = SECTORS.get(sym, 'Unknown')
            sector_counts[sec] += 1

    stocks = []
    for row in cur.fetchall():
        symbol, mcap, beat, peg, eps_g, rev_g = row
        peg = float(peg) if peg else 1.5
        beat = int(beat) if beat else 0
        eps_g = float(eps_g) if eps_g else 0
        rev_g = float(rev_g) if rev_g else 0
        sector = SECTORS.get(symbol, 'Unknown')
        sector_count = sector_counts.get(sector, 0)

        score = formula_func(peg, beat, eps_g, rev_g, sector, sector_count)
        stocks.append({'symbol': symbol, 'score': score, 'sector': sector})

    stocks.sort(key=lambda x: x['score'], reverse=True)
    return stocks

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

def run_backtest(formula_name, formula_func):
    positions = {}
    closed_trades = []
    total_commissions = 0

    # Para calcular drawdown
    portfolio_values = []
    peak_value = 0
    max_drawdown = 0

    # Para diversificación
    sector_history = []

    for i, week in enumerate(weeks[:-1]):
        # Calcular valor del portfolio antes de cambios
        portfolio_value = 0
        for sym, pos in positions.items():
            price, _ = get_monday_price(sym, week, 'close')
            if price:
                portfolio_value += price * pos['shares']

        if portfolio_value > 0:
            portfolio_values.append(portfolio_value)
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            drawdown = (peak_value - portfolio_value) / peak_value * 100 if peak_value > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        ranked = get_ranked_stocks(week, formula_func, set(positions.keys()))
        top50_symbols = set(s['symbol'] for s in ranked[:MAX_POSITIONS])
        ranked_dict = {s['symbol']: s['score'] for s in ranked}

        current_symbols = set(positions.keys())
        is_first_week = (i == 0)

        to_sell = current_symbols - top50_symbols
        to_buy = top50_symbols - current_symbols

        sells_list = [(s, ranked_dict.get(s, -9999)) for s in to_sell]
        sells_list.sort(key=lambda x: x[1])

        buys_list = [(s, ranked_dict.get(s, 0)) for s in to_buy]
        buys_list.sort(key=lambda x: x[1], reverse=True)

        sells_done = 0
        buys_done = 0

        max_sells = len(sells_list) if is_first_week else MAX_CHANGES
        for symbol, _ in sells_list:
            if sells_done >= max_sells:
                break
            if symbol in positions:
                pos = positions[symbol]
                price, date = get_monday_price(symbol, week, 'close')
                if price:
                    gross = price * pos['shares']
                    comm = gross * COMMISSION_RATE
                    total_commissions += comm
                    pnl = gross - comm - pos['cost_basis']
                    pct = (pnl / pos['cost_basis']) * 100
                    duration = (date - pos['entry_date']).days
                    closed_trades.append({'pnl': pnl, 'pct': pct, 'duration': duration})
                del positions[symbol]
                sells_done += 1

        max_buys = MAX_POSITIONS if is_first_week else MAX_CHANGES
        for symbol, _ in buys_list:
            if buys_done >= max_buys:
                break
            if len(positions) >= MAX_POSITIONS:
                break
            price, date = get_monday_price(symbol, week, 'close')
            if price and price > 0:
                shares = round(POSITION_SIZE / price)
                if shares > 0:
                    gross = price * shares
                    comm = gross * COMMISSION_RATE
                    total_commissions += comm
                    positions[symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date,
                        'cost_basis': gross + comm
                    }
                    buys_done += 1

        # Guardar distribución sectorial
        sectors = defaultdict(int)
        for sym in positions:
            sectors[SECTORS.get(sym, 'Unknown')] += 1
        sector_history.append(dict(sectors))

    # Cerrar posiciones restantes
    for symbol, pos in list(positions.items()):
        price, date = get_monday_price(symbol, weeks[-2], 'close')
        if price:
            gross = price * pos['shares']
            comm = gross * COMMISSION_RATE
            total_commissions += comm
            pnl = gross - comm - pos['cost_basis']
            pct = (pnl / pos['cost_basis']) * 100
            duration = (date - pos['entry_date']).days if date else 0
            closed_trades.append({'pnl': pnl, 'pct': pct, 'duration': duration})

    total_pnl = sum(t['pnl'] for t in closed_trades)
    winners = [t for t in closed_trades if t['pnl'] > 0]
    avg_duration = sum(t['duration'] for t in closed_trades) / len(closed_trades) if closed_trades else 0

    # Calcular diversificación promedio (número de sectores distintos)
    avg_sectors = sum(len(s) for s in sector_history) / len(sector_history) if sector_history else 0

    # Concentración máxima (% del sector más grande)
    max_concentration = 0
    for s in sector_history:
        if s:
            total = sum(s.values())
            max_sec = max(s.values()) if s else 0
            conc = (max_sec / total * 100) if total > 0 else 0
            if conc > max_concentration:
                max_concentration = conc

    return {
        'formula': formula_name,
        'trades': len(closed_trades),
        'winners': len(winners),
        'win_rate': len(winners) / len(closed_trades) * 100 if closed_trades else 0,
        'pnl': total_pnl,
        'pnl_per_trade': total_pnl / len(closed_trades) if closed_trades else 0,
        'avg_return': sum(t['pct'] for t in closed_trades) / len(closed_trades) if closed_trades else 0,
        'avg_duration': avg_duration,
        'max_drawdown': max_drawdown,
        'avg_sectors': avg_sectors,
        'max_concentration': max_concentration,
    }

print('='*100)
print('TEST COMPLETO DE FÓRMULAS (50 pos, max 2 cambios/semana)')
print('Incluye: Duración, Drawdown, Diversificación Sectorial')
print('='*100)

results = []
for name, func in FORMULAS.items():
    print(f'Probando {name}...', end=' ')
    r = run_backtest(name, func)
    results.append(r)
    print(f'PnL/trade: ${r["pnl_per_trade"]:,.0f}, MaxDD: {r["max_drawdown"]:.1f}%')

results.sort(key=lambda x: x['pnl_per_trade'], reverse=True)

print('\n' + '='*100)
print('RANKING POR PnL POR TRADE')
print('='*100)

print(f'\n{"#":<3} {"Formula":<14} {"Trades":<7} {"Win%":<7} {"PnL/Tr":<10} {"Ret%":<7} {"Durac":<7} {"MaxDD":<8} {"Sectors":<8} {"MaxConc":<8}')
print('-'*100)

for i, r in enumerate(results):
    print(f'{i+1:<3} {r["formula"]:<14} {r["trades"]:<7} {r["win_rate"]:<6.1f}% ${r["pnl_per_trade"]:<8,.0f} {r["avg_return"]:<6.1f}% {r["avg_duration"]:<6.0f}d {r["max_drawdown"]:<7.1f}% {r["avg_sectors"]:<7.1f} {r["max_concentration"]:<7.1f}%')

# Mejor por cada métrica
print('\n' + '='*100)
print('MEJOR POR CADA MÉTRICA')
print('='*100)

best_pnl = max(results, key=lambda x: x['pnl_per_trade'])
best_winrate = max(results, key=lambda x: x['win_rate'])
best_drawdown = min(results, key=lambda x: x['max_drawdown'])
best_diversified = max(results, key=lambda x: x['avg_sectors'])

print(f'Mejor PnL/Trade:      {best_pnl["formula"]} (${best_pnl["pnl_per_trade"]:,.0f})')
print(f'Mejor Win Rate:       {best_winrate["formula"]} ({best_winrate["win_rate"]:.1f}%)')
print(f'Menor Drawdown:       {best_drawdown["formula"]} ({best_drawdown["max_drawdown"]:.1f}%)')
print(f'Mayor Diversificación: {best_diversified["formula"]} ({best_diversified["avg_sectors"]:.1f} sectores)')

cur.close()
conn.close()
