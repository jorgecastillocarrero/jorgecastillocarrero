"""
Test de diferentes fórmulas de scoring para maximizar retorno por trade
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
MAX_POSITIONS = 50
MAX_CHANGES = 2  # Usar el óptimo encontrado

cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]

# Diferentes fórmulas de scoring
FORMULAS = {
    'ORIGINAL': lambda peg, beat, eps, rev: (1.5 - peg) * 50 + min(eps, 150) * 0.4 + min(rev, 100) * 0.6 + min(beat, 15) * 2,

    'PEG_HEAVY': lambda peg, beat, eps, rev: (1.5 - peg) * 100 + min(eps, 150) * 0.2 + min(rev, 100) * 0.3 + min(beat, 15) * 1,

    'GROWTH_HEAVY': lambda peg, beat, eps, rev: (1.5 - peg) * 20 + min(eps, 200) * 0.8 + min(rev, 150) * 1.0 + min(beat, 15) * 1,

    'BEAT_HEAVY': lambda peg, beat, eps, rev: (1.5 - peg) * 30 + min(eps, 150) * 0.3 + min(rev, 100) * 0.4 + min(beat, 30) * 5,

    'BALANCED': lambda peg, beat, eps, rev: (1.5 - peg) * 40 + min(eps, 150) * 0.5 + min(rev, 100) * 0.5 + min(beat, 20) * 3,

    'LOW_PEG_ONLY': lambda peg, beat, eps, rev: (1.5 - peg) * 100,

    'MOMENTUM': lambda peg, beat, eps, rev: (1.5 - peg) * 25 + min(eps, 300) * 1.0 + min(rev, 200) * 0.8 + min(beat, 10) * 1,

    'VALUE_GROWTH': lambda peg, beat, eps, rev: (1.0 - peg) * 80 + min(eps, 100) * 0.5 + min(rev, 80) * 0.5 + min(beat, 20) * 2,

    'QUALITY': lambda peg, beat, eps, rev: (1.5 - peg) * 30 + min(eps, 100) * 0.3 + min(rev, 80) * 0.3 + min(beat, 25) * 4,

    'AGGRESSIVE': lambda peg, beat, eps, rev: (1.5 - peg) * 60 + min(eps, 500) * 0.5 + min(rev, 200) * 0.8 + min(beat, 8) * 1,
}

def get_ranked_stocks(week_ending, formula_func):
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

    stocks = []
    for row in cur.fetchall():
        symbol, mcap, beat, peg, eps_g, rev_g = row
        peg = float(peg) if peg else 1.5
        beat = int(beat) if beat else 0
        eps_g = float(eps_g) if eps_g else 0
        rev_g = float(rev_g) if rev_g else 0

        score = formula_func(peg, beat, eps_g, rev_g)
        stocks.append({'symbol': symbol, 'score': score})

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

    for i, week in enumerate(weeks[:-1]):
        ranked = get_ranked_stocks(week, formula_func)
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
    avg_duration_winners = sum(t['duration'] for t in winners) / len(winners) if winners else 0
    losers = [t for t in closed_trades if t['pnl'] <= 0]
    avg_duration_losers = sum(t['duration'] for t in losers) / len(losers) if losers else 0

    return {
        'formula': formula_name,
        'trades': len(closed_trades),
        'winners': len(winners),
        'win_rate': len(winners) / len(closed_trades) * 100 if closed_trades else 0,
        'pnl': total_pnl,
        'pnl_per_trade': total_pnl / len(closed_trades) if closed_trades else 0,
        'avg_return': sum(t['pct'] for t in closed_trades) / len(closed_trades) if closed_trades else 0,
        'avg_winner': sum(t['pct'] for t in winners) / len(winners) if winners else 0,
        'avg_duration': avg_duration,
        'avg_duration_winners': avg_duration_winners,
        'avg_duration_losers': avg_duration_losers,
    }

print('='*90)
print('TEST DE FÓRMULAS DE SCORING (50 pos, max 2 cambios/semana)')
print('='*90)

results = []
for name, func in FORMULAS.items():
    print(f'Probando {name}...', end=' ')
    r = run_backtest(name, func)
    results.append(r)
    print(f'PnL/trade: ${r["pnl_per_trade"]:,.0f}')

# Ordenar por PnL per trade
results.sort(key=lambda x: x['pnl_per_trade'], reverse=True)

print('\n' + '='*90)
print('RANKING POR PnL POR TRADE')
print('='*90)

print(f'\n{"#":<3} {"Formula":<14} {"Trades":<7} {"Win%":<7} {"PnL/Trade":<11} {"Ret%":<7} {"DurMedia":<9} {"DurWin":<8} {"DurLose":<8}')
print('-'*95)

for i, r in enumerate(results):
    print(f'{i+1:<3} {r["formula"]:<14} {r["trades"]:<7} {r["win_rate"]:<6.1f}% ${r["pnl_per_trade"]:<9,.0f} {r["avg_return"]:<6.1f}% {r["avg_duration"]:<8.0f}d {r["avg_duration_winners"]:<7.0f}d {r["avg_duration_losers"]:<7.0f}d')

# Mejor fórmula
best = results[0]
print(f'\n*** GANADORA: {best["formula"]} ***')
print(f'    PnL por trade: ${best["pnl_per_trade"]:,.0f}')
print(f'    Win rate: {best["win_rate"]:.1f}%')
print(f'    PnL total: ${best["pnl"]:,.0f}')

# CAGR
capital = MAX_POSITIONS * POSITION_SIZE
print(f'\nCAGR de la mejor fórmula: {((1 + best["pnl"]/capital) ** (1/3) - 1) * 100:.1f}%')

cur.close()
conn.close()
