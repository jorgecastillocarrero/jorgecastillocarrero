"""
Estadísticas completas de MIN_DRAWDOWN
"""
import psycopg2
from datetime import timedelta
from collections import defaultdict

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
cur = conn.cursor()

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

cur.execute('SELECT symbol, sector FROM fmp_profiles WHERE sector IS NOT NULL')
SECTORS = {row[0]: row[1] for row in cur.fetchall()}

def get_ranked_stocks(week_ending, current_positions=None):
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
        # MIN_DRAWDOWN formula
        score = (1.0 - peg) * 60 + min(beat, 30) * 6 + min(eps_g, 80) * 0.2 + min(rev_g, 50) * 0.2
        stocks.append({'symbol': symbol, 'score': score, 'sector': SECTORS.get(symbol, 'Unknown')})

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

# Run backtest
positions = {}
closed_trades = []
total_commissions = 0
portfolio_values = []
peak_value = 0
max_drawdown = 0
sector_history = []

for i, week in enumerate(weeks[:-1]):
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

    ranked = get_ranked_stocks(week, set(positions.keys()))
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
                closed_trades.append({'symbol': symbol, 'pnl': pnl, 'pct': pct, 'duration': duration, 'sector': SECTORS.get(symbol, 'Unknown')})
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

    sectors = defaultdict(int)
    for sym in positions:
        sectors[SECTORS.get(sym, 'Unknown')] += 1
    sector_history.append(dict(sectors))

# Close remaining
for symbol, pos in list(positions.items()):
    price, date = get_monday_price(symbol, weeks[-2], 'close')
    if price:
        gross = price * pos['shares']
        comm = gross * COMMISSION_RATE
        total_commissions += comm
        pnl = gross - comm - pos['cost_basis']
        pct = (pnl / pos['cost_basis']) * 100
        duration = (date - pos['entry_date']).days if date else 0
        closed_trades.append({'symbol': symbol, 'pnl': pnl, 'pct': pct, 'duration': duration, 'sector': SECTORS.get(symbol, 'Unknown')})

# Statistics
total_pnl = sum(t['pnl'] for t in closed_trades)
winners = [t for t in closed_trades if t['pnl'] > 0]
losers = [t for t in closed_trades if t['pnl'] <= 0]

print('='*70)
print('MIN_DRAWDOWN - ESTADISTICAS COMPLETAS')
print('='*70)
print(f'Periodo: {weeks[0]} a {weeks[-1]}')
print(f'Semanas: {len(weeks)}')
print()
print('TRADES')
print('-'*70)
print(f'Total trades:        {len(closed_trades)}')
print(f'Ganadores:           {len(winners)} ({len(winners)/len(closed_trades)*100:.1f}%)')
print(f'Perdedores:          {len(losers)} ({len(losers)/len(closed_trades)*100:.1f}%)')
print()
print('RENTABILIDAD')
print('-'*70)
print(f'PnL Total:           ${total_pnl:,.0f}')
print(f'PnL/Trade:           ${total_pnl/len(closed_trades):,.0f}')
print(f'Comisiones:          ${total_commissions:,.0f}')
print(f'Retorno medio:       {sum(t["pct"] for t in closed_trades)/len(closed_trades):.1f}%')
print(f'Mejor trade:         ${max(t["pnl"] for t in closed_trades):,.0f} ({max(t["pct"] for t in closed_trades):.1f}%)')
print(f'Peor trade:          ${min(t["pnl"] for t in closed_trades):,.0f} ({min(t["pct"] for t in closed_trades):.1f}%)')
print()
print('GANADORES')
print('-'*70)
print(f'PnL medio ganador:   ${sum(t["pnl"] for t in winners)/len(winners):,.0f}')
print(f'Retorno medio:       {sum(t["pct"] for t in winners)/len(winners):.1f}%')
print()
print('PERDEDORES')
print('-'*70)
print(f'PnL medio perdedor:  ${sum(t["pnl"] for t in losers)/len(losers):,.0f}')
print(f'Retorno medio:       {sum(t["pct"] for t in losers)/len(losers):.1f}%')
print()
print('DURACION')
print('-'*70)
print(f'Duracion media:      {sum(t["duration"] for t in closed_trades)/len(closed_trades):.0f} dias')
print(f'Duracion maxima:     {max(t["duration"] for t in closed_trades)} dias')
print(f'Duracion minima:     {min(t["duration"] for t in closed_trades)} dias')
print()
print('RIESGO')
print('-'*70)
print(f'Max Drawdown:        {max_drawdown:.1f}%')
avg_sectors = sum(len(s) for s in sector_history) / len(sector_history)
print(f'Sectores promedio:   {avg_sectors:.1f}')
max_conc = max((max(s.values())/sum(s.values())*100 if s else 0) for s in sector_history)
print(f'Max concentracion:   {max_conc:.0f}%')
print()
print('DISTRIBUCION POR SECTOR')
print('-'*70)
sector_pnl = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'winners': 0})
for t in closed_trades:
    sector_pnl[t['sector']]['trades'] += 1
    sector_pnl[t['sector']]['pnl'] += t['pnl']
    if t['pnl'] > 0:
        sector_pnl[t['sector']]['winners'] += 1

for sec, data in sorted(sector_pnl.items(), key=lambda x: x[1]['pnl'], reverse=True):
    wr = data['winners']/data['trades']*100 if data['trades'] > 0 else 0
    print(f'{sec:<25} Trades: {data["trades"]:>3}  PnL: ${data["pnl"]:>10,.0f}  WinRate: {wr:.0f}%')

print()
print('TOP 10 MEJORES TRADES')
print('-'*70)
best = sorted(closed_trades, key=lambda x: x['pnl'], reverse=True)[:10]
for t in best:
    print(f'{t["symbol"]:<6} PnL: ${t["pnl"]:>8,.0f} ({t["pct"]:>5.1f}%)  Duracion: {t["duration"]:>3}d  Sector: {t["sector"]}')

print()
print('TOP 10 PEORES TRADES')
print('-'*70)
worst = sorted(closed_trades, key=lambda x: x['pnl'])[:10]
for t in worst:
    print(f'{t["symbol"]:<6} PnL: ${t["pnl"]:>8,.0f} ({t["pct"]:>5.1f}%)  Duracion: {t["duration"]:>3}d  Sector: {t["sector"]}')

cur.close()
conn.close()
