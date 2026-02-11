"""
Comparación detallada MIN_DRAWDOWN vs ORIGINAL
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

FORMULAS = {
    'ORIGINAL': lambda peg, beat, eps, rev: (1.5 - peg) * 50 + min(eps, 150) * 0.4 + min(rev, 100) * 0.6 + min(beat, 15) * 2,
    'MIN_DRAWDOWN': lambda peg, beat, eps, rev: (1.0 - peg) * 60 + min(beat, 30) * 6 + min(eps, 80) * 0.2 + min(rev, 50) * 0.2,
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

def run_backtest(formula_name, formula_func):
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
    avg_sectors = sum(len(s) for s in sector_history) / len(sector_history) if sector_history else 0
    max_conc = max((max(s.values())/sum(s.values())*100 if s else 0) for s in sector_history) if sector_history else 0

    # Sector breakdown
    sector_pnl = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'winners': 0})
    for t in closed_trades:
        sector_pnl[t['sector']]['trades'] += 1
        sector_pnl[t['sector']]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            sector_pnl[t['sector']]['winners'] += 1

    return {
        'name': formula_name,
        'trades': len(closed_trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners)/len(closed_trades)*100 if closed_trades else 0,
        'total_pnl': total_pnl,
        'pnl_per_trade': total_pnl/len(closed_trades) if closed_trades else 0,
        'commissions': total_commissions,
        'avg_return': sum(t["pct"] for t in closed_trades)/len(closed_trades) if closed_trades else 0,
        'best_trade_pnl': max(t["pnl"] for t in closed_trades) if closed_trades else 0,
        'best_trade_pct': max(t["pct"] for t in closed_trades) if closed_trades else 0,
        'worst_trade_pnl': min(t["pnl"] for t in closed_trades) if closed_trades else 0,
        'worst_trade_pct': min(t["pct"] for t in closed_trades) if closed_trades else 0,
        'avg_winner_pnl': sum(t["pnl"] for t in winners)/len(winners) if winners else 0,
        'avg_winner_pct': sum(t["pct"] for t in winners)/len(winners) if winners else 0,
        'avg_loser_pnl': sum(t["pnl"] for t in losers)/len(losers) if losers else 0,
        'avg_loser_pct': sum(t["pct"] for t in losers)/len(losers) if losers else 0,
        'avg_duration': sum(t["duration"] for t in closed_trades)/len(closed_trades) if closed_trades else 0,
        'max_duration': max(t["duration"] for t in closed_trades) if closed_trades else 0,
        'min_duration': min(t["duration"] for t in closed_trades) if closed_trades else 0,
        'max_drawdown': max_drawdown,
        'avg_sectors': avg_sectors,
        'max_concentration': max_conc,
        'sector_pnl': dict(sector_pnl),
        'closed_trades': closed_trades,
    }

# Run both
print('Calculando ORIGINAL...', flush=True)
orig = run_backtest('ORIGINAL', FORMULAS['ORIGINAL'])
print('Calculando MIN_DRAWDOWN...', flush=True)
mindd = run_backtest('MIN_DRAWDOWN', FORMULAS['MIN_DRAWDOWN'])

# Print comparison
print()
print('='*80)
print('COMPARACION: ORIGINAL vs MIN_DRAWDOWN')
print('='*80)
print(f'Periodo: {weeks[0]} a {weeks[-1]} ({len(weeks)} semanas)')
print()

def fmt(val, is_pct=False, is_money=False, is_days=False):
    if is_money:
        return f'${val:>10,.0f}'
    elif is_pct:
        return f'{val:>6.1f}%'
    elif is_days:
        return f'{val:>5.0f}d'
    else:
        return f'{val:>8}'

def compare_row(label, o_val, m_val, is_pct=False, is_money=False, is_days=False, higher_better=True):
    o_str = fmt(o_val, is_pct, is_money, is_days)
    m_str = fmt(m_val, is_pct, is_money, is_days)
    if higher_better:
        winner = 'MIN_DD' if m_val > o_val else ('ORIG' if o_val > m_val else 'IGUAL')
    else:
        winner = 'MIN_DD' if m_val < o_val else ('ORIG' if o_val < m_val else 'IGUAL')
    diff = m_val - o_val
    if is_money:
        diff_str = f'{diff:+,.0f}'
    elif is_pct:
        diff_str = f'{diff:+.1f}pp'
    else:
        diff_str = f'{diff:+.0f}'
    print(f'{label:<22} {o_str:>12} {m_str:>12}   {diff_str:>12}   {winner}')

print(f'{"METRICA":<22} {"ORIGINAL":>12} {"MIN_DRAWDOWN":>12}   {"DIFERENCIA":>12}   GANADOR')
print('-'*80)

print('\n--- TRADES ---')
compare_row('Total trades', orig['trades'], mindd['trades'])
compare_row('Ganadores', orig['winners'], mindd['winners'])
compare_row('Perdedores', orig['losers'], mindd['losers'], higher_better=False)
compare_row('Win Rate', orig['win_rate'], mindd['win_rate'], is_pct=True)

print('\n--- RENTABILIDAD ---')
compare_row('PnL Total', orig['total_pnl'], mindd['total_pnl'], is_money=True)
compare_row('PnL/Trade', orig['pnl_per_trade'], mindd['pnl_per_trade'], is_money=True)
compare_row('Comisiones', orig['commissions'], mindd['commissions'], is_money=True, higher_better=False)
compare_row('Retorno medio', orig['avg_return'], mindd['avg_return'], is_pct=True)
compare_row('Mejor trade PnL', orig['best_trade_pnl'], mindd['best_trade_pnl'], is_money=True)
compare_row('Mejor trade %', orig['best_trade_pct'], mindd['best_trade_pct'], is_pct=True)
compare_row('Peor trade PnL', orig['worst_trade_pnl'], mindd['worst_trade_pnl'], is_money=True, higher_better=False)
compare_row('Peor trade %', orig['worst_trade_pct'], mindd['worst_trade_pct'], is_pct=True, higher_better=False)

print('\n--- GANADORES ---')
compare_row('PnL medio ganador', orig['avg_winner_pnl'], mindd['avg_winner_pnl'], is_money=True)
compare_row('Retorno medio', orig['avg_winner_pct'], mindd['avg_winner_pct'], is_pct=True)

print('\n--- PERDEDORES ---')
compare_row('PnL medio perdedor', orig['avg_loser_pnl'], mindd['avg_loser_pnl'], is_money=True, higher_better=False)
compare_row('Retorno medio', orig['avg_loser_pct'], mindd['avg_loser_pct'], is_pct=True, higher_better=False)

print('\n--- DURACION ---')
compare_row('Duracion media', orig['avg_duration'], mindd['avg_duration'], is_days=True)
compare_row('Duracion maxima', orig['max_duration'], mindd['max_duration'], is_days=True)
compare_row('Duracion minima', orig['min_duration'], mindd['min_duration'], is_days=True)

print('\n--- RIESGO ---')
compare_row('Max Drawdown', orig['max_drawdown'], mindd['max_drawdown'], is_pct=True, higher_better=False)
compare_row('Sectores promedio', orig['avg_sectors'], mindd['avg_sectors'])
compare_row('Max concentracion', orig['max_concentration'], mindd['max_concentration'], is_pct=True, higher_better=False)

print('\n' + '='*80)
print('DISTRIBUCION POR SECTOR - PnL')
print('='*80)
all_sectors = set(orig['sector_pnl'].keys()) | set(mindd['sector_pnl'].keys())
print(f'{"SECTOR":<25} {"ORIG PnL":>12} {"ORIG WR":>8} {"MINDD PnL":>12} {"MINDD WR":>8}')
print('-'*80)
for sec in sorted(all_sectors, key=lambda x: orig['sector_pnl'].get(x, {}).get('pnl', 0), reverse=True):
    o = orig['sector_pnl'].get(sec, {'pnl': 0, 'trades': 0, 'winners': 0})
    m = mindd['sector_pnl'].get(sec, {'pnl': 0, 'trades': 0, 'winners': 0})
    o_wr = o['winners']/o['trades']*100 if o['trades'] > 0 else 0
    m_wr = m['winners']/m['trades']*100 if m['trades'] > 0 else 0
    print(f'{sec:<25} ${o["pnl"]:>10,.0f} {o_wr:>7.0f}% ${m["pnl"]:>10,.0f} {m_wr:>7.0f}%')

print('\n' + '='*80)
print('TOP 5 MEJORES TRADES - ORIGINAL')
print('='*80)
best_o = sorted(orig['closed_trades'], key=lambda x: x['pnl'], reverse=True)[:5]
for t in best_o:
    print(f'{t["symbol"]:<6} PnL: ${t["pnl"]:>8,.0f} ({t["pct"]:>5.1f}%)  {t["duration"]:>3}d  {t["sector"]}')

print('\n' + '='*80)
print('TOP 5 MEJORES TRADES - MIN_DRAWDOWN')
print('='*80)
best_m = sorted(mindd['closed_trades'], key=lambda x: x['pnl'], reverse=True)[:5]
for t in best_m:
    print(f'{t["symbol"]:<6} PnL: ${t["pnl"]:>8,.0f} ({t["pct"]:>5.1f}%)  {t["duration"]:>3}d  {t["sector"]}')

print('\n' + '='*80)
print('TOP 5 PEORES TRADES - ORIGINAL')
print('='*80)
worst_o = sorted(orig['closed_trades'], key=lambda x: x['pnl'])[:5]
for t in worst_o:
    print(f'{t["symbol"]:<6} PnL: ${t["pnl"]:>8,.0f} ({t["pct"]:>5.1f}%)  {t["duration"]:>3}d  {t["sector"]}')

print('\n' + '='*80)
print('TOP 5 PEORES TRADES - MIN_DRAWDOWN')
print('='*80)
worst_m = sorted(mindd['closed_trades'], key=lambda x: x['pnl'])[:5]
for t in worst_m:
    print(f'{t["symbol"]:<6} PnL: ${t["pnl"]:>8,.0f} ({t["pct"]:>5.1f}%)  {t["duration"]:>3}d  {t["sector"]}')

cur.close()
conn.close()
