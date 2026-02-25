"""
Backtesting con scoring basado en transcripts de earnings calls.
Analiza el guidance para extraer sentimiento positivo/negativo.
"""
import psycopg2
import re
import sys
from datetime import timedelta, date
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

# Keywords para análisis de guidance
POSITIVE_KEYWORDS = [
    'strong demand', 'exceed expectations', 'record revenue', 'significant growth',
    'momentum', 'accelerating', 'beat', 'outperform', 'robust', 'exceptional',
    'raising guidance', 'increasing outlook', 'ahead of plan', 'upside',
    'tailwind', 'market share gains', 'pricing power', 'margin expansion',
    'confident', 'optimistic', 'excited', 'tremendous', 'outstanding'
]

NEGATIVE_KEYWORDS = [
    'headwind', 'challenging', 'softness', 'weakness', 'decline',
    'below expectations', 'lowering guidance', 'reducing outlook', 'cautious',
    'uncertain', 'pressure', 'slowdown', 'decelerat', 'difficult',
    'concern', 'risk', 'delay', 'inventory buildup', 'margin compression',
    'disappointed', 'miss', 'shortfall'
]

# Cache de transcript scores
transcript_scores = {}

def extract_guidance_section(content):
    """Extrae la sección de guidance del transcript."""
    content_lower = content.lower()

    # Buscar sección de outlook/guidance
    patterns = [
        r'(outlook|guidance|looking ahead|for the (fourth|first|second|third) quarter)',
        r'(we expect|we anticipate|our expectation)',
        r'(revenue is expected|we are guiding)'
    ]

    best_start = -1
    for pattern in patterns:
        match = re.search(pattern, content_lower)
        if match:
            if best_start == -1 or match.start() < best_start:
                best_start = match.start()

    if best_start == -1:
        # Si no encuentra guidance específico, usar últimos 20% del texto
        best_start = int(len(content) * 0.8)

    # Tomar desde el inicio de guidance hasta 2000 chars después
    guidance_text = content[max(0, best_start - 200):best_start + 2000]
    return guidance_text.lower()

def calculate_transcript_score(symbol, as_of_date):
    """Calcula score basado en el transcript más reciente antes de la fecha."""
    cache_key = f"{symbol}_{as_of_date}"
    if cache_key in transcript_scores:
        return transcript_scores[cache_key]

    # Buscar transcript más reciente antes de la fecha
    cur.execute('''
        SELECT content, year, quarter
        FROM fmp_earnings_transcripts
        WHERE symbol = %s
        AND (year < %s OR (year = %s AND quarter <= %s))
        ORDER BY year DESC, quarter DESC
        LIMIT 1
    ''', (symbol, as_of_date.year, as_of_date.year, (as_of_date.month - 1) // 3 + 1))

    row = cur.fetchone()
    if not row:
        transcript_scores[cache_key] = 0
        return 0

    content = row[0]
    guidance = extract_guidance_section(content)

    # Contar keywords positivos y negativos
    positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in guidance)
    negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in guidance)

    # Score: diferencia normalizada (-10 a +10)
    total = positive_count + negative_count
    if total == 0:
        score = 0
    else:
        score = (positive_count - negative_count) / total * 10

    transcript_scores[cache_key] = score
    return score

# Cargar semanas
cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2024-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]
print(f'Periodo: {weeks[0]} a {weeks[-1]} ({len(weeks)} semanas)')

# Cargar sectores
cur.execute('SELECT symbol, sector FROM fmp_profiles WHERE sector IS NOT NULL')
SECTORS = {row[0]: row[1] for row in cur.fetchall()}

# Fórmulas a comparar
FORMULAS = {
    'MIN_DRAWDOWN': lambda peg, beat, eps, rev, ts:
        (1.0 - peg) * 60 + min(beat, 30) * 6 + min(eps, 80) * 0.2 + min(rev, 50) * 0.2,

    'WITH_TRANSCRIPT_V1': lambda peg, beat, eps, rev, ts:
        (1.0 - peg) * 50 + min(beat, 30) * 5 + min(eps, 80) * 0.2 + min(rev, 50) * 0.2 + ts * 3,

    'WITH_TRANSCRIPT_V2': lambda peg, beat, eps, rev, ts:
        (1.0 - peg) * 40 + min(beat, 30) * 4 + min(eps, 80) * 0.2 + min(rev, 50) * 0.2 + ts * 5,

    'TRANSCRIPT_HEAVY': lambda peg, beat, eps, rev, ts:
        (1.0 - peg) * 30 + min(beat, 30) * 3 + min(eps, 80) * 0.1 + min(rev, 50) * 0.1 + ts * 8,
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

        # Calcular transcript score
        ts = calculate_transcript_score(symbol, week_ending)

        score = formula_func(peg, beat, eps_g, rev_g, ts)
        stocks.append({'symbol': symbol, 'score': score, 'sector': SECTORS.get(symbol, 'Unknown'), 'ts': ts})

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
    transcript_used = 0

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

        # Contar cuántos tienen transcript score != 0
        with_ts = sum(1 for s in ranked[:MAX_POSITIONS] if s['ts'] != 0)
        transcript_used += with_ts

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
                price, dt = get_monday_price(symbol, week, 'close')
                if price:
                    gross = price * pos['shares']
                    comm = gross * COMMISSION_RATE
                    total_commissions += comm
                    pnl = gross - comm - pos['cost_basis']
                    pct = (pnl / pos['cost_basis']) * 100
                    duration = (dt - pos['entry_date']).days
                    closed_trades.append({'symbol': symbol, 'pnl': pnl, 'pct': pct, 'duration': duration})
                del positions[symbol]
                sells_done += 1

        max_buys = MAX_POSITIONS if is_first_week else MAX_CHANGES
        for symbol, _ in buys_list:
            if buys_done >= max_buys:
                break
            if len(positions) >= MAX_POSITIONS:
                break
            price, dt = get_monday_price(symbol, week, 'close')
            if price and price > 0:
                shares = round(POSITION_SIZE / price)
                if shares > 0:
                    gross = price * shares
                    comm = gross * COMMISSION_RATE
                    total_commissions += comm
                    positions[symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': dt,
                        'cost_basis': gross + comm
                    }
                    buys_done += 1

    # Close remaining
    for symbol, pos in list(positions.items()):
        price, dt = get_monday_price(symbol, weeks[-2], 'close')
        if price:
            gross = price * pos['shares']
            comm = gross * COMMISSION_RATE
            total_commissions += comm
            pnl = gross - comm - pos['cost_basis']
            pct = (pnl / pos['cost_basis']) * 100
            duration = (dt - pos['entry_date']).days if dt else 0
            closed_trades.append({'symbol': symbol, 'pnl': pnl, 'pct': pct, 'duration': duration})

    total_pnl = sum(t['pnl'] for t in closed_trades)
    winners = [t for t in closed_trades if t['pnl'] > 0]

    return {
        'formula': formula_name,
        'trades': len(closed_trades),
        'win_rate': len(winners) / len(closed_trades) * 100 if closed_trades else 0,
        'pnl': total_pnl,
        'pnl_per_trade': total_pnl / len(closed_trades) if closed_trades else 0,
        'avg_return': sum(t['pct'] for t in closed_trades) / len(closed_trades) if closed_trades else 0,
        'max_drawdown': max_drawdown,
        'transcript_coverage': transcript_used / (len(weeks) * MAX_POSITIONS) * 100,
    }

print('='*90)
print('BACKTEST CON TRANSCRIPT SCORING')
print('Periodo: 2024+ (transcripts disponibles desde 2024)')
print('='*90)

results = []
for name, func in FORMULAS.items():
    print(f'Probando {name}...', end=' ', flush=True)
    r = run_backtest(name, func)
    results.append(r)
    print(f'PnL/trade: ${r["pnl_per_trade"]:,.0f}, MaxDD: {r["max_drawdown"]:.1f}%, TS Coverage: {r["transcript_coverage"]:.0f}%')

results.sort(key=lambda x: x['pnl_per_trade'], reverse=True)

print('\n' + '='*90)
print('RANKING POR PnL/TRADE')
print('='*90)
print(f'\n{"#":<3} {"Formula":<20} {"Trades":<7} {"Win%":<7} {"PnL/Tr":<10} {"Ret%":<7} {"MaxDD":<8} {"TS Cov":<8}')
print('-'*90)

for i, r in enumerate(results):
    print(f'{i+1:<3} {r["formula"]:<20} {r["trades"]:<7} {r["win_rate"]:<6.1f}% ${r["pnl_per_trade"]:<8,.0f} {r["avg_return"]:<6.1f}% {r["max_drawdown"]:<7.1f}% {r["transcript_coverage"]:<7.0f}%')

# Comparación directa
print('\n' + '='*90)
print('MEJORA vs MIN_DRAWDOWN (baseline)')
print('='*90)
baseline = next(r for r in results if r['formula'] == 'MIN_DRAWDOWN')
for r in results:
    if r['formula'] != 'MIN_DRAWDOWN':
        diff = r['pnl_per_trade'] - baseline['pnl_per_trade']
        diff_pct = diff / baseline['pnl_per_trade'] * 100 if baseline['pnl_per_trade'] else 0
        dd_diff = r['max_drawdown'] - baseline['max_drawdown']
        print(f'{r["formula"]:<20}: PnL/trade {diff:+,.0f} ({diff_pct:+.1f}%), MaxDD {dd_diff:+.1f}pp')

cur.close()
conn.close()
