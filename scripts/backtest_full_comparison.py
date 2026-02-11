"""
Comparación completa de estrategias de backtesting.
Incluye: ORIGINAL, MIN_DRAWDOWN, MAX_RETURN, SQN, y versiones con TRANSCRIPTS.
Compara con límite de 2 cambios/semana vs sin límite (todos los trades).
"""
import psycopg2
import re
import math
from datetime import timedelta
from collections import defaultdict

conn = psycopg2.connect('postgresql://fmp:fmp123@localhost:5433/fmp_data')
cur = conn.cursor()

COMMISSION_RATE = 0.003
POSITION_SIZE = 20000
MAX_POSITIONS = 50

# Keywords para análisis de transcripts
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
    'concern', 'risk', 'delay', 'inventory buildup', 'margin compression'
]

# Cache de transcript scores
transcript_cache = {}

def extract_guidance_section(content):
    content_lower = content.lower()
    patterns = [
        r'(outlook|guidance|looking ahead|for the (fourth|first|second|third) quarter)',
        r'(we expect|we anticipate|our expectation)',
        r'(revenue is expected|we are guiding)'
    ]
    best_start = -1
    for pattern in patterns:
        match = re.search(pattern, content_lower)
        if match and (best_start == -1 or match.start() < best_start):
            best_start = match.start()
    if best_start == -1:
        best_start = int(len(content) * 0.8)
    return content[max(0, best_start - 200):best_start + 2000].lower()

def get_transcript_score(symbol, as_of_date):
    cache_key = f"{symbol}_{as_of_date}"
    if cache_key in transcript_cache:
        return transcript_cache[cache_key]

    # quarter es varchar 'Q1', 'Q2', etc - extraer número
    q_num = (as_of_date.month - 1) // 3 + 1
    q_str = f'Q{q_num}'

    cur.execute('''
        SELECT content FROM fmp_earnings_transcripts
        WHERE symbol = %s AND (year < %s OR (year = %s AND quarter <= %s))
        ORDER BY year DESC, quarter DESC LIMIT 1
    ''', (symbol, as_of_date.year, as_of_date.year, q_str))

    row = cur.fetchone()
    if not row:
        transcript_cache[cache_key] = 0
        return 0

    guidance = extract_guidance_section(row[0])
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in guidance)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in guidance)
    total = pos + neg
    score = (pos - neg) / total * 10 if total > 0 else 0
    transcript_cache[cache_key] = score
    return score

# Cargar datos
print('Cargando datos...', flush=True)
cur.execute('''
    SELECT DISTINCT week_ending FROM market_cap_weekly
    WHERE week_ending >= '2023-01-01'
    ORDER BY week_ending
''')
weeks = [row[0] for row in cur.fetchall()]

cur.execute('SELECT symbol, sector FROM fmp_profiles WHERE sector IS NOT NULL')
SECTORS = {row[0]: row[1] for row in cur.fetchall()}

# FORMULAS
# ts = transcript score (0 si no se usa)
FORMULAS = {
    # 1. ORIGINAL: fórmula base balanceada
    'ORIGINAL': lambda peg, beat, eps, rev, ts:
        (1.5 - peg) * 50 + min(eps, 150) * 0.4 + min(rev, 100) * 0.6 + min(beat, 15) * 2,

    # 2. MIN_DRAWDOWN: prioriza estabilidad
    'MIN_DRAWDOWN': lambda peg, beat, eps, rev, ts:
        (1.0 - peg) * 60 + min(beat, 30) * 6 + min(eps, 80) * 0.2 + min(rev, 50) * 0.2,

    # 3. MAX_RETURN: maximiza crecimiento agresivo
    'MAX_RETURN': lambda peg, beat, eps, rev, ts:
        (1.5 - peg) * 30 + min(eps, 200) * 0.8 + min(rev, 150) * 0.8 + min(beat, 10) * 1,

    # 4. SQN_OPTIMIZED: balance entre consistencia y retorno (bajo PEG + alto beat)
    'SQN_OPTIMIZED': lambda peg, beat, eps, rev, ts:
        (1.2 - peg) * 70 + min(beat, 25) * 5 + min(eps, 100) * 0.3 + min(rev, 80) * 0.3,

    # 5. ORIGINAL + TRANSCRIPTS: original mejorado con sentimiento
    'ORIG_TRANSCRIPT': lambda peg, beat, eps, rev, ts:
        (1.5 - peg) * 40 + min(eps, 150) * 0.3 + min(rev, 100) * 0.5 + min(beat, 15) * 2 + ts * 4,
}

def get_ranked_stocks(week_ending, formula_func, use_transcripts=False):
    cur.execute('''
        SELECT m.symbol, b.beat_streak, p.peg_ratio, e.eps_growth_yoy, r.revenue_growth_yoy
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
        symbol, beat, peg, eps_g, rev_g = row
        peg = float(peg) if peg else 1.5
        beat = int(beat) if beat else 0
        eps_g = float(eps_g) if eps_g else 0
        rev_g = float(rev_g) if rev_g else 0
        ts = get_transcript_score(symbol, week_ending) if use_transcripts else 0
        score = formula_func(peg, beat, eps_g, rev_g, ts)
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

def calculate_sqn(trades):
    """System Quality Number = (Mean R / StdDev R) * sqrt(N)"""
    if len(trades) < 2:
        return 0
    returns = [t['pct'] for t in trades]
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = math.sqrt(variance) if variance > 0 else 1
    sqn = (mean_r / std_r) * math.sqrt(len(trades))
    return sqn

def run_backtest(formula_name, formula_func, max_changes, use_transcripts=False):
    positions = {}
    closed_trades = []
    total_commissions = 0
    portfolio_values = []
    peak_value = 0
    max_drawdown = 0

    # Tracking por año - PnL realizado
    initial_capital = POSITION_SIZE * MAX_POSITIONS  # Capital inicial: $1,000,000

    for i, week in enumerate(weeks[:-1]):
        # Calcular valor portfolio
        portfolio_value = 0
        for sym, pos in positions.items():
            price, _ = get_monday_price(sym, week, 'close')
            if price:
                portfolio_value += price * pos['shares']

        if portfolio_value > 0:
            portfolio_values.append({'week': week, 'value': portfolio_value})
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            dd = (peak_value - portfolio_value) / peak_value * 100 if peak_value > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        ranked = get_ranked_stocks(week, formula_func, use_transcripts)
        top50_symbols = set(s['symbol'] for s in ranked[:MAX_POSITIONS])
        ranked_dict = {s['symbol']: s['score'] for s in ranked}

        current_symbols = set(positions.keys())
        is_first_week = (i == 0)

        to_sell = current_symbols - top50_symbols
        to_buy = top50_symbols - current_symbols

        sells_list = sorted([(s, ranked_dict.get(s, -9999)) for s in to_sell], key=lambda x: x[1])
        buys_list = sorted([(s, ranked_dict.get(s, 0)) for s in to_buy], key=lambda x: x[1], reverse=True)

        # Ejecutar ventas
        max_sells = len(sells_list) if is_first_week else (len(sells_list) if max_changes == 0 else max_changes)
        sells_done = 0
        for symbol, _ in sells_list:
            if max_changes > 0 and sells_done >= max_sells:
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
                    closed_trades.append({'symbol': symbol, 'pnl': pnl, 'pct': pct, 'duration': duration, 'close_year': dt.year if dt else week.year})
                del positions[symbol]
                sells_done += 1

        # Ejecutar compras
        max_buys = MAX_POSITIONS if is_first_week else (len(buys_list) if max_changes == 0 else max_changes)
        buys_done = 0
        for symbol, _ in buys_list:
            if max_changes > 0 and buys_done >= max_buys:
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

    # Cerrar posiciones restantes
    for symbol, pos in list(positions.items()):
        price, dt = get_monday_price(symbol, weeks[-2], 'close')
        if price:
            gross = price * pos['shares']
            comm = gross * COMMISSION_RATE
            total_commissions += comm
            pnl = gross - comm - pos['cost_basis']
            pct = (pnl / pos['cost_basis']) * 100
            duration = (dt - pos['entry_date']).days if dt else 0
            closed_trades.append({'symbol': symbol, 'pnl': pnl, 'pct': pct, 'duration': duration, 'close_year': dt.year if dt else weeks[-2].year})

    # Calcular métricas
    if not closed_trades:
        return None

    total_pnl = sum(t['pnl'] for t in closed_trades)
    winners = [t for t in closed_trades if t['pnl'] > 0]
    losers = [t for t in closed_trades if t['pnl'] <= 0]

    avg_winner = sum(t['pnl'] for t in winners) / len(winners) if winners else 0
    avg_loser = abs(sum(t['pnl'] for t in losers) / len(losers)) if losers else 1
    profit_factor = (sum(t['pnl'] for t in winners) / abs(sum(t['pnl'] for t in losers))) if losers and sum(t['pnl'] for t in losers) != 0 else 0

    # Calcular PnL por año (basado en trades cerrados)
    yearly_pnl = defaultdict(float)
    yearly_trades = defaultdict(int)
    for t in closed_trades:
        yearly_pnl[t['close_year']] += t['pnl']
        yearly_trades[t['close_year']] += 1

    # Rentabilidad anual = PnL anual / capital inicial
    yearly_returns = {}
    yearly_pnl_values = {}
    for year in sorted(yearly_pnl.keys()):
        yearly_pnl_values[year] = yearly_pnl[year]
        yearly_returns[year] = (yearly_pnl[year] / initial_capital) * 100

    # Calcular CAGR por trade (anualizado por duración)
    # CAGR_trade = ((1 + return%) ^ (365/días)) - 1
    cagr_per_trade = []
    for t in closed_trades:
        if t['duration'] > 0:
            ret_decimal = t['pct'] / 100
            annualized = ((1 + ret_decimal) ** (365 / t['duration']) - 1) * 100
            cagr_per_trade.append(annualized)

    avg_cagr = sum(cagr_per_trade) / len(cagr_per_trade) if cagr_per_trade else 0

    # También calcular CAGR del portfolio total (para referencia)
    first_date = weeks[0]
    last_date = weeks[-2]
    years_elapsed = (last_date - first_date).days / 365.25
    if years_elapsed > 0:
        total_return = (initial_capital + total_pnl) / initial_capital
        portfolio_cagr = (total_return ** (1 / years_elapsed) - 1) * 100
    else:
        portfolio_cagr = 0

    return {
        'formula': formula_name,
        'trades': len(closed_trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(closed_trades) * 100,
        'total_pnl': total_pnl,
        'pnl_per_trade': total_pnl / len(closed_trades),
        'avg_return': sum(t['pct'] for t in closed_trades) / len(closed_trades),
        'max_drawdown': max_drawdown,
        'sqn': calculate_sqn(closed_trades),
        'profit_factor': profit_factor,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'commissions': total_commissions,
        'avg_duration': sum(t['duration'] for t in closed_trades) / len(closed_trades),
        'yearly_returns': yearly_returns,
        'yearly_pnl': yearly_pnl_values,
        'yearly_trades': dict(yearly_trades),
        'cagr': avg_cagr,  # CAGR promedio por trade (anualizado)
        'portfolio_cagr': portfolio_cagr,  # CAGR del portfolio total
    }

# ============================================================================
# EJECUTAR BACKTESTS
# ============================================================================
print(f'\nPeriodo: {weeks[0]} a {weeks[-1]} ({len(weeks)} semanas)')
print('='*100)

results_2changes = []
results_unlimited = []

formulas_to_test = [
    ('ORIGINAL', FORMULAS['ORIGINAL'], False),
    ('MIN_DRAWDOWN', FORMULAS['MIN_DRAWDOWN'], False),
    ('MAX_RETURN', FORMULAS['MAX_RETURN'], False),
    ('SQN_OPTIMIZED', FORMULAS['SQN_OPTIMIZED'], False),
    ('ORIG_TRANSCRIPT', FORMULAS['ORIG_TRANSCRIPT'], True),
]

print('\n>>> BACKTEST CON MAX 2 CAMBIOS/SEMANA <<<')
print('-'*100)
for name, func, use_ts in formulas_to_test:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, func, max_changes=2, use_transcripts=use_ts)
    if r:
        results_2changes.append(r)
        print(f'PnL/trade: ${r["pnl_per_trade"]:,.0f}, WinRate: {r["win_rate"]:.1f}%, SQN: {r["sqn"]:.2f}')

print('\n>>> BACKTEST SIN LIMITE DE CAMBIOS <<<')
print('-'*100)
for name, func, use_ts in formulas_to_test:
    print(f'  {name}...', end=' ', flush=True)
    r = run_backtest(name, func, max_changes=0, use_transcripts=use_ts)
    if r:
        results_unlimited.append(r)
        print(f'PnL/trade: ${r["pnl_per_trade"]:,.0f}, WinRate: {r["win_rate"]:.1f}%, SQN: {r["sqn"]:.2f}')

# ============================================================================
# MOSTRAR RESULTADOS
# ============================================================================
def print_results_table(results, title):
    print('\n' + '='*160)
    print(title)
    print('='*160)

    # Ordenar por CAGR
    results_sorted = sorted(results, key=lambda x: x['cagr'], reverse=True)

    # Obtener todos los años
    all_years = set()
    for r in results:
        all_years.update(r['yearly_pnl'].keys())
    years_sorted = sorted(all_years)

    # Header con años (PnL en $)
    year_cols = ''.join([f'{y:>12}' for y in years_sorted])
    print(f'\n{"#":<2} {"ESTRATEGIA":<16} {"CAGR/TR":>8} {"PORT CAGR":>10} {"TOTAL PNL":>12} {year_cols} {"MAXDD":>7} {"SQN":>6} {"WIN%":>6}')
    print('-'*180)

    for i, r in enumerate(results_sorted):
        year_vals = ''.join([f'${r["yearly_pnl"].get(y, 0):>10,.0f}' for y in years_sorted])
        print(f'{i+1:<2} {r["formula"]:<16} {r["cagr"]:>7.1f}% {r["portfolio_cagr"]:>9.1f}% ${r["total_pnl"]:>10,.0f} {year_vals} {r["max_drawdown"]:>6.1f}% {r["sqn"]:>6.2f} {r["win_rate"]:>5.1f}%')

    # Resumen
    print('\n' + '-'*160)
    print(f'Capital inicial: ${POSITION_SIZE * MAX_POSITIONS:,} | Periodo: {len(weeks)} semanas (~{len(weeks)/52:.1f} años)')

    # Mejor por cada métrica
    print('\nMEJOR POR METRICA:')
    best_cagr = max(results, key=lambda x: x['cagr'])
    best_pnl = max(results, key=lambda x: x['total_pnl'])
    best_sqn = max(results, key=lambda x: x['sqn'])
    best_dd = min(results, key=lambda x: x['max_drawdown'])

    print(f'  CAGR/Trade:    {best_cagr["formula"]} ({best_cagr["cagr"]:.1f}%)')
    print(f'  Total PnL:     {best_pnl["formula"]} (${best_pnl["total_pnl"]:,.0f})')
    print(f'  SQN:           {best_sqn["formula"]} ({best_sqn["sqn"]:.2f})')
    print(f'  Min Drawdown:  {best_dd["formula"]} ({best_dd["max_drawdown"]:.1f}%)')

print_results_table(results_2changes, 'RESULTADOS CON MAX 2 CAMBIOS/SEMANA')
print_results_table(results_unlimited, 'RESULTADOS SIN LIMITE DE CAMBIOS (TODOS LOS TRADES)')

# ============================================================================
# COMPARACION ORIGINAL vs ORIG_TRANSCRIPT
# ============================================================================
print('\n' + '='*120)
print('COMPARACION: ORIGINAL vs ORIGINAL + TRANSCRIPTS')
print('='*120)

for mode, results in [('2 cambios/semana', results_2changes), ('Sin limite', results_unlimited)]:
    orig = next((r for r in results if r['formula'] == 'ORIGINAL'), None)
    orig_ts = next((r for r in results if r['formula'] == 'ORIG_TRANSCRIPT'), None)

    if orig and orig_ts:
        print(f'\n>>> {mode} <<<')
        print(f'{"METRICA":<20} {"ORIGINAL":>12} {"ORIG+TRANSCRIPT":>15} {"DIFERENCIA":>12} {"MEJOR":>10}')
        print('-'*75)

        metrics = [
            ('CAGR %', orig['cagr'], orig_ts['cagr'], True),
            ('Trades', orig['trades'], orig_ts['trades'], False),
            ('Win Rate %', orig['win_rate'], orig_ts['win_rate'], True),
            ('PnL/Trade $', orig['pnl_per_trade'], orig_ts['pnl_per_trade'], True),
            ('Max Drawdown %', orig['max_drawdown'], orig_ts['max_drawdown'], False),
            ('SQN', orig['sqn'], orig_ts['sqn'], True),
            ('Profit Factor', orig['profit_factor'], orig_ts['profit_factor'], True),
        ]

        for name, v1, v2, higher_better in metrics:
            diff = v2 - v1
            if 'PnL' in name or '$' in name:
                print(f'{name:<20} ${v1:>10,.0f} ${v2:>14,.0f} {diff:>+11,.0f} {"TRANSCRIPT" if (diff > 0) == higher_better else "ORIGINAL":>10}')
            elif '%' in name:
                print(f'{name:<20} {v1:>11.1f}% {v2:>14.1f}% {diff:>+10.1f}pp {"TRANSCRIPT" if (diff > 0) == higher_better else "ORIGINAL":>10}')
            else:
                print(f'{name:<20} {v1:>12.2f} {v2:>15.2f} {diff:>+12.2f} {"TRANSCRIPT" if (diff > 0) == higher_better else "ORIGINAL":>10}')

        # PnL anual en dinero
        all_years = sorted(set(orig['yearly_pnl'].keys()) | set(orig_ts['yearly_pnl'].keys()))
        print(f'\n  PnL por año (en $):')
        print(f'  {"AÑO":<6} {"ORIG TRADES":>12} {"ORIG PNL":>12} {"TS TRADES":>12} {"TS PNL":>12} {"DIFERENCIA":>12} {"MEJOR":>10}')
        for year in all_years:
            o_pnl = orig['yearly_pnl'].get(year, 0)
            t_pnl = orig_ts['yearly_pnl'].get(year, 0)
            o_trades = orig['yearly_trades'].get(year, 0)
            t_trades = orig_ts['yearly_trades'].get(year, 0)
            diff = t_pnl - o_pnl
            mejor = "TRANSCRIPT" if t_pnl > o_pnl else ("ORIGINAL" if o_pnl > t_pnl else "IGUAL")
            print(f'  {year:<6} {o_trades:>12} ${o_pnl:>10,.0f} {t_trades:>12} ${t_pnl:>10,.0f} ${diff:>+10,.0f} {mejor:>10}')

cur.close()
conn.close()

print('\n' + '='*120)
print('BACKTEST COMPLETADO')
print('='*120)
