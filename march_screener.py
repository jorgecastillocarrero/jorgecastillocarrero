import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy
from datetime import datetime, timedelta

fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')
railway_engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

print("=" * 120)
print("  SCREENER ESTACIONALIDAD MARZO 2026")
print("  Filtros: WinRate >= 70% en Marzo (10 anios), MCap > $7B, US (NYSE/NASDAQ)")
print("=" * 120)

# STEP 1: Get universe of US stocks (NYSE + NASDAQ, no ETF, no OTC)
print("\n[1/5] Obteniendo universo de acciones US (NYSE + NASDAQ)...")
with fmp_engine.connect() as conn:
    r = conn.execute(sqlalchemy.text(
        "SELECT symbol FROM fmp_profiles "
        "WHERE country = 'US' AND exchange IN ('NYSE', 'NASDAQ') "
        "AND is_etf = false AND is_fund = false AND is_actively_trading = true"
    ))
    us_symbols = [row[0] for row in r.fetchall()]
    print(f"  Universo: {len(us_symbols)} acciones")

# STEP 2: Calculate March seasonality (win rate over 10 years: 2016-2025)
print("\n[2/5] Calculando estacionalidad de Marzo (2016-2025)...")
print("  Esto puede tardar unos minutos con 87M registros...")

with fmp_engine.connect() as conn:
    # For each year, get last trading day of Feb and last trading day of March
    # Then calculate return for each symbol
    # Efficient approach: single query with window functions

    r = conn.execute(sqlalchemy.text("""
        WITH feb_close AS (
            SELECT symbol, EXTRACT(YEAR FROM date) as yr,
                   close as feb_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol, EXTRACT(YEAR FROM date) ORDER BY date DESC) as rn
            FROM fmp_price_history
            WHERE EXTRACT(MONTH FROM date) = 2
            AND EXTRACT(YEAR FROM date) BETWEEN 2016 AND 2025
            AND symbol = ANY(:symbols)
        ),
        mar_close AS (
            SELECT symbol, EXTRACT(YEAR FROM date) as yr,
                   close as mar_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol, EXTRACT(YEAR FROM date) ORDER BY date DESC) as rn
            FROM fmp_price_history
            WHERE EXTRACT(MONTH FROM date) = 3
            AND EXTRACT(YEAR FROM date) BETWEEN 2016 AND 2025
            AND symbol = ANY(:symbols)
        )
        SELECT f.symbol,
               COUNT(*) as years,
               SUM(CASE WHEN m.mar_close > f.feb_close THEN 1 ELSE 0 END) as wins,
               ROUND(AVG((m.mar_close - f.feb_close) / NULLIF(f.feb_close, 0) * 100)::numeric, 2) as avg_return
        FROM feb_close f
        JOIN mar_close m ON f.symbol = m.symbol AND f.yr = m.yr AND m.rn = 1
        WHERE f.rn = 1
        GROUP BY f.symbol
        HAVING COUNT(*) >= 6
        ORDER BY SUM(CASE WHEN m.mar_close > f.feb_close THEN 1 ELSE 0 END)::float / COUNT(*) DESC,
                 AVG((m.mar_close - f.feb_close) / NULLIF(f.feb_close, 0) * 100) DESC
    """), {"symbols": us_symbols})

    candidates = []
    for row in r.fetchall():
        sym, years, wins, avg_ret = row
        win_rate = wins / years * 100
        if win_rate >= 70:
            candidates.append({
                'symbol': sym,
                'years': int(years),
                'wins': int(wins),
                'win_rate': round(win_rate, 0),
                'avg_ret': float(avg_ret) if avg_ret else 0
            })

    print(f"  Candidatos con WinRate >= 70%: {len(candidates)}")

# STEP 3: Filter by market cap (use Railway price_history for recent prices or FMP profiles)
print("\n[3/5] Filtrando por capitalizacion (MCap > $7B aprox)...")

# Use railway for current MA200 and last price
with railway_engine.connect() as conn:
    filtered = []
    for c in candidates:
        sym = c['symbol']

        # Get last price and MA200
        r = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
        row = r.fetchone()
        if not row:
            continue
        last_price = float(row[0])

        # MA200
        r2 = conn.execute(sqlalchemy.text(
            "SELECT AVG(sub.close) FROM ("
            "  SELECT ph.close FROM price_history ph "
            "  JOIN symbols s ON s.id = ph.symbol_id "
            "  WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 200"
            ") sub"
        ), {"sym": sym})
        row2 = r2.fetchone()
        if not row2 or not row2[0]:
            continue
        ma200 = float(row2[0])
        vs_ma200 = ((last_price - ma200) / ma200) * 100

        c['last_price'] = last_price
        c['ma200'] = ma200
        c['vs_ma200'] = vs_ma200
        filtered.append(c)

    print(f"  Con datos en Railway: {len(filtered)}")

# STEP 4: Get EPS data from FMP
print("\n[4/5] Obteniendo datos de EPS...")
with fmp_engine.connect() as conn:
    for c in filtered:
        sym = c['symbol']
        r = conn.execute(sqlalchemy.text(
            "SELECT eps_actual, eps_estimated, revenue_actual, revenue_estimated, date "
            "FROM fmp_earnings "
            "WHERE symbol = :sym AND date < '2026-02-27' "
            "ORDER BY date DESC LIMIT 1"
        ), {"sym": sym})
        row = r.fetchone()
        if row and row[0] is not None and row[1] is not None and row[1] != 0:
            eps_act, eps_est = float(row[0]), float(row[1])
            c['eps_surprise'] = ((eps_act - eps_est) / abs(eps_est)) * 100
            c['eps_beat'] = eps_act > eps_est

            # YoY
            r2 = conn.execute(sqlalchemy.text(
                "SELECT eps_actual FROM fmp_earnings "
                "WHERE symbol = :sym AND date < :cutoff "
                "ORDER BY date DESC LIMIT 1"
            ), {"sym": sym, "cutoff": str(row[4] - timedelta(days=300))})
            yoy_row = r2.fetchone()
            if yoy_row and yoy_row[0] and yoy_row[0] != 0:
                c['eps_yoy'] = ((eps_act - float(yoy_row[0])) / abs(float(yoy_row[0]))) * 100
            else:
                c['eps_yoy'] = None

            # Revenue surprise
            if row[2] and row[3] and row[3] != 0:
                c['rev_surprise'] = ((float(row[2]) - float(row[3])) / float(row[3])) * 100
            else:
                c['rev_surprise'] = None
        else:
            c['eps_surprise'] = None
            c['eps_beat'] = None
            c['eps_yoy'] = None
            c['rev_surprise'] = None

# Get sector from FMP profiles
with fmp_engine.connect() as conn:
    for c in filtered:
        r = conn.execute(sqlalchemy.text(
            "SELECT sector, industry FROM fmp_profiles WHERE symbol = :sym LIMIT 1"
        ), {"sym": c['symbol']})
        row = r.fetchone()
        if row:
            c['sector'] = row[0] if row[0] else 'N/A'
            c['industry'] = row[1] if row[1] else 'N/A'
        else:
            c['sector'] = 'N/A'
            c['industry'] = 'N/A'

# STEP 5: Apply filters and rank
print("\n[5/5] Aplicando filtros y ranking...")

# Filter: MA200 > 0, EPS Beat, Surprise > 5%, YoY > 0
combo_ideal = [c for c in filtered if
    c['vs_ma200'] > 0 and
    c.get('eps_beat') == True and
    c.get('eps_surprise') is not None and c['eps_surprise'] > 5 and
    c.get('eps_yoy') is not None and c['eps_yoy'] > 0]

combo_ma200_beat = [c for c in filtered if
    c['vs_ma200'] > 0 and
    c.get('eps_beat') == True]

only_ma200 = [c for c in filtered if c['vs_ma200'] > 0]

# Print all with MA200 > 0
print(f"\n{'='*160}")
print(f"  CANDIDATOS MARZO 2026 - MA200 > 0 ({len(only_ma200)} acciones)")
print(f"{'='*160}")

only_ma200.sort(key=lambda x: x['win_rate'] * 100 + x['avg_ret'], reverse=True)

print(f"  {'#':>2s}  {'Symbol':8s} | {'Sector':25s} | {'WR':>4s} | {'Anos':>4s} | {'AvgRet':>7s} | {'vsMA200':>7s} | {'EPSsurp':>7s} | {'EPSyoy':>7s} | {'Beat':>4s} | {'Precio':>8s}")
print(f"  {'':>2s}  {'-'*8} | {'-'*25} | {'-'*4} | {'-'*4} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*4} | {'-'*8}")

for i, c in enumerate(only_ma200):
    sv = "+" if c['vs_ma200'] >= 0 else ""
    se = f"+{c['eps_surprise']:>5.1f}%" if c.get('eps_surprise') is not None and c['eps_surprise'] >= 0 else (f"{c['eps_surprise']:>6.1f}%" if c.get('eps_surprise') is not None else "   N/A ")
    sy = f"+{c['eps_yoy']:>5.1f}%" if c.get('eps_yoy') is not None and c['eps_yoy'] >= 0 else (f"{c['eps_yoy']:>6.1f}%" if c.get('eps_yoy') is not None else "   N/A ")
    bt = " YES" if c.get('eps_beat') == True else ("  NO" if c.get('eps_beat') == False else " N/A")
    print(f"  {i+1:>2d}. {c['symbol']:8s} | {c['sector'][:25]:25s} | {c['win_rate']:>3.0f}% | {c['years']:>4d} | {c['avg_ret']:>+6.2f}% | {sv}{c['vs_ma200']:>5.1f}% | {se} | {sy} | {bt} | {c['last_price']:>8.2f}")

# Print COMBO IDEAL
print(f"\n\n{'='*160}")
print(f"  COMBO IDEAL MARZO 2026: MA200>0 + Beat + Surprise>5% + YoY>0 ({len(combo_ideal)} acciones)")
print(f"{'='*160}")

combo_ideal.sort(key=lambda x: x['win_rate'] * 100 + x['avg_ret'], reverse=True)

print(f"  {'#':>2s}  {'Symbol':8s} | {'Sector':25s} | {'Industry':25s} | {'WR':>4s} | {'AvgRet':>7s} | {'vsMA200':>7s} | {'EPSsurp':>7s} | {'EPSyoy':>7s} | {'Precio':>8s}")
print(f"  {'':>2s}  {'-'*8} | {'-'*25} | {'-'*25} | {'-'*4} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*8}")

for i, c in enumerate(combo_ideal):
    sv = "+" if c['vs_ma200'] >= 0 else ""
    print(f"  {i+1:>2d}. {c['symbol']:8s} | {c['sector'][:25]:25s} | {c['industry'][:25]:25s} | {c['win_rate']:>3.0f}% | {c['avg_ret']:>+6.2f}% | {sv}{c['vs_ma200']:>5.1f}% | +{c['eps_surprise']:>5.1f}% | +{c['eps_yoy']:>5.1f}% | {c['last_price']:>8.2f}")

# Sector distribution of combo ideal
print(f"\n  Distribucion sectorial:")
from collections import Counter
sector_counts = Counter(c['sector'] for c in combo_ideal)
for sec, cnt in sector_counts.most_common():
    print(f"    {sec:25s}: {cnt}")
