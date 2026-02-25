import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy
from datetime import timedelta
from collections import defaultdict

fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')
railway_engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

print("=" * 120)
print("  SCREENER MARZO 2026 - Lista amplia 30 candidatos diversificados")
print("=" * 120)

# STEP 1: Get full list of March seasonality candidates from FMP
print("\n[1/4] Calculando estacionalidad Marzo (2016-2025) para universo US NYSE+NASDAQ...")

with fmp_engine.connect() as conn:
    r = conn.execute(sqlalchemy.text(
        "SELECT symbol FROM fmp_profiles "
        "WHERE country = 'US' AND exchange IN ('NYSE', 'NASDAQ') "
        "AND is_etf = false AND is_fund = false AND is_actively_trading = true"
    ))
    us_symbols = [row[0] for row in r.fetchall()]
    print(f"  Universo: {len(us_symbols)} acciones")

    r = conn.execute(sqlalchemy.text("""
        WITH feb_close AS (
            SELECT symbol, EXTRACT(YEAR FROM date) as yr, close as feb_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol, EXTRACT(YEAR FROM date) ORDER BY date DESC) as rn
            FROM fmp_price_history
            WHERE EXTRACT(MONTH FROM date) = 2 AND EXTRACT(YEAR FROM date) BETWEEN 2016 AND 2025
            AND symbol = ANY(:symbols)
        ),
        mar_close AS (
            SELECT symbol, EXTRACT(YEAR FROM date) as yr, close as mar_close,
                   ROW_NUMBER() OVER (PARTITION BY symbol, EXTRACT(YEAR FROM date) ORDER BY date DESC) as rn
            FROM fmp_price_history
            WHERE EXTRACT(MONTH FROM date) = 3 AND EXTRACT(YEAR FROM date) BETWEEN 2016 AND 2025
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
           AND SUM(CASE WHEN m.mar_close > f.feb_close THEN 1 ELSE 0 END)::float / COUNT(*) >= 0.7
        ORDER BY SUM(CASE WHEN m.mar_close > f.feb_close THEN 1 ELSE 0 END)::float / COUNT(*) DESC,
                 AVG((m.mar_close - f.feb_close) / NULLIF(f.feb_close, 0) * 100) DESC
    """), {"symbols": us_symbols})

    seasonality = {}
    for row in r.fetchall():
        sym, years, wins, avg_ret = row
        seasonality[sym] = {
            'years': int(years), 'wins': int(wins),
            'win_rate': round(wins / years * 100, 0),
            'avg_ret': float(avg_ret) if avg_ret else 0
        }
    print(f"  Candidatos WR >= 70%: {len(seasonality)}")

# STEP 2: Enrich with sector, market cap, MA200, EPS
print("\n[2/4] Enriqueciendo con sector, MCap, MA200, EPS...")

all_data = []
with fmp_engine.connect() as fconn, railway_engine.connect() as rconn:
    for sym, seas in seasonality.items():
        # Profile
        r = fconn.execute(sqlalchemy.text(
            "SELECT sector, industry, mkt_cap FROM fmp_profiles WHERE symbol = :sym LIMIT 1"
        ), {"sym": sym})
        prof = r.fetchone()
        if not prof:
            continue
        sector = prof[0] if prof[0] else 'N/A'
        industry = prof[1] if prof[1] else 'N/A'
        mkt_cap = float(prof[2]) / 1e9 if prof[2] else None

        # Railway: mkt_cap fallback + last price + MA200
        r2 = rconn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
        row2 = r2.fetchone()
        if not row2:
            continue
        last_price = float(row2[0])

        if mkt_cap is None or mkt_cap < 1:
            r_mc = rconn.execute(sqlalchemy.text(
                "SELECT market_cap FROM fundamentals "
                "WHERE symbol_id = (SELECT id FROM symbols WHERE code = :sym) "
                "ORDER BY data_date DESC LIMIT 1"
            ), {"sym": sym})
            mc_row = r_mc.fetchone()
            if mc_row and mc_row[0]:
                mkt_cap = float(mc_row[0]) / 1e9

        if mkt_cap is None or mkt_cap < 7:
            continue

        r3 = rconn.execute(sqlalchemy.text(
            "SELECT AVG(sub.close) FROM ("
            "  SELECT ph.close FROM price_history ph "
            "  JOIN symbols s ON s.id = ph.symbol_id "
            "  WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 200"
            ") sub"
        ), {"sym": sym})
        row3 = r3.fetchone()
        if not row3 or not row3[0]:
            continue
        ma200 = float(row3[0])
        vs_ma200 = ((last_price - ma200) / ma200) * 100

        if vs_ma200 <= 0:
            continue

        # EPS from FMP
        r4 = fconn.execute(sqlalchemy.text(
            "SELECT eps_actual, eps_estimated, revenue_actual, revenue_estimated, date "
            "FROM fmp_earnings WHERE symbol = :sym AND date < '2026-02-27' "
            "ORDER BY date DESC LIMIT 1"
        ), {"sym": sym})
        eps_row = r4.fetchone()
        eps_surprise = None
        eps_beat = None
        eps_yoy = None

        if eps_row and eps_row[0] is not None and eps_row[1] is not None and eps_row[1] != 0:
            eps_act, eps_est = float(eps_row[0]), float(eps_row[1])
            eps_surprise = ((eps_act - eps_est) / abs(eps_est)) * 100
            eps_beat = eps_act > eps_est

            r5 = fconn.execute(sqlalchemy.text(
                "SELECT eps_actual FROM fmp_earnings "
                "WHERE symbol = :sym AND date < :cutoff ORDER BY date DESC LIMIT 1"
            ), {"sym": sym, "cutoff": str(eps_row[4] - timedelta(days=300))})
            yoy_row = r5.fetchone()
            if yoy_row and yoy_row[0] and yoy_row[0] != 0:
                eps_yoy = ((eps_act - float(yoy_row[0])) / abs(float(yoy_row[0]))) * 100

        all_data.append({
            'symbol': sym, 'sector': sector, 'industry': industry[:30],
            'mkt_cap': mkt_cap, 'last_price': last_price,
            'win_rate': seas['win_rate'], 'avg_ret': seas['avg_ret'],
            'years': seas['years'], 'wins': seas['wins'],
            'ma200': ma200, 'vs_ma200': vs_ma200,
            'eps_surprise': eps_surprise, 'eps_beat': eps_beat, 'eps_yoy': eps_yoy,
        })

print(f"  Total con MCap>$7B y MA200>0: {len(all_data)}")

# STEP 3: Score and select top 30 diversified
print("\n[3/4] Scoring y seleccion diversificada de 30...")

# Score: combo_ideal gets highest, then WR, then avg_ret
for d in all_data:
    score = 0
    score += d['win_rate'] * 2  # WR weight
    score += d['avg_ret'] * 3   # avg return weight
    score += min(d['vs_ma200'], 50) * 0.5  # MA200 strength (cap at 50%)
    if d['eps_beat'] is True:
        score += 15
    if d['eps_surprise'] is not None and d['eps_surprise'] > 5:
        score += 10
    if d['eps_yoy'] is not None and d['eps_yoy'] > 0:
        score += 10
    d['score'] = score

all_data.sort(key=lambda x: x['score'], reverse=True)

# Select top 30 ensuring sector diversity (max 5 per sector, at least 7 sectors)
selected = []
sector_count = defaultdict(int)
MAX_PER_SECTOR = 5

# First pass: pick best from each sector
sectors_seen = set()
for d in all_data:
    sec = d['sector']
    if sec not in sectors_seen and len(selected) < 30:
        selected.append(d)
        sectors_seen.add(sec)
        sector_count[sec] += 1

# Second pass: fill to 30 with best scores respecting max per sector
for d in all_data:
    if d in selected:
        continue
    sec = d['sector']
    if sector_count[sec] < MAX_PER_SECTOR and len(selected) < 30:
        selected.append(d)
        sector_count[sec] += 1

selected.sort(key=lambda x: x['score'], reverse=True)

# STEP 4: Print
print(f"\n[4/4] Lista final de 30 candidatos:\n")

print("=" * 165)
print("  30 CANDIDATOS MARZO 2026 - Diversificados por sector")
print("  Criterios: WR>=70% (10a), MA200>0, MCap>$7B, diversificacion sectorial")
print("=" * 165)
print(f"  {'#':>2s}  {'Symbol':8s} | {'Sector':22s} | {'Industry':30s} | {'MCap$B':>7s} | {'WR':>4s} | {'AvgRet':>7s} | {'vsMA200':>7s} | {'EPSsurp':>7s} | {'EPSyoy':>7s} | {'Score':>5s}")
print(f"  {'':>2s}  {'-'*8} | {'-'*22} | {'-'*30} | {'-'*7} | {'-'*4} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*5}")

for i, d in enumerate(selected):
    mc = f"{d['mkt_cap']:>6.1f}" if d['mkt_cap'] else "   N/A"
    sv = "+" if d['vs_ma200'] >= 0 else ""
    se = f"+{d['eps_surprise']:>5.1f}%" if d.get('eps_surprise') is not None and d['eps_surprise'] >= 0 else (f"{d['eps_surprise']:>6.1f}%" if d.get('eps_surprise') is not None else "   N/A ")
    sy = f"+{d['eps_yoy']:>5.1f}%" if d.get('eps_yoy') is not None and d['eps_yoy'] >= 0 else (f"{d['eps_yoy']:>6.1f}%" if d.get('eps_yoy') is not None else "   N/A ")

    combo = ""
    if d['eps_beat'] is True and d.get('eps_surprise') is not None and d['eps_surprise'] > 5 and d.get('eps_yoy') is not None and d['eps_yoy'] > 0 and d['vs_ma200'] > 0:
        combo = " ***"

    sec = (d['sector'] if d['sector'] else 'N/A')[:22]
    print(f"  {i+1:>2d}. {d['symbol']:8s} | {sec:22s} | {d['industry']:30s} | {mc} | {d['win_rate']:>3.0f}% | {d['avg_ret']:>+6.2f}% | {sv}{d['vs_ma200']:>5.1f}% | {se} | {sy} | {d['score']:>5.0f}{combo}")

print(f"\n  *** = COMBO IDEAL (MA200>0 + Beat + Surprise>5% + YoY>0)")

# Sector summary
print(f"\n  Distribucion sectorial:")
sec_summary = defaultdict(list)
for d in selected:
    sec_summary[d['sector']].append(d['symbol'])
for sec in sorted(sec_summary.keys(), key=lambda s: len(sec_summary[s]), reverse=True):
    syms = sec_summary[sec]
    print(f"    {sec:25s} ({len(syms)}): {', '.join(syms)}")

# Highlight gold/mining specifically
print(f"\n  Oro / Mineras en la lista:")
gold = [d for d in selected if 'gold' in (d['industry'] or '').lower() or 'mining' in (d['industry'] or '').lower() or 'precious' in (d['industry'] or '').lower()]
if gold:
    for d in gold:
        print(f"    {d['symbol']:8s} | {d['industry']:30s} | WR {d['win_rate']:.0f}% | MA200 +{d['vs_ma200']:.1f}%")
else:
    print("    Ninguna directamente de oro/mineria en top 30. Candidatos de oro disponibles:")
    # Show gold candidates that didn't make the cut
    gold_all = [d for d in all_data if 'gold' in (d['industry'] or '').lower() or 'mining' in (d['industry'] or '').lower() or 'precious' in (d['industry'] or '').lower() or 'silver' in (d['industry'] or '').lower()]
    for d in gold_all[:5]:
        se = f"+{d['eps_surprise']:>5.1f}%" if d.get('eps_surprise') is not None and d['eps_surprise'] >= 0 else "N/A"
        print(f"    {d['symbol']:8s} | {d['industry']:30s} | MCap {d['mkt_cap']:.1f}B | WR {d['win_rate']:.0f}% | AvgRet {d['avg_ret']:+.2f}% | MA200 +{d['vs_ma200']:.1f}% | EPSsurp {se}")
