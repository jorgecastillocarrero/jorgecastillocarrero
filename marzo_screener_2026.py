import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy
from datetime import timedelta
from collections import defaultdict

fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')
railway_engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway',
    pool_pre_ping=True, pool_recycle=60)

print("=" * 140)
print("  SCREENER CARTERA MENSUAL MARZO 2026")
print("  Criterios: Estacionalidad WR>=70% (10a) + MA200>0 + MCap>$7B + EPS + Diversificacion sectorial")
print("=" * 140)

# ================================================================
# PASO 1: ESTACIONALIDAD MARZO (FMP - local, rapido)
# ================================================================
print("\n[1/4] Calculando estacionalidad Marzo (2016-2025)...")

with fmp_engine.connect() as conn:
    r = conn.execute(sqlalchemy.text(
        "SELECT symbol FROM fmp_profiles "
        "WHERE country = 'US' AND exchange IN ('NYSE', 'NASDAQ') "
        "AND is_etf = false AND is_fund = false AND is_actively_trading = true"
    ))
    us_symbols = [row[0] for row in r.fetchall()]
    print(f"  Universo: {len(us_symbols)} acciones US")

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
        ORDER BY SUM(CASE WHEN m.mar_close > f.feb_close THEN 1 ELSE 0 END)::float / COUNT(*) DESC
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

# ================================================================
# PASO 2: FILTRAR CON FMP (profiles + EPS) - todo local
# ================================================================
print("\n[2/4] Filtrando con FMP (profiles, MCap, EPS)...")

seas_symbols = list(seasonality.keys())

with fmp_engine.connect() as fconn:
    # Batch profiles (sector/industry)
    tlist = "','".join(seas_symbols)
    r = fconn.execute(sqlalchemy.text(
        f"SELECT symbol, sector, industry FROM fmp_profiles WHERE symbol IN ('{tlist}')"
    ))
    prof_raw = {}
    for row in r.fetchall():
        sym, sector, industry = row
        prof_raw[sym] = {
            'sector': sector if sector else 'N/A',
            'industry': (industry if industry else 'N/A')[:35],
        }

    # Market cap from fmp_key_metrics (latest per symbol)
    r_mc = fconn.execute(sqlalchemy.text(f"""
        SELECT DISTINCT ON (symbol) symbol, market_cap
        FROM fmp_key_metrics
        WHERE symbol IN ('{tlist}') AND market_cap IS NOT NULL AND market_cap > 0
        ORDER BY symbol, date DESC
    """))
    mcaps = {}
    for row in r_mc.fetchall():
        mcaps[row[0]] = float(row[1]) / 1e9

    profiles = {}
    for sym in seas_symbols:
        if sym not in prof_raw:
            continue
        mc = mcaps.get(sym)
        if mc is not None and mc >= 7:
            profiles[sym] = {
                'sector': prof_raw[sym]['sector'],
                'industry': prof_raw[sym]['industry'],
                'mkt_cap': mc
            }
    print(f"  Con MCap >= $7B: {len(profiles)}")

    # Batch EPS
    prof_syms = list(profiles.keys())
    tlist2 = "','".join(prof_syms)
    # Latest earnings per symbol
    r_eps = fconn.execute(sqlalchemy.text(f"""
        SELECT DISTINCT ON (symbol) symbol, eps_actual, eps_estimated,
               revenue_actual, revenue_estimated, date
        FROM fmp_earnings
        WHERE symbol IN ('{tlist2}') AND date < '2026-02-28'
        ORDER BY symbol, date DESC
    """))
    eps_data = {}
    for row in r_eps.fetchall():
        sym, eps_act, eps_est, rev_act, rev_est, dt = row
        eps_surprise = None
        eps_beat = None
        eps_yoy = None
        rev_surprise = None
        if eps_act is not None and eps_est is not None and eps_est != 0:
            eps_surprise = ((float(eps_act) - float(eps_est)) / abs(float(eps_est))) * 100
            eps_beat = float(eps_act) > float(eps_est)
            if rev_act and rev_est and rev_est != 0:
                rev_surprise = ((float(rev_act) - float(rev_est)) / float(rev_est)) * 100
        eps_data[sym] = {
            'eps_surprise': eps_surprise, 'eps_beat': eps_beat,
            'eps_yoy': None, 'rev_surprise': rev_surprise, 'eps_date': dt
        }

    # EPS YoY (need previous year)
    for sym, ed in eps_data.items():
        if ed['eps_surprise'] is not None and ed['eps_date']:
            cutoff = str(ed['eps_date'] - timedelta(days=300))
            r_yoy = fconn.execute(sqlalchemy.text(
                "SELECT eps_actual FROM fmp_earnings "
                "WHERE symbol = :sym AND date < :cutoff ORDER BY date DESC LIMIT 1"
            ), {'sym': sym, 'cutoff': cutoff})
            yoy_row = r_yoy.fetchone()
            if yoy_row and yoy_row[0] and yoy_row[0] != 0:
                # Get current eps_actual
                r_cur = fconn.execute(sqlalchemy.text(
                    "SELECT eps_actual FROM fmp_earnings "
                    "WHERE symbol = :sym AND date < '2026-02-28' ORDER BY date DESC LIMIT 1"
                ), {'sym': sym})
                cur = r_cur.fetchone()
                if cur and cur[0]:
                    ed['eps_yoy'] = ((float(cur[0]) - float(yoy_row[0])) / abs(float(yoy_row[0]))) * 100

# ================================================================
# PASO 3: MA200 Y PRECIOS DESDE RAILWAY (batch por lotes)
# ================================================================
print("\n[3/4] Obteniendo MA200 y precios de Railway (por lotes)...")

candidates_syms = [s for s in prof_syms if s in profiles]
all_data = []

# Process in batches of 20 to avoid Railway timeout
BATCH = 20
for batch_start in range(0, len(candidates_syms), BATCH):
    batch = candidates_syms[batch_start:batch_start+BATCH]
    if (batch_start) % 100 == 0 and batch_start > 0:
        print(f"  ... procesando {batch_start}/{len(candidates_syms)}")

    with railway_engine.connect() as rconn:
        for sym in batch:
            try:
                # Last price
                r2 = rconn.execute(sqlalchemy.text(
                    "SELECT ph.close FROM price_history ph "
                    "JOIN symbols s ON s.id = ph.symbol_id "
                    "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
                ), {'sym': sym})
                row2 = r2.fetchone()
                if not row2:
                    continue
                last_price = float(row2[0])

                # MA200
                r3 = rconn.execute(sqlalchemy.text(
                    "SELECT AVG(sub.close) FROM ("
                    "  SELECT ph.close FROM price_history ph "
                    "  JOIN symbols s ON s.id = ph.symbol_id "
                    "  WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 200"
                    ") sub"
                ), {'sym': sym})
                row3 = r3.fetchone()
                if not row3 or not row3[0]:
                    continue
                ma200 = float(row3[0])
                vs_ma200 = ((last_price - ma200) / ma200) * 100

                if vs_ma200 <= 0:
                    continue

                # Feb return
                r_fe = rconn.execute(sqlalchemy.text(
                    "SELECT ph.close FROM price_history ph JOIN symbols s ON s.id = ph.symbol_id "
                    "WHERE s.code = :sym AND ph.date BETWEEN '2026-01-27' AND '2026-01-31' "
                    "ORDER BY ph.date DESC LIMIT 1"
                ), {'sym': sym})
                r_fc = rconn.execute(sqlalchemy.text(
                    "SELECT ph.close FROM price_history ph JOIN symbols s ON s.id = ph.symbol_id "
                    "WHERE s.code = :sym AND ph.date = '2026-02-26'"
                ), {'sym': sym})
                fe, fc = r_fe.fetchone(), r_fc.fetchone()
                feb_ret = ((float(fc[0]) - float(fe[0])) / float(fe[0])) * 100 if fe and fc else None

                seas = seasonality[sym]
                prof = profiles[sym]
                eps = eps_data.get(sym, {})

                all_data.append({
                    'symbol': sym,
                    'sector': prof['sector'], 'industry': prof['industry'],
                    'mkt_cap': prof['mkt_cap'], 'last_price': last_price,
                    'win_rate': seas['win_rate'], 'avg_ret': seas['avg_ret'],
                    'ma200': ma200, 'vs_ma200': vs_ma200,
                    'eps_surprise': eps.get('eps_surprise'),
                    'eps_beat': eps.get('eps_beat'),
                    'eps_yoy': eps.get('eps_yoy'),
                    'rev_surprise': eps.get('rev_surprise'),
                    'feb_ret': feb_ret,
                })
            except Exception:
                continue

print(f"  Total candidatos MA200>0: {len(all_data)}")

# ================================================================
# PASO 4: SCORING + SELECCION DIVERSIFICADA
# ================================================================
print("\n[4/4] Scoring y seleccion diversificada...")

for d in all_data:
    score = 0
    score += d['win_rate'] * 2
    score += d['avg_ret'] * 3
    score += min(d['vs_ma200'], 50) * 0.5
    if d['eps_beat'] is True:
        score += 15
    if d['eps_surprise'] is not None and d['eps_surprise'] > 5:
        score += 10
    if d['eps_yoy'] is not None and d['eps_yoy'] > 0:
        score += 10
    if d['rev_surprise'] is not None and d['rev_surprise'] > 0:
        score += 5
    d['score'] = score

all_data.sort(key=lambda x: x['score'], reverse=True)

selected = []
sector_count = defaultdict(int)
MAX_PER_SECTOR = 5

# First pass: best from each sector
sectors_seen = set()
for d in all_data:
    sec = d['sector']
    if sec not in sectors_seen and len(selected) < 30:
        selected.append(d)
        sectors_seen.add(sec)
        sector_count[sec] += 1

# Second pass: fill to 30
for d in all_data:
    if d in selected:
        continue
    sec = d['sector']
    if sector_count[sec] < MAX_PER_SECTOR and len(selected) < 30:
        selected.append(d)
        sector_count[sec] += 1

selected.sort(key=lambda x: x['score'], reverse=True)

# ================================================================
# IMPRIMIR
# ================================================================
print(f"\n{'=' * 175}")
print("  30 CANDIDATOS MARZO 2026 - Precios al 26/02/2026")
print("  Criterios: WR>=70% (10a), MA200>0, MCap>$7B, EPS, diversificacion sectorial (max 5/sector)")
print(f"{'=' * 175}")
hdr = f'  {"#":>2s}  {"Symbol":8s} | {"Sector":22s} | {"Industry":35s} | {"MCap$B":>7s} | {"WR":>4s} | {"AvgRet":>7s} | {"vsMA200":>7s} | {"EPSsurp":>7s} | {"EPSyoy":>7s} | {"FebRet":>7s} | {"Score":>5s}'
print(hdr)
print('  ' + '-' * 171)

for i, d in enumerate(selected):
    mc = f"{d['mkt_cap']:>6.1f}" if d['mkt_cap'] else "   N/A"
    se = f"+{d['eps_surprise']:>5.1f}%" if d.get('eps_surprise') is not None and d['eps_surprise'] >= 0 else (f"{d['eps_surprise']:>6.1f}%" if d.get('eps_surprise') is not None else "   N/A ")
    sy = f"+{d['eps_yoy']:>5.1f}%" if d.get('eps_yoy') is not None and d['eps_yoy'] >= 0 else (f"{d['eps_yoy']:>6.1f}%" if d.get('eps_yoy') is not None else "   N/A ")
    fr = f"{d['feb_ret']:>+6.1f}%" if d.get('feb_ret') is not None else "   N/A "
    sv = "+" if d['vs_ma200'] >= 0 else ""

    combo = ""
    if d['eps_beat'] is True and d.get('eps_surprise') is not None and d['eps_surprise'] > 5 and d.get('eps_yoy') is not None and d['eps_yoy'] > 0 and d['vs_ma200'] > 10:
        combo = " ***"
    elif d['eps_beat'] is True and d.get('eps_surprise') is not None and d['eps_surprise'] > 5:
        combo = " **"
    elif d['eps_beat'] is True:
        combo = " *"

    sec = (d['sector'] if d['sector'] else 'N/A')[:22]
    print(f'  {i+1:>2d}. {d["symbol"]:8s} | {sec:22s} | {d["industry"]:35s} | {mc} | {d["win_rate"]:>3.0f}% | {d["avg_ret"]:>+6.2f}% | {sv}{d["vs_ma200"]:>5.1f}% | {se} | {sy} | {fr} | {d["score"]:>5.0f}{combo}')

print()
print("  *** = COMBO IDEAL (MA200>+10% + Beat + Surprise>5% + YoY>0)")
print("  **  = Beat + Surprise>5%")
print("  *   = EPS Beat")

# Sector summary
print(f"\n  Distribucion sectorial ({len(set(d['sector'] for d in selected))} sectores):")
sec_summary = defaultdict(list)
for d in selected:
    sec_summary[d['sector']].append(d['symbol'])
for sec in sorted(sec_summary.keys(), key=lambda s: len(sec_summary[s]), reverse=True):
    syms = sec_summary[sec]
    print(f"    {sec:25s} ({len(syms)}): {', '.join(syms)}")

# Combo ideal
combo_ideal = [d for d in selected
    if d['eps_beat'] is True
    and d.get('eps_surprise') is not None and d['eps_surprise'] > 5
    and d.get('eps_yoy') is not None and d['eps_yoy'] > 0
    and d['vs_ma200'] > 10]

print(f"\n  COMBO IDEAL ({len(combo_ideal)} acciones):")
if combo_ideal:
    for d in combo_ideal:
        fr = f"Feb {d['feb_ret']:+.1f}%" if d['feb_ret'] is not None else ""
        print(f"    {d['symbol']:8s} | {d['sector'][:20]:20s} | WR {d['win_rate']:.0f}% | AvgRet {d['avg_ret']:+.2f}% | MA200 +{d['vs_ma200']:.1f}% | EPSsurp +{d['eps_surprise']:.1f}% | EPSyoy +{d['eps_yoy']:.1f}% | {fr}")
else:
    print("    Ninguna cumple todos los criterios combo ideal")

# Also show top with relaxed criteria (MA200>0, eps_beat)
relaxed = [d for d in selected if d['eps_beat'] is True and d['vs_ma200'] > 5]
relaxed.sort(key=lambda x: x['score'], reverse=True)
print(f"\n  TOP RELAJADO - MA200>+5% + Beat ({len(relaxed)} acciones):")
for d in relaxed[:15]:
    fr = f"Feb {d['feb_ret']:+.1f}%" if d['feb_ret'] is not None else ""
    se = f"Surp +{d['eps_surprise']:.1f}%" if d['eps_surprise'] is not None and d['eps_surprise'] > 0 else ""
    sy = f"YoY +{d['eps_yoy']:.0f}%" if d['eps_yoy'] is not None and d['eps_yoy'] > 0 else ""
    print(f"    {d['symbol']:8s} | {d['sector'][:20]:20s} | WR {d['win_rate']:.0f}% | MA200 +{d['vs_ma200']:.1f}% | {se:15s} | {sy:12s} | {fr}")
