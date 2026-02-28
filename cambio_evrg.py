import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy
import json

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway',
    pool_pre_ping=True, pool_recycle=60)
fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

candidates = ['CMS','WEC','MRK','ATO','ADM','AME','OHI']
current = ['GOOGL','VLO','DG','CVX','UPS','STZ','CHD','SCCO','ED']

results = []

with fmp_engine.connect() as fconn:
    for sym in candidates:
        with engine.connect() as conn:
            # Profile
            r = fconn.execute(sqlalchemy.text("SELECT sector, industry FROM fmp_profiles WHERE symbol=:s LIMIT 1"), {'s': sym})
            p = r.fetchone()
            sector = p[0] if p else 'N/A'
            industry = (p[1] if p else 'N/A') or 'N/A'

            # MCap
            r_mc = fconn.execute(sqlalchemy.text(
                "SELECT market_cap FROM fmp_key_metrics WHERE symbol=:s AND market_cap IS NOT NULL ORDER BY date DESC LIMIT 1"
            ), {'s': sym})
            mc_row = r_mc.fetchone()
            mkt_cap = round(float(mc_row[0])/1e9, 1) if mc_row else None

            # Returns jan, feb
            def get_ret(conn, sym, d1_from, d1_to, d2):
                r1 = conn.execute(sqlalchemy.text(
                    "SELECT ph.close FROM price_history ph JOIN symbols s ON s.id=ph.symbol_id "
                    "WHERE s.code=:sym AND ph.date BETWEEN :d1 AND :d2 ORDER BY ph.date DESC LIMIT 1"
                ), {'sym': sym, 'd1': d1_from, 'd2': d1_to})
                r2 = conn.execute(sqlalchemy.text(
                    "SELECT ph.close FROM price_history ph JOIN symbols s ON s.id=ph.symbol_id "
                    "WHERE s.code=:sym AND ph.date=:d"
                ), {'sym': sym, 'd': d2})
                e, c = r1.fetchone(), r2.fetchone()
                if e and c:
                    return round(((float(c[0]) - float(e[0])) / float(e[0])) * 100, 2)
                return None

            jan_ret = get_ret(conn, sym, '2025-12-27', '2025-12-31', '2026-01-30')
            feb_ret = get_ret(conn, sym, '2026-01-27', '2026-01-31', '2026-02-26')

            # MA200 + price
            r3 = conn.execute(sqlalchemy.text(
                "SELECT AVG(sub.close) FROM ("
                "  SELECT ph.close FROM price_history ph JOIN symbols s ON s.id=ph.symbol_id "
                "  WHERE s.code=:sym ORDER BY ph.date DESC LIMIT 200"
                ") sub"
            ), {'sym': sym})
            r4 = conn.execute(sqlalchemy.text(
                "SELECT ph.close FROM price_history ph JOIN symbols s ON s.id=ph.symbol_id "
                "WHERE s.code=:sym ORDER BY ph.date DESC LIMIT 1"
            ), {'sym': sym})
            row3, row4 = r3.fetchone(), r4.fetchone()
            ma200 = float(row3[0]) if row3 and row3[0] else None
            last_price = float(row4[0]) if row4 else None
            vs_ma200 = round(((last_price - ma200) / ma200) * 100, 1) if last_price and ma200 else None

            # Seasonality march
            r_seas = fconn.execute(sqlalchemy.text("""
                WITH feb_close AS (
                    SELECT EXTRACT(YEAR FROM date) as yr, close as feb_close,
                           ROW_NUMBER() OVER (PARTITION BY EXTRACT(YEAR FROM date) ORDER BY date DESC) as rn
                    FROM fmp_price_history WHERE EXTRACT(MONTH FROM date)=2 AND EXTRACT(YEAR FROM date) BETWEEN 2016 AND 2025 AND symbol=:sym
                ),
                mar_close AS (
                    SELECT EXTRACT(YEAR FROM date) as yr, close as mar_close,
                           ROW_NUMBER() OVER (PARTITION BY EXTRACT(YEAR FROM date) ORDER BY date DESC) as rn
                    FROM fmp_price_history WHERE EXTRACT(MONTH FROM date)=3 AND EXTRACT(YEAR FROM date) BETWEEN 2016 AND 2025 AND symbol=:sym
                )
                SELECT COUNT(*), SUM(CASE WHEN m.mar_close>f.feb_close THEN 1 ELSE 0 END),
                       ROUND(AVG((m.mar_close-f.feb_close)/NULLIF(f.feb_close,0)*100)::numeric,2)
                FROM feb_close f JOIN mar_close m ON f.yr=m.yr AND m.rn=1 WHERE f.rn=1
            """), {'sym': sym})
            seas = r_seas.fetchone()
            wr = int(round(float(seas[1])/float(seas[0])*100)) if seas and seas[0] and int(seas[0])>0 else None
            avg_mar = float(seas[2]) if seas and seas[2] else None

            # EPS
            r_eps = fconn.execute(sqlalchemy.text(
                "SELECT eps_actual, eps_estimated FROM fmp_earnings WHERE symbol=:s AND date<'2026-02-28' ORDER BY date DESC LIMIT 1"
            ), {'s': sym})
            eps = r_eps.fetchone()
            eps_surp = None
            eps_beat = False
            if eps and eps[0] is not None and eps[1] is not None and float(eps[1]) != 0:
                eps_surp = round(((float(eps[0])-float(eps[1]))/abs(float(eps[1])))*100, 1)
                eps_beat = float(eps[0]) > float(eps[1])

            # EPS YoY
            eps_yoy = None
            if eps and eps[0] is not None:
                from datetime import timedelta
                r_yoy = fconn.execute(sqlalchemy.text(
                    "SELECT eps_actual, date FROM fmp_earnings WHERE symbol=:s AND date<'2026-02-28' ORDER BY date DESC LIMIT 1"
                ), {'s': sym})
                cur_row = r_yoy.fetchone()
                if cur_row and cur_row[1]:
                    cutoff = str(cur_row[1] - timedelta(days=300))
                    r_prev = fconn.execute(sqlalchemy.text(
                        "SELECT eps_actual FROM fmp_earnings WHERE symbol=:s AND date<:cutoff ORDER BY date DESC LIMIT 1"
                    ), {'s': sym, 'cutoff': cutoff})
                    prev = r_prev.fetchone()
                    if prev and prev[0] and float(prev[0]) != 0:
                        eps_yoy = round(((float(eps[0]) - float(prev[0])) / abs(float(prev[0]))) * 100, 1)

            results.append({
                'sym': sym, 'sector': sector, 'industry': industry[:35],
                'mkt_cap': mkt_cap, 'wr': wr, 'avg_mar': avg_mar,
                'vs_ma200': vs_ma200, 'eps_surp': eps_surp, 'eps_beat': eps_beat,
                'eps_yoy': eps_yoy, 'jan_ret': jan_ret, 'feb_ret': feb_ret,
                'price': round(last_price, 2) if last_price else None,
            })

# Check sector overlap with current portfolio
with fmp_engine.connect() as fconn:
    current_sectors = {}
    for sym in current:
        r = fconn.execute(sqlalchemy.text("SELECT sector FROM fmp_profiles WHERE symbol=:s LIMIT 1"), {'s': sym})
        p = r.fetchone()
        if p: current_sectors[sym] = p[0]

sector_counts = {}
for s in current_sectors.values():
    sector_counts[s] = sector_counts.get(s, 0) + 1

# Generate HTML
def vc(v):
    if v is None: return 'neutral'
    if v > 0: return 'pos'
    if v < 0: return 'neg'
    return 'neutral'

def fmt(v, suffix='%'):
    if v is None: return '<span class="neutral">N/A</span>'
    cls = vc(v)
    sign = '+' if v > 0 else ''
    return f'<span class="{cls}">{sign}{v}{suffix}</span>'

html = """<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>Cambio EVRG - Cartera Marzo 2026</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; background: #fff; color: #222; margin: 20px; }
h1 { color: #1565c0; text-align: center; }
h2 { color: #333; border-bottom: 2px solid #1565c0; padding-bottom: 5px; margin-top: 25px; }
.subtitle { text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }
th { background: #1565c0; color: #fff; padding: 8px 6px; text-align: center; border: 1px solid #ccc; }
td { padding: 6px; text-align: center; border: 1px solid #ddd; }
tr:nth-child(even) { background: #f5f7fa; }
tr:hover { background: #e3f2fd; }
.pos { color: #2e7d32; font-weight: bold; }
.neg { color: #c62828; font-weight: bold; }
.neutral { color: #999; }
td.left { text-align: left; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
.badge-combo { background: #ffd600; color: #000; }
.badge-beat { background: #66bb6a; color: #000; }
.badge-90 { background: #00bcd4; color: #000; }
.badge-warn { background: #ff9800; color: #000; }
.note { background: #fffde7; padding: 10px 15px; border-radius: 8px; border-left: 4px solid #ffd600; margin-bottom: 20px; font-size: 13px; }
.sector-tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; color: #fff; }
.s-utility { background: #00897b; }
.s-health { background: #2e7d32; }
.s-consumer-def { background: #7b1fa2; }
.s-industrial { background: #e65100; }
.s-real-estate { background: #5e35b1; }
</style></head><body>

<h1>Reemplazo de EVRG - Cartera Marzo 2026</h1>
<p class="subtitle">Cartera actual (9): GOOGL, VLO, DG, CVX, UPS, STZ, CHD, SCCO, ED | Falta 1 plaza</p>

<div class="note">
<b>Sectores actuales:</b> """

for sect, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
    html += f"{sect} ({cnt}) | "
html += f"""<br><b>ED</b> ya cubre Utilities. Meter otra utility = 3 del mismo sector.
</div>

<h2>7 Candidatos disponibles</h2>
<table>
<tr>
<th>#</th><th>Symbol</th><th>Sector</th><th>Industry</th><th>MCap $B</th><th>Precio</th>
<th>WR Mar</th><th>Avg Mar</th><th>vs MA200</th><th>EPS Surp</th><th>EPS YoY</th>
<th>Ret Ene</th><th>Ret Feb</th><th>Ene+Feb</th><th>Notas</th>
</tr>
"""

for i, r in enumerate(results, 1):
    # Badges
    badges = ''
    is_combo = r['vs_ma200'] and r['vs_ma200'] > 10 and r['eps_beat'] and r['eps_surp'] and r['eps_surp'] > 5 and r['eps_yoy'] and r['eps_yoy'] > 0
    if is_combo: badges += ' <span class="badge badge-combo">COMBO</span>'
    if r['eps_beat']: badges += ' <span class="badge badge-beat">BEAT</span>'
    if r['wr'] and r['wr'] >= 90: badges += ' <span class="badge badge-90">WR90</span>'

    # Sector overlap warning
    sector_class = ''
    if r['sector'] == 'Utilities': sector_class = 's-utility'
    elif r['sector'] == 'Healthcare': sector_class = 's-health'
    elif r['sector'] == 'Consumer Defensive': sector_class = 's-consumer-def'
    elif r['sector'] == 'Industrials': sector_class = 's-industrial'
    elif r['sector'] == 'Real Estate': sector_class = 's-real-estate'

    notes = ''
    existing_count = sector_counts.get(r['sector'], 0)
    if existing_count >= 3:
        notes += '<span class="badge badge-warn">3+ sector</span> '
    elif existing_count >= 2:
        notes += f'<span class="badge badge-warn">{existing_count} en sector</span> '

    if r['sector'] not in [s for s in sector_counts]:
        notes += '<span class="badge" style="background:#2e7d32;color:#fff;">NUEVO SECTOR</span> '

    ene_feb = None
    if r['jan_ret'] is not None and r['feb_ret'] is not None:
        ene_feb = round(r['jan_ret'] + r['feb_ret'], 2)

    html += f'<tr><td>{i}</td><td><b>{r["sym"]}</b>{badges}</td>'
    html += f'<td class="left"><span class="sector-tag {sector_class}">{r["sector"]}</span></td>'
    html += f'<td class="left">{r["industry"]}</td>'
    html += f'<td>{r["mkt_cap"]}</td><td>${r["price"]}</td>'
    html += f'<td>{"<b>"+str(r["wr"])+"%</b>" if r["wr"] and r["wr"]>=80 else (str(r["wr"])+"%" if r["wr"] else "N/A")}</td>'
    html += f'<td>{fmt(r["avg_mar"])}</td>'
    html += f'<td>{fmt(r["vs_ma200"])}</td>'
    html += f'<td>{fmt(r["eps_surp"])}</td>'
    html += f'<td>{fmt(r["eps_yoy"])}</td>'
    html += f'<td>{fmt(r["jan_ret"])}</td>'
    html += f'<td>{fmt(r["feb_ret"])}</td>'
    html += f'<td>{fmt(ene_feb)}</td>'
    html += f'<td class="left">{notes}</td></tr>\n'

html += """</table>

<div class="note">
<b>Criterio de seleccion:</b> WR Mar &ge;70% | MA200 &gt; 0 | MCap &gt; $7B | Diversificacion sectorial<br>
<b>Leccion febrero:</b> MA200 &gt; +10% en entrada &rarr; +7.36% avg. Priorizar MA200 alto.
</div>

</body></html>"""

with open('cambio_evrg.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("OK -> cambio_evrg.html")
