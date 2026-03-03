import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway',
    pool_pre_ping=True, pool_recycle=60)
fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

symbols = ['GOOGL','VLO','DG','CVX','UPS','CMS','STZ','WEC','SCCO','MRK','ATO','CHD','ADM','ED','AME','OHI','EVRG']

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

results = []

with fmp_engine.connect() as fconn:
    for sym in symbols:
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

            # Returns
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

            # Seasonality
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
                'sym': sym, 'sector': sector, 'industry': industry[:30],
                'mkt_cap': mkt_cap, 'wr': wr, 'avg_mar': avg_mar,
                'vs_ma200': vs_ma200, 'eps_surp': eps_surp, 'eps_beat': eps_beat,
                'eps_yoy': eps_yoy, 'jan_ret': jan_ret, 'feb_ret': feb_ret,
                'price': round(last_price, 2) if last_price else None,
            })

# Print as pipe-delimited for easy parsing
for r in results:
    print(f"{r['sym']}|{r['sector']}|{r['industry']}|{r['mkt_cap']}|{r['wr']}|{r['avg_mar']}|{r['vs_ma200']}|{r['eps_surp']}|{r['eps_beat']}|{r['eps_yoy']}|{r['jan_ret']}|{r['feb_ret']}|{r['price']}")
