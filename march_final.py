import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

railway_engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')
fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# Combo ideal from screener + some extras with high win rate
combo_all = [
    # 90% WR
    {'symbol': 'CNX',  'win_rate': 90, 'avg_ret': 10.63},
    {'symbol': 'NEM',  'win_rate': 90, 'avg_ret': 7.84},
    {'symbol': 'RGLD', 'win_rate': 90, 'avg_ret': 7.61},
    {'symbol': 'JBSS', 'win_rate': 90, 'avg_ret': 7.07},
    {'symbol': 'ED',   'win_rate': 90, 'avg_ret': 6.06},
    {'symbol': 'NFG',  'win_rate': 90, 'avg_ret': 5.24},
    # 80% WR with MA200>0
    {'symbol': 'YORW', 'win_rate': 80, 'avg_ret': 4.58},
    {'symbol': 'SR',   'win_rate': 80, 'avg_ret': 3.85},
    {'symbol': 'AGX',  'win_rate': 80, 'avg_ret': 2.64},
    {'symbol': 'LLY',  'win_rate': 80, 'avg_ret': 2.38},
    {'symbol': 'GD',   'win_rate': 80, 'avg_ret': 1.81},
    {'symbol': 'QCOM', 'win_rate': 80, 'avg_ret': 1.43},
    {'symbol': 'LMT',  'win_rate': 80, 'avg_ret': 0.81},
    {'symbol': 'ADP',  'win_rate': 80, 'avg_ret': 0.19},
    # 70% WR combo ideal (MA200>0 + Beat + Surp>5% + YoY>0) - large/mid caps
    {'symbol': 'DY',   'win_rate': 70, 'avg_ret': 6.21},
    {'symbol': 'VIRT', 'win_rate': 70, 'avg_ret': 5.41},
    {'symbol': 'DG',   'win_rate': 70, 'avg_ret': 5.36},
    {'symbol': 'INTC', 'win_rate': 70, 'avg_ret': 5.21},
    {'symbol': 'FDX',  'win_rate': 70, 'avg_ret': 4.08},
    {'symbol': 'CACI', 'win_rate': 70, 'avg_ret': 2.24},
    {'symbol': 'VLO',  'win_rate': 70, 'avg_ret': 2.17},
    {'symbol': 'UI',   'win_rate': 70, 'avg_ret': 1.94},
    {'symbol': 'LRCX', 'win_rate': 70, 'avg_ret': 1.84},
    {'symbol': 'HCA',  'win_rate': 70, 'avg_ret': 1.48},
    {'symbol': 'GOOG', 'win_rate': 70, 'avg_ret': 1.24},
    {'symbol': 'TDY',  'win_rate': 70, 'avg_ret': 1.16},
    {'symbol': 'SPB',  'win_rate': 70, 'avg_ret': 0.96},
    {'symbol': 'KMI',  'win_rate': 70, 'avg_ret': 0.61},
    {'symbol': 'FHI',  'win_rate': 70, 'avg_ret': 0.54},
    {'symbol': 'EL',   'win_rate': 70, 'avg_ret': -0.34},
    {'symbol': 'FLS',  'win_rate': 70, 'avg_ret': -0.89},
]

with fmp_engine.connect() as fconn, railway_engine.connect() as rconn:
    results = []
    for c in combo_all:
        sym = c['symbol']

        # Market cap from FMP
        r = fconn.execute(sqlalchemy.text(
            "SELECT mkt_cap, sector, industry FROM fmp_profiles WHERE symbol = :sym LIMIT 1"
        ), {"sym": sym})
        prof = r.fetchone()
        mkt_cap = float(prof[0]) / 1e9 if prof and prof[0] else None
        sector = prof[1] if prof else 'N/A'
        industry = prof[2] if prof else 'N/A'

        # If no mkt_cap from profiles, estimate from Railway (price * shares_outstanding approx)
        # Skip if clearly micro/small cap (price < $5)
        r2 = rconn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
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
        ), {"sym": sym})
        row3 = r3.fetchone()
        if not row3 or not row3[0]:
            continue
        ma200 = float(row3[0])
        vs_ma200 = ((last_price - ma200) / ma200) * 100

        # EPS data
        r4 = fconn.execute(sqlalchemy.text(
            "SELECT eps_actual, eps_estimated, revenue_actual, revenue_estimated, date "
            "FROM fmp_earnings WHERE symbol = :sym AND date < '2026-02-27' "
            "ORDER BY date DESC LIMIT 1"
        ), {"sym": sym})
        eps_row = r4.fetchone()
        eps_surprise = None
        eps_beat = None
        eps_yoy = None
        rev_surprise = None

        if eps_row and eps_row[0] is not None and eps_row[1] is not None and eps_row[1] != 0:
            eps_act, eps_est = float(eps_row[0]), float(eps_row[1])
            eps_surprise = ((eps_act - eps_est) / abs(eps_est)) * 100
            eps_beat = eps_act > eps_est

            from datetime import timedelta
            r5 = fconn.execute(sqlalchemy.text(
                "SELECT eps_actual FROM fmp_earnings "
                "WHERE symbol = :sym AND date < :cutoff ORDER BY date DESC LIMIT 1"
            ), {"sym": sym, "cutoff": str(eps_row[4] - timedelta(days=300))})
            yoy_row = r5.fetchone()
            if yoy_row and yoy_row[0] and yoy_row[0] != 0:
                eps_yoy = ((eps_act - float(yoy_row[0])) / abs(float(yoy_row[0]))) * 100

            if eps_row[2] and eps_row[3] and eps_row[3] != 0:
                rev_surprise = ((float(eps_row[2]) - float(eps_row[3])) / float(eps_row[3])) * 100

        # Market cap from Railway fundamentals
        r6 = rconn.execute(sqlalchemy.text(
            "SELECT market_cap FROM fundamentals "
            "WHERE symbol_id = (SELECT id FROM symbols WHERE code = :sym) "
            "ORDER BY data_date DESC LIMIT 1"
        ), {"sym": sym})
        rec_row = r6.fetchone()
        if rec_row and rec_row[0] and (mkt_cap is None or mkt_cap == 0):
            mkt_cap = float(rec_row[0]) / 1e9
        recom = 'N/A'

        results.append({
            'symbol': sym,
            'sector': sector if sector else 'N/A',
            'industry': (industry if industry else 'N/A')[:25],
            'win_rate': c['win_rate'],
            'avg_ret': c['avg_ret'],
            'mkt_cap': mkt_cap,
            'last_price': last_price,
            'ma200': ma200,
            'vs_ma200': vs_ma200,
            'eps_surprise': eps_surprise,
            'eps_beat': eps_beat,
            'eps_yoy': eps_yoy,
            'rev_surprise': rev_surprise,
            'recom': recom,
        })

    # Filter: mkt_cap > 7B or unknown (we'll note it)
    large = [r for r in results if r['mkt_cap'] is None or r['mkt_cap'] >= 7]
    large.sort(key=lambda x: x['win_rate'] * 100 + x['avg_ret'], reverse=True)

    print("=" * 175)
    print("  SCREENER FINAL MARZO 2026 - Candidatos MCap >= $7B")
    print("  Filtros aplicados: WinRate >= 70%, MA200 > 0%, MCap > $7B")
    print("=" * 175)
    print(f"  {'#':>2s}  {'Symbol':8s} | {'Sector':22s} | {'Industry':25s} | {'MCap$B':>7s} | {'WR':>4s} | {'AvgRet':>7s} | {'vsMA200':>7s} | {'EPSsurp':>7s} | {'EPSyoy':>7s} | {'Recom':>10s} | {'Precio':>8s}")
    print(f"  {'':>2s}  {'-'*8} | {'-'*22} | {'-'*25} | {'-'*7} | {'-'*4} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*10} | {'-'*8}")

    for i, r in enumerate(large):
        mc = f"{r['mkt_cap']:>6.1f}" if r['mkt_cap'] else "   N/A"
        sv = "+" if r['vs_ma200'] >= 0 else ""
        se = f"+{r['eps_surprise']:>5.1f}%" if r['eps_surprise'] is not None and r['eps_surprise'] >= 0 else (f"{r['eps_surprise']:>6.1f}%" if r['eps_surprise'] is not None else "   N/A ")
        sy = f"+{r['eps_yoy']:>5.1f}%" if r['eps_yoy'] is not None and r['eps_yoy'] >= 0 else (f"{r['eps_yoy']:>6.1f}%" if r['eps_yoy'] is not None else "   N/A ")
        bt_mark = ""
        if r['eps_beat'] is True and r['eps_surprise'] is not None and r['eps_surprise'] > 5 and r['eps_yoy'] is not None and r['eps_yoy'] > 0:
            bt_mark = " ***"
        elif r['eps_beat'] is True:
            bt_mark = " *"
        elif r['vs_ma200'] > 0:
            bt_mark = ""

        sec = (r['sector'] if r['sector'] else 'N/A')[:22]
        rec = (r['recom'] if r['recom'] else 'N/A')[:10]
        print(f"  {i+1:>2d}. {r['symbol']:8s} | {sec:22s} | {r['industry']:25s} | {mc} | {r['win_rate']:>3.0f}% | {r['avg_ret']:>+6.2f}% | {sv}{r['vs_ma200']:>5.1f}% | {se} | {sy} | {rec:>10s} | {r['last_price']:>8.2f}{bt_mark}")

    print(f"\n  *** = COMBO IDEAL (MA200>0 + Beat + Surprise>5% + YoY>0)")
    print(f"  *   = EPS Beat")

    # Show combo ideal only
    combo = [r for r in large if r['eps_beat'] is True and r['eps_surprise'] is not None and r['eps_surprise'] > 5 and r['eps_yoy'] is not None and r['eps_yoy'] > 0 and r['vs_ma200'] > 0]
    combo.sort(key=lambda x: x['win_rate'] * 100 + x['avg_ret'], reverse=True)

    print(f"\n\n  {'='*120}")
    print(f"  TOP SELECCION MARZO 2026 - COMBO IDEAL ({len(combo)} acciones)")
    print(f"  {'='*120}")
    print(f"  {'#':>2s}  {'Symbol':8s} | {'Sector':22s} | {'MCap$B':>7s} | {'WR':>4s} | {'AvgRet':>7s} | {'vsMA200':>7s} | {'EPSsurp':>7s} | {'EPSyoy':>7s} | {'Recom':>10s}")
    print(f"  {'':>2s}  {'-'*8} | {'-'*22} | {'-'*7} | {'-'*4} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*10}")

    from collections import Counter
    for i, r in enumerate(combo):
        mc = f"{r['mkt_cap']:>6.1f}" if r['mkt_cap'] else "   N/A"
        sv = "+"
        rec = (r['recom'] if r['recom'] else 'N/A')[:10]
        sec = (r['sector'] if r['sector'] else 'N/A')[:22]
        print(f"  {i+1:>2d}. {r['symbol']:8s} | {sec:22s} | {mc} | {r['win_rate']:>3.0f}% | {r['avg_ret']:>+6.2f}% | {sv}{r['vs_ma200']:>5.1f}% | +{r['eps_surprise']:>5.1f}% | +{r['eps_yoy']:>5.1f}% | {rec:>10s}")

    print(f"\n  Sectores:")
    for sec, cnt in Counter(r['sector'] for r in combo).most_common():
        print(f"    {sec:25s}: {cnt}")
