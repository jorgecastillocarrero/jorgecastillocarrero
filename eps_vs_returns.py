import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')
fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

symbols_list = [
    'AXON','ENTG','NET','HUBS','STLD','ABBV','HEI','NOC','GE','UNP','TJX','PAA',
    'ESTC','ATI','NVDA','CLF','SAIA','THC','AMAT','ODFL','IDXX','ARES','HII','TDG',
    'ALGN','SYK','HBAN','LMT','CHD','TECH','REGN','BSX','AVGO'
]

info = {
    'AXON':  {'sector': 'Industrials',      'win_rate': 90, 'ma200_sel': -25.7, 'selected': False},
    'ENTG':  {'sector': 'Technology',        'win_rate': 90, 'ma200_sel': 43.6,  'selected': False},
    'NET':   {'sector': 'Technology',        'win_rate': 83, 'ma200_sel': -5.5,  'selected': True},
    'HUBS':  {'sector': 'Technology',        'win_rate': 80, 'ma200_sel': -41.8, 'selected': False},
    'STLD':  {'sector': 'Basic Materials',   'win_rate': 80, 'ma200_sel': 27.4,  'selected': True},
    'ABBV':  {'sector': 'Healthcare',        'win_rate': 80, 'ma200_sel': 7.4,   'selected': False},
    'HEI':   {'sector': 'Industrials',       'win_rate': 80, 'ma200_sel': 7.1,   'selected': True},
    'NOC':   {'sector': 'Industrials',       'win_rate': 80, 'ma200_sel': 25.4,  'selected': False},
    'GE':    {'sector': 'Industrials',       'win_rate': 80, 'ma200_sel': 9.5,   'selected': True},
    'UNP':   {'sector': 'Industrials',       'win_rate': 80, 'ma200_sel': 4.3,   'selected': False},
    'TJX':   {'sector': 'Consumer Cyclical', 'win_rate': 80, 'ma200_sel': 7.1,   'selected': True},
    'PAA':   {'sector': 'Energy',            'win_rate': 80, 'ma200_sel': 14.8,  'selected': True},
    'ESTC':  {'sector': 'Technology',        'win_rate': 71, 'ma200_sel': -18.0, 'selected': False},
    'ATI':   {'sector': 'Basic Materials',   'win_rate': 70, 'ma200_sel': 38.5,  'selected': False},
    'NVDA':  {'sector': 'Technology',        'win_rate': 70, 'ma200_sel': 15.0,  'selected': False},
    'CLF':   {'sector': 'Basic Materials',   'win_rate': 70, 'ma200_sel': 33.1,  'selected': False},
    'SAIA':  {'sector': 'Industrials',       'win_rate': 70, 'ma200_sel': 14.0,  'selected': False},
    'THC':   {'sector': 'Healthcare',        'win_rate': 70, 'ma200_sel': 4.7,   'selected': False},
    'AMAT':  {'sector': 'Technology',        'win_rate': 70, 'ma200_sel': 65.2,  'selected': False},
    'ODFL':  {'sector': 'Industrials',       'win_rate': 70, 'ma200_sel': 12.7,  'selected': False},
    'IDXX':  {'sector': 'Healthcare',        'win_rate': 70, 'ma200_sel': 11.1,  'selected': True},
    'ARES':  {'sector': 'Financial',         'win_rate': 70, 'ma200_sel': -7.8,  'selected': False},
    'HII':   {'sector': 'Industrials',       'win_rate': 70, 'ma200_sel': 52.0,  'selected': False},
    'TDG':   {'sector': 'Industrials',       'win_rate': 70, 'ma200_sel': 5.9,   'selected': False},
    'ALGN':  {'sector': 'Healthcare',        'win_rate': 70, 'ma200_sel': 2.8,   'selected': False},
    'SYK':   {'sector': 'Healthcare',        'win_rate': 70, 'ma200_sel': -4.9,  'selected': True},
    'HBAN':  {'sector': 'Financial',         'win_rate': 70, 'ma200_sel': 6.4,   'selected': False},
    'LMT':   {'sector': 'Industrials',       'win_rate': 70, 'ma200_sel': 32.0,  'selected': False},
    'CHD':   {'sector': 'Consumer Defensive','win_rate': 70, 'ma200_sel': 0.8,   'selected': False},
    'TECH':  {'sector': 'Healthcare',        'win_rate': 70, 'ma200_sel': 14.2,  'selected': False},
    'REGN':  {'sector': 'Healthcare',        'win_rate': 70, 'ma200_sel': 21.5,  'selected': True},
    'BSX':   {'sector': 'Healthcare',        'win_rate': 70, 'ma200_sel': -8.2,  'selected': False},
    'AVGO':  {'sector': 'Technology',        'win_rate': 70, 'ma200_sel': 8.9,   'selected': True},
}

# Get EPS data from FMP
eps_data = {}
with fmp_engine.connect() as fconn:
    for sym in symbols_list:
        r = fconn.execute(sqlalchemy.text(
            "SELECT eps_actual, eps_estimated, revenue_actual, revenue_estimated, date "
            "FROM fmp_earnings "
            "WHERE symbol = :sym AND date < '2026-01-30' "
            "ORDER BY date DESC LIMIT 1"
        ), {"sym": sym})
        rows = r.fetchall()
        if rows:
            latest = rows[0]
            eps_act, eps_est, rev_act, rev_est, report_date = latest
            surprise_pct = ((eps_act - eps_est) / abs(eps_est) * 100) if eps_est and eps_est != 0 else 0
            beat = eps_act > eps_est if eps_act is not None and eps_est is not None else False
            yoy = None
            # Get YoY from same quarter previous year
            r2 = fconn.execute(sqlalchemy.text(
                "SELECT eps_actual FROM fmp_earnings "
                "WHERE symbol = :sym AND date < :cutoff "
                "ORDER BY date DESC LIMIT 1"
            ), {"sym": sym, "cutoff": str(report_date - __import__('datetime').timedelta(days=300))})
            yoy_row = r2.fetchone()
            if yoy_row and yoy_row[0] and eps_act:
                if yoy_row[0] != 0:
                    yoy = ((eps_act - yoy_row[0]) / abs(yoy_row[0])) * 100

            rev_surprise = None
            if rev_act and rev_est and rev_est != 0:
                rev_surprise = ((rev_act - rev_est) / rev_est) * 100

            eps_data[sym] = {
                'actual': eps_act,
                'estimated': eps_est,
                'surprise_pct': surprise_pct,
                'beat': beat,
                'yoy': yoy,
                'rev_surprise': rev_surprise,
            }

# Get returns from Railway
with engine.connect() as conn:
    data = []
    for sym in symbols_list:
        r1 = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym AND ph.date = '2026-01-30'"
        ), {"sym": sym})
        row1 = r1.fetchone()

        r2 = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
        row2 = r2.fetchone()

        if row1 and row2 and sym in eps_data:
            pc = float(row1[0])
            pa = float(row2[0])
            ret = ((pa - pc) / pc) * 100
            e = eps_data[sym]
            c = info[sym]
            data.append((sym, c, e, ret))

    # Sort by return
    data.sort(key=lambda x: x[3], reverse=True)

    print("=" * 155)
    print("  CANDIDATOS FEB 2026: RETORNO vs EPS SURPRISE vs MA200  (ordenado por retorno)")
    print("=" * 155)
    print(f"  {'#':>2s}  {'Symbol':6s} | {'Sector':20s} | {'WR':>3s} | {'MA200sel':>8s} | {'EPSsurp%':>8s} | {'EPSyoy%':>8s} | {'RevSurp%':>8s} | {'Beat':>4s} | {'RETORNO':>8s} | Sel")
    print(f"  {'':>2s}  {'-'*6} | {'-'*20} | {'-'*3} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*4} | {'-'*8} | ---")

    for i, (sym, c, e, ret) in enumerate(data):
        sr = "+" if ret >= 0 else ""
        sm = "+" if c['ma200_sel'] >= 0 else ""
        se = f"+{e['surprise_pct']:>6.1f}%" if e['surprise_pct'] >= 0 else f"{e['surprise_pct']:>7.1f}%"
        sy = f"+{e['yoy']:>6.1f}%" if e['yoy'] is not None and e['yoy'] >= 0 else (f"{e['yoy']:>7.1f}%" if e['yoy'] is not None else "    N/A ")
        rv = f"+{e['rev_surprise']:>6.1f}%" if e['rev_surprise'] is not None and e['rev_surprise'] >= 0 else (f"{e['rev_surprise']:>7.1f}%" if e['rev_surprise'] is not None else "    N/A ")
        bt = " YES" if e['beat'] else "  NO"
        sel = " <--" if c['selected'] else ""
        print(f"  {i+1:>2d}. {sym:6s} | {c['sector']:20s} | {c['win_rate']:>3d} | {sm}{c['ma200_sel']:>6.1f}% | {se} | {sy} | {rv} | {bt} | {sr}{ret:>7.2f}% |{sel}")

    # Correlation analysis
    print(f"\n  {'='*100}")
    print(f"  ANALISIS POR GRUPOS")
    print(f"  {'='*100}")

    # EPS Beat vs Miss
    beat_rets = [x[3] for x in data if x[2]['beat']]
    miss_rets = [x[3] for x in data if not x[2]['beat']]
    avg_beat = sum(beat_rets)/len(beat_rets) if beat_rets else 0
    avg_miss = sum(miss_rets)/len(miss_rets) if miss_rets else 0
    print(f"\n  --- EPS Beat vs Miss ---")
    print(f"  Beat  ({len(beat_rets):>2d}): ret medio {avg_beat:>+7.2f}%  | ganadoras {sum(1 for r in beat_rets if r>0)}/{len(beat_rets)}")
    print(f"  Miss  ({len(miss_rets):>2d}): ret medio {avg_miss:>+7.2f}%  | ganadoras {sum(1 for r in miss_rets if r>0)}/{len(miss_rets)}")

    # EPS Surprise > 5% vs < 5%
    high_surp = [x for x in data if x[2]['surprise_pct'] > 5]
    low_surp = [x for x in data if x[2]['surprise_pct'] <= 5]
    avg_hs = sum(x[3] for x in high_surp)/len(high_surp) if high_surp else 0
    avg_ls = sum(x[3] for x in low_surp)/len(low_surp) if low_surp else 0
    print(f"\n  --- EPS Surprise > 5% vs <= 5% ---")
    print(f"  Surp >5%  ({len(high_surp):>2d}): ret medio {avg_hs:>+7.2f}%  | ganadoras {sum(1 for x in high_surp if x[3]>0)}/{len(high_surp)}")
    print(f"  Surp <=5% ({len(low_surp):>2d}): ret medio {avg_ls:>+7.2f}%  | ganadoras {sum(1 for x in low_surp if x[3]>0)}/{len(low_surp)}")

    # EPS YoY > 0 vs < 0
    yoy_pos = [x for x in data if x[2]['yoy'] is not None and x[2]['yoy'] > 0]
    yoy_neg = [x for x in data if x[2]['yoy'] is not None and x[2]['yoy'] <= 0]
    avg_yp = sum(x[3] for x in yoy_pos)/len(yoy_pos) if yoy_pos else 0
    avg_yn = sum(x[3] for x in yoy_neg)/len(yoy_neg) if yoy_neg else 0
    print(f"\n  --- EPS YoY Growth positivo vs negativo ---")
    print(f"  YoY > 0%  ({len(yoy_pos):>2d}): ret medio {avg_yp:>+7.2f}%  | ganadoras {sum(1 for x in yoy_pos if x[3]>0)}/{len(yoy_pos)}")
    print(f"  YoY <= 0% ({len(yoy_neg):>2d}): ret medio {avg_yn:>+7.2f}%  | ganadoras {sum(1 for x in yoy_neg if x[3]>0)}/{len(yoy_neg)}")

    # Revenue surprise > 0 vs < 0
    rev_pos = [x for x in data if x[2]['rev_surprise'] is not None and x[2]['rev_surprise'] > 0]
    rev_neg = [x for x in data if x[2]['rev_surprise'] is not None and x[2]['rev_surprise'] <= 0]
    avg_rp = sum(x[3] for x in rev_pos)/len(rev_pos) if rev_pos else 0
    avg_rn = sum(x[3] for x in rev_neg)/len(rev_neg) if rev_neg else 0
    print(f"\n  --- Revenue Surprise positivo vs negativo ---")
    print(f"  Rev > 0%  ({len(rev_pos):>2d}): ret medio {avg_rp:>+7.2f}%  | ganadoras {sum(1 for x in rev_pos if x[3]>0)}/{len(rev_pos)}")
    print(f"  Rev <= 0% ({len(rev_neg):>2d}): ret medio {avg_rn:>+7.2f}%  | ganadoras {sum(1 for x in rev_neg if x[3]>0)}/{len(rev_neg)}")

    # Combinado: MA200 > 0 AND Beat AND YoY > 0
    combo = [x for x in data if x[1]['ma200_sel'] > 0 and x[2]['beat'] and x[2]['yoy'] is not None and x[2]['yoy'] > 0]
    avg_combo = sum(x[3] for x in combo)/len(combo) if combo else 0
    print(f"\n  --- COMBO: MA200>0 + Beat + YoY>0 ---")
    print(f"  Combo     ({len(combo):>2d}): ret medio {avg_combo:>+7.2f}%  | ganadoras {sum(1 for x in combo if x[3]>0)}/{len(combo)}")
    print(f"  Acciones: {', '.join(x[0] for x in sorted(combo, key=lambda x: x[3], reverse=True))}")

    # Combo 2: MA200 > 0 AND Surprise > 5%
    combo2 = [x for x in data if x[1]['ma200_sel'] > 0 and x[2]['surprise_pct'] > 5]
    avg_combo2 = sum(x[3] for x in combo2)/len(combo2) if combo2 else 0
    print(f"\n  --- COMBO: MA200>0 + Surprise>5% ---")
    print(f"  Combo     ({len(combo2):>2d}): ret medio {avg_combo2:>+7.2f}%  | ganadoras {sum(1 for x in combo2 if x[3]>0)}/{len(combo2)}")
    print(f"  Acciones: {', '.join(x[0] for x in sorted(combo2, key=lambda x: x[3], reverse=True))}")

    # Combo 3: MA200 > 0 AND Beat AND Surprise > 5% AND YoY > 0
    combo3 = [x for x in data if x[1]['ma200_sel'] > 0 and x[2]['beat'] and x[2]['surprise_pct'] > 5 and x[2]['yoy'] is not None and x[2]['yoy'] > 0]
    avg_combo3 = sum(x[3] for x in combo3)/len(combo3) if combo3 else 0
    print(f"\n  --- COMBO IDEAL: MA200>0 + Beat + Surprise>5% + YoY>0 ---")
    print(f"  Combo     ({len(combo3):>2d}): ret medio {avg_combo3:>+7.2f}%  | ganadoras {sum(1 for x in combo3 if x[3]>0)}/{len(combo3)}")
    print(f"  Acciones: {', '.join(x[0] for x in sorted(combo3, key=lambda x: x[3], reverse=True))}")
