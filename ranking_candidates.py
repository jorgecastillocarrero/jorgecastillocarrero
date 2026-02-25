import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

candidates = {
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

with engine.connect() as conn:
    data = []
    for sym, c in candidates.items():
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

        r3 = conn.execute(sqlalchemy.text(
            "SELECT AVG(sub.close) FROM ("
            "  SELECT ph.close FROM price_history ph "
            "  JOIN symbols s ON s.id = ph.symbol_id "
            "  WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 200"
            ") sub"
        ), {"sym": sym})
        row3 = r3.fetchone()

        if row1 and row2 and row3:
            pc = float(row1[0])
            pa = float(row2[0])
            ma200 = float(row3[0]) if row3[0] else 0
            vs_ma = ((pa - ma200) / ma200) * 100 if ma200 > 0 else 0
            ret = ((pa - pc) / pc) * 100
            data.append((sym, c, pc, pa, ma200, vs_ma, ret))

    data.sort(key=lambda x: x[6], reverse=True)

    print("=" * 105)
    print("  RANKING CANDIDATOS FEBRERO 2026 (mejor a peor retorno)")
    print("=" * 105)
    hdr = f"  {'#':>2s}  {'Symbol':6s} | {'Sector':20s} | {'WR':>4s} | {'MA200sel':>8s} | {'MA200act':>9s} | {'vsMA200':>7s} | {'RETORNO':>8s} |"
    print(hdr)
    print(f"  {'':>2s}  {'-'*6} | {'-'*20} | {'-'*4} | {'-'*8} | {'-'*9} | {'-'*7} | {'-'*8} |")

    for i, (sym, c, pc, pa, ma200, vs_ma, ret) in enumerate(data):
        sr = "+" if ret >= 0 else ""
        sv = "+" if vs_ma >= 0 else ""
        sm = "+" if c['ma200_sel'] >= 0 else ""
        sel = " <--" if c['selected'] else ""
        print(f"  {i+1:>2d}. {sym:6s} | {c['sector']:20s} | {c['win_rate']:>3d}% | {sm}{c['ma200_sel']:>6.1f}% | {ma200:>9.2f} | {sv}{vs_ma:>5.1f}% | {sr}{ret:>7.2f}% |{sel}")
