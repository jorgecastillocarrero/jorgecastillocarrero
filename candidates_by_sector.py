import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy
from collections import defaultdict

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

candidates = {
    'AXON':  {'sector': 'Industrials',      'industria': 'Aerospace & Defense',    'win_rate': 90, 'avg_ret': 7.45, 'ma200_sel': -25.7, 'rsi': 24, 'recom': 'Buy',        'eps_fwd': 138.5, 'beat': 'No', 'selected': False},
    'ENTG':  {'sector': 'Technology',        'industria': 'Semiconductors',         'win_rate': 90, 'avg_ret': 6.63, 'ma200_sel': 43.6,  'rsi': 79, 'recom': 'Buy',        'eps_fwd': 70.4,  'beat': 'No', 'selected': False},
    'NET':   {'sector': 'Technology',        'industria': 'Cloud/Software',         'win_rate': 83, 'avg_ret': 13.26,'ma200_sel': -5.5,  'rsi': 46, 'recom': 'Buy',        'eps_fwd': 481.2, 'beat': 'Si', 'selected': True},
    'HUBS':  {'sector': 'Technology',        'industria': 'Software',               'win_rate': 80, 'avg_ret': 9.00, 'ma200_sel': -41.8, 'rsi': 21, 'recom': 'Strong Buy', 'eps_fwd': 19194.9,'beat': 'Si', 'selected': False},
    'STLD':  {'sector': 'Basic Materials',   'industria': 'Steel',                  'win_rate': 80, 'avg_ret': 6.96, 'ma200_sel': 27.4,  'rsi': 61, 'recom': 'Strong Buy', 'eps_fwd': 93.4,  'beat': 'Si', 'selected': True},
    'ABBV':  {'sector': 'Healthcare',        'industria': 'Drug Manufacturers',     'win_rate': 80, 'avg_ret': 4.64, 'ma200_sel': 7.4,   'rsi': 47, 'recom': 'Buy',        'eps_fwd': 977.5, 'beat': 'Si', 'selected': False},
    'HEI':   {'sector': 'Industrials',       'industria': 'Aerospace & Defense',    'win_rate': 80, 'avg_ret': 4.59, 'ma200_sel': 7.1,   'rsi': 34, 'recom': 'Buy',        'eps_fwd': 27.8,  'beat': 'Si', 'selected': True},
    'NOC':   {'sector': 'Industrials',       'industria': 'Aerospace & Defense',    'win_rate': 80, 'avg_ret': 3.33, 'ma200_sel': 25.4,  'rsi': 84, 'recom': 'Buy',        'eps_fwd': 4.9,   'beat': 'Si', 'selected': False},
    'GE':    {'sector': 'Industrials',       'industria': 'Aerospace & Defense',    'win_rate': 80, 'avg_ret': 2.78, 'ma200_sel': 9.5,   'rsi': 41, 'recom': 'Strong Buy', 'eps_fwd': 5.1,   'beat': 'Si', 'selected': True},
    'UNP':   {'sector': 'Industrials',       'industria': 'Railroads',              'win_rate': 80, 'avg_ret': 1.85, 'ma200_sel': 4.3,   'rsi': 54, 'recom': 'Buy',        'eps_fwd': 13.1,  'beat': 'Si', 'selected': False},
    'TJX':   {'sector': 'Consumer Cyclical', 'industria': 'Apparel Retail',         'win_rate': 80, 'avg_ret': 1.26, 'ma200_sel': 7.1,   'rsi': 19, 'recom': 'Strong Buy', 'eps_fwd': 14.0,  'beat': 'Si', 'selected': True},
    'PAA':   {'sector': 'Energy',            'industria': 'Oil & Gas Midstream',    'win_rate': 80, 'avg_ret': 0.33, 'ma200_sel': 14.8,  'rsi': 81, 'recom': 'Buy',        'eps_fwd': 65.2,  'beat': 'Si', 'selected': True},
    'ESTC':  {'sector': 'Technology',        'industria': 'Software',               'win_rate': 71, 'avg_ret': 2.82, 'ma200_sel': -18.0, 'rsi': 27, 'recom': 'None',       'eps_fwd': 374.4, 'beat': 'Si', 'selected': False},
    'ATI':   {'sector': 'Basic Materials',   'industria': 'Specialty Metals',       'win_rate': 70, 'avg_ret': 12.13,'ma200_sel': 38.5,  'rsi': 59, 'recom': 'Strong Buy', 'eps_fwd': 25.1,  'beat': 'Si', 'selected': False},
    'NVDA':  {'sector': 'Technology',        'industria': 'Semiconductors',         'win_rate': 70, 'avg_ret': 7.74, 'ma200_sel': 15.0,  'rsi': 61, 'recom': 'Strong Buy', 'eps_fwd': 89.7,  'beat': 'Si', 'selected': False},
    'CLF':   {'sector': 'Basic Materials',   'industria': 'Steel',                  'win_rate': 70, 'avg_ret': 7.16, 'ma200_sel': 33.1,  'rsi': 67, 'recom': 'Hold',       'eps_fwd': 106.0, 'beat': 'Si', 'selected': False},
    'SAIA':  {'sector': 'Industrials',       'industria': 'Trucking',               'win_rate': 70, 'avg_ret': 5.68, 'ma200_sel': 14.0,  'rsi': 34, 'recom': 'Buy',        'eps_fwd': 1.9,   'beat': 'Si', 'selected': False},
    'THC':   {'sector': 'Healthcare',        'industria': 'Hospitals',              'win_rate': 70, 'avg_ret': 5.63, 'ma200_sel': 4.7,   'rsi': 26, 'recom': 'Strong Buy', 'eps_fwd': 10.9,  'beat': 'Si', 'selected': False},
    'AMAT':  {'sector': 'Technology',        'industria': 'Semiconductors',         'win_rate': 70, 'avg_ret': 5.52, 'ma200_sel': 65.2,  'rsi': 78, 'recom': 'Buy',        'eps_fwd': 39.1,  'beat': 'Si', 'selected': False},
    'ODFL':  {'sector': 'Industrials',       'industria': 'Trucking',               'win_rate': 70, 'avg_ret': 5.11, 'ma200_sel': 12.7,  'rsi': 52, 'recom': 'Buy',        'eps_fwd': 2.0,   'beat': 'Si', 'selected': False},
    'IDXX':  {'sector': 'Healthcare',        'industria': 'Diagnostics & Research', 'win_rate': 70, 'avg_ret': 4.33, 'ma200_sel': 11.1,  'rsi': 33, 'recom': 'Buy',        'eps_fwd': 15.5,  'beat': 'Si', 'selected': True},
    'ARES':  {'sector': 'Financial',         'industria': 'Asset Management',       'win_rate': 70, 'avg_ret': 3.59, 'ma200_sel': -7.8,  'rsi': 11, 'recom': 'Buy',        'eps_fwd': 177.4, 'beat': 'Si', 'selected': False},
    'HII':   {'sector': 'Industrials',       'industria': 'Aerospace & Defense',    'win_rate': 70, 'avg_ret': 3.00, 'ma200_sel': 52.0,  'rsi': 77, 'recom': 'Buy',        'eps_fwd': 21.7,  'beat': 'Si', 'selected': False},
    'TDG':   {'sector': 'Industrials',       'industria': 'Aerospace & Defense',    'win_rate': 70, 'avg_ret': 2.61, 'ma200_sel': 5.9,   'rsi': 63, 'recom': 'Buy',        'eps_fwd': 43.4,  'beat': 'Si', 'selected': False},
    'ALGN':  {'sector': 'Healthcare',        'industria': 'Medical Devices',        'win_rate': 70, 'avg_ret': 2.56, 'ma200_sel': 2.8,   'rsi': 37, 'recom': 'Buy',        'eps_fwd': 114.0, 'beat': 'Si', 'selected': False},
    'SYK':   {'sector': 'Healthcare',        'industria': 'Medical Devices',        'win_rate': 70, 'avg_ret': 2.24, 'ma200_sel': -4.9,  'rsi': 36, 'recom': 'Buy',        'eps_fwd': 119.4, 'beat': 'Si', 'selected': True},
    'HBAN':  {'sector': 'Financial',         'industria': 'Banking',                'win_rate': 70, 'avg_ret': 2.10, 'ma200_sel': 6.4,   'rsi': 37, 'recom': 'Buy',        'eps_fwd': 38.7,  'beat': 'No', 'selected': False},
    'LMT':   {'sector': 'Industrials',       'industria': 'Aerospace & Defense',    'win_rate': 70, 'avg_ret': 2.06, 'ma200_sel': 32.0,  'rsi': 87, 'recom': 'Hold',       'eps_fwd': 45.4,  'beat': 'Si', 'selected': False},
    'CHD':   {'sector': 'Consumer Defensive','industria': 'Household Products',     'win_rate': 70, 'avg_ret': 1.58, 'ma200_sel': 0.8,   'rsi': 78, 'recom': 'Buy',        'eps_fwd': 17.5,  'beat': 'Si', 'selected': False},
    'TECH':  {'sector': 'Healthcare',        'industria': 'Life Sciences',          'win_rate': 70, 'avg_ret': 1.44, 'ma200_sel': 14.2,  'rsi': 48, 'recom': 'Buy',        'eps_fwd': 342.0, 'beat': 'No', 'selected': False},
    'REGN':  {'sector': 'Healthcare',        'industria': 'Biotechnology',          'win_rate': 70, 'avg_ret': 1.24, 'ma200_sel': 21.5,  'rsi': 30, 'recom': 'Buy',        'eps_fwd': 8.1,   'beat': 'Si', 'selected': True},
    'BSX':   {'sector': 'Healthcare',        'industria': 'Medical Devices',        'win_rate': 70, 'avg_ret': 1.03, 'ma200_sel': -8.2,  'rsi': 33, 'recom': 'Strong Buy', 'eps_fwd': 85.7,  'beat': 'Si', 'selected': False},
    'AVGO':  {'sector': 'Technology',        'industria': 'Semiconductors',         'win_rate': 70, 'avg_ret': 0.38, 'ma200_sel': 8.9,   'rsi': 49, 'recom': 'Strong Buy', 'eps_fwd': 198.7, 'beat': 'Si', 'selected': True},
}

syms = list(candidates.keys())

with engine.connect() as conn:
    data = []
    for sym in syms:
        c = candidates[sym]
        r_compra = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym AND ph.date = '2026-01-30'"
        ), {"sym": sym})
        row_compra = r_compra.fetchone()

        r_last = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
        row_last = r_last.fetchone()

        r_ma = conn.execute(sqlalchemy.text(
            "SELECT AVG(ph.close) FROM ("
            "  SELECT ph.close FROM price_history ph "
            "  JOIN symbols s ON s.id = ph.symbol_id "
            "  WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 200"
            ") ph"
        ), {"sym": sym})
        row_ma = r_ma.fetchone()

        if row_compra and row_last and row_ma:
            pc = float(row_compra[0])
            pa = float(row_last[0])
            ma200 = float(row_ma[0]) if row_ma[0] else 0
            vs_ma = ((pa - ma200) / ma200) * 100 if ma200 > 0 else 0
            ret = ((pa - pc) / pc) * 100
            data.append((sym, c, pc, pa, ma200, vs_ma, ret))

    # Agrupar por sector
    sectors = defaultdict(list)
    for item in data:
        sectors[item[1]['sector']].append(item)

    sector_avgs = {}
    for sec, items in sectors.items():
        avg = sum(x[6] for x in items) / len(items)
        sector_avgs[sec] = avg

    sorted_sectors = sorted(sector_avgs.keys(), key=lambda s: sector_avgs[s], reverse=True)

    print("=" * 160)
    print("  CANDIDATOS FEBRERO 2026 POR SECTOR (retorno 30/01 - 23/02)")
    print("=" * 160)

    for sec in sorted_sectors:
        items = sectors[sec]
        items.sort(key=lambda x: x[6], reverse=True)
        avg_ret = sector_avgs[sec]
        pos = sum(1 for x in items if x[6] > 0)
        sel_count = sum(1 for x in items if x[1]['selected'])

        print(f"\n  {'='*155}")
        s = "+" if avg_ret >= 0 else ""
        print(f"  {sec.upper()} ({len(items)} acciones, {sel_count} seleccionadas)  |  Media: {s}{avg_ret:.2f}%  |  Ganadoras: {pos}/{len(items)}")
        print(f"  {'='*155}")
        print(f"  {'Symbol':6s} | {'Industria':25s} | {'WR':>3s} | {'AvgRet':>7s} | {'MA200sel':>8s} | {'MA200act':>9s} | {'vsMA200':>7s} | {'RSI':>3s} | {'Recom':>10s} | {'P.Compra':>9s} | {'P.Actual':>9s} | {'RETORNO':>8s} | {'Sel':>3s}")
        print(f"  {'-'*6} | {'-'*25} | {'-'*3} | {'-'*7} | {'-'*8} | {'-'*9} | {'-'*7} | {'-'*3} | {'-'*10} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*3}")

        for sym, c, pc, pa, ma200, vs_ma, ret in items:
            sr = "+" if ret >= 0 else ""
            sv = "+" if vs_ma >= 0 else ""
            sm = "+" if c['ma200_sel'] >= 0 else ""
            sel = " *" if c['selected'] else "  "
            print(f"  {sym:6s} | {c['industria']:25s} | {c['win_rate']:>3d} | {c['avg_ret']:>+6.1f}% | {sm}{c['ma200_sel']:>6.1f}% | {ma200:>9.2f} | {sv}{vs_ma:>5.1f}% | {c['rsi']:>3d} | {c['recom']:>10s} | {pc:>9.2f} | {pa:>9.2f} | {sr}{ret:>7.2f}% |{sel}")

    print(f"\n\n  {'='*70}")
    print(f"  RANKING SECTORES")
    print(f"  {'='*70}")
    print(f"  {'Sector':20s} | {'Acciones':>8s} | {'Ganadoras':>9s} | {'Ret. Medio':>10s}")
    print(f"  {'-'*20} | {'-'*8} | {'-'*9} | {'-'*10}")
    for sec in sorted_sectors:
        items = sectors[sec]
        avg = sector_avgs[sec]
        pos = sum(1 for x in items if x[6] > 0)
        s = "+" if avg >= 0 else ""
        print(f"  {sec:20s} | {len(items):>8d} | {pos:>5d}/{len(items):<3d} | {s}{avg:>9.2f}%")
