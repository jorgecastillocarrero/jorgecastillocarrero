import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

# Todos los candidatos del screener completo (34 acciones)
candidates = {
    'AXON':  {'win_rate': 90, 'avg_ret': 7.45, 'mcap': 40.5,   'ma200_sel': -25.7, 'rsi': 24, 'recom': 'Buy',        'eps_fwd': 138.5, 'beat': 'No', 'selected': False},
    'ENTG':  {'win_rate': 90, 'avg_ret': 6.63, 'mcap': 18.6,   'ma200_sel': 43.6,  'rsi': 79, 'recom': 'Buy',        'eps_fwd': 70.4,  'beat': 'No', 'selected': False},
    'NET':   {'win_rate': 83, 'avg_ret': 13.26,'mcap': 63.2,   'ma200_sel': -5.5,  'rsi': 46, 'recom': 'Buy',        'eps_fwd': 481.2, 'beat': 'Si', 'selected': True},
    'HUBS':  {'win_rate': 80, 'avg_ret': 9.00, 'mcap': 14.8,   'ma200_sel': -41.8, 'rsi': 21, 'recom': 'Strong Buy', 'eps_fwd': 19194.9,'beat': 'Si', 'selected': False},
    'STLD':  {'win_rate': 80, 'avg_ret': 6.96, 'mcap': 26.8,   'ma200_sel': 27.4,  'rsi': 61, 'recom': 'Strong Buy', 'eps_fwd': 93.4,  'beat': 'Si', 'selected': True},
    'ABBV':  {'win_rate': 80, 'avg_ret': 4.64, 'mcap': 389.6,  'ma200_sel': 7.4,   'rsi': 47, 'recom': 'Buy',        'eps_fwd': 977.5, 'beat': 'Si', 'selected': False},
    'HEI':   {'win_rate': 80, 'avg_ret': 4.59, 'mcap': 46.3,   'ma200_sel': 7.1,   'rsi': 34, 'recom': 'Buy',        'eps_fwd': 27.8,  'beat': 'Si', 'selected': True},
    'NOC':   {'win_rate': 80, 'avg_ret': 3.33, 'mcap': 98.7,   'ma200_sel': 25.4,  'rsi': 84, 'recom': 'Buy',        'eps_fwd': 4.9,   'beat': 'Si', 'selected': False},
    'GE':    {'win_rate': 80, 'avg_ret': 2.78, 'mcap': 315.2,  'ma200_sel': 9.5,   'rsi': 41, 'recom': 'Strong Buy', 'eps_fwd': 5.1,   'beat': 'Si', 'selected': True},
    'UNP':   {'win_rate': 80, 'avg_ret': 1.85, 'mcap': 138.6,  'ma200_sel': 4.3,   'rsi': 54, 'recom': 'Buy',        'eps_fwd': 13.1,  'beat': 'Si', 'selected': False},
    'TJX':   {'win_rate': 80, 'avg_ret': 1.26, 'mcap': 164.1,  'ma200_sel': 7.1,   'rsi': 19, 'recom': 'Strong Buy', 'eps_fwd': 14.0,  'beat': 'Si', 'selected': True},
    'PAA':   {'win_rate': 80, 'avg_ret': 0.33, 'mcap': 13.9,   'ma200_sel': 14.8,  'rsi': 81, 'recom': 'Buy',        'eps_fwd': 65.2,  'beat': 'Si', 'selected': True},
    'ESTC':  {'win_rate': 71, 'avg_ret': 2.82, 'mcap': 7.2,    'ma200_sel': -18.0, 'rsi': 27, 'recom': 'None',       'eps_fwd': 374.4, 'beat': 'Si', 'selected': False},
    'ATI':   {'win_rate': 70, 'avg_ret': 12.13,'mcap': 16.8,   'ma200_sel': 38.5,  'rsi': 59, 'recom': 'Strong Buy', 'eps_fwd': 25.1,  'beat': 'Si', 'selected': False},
    'NVDA':  {'win_rate': 70, 'avg_ret': 7.74, 'mcap': 4687.0, 'ma200_sel': 15.0,  'rsi': 61, 'recom': 'Strong Buy', 'eps_fwd': 89.7,  'beat': 'Si', 'selected': False},
    'CLF':   {'win_rate': 70, 'avg_ret': 7.16, 'mcap': 8.1,    'ma200_sel': 33.1,  'rsi': 67, 'recom': 'Hold',       'eps_fwd': 106.0, 'beat': 'Si', 'selected': False},
    'SAIA':  {'win_rate': 70, 'avg_ret': 5.68, 'mcap': 9.1,    'ma200_sel': 14.0,  'rsi': 34, 'recom': 'Buy',        'eps_fwd': 1.9,   'beat': 'Si', 'selected': False},
    'THC':   {'win_rate': 70, 'avg_ret': 5.63, 'mcap': 16.7,   'ma200_sel': 4.7,   'rsi': 26, 'recom': 'Strong Buy', 'eps_fwd': 10.9,  'beat': 'Si', 'selected': False},
    'AMAT':  {'win_rate': 70, 'avg_ret': 5.52, 'mcap': 270.9,  'ma200_sel': 65.2,  'rsi': 78, 'recom': 'Buy',        'eps_fwd': 39.1,  'beat': 'Si', 'selected': False},
    'ODFL':  {'win_rate': 70, 'avg_ret': 5.11, 'mcap': 36.3,   'ma200_sel': 12.7,  'rsi': 52, 'recom': 'Buy',        'eps_fwd': 2.0,   'beat': 'Si', 'selected': False},
    'IDXX':  {'win_rate': 70, 'avg_ret': 4.33, 'mcap': 54.1,   'ma200_sel': 11.1,  'rsi': 33, 'recom': 'Buy',        'eps_fwd': 15.5,  'beat': 'Si', 'selected': True},
    'ARES':  {'win_rate': 70, 'avg_ret': 3.59, 'mcap': 49.7,   'ma200_sel': -7.8,  'rsi': 11, 'recom': 'Buy',        'eps_fwd': 177.4, 'beat': 'Si', 'selected': False},
    'HII':   {'win_rate': 70, 'avg_ret': 3.00, 'mcap': 16.8,   'ma200_sel': 52.0,  'rsi': 77, 'recom': 'Buy',        'eps_fwd': 21.7,  'beat': 'Si', 'selected': False},
    'TDG':   {'win_rate': 70, 'avg_ret': 2.61, 'mcap': 80.3,   'ma200_sel': 5.9,   'rsi': 63, 'recom': 'Buy',        'eps_fwd': 43.4,  'beat': 'Si', 'selected': False},
    'ALGN':  {'win_rate': 70, 'avg_ret': 2.56, 'mcap': 11.9,   'ma200_sel': 2.8,   'rsi': 37, 'recom': 'Buy',        'eps_fwd': 114.0, 'beat': 'Si', 'selected': False},
    'SYK':   {'win_rate': 70, 'avg_ret': 2.24, 'mcap': 135.5,  'ma200_sel': -4.9,  'rsi': 36, 'recom': 'Buy',        'eps_fwd': 119.4, 'beat': 'Si', 'selected': True},
    'HBAN':  {'win_rate': 70, 'avg_ret': 2.10, 'mcap': 27.2,   'ma200_sel': 6.4,   'rsi': 37, 'recom': 'Buy',        'eps_fwd': 38.7,  'beat': 'No', 'selected': False},
    'LMT':   {'win_rate': 70, 'avg_ret': 2.06, 'mcap': 144.0,  'ma200_sel': 32.0,  'rsi': 87, 'recom': 'Hold',       'eps_fwd': 45.4,  'beat': 'Si', 'selected': False},
    'CHD':   {'win_rate': 70, 'avg_ret': 1.58, 'mcap': 22.4,   'ma200_sel': 0.8,   'rsi': 78, 'recom': 'Buy',        'eps_fwd': 17.5,  'beat': 'Si', 'selected': False},
    'TECH':  {'win_rate': 70, 'avg_ret': 1.44, 'mcap': 10.0,   'ma200_sel': 14.2,  'rsi': 48, 'recom': 'Buy',        'eps_fwd': 342.0, 'beat': 'No', 'selected': False},
    'REGN':  {'win_rate': 70, 'avg_ret': 1.24, 'mcap': 79.4,   'ma200_sel': 21.5,  'rsi': 30, 'recom': 'Buy',        'eps_fwd': 8.1,   'beat': 'Si', 'selected': True},
    'BSX':   {'win_rate': 70, 'avg_ret': 1.03, 'mcap': 136.9,  'ma200_sel': -8.2,  'rsi': 33, 'recom': 'Strong Buy', 'eps_fwd': 85.7,  'beat': 'Si', 'selected': False},
    'AVGO':  {'win_rate': 70, 'avg_ret': 0.38, 'mcap': 1568.1, 'ma200_sel': 8.9,   'rsi': 49, 'recom': 'Strong Buy', 'eps_fwd': 198.7, 'beat': 'Si', 'selected': True},
}

syms = list(candidates.keys())

with engine.connect() as conn:
    results = []

    for sym in syms:
        c = candidates[sym]

        r_compra = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym AND ph.date = '2026-01-30'"
        ), {"sym": sym})
        row_compra = r_compra.fetchone()

        r_last = conn.execute(sqlalchemy.text(
            "SELECT ph.close, ph.date FROM price_history ph "
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
            fecha = row_last[1]
            ma200 = float(row_ma[0]) if row_ma[0] else 0
            vs_ma200_now = ((pa - ma200) / ma200) * 100 if ma200 > 0 else 0
            ret = ((pa - pc) / pc) * 100
            results.append((sym, c, pc, pa, ma200, vs_ma200_now, ret, fecha))

    results.sort(key=lambda x: x[6], reverse=True)

    print("=" * 175)
    print("  COMPORTAMIENTO DE TODOS LOS CANDIDATOS DEL SCREENER FEBRERO 2026 (desde 30/01 al 23/02)")
    print("=" * 175)
    print(f"  {'':3s} {'Symbol':6s} | {'WR':>3s} | {'AvgRet':>7s} | {'MA200sel':>8s} | {'MA200act':>9s} | {'vsMA200':>7s} | {'RSI':>3s} | {'Recom':>10s} | {'EPSfwd':>8s} | {'P.30ene':>9s} | {'P.23feb':>9s} | {'RETORNO':>8s} | {'Sel':>3s}")
    print(f"  {'':3s} {'-'*6} | {'-'*3} | {'-'*7} | {'-'*8} | {'-'*9} | {'-'*7} | {'-'*3} | {'-'*10} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*3}")

    sel_rets = []
    nosel_rets = []
    above_ma200_rets = []
    below_ma200_rets = []

    for i, (sym, c, pc, pa, ma200, vs_now, ret, fecha) in enumerate(results):
        s = "+" if ret >= 0 else ""
        vs_s = "+" if vs_now >= 0 else ""
        ma_sel_s = "+" if c['ma200_sel'] >= 0 else ""
        sel = " *" if c['selected'] else "  "
        rank = i + 1

        print(f"  {rank:2d}. {sym:6s} | {c['win_rate']:>3d} | {c['avg_ret']:>+6.1f}% | {ma_sel_s}{c['ma200_sel']:>6.1f}% | {ma200:>9.2f} | {vs_s}{vs_now:>5.1f}% | {c['rsi']:>3d} | {c['recom']:>10s} | {c['eps_fwd']:>+7.1f}% | {pc:>9.2f} | {pa:>9.2f} | {s}{ret:>7.2f}% |{sel}")

        if c['selected']:
            sel_rets.append(ret)
        else:
            nosel_rets.append(ret)

        if c['ma200_sel'] > 0:
            above_ma200_rets.append(ret)
        else:
            below_ma200_rets.append(ret)

    print(f"  {'':3s} {'-'*6} | {'-'*3} | {'-'*7} | {'-'*8} | {'-'*9} | {'-'*7} | {'-'*3} | {'-'*10} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*3}")

    avg_sel = sum(sel_rets) / len(sel_rets) if sel_rets else 0
    avg_nosel = sum(nosel_rets) / len(nosel_rets) if nosel_rets else 0
    avg_above = sum(above_ma200_rets) / len(above_ma200_rets) if above_ma200_rets else 0
    avg_below = sum(below_ma200_rets) / len(below_ma200_rets) if below_ma200_rets else 0
    avg_all = sum(r[6] for r in results) / len(results)

    pos_sel = sum(1 for r in sel_rets if r > 0)
    pos_nosel = sum(1 for r in nosel_rets if r > 0)
    pos_above = sum(1 for r in above_ma200_rets if r > 0)
    pos_below = sum(1 for r in below_ma200_rets if r > 0)

    wr80 = [r[6] for r in results if r[1]['win_rate'] >= 80]
    wr70 = [r[6] for r in results if r[1]['win_rate'] < 80]
    avg_wr80 = sum(wr80)/len(wr80) if wr80 else 0
    avg_wr70 = sum(wr70)/len(wr70) if wr70 else 0

    # Strong Buy vs Buy vs Hold
    sb_rets = [r[6] for r in results if r[1]['recom'] == 'Strong Buy']
    buy_rets = [r[6] for r in results if r[1]['recom'] == 'Buy']
    hold_rets = [r[6] for r in results if r[1]['recom'] in ('Hold', 'None')]

    # EPS Beat Si vs No
    beat_si = [r[6] for r in results if r[1]['beat'] == 'Si']
    beat_no = [r[6] for r in results if r[1]['beat'] == 'No']

    print(f"\n  {'='*80}")
    print(f"  ANALISIS COMPARATIVO")
    print(f"  {'='*80}")
    print(f"  Media TODOS los candidatos (34):          {avg_all:>+7.2f}%")
    print(f"  Media SELECCIONADAS (10):                 {avg_sel:>+7.2f}%  ({pos_sel}/10 ganadoras)")
    print(f"  Media NO seleccionadas (24):              {avg_nosel:>+7.2f}%  ({pos_nosel}/24 ganadoras)")
    print(f"")
    print(f"  --- Por Win Rate ---")
    print(f"  Win Rate >= 80% (12):                     {avg_wr80:>+7.2f}%  ({sum(1 for r in wr80 if r>0)}/{len(wr80)} ganadoras)")
    print(f"  Win Rate < 80%  (22):                     {avg_wr70:>+7.2f}%  ({sum(1 for r in wr70 if r>0)}/{len(wr70)} ganadoras)")
    print(f"")
    print(f"  --- Por MA200 al seleccionar ---")
    print(f"  MA200 > 0% (27):                          {avg_above:>+7.2f}%  ({pos_above}/27 ganadoras)")
    print(f"  MA200 < 0% (7):                           {avg_below:>+7.2f}%  ({pos_below}/7 ganadoras)")
    print(f"")
    print(f"  --- Por Recomendacion ---")
    avg_sb = sum(sb_rets)/len(sb_rets) if sb_rets else 0
    avg_buy = sum(buy_rets)/len(buy_rets) if buy_rets else 0
    avg_hold = sum(hold_rets)/len(hold_rets) if hold_rets else 0
    print(f"  Strong Buy ({len(sb_rets)}):                          {avg_sb:>+7.2f}%  ({sum(1 for r in sb_rets if r>0)}/{len(sb_rets)} ganadoras)")
    print(f"  Buy ({len(buy_rets)}):                                {avg_buy:>+7.2f}%  ({sum(1 for r in buy_rets if r>0)}/{len(buy_rets)} ganadoras)")
    print(f"  Hold/None ({len(hold_rets)}):                          {avg_hold:>+7.2f}%  ({sum(1 for r in hold_rets if r>0)}/{len(hold_rets)} ganadoras)")
    print(f"")
    print(f"  --- Por EPS Beat ---")
    avg_beat_si = sum(beat_si)/len(beat_si) if beat_si else 0
    avg_beat_no = sum(beat_no)/len(beat_no) if beat_no else 0
    print(f"  EPS Beat Si ({len(beat_si)}):                         {avg_beat_si:>+7.2f}%  ({sum(1 for r in beat_si if r>0)}/{len(beat_si)} ganadoras)")
    print(f"  EPS Beat No ({len(beat_no)}):                          {avg_beat_no:>+7.2f}%  ({sum(1 for r in beat_no if r>0)}/{len(beat_no)} ganadoras)")
    print(f"")

    # Top 10 real vs nuestra seleccion
    top10_real = results[:10]
    avg_top10 = sum(r[6] for r in top10_real) / 10
    print(f"  --- Top 10 real vs nuestra seleccion ---")
    print(f"  TOP 10 por retorno real:                  {avg_top10:>+7.2f}%")
    top10_syms = [r[0] for r in top10_real]
    our_in_top10 = [s for s in top10_syms if candidates[s]['selected']]
    print(f"  De nuestras 10, {len(our_in_top10)} estan en top 10: {', '.join(our_in_top10)}")
    missed = [s for s in top10_syms if not candidates[s]['selected']]
    print(f"  Nos perdimos: {', '.join(missed)}")
