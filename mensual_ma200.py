import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

criteria = {
    'AVGO': {'sector': 'Semiconductors',   'win_rate': '70%', 'avg_ret': '+0.4%',  'ma200_sel': '+8.9%',  'rsi': 49, 'recom': 'Strong Buy', 'eps_fwd': '+198.7%'},
    'SYK':  {'sector': 'Medical Devices',   'win_rate': '70%', 'avg_ret': '+2.2%',  'ma200_sel': '-4.9%',  'rsi': 36, 'recom': 'Buy',        'eps_fwd': '+119.4%'},
    'NET':  {'sector': 'Cloud/Software',    'win_rate': '83%', 'avg_ret': '+13.3%', 'ma200_sel': '-5.5%',  'rsi': 46, 'recom': 'Buy',        'eps_fwd': '+481.2%'},
    'STLD': {'sector': 'Steel',             'win_rate': '80%', 'avg_ret': '+7.0%',  'ma200_sel': '+27.4%', 'rsi': 61, 'recom': 'Strong Buy', 'eps_fwd': '+93.4%'},
    'TJX':  {'sector': 'Apparel Retail',    'win_rate': '80%', 'avg_ret': '+1.3%',  'ma200_sel': '+7.1%',  'rsi': 19, 'recom': 'Strong Buy', 'eps_fwd': '+14.0%'},
    'PAA':  {'sector': 'Oil & Gas Midstr.', 'win_rate': '80%', 'avg_ret': '+0.3%',  'ma200_sel': '+14.8%', 'rsi': 81, 'recom': 'Buy',        'eps_fwd': '+65.2%'},
    'REGN': {'sector': 'Biotechnology',     'win_rate': '70%', 'avg_ret': '+1.2%',  'ma200_sel': '+21.5%', 'rsi': 30, 'recom': 'Buy',        'eps_fwd': '+8.1%'},
    'GE':   {'sector': 'Aerospace',         'win_rate': '80%', 'avg_ret': '+2.8%',  'ma200_sel': '+9.5%',  'rsi': 41, 'recom': 'Strong Buy', 'eps_fwd': '+5.1%'},
    'IDXX': {'sector': 'Diagnostics',       'win_rate': '70%', 'avg_ret': '+4.3%',  'ma200_sel': '+11.1%', 'rsi': 33, 'recom': 'Buy',        'eps_fwd': '+15.5%'},
    'HEI':  {'sector': 'Aerospace',         'win_rate': '80%', 'avg_ret': '+4.6%',  'ma200_sel': '+7.1%',  'rsi': 34, 'recom': 'Buy',        'eps_fwd': '+27.8%'},
}

symbols = ['AVGO', 'SYK', 'NET', 'STLD', 'TJX', 'PAA', 'REGN', 'GE', 'IDXX', 'HEI']

with engine.connect() as conn:
    results = []
    total_inv = 0
    total_act = 0
    
    for sym in symbols:
        # Compra
        r = conn.execute(sqlalchemy.text(
            "SELECT shares, precio FROM compras "
            "WHERE account_code = 'CO3365' AND symbol = :sym AND asset_type = 'Mensual' "
            "ORDER BY fecha DESC LIMIT 1"
        ), {"sym": sym})
        compra = r.fetchone()
        
        # Ultimo cierre
        r2 = conn.execute(sqlalchemy.text(
            "SELECT ph.close, ph.date FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
        ultimo = r2.fetchone()
        
        # MA200: media de los ultimos 200 cierres
        r3 = conn.execute(sqlalchemy.text(
            "SELECT AVG(ph.close) FROM ("
            "  SELECT ph.close FROM price_history ph "
            "  JOIN symbols s ON s.id = ph.symbol_id "
            "  WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 200"
            ") ph"
        ), {"sym": sym})
        ma200_row = r3.fetchone()
        
        if compra and ultimo and ma200_row:
            shares, pc = compra
            pa, fecha = ultimo
            ma200 = float(ma200_row[0]) if ma200_row[0] else 0
            vs_ma200_now = ((pa - ma200) / ma200) * 100 if ma200 > 0 else 0
            ret = ((pa - pc) / pc) * 100
            inv = shares * pc
            act = shares * pa
            total_inv += inv
            total_act += act
            c = criteria[sym]
            results.append((sym, c, shares, pc, pa, ma200, vs_ma200_now, ret, inv, act))
    
    results.sort(key=lambda x: x[7], reverse=True)
    
    print("=" * 165)
    print("  RETORNOS MENSUAL FEBRERO 2026 + CRITERIOS DE SELECCION + MA200")
    print("=" * 165)
    print(f"  {'Symbol':6s} | {'Sector':18s} | {'WinRate':>7s} | {'AvgRet':>7s} | {'MA200 Sel':>9s} | {'MA200 Act':>9s} | {'vs MA200':>8s} | {'Recom':>10s} | {'P.Compra':>9s} | {'P.Actual':>9s} | {'Retorno':>8s} | {'P&L $':>10s}")
    print(f"  {'-'*6} | {'-'*18} | {'-'*7} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*10} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*10}")
    
    for sym, c, shares, pc, pa, ma200, vs_now, ret, inv, act in results:
        s = "+" if ret >= 0 else ""
        pnl = act - inv
        sp = "+" if pnl >= 0 else ""
        vs_s = "+" if vs_now >= 0 else ""
        print(f"  {sym:6s} | {c['sector']:18s} | {c['win_rate']:>7s} | {c['avg_ret']:>7s} | {c['ma200_sel']:>9s} | {ma200:>9.2f} | {vs_s}{vs_now:>6.1f}% | {c['recom']:>10s} | {pc:>9.2f} | {pa:>9.2f} | {s}{ret:>7.2f}% | {sp}{pnl:>9,.0f}")
    
    print(f"  {'-'*6} | {'-'*18} | {'-'*7} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*10} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*10}")
    
    ret_total = ((total_act - total_inv) / total_inv) * 100
    pnl_total = total_act - total_inv
    s = "+" if ret_total >= 0 else ""
    sp = "+" if pnl_total >= 0 else ""
    print(f"  {'TOTAL':6s} | {'':18s} | {'':>7s} | {'':>7s} | {'':>9s} | {'':>9s} | {'':>8s} | {'':>10s} | {total_inv:>9,.0f} | {total_act:>9,.0f} | {s}{ret_total:>7.2f}% | {sp}{pnl_total:>9,.0f}")

