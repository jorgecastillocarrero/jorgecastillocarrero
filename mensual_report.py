import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

# Criteria data from CSVs
criteria = {
    'AVGO': {'sector': 'Semiconductors',   'win_rate': '70%', 'avg_ret': '+0.4%',  'ma200': '+8.9%',  'rsi': 49, 'recom': 'Strong Buy', 'eps_fwd': '+198.7%', 'eps_beat': 'Si'},
    'SYK':  {'sector': 'Medical Devices',   'win_rate': '70%', 'avg_ret': '+2.2%',  'ma200': '-4.9%',  'rsi': 36, 'recom': 'Buy',        'eps_fwd': '+119.4%', 'eps_beat': 'Si'},
    'NET':  {'sector': 'Cloud/Software',    'win_rate': '83%', 'avg_ret': '+13.3%', 'ma200': '-5.5%',  'rsi': 46, 'recom': 'Buy',        'eps_fwd': '+481.2%', 'eps_beat': 'Si'},
    'STLD': {'sector': 'Steel',             'win_rate': '80%', 'avg_ret': '+7.0%',  'ma200': '+27.4%', 'rsi': 61, 'recom': 'Strong Buy', 'eps_fwd': '+93.4%',  'eps_beat': 'Si'},
    'TJX':  {'sector': 'Apparel Retail',    'win_rate': '80%', 'avg_ret': '+1.3%',  'ma200': '+7.1%',  'rsi': 19, 'recom': 'Strong Buy', 'eps_fwd': '+14.0%',  'eps_beat': 'Si'},
    'PAA':  {'sector': 'Oil & Gas Midstr.', 'win_rate': '80%', 'avg_ret': '+0.3%',  'ma200': '+14.8%', 'rsi': 81, 'recom': 'Buy',        'eps_fwd': '+65.2%',  'eps_beat': 'Si'},
    'REGN': {'sector': 'Biotechnology',     'win_rate': '70%', 'avg_ret': '+1.2%',  'ma200': '+21.5%', 'rsi': 30, 'recom': 'Buy',        'eps_fwd': '+8.1%',   'eps_beat': 'Si'},
    'GE':   {'sector': 'Aerospace',         'win_rate': '80%', 'avg_ret': '+2.8%',  'ma200': '+9.5%',  'rsi': 41, 'recom': 'Strong Buy', 'eps_fwd': '+5.1%',   'eps_beat': 'Si'},
    'IDXX': {'sector': 'Diagnostics',       'win_rate': '70%', 'avg_ret': '+4.3%',  'ma200': '+11.1%', 'rsi': 33, 'recom': 'Buy',        'eps_fwd': '+15.5%',  'eps_beat': 'Si'},
    'HEI':  {'sector': 'Aerospace',         'win_rate': '80%', 'avg_ret': '+4.6%',  'ma200': '+7.1%',  'rsi': 34, 'recom': 'Buy',        'eps_fwd': '+27.8%',  'eps_beat': 'Si'},
}

symbols = ['AVGO', 'SYK', 'NET', 'STLD', 'TJX', 'PAA', 'REGN', 'GE', 'IDXX', 'HEI']

with engine.connect() as conn:
    results = []
    total_inv = 0
    total_act = 0
    
    for sym in symbols:
        r = conn.execute(sqlalchemy.text(
            "SELECT shares, precio FROM compras "
            "WHERE account_code = 'CO3365' AND symbol = :sym AND asset_type = 'Mensual' "
            "ORDER BY fecha DESC LIMIT 1"
        ), {"sym": sym})
        compra = r.fetchone()
        
        r2 = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
        ultimo = r2.fetchone()
        
        if compra and ultimo:
            shares, pc = compra
            pa = ultimo[0]
            ret = ((pa - pc) / pc) * 100
            inv = shares * pc
            act = shares * pa
            total_inv += inv
            total_act += act
            c = criteria[sym]
            results.append((sym, c, shares, pc, pa, ret, inv, act))
    
    results.sort(key=lambda x: x[5], reverse=True)
    
    print("=" * 145)
    print("  RETORNOS MENSUAL FEBRERO 2026 + CRITERIOS DE SELECCION")
    print("=" * 145)
    print(f"  {'Symbol':6s} | {'Sector':18s} | {'WinRate':>7s} | {'AvgRet':>7s} | {'MA200':>7s} | {'RSI':>3s} | {'Recom':>10s} | {'P.Compra':>9s} | {'P.Actual':>9s} | {'Retorno':>8s} | {'P&L $':>10s}")
    print(f"  {'-'*6} | {'-'*18} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*3} | {'-'*10} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*10}")
    
    for sym, c, shares, pc, pa, ret, inv, act in results:
        s = "+" if ret >= 0 else ""
        pnl = act - inv
        sp = "+" if pnl >= 0 else ""
        print(f"  {sym:6s} | {c['sector']:18s} | {c['win_rate']:>7s} | {c['avg_ret']:>7s} | {c['ma200']:>7s} | {c['rsi']:>3d} | {c['recom']:>10s} | {pc:>9.2f} | {pa:>9.2f} | {s}{ret:>7.2f}% | {sp}{pnl:>9,.0f}")
    
    print(f"  {'-'*6} | {'-'*18} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*3} | {'-'*10} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*10}")
    
    ret_total = ((total_act - total_inv) / total_inv) * 100
    pnl_total = total_act - total_inv
    s = "+" if ret_total >= 0 else ""
    sp = "+" if pnl_total >= 0 else ""
    print(f"  {'TOTAL':6s} | {'':18s} | {'':>7s} | {'':>7s} | {'':>7s} | {'':>3s} | {'':>10s} | {total_inv:>9,.0f} | {total_act:>9,.0f} | {s}{ret_total:>7.2f}% | {sp}{pnl_total:>9,.0f}")
    print(f"  {'':6s} | {'':18s} | {'':>7s} | {'':>7s} | {'':>7s} | {'':>3s} | {'':>10s} | {'':>9s} | {'':>9s} | {'':>8s} | {'':>10s}")
    
    # Correlacion win_rate vs retorno real
    print("\n  OBSERVACIONES:")
    wins80 = [x for x in results if x[1]['win_rate'] == '80%']
    wins70 = [x for x in results if x[1]['win_rate'] in ('70%', '83%')]
    avg80 = sum(x[5] for x in wins80) / len(wins80) if wins80 else 0
    avg70 = sum(x[5] for x in wins70) / len(wins70) if wins70 else 0
    pos = sum(1 for x in results if x[5] > 0)
    neg = sum(1 for x in results if x[5] <= 0)
    print(f"  - Win Rate 80%+ (5 acciones): retorno medio {avg80:+.2f}%")
    print(f"  - Win Rate 70%  (5 acciones): retorno medio {avg70:+.2f}%")
    print(f"  - Ganadoras: {pos} | Perdedoras: {neg}")
    print(f"  - Mejor: {results[0][0]} ({results[0][5]:+.2f}%) | Peor: {results[-1][0]} ({results[-1][5]:+.2f}%)")

