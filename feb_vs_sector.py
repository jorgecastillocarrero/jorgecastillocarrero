import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy
import json

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')
fmp_engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

symbols = ['AVGO', 'SYK', 'NET', 'STLD', 'TJX', 'PAA', 'REGN', 'GE', 'IDXX', 'HEI']

sector_etfs = {
    'Technology': 'XLK', 'Healthcare': 'XLV', 'Industrials': 'XLI',
    'Consumer Cyclical': 'XLY', 'Energy': 'XLE', 'Basic Materials': 'XLB',
    'Financial Services': 'XLF', 'Consumer Defensive': 'XLP',
    'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Communication Services': 'XLC',
}

def get_return(conn, sym):
    r1 = conn.execute(sqlalchemy.text(
        "SELECT ph.close FROM price_history ph JOIN symbols s ON s.id = ph.symbol_id "
        "WHERE s.code = :sym AND ph.date BETWEEN '2026-01-27' AND '2026-01-31' "
        "ORDER BY ph.date DESC LIMIT 1"
    ), {'sym': sym})
    r2 = conn.execute(sqlalchemy.text(
        "SELECT ph.close FROM price_history ph JOIN symbols s ON s.id = ph.symbol_id "
        "WHERE s.code = :sym AND ph.date = '2026-02-26'"
    ), {'sym': sym})
    e, c = r1.fetchone(), r2.fetchone()
    if e and c:
        return ((float(c[0]) - float(e[0])) / float(e[0])) * 100
    return None

with engine.connect() as conn, fmp_engine.connect() as fconn:
    # Profiles
    profiles = {}
    for sym in symbols:
        r = fconn.execute(sqlalchemy.text(
            'SELECT sector, industry FROM fmp_profiles WHERE symbol = :sym LIMIT 1'
        ), {'sym': sym})
        row = r.fetchone()
        profiles[sym] = {'sector': row[0] if row else 'N/A', 'industry': row[1] if row else 'N/A'}

    # S&P 500 por industry
    with open('data/sp500_constituents.json') as f:
        sp500 = json.load(f)
    sp500_syms = [s['symbol'] for s in sp500]

    sp_profiles = {}
    tlist = "','".join(sp500_syms)
    r = fconn.execute(sqlalchemy.text(
        f"SELECT symbol, sector, industry FROM fmp_profiles WHERE symbol IN ('{tlist}')"
    ))
    for row in r.fetchall():
        sp_profiles[row[0]] = {'sector': row[1], 'industry': row[2]}

    industry_stocks = {}
    for sym2, prof in sp_profiles.items():
        ind = prof.get('industry')
        if ind:
            if ind not in industry_stocks:
                industry_stocks[ind] = []
            industry_stocks[ind].append(sym2)

    # SPY
    spy_ret = get_return(conn, 'SPY')

    # Calculate for each stock
    results = []
    for sym in symbols:
        stock_ret = get_return(conn, sym)
        sector = profiles[sym]['sector']
        industry = profiles[sym]['industry']
        sector_etf = sector_etfs.get(sector)

        # Sector ETF return
        sector_ret = get_return(conn, sector_etf) if sector_etf else None

        # Industry average return
        ind_stocks = industry_stocks.get(industry, [])
        ind_rets = []
        for isym in ind_stocks:
            ir = get_return(conn, isym)
            if ir is not None:
                ind_rets.append(ir)
        ind_ret = sum(ind_rets) / len(ind_rets) if ind_rets else None
        n_ind = len(ind_rets)

        vs_sector = stock_ret - sector_ret if stock_ret is not None and sector_ret is not None else None
        vs_industry = stock_ret - ind_ret if stock_ret is not None and ind_ret is not None else None

        results.append({
            'symbol': sym, 'sector': sector, 'industry': industry,
            'sector_etf': sector_etf or 'N/A',
            'stock_ret': stock_ret, 'sector_ret': sector_ret,
            'ind_ret': ind_ret, 'n_ind': n_ind,
            'vs_sector': vs_sector, 'vs_industry': vs_industry,
        })

    results.sort(key=lambda x: x['stock_ret'] or 0, reverse=True)

    print('=' * 165)
    print('  CARTERA MENSUAL FEB 2026 vs SECTOR (ETF) vs SUBSECTOR (media S&P500 industry)')
    print('  Periodo: 30/01 cierre -> 26/02 cierre | SPY: %+.2f%%' % spy_ret)
    print('=' * 165)
    hdr = '  #   Symbol | Accion  | Sector(ETF)   | vs Sector  | Subsector  | vs Subsec  | Sector               | Industry                       | N'
    print(hdr)
    print('  ' + '-' * 161)

    for i, r in enumerate(results, 1):
        sr = '%+6.2f%%' % r['stock_ret'] if r['stock_ret'] is not None else '   N/A '
        sec_r = '%+5.1f%% %s' % (r['sector_ret'], r['sector_etf']) if r['sector_ret'] is not None else '  N/A       '
        vs_s = '%+6.2f%%' % r['vs_sector'] if r['vs_sector'] is not None else '   N/A '
        ind_r = '%+6.2f%%' % r['ind_ret'] if r['ind_ret'] is not None else '   N/A '
        vs_i = '%+6.2f%%' % r['vs_industry'] if r['vs_industry'] is not None else '   N/A '

        bs = '+' if r['vs_sector'] is not None and r['vs_sector'] > 0 else '-' if r['vs_sector'] is not None else ' '
        bi = '+' if r['vs_industry'] is not None and r['vs_industry'] > 0 else '-' if r['vs_industry'] is not None else ' '

        print(f'  {i:>2d}. {r["symbol"]:6s} | {sr} | {sec_r:13s} | {bs} {vs_s} | {ind_r:10s} | {bi} {vs_i} | {r["sector"][:20]:20s} | {r["industry"][:30]:30s} | {r["n_ind"]:>2d}')

    print('  ' + '-' * 161)

    avg_stock = sum(r['stock_ret'] for r in results if r['stock_ret'] is not None) / len(results)
    vals_vs_sec = [r['vs_sector'] for r in results if r['vs_sector'] is not None]
    vals_vs_ind = [r['vs_industry'] for r in results if r['vs_industry'] is not None]
    avg_vs_sec = sum(vals_vs_sec) / len(vals_vs_sec) if vals_vs_sec else 0
    avg_vs_ind = sum(vals_vs_ind) / len(vals_vs_ind) if vals_vs_ind else 0
    beat_sec = sum(1 for v in vals_vs_sec if v > 0)
    beat_ind = sum(1 for v in vals_vs_ind if v > 0)

    print(f'      MEDIA  | {avg_stock:>+6.2f}% |               |   {avg_vs_sec:>+6.2f}% |            |   {avg_vs_ind:>+6.2f}% |')
    print(f'      Baten su sector: {beat_sec}/10 | Baten su subsector: {beat_ind}/10')
