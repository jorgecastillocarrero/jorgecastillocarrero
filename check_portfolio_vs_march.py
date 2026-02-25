import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

march_candidates = [
    'NFG', 'NEM', 'RGLD', 'CMS', 'UPS', 'STZ', 'DY', 'WEC', 'LLY', 'FDX',
    'MRK', 'INTC', 'DG', 'SCCO', 'LRCX', 'HSY', 'ADM', 'ED', 'ATO', 'CVX',
    'CL', 'EVRG', 'FAST', 'UI', 'AME', 'CTRA', 'OHI', 'VIRT', 'DHI', 'CMCSA'
]

with engine.connect() as conn:
    # Get all current holdings
    r = conn.execute(sqlalchemy.text(
        "SELECT h.account_code, h.symbol, h.asset_type, h.shares, h.currency "
        "FROM holding_diario h "
        "WHERE h.fecha = (SELECT MAX(fecha) FROM holding_diario) "
        "ORDER BY h.account_code, h.asset_type, h.symbol"
    ))

    holdings = {}
    all_symbols = set()
    for row in r.fetchall():
        acc, sym, atype, shares, curr = row
        all_symbols.add(sym)
        if sym not in holdings:
            holdings[sym] = []
        holdings[sym].append({'account': acc, 'type': atype, 'shares': shares, 'currency': curr})

    print("=" * 100)
    print("  CRUCE: CARTERA ACTUAL vs CANDIDATOS MARZO 2026")
    print("=" * 100)

    # Check overlap
    overlap = [s for s in march_candidates if s in all_symbols]
    no_overlap = [s for s in march_candidates if s not in all_symbols]

    print(f"\n  Candidatos Marzo propuestos:     {len(march_candidates)}")
    print(f"  Simbolos en cartera actual:      {len(all_symbols)}")
    print(f"  SOLAPAMIENTO:                    {len(overlap)}")
    print(f"  SIN SOLAPAR (nuevos):            {len(no_overlap)}")

    if overlap:
        print(f"\n  {'='*95}")
        print(f"  CANDIDATOS QUE YA TIENES EN CARTERA ({len(overlap)})")
        print(f"  {'='*95}")
        print(f"  {'Symbol':8s} | {'Cuenta':10s} | {'Tipo':15s} | {'Shares':>10s} | {'Moneda':>6s}")
        print(f"  {'-'*8} | {'-'*10} | {'-'*15} | {'-'*10} | {'-'*6}")
        for sym in sorted(overlap):
            for h in holdings[sym]:
                print(f"  {sym:8s} | {h['account']:10s} | {h['type']:15s} | {h['shares']:>10,.0f} | {h['currency']:>6s}")

    print(f"\n  {'='*95}")
    print(f"  CANDIDATOS NUEVOS - NO en cartera ({len(no_overlap)})")
    print(f"  {'='*95}")
    print(f"  {', '.join(sorted(no_overlap))}")

    # Now show full portfolio by account and type for context
    print(f"\n\n  {'='*95}")
    print(f"  CARTERA COMPLETA POR CUENTA Y TIPO")
    print(f"  {'='*95}")

    by_account = {}
    for sym, hlist in holdings.items():
        for h in hlist:
            key = (h['account'], h['type'])
            if key not in by_account:
                by_account[key] = []
            by_account[key].append(sym)

    for (acc, atype) in sorted(by_account.keys()):
        syms = sorted(by_account[(acc, atype)])
        print(f"\n  {acc} - {atype} ({len(syms)}):")
        print(f"    {', '.join(syms)}")
