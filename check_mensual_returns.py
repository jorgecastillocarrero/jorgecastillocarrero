import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

symbols = ['AVGO', 'SYK', 'NET', 'STLD', 'TJX', 'PAA', 'REGN', 'GE', 'IDXX', 'HEI']

with engine.connect() as conn:
    # Get purchase info from compras
    print("=== RETORNOS MENSUAL FEBRERO 2026 ===\n")
    
    total_invertido = 0
    total_actual = 0
    
    results = []
    for sym in symbols:
        # Purchase price and shares from compras
        r = conn.execute(sqlalchemy.text(
            "SELECT fecha, shares, precio, currency FROM compras "
            "WHERE account_code = 'CO3365' AND symbol = :sym AND asset_type = 'Mensual' "
            "ORDER BY fecha DESC LIMIT 1"
        ), {"sym": sym})
        compra = r.fetchone()
        
        # Latest closing price from price_history
        r2 = conn.execute(sqlalchemy.text(
            "SELECT ph.date, ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :sym ORDER BY ph.date DESC LIMIT 1"
        ), {"sym": sym})
        ultimo = r2.fetchone()
        
        if compra and ultimo:
            fecha_compra, shares, precio_compra, currency = compra
            fecha_cierre, precio_cierre = ultimo
            
            valor_compra = shares * precio_compra
            valor_actual = shares * precio_cierre
            retorno_pct = ((precio_cierre - precio_compra) / precio_compra) * 100
            retorno_usd = valor_actual - valor_compra
            
            total_invertido += valor_compra
            total_actual += valor_actual
            
            results.append((sym, shares, precio_compra, precio_cierre, fecha_cierre, retorno_pct, retorno_usd, valor_compra, valor_actual))
    
    # Sort by return
    results.sort(key=lambda x: x[5], reverse=True)
    
    print(f"  {'Symbol':8s} | {'Shares':>8s} | {'P.Compra':>10s} | {'P.Actual':>10s} | {'Fecha':>12s} | {'Retorno%':>9s} | {'Retorno$':>12s}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*9} | {'-'*12}")
    
    for sym, shares, pc, pa, fecha, ret_pct, ret_usd, vc, va in results:
        signo = "+" if ret_pct >= 0 else ""
        print(f"  {sym:8s} | {shares:>8,.0f} | {pc:>10.2f} | {pa:>10.2f} | {str(fecha):>12s} | {signo}{ret_pct:>8.2f}% | {signo}{ret_usd:>11,.2f}")
    
    print(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*9} | {'-'*12}")
    
    retorno_total_pct = ((total_actual - total_invertido) / total_invertido) * 100
    retorno_total_usd = total_actual - total_invertido
    signo = "+" if retorno_total_pct >= 0 else ""
    print(f"  {'TOTAL':8s} | {'':>8s} | {total_invertido:>10,.0f} | {total_actual:>10,.0f} | {'':>12s} | {signo}{retorno_total_pct:>8.2f}% | {signo}{retorno_total_usd:>11,.2f}")

