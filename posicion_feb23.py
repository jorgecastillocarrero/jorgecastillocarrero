import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

with engine.connect() as conn:
    # Tipos de cambio del 23/02
    fx = {}
    for pair in ['EURUSD=X', 'CADEUR=X', 'CHFEUR=X']:
        r = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :pair AND ph.date = '2026-02-23'"
        ), {"pair": pair})
        row = r.fetchone()
        fx[pair] = row[0] if row else None

    eurusd = fx['EURUSD=X']
    cadeur = fx['CADEUR=X']
    chfeur = fx['CHFEUR=X']

    print("=" * 95)
    print("  POSICION 23/02/2026")
    print("=" * 95)
    print(f"  FX: EUR/USD={eurusd:.4f} | CAD/EUR={cadeur:.4f} | CHF/EUR={chfeur:.4f}")

    cuentas = ['IB', 'RCO951', 'CO3365', 'LACAIXA']
    gran_total_eur = 0

    for cuenta in cuentas:
        r = conn.execute(sqlalchemy.text(
            "SELECT symbol, shares, precio_entrada, currency, asset_type "
            "FROM holding_diario WHERE account_code = :cuenta AND fecha::date = '2026-02-23' "
            "ORDER BY asset_type, symbol"
        ), {"cuenta": cuenta})
        holdings = r.fetchall()

        # Cash
        r_cash = conn.execute(sqlalchemy.text(
            "SELECT currency, saldo FROM cash_diario "
            "WHERE account_code = :cuenta AND fecha::date = '2026-02-23'"
        ), {"cuenta": cuenta})
        cash = {row[0]: row[1] for row in r_cash.fetchall()}
        cash_eur = cash.get('EUR', 0)
        cash_usd = cash.get('USD', 0)
        cash_total_eur = cash_eur + (cash_usd / eurusd)

        print(f"\n{'=' * 95}")
        print(f"  {cuenta}")
        print(f"{'=' * 95}")

        total_holding_eur = 0

        if holdings:
            print(f"  {'Symbol':10s} | {'Shares':>10s} | {'Precio':>10s} | {'Moneda':>6s} | {'Valor EUR':>14s} | {'Tipo'}")
            print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*6} | {'-'*14} | {'-'*15}")

            for h in holdings:
                sym, shares, precio, currency, asset_type = h
                if precio and shares:
                    valor_local = shares * precio
                    if currency == 'EUR':
                        valor_eur = valor_local
                    elif currency == 'USD':
                        valor_eur = valor_local / eurusd
                    elif currency == 'CAD':
                        valor_eur = valor_local * cadeur
                    elif currency == 'CHF':
                        valor_eur = valor_local * chfeur
                    else:
                        valor_eur = valor_local / eurusd  # fallback USD

                    total_holding_eur += valor_eur
                    print(f"  {sym:10s} | {shares:>10,.0f} | {precio:>10.3f} | {currency:>6s} | {valor_eur:>14,.2f} | {asset_type}")
                else:
                    print(f"  {sym:10s} | {shares:>10,.0f} | {'N/A':>10s} | {currency:>6s} | {'--':>14s} | {asset_type}")

        print(f"  {'-'*95}")
        print(f"  Holdings:  {total_holding_eur:>14,.2f} EUR")
        print(f"  Cash EUR:  {cash_eur:>14,.2f}")
        print(f"  Cash USD:  {cash_usd:>14,.2f} ({cash_usd/eurusd:>14,.2f} EUR)")
        total_cuenta = total_holding_eur + cash_total_eur
        print(f"  TOTAL:     {total_cuenta:>14,.2f} EUR")

        gran_total_eur += total_cuenta

    print(f"\n{'=' * 95}")
    print(f"  TOTAL CARTERA:  {gran_total_eur:>14,.2f} EUR")
    print(f"{'=' * 95}")
