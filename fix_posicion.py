import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')

def calcular_posicion(conn, fecha):
    # FX rates (buscar fecha exacta o la mas cercana anterior)
    fx = {}
    for pair in ['EURUSD=X', 'CADEUR=X', 'CHFEUR=X']:
        r = conn.execute(sqlalchemy.text(
            "SELECT ph.close FROM price_history ph "
            "JOIN symbols s ON s.id = ph.symbol_id "
            "WHERE s.code = :pair AND ph.date <= :fecha ORDER BY ph.date DESC LIMIT 1"
        ), {"pair": pair, "fecha": fecha})
        row = r.fetchone()
        if row:
            fx[pair] = row[0]

    eurusd = fx.get('EURUSD=X', 1.05)
    cadeur = fx.get('CADEUR=X', 0.68)
    chfeur = fx.get('CHFEUR=X', 1.05)

    cuentas = ['IB', 'RCO951', 'CO3365', 'LACAIXA']
    resultados = []

    for cuenta in cuentas:
        # Holdings - usar precio de cierre de price_history, no precio_entrada
        r = conn.execute(sqlalchemy.text(
            "SELECT h.symbol, h.shares, h.currency, "
            "  COALESCE("
            "    (SELECT ph.close FROM price_history ph "
            "     JOIN symbols s ON s.id = ph.symbol_id "
            "     WHERE s.code = h.symbol AND ph.date <= :fecha ORDER BY ph.date DESC LIMIT 1), "
            "    h.precio_entrada"
            "  ) as precio_cierre "
            "FROM holding_diario h "
            "WHERE h.account_code = :cuenta AND h.fecha::date = :fecha"
        ), {"cuenta": cuenta, "fecha": fecha})

        holding_eur = 0
        for h in r.fetchall():
            sym, shares, currency, precio = h
            if precio and shares:
                valor_local = shares * precio
                if currency == 'EUR':
                    holding_eur += valor_local
                elif currency == 'USD':
                    holding_eur += valor_local / eurusd
                elif currency == 'CAD':
                    holding_eur += valor_local * cadeur
                elif currency == 'CHF':
                    holding_eur += valor_local * chfeur
                else:
                    holding_eur += valor_local / eurusd

        # Cash
        r2 = conn.execute(sqlalchemy.text(
            "SELECT currency, saldo FROM cash_diario "
            "WHERE account_code = :cuenta AND fecha::date = :fecha"
        ), {"cuenta": cuenta, "fecha": fecha})
        cash_eur = 0
        for c in r2.fetchall():
            if c[0] == 'EUR':
                cash_eur += c[1]
            elif c[0] == 'USD':
                cash_eur += c[1] / eurusd
            elif c[0] == 'CAD':
                cash_eur += c[1] * cadeur
            elif c[0] == 'CHF':
                cash_eur += c[1] * chfeur

        total_eur = holding_eur + cash_eur
        resultados.append({
            "fecha": fecha,
            "account": cuenta,
            "holding_eur": holding_eur,
            "cash_eur": cash_eur,
            "total_eur": total_eur
        })

    return resultados


with engine.begin() as conn:
    for fecha in ['2026-02-20', '2026-02-23']:
        print(f"\n=== Posicion {fecha} ===")

        conn.execute(sqlalchemy.text(
            "DELETE FROM posicion WHERE fecha::date = :fecha"
        ), {"fecha": fecha})

        resultados = calcular_posicion(conn, fecha)
        total_fecha = 0
        for r in resultados:
            conn.execute(sqlalchemy.text(
                "INSERT INTO posicion (fecha, account_code, holding_eur, cash_eur, total_eur) "
                "VALUES (:fecha, :account, :holding, :cash, :total)"
            ), {
                "fecha": r["fecha"],
                "account": r["account"],
                "holding": r["holding_eur"],
                "cash": r["cash_eur"],
                "total": r["total_eur"]
            })
            total_fecha += r["total_eur"]
            print(f"  {r['account']:8s} | holding: {r['holding_eur']:>14,.2f} | cash: {r['cash_eur']:>14,.2f} | total: {r['total_eur']:>14,.2f}")
        print(f"  {'TOTAL':8s} | {' ':>14s} | {' ':>14s} | {total_fecha:>14,.2f} EUR")
