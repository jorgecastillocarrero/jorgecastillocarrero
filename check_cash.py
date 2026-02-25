import psycopg2

conn = psycopg2.connect('postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway')
cur = conn.cursor()

# Check schema
print("=== SCHEMA cash_diario ===")
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'cash_diario'")
for r in cur.fetchall():
    print(r[0])

print("\n=== SAMPLE DATA ===")
cur.execute("SELECT * FROM cash_diario WHERE account_code = 'RCO951' LIMIT 3")
cols = [desc[0] for desc in cur.description]
print(cols)
for r in cur.fetchall():
    print(r)

conn.close()
