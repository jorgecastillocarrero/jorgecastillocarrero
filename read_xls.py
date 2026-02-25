import pandas as pd

df = pd.read_excel(r"C:\Users\usuario\Downloads\bolsa (36).xls")
print("=== Columnas ===")
print(list(df.columns))
print(f"\n=== Filas: {len(df)} ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 30)
print(df.to_string(index=False))
