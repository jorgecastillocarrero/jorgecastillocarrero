"""
Analyze R4 accounts to understand where the IB deposits came from
"""
import pandas as pd

# Read the R4 reports
print("=" * 80)
print("ANALISIS DE CUENTAS RENTA 4 - ORIGEN DE TRANSFERENCIAS A IB")
print("=" * 80)

# CO3365
print("\n[1] CO3365 - INFORME PATRIMONIO")
print("-" * 60)

df_co3365 = pd.read_excel(
    r"C:\Users\usuario\Downloads\informePatrimonio.xls",
    sheet_name=0,
    header=None
)

# Find the data rows
print("\nBuscando efectivo y posiciones...")
for i, row in df_co3365.iterrows():
    row_str = str(row.values)
    if 'EUR' in row_str or 'USD' in row_str or 'Efectivo' in row_str.lower() or 'cash' in row_str.lower():
        print(f"Row {i}: {row.values}")

print("\n" + "=" * 80)
print("[2] RCO951 - INFORME PATRIMONIO")
print("-" * 60)

df_rco951 = pd.read_excel(
    r"C:\Users\usuario\Downloads\informePatrimonio (1).xls",
    sheet_name=0,
    header=None
)

# Show all rows to find cash/totals
print("\nUltimas filas del informe (buscando efectivo y totales):")
for i in range(len(df_rco951) - 20, len(df_rco951)):
    if i >= 0:
        row = df_rco951.iloc[i]
        if pd.notna(row.values).any():
            print(f"Row {i}: {[x for x in row.values if pd.notna(x)]}")

# Try to read all sheets
print("\n" + "=" * 80)
print("HOJAS DISPONIBLES EN LOS ARCHIVOS")
print("-" * 60)

xl_co3365 = pd.ExcelFile(r"C:\Users\usuario\Downloads\informePatrimonio.xls")
print(f"\nCO3365 sheets: {xl_co3365.sheet_names}")

xl_rco951 = pd.ExcelFile(r"C:\Users\usuario\Downloads\informePatrimonio (1).xls")
print(f"RCO951 sheets: {xl_rco951.sheet_names}")

# Read additional sheets if available
for sheet in xl_co3365.sheet_names:
    if sheet != xl_co3365.sheet_names[0]:
        print(f"\n--- CO3365 Sheet: {sheet} ---")
        df = pd.read_excel(xl_co3365, sheet_name=sheet, header=None)
        print(df.head(20).to_string())

for sheet in xl_rco951.sheet_names:
    if sheet != xl_rco951.sheet_names[0]:
        print(f"\n--- RCO951 Sheet: {sheet} ---")
        df = pd.read_excel(xl_rco951, sheet_name=sheet, header=None)
        print(df.head(30).to_string())
