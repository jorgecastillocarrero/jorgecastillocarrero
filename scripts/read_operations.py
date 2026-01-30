"""
Read the operations file to see what was sold in January 2026
"""
import pandas as pd

print("=" * 80)
print("OPERACIONES ENERO 2026 - BOLSA (22).xls")
print("=" * 80)

try:
    # Try to read the operations file
    xl = pd.ExcelFile(r"C:\Users\usuario\Downloads\bolsa (22).xls")
    print(f"\nHojas disponibles: {xl.sheet_names}")

    for sheet in xl.sheet_names:
        print(f"\n{'=' * 60}")
        print(f"HOJA: {sheet}")
        print("=" * 60)
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', None)
        print(df.to_string())

except Exception as e:
    print(f"Error: {e}")

# Also check if there are historical patrimonio files
print("\n" + "=" * 80)
print("BUSCANDO INFORMES HISTORICOS DE PATRIMONIO")
print("=" * 80)

import os
downloads_dir = r"C:\Users\usuario\Downloads"
for f in os.listdir(downloads_dir):
    if 'patrimonio' in f.lower() or 'cartera' in f.lower():
        print(f"  {f}")
