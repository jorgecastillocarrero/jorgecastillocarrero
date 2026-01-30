"""
Read actual broker reports to get correct share counts
"""
import pandas as pd
import sys
import os

os.chdir('C:/Users/usuario/financial-data-project')

# Read CO3365 report
print("=" * 80)
print("CO3365 - INFORME PATRIMONIO")
print("=" * 80)

try:
    df_co3365 = pd.read_excel(
        r"C:\Users\usuario\Downloads\informePatrimonio.xls",
        sheet_name=0
    )
    print("\nColumnas:", df_co3365.columns.tolist())
    print("\n")
    print(df_co3365.to_string())
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("RCO951 - INFORME PATRIMONIO")
print("=" * 80)

try:
    df_rco951 = pd.read_excel(
        r"C:\Users\usuario\Downloads\informePatrimonio (1).xls",
        sheet_name=0
    )
    print("\nColumnas:", df_rco951.columns.tolist())
    print("\n")
    # Show all rows
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(df_rco951.to_string())
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("LA CAIXA")
print("=" * 80)

try:
    df_lacaixa = pd.read_excel(
        r"C:\Users\usuario\Downloads\CarteraValores_0020001961020010700538390087_20260127_0627.xls",
        sheet_name=0
    )
    print("\nColumnas:", df_lacaixa.columns.tolist())
    print("\n")
    print(df_lacaixa.to_string())
except Exception as e:
    print(f"Error: {e}")
