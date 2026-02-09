"""Check API fields for key-metrics and ratios"""
import requests
import json

API_KEY = "PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf"

# Key Metrics
url = f"https://financialmodelingprep.com/stable/key-metrics?symbol=AAPL&apikey={API_KEY}"
resp = requests.get(url)
data = resp.json()
print("KEY METRICS FIELDS:")
if data and isinstance(data, list):
    for k in data[0].keys():
        print(f"  {k}")
else:
    print(f"  Error: {data}")

print()

# Ratios
url = f"https://financialmodelingprep.com/stable/ratios?symbol=AAPL&apikey={API_KEY}"
resp = requests.get(url)
data = resp.json()
print("RATIOS FIELDS:")
if data and isinstance(data, list):
    for k in data[0].keys():
        print(f"  {k}")
else:
    print(f"  Error: {data}")
