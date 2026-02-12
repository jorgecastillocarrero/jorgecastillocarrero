"""
Monitor 2003-2008 download and then continue with older periods.
"""
import subprocess
import time
import os

LOG_2003 = 'logs/fmp_prices_2003_2008.log'

def check_if_complete(log_file):
    """Check if download is complete."""
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r') as f:
        content = f.read()
        return 'Finalizado' in content or '100.0%]' in content

def get_progress(log_file):
    """Get current progress."""
    if not os.path.exists(log_file):
        return "No log"
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if '%]' in line:
                return line.strip()[:60]
    return "Running..."

print("="*70)
print("MONITOR: Esperando que termine 2003-2008, luego continúa con anteriores")
print("="*70)

# Wait for 2003-2008 to complete
while not check_if_complete(LOG_2003):
    progress = get_progress(LOG_2003)
    print(f"[{time.strftime('%H:%M:%S')}] 2003-2008: {progress}", flush=True)
    time.sleep(60)

print("\n2003-2008 COMPLETADO! Iniciando 1997-2002 y 1991-1996...")

# Run next periods
os.chdir('C:/Users/usuario/financial-data-project')
subprocess.run(['py', '-3', 'scripts/download_historical_prices.py'], check=True)

print("\nTODAS LAS DESCARGAS COMPLETADAS!")
