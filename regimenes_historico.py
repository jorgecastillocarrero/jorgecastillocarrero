"""
Calculo historico de regimenes semanales + rentabilidad SPY (Fri->Fri)
Desde 2001 hasta 2026. Sin backtest, solo datos para verificar.

Flujo por semana:
  Senal:   Jue W cierre (datos para calcular regimen)
  Trading: Vie W open -> Vie W+1 open (rentabilidad real)
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
import sys

# Forzar UTF-8
sys.stdout.reconfigure(encoding='utf-8')

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# --- Constituyentes S&P 500 ---
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
symbols = [s['symbol'] for s in sp500]

profiles = pd.read_sql("SELECT symbol, industry FROM fmp_profiles WHERE symbol = ANY(%(syms)s)",
                        engine, params={'syms': symbols})
sym_to_sub = dict(zip(profiles['symbol'], profiles['industry']))

# --- Precios subsectores ---
print("Cargando precios subsectores...")
prices = pd.read_sql("""
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol = ANY(%(syms)s) AND date BETWEEN '1998-01-01' AND '2026-02-28'
    ORDER BY date
""", engine, params={'syms': symbols})
prices['date'] = pd.to_datetime(prices['date'])
prices['subsector'] = prices['symbol'].map(sym_to_sub)
prices = prices.dropna(subset=['subsector'])
sub_counts = prices.groupby('subsector')['symbol'].nunique()
valid_subs = sub_counts[sub_counts >= 3].index
prices = prices[prices['subsector'].isin(valid_subs)]
print(f"  Subsectores validos: {len(valid_subs)}")

# Semanal viernes
weekly = prices.set_index('date').groupby('subsector').resample('W-FRI')['close'].mean().reset_index()
weekly = weekly.rename(columns={'close': 'avg_close'}).sort_values(['subsector', 'date'])

def calc_metrics(grp):
    grp = grp.sort_values('date')
    high_52 = grp['avg_close'].rolling(52, min_periods=10).max()
    grp['drawdown_52w'] = (grp['avg_close'] / high_52 - 1) * 100
    delta = grp['avg_close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    grp['rsi_14w'] = 100 - (100 / (1 + rs))
    return grp

print("Calculando metricas tecnicas...")
weekly = weekly.groupby('subsector', group_keys=False).apply(calc_metrics)
dd_wide = weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = weekly.pivot(index='date', columns='subsector', values='rsi_14w')

# --- SPY diario ---
print("Cargando SPY...")
spy_daily = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '1998-01-01' AND '2026-02-28'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100

spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

# SPY viernes open para rentabilidad trading
# Si viernes es festivo, usa el siguiente dia habil (lunes)
spy_fri = spy_daily[['open']].copy()
spy_fri_only = spy_fri[spy_fri.index.dayofweek == 4]  # solo viernes
# Resample W-FRI para cubrir festivos (toma el primer open de la semana terminando viernes)
spy_weekly_fri_open = spy_daily[['open']].resample('W-FRI').first().dropna()

# --- VIX ---
vix_df = pd.read_sql("""
    SELECT date, close as vix FROM price_history_vix
    WHERE symbol='^VIX' ORDER BY date
""", engine)
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.set_index('date').sort_index()

# --- Mapeo senal -> viernes trading ---
# Senal: jueves cierre (o dia anterior si festivo)
# Trading: viernes open (o dia siguiente si festivo) -> siguiente viernes open
# En los datos, fecha_senal es viernes (W-FRI resample), trading = ese viernes open -> siguiente viernes open

def find_nearest_trading_day(target, direction='forward', tolerance=4):
    """Busca el dia de trading mas cercano a target.
    direction='forward': busca target o dias posteriores (para entrada)
    direction='backward': busca target o dias anteriores (para senal)
    """
    all_dates = spy_daily.index.tolist()
    if direction == 'forward':
        candidates = [(abs((d - target).days), d) for d in all_dates if d >= target and (d - target).days <= tolerance]
    else:
        candidates = [(abs((d - target).days), d) for d in all_dates if d <= target and (target - d).days <= tolerance]
    if candidates:
        return min(candidates, key=lambda x: x[0])[1]
    return None

# --- Calcular regimen para cada viernes desde 2001 ---
print("Calculando regimenes...")
fridays = dd_wide.index[dd_wide.index >= '2001-01-01']
print(f"  Viernes a procesar: {len(fridays)}")

results = []
prev_vix_val = None
for i, fri in enumerate(fridays):
    # Datos brutos
    dd_row = dd_wide.loc[fri]
    rsi_row = rsi_wide.loc[fri]
    n_total = dd_row.notna().sum()
    if n_total == 0:
        continue

    n_dd_h = int((dd_row > -10).sum())
    n_dd_d = int((dd_row < -20).sum())
    n_rsi_t = int(rsi_row.notna().sum())
    n_rsi_55 = int((rsi_row > 55).sum())
    pct_dd_h = n_dd_h / n_total * 100
    pct_dd_d = n_dd_d / n_total * 100
    pct_rsi = n_rsi_55 / n_rsi_t * 100 if n_rsi_t > 0 else 50

    # SPY
    spy_dates = spy_w.index[spy_w.index <= fri]
    if len(spy_dates) == 0:
        continue
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_close = spy_last['close']
    spy_ma = spy_last['ma200']
    spy_above = spy_last['above_ma200']
    spy_dist = spy_last['dist_ma200']
    spy_mom = spy_last['mom_10w']
    if not pd.notna(spy_mom): spy_mom = 0
    if not pd.notna(spy_dist): spy_dist = 0

    # VIX
    vix_dates_f = vix_df.index[vix_df.index <= fri]
    vix_val = vix_df.loc[vix_dates_f[-1], 'vix'] if len(vix_dates_f) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scores
    if pct_dd_h >= 75: s_bdd = 2.0
    elif pct_dd_h >= 60: s_bdd = 1.0
    elif pct_dd_h >= 45: s_bdd = 0.0
    elif pct_dd_h >= 30: s_bdd = -1.0
    elif pct_dd_h >= 15: s_bdd = -2.0
    else: s_bdd = -3.0

    if pct_rsi >= 75: s_brsi = 2.0
    elif pct_rsi >= 60: s_brsi = 1.0
    elif pct_rsi >= 45: s_brsi = 0.0
    elif pct_rsi >= 30: s_brsi = -1.0
    elif pct_rsi >= 15: s_brsi = -2.0
    else: s_brsi = -3.0

    if pct_dd_d <= 5: s_ddp = 1.5
    elif pct_dd_d <= 15: s_ddp = 0.5
    elif pct_dd_d <= 30: s_ddp = -0.5
    elif pct_dd_d <= 50: s_ddp = -1.5
    else: s_ddp = -2.5

    if spy_above and spy_dist > 5: s_spy = 1.5
    elif spy_above: s_spy = 0.5
    elif spy_dist > -5: s_spy = -0.5
    elif spy_dist > -15: s_spy = -1.5
    else: s_spy = -2.5

    if spy_mom > 5: s_mom = 1.0
    elif spy_mom > 0: s_mom = 0.5
    elif spy_mom > -5: s_mom = -0.5
    elif spy_mom > -15: s_mom = -1.0
    else: s_mom = -1.5

    total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

    is_burbuja = (total >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'

    vix_veto = ''
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        vix_veto = f'{regime}->NEUTRAL'
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        vix_veto = 'NEUTRAL->CAUTIOUS'
        regime = 'CAUTIOUS'

    # CAPITULACION: dentro de PANICO, si VIX baja vs semana anterior → rebote
    # RECOVERY: dentro de BEARISH, si VIX baja vs semana anterior → reversion
    vix_delta = vix_val - prev_vix_val if prev_vix_val is not None else 0
    if regime == 'PANICO' and prev_vix_val is not None and vix_delta < 0:
        regime = 'CAPITULACION'
    elif regime == 'BEARISH' and prev_vix_val is not None and vix_delta < 0:
        regime = 'RECOVERY'
    prev_vix_val = vix_val

    # Rentabilidad SPY Fri->Fri
    # Senal: jueves cierre (fecha_senal es viernes del resample)
    # Entrada: viernes open (mismo dia que fecha_senal, o siguiente habil si festivo)
    # Salida: siguiente viernes open (fri + 7 dias, o siguiente habil si festivo)
    fri_entry = find_nearest_trading_day(fri, direction='forward', tolerance=4)
    fri_exit_target = fri + pd.Timedelta(days=7)
    fri_exit = find_nearest_trading_day(fri_exit_target, direction='forward', tolerance=4)

    spy_ret = None
    spy_entry = None
    spy_exit = None
    if fri_entry and fri_exit and fri_entry in spy_daily.index and fri_exit in spy_daily.index:
        spy_entry = spy_daily.loc[fri_entry, 'open']
        spy_exit = spy_daily.loc[fri_exit, 'open']
        spy_ret = (spy_exit / spy_entry - 1) * 100

    results.append({
        'fecha_senal': fri,
        'year': fri.year,
        'sem': fri.isocalendar()[1],
        'n_sub': n_total,
        'dd_h': n_dd_h,
        'pct_dd_h': pct_dd_h,
        'dd_d': n_dd_d,
        'pct_dd_d': pct_dd_d,
        'rsi55': n_rsi_55,
        'pct_rsi': pct_rsi,
        'spy_close': spy_close,
        'spy_ma200': spy_ma,
        'spy_dist': spy_dist,
        'spy_mom': spy_mom,
        'vix': vix_val,
        'vix_delta': vix_delta,
        's_bdd': s_bdd,
        's_brsi': s_brsi,
        's_ddp': s_ddp,
        's_spy': s_spy,
        's_mom': s_mom,
        'total': total,
        'regime': regime,
        'vix_veto': vix_veto,
        'fri_entry': fri_entry,
        'fri_exit': fri_exit,
        'spy_entry_open': spy_entry,
        'spy_exit_open': spy_exit,
        'spy_ret_pct': spy_ret,
    })

df = pd.DataFrame(results)
print(f"\nTotal semanas calculadas: {len(df)}")
print(f"Rango: {df['fecha_senal'].min().strftime('%d/%m/%Y')} - {df['fecha_senal'].max().strftime('%d/%m/%Y')}")
print(f"Con rentabilidad SPY: {df['spy_ret_pct'].notna().sum()}/{len(df)}")

# --- Guardar CSV para verificacion ---
csv_path = 'data/regimenes_historico.csv'
df.to_csv(csv_path, index=False, float_format='%.4f')
print(f"\nGuardado en {csv_path}")

# --- Resumen por regimen ---
print(f"\n{'='*100}")
print(f"  RESUMEN POR REGIMEN - SPY Rent. Fri->Fri")
print(f"{'='*100}")
print(f"  {'Regimen':<12} {'N':>5} {'%Sem':>6} | {'Avg%':>7} {'Med%':>7} {'Std%':>7} {'WR%':>6} | {'Total%':>8}")
print(f"  {'-'*12} {'-'*5} {'-'*6} | {'-'*7} {'-'*7} {'-'*7} {'-'*6} | {'-'*8}")

for reg in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'RECOVERY', 'CRISIS', 'PANICO', 'CAPITULACION']:
    mask = (df['regime'] == reg) & df['spy_ret_pct'].notna()
    sub = df[mask]
    n = len(sub)
    if n == 0:
        print(f"  {reg:<12} {0:>5} {0:>5.1f}% |")
        continue
    avg = sub['spy_ret_pct'].mean()
    med = sub['spy_ret_pct'].median()
    std = sub['spy_ret_pct'].std()
    wr = (sub['spy_ret_pct'] > 0).mean() * 100
    tot = sub['spy_ret_pct'].sum()
    pct_sem = n / len(df[df['spy_ret_pct'].notna()]) * 100
    print(f"  {reg:<12} {n:>5} {pct_sem:>5.1f}% | {avg:>+7.2f} {med:>+7.2f} {std:>7.2f} {wr:>5.1f}% | {tot:>+8.1f}")

total_n = df['spy_ret_pct'].notna().sum()
total_avg = df.loc[df['spy_ret_pct'].notna(), 'spy_ret_pct'].mean()
total_sum = df.loc[df['spy_ret_pct'].notna(), 'spy_ret_pct'].sum()
print(f"  {'-'*12} {'-'*5} {'-'*6} | {'-'*7} {'-'*7} {'-'*7} {'-'*6} | {'-'*8}")
print(f"  {'TOTAL':<12} {total_n:>5} {'100%':>6} | {total_avg:>+7.2f} {'':>7} {'':>7} {'':>6} | {total_sum:>+8.1f}")

# --- Resumen por anio ---
print(f"\n{'='*100}")
print(f"  RESUMEN POR AÑO - SPY Rent. Fri->Fri")
print(f"{'='*100}")
print(f"  {'Año':>4} | {'N':>4} | {'Avg%':>7} {'Sum%':>8} {'WR%':>6} | {'BUB':>3} {'GOL':>3} {'ALC':>3} {'NEU':>3} {'CAU':>3} {'BEA':>3} {'CRI':>3} {'PAN':>3} {'CAP':>3}")
print(f"  {'-'*4} | {'-'*4} | {'-'*7} {'-'*8} {'-'*6} | {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*3}")

for year in sorted(df['year'].unique()):
    mask = (df['year'] == year) & df['spy_ret_pct'].notna()
    sub = df[mask]
    n = len(sub)
    if n == 0:
        continue
    avg = sub['spy_ret_pct'].mean()
    s = sub['spy_ret_pct'].sum()
    wr = (sub['spy_ret_pct'] > 0).mean() * 100

    reg_counts = df[df['year'] == year]['regime'].value_counts()
    bub = reg_counts.get('BURBUJA', 0)
    gol = reg_counts.get('GOLDILOCKS', 0)
    alc = reg_counts.get('ALCISTA', 0)
    neu = reg_counts.get('NEUTRAL', 0)
    cau = reg_counts.get('CAUTIOUS', 0)
    bea = reg_counts.get('BEARISH', 0)
    rec = reg_counts.get('RECOVERY', 0)
    cri = reg_counts.get('CRISIS', 0)
    pan = reg_counts.get('PANICO', 0)
    cap = reg_counts.get('CAPITULACION', 0)

    print(f"  {year:>4} | {n:>4} | {avg:>+7.2f} {s:>+8.1f} {wr:>5.1f}% | {bub:>3} {gol:>3} {alc:>3} {neu:>3} {cau:>3} {bea:>3} {rec:>3} {cri:>3} {pan:>3} {cap:>3}")

# --- Primeras y ultimas filas del CSV ---
print(f"\n{'='*100}")
print(f"  PRIMERAS 5 SEMANAS (verificacion)")
print(f"{'='*100}")
print(f"  {'Fecha':>10} {'Sem':>3} | {'DD_H':>4} {'%':>5} {'DD_D':>4} {'%':>5} {'RSI':>4} {'%':>5} | {'SPY':>7} {'Dist':>6} {'Mom':>6} {'VIX':>5} | {'BDD':>4} {'BRS':>4} {'DDP':>4} {'SPY':>4} {'MOM':>4} {'TOT':>5} | {'REG':>10} | {'Entrada':>10} {'Salida':>10} {'Ret%':>7}")
for _, r in df.head(5).iterrows():
    entry_str = r['fri_entry'].strftime('%d/%m/%Y') if pd.notna(r.get('fri_entry')) and r['fri_entry'] is not None else '---'
    exit_str = r['fri_exit'].strftime('%d/%m/%Y') if pd.notna(r.get('fri_exit')) and r['fri_exit'] is not None else '---'
    ret_str = f"{r['spy_ret_pct']:>+7.2f}" if pd.notna(r['spy_ret_pct']) else '    ---'
    print(f"  {r['fecha_senal'].strftime('%d/%m/%Y'):>10} {r['sem']:>3} | {r['dd_h']:>4} {r['pct_dd_h']:>5.1f} {r['dd_d']:>4} {r['pct_dd_d']:>5.1f} {r['rsi55']:>4} {r['pct_rsi']:>5.1f} | {r['spy_close']:>7.1f} {r['spy_dist']:>+6.1f} {r['spy_mom']:>+6.1f} {r['vix']:>5.1f} | {r['s_bdd']:>+4.1f} {r['s_brsi']:>+4.1f} {r['s_ddp']:>+4.1f} {r['s_spy']:>+4.1f} {r['s_mom']:>+4.1f} {r['total']:>+5.1f} | {r['regime']:>10} | {entry_str:>10} {exit_str:>10} {ret_str}")

print(f"\n  ULTIMAS 5 SEMANAS (2026)")
print(f"  {'Fecha':>10} {'Sem':>3} | {'DD_H':>4} {'%':>5} {'DD_D':>4} {'%':>5} {'RSI':>4} {'%':>5} | {'SPY':>7} {'Dist':>6} {'Mom':>6} {'VIX':>5} | {'BDD':>4} {'BRS':>4} {'DDP':>4} {'SPY':>4} {'MOM':>4} {'TOT':>5} | {'REG':>10} | {'Entrada':>10} {'Salida':>10} {'Ret%':>7}")
for _, r in df.tail(5).iterrows():
    entry_str = r['fri_entry'].strftime('%d/%m/%Y') if pd.notna(r.get('fri_entry')) and r['fri_entry'] is not None else '---'
    exit_str = r['fri_exit'].strftime('%d/%m/%Y') if pd.notna(r.get('fri_exit')) and r['fri_exit'] is not None else '---'
    ret_str = f"{r['spy_ret_pct']:>+7.2f}" if pd.notna(r['spy_ret_pct']) else '    ---'
    print(f"  {r['fecha_senal'].strftime('%d/%m/%Y'):>10} {r['sem']:>3} | {r['dd_h']:>4} {r['pct_dd_h']:>5.1f} {r['dd_d']:>4} {r['pct_dd_d']:>5.1f} {r['rsi55']:>4} {r['pct_rsi']:>5.1f} | {r['spy_close']:>7.1f} {r['spy_dist']:>+6.1f} {r['spy_mom']:>+6.1f} {r['vix']:>5.1f} | {r['s_bdd']:>+4.1f} {r['s_brsi']:>+4.1f} {r['s_ddp']:>+4.1f} {r['s_spy']:>+4.1f} {r['s_mom']:>+4.1f} {r['total']:>+5.1f} | {r['regime']:>10} | {entry_str:>10} {exit_str:>10} {ret_str}")
