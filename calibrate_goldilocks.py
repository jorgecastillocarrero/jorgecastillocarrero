"""Calibrar umbral GOLDILOCKS vs ALCISTA"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

print("Cargando datos...")
df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history
    WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY symbol, date
""", engine)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['subsector'] = df_all['symbol'].map(ticker_to_sub)
df_all = df_all.dropna(subset=['subsector'])
df_all['week'] = df_all['date'].dt.isocalendar().week.astype(int)
df_all['year'] = df_all['date'].dt.year

df_weekly = df_all.sort_values('date').groupby(['symbol', 'year', 'week']).last().reset_index()
df_weekly = df_weekly.sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])

sub_weekly = df_weekly.groupby(['subsector', 'date']).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean'), avg_return=('return', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])
date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_metrics)
returns_wide = sub_weekly.pivot(index='date', columns='subsector', values='avg_return')
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
                      skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

dates = returns_wide.index[returns_wide.index >= '2001-01-01']

print("Calculando scores...")
records = []
for date in dates:
    dd_row = dd_wide.loc[date] if date in dd_wide.index else None
    rsi_row = rsi_wide.loc[date] if date in rsi_wide.index else None
    if dd_row is None or rsi_row is None:
        continue
    n_total = dd_row.notna().sum()
    n_rsi = rsi_row.notna().sum()
    if n_total == 0 or n_rsi == 0:
        continue

    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi55 = (rsi_row > 55).sum() / n_rsi * 100

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) == 0:
        continue
    spy_last = spy_w.loc[spy_dates[-1]]
    spy_above = spy_last.get('above_ma200', 0)
    spy_mom = spy_last.get('mom_10w', 0)
    spy_dist = spy_last.get('dist_ma200', 0)
    if not pd.notna(spy_mom): spy_mom = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    if pct_dd_healthy >= 75: s1 = 2.0
    elif pct_dd_healthy >= 60: s1 = 1.0
    elif pct_dd_healthy >= 45: s1 = 0.0
    elif pct_dd_healthy >= 30: s1 = -1.0
    else: s1 = -2.0

    if pct_rsi55 >= 75: s2 = 2.0
    elif pct_rsi55 >= 60: s2 = 1.0
    elif pct_rsi55 >= 45: s2 = 0.0
    elif pct_rsi55 >= 30: s2 = -1.0
    else: s2 = -2.0

    if pct_dd_deep <= 5: s3 = 1.5
    elif pct_dd_deep <= 15: s3 = 0.5
    elif pct_dd_deep <= 30: s3 = -0.5
    else: s3 = -1.5

    if spy_above and spy_dist > 5: s4 = 1.5
    elif spy_above: s4 = 0.5
    elif spy_dist > -5: s4 = -0.5
    else: s4 = -1.5

    if spy_mom > 5: s5 = 1.0
    elif spy_mom > 0: s5 = 0.5
    elif spy_mom > -5: s5 = -0.5
    else: s5 = -1.0

    total = s1 + s2 + s3 + s4 + s5

    # VIX override
    if vix_val >= 30 and total >= 4.0:
        total_adj = 0.4  # forzar NEUTRAL
    elif vix_val >= 35 and total >= 0.5 and total < 4.0:
        total_adj = -0.1  # forzar CAUTIOUS
    else:
        total_adj = total

    # Retorno medio del mercado esa semana
    ret_row = returns_wide.loc[date] if date in returns_wide.index else None
    avg_ret = ret_row.mean() if ret_row is not None else 0

    records.append({
        'date': date, 'score': total_adj, 'score_raw': total,
        'avg_ret': avg_ret, 'vix': vix_val,
        'dd_h': pct_dd_healthy, 'rsi55': pct_rsi55,
    })

df = pd.DataFrame(records)

# Probar distintos umbrales GOLDILOCKS
print("\n" + "=" * 110)
print("DISTRIBUCION REGIMENES CON DISTINTOS UMBRALES GOLDILOCKS")
print("=" * 110)
print(f"\n{'Gold_th':>8} {'BURB':>6} {'GOLD':>6} {'ALCI':>6} {'NEUT':>6} {'CAUT':>6} {'BEAR':>6} {'CRIS':>6}"
      f" | {'GOLD avg%':>10} {'ALCI avg%':>10} {'GOLD WR':>8} {'ALCI WR':>8} {'Esta sem':>10}")
print("-" * 110)

target_score = df[df['date'] <= '2026-02-21']['score'].iloc[-1]

for gold_th in [5.0, 5.5, 6.0, 6.5, 7.0]:
    counts = {'BURBUJA': 0, 'GOLDILOCKS': 0, 'ALCISTA': 0, 'NEUTRAL': 0,
              'CAUTIOUS': 0, 'BEARISH': 0, 'CRISIS': 0}
    gold_rets = []
    alci_rets = []

    for _, r in df.iterrows():
        s = r['score']
        is_burb = (r['score_raw'] >= 8.0 and r['dd_h'] >= 85 and r['rsi55'] >= 90)

        if is_burb: reg = 'BURBUJA'
        elif s >= gold_th: reg = 'GOLDILOCKS'
        elif s >= 4.0: reg = 'ALCISTA'
        elif s >= 0.5: reg = 'NEUTRAL'
        elif s >= -1.5: reg = 'CAUTIOUS'
        elif s >= -3.0: reg = 'BEARISH'
        else: reg = 'CRISIS'

        counts[reg] += 1
        if reg == 'GOLDILOCKS':
            gold_rets.append(r['avg_ret'])
        elif reg == 'ALCISTA':
            alci_rets.append(r['avg_ret'])

    gold_avg = np.mean(gold_rets) * 100 if gold_rets else 0
    alci_avg = np.mean(alci_rets) * 100 if alci_rets else 0
    gold_wr = (np.array(gold_rets) > 0).mean() * 100 if gold_rets else 0
    alci_wr = (np.array(alci_rets) > 0).mean() * 100 if alci_rets else 0

    is_burb_now = (target_score >= 8.0)
    if is_burb_now: this_week = 'BURBUJA'
    elif target_score >= gold_th: this_week = 'GOLDILOCKS'
    elif target_score >= 4.0: this_week = 'ALCISTA'
    else: this_week = 'NEUTRAL'

    marker = " <<<" if 100 <= counts['GOLDILOCKS'] <= 200 else ""
    print(f"  >={gold_th:.1f}  {counts['BURBUJA']:>5} {counts['GOLDILOCKS']:>5} {counts['ALCISTA']:>5} "
          f"{counts['NEUTRAL']:>5} {counts['CAUTIOUS']:>5} {counts['BEARISH']:>5} {counts['CRISIS']:>5}"
          f" | {gold_avg:>+9.3f}% {alci_avg:>+9.3f}% {gold_wr:>7.1f}% {alci_wr:>7.1f}% {this_week:>10}{marker}")

# Detalle por ano para las mejores opciones
print("\n" + "=" * 110)
print("DETALLE ANUAL: GOLD>=6.0 vs GOLD>=6.5")
print("=" * 110)

for gold_th in [6.0, 6.5]:
    print(f"\n  GOLDILOCKS >= {gold_th}")
    print(f"  {'Ano':>5} {'BURB':>5} {'GOLD':>5} {'ALCI':>5} {'NEUT':>5} {'otros':>5}")
    print(f"  {'-'*35}")
    for yr in range(2001, 2027):
        yr_df = df[df['date'].dt.year == yr]
        bc = gc = ac = nc = oc = 0
        for _, r in yr_df.iterrows():
            s = r['score']
            is_burb = (r['score_raw'] >= 8.0 and r['dd_h'] >= 85 and r['rsi55'] >= 90)
            if is_burb: bc += 1
            elif s >= gold_th: gc += 1
            elif s >= 4.0: ac += 1
            elif s >= 0.5: nc += 1
            else: oc += 1
        if gc + ac > 0:
            print(f"  {yr:>5} {bc:>5} {gc:>5} {ac:>5} {nc:>5} {oc:>5}")
