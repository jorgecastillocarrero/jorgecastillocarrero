"""
Resumen ejecutivo ano a ano de los regimenes de mercado.
Para verificar que la clasificacion cuadra con la realidad historica.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)

# ================================================================
# CARGAR DATOS (misma lógica que report_compound.py)
# ================================================================
print("Cargando datos...")

ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id
all_tickers = list(ticker_to_sub.keys())
tlist = "','".join(all_tickers)

df_all = pd.read_sql(f"""
    SELECT symbol, date, close, high, low
    FROM fmp_price_history WHERE symbol IN ('{tlist}')
    AND date BETWEEN '2000-01-01' AND '2026-02-21' ORDER BY symbol, date
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
    avg_low=('low', 'mean')).reset_index()
sub_weekly = sub_weekly.sort_values(['subsector', 'date'])

date_counts = sub_weekly.groupby('date')['subsector'].count()
valid_dates = date_counts[date_counts >= 40].index
sub_weekly = sub_weekly[sub_weekly['date'].isin(valid_dates)]

def calc_price_metrics(g):
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

sub_weekly = sub_weekly.groupby('subsector', group_keys=False).apply(calc_price_metrics)
dd_wide = sub_weekly.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_weekly.pivot(index='date', columns='subsector', values='rsi_14w')

spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-21' ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100
spy_w = spy_daily.resample('W-FRI').last().dropna(subset=['ma200'])
spy_w['mom_10w'] = spy_w['close'].pct_change(10) * 100
spy_w['ret_spy'] = spy_w['close'].pct_change()

vix_df = pd.read_csv('C:/Users/usuario/financial-data-project/data/vix_weekly.csv',
    skiprows=3, header=None, names=['date', 'close', 'high', 'low', 'open', 'volume'])
vix_df['date'] = pd.to_datetime(vix_df['date'], format='%Y-%m-%d')
vix_df = vix_df.dropna(subset=['date']).set_index('date')
vix_df = vix_df.rename(columns={'close': 'vix'})

# ================================================================
# CLASIFICAR REGIMEN PARA CADA SEMANA
# ================================================================
def classify_regime(date):
    prev_dates = dd_wide.index[dd_wide.index <= date]
    if len(prev_dates) == 0: return 'NEUTRAL', 0.0, {}
    last_date = prev_dates[-1]
    dd_row = dd_wide.loc[last_date]
    rsi_row = rsi_wide.loc[last_date]
    n_total = dd_row.notna().sum()
    if n_total == 0: return 'NEUTRAL', 0.0, {}

    pct_dd_healthy = (dd_row > -10).sum() / n_total * 100
    pct_dd_deep = (dd_row < -20).sum() / n_total * 100
    pct_rsi_above55 = (rsi_row > 55).sum() / rsi_row.notna().sum() * 100 if rsi_row.notna().sum() > 0 else 50

    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) > 0:
        spy_last = spy_w.loc[spy_dates[-1]]
        spy_above_ma200 = spy_last.get('above_ma200', 0.5)
        spy_mom_10w = spy_last.get('mom_10w', 0)
        spy_dist = spy_last.get('dist_ma200', 0)
        spy_close = spy_last.get('close', 0)
    else:
        spy_above_ma200 = 0.5; spy_mom_10w = 0; spy_dist = 0; spy_close = 0
    if not pd.notna(spy_mom_10w): spy_mom_10w = 0
    if not pd.notna(spy_dist): spy_dist = 0

    vix_dates = vix_df.index[vix_df.index <= date]
    vix_val = vix_df.loc[vix_dates[-1], 'vix'] if len(vix_dates) > 0 else 20
    if not pd.notna(vix_val): vix_val = 20

    # Scoring extendido
    if pct_dd_healthy >= 75: s1 = 2.0
    elif pct_dd_healthy >= 60: s1 = 1.0
    elif pct_dd_healthy >= 45: s1 = 0.0
    elif pct_dd_healthy >= 30: s1 = -1.0
    elif pct_dd_healthy >= 15: s1 = -2.0
    else: s1 = -3.0

    if pct_rsi_above55 >= 75: s2 = 2.0
    elif pct_rsi_above55 >= 60: s2 = 1.0
    elif pct_rsi_above55 >= 45: s2 = 0.0
    elif pct_rsi_above55 >= 30: s2 = -1.0
    elif pct_rsi_above55 >= 15: s2 = -2.0
    else: s2 = -3.0

    if pct_dd_deep <= 5: s3 = 1.5
    elif pct_dd_deep <= 15: s3 = 0.5
    elif pct_dd_deep <= 30: s3 = -0.5
    elif pct_dd_deep <= 50: s3 = -1.5
    else: s3 = -2.5

    if spy_above_ma200 and spy_dist > 5: s4 = 1.5
    elif spy_above_ma200: s4 = 0.5
    elif spy_dist > -5: s4 = -0.5
    elif spy_dist > -15: s4 = -1.5
    else: s4 = -2.5

    if spy_mom_10w > 5: s5 = 1.0
    elif spy_mom_10w > 0: s5 = 0.5
    elif spy_mom_10w > -5: s5 = -0.5
    elif spy_mom_10w > -15: s5 = -1.0
    else: s5 = -1.5

    total = s1 + s2 + s3 + s4 + s5

    is_burbuja = (total >= 8.0 and pct_dd_healthy >= 85 and pct_rsi_above55 >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'

    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        regime = 'CAUTIOUS'

    details = {
        'score': total, 'vix': vix_val, 'spy_close': spy_close,
        'spy_dist': spy_dist, 'spy_mom': spy_mom_10w,
        'dd_healthy': pct_dd_healthy, 'dd_deep': pct_dd_deep,
        'rsi_broad': pct_rsi_above55,
    }
    return regime, total, details

# Calcular para todas las semanas
print("Clasificando regimenes semana a semana...")
all_weeks = []
for date in dd_wide.index:
    if date.year < 2001: continue
    regime, score, details = classify_regime(date)

    # SPY return semanal
    spy_ret = 0
    spy_dates = spy_w.index[spy_w.index <= date]
    if len(spy_dates) >= 2:
        spy_ret = (spy_w.loc[spy_dates[-1], 'close'] / spy_w.loc[spy_dates[-2], 'close'] - 1) * 100

    all_weeks.append({
        'date': date, 'year': date.year, 'month': date.month,
        'week_num': date.isocalendar()[1],
        'regime': regime, 'score': score,
        'spy_ret': spy_ret, **details
    })

df = pd.DataFrame(all_weeks)

# ================================================================
# RESUMEN EJECUTIVO AÑO A AÑO
# ================================================================
REGIME_ORDER = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA', 'NEUTRAL', 'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO']
REGIME_SYMBOL = {
    'BURBUJA': 'B', 'GOLDILOCKS': 'G', 'ALCISTA': 'A',
    'NEUTRAL': 'N', 'CAUTIOUS': 'c', 'BEARISH': 'b',
    'CRISIS': 'X', 'PANICO': '!'
}

print("\n" + "=" * 130)
print("  RESUMEN EJECUTIVO: REGIMENES DE MERCADO ANO A ANO (2001-2026)")
print("  Leyenda: B=BURBUJA G=GOLDILOCKS A=ALCISTA N=NEUTRAL c=CAUTIOUS b=BEARISH X=CRISIS !=PANICO")
print("=" * 130)

for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year].sort_values('date')
    if len(yr) == 0: continue

    # SPY annual
    spy_yr = spy_w[(spy_w.index.year == year) & spy_w['ret_spy'].notna()]
    spy_annual = (1 + spy_yr['ret_spy']).prod() - 1 if len(spy_yr) > 0 else 0
    spy_start = spy_yr.iloc[0]['close'] if len(spy_yr) > 0 else 0
    spy_end = spy_yr.iloc[-1]['close'] if len(spy_yr) > 0 else 0

    # Conteo regimenes
    rc = yr['regime'].value_counts()

    # Timeline mensual
    monthly_regimes = []
    for month in range(1, 13):
        m_data = yr[yr['month'] == month]
        if len(m_data) == 0:
            monthly_regimes.append('   ')
            continue
        # Regimen dominante del mes
        dom = m_data['regime'].value_counts().index[0]
        monthly_regimes.append(f" {REGIME_SYMBOL[dom]} ")

    timeline = '|'.join(monthly_regimes)

    # Score promedio y rango
    avg_score = yr['score'].mean()
    min_score = yr['score'].min()
    max_score = yr['score'].max()

    # VIX promedio
    avg_vix = yr['vix'].mean()

    print(f"\n{'-'*130}")
    print(f"  {year}  |  SPY: {spy_annual*100:+.1f}% ({spy_start:.0f} -> {spy_end:.0f})  |  "
          f"Score: avg {avg_score:+.1f} [{min_score:+.1f} a {max_score:+.1f}]  |  VIX avg: {avg_vix:.0f}")
    print(f"  {'':4s}  |  Ene |Feb |Mar |Abr |May |Jun |Jul |Ago |Sep |Oct |Nov |Dic |")
    print(f"  {'':4s}  |{timeline}|")

    # Conteo de semanas por regimen
    regime_str = '  '.join(f"{r}:{rc.get(r,0)}" for r in REGIME_ORDER if rc.get(r, 0) > 0)
    print(f"  {'':4s}  |  {len(yr)} semanas: {regime_str}")

    # Transiciones importantes (cambios de regimen)
    transitions = []
    prev_reg = None
    for _, row in yr.iterrows():
        if row['regime'] != prev_reg:
            if prev_reg is not None:
                transitions.append((row['date'].strftime('%d/%m'), prev_reg, row['regime']))
            prev_reg = row['regime']

    if transitions:
        # Mostrar solo transiciones significativas (cambio de zona positiva a negativa o vice versa)
        positive = {'BURBUJA', 'GOLDILOCKS', 'ALCISTA'}
        neutral = {'NEUTRAL'}
        negative = {'CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO'}

        def zone(r):
            if r in positive: return 'POS'
            if r in neutral: return 'NEU'
            return 'NEG'

        sig_trans = []
        for date_str, fr, to in transitions:
            if zone(fr) != zone(to) or (fr in negative and to in negative and fr != to):
                sig_trans.append(f"{date_str}: {fr}->{to}")

        if sig_trans:
            # Mostrar en lineas de max 3 transiciones
            for i in range(0, len(sig_trans), 4):
                chunk = '   '.join(sig_trans[i:i+4])
                if i == 0:
                    print(f"  {'':4s}  |  Transiciones: {chunk}")
                else:
                    print(f"  {'':4s}  |                {chunk}")

    # Detalle trimestral
    for q in range(1, 5):
        q_data = yr[yr['month'].between((q-1)*3+1, q*3)]
        if len(q_data) == 0: continue
        q_rc = q_data['regime'].value_counts()
        q_spy = q_data['spy_ret'].sum()
        q_regimes = ', '.join(f"{r}:{q_rc.get(r,0)}" for r in REGIME_ORDER if q_rc.get(r, 0) > 0)
        q_name = ['Q1 (Ene-Mar)', 'Q2 (Abr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dic)'][q-1]
        q_vix = q_data['vix'].mean()
        print(f"  {'':4s}  |  {q_name}: SPY {q_spy:+5.1f}%  VIX {q_vix:4.0f}  | {q_regimes}")

# ================================================================
# TABLA RESUMEN GLOBAL
# ================================================================
print(f"\n{'='*130}")
print(f"  TABLA RESUMEN: SEMANAS POR REGIMEN Y ANO")
print(f"{'='*130}")

header = f"  {'Año':>5} {'SPY%':>7}"
for r in REGIME_ORDER:
    abbr = r[:4]
    header += f" {abbr:>5}"
header += f" {'TOTAL':>6}  {'Zona+':>6} {'ZonaN':>6} {'Zona-':>6}"
print(header)
print(f"  {'-'*110}")

totals = {r: 0 for r in REGIME_ORDER}
total_weeks = 0

for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year]
    rc = yr['regime'].value_counts()
    spy_yr = spy_w[(spy_w.index.year == year) & spy_w['ret_spy'].notna()]
    spy_annual = ((1 + spy_yr['ret_spy']).prod() - 1) * 100 if len(spy_yr) > 0 else 0

    line = f"  {year:>5} {spy_annual:>+6.1f}%"
    for r in REGIME_ORDER:
        n = rc.get(r, 0)
        totals[r] += n
        total_weeks += n
        line += f" {n:>5}" if n > 0 else f" {'·':>5}"
    n_total = len(yr)
    n_pos = sum(rc.get(r, 0) for r in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA'])
    n_neu = rc.get('NEUTRAL', 0)
    n_neg = sum(rc.get(r, 0) for r in ['CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO'])
    line += f" {n_total:>6}  {n_pos:>6} {n_neu:>6} {n_neg:>6}"
    print(line)

# Totales
print(f"  {'-'*110}")
line = f"  {'TOTAL':>5} {'':>7}"
for r in REGIME_ORDER:
    line += f" {totals[r]:>5}"
n_all = sum(totals.values())
n_pos = sum(totals[r] for r in ['BURBUJA', 'GOLDILOCKS', 'ALCISTA'])
n_neu = totals['NEUTRAL']
n_neg = sum(totals[r] for r in ['CAUTIOUS', 'BEARISH', 'CRISIS', 'PANICO'])
line += f" {n_all:>6}  {n_pos:>6} {n_neu:>6} {n_neg:>6}"
print(line)

line = f"  {'%':>5} {'':>7}"
for r in REGIME_ORDER:
    pct = totals[r] / n_all * 100
    line += f" {pct:>4.1f}%"
line += f" {'100%':>6}  {n_pos/n_all*100:>5.1f}% {n_neu/n_all*100:>5.1f}% {n_neg/n_all*100:>5.1f}%"
print(line)
