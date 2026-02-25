"""
SISTEMA DE SCORING COMPREHENSIVO S&P 500 v3
=============================================
REGLA: Siempre 1 = menos favorable, 10 = mas favorable para el mercado

7 Sub-scores (1-10) -> Score Compuesto (1-10)

Sub-scores:
  1. MARKET SCORE:   Media de 3 sub-scores: MA5 + MA10 + MA200 (distancia SPY)
  2. VIX SCORE:      VIX bajo=10(favorable), VIX alto=1(desfavorable)
  3. RSI SCORE:      RSI alto=10(mercado fuerte), RSI bajo=1(mercado debil)
  4. EPS GROWTH:     Crecimiento alto=10, contraccion=1
  5. EARNINGS BEAT:  Beat rate alto=10, bajo=1
  6. INFLATION:      Inflacion baja/cayendo=10(favorable), subiendo=1(desfavorable)
  7. SENTIMENT:      AAII Bull/Bear Spread directo (optimismo=10, pesimismo=1)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

print("=" * 140)
print("  SISTEMA DE SCORING COMPREHENSIVO S&P 500 v2")
print("  Regla: 1 = menos favorable | 10 = mas favorable")
print("=" * 140)

# ============================================================
# [1/7] LOAD ALL DATA
# ============================================================
print("\n[1/7] Cargando datos...")

with engine.connect() as conn:
    spy = pd.read_sql("""SELECT date, open, close FROM fmp_price_history
        WHERE symbol = 'SPY' AND date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])
    vix = pd.read_sql("SELECT date, close as vix FROM price_history_vix WHERE date >= '2003-01-01' ORDER BY date",
        conn, parse_dates=['date'])
    tip = pd.read_sql("""SELECT date, close as tip_close FROM fmp_price_history
        WHERE symbol = 'TIP' AND date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])
    ief = pd.read_sql("""SELECT date, close as ief_close FROM fmp_price_history
        WHERE symbol = 'IEF' AND date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])
    aaii = pd.read_sql("""SELECT date, bullish, bearish, bull_bear_spread
        FROM sentiment_aaii WHERE date >= '2003-01-01' ORDER BY date""",
        conn, parse_dates=['date'])

    with open('data/sp500_symbols.txt') as f:
        sp500_syms = [line.strip() for line in f if line.strip()]

    earnings = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated, revenue_actual, revenue_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s) AND eps_actual IS NOT NULL AND eps_estimated IS NOT NULL
        AND date >= '2004-01-01'
        ORDER BY date""", conn, params={"syms": sp500_syms}, parse_dates=['date'])

print(f"  SPY: {len(spy)} dias | VIX: {len(vix)} | TIP: {len(tip)} | IEF: {len(ief)} | AAII: {len(aaii)} | Earnings: {len(earnings)}")

# ============================================================
# [2/7] CALCULATE INDICATORS
# ============================================================
print("\n[2/7] Calculando indicadores...")

spy = spy.sort_values('date').reset_index(drop=True)
c = spy['close']

# MAs + distances (MA5=1sem, MA10=2sem, MA200=largo plazo)
spy['ma5'] = c.rolling(5).mean()
spy['ma10'] = c.rolling(10).mean()
spy['ma200'] = c.rolling(200).mean()
spy['dist_ma200'] = (c - spy['ma200']) / spy['ma200'] * 100
spy['dist_ma10'] = (c - spy['ma10']) / spy['ma10'] * 100
spy['dist_ma5'] = (c - spy['ma5']) / spy['ma5'] * 100

# RSI(14)
delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta).clip(lower=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
spy['rsi'] = 100 - (100 / (1 + rs))

# Merge VIX
spy = spy.merge(vix, on='date', how='left')
spy['vix'] = spy['vix'].ffill()

# Merge TIP + IEF -> Inflation expectation
spy = spy.merge(tip, on='date', how='left')
spy['tip_close'] = spy['tip_close'].ffill()
spy = spy.merge(ief, on='date', how='left')
spy['ief_close'] = spy['ief_close'].ffill()
spy['tip_ief_ratio'] = spy['tip_close'] / spy['ief_close']
spy['tip_ief_change_20d'] = (spy['tip_ief_ratio'] / spy['tip_ief_ratio'].shift(20) - 1) * 100

# Merge AAII Sentiment (semanal, jueves -> ffill al resto de dias)
spy = spy.merge(aaii[['date', 'bull_bear_spread']], on='date', how='left')
spy['bull_bear_spread'] = spy['bull_bear_spread'].ffill()

spy = spy.dropna(subset=['ma200', 'rsi']).reset_index(drop=True)
print(f"  SPY con indicadores: {len(spy)} dias ({spy['date'].min().date()} a {spy['date'].max().date()})")

# ============================================================
# [3/7] EARNINGS QUARTERLY DATA
# ============================================================
print("\n[3/7] Calculando EPS Growth y Earnings Beat Rate...")

earnings['quarter'] = earnings['date'].dt.to_period('Q')
earnings['beat'] = (earnings['eps_actual'] > earnings['eps_estimated']).astype(int)
earnings['eps_surprise_pct'] = np.where(
    earnings['eps_estimated'].abs() > 0.01,
    (earnings['eps_actual'] - earnings['eps_estimated']) / earnings['eps_estimated'].abs() * 100,
    np.where(earnings['eps_actual'] > earnings['eps_estimated'], 100, -100)
)

quarterly = earnings.groupby('quarter').agg(
    n_earnings=('beat', 'count'),
    beat_rate=('beat', 'mean'),
    avg_surprise=('eps_surprise_pct', 'mean'),
    total_eps=('eps_actual', 'sum'),
).reset_index()
quarterly['quarter_str'] = quarterly['quarter'].astype(str)
quarterly = quarterly[quarterly['n_earnings'] >= 50].reset_index(drop=True)
quarterly['eps_yoy'] = quarterly['total_eps'] / quarterly['total_eps'].shift(4) - 1
quarterly['eps_yoy_pct'] = quarterly['eps_yoy'] * 100
quarterly['quarter_end'] = quarterly['quarter'].apply(lambda q: q.end_time.date())
quarterly = quarterly.sort_values('quarter_end').reset_index(drop=True)

print(f"  Trimestres: {len(quarterly)}")
print(f"\n  Ultimos 8 trimestres:")
print(f"  {'Quarter':>8s} | {'N':>5s} | {'BeatRate':>8s} | {'Surprise':>9s} | {'EPS YoY':>8s}")
print(f"  {'-'*8} | {'-'*5} | {'-'*8} | {'-'*9} | {'-'*8}")
for _, qr in quarterly.tail(8).iterrows():
    yoy = f"{qr['eps_yoy_pct']:+.1f}%" if pd.notna(qr['eps_yoy_pct']) else "N/A"
    print(f"  {qr['quarter_str']:>8s} | {qr['n_earnings']:>5.0f} | {qr['beat_rate']*100:>7.1f}% | {qr['avg_surprise']:>+8.1f}% | {yoy:>8s}")

def get_quarterly_data_for_date(target_date):
    td = target_date.date() if hasattr(target_date, 'date') else target_date
    valid = quarterly[quarterly['quarter_end'] <= td]
    return valid.iloc[-1] if len(valid) > 0 else None

# ============================================================
# [4/7] DEFINE ALL 7 SUB-SCORES (1=desfavorable, 10=favorable)
# ============================================================
print("\n[4/7] Definiendo 7 sub-scores (1=desfavorable, 10=favorable)...\n")

score_defs = """
  ===========================================================================================
  REGLA UNIVERSAL: 1 = menos favorable para el mercado | 10 = mas favorable
  ===========================================================================================

  === SUB-SCORE 1: MARKET (media de MA5 + MA10 + MA200) ===
  MA5 = 1 semana, MA10 = 2 semanas, MA200 = largo plazo
  MISMA ESCALA para las 3 medias (0% = zona de peligro = 5):

   1 | dist < -30%         CRASH
   2 | dist -30% a -20%    BEAR FUERTE
   3 | dist -20% a -10%    BEAR
   4 | dist -10% a -7%     CORRECCION
   5 | dist -7% a +7%      ZONA PELIGRO / NEUTRAL  (0% esta aqui)
   6 | dist +7% a +10%     LIGERAMENTE ALCISTA
   7 | dist +10% a +20%    ALCISTA
   8 | dist +20% a +30%    MUY ALCISTA
   9 | dist +30% a +40%    BULL EXTREMO
  10 | dist > +40%         EUFORIA

  MARKET SCORE = round(media(MA5_score, MA10_score, MA200_score))

  === SUB-SCORE 2: VIX (bajo=favorable, VIX 18-20 elevado = 4) ===
  10 | VIX < 12       COMPLACENCIA    (mas favorable)
   9 | VIX 12-14      MUY BAJO
   8 | VIX 14-15      BAJO
   7 | VIX 15-16      NORMAL BAJO
   6 | VIX 16-17      NORMAL
   5 | VIX 17-18      NORMAL ALTO
   4 | VIX 18-20      ELEVADO
   3 | VIX 20-25      ALTO
   2 | VIX 25-35      PANICO
   1 | VIX > 35       CRASH           (menos favorable)

  === SUB-SCORE 3: RSI(14) SPY (RSI 50 = frontera, <50 = desfavorable) ===
   1 | RSI < 20       CRASH           (mercado muy debil)
   2 | RSI 20-30      MUY DEBIL
   3 | RSI 30-40      DEBIL
   4 | RSI 40-50      FLOJO           (RSI 49 = 4)
   5 | RSI 50-55      NEUTRAL
   6 | RSI 55-62      MODERADO
   7 | RSI 62-70      FUERTE
   8 | RSI 70-78      MUY FUERTE
   9 | RSI 78-85      EXTREMO
  10 | RSI > 85       EUFORIA         (mercado muy fuerte)

  === SUB-SCORE 4: EPS GROWTH YoY (SP500 agregado) ===
   1 | YoY < -20%     CONTRACCION SEVERA
   2 | YoY -20% a -10%
   3 | YoY -10% a -3%
   4 | YoY -3% a +3%  ESTANCAMIENTO
   5 | YoY +3% a +8%
   6 | YoY +8% a +15% CRECIMIENTO SANO
   7 | YoY +15% a +25%
   8 | YoY +25% a +40% BOOM
   9 | YoY +40% a +80%
  10 | YoY > +80%     EXPANSION MAXIMA

  === SUB-SCORE 5: EARNINGS BEAT RATE (SP500) ===
   1 | Beat rate < 40%
   2 | 40-50%
   3 | 50-55%
   4 | 55-60%
   5 | 60-65%
   6 | 65-70%
   7 | 70-75%         BUENO
   8 | 75-80%         EXCELENTE
   9 | 80-85%         EXCEPCIONAL
  10 | > 85%          HISTORICO

  === SUB-SCORE 6: INFLACION (TIP/IEF ratio 20d change) ===
  10 | cambio < -3%   DEFLACION FUERTE (favorable: Fed dovish)
   9 | cambio -3% a -2%
   8 | cambio -2% a -1%
   7 | cambio -1% a -0.3%
   6 | cambio -0.3% a +0.3%  NEUTRAL
   5 | cambio +0.3% a +1%
   4 | cambio +1% a +2%
   3 | cambio +2% a +3%
   2 | cambio +3% a +5%
   1 | cambio > +5%   SHOCK INFLACION (desfavorable: Fed hawkish)

  === SUB-SCORE 7: SENTIMENT AAII (Bull-Bear Spread, directo) ===
   1 | spread < -25    PANICO EXTREMO
   2 | spread -25 a -15
   3 | spread -15 a -5
   4 | spread -5 a 0    LIGERAMENTE BEARISH
   5 | spread 0 a +5    NEUTRAL
   6 | spread +5 a +15
   7 | spread +15 a +25  OPTIMISTA
   8 | spread +25 a +35
   9 | spread +35 a +50
  10 | spread > +50     EUFORIA ALCISTA
"""
print(score_defs)

# ============================================================
# SCORE FUNCTIONS
# ============================================================

# 1. MARKET SCORE = media(MA20, MA50, MA200)
# MISMA ESCALA para las 3 medias: 0% = 5 (zona peligro)
def calc_ma_score(dist):
    """Misma funcion para MA20, MA50 y MA200"""
    if pd.isna(dist): return 5
    if dist < -30: return 1
    if dist < -20: return 2
    if dist < -10: return 3
    if dist < -7:  return 4
    if dist < 7:   return 5   # -7% a +7% = zona peligro/neutral (0% aqui)
    if dist < 10:  return 6   # +7% a +10%
    if dist < 20:  return 7   # +10% a +20%
    if dist < 30:  return 8   # +20% a +30%
    if dist < 40:  return 9   # +30% a +40%
    return 10                  # > +40%

def calc_market_score(row):
    s200 = calc_ma_score(row['dist_ma200'])
    s10 = calc_ma_score(row['dist_ma10'])
    s5 = calc_ma_score(row['dist_ma5'])
    avg = (s200 + s10 + s5) / 3
    return int(round(avg))

# 2. VIX SCORE (bajo=favorable=10, VIX 18-20 elevado = 4)
def calc_vix_score(v):
    if pd.isna(v): return 5
    if v > 35:  return 1
    if v > 25:  return 2
    if v > 20:  return 3
    if v > 18:  return 4   # VIX 18-20 = ELEVADO = 4
    if v > 17:  return 5
    if v > 16:  return 6
    if v > 15:  return 7
    if v > 14:  return 8
    if v > 12:  return 9
    return 10  # VIX < 12

# 3. RSI SCORE (RSI 50 = frontera, <50 = desfavorable)
def calc_rsi_score(rsi):
    if pd.isna(rsi): return 5
    if rsi < 20: return 1
    if rsi < 30: return 2
    if rsi < 40: return 3
    if rsi < 50: return 4   # RSI 49 = 4 (por debajo de 50 = flojo)
    if rsi < 55: return 5
    if rsi < 62: return 6
    if rsi < 70: return 7
    if rsi < 78: return 8
    if rsi < 85: return 9
    return 10

# 4. EPS GROWTH SCORE (crecimiento=favorable=10)
def calc_eps_growth_score(eps_yoy_pct):
    if pd.isna(eps_yoy_pct): return 5
    if eps_yoy_pct < -20: return 1
    if eps_yoy_pct < -10: return 2
    if eps_yoy_pct < -3:  return 3
    if eps_yoy_pct < 3:   return 4
    if eps_yoy_pct < 8:   return 5
    if eps_yoy_pct < 15:  return 6
    if eps_yoy_pct < 25:  return 7
    if eps_yoy_pct < 40:  return 8
    if eps_yoy_pct < 80:  return 9
    return 10

# 5. BEAT RATE SCORE (alto=favorable=10)
def calc_beat_rate_score(beat_rate):
    if pd.isna(beat_rate): return 5
    br = beat_rate * 100
    if br < 40: return 1
    if br < 50: return 2
    if br < 55: return 3
    if br < 60: return 4
    if br < 65: return 5
    if br < 70: return 6
    if br < 75: return 7
    if br < 80: return 8
    if br < 85: return 9
    return 10

# 6. INFLATION SCORE (INVERTIDO: inflacion baja/cayendo=10=favorable, subiendo=1=desfavorable)
def calc_inflation_score(tip_ief_chg_20d):
    if pd.isna(tip_ief_chg_20d): return 6
    if tip_ief_chg_20d > 5:   return 1
    if tip_ief_chg_20d > 3:   return 2
    if tip_ief_chg_20d > 2:   return 3
    if tip_ief_chg_20d > 1:   return 4
    if tip_ief_chg_20d > 0.3: return 5
    if tip_ief_chg_20d > -0.3: return 6
    if tip_ief_chg_20d > -1:  return 7
    if tip_ief_chg_20d > -2:  return 8
    if tip_ief_chg_20d > -3:  return 9
    return 10

# 7. SENTIMENT SCORE (AAII Bull-Bear Spread, directo: optimismo=favorable=10)
def calc_sentiment_score(spread):
    if pd.isna(spread): return 5
    if spread < -25: return 1
    if spread < -15: return 2
    if spread < -5:  return 3
    if spread < 0:   return 4
    if spread < 5:   return 5
    if spread < 15:  return 6
    if spread < 25:  return 7
    if spread < 35:  return 8
    if spread < 50:  return 9
    return 10

# ============================================================
# APPLY ALL SCORES
# ============================================================
print("  Aplicando scores...")

spy['ma200_score'] = spy['dist_ma200'].apply(calc_ma_score)
spy['ma10_score'] = spy['dist_ma10'].apply(calc_ma_score)
spy['ma5_score'] = spy['dist_ma5'].apply(calc_ma_score)
spy['market_score'] = spy.apply(calc_market_score, axis=1)
spy['vix_score'] = spy['vix'].apply(calc_vix_score)
spy['rsi_score'] = spy['rsi'].apply(calc_rsi_score)
spy['inflation_score'] = spy['tip_ief_change_20d'].apply(calc_inflation_score)
spy['sentiment_score'] = spy['bull_bear_spread'].apply(calc_sentiment_score)

# EPS and Beat Rate from quarterly
spy['eps_growth_score'] = 5
spy['beat_rate_score'] = 5
spy['eps_yoy_pct'] = np.nan
spy['beat_rate_val'] = np.nan

for i, row in spy.iterrows():
    qd = get_quarterly_data_for_date(row['date'])
    if qd is not None:
        spy.at[i, 'eps_growth_score'] = calc_eps_growth_score(qd['eps_yoy_pct'] if pd.notna(qd.get('eps_yoy_pct')) else np.nan)
        spy.at[i, 'beat_rate_score'] = calc_beat_rate_score(qd['beat_rate'])
        spy.at[i, 'eps_yoy_pct'] = qd['eps_yoy_pct'] if pd.notna(qd.get('eps_yoy_pct')) else np.nan
        spy.at[i, 'beat_rate_val'] = qd['beat_rate']

# COMPOSITE SCORE (1-10) = media ponderada, TODOS ya alineados (alto=favorable)
# Pesos: Market 30%, VIX 20%, RSI 10%, EPS 15%, Beat 10%, Inflation 5%, Sentiment 10%
weights = {
    'market_score': 0.30,
    'vix_score': 0.20,
    'rsi_score': 0.10,
    'eps_growth_score': 0.15,
    'beat_rate_score': 0.10,
    'inflation_score': 0.05,
    'sentiment_score': 0.10,
}

# No need to invert anything - ALL scores are already 1=bad, 10=good
spy['composite_raw'] = sum(spy[col] * w for col, w in weights.items())

# Normalize to 1-10
comp_min = spy['composite_raw'].quantile(0.01)
comp_max = spy['composite_raw'].quantile(0.99)
spy['composite_score'] = ((spy['composite_raw'] - comp_min) / (comp_max - comp_min) * 9 + 1).clip(1, 10).round().astype(int)

print("  Todos los scores calculados")

# ============================================================
# [5/7] DISTRIBUTION
# ============================================================
print(f"\n[5/7] Distribucion de scores (2003-2026)\n")

spy_dist = spy.copy()

score_cols = ['market_score', 'ma200_score', 'ma10_score', 'ma5_score',
              'vix_score', 'rsi_score', 'eps_growth_score',
              'beat_rate_score', 'inflation_score', 'sentiment_score', 'composite_score']
score_names = ['MARKET (media)', 'MA200', 'MA10 (2sem)', 'MA5 (1sem)',
               'VIX', 'RSI', 'EPS_GROWTH',
               'BEAT_RATE', 'INFLATION', 'SENTIMENT (AAII)', 'COMPOSITE']

for col, name in zip(score_cols, score_names):
    print(f"\n  {name} SCORE:")
    for score in range(1, 11):
        cnt = (spy_dist[col] == score).sum()
        pct = cnt / len(spy_dist) * 100
        bar = '#' * int(pct * 1.5)
        label_1 = "desfavorable" if score == 1 else ""
        label_10 = "favorable" if score == 10 else ""
        print(f"    {score:>2d}: {cnt:>5d} ({pct:>5.1f}%) {bar} {label_1}{label_10}")

# ============================================================
# [6/7] WEEKLY BACKTEST
# ============================================================
print(f"\n[6/7] Backtest semanal MR_1w por cada score...\n")

with open('data/sp500_constituents.json') as f:
    current_members = json.load(f)
with open('data/sp500_historical_changes.json') as f:
    all_changes = json.load(f)
all_changes.sort(key=lambda x: x.get('date', ''), reverse=True)
current_set = {d['symbol'] for d in current_members}
all_sp500_symbols = set(current_set)
for ch in all_changes:
    if ch.get('date', '') >= '2004-01-01':
        if ch.get('removedTicker'): all_sp500_symbols.add(ch['removedTicker'])
        if ch.get('symbol'): all_sp500_symbols.add(ch['symbol'])

def get_sp500_at_date(target_date):
    members = set(current_set)
    for ch in all_changes:
        if ch.get('date', '') > str(target_date):
            if ch.get('symbol') and ch['symbol'] in members: members.discard(ch['symbol'])
            if ch.get('removedTicker'): members.add(ch['removedTicker'])
    return members

sp500_cache = {}
def get_sp500_cached(d):
    k = str(d)[:7]
    if k not in sp500_cache: sp500_cache[k] = get_sp500_at_date(d)
    return sp500_cache[k]

print("  Cargando precios acciones SP500...")
with engine.connect() as conn:
    df = pd.read_sql("""SELECT symbol, date, open, close FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2003-01-01'
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

def calc_ind(g):
    g = g.sort_values('date').copy()
    g['ret_1w'] = g['close'] / g['close'].shift(5) - 1
    return g

df = df.groupby('symbol', group_keys=False).apply(calc_ind)
df_indexed = df.set_index(['symbol', 'date']).sort_index()
df['weekday'] = df['date'].dt.weekday

fridays = np.sort(df[df['weekday'] == 4]['date'].unique())
mondays = np.sort(df[df['weekday'] == 0]['date'].unique())
spy_indexed = spy.set_index('date')

weeks = []
for fri in fridays:
    next_mons = mondays[mondays > fri]
    if len(next_mons) < 2: continue
    fri_ts = pd.Timestamp(fri)
    if fri_ts not in spy_indexed.index: continue
    sr = spy_indexed.loc[fri_ts]
    if pd.isna(sr.get('market_score')): continue
    w = {'signal_date': fri_ts, 'entry_date': pd.Timestamp(next_mons[0]), 'exit_date': pd.Timestamp(next_mons[1])}
    for sc in ['market_score', 'ma200_score', 'ma10_score', 'ma5_score',
               'vix_score', 'rsi_score', 'eps_growth_score', 'beat_rate_score',
               'inflation_score', 'sentiment_score', 'composite_score']:
        w[sc] = int(sr[sc])
    w['vix'] = sr['vix']
    w['rsi'] = sr['rsi']
    w['dist_ma200'] = sr['dist_ma200']
    weeks.append(w)

weeks_df = pd.DataFrame(weeks)
weeks_df = weeks_df[weeks_df['signal_date'] >= '2004-01-01'].reset_index(drop=True)

# Pre-build snapshots
friday_data = {}
monday_data = {}
for _, row in weeks_df.iterrows():
    sig = row['signal_date']
    if sig not in friday_data and sig in df_indexed.index.get_level_values('date'):
        friday_data[sig] = df_indexed.xs(sig, level='date', drop_level=True)
    for d in [row['entry_date'], row['exit_date']]:
        if d not in monday_data and d in df_indexed.index.get_level_values('date'):
            monday_data[d] = df_indexed.xs(d, level='date', drop_level=True)

N = 10
results = []

for _, wrow in weeks_df.iterrows():
    sig = wrow['signal_date']
    entry = wrow['entry_date']
    exit_d = wrow['exit_date']
    if sig not in friday_data or entry not in monday_data or exit_d not in monday_data: continue

    fri = friday_data[sig]
    en = monday_data[entry]
    ex = monday_data[exit_d]
    sp500 = get_sp500_cached(sig)
    elig = [s for s in fri.index if s in sp500]
    if len(elig) < 100: continue

    ret1w = fri.loc[elig, 'ret_1w'].dropna().sort_values()
    if len(ret1w) < N*2: continue

    long_syms = ret1w.head(N).index.tolist()
    short_syms = ret1w.tail(N).index.tolist()

    lr, sr_list = [], []
    for sym in long_syms:
        if sym in en.index and sym in ex.index:
            ep, xp = en.loc[sym, 'open'], ex.loc[sym, 'open']
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                lr.append((xp - ep) / ep)
    for sym in short_syms:
        if sym in en.index and sym in ex.index:
            ep, xp = en.loc[sym, 'open'], ex.loc[sym, 'open']
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                sr_list.append((ep - xp) / ep)

    if len(lr) >= 5 and len(sr_list) >= 5:
        r = {'date': sig, 'long_ret': np.mean(lr), 'short_ret': np.mean(sr_list), 'ls_ret': (np.mean(lr) + np.mean(sr_list)) / 2}
        for sc in ['market_score', 'ma200_score', 'ma10_score', 'ma5_score',
                    'vix_score', 'rsi_score', 'eps_growth_score', 'beat_rate_score',
                    'inflation_score', 'sentiment_score', 'composite_score']:
            r[sc] = wrow[sc]
        results.append(r)

rdf = pd.DataFrame(results)
print(f"  Semanas con resultados: {len(rdf)}")

# ============================================================
# RESULTS TABLES
# ============================================================

def print_score_table(data, score_col, ret_col, title):
    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"  (1=desfavorable ... 10=favorable)")
    print(f"{'='*130}")
    print(f"  {'Score':>6s} | {'Sem':>5s} | {'%':>5s} | {'AvgRet%':>8s} | {'Anual%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'PF':>5s}")
    print(f"  {'-'*6} | {'-'*5} | {'-'*5} | {'-'*8} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*5}")

    for score in range(1, 11):
        subset = data[data[score_col] == score][ret_col].dropna()
        if len(subset) < 5: continue
        rets = subset.values
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
        wr = np.mean(rets > 0) * 100
        pf = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0
        pct = len(subset) / len(data) * 100
        annual = np.mean(rets) * 52 * 100
        marker = " ***" if sharpe > 0.5 else (" **" if sharpe > 0.2 else (" *" if sharpe > 0 else ""))
        print(f"  {score:>6d} | {len(subset):>5d} | {pct:>4.1f}% | {np.mean(rets)*100:>+7.3f}% | {annual:>+6.1f}% | {sharpe:>6.2f} | {wr:>4.0f}% | {pf:>5.2f}{marker}")

# Print tables for each score x each side
all_score_cols = [
    ('composite_score', 'COMPOSITE'),
    ('market_score', 'MARKET (media MAs)'),
    ('ma200_score', 'MA200'),
    ('ma10_score', 'MA10 (2sem)'),
    ('ma5_score', 'MA5 (1sem)'),
    ('vix_score', 'VIX'),
    ('rsi_score', 'RSI'),
    ('eps_growth_score', 'EPS GROWTH'),
    ('beat_rate_score', 'BEAT RATE'),
    ('inflation_score', 'INFLACION'),
    ('sentiment_score', 'SENTIMENT (AAII)'),
]

for side_label, ret_col in [('LONG', 'long_ret'), ('SHORT', 'short_ret'), ('L/S', 'ls_ret')]:
    print(f"\n\n{'#'*130}")
    print(f"  === {side_label} (MR_1w) ===")
    print(f"{'#'*130}")
    for sc_col, sc_name in all_score_cols:
        print_score_table(rdf, sc_col, ret_col, f"{side_label} POR {sc_name}")

# ============================================================
# [7/7] CURRENT STATE
# ============================================================
print(f"\n\n{'='*130}")
print(f"  ESTADO ACTUAL DEL S&P 500")
print(f"{'='*130}")

latest = spy.iloc[-1]
latest_qd = get_quarterly_data_for_date(latest['date'])

print(f"""
  Fecha ultimo dato: {latest['date'].date()}
  SPY: {latest['close']:.2f}

  ===== INDICADORES Y SUB-SCORES =====

  MARKET SCORE (media de MAs): {int(latest['market_score']):>2d} / 10
    MA5:   {latest['ma5']:.2f}  (dist: {latest['dist_ma5']:+.1f}%)  -> sub-score MA5:   {int(latest['ma5_score'])}
    MA10:  {latest['ma10']:.2f}  (dist: {latest['dist_ma10']:+.1f}%)  -> sub-score MA10:  {int(latest['ma10_score'])}
    MA200: {latest['ma200']:.2f}  (dist: {latest['dist_ma200']:+.1f}%)  -> sub-score MA200: {int(latest['ma200_score'])}

  VIX SCORE:          {int(latest['vix_score']):>2d} / 10  (VIX={latest['vix']:.1f})
    VIX 19.09 (20/02) -> {calc_vix_score(19.09):>2d} / 10

  RSI SCORE:          {int(latest['rsi_score']):>2d} / 10  (RSI={latest['rsi']:.1f})""")

if latest_qd is not None:
    yoy = f"{latest_qd['eps_yoy_pct']:+.1f}%" if pd.notna(latest_qd.get('eps_yoy_pct')) else "N/A"
    print(f"""
  EPS GROWTH SCORE:   {int(latest['eps_growth_score']):>2d} / 10  (YoY: {yoy}, Q: {latest_qd['quarter_str']})
  BEAT RATE SCORE:    {int(latest['beat_rate_score']):>2d} / 10  (Beat: {latest_qd['beat_rate']*100:.1f}%)""")

print(f"""
  INFLATION SCORE:    {int(latest['inflation_score']):>2d} / 10  (TIP/IEF 20d: {latest['tip_ief_change_20d']:+.2f}%)
  SENTIMENT SCORE:    {int(latest['sentiment_score']):>2d} / 10  (AAII Spread: {latest['bull_bear_spread']:+.1f})

  ========================================
  >> COMPOSITE SCORE: {int(latest['composite_score']):>2d} / 10 <<
  ========================================
""")

# Historical performance for current composite
cs = int(latest['composite_score'])
print(f"  RENDIMIENTO HISTORICO con Composite Score {cs}:")
mask = rdf['composite_score'] == cs
if mask.sum() >= 5:
    ls = rdf[mask]
    for side, col in [('LONG', 'long_ret'), ('SHORT', 'short_ret'), ('L/S', 'ls_ret')]:
        rets = ls[col].values
        sh = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
        wr = np.mean(rets > 0) * 100
        ann = np.mean(rets) * 52 * 100
        ok = "SI" if sh > 0.3 else "NO"
        print(f"    {side:>5s}: {mask.sum()} sem | avg {np.mean(rets)*100:+.3f}%/sem | {ann:+.0f}% anual | Sharpe {sh:.2f} | WR {wr:.0f}% | Operar: {ok}")

# Summary chart
print(f"\n  RESUMEN TODAS LAS SCORES (1=desfavorable ... 10=favorable):")
print(f"  {'Score':>6s} | {'LONG Sharpe':>12s} | {'SHORT Sharpe':>12s} | {'L/S Sharpe':>12s}")
print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*12}")
for sc in range(1, 11):
    m = rdf['composite_score'] == sc
    if m.sum() < 5: continue
    l_sh = (rdf[m]['long_ret'].mean() / rdf[m]['long_ret'].std()) * np.sqrt(52) if rdf[m]['long_ret'].std() > 0 else 0
    s_sh = (rdf[m]['short_ret'].mean() / rdf[m]['short_ret'].std()) * np.sqrt(52) if rdf[m]['short_ret'].std() > 0 else 0
    ls_sh = (rdf[m]['ls_ret'].mean() / rdf[m]['ls_ret'].std()) * np.sqrt(52) if rdf[m]['ls_ret'].std() > 0 else 0
    l_bar = '+' * max(0, int(l_sh * 3)) if l_sh > 0 else '-' * max(0, int(-l_sh * 3))
    s_bar = '+' * max(0, int(s_sh * 3)) if s_sh > 0 else '-' * max(0, int(-s_sh * 3))
    arrow = " <<<" if sc == cs else ""
    print(f"  {sc:>6d} | {l_sh:>+6.2f} {l_bar:<5s} | {s_sh:>+6.2f} {s_bar:<5s} | {ls_sh:>+6.2f}{arrow}")

# Decision
print(f"\n  DECISION PARA COMPOSITE {cs}/10:")
if mask.sum() >= 5:
    ls = rdf[mask]
    l_sh = (ls['long_ret'].mean() / ls['long_ret'].std()) * np.sqrt(52) if ls['long_ret'].std() > 0 else 0
    s_sh = (ls['short_ret'].mean() / ls['short_ret'].std()) * np.sqrt(52) if ls['short_ret'].std() > 0 else 0
    if l_sh > 0.3 and s_sh > 0.3:
        print(f"  >> LONG + SHORT")
    elif l_sh > 0.3:
        print(f"  >> SOLO LONG (short Sharpe {s_sh:.2f} no rentable)")
    elif s_sh > 0.3:
        print(f"  >> SOLO SHORT")
    else:
        print(f"  >> CASH / REDUCIR EXPOSICION")

print(f"\n{'='*130}")
print(f"  FIN DEL ANALISIS")
print(f"{'='*130}")
