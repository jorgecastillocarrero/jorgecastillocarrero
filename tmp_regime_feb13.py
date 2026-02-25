import sqlalchemy
import pandas as pd
import numpy as np
import json

engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

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
    earnings = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s) AND eps_actual IS NOT NULL AND eps_estimated IS NOT NULL
        AND date >= '2004-01-01' ORDER BY date""", conn, params={"syms": sp500_syms}, parse_dates=['date'])

spy = spy.sort_values('date').reset_index(drop=True)
c = spy['close']
spy['ma5'] = c.rolling(5).mean()
spy['ma10'] = c.rolling(10).mean()
spy['ma200'] = c.rolling(200).mean()
spy['dist_ma200'] = (c - spy['ma200']) / spy['ma200'] * 100
spy['dist_ma10'] = (c - spy['ma10']) / spy['ma10'] * 100
spy['dist_ma5'] = (c - spy['ma5']) / spy['ma5'] * 100
delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta).clip(lower=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
spy['rsi'] = 100 - (100 / (1 + rs))
spy = spy.merge(vix, on='date', how='left'); spy['vix'] = spy['vix'].ffill()
spy = spy.merge(tip, on='date', how='left'); spy['tip_close'] = spy['tip_close'].ffill()
spy = spy.merge(ief, on='date', how='left'); spy['ief_close'] = spy['ief_close'].ffill()
spy['tip_ief_ratio'] = spy['tip_close'] / spy['ief_close']
spy['tip_ief_change_20d'] = (spy['tip_ief_ratio'] / spy['tip_ief_ratio'].shift(20) - 1) * 100
spy = spy.merge(aaii[['date', 'bull_bear_spread']], on='date', how='left')
spy['bull_bear_spread'] = spy['bull_bear_spread'].ffill()
spy = spy.dropna(subset=['ma200', 'rsi']).reset_index(drop=True)

earnings['quarter'] = earnings['date'].dt.to_period('Q')
earnings['beat'] = (earnings['eps_actual'] > earnings['eps_estimated']).astype(int)
quarterly = earnings.groupby('quarter').agg(
    n_earnings=('beat', 'count'), beat_rate=('beat', 'mean'), total_eps=('eps_actual', 'sum'),
).reset_index()
quarterly = quarterly[quarterly['n_earnings'] >= 50].reset_index(drop=True)
quarterly['eps_yoy'] = quarterly['total_eps'] / quarterly['total_eps'].shift(4) - 1
quarterly['eps_yoy_pct'] = quarterly['eps_yoy'] * 100
quarterly['quarter_end'] = quarterly['quarter'].apply(lambda q: q.end_time.date())
quarterly = quarterly.sort_values('quarter_end').reset_index(drop=True)

def get_quarterly_data_for_date(target_date):
    td = target_date.date() if hasattr(target_date, 'date') else target_date
    valid = quarterly[quarterly['quarter_end'] <= td]
    return valid.iloc[-1] if len(valid) > 0 else None

def calc_ma_score(dist):
    if pd.isna(dist): return 5
    for t, s in [(-30,1),(-20,2),(-10,3),(-7,4),(7,5),(10,6),(20,7),(30,8),(40,9)]:
        if dist < t: return s
    return 10

def calc_market_score(row):
    return int(round((calc_ma_score(row['dist_ma200']) + calc_ma_score(row['dist_ma10']) + calc_ma_score(row['dist_ma5'])) / 3))

def calc_vix_score(v):
    if pd.isna(v): return 5
    for t, s in [(35,1),(25,2),(20,3),(18,4),(17,5),(16,6),(15,7),(14,8),(12,9)]:
        if v > t: return s
    return 10

def calc_rsi_score(rsi):
    if pd.isna(rsi): return 5
    for t, s in [(20,1),(30,2),(40,3),(50,4),(55,5),(62,6),(70,7),(78,8),(85,9)]:
        if rsi < t: return s
    return 10

def calc_eps_growth_score(v):
    if pd.isna(v): return 5
    for t, s in [(-20,1),(-10,2),(-3,3),(3,4),(8,5),(15,6),(25,7),(40,8),(80,9)]:
        if v < t: return s
    return 10

def calc_beat_rate_score(br):
    if pd.isna(br): return 5
    br *= 100
    for t, s in [(40,1),(50,2),(55,3),(60,4),(65,5),(70,6),(75,7),(80,8),(85,9)]:
        if br < t: return s
    return 10

def calc_inflation_score(v):
    if pd.isna(v): return 6
    for t, s in [(5,1),(3,2),(2,3),(1,4),(0.3,5),(-0.3,6),(-1,7),(-2,8),(-3,9)]:
        if v > t: return s
    return 10

def calc_sentiment_score(v):
    if pd.isna(v): return 5
    for t, s in [(-25,1),(-15,2),(-5,3),(0,4),(5,5),(15,6),(25,7),(35,8),(50,9)]:
        if v < t: return s
    return 10

spy['market_score'] = spy.apply(calc_market_score, axis=1)
spy['vix_score'] = spy['vix'].apply(calc_vix_score)
spy['rsi_score'] = spy['rsi'].apply(calc_rsi_score)
spy['inflation_score'] = spy['tip_ief_change_20d'].apply(calc_inflation_score)
spy['sentiment_score'] = spy['bull_bear_spread'].apply(calc_sentiment_score)
spy['eps_growth_score'] = 5
spy['beat_rate_score'] = 5
for i, r in spy.iterrows():
    q = get_quarterly_data_for_date(r['date'])
    if q is not None:
        spy.at[i, 'eps_growth_score'] = calc_eps_growth_score(q['eps_yoy_pct'] if pd.notna(q.get('eps_yoy_pct')) else np.nan)
        spy.at[i, 'beat_rate_score'] = calc_beat_rate_score(q['beat_rate'])

w = {'market_score': 0.30, 'vix_score': 0.20, 'rsi_score': 0.10,
     'eps_growth_score': 0.15, 'beat_rate_score': 0.10, 'inflation_score': 0.05, 'sentiment_score': 0.10}
spy['composite_raw'] = sum(spy[col] * wt for col, wt in w.items())
comp_min = spy['composite_raw'].quantile(0.01)
comp_max = spy['composite_raw'].quantile(0.99)
spy['composite_score'] = ((spy['composite_raw'] - comp_min) / (comp_max - comp_min) * 9 + 1).clip(1, 10).round().astype(int)

feb13 = spy[spy['date'] == '2026-02-13'].iloc[0]
qd = get_quarterly_data_for_date(feb13['date'])

ms = int(feb13['market_score'])
vs = int(feb13['vix_score'])
rs_sc = int(feb13['rsi_score'])
eps_sc = int(feb13['eps_growth_score'])
beat_sc = int(feb13['beat_rate_score'])
inf_sc = int(feb13['inflation_score'])
sent_sc = int(feb13['sentiment_score'])
comp = int(feb13['composite_score'])

print('='*80)
print('  REGIMEN DE MERCADO - Viernes 13 Febrero 2026')
print('='*80)
print(f'  SPY close:         ${feb13["close"]:.2f}')
print(f'  SPY dist MA200:    {feb13["dist_ma200"]:>+.2f}%')
print(f'  SPY dist MA10:     {feb13["dist_ma10"]:>+.2f}%')
print(f'  SPY dist MA5:      {feb13["dist_ma5"]:>+.2f}%')
print(f'  SPY RSI(14):       {feb13["rsi"]:.1f}')
print(f'  VIX:               {feb13["vix"]:.1f}')
print(f'  TIP/IEF chg 20d:  {feb13["tip_ief_change_20d"]:>+.2f}%')
print(f'  AAII spread:       {feb13["bull_bear_spread"]:>+.1f}')
if qd is not None:
    print(f'  EPS YoY growth:    {qd["eps_yoy_pct"]:>+.1f}%')
    print(f'  Beat rate:         {qd["beat_rate"]*100:.1f}%')
print()
print(f'  Sub-scores (peso):')
print(f'    Market/MA (30%):     {ms}/10')
print(f'    VIX (20%):           {vs}/10')
print(f'    RSI (10%):           {rs_sc}/10')
print(f'    EPS growth (15%):    {eps_sc}/10')
print(f'    Beat rate (10%):     {beat_sc}/10')
print(f'    Inflation (5%):      {inf_sc}/10')
print(f'    Sentiment (10%):     {sent_sc}/10')
print()
regime = "BEARISH" if comp <= 3 else "NEUTRAL" if comp <= 6 else "BULLISH"
print(f'  >>> COMPOSITE SCORE:   {comp}/10  ({regime})')
print(f'  >>> Long strategy:     RANGE (Range Breakout)')
print(f'  >>> Short strategy:    MOM_4w (Momentum 4sem)')
