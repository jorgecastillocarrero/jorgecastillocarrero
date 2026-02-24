"""
Analisis: 5 longs vs 10 longs en Pattern 2.
Re-ejecuta el scoring solo para semanas P2 con 5 y 10 longs.
Usa la misma infraestructura que backtest_pattern_picks.py.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
import json
import warnings
import time
warnings.filterwarnings('ignore')

from regime_pattern_longonly import OPERABLE_WEEKS_LONG

t0 = time.time()
engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

POS_SIZE = 50_000
COST = 0.003

LONG_W = {
    'sc_ret12w': 0.20, 'sc_rsi_inv': 0.10, 'sc_psar_bear': 0.12,
    'sc_st_bear': 0.08, 'sc_bb_inv': 0.08, 'sc_stoch_inv': 0.08,
    'sc_ret1w_inv': 0.12, 'sc_vol': 0.05, 'sc_margin': 0.07,
    'sc_beats': 0.05, 'sc_epsgr': 0.05,
}

print("=" * 160)
print("  ANALISIS: 5 LONGS vs 10 LONGS EN PATTERN 2")
print(f"  Semanas: {sorted(OPERABLE_WEEKS_LONG)}")
print("=" * 160)

# ============================================================
# [1] S&P 500 MEMBERSHIP (same as backtest_pattern_picks.py)
# ============================================================
print("\n[1/5] Cargando constituyentes S&P 500...")
with open('data/sp500_constituents.json') as f:
    current_members = json.load(f)
with open('data/sp500_historical_changes.json') as f:
    all_changes = json.load(f)
all_changes.sort(key=lambda x: x.get('date', ''), reverse=True)
current_set = {d['symbol'] for d in current_members}
all_sp500_symbols = set(current_set)
for ch in all_changes:
    if ch.get('date', '') >= '2003-01-01':
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

print(f"  Simbolos historicos: {len(all_sp500_symbols)}")

# ============================================================
# [2] LOAD PRICES + COMPUTE INDICATORS
# ============================================================
print(f"\n[2/5] Cargando precios... t={time.time()-t0:.0f}s")
with engine.connect() as conn:
    df = pd.read_sql("""SELECT symbol, date, open, high, low, close FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2003-01-01'
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])
print(f"  Registros: {len(df):,} | t={time.time()-t0:.0f}s")

print(f"[3/5] Calculando indicadores... t={time.time()-t0:.0f}s")
df = df.sort_values(['symbol', 'date'])
g = df.groupby('symbol')

# Returns
df['ret_1w'] = g['close'].pct_change(5)
df['ret_12w'] = g['close'].pct_change(60)

# RSI(14)
def calc_rsi(s, p=14):
    d = s.diff(); gain = d.where(d>0,0); loss = -d.where(d<0,0)
    ag = gain.rolling(p).mean(); al = loss.rolling(p).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))
df['rsi_14'] = g['close'].transform(lambda x: calc_rsi(x))

# Volatility
df['vol_5d'] = g['close'].transform(lambda x: x.pct_change().rolling(5).std())

# Bollinger Bands %B
df['bb_ma20'] = g['close'].transform(lambda x: x.rolling(20).mean())
df['bb_std20'] = g['close'].transform(lambda x: x.rolling(20).std())
df['bb_pctb'] = (df['close'] - (df['bb_ma20'] - 2*df['bb_std20'])) / (4*df['bb_std20'])

# Stochastic %K
df['low_14'] = g['low'].transform(lambda x: x.rolling(14).min())
df['high_14'] = g['high'].transform(lambda x: x.rolling(14).max())
df['stoch_k'] = (df['close'] - df['low_14']) / (df['high_14'] - df['low_14']) * 100

# Parabolic SAR
def calc_psar_dist(group):
    c = group['close'].values; h = group['high'].values; l = group['low'].values
    n = len(c)
    if n < 5: return pd.Series(np.full(n, np.nan), index=group.index)
    sar = np.full(n, np.nan); bull = True; af = 0.02; ep = h[0]; sar[0] = l[0]
    for i in range(1, n):
        if bull:
            sar[i] = sar[i-1] + af*(ep - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[max(0,i-2)])
            if l[i] < sar[i]: bull=False; sar[i]=ep; ep=l[i]; af=0.02
            else:
                if h[i]>ep: ep=h[i]; af=min(af+0.02,0.20)
        else:
            sar[i] = sar[i-1] + af*(ep - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[max(0,i-2)])
            if h[i] > sar[i]: bull=True; sar[i]=ep; ep=h[i]; af=0.02
            else:
                if l[i]<ep: ep=l[i]; af=min(af+0.02,0.20)
    return pd.Series((c - sar)/c*100, index=group.index)

df['psar_dist'] = df.groupby('symbol', group_keys=False).apply(calc_psar_dist)

# SuperTrend
def calc_st_dist(group, period=10, mult=3):
    c=group['close'].values; h=group['high'].values; l=group['low'].values; n=len(c)
    if n < period+1: return pd.Series(np.full(n, np.nan), index=group.index)
    hl2=(h+l)/2; tr=np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    tr[0]=h[0]-l[0]; atr=pd.Series(tr).rolling(period).mean().values
    ub=hl2+mult*atr; lb=hl2-mult*atr; st=np.full(n,np.nan); d=np.zeros(n)
    st[period]=ub[period]; d[period]=-1
    for i in range(period+1, n):
        if c[i-1]>st[i-1]: st[i]=max(lb[i],st[i-1]) if d[i-1]==1 else lb[i]; d[i]=1
        else: st[i]=min(ub[i],st[i-1]) if d[i-1]==-1 else ub[i]; d[i]=-1
    return pd.Series((c-st)/c*100, index=group.index)

df['st_dist'] = df.groupby('symbol', group_keys=False).apply(calc_st_dist)
print(f"  Indicadores tecnicos OK | t={time.time()-t0:.0f}s")

# Fundamentals
print(f"[4/5] Cargando fundamentales... t={time.time()-t0:.0f}s")
with engine.connect() as conn:
    ratios = pd.read_sql("""SELECT symbol, date, net_profit_margin, dividend_yield, payout_ratio, debt_equity_ratio
        FROM fmp_ratios WHERE symbol = ANY(%(syms)s)
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])
    earnings = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated
        FROM fmp_earnings WHERE symbol = ANY(%(syms)s)
        ORDER BY symbol, date""", conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

ratios = ratios.sort_values('date')
df = df.reset_index(drop=True).sort_values('date')
df = pd.merge_asof(df, ratios.rename(columns={'net_profit_margin':'net_margin','dividend_yield':'div_yield',
    'payout_ratio':'payout','debt_equity_ratio':'de_ratio'}), on='date', by='symbol', direction='backward')

earnings['beat'] = (earnings['eps_actual'] > earnings['eps_estimated']).astype(int)
earnings['eps_growth_yoy'] = earnings.groupby('symbol')['eps_actual'].pct_change(4)
earnings['beats_4q'] = earnings.groupby('symbol')['beat'].transform(lambda x: x.rolling(4, min_periods=1).sum())
earnings = earnings.sort_values('date')
df = pd.merge_asof(df, earnings[['symbol','date','beats_4q','eps_growth_yoy']],
    on='date', by='symbol', direction='backward')
print(f"  Fundamentales OK | t={time.time()-t0:.0f}s")

# ============================================================
# [5] BUILD FRIDAY SNAPSHOTS + SCORE + COMPARE 5L vs 10L
# ============================================================
print(f"\n[5/5] Scoring y picks... t={time.time()-t0:.0f}s")

# Pivot to date snapshots
df['dow'] = df['date'].dt.dayofweek
df['woy'] = df['date'].dt.isocalendar().week.astype(int)
df = df.set_index(['symbol','date']).sort_index()

# Get all Fridays in P2 weeks
fridays = df.reset_index()
fridays = fridays[(fridays['dow']==4) & (fridays['woy'].isin(OPERABLE_WEEKS_LONG))]
signal_dates = sorted(fridays['date'].unique())
print(f"  Signal dates (viernes P2): {len(signal_dates)}")

# Build date snapshots
all_dates = sorted(df.reset_index()['date'].unique())
date_idx = {d: i for i, d in enumerate(all_dates)}

def get_snap(target_date):
    """Get cross-sectional snapshot for a date."""
    try:
        return df.xs(target_date, level='date')
    except KeyError:
        return pd.DataFrame()

def score_and_pick(sig_date, n_long):
    """Score cross-section and pick top n_long stocks."""
    snap = get_snap(sig_date)
    if len(snap) < 100:
        return None

    # Find entry/exit dates
    idx = date_idx.get(sig_date)
    if idx is None:
        return None
    entry_date = None
    exit_date = None
    for j in range(idx+1, min(idx+4, len(all_dates))):
        if all_dates[j].weekday() < 5:
            entry_date = all_dates[j]
            break
    if entry_date is None:
        return None
    eidx = date_idx[entry_date]
    for j in range(eidx+4, min(eidx+8, len(all_dates))):
        if all_dates[j].weekday() < 5:
            exit_date = all_dates[j]
            break
    if exit_date is None:
        return None

    # S&P 500 members
    sp500 = get_sp500_cached(sig_date)
    eligible = [s for s in snap.index if s in sp500]
    if len(eligible) < 100:
        return None

    cross = snap.loc[eligible].copy()
    cross = cross.dropna(subset=['ret_12w', 'rsi_14', 'bb_pctb', 'stoch_k'])
    if len(cross) < 50:
        return None

    # LONG component scores
    cross['sc_ret12w'] = cross['ret_12w'].rank(pct=True) * 100
    cross['sc_ret1w_inv'] = (1 - cross['ret_1w'].rank(pct=True)) * 100
    cross['sc_psar_bear'] = np.where(cross['psar_dist'].isna(), 50,
                                      np.where(cross['psar_dist'] < 0, 100, 0))
    cross['sc_st_bear'] = np.where(cross['st_dist'].isna(), 50,
                                    np.where(cross['st_dist'] < 0, 100, 0))
    cross['sc_rsi_inv'] = (1 - cross['rsi_14'].rank(pct=True)) * 100
    cross['sc_bb_inv'] = (1 - cross['bb_pctb'].rank(pct=True)) * 100
    cross['sc_stoch_inv'] = (1 - cross['stoch_k'].rank(pct=True)) * 100
    cross['sc_vol'] = cross['vol_5d'].rank(pct=True, na_option='keep').fillna(0.5) * 100
    cross['sc_margin'] = cross['net_margin'].rank(pct=True, na_option='keep').fillna(0.5) * 100
    cross['sc_beats'] = cross['beats_4q'].fillna(2) / 4 * 100
    cross['sc_epsgr'] = cross['eps_growth_yoy'].rank(pct=True, na_option='keep').fillna(0.5) * 100

    cross['LONG_SCORE'] = sum(cross[col] * w for col, w in LONG_W.items())

    # Filter
    long_mask = (cross['ret_12w'] > 0.10) & ((cross['psar_dist'].fillna(1) < 0) | (cross['st_dist'].fillna(1) < 0))
    cands = cross[long_mask].dropna(subset=['LONG_SCORE'])
    if len(cands) < n_long:
        cands = cross[cross['ret_12w'] > 0].dropna(subset=['LONG_SCORE'])
    if len(cands) < n_long:
        cands = cross.dropna(subset=['LONG_SCORE'])

    picks = cands.nlargest(min(n_long, len(cands)), 'LONG_SCORE')

    # Get returns
    en_snap = get_snap(entry_date)
    ex_snap = get_snap(exit_date)
    if len(en_snap) == 0 or len(ex_snap) == 0:
        return None

    pnls = []
    rets = []
    symbols = []
    ranks = []
    for rank, sym in enumerate(picks.index, 1):
        if sym not in en_snap.index or sym not in ex_snap.index:
            continue
        ep = en_snap.loc[sym, 'open'] if 'open' in en_snap.columns else np.nan
        xp = ex_snap.loc[sym, 'open'] if 'open' in ex_snap.columns else np.nan
        if isinstance(ep, pd.Series): ep = ep.iloc[0]
        if isinstance(xp, pd.Series): xp = xp.iloc[0]
        if pd.isna(ep) or pd.isna(xp) or ep <= 0 or xp <= 0:
            continue
        gross = (xp - ep) / ep
        net = gross - COST
        pnl = POS_SIZE * net
        pnls.append(pnl)
        rets.append(net * 100)
        symbols.append(sym)
        ranks.append(rank)

    if len(pnls) == 0:
        return None

    woy = pd.Timestamp(sig_date).isocalendar()[1]
    year = pd.Timestamp(sig_date).year

    return {
        'Signal_Date': sig_date, 'Year': year, 'Week': woy,
        'N_Longs': n_long, 'Actual_Picks': len(pnls),
        'Long_PnL': sum(pnls), 'Long_Ret': np.mean(rets),
        'picks': list(zip(ranks, symbols, pnls, rets)),
    }

# Run for 5 and 10 longs
results = []
for i, sig_date in enumerate(signal_dates):
    if i % 50 == 0:
        print(f"  Procesando {i+1}/{len(signal_dates)} | t={time.time()-t0:.0f}s")
    for n_l in [5, 10]:
        r = score_and_pick(sig_date, n_l)
        if r:
            results.append(r)

rdf = pd.DataFrame(results)
print(f"\n  Resultados: {len(rdf)} filas | t={time.time()-t0:.0f}s")

# ============================================================
# RESULTADOS
# ============================================================
print(f"\n\n{'='*160}")
print("  COMPARATIVA: 5 LONGS vs 10 LONGS EN PATTERN 2")
print("=" * 160)

for n_l in [5, 10]:
    sub = rdf[rdf['N_Longs'] == n_l]
    n = len(sub)
    if n == 0: continue
    pnl = sub['Long_PnL'].sum()
    wr = (sub['Long_PnL'] > 0).sum() / n * 100
    rets = sub['Long_Ret'].values / 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
    capital = n_l * POS_SIZE

    print(f"\n  {n_l} LONGS x $50K = ${capital:,}/semana:")
    print(f"    Semanas: {n} | P&L: ${pnl:>+,.0f} | P&L/sem: ${pnl/n:>+,.0f}")
    print(f"    WR: {wr:.0f}% | Sharpe: {sharpe:>+.2f} | Ret/sem: {np.mean(rets)*100:>+.3f}%")

# Picks 1-5 vs 6-10
print(f"\n\n{'='*160}")
print("  DETALLE: PICKS 1-5 (top) vs PICKS 6-10 (second tier)")
print("=" * 160)

r5 = rdf[rdf['N_Longs']==5].set_index('Signal_Date')
r10 = rdf[rdf['N_Longs']==10].set_index('Signal_Date')
common = r5.index.intersection(r10.index)

pnl_5 = r5.loc[common, 'Long_PnL']
pnl_10 = r10.loc[common, 'Long_PnL']
pnl_extra = pnl_10 - pnl_5

print(f"\n  Semanas comparables: {len(common)}")
print(f"\n  {'Grupo':>15s} | {'Total P&L':>12} | {'P&L/sem':>10} | {'WR%':>5} | {'Sharpe':>7}")
print(f"  {'-'*15} | {'-'*12} | {'-'*10} | {'-'*5} | {'-'*7}")

for label, series in [('Top 5 (1-5)', pnl_5), ('Extra 5 (6-10)', pnl_extra), ('All 10', pnl_10)]:
    t = series.sum(); m = series.mean(); wr = (series>0).sum()/len(series)*100
    r = series.values / (5*POS_SIZE)  # approx ret
    sh = (np.mean(r)/np.std(r))*np.sqrt(52) if np.std(r)>0 else 0
    print(f"  {label:>15s} | ${t:>+11,.0f} | ${m:>+9,.0f} | {wr:>4.0f}% | {sh:>+6.2f}")

# Año a año
print(f"\n\n{'='*160}")
print("  AÑO A AÑO: 5 LONGS vs 10 LONGS")
print("=" * 160)
print(f"\n  {'Año':>6} | {'5L P&L':>11} | {'10L P&L':>11} | {'6-10 P&L':>11} | {'5L WR':>5} | {'10L WR':>5} | Mejor")
print(f"  {'-'*6} | {'-'*11} | {'-'*11} | {'-'*11} | {'-'*5} | {'-'*5} | {'-'*6}")

r5y = rdf[rdf['N_Longs']==5].groupby('Year').agg(
    pnl=('Long_PnL','sum'), n=('Long_PnL','count'), wins=('Long_PnL', lambda x: (x>0).sum()))
r10y = rdf[rdf['N_Longs']==10].groupby('Year').agg(
    pnl=('Long_PnL','sum'), n=('Long_PnL','count'), wins=('Long_PnL', lambda x: (x>0).sum()))

n10_better = 0
for year in sorted(set(r5y.index) & set(r10y.index)):
    p5=r5y.loc[year,'pnl']; p10=r10y.loc[year,'pnl']
    n5=r5y.loc[year,'n']; n10=r10y.loc[year,'n']
    w5=r5y.loc[year,'wins']/n5*100; w10=r10y.loc[year,'wins']/n10*100
    delta=p10-p5; better="10L" if p10>p5 else "5L"
    if p10>p5: n10_better+=1
    print(f"  {year:>6} | ${p5:>+10,.0f} | ${p10:>+10,.0f} | ${delta:>+10,.0f} | {w5:>4.0f}% | {w10:>4.0f}% | {better}")

ny = len(set(r5y.index)&set(r10y.index))
print(f"\n  10 Longs mejor en {n10_better}/{ny} años ({n10_better/ny*100:.0f}%)")

# Por semana del año
print(f"\n\n{'='*160}")
print("  POR SEMANA: DONDE LOS PICKS 6-10 APORTAN VALOR")
print("=" * 160)
print(f"\n  {'Sem':>4} | {'N':>3} | {'5L P&L':>11} | {'10L P&L':>11} | {'6-10 P&L':>11} | {'6-10 Sharpe':>11} | Veredicto")
print(f"  {'-'*4} | {'-'*3} | {'-'*11} | {'-'*11} | {'-'*11} | {'-'*11} | {'-'*20}")

for w in sorted(OPERABLE_WEEKS_LONG):
    s5 = rdf[(rdf['N_Longs']==5) & (rdf['Week']==w)]
    s10 = rdf[(rdf['N_Longs']==10) & (rdf['Week']==w)]
    if len(s5)==0 or len(s10)==0: continue
    n=len(s5); p5=s5['Long_PnL'].sum(); p10=s10['Long_PnL'].sum()
    pe = p10 - p5
    extra_pnls = s10['Long_PnL'].values - s5['Long_PnL'].values
    extra_rets = extra_pnls / (5*POS_SIZE)
    esh = (np.mean(extra_rets)/np.std(extra_rets))*np.sqrt(52) if len(extra_rets)>1 and np.std(extra_rets)>0 else 0
    verdict = "AÑADIR 10L" if pe > 0 and esh > 0.3 else "NEUTRAL" if pe > -5000 else "SOLO 5L"
    print(f"  {w:>4} | {n:>3} | ${p5:>+10,.0f} | ${p10:>+10,.0f} | ${pe:>+10,.0f} | {esh:>+10.2f} | {verdict}")

# Conclusion
total_5 = rdf[rdf['N_Longs']==5]['Long_PnL'].sum()
total_10 = rdf[rdf['N_Longs']==10]['Long_PnL'].sum()
extra = total_10 - total_5

print(f"\n\n{'='*160}")
print("  CONCLUSION FINAL")
print("=" * 160)
print(f"\n  5 Longs x $50K ($250K/sem):  ${total_5:>+,.0f}  | P&L/sem: ${total_5/len(rdf[rdf['N_Longs']==5]):>+,.0f}")
print(f"  10 Longs x $50K ($500K/sem): ${total_10:>+,.0f}  | P&L/sem: ${total_10/len(rdf[rdf['N_Longs']==10]):>+,.0f}")
print(f"  Valor picks 6-10:            ${extra:>+,.0f}")
print(f"  Capital extra:               +$250K/semana (de $250K a $500K)")

if extra > 0:
    n10 = len(rdf[rdf['N_Longs']==10])
    roi_extra = extra / (250_000 * n10) * 100
    print(f"\n  >>> 10 LONGS SUMAN VALOR: +${extra:>,.0f} extra ({roi_extra:.3f}%/sem sobre capital extra)")
    print(f"  >>> RECOMENDACION: USAR 10 LONGS en Pattern 2")
else:
    print(f"\n  >>> PICKS 6-10 RESTAN VALOR: ${extra:>,.0f}")
    print(f"  >>> RECOMENDACION: MANTENER 5 LONGS")

print(f"\n  Tiempo total: {time.time()-t0:.0f}s")
