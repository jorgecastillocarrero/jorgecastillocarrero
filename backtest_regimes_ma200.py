"""
ANALISIS DE REGIMENES DE MERCADO basado en MA200
=================================================
Definir tipos de mercado usando SPY vs MA200:
- Distancia a MA200 (%, por rangos)
- Pendiente de MA200 (subiendo/bajando)
- Velocidad de aproximacion/alejamiento
- Cruce reciente
Cruzar con rendimiento LONG, SHORT y L/S
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
print("  ANALISIS DE REGIMENES: SPY vs MA200")
print("  Definir tipos de mercado y rendimiento de cada estrategia por regimen")
print("=" * 140)

# ============================================================
# Load data
# ============================================================
print("\n[1/5] Cargando datos...")

with engine.connect() as conn:
    spy = pd.read_sql("""SELECT date, open, high, low, close, volume
        FROM fmp_price_history WHERE symbol = 'SPY' AND date >= '2003-01-01'
        ORDER BY date""", conn, parse_dates=['date'])
    vix = pd.read_sql("SELECT date, close as vix FROM price_history_vix WHERE date >= '2003-01-01' ORDER BY date",
                       conn, parse_dates=['date'])

spy = spy.sort_values('date').reset_index(drop=True)
c = spy['close']

# MA200 and derivatives
spy['ma200'] = c.rolling(200).mean()
spy['ma50'] = c.rolling(50).mean()
spy['ma20'] = c.rolling(20).mean()

# Distance to MA200 (%)
spy['dist_ma200'] = (c - spy['ma200']) / spy['ma200'] * 100

# MA200 slope (is it rising or falling?) - 20-day change in MA200
spy['ma200_slope'] = (spy['ma200'] - spy['ma200'].shift(20)) / spy['ma200'].shift(20) * 100

# Speed: how fast is price approaching/leaving MA200
spy['dist_ma200_change'] = spy['dist_ma200'] - spy['dist_ma200'].shift(5)

# Recent cross: days since last MA200 cross
spy['above_ma200'] = c > spy['ma200']
spy['cross'] = spy['above_ma200'] != spy['above_ma200'].shift(1)

# SPY returns
spy['ret_1w'] = c / c.shift(5) - 1
spy['ret_4w'] = c / c.shift(20) - 1
spy['ret_12w'] = c / c.shift(60) - 1

# Merge VIX
spy = spy.merge(vix, on='date', how='left')
spy['vix'] = spy['vix'].ffill()

spy = spy.dropna(subset=['ma200', 'ma200_slope']).reset_index(drop=True)

print(f"  SPY: {len(spy)} dias ({spy['date'].min().date()} a {spy['date'].max().date()})")

# ============================================================
# Define market regimes
# ============================================================
print("\n[2/5] Definiendo regimenes de mercado...")

# Regime based on dist_ma200 + slope
def classify_regime(row):
    dist = row['dist_ma200']
    slope = row['ma200_slope']

    if dist > 10 and slope > 0:
        return 'BULL_FUERTE'      # Far above rising MA200
    elif dist > 0 and slope > 0:
        return 'BULL_SANO'        # Above rising MA200
    elif dist > 0 and slope <= 0:
        return 'BULL_DEBIL'       # Above but MA200 falling (danger)
    elif dist > -5 and dist <= 0 and slope > 0:
        return 'CORRECCION'       # Just below rising MA200 (pullback)
    elif dist > -5 and dist <= 0 and slope <= 0:
        return 'TRANSICION'       # Just below falling MA200
    elif dist > -10 and dist <= -5:
        return 'BEAR_MODERADO'    # Significantly below MA200
    else:
        return 'BEAR_FUERTE'      # Far below MA200 (crisis)

spy['regime'] = spy.apply(classify_regime, axis=1)

# Count days per regime
regime_counts = spy['regime'].value_counts()
print(f"\n  Distribucion de regimenes (dias):")
total_days = len(spy)
regime_order = ['BULL_FUERTE', 'BULL_SANO', 'BULL_DEBIL', 'CORRECCION', 'TRANSICION', 'BEAR_MODERADO', 'BEAR_FUERTE']
for r in regime_order:
    cnt = regime_counts.get(r, 0)
    pct = cnt / total_days * 100
    bar = '#' * int(pct)
    print(f"    {r:>15s}: {cnt:>5d} dias ({pct:>5.1f}%) {bar}")

# ============================================================
# Also analyze by simple dist_ma200 buckets
# ============================================================
print(f"\n  Distribucion por distancia a MA200:")
dist_buckets = [
    (-100, -20, 'SPY <-20% MA200 (crash)'),
    (-20, -10, 'SPY -20%/-10% MA200'),
    (-10, -5,  'SPY -10%/-5% MA200'),
    (-5, 0,    'SPY -5%/0% MA200'),
    (0, 3,     'SPY 0%/+3% MA200'),
    (3, 7,     'SPY +3%/+7% MA200'),
    (7, 12,    'SPY +7%/+12% MA200'),
    (12, 20,   'SPY +12%/+20% MA200'),
    (20, 100,  'SPY >+20% MA200'),
]

for lo, hi, label in dist_buckets:
    mask = (spy['dist_ma200'] >= lo) & (spy['dist_ma200'] < hi)
    cnt = mask.sum()
    pct = cnt / total_days * 100
    bar = '#' * int(pct)
    print(f"    {label:>28s}: {cnt:>5d} dias ({pct:>5.1f}%) {bar}")

# Current state
latest = spy.iloc[-1]
print(f"\n  ESTADO ACTUAL (ultimo dato {latest['date'].date()}):")
print(f"    SPY: {latest['close']:.2f}")
print(f"    MA200: {latest['ma200']:.2f}")
print(f"    Distancia MA200: {latest['dist_ma200']:+.1f}%")
print(f"    Pendiente MA200 (20d): {latest['ma200_slope']:+.2f}%")
print(f"    VIX: {latest['vix']:.1f}")
print(f"    Regimen: {latest['regime']}")

# ============================================================
# Load stock data and calculate weekly returns by regime
# ============================================================
print(f"\n[3/5] Calculando rendimiento semanal por regimen...")

with open('data/sp500_constituents.json') as f:
    current_members = json.load(f)
with open('data/sp500_historical_changes.json') as f:
    all_changes = json.load(f)

all_changes.sort(key=lambda x: x.get('date', ''), reverse=True)
current_set = {d['symbol'] for d in current_members}
all_sp500_symbols = set(current_set)
for c_change in all_changes:
    if c_change.get('date', '') >= '2004-01-01':
        if c_change.get('removedTicker'):
            all_sp500_symbols.add(c_change['removedTicker'])
        if c_change.get('symbol'):
            all_sp500_symbols.add(c_change['symbol'])

def get_sp500_at_date(target_date):
    members = set(current_set)
    target_str = str(target_date)
    for ch in all_changes:
        if ch.get('date', '') > target_str:
            if ch.get('symbol') and ch['symbol'] in members:
                members.discard(ch['symbol'])
            if ch.get('removedTicker'):
                members.add(ch['removedTicker'])
    return members

sp500_cache = {}
def get_sp500_cached(sig_date):
    mk = str(sig_date)[:7]
    if mk not in sp500_cache:
        sp500_cache[mk] = get_sp500_at_date(sig_date)
    return sp500_cache[mk]

with engine.connect() as conn:
    df = pd.read_sql("""
        SELECT symbol, date, open, close FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2004-01-01'
        ORDER BY symbol, date
    """, conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

def calc_ind(group):
    g = group.sort_values('date').copy()
    g['ret_1w'] = g['close'] / g['close'].shift(5) - 1
    g['ma200'] = g['close'].rolling(200).mean()
    g['dist_ma200'] = (g['close'] - g['ma200']) / g['ma200'] * 100
    delta = g['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g['rsi'] = 100 - (100 / (1 + rs))
    return g

df = df.groupby('symbol', group_keys=False).apply(calc_ind)
df_indexed = df.set_index(['symbol', 'date']).sort_index()

# Build weekly pairs
df['weekday'] = df['date'].dt.weekday
fridays = np.sort(df[df['weekday'] == 4]['date'].unique())
mondays = np.sort(df[df['weekday'] == 0]['date'].unique())

spy_indexed = spy.set_index('date')

weeks = []
for fri in fridays:
    next_mons = mondays[mondays > fri]
    if len(next_mons) < 2:
        continue
    fri_ts = pd.Timestamp(fri)
    if fri_ts not in spy_indexed.index:
        continue
    spy_row = spy_indexed.loc[fri_ts]
    if pd.isna(spy_row.get('regime')):
        continue
    weeks.append({
        'signal_date': fri_ts,
        'entry_date': pd.Timestamp(next_mons[0]),
        'exit_date': pd.Timestamp(next_mons[1]),
        'regime': spy_row['regime'],
        'dist_ma200': spy_row['dist_ma200'],
        'ma200_slope': spy_row['ma200_slope'],
        'vix': spy_row['vix'] if pd.notna(spy_row.get('vix')) else None,
        'spy_ret_4w': spy_row['ret_4w'] if pd.notna(spy_row.get('ret_4w')) else 0,
    })

weeks_df = pd.DataFrame(weeks)
weeks_df = weeks_df[weeks_df['signal_date'] >= '2005-01-01'].reset_index(drop=True)

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

print(f"  Semanas: {len(weeks_df)}")

# Run strategies for each week
N = 10
strategies = {
    'MR_1w': {'rank': 'ret_1w'},       # Long: lowest 1w ret (oversold), Short: highest 1w ret
    'MR_RSI': {'rank': 'rsi'},          # Long: lowest RSI, Short: highest RSI
}

week_results = []

for _, wrow in weeks_df.iterrows():
    sig = wrow['signal_date']
    entry = wrow['entry_date']
    exit_d = wrow['exit_date']

    if sig not in friday_data or entry not in monday_data or exit_d not in monday_data:
        continue

    fri = friday_data[sig]
    entry_snap = monday_data[entry]
    exit_snap = monday_data[exit_d]

    sp500 = get_sp500_cached(sig)
    eligible = [s for s in fri.index if s in sp500]
    if len(eligible) < 100:
        continue

    fri_e = fri.loc[eligible]

    for strat_name, strat_cfg in strategies.items():
        rank_col = strat_cfg['rank']
        ranked = fri_e[rank_col].dropna().sort_values(ascending=True)
        if len(ranked) < N * 2:
            continue

        long_syms = ranked.head(N).index.tolist()     # Most oversold
        short_syms = ranked.tail(N).index.tolist()     # Most overbought

        long_rets, short_rets = [], []
        for sym in long_syms:
            if sym in entry_snap.index and sym in exit_snap.index:
                ep = entry_snap.loc[sym, 'open']
                xp = exit_snap.loc[sym, 'open']
                if pd.notna(ep) and pd.notna(xp) and ep > 0:
                    long_rets.append((xp - ep) / ep)
        for sym in short_syms:
            if sym in entry_snap.index and sym in exit_snap.index:
                ep = entry_snap.loc[sym, 'open']
                xp = exit_snap.loc[sym, 'open']
                if pd.notna(ep) and pd.notna(xp) and ep > 0:
                    short_rets.append((ep - xp) / ep)

        if len(long_rets) >= 5 and len(short_rets) >= 5:
            week_results.append({
                'date': sig,
                'strategy': strat_name,
                'regime': wrow['regime'],
                'dist_ma200': wrow['dist_ma200'],
                'ma200_slope': wrow['ma200_slope'],
                'vix': wrow['vix'],
                'spy_ret_4w': wrow['spy_ret_4w'],
                'long_ret': np.mean(long_rets),
                'short_ret': np.mean(short_rets),
                'ls_ret': (np.mean(long_rets) + np.mean(short_rets)) / 2,
            })

rdf = pd.DataFrame(week_results)
print(f"  Resultados semanales: {len(rdf)}")

# ============================================================
# Results by regime
# ============================================================
print(f"\n[4/5] Rendimiento por regimen de mercado\n")

def print_regime_table(data, ret_col, title):
    print(f"\n{'='*140}")
    print(f"  {title}")
    print(f"{'='*140}")
    print(f"\n  {'Regimen':>15s} | {'Sem':>5s} | {'%Tiempo':>7s} | {'AvgRet%':>8s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'AvgWin%':>7s} | {'AvgLoss%':>8s} | {'PF':>5s} | {'Dist MA200':>10s} | {'Slope':>6s}")
    print(f"  {'-'*15} | {'-'*5} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*7} | {'-'*8} | {'-'*5} | {'-'*10} | {'-'*6}")

    for regime in regime_order:
        subset = data[data['regime'] == regime][ret_col].dropna()
        if len(subset) < 10:
            continue
        rets = subset.values
        cum = np.cumprod(1 + rets)
        years = len(rets) / 52
        cagr = (cum[-1] ** (1/years) - 1) * 100 if years > 0.5 else np.nan
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
        wr = np.mean(rets > 0) * 100
        avg_w = np.mean(rets[rets > 0]) * 100 if (rets > 0).any() else 0
        avg_l = np.mean(rets[rets < 0]) * 100 if (rets < 0).any() else 0
        pf = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0
        pct = len(subset) / len(data) * 100

        avg_dist = data[data['regime'] == regime]['dist_ma200'].mean()
        avg_slope = data[data['regime'] == regime]['ma200_slope'].mean()

        cagr_str = f"{cagr:>+6.1f}%" if not np.isnan(cagr) else "   N/A "
        marker = " ***" if sharpe > 0.5 else (" **" if sharpe > 0.2 else (" *" if sharpe > 0 else ""))
        print(f"  {regime:>15s} | {len(subset):>5d} | {pct:>6.1f}% | {np.mean(rets)*100:>+7.3f}% | {cagr_str} | {sharpe:>6.2f} | {wr:>4.0f}% | {avg_w:>+6.2f}% | {avg_l:>+7.2f}% | {pf:>5.2f} | {avg_dist:>+9.1f}% | {avg_slope:>+5.2f}%{marker}")

# MR_1w strategy
mr1w = rdf[rdf['strategy'] == 'MR_1w']

print_regime_table(mr1w, 'long_ret', 'LONG MR_1w (comprar 10 mas oversold) POR REGIMEN')
print_regime_table(mr1w, 'short_ret', 'SHORT MR_1w (shortear 10 mas overbought) POR REGIMEN')
print_regime_table(mr1w, 'ls_ret', 'LONG/SHORT MR_1w COMBINADO POR REGIMEN')

# ============================================================
# By distance to MA200 buckets (more granular)
# ============================================================
print(f"\n\n[5/5] Rendimiento por distancia a MA200 (granular)\n")

def print_dist_table(data, ret_col, title):
    print(f"\n{'='*140}")
    print(f"  {title}")
    print(f"{'='*140}")
    print(f"\n  {'Dist MA200':>28s} | {'Sem':>5s} | {'%Tiempo':>7s} | {'AvgRet%':>8s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'PF':>5s} | {'AvgVIX':>6s}")
    print(f"  {'-'*28} | {'-'*5} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*5} | {'-'*6}")

    for lo, hi, label in dist_buckets:
        mask = (data['dist_ma200'] >= lo) & (data['dist_ma200'] < hi)
        subset = data[mask][ret_col].dropna()
        if len(subset) < 10:
            continue
        rets = subset.values
        cum = np.cumprod(1 + rets)
        years = len(rets) / 52
        cagr = (cum[-1] ** (1/years) - 1) * 100 if years > 0.5 else np.nan
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
        wr = np.mean(rets > 0) * 100
        pf = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0
        pct = len(subset) / len(data) * 100
        avg_vix = data[mask]['vix'].mean()

        cagr_str = f"{cagr:>+6.1f}%" if not np.isnan(cagr) else "   N/A "
        marker = " ***" if sharpe > 0.5 else (" **" if sharpe > 0.2 else (" *" if sharpe > 0 else ""))
        print(f"  {label:>28s} | {len(subset):>5d} | {pct:>6.1f}% | {np.mean(rets)*100:>+7.3f}% | {cagr_str} | {sharpe:>6.2f} | {wr:>4.0f}% | {pf:>5.2f} | {avg_vix:>5.1f}{marker}")

print_dist_table(mr1w, 'long_ret', 'LONG MR_1w POR DISTANCIA A MA200')
print_dist_table(mr1w, 'short_ret', 'SHORT MR_1w POR DISTANCIA A MA200')
print_dist_table(mr1w, 'ls_ret', 'LONG/SHORT MR_1w POR DISTANCIA A MA200')

# ============================================================
# MA200 Slope analysis
# ============================================================
print(f"\n\n{'='*140}")
print(f"  IMPACTO DE LA PENDIENTE DE MA200 (subiendo vs bajando)")
print(f"{'='*140}")

for side, col in [('LONG', 'long_ret'), ('SHORT', 'short_ret'), ('L/S', 'ls_ret')]:
    print(f"\n  {side}:")
    for slope_label, slope_filter in [('MA200 subiendo (slope>0)', mr1w['ma200_slope'] > 0),
                                       ('MA200 bajando (slope<0)', mr1w['ma200_slope'] <= 0)]:
        subset = mr1w[slope_filter][col].dropna()
        if len(subset) < 20:
            continue
        rets = subset.values
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
        wr = np.mean(rets > 0) * 100
        cum = np.cumprod(1 + rets)
        years = len(rets) / 52
        cagr = (cum[-1] ** (1/years) - 1) * 100 if years > 0.5 else 0
        print(f"    {slope_label:>30s}: {len(subset):>5d} sem | CAGR {cagr:>+6.1f}% | Sharpe {sharpe:>6.2f} | WR {wr:>4.0f}%")

# ============================================================
# Combined: dist_ma200 + slope
# ============================================================
print(f"\n\n{'='*140}")
print(f"  MATRIZ: Distancia MA200 x Pendiente MA200 -> LONG Sharpe")
print(f"{'='*140}")

dist_ranges = [(-100, -5, 'Dist<-5%'), (-5, 0, 'Dist -5%/0'), (0, 5, 'Dist 0/+5%'), (5, 10, 'Dist +5%/+10%'), (10, 100, 'Dist>+10%')]
slope_ranges = [(-100, 0, 'Slope<0'), (0, 100, 'Slope>0')]

print(f"\n  {'':>15s}", end='')
for _, _, sl in slope_ranges:
    print(f" | {sl:>15s}", end='')
print()
print(f"  {'-'*15}", end='')
for _ in slope_ranges:
    print(f" | {'-'*15}", end='')
print()

for dlo, dhi, dlabel in dist_ranges:
    print(f"  {dlabel:>15s}", end='')
    for slo, shi, slabel in slope_ranges:
        mask = (mr1w['dist_ma200'] >= dlo) & (mr1w['dist_ma200'] < dhi) & \
               (mr1w['ma200_slope'] >= slo) & (mr1w['ma200_slope'] < shi)
        subset = mr1w[mask]['long_ret'].dropna()
        if len(subset) < 10:
            print(f" | {'N/A':>15s}", end='')
        else:
            rets = subset.values
            sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
            wr = np.mean(rets > 0) * 100
            n = len(subset)
            print(f" | Sh{sharpe:>+5.2f} W{wr:.0f}% n{n}", end='')
    print()

# Same for SHORT
print(f"\n  MATRIZ: Distancia MA200 x Pendiente MA200 -> SHORT Sharpe")
print(f"\n  {'':>15s}", end='')
for _, _, sl in slope_ranges:
    print(f" | {sl:>15s}", end='')
print()
print(f"  {'-'*15}", end='')
for _ in slope_ranges:
    print(f" | {'-'*15}", end='')
print()

for dlo, dhi, dlabel in dist_ranges:
    print(f"  {dlabel:>15s}", end='')
    for slo, shi, slabel in slope_ranges:
        mask = (mr1w['dist_ma200'] >= dlo) & (mr1w['dist_ma200'] < dhi) & \
               (mr1w['ma200_slope'] >= slo) & (mr1w['ma200_slope'] < shi)
        subset = mr1w[mask]['short_ret'].dropna()
        if len(subset) < 10:
            print(f" | {'N/A':>15s}", end='')
        else:
            rets = subset.values
            sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
            wr = np.mean(rets > 0) * 100
            n = len(subset)
            print(f" | Sh{sharpe:>+5.2f} W{wr:.0f}% n{n}", end='')
    print()

# ============================================================
# Summary and recommendation
# ============================================================
print(f"\n\n{'='*140}")
print(f"  RESUMEN: DEFINICION DE REGIMENES PARA SISTEMA ADAPTATIVO")
print(f"{'='*140}")
print(f"""
  Basado en MA200 (distancia + pendiente):

  REGIMEN          | DIST MA200  | SLOPE MA200 | ACCION          | LONG  | SHORT
  -----------------+-------------+-------------+-----------------+-------+-------
  BULL FUERTE      | > +10%      | Subiendo    | Long agresivo   | SI    | NO
  BULL SANO        | 0% a +10%   | Subiendo    | Long normal     | SI    | NO
  BULL DEBIL       | > 0%        | Bajando     | Long cauteloso  | SI    | NO
  CORRECCION       | -5% a 0%    | Subiendo    | Long oportunista| SI    | VIX?
  TRANSICION       | -5% a 0%    | Bajando     | Reducir longs   | MIN   | VIX?
  BEAR MODERADO    | -10% a -5%  | Bajando     | L/S o cash      | MIN   | SI
  BEAR FUERTE      | < -10%      | Bajando     | Short agresivo  | SI(MR)| SI

  ESTADO ACTUAL ({latest['date'].date()}):
    Distancia MA200: {latest['dist_ma200']:+.1f}%
    Pendiente MA200: {latest['ma200_slope']:+.2f}%
    Regimen: {latest['regime']}
""")
