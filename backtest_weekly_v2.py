"""
BACKTEST v2: Weekly Long/Short - S&P 500
==========================================
Strategies: Momentum + Mean Reversion + Multi-Factor
Signal: Friday close
Entry: Monday open OR Monday close
Exit: Next Monday (same time)
Holding: Exactly 1 week
Universe: S&P 500 (historical constituents)
Period: From 2005 (extendable)

Technical indicators: RSI, Bollinger Bands, distance from MA20/MA50/MA200
Quality indicators: EPS beat, EPS surprise, EPS YoY growth
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# ============================================================
# STEP 1: Build historical S&P 500 membership
# ============================================================
print("=" * 130)
print("  BACKTEST v2: Weekly Long/Short - S&P 500")
print("  Momentum + Mean Reversion + Multi-Factor")
print("=" * 130)

print("\n[1/6] Reconstruyendo composicion historica S&P 500...")

with open('data/sp500_constituents.json') as f:
    current_members = json.load(f)
with open('data/sp500_historical_changes.json') as f:
    all_changes = json.load(f)

# Sort changes by date descending (most recent first)
all_changes.sort(key=lambda x: x.get('date', ''), reverse=True)

# Build membership at any date by working backwards from current
# current_set at today -> remove recent additions, add recent removals -> past set
current_set = {d['symbol'] for d in current_members}

# Get all unique symbols that were ever in SP500 during our period
all_sp500_symbols = set(current_set)
for c in all_changes:
    if c.get('date', '') >= '2004-01-01':
        if c.get('removedTicker'):
            all_sp500_symbols.add(c['removedTicker'])
        if c.get('symbol'):
            all_sp500_symbols.add(c['symbol'])

print(f"  Miembros actuales: {len(current_set)}")
print(f"  Total simbolos historicos SP500 (2004+): {len(all_sp500_symbols)}")

# Build a function to get SP500 members at a given date
def get_sp500_at_date(target_date):
    """Reconstruct SP500 membership at a given date"""
    members = set(current_set)
    target_str = str(target_date)
    for c in all_changes:
        change_date = c.get('date', '')
        if change_date > target_str:
            # This change happened AFTER target date, so reverse it
            added_sym = c.get('symbol')
            removed_sym = c.get('removedTicker')
            if added_sym and added_sym in members:
                members.discard(added_sym)
            if removed_sym:
                members.add(removed_sym)
    return members

# Test
sp500_2020 = get_sp500_at_date('2020-01-01')
sp500_2010 = get_sp500_at_date('2010-01-01')
print(f"  SP500 al 2010-01-01: {len(sp500_2010)} miembros")
print(f"  SP500 al 2020-01-01: {len(sp500_2020)} miembros")

# ============================================================
# STEP 2: Load price data for all historical SP500 stocks
# ============================================================
print("\n[2/6] Cargando datos de precio...")

symbols_list = list(all_sp500_symbols)

with engine.connect() as conn:
    df = pd.read_sql("""
        SELECT symbol, date, open, high, low, close, volume
        FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s)
        AND date >= '2004-01-01'
        ORDER BY symbol, date
    """, conn, params={"syms": symbols_list}, parse_dates=['date'])

print(f"  Registros: {len(df):,}")
print(f"  Simbolos con datos: {df['symbol'].nunique()}")
print(f"  Rango: {df['date'].min().date()} a {df['date'].max().date()}")

# ============================================================
# STEP 3: Calculate all technical indicators (per stock)
# ============================================================
print("\n[3/6] Calculando indicadores tecnicos por accion...")

# Group by symbol and calculate indicators
def calc_indicators(group):
    g = group.sort_values('date').copy()
    c = g['close']

    # Moving averages
    g['ma20'] = c.rolling(20).mean()
    g['ma50'] = c.rolling(50).mean()
    g['ma200'] = c.rolling(200).mean()

    # Distance from MAs (%)
    g['dist_ma20'] = (c - g['ma20']) / g['ma20'] * 100
    g['dist_ma50'] = (c - g['ma50']) / g['ma50'] * 100
    g['dist_ma200'] = (c - g['ma200']) / g['ma200'] * 100

    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    g['bb_mid'] = g['ma20']
    bb_std = c.rolling(20).std()
    g['bb_upper'] = g['bb_mid'] + 2 * bb_std
    g['bb_lower'] = g['bb_mid'] - 2 * bb_std
    g['bb_pct'] = (c - g['bb_lower']) / (g['bb_upper'] - g['bb_lower'])  # 0=lower, 1=upper

    # Momentum returns (lookback)
    g['ret_1w'] = c / c.shift(5) - 1
    g['ret_4w'] = c / c.shift(20) - 1
    g['ret_12w'] = c / c.shift(60) - 1
    g['ret_26w'] = c / c.shift(130) - 1
    g['ret_52w'] = c / c.shift(252) - 1

    # Volatility (20d)
    g['vol_20d'] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Average volume 20d
    g['avg_vol_20d'] = g['volume'].rolling(20).mean()

    return g

print("  Calculando (esto tarda ~1 min)...")
df = df.groupby('symbol', group_keys=False).apply(calc_indicators)
print(f"  Indicadores calculados para {df['symbol'].nunique()} simbolos")

# ============================================================
# STEP 4: Load EPS data
# ============================================================
print("\n[4/6] Cargando datos EPS...")

with engine.connect() as conn:
    eps_df = pd.read_sql("""
        SELECT symbol, date, eps_actual, eps_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s)
        AND date >= '2003-01-01'
        ORDER BY symbol, date
    """, conn, params={"syms": symbols_list}, parse_dates=['date'])

# Calculate EPS metrics per earning report
eps_df['eps_beat'] = eps_df['eps_actual'] > eps_df['eps_estimated']
eps_df['eps_surprise'] = np.where(
    eps_df['eps_estimated'].abs() > 0.01,
    (eps_df['eps_actual'] - eps_df['eps_estimated']) / eps_df['eps_estimated'].abs() * 100,
    np.nan
)

# EPS YoY: compare to same quarter prior year (shift 4 quarters)
eps_df = eps_df.sort_values(['symbol', 'date'])
eps_df['eps_actual_4q_ago'] = eps_df.groupby('symbol')['eps_actual'].shift(4)
eps_df['eps_yoy'] = np.where(
    eps_df['eps_actual_4q_ago'].abs() > 0.01,
    (eps_df['eps_actual'] - eps_df['eps_actual_4q_ago']) / eps_df['eps_actual_4q_ago'].abs() * 100,
    np.nan
)

print(f"  Registros EPS: {len(eps_df):,}")
print(f"  Simbolos con EPS: {eps_df['symbol'].nunique()}")

# For each stock, get the LATEST EPS data available BEFORE a given date
# We'll merge this at the weekly level later

# ============================================================
# STEP 5: Build weekly trading structure and run backtest
# ============================================================
print("\n[5/6] Construyendo semanas de trading y ejecutando backtest...")

# Identify all Fridays and following Mondays
df['weekday'] = df['date'].dt.weekday

# Get unique trading dates
trading_dates = df[df['symbol'] == 'AAPL']['date'].sort_values().values

# Find Friday-Monday pairs
fridays = df[df['weekday'] == 4]['date'].unique()
mondays = df[df['weekday'] == 0]['date'].unique()
fridays = np.sort(fridays)
mondays = np.sort(mondays)

# Build signal weeks: signal on Friday, enter next Monday, exit Monday after
weeks = []
for fri in fridays:
    # Next Monday after this Friday
    next_mondays = mondays[mondays > fri]
    if len(next_mondays) < 2:
        continue
    entry_mon = next_mondays[0]
    exit_mon = next_mondays[1]
    weeks.append({
        'signal_date': pd.Timestamp(fri),
        'entry_date': pd.Timestamp(entry_mon),
        'exit_date': pd.Timestamp(exit_mon),
    })

weeks_df = pd.DataFrame(weeks)
# Filter to backtest period
weeks_df = weeks_df[weeks_df['signal_date'] >= '2005-01-01'].reset_index(drop=True)
print(f"  Total semanas de trading: {len(weeks_df)}")
print(f"  Primera: signal {weeks_df.iloc[0]['signal_date'].date()}, entry {weeks_df.iloc[0]['entry_date'].date()}, exit {weeks_df.iloc[0]['exit_date'].date()}")
print(f"  Ultima: signal {weeks_df.iloc[-1]['signal_date'].date()}, entry {weeks_df.iloc[-1]['entry_date'].date()}, exit {weeks_df.iloc[-1]['exit_date'].date()}")

# Pre-index price data for fast lookup
df_indexed = df.set_index(['symbol', 'date']).sort_index()

# Function to get latest EPS data for a symbol before a date
eps_indexed = eps_df.set_index(['symbol', 'date']).sort_index()

def get_eps_for_stock(sym, before_date):
    """Get most recent EPS data for a stock before a given date"""
    try:
        sym_eps = eps_indexed.loc[sym]
        mask = sym_eps.index <= before_date
        if mask.any():
            latest = sym_eps[mask].iloc[-1]
            return {
                'eps_beat': latest['eps_beat'],
                'eps_surprise': latest['eps_surprise'],
                'eps_yoy': latest['eps_yoy'],
            }
    except KeyError:
        pass
    return {'eps_beat': None, 'eps_surprise': None, 'eps_yoy': None}

# ============================================================
# Define strategies
# ============================================================

N_LONG = 10
N_SHORT = 10
MIN_ELIGIBLE = 100

strategies = {
    # MOMENTUM strategies
    'MOM_4w':   {'type': 'momentum', 'rank_col': 'ret_4w',  'filters': None},
    'MOM_12w':  {'type': 'momentum', 'rank_col': 'ret_12w', 'filters': None},
    'MOM_26w':  {'type': 'momentum', 'rank_col': 'ret_26w', 'filters': None},
    'MOM_52w':  {'type': 'momentum', 'rank_col': 'ret_52w', 'filters': None},

    # MOMENTUM + MA200 filter
    'MOM_12w+MA200': {'type': 'momentum_filtered', 'rank_col': 'ret_12w',
                      'long_filter': 'dist_ma200 > 0', 'short_filter': 'dist_ma200 < 0'},
    'MOM_26w+MA200': {'type': 'momentum_filtered', 'rank_col': 'ret_26w',
                      'long_filter': 'dist_ma200 > 0', 'short_filter': 'dist_ma200 < 0'},

    # MEAN REVERSION strategies
    'MR_RSI':      {'type': 'mean_reversion', 'rank_col': 'rsi', 'filters': None},
    'MR_BB':       {'type': 'mean_reversion', 'rank_col': 'bb_pct', 'filters': None},
    'MR_dist20':   {'type': 'mean_reversion', 'rank_col': 'dist_ma20', 'filters': None},
    'MR_1w':       {'type': 'mean_reversion', 'rank_col': 'ret_1w', 'filters': None},
    'MR_4w':       {'type': 'mean_reversion', 'rank_col': 'ret_4w', 'filters': None},

    # MEAN REVERSION + MA200 (only revert stocks in long-term trend)
    'MR_RSI+MA200': {'type': 'mr_filtered', 'rank_col': 'rsi',
                     'long_filter': 'dist_ma200 > 0', 'short_filter': 'dist_ma200 < 0'},
    'MR_1w+MA200':  {'type': 'mr_filtered', 'rank_col': 'ret_1w',
                     'long_filter': 'dist_ma200 > 0', 'short_filter': 'dist_ma200 < 0'},

    # MULTI-FACTOR (composite score)
    'MULTI_MomQual':  {'type': 'multi_factor', 'factors': ['ret_12w', 'dist_ma200', 'eps_score']},
    'MULTI_MR_Qual':  {'type': 'multi_factor_mr', 'factors': ['rsi', 'dist_ma20', 'dist_ma200', 'eps_score']},
}

# ============================================================
# Run all strategies
# ============================================================
print(f"\n  Ejecutando {len(strategies)} estrategias...")

# Pre-build Friday snapshots for speed
print("  Construyendo snapshots de viernes...")
friday_data = {}
for _, row in weeks_df.iterrows():
    sig_date = row['signal_date']
    snap = df_indexed.xs(sig_date, level='date', drop_level=True) if sig_date in df_indexed.index.get_level_values('date') else pd.DataFrame()
    if not snap.empty:
        friday_data[sig_date] = snap

# Pre-build Monday data
monday_data = {}
for _, row in weeks_df.iterrows():
    for d in [row['entry_date'], row['exit_date']]:
        if d not in monday_data:
            snap = df_indexed.xs(d, level='date', drop_level=True) if d in df_indexed.index.get_level_values('date') else pd.DataFrame()
            if not snap.empty:
                monday_data[d] = snap

print(f"  Snapshots: {len(friday_data)} viernes, {len(monday_data)} lunes")

# Pre-cache SP500 membership per signal date (cache by month to save time)
sp500_cache = {}

def get_sp500_cached(sig_date):
    month_key = str(sig_date)[:7]
    if month_key not in sp500_cache:
        sp500_cache[month_key] = get_sp500_at_date(sig_date)
    return sp500_cache[month_key]

# Pre-cache EPS data: for each symbol, get chronological list of EPS events
eps_by_symbol = {}
for sym, grp in eps_df.groupby('symbol'):
    eps_by_symbol[sym] = grp.sort_values('date')[['date', 'eps_beat', 'eps_surprise', 'eps_yoy']].values

def get_eps_fast(sym, before_date):
    if sym not in eps_by_symbol:
        return None, None, None
    data = eps_by_symbol[sym]
    mask = data[:, 0] <= before_date
    if not mask.any():
        return None, None, None
    last = data[mask][-1]
    return last[1], last[2], last[3]

results = {}

for strat_name, strat_config in strategies.items():
    weekly_rets_ls = []
    weekly_rets_long = []
    weekly_rets_short = []
    trade_weeks_list = []
    picks_long = []
    picks_short = []

    for entry_mode in ['mon_open', 'mon_close']:
        key = f"{strat_name}__{entry_mode}"
        w_ls, w_l, w_s = [], [], []
        tw = []

        for idx, wrow in weeks_df.iterrows():
            sig_date = wrow['signal_date']
            entry_date = wrow['entry_date']
            exit_date = wrow['exit_date']

            if sig_date not in friday_data:
                continue
            if entry_date not in monday_data or exit_date not in monday_data:
                continue

            fri_snap = friday_data[sig_date]
            entry_snap = monday_data[entry_date]
            exit_snap = monday_data[exit_date]

            # Get SP500 members for this date
            sp500_members = get_sp500_cached(sig_date)

            # Filter to SP500 members with data
            eligible_syms = [s for s in fri_snap.index if s in sp500_members]
            if len(eligible_syms) < MIN_ELIGIBLE:
                continue

            fri_eligible = fri_snap.loc[eligible_syms].copy()

            # Add EPS data
            if strat_config['type'] in ('multi_factor', 'multi_factor_mr') or \
               'eps' in str(strat_config.get('long_filter', '')) or \
               'eps' in str(strat_config.get('short_filter', '')):
                eps_data = {}
                for sym in eligible_syms:
                    beat, surp, yoy = get_eps_fast(sym, sig_date)
                    eps_score = 0
                    if beat == True: eps_score += 1
                    if surp is not None and surp > 5: eps_score += 1
                    if yoy is not None and yoy > 0: eps_score += 1
                    eps_data[sym] = eps_score
                fri_eligible['eps_score'] = pd.Series(eps_data)

            # Apply strategy logic
            stype = strat_config['type']
            rank_col = strat_config.get('rank_col')

            if stype == 'momentum':
                ranked = fri_eligible[rank_col].dropna().sort_values(ascending=False)
                long_syms = ranked.head(N_LONG).index.tolist()
                short_syms = ranked.tail(N_SHORT).index.tolist()

            elif stype == 'momentum_filtered':
                # Long: filter first, then rank by momentum
                long_pool = fri_eligible.query(strat_config['long_filter'])[rank_col].dropna().sort_values(ascending=False)
                short_pool = fri_eligible.query(strat_config['short_filter'])[rank_col].dropna().sort_values(ascending=True)
                if len(long_pool) < N_LONG or len(short_pool) < N_SHORT:
                    continue
                long_syms = long_pool.head(N_LONG).index.tolist()
                short_syms = short_pool.head(N_SHORT).index.tolist()

            elif stype == 'mean_reversion':
                # Long: buy the MOST oversold (lowest RSI, lowest BB%, most negative returns)
                ranked = fri_eligible[rank_col].dropna().sort_values(ascending=True)
                long_syms = ranked.head(N_LONG).index.tolist()
                short_syms = ranked.tail(N_SHORT).index.tolist()

            elif stype == 'mr_filtered':
                # Mean reversion but only for stocks in the right long-term trend
                long_pool = fri_eligible.query(strat_config['long_filter'])[rank_col].dropna().sort_values(ascending=True)
                short_pool = fri_eligible.query(strat_config['short_filter'])[rank_col].dropna().sort_values(ascending=False)
                if len(long_pool) < N_LONG or len(short_pool) < N_SHORT:
                    continue
                long_syms = long_pool.head(N_LONG).index.tolist()
                short_syms = short_pool.head(N_SHORT).index.tolist()

            elif stype == 'multi_factor':
                # Composite score: normalize and combine factors
                scores = pd.DataFrame(index=fri_eligible.index)
                for factor in strat_config['factors']:
                    col = fri_eligible[factor].dropna()
                    # Percentile rank (0-1)
                    scores[factor] = col.rank(pct=True)
                scores['composite'] = scores.mean(axis=1)
                scores = scores['composite'].dropna().sort_values(ascending=False)
                long_syms = scores.head(N_LONG).index.tolist()
                short_syms = scores.tail(N_SHORT).index.tolist()

            elif stype == 'multi_factor_mr':
                # Mean reversion multi-factor: LOW RSI + LOW dist_ma20 + HIGH dist_ma200 + HIGH eps_score
                scores = pd.DataFrame(index=fri_eligible.index)
                for factor in strat_config['factors']:
                    col = fri_eligible[factor].dropna()
                    if factor in ('rsi', 'dist_ma20'):
                        # Inverse rank for mean reversion (lower = better for longs)
                        scores[factor] = 1 - col.rank(pct=True)
                    else:
                        scores[factor] = col.rank(pct=True)
                scores['composite'] = scores.mean(axis=1)
                scores = scores['composite'].dropna().sort_values(ascending=False)
                long_syms = scores.head(N_LONG).index.tolist()
                short_syms = scores.tail(N_SHORT).index.tolist()

            else:
                continue

            # Calculate returns
            long_ret_list = []
            short_ret_list = []

            entry_col = 'open' if entry_mode == 'mon_open' else 'close'

            for sym in long_syms:
                if sym in entry_snap.index and sym in exit_snap.index:
                    ep = entry_snap.loc[sym, entry_col]
                    xp = exit_snap.loc[sym, entry_col]  # Exit at same type (open or close)
                    if pd.notna(ep) and pd.notna(xp) and ep > 0:
                        long_ret_list.append((xp - ep) / ep)

            for sym in short_syms:
                if sym in entry_snap.index and sym in exit_snap.index:
                    ep = entry_snap.loc[sym, entry_col]
                    xp = exit_snap.loc[sym, entry_col]
                    if pd.notna(ep) and pd.notna(xp) and ep > 0:
                        short_ret_list.append((ep - xp) / ep)

            if len(long_ret_list) >= 5 and len(short_ret_list) >= 5:
                avg_l = np.mean(long_ret_list)
                avg_s = np.mean(short_ret_list)
                avg_ls = (avg_l + avg_s) / 2

                w_ls.append(avg_ls)
                w_l.append(avg_l)
                w_s.append(avg_s)
                tw.append(sig_date)

        if not w_ls:
            continue

        rets = np.array(w_ls)
        lr = np.array(w_l)
        sr = np.array(w_s)

        cum = np.cumprod(1 + rets)
        cum_l = np.cumprod(1 + lr)
        cum_s = np.cumprod(1 + sr)
        years = len(rets) / 52

        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min() * 100

        peak_l = np.maximum.accumulate(cum_l)
        max_dd_l = ((cum_l - peak_l) / peak_l).min() * 100

        results[key] = {
            'strategy': strat_name,
            'entry': entry_mode,
            'weeks': len(rets),
            'years': years,
            'total_ret': (cum[-1] - 1) * 100,
            'cagr': (cum[-1] ** (1/years) - 1) * 100 if years > 0 else 0,
            'avg_weekly': np.mean(rets) * 100,
            'sharpe': (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0,
            'win_rate': np.mean(rets > 0) * 100,
            'max_dd': max_dd,
            # Long
            'long_total': (cum_l[-1] - 1) * 100,
            'long_cagr': (cum_l[-1] ** (1/years) - 1) * 100 if years > 0 else 0,
            'long_avg': np.mean(lr) * 100,
            'long_sharpe': (np.mean(lr) / np.std(lr)) * np.sqrt(52) if np.std(lr) > 0 else 0,
            'long_win': np.mean(lr > 0) * 100,
            'long_max_dd': max_dd_l,
            # Short
            'short_total': (cum_s[-1] - 1) * 100,
            'short_cagr': (cum_s[-1] ** (1/years) - 1) * 100 if years > 0 else 0,
            'short_avg': np.mean(sr) * 100,
            'short_sharpe': (np.mean(sr) / np.std(sr)) * np.sqrt(52) if np.std(sr) > 0 else 0,
            'short_win': np.mean(sr > 0) * 100,
            # Raw data
            'cum': cum,
            'cum_l': cum_l,
            'cum_s': cum_s,
            'rets': rets,
            'long_rets': lr,
            'short_rets': sr,
            'trade_weeks': tw,
        }

    print(f"  {strat_name} completado")

# ============================================================
# STEP 6: Print Results
# ============================================================
print(f"\n[6/6] Resultados:\n")

# Sort by Sharpe
sorted_results = sorted(results.values(), key=lambda x: x['sharpe'], reverse=True)

# PART A: Long/Short Combined
print("=" * 160)
print("  RANKING LONG/SHORT COMBINADO (ordenado por Sharpe)")
print("=" * 160)
print(f"\n  {'#':>3s} {'Estrategia':>20s} | {'Entry':>10s} | {'Sem':>5s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'MaxDD%':>7s} | {'L_CAGR%':>7s} | {'L_Sharpe':>8s} | {'S_CAGR%':>7s} | {'S_Sharpe':>8s}")
print(f"  {'':>3s} {'-'*20} | {'-'*10} | {'-'*5} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*7} | {'-'*7} | {'-'*8} | {'-'*7} | {'-'*8}")

for i, r in enumerate(sorted_results):
    marker = " ***" if r['sharpe'] > 0.5 else (" **" if r['sharpe'] > 0.3 else (" *" if r['sharpe'] > 0 else ""))
    print(f"  {i+1:>3d} {r['strategy']:>20s} | {r['entry']:>10s} | {r['weeks']:>5d} | {r['cagr']:>+6.1f}% | {r['sharpe']:>6.2f} | {r['win_rate']:>4.0f}% | {r['max_dd']:>6.1f}% | {r['long_cagr']:>+6.1f}% | {r['long_sharpe']:>8.2f} | {r['short_cagr']:>+6.1f}% | {r['short_sharpe']:>8.2f}{marker}")

# PART B: Best of each type
print(f"\n\n{'='*130}")
print(f"  MEJOR POR CATEGORIA")
print(f"{'='*130}")

categories = {
    'MOMENTUM': [r for r in sorted_results if r['strategy'].startswith('MOM_')],
    'MEAN REVERSION': [r for r in sorted_results if r['strategy'].startswith('MR_')],
    'MULTI-FACTOR': [r for r in sorted_results if r['strategy'].startswith('MULTI_')],
}

for cat_name, cat_results in categories.items():
    if not cat_results:
        continue
    best = cat_results[0]
    print(f"\n  {cat_name}:")
    print(f"    Mejor: {best['strategy']} / {best['entry']}")
    print(f"    L/S: CAGR {best['cagr']:+.1f}%, Sharpe {best['sharpe']:.2f}, WR {best['win_rate']:.0f}%, MaxDD {best['max_dd']:.1f}%")
    print(f"    Long: CAGR {best['long_cagr']:+.1f}%, Sharpe {best['long_sharpe']:.2f}")
    print(f"    Short: CAGR {best['short_cagr']:+.1f}%, Sharpe {best['short_sharpe']:.2f}")

# PART C: Long-Only ranking (since shorts tend to lose)
print(f"\n\n{'='*160}")
print(f"  RANKING LONG-ONLY (ordenado por Sharpe de la pata larga)")
print(f"{'='*160}")

long_sorted = sorted(results.values(), key=lambda x: x['long_sharpe'], reverse=True)

print(f"\n  {'#':>3s} {'Estrategia':>20s} | {'Entry':>10s} | {'L_CAGR%':>7s} | {'L_Sharpe':>8s} | {'L_WR%':>6s} | {'L_MaxDD%':>8s} | {'L_AvgSem%':>9s}")
print(f"  {'':>3s} {'-'*20} | {'-'*10} | {'-'*7} | {'-'*8} | {'-'*6} | {'-'*8} | {'-'*9}")

for i, r in enumerate(long_sorted[:15]):
    print(f"  {i+1:>3d} {r['strategy']:>20s} | {r['entry']:>10s} | {r['long_cagr']:>+6.1f}% | {r['long_sharpe']:>8.2f} | {r['long_win']:>5.0f}% | {r['long_max_dd']:>7.1f}% | {r['long_avg']:>+8.3f}%")

# PART D: Short-Only ranking
print(f"\n\n{'='*160}")
print(f"  RANKING SHORT-ONLY (ordenado por Sharpe de la pata corta)")
print(f"{'='*160}")

short_sorted = sorted(results.values(), key=lambda x: x['short_sharpe'], reverse=True)

print(f"\n  {'#':>3s} {'Estrategia':>20s} | {'Entry':>10s} | {'S_CAGR%':>7s} | {'S_Sharpe':>8s} | {'S_WR%':>6s} | {'S_AvgSem%':>9s}")
print(f"  {'':>3s} {'-'*20} | {'-'*10} | {'-'*7} | {'-'*8} | {'-'*6} | {'-'*9}")

for i, r in enumerate(short_sorted[:15]):
    print(f"  {i+1:>3d} {r['strategy']:>20s} | {r['entry']:>10s} | {r['short_cagr']:>+6.1f}% | {r['short_sharpe']:>8.2f} | {r['short_win']:>5.0f}% | {r['short_avg']:>+8.3f}%")

# PART E: Annual breakdown for best overall
print(f"\n\n{'='*130}")
best_overall = sorted_results[0]
print(f"  DESGLOSE ANUAL: {best_overall['strategy']} / {best_overall['entry']}")
print(f"{'='*130}")

annual = defaultdict(lambda: {'ls': [], 'l': [], 's': []})
for tw, r_ls, r_l, r_s in zip(best_overall['trade_weeks'], best_overall['rets'], best_overall['long_rets'], best_overall['short_rets']):
    yr = tw.year
    annual[yr]['ls'].append(r_ls)
    annual[yr]['l'].append(r_l)
    annual[yr]['s'].append(r_s)

print(f"\n  {'Ano':>6s} | {'Sem':>4s} | {'L/S%':>7s} | {'L/S WR':>6s} | {'Long%':>7s} | {'Short%':>7s}")
print(f"  {'-'*6} | {'-'*4} | {'-'*7} | {'-'*6} | {'-'*7} | {'-'*7}")

for yr in sorted(annual.keys()):
    a = annual[yr]
    ls = np.array(a['ls'])
    lo = np.array(a['l'])
    sh = np.array(a['s'])
    ls_ret = (np.cumprod(1 + ls)[-1] - 1) * 100
    ls_wr = np.mean(ls > 0) * 100
    lo_ret = (np.cumprod(1 + lo)[-1] - 1) * 100
    sh_ret = (np.cumprod(1 + sh)[-1] - 1) * 100
    marker = " ***" if ls_ret > 15 else (" **" if ls_ret > 5 else (" *" if ls_ret > 0 else ""))
    print(f"  {yr:>6d} | {len(ls):>4d} | {ls_ret:>+6.1f}% | {ls_wr:>5.0f}% | {lo_ret:>+6.1f}% | {sh_ret:>+6.1f}%{marker}")

print(f"\n{'='*130}")
print(f"  FIN DEL BACKTEST")
print(f"{'='*130}")
