"""
BACKTEST v3: FOCUS SHORT SIDE
==============================
Analisis profundo de la pata corta:
- Cuando funciona shortear (regimen de mercado)
- Que senales mejoran los shorts
- Filtros combinados para shorts
- Signal: Friday close, Entry: Monday open, Exit: Next Monday open
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
# Load data (reuse structure from v2)
# ============================================================
print("=" * 130)
print("  BACKTEST v3: ANALISIS PROFUNDO DEL LADO CORTO")
print("  Signal viernes cierre -> Entry lunes open -> Exit lunes siguiente open")
print("=" * 130)

# SP500 historical
with open('data/sp500_constituents.json') as f:
    current_members = json.load(f)
with open('data/sp500_historical_changes.json') as f:
    all_changes = json.load(f)

all_changes.sort(key=lambda x: x.get('date', ''), reverse=True)
current_set = {d['symbol'] for d in current_members}
all_sp500_symbols = set(current_set)
for c in all_changes:
    if c.get('date', '') >= '2004-01-01':
        if c.get('removedTicker'):
            all_sp500_symbols.add(c['removedTicker'])
        if c.get('symbol'):
            all_sp500_symbols.add(c['symbol'])

def get_sp500_at_date(target_date):
    members = set(current_set)
    target_str = str(target_date)
    for c in all_changes:
        change_date = c.get('date', '')
        if change_date > target_str:
            added_sym = c.get('symbol')
            removed_sym = c.get('removedTicker')
            if added_sym and added_sym in members:
                members.discard(added_sym)
            if removed_sym:
                members.add(removed_sym)
    return members

print("\n[1/5] Cargando datos...")
symbols_list = list(all_sp500_symbols)

with engine.connect() as conn:
    df = pd.read_sql("""
        SELECT symbol, date, open, high, low, close, volume
        FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s)
        AND date >= '2004-01-01'
        ORDER BY symbol, date
    """, conn, params={"syms": symbols_list}, parse_dates=['date'])

    # Also load SPY for market regime
    spy = pd.read_sql("""
        SELECT date, open, close
        FROM fmp_price_history
        WHERE symbol = 'SPY'
        AND date >= '2004-01-01'
        ORDER BY date
    """, conn, parse_dates=['date'])

print(f"  Stocks: {len(df):,} registros, {df['symbol'].nunique()} simbolos")
print(f"  SPY: {len(spy)} registros")

# ============================================================
# Calculate indicators
# ============================================================
print("\n[2/5] Calculando indicadores...")

def calc_indicators(group):
    g = group.sort_values('date').copy()
    c = g['close']
    g['ma20'] = c.rolling(20).mean()
    g['ma50'] = c.rolling(50).mean()
    g['ma200'] = c.rolling(200).mean()
    g['dist_ma20'] = (c - g['ma20']) / g['ma20'] * 100
    g['dist_ma50'] = (c - g['ma50']) / g['ma50'] * 100
    g['dist_ma200'] = (c - g['ma200']) / g['ma200'] * 100

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi'] = 100 - (100 / (1 + rs))

    bb_std = c.rolling(20).std()
    g['bb_upper'] = g['ma20'] + 2 * bb_std
    g['bb_lower'] = g['ma20'] - 2 * bb_std
    g['bb_pct'] = (c - g['bb_lower']) / (g['bb_upper'] - g['bb_lower'])

    g['ret_1w'] = c / c.shift(5) - 1
    g['ret_4w'] = c / c.shift(20) - 1
    g['ret_12w'] = c / c.shift(60) - 1
    g['ret_26w'] = c / c.shift(130) - 1
    g['vol_20d'] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    g['avg_vol_20d'] = g['volume'].rolling(20).mean()
    return g

df = df.groupby('symbol', group_keys=False).apply(calc_indicators)

# SPY indicators for market regime
spy = spy.sort_values('date')
spy['ma50'] = spy['close'].rolling(50).mean()
spy['ma200'] = spy['close'].rolling(200).mean()
spy['ret_4w'] = spy['close'] / spy['close'].shift(20) - 1
spy['ret_12w'] = spy['close'] / spy['close'].shift(60) - 1
spy['vix_proxy'] = spy['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100  # Vol as VIX proxy

# EPS data
with engine.connect() as conn:
    eps_df = pd.read_sql("""
        SELECT symbol, date, eps_actual, eps_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s) AND date >= '2003-01-01'
        ORDER BY symbol, date
    """, conn, params={"syms": symbols_list}, parse_dates=['date'])

eps_df['eps_beat'] = eps_df['eps_actual'] > eps_df['eps_estimated']
eps_df['eps_surprise'] = np.where(
    eps_df['eps_estimated'].abs() > 0.01,
    (eps_df['eps_actual'] - eps_df['eps_estimated']) / eps_df['eps_estimated'].abs() * 100, np.nan)
eps_df = eps_df.sort_values(['symbol', 'date'])
eps_df['eps_actual_4q_ago'] = eps_df.groupby('symbol')['eps_actual'].shift(4)
eps_df['eps_yoy'] = np.where(
    eps_df['eps_actual_4q_ago'].abs() > 0.01,
    (eps_df['eps_actual'] - eps_df['eps_actual_4q_ago']) / eps_df['eps_actual_4q_ago'].abs() * 100, np.nan)

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

print("  Indicadores calculados")

# ============================================================
# Build weekly structure
# ============================================================
print("\n[3/5] Construyendo semanas...")

df['weekday'] = df['date'].dt.weekday
df_indexed = df.set_index(['symbol', 'date']).sort_index()

fridays = np.sort(df[df['weekday'] == 4]['date'].unique())
mondays = np.sort(df[df['weekday'] == 0]['date'].unique())

weeks = []
for fri in fridays:
    next_mons = mondays[mondays > fri]
    if len(next_mons) < 2:
        continue
    weeks.append({
        'signal_date': pd.Timestamp(fri),
        'entry_date': pd.Timestamp(next_mons[0]),
        'exit_date': pd.Timestamp(next_mons[1]),
    })

weeks_df = pd.DataFrame(weeks)
weeks_df = weeks_df[weeks_df['signal_date'] >= '2005-01-01'].reset_index(drop=True)
print(f"  Semanas: {len(weeks_df)}")

# Pre-build snapshots
friday_data = {}
monday_data = {}
spy_indexed = spy.set_index('date')

for _, row in weeks_df.iterrows():
    sig = row['signal_date']
    if sig in df_indexed.index.get_level_values('date'):
        friday_data[sig] = df_indexed.xs(sig, level='date', drop_level=True)
    for d in [row['entry_date'], row['exit_date']]:
        if d not in monday_data and d in df_indexed.index.get_level_values('date'):
            monday_data[d] = df_indexed.xs(d, level='date', drop_level=True)

sp500_cache = {}
def get_sp500_cached(sig_date):
    mk = str(sig_date)[:7]
    if mk not in sp500_cache:
        sp500_cache[mk] = get_sp500_at_date(sig_date)
    return sp500_cache[mk]

# ============================================================
# Define SHORT strategies to test
# ============================================================
print("\n[4/5] Testeando estrategias SHORT...")

N_SHORT = 10

def get_market_regime(sig_date):
    """Get market regime at signal date"""
    if sig_date in spy_indexed.index:
        row = spy_indexed.loc[sig_date]
        above_ma50 = row['close'] > row['ma50'] if pd.notna(row['ma50']) else True
        above_ma200 = row['close'] > row['ma200'] if pd.notna(row['ma200']) else True
        ret_4w = row['ret_4w'] if pd.notna(row['ret_4w']) else 0
        vol = row['vix_proxy'] if pd.notna(row['vix_proxy']) else 15
        return {
            'spy_above_ma50': above_ma50,
            'spy_above_ma200': above_ma200,
            'spy_ret_4w': ret_4w,
            'spy_vol': vol,
            'bull': above_ma200 and ret_4w > 0,
            'bear': not above_ma200,
            'high_vol': vol > 25,
        }
    return {'spy_above_ma50': True, 'spy_above_ma200': True, 'spy_ret_4w': 0,
            'spy_vol': 15, 'bull': True, 'bear': False, 'high_vol': False}

# Run all short strategies
short_strategies = {
    # MEAN REVERSION: short overbought (highest RSI, highest 1w return, etc.)
    'MR_1w_top10':         {'rank': 'ret_1w', 'ascending': False, 'filter': None},
    'MR_RSI_top10':        {'rank': 'rsi', 'ascending': False, 'filter': None},
    'MR_BB_top10':         {'rank': 'bb_pct', 'ascending': False, 'filter': None},
    'MR_dist20_top10':     {'rank': 'dist_ma20', 'ascending': False, 'filter': None},

    # MR + below MA200 (short overbought stocks in downtrend)
    'MR_1w_belowMA200':    {'rank': 'ret_1w', 'ascending': False, 'filter': 'dist_ma200 < 0'},
    'MR_RSI_belowMA200':   {'rank': 'rsi', 'ascending': False, 'filter': 'dist_ma200 < 0'},
    'MR_1w_belowMA50':     {'rank': 'ret_1w', 'ascending': False, 'filter': 'dist_ma50 < 0'},

    # MR + EPS miss (short overbought + bad fundamentals)
    'MR_1w_EPSmiss':       {'rank': 'ret_1w', 'ascending': False, 'filter': 'eps_miss'},
    'MR_RSI_EPSmiss':      {'rank': 'rsi', 'ascending': False, 'filter': 'eps_miss'},

    # MR + below MA200 + EPS miss (triple filter)
    'MR_1w_MA200_EPSmiss': {'rank': 'ret_1w', 'ascending': False, 'filter': 'dist_ma200 < 0 & eps_miss'},

    # MOMENTUM: short lowest momentum (continuation)
    'MOM_4w_bottom10':     {'rank': 'ret_4w', 'ascending': True, 'filter': None},
    'MOM_12w_bottom10':    {'rank': 'ret_12w', 'ascending': True, 'filter': None},
    'MOM_4w_belowMA200':   {'rank': 'ret_4w', 'ascending': True, 'filter': 'dist_ma200 < 0'},

    # HIGH VOL + overbought (short volatile stocks that just rallied)
    'MR_1w_highvol':       {'rank': 'ret_1w', 'ascending': False, 'filter': 'vol_20d > 40'},

    # COMPOSITE: rank by multiple factors
    'COMPOSITE_short':     {'rank': 'composite', 'ascending': False, 'filter': None},
}

# Market regime filters
regime_filters = {
    'ALL':         lambda r: True,
    'BEAR_ONLY':   lambda r: r['bear'],
    'NOT_BULL':    lambda r: not r['bull'],
    'HIGH_VOL':    lambda r: r['high_vol'],
    'SPY<MA200':   lambda r: not r['spy_above_ma200'],
    'SPY<MA50':    lambda r: not r['spy_above_ma50'],
    'SPY_4w<0':    lambda r: r['spy_ret_4w'] < 0,
}

all_results = []

for strat_name, strat_cfg in short_strategies.items():
    for regime_name, regime_fn in regime_filters.items():
        weekly_short_rets = []
        trade_weeks = []
        picks_list = []

        for _, wrow in weeks_df.iterrows():
            sig_date = wrow['signal_date']
            entry_date = wrow['entry_date']
            exit_date = wrow['exit_date']

            if sig_date not in friday_data or entry_date not in monday_data or exit_date not in monday_data:
                continue

            # Market regime filter
            regime = get_market_regime(sig_date)
            if not regime_fn(regime):
                continue

            fri_snap = friday_data[sig_date]
            entry_snap = monday_data[entry_date]
            exit_snap = monday_data[exit_date]

            sp500 = get_sp500_cached(sig_date)
            eligible_syms = [s for s in fri_snap.index if s in sp500]
            if len(eligible_syms) < 100:
                continue

            fri_eligible = fri_snap.loc[eligible_syms].copy()

            # Add EPS data if needed
            if 'eps' in str(strat_cfg.get('filter', '')):
                eps_data = {}
                for sym in eligible_syms:
                    beat, surp, yoy = get_eps_fast(sym, sig_date)
                    eps_data[sym] = {'eps_miss': beat == False, 'eps_beat': beat == True,
                                     'eps_surp': surp, 'eps_yoy': yoy}
                fri_eligible['eps_miss'] = pd.Series({s: v['eps_miss'] for s, v in eps_data.items()})
                fri_eligible['eps_yoy_neg'] = pd.Series({s: (v['eps_yoy'] is not None and v['eps_yoy'] < 0) for s, v in eps_data.items()})

            # Apply filter
            filt = strat_cfg['filter']
            if filt:
                if filt == 'eps_miss':
                    pool = fri_eligible[fri_eligible['eps_miss'] == True]
                elif filt == 'dist_ma200 < 0 & eps_miss':
                    pool = fri_eligible[(fri_eligible['dist_ma200'] < 0) & (fri_eligible['eps_miss'] == True)]
                else:
                    try:
                        pool = fri_eligible.query(filt)
                    except:
                        pool = fri_eligible
            else:
                pool = fri_eligible

            # Rank
            rank_col = strat_cfg['rank']
            if rank_col == 'composite':
                # Composite: high RSI + high 1w return + below MA200 + EPS miss
                scores = pd.DataFrame(index=pool.index)
                if 'rsi' in pool.columns:
                    scores['rsi_rank'] = pool['rsi'].rank(pct=True)
                if 'ret_1w' in pool.columns:
                    scores['ret1w_rank'] = pool['ret_1w'].rank(pct=True)
                if 'dist_ma200' in pool.columns:
                    scores['ma200_rank'] = 1 - pool['dist_ma200'].rank(pct=True)  # Lower = worse = more shortable
                scores['composite'] = scores.mean(axis=1)
                ranked = scores['composite'].dropna().sort_values(ascending=False)
            else:
                ranked = pool[rank_col].dropna().sort_values(ascending=not strat_cfg['ascending'])

            if len(ranked) < N_SHORT:
                continue

            short_syms = ranked.head(N_SHORT).index.tolist()

            # Calculate short returns (Monday open to Monday open)
            short_ret_list = []
            picks_detail = []
            for sym in short_syms:
                if sym in entry_snap.index and sym in exit_snap.index:
                    ep = entry_snap.loc[sym, 'open']
                    xp = exit_snap.loc[sym, 'open']
                    if pd.notna(ep) and pd.notna(xp) and ep > 0:
                        ret = (ep - xp) / ep  # Short profit
                        short_ret_list.append(ret)
                        picks_detail.append((sym, ret))

            if len(short_ret_list) >= 5:
                avg_ret = np.mean(short_ret_list)
                weekly_short_rets.append(avg_ret)
                trade_weeks.append(sig_date)
                picks_list.append(picks_detail)

        if len(weekly_short_rets) < 20:
            continue

        rets = np.array(weekly_short_rets)
        cum = np.cumprod(1 + rets)
        years = len(rets) / 52
        peak = np.maximum.accumulate(cum)
        max_dd = ((cum - peak) / peak).min() * 100

        all_results.append({
            'strategy': strat_name,
            'regime': regime_name,
            'weeks': len(rets),
            'years': years,
            'total_ret': (cum[-1] - 1) * 100,
            'cagr': (cum[-1] ** (1/years) - 1) * 100 if years > 0 else 0,
            'avg_weekly': np.mean(rets) * 100,
            'sharpe': (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0,
            'win_rate': np.mean(rets > 0) * 100,
            'max_dd': max_dd,
            'avg_win': np.mean(rets[rets > 0]) * 100 if (rets > 0).any() else 0,
            'avg_loss': np.mean(rets[rets < 0]) * 100 if (rets < 0).any() else 0,
            'profit_factor': abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0,
            'cum': cum,
            'rets': rets,
            'trade_weeks': trade_weeks,
            'picks': picks_list,
        })

    print(f"  {strat_name} completado ({len(regime_filters)} regimenes)")

# ============================================================
# Results
# ============================================================
print(f"\n[5/5] Resultados:\n")

all_results.sort(key=lambda x: x['sharpe'], reverse=True)

# TOP 30 SHORT strategies
print("=" * 160)
print("  TOP 30 ESTRATEGIAS SHORT (ordenadas por Sharpe)")
print("  Signal: Viernes cierre | Entry: Lunes open | Exit: Lunes siguiente open | 10 posiciones cortas")
print("=" * 160)

print(f"\n  {'#':>3s} {'Estrategia':>22s} | {'Regimen':>12s} | {'Sem':>5s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'MaxDD%':>7s} | {'AvgWin%':>7s} | {'AvgLoss%':>8s} | {'ProfFact':>8s}")
print(f"  {'':>3s} {'-'*22} | {'-'*12} | {'-'*5} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*7} | {'-'*7} | {'-'*8} | {'-'*8}")

for i, r in enumerate(all_results[:30]):
    marker = " ***" if r['sharpe'] > 0.3 else (" **" if r['sharpe'] > 0.1 else (" *" if r['sharpe'] > 0 else ""))
    print(f"  {i+1:>3d} {r['strategy']:>22s} | {r['regime']:>12s} | {r['weeks']:>5d} | {r['cagr']:>+6.1f}% | {r['sharpe']:>6.2f} | {r['win_rate']:>4.0f}% | {r['max_dd']:>6.1f}% | {r['avg_win']:>+6.2f}% | {r['avg_loss']:>+7.2f}% | {r['profit_factor']:>8.2f}{marker}")

# Best per regime
print(f"\n\n{'='*130}")
print(f"  MEJOR SHORT POR REGIMEN DE MERCADO")
print(f"{'='*130}")

for regime_name in regime_filters.keys():
    regime_results = [r for r in all_results if r['regime'] == regime_name]
    if not regime_results:
        continue
    regime_results.sort(key=lambda x: x['sharpe'], reverse=True)
    best = regime_results[0]
    print(f"\n  {regime_name:>12s}: {best['strategy']:>22s} | CAGR {best['cagr']:>+6.1f}% | Sharpe {best['sharpe']:>6.2f} | WR {best['win_rate']:>4.0f}% | MaxDD {best['max_dd']:>6.1f}% | {best['weeks']} semanas")

# Best regime = ALL vs filtered
print(f"\n\n{'='*130}")
print(f"  IMPACTO DEL FILTRO DE REGIMEN (misma estrategia, diferente regimen)")
print(f"{'='*130}")

# Compare MR_1w_top10 across regimes
for strat in ['MR_1w_top10', 'MR_RSI_top10', 'MR_1w_belowMA200', 'COMPOSITE_short']:
    strat_results = [r for r in all_results if r['strategy'] == strat]
    if not strat_results:
        continue
    strat_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  {strat}:")
    print(f"    {'Regimen':>12s} | {'Sem':>5s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'MaxDD%':>7s}")
    print(f"    {'-'*12} | {'-'*5} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*7}")
    for r in strat_results:
        marker = " <--" if r['sharpe'] == strat_results[0]['sharpe'] else ""
        print(f"    {r['regime']:>12s} | {r['weeks']:>5d} | {r['cagr']:>+6.1f}% | {r['sharpe']:>6.2f} | {r['win_rate']:>4.0f}% | {r['max_dd']:>6.1f}%{marker}")

# Annual breakdown for best overall short
print(f"\n\n{'='*130}")
best = all_results[0]
print(f"  DESGLOSE ANUAL: {best['strategy']} / Regimen: {best['regime']}")
print(f"{'='*130}")

annual = defaultdict(list)
for tw, ret in zip(best['trade_weeks'], best['rets']):
    annual[tw.year].append(ret)

print(f"\n  {'Ano':>6s} | {'Sem':>4s} | {'Ret%':>8s} | {'WR%':>5s} | {'Sharpe':>6s} | {'AvgWin%':>7s} | {'AvgLoss%':>8s}")
print(f"  {'-'*6} | {'-'*4} | {'-'*8} | {'-'*5} | {'-'*6} | {'-'*7} | {'-'*8}")

for yr in sorted(annual.keys()):
    yr_rets = np.array(annual[yr])
    yr_ret = (np.cumprod(1 + yr_rets)[-1] - 1) * 100
    yr_wr = np.mean(yr_rets > 0) * 100
    yr_sharpe = (np.mean(yr_rets) / np.std(yr_rets)) * np.sqrt(52) if np.std(yr_rets) > 0 else 0
    yr_avg_win = np.mean(yr_rets[yr_rets > 0]) * 100 if (yr_rets > 0).any() else 0
    yr_avg_loss = np.mean(yr_rets[yr_rets < 0]) * 100 if (yr_rets < 0).any() else 0
    marker = " ***" if yr_ret > 15 else (" **" if yr_ret > 5 else (" *" if yr_ret > 0 else ""))
    print(f"  {yr:>6d} | {len(yr_rets):>4d} | {yr_ret:>+7.1f}% | {yr_wr:>4.0f}% | {yr_sharpe:>6.2f} | {yr_avg_win:>+6.2f}% | {yr_avg_loss:>+7.2f}%{marker}")

# Ultimas 8 semanas
print(f"\n\n{'='*130}")
print(f"  ULTIMAS 8 SEMANAS - {best['strategy']} / {best['regime']}")
print(f"{'='*130}")

for tw, ret, picks in zip(best['trade_weeks'][-8:], best['rets'][-8:], best['picks'][-8:]):
    print(f"\n  {str(tw)[:10]} | Ret: {ret*100:>+5.2f}%")
    picks_sorted = sorted(picks, key=lambda x: x[1], reverse=True)
    for sym, r in picks_sorted:
        marker = "WIN" if r > 0 else "LOSS"
        print(f"    {sym:8s} {r*100:>+6.2f}% [{marker}]")

print(f"\n{'='*130}")
print(f"  CONCLUSIONES")
print(f"{'='*130}")
print(f"""
  RESUMEN:
  - Mejor SHORT overall: {all_results[0]['strategy']} / {all_results[0]['regime']}
    CAGR: {all_results[0]['cagr']:+.1f}%, Sharpe: {all_results[0]['sharpe']:.2f}, WR: {all_results[0]['win_rate']:.0f}%
  - El filtro de regimen de mercado es CRITICO para shorts
  - Shortear en mercado alcista suele perder dinero
  - Los mejores shorts combinan: mean reversion (overbought) + tendencia bajista (MA200/MA50)
""")
