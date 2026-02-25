"""
Backtest Sector Rotation - Capa 1: ETFs Sectoriales
=====================================================
Estrategia: Detectar flujos de dinero entre sectores
- Long mejor sector, Short peor sector
- Señales: momentum, volatilidad relativa, flujos (volumen)
- Rebalanceo semanal (viernes cierre)
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────
DB_URI = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'
START_DATE = '2000-01-01'
END_DATE = '2026-02-07'

# 9 sectores con historial largo (desde 1998)
ETF_SECTOR = {
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Cons. Discret.',
    'XLP': 'Cons. Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials'
}
ETFS = list(ETF_SECTOR.keys())

CAPITAL_PER_SIDE = 500_000  # $500K long + $500K short

# ── Data Loading ────────────────────────────────────────
print("=" * 70)
print("BACKTEST SECTOR ROTATION - CAPA 1: ETFs SECTORIALES")
print("=" * 70)

conn = psycopg2.connect(DB_URI)

# Load sector ETF prices
placeholders = ','.join([f"'{e}'" for e in ETFS])
query = f"""
SELECT symbol, date, open, high, low, close, volume
FROM fmp_price_history
WHERE symbol IN ({placeholders})
  AND date >= '{START_DATE}' AND date <= '{END_DATE}'
ORDER BY date, symbol
"""
df = pd.read_sql(query, conn)
df['date'] = pd.to_datetime(df['date'])

# Load SPY as benchmark
spy = pd.read_sql(f"""
SELECT date, close as spy_close
FROM fmp_price_history
WHERE symbol = 'SPY' AND date >= '{START_DATE}' AND date <= '{END_DATE}'
ORDER BY date
""", conn)
spy['date'] = pd.to_datetime(spy['date'])

conn.close()
print(f"\nDatos cargados: {len(df)} filas, {df.date.min().date()} a {df.date.max().date()}")
print(f"ETFs: {', '.join(ETFS)}")

# ── Pivot to wide format ────────────────────────────────
prices = df.pivot_table(index='date', columns='symbol', values='close')
volumes = df.pivot_table(index='date', columns='symbol', values='volume')
prices = prices[ETFS].dropna()
volumes = volumes[ETFS].reindex(prices.index).fillna(0)

# Merge SPY
spy = spy.set_index('date')
prices = prices.join(spy, how='left').ffill()

print(f"Fechas con datos completos: {len(prices)} ({prices.index[0].date()} a {prices.index[-1].date()})")

# ── Weekly Returns ──────────────────────────────────────
# Resample to weekly (Friday close)
weekly_prices = prices.resample('W-FRI').last().dropna()
weekly_volumes = volumes.resample('W-FRI').sum()
weekly_volumes = weekly_volumes.reindex(weekly_prices.index).fillna(0)

weekly_ret = weekly_prices[ETFS].pct_change()
spy_weekly_ret = weekly_prices['spy_close'].pct_change()

print(f"Semanas: {len(weekly_prices)}")

# ── Signal Construction ─────────────────────────────────
def compute_signals(weekly_prices, weekly_ret, weekly_volumes, spy_weekly_ret):
    """Compute ranking signals for each ETF each week."""
    signals = {}

    for lookback_label, lookback in [('1w', 1), ('2w', 2), ('4w', 4), ('8w', 8), ('12w', 12)]:
        # Momentum: return over lookback period
        mom = weekly_ret.rolling(lookback).sum()
        signals[f'mom_{lookback_label}'] = mom

        # Relative strength vs SPY
        spy_mom = spy_weekly_ret.rolling(lookback).sum()
        rel_strength = mom.sub(spy_mom, axis=0)
        signals[f'rs_{lookback_label}'] = rel_strength

    # Volatility (4-week rolling)
    vol_4w = weekly_ret.rolling(4).std()
    signals['vol_4w'] = vol_4w

    # Volatility (8-week rolling)
    vol_8w = weekly_ret.rolling(8).std()
    signals['vol_8w'] = vol_8w

    # Volume change (proxy for flows) - 4w avg vs 12w avg
    vol_ratio = weekly_volumes.rolling(4).mean() / weekly_volumes.rolling(12).mean()
    signals['vol_flow_4v12'] = vol_ratio

    # Mean reversion signal: 1-week return (for contrarian)
    signals['mean_rev_1w'] = -weekly_ret  # negative = buy losers

    # Momentum acceleration: 4w mom - 8w mom
    signals['mom_accel'] = signals['mom_4w'] - signals['mom_8w']

    # Risk-adjusted momentum: mom / vol
    signals['sharpe_4w'] = signals['mom_4w'] / (vol_4w + 1e-10)
    signals['sharpe_8w'] = signals['mom_8w'] / (vol_8w + 1e-10)

    return signals

signals = compute_signals(weekly_prices, weekly_ret, weekly_volumes, spy_weekly_ret)
print(f"\nSeñales calculadas: {list(signals.keys())}")

# ── Backtest Engine ─────────────────────────────────────
def run_backtest(signal_df, weekly_ret, n_long=1, n_short=1, label=""):
    """
    Each week, rank sectors by signal.
    Go Long top n_long, Short bottom n_short.
    Equal weight within each side.
    """
    results = []

    for i in range(1, len(weekly_ret)):
        date = weekly_ret.index[i]
        prev_date = weekly_ret.index[i - 1]

        # Signal from previous week (known at rebalance time)
        sig = signal_df.loc[prev_date]
        if sig.isna().all():
            continue

        # Rank sectors (higher signal = better)
        ranked = sig.dropna().sort_values(ascending=False)
        if len(ranked) < n_long + n_short:
            continue

        long_etfs = ranked.head(n_long).index.tolist()
        short_etfs = ranked.tail(n_short).index.tolist()

        # Returns for this week
        week_rets = weekly_ret.loc[date]

        long_ret = week_rets[long_etfs].mean()
        short_ret = -week_rets[short_etfs].mean()  # short = negative of return

        total_ret = (long_ret + short_ret) / 2  # equal capital each side
        long_only_ret = long_ret

        results.append({
            'date': date,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'total_ret': total_ret,
            'long_etfs': ','.join(long_etfs),
            'short_etfs': ','.join(short_etfs)
        })

    if not results:
        return None

    res = pd.DataFrame(results).set_index('date')

    # Metrics
    n_weeks = len(res)
    total_pnl = (res['total_ret'] * CAPITAL_PER_SIDE).sum()
    long_pnl = (res['long_ret'] * CAPITAL_PER_SIDE).sum()
    short_pnl = (res['short_ret'] * CAPITAL_PER_SIDE).sum()

    ann_ret = res['total_ret'].mean() * 52
    ann_vol = res['total_ret'].std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    win_rate = (res['total_ret'] > 0).mean()

    # Max drawdown
    cum = (1 + res['total_ret']).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()

    # Annual breakdown
    res['year'] = res.index.year
    annual = res.groupby('year')['total_ret'].agg(['sum', 'count', 'mean', 'std'])
    annual['pnl'] = annual['sum'] * CAPITAL_PER_SIDE
    positive_years = (annual['pnl'] > 0).sum()
    total_years = len(annual)

    return {
        'label': label,
        'n_weeks': n_weeks,
        'total_pnl': total_pnl,
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'pos_years': f"{positive_years}/{total_years}",
        'annual': annual,
        'results': res
    }

# ── Run All Signal Variants ─────────────────────────────
print("\n" + "=" * 70)
print("RESULTADOS: 1 LONG + 1 SHORT (sector ETFs)")
print("=" * 70)

all_results = []
for sig_name, sig_df in sorted(signals.items()):
    result = run_backtest(sig_df, weekly_ret, n_long=1, n_short=1, label=sig_name)
    if result:
        all_results.append(result)

# Sort by Sharpe
all_results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Signal':<20} {'PnL':>12} {'Long PnL':>12} {'Short PnL':>12} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'Years+':>8}")
print("-" * 100)
for r in all_results:
    print(f"{r['label']:<20} ${r['total_pnl']:>10,.0f} ${r['long_pnl']:>10,.0f} ${r['short_pnl']:>10,.0f} {r['sharpe']:>8.2f} {r['win_rate']:>7.1%} {r['max_dd']:>7.1%} {r['pos_years']:>8}")

# ── Test with 2 Long + 2 Short ─────────────────────────
print("\n" + "=" * 70)
print("RESULTADOS: 2 LONG + 2 SHORT")
print("=" * 70)

all_results_2 = []
for sig_name, sig_df in sorted(signals.items()):
    result = run_backtest(sig_df, weekly_ret, n_long=2, n_short=2, label=sig_name)
    if result:
        all_results_2.append(result)

all_results_2.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Signal':<20} {'PnL':>12} {'Long PnL':>12} {'Short PnL':>12} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'Years+':>8}")
print("-" * 100)
for r in all_results_2:
    print(f"{r['label']:<20} ${r['total_pnl']:>10,.0f} ${r['long_pnl']:>10,.0f} ${r['short_pnl']:>10,.0f} {r['sharpe']:>8.2f} {r['win_rate']:>7.1%} {r['max_dd']:>7.1%} {r['pos_years']:>8}")

# ── Best signal: annual breakdown ───────────────────────
best = all_results[0]
print(f"\n{'=' * 70}")
print(f"MEJOR SEÑAL: {best['label']} (1L + 1S)")
print(f"{'=' * 70}")
print(f"\nPnL Total: ${best['total_pnl']:,.0f}")
print(f"  Long:  ${best['long_pnl']:,.0f}")
print(f"  Short: ${best['short_pnl']:,.0f}")
print(f"Sharpe: {best['sharpe']:.2f}")
print(f"Win Rate: {best['win_rate']:.1%}")
print(f"Max DD: {best['max_dd']:.1%}")

print(f"\n{'Year':<8} {'PnL':>12} {'Weeks':>8} {'Avg Ret':>10} {'StdDev':>10}")
print("-" * 52)
for year, row in best['annual'].iterrows():
    pnl = row['pnl']
    marker = " +" if pnl > 0 else " -"
    print(f"{year:<8} ${pnl:>10,.0f} {row['count']:>8.0f} {row['mean']:>9.2%} {row['std']:>9.2%}{marker}")

# ── Composite Signal ────────────────────────────────────
print(f"\n{'=' * 70}")
print("SEÑAL COMPUESTA (momentum + rel.strength + flows)")
print(f"{'=' * 70}")

# Normalize signals to z-scores for combination
def zscore_signal(sig_df):
    """Convert to cross-sectional z-score each week."""
    mean = sig_df.mean(axis=1)
    std = sig_df.std(axis=1)
    return sig_df.sub(mean, axis=0).div(std + 1e-10, axis=0)

# Composite: combine best performing individual signals
composite_configs = [
    ("mom4w+rs4w", {'mom_4w': 0.5, 'rs_4w': 0.5}),
    ("mom4w+rs4w+flow", {'mom_4w': 0.4, 'rs_4w': 0.4, 'vol_flow_4v12': 0.2}),
    ("mom8w+rs4w", {'mom_8w': 0.5, 'rs_4w': 0.5}),
    ("sharpe4w+rs4w", {'sharpe_4w': 0.5, 'rs_4w': 0.5}),
    ("sharpe8w+rs4w+flow", {'sharpe_8w': 0.4, 'rs_4w': 0.4, 'vol_flow_4v12': 0.2}),
    ("mom4w+rs8w+flow+accel", {'mom_4w': 0.3, 'rs_8w': 0.3, 'vol_flow_4v12': 0.2, 'mom_accel': 0.2}),
    ("all_momentum", {'mom_4w': 0.25, 'mom_8w': 0.25, 'rs_4w': 0.25, 'sharpe_4w': 0.25}),
]

composite_results = []
for label, weights in composite_configs:
    composite = None
    for sig_name, weight in weights.items():
        z = zscore_signal(signals[sig_name])
        if composite is None:
            composite = z * weight
        else:
            composite = composite + z * weight

    for n in [1, 2]:
        r = run_backtest(composite, weekly_ret, n_long=n, n_short=n, label=f"{label} ({n}L{n}S)")
        if r:
            composite_results.append(r)

composite_results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Signal':<35} {'PnL':>12} {'Long PnL':>12} {'Short PnL':>12} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'Yr+':>6}")
print("-" * 110)
for r in composite_results:
    print(f"{r['label']:<35} ${r['total_pnl']:>10,.0f} ${r['long_pnl']:>10,.0f} ${r['short_pnl']:>10,.0f} {r['sharpe']:>8.2f} {r['win_rate']:>7.1%} {r['max_dd']:>7.1%} {r['pos_years']:>6}")

# ── Best composite: annual breakdown ────────────────────
best_comp = composite_results[0]
print(f"\n{'=' * 70}")
print(f"MEJOR COMPUESTA: {best_comp['label']}")
print(f"{'=' * 70}")
print(f"\nPnL Total: ${best_comp['total_pnl']:,.0f}")
print(f"  Long:  ${best_comp['long_pnl']:,.0f}")
print(f"  Short: ${best_comp['short_pnl']:,.0f}")
print(f"Sharpe: {best_comp['sharpe']:.2f}")
print(f"Win Rate: {best_comp['win_rate']:.1%}")
print(f"Max DD: {best_comp['max_dd']:.1%}")

print(f"\n{'Year':<8} {'PnL':>12} {'Weeks':>8} {'Avg Ret':>10}")
print("-" * 42)
for year, row in best_comp['annual'].iterrows():
    pnl = row['pnl']
    marker = " +" if pnl > 0 else " -"
    print(f"{year:<8} ${pnl:>10,.0f} {row['count']:>8.0f} {row['mean']:>9.2%}{marker}")

# ── Frequency analysis: which sectors appear most? ──────
print(f"\n{'=' * 70}")
print(f"FRECUENCIA DE SECTORES (mejor señal: {best['label']})")
print(f"{'=' * 70}")

long_counts = best['results']['long_etfs'].value_counts()
short_counts = best['results']['short_etfs'].value_counts()

print(f"\n{'Sector':<8} {'Long freq':>12} {'Short freq':>12} {'Net':>8}")
print("-" * 44)
for etf in ETFS:
    l = long_counts.get(etf, 0)
    s = short_counts.get(etf, 0)
    name = ETF_SECTOR[etf]
    print(f"{etf:<8} {l:>12} {s:>12} {l-s:>8}  ({name})")
