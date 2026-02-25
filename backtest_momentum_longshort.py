"""
BACKTEST: Weekly Long/Short Momentum - S&P 500
===============================================
Strategy: Each week, go LONG top 10 momentum stocks, SHORT bottom 10.
Entry: Monday (test both Open and Close)
Exit: Friday Close
Universe: S&P 500 (using dateFirstAdded to filter per-week eligibility)
Momentum: Past N-week return (test 4w, 12w, 26w, 52w)
Period: 2005-2026 (max available)

Survivorship Bias Note:
- Uses CURRENT S&P 500 constituents only
- dateFirstAdded filters stocks that weren't yet in the index
- Stocks REMOVED from the index are NOT included (bias towards survivors)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# Load S&P 500 symbols with dateFirstAdded
with open('data/sp500_constituents.json') as f:
    sp500_data = json.load(f)

SP500 = {}
for d in sp500_data:
    sym = d['symbol']
    dfa = d.get('dateFirstAdded', '1957-01-01')
    try:
        SP500[sym] = pd.Timestamp(dfa)
    except:
        SP500[sym] = pd.Timestamp('1957-01-01')

symbols_list = list(SP500.keys())

print("=" * 130)
print("  BACKTEST: Weekly Long/Short Momentum - S&P 500")
print("  Long top 10 + Short bottom 10, rotacion semanal")
print("  Universo: 503 constituyentes actuales (con filtro dateFirstAdded)")
print("=" * 130)

# ============================================================
# STEP 1: Load all price data
# ============================================================
print("\n[1/5] Cargando datos de precio S&P 500...")

with engine.connect() as conn:
    df = pd.read_sql("""
        SELECT symbol, date, open, close
        FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s)
        AND date >= '2004-01-01'
        ORDER BY symbol, date
    """, conn, params={"syms": symbols_list}, parse_dates=['date'])

print(f"  Registros cargados: {len(df):,}")
print(f"  Simbolos: {df['symbol'].nunique()}")
print(f"  Rango: {df['date'].min().date()} a {df['date'].max().date()}")

# ============================================================
# STEP 2: Build weekly structure (Monday open/close, Friday close)
# ============================================================
print("\n[2/5] Construyendo estructura semanal...")

df['iso_year'] = df['date'].dt.isocalendar().year.astype(int)
df['iso_week'] = df['date'].dt.isocalendar().week.astype(int)

# First trading day of week = entry, Last trading day = exit
first_day = df.groupby(['symbol', 'iso_year', 'iso_week']).first().reset_index()
last_day = df.groupby(['symbol', 'iso_year', 'iso_week']).last().reset_index()

weekly = first_day[['symbol', 'iso_year', 'iso_week', 'date', 'open', 'close']].rename(
    columns={'date': 'mon_date', 'open': 'mon_open', 'close': 'mon_close'}
)
weekly = weekly.merge(
    last_day[['symbol', 'iso_year', 'iso_week', 'close']].rename(columns={'close': 'fri_close'}),
    on=['symbol', 'iso_year', 'iso_week']
)

# Weekly key for consistent ordering
weekly['week_key'] = weekly['iso_year'] * 100 + weekly['iso_week']

# Filter: only include stock in weeks AFTER its dateFirstAdded
weekly['first_added'] = weekly['symbol'].map(SP500)
weekly = weekly[weekly['mon_date'] >= weekly['first_added']].copy()

print(f"  Semanas-simbolo (tras filtro dateFirstAdded): {len(weekly):,}")

# ============================================================
# STEP 3: Calculate momentum lookback returns
# ============================================================
print("\n[3/5] Calculando momentum...")

# Pivot to get weekly close by symbol
pivot_close = weekly.pivot_table(index='week_key', columns='symbol', values='fri_close')
pivot_close = pivot_close.sort_index()

# Also create eligibility matrix (1 if stock is eligible that week, NaN otherwise)
# Stock is eligible if it has data (was in SP500 and had price)
eligible = pivot_close.notna().astype(float)
eligible = eligible.replace(0, np.nan)

lookbacks = {'4w': 4, '12w': 12, '26w': 26, '52w': 52}
momentum = {}

for name, periods in lookbacks.items():
    ret = pivot_close / pivot_close.shift(periods) - 1
    # Mask stocks that don't have enough history for this lookback
    momentum[name] = ret * eligible
    valid = ret.notna().sum(axis=1)
    print(f"  {name}: avg eligible stocks/week = {valid[valid > 0].mean():.0f}")

# ============================================================
# STEP 4: Run backtest for each configuration
# ============================================================
print("\n[4/5] Ejecutando backtest (8 configuraciones)...")

# Pivot for entry/exit prices
pivot_mon_open = weekly.pivot_table(index='week_key', columns='symbol', values='mon_open')
pivot_mon_close = weekly.pivot_table(index='week_key', columns='symbol', values='mon_close')
pivot_fri_close = weekly.pivot_table(index='week_key', columns='symbol', values='fri_close')

# Align indices
common_idx = pivot_close.index.intersection(pivot_mon_open.index).intersection(pivot_mon_close.index)
pivot_mon_open = pivot_mon_open.loc[common_idx]
pivot_mon_close = pivot_mon_close.loc[common_idx]
pivot_fri_close = pivot_fri_close.loc[common_idx]

# Week dates lookup
week_dates = weekly.drop_duplicates('week_key').set_index('week_key')['mon_date'].sort_index()

N_LONG = 10
N_SHORT = 10
MIN_STOCKS = 100  # Need at least 100 eligible stocks to trade

results_summary = []

for mom_name, mom_periods in lookbacks.items():
    mom = momentum[mom_name].reindex(common_idx)

    for entry_type in ['mon_open', 'mon_close']:
        entry_prices = pivot_mon_open if entry_type == 'mon_open' else pivot_mon_close
        exit_prices = pivot_fri_close

        weekly_returns = []
        weekly_long_rets = []
        weekly_short_rets = []
        trade_weeks = []
        long_picks_history = []
        short_picks_history = []

        for i in range(1, len(common_idx)):
            wk = common_idx[i]
            prev_wk = common_idx[i - 1]

            if prev_wk not in mom.index:
                continue

            # Get momentum ranking from PREVIOUS week's data
            mom_row = mom.loc[prev_wk].dropna()
            if len(mom_row) < MIN_STOCKS:
                continue

            # Rank by momentum
            ranked = mom_row.sort_values(ascending=False)
            long_symbols = ranked.head(N_LONG).index.tolist()
            short_symbols = ranked.tail(N_SHORT).index.tolist()

            # Calculate returns
            long_ret_list = []
            short_ret_list = []

            for sym in long_symbols:
                if sym in entry_prices.columns and sym in exit_prices.columns:
                    ep = entry_prices.loc[wk, sym]
                    xp = exit_prices.loc[wk, sym]
                    if pd.notna(ep) and pd.notna(xp) and ep > 0:
                        long_ret_list.append((xp - ep) / ep)

            for sym in short_symbols:
                if sym in entry_prices.columns and sym in exit_prices.columns:
                    ep = entry_prices.loc[wk, sym]
                    xp = exit_prices.loc[wk, sym]
                    if pd.notna(ep) and pd.notna(xp) and ep > 0:
                        short_ret_list.append((ep - xp) / ep)

            if len(long_ret_list) >= 5 and len(short_ret_list) >= 5:
                avg_long = np.mean(long_ret_list)
                avg_short = np.mean(short_ret_list)
                combined = (avg_long + avg_short) / 2

                weekly_returns.append(combined)
                weekly_long_rets.append(avg_long)
                weekly_short_rets.append(avg_short)
                trade_weeks.append(wk)
                long_picks_history.append(long_symbols)
                short_picks_history.append(short_symbols)

        if not weekly_returns:
            continue

        rets = np.array(weekly_returns)
        long_rets = np.array(weekly_long_rets)
        short_rets = np.array(weekly_short_rets)

        # Metrics
        total_weeks = len(rets)
        win_rate = np.mean(rets > 0) * 100
        avg_weekly = np.mean(rets) * 100
        std_weekly = np.std(rets) * 100
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
        cum = np.cumprod(1 + rets)
        total_return = (cum[-1] - 1) * 100
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min() * 100
        years = total_weeks / 52
        cagr = (cum[-1] ** (1 / years) - 1) * 100 if years > 0 else 0

        # Long-only metrics
        long_cum = np.cumprod(1 + long_rets)
        long_total = (long_cum[-1] - 1) * 100
        long_cagr = (long_cum[-1] ** (1 / years) - 1) * 100 if years > 0 else 0
        long_sharpe = (np.mean(long_rets) / np.std(long_rets)) * np.sqrt(52) if np.std(long_rets) > 0 else 0
        long_peak = np.maximum.accumulate(long_cum)
        long_max_dd = ((long_cum - long_peak) / long_peak).min() * 100

        # Short-only metrics
        short_cum = np.cumprod(1 + short_rets)
        short_total = (short_cum[-1] - 1) * 100
        short_cagr = (short_cum[-1] ** (1 / years) - 1) * 100 if years > 0 else 0
        short_sharpe = (np.mean(short_rets) / np.std(short_rets)) * np.sqrt(52) if np.std(short_rets) > 0 else 0

        first_date = week_dates.get(trade_weeks[0], None)
        last_date = week_dates.get(trade_weeks[-1], None)

        results_summary.append({
            'momentum': mom_name,
            'entry': entry_type,
            'weeks': total_weeks,
            'years': years,
            'first_date': first_date,
            'last_date': last_date,
            'total_return': total_return,
            'cagr': cagr,
            'avg_weekly': avg_weekly,
            'std_weekly': std_weekly,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_dd': max_dd,
            # Long only
            'long_total': long_total,
            'long_cagr': long_cagr,
            'long_avg': np.mean(long_rets) * 100,
            'long_win': np.mean(long_rets > 0) * 100,
            'long_sharpe': long_sharpe,
            'long_max_dd': long_max_dd,
            # Short only
            'short_total': short_total,
            'short_cagr': short_cagr,
            'short_avg': np.mean(short_rets) * 100,
            'short_win': np.mean(short_rets > 0) * 100,
            'short_sharpe': short_sharpe,
            # For detailed analysis
            'cum_returns': cum,
            'long_cum': long_cum,
            'short_cum': short_cum,
            'weekly_rets': rets,
            'weekly_long_rets': long_rets,
            'weekly_short_rets': short_rets,
            'trade_weeks': trade_weeks,
            'long_picks': long_picks_history,
            'short_picks': short_picks_history,
        })

        print(f"  {mom_name:>4s} / {entry_type:>10s}: {total_weeks} semanas, CAGR {cagr:>+5.1f}%, Sharpe {sharpe:.2f}, WR {win_rate:.0f}%, MaxDD {max_dd:.1f}%")

# ============================================================
# STEP 5: Print Results
# ============================================================
print(f"\n[5/5] Resultados:\n")

print("=" * 160)
print("  RESUMEN COMPLETO: BACKTEST LONG/SHORT MOMENTUM SEMANAL - S&P 500")
print("  Estrategia: Long Top 10 + Short Bottom 10, rotacion semanal")
print("=" * 160)

results_summary.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n  {'Mom':>4s} | {'Entry':>10s} | {'Sem':>5s} | {'Anos':>4s} | {'RetTotal%':>9s} | {'CAGR%':>6s} | {'Avg/Sem%':>8s} | {'Sharpe':>6s} | {'WR%':>5s} | {'MaxDD%':>7s} || {'L_CAGR%':>7s} | {'L_Sharpe':>8s} | {'L_WR%':>5s} | {'L_MaxDD%':>8s} || {'S_CAGR%':>7s} | {'S_Sharpe':>8s} | {'S_WR%':>5s}")
print(f"  {'-'*4} | {'-'*10} | {'-'*5} | {'-'*4} | {'-'*9} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*5} | {'-'*7} || {'-'*7} | {'-'*8} | {'-'*5} | {'-'*8} || {'-'*7} | {'-'*8} | {'-'*5}")

for r in results_summary:
    print(f"  {r['momentum']:>4s} | {r['entry']:>10s} | {r['weeks']:>5d} | {r['years']:>4.1f} | {r['total_return']:>+8.1f}% | {r['cagr']:>+5.1f}% | {r['avg_weekly']:>+7.3f}% | {r['sharpe']:>6.2f} | {r['win_rate']:>4.0f}% | {r['max_dd']:>6.1f}% || {r['long_cagr']:>+6.1f}% | {r['long_sharpe']:>8.2f} | {r['long_win']:>4.0f}% | {r['long_max_dd']:>7.1f}% || {r['short_cagr']:>+6.1f}% | {r['short_sharpe']:>8.2f} | {r['short_win']:>4.0f}%")

# Best config
best = results_summary[0]
print(f"\n  MEJOR CONFIGURACION (por Sharpe): Momentum {best['momentum']}, Entry {best['entry']}")
print(f"    CAGR: {best['cagr']:+.1f}%, Sharpe: {best['sharpe']:.2f}, WR: {best['win_rate']:.0f}%, MaxDD: {best['max_dd']:.1f}%")

# ============================================================
# Long/Short vs Long-Only vs Short-Only comparison
# ============================================================
print(f"\n\n{'='*130}")
print(f"  COMPARACION: Long/Short vs Long-Only vs Short-Only (mejor momentum: {best['momentum']})")
print(f"{'='*130}")

# Get both entry types for best momentum
best_mom_results = [r for r in results_summary if r['momentum'] == best['momentum']]

for r in best_mom_results:
    print(f"\n  Entry: {r['entry']}  ({r['first_date']} a {r['last_date']}, {r['weeks']} semanas)")
    print(f"  {'Estrategia':>15s} | {'Ret.Total%':>10s} | {'CAGR%':>7s} | {'Avg/Sem%':>8s} | {'Sharpe':>6s} | {'WR%':>5s} | {'MaxDD%':>7s}")
    print(f"  {'-'*15} | {'-'*10} | {'-'*7} | {'-'*8} | {'-'*6} | {'-'*5} | {'-'*7}")
    print(f"  {'Long/Short':>15s} | {r['total_return']:>+9.1f}% | {r['cagr']:>+6.1f}% | {r['avg_weekly']:>+7.3f}% | {r['sharpe']:>6.2f} | {r['win_rate']:>4.0f}% | {r['max_dd']:>6.1f}%")
    print(f"  {'Long Only':>15s} | {r['long_total']:>+9.1f}% | {r['long_cagr']:>+6.1f}% | {r['long_avg']:>+7.3f}% | {r['long_sharpe']:>6.2f} | {r['long_win']:>4.0f}% | {r['long_max_dd']:>6.1f}%")
    print(f"  {'Short Only':>15s} | {r['short_total']:>+9.1f}% | {r['short_cagr']:>+6.1f}% | {r['short_avg']:>+7.3f}% | {r['short_sharpe']:>6.2f} | {r['short_win']:>4.0f}% | {'N/A':>7s}")

# ============================================================
# Annual breakdown for BEST config
# ============================================================
print(f"\n\n{'='*130}")
print(f"  DESGLOSE ANUAL: {best['momentum']} / {best['entry']}")
print(f"{'='*130}")

rets = best['weekly_rets']
long_r = best['weekly_long_rets']
short_r = best['weekly_short_rets']
weeks = best['trade_weeks']

annual = defaultdict(lambda: {'ls': [], 'long': [], 'short': []})
for wk, r_ls, r_l, r_s in zip(weeks, rets, long_r, short_r):
    yr = wk // 100
    annual[yr]['ls'].append(r_ls)
    annual[yr]['long'].append(r_l)
    annual[yr]['short'].append(r_s)

print(f"\n  {'Ano':>6s} | {'Sem':>4s} | {'L/S Ret%':>8s} | {'L/S WR%':>7s} | {'L/S Sharpe':>10s} | {'Long Ret%':>9s} | {'Long WR%':>8s} | {'Short Ret%':>10s} | {'Short WR%':>9s}")
print(f"  {'-'*6} | {'-'*4} | {'-'*8} | {'-'*7} | {'-'*10} | {'-'*9} | {'-'*8} | {'-'*10} | {'-'*9}")

total_positive_years = 0
total_years = 0
for yr in sorted(annual.keys()):
    a = annual[yr]
    ls = np.array(a['ls'])
    lo = np.array(a['long'])
    sh = np.array(a['short'])

    ls_ret = (np.cumprod(1 + ls)[-1] - 1) * 100
    ls_wr = np.mean(ls > 0) * 100
    ls_sharpe = (np.mean(ls) / np.std(ls)) * np.sqrt(52) if np.std(ls) > 0 else 0

    lo_ret = (np.cumprod(1 + lo)[-1] - 1) * 100
    lo_wr = np.mean(lo > 0) * 100

    sh_ret = (np.cumprod(1 + sh)[-1] - 1) * 100
    sh_wr = np.mean(sh > 0) * 100

    marker = " ***" if ls_ret > 20 else (" **" if ls_ret > 10 else (" *" if ls_ret > 5 else (" -" if ls_ret < -5 else "")))
    total_years += 1
    if ls_ret > 0:
        total_positive_years += 1

    print(f"  {yr:>6d} | {len(ls):>4d} | {ls_ret:>+7.1f}% | {ls_wr:>6.1f}% | {ls_sharpe:>10.2f} | {lo_ret:>+8.1f}% | {lo_wr:>7.1f}% | {sh_ret:>+9.1f}% | {sh_wr:>8.1f}%{marker}")

print(f"\n  Anos positivos: {total_positive_years}/{total_years} ({total_positive_years/total_years*100:.0f}%)")

# ============================================================
# Drawdown analysis
# ============================================================
print(f"\n\n{'='*130}")
print(f"  ANALISIS DE DRAWDOWN: {best['momentum']} / {best['entry']}")
print(f"{'='*130}")

cum = best['cum_returns']
peak = np.maximum.accumulate(cum)
dd = (cum - peak) / peak

# Find top 5 drawdowns
dd_series = pd.Series(dd, index=best['trade_weeks'])

# Find drawdown periods
in_dd = dd < 0
dd_start = None
dd_periods = []
for i, (wk, d) in enumerate(zip(best['trade_weeks'], dd)):
    if d < 0 and dd_start is None:
        dd_start = i
    elif d >= 0 and dd_start is not None:
        min_dd = dd[dd_start:i].min()
        min_idx = dd_start + np.argmin(dd[dd_start:i])
        dd_periods.append({
            'start_wk': best['trade_weeks'][dd_start],
            'end_wk': best['trade_weeks'][i],
            'trough_wk': best['trade_weeks'][min_idx],
            'max_dd': min_dd * 100,
            'duration': i - dd_start
        })
        dd_start = None

dd_periods.sort(key=lambda x: x['max_dd'])

print(f"\n  Top 5 peores drawdowns:")
print(f"  {'#':>3s} | {'MaxDD%':>7s} | {'Duracion':>8s} | {'Inicio':>12s} | {'Valle':>12s} | {'Recuperacion':>12s}")
print(f"  {'-'*3} | {'-'*7} | {'-'*8} | {'-'*12} | {'-'*12} | {'-'*12}")

for i, dp in enumerate(dd_periods[:5]):
    start_dt = week_dates.get(dp['start_wk'], 'N/A')
    trough_dt = week_dates.get(dp['trough_wk'], 'N/A')
    end_dt = week_dates.get(dp['end_wk'], 'N/A')
    print(f"  {i+1:>3d} | {dp['max_dd']:>6.1f}% | {dp['duration']:>5d} sem | {str(start_dt)[:10]:>12s} | {str(trough_dt)[:10]:>12s} | {str(end_dt)[:10]:>12s}")

# ============================================================
# Last 8 weeks detail
# ============================================================
print(f"\n\n{'='*130}")
print(f"  ULTIMAS 8 SEMANAS (detalle picks)")
print(f"{'='*130}")

last_8 = list(zip(
    best['trade_weeks'][-8:],
    best['weekly_rets'][-8:],
    best['weekly_long_rets'][-8:],
    best['weekly_short_rets'][-8:],
    best['long_picks'][-8:],
    best['short_picks'][-8:],
))

mom_data = momentum[best['momentum']].reindex(common_idx)
entry_piv = pivot_mon_open if best['entry'] == 'mon_open' else pivot_mon_close

for wk, ret_ls, ret_l, ret_s, longs, shorts in last_8:
    wk_date = week_dates.get(wk, 'N/A')
    idx = list(common_idx).index(wk)
    prev_wk = common_idx[idx - 1] if idx > 0 else None
    mom_row = mom_data.loc[prev_wk] if prev_wk is not None else pd.Series()

    print(f"\n  Semana {wk} ({str(wk_date)[:10]}) | L/S: {ret_ls*100:>+5.2f}% | Long: {ret_l*100:>+5.2f}% | Short: {ret_s*100:>+5.2f}%")

    print(f"    LONG (top momentum):", end="")
    for sym in longs[:5]:
        m = mom_row.get(sym, 0)
        ep = entry_piv.loc[wk, sym] if sym in entry_piv.columns else None
        xp = pivot_fri_close.loc[wk, sym] if sym in pivot_fri_close.columns else None
        if pd.notna(ep) and pd.notna(xp) and ep > 0:
            r = (xp - ep) / ep * 100
            print(f"  {sym}({m*100:+.0f}%/{r:+.1f}%)", end="")
    print()

    print(f"    SHORT (peor momentum):", end="")
    for sym in shorts[:5]:
        m = mom_row.get(sym, 0)
        ep = entry_piv.loc[wk, sym] if sym in entry_piv.columns else None
        xp = pivot_fri_close.loc[wk, sym] if sym in pivot_fri_close.columns else None
        if pd.notna(ep) and pd.notna(xp) and ep > 0:
            r = (ep - xp) / ep * 100
            print(f"  {sym}({m*100:+.0f}%/{r:+.1f}%)", end="")
    print()

# ============================================================
# Money Management example
# ============================================================
print(f"\n\n{'='*130}")
print(f"  EJEMPLO MONEY MANAGEMENT: Capital $100,000")
print(f"{'='*130}")

capital = 100000
print(f"\n  Capital inicial: ${capital:,.0f}")
print(f"  Posiciones largas: {N_LONG} x ${capital/N_LONG:,.0f} = ${capital:,.0f}")
print(f"  Posiciones cortas: {N_SHORT} x ${capital/N_SHORT:,.0f} = ${capital:,.0f}")
print(f"  Exposicion total: ${capital*2:,.0f} (2x leverage)")
print(f"  Exposicion neta: $0 (market neutral)")

final_ls = capital * best['cum_returns'][-1]
final_lo = capital * best['long_cum'][-1]
print(f"\n  Resultado {best['years']:.1f} anos:")
print(f"    Long/Short: ${capital:,.0f} -> ${final_ls:,.0f} ({best['total_return']:+.1f}%)")
print(f"    Long Only:  ${capital:,.0f} -> ${final_lo:,.0f} ({best['long_total']:+.1f}%)")

# ============================================================
# Notes
# ============================================================
print(f"\n\n{'='*130}")
print(f"  NOTAS Y LIMITACIONES")
print(f"{'='*130}")
print(f"""
  SURVIVORSHIP BIAS:
  - Se usan los 503 constituyentes ACTUALES del S&P 500
  - Cada accion solo se incluye desde su fecha de incorporacion (dateFirstAdded)
  - Las acciones ELIMINADAS del indice NO estan (Enron, Lehman, GE cuando cayo, etc.)
  - Esto SOBREESTIMA los retornos, especialmente del lado corto

  COSTES NO INCLUIDOS:
  - Comisiones de broker (tipico $0.005/accion = ~$1/operacion)
  - Spread bid/ask (tipico 0.01-0.05% para SP500 large caps)
  - Slippage (impacto en precio por tamano de orden)
  - Coste de prestamo de acciones para shorts (tipico 0.5-3% anual)
  - Impacto estimado: -2% a -5% anual sobre retornos brutos

  OTROS:
  - Momentum calculado con precios de cierre (close), no ajustados por dividendos
  - Semanas parciales (festivos) tratadas como semanas normales
  - No se consideran earnings dates, ex-dividend, corporate actions
  - Resultados pasados NO garantizan resultados futuros
""")
