"""
ANALISIS VIX: Umbrales optimos para sistema adaptativo Long/Short
=================================================================
- Distribucion historica del VIX
- Rendimiento de shorts por nivel de VIX
- Buscar umbral optimo para activar pata corta
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

# ============================================================
# Load VIX + SPY + Stocks
# ============================================================
print("=" * 130)
print("  ANALISIS VIX: Umbrales para sistema adaptativo Long/Short")
print("=" * 130)

print("\n[1/4] Cargando datos...")

with engine.connect() as conn:
    vix = pd.read_sql("SELECT date, open, high, low, close FROM price_history_vix ORDER BY date",
                       conn, parse_dates=['date'])
    spy = pd.read_sql("""SELECT date, open, close FROM fmp_price_history
                         WHERE symbol = 'SPY' AND date >= '2004-01-01' ORDER BY date""",
                       conn, parse_dates=['date'])

print(f"  VIX: {len(vix)} dias ({vix['date'].min().date()} a {vix['date'].max().date()})")
print(f"  SPY: {len(spy)} dias")

# ============================================================
# VIX Distribution
# ============================================================
print(f"\n[2/4] Distribucion historica del VIX\n")

vix_post2005 = vix[vix['date'] >= '2005-01-01']['close']

print(f"  Estadisticas VIX (2005-2026):")
print(f"    Media:    {vix_post2005.mean():.1f}")
print(f"    Mediana:  {vix_post2005.median():.1f}")
print(f"    Std:      {vix_post2005.std():.1f}")
print(f"    Min:      {vix_post2005.min():.1f}")
print(f"    Max:      {vix_post2005.max():.1f}")
print(f"    P10:      {vix_post2005.quantile(0.10):.1f}")
print(f"    P25:      {vix_post2005.quantile(0.25):.1f}")
print(f"    P50:      {vix_post2005.quantile(0.50):.1f}")
print(f"    P75:      {vix_post2005.quantile(0.75):.1f}")
print(f"    P90:      {vix_post2005.quantile(0.90):.1f}")

# Distribution by ranges
ranges = [(0, 12), (12, 15), (15, 18), (18, 20), (20, 22), (22, 25), (25, 30), (30, 40), (40, 100)]
print(f"\n  Distribucion por rangos:")
print(f"  {'Rango VIX':>12s} | {'Dias':>6s} | {'% Tiempo':>8s} | {'Histograma'}")
print(f"  {'-'*12} | {'-'*6} | {'-'*8} | {'-'*40}")

total = len(vix_post2005)
for lo, hi in ranges:
    count = ((vix_post2005 >= lo) & (vix_post2005 < hi)).sum()
    pct = count / total * 100
    bar = '#' * int(pct)
    label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
    marker = ""
    if 18 <= 19.09 < 20 and lo == 18:
        marker = " <-- VIX ACTUAL (19.09)"
    print(f"  {label:>12s} | {count:>6d} | {pct:>7.1f}% | {bar}{marker}")

# ============================================================
# VIX level vs SHORT performance (weekly)
# ============================================================
print(f"\n[3/4] Rendimiento SHORT por nivel de VIX...")

# Load stock data and rebuild weekly short returns
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
            if c.get('symbol') and c['symbol'] in members:
                members.discard(c['symbol'])
            if c.get('removedTicker'):
                members.add(c['removedTicker'])
    return members

sp500_cache = {}
def get_sp500_cached(sig_date):
    mk = str(sig_date)[:7]
    if mk not in sp500_cache:
        sp500_cache[mk] = get_sp500_at_date(sig_date)
    return sp500_cache[mk]

print("  Cargando precios stocks...")
with engine.connect() as conn:
    df = pd.read_sql("""
        SELECT symbol, date, open, close FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2004-01-01'
        ORDER BY symbol, date
    """, conn, params={"syms": list(all_sp500_symbols)}, parse_dates=['date'])

# Calculate 1w return for mean reversion signal
print("  Calculando indicadores...")
def calc_ret1w(group):
    g = group.sort_values('date').copy()
    g['ret_1w'] = g['close'] / g['close'].shift(5) - 1
    g['rsi'] = None
    delta = g['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g['rsi'] = 100 - (100 / (1 + rs))
    g['ma200'] = g['close'].rolling(200).mean()
    g['dist_ma200'] = (g['close'] - g['ma200']) / g['ma200'] * 100
    return g

df = df.groupby('symbol', group_keys=False).apply(calc_ret1w)
df_indexed = df.set_index(['symbol', 'date']).sort_index()

# Build weekly pairs and VIX at signal
vix_indexed = vix.set_index('date')['close']

df['weekday'] = df['date'].dt.weekday
fridays = np.sort(df[df['weekday'] == 4]['date'].unique())
mondays = np.sort(df[df['weekday'] == 0]['date'].unique())

weeks = []
for fri in fridays:
    next_mons = mondays[mondays > fri]
    if len(next_mons) < 2:
        continue
    # Get VIX on Friday
    vix_val = vix_indexed.get(pd.Timestamp(fri), None)
    if vix_val is None or pd.isna(vix_val):
        # Try day before
        for delta in range(1, 4):
            vix_val = vix_indexed.get(pd.Timestamp(fri) - pd.Timedelta(days=delta), None)
            if vix_val is not None and not pd.isna(vix_val):
                break
    if vix_val is None or pd.isna(vix_val):
        continue
    weeks.append({
        'signal_date': pd.Timestamp(fri),
        'entry_date': pd.Timestamp(next_mons[0]),
        'exit_date': pd.Timestamp(next_mons[1]),
        'vix': float(vix_val),
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

print(f"  Semanas con VIX: {len(weeks_df)}")

# Run SHORT strategy (MR_1w: short top 10 highest 1w return) for each week
# Also run LONG strategy (MR_1w: long bottom 10 lowest 1w return)
# Also run COMPOSITE short (RSI + ret_1w + dist_ma200)
N = 10

print("  Calculando retornos semanales...")

results_by_vix = []

for _, wrow in weeks_df.iterrows():
    sig = wrow['signal_date']
    entry = wrow['entry_date']
    exit_d = wrow['exit_date']
    vix_val = wrow['vix']

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

    # MR_1w SHORT: short top 10 highest 1w return (overbought)
    ret1w = fri_e['ret_1w'].dropna().sort_values(ascending=False)
    short_syms = ret1w.head(N).index.tolist()
    long_syms = ret1w.tail(N).index.tolist()

    # COMPOSITE SHORT
    scores = pd.DataFrame(index=fri_e.index)
    if 'rsi' in fri_e.columns:
        scores['rsi_r'] = fri_e['rsi'].rank(pct=True)
    if 'ret_1w' in fri_e.columns:
        scores['ret1w_r'] = fri_e['ret_1w'].rank(pct=True)
    if 'dist_ma200' in fri_e.columns:
        scores['ma200_r'] = 1 - fri_e['dist_ma200'].rank(pct=True)
    scores['comp'] = scores.mean(axis=1)
    comp_ranked = scores['comp'].dropna().sort_values(ascending=False)
    comp_short_syms = comp_ranked.head(N).index.tolist()

    # Calculate returns
    def calc_short_ret(syms):
        rets = []
        for sym in syms:
            if sym in entry_snap.index and sym in exit_snap.index:
                ep = entry_snap.loc[sym, 'open']
                xp = exit_snap.loc[sym, 'open']
                if pd.notna(ep) and pd.notna(xp) and ep > 0:
                    rets.append((ep - xp) / ep)
        return np.mean(rets) if rets else None

    def calc_long_ret(syms):
        rets = []
        for sym in syms:
            if sym in entry_snap.index and sym in exit_snap.index:
                ep = entry_snap.loc[sym, 'open']
                xp = exit_snap.loc[sym, 'open']
                if pd.notna(ep) and pd.notna(xp) and ep > 0:
                    rets.append((xp - ep) / ep)
        return np.mean(rets) if rets else None

    short_ret = calc_short_ret(short_syms)
    long_ret = calc_long_ret(long_syms)
    comp_short_ret = calc_short_ret(comp_short_syms)

    if short_ret is not None and long_ret is not None:
        results_by_vix.append({
            'date': sig,
            'vix': vix_val,
            'short_mr1w': short_ret,
            'long_mr1w': long_ret,
            'ls_mr1w': (long_ret + short_ret) / 2,
            'comp_short': comp_short_ret,
            'ls_comp': (long_ret + comp_short_ret) / 2 if comp_short_ret else None,
        })

rdf = pd.DataFrame(results_by_vix)
print(f"  Semanas con resultados: {len(rdf)}")

# ============================================================
# Analysis by VIX buckets
# ============================================================
print(f"\n[4/4] Resultados por nivel de VIX\n")

vix_buckets = [
    (0, 12, 'VIX < 12 (muy bajo)'),
    (12, 15, 'VIX 12-15 (bajo)'),
    (15, 18, 'VIX 15-18 (normal)'),
    (18, 20, 'VIX 18-20 (elevado)'),
    (20, 22, 'VIX 20-22 (alto)'),
    (22, 25, 'VIX 22-25 (muy alto)'),
    (25, 30, 'VIX 25-30 (panico leve)'),
    (30, 40, 'VIX 30-40 (panico)'),
    (40, 100, 'VIX 40+ (crash)'),
]

# SHORT MR_1w by VIX bucket
print("=" * 140)
print("  SHORT MR_1w (short top 10 overbought) POR NIVEL DE VIX")
print("=" * 140)
print(f"\n  {'Rango VIX':>25s} | {'Sem':>5s} | {'%Tiempo':>7s} | {'AvgRet%':>7s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'AvgWin%':>7s} | {'AvgLoss%':>8s} | {'PF':>5s}")
print(f"  {'-'*25} | {'-'*5} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*7} | {'-'*8} | {'-'*5}")

for lo, hi, label in vix_buckets:
    mask = (rdf['vix'] >= lo) & (rdf['vix'] < hi)
    subset = rdf[mask]['short_mr1w'].dropna()
    if len(subset) < 5:
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
    pct_time = len(subset) / len(rdf) * 100
    cagr_str = f"{cagr:>+6.1f}%" if not np.isnan(cagr) else "   N/A "
    marker = " ***" if sharpe > 0.3 else (" **" if sharpe > 0 else "")
    print(f"  {label:>25s} | {len(subset):>5d} | {pct_time:>6.1f}% | {np.mean(rets)*100:>+6.3f}% | {cagr_str} | {sharpe:>6.2f} | {wr:>4.0f}% | {avg_w:>+6.2f}% | {avg_l:>+7.2f}% | {pf:>5.2f}{marker}")

# COMPOSITE SHORT by VIX bucket
print(f"\n\n{'='*140}")
print("  COMPOSITE SHORT (RSI + ret_1w + dist_ma200 inverso) POR NIVEL DE VIX")
print("=" * 140)
print(f"\n  {'Rango VIX':>25s} | {'Sem':>5s} | {'%Tiempo':>7s} | {'AvgRet%':>7s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'PF':>5s}")
print(f"  {'-'*25} | {'-'*5} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*5}")

for lo, hi, label in vix_buckets:
    mask = (rdf['vix'] >= lo) & (rdf['vix'] < hi)
    subset = rdf[mask]['comp_short'].dropna()
    if len(subset) < 5:
        continue
    rets = subset.values
    cum = np.cumprod(1 + rets)
    years = len(rets) / 52
    cagr = (cum[-1] ** (1/years) - 1) * 100 if years > 0.5 else np.nan
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
    wr = np.mean(rets > 0) * 100
    pf = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0
    pct_time = len(subset) / len(rdf) * 100
    cagr_str = f"{cagr:>+6.1f}%" if not np.isnan(cagr) else "   N/A "
    marker = " ***" if sharpe > 0.3 else (" **" if sharpe > 0 else "")
    print(f"  {label:>25s} | {len(subset):>5d} | {pct_time:>6.1f}% | {np.mean(rets)*100:>+6.3f}% | {cagr_str} | {sharpe:>6.2f} | {wr:>4.0f}% | {pf:>5.2f}{marker}")

# LONG MR_1w by VIX bucket
print(f"\n\n{'='*140}")
print("  LONG MR_1w (long bottom 10 oversold) POR NIVEL DE VIX")
print("=" * 140)
print(f"\n  {'Rango VIX':>25s} | {'Sem':>5s} | {'%Tiempo':>7s} | {'AvgRet%':>7s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'PF':>5s}")
print(f"  {'-'*25} | {'-'*5} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*5}")

for lo, hi, label in vix_buckets:
    mask = (rdf['vix'] >= lo) & (rdf['vix'] < hi)
    subset = rdf[mask]['long_mr1w'].dropna()
    if len(subset) < 5:
        continue
    rets = subset.values
    cum = np.cumprod(1 + rets)
    years = len(rets) / 52
    cagr = (cum[-1] ** (1/years) - 1) * 100 if years > 0.5 else np.nan
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
    wr = np.mean(rets > 0) * 100
    pf = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0
    pct_time = len(subset) / len(rdf) * 100
    cagr_str = f"{cagr:>+6.1f}%" if not np.isnan(cagr) else "   N/A "
    marker = " ***" if sharpe > 0.3 else (" **" if sharpe > 0 else "")
    print(f"  {label:>25s} | {len(subset):>5d} | {pct_time:>6.1f}% | {np.mean(rets)*100:>+6.3f}% | {cagr_str} | {sharpe:>6.2f} | {wr:>4.0f}% | {pf:>5.2f}{marker}")

# LONG/SHORT COMBINED by VIX bucket
print(f"\n\n{'='*140}")
print("  LONG/SHORT COMBINADO (MR_1w) POR NIVEL DE VIX")
print("=" * 140)
print(f"\n  {'Rango VIX':>25s} | {'Sem':>5s} | {'%Tiempo':>7s} | {'AvgRet%':>7s} | {'CAGR%':>7s} | {'Sharpe':>6s} | {'WR%':>5s} | {'PF':>5s}")
print(f"  {'-'*25} | {'-'*5} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*5} | {'-'*5}")

for lo, hi, label in vix_buckets:
    mask = (rdf['vix'] >= lo) & (rdf['vix'] < hi)
    subset = rdf[mask]['ls_mr1w'].dropna()
    if len(subset) < 5:
        continue
    rets = subset.values
    cum = np.cumprod(1 + rets)
    years = len(rets) / 52
    cagr = (cum[-1] ** (1/years) - 1) * 100 if years > 0.5 else np.nan
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(52) if np.std(rets) > 0 else 0
    wr = np.mean(rets > 0) * 100
    pf = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if (rets < 0).any() and np.sum(rets[rets < 0]) != 0 else 0
    pct_time = len(subset) / len(rdf) * 100
    cagr_str = f"{cagr:>+6.1f}%" if not np.isnan(cagr) else "   N/A "
    marker = " ***" if sharpe > 0.3 else (" **" if sharpe > 0 else "")
    print(f"  {label:>25s} | {len(subset):>5d} | {pct_time:>6.1f}% | {np.mean(rets)*100:>+6.3f}% | {cagr_str} | {sharpe:>6.2f} | {wr:>4.0f}% | {pf:>5.2f}{marker}")

# ============================================================
# Cumulative threshold analysis: "activate shorts when VIX >= X"
# ============================================================
print(f"\n\n{'='*140}")
print("  UMBRAL OPTIMO: Activar shorts cuando VIX >= X")
print("  Pregunta: a partir de que VIX merece la pena shortear?")
print("=" * 140)

thresholds = [12, 14, 15, 16, 17, 18, 19, 20, 22, 25, 28, 30, 35]

print(f"\n  {'Umbral VIX':>10s} | {'Sem':>5s} | {'%Tiempo':>7s} | {'S_AvgRet%':>9s} | {'S_Sharpe':>8s} | {'S_WR%':>6s} | {'L/S_AvgRet%':>11s} | {'L/S_Sharpe':>10s} | {'L/S_WR%':>8s}")
print(f"  {'-'*10} | {'-'*5} | {'-'*7} | {'-'*9} | {'-'*8} | {'-'*6} | {'-'*11} | {'-'*10} | {'-'*8}")

for thresh in thresholds:
    mask = rdf['vix'] >= thresh
    if mask.sum() < 10:
        continue

    s_rets = rdf[mask]['short_mr1w'].dropna().values
    ls_rets = rdf[mask]['ls_mr1w'].dropna().values
    pct_time = mask.sum() / len(rdf) * 100

    s_sharpe = (np.mean(s_rets) / np.std(s_rets)) * np.sqrt(52) if np.std(s_rets) > 0 else 0
    s_wr = np.mean(s_rets > 0) * 100
    ls_sharpe = (np.mean(ls_rets) / np.std(ls_rets)) * np.sqrt(52) if np.std(ls_rets) > 0 else 0
    ls_wr = np.mean(ls_rets > 0) * 100

    marker = " <-- OPTIMO" if s_sharpe > 0.1 and pct_time > 10 else (" ***" if s_sharpe > 0.3 else (" **" if s_sharpe > 0 else ""))
    print(f"  VIX >= {thresh:>2d} | {mask.sum():>5d} | {pct_time:>6.1f}% | {np.mean(s_rets)*100:>+8.3f}% | {s_sharpe:>8.2f} | {s_wr:>5.0f}% | {np.mean(ls_rets)*100:>+10.3f}% | {ls_sharpe:>10.2f} | {ls_wr:>7.0f}%{marker}")

# ============================================================
# VIX 19.09 context
# ============================================================
print(f"\n\n{'='*140}")
print(f"  ESTADO ACTUAL: VIX = 19.09 (viernes 20/02/2026)")
print(f"{'='*140}")

# Percentile
pct_rank = (vix_post2005 < 19.09).mean() * 100
print(f"\n  VIX 19.09 esta en el percentil {pct_rank:.0f}% (historico 2005-2026)")
print(f"  Es decir, el {100-pct_rank:.0f}% del tiempo el VIX esta POR ENCIMA de 19.09")

# What the data says for VIX around 18-20
mask_1820 = (rdf['vix'] >= 18) & (rdf['vix'] < 20)
if mask_1820.sum() > 0:
    s18 = rdf[mask_1820]['short_mr1w'].dropna().values
    l18 = rdf[mask_1820]['long_mr1w'].dropna().values
    ls18 = rdf[mask_1820]['ls_mr1w'].dropna().values
    print(f"\n  Rendimiento historico cuando VIX 18-20 ({mask_1820.sum()} semanas):")
    print(f"    SHORT MR_1w:  avg {np.mean(s18)*100:+.3f}%/sem, Sharpe {(np.mean(s18)/np.std(s18))*np.sqrt(52):.2f}, WR {np.mean(s18>0)*100:.0f}%")
    print(f"    LONG MR_1w:   avg {np.mean(l18)*100:+.3f}%/sem, Sharpe {(np.mean(l18)/np.std(l18))*np.sqrt(52):.2f}, WR {np.mean(l18>0)*100:.0f}%")
    print(f"    L/S MR_1w:    avg {np.mean(ls18)*100:+.3f}%/sem, Sharpe {(np.mean(ls18)/np.std(ls18))*np.sqrt(52):.2f}, WR {np.mean(ls18>0)*100:.0f}%")

# What happens AFTER VIX is at 18-20 level - does it go up or down?
print(f"\n  Que suele pasar DESPUES de VIX 18-20:")
vix_ts = vix.set_index('date')['close'].sort_index()
vix_1820_dates = vix_ts[(vix_ts >= 18) & (vix_ts < 20)].index
next_week_vix = []
for d in vix_1820_dates:
    future = vix_ts[vix_ts.index > d]
    if len(future) >= 5:
        next_week_vix.append(future.iloc[4])  # ~1 week later

if next_week_vix:
    nw = np.array(next_week_vix)
    print(f"    VIX 1 semana despues (avg): {np.mean(nw):.1f}")
    print(f"    VIX sube (>20): {np.mean(nw > 20)*100:.0f}% de las veces")
    print(f"    VIX baja (<18): {np.mean(nw < 18)*100:.0f}% de las veces")
    print(f"    VIX explota (>25): {np.mean(nw > 25)*100:.0f}% de las veces")
    print(f"    VIX explota (>30): {np.mean(nw > 30)*100:.0f}% de las veces")

print(f"\n{'='*140}")
print(f"  RECOMENDACION SISTEMA ADAPTATIVO")
print(f"{'='*140}")
print(f"""
  Basado en el analisis:

  VIX < 15:  SOLO LONG (mercado tranquilo, shorts pierden)
  VIX 15-18: SOLO LONG (normal, shorts en breakeven)
  VIX 18-20: ZONA GRIS - long seguro, short marginal
  VIX 20+:   LONG + SHORT (shorts empiezan a funcionar)
  VIX 25+:   LONG + SHORT AGRESIVO (shorts muy rentables)
  VIX 30+:   SHORT DOMINANTE (crisis, shorts dan 2-3x)

  VIX ACTUAL: 19.09 -> ZONA GRIS (entre 18-20)
""")
