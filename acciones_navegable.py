"""
HTML navegable: Acciones S&P 500 por semana con regimen de mercado.
Ranking por score compuesto, rendimientos Vie->Vie, estadisticas por posicion y regimen.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import build_weekly_events
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
engine = create_engine(FMP_DB)
MAX_CONTRIBUTION = 4.0

REGIME_COLORS = {
    'BURBUJA': '#e91e63', 'GOLDILOCKS': '#4caf50', 'ALCISTA': '#2196f3',
    'NEUTRAL': '#ff9800', 'CAUTIOUS': '#ff5722', 'BEARISH': '#795548',
    'CRISIS': '#9c27b0', 'PANICO': '#f44336', 'CAPITULACION': '#00bcd4',
    'RECOVERY': '#8bc34a',
}

def score_fair(active_events):
    contributions = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP: continue
        for subsec, impact in EVENT_SUBSECTOR_MAP[evt_type]['impacto'].items():
            if subsec not in contributions: contributions[subsec] = []
            contributions[subsec].append(intensity * impact)
    scores = {}
    for sub_id in SUBSECTORS:
        if sub_id not in contributions or len(contributions[sub_id]) == 0:
            scores[sub_id] = 5.0
        else:
            avg = np.mean(contributions[sub_id])
            scores[sub_id] = max(0.0, min(10.0, 5.0 + (avg / MAX_CONTRIBUTION) * 5.0))
    return scores

def adjust_score_by_price(scores, dd_row, rsi_row):
    adjusted = {}
    for sub_id, score in scores.items():
        dd_val = dd_row.get(sub_id, 0) if dd_row is not None else 0
        rsi_val = rsi_row.get(sub_id, 50) if rsi_row is not None else 50
        if not pd.notna(dd_val): dd_val = 0
        if not pd.notna(rsi_val): rsi_val = 50
        if score < 5.0:
            dd_factor = np.clip((abs(dd_val) - 15) / 30, 0, 1)
            rsi_factor = np.clip((35 - rsi_val) / 20, 0, 1)
            oversold = max(dd_factor, rsi_factor)
            adjusted[sub_id] = score + (5.0 - score) * oversold * 0.5
        elif score > 5.0:
            rsi_factor = np.clip((rsi_val - 70) / 15, 0, 1)
            adjusted[sub_id] = score - (score - 5.0) * rsi_factor * 0.5
        else:
            adjusted[sub_id] = score
    return adjusted

# ================================================================
print("Cargando datos base...")
ticker_to_sub = {}
sub_labels = {}
for sub_id, sub_data in SUBSECTORS.items():
    sub_labels[sub_id] = sub_data['label']
    for t in sub_data['tickers']:
        ticker_to_sub[t] = sub_id

# --- Composicion historica S&P 500 ---
import json as _json
with open('data/sp500_constituents.json') as f:
    sp500_current = {s['symbol'] for s in _json.load(f)}
with open('data/sp500_historical_changes.json') as f:
    sp500_changes = _json.load(f)

# Construir set de miembros del S&P 500 por fecha
# Partimos del presente y retrocedemos deshaciendo cambios
sp500_changes_sorted = sorted(sp500_changes, key=lambda c: c.get('date', c.get('dateAdded', '')), reverse=True)
members_snapshots = {}  # date_str -> set of tickers justo ANTES del cambio

current_members = set(sp500_current)
for chg in sp500_changes_sorted:
    chg_date = chg.get('date', chg.get('dateAdded', ''))
    added = chg.get('symbol', '')
    removed = chg.get('removedTicker', '')
    # Deshacer: quitar el que fue añadido, poner el que fue quitado
    if added and added in current_members:
        current_members.discard(added)
    if removed:
        current_members.add(removed)
    members_snapshots[chg_date] = set(current_members)

def get_sp500_members(date_str):
    """Devuelve los miembros del S&P 500 en una fecha dada."""
    # Buscar el snapshot mas cercano posterior o igual
    relevant_dates = sorted(members_snapshots.keys())
    members = set(sp500_current)  # default: actual
    for d in reversed(relevant_dates):
        if d <= date_str:
            members = members_snapshots[d]
            break
    return members

print(f"  Cambios historicos: {len(sp500_changes)}")
print(f"  Miembros actuales: {len(sp500_current)}")
# Test: verificar composicion en distintas epocas
for test_date in ['2001-01-01', '2005-01-01', '2010-01-01', '2020-01-01', '2026-01-01']:
    m = get_sp500_members(test_date)
    print(f"    SP500 en {test_date}: {len(m)} miembros")

# Recopilar TODOS los tickers historicos que necesitamos
all_historical_tickers = set()
for d, members in members_snapshots.items():
    all_historical_tickers |= members
all_historical_tickers |= sp500_current
# Añadir subsector mapping para tickers que ya tenemos
# Para historicos sin subsector, asignaremos 'other'
print(f"  Tickers historicos totales: {len(all_historical_tickers)}")

# Tickers a cargar = todos los que tenemos en subsectors + historicos
all_tickers_to_load = sorted(all_historical_tickers)
ticker_to_idx = {t: i for i, t in enumerate(all_tickers_to_load)}
print(f"  Tickers a cargar: {len(all_tickers_to_load)}")

# Cargar en lotes
batch_size = 100
frames = []
for i in range(0, len(all_tickers_to_load), batch_size):
    batch = all_tickers_to_load[i:i+batch_size]
    tlist = "','".join(batch)
    df_batch = pd.read_sql(f"""
        SELECT symbol, date, open, close, high, low
        FROM fmp_price_history WHERE symbol IN ('{tlist}')
        AND date BETWEEN '2000-01-01' AND '2026-12-31' ORDER BY symbol, date
    """, engine)
    frames.append(df_batch)
    if (i // batch_size + 1) % 5 == 0:
        print(f"    Lote {i//batch_size+1}: {sum(len(f) for f in frames)} registros acumulados")
df_all = pd.concat(frames, ignore_index=True)
df_all['date'] = pd.to_datetime(df_all['date'])
# Optimizar tipos de datos para reducir memoria
df_all['open'] = df_all['open'].astype('float32')
df_all['close'] = df_all['close'].astype('float32')
df_all['high'] = df_all['high'].astype('float32')
df_all['low'] = df_all['low'].astype('float32')
df_all['symbol'] = df_all['symbol'].astype('category')
# Subsector: usar mapa existente, 'other' para historicos sin mapa
df_all['subsector'] = df_all['symbol'].map(lambda s: ticker_to_sub.get(s, 'other')).astype('category')
loaded_symbols = set(df_all['symbol'].unique())
print(f"  Total precios: {len(df_all)} registros ({len(loaded_symbols)} simbolos con datos)")

# Weekly aggregation por symbol (sin Grouper para evitar memory error)
print("  Agregando semanal...")
df_all['week_fri'] = df_all['date'].dt.to_period('W-FRI').dt.end_time.dt.normalize()
df_weekly = df_all.sort_values('date').groupby(
    ['symbol', 'week_fri'], observed=True)[['open', 'close', 'high', 'low']].last().reset_index()
df_weekly = df_weekly.rename(columns={'week_fri': 'date'})
df_weekly['symbol'] = df_weekly['symbol'].astype(str)
df_weekly = df_weekly.dropna(subset=['close']).sort_values(['symbol', 'date'])
df_weekly['prev_close'] = df_weekly.groupby('symbol')['close'].shift(1)
df_weekly['return'] = df_weekly['close'] / df_weekly['prev_close'] - 1
df_weekly = df_weekly.dropna(subset=['return'])

def calc_stock_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['high'].rolling(52, min_periods=26).max()
    g['dd_52w'] = (g['close'] / g['high_52w'] - 1) * 100
    delta = g['close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    g['mom_12w'] = g['close'].pct_change(12) * 100
    return g

print("Calculando metricas acciones...")
df_weekly = df_weekly.groupby('symbol', group_keys=False).apply(calc_stock_metrics)

# Fri open -> Fri open returns per stock
# Usamos df_weekly que ya tiene open del ultimo dia de la semana (viernes)
print("Calculando retornos acciones (Fri open -> Fri open)...")
stock_fri = df_weekly[['symbol', 'date', 'open']].dropna(subset=['open']).copy()
stock_fri = stock_fri.rename(columns={'open': 'open_fri'})
stock_fri = stock_fri.sort_values(['symbol', 'date'])
stock_fri['next_open'] = stock_fri.groupby('symbol')['open_fri'].shift(-1)
stock_fri['ret_pct'] = (stock_fri['next_open'] / stock_fri['open_fri'] - 1) * 100
# Detectar tickers con datos corruptos (saltos >10x entre semanas consecutivas)
stock_fri['ratio'] = stock_fri['next_open'] / stock_fri['open_fri']
bad_tickers = set(stock_fri[(stock_fri['ratio'] > 10) | (stock_fri['ratio'] < 0.1)]['symbol'].unique())
if bad_tickers:
    print(f"  Tickers con datos corruptos excluidos ({len(bad_tickers)}): {sorted(bad_tickers)[:20]}...")
    stock_fri.loc[stock_fri['symbol'].isin(bad_tickers), 'ret_pct'] = np.nan
# Capear retornos restantes a ±100%
n_capped = ((stock_fri['ret_pct'] > 100) | (stock_fri['ret_pct'] < -100)).sum()
stock_fri['ret_pct'] = stock_fri['ret_pct'].clip(-100, 100)
if n_capped > 0:
    print(f"  Retornos capeados a ±100%: {n_capped}")
stock_ret_wide = stock_fri.pivot(index='date', columns='symbol', values='ret_pct')
stock_open_wide = stock_fri.pivot(index='date', columns='symbol', values='open_fri')
stock_next_open_wide = stock_fri.pivot(index='date', columns='symbol', values='next_open')
print(f"  Stock returns: {stock_ret_wide.shape}")

# Subsector metrics for FV scoring (excluir 'other')
# Calcular ANTES de liberar df_all
df_all_known = df_all[df_all['subsector'] != 'other'].copy()
del df_all
import gc; gc.collect()
print("  Memoria liberada (df_all)")
df_all_known['week_fri'] = df_all_known['date'].dt.to_period('W-FRI').dt.end_time.dt.normalize()
df_all_known['subsector'] = df_all_known['subsector'].astype(str)
sub_agg = df_all_known.sort_values('date').groupby(
    ['subsector', 'week_fri'], observed=True).agg(
    avg_close=('close', 'mean'), avg_high=('high', 'mean'),
    avg_low=('low', 'mean')).reset_index().dropna(subset=['avg_close'])
sub_agg = sub_agg.rename(columns={'week_fri': 'date'})
sub_agg = sub_agg.sort_values(['subsector', 'date'])

def calc_sub_metrics(g):
    g = g.sort_values('date').copy()
    g['high_52w'] = g['avg_high'].rolling(52, min_periods=26).max()
    g['drawdown_52w'] = (g['avg_close'] / g['high_52w'] - 1) * 100
    delta = g['avg_close'].diff()
    gain = delta.where(delta > 0, 0); loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14w'] = 100 - (100 / (1 + rs))
    return g

print("Calculando metricas subsectores (para FV)...")
sub_agg = sub_agg.groupby('subsector', group_keys=False).apply(calc_sub_metrics)
dd_wide = sub_agg.pivot(index='date', columns='subsector', values='drawdown_52w')
rsi_wide = sub_agg.pivot(index='date', columns='subsector', values='rsi_14w')

weekly_events = build_weekly_events('2000-01-01', '2026-12-31')
reg_df = pd.read_csv('data/regimenes_historico.csv')
reg_df['fecha_senal'] = pd.to_datetime(reg_df['fecha_senal'])

# ================================================================
print("Generando datos por semana...")
all_weeks = []
prev_ranks = {}

for idx, (_, reg_row) in enumerate(reg_df.iterrows()):
    signal_date = reg_row['fecha_senal']
    if signal_date not in dd_wide.index:
        continue

    regime = reg_row['regime']
    year = int(reg_row['year'])
    sem = int(reg_row['sem'])

    # FV scores
    if signal_date in weekly_events.index:
        evt_date = signal_date
    else:
        nearest_idx = weekly_events.index.get_indexer([signal_date], method='nearest')[0]
        evt_date = weekly_events.index[nearest_idx]
    events_row = weekly_events.loc[evt_date]
    active = {col: float(events_row[col]) for col in events_row.index if events_row[col] > 0}

    scores_evt = score_fair(active)
    prev_dates = dd_wide.index[dd_wide.index < signal_date]
    dd_row_prev = dd_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    rsi_row_prev = rsi_wide.loc[prev_dates[-1]] if len(prev_dates) > 0 else None
    scores_adj = adjust_score_by_price(scores_evt, dd_row_prev, rsi_row_prev)

    # Latest stock data: solo acciones que cotizaron en la semana de senal
    latest = df_weekly[df_weekly['date'] == signal_date].copy()

    # Filtrar por composicion historica del S&P 500
    sp500_at_date = get_sp500_members(signal_date.strftime('%Y-%m-%d'))
    latest = latest[latest['symbol'].isin(sp500_at_date & loaded_symbols)]

    # Returns and prices
    ret_row = stock_ret_wide.loc[signal_date] if signal_date in stock_ret_wide.index else None
    open_row = stock_open_wide.loc[signal_date] if signal_date in stock_open_wide.index else None
    nopen_row = stock_next_open_wide.loc[signal_date] if signal_date in stock_next_open_wide.index else None

    # Score ALL stocks
    stocks = []
    for _, stk in latest.iterrows():
        sym = stk['symbol']
        if sym not in ticker_to_idx: continue
        if sym in bad_tickers: continue  # excluir tickers con datos corruptos
        sub = ticker_to_sub.get(sym)
        if sub is None: sub = 'other'

        fv = scores_adj.get(sub, 5.0)
        mom = float(stk['mom_12w']) if pd.notna(stk.get('mom_12w')) else 0
        rsi = float(stk['rsi_14w']) if pd.notna(stk.get('rsi_14w')) else 50
        dd = float(stk['dd_52w']) if pd.notna(stk.get('dd_52w')) else 0

        # Precios open viernes
        o_fri = None
        if open_row is not None and sym in open_row.index and pd.notna(open_row[sym]):
            o_fri = round(float(open_row[sym]), 2)
        n_fri = None
        if nopen_row is not None and sym in nopen_row.index and pd.notna(nopen_row[sym]):
            n_fri = round(float(nopen_row[sym]), 2)
        if o_fri is None or o_fri <= 0: continue  # excluir acciones sin cotización válida

        composite = (np.clip((fv - 5.0) / 4.0, 0, 1) * 3.0 +
                     np.clip((mom + 20) / 60, 0, 1) * 3.0 +
                     np.clip((rsi - 30) / 50, 0, 1) * 2.0 +
                     np.clip((dd + 30) / 30, 0, 1) * 2.0)

        ret = None
        if ret_row is not None and sym in ret_row.index and pd.notna(ret_row[sym]):
            ret = round(float(ret_row[sym]), 2)

        tidx = ticker_to_idx[sym]
        stocks.append([tidx, round(composite, 1), round(fv, 1),
                       int(round(mom)), int(round(rsi)), int(round(dd)),
                       o_fri, n_fri, ret])

    stocks.sort(key=lambda x: -x[1])

    # Rank changes
    cur_ranks = {}
    chg = {}
    for rank, s in enumerate(stocks, 1):
        tidx = s[0]
        cur_ranks[tidx] = rank
        if tidx in prev_ranks:
            chg[str(tidx)] = prev_ranks[tidx] - rank
    prev_ranks = cur_ranks

    spy_ret = round(float(reg_row['spy_ret_pct']), 2) if pd.notna(reg_row['spy_ret_pct']) else None

    all_weeks.append({
        'd': signal_date.strftime('%Y-%m-%d'),
        'y': year, 'w': sem,
        'r': regime, 't': round(float(reg_row['total']), 1),
        'sp': int(round(float(reg_row['spy_close']))),
        'v': int(round(float(reg_row['vix']))),
        'sr': spy_ret,
        's': stocks,
        'c': chg,
    })

    if (idx + 1) % 200 == 0:
        print(f"  Procesadas {idx + 1} semanas...")

print(f"  Total semanas: {len(all_weeks)}")

# ================================================================
# Estadisticas agrupadas por decenas (top 3 + bottom 3)
# ================================================================
print("Calculando estadisticas por grupos de 10...")
group_data = {}  # regime -> group_label -> [rets]

for week in all_weeks:
    regime = week['r']
    n = len(week['s'])
    if n == 0: continue
    if regime not in group_data:
        group_data[regime] = {}

    for rank_idx, stk in enumerate(week['s']):
        ret = stk[8]
        if ret is None: continue
        pos = rank_idx + 1  # 1-based

        # Top groups: 1-10, 11-20, 21-30
        # Bottom groups: last 30 split in 3 groups of 10
        if pos <= 10:
            grp = '1-10'
        elif pos <= 20:
            grp = '11-20'
        elif pos <= 30:
            grp = '21-30'
        elif pos > n - 10:
            grp = 'Bot 10'
        elif pos > n - 20:
            grp = 'Bot 20'
        elif pos > n - 30:
            grp = 'Bot 30'
        else:
            grp = '31-resto'

        if grp not in group_data[regime]:
            group_data[regime][grp] = []
        group_data[regime][grp].append(ret)

def summarize_groups(data_dict):
    out = {}
    for regime in data_dict:
        out[regime] = {}
        for grp, rets in data_dict[regime].items():
            if rets:
                out[regime][grp] = {
                    'a': round(float(np.mean(rets)), 3),
                    'n': len(rets),
                    'w': round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                }
    return out

group_stats = summarize_groups(group_data)
print(f"  Regimenes: {list(group_stats.keys())}")
for regime in ['ALCISTA', 'NEUTRAL', 'BEARISH']:
    if regime in group_stats:
        print(f"  {regime}: {list(group_stats[regime].keys())}")

# Rank stats: posiciones individuales 1-20
print("Calculando estadisticas por posicion individual (top 20)...")
rank_data = {}
for week in all_weeks:
    regime = week['r']
    if regime not in rank_data:
        rank_data[regime] = {}
    for rank_idx, stk in enumerate(week['s'][:20]):
        ret = stk[8]
        if ret is None: continue
        pos = rank_idx + 1
        if pos not in rank_data[regime]:
            rank_data[regime][pos] = []
        rank_data[regime][pos].append(ret)

rank_stats = {}
for regime in rank_data:
    rank_stats[regime] = {}
    for pos, rets in rank_data[regime].items():
        if rets:
            rank_stats[regime][str(pos)] = {
                'a': round(float(np.mean(rets)), 3),
                'n': len(rets),
                'w': round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
            }

# ================================================================
# Estadisticas anuales (SPY) por año y por régimen dentro de cada año
# ================================================================
print("Calculando estadisticas anuales...")
yearly_stats = {}  # year -> {total: {ret, n, wr}, regimes: {regime: {ret, n, wr}}}

from collections import defaultdict
year_weeks = defaultdict(list)  # year -> [(spy_ret, regime)]
for week in all_weeks:
    if week['sr'] is not None:
        year_weeks[week['y']].append((week['sr'], week['r']))

for year in sorted(year_weeks.keys()):
    entries = year_weeks[year]
    rets_all = [e[0] for e in entries]
    # Retorno compuesto anual
    compound = 1.0
    for r in rets_all:
        compound *= (1 + r / 100)
    yearly_stats[year] = {
        'total': {
            'r': round((compound - 1) * 100, 2),
            'n': len(rets_all),
            'a': round(float(np.mean(rets_all)), 3),
            'w': round(sum(1 for r in rets_all if r > 0) / len(rets_all) * 100, 1),
        },
        'regimes': {}
    }
    # Por regimen dentro del año
    reg_rets = defaultdict(list)
    for ret, reg in entries:
        reg_rets[reg].append(ret)
    for reg in reg_rets:
        rr = reg_rets[reg]
        comp_r = 1.0
        for r in rr:
            comp_r *= (1 + r / 100)
        yearly_stats[year]['regimes'][reg] = {
            'r': round((comp_r - 1) * 100, 2),
            'n': len(rr),
            'a': round(float(np.mean(rr)), 3),
            'w': round(sum(1 for r in rr if r > 0) / len(rr) * 100, 1),
        }

print(f"  Años: {min(yearly_stats.keys())}-{max(yearly_stats.keys())}")

# ================================================================
# HTML
# ================================================================
print("Generando HTML...")

ticker_info = [{'t': t, 's': sub_labels.get(ticker_to_sub.get(t, ''), '')} for t in all_tickers_to_load]
tickers_json = json.dumps(ticker_info)
weeks_json = json.dumps(all_weeks)
rank_json = json.dumps(rank_stats)
group_json = json.dumps(group_stats)
rc_json = json.dumps(REGIME_COLORS)
yearly_json = json.dumps(yearly_stats)

print(f"  weeks JSON: {len(weeks_json)/1024/1024:.1f} MB")

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>Acciones S&P 500 - Navegable</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #fff; color: #222; margin: 20px; }}
h1 {{ color: #1565c0; text-align: center; margin-bottom: 5px; }}
h2 {{ color: #333; margin-top: 20px; margin-bottom: 8px; border-bottom: 2px solid #1565c0; padding-bottom: 4px; }}
.sub {{ text-align: center; color: #666; margin-bottom: 12px; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 11px; }}
th {{ background: #1565c0; color: #fff; padding: 4px 3px; text-align: center; border: 1px solid #ccc; cursor: pointer; font-size: 10px; }}
th:hover {{ background: #0d47a1; }}
td {{ padding: 3px 4px; text-align: center; border: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #f5f7fa; }}
tr:hover {{ background: #e3f2fd; }}
.pos {{ color: #2e7d32; font-weight: bold; }}
.neg {{ color: #c62828; font-weight: bold; }}
.neutral {{ color: #999; }}
td.left {{ text-align: left; }}
.sbox {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }}
.sc {{ background: #f5f7fa; border: 1px solid #ddd; border-radius: 8px; padding: 8px; flex: 1; min-width: 90px; text-align: center; }}
.sc h4 {{ margin: 0 0 3px 0; color: #1565c0; font-size: 10px; }}
.sc .v {{ font-size: 17px; font-weight: bold; }}
.rb {{ display: inline-block; padding: 4px 16px; border-radius: 5px; font-size: 16px; font-weight: bold; color: #fff; }}
.note {{ background: #fffde7; padding: 7px 10px; border-radius: 6px; border-left: 4px solid #ffd600; margin-bottom: 10px; font-size: 11px; }}
.nb {{ text-align: center; margin: 12px 0 16px 0; padding: 12px; background: #f5f7fa; border-radius: 8px; border: 1px solid #ddd; }}
.nb select {{ font-size: 13px; padding: 5px 10px; border-radius: 4px; border: 1px solid #ccc; margin: 0 6px; }}
.nb button {{ font-size: 12px; padding: 5px 14px; border-radius: 4px; border: 1px solid #1565c0; background: #1565c0; color: #fff; cursor: pointer; margin: 0 3px; }}
.nb button:hover {{ background: #0d47a1; }}
.yb {{ margin-top: 6px; }}
.yb button {{ font-size: 10px; padding: 2px 7px; border-radius: 3px; border: 1px solid #999; background: #fff; color: #333; cursor: pointer; margin: 1px; }}
.yb button:hover, .yb button.active {{ background: #1565c0; color: #fff; border-color: #1565c0; }}
.retb {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-left: 8px; }}
.cu {{ color: #2e7d32; font-weight: bold; font-size: 10px; }}
.cd {{ color: #c62828; font-weight: bold; font-size: 10px; }}
.cs {{ color: #bbb; font-size: 10px; }}
.fh {{ background: #e8f5e9; }}
.fl {{ background: #ffebee; }}
.st {{ max-height: 600px; overflow-y: auto; border: 1px solid #ccc; }}
.st table {{ margin-bottom: 0; }}
.st th {{ position: sticky; top: 0; z-index: 1; }}
.qt {{ width: auto !important; margin: 0 auto !important; min-width: 500px; }}
.qt td {{ padding: 5px 8px; font-size: 12px; }}
.qt th {{ font-size: 10px; }}
#ss {{ padding: 4px 8px; font-size: 11px; border: 1px solid #ccc; border-radius: 3px; width: 160px; margin-right: 8px; }}
</style></head><body>
<h1>Acciones S&P 500 - Navegador Semanal</h1>
<p class="sub">Ranking por score compuesto (FV + Mom + RSI + DD) | Trading: viernes open &rarr; viernes open</p>
<div class="nb">
<button onclick="pW()">&larr;</button>
<select id="ws" onchange="lW()"></select>
<button onclick="nW()">&rarr;</button>
<div class="yb" id="yb"></div>
<div class="yb" id="rb" style="margin-top:4px;"></div>
</div>
<div id="ct"></div>
<div id="yt"></div>
<script>
const T={tickers_json};
const W={weeks_json};
const RC={rc_json};
const RS={rank_json};
const YS={yearly_json};
const GS={group_json};
const sel=document.getElementById('ws');
W.forEach((w,i)=>{{const o=document.createElement('option');o.value=i;const r=w.sr!==null?(w.sr>=0?'+':'')+w.sr.toFixed(2)+'%':'-';o.text=w.y+' S'+w.w+' ('+w.d+') '+w.r+' '+r;sel.appendChild(o);}});
const yrs=[...new Set(W.map(w=>w.y))].sort();
const yb=document.getElementById('yb');
yrs.forEach(y=>{{const b=document.createElement('button');b.textContent=y;b.onclick=()=>fY(y);yb.appendChild(b);}});
const rb=document.getElementById('rb');
const RO=['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION'];
const ab=document.createElement('button');ab.textContent='TODOS';ab.className='active';ab.onclick=()=>rF();rb.appendChild(ab);
RO.forEach(r=>{{const b=document.createElement('button');b.style.background=RC[r]||'#666';b.style.color='#fff';b.style.borderColor=RC[r]||'#666';b.textContent=r;b.onclick=()=>fR(r);rb.appendChild(b);}});
let aF=null,fI=W.map((_,i)=>i);
function apF(){{sel.innerHTML='';fI.forEach(i=>{{const w=W[i];const o=document.createElement('option');o.value=i;const r=w.sr!==null?(w.sr>=0?'+':'')+w.sr.toFixed(2)+'%':'-';o.text=w.y+' S'+w.w+' ('+w.d+') '+w.r+' '+r;sel.appendChild(o);}});if(fI.length>0){{sel.value=fI[fI.length-1];lW();}}
document.querySelectorAll('#yb button').forEach(b=>{{b.classList.toggle('active',aF&&aF.t==='y'&&parseInt(b.textContent)===aF.v);}});
document.querySelectorAll('#rb button').forEach(b=>{{if(b.textContent==='TODOS')b.classList.toggle('active',!aF);else b.classList.toggle('active',aF&&aF.t==='r'&&b.textContent===aF.v);}});}}
function fY(y){{aF={{t:'y',v:y}};fI=W.map((w,i)=>w.y===y?i:-1).filter(i=>i>=0);apF();}}
function fR(r){{aF={{t:'r',v:r}};fI=W.map((w,i)=>w.r===r?i:-1).filter(i=>i>=0);apF();}}
function rF(){{aF=null;fI=W.map((_,i)=>i);apF();}}
sel.value=W.length-1;lW();
function pW(){{const c=fI.indexOf(parseInt(sel.value));if(c>0){{sel.value=fI[c-1];lW();}}}}
function nW(){{const c=fI.indexOf(parseInt(sel.value));if(c<fI.length-1){{sel.value=fI[c+1];lW();}}}}
function vc(v){{return v>0?'pos':v<0?'neg':'neutral';}}
function fm(v,d){{d=d||1;return(v>=0?'+':'')+v.toFixed(d);}}
function lW(){{
const w=W[sel.value];const rc=RC[w.r]||'#666';
const rh=w.sr!==null?'<span class="retb" style="background:'+(w.sr>=0?'#e8f5e9;color:#2e7d32':'#ffebee;color:#c62828')+';">SPY: '+fm(w.sr,2)+'%</span>':'<span class="retb" style="background:#f5f5f5;color:#999;">SPY: -</span>';
let h='<div style="text-align:center;margin:10px 0;"><span class="rb" style="background:'+rc+';">'+w.r+'</span> <span style="font-size:16px;margin-left:10px;font-weight:bold;">Score: '+fm(w.t)+'</span>'+rh+'</div>';
h+='<div class="sbox"><div class="sc"><h4>SPY</h4><div class="v">$'+w.sp+'</div></div><div class="sc"><h4>VIX</h4><div class="v">'+w.v+'</div></div><div class="sc"><h4>Acciones</h4><div class="v">'+w.s.length+'</div></div></div>';
h+='<h2>Ranking Acciones ('+w.s.length+')</h2>';
h+='<div style="margin-bottom:6px;"><input type="text" id="ss" placeholder="Buscar ticker..." onkeyup="fS()"></div>';
h+='<div class="st"><table id="sT"><tr><th onclick="sC(0)">#</th><th>Chg</th><th onclick="sC(2)">Ticker</th><th onclick="sC(3)">Subsector</th><th onclick="sC(4)">Score</th><th onclick="sC(5)">FV</th><th onclick="sC(6)">Mom</th><th onclick="sC(7)">RSI</th><th onclick="sC(8)">DD</th><th onclick="sC(9)">Open Fri</th><th onclick="sC(10)">Open Fri+1</th><th onclick="sC(11)">Ret%</th></tr>';
w.s.forEach((s,i)=>{{
const ti=T[s[0]];const fc=s[2]>=5.5?' class="fh"':s[2]<4.5?' class="fl"':'';
const rt=s[8]!==null?'<span class="'+vc(s[8])+'">'+fm(s[8],2)+'%</span>':'<span class="neutral">-</span>';
const op=s[6]!==null?'$'+s[6].toFixed(2):'<span class="neutral">-</span>';
const np=s[7]!==null?'$'+s[7].toFixed(2):'<span class="neutral">-</span>';
let cg='<span class="cs">-</span>';const cv=w.c[String(s[0])];
if(cv!==undefined){{if(cv>0)cg='<span class="cu">&#x25B2;'+cv+'</span>';else if(cv<0)cg='<span class="cd">&#x25BC;'+Math.abs(cv)+'</span>';else cg='<span class="cs">=</span>';}}
h+='<tr'+fc+' data-t="'+ti.t.toLowerCase()+'" data-r="'+(s[8]!==null?s[8]:-999)+'"><td>'+(i+1)+'</td><td>'+cg+'</td><td><b>'+ti.t+'</b></td><td class="left" style="font-size:10px;">'+ti.s+'</td><td><b>'+s[1]+'</b></td><td class="'+vc(s[2]-5)+'">'+s[2]+'</td><td class="'+vc(s[3])+'">'+fm(s[3],0)+'%</td><td>'+s[4]+'</td><td class="'+vc(s[5])+'">'+fm(s[5],0)+'%</td><td>'+op+'</td><td>'+np+'</td><td>'+rt+'</td></tr>';
}});
h+='</table></div>';
// Group stats (groups of 10)
h+='<h2>Rendimiento por Grupo de Posicion ('+w.r+')</h2>';
const gs=GS[w.r];
if(gs){{
h+='<div class="note">Grupos por posicion en ranking (score compuesto). Top 3 grupos de 10 + medio + bottom 3 grupos de 10. Historico de todas las semanas en regimen '+w.r+'.</div>';
h+='<table class="qt"><tr><th>Grupo</th><th>N</th><th>Ret Medio</th><th>Win Rate</th></tr>';
const gl=['1-10','11-20','21-30','31-resto','Bot 30','Bot 20','Bot 10'];
const gn=['Top 1-10','Top 11-20','Top 21-30','31-Resto','Bot 30','Bot 20','Bot 10'];
for(let g=0;g<gl.length;g++){{const gd=gs[gl[g]];if(gd){{const wc=gd.w>=55?'pos':gd.w<48?'neg':'neutral';const bg=g<=2?' style="background:#e8f5e9;"':g>=4?' style="background:#ffebee;"':'';h+='<tr'+bg+'><td class="left"><b>'+gn[g]+'</b></td><td>'+gd.n+'</td><td class="'+vc(gd.a)+'">'+fm(gd.a,3)+'%</td><td class="'+wc+'">'+gd.w.toFixed(1)+'%</td></tr>';}}}}
h+='</table>';}}
// Group comparison across regimes
h+='<details><summary style="cursor:pointer;color:#1565c0;font-weight:bold;margin:6px 0;">Comparativa grupos - Todos los regimenes</summary>';
h+='<table class="qt"><tr><th>Grupo</th>';
const ar=RO.filter(r=>GS[r]);
ar.forEach(r=>{{h+='<th style="background:'+(RC[r]||'#666')+';font-size:9px;">'+r+'</th>';}});h+='</tr>';
const gl2=['1-10','11-20','21-30','31-resto','Bot 30','Bot 20','Bot 10'];
const gn2=['1-10','11-20','21-30','Medio','B30','B20','B10'];
for(let g=0;g<gl2.length;g++){{h+='<tr><td><b>'+gn2[g]+'</b></td>';ar.forEach(r=>{{const gd=GS[r]&&GS[r][gl2[g]];if(gd)h+='<td class="'+vc(gd.a)+'" title="N='+gd.n+' WR='+gd.w+'%">'+fm(gd.a,3)+'%</td>';else h+='<td class="neutral">-</td>';}});h+='</tr>';}}
h+='</table></details>';
// Top 20 positions
h+='<details><summary style="cursor:pointer;color:#1565c0;font-weight:bold;margin:6px 0;">Posiciones 1-20 detalle ('+w.r+')</summary>';
const rs=RS[w.r];
if(rs){{h+='<table class="qt"><tr><th>Pos</th><th>N</th><th>Ret Medio</th><th>Win Rate</th></tr>';
for(let p=1;p<=20;p++){{const pd=rs[String(p)];if(pd){{const wc=pd.w>=55?'pos':pd.w<48?'neg':'neutral';h+='<tr><td><b>'+p+'</b></td><td>'+pd.n+'</td><td class="'+vc(pd.a)+'">'+fm(pd.a,3)+'%</td><td class="'+wc+'">'+pd.w.toFixed(1)+'%</td></tr>';}}}}
h+='</table>';}}
h+='</details>';
document.getElementById('ct').innerHTML=h;
}}
function fS(){{const q=document.getElementById('ss').value.toLowerCase();document.querySelectorAll('#sT tr[data-t]').forEach(r=>{{r.style.display=q===''||r.dataset.t.includes(q)?'':'none';}});}}
let sD={{}};
function sC(c){{const t=document.getElementById('sT');if(!t)return;const rows=Array.from(t.querySelectorAll('tr[data-t]'));const d=sD[c]=!(sD[c]||false);rows.sort((a,b)=>{{let va=a.cells[c].textContent.replace(/[\\$%+=\\-\\u25B2\\u25BC]/g,'').trim();let vb=b.cells[c].textContent.replace(/[\\$%+=\\-\\u25B2\\u25BC]/g,'').trim();let na=parseFloat(va),nb=parseFloat(vb);if(!isNaN(na)&&!isNaN(nb))return d?na-nb:nb-na;return d?va.localeCompare(vb):vb.localeCompare(va);}});rows.forEach((r,i)=>{{r.cells[0].textContent=i+1;t.appendChild(r);}});}}
// Yearly stats table
(function(){{
const RO=['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION'];
const yrs=Object.keys(YS).sort();
let h='<h2>Rendimiento Anual SPY (compuesto semanal)</h2>';
h+='<div class="note">Retorno compuesto de las semanas de cada a&ntilde;o (Fri open &rarr; Fri open). Desglose por r&eacute;gimen dentro de cada a&ntilde;o.</div>';
h+='<table class="qt"><tr><th>A&ntilde;o</th><th>Sem</th><th>Ret Compuesto</th><th>Ret Medio/Sem</th><th>Win Rate</th></tr>';
yrs.forEach(y=>{{
const d=YS[y].total;
const bg=d.r>=0?'background:#e8f5e9;':'background:#ffebee;';
h+='<tr style="'+bg+'font-weight:bold;"><td>'+y+'</td><td>'+d.n+'</td><td class="'+vc(d.r)+'">'+fm(d.r,2)+'%</td><td class="'+vc(d.a)+'">'+fm(d.a,3)+'%</td><td>'+(d.w>=55?'<span class="pos">':d.w<48?'<span class="neg">':'<span>')+d.w.toFixed(1)+'%</span></td></tr>';
const regs=YS[y].regimes;
RO.forEach(r=>{{if(regs[r]){{const rd=regs[r];h+='<tr><td style="padding-left:20px;color:'+(RC[r]||'#666')+';font-size:11px;">'+r+'</td><td style="font-size:11px;">'+rd.n+'</td><td style="font-size:11px;" class="'+vc(rd.r)+'">'+fm(rd.r,2)+'%</td><td style="font-size:11px;" class="'+vc(rd.a)+'">'+fm(rd.a,3)+'%</td><td style="font-size:11px;">'+(rd.w>=55?'<span class="pos">':rd.w<48?'<span class="neg">':'<span>')+rd.w.toFixed(1)+'%</span></td></tr>';}}}});
}});
h+='</table>';
document.getElementById('yt').innerHTML=h;
}})();
</script></body></html>"""

with open('acciones_navegable.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"OK -> acciones_navegable.html ({len(all_weeks)} semanas, {len(html)/1024/1024:.1f} MB)")
