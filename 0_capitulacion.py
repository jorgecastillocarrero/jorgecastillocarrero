"""
0_capitulacion.py - Backtest estrategia CAPITULACION
=====================================================
Senal: Jueves close detecta CAPITULACION (PANICO + VIX bajando)
Trade: Viernes open -> Viernes open siguiente

Estrategias:
  A) BASELINE: Short 5 resistentes + Long 5 oversold (puro drawdown)
  B) SECTOR:   Short 5 defensivas resistentes + Long 5 ciclicas oversold + calidad
     - SHORT: defensivas (Staples, Utilities, Healthcare) con menor drawdown
     - LONG: ciclicas (Banks, Tech, Energy...) con mayor drawdown + EPS positivo + beat/ROE
     - Si la semana resulta ser PANICO/CRISIS, las perdidas son minimas vs benchmark
       porque los shorts en defensivas compensan los longs en ciclicas
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
from collections import Counter

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# ============================================================
# 1. CARGAR DATOS BASE
# ============================================================
print("Cargando datos...")
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
tickers = [s['symbol'] for s in sp500]

profiles = pd.read_sql(
    "SELECT symbol, industry, sector FROM fmp_profiles WHERE symbol IN ('"
    + "','".join(tickers) + "')", engine)
ticker_to_sector = dict(zip(profiles['symbol'], profiles['sector']))
ticker_to_ind = dict(zip(profiles['symbol'], profiles['industry']))
all_tickers = profiles['symbol'].tolist()
tlist = "','".join(all_tickers)

# Clasificacion sectores
DEFENSIVE = {'Consumer Defensive', 'Utilities', 'Healthcare'}
CYCLICAL = {'Financial Services', 'Consumer Cyclical', 'Technology',
            'Energy', 'Basic Materials', 'Industrials', 'Communication Services'}

# Fechas CAPITULACION jueves
df_thu = pd.read_csv('data/regimenes_jueves.csv')
df_thu['fecha_senal'] = pd.to_datetime(df_thu['fecha_senal'])
cap_rows = df_thu[df_thu['regime'] == 'CAPITULACION'].copy()
cap_dates = cap_rows['fecha_senal'].tolist()
print(f'Semanas CAPITULACION: {len(cap_dates)}')

# SPY trading dates
spy = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '2000-01-01' AND '2026-02-28' ORDER BY date
""", engine)
spy['date'] = pd.to_datetime(spy['date'])
spy = spy.set_index('date').sort_index()
spy_dates = set(spy.index.tolist())

# ============================================================
# 2. PRECARGAR DATOS FUNDAMENTALES
# ============================================================
print("Cargando earnings...")
all_earnings = pd.read_sql(f"""
    SELECT symbol, date, eps_actual, eps_estimated FROM fmp_earnings
    WHERE symbol IN ('{tlist}') ORDER BY symbol, date
""", engine)
all_earnings['date'] = pd.to_datetime(all_earnings['date'])

# Indexar por symbol para busquedas rapidas
earnings_idx = {}
for sym, group in all_earnings.groupby('symbol'):
    earnings_idx[sym] = group.sort_values('date').reset_index(drop=True)

print("Cargando key_metrics (ROE)...")
all_metrics = pd.read_sql(f"""
    SELECT symbol, date, roe FROM fmp_key_metrics
    WHERE symbol IN ('{tlist}') ORDER BY symbol, date
""", engine)
all_metrics['date'] = pd.to_datetime(all_metrics['date'])

metrics_idx = {}
for sym, group in all_metrics.groupby('symbol'):
    metrics_idx[sym] = group.sort_values('date').reset_index(drop=True)

print(f"  Earnings: {len(all_earnings):,} registros, {len(earnings_idx)} symbols")
print(f"  Metrics:  {len(all_metrics):,} registros, {len(metrics_idx)} symbols")

# ============================================================
# 3. PERIODOS DE TRADING
# ============================================================
trade_periods = []
for thu in cap_dates:
    fri_entry = None
    for d in range(1, 5):
        c = thu + pd.Timedelta(days=d)
        if c in spy_dates:
            fri_entry = c
            break
    fri_exit = None
    for d in range(8, 12):
        c = thu + pd.Timedelta(days=d)
        if c in spy_dates:
            fri_exit = c
            break
    if fri_entry and fri_exit:
        trade_periods.append((thu, fri_entry, fri_exit))
    else:
        print(f'  Sin trading: {thu.strftime("%Y-%m-%d")} -> entry:{fri_entry} exit:{fri_exit}')

print(f'Periodos de trading validos: {len(trade_periods)} de {len(cap_dates)}')

# ============================================================
# 4. FUNCIONES DE SELECCION
# ============================================================
def get_quality(sym, date):
    """Filtro calidad: EPS positivo AND (EPS beat OR ROE > 0)"""
    eps_pos = False
    eps_beat = False
    roe_ok = False
    roe_val = None

    earn = earnings_idx.get(sym)
    if earn is not None:
        mask = earn['date'] <= date
        valid = earn[mask]
        if len(valid) > 0:
            last = valid.iloc[-1]
            eps_pos = pd.notna(last['eps_actual']) and last['eps_actual'] > 0
            eps_beat = (pd.notna(last['eps_actual']) and pd.notna(last['eps_estimated'])
                       and last['eps_actual'] > last['eps_estimated'])

    met = metrics_idx.get(sym)
    if met is not None:
        mask = met['date'] <= date
        valid = met[mask]
        if len(valid) > 0:
            roe_val = valid.iloc[-1]['roe']
            roe_ok = pd.notna(roe_val) and roe_val > 0

    quality = eps_pos and (eps_beat or roe_ok)
    return quality, eps_pos, eps_beat, roe_val


def select_baseline(dd, tradeable, n=5):
    """Estrategia A: puro drawdown, sin filtro sector/calidad"""
    short = dd[tradeable].sort_values(ascending=False).head(n).index.tolist()
    long = dd[tradeable].sort_values(ascending=True).head(n).index.tolist()
    return short, long


def select_sector(dd, tradeable, thu, n=5):
    """Estrategia B: Short defensivas resistentes + Long ciclicas oversold con calidad"""
    # SHORTS: defensivas con menor drawdown (las que menos han caido)
    defensive = [s for s in tradeable if ticker_to_sector.get(s) in DEFENSIVE]
    if len(defensive) >= n:
        short = dd[defensive].sort_values(ascending=False).head(n).index.tolist()
    else:
        # Fallback: si no hay suficientes defensivas, rellenar con las mas resistentes
        short = dd[tradeable].sort_values(ascending=False).head(n).index.tolist()

    # LONGS: ciclicas con mas drawdown + filtro calidad
    cyclical = [s for s in tradeable if ticker_to_sector.get(s) in CYCLICAL]
    candidates = dd[cyclical].sort_values(ascending=True)  # mas oversold primero

    long = []
    long_quality_info = []
    for sym in candidates.index:
        quality, eps_p, eps_b, roe = get_quality(sym, thu)
        if quality:
            long.append(sym)
            long_quality_info.append((sym, dd[sym], eps_p, eps_b, roe))
            if len(long) >= n:
                break

    # Fallback: si no hay suficientes con calidad, rellenar ciclicas sin filtro
    if len(long) < n:
        for sym in candidates.index:
            if sym not in long:
                long.append(sym)
                long_quality_info.append((sym, dd[sym], False, False, None))
                if len(long) >= n:
                    break

    return short, long, long_quality_info


# ============================================================
# 5. BACKTEST PERIODO A PERIODO
# ============================================================
print(f'\n{"="*150}')
print('BACKTEST CAPITULACION - Comparativa A (Baseline) vs B (Sector+Calidad)')
print(f'{"="*150}')

results_a = []
results_b = []

for i, (thu, entry, exit_d) in enumerate(trade_periods):
    cap_info = cap_rows[cap_rows['fecha_senal'] == thu].iloc[0]
    vix = cap_info['vix']
    vix_delta = vix - cap_info['prev_vix'] if pd.notna(cap_info.get('prev_vix', np.nan)) else 0

    # Cargar precios historicos (380 dias para drawdown 52w)
    dd_start = thu - pd.Timedelta(days=380)
    hist = pd.read_sql(f"""
        SELECT symbol, date, close, high FROM fmp_price_history
        WHERE symbol IN ('{tlist}')
        AND date BETWEEN '{dd_start.strftime('%Y-%m-%d')}' AND '{thu.strftime('%Y-%m-%d')}'
        ORDER BY symbol, date
    """, engine)
    hist['date'] = pd.to_datetime(hist['date'])

    # Drawdown 52w
    max_52w = hist.groupby('symbol')['high'].max()
    last_close = hist.sort_values('date').groupby('symbol')['close'].last()
    common = max_52w.index.intersection(last_close.index)
    drawdown = (last_close[common] / max_52w[common] - 1) * 100

    # MA200 distance (indicador adicional)
    ma200_series = hist.sort_values('date').groupby('symbol')['close'].apply(
        lambda x: x.rolling(200, min_periods=150).mean().iloc[-1] if len(x) >= 150 else np.nan)
    dist_ma200 = ((last_close / ma200_series - 1) * 100).dropna()

    # Precios entry/exit
    p_entry = pd.read_sql(f"""
        SELECT symbol, open FROM fmp_price_history
        WHERE date = '{entry.strftime('%Y-%m-%d')}' AND symbol IN ('{tlist}')
    """, engine).set_index('symbol')['open']

    p_exit = pd.read_sql(f"""
        SELECT symbol, open FROM fmp_price_history
        WHERE date = '{exit_d.strftime('%Y-%m-%d')}' AND symbol IN ('{tlist}')
    """, engine).set_index('symbol')['open']

    tradeable = p_entry.index.intersection(p_exit.index).intersection(drawdown.index).tolist()
    ret = (p_exit[tradeable] / p_entry[tradeable] - 1) * 100
    dd = drawdown[tradeable]

    # SPY return y gap
    spy_ret = (spy.loc[exit_d, 'open'] / spy.loc[entry, 'open'] - 1) * 100
    gap = (spy.loc[entry, 'open'] / spy.loc[thu, 'close'] - 1) * 100 if thu in spy.index else 0

    # --- ESTRATEGIA A: BASELINE ---
    short_a, long_a = select_baseline(dd, tradeable)
    short_ret_a = -ret[short_a].mean()
    long_ret_a = ret[long_a].mean()
    total_a = (long_ret_a + short_ret_a) / 2

    results_a.append({
        'fecha': thu, 'entry': entry, 'exit': exit_d, 'vix': vix, 'vix_delta': vix_delta,
        'gap_pct': gap, 'spy_ret': spy_ret,
        'long_ret': long_ret_a, 'short_ret': short_ret_a, 'total': total_a,
        'long_syms': long_a, 'short_syms': short_a,
    })

    # --- ESTRATEGIA B: SECTOR + CALIDAD ---
    short_b, long_b, long_b_info = select_sector(dd, tradeable, thu)
    short_ret_b = -ret[short_b].mean() if len(short_b) > 0 else 0
    long_ret_b = ret[long_b].mean() if len(long_b) > 0 else 0
    total_b = (long_ret_b + short_ret_b) / 2

    # Contar cuantos longs pasaron filtro calidad
    n_quality = sum(1 for info in long_b_info if info[2])  # eps_pos = True

    results_b.append({
        'fecha': thu, 'entry': entry, 'exit': exit_d, 'vix': vix, 'vix_delta': vix_delta,
        'gap_pct': gap, 'spy_ret': spy_ret,
        'long_ret': long_ret_b, 'short_ret': short_ret_b, 'total': total_b,
        'long_syms': long_b, 'short_syms': short_b,
        'n_quality': n_quality,
    })

    # MA200 info para los longs de B
    ma_info = []
    for s in long_b[:3]:
        if s in dist_ma200.index:
            ma_info.append(f'{s}(ma:{dist_ma200[s]:+.0f}%)')
        else:
            ma_info.append(f'{s}')

    # Print periodo
    la = ' '.join([f'{s}' for s in long_a[:3]])
    lb = ' '.join(ma_info)
    sb = ' '.join([f'{s}({ticker_to_sector.get(s,"?")[:6]})' for s in short_b[:3]])
    winner = 'A' if total_a > total_b else 'B' if total_b > total_a else '='
    print(f'{i+1:>2}. {thu.strftime("%Y-%m-%d")} VIX:{vix:>4.0f} | '
          f'A:{total_a:>+5.1f}% B:{total_b:>+5.1f}% [{winner}] SPY:{spy_ret:>+5.1f}% | '
          f'A-L:[{la}] B-L:[{lb}] B-S:[{sb}] q:{n_quality}/5')


# ============================================================
# 6. RESUMEN COMPARATIVO
# ============================================================
dfa = pd.DataFrame(results_a)
dfb = pd.DataFrame(results_b)

print(f'\n{"="*90}')
print('RESUMEN COMPARATIVO - CAPITULACION')
print(f'{"="*90}')
print(f'Semanas: {len(dfa)}')

print(f'\n{"Metrica":<20} {"A:Baseline":>14} {"B:Sector+Q":>14} {"SPY":>14}')
print('-'*65)
print(f'{"Avg %":<20} {dfa["total"].mean():>+13.2f}% {dfb["total"].mean():>+13.2f}% {dfa["spy_ret"].mean():>+13.2f}%')
print(f'{"Median %":<20} {dfa["total"].median():>+13.2f}% {dfb["total"].median():>+13.2f}% {dfa["spy_ret"].median():>+13.2f}%')
print(f'{"WR %":<20} {(dfa["total"]>0).mean()*100:>12.1f}% {(dfb["total"]>0).mean()*100:>12.1f}% {(dfa["spy_ret"]>0).mean()*100:>12.1f}%')
print(f'{"Best %":<20} {dfa["total"].max():>+13.2f}% {dfb["total"].max():>+13.2f}% {dfa["spy_ret"].max():>+13.2f}%')
print(f'{"Worst %":<20} {dfa["total"].min():>+13.2f}% {dfb["total"].min():>+13.2f}% {dfa["spy_ret"].min():>+13.2f}%')
print(f'{"Total acum %":<20} {dfa["total"].sum():>+13.1f}% {dfb["total"].sum():>+13.1f}% {dfa["spy_ret"].sum():>+13.1f}%')

# Sharpe
sha = dfa["total"].mean() / dfa["total"].std() * np.sqrt(52) if dfa["total"].std() > 0 else 0
shb = dfb["total"].mean() / dfb["total"].std() * np.sqrt(52) if dfb["total"].std() > 0 else 0
print(f'{"Sharpe (anual)":<20} {sha:>13.2f}  {shb:>13.2f}')

print(f'\n{"Desglose":<20} {"A-Long":>10} {"A-Short":>10} {"B-Long":>10} {"B-Short":>10}')
print('-'*65)
print(f'{"Avg %":<20} {dfa["long_ret"].mean():>+9.2f}% {dfa["short_ret"].mean():>+9.2f}% {dfb["long_ret"].mean():>+9.2f}% {dfb["short_ret"].mean():>+9.2f}%')
print(f'{"WR %":<20} {(dfa["long_ret"]>0).mean()*100:>8.1f}% {(dfa["short_ret"]>0).mean()*100:>8.1f}% {(dfb["long_ret"]>0).mean()*100:>8.1f}% {(dfb["short_ret"]>0).mean()*100:>8.1f}%')
print(f'{"Total acum %":<20} {dfa["long_ret"].sum():>+9.1f}% {dfa["short_ret"].sum():>+9.1f}% {dfb["long_ret"].sum():>+9.1f}% {dfb["short_ret"].sum():>+9.1f}%')

# Gap medio
print(f'\nGap apertura viernes: {dfa["gap_pct"].mean():+.2f}% medio')
print(f'Longs B con filtro calidad: {dfb["n_quality"].mean():.1f}/5 media')

# ============================================================
# 7. VENTAJA vs BENCHMARK EN SEMANAS MALAS
# ============================================================
print(f'\n{"="*70}')
print('PROTECCION EN SEMANAS MALAS (SPY negativo)')
print(f'{"="*70}')
spy_neg = dfa['spy_ret'] < 0
n_neg = spy_neg.sum()
if n_neg > 0:
    print(f'Semanas SPY negativo: {n_neg} de {len(dfa)}')
    print(f'  SPY medio:      {dfa.loc[spy_neg, "spy_ret"].mean():+.2f}%')
    print(f'  A medio:        {dfa.loc[spy_neg, "total"].mean():+.2f}%')
    print(f'  B medio:        {dfb.loc[spy_neg, "total"].mean():+.2f}%')
    print(f'  B WR:           {(dfb.loc[spy_neg, "total"]>0).mean()*100:.1f}%')
    print(f'  Ventaja B vs SPY: {dfb.loc[spy_neg, "total"].mean() - dfa.loc[spy_neg, "spy_ret"].mean():+.2f}%')

# ============================================================
# 8. ACCIONES MAS FRECUENTES - ESTRATEGIA B
# ============================================================
print(f'\n{"="*70}')
print('ACCIONES MAS FRECUENTES - ESTRATEGIA B (Sector+Calidad)')
print(f'{"="*70}')

long_counter = Counter()
short_counter = Counter()
for r in results_b:
    for s in r['long_syms']:
        long_counter[s] += 1
    for s in r['short_syms']:
        short_counter[s] += 1

print(f'\nLONG ciclicas oversold + calidad:')
for sym, count in long_counter.most_common(20):
    sector = ticker_to_sector.get(sym, '?')
    ind = ticker_to_ind.get(sym, '?')
    print(f'  {sym:<6} {count:>2}x  {sector[:25]:<25} {ind[:30]}')

print(f'\nSHORT defensivas resistentes:')
for sym, count in short_counter.most_common(20):
    sector = ticker_to_sector.get(sym, '?')
    ind = ticker_to_ind.get(sym, '?')
    print(f'  {sym:<6} {count:>2}x  {sector[:25]:<25} {ind[:30]}')

# ============================================================
# 9. DETALLE PERIODO A PERIODO (A vs B)
# ============================================================
print(f'\n{"="*110}')
print(f'{"Fecha":<12} {"VIX":>4} {"Gap":>6} {"SPY":>6} | {"A-Tot":>6} {"A-L":>6} {"A-S":>6} | {"B-Tot":>6} {"B-L":>6} {"B-S":>6} | {"Mejor":>5}')
print('-'*110)
wins_a = wins_b = ties = 0
for idx in range(len(dfa)):
    a = dfa.iloc[idx]
    b = dfb.iloc[idx]
    if a['total'] > b['total']:
        better = 'A'
        wins_a += 1
    elif b['total'] > a['total']:
        better = 'B'
        wins_b += 1
    else:
        better = '='
        ties += 1
    print(f'{a["fecha"].strftime("%Y-%m-%d"):<12} {a["vix"]:>4.0f} {a["gap_pct"]:>+5.1f}% {a["spy_ret"]:>+5.1f}% | '
          f'{a["total"]:>+5.1f}% {a["long_ret"]:>+5.1f}% {a["short_ret"]:>+5.1f}% | '
          f'{b["total"]:>+5.1f}% {b["long_ret"]:>+5.1f}% {b["short_ret"]:>+5.1f}% | '
          f'  {better}')

print('-'*110)
print(f'A gana: {wins_a}x | B gana: {wins_b}x | Empate: {ties}x')
