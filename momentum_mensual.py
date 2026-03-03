"""
HTML navegable: Top/Bottom 25 acciones S&P 500 por mes.
Genera 4 HTMLs, uno por ventana de retorno (1M, 3M, 6M, 12M).
Para cada mes rankea acciones por retorno de la ventana y muestra
su rendimiento el mes siguiente. Historico 2001-2026.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sector_event_map import SUBSECTORS
import json, sys, io, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FMP_DB = "postgresql://fmp:fmp123@localhost:5433/fmp_data"

REGIME_COLORS = {
    'BURBUJA': '#e91e63', 'GOLDILOCKS': '#4caf50', 'ALCISTA': '#2196f3',
    'NEUTRAL': '#ff9800', 'CAUTIOUS': '#ff5722', 'BEARISH': '#795548',
    'CRISIS': '#9c27b0', 'PANICO': '#f44336', 'CAPITULACION': '#00bcd4',
    'RECOVERY': '#8bc34a',
}
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']
engine = create_engine(FMP_DB)

# ================================================================
# Subsector mapping
# ================================================================
ticker_to_sub = {}
for sub_id, sub_data in SUBSECTORS.items():
    label = sub_data['label']
    for t in sub_data['tickers']:
        ticker_to_sub[t] = label

# ================================================================
# Composicion historica S&P 500
# ================================================================
print("Cargando composicion S&P 500...")
with open('data/sp500_constituents.json') as f:
    sp500_current = {s['symbol'] for s in json.load(f)}
with open('data/sp500_historical_changes.json') as f:
    sp500_changes = json.load(f)

sp500_changes_sorted = sorted(sp500_changes, key=lambda c: c.get('date', c.get('dateAdded', '')), reverse=True)
members_snapshots = {}
current_members = set(sp500_current)
for chg in sp500_changes_sorted:
    chg_date = chg.get('date', chg.get('dateAdded', ''))
    added = chg.get('symbol', '')
    removed = chg.get('removedTicker', '')
    if added and added in current_members:
        current_members.discard(added)
    if removed:
        current_members.add(removed)
    members_snapshots[chg_date] = set(current_members)

def get_sp500_members(date_str):
    relevant_dates = sorted(members_snapshots.keys())
    members = set(sp500_current)
    for d in reversed(relevant_dates):
        if d <= date_str:
            members = members_snapshots[d]
            break
    return members

all_historical_tickers = set()
for d, members in members_snapshots.items():
    all_historical_tickers |= members
all_historical_tickers |= sp500_current
all_tickers = sorted(all_historical_tickers)
print(f"  Miembros actuales: {len(sp500_current)}, historicos: {len(all_tickers)}")

# ================================================================
# Cargar industria/sector de fmp_profiles
# ================================================================
print("Cargando sectores...")
batch_size = 100
sector_frames = []
for i in range(0, len(all_tickers), batch_size):
    batch = all_tickers[i:i+batch_size]
    tlist = "','".join(batch)
    df_s = pd.read_sql(f"""
        SELECT DISTINCT ON (symbol) symbol, industry, sector
        FROM fmp_profiles WHERE symbol IN ('{tlist}') ORDER BY symbol
    """, engine)
    sector_frames.append(df_s)
df_sectors = pd.concat(sector_frames, ignore_index=True) if sector_frames else pd.DataFrame()
ticker_industry = dict(zip(df_sectors['symbol'], df_sectors['industry'].fillna('')))
ticker_sector = dict(zip(df_sectors['symbol'], df_sectors['sector'].fillna('')))
print(f"  Perfiles: {len(df_sectors)}")

# ================================================================
# Cargar VIX diario
# ================================================================
print("Cargando VIX...")
df_vix = pd.read_sql("SELECT date, close as vix FROM price_history_vix WHERE symbol='^VIX' ORDER BY date", engine)
df_vix['date'] = pd.to_datetime(df_vix['date'])
df_vix = df_vix.set_index('date').sort_index()
print(f"  VIX: {len(df_vix)} registros")

def get_subsector(sym):
    if sym in ticker_to_sub:
        return ticker_to_sub[sym]
    ind = ticker_industry.get(sym, '')
    return ind if ind else ticker_sector.get(sym, '')

# ================================================================
# Cargar precios diarios
# ================================================================
print("Cargando precios diarios...")
frames = []
for i in range(0, len(all_tickers), batch_size):
    batch = all_tickers[i:i+batch_size]
    tlist = "','".join(batch)
    df_batch = pd.read_sql(f"""
        SELECT symbol, date, close
        FROM fmp_price_history WHERE symbol IN ('{tlist}')
        AND date BETWEEN '1999-01-01' AND '2026-12-31' ORDER BY symbol, date
    """, engine)
    frames.append(df_batch)
    if (i // batch_size + 1) % 5 == 0:
        print(f"  Lote {i//batch_size+1}: {sum(len(f) for f in frames)} registros")
df = pd.concat(frames, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df['close'] = df['close'].astype('float32')
df['symbol'] = df['symbol'].astype('category')
loaded_symbols = set(df['symbol'].unique())
print(f"  Total: {len(df)} registros ({len(loaded_symbols)} simbolos)")

# ================================================================
# Close mensual (ultimo dia de trading de cada mes)
# ================================================================
print("Calculando cierres mensuales...")
df['month'] = df['date'].dt.to_period('M')
monthly = df.sort_values('date').groupby(['symbol', 'month'], observed=True).agg(
    close=('close', 'last'),
    last_date=('date', 'last')
).reset_index()
monthly['symbol'] = monthly['symbol'].astype(str)
monthly = monthly.sort_values(['symbol', 'month'])

# ================================================================
# Extraer SPY diario y precios mensuales por subsector (ANTES de liberar df)
# ================================================================
print("Extrayendo SPY diario y subsectores mensuales...")

# SPY diario (cargado aparte, no esta en all_tickers - es ETF)
spy_daily = pd.read_sql("""
    SELECT date, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '1998-01-01' AND '2026-12-31'
    ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()
spy_daily['ma200'] = spy_daily['close'].rolling(200).mean()
spy_daily['above_ma200'] = (spy_daily['close'] > spy_daily['ma200']).astype(int)
spy_daily['dist_ma200'] = (spy_daily['close'] / spy_daily['ma200'] - 1) * 100

# SPY mensual (ultimo close del mes)
spy_monthly = spy_daily[['close', 'ma200', 'above_ma200', 'dist_ma200']].resample('ME').last().dropna(subset=['ma200'])
spy_monthly.index = spy_monthly.index.to_period('M')
spy_monthly['mom_3m'] = spy_monthly['close'].pct_change(3) * 100
print(f"  SPY diario: {len(spy_daily)} | SPY mensual: {len(spy_monthly)}")

# Precios mensuales por subsector
df['subsector'] = df['symbol'].apply(get_subsector)
df_with_sub = df[df['subsector'] != ''].copy()
sub_counts = df_with_sub.groupby('subsector')['symbol'].nunique()
valid_subs = sub_counts[sub_counts >= 3].index
df_with_sub = df_with_sub[df_with_sub['subsector'].isin(valid_subs)]
sub_monthly = df_with_sub.groupby(['subsector', 'month']).agg(avg_close=('close', 'mean')).reset_index()
sub_monthly = sub_monthly.sort_values(['subsector', 'month'])
print(f"  Subsectores validos: {len(valid_subs)} | Registros mensuales: {len(sub_monthly)}")

del df, df_with_sub; gc.collect()
print("  Memoria liberada")

# ================================================================
# Calcular retornos multi-ventana: 1M, 3M, 6M, 12M
# ================================================================
print("Calculando retornos multi-ventana...")
WINDOWS = [1, 3, 6, 12]

for w in WINDOWS:
    col = f'ret_{w}m'
    monthly[col] = monthly.groupby('symbol')['close'].transform(
        lambda x: (x / x.shift(w) - 1) * 100
    )

# Retorno 1M del mes SIGUIENTE (para evaluar momentum/reversion)
monthly['next_ret_1m'] = monthly.groupby('symbol')['ret_1m'].shift(-1)

# Filtrar extremos por ventana (splits, datos corruptos)
for w in WINDOWS:
    col = f'ret_{w}m'
    threshold_hi = 300 + w * 50  # mas tolerante para ventanas largas
    threshold_lo = -80
    extreme = monthly[(monthly[col] > threshold_hi) | (monthly[col] < threshold_lo)]
    if len(extreme) > 0:
        n_ext = len(extreme)
        monthly.loc[(monthly[col] > threshold_hi) | (monthly[col] < threshold_lo), col] = np.nan
        print(f"  {col}: {n_ext} valores extremos eliminados")

# Capear
for w in WINDOWS:
    col = f'ret_{w}m'
    monthly[col] = monthly[col].clip(-80, 500)
monthly['next_ret_1m'] = monthly['next_ret_1m'].clip(-80, 200)

print(f"  Registros mensuales: {len(monthly)}")

# ================================================================
# Calcular metricas mensuales por subsector (DD 12M, RSI 6M)
# ================================================================
print("Calculando metricas mensuales por subsector...")

def calc_sub_monthly_metrics(grp):
    grp = grp.sort_values('month')
    high_12 = grp['avg_close'].rolling(12, min_periods=6).max()
    grp['drawdown_12m'] = (grp['avg_close'] / high_12 - 1) * 100
    delta = grp['avg_close'].diff()
    gain = delta.clip(lower=0).rolling(6, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(6, min_periods=3).mean()
    rs = gain / loss.replace(0, np.nan)
    grp['rsi_6m'] = 100 - (100 / (1 + rs))
    return grp

sub_monthly = sub_monthly.groupby('subsector', group_keys=False).apply(calc_sub_monthly_metrics)
dd_monthly = sub_monthly.pivot(index='month', columns='subsector', values='drawdown_12m')
rsi_monthly = sub_monthly.pivot(index='month', columns='subsector', values='rsi_6m')
print(f"  Meses con metricas: {len(dd_monthly)}")

# ================================================================
# VIX mensual (ultimo close del mes)
# ================================================================
vix_monthly = df_vix[['vix']].resample('ME').last().dropna()
vix_monthly.index = vix_monthly.index.to_period('M')
print(f"  VIX mensual: {len(vix_monthly)}")

# ================================================================
# Calcular regimenes mensuales
# ================================================================
print("Calculando regimenes mensuales...")

def compute_regimes_monthly():
    months = sorted(dd_monthly.index[dd_monthly.index >= pd.Period('2001-01', 'M')])
    regimes = {}
    prev_vix = None

    for month in months:
        if month not in dd_monthly.index:
            continue
        dd_row = dd_monthly.loc[month]
        rsi_row = rsi_monthly.loc[month] if month in rsi_monthly.index else pd.Series(dtype=float)

        n_total = dd_row.notna().sum()
        if n_total == 0:
            continue

        n_dd_h = int((dd_row > -10).sum())
        n_dd_d = int((dd_row < -20).sum())
        n_rsi_t = int(rsi_row.notna().sum())
        n_rsi_55 = int((rsi_row > 55).sum())
        pct_dd_h = n_dd_h / n_total * 100
        pct_dd_d = n_dd_d / n_total * 100
        pct_rsi = n_rsi_55 / n_rsi_t * 100 if n_rsi_t > 0 else 50

        # SPY
        if month not in spy_monthly.index:
            continue
        spy_row = spy_monthly.loc[month]
        spy_above = spy_row['above_ma200']
        spy_dist = spy_row['dist_ma200']
        spy_mom = spy_row['mom_3m']
        if not pd.notna(spy_mom): spy_mom = 0
        if not pd.notna(spy_dist): spy_dist = 0

        # VIX
        vix_val = vix_monthly.loc[month, 'vix'] if month in vix_monthly.index else 20
        if not pd.notna(vix_val): vix_val = 20

        # Scores (mismos umbrales que semanal)
        if pct_dd_h >= 75: s_bdd = 2.0
        elif pct_dd_h >= 60: s_bdd = 1.0
        elif pct_dd_h >= 45: s_bdd = 0.0
        elif pct_dd_h >= 30: s_bdd = -1.0
        elif pct_dd_h >= 15: s_bdd = -2.0
        else: s_bdd = -3.0

        if pct_rsi >= 75: s_brsi = 2.0
        elif pct_rsi >= 60: s_brsi = 1.0
        elif pct_rsi >= 45: s_brsi = 0.0
        elif pct_rsi >= 30: s_brsi = -1.0
        elif pct_rsi >= 15: s_brsi = -2.0
        else: s_brsi = -3.0

        if pct_dd_d <= 5: s_ddp = 1.5
        elif pct_dd_d <= 15: s_ddp = 0.5
        elif pct_dd_d <= 30: s_ddp = -0.5
        elif pct_dd_d <= 50: s_ddp = -1.5
        else: s_ddp = -2.5

        if spy_above and spy_dist > 5: s_spy = 1.5
        elif spy_above: s_spy = 0.5
        elif spy_dist > -5: s_spy = -0.5
        elif spy_dist > -15: s_spy = -1.5
        else: s_spy = -2.5

        if spy_mom > 5: s_mom = 1.0
        elif spy_mom > 0: s_mom = 0.5
        elif spy_mom > -5: s_mom = -0.5
        elif spy_mom > -15: s_mom = -1.0
        else: s_mom = -1.5

        total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

        is_burbuja = (total >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90)
        if is_burbuja: regime = 'BURBUJA'
        elif total >= 7.0: regime = 'GOLDILOCKS'
        elif total >= 4.0: regime = 'ALCISTA'
        elif total >= 0.5: regime = 'NEUTRAL'
        elif total >= -2.0: regime = 'CAUTIOUS'
        elif total >= -5.0: regime = 'BEARISH'
        elif total >= -9.0: regime = 'CRISIS'
        else: regime = 'PANICO'

        # VIX veto
        if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
            regime = 'NEUTRAL'
        elif vix_val >= 35 and regime == 'NEUTRAL':
            regime = 'CAUTIOUS'

        # CAPITULACION / RECOVERY
        vix_delta = vix_val - prev_vix if prev_vix is not None else 0
        if regime == 'PANICO' and prev_vix is not None and vix_delta < 0:
            regime = 'CAPITULACION'
        elif regime == 'BEARISH' and prev_vix is not None and vix_delta < 0:
            regime = 'RECOVERY'
        prev_vix = vix_val

        regimes[month] = {
            'regime': regime, 'total': round(total, 1), 'vix': round(float(vix_val), 1),
            'pct_dd_h': round(pct_dd_h, 1), 'pct_rsi': round(pct_rsi, 1),
            'scores': {'bdd': s_bdd, 'brsi': s_brsi, 'ddp': s_ddp, 'spy': s_spy, 'mom': s_mom},
        }

    return regimes

month_regimes = compute_regimes_monthly()
print(f"  Regimenes calculados: {len(month_regimes)} meses")
# Resumen
from collections import Counter
reg_counts = Counter(r['regime'] for r in month_regimes.values())
for r in REGIME_ORDER:
    if r in reg_counts:
        print(f"    {r}: {reg_counts[r]}")

# ================================================================
# Generar datos y HTML para cada ventana
# ================================================================

WINDOW_LABELS = {1: '1 Mes', 3: '3 Meses', 6: '6 Meses', 12: '12 Meses (Anual)'}
WINDOW_FILES = {1: 'momentum_mensual_1m.html', 3: 'momentum_mensual_3m.html',
                6: 'momentum_mensual_6m.html', 12: 'momentum_mensual_12m.html'}

months_list = sorted(monthly['month'].unique())
all_yearly_results = {}  # guardar yearly_stats por ventana para resumen final

for window in WINDOWS:
    ret_col = f'ret_{window}m'
    label = WINDOW_LABELS[window]
    outfile = WINDOW_FILES[window]
    print(f"\n{'='*60}")
    print(f"Generando {outfile} (ventana {label})...")

    all_months_data = []

    for month in months_list:
        year = month.year
        m_num = month.month
        if year < 2001:
            continue

        month_data = monthly[monthly['month'] == month].copy()
        month_data = month_data.dropna(subset=[ret_col])
        if len(month_data) < 50:
            continue

        last_day = month_data['last_date'].max()
        if pd.isna(last_day):
            continue
        sp500_at_date = get_sp500_members(last_day.strftime('%Y-%m-%d'))
        month_data = month_data[month_data['symbol'].isin(sp500_at_date & loaded_symbols)]

        if len(month_data) < 50:
            continue

        month_data = month_data.sort_values(ret_col, ascending=False)

        top25 = month_data.head(25)
        bottom25 = month_data.tail(25)

        # Mes de operacion = mes siguiente al de la senal
        trade_month = month + 1  # Period arithmetic
        trade_year = trade_month.year
        trade_m_num = trade_month.month

        def make_stocks(df_slice):
            stocks = []
            for _, row in df_slice.iterrows():
                sym = row['symbol']
                nr = round(float(row['next_ret_1m']), 2) if pd.notna(row['next_ret_1m']) else None
                stocks.append([
                    sym,
                    get_subsector(sym),
                    round(float(row[ret_col]), 2),
                    nr,
                    round(float(row['close']), 2),
                ])
            return stocks

        top_stocks = make_stocks(top25)
        bot_stocks = make_stocks(bottom25)

        def calc_stats(rets):
            if not rets:
                return {'a': None, 'w': None, 'n': 0}
            return {
                'a': round(float(np.mean(rets)), 2),
                'w': round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                'n': len(rets),
            }

        top_next = [s[3] for s in top_stocks if s[3] is not None]
        bot_next = [s[3] for s in bot_stocks if s[3] is not None]
        top10_next = [s[3] for s in top_stocks[:10] if s[3] is not None]
        bot10_next = [s[3] for s in bot_stocks[-10:] if s[3] is not None]

        # Estrategia conjunta: Top 10 + Bot 10 (20 acciones)
        combo_next = top10_next + bot10_next

        # Regimen del mes senal
        reg_info = month_regimes.get(month, {})
        reg = reg_info.get('regime', '')
        rsc = reg_info.get('total', 0)
        rvx = reg_info.get('vix', 0)

        all_months_data.append({
            'm': str(trade_month), 'y': trade_year, 'mn': trade_m_num,
            'n': len(month_data),
            'top': top_stocks, 'bot': bot_stocks,
            'ts': calc_stats(top_next), 'bs': calc_stats(bot_next),
            't10': calc_stats(top10_next), 'b10': calc_stats(bot10_next),
            'cmb': calc_stats(combo_next),
            'reg': reg, 'rsc': rsc, 'rvx': rvx,
        })

    print(f"  Meses procesados: {len(all_months_data)}")

    # Estadisticas anuales con gestion monetaria (capital fijo, NO compounding)
    COST_PCT = 0.3  # slippage + comisiones por operacion
    CAP_PER_STOCK = 20000
    GROUPS = {'top': 'ts', 'bot': 'bs', 't10': 't10', 'b10': 'b10', 'cmb': 'cmb'}
    N_STOCKS = {'top': 25, 'bot': 25, 't10': 10, 'b10': 10, 'cmb': 20}

    # Capital fijo por grupo (se invierte siempre la misma cantidad cada mes)
    INIT_CAP = {k: CAP_PER_STOCK * N_STOCKS[k] for k in GROUPS}
    # PnL acumulado
    pnl = {k: 0.0 for k in GROUPS}

    # Recopilar datos por año y trackear PnL
    yearly_data = {}
    all_months_sorted = sorted(all_months_data, key=lambda m: m['m'])

    for md in all_months_sorted:
        y = md['y']
        if y not in yearly_data:
            yearly_data[y] = {k: [] for k in GROUPS}
            yearly_data[y].update({k + '_pnl': [] for k in GROUPS})

        for grp, stat_key in GROUPS.items():
            gross = md[stat_key]['a']
            if gross is not None:
                yearly_data[y][grp].append(gross)
                net = gross - COST_PCT
                month_pnl = net / 100 * INIT_CAP[grp]  # PnL en $ sobre capital fijo
                yearly_data[y][grp + '_pnl'].append(month_pnl)
                pnl[grp] += month_pnl

        # Snapshot PnL acumulado al final de cada mes
        yearly_data[y]['pnl_end'] = dict(pnl)

    yearly_stats = {}
    for y in sorted(yearly_data.keys()):
        tr, br = yearly_data[y]['top'], yearly_data[y]['bot']
        t10, b10 = yearly_data[y]['t10'], yearly_data[y]['b10']
        if not tr or not br: continue
        ta, ba = float(np.mean(tr)), float(np.mean(br))
        t10a = float(np.mean(t10)) if t10 else 0
        b10a = float(np.mean(b10)) if b10 else 0

        cmb = yearly_data[y].get('cmb', [])
        cmba = float(np.mean(cmb)) if cmb else 0

        ys = {
            'ta': round(ta, 2), 'tw': round(sum(1 for r in tr if r > 0) / len(tr) * 100, 1),
            'ba': round(ba, 2), 'bw': round(sum(1 for r in br if r > 0) / len(br) * 100, 1),
            'n': len(tr), 'd': round(ta - ba, 2),
            't10a': round(t10a, 2), 't10w': round(sum(1 for r in t10 if r > 0) / len(t10) * 100, 1) if t10 else 0,
            'b10a': round(b10a, 2), 'b10w': round(sum(1 for r in b10 if r > 0) / len(b10) * 100, 1) if b10 else 0,
            'd10': round(t10a - b10a, 2),
            'cmba': round(cmba, 2), 'cmbw': round(sum(1 for r in cmb if r > 0) / len(cmb) * 100, 1) if cmb else 0,
        }
        # PnL anual y neto % sobre capital fijo
        for grp in GROUPS:
            year_pnl = sum(yearly_data[y][grp + '_pnl'])
            ys[grp + '_net'] = round(year_pnl / INIT_CAP[grp] * 100, 2)  # % neto anual
            ys[grp + '_pnl'] = round(year_pnl)  # PnL $ del año
        # Capital = inicial + PnL acumulado
        pnl_end = yearly_data[y]['pnl_end']
        for grp in GROUPS:
            ys['cap_' + grp] = round(INIT_CAP[grp] + pnl_end[grp])
        yearly_stats[y] = ys

    # Capital final
    for grp, label in [('top','Top25'), ('bot','Bot25'), ('t10','Top10'), ('b10','Bot10'), ('cmb','COMBO')]:
        final = INIT_CAP[grp] + pnl[grp]
        print(f"  Capital final {label}: ${final:,.0f} (inicio ${INIT_CAP[grp]:,} | PnL ${pnl[grp]:,.0f})")

    # Estadisticas globales
    decile_data = {'top': [], 'bot': [], 't10': [], 'b10': [], 'cmb': []}
    for md in all_months_data:
        decile_data['top'].extend([s[3] for s in md['top'] if s[3] is not None])
        decile_data['bot'].extend([s[3] for s in md['bot'] if s[3] is not None])
        t10r = [s[3] for s in md['top'][:10] if s[3] is not None]
        b10r = [s[3] for s in md['bot'][-10:] if s[3] is not None]
        decile_data['t10'].extend(t10r)
        decile_data['b10'].extend(b10r)
        decile_data['cmb'].extend(t10r + b10r)

    global_stats = {}
    for k in ['top', 'bot', 't10', 'b10', 'cmb']:
        d = decile_data[k]
        if d:
            global_stats[k+'_avg'] = round(float(np.mean(d)), 3)
            global_stats[k+'_wr'] = round(sum(1 for r in d if r > 0) / len(d) * 100, 1)
            global_stats[k+'_n'] = len(d)
        else:
            global_stats[k+'_avg'] = 0; global_stats[k+'_wr'] = 0; global_stats[k+'_n'] = 0
    global_stats['diff'] = round(global_stats['top_avg'] - global_stats['bot_avg'], 3)
    global_stats['diff10'] = round(global_stats['t10_avg'] - global_stats['b10_avg'], 3)

    print(f"  Top 25 ret mes: {global_stats['top_avg']:.3f}% | Bot 25: {global_stats['bot_avg']:.3f}% | Diff: {global_stats['diff']:.3f}%")
    print(f"  Top 10 ret mes: {global_stats['t10_avg']:.3f}% | Bot 10: {global_stats['b10_avg']:.3f}% | Diff: {global_stats['diff10']:.3f}%")
    print(f"  COMBO (T10+B10): {global_stats['cmb_avg']:.3f}% | WR: {global_stats['cmb_wr']:.1f}%")

    # ================================================================
    # Stats por regimen
    # ================================================================
    regime_stats = {}
    for reg in REGIME_ORDER:
        reg_months = [md for md in all_months_data if md.get('reg') == reg]
        if not reg_months:
            continue
        rs = {'n': len(reg_months)}
        for grp, stat_key in [('top','ts'), ('bot','bs'), ('t10','t10'), ('b10','b10'), ('cmb','cmb')]:
            rets = [md[stat_key]['a'] for md in reg_months if md[stat_key]['a'] is not None]
            if rets:
                rs[grp + '_avg'] = round(float(np.mean(rets)), 2)
                rs[grp + '_wr'] = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
            else:
                rs[grp + '_avg'] = 0; rs[grp + '_wr'] = 0
        regime_stats[reg] = rs

    # HTML
    all_yearly_results[window] = yearly_stats

    # ================================================================
    months_json = json.dumps(all_months_data)
    yearly_json = json.dumps(yearly_stats)
    global_json = json.dumps(global_stats)
    regime_stats_json = json.dumps(regime_stats)
    regime_colors_json = json.dumps(REGIME_COLORS)
    regime_order_json = json.dumps(REGIME_ORDER)

    ret_label = f'Señal {label}'
    ret_col_header = f'Se\\u00f1al {window}M %'

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>Momentum {label} S&P 500 - Top/Bottom 25</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #fff; color: #222; max-width: 1500px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #1565c0; text-align: center; margin-bottom: 5px; }}
h2 {{ color: #333; margin-top: 20px; margin-bottom: 8px; border-bottom: 2px solid #1565c0; padding-bottom: 4px; font-size: 15px; }}
.sub {{ text-align: center; color: #666; margin-bottom: 14px; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 11px; }}
th {{ background: #1565c0; color: #fff; padding: 5px 4px; text-align: center; border: 1px solid #ccc; cursor: pointer; font-size: 10px; white-space: nowrap; }}
th:hover {{ background: #0d47a1; }}
td {{ padding: 4px 5px; text-align: center; border: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #f5f7fa; }}
tr:hover {{ background: #e3f2fd; }}
.pos {{ color: #2e7d32; font-weight: bold; }}
.neg {{ color: #c62828; font-weight: bold; }}
.neutral {{ color: #999; }}
td.left {{ text-align: left; }}
.sbox {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; justify-content: center; }}
.sc {{ background: #f5f7fa; border: 1px solid #ddd; border-radius: 8px; padding: 10px 18px; text-align: center; min-width: 130px; }}
.sc h4 {{ margin: 0 0 4px 0; color: #1565c0; font-size: 11px; }}
.sc .v {{ font-size: 20px; font-weight: bold; }}
.sc .d {{ font-size: 10px; color: #666; margin-top: 2px; }}
.note {{ background: #fffde7; padding: 8px 12px; border-radius: 6px; border-left: 4px solid #ffd600; margin-bottom: 12px; font-size: 11px; }}
.nb {{ text-align: center; margin: 12px 0 16px 0; padding: 14px; background: #f5f7fa; border-radius: 8px; border: 1px solid #ddd; }}
.nb select {{ font-size: 13px; padding: 5px 10px; border-radius: 4px; border: 1px solid #ccc; margin: 0 6px; }}
.nb button {{ font-size: 12px; padding: 5px 14px; border-radius: 4px; border: 1px solid #1565c0; background: #1565c0; color: #fff; cursor: pointer; margin: 0 3px; }}
.nb button:hover {{ background: #0d47a1; }}
.yb {{ margin-top: 8px; }}
.yb button {{ font-size: 10px; padding: 3px 8px; border-radius: 3px; border: 1px solid #999; background: #fff; color: #333; cursor: pointer; margin: 2px; }}
.yb button:hover, .yb button.active {{ background: #1565c0; color: #fff; border-color: #1565c0; }}
.st {{ max-height: 750px; overflow-y: auto; border: 1px solid #ccc; border-radius: 4px; }}
.st table {{ margin-bottom: 0; }}
.st th {{ position: sticky; top: 0; z-index: 1; }}
.qt {{ width: auto !important; margin: 0 auto !important; min-width: 700px; }}
.qt td {{ padding: 5px 10px; font-size: 12px; }}
.qt th {{ font-size: 11px; padding: 6px 8px; }}
.cols {{ display: flex; gap: 20px; }}
.cols > div {{ flex: 1; min-width: 0; }}
@media (max-width: 1000px) {{ .cols {{ flex-direction: column; }} }}
.gh {{ text-align: center; margin: 16px 0; padding: 14px; background: #e3f2fd; border-radius: 8px; border: 1px solid #90caf9; }}
.gh .t {{ font-size: 13px; color: #1565c0; font-weight: bold; margin-bottom: 6px; }}
.gh .r {{ display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }}
.gh .c {{ text-align: center; }}
.gh .c .l {{ font-size: 11px; color: #666; }}
.gh .c .v {{ font-size: 22px; font-weight: bold; }}
.lnk {{ text-align: center; margin: 14px 0; padding: 10px; background: #f5f7fa; border-radius: 8px; border: 1px solid #ddd; font-size: 14px; }}
.lnk a {{ margin: 0 6px; color: #1565c0; text-decoration: none; font-weight: bold; padding: 6px 18px; border: 2px solid #1565c0; border-radius: 6px; display: inline-block; }}
.lnk a:hover {{ background: #e3f2fd; }}
.lnk a.act {{ background: #1565c0; color: #fff; }}
.rb {{ display: inline-block; padding: 3px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; color: #fff; margin: 0 2px; }}
.rb2 {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold; color: #fff; }}
.rfb {{ margin-top: 6px; }}
.rfb button {{ font-size: 10px; padding: 3px 8px; border-radius: 3px; border: 1px solid #999; color: #fff; cursor: pointer; margin: 2px; }}
.rfb button:hover, .rfb button.active {{ opacity: 0.8; box-shadow: 0 0 0 2px #333; }}
</style></head><body>
<h1>Momentum S&P 500 - Ventana {label}</h1>
<p class="sub">Top/Bottom 25 acciones por se&ntilde;al {label.lower()} | Mes = mes de operaci&oacute;n | 2001-2026</p>
<div class="lnk">
<a href="momentum_mensual_1m.html" {"class=act" if window == 1 else ""}>1 Mes</a>
<a href="momentum_mensual_3m.html" {"class=act" if window == 3 else ""}>3 Meses</a>
<a href="momentum_mensual_6m.html" {"class=act" if window == 6 else ""}>6 Meses</a>
<a href="momentum_mensual_12m.html" {"class=act" if window == 12 else ""}>12 Meses</a>
<a href="momentum_mensual_mix.html">MIX 12M+3M</a>
</div>

<div id="gs"></div>
<div class="nb">
<button onclick="pM()">&larr; Anterior</button>
<select id="ms" onchange="lM()"></select>
<button onclick="nM()">Siguiente &rarr;</button>
<div class="yb" id="yb"></div>
<div class="rfb" id="rfb"></div>
</div>
<div id="ct"></div>
<div id="yt"></div>
<div id="rt"></div>
<script>
const WL='{label}';
const WC='{ret_col_header}';
const MN=['','Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];
const D={months_json};
const YS={yearly_json};
const GS={global_json};
const RC={regime_colors_json};
const RO={regime_order_json};
const RS={regime_stats_json};
const sel=document.getElementById('ms');
D.forEach((m,i)=>{{const o=document.createElement('option');o.value=i;o.text=m.y+'-'+(m.mn<10?'0':'')+m.mn+' '+MN[m.mn]+(m.reg?' ['+m.reg+']':'');sel.appendChild(o);}});
const yrs=[...new Set(D.map(m=>m.y))].sort();
const yb=document.getElementById('yb');
const ab=document.createElement('button');ab.textContent='TODOS';ab.className='active';ab.onclick=()=>rF();yb.appendChild(ab);
yrs.forEach(y=>{{const b=document.createElement('button');b.textContent=y;b.onclick=()=>fY(y);yb.appendChild(b);}});
const rfb=document.getElementById('rfb');
RO.forEach(r=>{{const b=document.createElement('button');b.textContent=r;b.style.background=RC[r];b.style.borderColor=RC[r];b.onclick=()=>fReg(r);rfb.appendChild(b);}});
let aF=null,fI=D.map((_,i)=>i);
function apF(){{sel.innerHTML='';fI.forEach(i=>{{const m=D[i];const o=document.createElement('option');o.value=i;o.text=m.y+'-'+(m.mn<10?'0':'')+m.mn+' '+MN[m.mn]+(m.reg?' ['+m.reg+']':'');sel.appendChild(o);}});if(fI.length>0){{sel.value=fI[fI.length-1];lM();}}
document.querySelectorAll('#yb button').forEach(b=>{{if(b.textContent==='TODOS')b.classList.toggle('active',!aF);else b.classList.toggle('active',aF&&aF.type==='year'&&parseInt(b.textContent)===aF.value);}});
document.querySelectorAll('#rfb button').forEach(b=>{{b.classList.toggle('active',aF&&aF.type==='regime'&&b.textContent===aF.value);}});}}
function fY(y){{aF={{type:'year',value:y}};fI=D.map((m,i)=>m.y===y?i:-1).filter(i=>i>=0);apF();}}
function fReg(r){{aF={{type:'regime',value:r}};fI=D.map((m,i)=>m.reg===r?i:-1).filter(i=>i>=0);apF();}}
function rF(){{aF=null;fI=D.map((_,i)=>i);apF();}}
sel.value=D.length-1;
function pM(){{const c=fI.indexOf(parseInt(sel.value));if(c>0){{sel.value=fI[c-1];lM();}}}}
function nM(){{const c=fI.indexOf(parseInt(sel.value));if(c<fI.length-1){{sel.value=fI[c+1];lM();}}}}
function vc(v){{return v>0?'pos':v<0?'neg':'neutral';}}
function fm(v,d){{d=d||2;return(v>=0?'+':'')+v.toFixed(d);}}
function mkT(stocks,id){{
let h='<div class="st"><table id="'+id+'"><tr><th onclick="sC(\\''+id+'\\',0)">#</th><th onclick="sC(\\''+id+'\\',1)">Ticker</th><th onclick="sC(\\''+id+'\\',2)">Industria</th><th onclick="sC(\\''+id+'\\',3)">'+WC+'</th><th onclick="sC(\\''+id+'\\',4)">Ret Mes %</th><th onclick="sC(\\''+id+'\\',5)">Cierre $</th></tr>';
stocks.forEach((s,i)=>{{
const rc=vc(s[2]);const nc=s[3]!==null?vc(s[3]):'neutral';
const nr=s[3]!==null?'<span class="'+nc+'">'+fm(s[3])+'%</span>':'<span class="neutral">-</span>';
h+='<tr><td>'+(i+1)+'</td><td><b>'+s[0]+'</b></td><td class="left" style="font-size:10px;">'+s[1]+'</td><td class="'+rc+'">'+fm(s[2])+'%</td><td>'+nr+'</td><td>$'+s[4].toFixed(2)+'</td></tr>';
}});
h+='</table></div>';return h;
}}
// Global stats header
(function(){{
const g=GS;
const d25=g.diff;const e25=d25>0?'MOMENTUM':'REVERSION';const c25=d25>0?'pos':'neg';
const d10=g.diff10;const e10=d10>0?'MOMENTUM':'REVERSION';const c10=d10>0?'pos':'neg';
let h='<div class="gh"><div class="t">Resultado Global - Ventana '+WL+' ('+D.length+' meses, 2001-2026)</div>';
h+='<div class="r" style="margin-bottom:8px;">';
h+='<div class="c"><div class="l">Top 25 Ret Mes</div><div class="v '+vc(g.top_avg)+'">'+fm(g.top_avg,3)+'%</div><div class="l">WR: '+g.top_wr.toFixed(1)+'%</div></div>';
h+='<div class="c"><div class="l">Bot 25 Ret Mes</div><div class="v '+vc(g.bot_avg)+'">'+fm(g.bot_avg,3)+'%</div><div class="l">WR: '+g.bot_wr.toFixed(1)+'%</div></div>';
h+='<div class="c"><div class="l">Top-Bot 25</div><div class="v '+c25+'">'+fm(d25,3)+'%</div><div class="l" style="font-weight:bold;color:'+(d25>0?'#2e7d32':'#c62828')+';">'+e25+'</div></div>';
h+='</div><div class="r">';
h+='<div class="c"><div class="l">Top 10 Ret Mes</div><div class="v '+vc(g.t10_avg)+'">'+fm(g.t10_avg,3)+'%</div><div class="l">WR: '+g.t10_wr.toFixed(1)+'%</div></div>';
h+='<div class="c"><div class="l">Bot 10 Ret Mes</div><div class="v '+vc(g.b10_avg)+'">'+fm(g.b10_avg,3)+'%</div><div class="l">WR: '+g.b10_wr.toFixed(1)+'%</div></div>';
h+='<div class="c"><div class="l">Top-Bot 10</div><div class="v '+c10+'">'+fm(d10,3)+'%</div><div class="l" style="font-weight:bold;color:'+(d10>0?'#2e7d32':'#c62828')+';">'+e10+'</div></div>';
h+='</div><div class="r" style="margin-top:8px;border-top:2px solid #90caf9;padding-top:8px;">';
h+='<div class="c"><div class="l" style="font-weight:bold;color:#1565c0;">COMBO T10+B10 (20 acc, $400K)</div><div class="v '+vc(g.cmb_avg)+'">'+fm(g.cmb_avg,3)+'%</div><div class="l">WR: '+g.cmb_wr.toFixed(1)+'%</div></div>';
h+='</div></div>';
document.getElementById('gs').innerHTML=h;
}})();
function lM(){{
const m=D[sel.value];
let h='<div style="text-align:center;margin:12px 0;font-size:22px;font-weight:bold;color:#1565c0;">'+MN[m.mn]+' '+m.y+'</div>';
if(m.reg){{const rc=RC[m.reg]||'#999';h+='<div style="text-align:center;margin-bottom:10px;"><span class="rb" style="background:'+rc+';">'+m.reg+'</span> <span style="font-size:13px;color:#666;">Score: '+m.rsc+' | VIX: '+m.rvx+'</span></div>';}}
h+='<div class="sbox">';
h+='<div class="sc"><h4>Acciones S&P 500</h4><div class="v">'+m.n+'</div></div>';
if(m.ts.a!==null){{h+='<div class="sc" style="border-color:#2e7d32;"><h4>Top 25 &rarr; Ret Mes</h4><div class="v '+vc(m.ts.a)+'">'+fm(m.ts.a)+'%</div><div class="d">WR: '+m.ts.w.toFixed(1)+'%</div></div>';}}
if(m.bs.a!==null){{h+='<div class="sc" style="border-color:#c62828;"><h4>Bottom 25 &rarr; Ret Mes</h4><div class="v '+vc(m.bs.a)+'">'+fm(m.bs.a)+'%</div><div class="d">WR: '+m.bs.w.toFixed(1)+'%</div></div>';}}
if(m.ts.a!==null&&m.bs.a!==null){{const d=m.ts.a-m.bs.a;h+='<div class="sc" style="border-color:'+(d>0?'#2e7d32':'#c62828')+';"><h4>Top-Bot 25</h4><div class="v '+vc(d)+'">'+fm(d)+'%</div><div class="d" style="font-weight:bold;">'+(d>0?'Momentum':'Reversion')+'</div></div>';}}
h+='</div>';
h+='<div class="sbox">';
if(m.t10.a!==null){{h+='<div class="sc" style="border-color:#2e7d32;"><h4>Top 10 &rarr; Ret Mes</h4><div class="v '+vc(m.t10.a)+'">'+fm(m.t10.a)+'%</div><div class="d">WR: '+m.t10.w.toFixed(1)+'%</div></div>';}}
if(m.b10.a!==null){{h+='<div class="sc" style="border-color:#c62828;"><h4>Bottom 10 &rarr; Ret Mes</h4><div class="v '+vc(m.b10.a)+'">'+fm(m.b10.a)+'%</div><div class="d">WR: '+m.b10.w.toFixed(1)+'%</div></div>';}}
if(m.t10.a!==null&&m.b10.a!==null){{const d=m.t10.a-m.b10.a;h+='<div class="sc" style="border-color:'+(d>0?'#2e7d32':'#c62828')+';"><h4>Top-Bot 10</h4><div class="v '+vc(d)+'">'+fm(d)+'%</div><div class="d" style="font-weight:bold;">'+(d>0?'Momentum':'Reversion')+'</div></div>';}}
if(m.cmb.a!==null){{h+='<div class="sc" style="border-color:#1565c0;border-width:2px;"><h4>COMBO T10+B10</h4><div class="v '+vc(m.cmb.a)+'">'+fm(m.cmb.a)+'%</div><div class="d">WR: '+m.cmb.w.toFixed(1)+'% (20 acc)</div></div>';}}
h+='</div>';
h+='<div class="cols">';
h+='<div><h2 style="color:#2e7d32;border-color:#2e7d32;">Top 25 Ganadoras ('+WL+')</h2>';
h+=mkT(m.top,'tT');
h+='</div>';
h+='<div><h2 style="color:#c62828;border-color:#c62828;">Bottom 25 Perdedoras ('+WL+')</h2>';
h+=mkT(m.bot,'bT');
h+='</div></div>';
document.getElementById('ct').innerHTML=h;
}}
let sD={{}};
function sC(tid,c){{const t=document.getElementById(tid);if(!t)return;const rows=Array.from(t.querySelectorAll('tr')).slice(1);const k=tid+'_'+c;const d=sD[k]=!(sD[k]||false);rows.sort((a,b)=>{{let va=a.cells[c].textContent.replace(/[\\$%+=\\-]/g,'').trim();let vb=b.cells[c].textContent.replace(/[\\$%+=\\-]/g,'').trim();let na=parseFloat(va),nb=parseFloat(vb);if(!isNaN(na)&&!isNaN(nb))return d?na-nb:nb-na;return d?va.localeCompare(vb):vb.localeCompare(va);}});rows.forEach((r,i)=>{{r.cells[0].textContent=i+1;t.appendChild(r);}});}}
// Yearly stats table
function fK(v){{if(v>=1e6)return '$'+( v/1e6).toFixed(2)+'M';if(v>=1e3)return '$'+(v/1e3).toFixed(0)+'K';return '$'+v.toFixed(0);}}
(function(){{
const yrs=Object.keys(YS).map(Number).sort();
let h='<h2>Historico Anual: Momentum vs Reversion (Ventana '+WL+')</h2>';
h+='<div class="note">Se&ntilde;al: retorno '+WL.toLowerCase()+' del periodo anterior. Neto = bruto - 0.3% (slippage + comisiones). <b>Capital fijo</b>: $20,000/acci&oacute;n cada mes (NO compounding). PnL acumulado.<br><b>COMBO</b> = comprar Top 10 + Bottom 10 cada mes (20 acciones, $400K/mes).</div>';
h+='<table class="qt"><tr><th rowspan=2>A&ntilde;o</th><th rowspan=2>M</th><th colspan=4 style="background:#2e7d32;">Top 25 ($500K/mes)</th><th colspan=4 style="background:#c62828;">Bottom 25 ($500K/mes)</th><th colspan=4 style="background:#1565c0;">Top 10 ($200K/mes)</th><th colspan=4 style="background:#795548;">Bottom 10 ($200K/mes)</th><th colspan=4 style="background:#e65100;">COMBO T10+B10 ($400K/mes)</th></tr>';
h+='<tr><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th></tr>';
yrs.forEach(y=>{{
const d=YS[y];if(!d)return;
const bg=d.d<0?'background:#ffebee;':'background:#e8f5e9;';
h+='<tr style="'+bg+'"><td><b>'+y+'</b></td><td>'+d.n+'</td>';
const gs=['top','bot','t10','b10','cmb'];
const as_=['ta','ba','t10a','b10a','cmba'];
const ns=['top_net','bot_net','t10_net','b10_net','cmb_net'];
const ps=['top_pnl','bot_pnl','t10_pnl','b10_pnl','cmb_pnl'];
const cs=['cap_top','cap_bot','cap_t10','cap_b10','cap_cmb'];
for(let i=0;i<5;i++){{
h+='<td class="'+vc(d[as_[i]])+'">'+fm(d[as_[i]])+'%</td>';
h+='<td class="'+vc(d[ns[i]])+'">'+fm(d[ns[i]])+'%</td>';
h+='<td class="'+vc(d[ps[i]])+'">'+fK(d[ps[i]])+'</td>';
h+='<td style="font-weight:bold;">'+fK(d[cs[i]])+'</td>';
}}
h+='</tr>';
}});
// Final row
const ly=yrs[yrs.length-1];const ld=YS[ly];
if(ld){{
h+='<tr style="font-weight:bold;background:#e3f2fd;border-top:3px solid #1565c0;">';
h+='<td>FINAL</td><td>'+yrs.length+' a&ntilde;os</td>';
const cs=['cap_top','cap_bot','cap_t10','cap_b10','cap_cmb'];
for(let i=0;i<5;i++){{
h+='<td colspan=3></td><td style="font-size:14px;">'+fK(ld[cs[i]])+'</td>';
}}
h+='</tr>';
}}
h+='</table>';
// Regime distribution per year
h+='<h2>Distribuci\\u00f3n de Reg\\u00edmenes por A\\u00f1o</h2>';
h+='<div style="font-size:11px;margin-bottom:8px;">';
yrs.forEach(y=>{{
const ym=D.filter(m=>m.y===y);
const rc={{}};ym.forEach(m=>{{if(m.reg)rc[m.reg]=(rc[m.reg]||0)+1;}});
h+='<div style="margin:2px 0;"><b>'+y+'</b>: ';
RO.forEach(r=>{{if(rc[r])h+='<span class="rb2" style="background:'+(RC[r]||'#999')+';">'+r+' '+rc[r]+'</span> ';}});
h+='</div>';
}});
h+='</div>';
document.getElementById('yt').innerHTML=h;
}})();
// Regime stats table
(function(){{
const regs=Object.keys(RS);
if(regs.length===0)return;
let h='<h2>Rendimiento por R\\u00e9gimen de Mercado (Ventana '+WL+')</h2>';
h+='<div class="note">Retorno medio mensual (%) y Win Rate por grupo de acciones, seg\\u00fan el r\\u00e9gimen de mercado del mes se\\u00f1al.</div>';
h+='<table class="qt" style="min-width:600px;"><tr><th style="background:#333;">R\\u00e9gimen</th><th style="background:#333;">N</th><th style="background:#2e7d32;">Top25 Avg</th><th style="background:#2e7d32;">Top25 WR</th><th style="background:#c62828;">Bot25 Avg</th><th style="background:#c62828;">Bot25 WR</th><th style="background:#1565c0;">T10 Avg</th><th style="background:#1565c0;">T10 WR</th><th style="background:#795548;">B10 Avg</th><th style="background:#795548;">B10 WR</th><th style="background:#e65100;">CMB Avg</th><th style="background:#e65100;">CMB WR</th></tr>';
RO.forEach(r=>{{
const s=RS[r];if(!s)return;
const rc=RC[r]||'#999';
h+='<tr><td><span class="rb2" style="background:'+rc+';">'+r+'</span></td><td>'+s.n+'</td>';
['top','bot','t10','b10','cmb'].forEach(g=>{{
const a=s[g+'_avg'];const w=s[g+'_wr'];
h+='<td class="'+vc(a)+'">'+fm(a)+'%</td><td>'+w.toFixed(1)+'%</td>';
}});
h+='</tr>';
}});
h+='</table>';
document.getElementById('rt').innerHTML=h;
}})();
lM();
</script></body></html>"""

    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  -> {outfile} ({len(all_months_data)} meses, {len(html)/1024/1024:.1f} MB)")

# ================================================================
# HTML MIXTO: Top 10 de 12M + Bottom 10 de 3M
# ================================================================
print(f"\n{'='*60}")
print("Generando momentum_mensual_mix.html (Top10 12M + Bot10 3M)...")

mix_months_data = []
for month in months_list:
    year = month.year
    m_num = month.month
    if year < 2001:
        continue

    month_data = monthly[monthly['month'] == month].copy()
    last_day = month_data['last_date'].max()
    if pd.isna(last_day):
        continue
    sp500_at_date = get_sp500_members(last_day.strftime('%Y-%m-%d'))
    month_data = month_data[month_data['symbol'].isin(sp500_at_date & loaded_symbols)]

    # Necesitamos ret_12m y ret_3m validos
    data_12m = month_data.dropna(subset=['ret_12m']).copy()
    data_3m = month_data.dropna(subset=['ret_3m']).copy()

    if len(data_12m) < 50 or len(data_3m) < 50:
        continue

    # Top 10 por ventana 12M (momentum: mejores rendimiento anual)
    top10_12m = data_12m.sort_values('ret_12m', ascending=False).head(10)
    # Bottom 10 por ventana 3M (reversion: peores rendimiento trimestral)
    bot10_3m = data_3m.sort_values('ret_3m', ascending=True).head(10)

    trade_month = month + 1
    trade_year = trade_month.year
    trade_m_num = trade_month.month

    def make_mix_stocks(df_slice, ret_col_name):
        stocks = []
        for _, row in df_slice.iterrows():
            sym = row['symbol']
            nr = round(float(row['next_ret_1m']), 2) if pd.notna(row['next_ret_1m']) else None
            stocks.append([
                sym,
                get_subsector(sym),
                round(float(row[ret_col_name]), 2),
                nr,
                round(float(row['close']), 2),
            ])
        return stocks

    top_stocks = make_mix_stocks(top10_12m, 'ret_12m')
    bot_stocks = make_mix_stocks(bot10_3m, 'ret_3m')

    def calc_stats(rets):
        if not rets:
            return {'a': None, 'w': None, 'n': 0}
        return {
            'a': round(float(np.mean(rets)), 2),
            'w': round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
            'n': len(rets),
        }

    top_next = [s[3] for s in top_stocks if s[3] is not None]
    bot_next = [s[3] for s in bot_stocks if s[3] is not None]
    combo_next = top_next + bot_next

    # Regimen del mes senal
    reg_info = month_regimes.get(month, {})
    reg = reg_info.get('regime', '')
    rsc = reg_info.get('total', 0)
    rvx = reg_info.get('vix', 0)

    mix_months_data.append({
        'm': str(trade_month), 'y': trade_year, 'mn': trade_m_num,
        'n': len(month_data),
        'top': top_stocks, 'bot': bot_stocks,
        'ts': calc_stats(top_next), 'bs': calc_stats(bot_next),
        'cmb': calc_stats(combo_next),
        'reg': reg, 'rsc': rsc, 'rvx': rvx,
    })

print(f"  Meses procesados: {len(mix_months_data)}")

# Gestion monetaria mix (capital fijo, NO compounding)
COST_PCT = 0.3
CAP_PER_STOCK = 20000
MIX_GROUPS = {'top': 'ts', 'bot': 'bs', 'cmb': 'cmb'}
MIX_N = {'top': 10, 'bot': 10, 'cmb': 20}
MIX_INIT = {k: CAP_PER_STOCK * MIX_N[k] for k in MIX_GROUPS}
pnl_mix = {k: 0.0 for k in MIX_GROUPS}

yearly_mix = {}
all_mix_sorted = sorted(mix_months_data, key=lambda m: m['m'])

for md in all_mix_sorted:
    y = md['y']
    if y not in yearly_mix:
        yearly_mix[y] = {k: [] for k in MIX_GROUPS}
        yearly_mix[y].update({k + '_pnl': [] for k in MIX_GROUPS})

    for grp, stat_key in MIX_GROUPS.items():
        gross = md[stat_key]['a']
        if gross is not None:
            yearly_mix[y][grp].append(gross)
            net = gross - COST_PCT
            month_pnl = net / 100 * MIX_INIT[grp]
            yearly_mix[y][grp + '_pnl'].append(month_pnl)
            pnl_mix[grp] += month_pnl
    yearly_mix[y]['pnl_end'] = dict(pnl_mix)

yearly_mix_stats = {}
for y in sorted(yearly_mix.keys()):
    tr, br = yearly_mix[y]['top'], yearly_mix[y]['bot']
    cmb = yearly_mix[y].get('cmb', [])
    if not tr or not br:
        continue

    ta, ba = float(np.mean(tr)), float(np.mean(br))
    cmba = float(np.mean(cmb)) if cmb else 0
    ys = {
        'ta': round(ta, 2), 'tw': round(sum(1 for r in tr if r > 0) / len(tr) * 100, 1),
        'ba': round(ba, 2), 'bw': round(sum(1 for r in br if r > 0) / len(br) * 100, 1),
        'n': len(tr), 'd': round(ta - ba, 2),
        'cmba': round(cmba, 2), 'cmbw': round(sum(1 for r in cmb if r > 0) / len(cmb) * 100, 1) if cmb else 0,
    }
    for grp in MIX_GROUPS:
        year_pnl = sum(yearly_mix[y][grp + '_pnl'])
        ys[grp + '_net'] = round(year_pnl / MIX_INIT[grp] * 100, 2)
        ys[grp + '_pnl'] = round(year_pnl)
    pnl_end = yearly_mix[y]['pnl_end']
    for grp in MIX_GROUPS:
        ys['cap_' + grp] = round(MIX_INIT[grp] + pnl_end[grp])
    yearly_mix_stats[y] = ys

for grp, label in [('top','Top10 12M'), ('bot','Bot10 3M'), ('cmb','MIX')]:
    final = MIX_INIT[grp] + pnl_mix[grp]
    print(f"  Capital final {label}: ${final:,.0f} (inicio ${MIX_INIT[grp]:,} | PnL ${pnl_mix[grp]:,.0f})")

# Global stats mix
mix_decile = {'top': [], 'bot': [], 'cmb': []}
for md in mix_months_data:
    t = [s[3] for s in md['top'] if s[3] is not None]
    b = [s[3] for s in md['bot'] if s[3] is not None]
    mix_decile['top'].extend(t)
    mix_decile['bot'].extend(b)
    mix_decile['cmb'].extend(t + b)

mix_global = {}
for k in ['top', 'bot', 'cmb']:
    d = mix_decile[k]
    if d:
        mix_global[k+'_avg'] = round(float(np.mean(d)), 3)
        mix_global[k+'_wr'] = round(sum(1 for r in d if r > 0) / len(d) * 100, 1)
        mix_global[k+'_n'] = len(d)
    else:
        mix_global[k+'_avg'] = 0; mix_global[k+'_wr'] = 0; mix_global[k+'_n'] = 0
mix_global['diff'] = round(mix_global['top_avg'] - mix_global['bot_avg'], 3)

print(f"  Top10 12M ret mes: {mix_global['top_avg']:.3f}% | Bot10 3M: {mix_global['bot_avg']:.3f}%")
print(f"  MIX (T10 12M + B10 3M): {mix_global['cmb_avg']:.3f}% | WR: {mix_global['cmb_wr']:.1f}%")

# Stats por regimen MIX
mix_regime_stats = {}
for reg in REGIME_ORDER:
    reg_months = [md for md in mix_months_data if md.get('reg') == reg]
    if not reg_months:
        continue
    rs = {'n': len(reg_months)}
    for grp, stat_key in [('top','ts'), ('bot','bs'), ('cmb','cmb')]:
        rets = [md[stat_key]['a'] for md in reg_months if md[stat_key]['a'] is not None]
        if rets:
            rs[grp + '_avg'] = round(float(np.mean(rets)), 2)
            rs[grp + '_wr'] = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
        else:
            rs[grp + '_avg'] = 0; rs[grp + '_wr'] = 0
    mix_regime_stats[reg] = rs

# HTML mix
mix_months_json = json.dumps(mix_months_data)
mix_yearly_json = json.dumps(yearly_mix_stats)
mix_global_json = json.dumps(mix_global)
mix_regime_stats_json = json.dumps(mix_regime_stats)
mix_regime_colors_json = json.dumps(REGIME_COLORS)
mix_regime_order_json = json.dumps(REGIME_ORDER)

mix_html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8">
<title>Momentum MIX S&P 500 - Top10 12M + Bot10 3M</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #fff; color: #222; max-width: 1500px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #e65100; text-align: center; margin-bottom: 5px; }}
h2 {{ color: #333; margin-top: 20px; margin-bottom: 8px; border-bottom: 2px solid #e65100; padding-bottom: 4px; font-size: 15px; }}
.sub {{ text-align: center; color: #666; margin-bottom: 14px; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 11px; }}
th {{ background: #e65100; color: #fff; padding: 5px 4px; text-align: center; border: 1px solid #ccc; cursor: pointer; font-size: 10px; white-space: nowrap; }}
th:hover {{ background: #bf360c; }}
td {{ padding: 4px 5px; text-align: center; border: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #fff3e0; }}
tr:hover {{ background: #ffe0b2; }}
.pos {{ color: #2e7d32; font-weight: bold; }}
.neg {{ color: #c62828; font-weight: bold; }}
.neutral {{ color: #999; }}
td.left {{ text-align: left; }}
.sbox {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; justify-content: center; }}
.sc {{ background: #fff3e0; border: 1px solid #ddd; border-radius: 8px; padding: 10px 18px; text-align: center; min-width: 130px; }}
.sc h4 {{ margin: 0 0 4px 0; color: #e65100; font-size: 11px; }}
.sc .v {{ font-size: 20px; font-weight: bold; }}
.sc .d {{ font-size: 10px; color: #666; margin-top: 2px; }}
.note {{ background: #fff3e0; padding: 8px 12px; border-radius: 6px; border-left: 4px solid #e65100; margin-bottom: 12px; font-size: 11px; }}
.nb {{ text-align: center; margin: 12px 0 16px 0; padding: 14px; background: #fff3e0; border-radius: 8px; border: 1px solid #ddd; }}
.nb select {{ font-size: 13px; padding: 5px 10px; border-radius: 4px; border: 1px solid #ccc; margin: 0 6px; }}
.nb button {{ font-size: 12px; padding: 5px 14px; border-radius: 4px; border: 1px solid #e65100; background: #e65100; color: #fff; cursor: pointer; margin: 0 3px; }}
.nb button:hover {{ background: #bf360c; }}
.yb {{ margin-top: 8px; }}
.yb button {{ font-size: 10px; padding: 3px 8px; border-radius: 3px; border: 1px solid #999; background: #fff; color: #333; cursor: pointer; margin: 2px; }}
.yb button:hover, .yb button.active {{ background: #e65100; color: #fff; border-color: #e65100; }}
.st {{ max-height: 750px; overflow-y: auto; border: 1px solid #ccc; border-radius: 4px; }}
.st table {{ margin-bottom: 0; }}
.st th {{ position: sticky; top: 0; z-index: 1; }}
.qt {{ width: auto !important; margin: 0 auto !important; min-width: 600px; }}
.qt td {{ padding: 5px 10px; font-size: 12px; }}
.qt th {{ font-size: 11px; padding: 6px 8px; }}
.cols {{ display: flex; gap: 20px; }}
.cols > div {{ flex: 1; min-width: 0; }}
@media (max-width: 1000px) {{ .cols {{ flex-direction: column; }} }}
.gh {{ text-align: center; margin: 16px 0; padding: 14px; background: #fff3e0; border-radius: 8px; border: 2px solid #e65100; }}
.gh .t {{ font-size: 14px; color: #e65100; font-weight: bold; margin-bottom: 6px; }}
.gh .r {{ display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; }}
.gh .c {{ text-align: center; }}
.gh .c .l {{ font-size: 11px; color: #666; }}
.gh .c .v {{ font-size: 22px; font-weight: bold; }}
.lnk {{ text-align: center; margin: 14px 0; padding: 10px; background: #f5f7fa; border-radius: 8px; border: 1px solid #ddd; font-size: 14px; }}
.lnk a {{ margin: 0 6px; color: #1565c0; text-decoration: none; font-weight: bold; padding: 6px 18px; border: 2px solid #1565c0; border-radius: 6px; display: inline-block; }}
.lnk a:hover {{ background: #e3f2fd; }}
.lnk a.act {{ background: #e65100; color: #fff; border-color: #e65100; }}
.rb {{ display: inline-block; padding: 3px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; color: #fff; margin: 0 2px; }}
.rb2 {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold; color: #fff; }}
.rfb {{ margin-top: 6px; }}
.rfb button {{ font-size: 10px; padding: 3px 8px; border-radius: 3px; border: 1px solid #999; color: #fff; cursor: pointer; margin: 2px; }}
.rfb button:hover, .rfb button.active {{ opacity: 0.8; box-shadow: 0 0 0 2px #333; }}
</style></head><body>
<h1>Estrategia MIX: Top10 12M + Bot10 3M</h1>
<p class="sub">Momentum 12M (mejores) + Reversi&oacute;n 3M (peores) | 20 acciones | $400K | 2001-2026</p>
<div class="lnk">
<a href="momentum_mensual_1m.html">1 Mes</a>
<a href="momentum_mensual_3m.html">3 Meses</a>
<a href="momentum_mensual_6m.html">6 Meses</a>
<a href="momentum_mensual_12m.html">12 Meses</a>
<a href="momentum_mensual_mix.html" class=act>MIX 12M+3M</a>
</div>

<div id="gs"></div>
<div class="nb">
<button onclick="pM()">&larr; Anterior</button>
<select id="ms" onchange="lM()"></select>
<button onclick="nM()">Siguiente &rarr;</button>
<div class="yb" id="yb"></div>
<div class="rfb" id="rfb"></div>
</div>
<div id="ct"></div>
<div id="yt"></div>
<div id="rt"></div>
<script>
const MN=['','Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];
const D={mix_months_json};
const YS={mix_yearly_json};
const GS={mix_global_json};
const RC={mix_regime_colors_json};
const RO={mix_regime_order_json};
const RS={mix_regime_stats_json};
const sel=document.getElementById('ms');
D.forEach((m,i)=>{{const o=document.createElement('option');o.value=i;o.text=m.y+'-'+(m.mn<10?'0':'')+m.mn+' '+MN[m.mn]+(m.reg?' ['+m.reg+']':'');sel.appendChild(o);}});
const yrs=[...new Set(D.map(m=>m.y))].sort();
const yb=document.getElementById('yb');
const ab=document.createElement('button');ab.textContent='TODOS';ab.className='active';ab.onclick=()=>rF();yb.appendChild(ab);
yrs.forEach(y=>{{const b=document.createElement('button');b.textContent=y;b.onclick=()=>fY(y);yb.appendChild(b);}});
const rfb=document.getElementById('rfb');
RO.forEach(r=>{{const b=document.createElement('button');b.textContent=r;b.style.background=RC[r];b.style.borderColor=RC[r];b.onclick=()=>fReg(r);rfb.appendChild(b);}});
let aF=null,fI=D.map((_,i)=>i);
function apF(){{sel.innerHTML='';fI.forEach(i=>{{const m=D[i];const o=document.createElement('option');o.value=i;o.text=m.y+'-'+(m.mn<10?'0':'')+m.mn+' '+MN[m.mn]+(m.reg?' ['+m.reg+']':'');sel.appendChild(o);}});if(fI.length>0){{sel.value=fI[fI.length-1];lM();}}
document.querySelectorAll('#yb button').forEach(b=>{{if(b.textContent==='TODOS')b.classList.toggle('active',!aF);else b.classList.toggle('active',aF&&aF.type==='year'&&parseInt(b.textContent)===aF.value);}});
document.querySelectorAll('#rfb button').forEach(b=>{{b.classList.toggle('active',aF&&aF.type==='regime'&&b.textContent===aF.value);}});}}
function fY(y){{aF={{type:'year',value:y}};fI=D.map((m,i)=>m.y===y?i:-1).filter(i=>i>=0);apF();}}
function fReg(r){{aF={{type:'regime',value:r}};fI=D.map((m,i)=>m.reg===r?i:-1).filter(i=>i>=0);apF();}}
function rF(){{aF=null;fI=D.map((_,i)=>i);apF();}}
sel.value=D.length-1;
function pM(){{const c=fI.indexOf(parseInt(sel.value));if(c>0){{sel.value=fI[c-1];lM();}}}}
function nM(){{const c=fI.indexOf(parseInt(sel.value));if(c<fI.length-1){{sel.value=fI[c+1];lM();}}}}
function vc(v){{return v>0?'pos':v<0?'neg':'neutral';}}
function fm(v,d){{d=d||2;return(v>=0?'+':'')+v.toFixed(d);}}
function mkT(stocks,id,hdr){{
let h='<div class="st"><table id="'+id+'"><tr><th onclick="sC(\\''+id+'\\',0)">#</th><th onclick="sC(\\''+id+'\\',1)">Ticker</th><th onclick="sC(\\''+id+'\\',2)">Industria</th><th onclick="sC(\\''+id+'\\',3)">'+hdr+'</th><th onclick="sC(\\''+id+'\\',4)">Ret Mes %</th><th onclick="sC(\\''+id+'\\',5)">Cierre $</th></tr>';
stocks.forEach((s,i)=>{{
const rc=vc(s[2]);const nc=s[3]!==null?vc(s[3]):'neutral';
const nr=s[3]!==null?'<span class="'+nc+'">'+fm(s[3])+'%</span>':'<span class="neutral">-</span>';
h+='<tr><td>'+(i+1)+'</td><td><b>'+s[0]+'</b></td><td class="left" style="font-size:10px;">'+s[1]+'</td><td class="'+rc+'">'+fm(s[2])+'%</td><td>'+nr+'</td><td>$'+s[4].toFixed(2)+'</td></tr>';
}});
h+='</table></div>';return h;
}}
// Global stats header
(function(){{
const g=GS;
let h='<div class="gh"><div class="t">Resultado Global - MIX Top10 12M + Bot10 3M ('+D.length+' meses, 2001-2026)</div>';
h+='<div class="r">';
h+='<div class="c"><div class="l">Top 10 (12M Momentum)</div><div class="v '+vc(g.top_avg)+'">'+fm(g.top_avg,3)+'%</div><div class="l">WR: '+g.top_wr.toFixed(1)+'%</div></div>';
h+='<div class="c"><div class="l">Bot 10 (3M Reversi\\u00f3n)</div><div class="v '+vc(g.bot_avg)+'">'+fm(g.bot_avg,3)+'%</div><div class="l">WR: '+g.bot_wr.toFixed(1)+'%</div></div>';
h+='<div class="c" style="border-left:2px solid #e65100;padding-left:20px;"><div class="l" style="font-weight:bold;color:#e65100;">MIX COMBINADO (20 acc, $400K)</div><div class="v '+vc(g.cmb_avg)+'">'+fm(g.cmb_avg,3)+'%</div><div class="l">WR: '+g.cmb_wr.toFixed(1)+'%</div></div>';
h+='</div></div>';
document.getElementById('gs').innerHTML=h;
}})();
function lM(){{
const m=D[sel.value];
let h='<div style="text-align:center;margin:12px 0;font-size:22px;font-weight:bold;color:#e65100;">'+MN[m.mn]+' '+m.y+'</div>';
if(m.reg){{const rc=RC[m.reg]||'#999';h+='<div style="text-align:center;margin-bottom:10px;"><span class="rb" style="background:'+rc+';">'+m.reg+'</span> <span style="font-size:13px;color:#666;">Score: '+m.rsc+' | VIX: '+m.rvx+'</span></div>';}}
h+='<div class="sbox">';
h+='<div class="sc"><h4>Acciones S&P 500</h4><div class="v">'+m.n+'</div></div>';
if(m.ts.a!==null){{h+='<div class="sc" style="border-color:#2e7d32;"><h4>Top10 12M &rarr; Ret Mes</h4><div class="v '+vc(m.ts.a)+'">'+fm(m.ts.a)+'%</div><div class="d">WR: '+m.ts.w.toFixed(1)+'%</div></div>';}}
if(m.bs.a!==null){{h+='<div class="sc" style="border-color:#c62828;"><h4>Bot10 3M &rarr; Ret Mes</h4><div class="v '+vc(m.bs.a)+'">'+fm(m.bs.a)+'%</div><div class="d">WR: '+m.bs.w.toFixed(1)+'%</div></div>';}}
if(m.cmb.a!==null){{h+='<div class="sc" style="border-color:#e65100;border-width:2px;"><h4>MIX COMBINADO</h4><div class="v '+vc(m.cmb.a)+'">'+fm(m.cmb.a)+'%</div><div class="d">WR: '+m.cmb.w.toFixed(1)+'% (20 acc)</div></div>';}}
h+='</div>';
h+='<div class="cols">';
h+='<div><h2 style="color:#2e7d32;border-color:#2e7d32;">Top 10 - Momentum 12M (mejores a\\u00f1o)</h2>';
h+=mkT(m.top,'tT','Se\\u00f1al 12M %');
h+='</div>';
h+='<div><h2 style="color:#c62828;border-color:#c62828;">Bottom 10 - Reversi\\u00f3n 3M (peores trimestre)</h2>';
h+=mkT(m.bot,'bT','Se\\u00f1al 3M %');
h+='</div></div>';
document.getElementById('ct').innerHTML=h;
}}
let sD={{}};
function sC(tid,c){{const t=document.getElementById(tid);if(!t)return;const rows=Array.from(t.querySelectorAll('tr')).slice(1);const k=tid+'_'+c;const d=sD[k]=!(sD[k]||false);rows.sort((a,b)=>{{let va=a.cells[c].textContent.replace(/[\\$%+=\\-]/g,'').trim();let vb=b.cells[c].textContent.replace(/[\\$%+=\\-]/g,'').trim();let na=parseFloat(va),nb=parseFloat(vb);if(!isNaN(na)&&!isNaN(nb))return d?na-nb:nb-na;return d?va.localeCompare(vb):vb.localeCompare(va);}});rows.forEach((r,i)=>{{r.cells[0].textContent=i+1;t.appendChild(r);}});}}
// Yearly stats table
function fK(v){{if(v>=1e6)return '$'+( v/1e6).toFixed(2)+'M';if(v>=1e3)return '$'+(v/1e3).toFixed(0)+'K';return '$'+v.toFixed(0);}}
(function(){{
const yrs=Object.keys(YS).map(Number).sort();
let h='<h2>Hist\\u00f3rico Anual: MIX Top10 12M + Bot10 3M</h2>';
h+='<div class="note"><b>Estrategia MIX</b>: cada mes se compran las 10 acciones con mejor rendimiento 12M (momentum) y las 10 con peor rendimiento 3M (reversi\\u00f3n a la media). Total 20 acciones, $20K por acci\\u00f3n. <b>Capital fijo $400K/mes</b> (NO compounding). Neto = bruto - 0.3%.</div>';
h+='<table class="qt"><tr><th rowspan=2>A\\u00f1o</th><th rowspan=2>M</th><th colspan=4 style="background:#2e7d32;">Top10 12M ($200K/mes)</th><th colspan=4 style="background:#c62828;">Bot10 3M ($200K/mes)</th><th colspan=4 style="background:#e65100;">MIX Combinado ($400K/mes)</th></tr>';
h+='<tr><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th><th>Bruto</th><th>Neto</th><th>PnL $</th><th>Acum</th></tr>';
yrs.forEach(y=>{{
const d=YS[y];if(!d)return;
const bg=d.d>0?'background:#e8f5e9;':'background:#ffebee;';
h+='<tr style="'+bg+'"><td><b>'+y+'</b></td><td>'+d.n+'</td>';
const gs=['top','bot','cmb'];
const as_=['ta','ba','cmba'];
const ns=['top_net','bot_net','cmb_net'];
const ps=['top_pnl','bot_pnl','cmb_pnl'];
const cs=['cap_top','cap_bot','cap_cmb'];
for(let i=0;i<3;i++){{
h+='<td class="'+vc(d[as_[i]])+'">'+fm(d[as_[i]])+'%</td>';
h+='<td class="'+vc(d[ns[i]])+'">'+fm(d[ns[i]])+'%</td>';
h+='<td class="'+vc(d[ps[i]])+'">'+fK(d[ps[i]])+'</td>';
h+='<td style="font-weight:bold;">'+fK(d[cs[i]])+'</td>';
}}
h+='</tr>';
}});
const ly=yrs[yrs.length-1];const ld=YS[ly];
if(ld){{
h+='<tr style="font-weight:bold;background:#ffe0b2;border-top:3px solid #e65100;">';
h+='<td>FINAL</td><td>'+yrs.length+' a\\u00f1os</td>';
const cs=['cap_top','cap_bot','cap_cmb'];
for(let i=0;i<3;i++){{
h+='<td colspan=3></td><td style="font-size:14px;">'+fK(ld[cs[i]])+'</td>';
}}
h+='</tr>';
}}
h+='</table>';
// Regime distribution per year
h+='<h2>Distribuci\\u00f3n de Reg\\u00edmenes por A\\u00f1o</h2>';
h+='<div style="font-size:11px;margin-bottom:8px;">';
yrs.forEach(y=>{{
const ym=D.filter(m=>m.y===y);
const rc={{}};ym.forEach(m=>{{if(m.reg)rc[m.reg]=(rc[m.reg]||0)+1;}});
h+='<div style="margin:2px 0;"><b>'+y+'</b>: ';
RO.forEach(r=>{{if(rc[r])h+='<span class="rb2" style="background:'+(RC[r]||'#999')+';">'+r+' '+rc[r]+'</span> ';}});
h+='</div>';
}});
h+='</div>';
document.getElementById('yt').innerHTML=h;
}})();
// Regime stats table
(function(){{
const regs=Object.keys(RS);
if(regs.length===0)return;
let h='<h2>Rendimiento por R\\u00e9gimen de Mercado (MIX)</h2>';
h+='<div class="note">Retorno medio mensual (%) y Win Rate por grupo de acciones, seg\\u00fan el r\\u00e9gimen de mercado del mes se\\u00f1al.</div>';
h+='<table class="qt" style="min-width:400px;"><tr><th style="background:#333;">R\\u00e9gimen</th><th style="background:#333;">N</th><th style="background:#2e7d32;">T10 12M Avg</th><th style="background:#2e7d32;">T10 WR</th><th style="background:#c62828;">B10 3M Avg</th><th style="background:#c62828;">B10 WR</th><th style="background:#e65100;">MIX Avg</th><th style="background:#e65100;">MIX WR</th></tr>';
RO.forEach(r=>{{
const s=RS[r];if(!s)return;
const rcl=RC[r]||'#999';
h+='<tr><td><span class="rb2" style="background:'+rcl+';">'+r+'</span></td><td>'+s.n+'</td>';
['top','bot','cmb'].forEach(g=>{{
const a=s[g+'_avg'];const w=s[g+'_wr'];
h+='<td class="'+vc(a)+'">'+fm(a)+'%</td><td>'+w.toFixed(1)+'%</td>';
}});
h+='</tr>';
}});
h+='</table>';
document.getElementById('rt').innerHTML=h;
}})();
lM();
</script></body></html>"""

with open('momentum_mensual_mix.html', 'w', encoding='utf-8') as f:
    f.write(mix_html)
print(f"  -> momentum_mensual_mix.html ({len(mix_months_data)} meses, {len(mix_html)/1024/1024:.1f} MB)")

print("\nTodos los HTMLs generados.")

# ================================================================
# RESUMEN: Rentabilidad anualizada (media simple de % neto anual)
# ================================================================
print(f"\n{'='*80}")
print("RENTABILIDAD ANUALIZADA (media simple % neto anual, capital fijo, -0.3% costes)")
print(f"{'='*80}")

STRAT_NAMES = {'top': 'Top 25', 'bot': 'Bot 25', 't10': 'Top 10', 'b10': 'Bot 10', 'cmb': 'COMBO T10+B10'}
MIX_STRAT_NAMES = {'top': 'Top10 12M', 'bot': 'Bot10 3M', 'cmb': 'MIX Combinado'}

for window in WINDOWS:
    label = WINDOW_LABELS[window]
    ys = all_yearly_results[window]
    years = sorted(ys.keys())
    n_years = len(years)
    print(f"\n--- Ventana {label} ({n_years} anos) ---")
    for grp in ['top', 'bot', 't10', 'b10', 'cmb']:
        yearly_nets = [ys[y][grp + '_net'] for y in years]
        total = sum(yearly_nets)
        avg = total / n_years
        print(f"  {STRAT_NAMES[grp]:16s}: {' + '.join(f'{v:+.2f}%' for v in yearly_nets)}")
        print(f"  {'':16s}  TOTAL: {total:+.2f}% / {n_years} anos = {avg:+.2f}% anualizado")

# MIX
years_mix = sorted(yearly_mix_stats.keys())
n_years_mix = len(years_mix)
print(f"\n--- MIX: Top10 12M + Bot10 3M ({n_years_mix} anos) ---")
for grp in ['top', 'bot', 'cmb']:
    yearly_nets = [yearly_mix_stats[y][grp + '_net'] for y in years_mix]
    total = sum(yearly_nets)
    avg = total / n_years_mix
    print(f"  {MIX_STRAT_NAMES[grp]:16s}: {' + '.join(f'{v:+.2f}%' for v in yearly_nets)}")
    print(f"  {'':16s}  TOTAL: {total:+.2f}% / {n_years_mix} anos = {avg:+.2f}% anualizado")

# Tabla comparativa final
print(f"\n{'='*80}")
print("COMPARATIVA FINAL - Rentabilidad anualizada media")
print(f"{'='*80}")
print(f"{'Estrategia':30s} | {'1M':>8s} | {'3M':>8s} | {'6M':>8s} | {'12M':>8s} | {'MIX':>8s}")
print(f"{'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

for grp, name in [('top','Top 25'), ('bot','Bot 25'), ('t10','Top 10'), ('b10','Bot 10'), ('cmb','COMBO T10+B10')]:
    vals = []
    for w in WINDOWS:
        ys = all_yearly_results[w]
        years = sorted(ys.keys())
        avg = sum(ys[y][grp + '_net'] for y in years) / len(years)
        vals.append(f'{avg:+.2f}%')
    # MIX no tiene top25/bot25, solo top/bot/cmb
    if grp in ['top', 'bot', 'cmb']:
        mix_grp = grp
        mix_avg = sum(yearly_mix_stats[y][mix_grp + '_net'] for y in years_mix) / n_years_mix
        vals.append(f'{mix_avg:+.2f}%')
    else:
        vals.append('  -  ')
    print(f"{name:30s} | {vals[0]:>8s} | {vals[1]:>8s} | {vals[2]:>8s} | {vals[3]:>8s} | {vals[4]:>8s}")

# Fila MIX separada
print(f"{'MIX (T10 12M + B10 3M)':30s} | {'  -  ':>8s} | {'  -  ':>8s} | {'  -  ':>8s} | {'  -  ':>8s} | ", end='')
mix_cmb_avg = sum(yearly_mix_stats[y]['cmb_net'] for y in years_mix) / n_years_mix
print(f"{mix_cmb_avg:+.2f}%")

# ================================================================
# Inyectar tabla comparativa en los 5 HTMLs
# ================================================================
print("\nInyectando tabla comparativa en los HTMLs...")

# Construir datos de la tabla: por cada estrategia, rentabilidad anualizada por ventana
summary_rows = []
for grp, name in [('top','Top 25'), ('bot','Bot 25'), ('t10','Top 10'), ('b10','Bot 10'), ('cmb','COMBO T10+B10')]:
    row = {'name': name, 'vals': {}}
    for w in WINDOWS:
        ys = all_yearly_results[w]
        yrs_w = sorted(ys.keys())
        row['vals'][str(w)] = round(sum(ys[y][grp + '_net'] for y in yrs_w) / len(yrs_w), 2)
    if grp in ['top', 'bot', 'cmb']:
        row['vals']['mix'] = round(sum(yearly_mix_stats[y][grp + '_net'] for y in years_mix) / n_years_mix, 2)
    else:
        row['vals']['mix'] = None
    summary_rows.append(row)

# Fila MIX especial
mix_row = {'name': 'MIX (T10 12M + B10 3M)', 'vals': {}}
for w in WINDOWS:
    mix_row['vals'][str(w)] = None
mix_row['vals']['mix'] = round(sum(yearly_mix_stats[y]['cmb_net'] for y in years_mix) / n_years_mix, 2)
summary_rows.append(mix_row)

# Detalle anual por estrategia y ventana (para tabla desplegable)
detail_data = {}
for w in WINDOWS:
    ys = all_yearly_results[w]
    yrs_w = sorted(ys.keys())
    for grp in ['top', 'bot', 't10', 'b10', 'cmb']:
        key = f'{grp}_{w}'
        detail_data[key] = {str(y): ys[y][grp + '_net'] for y in yrs_w}

# Detalle MIX
for grp in ['top', 'bot', 'cmb']:
    key = f'{grp}_mix'
    detail_data[key] = {str(y): yearly_mix_stats[y][grp + '_net'] for y in years_mix}

summary_json = json.dumps(summary_rows)
detail_json = json.dumps(detail_data)

# HTML/JS del bloque comparativo
summary_block = """
<div id="cmp" style="margin-top:30px;"></div>
<script>
(function(){
const SR=""" + summary_json + """;
const DT=""" + detail_json + """;
const WS=['1','3','6','12','mix'];
const WL={'1':'1 Mes','3':'3 Meses','6':'6 Meses','12':'12 Meses','mix':'MIX 12M+3M'};
const GK={'Top 25':'top','Bot 25':'bot','Top 10':'t10','Bot 10':'b10','COMBO T10+B10':'cmb','MIX (T10 12M + B10 3M)':'cmb'};
function vc(v){return v>0?'pos':v<0?'neg':'neutral';}
function fm(v){return v!==null?(v>=0?'+':'')+v.toFixed(2)+'%':'-';}
// find best value per column
let best={};
WS.forEach(w=>{let mx=-999;SR.forEach(r=>{if(r.vals[w]!==null&&r.vals[w]>mx)mx=r.vals[w];});best[w]=mx;});
let h='<h2 style="border-color:#e65100;color:#e65100;">Comparativa: Rentabilidad Anualizada (media simple % neto, capital fijo)</h2>';
h+='<div class="note">Rentabilidad anualizada = suma de % neto anual / n\\u00famero de a\\u00f1os. Capital fijo $20K/acci\\u00f3n cada mes (NO compounding). Neto = bruto - 0.3%. Click en cualquier celda para ver detalle anual.</div>';
h+='<table class="qt" style="min-width:500px;"><tr><th style="background:#e65100;">Estrategia</th>';
WS.forEach(w=>{h+='<th style="background:#e65100;">'+WL[w]+'</th>';});
h+='</tr>';
SR.forEach((r,ri)=>{
const isMix=r.name.indexOf('MIX')===0;
const bg=isMix?'background:#fff3e0;font-weight:bold;':'';
h+='<tr style="'+bg+'">';
h+='<td style="text-align:left;font-weight:bold;">'+r.name+'</td>';
WS.forEach(w=>{
const v=r.vals[w];
if(v===null){h+='<td style="color:#999;">-</td>';}
else{
const cls=vc(v);
const isBest=Math.abs(v-best[w])<0.01;
const style=isBest?'font-size:14px;text-decoration:underline;':'';
const gk=GK[r.name];
const dk=gk+'_'+w;
h+='<td class="'+cls+'" style="cursor:pointer;'+style+'" onclick="toggleDetail(\\''+dk+'\\',this)">'+fm(v)+(isBest?' \\u2605':'')+'</td>';
}
});
h+='</tr>';
});
h+='</table>';
h+='<div id="dtl" style="margin-top:10px;"></div>';
document.getElementById('cmp').innerHTML=h;

// Detail toggle
window.toggleDetail=function(dk,el){
const dtl=document.getElementById('dtl');
if(dtl.dataset.active===dk){dtl.innerHTML='';dtl.dataset.active='';return;}
dtl.dataset.active=dk;
const data=DT[dk];
if(!data){dtl.innerHTML='<div class="note">No hay datos para esta combinaci\\u00f3n.</div>';return;}
const yrs=Object.keys(data).map(Number).sort();
const total=yrs.reduce((s,y)=>s+data[y],0);
const avg=total/yrs.length;
let dh='<h2 style="font-size:13px;border-color:#795548;color:#795548;">Detalle anual: '+dk.replace('_',' \\u2192 ventana ')+'</h2>';
dh+='<table class="qt" style="min-width:300px;max-width:700px;"><tr><th style="background:#795548;">A\\u00f1o</th><th style="background:#795548;">Neto %</th><th style="background:#795548;">Acumulado %</th></tr>';
let acum=0;
yrs.forEach(y=>{
const v=data[y];acum+=v;
dh+='<tr><td><b>'+y+'</b></td><td class="'+vc(v)+'">'+fm(v)+'</td><td class="'+vc(acum)+'">'+fm(acum)+'</td></tr>';
});
dh+='<tr style="font-weight:bold;background:#efebe9;border-top:2px solid #795548;"><td>MEDIA</td><td class="'+vc(avg)+'">'+fm(avg)+'</td><td class="'+vc(total)+'">'+fm(total)+' ('+yrs.length+' a\\u00f1os)</td></tr>';
dh+='</table>';
dtl.innerHTML=dh;
};
})();
</script>
"""

# Inyectar en cada HTML antes de </body>
all_files = [WINDOW_FILES[w] for w in WINDOWS] + ['momentum_mensual_mix.html']
for fpath in all_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('</body></html>', summary_block + '</body></html>')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  {fpath} actualizado")

print("\nDashboards actualizados con tabla comparativa.")
