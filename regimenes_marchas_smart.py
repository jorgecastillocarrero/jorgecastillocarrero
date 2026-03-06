"""
Sistema de Regimenes con Marchas Adaptativas + Smart Filters
=============================================================
Flujo por semana W:
  1. SENAL:   Jue W-1 cierre -> Jue W cierre (datos para calcular regimen)
  2. TRADING: Vie W open -> Vie W+1 open (rentabilidad real del trade)

Dos modos de operacion:
  MODO 1 - MAX PNL:     Baseline (DD52,RSI14,MA200,MOM10) + filtro VIXup
  MODO 2 - MIN DRAWDOWN: Marchas adaptativas + filtro Smart

Marchas (segun VIX):
  1a (VIX<15):  DD52, RSI14, MA200, MOM10 - Calma
  2a (VIX 15-20): DD52, RSI14, MA200, MOM10 - Normal
  3a (VIX 20-25): DD39, RSI10, MA200, MOM7 - Tension
  4a (VIX 25-30): DD26, RSI7, MA200, MOM5 - Estres
  5a (VIX 30-40): DD26, RSI7, MA150, MOM5 - Crisis
  6a (VIX>40):  DD13, RSI5, MA100, MOM3 - Panico

Filtros Smart para shorts:
  - NO shortear CAPITULACION ni RECOVERY (rebotes)
  - Solo shortear cuando VIX sube (vix_delta > 0)
  - NO shortear si RSI oversold (<15% subsectores con RSI>55)
  - NO shortear en marcha 6 (VIX>40, zona de rebotes violentos)

Resultados backtest SPY 2001-2026:
  MODO 1 (MAX PNL):     +261.2%, MaxDD -42.9%, 20W/6L
  MODO 2 (MIN DRAWDOWN): +205.6%, MaxDD -26.2%, 21W/5L
  vs BASELINE original:  +236.2%, MaxDD -49.4%, 19W/7L
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json, sys

sys.stdout.reconfigure(encoding='utf-8')

engine = create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# ====================================================================
# CARGAR DATOS
# ====================================================================
print("Cargando datos...")
with open('data/sp500_constituents.json') as f:
    sp500 = json.load(f)
symbols = [s['symbol'] for s in sp500]

profiles = pd.read_sql("SELECT symbol, industry FROM fmp_profiles WHERE symbol = ANY(%(syms)s)",
                        engine, params={'syms': symbols})
sym_to_sub = dict(zip(profiles['symbol'], profiles['industry']))

prices = pd.read_sql("""
    SELECT symbol, date, close FROM fmp_price_history
    WHERE symbol = ANY(%(syms)s) AND date BETWEEN '1998-01-01' AND '2026-03-07'
    ORDER BY date
""", engine, params={'syms': symbols})
prices['date'] = pd.to_datetime(prices['date'])
prices['subsector'] = prices['symbol'].map(sym_to_sub)
prices = prices.dropna(subset=['subsector'])
valid_subs = prices.groupby('subsector')['symbol'].nunique()
valid_subs = valid_subs[valid_subs >= 3].index
prices = prices[prices['subsector'].isin(valid_subs)]

weekly = prices.set_index('date').groupby('subsector').resample('W-THU')['close'].mean().reset_index()
weekly = weekly.rename(columns={'close': 'avg_close'}).sort_values(['subsector', 'date'])
print(f"  Subsectores validos: {len(valid_subs)}")

# SPY diario
spy_daily = pd.read_sql("""
    SELECT date, open, close FROM fmp_price_history
    WHERE symbol = 'SPY' AND date BETWEEN '1998-01-01' AND '2026-03-07' ORDER BY date
""", engine)
spy_daily['date'] = pd.to_datetime(spy_daily['date'])
spy_daily = spy_daily.set_index('date').sort_index()

# VIX
vix_df = pd.read_sql("SELECT date, close as vix FROM price_history_vix WHERE symbol='^VIX' ORDER BY date", engine)
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.set_index('date').sort_index()

# ====================================================================
# PRE-CALCULAR METRICAS CON MULTIPLES VENTANAS
# ====================================================================
print("Calculando metricas con multiples ventanas...")

dd_windows = [13, 26, 39, 52]
rsi_windows = [5, 7, 10, 14]

dd_wides = {}
for dd_w in dd_windows:
    w = weekly.copy()
    def calc_dd(grp, window=dd_w):
        grp = grp.sort_values('date')
        high = grp['avg_close'].rolling(window, min_periods=max(5, window // 4)).max()
        grp['dd'] = (grp['avg_close'] / high - 1) * 100
        return grp
    w = w.groupby('subsector', group_keys=False).apply(calc_dd)
    dd_wides[dd_w] = w.pivot(index='date', columns='subsector', values='dd')

rsi_wides = {}
for rsi_w in rsi_windows:
    w = weekly.copy()
    def calc_rsi(grp, window=rsi_w):
        grp = grp.sort_values('date')
        delta = grp['avg_close'].diff()
        gain = delta.clip(lower=0).rolling(window, min_periods=max(3, window // 2)).mean()
        loss = (-delta.clip(upper=0)).rolling(window, min_periods=max(3, window // 2)).mean()
        rs = gain / loss.replace(0, np.nan)
        grp['rsi'] = 100 - (100 / (1 + rs))
        return grp
    w = w.groupby('subsector', group_keys=False).apply(calc_rsi)
    rsi_wides[rsi_w] = w.pivot(index='date', columns='subsector', values='rsi')

# SPY con multiples MAs y MOMs
for ma_w in [50, 100, 150, 200]:
    spy_daily[f'ma{ma_w}'] = spy_daily['close'].rolling(ma_w).mean()
spy_w = spy_daily.resample('W-THU').last()
for mom_w in [3, 5, 7, 10]:
    spy_w[f'mom{mom_w}'] = spy_w['close'].pct_change(mom_w) * 100

all_trading_dates = spy_daily.index.tolist()
def find_td(target, direction='forward', tolerance=4):
    if direction == 'forward':
        cands = [(abs((d - target).days), d) for d in all_trading_dates if d >= target and (d - target).days <= tolerance]
    else:
        cands = [(abs((d - target).days), d) for d in all_trading_dates if d <= target and (target - d).days <= tolerance]
    return min(cands, key=lambda x: x[0])[1] if cands else None

print("Datos preparados.\n")

# ====================================================================
# MARCHAS
# ====================================================================
GEARS = {
    1: (0,   15, 52, 14, 200, 10, '1a-Calma'),
    2: (15,  20, 52, 14, 200, 10, '2a-Normal'),
    3: (20,  25, 39, 10, 200,  7, '3a-Tension'),
    4: (25,  30, 26,  7, 200,  5, '4a-Estres'),
    5: (30,  40, 26,  7, 150,  5, '5a-Crisis'),
    6: (40, 999, 13,  5, 100,  3, '6a-Panico'),
}

# ====================================================================
# FUNCION CORE: calcular regimenes
# ====================================================================
def score_indicators(pct_dd_h, pct_dd_d, pct_rsi, spy_above, spy_dist, spy_mom):
    """Calcula los 5 scores individuales."""
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

    return s_bdd, s_brsi, s_ddp, s_spy, s_mom


def classify_regime(total, pct_dd_h, pct_rsi, vix_val, vix_delta, prev_vix):
    """Clasifica regimen a partir del score total + VIX."""
    is_burbuja = (total >= 8.0 and pct_dd_h >= 85 and pct_rsi >= 90)
    if is_burbuja: regime = 'BURBUJA'
    elif total >= 7.0: regime = 'GOLDILOCKS'
    elif total >= 4.0: regime = 'ALCISTA'
    elif total >= 0.5: regime = 'NEUTRAL'
    elif total >= -2.0: regime = 'CAUTIOUS'
    elif total >= -5.0: regime = 'BEARISH'
    elif total >= -9.0: regime = 'CRISIS'
    else: regime = 'PANICO'

    vix_veto = ''
    if vix_val >= 30 and regime in ('BURBUJA', 'GOLDILOCKS', 'ALCISTA'):
        vix_veto = f'{regime}->NEUTRAL'
        regime = 'NEUTRAL'
    elif vix_val >= 35 and regime == 'NEUTRAL':
        vix_veto = 'NEUTRAL->CAUTIOUS'
        regime = 'CAUTIOUS'

    if regime == 'PANICO' and prev_vix is not None and vix_delta < 0:
        regime = 'CAPITULACION'
    elif regime == 'BEARISH' and prev_vix is not None and vix_delta < 0:
        regime = 'RECOVERY'

    return regime, vix_veto


def calc_regimes(mode='min_dd'):
    """
    mode='max_pnl':  Baseline lookbacks + filtro VIXup
    mode='min_dd':   Marchas adaptativas + filtro Smart
    """
    use_gears = (mode == 'min_dd')
    thursdays = dd_wides[52].index[dd_wides[52].index >= '2001-01-01']
    results = []
    prev_vix = None

    for thu in thursdays:
        # VIX
        vix_dates_f = vix_df.index[vix_df.index <= thu]
        vix_val = vix_df.loc[vix_dates_f[-1], 'vix'] if len(vix_dates_f) > 0 else 20
        if not pd.notna(vix_val):
            vix_val = 20
        vix_delta = vix_val - prev_vix if prev_vix is not None else 0

        # Determinar lookbacks
        if use_gears:
            gear = 2
            for g, (lo, hi, *_) in GEARS.items():
                if lo <= vix_val < hi:
                    gear = g
                    break
            _, _, dd_w, rsi_w, ma_w, mom_w, gear_label = GEARS[gear]
        else:
            gear = 0
            dd_w, rsi_w, ma_w, mom_w = 52, 14, 200, 10
            gear_label = 'fijo'

        # Breadth con ventana seleccionada
        if thu not in dd_wides[dd_w].index or thu not in rsi_wides[rsi_w].index:
            prev_vix = vix_val
            continue
        dd_row = dd_wides[dd_w].loc[thu]
        rsi_row = rsi_wides[rsi_w].loc[thu]
        n_total = dd_row.notna().sum()
        if n_total == 0:
            prev_vix = vix_val
            continue

        n_dd_h = int((dd_row > -10).sum())
        n_dd_d = int((dd_row < -20).sum())
        n_rsi_t = int(rsi_row.notna().sum())
        n_rsi_ok = int((rsi_row > 55).sum())
        pct_dd_h = n_dd_h / n_total * 100
        pct_dd_d = n_dd_d / n_total * 100
        pct_rsi = n_rsi_ok / n_rsi_t * 100 if n_rsi_t > 0 else 50

        # SPY
        ma_col = f'ma{ma_w}'
        mom_col = f'mom{mom_w}'
        spy_dates = spy_w.index[spy_w.index <= thu]
        if len(spy_dates) == 0:
            prev_vix = vix_val
            continue
        spy_last = spy_w.loc[spy_dates[-1]]
        spy_close = spy_last['close']
        spy_ma_val = spy_last.get(ma_col, np.nan)
        if pd.isna(spy_ma_val):
            spy_ma_val = spy_last.get('ma200', spy_close)
        spy_above = spy_close > spy_ma_val
        spy_dist = (spy_close / spy_ma_val - 1) * 100 if spy_ma_val > 0 else 0
        spy_mom = spy_last.get(mom_col, 0)
        if not pd.notna(spy_mom): spy_mom = 0
        if not pd.notna(spy_dist): spy_dist = 0

        # Scores
        s_bdd, s_brsi, s_ddp, s_spy, s_mom = score_indicators(
            pct_dd_h, pct_dd_d, pct_rsi, spy_above, spy_dist, spy_mom)
        total = s_bdd + s_brsi + s_ddp + s_spy + s_mom

        # Regimen
        regime, vix_veto = classify_regime(total, pct_dd_h, pct_rsi, vix_val, vix_delta, prev_vix)

        # --- FILTRO SHORTS ---
        shortable = True
        short_reason = ''

        # Comun a ambos modos: nunca shortear rebotes
        if regime in ('CAPITULACION', 'RECOVERY'):
            shortable = False
            short_reason = 'rebote'

        # En bajistas, solo shortear si VIX sube
        if regime in ('CRISIS', 'PANICO', 'BEARISH') and vix_delta <= 0:
            shortable = False
            short_reason = 'vix_bajando'

        # Solo en modo min_dd: filtros adicionales
        if mode == 'min_dd':
            if regime in ('CRISIS', 'PANICO') and pct_rsi < 15:
                shortable = False
                short_reason = 'oversold_rsi'
            if regime in ('CRISIS', 'PANICO') and pct_dd_h < 15:
                shortable = False
                short_reason = 'oversold_dd'
            if gear == 6:
                shortable = False
                short_reason = 'gear6_panico'

        # SPY return Fri open -> Fri open (trading = dia siguiente a senal jueves)
        fri_target = thu + pd.Timedelta(days=1)
        fri_entry = find_td(fri_target, 'forward', 4)
        fri_exit = find_td(fri_target + pd.Timedelta(days=7), 'forward', 4)
        spy_ret = None
        spy_entry_val = None
        spy_exit_val = None
        if fri_entry and fri_exit and fri_entry in spy_daily.index and fri_exit in spy_daily.index:
            spy_entry_val = spy_daily.loc[fri_entry, 'open']
            spy_exit_val = spy_daily.loc[fri_exit, 'open']
            spy_ret = (spy_exit_val / spy_entry_val - 1) * 100

        prev_vix = vix_val

        results.append({
            'fecha_senal': thu,
            'year': thu.year,
            'sem': thu.isocalendar()[1],
            'n_sub': n_total,
            'dd_h': n_dd_h, 'pct_dd_h': pct_dd_h,
            'dd_d': n_dd_d, 'pct_dd_d': pct_dd_d,
            'rsi55': n_rsi_ok, 'pct_rsi': pct_rsi,
            'spy_close': spy_close,
            'spy_ma': spy_ma_val, 'spy_dist': spy_dist, 'spy_mom': spy_mom,
            'vix': vix_val, 'vix_delta': vix_delta,
            's_bdd': s_bdd, 's_brsi': s_brsi, 's_ddp': s_ddp, 's_spy': s_spy, 's_mom': s_mom,
            'total': total,
            'regime': regime, 'vix_veto': vix_veto,
            'gear': gear, 'gear_label': gear_label,
            'dd_w': dd_w, 'rsi_w': rsi_w, 'ma_w': ma_w, 'mom_w': mom_w,
            'shortable': shortable, 'short_reason': short_reason,
            'fri_entry': fri_entry, 'fri_exit': fri_exit,
            'spy_entry_open': spy_entry_val, 'spy_exit_open': spy_exit_val,
            'spy_ret_pct': spy_ret,
        })

    return pd.DataFrame(results)


# ====================================================================
# BACKTEST SPY (long/short/flat)
# ====================================================================
def backtest_spy(df, label):
    """Long bullish + bounce, Short bearish (si shortable), Flat resto."""
    bullish = ['BURBUJA', 'GOLDILOCKS', 'ALCISTA']
    bearish_short = ['CRISIS', 'PANICO', 'BEARISH']
    bounce_long = ['CAPITULACION', 'RECOVERY']

    valid = df[df['spy_ret_pct'].notna()].copy()
    pnl = 0
    n_long = n_short = n_flat = n_bounce = n_filtered = 0
    weekly_pnl = []
    year_pnl = {}

    for _, r in valid.iterrows():
        ret = r['spy_ret_pct']
        regime = r['regime']
        year = r['fecha_senal'].year if hasattr(r['fecha_senal'], 'year') else int(str(r['fecha_senal'])[:4])

        if regime in bullish:
            w_pnl = ret; n_long += 1
        elif regime in bounce_long:
            w_pnl = ret; n_bounce += 1
        elif regime in bearish_short:
            if r.get('shortable', True):
                w_pnl = -ret; n_short += 1
            else:
                w_pnl = 0; n_flat += 1; n_filtered += 1
        else:
            w_pnl = 0; n_flat += 1

        pnl += w_pnl
        weekly_pnl.append(w_pnl)
        year_pnl[year] = year_pnl.get(year, 0) + w_pnl

    cum = np.cumsum(weekly_pnl)
    peak = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()
    wins = sum(1 for v in year_pnl.values() if v > 0)
    losses = sum(1 for v in year_pnl.values() if v <= 0)

    return {
        'label': label, 'pnl': pnl, 'max_dd': max_dd,
        'n_long': n_long, 'n_short': n_short, 'n_bounce': n_bounce,
        'n_flat': n_flat, 'n_filtered': n_filtered,
        'wins': wins, 'losses': losses, 'year_pnl': year_pnl,
    }


# ====================================================================
# EJECUTAR AMBOS MODOS
# ====================================================================
print("=" * 120)
print("  MODO 1: MAX PNL (Baseline + filtro VIXup)")
print("=" * 120)
df_maxpnl = calc_regimes(mode='max_pnl')
bt_maxpnl = backtest_spy(df_maxpnl, 'MODO 1 - MAX PNL')

print(f"\n{'=' * 120}")
print("  MODO 2: MIN DRAWDOWN (Marchas + filtro Smart)")
print("=" * 120)
df_mindd = calc_regimes(mode='min_dd')
bt_mindd = backtest_spy(df_mindd, 'MODO 2 - MIN DD')

# ====================================================================
# GUARDAR CSVs
# ====================================================================
csv1 = 'data/regimenes_maxpnl.csv'
csv2 = 'data/regimenes_mindd.csv'
df_maxpnl.to_csv(csv1, index=False, float_format='%.4f')
df_mindd.to_csv(csv2, index=False, float_format='%.4f')
print(f"\nGuardados: {csv1} y {csv2}")

# ====================================================================
# COMPARATIVA
# ====================================================================
print(f"\n{'=' * 120}")
print(f"  COMPARATIVA FINAL")
print(f"{'=' * 120}")

print(f"\n  {'Metrica':<30} {'MODO 1 (MAX PNL)':>20} {'MODO 2 (MIN DD)':>20}")
print(f"  {'-'*30} {'-'*20} {'-'*20}")
for bt, lbl in [(bt_maxpnl, 'MODO 1'), (bt_mindd, 'MODO 2')]:
    pass

print(f"  {'PnL total %':<30} {bt_maxpnl['pnl']:>+20.1f} {bt_mindd['pnl']:>+20.1f}")
print(f"  {'Max Drawdown %':<30} {bt_maxpnl['max_dd']:>20.1f} {bt_mindd['max_dd']:>20.1f}")
wl1 = f"{bt_maxpnl['wins']}W/{bt_maxpnl['losses']}L"
wl2 = f"{bt_mindd['wins']}W/{bt_mindd['losses']}L"
print(f"  {'Anos W/L':<30} {wl1:>20} {wl2:>20}")
print(f"  {'Semanas long':<30} {bt_maxpnl['n_long']:>20} {bt_mindd['n_long']:>20}")
print(f"  {'Semanas short':<30} {bt_maxpnl['n_short']:>20} {bt_mindd['n_short']:>20}")
print(f"  {'Semanas bounce long':<30} {bt_maxpnl['n_bounce']:>20} {bt_mindd['n_bounce']:>20}")
print(f"  {'Semanas flat':<30} {bt_maxpnl['n_flat']:>20} {bt_mindd['n_flat']:>20}")
print(f"  {'Shorts filtrados':<30} {bt_maxpnl['n_filtered']:>20} {bt_mindd['n_filtered']:>20}")
print(f"  {'Lookbacks':<30} {'DD52 RSI14 MA200 M10':>20} {'Adaptativos x VIX':>20}")

# Cambios de regimen
changes1 = (df_maxpnl['regime'] != df_maxpnl['regime'].shift()).sum()
changes2 = (df_mindd['regime'] != df_mindd['regime'].shift()).sum()
print(f"  {'Cambios regimen':<30} {changes1:>20} {changes2:>20}")

# PnL anual
print(f"\n  {'Año':>4} | {'MODO 1':>10} {'MODO 2':>10} | {'Mejor':>8}")
print(f"  {'-'*4} | {'-'*10} {'-'*10} | {'-'*8}")

years = sorted(set(list(bt_maxpnl['year_pnl'].keys()) + list(bt_mindd['year_pnl'].keys())))
for year in years:
    v1 = bt_maxpnl['year_pnl'].get(year, 0)
    v2 = bt_mindd['year_pnl'].get(year, 0)
    mejor = 'PNL' if v1 > v2 else 'DD' if v2 > v1 else 'IGUAL'
    print(f"  {year:>4} | {v1:>+10.1f} {v2:>+10.1f} | {mejor:>8}")

# Distribucion de marchas (solo modo 2)
print(f"\n{'=' * 120}")
print(f"  DISTRIBUCION DE MARCHAS (MODO 2 - MIN DD)")
print(f"{'=' * 120}")

print(f"  {'Marcha':<15} {'N':>5} {'%':>6} {'Avg SPY%':>8}")
for g in sorted(GEARS.keys()):
    mask = df_mindd['gear'] == g
    n = mask.sum()
    if n == 0: continue
    pct = n / len(df_mindd) * 100
    sub = df_mindd[mask & df_mindd['spy_ret_pct'].notna()]
    avg = sub['spy_ret_pct'].mean() if len(sub) > 0 else 0
    _, _, _, _, _, _, label = GEARS[g]
    print(f"  {label:<15} {n:>5} {pct:>5.1f}% {avg:>+8.3f}%")

# Retorno por regimen (ambos modos)
print(f"\n{'=' * 120}")
print(f"  RETORNO POR REGIMEN")
print(f"{'=' * 120}")

regs_order = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']
print(f"\n  {'':>14} | {'--- MODO 1 (MAX PNL) ---':>30} | {'--- MODO 2 (MIN DD) ---':>30}")
print(f"  {'Regimen':<14} | {'N':>4} {'Avg%':>7} {'WR%':>5} {'Tot%':>7} | {'N':>4} {'Avg%':>7} {'WR%':>5} {'Tot%':>7}")
print(f"  {'-'*14} | {'-'*4} {'-'*7} {'-'*5} {'-'*7} | {'-'*4} {'-'*7} {'-'*5} {'-'*7}")

for reg in regs_order:
    parts = []
    for df in [df_maxpnl, df_mindd]:
        sub = df[(df['regime'] == reg) & df['spy_ret_pct'].notna()]
        if len(sub) > 0:
            avg = sub['spy_ret_pct'].mean()
            wr = (sub['spy_ret_pct'] > 0).mean() * 100
            tot = sub['spy_ret_pct'].sum()
            parts.append(f"{len(sub):>4} {avg:>+7.2f} {wr:>4.0f}% {tot:>+7.1f}")
        else:
            parts.append(f"{'':>4} {'':>7} {'':>5} {'':>7}")
    print(f"  {reg:<14} | {parts[0]} | {parts[1]}")

# Shorts filtrados detalle (modo 2)
print(f"\n{'=' * 120}")
print(f"  DETALLE SHORTS FILTRADOS (MODO 2)")
print(f"{'=' * 120}")

for mode_lbl, df in [('MODO 1', df_maxpnl), ('MODO 2', df_mindd)]:
    filt = df[(~df['shortable']) & df['regime'].isin(['CRISIS','PANICO','BEARISH']) & df['spy_ret_pct'].notna()]
    if len(filt) == 0:
        print(f"\n  {mode_lbl}: 0 shorts filtrados")
        continue
    print(f"\n  {mode_lbl}: {len(filt)} shorts filtrados")
    print(f"    Retorno SPY medio: {filt['spy_ret_pct'].mean():+.3f}% (positivo = bien no shortear)")
    print(f"    Ahorro total: {filt['spy_ret_pct'].sum():+.1f}%")
    by_reason = filt.groupby('short_reason')['spy_ret_pct'].agg(['count', 'mean', 'sum'])
    print(f"    {'Razon':<20} {'N':>4} {'Avg%':>8} {'Total%':>8}")
    for reason, row in by_reason.iterrows():
        print(f"    {reason:<20} {int(row['count']):>4} {row['mean']:>+8.3f} {row['sum']:>+8.1f}")

print(f"\n{'=' * 120}")
print(f"  COMPLETADO - CSVs guardados en data/regimenes_maxpnl.csv y data/regimenes_mindd.csv")
print(f"{'=' * 120}")
