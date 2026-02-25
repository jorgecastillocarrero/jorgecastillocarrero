import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
import json
import time

t0 = time.time()
engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# Load SP500 constituents
with open('data/sp500_constituents.json') as f:
    members = json.load(f)
sp500_syms = [m['symbol'] for m in members]

print(f"S&P 500: {len(sp500_syms)} acciones")

# ============================================================
# LOAD DATA
# ============================================================
with engine.connect() as conn:
    prices = pd.read_sql("""SELECT symbol, date, open, high, low, close, volume
        FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-06-01' AND date <= '2026-02-13'
        ORDER BY symbol, date""", conn, params={"syms": sp500_syms}, parse_dates=['date'])

    earn = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s) AND eps_actual IS NOT NULL AND eps_estimated IS NOT NULL
        AND date >= '2023-01-01' ORDER BY symbol, date DESC""",
        conn, params={"syms": sp500_syms}, parse_dates=['date'])

    ratios = pd.read_sql("""SELECT symbol, date, pe_ratio, price_earnings_to_growth_ratio,
        net_profit_margin, operating_profit_margin, dividend_yield, payout_ratio,
        debt_equity_ratio
        FROM fmp_ratios
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-01-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": sp500_syms}, parse_dates=['date'])

    metrics = pd.read_sql("""SELECT symbol, date, market_cap
        FROM fmp_key_metrics
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-06-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": sp500_syms}, parse_dates=['date'])

print(f"Prices: {len(prices):,} | Earnings: {len(earn):,} | Ratios: {len(ratios):,} | t={time.time()-t0:.0f}s")

# ============================================================
# PARABOLIC SAR & SUPERTREND
# ============================================================
def calc_parabolic_sar(high, low, close, af_start=0.02, af_max=0.20, af_step=0.02):
    n = len(close)
    if n < 3: return np.full(n, np.nan), np.full(n, np.nan)
    sar = np.full(n, np.nan); trend = np.ones(n); af_arr = np.zeros(n); ep = np.zeros(n)
    h = high.values; l = low.values
    sar[0] = l[0]; trend[0] = 1; af_arr[0] = af_start; ep[0] = h[0]
    for i in range(1, n):
        if trend[i-1] == 1:
            sar[i] = sar[i-1] + af_arr[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[i-2] if i >= 2 else l[i-1])
            if l[i] < sar[i]:
                trend[i] = -1; sar[i] = ep[i-1]; ep[i] = l[i]; af_arr[i] = af_start
            else:
                trend[i] = 1
                if h[i] > ep[i-1]: ep[i] = h[i]; af_arr[i] = min(af_arr[i-1] + af_step, af_max)
                else: ep[i] = ep[i-1]; af_arr[i] = af_arr[i-1]
        else:
            sar[i] = sar[i-1] + af_arr[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], h[i-1], h[i-2] if i >= 2 else h[i-1])
            if h[i] > sar[i]:
                trend[i] = 1; sar[i] = ep[i-1]; ep[i] = h[i]; af_arr[i] = af_start
            else:
                trend[i] = -1
                if l[i] < ep[i-1]: ep[i] = l[i]; af_arr[i] = min(af_arr[i-1] + af_step, af_max)
                else: ep[i] = ep[i-1]; af_arr[i] = af_arr[i-1]
    return sar, trend

def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    n = len(close)
    if n < period + 1: return np.full(n, np.nan), np.full(n, np.nan)
    h = high.values; l = low.values; c = close.values
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    atr = pd.Series(tr).rolling(period).mean().values
    hl2 = (h + l) / 2; upper = hl2 + multiplier * atr; lower = hl2 - multiplier * atr
    st = np.full(n, np.nan); direction = np.ones(n)
    final_upper = upper.copy(); final_lower = lower.copy()
    start = period; st[start] = lower[start]; direction[start] = 1
    for i in range(start + 1, n):
        if lower[i] > final_lower[i-1] or c[i-1] < final_lower[i-1]: final_lower[i] = lower[i]
        else: final_lower[i] = final_lower[i-1]
        if upper[i] < final_upper[i-1] or c[i-1] > final_upper[i-1]: final_upper[i] = upper[i]
        else: final_upper[i] = final_upper[i-1]
        if direction[i-1] == 1:
            if c[i] < final_lower[i]: direction[i] = -1; st[i] = final_upper[i]
            else: direction[i] = 1; st[i] = final_lower[i]
        else:
            if c[i] > final_upper[i]: direction[i] = 1; st[i] = final_lower[i]
            else: direction[i] = -1; st[i] = final_upper[i]
    return st, direction

# ============================================================
# COMPUTE INDICATORS FOR ALL S&P 500
# ============================================================
print(f"\nCalculando indicadores para {len(sp500_syms)} acciones...")

all_results = []
target = pd.Timestamp('2026-02-13')

for sym in sp500_syms:
    sp = prices[prices['symbol'] == sym].sort_values('date')
    if len(sp) < 60:
        continue

    c = sp['close']; h = sp['high']; l = sp['low']; v = sp['volume']
    n = len(sp)
    close_price = c.iloc[-1]

    # MAs
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1] if n >= 50 else np.nan
    ma200 = c.rolling(200).mean().iloc[-1] if n >= 200 else np.nan
    dist_ma200 = (close_price - ma200) / ma200 * 100 if pd.notna(ma200) else np.nan

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss_s.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # Returns
    ret_1w = (close_price / c.iloc[-6] - 1) * 100 if n >= 6 else np.nan
    ret_4w = (close_price / c.iloc[-21] - 1) * 100 if n >= 21 else np.nan
    ret_12w = (close_price / c.iloc[-61] - 1) * 100 if n >= 61 else np.nan

    # Volatility
    daily_ret = c.pct_change()
    vol_20d = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

    # ATR%
    tr_vals = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_pct = tr_vals.rolling(14).mean().iloc[-1] / close_price * 100

    # Bollinger %B
    bb_mid = c.rolling(20).mean().iloc[-1]; bb_std = c.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std; bb_lower = bb_mid - 2 * bb_std
    bb_pctb = (close_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # Stochastic
    low_14 = l.tail(14).min(); high_14 = h.tail(14).max()
    stoch_k = (close_price - low_14) / (high_14 - low_14) * 100 if (high_14 - low_14) > 0 else 50

    # PSAR
    sar_vals, sar_trend = calc_parabolic_sar(h, l, c)
    psar_dist = (close_price - sar_vals[-1]) / close_price * 100 if pd.notna(sar_vals[-1]) else np.nan
    psar_bull = 1 if sar_trend[-1] == 1 else 0

    # SuperTrend
    st_vals, st_dir = calc_supertrend(h, l, c)
    st_dist = (close_price - st_vals[-1]) / close_price * 100 if pd.notna(st_vals[-1]) else np.nan
    st_bull = 1 if st_dir[-1] == 1 else 0

    # Earnings
    sym_earn = earn[earn['symbol'] == sym]
    past_earn = sym_earn[sym_earn['date'] <= target]
    if len(past_earn) > 0:
        last_earn = past_earn.iloc[0]
        days_since_earn = (target - last_earn['date']).days
        eps_surprise = ((last_earn['eps_actual'] - last_earn['eps_estimated']) / abs(last_earn['eps_estimated']) * 100
                       if pd.notna(last_earn['eps_estimated']) and last_earn['eps_estimated'] != 0 else np.nan)
        beats_4q = sum(1 for _, e in past_earn.head(4).iterrows()
                      if pd.notna(e['eps_actual']) and pd.notna(e['eps_estimated']) and e['eps_actual'] > e['eps_estimated'])
        if len(past_earn) >= 5 and pd.notna(past_earn.iloc[4]['eps_actual']) and past_earn.iloc[4]['eps_actual'] != 0:
            eps_growth_yoy = ((past_earn.iloc[0]['eps_actual'] - past_earn.iloc[4]['eps_actual'])
                             / abs(past_earn.iloc[4]['eps_actual']) * 100)
        else:
            eps_growth_yoy = np.nan
    else:
        days_since_earn = np.nan; eps_surprise = np.nan; beats_4q = np.nan; eps_growth_yoy = np.nan

    # Fundamentals
    sym_rat = ratios[ratios['symbol'] == sym]
    if len(sym_rat) > 0:
        rat = sym_rat.iloc[0]
        peg = rat.get('price_earnings_to_growth_ratio', np.nan)
        net_margin = rat.get('net_profit_margin', np.nan)
        op_margin = rat.get('operating_profit_margin', np.nan)
        div_yield = rat.get('dividend_yield', np.nan)
        payout = rat.get('payout_ratio', np.nan)
        de_ratio = rat.get('debt_equity_ratio', np.nan)
    else:
        peg = net_margin = op_margin = div_yield = payout = de_ratio = np.nan

    sym_met = metrics[metrics['symbol'] == sym]
    mktcap = sym_met.iloc[0].get('market_cap', np.nan) if len(sym_met) > 0 else np.nan

    all_results.append({
        'symbol': sym, 'close': close_price,
        'dist_ma200': dist_ma200, 'rsi': rsi,
        'ret_1w': ret_1w, 'ret_4w': ret_4w, 'ret_12w': ret_12w,
        'vol_20d': vol_20d, 'atr_pct': atr_pct,
        'bb_pctb': bb_pctb, 'stoch_k': stoch_k,
        'psar_dist': psar_dist, 'psar_bull': psar_bull,
        'st_dist': st_dist, 'st_bull': st_bull,
        'days_since_earn': days_since_earn, 'eps_surprise': eps_surprise,
        'beats_4q': beats_4q, 'eps_growth_yoy': eps_growth_yoy,
        'peg': peg, 'net_margin': net_margin, 'op_margin': op_margin,
        'div_yield': div_yield, 'payout': payout, 'de_ratio': de_ratio,
        'mktcap': mktcap,
    })

df = pd.DataFrame(all_results)
print(f"Acciones con datos: {len(df)} | t={time.time()-t0:.0f}s")

# ============================================================
# SCORING: LONG PATTERN
# "Pullback en tendencia fuerte"
# - Momentum 12w alto            (ret_12w top quintile)
# - RSI bajo                     (oversold)
# - PSAR BEAR                    (señal tecnica "rota")
# - SuperTrend BEAR
# - BB%B bajo
# - Stochastic bajo
# - Ret 1w bajo                  (caida reciente)
# - Volatilidad alta             (accion en movimiento)
# - Margenes altos
# - EPS growth positivo / beats
# - PEG bajo (growth razonable)
# ============================================================
print(f"\n{'='*160}")
print(f"  SCORING PATRON LONG: 'Pullback en tendencia fuerte'")
print(f"{'='*160}")

def percentile_rank(series):
    """Return 0-100 percentile rank for each value."""
    return series.rank(pct=True, na_option='keep') * 100

# Compute component scores (all 0-100, higher = better match for LONG)
df['sc_ret12w'] = percentile_rank(df['ret_12w'])              # High momentum = high score
df['sc_rsi_inv'] = 100 - percentile_rank(df['rsi'])           # Low RSI = high score
df['sc_psar_bear'] = (1 - df['psar_bull']) * 100              # PSAR BEAR = 100, BULL = 0
df['sc_st_bear'] = (1 - df['st_bull']) * 100                  # ST BEAR = 100, BULL = 0
df['sc_bb_inv'] = 100 - percentile_rank(df['bb_pctb'])        # Low BB%B = high score
df['sc_stoch_inv'] = 100 - percentile_rank(df['stoch_k'])     # Low Stoch = high score
df['sc_ret1w_inv'] = 100 - percentile_rank(df['ret_1w'])      # Low ret 1w (pullback) = high score
df['sc_vol'] = percentile_rank(df['vol_20d'])                  # High vol = high score
df['sc_margin'] = percentile_rank(df['net_margin'])            # High margin = high score
df['sc_beats'] = df['beats_4q'].fillna(0) / 4 * 100           # 4/4 = 100
df['sc_epsgr'] = percentile_rank(df['eps_growth_yoy'])         # High EPS growth = high

# Composite LONG score (weighted)
df['LONG_SCORE'] = (
    df['sc_ret12w'] * 0.20 +          # Momentum LP (mas importante)
    df['sc_rsi_inv'] * 0.10 +         # RSI oversold
    df['sc_psar_bear'] * 0.12 +       # PSAR bear
    df['sc_st_bear'] * 0.08 +         # SuperTrend bear
    df['sc_bb_inv'] * 0.08 +          # BB%B bajo
    df['sc_stoch_inv'] * 0.08 +       # Stoch bajo
    df['sc_ret1w_inv'] * 0.12 +       # Pullback reciente
    df['sc_vol'] * 0.05 +             # Volatilidad
    df['sc_margin'] * 0.07 +          # Margenes
    df['sc_beats'] * 0.05 +           # Beats
    df['sc_epsgr'] * 0.05             # EPS growth
)

# ============================================================
# SCORING: SHORT PATTERN
# "Overbought sin fundamento"
# - PSAR BULL                    (señal tecnica "confirmada" -> ripe for reversal)
# - Stochastic alto
# - Ret 1w alto                  (rally reciente)
# - Momentum 12w BAJO            (sin tendencia larga)
# - Alto dividend yield / payout
# - Margenes bajos
# - Mas deuda
# - RSI alto
# ============================================================
print(f"\n{'='*160}")
print(f"  SCORING PATRON SHORT: 'Overbought sin fundamento'")
print(f"{'='*160}")

# Component scores for SHORT (higher = better match for SHORT)
df['sc_psar_bull_s'] = df['psar_bull'] * 100                   # PSAR BULL = 100
df['sc_st_bull_s'] = df['st_bull'] * 100                       # ST BULL = 100
df['sc_stoch_s'] = percentile_rank(df['stoch_k'])              # High Stoch = high score
df['sc_rsi_s'] = percentile_rank(df['rsi'])                    # High RSI = high score
df['sc_ret1w_s'] = percentile_rank(df['ret_1w'])               # High ret 1w (rally) = high score
df['sc_ret12w_inv_s'] = 100 - percentile_rank(df['ret_12w'])   # LOW momentum LP = high score
df['sc_divyield_s'] = percentile_rank(df['div_yield'])         # High div yield = high score
df['sc_payout_s'] = percentile_rank(df['payout'])              # High payout = high score
df['sc_margin_inv_s'] = 100 - percentile_rank(df['net_margin']) # LOW margin = high score
df['sc_debt_s'] = percentile_rank(df['de_ratio'])              # High debt = high score
df['sc_bb_s'] = percentile_rank(df['bb_pctb'])                 # High BB%B = high score

# Composite SHORT score (weighted)
df['SHORT_SCORE'] = (
    df['sc_psar_bull_s'] * 0.15 +     # PSAR bull (mas importante)
    df['sc_st_bull_s'] * 0.08 +       # ST bull
    df['sc_stoch_s'] * 0.12 +         # Stoch alto
    df['sc_rsi_s'] * 0.08 +           # RSI alto
    df['sc_bb_s'] * 0.07 +            # BB%B alto
    df['sc_ret1w_s'] * 0.12 +         # Rally reciente
    df['sc_ret12w_inv_s'] * 0.15 +    # Sin momentum LP (mas importante)
    df['sc_divyield_s'] * 0.05 +      # Alto dividendo
    df['sc_payout_s'] * 0.05 +        # Alto payout
    df['sc_margin_inv_s'] * 0.08 +    # Margenes bajos
    df['sc_debt_s'] * 0.05            # Mas deuda
)

# ============================================================
# TOP 5 LONG PICKS
# ============================================================
# Filter: must have ret_12w > 10% (need strong momentum) and PSAR BEAR or ST BEAR
long_candidates = df[(df['ret_12w'] > 10) & ((df['psar_bull'] == 0) | (df['st_bull'] == 0))].copy()
long_top5 = long_candidates.nlargest(5, 'LONG_SCORE')

print(f"\n\n{'='*160}")
print(f"  TOP 5 LONG PICKS - 'Pullback en tendencia fuerte'")
print(f"  Filtro: ret_12w > 10% AND (PSAR BEAR OR ST BEAR)")
print(f"  Candidatos que pasan filtro: {len(long_candidates)}")
print(f"{'='*160}")

print(f"\n  {'#':>3} {'Sym':>6} {'Score':>6} | {'Close':>8} | {'Ret1w':>7} {'Ret4w':>7} {'Ret12w':>7} | {'RSI':>5} {'BB%B':>5} {'Stoch':>5} | {'PSAR':>6} {'SAR%':>6} {'ST':>5} {'ST%':>6} | {'dMA200':>7} | {'Vol20':>6} | {'Beats':>5} {'EPSg%':>7} {'Marg%':>6}")
print(f"  {'-'*3} {'-'*6} {'-'*6} | {'-'*8} | {'-'*7} {'-'*7} {'-'*7} | {'-'*5} {'-'*5} {'-'*5} | {'-'*6} {'-'*6} {'-'*5} {'-'*6} | {'-'*7} | {'-'*6} | {'-'*5} {'-'*7} {'-'*6}")

for i, (_, r) in enumerate(long_top5.iterrows(), 1):
    pt = "BEAR" if r['psar_bull'] == 0 else "BULL"
    sd = "BEAR" if r['st_bull'] == 0 else "BULL"
    bt = f"{int(r['beats_4q'])}/4" if pd.notna(r['beats_4q']) else " N/A"
    eg = f"{r['eps_growth_yoy']:>+6.0f}%" if pd.notna(r['eps_growth_yoy']) else "   N/A"
    nm = f"{r['net_margin']*100:>5.1f}%" if pd.notna(r['net_margin']) else "  N/A"
    print(f"  {i:>3} {r['symbol']:>6} {r['LONG_SCORE']:>5.1f} | ${r['close']:>7.2f} | {r['ret_1w']:>+6.1f}% {r['ret_4w']:>+6.1f}% {r['ret_12w']:>+6.1f}% | {r['rsi']:>5.1f} {r['bb_pctb']:>5.2f} {r['stoch_k']:>5.1f} | {pt:>6} {r['psar_dist']:>+5.1f}% {sd:>5} {r['st_dist']:>+5.1f}% | {r['dist_ma200']:>+6.1f}% | {r['vol_20d']:>5.0f}% | {bt:>5} {eg} {nm}")

# Component scores for top 5
print(f"\n  Desglose scores (0-100):")
print(f"  {'Sym':>6} | {'Mom12':>5} {'RSI':>5} {'PSAR':>5} {'ST':>5} {'BB':>5} {'Stoch':>5} {'Pull1w':>6} {'Vol':>5} {'Marg':>5} {'Beats':>5} {'EPSg':>5}")
for _, r in long_top5.iterrows():
    print(f"  {r['symbol']:>6} | {r['sc_ret12w']:>5.0f} {r['sc_rsi_inv']:>5.0f} {r['sc_psar_bear']:>5.0f} {r['sc_st_bear']:>5.0f} {r['sc_bb_inv']:>5.0f} {r['sc_stoch_inv']:>5.0f} {r['sc_ret1w_inv']:>6.0f} {r['sc_vol']:>5.0f} {r['sc_margin']:>5.0f} {r['sc_beats']:>5.0f} {r['sc_epsgr']:>5.0f}")

# ============================================================
# TOP 5 SHORT PICKS
# ============================================================
# Filter: must have PSAR BULL and ret_12w < 15% (no strong momentum) and stoch > 50
short_candidates = df[(df['psar_bull'] == 1) & (df['ret_12w'] < 15) & (df['stoch_k'] > 50)].copy()
short_top5 = short_candidates.nlargest(5, 'SHORT_SCORE')

print(f"\n\n{'='*160}")
print(f"  TOP 5 SHORT PICKS - 'Overbought sin fundamento'")
print(f"  Filtro: PSAR BULL AND ret_12w < 15% AND Stoch > 50")
print(f"  Candidatos que pasan filtro: {len(short_candidates)}")
print(f"{'='*160}")

print(f"\n  {'#':>3} {'Sym':>6} {'Score':>6} | {'Close':>8} | {'Ret1w':>7} {'Ret4w':>7} {'Ret12w':>7} | {'RSI':>5} {'BB%B':>5} {'Stoch':>5} | {'PSAR':>6} {'SAR%':>6} {'ST':>5} {'ST%':>6} | {'dMA200':>7} | {'DivY%':>6} {'Pout%':>6} {'D/E':>6} {'Marg%':>6}")
print(f"  {'-'*3} {'-'*6} {'-'*6} | {'-'*8} | {'-'*7} {'-'*7} {'-'*7} | {'-'*5} {'-'*5} {'-'*5} | {'-'*6} {'-'*6} {'-'*5} {'-'*6} | {'-'*7} | {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

for i, (_, r) in enumerate(short_top5.iterrows(), 1):
    pt = "BULL" if r['psar_bull'] == 1 else "BEAR"
    sd = "BULL" if r['st_bull'] == 1 else "BEAR"
    dy = f"{r['div_yield']*100:>5.1f}%" if pd.notna(r['div_yield']) else "  N/A"
    po = f"{r['payout']*100:>5.0f}%" if pd.notna(r['payout']) else "  N/A"
    de = f"{r['de_ratio']:>5.2f}" if pd.notna(r['de_ratio']) else "  N/A"
    nm = f"{r['net_margin']*100:>5.1f}%" if pd.notna(r['net_margin']) else "  N/A"
    print(f"  {i:>3} {r['symbol']:>6} {r['SHORT_SCORE']:>5.1f} | ${r['close']:>7.2f} | {r['ret_1w']:>+6.1f}% {r['ret_4w']:>+6.1f}% {r['ret_12w']:>+6.1f}% | {r['rsi']:>5.1f} {r['bb_pctb']:>5.2f} {r['stoch_k']:>5.1f} | {pt:>6} {r['psar_dist']:>+5.1f}% {sd:>5} {r['st_dist']:>+5.1f}% | {r['dist_ma200']:>+6.1f}% | {dy} {po} {de} {nm}")

print(f"\n  Desglose scores (0-100):")
print(f"  {'Sym':>6} | {'PSAR':>5} {'ST':>5} {'Stoch':>5} {'RSI':>5} {'BB':>5} {'Rally':>5} {'NoMom':>5} {'DivY':>5} {'Pout':>5} {'-Marg':>5} {'Debt':>5}")
for _, r in short_top5.iterrows():
    print(f"  {r['symbol']:>6} | {r['sc_psar_bull_s']:>5.0f} {r['sc_st_bull_s']:>5.0f} {r['sc_stoch_s']:>5.0f} {r['sc_rsi_s']:>5.0f} {r['sc_bb_s']:>5.0f} {r['sc_ret1w_s']:>5.0f} {r['sc_ret12w_inv_s']:>5.0f} {r['sc_divyield_s']:>5.0f} {r['sc_payout_s']:>5.0f} {r['sc_margin_inv_s']:>5.0f} {r['sc_debt_s']:>5.0f}")

# ============================================================
# VERIFY: How would these have performed Feb 17 -> Feb 23?
# ============================================================
print(f"\n\n{'='*160}")
print(f"  VERIFICACION: Rendimiento real Open 17-Feb -> Open 23-Feb")
print(f"{'='*160}")

with engine.connect() as conn:
    verify = pd.read_sql("""SELECT symbol, date, open
        FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s)
          AND date IN ('2026-02-17', '2026-02-23')
        ORDER BY symbol, date""",
        conn, params={"syms": list(long_top5['symbol']) + list(short_top5['symbol'])},
        parse_dates=['date'])

entry = verify[verify['date'] == pd.Timestamp('2026-02-17')][['symbol', 'open']].rename(columns={'open': 'entry'})
exit_ = verify[verify['date'] == pd.Timestamp('2026-02-23')][['symbol', 'open']].rename(columns={'open': 'exit'})
perf = entry.merge(exit_, on='symbol')
perf['ret_pct'] = (perf['exit'] / perf['entry'] - 1) * 100

print(f"\n  --- 5 LONGS ---")
long_pnl_total = 0
for _, r in long_top5.iterrows():
    sym = r['symbol']
    p = perf[perf['symbol'] == sym]
    if len(p) > 0:
        ret = p.iloc[0]['ret_pct']
        pnl = 50000 * ret / 100
        long_pnl_total += pnl
        print(f"  LONG  {sym:>6}: entry ${p.iloc[0]['entry']:>8.2f} -> exit ${p.iloc[0]['exit']:>8.2f} = {ret:>+6.2f}% -> P&L ${pnl:>+9,.0f}")
    else:
        print(f"  LONG  {sym:>6}: Sin datos")

print(f"\n  --- 5 SHORTS ---")
short_pnl_total = 0
for _, r in short_top5.iterrows():
    sym = r['symbol']
    p = perf[perf['symbol'] == sym]
    if len(p) > 0:
        ret = -p.iloc[0]['ret_pct']  # short: profit when price drops
        pnl = 50000 * ret / 100
        short_pnl_total += pnl
        print(f"  SHORT {sym:>6}: entry ${p.iloc[0]['entry']:>8.2f} -> exit ${p.iloc[0]['exit']:>8.2f} = {-ret:>+6.2f}% (short gana {ret:>+6.2f}%) -> P&L ${pnl:>+9,.0f}")
    else:
        print(f"  SHORT {sym:>6}: Sin datos")

total_pnl = long_pnl_total + short_pnl_total
print(f"\n  RESUMEN:")
print(f"    P&L Longs:  ${long_pnl_total:>+10,.0f}  (5 x $50K)")
print(f"    P&L Shorts: ${short_pnl_total:>+10,.0f}  (5 x $50K)")
print(f"    P&L TOTAL:  ${total_pnl:>+10,.0f}  ($500K invertidos)")
print(f"    Retorno:    {total_pnl/500000*100:>+.2f}%")

# Also show what the actual top/bot 25 included vs our picks
print(f"\n  Nuestros LONG picks que estaban en el TOP 25 real de subidas:")
top25_real = ['OMC','TPL','GPN','MRNA','GRMN','OXY','CIEN','MOH','FIX','DE',
              'CSGP','GE','SNDK','GLW','FSLR','EBAY','MGM','DPZ','GOOGL','PWR',
              'GOOG','AMAT','LRCX','BAX','EXPD']
for sym in long_top5['symbol']:
    hit = "SI - en TOP 25" if sym in top25_real else "NO"
    print(f"    {sym:>6}: {hit}")

print(f"\n  Nuestros SHORT picks que estaban en el TOP 25 real de bajadas:")
bot25_real = ['EPAM','POOL','AKAM','GPC','IP','PANW','ARES','DDOG','CPB','PKG',
              'CRWD','PAYC','BX','SW','FTNT','ORCL','WDAY','KR','WMT','INVH',
              'NOW','APTV','DOW','CAG','ANET']
for sym in short_top5['symbol']:
    hit = "SI - en TOP 25" if sym in bot25_real else "NO"
    print(f"    {sym:>6}: {hit}")

print(f"\n  Tiempo total: {time.time()-t0:.0f}s")
