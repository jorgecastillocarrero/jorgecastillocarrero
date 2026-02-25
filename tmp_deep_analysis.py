import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
import json

engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

TOP25 = ['OMC','TPL','GPN','MRNA','GRMN','OXY','CIEN','MOH','FIX','DE',
         'CSGP','GE','SNDK','GLW','FSLR','EBAY','MGM','DPZ','GOOGL','PWR',
         'GOOG','AMAT','LRCX','BAX','EXPD']
BOT25 = ['EPAM','POOL','AKAM','GPC','IP','PANW','ARES','DDOG','CPB','PKG',
         'CRWD','PAYC','BX','SW','FTNT','ORCL','WDAY','KR','WMT','INVH',
         'NOW','APTV','DOW','CAG','ANET']

all_syms = list(set(TOP25 + BOT25))

with engine.connect() as conn:
    # Longer price history for 200-day indicators
    prices = pd.read_sql("""SELECT symbol, date, open, high, low, close, volume
        FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-06-01' AND date <= '2026-02-13'
        ORDER BY symbol, date""", conn, params={"syms": all_syms}, parse_dates=['date'])

    # Earnings
    earn = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated, revenue_actual, revenue_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s) AND date >= '2023-01-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": all_syms}, parse_dates=['date'])

    # Key metrics
    metrics = pd.read_sql("""SELECT symbol, date, pe_ratio, pb_ratio, ev_to_ebitda,
        roe, revenue_per_share, net_income_per_share, market_cap
        FROM fmp_key_metrics
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-01-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": all_syms}, parse_dates=['date'])

    # Ratios (PEG, dividend, margins, etc.)
    ratios = pd.read_sql("""SELECT symbol, date, pe_ratio, price_to_book_ratio,
        price_earnings_to_growth_ratio, return_on_equity, return_on_assets,
        debt_equity_ratio, current_ratio, dividend_yield, payout_ratio,
        net_profit_margin, operating_profit_margin, gross_profit_margin,
        ev_to_ebitda, price_to_free_cash_flow_ratio
        FROM fmp_ratios
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-01-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": all_syms}, parse_dates=['date'])

    # Dividends
    dividends = pd.read_sql("""SELECT symbol, date, dividend, declaration_date, record_date, payment_date
        FROM fmp_dividends
        WHERE symbol = ANY(%(syms)s) AND date >= '2025-01-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": all_syms}, parse_dates=['date'])

print(f"Prices: {len(prices):,} | Earnings: {len(earn)} | Metrics: {len(metrics)} | Ratios: {len(ratios)} | Dividends: {len(dividends)}")

# ============================================================
# PARABOLIC SAR & SUPERTREND functions
# ============================================================
def calc_parabolic_sar(high, low, close, af_start=0.02, af_max=0.20, af_step=0.02):
    n = len(close)
    if n < 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    sar = np.full(n, np.nan)
    trend = np.ones(n)
    af_arr = np.zeros(n)
    ep = np.zeros(n)
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
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    h = high.values; l = low.values; c = close.values
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    atr = pd.Series(tr).rolling(period).mean().values
    hl2 = (h + l) / 2
    upper = hl2 + multiplier * atr; lower = hl2 - multiplier * atr
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
# COMPUTE ALL INDICATORS
# ============================================================
results = []

for sym in all_syms:
    sp = prices[prices['symbol'] == sym].sort_values('date').copy()
    if len(sp) < 30:
        continue

    c = sp['close']; h = sp['high']; l = sp['low']; v = sp['volume']
    last = sp.iloc[-1]
    close_price = last['close']
    n = len(sp)

    # === MOVING AVERAGES ===
    ma5 = c.rolling(5).mean().iloc[-1]
    ma10 = c.rolling(10).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1] if n >= 50 else np.nan
    ma200 = c.rolling(200).mean().iloc[-1] if n >= 200 else np.nan

    dist_ma5 = (close_price - ma5) / ma5 * 100
    dist_ma20 = (close_price - ma20) / ma20 * 100
    dist_ma50 = (close_price - ma50) / ma50 * 100 if pd.notna(ma50) else np.nan
    dist_ma200 = (close_price - ma200) / ma200 * 100 if pd.notna(ma200) else np.nan

    # MA crossovers
    ma5_above_ma20 = 1 if ma5 > ma20 else 0
    ma20_above_ma50 = 1 if pd.notna(ma50) and ma20 > ma50 else (np.nan if pd.isna(ma50) else 0)
    ma50_above_ma200 = 1 if pd.notna(ma200) and pd.notna(ma50) and ma50 > ma200 else (np.nan if pd.isna(ma200) else 0)

    # === RSI ===
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss_s.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # === RETURNS ===
    ret_1w = (close_price / c.iloc[-6] - 1) * 100 if n >= 6 else np.nan
    ret_2w = (close_price / c.iloc[-11] - 1) * 100 if n >= 11 else np.nan
    ret_4w = (close_price / c.iloc[-21] - 1) * 100 if n >= 21 else np.nan
    ret_12w = (close_price / c.iloc[-61] - 1) * 100 if n >= 61 else np.nan

    # === VOLATILITY ===
    daily_ret = c.pct_change()
    vol_5d = daily_ret.tail(5).std() * np.sqrt(252) * 100
    vol_20d = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
    vol_60d = daily_ret.rolling(60).std().iloc[-1] * np.sqrt(252) * 100 if n >= 60 else np.nan

    # ATR(14)
    tr_vals = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_14 = tr_vals.rolling(14).mean().iloc[-1]
    atr_pct = atr_14 / close_price * 100  # ATR as % of price

    # Vol compression: vol_5d vs vol_60d
    vol_compression = vol_5d / vol_60d if pd.notna(vol_60d) and vol_60d > 0 else np.nan

    # Volume
    vol_avg_20 = v.rolling(20).mean().iloc[-1]
    vol_ratio = v.iloc[-1] / vol_avg_20 if vol_avg_20 > 0 else np.nan
    vol_avg_5 = v.tail(5).mean()
    vol_5v20 = vol_avg_5 / vol_avg_20 if vol_avg_20 > 0 else np.nan

    # === BOLLINGER ===
    bb_mid = c.rolling(20).mean().iloc[-1]
    bb_std = c.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std; bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_pctb = (close_price - bb_lower) / bb_width if bb_width > 0 else 0.5
    bb_width_pct = bb_width / bb_mid * 100 if bb_mid > 0 else np.nan  # Band width as % of price

    # === STOCHASTIC ===
    if n >= 14:
        low_14 = l.tail(14).min(); high_14 = h.tail(14).max()
        stoch_k = (close_price - low_14) / (high_14 - low_14) * 100 if (high_14 - low_14) > 0 else 50
    else:
        stoch_k = np.nan

    # === PARABOLIC SAR ===
    sar_vals, sar_trend = calc_parabolic_sar(h, l, c)
    psar_dist = (close_price - sar_vals[-1]) / close_price * 100 if pd.notna(sar_vals[-1]) else np.nan
    psar_trend = int(sar_trend[-1]) if pd.notna(sar_trend[-1]) else np.nan  # +1=bull, -1=bear

    # === SUPERTREND ===
    st_vals, st_dir = calc_supertrend(h, l, c)
    st_dist = (close_price - st_vals[-1]) / close_price * 100 if pd.notna(st_vals[-1]) else np.nan
    st_direction = int(st_dir[-1]) if pd.notna(st_dir[-1]) else np.nan  # +1=bull, -1=bear

    # Range position
    h5 = h.tail(5).max(); l5 = l.tail(5).min(); rng = h5 - l5
    range_pos = (close_price - l5) / rng if rng > 0 else 0.5

    # === EARNINGS ===
    sym_earn = earn[earn['symbol'] == sym]
    target = pd.Timestamp('2026-02-13')
    past_earn = sym_earn[sym_earn['date'] <= target]
    future_earn = sym_earn[sym_earn['date'] > target]

    if len(past_earn) > 0:
        last_earn = past_earn.iloc[0]
        days_since_earn = (target - last_earn['date']).days
        eps_surprise = ((last_earn['eps_actual'] - last_earn['eps_estimated']) / abs(last_earn['eps_estimated']) * 100
                       if pd.notna(last_earn['eps_estimated']) and last_earn['eps_estimated'] != 0 else np.nan)
        rev_surprise = ((last_earn['revenue_actual'] - last_earn['revenue_estimated']) / abs(last_earn['revenue_estimated']) * 100
                       if pd.notna(last_earn['revenue_estimated']) and last_earn['revenue_estimated'] != 0 else np.nan)
        # Consecutive beats
        beats_4q = sum(1 for _, e in past_earn.head(4).iterrows()
                      if pd.notna(e['eps_actual']) and pd.notna(e['eps_estimated']) and e['eps_actual'] > e['eps_estimated'])
        # EPS growth YoY (last Q vs same Q year ago)
        if len(past_earn) >= 5:
            curr_eps = past_earn.iloc[0]['eps_actual']
            yago_eps = past_earn.iloc[4]['eps_actual'] if len(past_earn) > 4 else np.nan
            eps_growth_yoy = ((curr_eps - yago_eps) / abs(yago_eps) * 100
                             if pd.notna(yago_eps) and yago_eps != 0 else np.nan)
        else:
            eps_growth_yoy = np.nan
    else:
        days_since_earn = np.nan; eps_surprise = np.nan; rev_surprise = np.nan
        beats_4q = np.nan; eps_growth_yoy = np.nan

    in_earnings_window = 1 if pd.notna(days_since_earn) and abs(days_since_earn) <= 7 else 0

    # === FUNDAMENTALS ===
    sym_rat = ratios[ratios['symbol'] == sym]
    if len(sym_rat) > 0:
        rat = sym_rat.iloc[0]
        pe = rat.get('pe_ratio', np.nan)
        ptb = rat.get('price_to_book_ratio', np.nan)
        peg = rat.get('price_earnings_to_growth_ratio', np.nan)
        roe = rat.get('return_on_equity', np.nan)
        roa = rat.get('return_on_assets', np.nan)
        de_ratio = rat.get('debt_equity_ratio', np.nan)
        curr_ratio = rat.get('current_ratio', np.nan)
        div_yield = rat.get('dividend_yield', np.nan)
        payout = rat.get('payout_ratio', np.nan)
        net_margin = rat.get('net_profit_margin', np.nan)
        op_margin = rat.get('operating_profit_margin', np.nan)
        gross_margin = rat.get('gross_profit_margin', np.nan)
        ev_ebitda = rat.get('ev_to_ebitda', np.nan)
        p_fcf = rat.get('price_to_free_cash_flow_ratio', np.nan)
    else:
        pe = ptb = peg = roe = roa = de_ratio = curr_ratio = np.nan
        div_yield = payout = net_margin = op_margin = gross_margin = np.nan
        ev_ebitda = p_fcf = np.nan

    sym_met = metrics[metrics['symbol'] == sym]
    mktcap = sym_met.iloc[0].get('market_cap', np.nan) if len(sym_met) > 0 else np.nan

    # Dividends
    sym_div = dividends[dividends['symbol'] == sym]
    has_dividend = 1 if len(sym_div) > 0 else 0
    if len(sym_div) > 0:
        last_div_date = sym_div.iloc[0]['date']
        days_since_div = (target - last_div_date).days
    else:
        days_since_div = np.nan

    side = 'LONG' if sym in TOP25 else 'SHORT'
    rank = TOP25.index(sym) + 1 if sym in TOP25 else BOT25.index(sym) + 1

    results.append({
        'symbol': sym, 'side': side, 'rank': rank, 'close': close_price,
        # MAs
        'dist_ma5': dist_ma5, 'dist_ma20': dist_ma20, 'dist_ma50': dist_ma50, 'dist_ma200': dist_ma200,
        'ma5_x_ma20': ma5_above_ma20, 'ma20_x_ma50': ma20_above_ma50, 'ma50_x_ma200': ma50_above_ma200,
        # Oscillators
        'rsi': rsi, 'bb_pctb': bb_pctb, 'bb_width_pct': bb_width_pct, 'stoch_k': stoch_k, 'range_pos': range_pos,
        # PSAR & SuperTrend
        'psar_dist': psar_dist, 'psar_trend': psar_trend, 'st_dist': st_dist, 'st_dir': st_direction,
        # Returns
        'ret_1w': ret_1w, 'ret_2w': ret_2w, 'ret_4w': ret_4w, 'ret_12w': ret_12w,
        # Volatility
        'vol_5d': vol_5d, 'vol_20d': vol_20d, 'vol_60d': vol_60d, 'atr_pct': atr_pct,
        'vol_compress': vol_compression, 'vol_ratio': vol_ratio, 'vol_5v20': vol_5v20,
        # Earnings
        'days_since_earn': days_since_earn, 'in_earn_window': in_earnings_window,
        'eps_surprise': eps_surprise, 'rev_surprise': rev_surprise,
        'beats_4q': beats_4q, 'eps_growth_yoy': eps_growth_yoy,
        # Fundamentals
        'pe': pe, 'ptb': ptb, 'peg': peg, 'roe': roe, 'roa': roa,
        'de_ratio': de_ratio, 'curr_ratio': curr_ratio,
        'div_yield': div_yield, 'payout': payout, 'has_dividend': has_dividend,
        'net_margin': net_margin, 'op_margin': op_margin, 'gross_margin': gross_margin,
        'ev_ebitda': ev_ebitda, 'p_fcf': p_fcf, 'mktcap_B': mktcap / 1e9 if pd.notna(mktcap) else np.nan,
        'days_since_div': days_since_div,
    })

rdf = pd.DataFrame(results)
longs = rdf[rdf['side'] == 'LONG']
shorts = rdf[rdf['side'] == 'SHORT']

# ============================================================
# A) INDICADORES TECNICOS EXTENDIDOS
# ============================================================
print("=" * 160)
print("  A) INDICADORES TECNICOS EXTENDIDOS - PSAR, SuperTrend, Volatilidad")
print("=" * 160)

for side_name, sdf in [('LONG (subieron)', longs), ('SHORT (bajaron)', shorts)]:
    sdf = sdf.sort_values('rank')
    print(f"\n  --- {side_name} ---")
    print(f"  {'#':>3} {'Sym':>6} | {'PSAR%':>7} {'Trend':>5} | {'ST%':>7} {'Dir':>4} | {'ATR%':>5} | {'Vol5d':>6} {'Vol20':>6} {'Vol60':>6} {'Compr':>6} | {'BBw%':>5} | {'VolR':>5} {'V5/20':>5}")
    print(f"  {'-'*3} {'-'*6} | {'-'*7} {'-'*5} | {'-'*7} {'-'*4} | {'-'*5} | {'-'*6} {'-'*6} {'-'*6} {'-'*6} | {'-'*5} | {'-'*5} {'-'*5}")
    for _, r in sdf.iterrows():
        pt = "BULL" if r['psar_trend'] == 1 else "BEAR" if r['psar_trend'] == -1 else "N/A"
        sd = "BULL" if r['st_dir'] == 1 else "BEAR" if r['st_dir'] == -1 else "N/A"
        vc = f"{r['vol_compress']:.2f}" if pd.notna(r['vol_compress']) else " N/A"
        v60 = f"{r['vol_60d']:.0f}%" if pd.notna(r['vol_60d']) else "  N/A"
        print(f"  {int(r['rank']):>3} {r['symbol']:>6} | {r['psar_dist']:>+6.1f}% {pt:>5} | {r['st_dist']:>+6.1f}% {sd:>4} | {r['atr_pct']:>4.1f}% | {r['vol_5d']:>5.0f}% {r['vol_20d']:>5.0f}% {v60:>6} {vc:>6} | {r['bb_width_pct']:>4.1f}% | {r['vol_ratio']:>4.1f}x {r['vol_5v20']:>4.1f}x")

    # Averages
    bull_psar = (sdf['psar_trend'] == 1).sum()
    bull_st = (sdf['st_dir'] == 1).sum()
    n = len(sdf)
    print(f"\n  MEDIAS {side_name}:")
    print(f"    PSAR dist: {sdf['psar_dist'].mean():>+.1f}% | PSAR Bull: {bull_psar}/{n} ({bull_psar/n*100:.0f}%)")
    print(f"    ST dist:   {sdf['st_dist'].mean():>+.1f}% | ST Bull:   {bull_st}/{n} ({bull_st/n*100:.0f}%)")
    print(f"    ATR%:      {sdf['atr_pct'].mean():.1f}% | Vol5d: {sdf['vol_5d'].mean():.0f}% | Vol20d: {sdf['vol_20d'].mean():.0f}% | Vol compress: {sdf['vol_compress'].mean():.2f}")
    print(f"    BB width:  {sdf['bb_width_pct'].mean():.1f}% | Vol ratio: {sdf['vol_ratio'].mean():.2f}x")

# ============================================================
# B) FUNDAMENTALES EXTENDIDOS
# ============================================================
print(f"\n\n{'='*160}")
print("  B) FUNDAMENTALES EXTENDIDOS - PEG, Margenes, Dividendos")
print("=" * 160)

for side_name, sdf in [('LONG (subieron)', longs), ('SHORT (bajaron)', shorts)]:
    sdf = sdf.sort_values('rank')
    print(f"\n  --- {side_name} ---")
    print(f"  {'#':>3} {'Sym':>6} | {'P/E':>7} | {'PEG':>7} | {'EV/EB':>7} | {'P/FCF':>7} | {'ROE%':>7} | {'NetMrg%':>7} | {'OpMrg%':>7} | {'D/E':>6} | {'DivY%':>6} {'Payout%':>7} | {'MktCap':>8}")
    print(f"  {'-'*3} {'-'*6} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*6} {'-'*7} | {'-'*8}")
    for _, r in sdf.iterrows():
        def fmt(v, f='.1f', w=7):
            return f"{v:{f}}" if pd.notna(v) and abs(v) < 9999 else "N/A"
        pe_s = fmt(r['pe']); peg_s = fmt(r['peg']); ev_s = fmt(r['ev_ebitda'])
        pfcf_s = fmt(r['p_fcf']); roe_s = fmt(r['roe']*100 if pd.notna(r['roe']) else np.nan)
        nm_s = fmt(r['net_margin']*100 if pd.notna(r['net_margin']) else np.nan)
        om_s = fmt(r['op_margin']*100 if pd.notna(r['op_margin']) else np.nan)
        de_s = fmt(r['de_ratio'], '.2f', 6)
        dy_s = fmt(r['div_yield']*100 if pd.notna(r['div_yield']) else np.nan, '.2f', 6)
        po_s = fmt(r['payout']*100 if pd.notna(r['payout']) else np.nan)
        mc_s = f"${r['mktcap_B']:.0f}B" if pd.notna(r['mktcap_B']) else "N/A"
        print(f"  {int(r['rank']):>3} {r['symbol']:>6} | {pe_s:>7} | {peg_s:>7} | {ev_s:>7} | {pfcf_s:>7} | {roe_s:>7} | {nm_s:>7} | {om_s:>7} | {de_s:>6} | {dy_s:>6} {po_s:>7} | {mc_s:>8}")

# ============================================================
# C) EARNINGS WINDOW ANALYSIS
# ============================================================
print(f"\n\n{'='*160}")
print("  C) ANALISIS POR VENTANA DE EARNINGS")
print("=" * 160)

for side_name, sdf in [('LONG (subieron)', longs), ('SHORT (bajaron)', shorts)]:
    sdf = sdf.sort_values('rank')
    print(f"\n  --- {side_name} ---")
    print(f"  {'#':>3} {'Sym':>6} | {'DaysEarn':>8} | {'InWin':>5} | {'EPSsurp%':>8} | {'REVsurp%':>8} | {'Beats':>5} | {'EPSgYoY%':>8}")
    print(f"  {'-'*3} {'-'*6} | {'-'*8} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*5} | {'-'*8}")
    for _, r in sdf.iterrows():
        de_s = f"{int(r['days_since_earn']):>7}d" if pd.notna(r['days_since_earn']) else "     N/A"
        iw = "SI" if r['in_earn_window'] == 1 else "NO"
        es = f"{r['eps_surprise']:>+7.1f}%" if pd.notna(r['eps_surprise']) else "     N/A"
        rs = f"{r['rev_surprise']:>+7.1f}%" if pd.notna(r['rev_surprise']) else "     N/A"
        bt = f"{int(r['beats_4q'])}/4" if pd.notna(r['beats_4q']) else "  N/A"
        eg = f"{r['eps_growth_yoy']:>+7.1f}%" if pd.notna(r['eps_growth_yoy']) else "     N/A"
        print(f"  {int(r['rank']):>3} {r['symbol']:>6} | {de_s} | {iw:>5} | {es} | {rs} | {bt:>5} | {eg}")

# ============================================================
# D) COMPARATIVA COMPLETA
# ============================================================
print(f"\n\n{'='*160}")
print("  D) COMPARATIVA COMPLETA LONG vs SHORT")
print("=" * 160)

compare = [
    # Technicals
    ('PSAR dist %', 'psar_dist', '.1f'),
    ('PSAR Bull %', None, None),  # special
    ('SuperTrend dist %', 'st_dist', '.1f'),
    ('SuperTrend Bull %', None, None),  # special
    ('ATR %', 'atr_pct', '.2f'),
    ('Vol 5d ann%', 'vol_5d', '.0f'),
    ('Vol 20d ann%', 'vol_20d', '.0f'),
    ('Vol 60d ann%', 'vol_60d', '.0f'),
    ('Vol compress (5d/60d)', 'vol_compress', '.2f'),
    ('BB width %', 'bb_width_pct', '.1f'),
    ('BB %B', 'bb_pctb', '.2f'),
    ('Stochastic %K', 'stoch_k', '.1f'),
    ('RSI(14)', 'rsi', '.1f'),
    ('Range Pos', 'range_pos', '.2f'),
    ('', None, None),
    # Returns
    ('Ret 1 sem %', 'ret_1w', '.1f'),
    ('Ret 2 sem %', 'ret_2w', '.1f'),
    ('Ret 4 sem %', 'ret_4w', '.1f'),
    ('Ret 12 sem %', 'ret_12w', '.1f'),
    ('', None, None),
    # Volume
    ('Vol ratio (last/avg20)', 'vol_ratio', '.2f'),
    ('Vol 5d/20d', 'vol_5v20', '.2f'),
    ('', None, None),
    # Earnings
    ('Dias desde earnings', 'days_since_earn', '.0f'),
    ('En ventana earnings', None, None),  # special
    ('EPS Surprise %', 'eps_surprise', '.1f'),
    ('Rev Surprise %', 'rev_surprise', '.1f'),
    ('Beats (de 4Q)', 'beats_4q', '.1f'),
    ('EPS growth YoY %', 'eps_growth_yoy', '.1f'),
    ('', None, None),
    # Fundamentals
    ('P/E', 'pe', '.1f'),
    ('PEG', 'peg', '.1f'),
    ('EV/EBITDA', 'ev_ebitda', '.1f'),
    ('P/FCF', 'p_fcf', '.1f'),
    ('P/B', 'ptb', '.1f'),
    ('ROE %', 'roe', '.3f'),
    ('ROA %', 'roa', '.3f'),
    ('Net Margin %', 'net_margin', '.3f'),
    ('Op Margin %', 'op_margin', '.3f'),
    ('D/E', 'de_ratio', '.2f'),
    ('Div Yield %', 'div_yield', '.4f'),
    ('Payout %', 'payout', '.3f'),
    ('Has Dividend', 'has_dividend', '.0f'),
    ('Market Cap $B', 'mktcap_B', '.0f'),
]

print(f"\n  {'Metrica':<28} | {'LONG':>12} | {'SHORT':>12} | {'Diff':>10} | Interpretacion")
print(f"  {'-'*28} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*50}")

for label, col, fmt in compare:
    if label == '':
        print(f"  {'':>28} |")
        continue

    if label == 'PSAR Bull %':
        lv = (longs['psar_trend'] == 1).mean() * 100
        sv = (shorts['psar_trend'] == 1).mean() * 100
        diff = lv - sv
        interp = "Mas longs en BULL SAR" if diff > 10 else "Mas shorts en BULL SAR" if diff < -10 else ""
        print(f"  {label:<28} | {lv:>11.0f}% | {sv:>11.0f}% | {diff:>+9.0f}% | {interp}")
        continue
    if label == 'SuperTrend Bull %':
        lv = (longs['st_dir'] == 1).mean() * 100
        sv = (shorts['st_dir'] == 1).mean() * 100
        diff = lv - sv
        interp = "Mas longs en BULL ST" if diff > 10 else "Mas shorts en BULL ST" if diff < -10 else ""
        print(f"  {label:<28} | {lv:>11.0f}% | {sv:>11.0f}% | {diff:>+9.0f}% | {interp}")
        continue
    if label == 'En ventana earnings':
        lv = longs['in_earn_window'].mean() * 100
        sv = shorts['in_earn_window'].mean() * 100
        diff = lv - sv
        print(f"  {label:<28} | {lv:>11.0f}% | {sv:>11.0f}% | {diff:>+9.0f}% | Ambos post-earnings")
        continue

    lm = longs[col].mean()
    sm = shorts[col].mean()

    if pd.isna(lm) and pd.isna(sm):
        print(f"  {label:<28} | {'N/A':>12} | {'N/A':>12} | {'N/A':>10} |")
        continue

    diff = lm - sm if pd.notna(lm) and pd.notna(sm) else np.nan

    # Auto-interpret
    interp = ""
    if col == 'psar_dist':
        interp = "Longs mas lejos de SAR (oversold)" if diff < -1 else "Shorts mas lejos de SAR" if diff > 1 else ""
    elif col == 'st_dist':
        interp = "Longs debajo SuperTrend" if lm < 0 and sm > 0 else ""
    elif col == 'atr_pct':
        interp = "Longs mas volatiles (ATR)" if diff > 0.3 else "Shorts mas volatiles" if diff < -0.3 else ""
    elif col == 'vol_compress':
        interp = "Longs vol comprimida -> expansion?" if diff < -0.2 else "Shorts vol comprimida" if diff > 0.2 else ""
    elif col == 'bb_width_pct':
        interp = "Longs bandas mas anchas" if diff > 1 else "Shorts bandas mas anchas" if diff < -1 else ""
    elif col in ('ret_1w', 'ret_2w'):
        interp = "Longs habian caido -> mean reversion" if diff < -2 else "Longs habian subido -> momentum" if diff > 2 else ""
    elif col == 'ret_12w':
        interp = "Longs mucho mas momentum LP" if diff > 10 else ""
    elif col == 'peg':
        if pd.notna(diff):
            interp = "Longs PEG mas bajo -> mas value/growth" if diff < -1 else "Shorts PEG mas bajo" if diff > 1 else ""
    elif col == 'div_yield':
        if pd.notna(diff):
            interp = "Longs pagan mas dividendo" if diff > 0.005 else "Shorts pagan mas dividendo" if diff < -0.005 else ""
    elif col == 'eps_surprise':
        if pd.notna(diff):
            interp = "Longs mejor EPS surprise" if diff > 3 else "Shorts mejor surprise" if diff < -3 else ""

    lm_s = f"{lm:{fmt}}" if pd.notna(lm) else "N/A"
    sm_s = f"{sm:{fmt}}" if pd.notna(sm) else "N/A"
    diff_s = f"{diff:+{fmt}}" if pd.notna(diff) else "N/A"
    print(f"  {label:<28} | {lm_s:>12} | {sm_s:>12} | {diff_s:>10} | {interp}")

# ============================================================
# E) SEGMENTACION: EARNINGS vs NO-EARNINGS
# ============================================================
print(f"\n\n{'='*160}")
print("  E) SEGMENTACION: ACCIONES CON EARNINGS RECIENTES (<10d) vs SIN EARNINGS")
print("=" * 160)

# In this case almost all are in earnings window, but let's check
for side_name, sdf in [('LONG', longs), ('SHORT', shorts)]:
    in_earn = sdf[sdf['days_since_earn'].abs() <= 7] if 'days_since_earn' in sdf.columns else pd.DataFrame()
    not_earn = sdf[sdf['days_since_earn'].abs() > 7] if 'days_since_earn' in sdf.columns else pd.DataFrame()

    print(f"\n  --- {side_name}: {len(in_earn)} en ventana earnings, {len(not_earn)} fuera ---")
    if len(in_earn) > 0 and len(not_earn) > 0:
        key_metrics = ['rsi', 'ret_1w', 'ret_4w', 'ret_12w', 'vol_20d', 'bb_pctb', 'stoch_k',
                       'psar_dist', 'st_dist', 'eps_surprise', 'atr_pct']
        print(f"  {'Metrica':<20} | {'Con earnings':>12} | {'Sin earnings':>12}")
        print(f"  {'-'*20} | {'-'*12} | {'-'*12}")
        for m in key_metrics:
            ie = in_earn[m].mean()
            ne = not_earn[m].mean()
            ie_s = f"{ie:.1f}" if pd.notna(ie) else "N/A"
            ne_s = f"{ne:.1f}" if pd.notna(ne) else "N/A"
            print(f"  {m:<20} | {ie_s:>12} | {ne_s:>12}")
    elif len(in_earn) > 0:
        print(f"    TODAS en ventana de earnings - no hay grupo de control")
        # List the ones with biggest EPS surprise
        sorted_earn = in_earn.sort_values('eps_surprise', ascending=False)
        print(f"    Top EPS Surprises:")
        for _, r in sorted_earn.head(10).iterrows():
            es = f"{r['eps_surprise']:>+.1f}%" if pd.notna(r['eps_surprise']) else "N/A"
            rs = f"{r['rev_surprise']:>+.1f}%" if pd.notna(r['rev_surprise']) else "N/A"
            print(f"      {r['symbol']:>6}: EPS {es} | Rev {rs} | RSI {r['rsi']:.0f} | Ret1w {r['ret_1w']:>+.1f}%")

# ============================================================
# F) PATRON RESUMEN
# ============================================================
print(f"\n\n{'='*160}")
print("  F) PATRONES IDENTIFICADOS")
print("=" * 160)

print(f"""
  PATRON LADO LARGO (acciones que subieron):
  ==========================================
  - PSAR: {(longs['psar_trend']==1).mean()*100:.0f}% bullish (dist media {longs['psar_dist'].mean():+.1f}%)
  - SuperTrend: {(longs['st_dir']==1).mean()*100:.0f}% bullish (dist media {longs['st_dist'].mean():+.1f}%)
  - RSI medio: {longs['rsi'].mean():.1f} | BB%B: {longs['bb_pctb'].mean():.2f} | Stoch: {longs['stoch_k'].mean():.1f}
  - Ret 1w: {longs['ret_1w'].mean():+.1f}% | Ret 4w: {longs['ret_4w'].mean():+.1f}% | Ret 12w: {longs['ret_12w'].mean():+.1f}%
  - Volatilidad 20d: {longs['vol_20d'].mean():.0f}% | ATR: {longs['atr_pct'].mean():.1f}% | Vol compress: {longs['vol_compress'].mean():.2f}
  - Dist MA200: {longs['dist_ma200'].mean():+.1f}% | Dist MA20: {longs['dist_ma20'].mean():+.1f}%
  - Earnings: {longs['in_earn_window'].mean()*100:.0f}% en ventana | Beats: {longs['beats_4q'].mean():.1f}/4
  - Con dividendo: {longs['has_dividend'].mean()*100:.0f}%

  PATRON LADO CORTO (acciones que bajaron):
  ==========================================
  - PSAR: {(shorts['psar_trend']==1).mean()*100:.0f}% bullish (dist media {shorts['psar_dist'].mean():+.1f}%)
  - SuperTrend: {(shorts['st_dir']==1).mean()*100:.0f}% bullish (dist media {shorts['st_dist'].mean():+.1f}%)
  - RSI medio: {shorts['rsi'].mean():.1f} | BB%B: {shorts['bb_pctb'].mean():.2f} | Stoch: {shorts['stoch_k'].mean():.1f}
  - Ret 1w: {shorts['ret_1w'].mean():+.1f}% | Ret 4w: {shorts['ret_4w'].mean():+.1f}% | Ret 12w: {shorts['ret_12w'].mean():+.1f}%
  - Volatilidad 20d: {shorts['vol_20d'].mean():.0f}% | ATR: {shorts['atr_pct'].mean():.1f}% | Vol compress: {shorts['vol_compress'].mean():.2f}
  - Dist MA200: {shorts['dist_ma200'].mean():+.1f}% | Dist MA20: {shorts['dist_ma20'].mean():+.1f}%
  - Earnings: {shorts['in_earn_window'].mean()*100:.0f}% en ventana | Beats: {shorts['beats_4q'].mean():.1f}/4
  - Con dividendo: {shorts['has_dividend'].mean()*100:.0f}%

  DIFERENCIAS CLAVE (Long - Short):
  ==================================
  - PSAR dist:      {longs['psar_dist'].mean() - shorts['psar_dist'].mean():+.1f}% (longs {'mas oversold' if longs['psar_dist'].mean() < shorts['psar_dist'].mean() else 'mas bullish'} por SAR)
  - ST dist:        {longs['st_dist'].mean() - shorts['st_dist'].mean():+.1f}%
  - RSI:            {longs['rsi'].mean() - shorts['rsi'].mean():+.1f} (longs {'mas oversold' if longs['rsi'].mean() < shorts['rsi'].mean() else 'mas overbought'})
  - Stochastic:     {longs['stoch_k'].mean() - shorts['stoch_k'].mean():+.1f}
  - Ret 1w:         {longs['ret_1w'].mean() - shorts['ret_1w'].mean():+.1f}% (longs {'caian mas -> mean reversion' if longs['ret_1w'].mean() < shorts['ret_1w'].mean() else 'subian mas'})
  - Ret 12w:        {longs['ret_12w'].mean() - shorts['ret_12w'].mean():+.1f}% (longs {'mas momentum LP' if longs['ret_12w'].mean() > shorts['ret_12w'].mean() else 'menos momentum'})
  - Volatilidad:    {longs['vol_20d'].mean() - shorts['vol_20d'].mean():+.0f}% (longs {'mas volatiles' if longs['vol_20d'].mean() > shorts['vol_20d'].mean() else 'menos volatiles'})
  - Vol compress:   {longs['vol_compress'].mean() - shorts['vol_compress'].mean():+.2f}
  - Dist MA200:     {longs['dist_ma200'].mean() - shorts['dist_ma200'].mean():+.1f}%
""")
