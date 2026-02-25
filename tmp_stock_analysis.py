import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sqlalchemy
import pandas as pd
import numpy as np
import json

engine = sqlalchemy.create_engine('postgresql://fmp:fmp123@localhost:5433/fmp_data')

# Top 25 subidas y bajadas
TOP25 = ['OMC','TPL','GPN','MRNA','GRMN','OXY','CIEN','MOH','FIX','DE',
         'CSGP','GE','SNDK','GLW','FSLR','EBAY','MGM','DPZ','GOOGL','PWR',
         'GOOG','AMAT','LRCX','BAX','EXPD']
BOT25 = ['EPAM','POOL','AKAM','GPC','IP','PANW','ARES','DDOG','CPB','PKG',
         'CRWD','PAYC','BX','SW','FTNT','ORCL','WDAY','KR','WMT','INVH',
         'NOW','APTV','DOW','CAG','ANET']

all_syms = list(set(TOP25 + BOT25))

with engine.connect() as conn:
    # Price history for indicators (need ~250 days before Feb 13)
    prices = pd.read_sql("""SELECT symbol, date, open, high, low, close, volume
        FROM fmp_price_history
        WHERE symbol = ANY(%(syms)s) AND date >= '2025-01-01' AND date <= '2026-02-13'
        ORDER BY symbol, date""", conn, params={"syms": all_syms}, parse_dates=['date'])

    # Earnings data
    earn = pd.read_sql("""SELECT symbol, date, eps_actual, eps_estimated, revenue_actual, revenue_estimated
        FROM fmp_earnings
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-01-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": all_syms}, parse_dates=['date'])

    # Key metrics
    metrics = pd.read_sql("""SELECT symbol, date, pe_ratio, pb_ratio, ev_to_ebitda,
        roe, revenue_per_share, net_income_per_share, market_cap
        FROM fmp_key_metrics
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-06-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": all_syms}, parse_dates=['date'])

    # Ratios
    ratios = pd.read_sql("""SELECT symbol, date, pe_ratio, price_to_book_ratio,
        return_on_equity, debt_equity_ratio, current_ratio, dividend_yield,
        net_profit_margin, ev_to_ebitda as rat_ev_ebitda
        FROM fmp_ratios
        WHERE symbol = ANY(%(syms)s) AND date >= '2024-06-01'
        ORDER BY symbol, date DESC""", conn, params={"syms": all_syms}, parse_dates=['date'])

print(f"Prices: {len(prices):,} rows, Earnings: {len(earn)}, Metrics: {len(metrics)}, Ratios: {len(ratios)}")

# ============================================================
# COMPUTE INDICATORS per stock as of Feb 13
# ============================================================
results = []

for sym in all_syms:
    sp = prices[prices['symbol'] == sym].sort_values('date').copy()
    if len(sp) < 20:
        continue

    c = sp['close']
    h = sp['high']
    l = sp['low']
    v = sp['volume']
    last = sp.iloc[-1]

    # Price and MAs
    ma5 = c.rolling(5).mean().iloc[-1]
    ma10 = c.rolling(10).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1] if len(sp) >= 50 else np.nan
    ma200 = c.rolling(200).mean().iloc[-1] if len(sp) >= 200 else np.nan

    close_price = last['close']
    dist_ma5 = (close_price - ma5) / ma5 * 100 if pd.notna(ma5) else np.nan
    dist_ma10 = (close_price - ma10) / ma10 * 100 if pd.notna(ma10) else np.nan
    dist_ma20 = (close_price - ma20) / ma20 * 100 if pd.notna(ma20) else np.nan
    dist_ma50 = (close_price - ma50) / ma50 * 100 if pd.notna(ma50) else np.nan
    dist_ma200 = (close_price - ma200) / ma200 * 100 if pd.notna(ma200) else np.nan

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss_s.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # Returns
    ret_1w = (close_price / c.iloc[-6] - 1) * 100 if len(sp) >= 6 else np.nan
    ret_2w = (close_price / c.iloc[-11] - 1) * 100 if len(sp) >= 11 else np.nan
    ret_4w = (close_price / c.iloc[-21] - 1) * 100 if len(sp) >= 21 else np.nan
    ret_12w = (close_price / c.iloc[-61] - 1) * 100 if len(sp) >= 61 else np.nan

    # Volume
    vol_avg_20 = v.rolling(20).mean().iloc[-1]
    vol_last = v.iloc[-1]
    vol_ratio = vol_last / vol_avg_20 if vol_avg_20 > 0 else np.nan

    # Volatility
    daily_ret = c.pct_change()
    vol_20d = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100  # annualized

    # Bollinger %B
    bb_mid = c.rolling(20).mean().iloc[-1]
    bb_std = c.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pctb = (close_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # Stochastic
    if len(sp) >= 14:
        low_14 = l.tail(14).min()
        high_14 = h.tail(14).max()
        stoch_k = (close_price - low_14) / (high_14 - low_14) * 100 if (high_14 - low_14) > 0 else 50
    else:
        stoch_k = np.nan

    # Range position (5d)
    h5 = h.tail(5).max()
    l5 = l.tail(5).min()
    rng = h5 - l5
    range_pos = (close_price - l5) / rng if rng > 0 else 0.5

    # Earnings
    sym_earn = earn[earn['symbol'] == sym].head(4)
    if len(sym_earn) > 0:
        last_earn = sym_earn.iloc[0]
        eps_surprise = ((last_earn['eps_actual'] - last_earn['eps_estimated']) / abs(last_earn['eps_estimated']) * 100
                       if pd.notna(last_earn['eps_estimated']) and last_earn['eps_estimated'] != 0 else np.nan)
        beats = sum(1 for _, e in sym_earn.iterrows() if pd.notna(e['eps_actual']) and pd.notna(e['eps_estimated']) and e['eps_actual'] > e['eps_estimated'])
        n_earn = len(sym_earn)
        days_since_earn = (pd.Timestamp('2026-02-13') - last_earn['date']).days
    else:
        eps_surprise = np.nan
        beats = np.nan
        n_earn = 0
        days_since_earn = np.nan

    # Key metrics (most recent)
    sym_met = metrics[metrics['symbol'] == sym]
    if len(sym_met) > 0:
        m = sym_met.iloc[0]
        pe = m.get('pe_ratio', np.nan)
        pb = m.get('pb_ratio', np.nan)
        ev_ebitda = m.get('ev_to_ebitda', np.nan)
        roe = m.get('roe', np.nan)
        mktcap = m.get('market_cap', np.nan)
    else:
        pe = pb = ev_ebitda = roe = mktcap = np.nan

    # Ratios
    sym_rat = ratios[ratios['symbol'] == sym]
    if len(sym_rat) > 0:
        rat = sym_rat.iloc[0]
        de_ratio = rat.get('debt_equity_ratio', np.nan)
        curr_ratio = rat.get('current_ratio', np.nan)
        div_yield = rat.get('dividend_yield', np.nan)
        net_margin = rat.get('net_profit_margin', np.nan)
    else:
        de_ratio = curr_ratio = div_yield = net_margin = np.nan

    side = 'LONG' if sym in TOP25 else 'SHORT'
    rank_in_list = TOP25.index(sym) + 1 if sym in TOP25 else BOT25.index(sym) + 1

    results.append({
        'symbol': sym, 'side': side, 'rank': rank_in_list,
        'close': close_price,
        'dist_ma5': dist_ma5, 'dist_ma10': dist_ma10, 'dist_ma20': dist_ma20,
        'dist_ma50': dist_ma50, 'dist_ma200': dist_ma200,
        'rsi': rsi,
        'ret_1w': ret_1w, 'ret_2w': ret_2w, 'ret_4w': ret_4w, 'ret_12w': ret_12w,
        'vol_ratio': vol_ratio, 'vol_annual': vol_20d,
        'bb_pctb': bb_pctb, 'stoch_k': stoch_k, 'range_pos': range_pos,
        'pe': pe, 'pb': pb, 'ev_ebitda': ev_ebitda, 'roe': roe,
        'de_ratio': de_ratio, 'curr_ratio': curr_ratio, 'div_yield': div_yield, 'net_margin': net_margin,
        'mktcap_B': mktcap / 1e9 if pd.notna(mktcap) else np.nan,
        'eps_surprise_pct': eps_surprise, 'beats_4q': beats, 'n_earn': n_earn,
        'days_since_earn': days_since_earn,
    })

rdf = pd.DataFrame(results)

# ============================================================
# PRINT DETAILED TABLES
# ============================================================

for side, label in [('LONG', 'TOP 25 SUBIDAS (candidatas a LONG)'), ('SHORT', 'TOP 25 BAJADAS (candidatas a SHORT)')]:
    sdf = rdf[rdf['side'] == side].sort_values('rank')

    print(f"\n{'='*180}")
    print(f"  {label}")
    print(f"{'='*180}")

    # Table 1: Price & MAs
    print(f"\n  --- Precio y Medias Moviles ---")
    print(f"  {'#':>3} {'Sym':>6} | {'Close':>8} | {'dMA5%':>7} | {'dMA10%':>7} | {'dMA20%':>7} | {'dMA50%':>7} | {'dMA200%':>8} | {'RSI':>5} | {'BB%B':>5} | {'Stoch':>5} | {'Range':>5}")
    print(f"  {'-'*3} {'-'*6} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*8} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5}")
    for _, r in sdf.iterrows():
        print(f"  {int(r['rank']):>3} {r['symbol']:>6} | ${r['close']:>7.2f} | {r['dist_ma5']:>+6.1f}% | {r['dist_ma10']:>+6.1f}% | {r['dist_ma20']:>+6.1f}% | {r['dist_ma50']:>+6.1f}% | {r['dist_ma200']:>+7.1f}% | {r['rsi']:>5.1f} | {r['bb_pctb']:>5.2f} | {r['stoch_k']:>5.1f} | {r['range_pos']:>5.2f}")

    # Averages
    print(f"  {'':>3} {'MEDIA':>6} | {'':>8} | {sdf['dist_ma5'].mean():>+6.1f}% | {sdf['dist_ma10'].mean():>+6.1f}% | {sdf['dist_ma20'].mean():>+6.1f}% | {sdf['dist_ma50'].mean():>+6.1f}% | {sdf['dist_ma200'].mean():>+7.1f}% | {sdf['rsi'].mean():>5.1f} | {sdf['bb_pctb'].mean():>5.2f} | {sdf['stoch_k'].mean():>5.1f} | {sdf['range_pos'].mean():>5.2f}")

    # Table 2: Returns
    print(f"\n  --- Retornos previos ---")
    print(f"  {'#':>3} {'Sym':>6} | {'Ret 1w%':>8} | {'Ret 2w%':>8} | {'Ret 4w%':>8} | {'Ret 12w%':>9} | {'Vol ann%':>8} | {'Vol ratio':>9}")
    print(f"  {'-'*3} {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*8} | {'-'*9}")
    for _, r in sdf.iterrows():
        vr = f"{r['vol_ratio']:>8.2f}x" if pd.notna(r['vol_ratio']) else "     N/A"
        print(f"  {int(r['rank']):>3} {r['symbol']:>6} | {r['ret_1w']:>+7.1f}% | {r['ret_2w']:>+7.1f}% | {r['ret_4w']:>+7.1f}% | {r['ret_12w']:>+8.1f}% | {r['vol_annual']:>7.1f}% | {vr}")
    print(f"  {'':>3} {'MEDIA':>6} | {sdf['ret_1w'].mean():>+7.1f}% | {sdf['ret_2w'].mean():>+7.1f}% | {sdf['ret_4w'].mean():>+7.1f}% | {sdf['ret_12w'].mean():>+8.1f}% | {sdf['vol_annual'].mean():>7.1f}% | {sdf['vol_ratio'].mean():>8.2f}x")

    # Table 3: Fundamentals
    print(f"\n  --- Fundamentales ---")
    print(f"  {'#':>3} {'Sym':>6} | {'P/E':>8} | {'P/B':>7} | {'EV/EBITDA':>9} | {'ROE%':>7} | {'D/E':>7} | {'MktCap B':>9} | {'EPS surp%':>9} | {'Beats/4Q':>8} | {'DaysSince':>9}")
    print(f"  {'-'*3} {'-'*6} | {'-'*8} | {'-'*7} | {'-'*9} | {'-'*7} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*9}")
    for _, r in sdf.iterrows():
        pe_s = f"{r['pe']:>8.1f}" if pd.notna(r['pe']) else "     N/A"
        pb_s = f"{r['pb']:>7.1f}" if pd.notna(r['pb']) else "    N/A"
        ev_s = f"{r['ev_ebitda']:>9.1f}" if pd.notna(r['ev_ebitda']) else "      N/A"
        roe_s = f"{r['roe']*100:>6.1f}%" if pd.notna(r['roe']) else "    N/A"
        de_s = f"{r['de_ratio']:>7.2f}" if pd.notna(r['de_ratio']) else "    N/A"
        mc_s = f"${r['mktcap_B']:>7.1f}B" if pd.notna(r['mktcap_B']) else "      N/A"
        eps_s = f"{r['eps_surprise_pct']:>+8.1f}%" if pd.notna(r['eps_surprise_pct']) else "      N/A"
        bt_s = f"{int(r['beats_4q'])}/{int(r['n_earn'])}" if pd.notna(r['beats_4q']) else "  N/A"
        ds_s = f"{int(r['days_since_earn']):>8}d" if pd.notna(r['days_since_earn']) else "      N/A"
        print(f"  {int(r['rank']):>3} {r['symbol']:>6} | {pe_s} | {pb_s} | {ev_s} | {roe_s} | {de_s} | {mc_s} | {eps_s} | {bt_s:>8} | {ds_s}")

# ============================================================
# PATTERN ANALYSIS - Compare averages
# ============================================================
print(f"\n\n{'='*180}")
print(f"  COMPARATIVA MEDIAS: LONG (subidas) vs SHORT (bajadas)")
print(f"{'='*180}")

longs = rdf[rdf['side'] == 'LONG']
shorts = rdf[rdf['side'] == 'SHORT']

metrics_to_compare = [
    ('dist_ma5', 'Dist MA5 %', '.1f'),
    ('dist_ma10', 'Dist MA10 %', '.1f'),
    ('dist_ma20', 'Dist MA20 %', '.1f'),
    ('dist_ma50', 'Dist MA50 %', '.1f'),
    ('dist_ma200', 'Dist MA200 %', '.1f'),
    ('rsi', 'RSI(14)', '.1f'),
    ('bb_pctb', 'Bollinger %B', '.2f'),
    ('stoch_k', 'Stochastic %K', '.1f'),
    ('range_pos', 'Range Pos (0-1)', '.2f'),
    ('ret_1w', 'Ret 1 sem %', '.1f'),
    ('ret_2w', 'Ret 2 sem %', '.1f'),
    ('ret_4w', 'Ret 4 sem %', '.1f'),
    ('ret_12w', 'Ret 12 sem %', '.1f'),
    ('vol_annual', 'Volatilidad anual %', '.1f'),
    ('vol_ratio', 'Vol ratio (vs avg20)', '.2f'),
    ('pe', 'P/E', '.1f'),
    ('pb', 'P/B', '.1f'),
    ('ev_ebitda', 'EV/EBITDA', '.1f'),
    ('mktcap_B', 'Market Cap ($B)', '.1f'),
    ('eps_surprise_pct', 'EPS Surprise %', '.1f'),
    ('days_since_earn', 'Dias desde earnings', '.0f'),
]

print(f"\n  {'Metrica':<25} | {'LONG (subidas)':>15} | {'SHORT (bajadas)':>15} | {'Diferencia':>12} | {'Patron'}")
print(f"  {'-'*25} | {'-'*15} | {'-'*15} | {'-'*12} | {'-'*40}")

for col, label, fmt in metrics_to_compare:
    lm = longs[col].mean()
    sm = shorts[col].mean()
    diff = lm - sm

    # Detect pattern
    pattern = ""
    if col.startswith('dist_ma'):
        if lm < sm - 1: pattern = "Longs MAS LEJOS debajo MA -> oversold"
        elif lm > sm + 1: pattern = "Longs MAS ARRIBA de MA -> momentum"
    elif col == 'rsi':
        if lm < sm - 3: pattern = "Longs MAS oversold"
        elif lm > sm + 3: pattern = "Longs MAS overbought"
    elif col == 'ret_1w':
        if lm < sm - 1: pattern = "Longs CAIAN MAS -> mean reversion!"
        elif lm > sm + 1: pattern = "Longs SUBIAN MAS -> momentum!"
    elif col == 'ret_4w':
        if lm < sm - 2: pattern = "Longs peor 4w -> mean reversion"
        elif lm > sm + 2: pattern = "Longs mejor 4w -> momentum"
    elif col == 'bb_pctb':
        if lm < sm - 0.05: pattern = "Longs MAS cerca de lower band"
        elif lm > sm + 0.05: pattern = "Longs MAS cerca de upper band"
    elif col == 'vol_annual':
        if lm > sm + 3: pattern = "Longs MAS volatiles"
        elif lm < sm - 3: pattern = "Longs MENOS volatiles"
    elif col == 'pe':
        if abs(diff) > 5: pattern = f"{'Longs mas baratos' if diff < 0 else 'Longs mas caros'} por P/E"
    elif col == 'days_since_earn':
        if abs(diff) > 10: pattern = f"{'Longs mas recientes' if diff < 0 else 'Shorts mas recientes'} earnings"
    elif col == 'eps_surprise_pct':
        if abs(diff) > 2: pattern = f"{'Longs mejor' if diff > 0 else 'Shorts mejor'} surprise"

    lm_s = f"{lm:{fmt}}" if pd.notna(lm) else "N/A"
    sm_s = f"{sm:{fmt}}" if pd.notna(sm) else "N/A"
    diff_s = f"{diff:+{fmt}}" if pd.notna(diff) else "N/A"

    print(f"  {label:<25} | {lm_s:>15} | {sm_s:>15} | {diff_s:>12} | {pattern}")

# Count how many in each group were above/below key levels
print(f"\n\n  --- Distribucion de niveles clave ---")
for side, sdf, label in [('LONG', longs, 'Subidas'), ('SHORT', shorts, 'Bajadas')]:
    n = len(sdf)
    print(f"\n  {label} (n={n}):")
    print(f"    RSI < 30 (oversold):     {(sdf['rsi'] < 30).sum():>3}/{n} ({(sdf['rsi'] < 30).mean()*100:.0f}%)")
    print(f"    RSI < 40:                {(sdf['rsi'] < 40).sum():>3}/{n} ({(sdf['rsi'] < 40).mean()*100:.0f}%)")
    print(f"    RSI 40-60 (neutral):     {((sdf['rsi'] >= 40) & (sdf['rsi'] <= 60)).sum():>3}/{n}")
    print(f"    RSI > 60 (overbought):   {(sdf['rsi'] > 60).sum():>3}/{n} ({(sdf['rsi'] > 60).mean()*100:.0f}%)")
    print(f"    RSI > 70:                {(sdf['rsi'] > 70).sum():>3}/{n} ({(sdf['rsi'] > 70).mean()*100:.0f}%)")
    print(f"    Below MA20:              {(sdf['dist_ma20'] < 0).sum():>3}/{n} ({(sdf['dist_ma20'] < 0).mean()*100:.0f}%)")
    print(f"    Below MA50:              {(sdf['dist_ma50'] < 0).sum():>3}/{n} ({(sdf['dist_ma50'] < 0).mean()*100:.0f}%)")
    print(f"    Below MA200:             {(sdf['dist_ma200'].dropna() < 0).sum():>3}/{len(sdf['dist_ma200'].dropna())}")
    print(f"    BB %B < 0.2 (oversold):  {(sdf['bb_pctb'] < 0.2).sum():>3}/{n}")
    print(f"    BB %B > 0.8 (overbought):{(sdf['bb_pctb'] > 0.8).sum():>3}/{n}")
    print(f"    Stoch < 20 (oversold):   {(sdf['stoch_k'] < 20).sum():>3}/{n}")
    print(f"    Stoch > 80 (overbought): {(sdf['stoch_k'] > 80).sum():>3}/{n}")
    print(f"    Ret 1w < -3%:            {(sdf['ret_1w'] < -3).sum():>3}/{n}")
    print(f"    Ret 1w > +3%:            {(sdf['ret_1w'] > 3).sum():>3}/{n}")
    print(f"    Ret 4w < -5%:            {(sdf['ret_4w'] < -5).sum():>3}/{n}")
    print(f"    Ret 4w > +5%:            {(sdf['ret_4w'] > 5).sum():>3}/{n}")
    print(f"    Vol ratio > 1.5 (spike): {(sdf['vol_ratio'] > 1.5).sum():>3}/{n}")
    print(f"    Days since earn < 10:    {(sdf['days_since_earn'] < 10).sum():>3}/{n} (earnings recientes)")
