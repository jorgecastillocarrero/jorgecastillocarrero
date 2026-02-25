"""
Backtest: Event-Driven Sub-Sector Anticipation Strategy
========================================================
Layer 1: Detect world events from market data proxies
         -> Score sub-sectors using event impact map
         -> Aggregate to sector ETF level
         -> Long best sectors, Short worst sectors

Uses proxy-based event detection (oil, gold, bonds, copper, VIX, banks)
to anticipate money flows into/out of sectors BEFORE momentum shows up.
"""

import pandas as pd
import numpy as np
import psycopg2
from collections import Counter
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP

DB = 'postgresql://fmp:fmp123@localhost:5433/fmp_data'

# ── SECTOR ETFs ──
SECTOR_ETFS = ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']

# ── PROXY TICKERS for event detection ──
PROXY_TICKERS = [
    'SPY',   # Market
    'XLE',   # Oil proxy (1998+, before USO exists)
    'USO',   # Oil ETF (2006+)
    'NEM',   # Gold miner (1991+, before GLD)
    'GLD',   # Gold ETF (2004+)
    'FCX',   # Copper/China (1995+)
    'TLT',   # Bonds/Rates (2002+)
    'HYG',   # Credit stress (2007+)
    'XLF',   # Banks (1998+)
    'KRE',   # Regional banks (2006+)
    'ITB',   # Homebuilders (2006+)
    'XLY',   # Consumer discretionary
    'XLP',   # Consumer staples
]


def load_weekly_prices():
    """Load weekly (Friday close) prices for proxies + sector ETFs."""
    conn = psycopg2.connect(DB)

    all_syms = list(set(PROXY_TICKERS + SECTOR_ETFS))
    ph = ','.join(['%s'] * len(all_syms))

    df = pd.read_sql(
        f"SELECT symbol, date, close, volume FROM fmp_price_history "
        f"WHERE symbol IN ({ph}) ORDER BY date",
        conn, params=all_syms, parse_dates=['date']
    )
    conn.close()

    prices = df.pivot_table(index='date', columns='symbol', values='close')
    volumes = df.pivot_table(index='date', columns='symbol', values='volume')

    # Resample to weekly Friday
    prices_w = prices.resample('W-FRI').last()
    volumes_w = volumes.resample('W-FRI').sum()

    return prices_w, volumes_w


# ═══════════════════════════════════════════════════════════════
# SIGNAL CALCULATION
# ═══════════════════════════════════════════════════════════════

def calculate_signals(prices):
    """Calculate z-scored signals from proxy prices."""
    sig = pd.DataFrame(index=prices.index)

    def mom(s, n):
        return s.pct_change(n)

    def zs(s, w=52):
        m = s.rolling(w, min_periods=20).mean()
        sd = s.rolling(w, min_periods=20).std()
        return (s - m) / sd.replace(0, np.nan)

    # ── OIL ── (USO preferred, XLE fallback for pre-2006)
    if 'USO' in prices.columns:
        sig['oil_m4'] = zs(mom(prices['USO'], 4))
        sig['oil_m12'] = zs(mom(prices['USO'], 12))
    if 'XLE' in prices.columns:
        xle_m4 = zs(mom(prices['XLE'], 4))
        xle_m12 = zs(mom(prices['XLE'], 12))
        if 'oil_m4' in sig.columns:
            sig['oil_m4'] = sig['oil_m4'].fillna(xle_m4)
            sig['oil_m12'] = sig['oil_m12'].fillna(xle_m12)
        else:
            sig['oil_m4'] = xle_m4
            sig['oil_m12'] = xle_m12

    # ── GOLD ── (GLD preferred, NEM fallback for pre-2004)
    if 'GLD' in prices.columns:
        sig['gold_m4'] = zs(mom(prices['GLD'], 4))
        sig['gold_m12'] = zs(mom(prices['GLD'], 12))
    if 'NEM' in prices.columns:
        nem_m4 = zs(mom(prices['NEM'], 4))
        nem_m12 = zs(mom(prices['NEM'], 12))
        if 'gold_m4' in sig.columns:
            sig['gold_m4'] = sig['gold_m4'].fillna(nem_m4)
            sig['gold_m12'] = sig['gold_m12'].fillna(nem_m12)
        else:
            sig['gold_m4'] = nem_m4
            sig['gold_m12'] = nem_m12

    # ── COPPER / CHINA ──
    if 'FCX' in prices.columns:
        sig['copper_m4'] = zs(mom(prices['FCX'], 4))
        sig['copper_m12'] = zs(mom(prices['FCX'], 12))

    # ── BONDS / RATES ── (TLT down = rates up)
    if 'TLT' in prices.columns:
        sig['bonds_m4'] = zs(mom(prices['TLT'], 4))
        sig['bonds_m12'] = zs(mom(prices['TLT'], 12))

    # ── CREDIT STRESS ── (HYG vs TLT relative)
    if 'HYG' in prices.columns and 'TLT' in prices.columns:
        spread = mom(prices['HYG'], 4) - mom(prices['TLT'], 4)
        sig['credit_m4'] = zs(spread)

    # ── MARKET ──
    if 'SPY' in prices.columns:
        spy_ret = prices['SPY'].pct_change()
        sig['vol_4w'] = zs(spy_ret.rolling(20).std())
        sig['spy_m4'] = zs(mom(prices['SPY'], 4))
        sig['spy_m12'] = zs(mom(prices['SPY'], 12))

    # ── BANKS ──
    if 'XLF' in prices.columns:
        sig['banks_m4'] = zs(mom(prices['XLF'], 4))
    if 'KRE' in prices.columns:
        kre_m4 = zs(mom(prices['KRE'], 4))
        if 'banks_m4' in sig.columns:
            sig['regbanks_m4'] = kre_m4
        else:
            sig['banks_m4'] = kre_m4

    # ── HOUSING ──
    if 'ITB' in prices.columns:
        sig['housing_m4'] = zs(mom(prices['ITB'], 4))

    # ── CONSUMER APPETITE (XLY/XLP ratio) ──
    if 'XLY' in prices.columns and 'XLP' in prices.columns:
        ratio = prices['XLY'] / prices['XLP']
        sig['consumer_m4'] = zs(mom(ratio, 4))

    return sig


# ═══════════════════════════════════════════════════════════════
# EVENT DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_events(row):
    """Detect active events from a row of signals.

    Returns dict: event_name -> intensity (typically 0.3 to 2.0)
    """
    events = {}

    def g(col):
        v = row.get(col, np.nan)
        return 0 if pd.isna(v) else float(v)

    def clp(v):
        return max(0.3, min(2.0, abs(v)))

    oil   = g('oil_m4')
    gold  = g('gold_m4')
    bonds = g('bonds_m4')
    vol   = g('vol_4w')
    spy   = g('spy_m4')
    copper = g('copper_m4')
    banks = g('banks_m4')
    regb  = g('regbanks_m4')
    hous  = g('housing_m4')
    cons  = g('consumer_m4')
    credit = g('credit_m4')

    # ── OIL ──
    if oil > 1.0:
        events['precio_petroleo_sube'] = clp(oil)
    if oil < -1.0:
        events['precio_petroleo_baja'] = clp(oil)

    # ── RATES ── (bonds down = rates up)
    if bonds < -1.0:
        events['subida_tipos_interes'] = clp(bonds)
    if bonds > 1.0:
        events['bajada_tipos_interes'] = clp(bonds)

    # ── INFLATION ── (gold up + oil up + bonds down)
    if gold > 0.7 and oil > 0.5 and bonds < -0.3:
        events['inflacion_alta'] = clp((gold + abs(bonds)) / 2)

    # ── GEOPOLITICS ──
    # Middle East war: oil spike + gold spike + volatility
    if oil > 1.0 and gold > 0.8 and vol > 0.5:
        events['guerra_medio_oriente'] = clp(min(oil, gold))

    # Russia/Ukraine: oil very high + gold + vol
    if oil > 1.5 and gold > 1.0 and vol > 0.8:
        events['guerra_rusia_ucrania'] = clp(oil * 0.6)

    # China/Taiwan: copper down + gold up (risk-off + China impact)
    if copper < -0.8 and gold > 0.5 and spy < 0:
        events['tension_china_taiwan'] = clp(abs(copper) * 0.7)

    # Terrorism/sudden shock: vol spike + gold up but oil neutral
    if vol > 1.5 and gold > 0.5 and abs(oil) < 1.0:
        events['terrorismo'] = clp(vol * 0.5)

    # Sanctions: detected indirectly through multiple signals
    if oil > 0.5 and copper < -0.5:
        events['sanciones_comerciales'] = clp(0.5)

    # ── ECONOMIC CYCLE ──
    # Recession: SPY down + vol up + credit stress
    if spy < -1.0 and vol > 0.8:
        intensity = clp(abs(spy))
        if credit < -0.5:
            intensity = min(2.0, intensity * 1.3)
        events['recesion'] = intensity

    # Growth: SPY up + vol low + copper up
    if spy > 0.8 and vol < 0 and copper > 0:
        events['crecimiento_economico'] = clp(spy)

    # Banking crisis: regional banks crash + vol + credit stress
    bank_sig = min(banks, regb) if regb != 0 else banks
    if bank_sig < -1.5 and vol > 0.8:
        events['crisis_bancaria'] = clp(abs(bank_sig))

    # Fiscal stimulus: copper up + SPY up + construction/infra
    if copper > 0.8 and spy > 0.5:
        events['estimulo_fiscal'] = clp(copper * 0.5)

    # ── CHINA / COPPER ──
    if copper > 1.0:
        events['demanda_china_fuerte'] = clp(copper)
    if copper < -1.0:
        events['china_desaceleracion'] = clp(copper)

    # ── HOUSING ──
    if hous > 1.0:
        events['boom_inmobiliario'] = clp(hous)
    if hous < -1.0:
        events['crisis_inmobiliaria'] = clp(hous)

    # ── CONSUMER ──
    if cons > 1.0:
        events['confianza_consumidor_alta'] = clp(cons)
    if cons < -1.0:
        events['crisis_consumo'] = clp(cons)

    # ── PANDEMIC ── (extreme vol + SPY crash)
    if vol > 2.0 and spy < -2.0:
        events['pandemia'] = clp(vol * 0.7)

    # ── ENERGY TRANSITION ── (oil falling, SPY stable)
    if oil < -0.8 and spy > -0.3:
        events['transicion_energetica'] = clp(abs(oil) * 0.4)

    return events


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

def score_subsectors(events):
    """Score each sub-sector based on detected events.

    score = sum(event_intensity * impact_score)
    """
    scores = {}
    for event_name, intensity in events.items():
        if event_name not in EVENT_SUBSECTOR_MAP:
            continue
        for subsec, impact in EVENT_SUBSECTOR_MAP[event_name]['impacto'].items():
            scores[subsec] = scores.get(subsec, 0) + intensity * impact
    return scores


def aggregate_to_sectors(subsector_scores):
    """Aggregate sub-sector scores to sector ETF level.

    Weighted by number of impacts and max absolute score within sector.
    """
    sector_scores = {}
    sector_details = {}

    for subsec_id, subsec_data in SUBSECTORS.items():
        etf = subsec_data['etf']
        score = subsector_scores.get(subsec_id, 0)
        if score != 0:
            sector_scores.setdefault(etf, []).append(score)
            sector_details.setdefault(etf, []).append((subsec_id, score))

    # Aggregate: sum (not average) because more impacts = stronger signal
    result = {}
    for etf in SECTOR_ETFS:
        subscores = sector_scores.get(etf, [])
        if subscores:
            # Use sum of scores - more sub-sectors impacted = stronger signal
            result[etf] = sum(subscores)
        else:
            result[etf] = 0

    return result, sector_details


# ═══════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════

def backtest(signals, etf_returns,
             n_long=3, n_short=3,
             capital_per_side=500_000,
             momentum_decay=0.6,
             min_score=0.3):
    """Backtest event-driven sector rotation.

    Args:
        signals: proxy signals DataFrame
        etf_returns: weekly returns of sector ETFs
        n_long/n_short: max positions each side
        capital_per_side: $ per side
        momentum_decay: persistence of previous score (0=no memory, 1=full memory)
        min_score: minimum absolute score to open position
    """
    common = signals.index.intersection(etf_returns.index).sort_values()
    results = []
    prev_sector_scores = {etf: 0 for etf in SECTOR_ETFS}

    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]

        if i < 52:  # warmup
            continue

        # 1. Detect events
        events = detect_events(signals.loc[date])

        # 2. Score sub-sectors
        subsec_scores = score_subsectors(events)

        # 3. Aggregate to sector ETFs
        raw_sector, details = aggregate_to_sectors(subsec_scores)

        # 4. Blend with previous scores (momentum persistence)
        blended = {}
        for etf in SECTOR_ETFS:
            raw = raw_sector.get(etf, 0)
            prev = prev_sector_scores.get(etf, 0)
            blended[etf] = raw + momentum_decay * prev
        prev_sector_scores = blended.copy()

        # 5. Rank and select
        ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        longs = [(etf, sc) for etf, sc in ranked[:n_long] if sc > min_score]
        shorts = [(etf, sc) for etf, sc in ranked[-n_short:] if sc < -min_score]

        # 6. Calculate next week PnL
        long_pnl = 0
        short_pnl = 0
        n_l = len(longs)
        n_s = len(shorts)

        if n_l > 0:
            cap = capital_per_side / n_l
            for etf, _ in longs:
                r = etf_returns.loc[next_date].get(etf, 0)
                if pd.isna(r):
                    r = 0
                long_pnl += cap * r

        if n_s > 0:
            cap = capital_per_side / n_s
            for etf, _ in shorts:
                r = etf_returns.loc[next_date].get(etf, 0)
                if pd.isna(r):
                    r = 0
                short_pnl += cap * (-r)

        results.append({
            'date': next_date,
            'year': next_date.year,
            'n_events': len(events),
            'events': '|'.join(events.keys()) if events else '',
            'n_longs': n_l,
            'n_shorts': n_s,
            'longs': ','.join(e for e, _ in longs),
            'shorts': ','.join(e for e, _ in shorts),
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'total_pnl': long_pnl + short_pnl,
            'scores': blended.copy(),
        })

    return pd.DataFrame(results)


def print_results(results, label=""):
    """Print annual results table."""
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    print(f"\n  {'Year':>6s} {'PnL':>12s} {'Long':>10s} {'Short':>10s} {'Weeks':>6s} {'Active':>6s} {'Win%':>6s}")
    print("  " + "-" * 62)

    total = 0
    yrs_pos = 0
    yrs_tot = 0
    all_weekly = []

    for year in sorted(results['year'].unique()):
        yr = results[results['year'] == year]
        pnl = yr['total_pnl'].sum()
        lpnl = yr['long_pnl'].sum()
        spnl = yr['short_pnl'].sum()
        n = len(yr)
        active = len(yr[(yr['n_longs'] > 0) | (yr['n_shorts'] > 0)])
        wins = len(yr[yr['total_pnl'] > 0])
        wpct = wins / n * 100 if n > 0 else 0

        total += pnl
        yrs_tot += 1
        if pnl > 0:
            yrs_pos += 1
        all_weekly.extend(yr['total_pnl'].tolist())

        s = '+' if pnl > 0 else '-' if pnl < 0 else ' '
        print(f"  {year:>6d} {s}${abs(pnl):>10,.0f} ${lpnl:>+9,.0f} ${spnl:>+9,.0f} {n:>6d} {active:>6d} {wpct:>5.1f}%")

    print("  " + "-" * 62)

    aw = np.array(all_weekly)
    sharpe = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
    cum = np.cumsum(aw)
    dd = (cum - np.maximum.accumulate(cum)).min()

    print(f"  {'TOTAL':>6s}  ${total:>11,.0f}   Sharpe: {sharpe:.2f}   MaxDD: ${dd:,.0f}")
    print(f"  Positive years: {yrs_pos}/{yrs_tot}   Avg weekly: ${aw.mean():,.0f}")

    return total, sharpe, yrs_pos, yrs_tot


# ═══════════════════════════════════════════════════════════════
# BASELINE: Pure momentum (for comparison)
# ═══════════════════════════════════════════════════════════════

def backtest_momentum_baseline(prices_w, etf_returns, n_long=3, n_short=3,
                                capital_per_side=500_000, lookback=4):
    """Simple sector momentum baseline: long winners, short losers."""
    common = prices_w.index.intersection(etf_returns.index).sort_values()
    results = []

    for i in range(len(common) - 1):
        date = common[i]
        next_date = common[i + 1]
        if i < 52:
            continue

        # Momentum of each sector ETF
        moms = {}
        for etf in SECTOR_ETFS:
            if etf in prices_w.columns:
                if i >= lookback:
                    cur = prices_w.loc[date, etf]
                    prev_date = common[i - lookback]
                    prev = prices_w.loc[prev_date, etf]
                    if pd.notna(cur) and pd.notna(prev) and prev > 0:
                        moms[etf] = cur / prev - 1

        if len(moms) < 5:
            continue

        ranked = sorted(moms.items(), key=lambda x: x[1], reverse=True)
        longs = ranked[:n_long]
        shorts = ranked[-n_short:]

        long_pnl = 0
        short_pnl = 0
        cap_l = capital_per_side / len(longs)
        cap_s = capital_per_side / len(shorts)

        for etf, _ in longs:
            r = etf_returns.loc[next_date].get(etf, 0)
            if pd.isna(r): r = 0
            long_pnl += cap_l * r

        for etf, _ in shorts:
            r = etf_returns.loc[next_date].get(etf, 0)
            if pd.isna(r): r = 0
            short_pnl += cap_s * (-r)

        results.append({
            'date': next_date, 'year': next_date.year,
            'n_events': 0, 'events': '',
            'n_longs': len(longs), 'n_shorts': len(shorts),
            'longs': ','.join(e for e, _ in longs),
            'shorts': ','.join(e for e, _ in shorts),
            'long_pnl': long_pnl, 'short_pnl': short_pnl,
            'total_pnl': long_pnl + short_pnl,
            'scores': {},
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BACKTEST: Event-Driven Sub-Sector Anticipation")
    print("  30 events x 49 sub-sectors -> 9 sector ETFs")
    print("=" * 70)

    # 1. Load prices
    print("\n1. Loading weekly prices...")
    prices_w, volumes_w = load_weekly_prices()
    print(f"   {len(prices_w.columns)} tickers, {len(prices_w)} weeks")
    print(f"   {prices_w.index[0].strftime('%Y-%m-%d')} -> {prices_w.index[-1].strftime('%Y-%m-%d')}")

    # 2. Calculate signals
    print("\n2. Calculating proxy signals...")
    signals = calculate_signals(prices_w)
    valid_sigs = {col: signals[col].dropna() for col in signals.columns}
    for col in signals.columns:
        v = valid_sigs[col]
        print(f"   {col:18s}: {len(v):5d} weeks  ({v.index[0].strftime('%Y-%m-%d')} ->)")

    # 3. ETF returns
    etf_returns = prices_w[SECTOR_ETFS].pct_change()

    # 4. Validate against known events
    print("\n3. Event detection validation:")
    known = [
        ('2001-09-14', '9/11 Attacks'),
        ('2003-03-21', 'Iraq War'),
        ('2008-09-19', 'Lehman Brothers'),
        ('2008-10-10', 'Crisis peak'),
        ('2011-08-05', 'US downgrade'),
        ('2015-08-21', 'China devaluation'),
        ('2016-02-12', 'Oil crash'),
        ('2018-12-21', 'Rate hike selloff'),
        ('2020-03-20', 'COVID crash'),
        ('2022-02-25', 'Russia/Ukraine'),
        ('2022-06-17', 'Rate hikes'),
        ('2023-03-17', 'SVB crisis'),
    ]
    for ds, label in known:
        target = pd.Timestamp(ds)
        idx = signals.index.get_indexer([target], method='nearest')[0]
        d = signals.index[idx]
        evts = detect_events(signals.loc[d])
        estr = ', '.join(f'{k}({v:.1f})' for k, v in sorted(evts.items(), key=lambda x: -x[1]))
        print(f"   {d.strftime('%Y-%m-%d')} [{label:22s}]: {estr or '(none)'}")

    # 5. Backtest event strategy
    print("\n4. Backtesting event strategy...")

    # Test multiple configs
    configs = [
        ("3L+3S decay=0.6 thr=0.3", 3, 3, 0.6, 0.3),
        ("3L+3S decay=0.7 thr=0.5", 3, 3, 0.7, 0.5),
        ("3L+3S decay=0.5 thr=0.3", 3, 3, 0.5, 0.3),
        ("4L+4S decay=0.6 thr=0.3", 4, 4, 0.6, 0.3),
        ("2L+2S decay=0.6 thr=0.5", 2, 2, 0.6, 0.5),
    ]

    best_sharpe = -999
    best_config = None
    best_results = None

    for label, nl, ns, decay, thr in configs:
        res = backtest(signals, etf_returns, n_long=nl, n_short=ns,
                       momentum_decay=decay, min_score=thr)
        if res.empty:
            continue

        aw = res['total_pnl'].values
        total = aw.sum()
        sharpe = aw.mean() / aw.std() * np.sqrt(52) if aw.std() > 0 else 0
        active = len(res[(res['n_longs'] > 0) | (res['n_shorts'] > 0)])
        active_pct = active / len(res) * 100

        print(f"   {label:30s}: PnL ${total:>+11,.0f}  Sharpe {sharpe:>+.2f}  Active {active_pct:.0f}%")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_config = label
            best_results = res

    # 6. Print best config details
    print(f"\n   Best config: {best_config}")
    total, sharpe, yp, yt = print_results(best_results, f"EVENT STRATEGY: {best_config}")

    # 7. Baseline comparison
    print("\n5. Baseline: Pure Sector Momentum (4-week)...")
    base_res = backtest_momentum_baseline(prices_w, etf_returns, n_long=3, n_short=3)
    if not base_res.empty:
        print_results(base_res, "BASELINE: 4-week Momentum 3L+3S")

    # 8. Event frequency
    print(f"\n6. Event Frequency (in best config):")
    ecounts = Counter()
    for estr in best_results['events']:
        if estr:
            for e in estr.split('|'):
                ecounts[e] += 1

    for ev, cnt in ecounts.most_common(20):
        pct = cnt / len(best_results) * 100
        # Get impacted sub-sectors
        n_impacts = len(EVENT_SUBSECTOR_MAP.get(ev, {}).get('impacto', {}))
        print(f"   {ev:35s}: {cnt:4d} weeks ({pct:5.1f}%)  -> {n_impacts} sub-sectors")

    # 9. Sector selection frequency
    print(f"\n7. Sector Selection Frequency:")
    lc = Counter()
    sc = Counter()
    for _, row in best_results.iterrows():
        if row['longs']:
            for s in row['longs'].split(','):
                lc[s] += 1
        if row['shorts']:
            for s in row['shorts'].split(','):
                sc[s] += 1

    print(f"   {'LONG':20s}  {'SHORT':20s}")
    ll = lc.most_common(9)
    sl = sc.most_common(9)
    for i in range(max(len(ll), len(sl))):
        ls = f"{ll[i][0]} ({ll[i][1]})" if i < len(ll) else ""
        ss = f"{sl[i][0]} ({sl[i][1]})" if i < len(sl) else ""
        print(f"   {ls:20s}  {ss:20s}")

    # 10. Sub-sector score analysis for latest period
    print(f"\n8. Latest Sub-Sector Scores (last available date):")
    last_date = signals.index[-1]
    evts = detect_events(signals.loc[last_date])
    if evts:
        print(f"   Date: {last_date.strftime('%Y-%m-%d')}")
        print(f"   Active events: {', '.join(f'{k}({v:.1f})' for k, v in sorted(evts.items(), key=lambda x: -x[1]))}")
        sscore = score_subsectors(evts)
        sector_agg, details = aggregate_to_sectors(sscore)
        ranked = sorted(sector_agg.items(), key=lambda x: x[1], reverse=True)
        print(f"\n   {'ETF':>5s} {'Score':>8s}  Sub-sector breakdown")
        print("   " + "-" * 60)
        for etf, score in ranked:
            det = details.get(etf, [])
            det_str = ", ".join(f"{s}({sc:+.1f})" for s, sc in sorted(det, key=lambda x: -x[1]))
            print(f"   {etf:>5s} {score:>+8.1f}  {det_str}")
    else:
        print(f"   No events detected on {last_date.strftime('%Y-%m-%d')}")

    return best_results


if __name__ == '__main__':
    main()
