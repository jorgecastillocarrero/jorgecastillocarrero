#!/usr/bin/env python3
"""
EPS Analysis for February 2026 Seasonality Strategy - 33 Candidate Stocks
=========================================================================
Queries TWO databases:
  1. Railway PostgreSQL (fundamentals: TTM EPS, earnings_growth, q0 quarterly data)
  2. FMP Local Docker PostgreSQL (fmp_earnings: granular quarterly earnings)

For each symbol, retrieves the MOST RECENT earnings report BEFORE January 30, 2026.
Also computes YoY EPS growth by comparing to the same quarter one year earlier.
"""

import psycopg2
from decimal import Decimal
from datetime import date

# ── Configuration ──────────────────────────────────────────────────────────────

RAILWAY_DSN = "postgresql://postgres:TWevONOeueNlJYYDmVGNFVLQKnQwGuWN@shuttle.proxy.rlwy.net:53628/railway"
FMP_DSN     = "postgresql://fmp:fmp123@localhost:5433/fmp_data"

CUTOFF_DATE = date(2026, 1, 30)

SYMBOLS = [
    "AXON", "ENTG", "NET",  "HUBS", "STLD", "ABBV", "HEI",  "NOC",
    "GE",   "UNP",  "TJX",  "PAA",  "ESTC", "ATI",  "NVDA", "CLF",
    "SAIA", "THC",  "AMAT", "ODFL", "IDXX", "ARES", "HII",  "TDG",
    "ALGN", "SYK",  "HBAN", "LMT",  "CHD",  "TECH", "REGN", "BSX",
    "AVGO",
]

# ── Helper ─────────────────────────────────────────────────────────────────────

def to_float(v):
    """Convert Decimal/int/float to float, or return None."""
    if v is None:
        return None
    return float(v)

def fmt(v, decimals=2, pct=False):
    """Format a number for display."""
    if v is None:
        return "N/A"
    if pct:
        return f"{v:+.{decimals}f}%"
    return f"{v:.{decimals}f}"

def fmt_rev(v):
    """Format revenue in billions."""
    if v is None:
        return "N/A"
    return f"${v/1e9:.2f}B"

def fmt_surprise(v):
    """Format surprise with sign."""
    if v is None:
        return "N/A"
    return f"{v:+.4f}"

# ── Database 1: FMP Docker (fmp_earnings) ──────────────────────────────────────

def query_fmp_earnings(symbols, cutoff):
    """
    For each symbol, get:
      - Most recent earnings BEFORE cutoff (the 'current' quarter)
      - The earnings from ~1 year earlier (for YoY calculation)
    """
    conn = psycopg2.connect(FMP_DSN)
    cur = conn.cursor()

    results = {}
    placeholders = ",".join(["%s"] * len(symbols))

    # ── Most recent earnings before cutoff for each symbol ──
    cur.execute(f"""
        SELECT DISTINCT ON (symbol)
            symbol, date, eps_actual, eps_estimated,
            revenue_actual, revenue_estimated
        FROM fmp_earnings
        WHERE symbol IN ({placeholders})
          AND date < %s
          AND eps_actual IS NOT NULL
        ORDER BY symbol, date DESC
    """, symbols + [cutoff])

    for row in cur.fetchall():
        sym, dt, eps_a, eps_e, rev_a, rev_e = row
        eps_a = to_float(eps_a)
        eps_e = to_float(eps_e)
        rev_a = to_float(rev_a)
        rev_e = to_float(rev_e)

        surprise = None
        surprise_pct = None
        if eps_a is not None and eps_e is not None and eps_e != 0:
            surprise = eps_a - eps_e
            surprise_pct = (surprise / abs(eps_e)) * 100
        elif eps_a is not None and eps_e is not None:
            surprise = eps_a - eps_e

        rev_surprise_pct = None
        if rev_a is not None and rev_e is not None and rev_e != 0:
            rev_surprise_pct = ((rev_a - rev_e) / abs(rev_e)) * 100

        beat = None
        if eps_a is not None and eps_e is not None:
            beat = eps_a > eps_e

        results[sym] = {
            "date":             dt,
            "eps_actual":       eps_a,
            "eps_estimated":    eps_e,
            "eps_surprise":     surprise,
            "eps_surprise_pct": surprise_pct,
            "eps_beat":         beat,
            "rev_actual":       rev_a,
            "rev_estimated":    rev_e,
            "rev_surprise_pct": rev_surprise_pct,
        }

    # ── YoY: for each symbol get the earnings ~1 year before the current quarter ──
    for sym, data in results.items():
        if data["date"] is None:
            continue
        # Look for earnings between 330 and 400 days before current date
        yoy_start = date(data["date"].year - 1, max(data["date"].month - 2, 1), 1)
        yoy_end   = date(data["date"].year - 1, min(data["date"].month + 2, 12), 28)

        cur.execute("""
            SELECT date, eps_actual
            FROM fmp_earnings
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
              AND eps_actual IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
        """, (sym, yoy_start, yoy_end))

        row = cur.fetchone()
        if row and row[1] is not None:
            yoy_eps = to_float(row[1])
            data["yoy_eps_prev"]  = yoy_eps
            data["yoy_date_prev"] = row[0]
            if yoy_eps != 0:
                data["yoy_growth_pct"] = ((data["eps_actual"] - yoy_eps) / abs(yoy_eps)) * 100
            else:
                data["yoy_growth_pct"] = None
        else:
            data["yoy_eps_prev"]   = None
            data["yoy_date_prev"]  = None
            data["yoy_growth_pct"] = None

    conn.close()
    return results


# ── Database 2: Railway PostgreSQL (fundamentals) ─────────────────────────────

def query_railway_fundamentals(symbols, cutoff):
    """
    Get TTM EPS, earnings_growth, pe_ratio, and q0 quarterly data from Railway.
    Two queries:
      1. Latest fundamentals BEFORE cutoff (for TTM EPS, earnings_growth context)
      2. Latest fundamentals overall (may include q0 data reported after cutoff)
    """
    conn = psycopg2.connect(RAILWAY_DSN)
    cur = conn.cursor()

    placeholders = ",".join(["%s"] * len(symbols))

    # ── Latest fundamentals snapshot BEFORE cutoff ──
    cur.execute(f"""
        SELECT DISTINCT ON (s.code)
            s.code, f.data_date, f.eps, f.earnings_growth, f.pe_ratio, f.revenue
        FROM fundamentals f
        JOIN symbols s ON f.symbol_id = s.id
        WHERE s.code IN ({placeholders})
          AND f.data_date <= %s
        ORDER BY s.code, f.data_date DESC
    """, symbols + [cutoff])

    results = {}
    for row in cur.fetchall():
        code, dt, eps, eg, pe, rev = row
        results[code] = {
            "rwy_date":            dt,
            "rwy_eps_ttm":         to_float(eps),
            "rwy_earnings_growth": to_float(eg),
            "rwy_pe_ratio":        to_float(pe),
            "rwy_revenue":         to_float(rev),
        }

    # ── Latest fundamentals with q0 data (may be after cutoff, from recent reporting) ──
    cur.execute(f"""
        SELECT DISTINCT ON (s.code)
            s.code, f.data_date,
            f.q0_eps_actual, f.q0_eps_estimate, f.q0_eps_difference,
            f.q0_eps_surprise_pct, f.q0_quarter_end
        FROM fundamentals f
        JOIN symbols s ON f.symbol_id = s.id
        WHERE s.code IN ({placeholders})
          AND f.q0_eps_actual IS NOT NULL
        ORDER BY s.code, f.data_date DESC
    """, symbols)

    for row in cur.fetchall():
        code, dt, q0a, q0e, q0d, q0s, q0end = row
        if code not in results:
            results[code] = {}
        results[code]["rwy_q0_actual"]      = to_float(q0a)
        results[code]["rwy_q0_estimate"]    = to_float(q0e)
        results[code]["rwy_q0_difference"]  = to_float(q0d)
        results[code]["rwy_q0_surprise_pct"]= to_float(q0s)
        results[code]["rwy_q0_quarter_end"] = q0end

    conn.close()
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 120)
    print("EPS ANALYSIS - February 2026 Seasonality Strategy (33 Candidates)")
    print(f"Cutoff date: {CUTOFF_DATE} (most recent earnings BEFORE this date)")
    print("=" * 120)

    # Query both databases
    print("\n[1/2] Querying FMP Docker PostgreSQL (fmp_earnings) ...")
    fmp_data = query_fmp_earnings(SYMBOLS, CUTOFF_DATE)
    print(f"      Retrieved earnings data for {len(fmp_data)} symbols.")

    print("[2/2] Querying Railway PostgreSQL (fundamentals) ...")
    rwy_data = query_railway_fundamentals(SYMBOLS, CUTOFF_DATE)
    print(f"      Retrieved fundamentals for {len(rwy_data)} symbols.")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 1: FMP Earnings Data (most recent quarter before Jan 30, 2026)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 140)
    print("TABLE 1: FMP EARNINGS - Most Recent Quarter BEFORE Jan 30, 2026")
    print("=" * 140)
    header = (
        f"{'Symbol':<7} {'Report Date':>11} "
        f"{'EPS Act':>8} {'EPS Est':>8} {'Surprise':>9} {'Surpr %':>9} {'Beat':>5} "
        f"{'Rev Act':>11} {'Rev Est':>11} {'Rev Surp%':>9} "
        f"{'YoY Prev':>9} {'YoY Date':>11} {'YoY Grw%':>9}"
    )
    print(header)
    print("-" * 140)

    beat_count = 0
    miss_count = 0
    total_with_data = 0

    for sym in SYMBOLS:
        d = fmp_data.get(sym)
        if d is None:
            print(f"{sym:<7} {'--- no data ---'}")
            continue

        total_with_data += 1
        if d["eps_beat"] is True:
            beat_count += 1
            beat_str = "YES"
        elif d["eps_beat"] is False:
            miss_count += 1
            beat_str = "NO"
        else:
            beat_str = "N/A"

        print(
            f"{sym:<7} {str(d['date']):>11} "
            f"{fmt(d['eps_actual']):>8} {fmt(d['eps_estimated']):>8} "
            f"{fmt_surprise(d['eps_surprise']):>9} "
            f"{fmt(d['eps_surprise_pct'], 2, pct=True):>9} {beat_str:>5} "
            f"{fmt_rev(d['rev_actual']):>11} {fmt_rev(d['rev_estimated']):>11} "
            f"{fmt(d['rev_surprise_pct'], 2, pct=True):>9} "
            f"{fmt(d.get('yoy_eps_prev')):>9} "
            f"{str(d.get('yoy_date_prev', 'N/A')):>11} "
            f"{fmt(d.get('yoy_growth_pct'), 1, pct=True):>9}"
        )

    print("-" * 140)
    print(f"Beat rate: {beat_count}/{total_with_data} ({beat_count/total_with_data*100:.1f}%)   |   "
          f"Miss: {miss_count}/{total_with_data} ({miss_count/total_with_data*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 2: Railway Fundamentals (TTM EPS, Earnings Growth, PE)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 120)
    print("TABLE 2: RAILWAY FUNDAMENTALS - Snapshot as of ~Jan 26-28, 2026")
    print("=" * 120)
    header2 = (
        f"{'Symbol':<7} {'Date':>11} "
        f"{'EPS(TTM)':>9} {'Earn Grw':>9} {'P/E':>8} {'Revenue(TTM)':>13}"
    )
    print(header2)
    print("-" * 120)

    for sym in SYMBOLS:
        d = rwy_data.get(sym, {})
        rwy_date = d.get("rwy_date", None)
        print(
            f"{sym:<7} {str(rwy_date) if rwy_date else 'N/A':>11} "
            f"{fmt(d.get('rwy_eps_ttm')):>9} "
            f"{fmt(d.get('rwy_earnings_growth'), 1, pct=True) if d.get('rwy_earnings_growth') is not None else 'N/A':>9} "
            f"{fmt(d.get('rwy_pe_ratio'), 1):>8} "
            f"{fmt_rev(d.get('rwy_revenue')):>13}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 3: Railway q0 Quarterly Data (latest available, may be post-cutoff)
    # ══════════════════════════════════════════════════════════════════════════
    q0_symbols = [sym for sym in SYMBOLS if rwy_data.get(sym, {}).get("rwy_q0_actual") is not None]
    print("\n\n")
    print("=" * 110)
    print("TABLE 3: RAILWAY Q0 QUARTERLY EARNINGS (latest available, may include post-Jan-30 reports)")
    print(f"         ({len(q0_symbols)} of 33 symbols have q0 data)")
    print("=" * 110)
    header3 = (
        f"{'Symbol':<7} {'Qtr End':>11} "
        f"{'Q0 Actual':>10} {'Q0 Est':>10} {'Q0 Diff':>10} {'Q0 Surpr%':>10}"
    )
    print(header3)
    print("-" * 110)

    for sym in SYMBOLS:
        d = rwy_data.get(sym, {})
        if d.get("rwy_q0_actual") is None:
            continue
        print(
            f"{sym:<7} {str(d.get('rwy_q0_quarter_end', 'N/A')):>11} "
            f"{fmt(d.get('rwy_q0_actual')):>10} "
            f"{fmt(d.get('rwy_q0_estimate')):>10} "
            f"{fmt_surprise(d.get('rwy_q0_difference')):>10} "
            f"{fmt(d.get('rwy_q0_surprise_pct'), 2, pct=True):>10}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 4: Combined Summary - Key EPS Metrics Per Symbol
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 150)
    print("TABLE 4: COMBINED SUMMARY - EPS Metrics for Seasonality Strategy")
    print("=" * 150)
    header4 = (
        f"{'Symbol':<7} "
        f"{'EPS Act':>8} {'EPS Est':>8} {'Surpr%':>8} {'Beat':>5} "
        f"{'YoY Grw%':>9} "
        f"{'TTM EPS':>8} {'Earn Grw':>9} {'P/E':>7} "
        f"{'Rev Surp%':>9} "
        f"{'Score':>6}"
    )
    print(header4)
    print("-" * 150)

    # Score: positive surprise + YoY growth + beat = higher score
    scores = {}
    for sym in SYMBOLS:
        fmp = fmp_data.get(sym, {})
        rwy = rwy_data.get(sym, {})

        eps_a = fmp.get("eps_actual")
        eps_e = fmp.get("eps_estimated")
        surp_pct = fmp.get("eps_surprise_pct")
        beat = fmp.get("eps_beat")
        yoy = fmp.get("yoy_growth_pct")
        ttm = rwy.get("rwy_eps_ttm")
        eg  = rwy.get("rwy_earnings_growth")
        pe  = rwy.get("rwy_pe_ratio")
        rev_s = fmp.get("rev_surprise_pct")

        # Simple composite score: surprise% (capped) + YoY growth% (capped) + beat bonus
        score = 0
        if surp_pct is not None:
            score += min(max(surp_pct, -20), 20)  # cap at +/-20
        if yoy is not None:
            score += min(max(yoy * 0.3, -15), 15)  # weighted, capped
        if beat is True:
            score += 2
        if rev_s is not None and rev_s > 0:
            score += min(rev_s, 5)
        scores[sym] = score

        beat_str = "YES" if beat is True else ("NO" if beat is False else "N/A")
        eg_str = fmt(eg, 1, pct=True) if eg is not None else "N/A"

        print(
            f"{sym:<7} "
            f"{fmt(eps_a):>8} {fmt(eps_e):>8} "
            f"{fmt(surp_pct, 2, pct=True):>8} {beat_str:>5} "
            f"{fmt(yoy, 1, pct=True):>9} "
            f"{fmt(ttm):>8} {eg_str:>9} {fmt(pe, 1):>7} "
            f"{fmt(rev_s, 2, pct=True):>9} "
            f"{score:>6.1f}"
        )

    print("-" * 150)

    # Ranking by score
    print("\n\n")
    print("=" * 80)
    print("RANKING BY EPS COMPOSITE SCORE (Higher = Better)")
    print("=" * 80)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (sym, sc) in enumerate(ranked, 1):
        fmp = fmp_data.get(sym, {})
        beat_str = "BEAT" if fmp.get("eps_beat") is True else ("MISS" if fmp.get("eps_beat") is False else "N/A")
        surp = fmp.get("eps_surprise_pct")
        yoy = fmp.get("yoy_growth_pct")
        print(
            f"  {i:2d}. {sym:<6}  Score: {sc:>6.1f}  |  "
            f"Surprise: {fmt(surp, 2, pct=True):>9}  |  "
            f"YoY: {fmt(yoy, 1, pct=True):>9}  |  {beat_str}"
        )

    # Summary stats
    print("\n")
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    surprises = [fmp_data[s]["eps_surprise_pct"] for s in SYMBOLS if s in fmp_data and fmp_data[s]["eps_surprise_pct"] is not None]
    yoys      = [fmp_data[s]["yoy_growth_pct"]   for s in SYMBOLS if s in fmp_data and fmp_data[s].get("yoy_growth_pct") is not None]
    rev_surps = [fmp_data[s]["rev_surprise_pct"]  for s in SYMBOLS if s in fmp_data and fmp_data[s].get("rev_surprise_pct") is not None]

    beats = sum(1 for s in SYMBOLS if s in fmp_data and fmp_data[s].get("eps_beat") is True)
    total = sum(1 for s in SYMBOLS if s in fmp_data and fmp_data[s].get("eps_beat") is not None)

    print(f"  Symbols with earnings data: {len(fmp_data)}/33")
    print(f"  EPS Beat Rate:   {beats}/{total} ({beats/total*100:.1f}%)" if total > 0 else "  EPS Beat Rate: N/A")
    if surprises:
        print(f"  Avg EPS Surprise%:  {sum(surprises)/len(surprises):+.2f}%")
        print(f"  Median EPS Surprise%: {sorted(surprises)[len(surprises)//2]:+.2f}%")
        print(f"  Min / Max Surprise%: {min(surprises):+.2f}% / {max(surprises):+.2f}%")
    if yoys:
        print(f"  Avg YoY EPS Growth: {sum(yoys)/len(yoys):+.1f}%")
        positive_yoy = sum(1 for y in yoys if y > 0)
        print(f"  Positive YoY Growth: {positive_yoy}/{len(yoys)} ({positive_yoy/len(yoys)*100:.1f}%)")
    if rev_surps:
        print(f"  Avg Revenue Surprise%: {sum(rev_surps)/len(rev_surps):+.2f}%")
        rev_beats = sum(1 for r in rev_surps if r > 0)
        print(f"  Revenue Beat Rate: {rev_beats}/{len(rev_surps)} ({rev_beats/len(rev_surps)*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
