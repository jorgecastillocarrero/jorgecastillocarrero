"""
Validate: Do the 43 event types + 49 subsectors produce correct scores?
Check each event type, its subsector impacts, and whether the scores
match reality for known historical periods.
"""
import pandas as pd
import numpy as np
from collections import Counter
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP
from event_calendar import EVENT_CALENDAR, build_weekly_events

SECTOR_ETFS = ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']


# ═══════════════════════════════════════════════════════════════
# 1. EVENT MAP COVERAGE
# ═══════════════════════════════════════════════════════════════
print(f"{'=' * 100}")
print(f"  1. EVENT MAP: 43 event types x 49 subsectors")
print(f"{'=' * 100}")

all_event_types = sorted(EVENT_SUBSECTOR_MAP.keys())
print(f"\n  {len(all_event_types)} event types defined in EVENT_SUBSECTOR_MAP:")

for i, evt in enumerate(all_event_types):
    impacts = EVENT_SUBSECTOR_MAP[evt]['impacto']
    pos = sum(1 for v in impacts.values() if v > 0)
    neg = sum(1 for v in impacts.values() if v < 0)
    pos_sum = sum(v for v in impacts.values() if v > 0)
    neg_sum = sum(v for v in impacts.values() if v < 0)
    print(f"  {i+1:>3d}. {evt:>35s}: {len(impacts):>2d} subsectors ({pos}+ {neg}-) net={pos_sum+neg_sum:>+3d}")

# Subsectors with NO events mapped
all_subsectors = set(SUBSECTORS.keys())
mapped_subsectors = set()
for evt_data in EVENT_SUBSECTOR_MAP.values():
    mapped_subsectors.update(evt_data['impacto'].keys())

unmapped = all_subsectors - mapped_subsectors
if unmapped:
    print(f"\n  WARNING: {len(unmapped)} subsectors with NO events mapped:")
    for s in sorted(unmapped):
        print(f"    - {s} ({SUBSECTORS[s]['etf']}: {SUBSECTORS[s]['label']})")

# Events per subsector
print(f"\n  Events per subsector:")
subsec_event_count = Counter()
subsec_pos_events = Counter()
subsec_neg_events = Counter()
for evt_name, evt_data in EVENT_SUBSECTOR_MAP.items():
    for subsec, impact in evt_data['impacto'].items():
        subsec_event_count[subsec] += 1
        if impact > 0:
            subsec_pos_events[subsec] += 1
        else:
            subsec_neg_events[subsec] += 1

current_etf = None
for subsec_id, subsec_data in SUBSECTORS.items():
    etf = subsec_data['etf']
    if etf != current_etf:
        print(f"\n  {etf}:")
        current_etf = etf
    total = subsec_event_count.get(subsec_id, 0)
    pos = subsec_pos_events.get(subsec_id, 0)
    neg = subsec_neg_events.get(subsec_id, 0)
    balance = "BALANCED" if abs(pos - neg) <= 1 else "POS BIAS" if pos > neg else "NEG BIAS"
    print(f"    {subsec_data['label']:>35s}: {total:>2d} events ({pos:>2d}+ {neg:>2d}-)  {balance}")


# ═══════════════════════════════════════════════════════════════
# 2. CALENDAR COVERAGE
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 100}")
print(f"  2. EVENT CALENDAR: {len(EVENT_CALENDAR)} entries")
print(f"{'=' * 100}")

cal_types = Counter()
cal_years = Counter()
for entry in EVENT_CALENDAR:
    evt_type = entry[0]
    start = entry[1]
    year = int(start[:4])
    cal_types[evt_type] += 1
    cal_years[year] += 1

print(f"\n  Event types used in calendar:")
for evt, cnt in cal_types.most_common():
    in_map = "OK" if evt in EVENT_SUBSECTOR_MAP else "NOT IN MAP!"
    n_impacts = len(EVENT_SUBSECTOR_MAP.get(evt, {}).get('impacto', {}))
    print(f"    {evt:>35s}: {cnt:>3d} entries  ({n_impacts} subsectors)  {in_map}")

# Event types in map but NOT in calendar
map_types = set(EVENT_SUBSECTOR_MAP.keys())
cal_type_set = set(cal_types.keys())
unused = map_types - cal_type_set
if unused:
    print(f"\n  WARNING: {len(unused)} event types defined but NEVER used in calendar:")
    for evt in sorted(unused):
        n = len(EVENT_SUBSECTOR_MAP[evt]['impacto'])
        print(f"    - {evt} ({n} subsectors impacted)")

# Calendar entries not in map
not_in_map = cal_type_set - map_types
if not_in_map:
    print(f"\n  ERROR: {len(not_in_map)} calendar entries with NO event map:")
    for evt in sorted(not_in_map):
        print(f"    - {evt} ({cal_types[evt]} entries)")

print(f"\n  Events per year:")
for year in sorted(cal_years.keys()):
    bar = "#" * cal_years[year]
    print(f"    {year}: {cal_years[year]:>3d} {bar}")


# ═══════════════════════════════════════════════════════════════
# 3. SUBSECTOR SCORES vs REALITY for key periods
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 100}")
print(f"  3. SUBSECTOR SCORES vs REALITY")
print(f"{'=' * 100}")

def score_week(active_events):
    scores = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP:
            continue
        impacts = EVENT_SUBSECTOR_MAP[evt_type]['impacto']
        for subsec, impact in impacts.items():
            scores[subsec] = scores.get(subsec, 0) + intensity * impact
    return scores

weekly_events = build_weekly_events('1999-01-01', '2026-03-01')

# Key test dates
test_periods = [
    ("2008-09-19", "Lehman collapse (peak crisis)"),
    ("2008-03-14", "Bear Stearns collapse"),
    ("2020-03-20", "COVID crash peak"),
    ("2020-11-13", "Post-vaccine rally"),
    ("2022-06-17", "Fed hikes + inflation"),
    ("2007-08-10", "Subprime starts"),
    ("2003-03-21", "Iraq war starts"),
    ("2025-02-07", "Current period"),
]

for test_date, description in test_periods:
    td = pd.Timestamp(test_date)
    # Find nearest Friday
    if td not in weekly_events.index:
        nearest = weekly_events.index[weekly_events.index.get_indexer([td], method='nearest')[0]]
        td = nearest

    events_row = weekly_events.loc[td]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}

    if not active:
        print(f"\n  {description} ({td.strftime('%Y-%m-%d')}): No events active")
        continue

    scores = score_week(active)

    # Aggregate to sectors
    sector_agg = {}
    for subsec_id, subsec_data in SUBSECTORS.items():
        etf = subsec_data['etf']
        sc = scores.get(subsec_id, 0)
        if sc != 0:
            sector_agg.setdefault(etf, []).append((subsec_id, subsec_data['label'], sc))

    print(f"\n  {description} ({td.strftime('%Y-%m-%d')})")
    print(f"  Active: {', '.join(f'{k}({v:.0f})' for k,v in active.items())}")
    print(f"  {'Subsector':>35s} {'Score':>7}  {'ETF':>5}")
    print("  " + "-" * 55)

    # Sort all subsectors by score
    all_scores = [(subsec_id, SUBSECTORS[subsec_id]['label'], SUBSECTORS[subsec_id]['etf'], scores.get(subsec_id, 0))
                  for subsec_id in SUBSECTORS]
    all_scores.sort(key=lambda x: x[3], reverse=True)

    # Show top 10 and bottom 10
    print("  TOP 10 (should be LONG):")
    for subsec_id, label, etf, sc in all_scores[:10]:
        marker = " <<" if sc > 2 else ""
        print(f"  {label:>35s} {sc:>+7.1f}  {etf:>5}{marker}")

    print("  ...")
    print("  BOTTOM 10 (should be SHORT):")
    for subsec_id, label, etf, sc in all_scores[-10:]:
        marker = " <<" if sc < -2 else ""
        print(f"  {label:>35s} {sc:>+7.1f}  {etf:>5}{marker}")

    # Net score
    total_pos = sum(sc for _, _, _, sc in all_scores if sc > 0)
    total_neg = sum(sc for _, _, _, sc in all_scores if sc < 0)
    n_pos = sum(1 for _, _, _, sc in all_scores if sc > 0)
    n_neg = sum(1 for _, _, _, sc in all_scores if sc < 0)
    n_zero = sum(1 for _, _, _, sc in all_scores if sc == 0)
    print(f"  Net: {n_pos} positive (sum={total_pos:+.0f}), {n_neg} negative (sum={total_neg:+.0f}), {n_zero} neutral")
    print(f"  Short ratio: {abs(total_neg)/(total_pos+abs(total_neg))*100:.0f}%" if (total_pos+abs(total_neg)) > 0 else "")


# ═══════════════════════════════════════════════════════════════
# 4. BIAS CHECK: across all weeks, how balanced are scores?
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 100}")
print(f"  4. SYSTEMATIC BIAS CHECK")
print(f"{'=' * 100}")

all_subsec_totals = Counter()
all_subsec_weeks = Counter()

for date in weekly_events.index:
    if date.year < 2000:
        continue
    events_row = weekly_events.loc[date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    if not active:
        continue
    scores = score_week(active)
    for subsec, sc in scores.items():
        all_subsec_totals[subsec] += sc
        all_subsec_weeks[subsec] += 1

print(f"\n  Cumulative subsector scores (2000-2026) - sorted by total:")
print(f"  {'Subsector':>35s} {'ETF':>5} {'Total':>8} {'Weeks':>6} {'Avg/wk':>8} {'Bias':>10}")
print("  " + "-" * 80)

by_total = sorted(all_subsec_totals.items(), key=lambda x: x[1], reverse=True)
for subsec, total in by_total:
    if subsec in SUBSECTORS:
        etf = SUBSECTORS[subsec]['etf']
        label = SUBSECTORS[subsec]['label']
        weeks = all_subsec_weeks[subsec]
        avg = total / weeks if weeks > 0 else 0
        bias = "STRONG POS" if avg > 3 else "POS" if avg > 1 else "STRONG NEG" if avg < -3 else "NEG" if avg < -1 else "OK"
        print(f"  {label:>35s} {etf:>5} {total:>+8.0f} {weeks:>6d} {avg:>+7.1f} {bias:>10s}")

# Zero-score subsectors
zero_subsecs = [s for s in SUBSECTORS if s not in all_subsec_totals]
if zero_subsecs:
    print(f"\n  Subsectors with ZERO score across all weeks:")
    for s in zero_subsecs:
        print(f"    - {SUBSECTORS[s]['label']} ({SUBSECTORS[s]['etf']})")

print(f"\n{'=' * 100}")
