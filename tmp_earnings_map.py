"""Mapear patrones a temporadas de earnings para confirmar la hipotesis."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from regime_pattern_seasonal import OPERABLE_WEEKS, CLUSTERS
from regime_pattern_longonly import OPERABLE_WEEKS_LONG, CLUSTERS_LONG

# Earnings seasons (approximate weeks):
# Q4 earnings: Jan-Feb (weeks ~3-8)
# Q1 earnings: Apr-May (weeks ~14-20)
# Q2 earnings: Jul-Aug (weeks ~28-34)
# Q3 earnings: Oct-Nov (weeks ~40-47)

EARNINGS_SEASONS = {
    'Q4_earnings': list(range(3, 9)),    # Sem 3-8 (Jan-Feb)
    'Q1_earnings': list(range(14, 21)),  # Sem 14-20 (Apr-May)
    'Q2_earnings': list(range(28, 35)),  # Sem 28-34 (Jul-Aug)
    'Q3_earnings': list(range(40, 48)),  # Sem 40-47 (Oct-Nov)
}

# Flatten
all_earnings_weeks = set()
for weeks in EARNINGS_SEASONS.values():
    all_earnings_weeks.update(weeks)

# Pre-earnings: 2-3 weeks before each season
PRE_EARNINGS = {
    'Pre_Q4': [1, 2, 3],         # Sem 1-3 (antes de Q4 earnings en Jan)
    'Pre_Q1': [11, 12, 13],      # Sem 11-13 (antes de Q1 earnings en Apr)
    'Pre_Q2': [25, 26, 27],      # Sem 25-27 (antes de Q2 earnings en Jul)
    'Pre_Q3': [37, 38, 39],      # Sem 37-39 (antes de Q3 earnings en Oct)
}

all_pre_weeks = set()
for weeks in PRE_EARNINGS.values():
    all_pre_weeks.update(weeks)

print("=" * 140)
print("  MAPA: PATRONES vs TEMPORADAS DE EARNINGS")
print("=" * 140)

print(f"\n  CALENDARIO ANUAL:")
print(f"  {'Sem':>4} | {'Mes':>6} | {'Earnings?':>12} | {'Pattern':>12} | {'Tipo':>15} | Cluster")
print(f"  {'-'*4} | {'-'*6} | {'-'*12} | {'-'*12} | {'-'*15} | {'-'*35}")

import datetime

p1_in_earnings = 0
p1_in_pre = 0
p1_other = 0
p2_in_earnings = 0
p2_in_pre = 0
p2_other = 0

for w in range(1, 53):
    approx = datetime.date(2025, 1, 1) + datetime.timedelta(weeks=w-1)
    month = approx.strftime('%b')

    # Earnings season?
    earn_label = ""
    for season, weeks in EARNINGS_SEASONS.items():
        if w in weeks:
            earn_label = season
            break

    pre_label = ""
    for pre, weeks in PRE_EARNINGS.items():
        if w in weeks:
            pre_label = pre
            break

    # Pattern
    if w in OPERABLE_WEEKS:
        pattern = "P1 (L+S)"
        cluster = next((c for c, d in CLUSTERS.items() if w in d['weeks']), '')
        if earn_label: p1_in_earnings += 1
        elif pre_label: p1_in_pre += 1
        else: p1_other += 1
    elif w in OPERABLE_WEEKS_LONG:
        pattern = "P2 (Long)"
        cluster = next((c for c, d in CLUSTERS_LONG.items() if w in d['weeks']), '')
        if earn_label: p2_in_earnings += 1
        elif pre_label: p2_in_pre += 1
        else: p2_other += 1
    else:
        pattern = ""
        cluster = ""

    if pattern or earn_label or pre_label:
        tipo = ""
        if earn_label and pre_label:
            tipo = f"{earn_label}+{pre_label}"
        elif earn_label:
            tipo = earn_label
        elif pre_label:
            tipo = pre_label
        else:
            tipo = "No earnings"

        print(f"  {w:>4} | {month:>6} | {earn_label:>12} | {pattern:>12} | {tipo:>15} | {cluster}")

print(f"\n\n{'='*140}")
print("  RESUMEN: RELACION PATRONES - EARNINGS")
print("=" * 140)

print(f"\n  Pattern 1 (L+S, 14 semanas):")
print(f"    En temporada earnings:    {p1_in_earnings} semanas")
print(f"    Pre-earnings:             {p1_in_pre} semanas")
print(f"    Fuera de earnings:        {p1_other} semanas")
print(f"    % relacionado earnings:   {(p1_in_earnings+p1_in_pre)/14*100:.0f}%")

print(f"\n  Pattern 2 (Solo Longs, 10 semanas):")
print(f"    En temporada earnings:    {p2_in_earnings} semanas")
print(f"    Pre-earnings:             {p2_in_pre} semanas")
print(f"    Fuera de earnings:        {p2_other} semanas")
print(f"    % relacionado earnings:   {(p2_in_earnings+p2_in_pre)/10*100:.0f}%")

# Detailed mapping
print(f"\n\n{'='*140}")
print("  DETALLE POR TEMPORADA DE EARNINGS")
print("=" * 140)

for season_name in ['Q4', 'Q1', 'Q2', 'Q3']:
    earn_key = f'{season_name}_earnings'
    pre_key = f'Pre_{season_name}'
    earn_weeks = EARNINGS_SEASONS[earn_key]
    pre_weeks = PRE_EARNINGS[pre_key]

    print(f"\n  --- {season_name} EARNINGS ---")
    print(f"  Pre-earnings (sem {pre_weeks[0]}-{pre_weeks[-1]}):")
    for w in pre_weeks:
        if w in OPERABLE_WEEKS:
            print(f"    Sem {w}: P1 (L+S)")
        elif w in OPERABLE_WEEKS_LONG:
            print(f"    Sem {w}: P2 (Long)")
        else:
            print(f"    Sem {w}: NO OPERAR")

    print(f"  Durante earnings (sem {earn_weeks[0]}-{earn_weeks[-1]}):")
    for w in earn_weeks:
        if w in OPERABLE_WEEKS:
            print(f"    Sem {w}: P1 (L+S)")
        elif w in OPERABLE_WEEKS_LONG:
            print(f"    Sem {w}: P2 (Long)")
        else:
            print(f"    Sem {w}: NO OPERAR")

# Visual timeline
print(f"\n\n{'='*140}")
print("  TIMELINE VISUAL (E=Earnings, p=Pre-earnings, .=nada)")
print("=" * 140)

line1 = "  Sem:     "
line2 = "  Earnings:"
line3 = "  Pattern: "

for w in range(1, 53):
    line1 += f"{w:>3}"

    if w in all_earnings_weeks:
        line2 += "  E"
    elif w in all_pre_weeks:
        line2 += "  p"
    else:
        line2 += "  ."

    if w in OPERABLE_WEEKS:
        line3 += " P1"
    elif w in OPERABLE_WEEKS_LONG:
        line3 += " P2"
    else:
        line3 += "  ."

print(line1)
print(line2)
print(line3)

# Count weeks by category
print(f"\n\n{'='*140}")
print("  TABLA CRUZADA: EARNINGS × PATTERN")
print("=" * 140)

categories = {
    'Earnings + P1': len([w for w in range(1,53) if w in all_earnings_weeks and w in OPERABLE_WEEKS]),
    'Earnings + P2': len([w for w in range(1,53) if w in all_earnings_weeks and w in OPERABLE_WEEKS_LONG]),
    'Earnings + Nada': len([w for w in range(1,53) if w in all_earnings_weeks and w not in OPERABLE_WEEKS and w not in OPERABLE_WEEKS_LONG]),
    'Pre-earn + P1': len([w for w in range(1,53) if w in all_pre_weeks and w in OPERABLE_WEEKS]),
    'Pre-earn + P2': len([w for w in range(1,53) if w in all_pre_weeks and w in OPERABLE_WEEKS_LONG]),
    'Pre-earn + Nada': len([w for w in range(1,53) if w in all_pre_weeks and w not in OPERABLE_WEEKS and w not in OPERABLE_WEEKS_LONG]),
    'Fuera + P1': len([w for w in range(1,53) if w not in all_earnings_weeks and w not in all_pre_weeks and w in OPERABLE_WEEKS]),
    'Fuera + P2': len([w for w in range(1,53) if w not in all_earnings_weeks and w not in all_pre_weeks and w in OPERABLE_WEEKS_LONG]),
    'Fuera + Nada': len([w for w in range(1,53) if w not in all_earnings_weeks and w not in all_pre_weeks and w not in OPERABLE_WEEKS and w not in OPERABLE_WEEKS_LONG]),
}

print(f"\n  {'Categoria':>25s} | {'Semanas':>8}")
print(f"  {'-'*25} | {'-'*8}")
for cat, n in categories.items():
    print(f"  {cat:>25s} | {n:>8}")
print(f"  {'-'*25} | {'-'*8}")
print(f"  {'TOTAL':>25s} | {sum(categories.values()):>8}")
