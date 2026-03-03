"""
Diagnostico: cuantas semanas del backtest tienen eventos activos?
Cuantas realmente generan scores diferenciados (> 5.5 o < 3.5)?
"""
import pandas as pd
import numpy as np
from event_calendar import build_weekly_events
from sector_event_map import SUBSECTORS, EVENT_SUBSECTOR_MAP

MAX_CONTRIBUTION = 4.0

def score_fair(active_events):
    contributions = {}
    for evt_type, intensity in active_events.items():
        if intensity == 0 or evt_type not in EVENT_SUBSECTOR_MAP: continue
        for subsec, impact in EVENT_SUBSECTOR_MAP[evt_type]['impacto'].items():
            if subsec not in contributions: contributions[subsec] = []
            contributions[subsec].append(intensity * impact)
    scores = {}
    for sub_id in SUBSECTORS:
        if sub_id not in contributions or len(contributions[sub_id]) == 0:
            scores[sub_id] = 5.0
        else:
            avg = np.mean(contributions[sub_id])
            scores[sub_id] = max(0.0, min(10.0, 5.0 + (avg / MAX_CONTRIBUTION) * 5.0))
    return scores

weekly_events = build_weekly_events('2000-01-01', '2026-02-21')

print(f"Calendario de eventos: {len(weekly_events)} semanas ({weekly_events.index[0].strftime('%Y-%m-%d')} a {weekly_events.index[-1].strftime('%Y-%m-%d')})")
print(f"Columnas (tipos de evento): {list(weekly_events.columns)}")

# Analizar semana a semana
results = []
for date in weekly_events.index:
    if date.year < 2001: continue
    events_row = weekly_events.loc[date]
    active = {col: events_row[col] for col in events_row.index if events_row[col] > 0}
    n_active = len(active)

    if n_active > 0:
        scores = score_fair(active)
        max_score = max(scores.values())
        min_score = min(scores.values())
        n_above_55 = sum(1 for s in scores.values() if s > 5.5)
        n_below_35 = sum(1 for s in scores.values() if s < 3.5)
        n_above_65 = sum(1 for s in scores.values() if s > 6.5)
        score_range = max_score - min_score
    else:
        max_score = min_score = 5.0
        n_above_55 = n_below_35 = n_above_65 = 0
        score_range = 0

    results.append({
        'date': date, 'year': date.year, 'n_events': n_active,
        'max_score': max_score, 'min_score': min_score, 'score_range': score_range,
        'n_above_55': n_above_55, 'n_below_35': n_below_35, 'n_above_65': n_above_65,
    })

df = pd.DataFrame(results)

print(f"\n{'='*100}")
print(f"  DIAGNOSTICO: EVENTOS ACTIVOS EN EL BACKTEST")
print(f"{'='*100}")

print(f"\n  Total semanas (2001-2026): {len(df)}")
print(f"  Semanas con 0 eventos: {(df['n_events'] == 0).sum()} ({(df['n_events'] == 0).mean()*100:.1f}%)")
print(f"  Semanas con >= 1 evento: {(df['n_events'] > 0).sum()} ({(df['n_events'] > 0).mean()*100:.1f}%)")

print(f"\n  Semanas con score_range = 0 (todos FV = 5.0): {(df['score_range'] == 0).sum()}")
print(f"  Semanas con algun FV > 5.5 (pool LONG): {(df['n_above_55'] > 0).sum()} ({(df['n_above_55'] > 0).mean()*100:.1f}%)")
print(f"  Semanas con algun FV > 6.5 (pool LONG fuerte): {(df['n_above_65'] > 0).sum()} ({(df['n_above_65'] > 0).mean()*100:.1f}%)")
print(f"  Semanas con algun FV < 3.5 (pool SHORT): {(df['n_below_35'] > 0).sum()} ({(df['n_below_35'] > 0).mean()*100:.1f}%)")

print(f"\n  Distribucion de numero de eventos activos por semana:")
for n in sorted(df['n_events'].unique()):
    count = (df['n_events'] == n).sum()
    pct = count / len(df) * 100
    bar = '#' * int(pct)
    print(f"    {n:>2} eventos: {count:>4} semanas ({pct:>5.1f}%) {bar}")

print(f"\n  Distribucion de score range (max - min FV):")
for label, lo, hi in [('0 (sin diferencia)', -0.01, 0.01), ('0-1 (poca)', 0.01, 1),
                        ('1-3 (moderada)', 1, 3), ('3-5 (alta)', 3, 5), ('5+ (muy alta)', 5, 20)]:
    count = ((df['score_range'] > lo) & (df['score_range'] <= hi)).sum()
    pct = count / len(df) * 100
    print(f"    {label:<25s}: {count:>4} semanas ({pct:>5.1f}%)")

# Por ano
print(f"\n  POR ANO:")
print(f"  {'Ano':>5} {'N sem':>6} {'0 evt':>6} {'Con evt':>7} {'FV>5.5':>7} {'FV>6.5':>7} {'FV<3.5':>7} {'Rango avg':>10}")
print(f"  {'-'*65}")
for year in sorted(df['year'].unique()):
    yr = df[df['year'] == year]
    n = len(yr)
    n0 = (yr['n_events'] == 0).sum()
    nev = (yr['n_events'] > 0).sum()
    n55 = (yr['n_above_55'] > 0).sum()
    n65 = (yr['n_above_65'] > 0).sum()
    n35 = (yr['n_below_35'] > 0).sum()
    avg_range = yr['score_range'].mean()
    print(f"  {year:>5} {n:>6} {n0:>6} {nev:>7} {n55:>7} {n65:>7} {n35:>7} {avg_range:>9.2f}")

# Ultimas semanas de 2026
print(f"\n  ULTIMAS SEMANAS 2025-2026:")
recent = df[df['date'] >= '2025-12-01'].tail(15)
for _, row in recent.iterrows():
    print(f"    {row['date'].strftime('%Y-%m-%d')}: {row['n_events']} eventos, "
          f"range={row['score_range']:.2f}, FV>5.5={row['n_above_55']}, FV<3.5={row['n_below_35']}")
