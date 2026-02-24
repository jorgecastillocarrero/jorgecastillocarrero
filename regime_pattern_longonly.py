"""
REGIMEN DE MERCADO: Pattern Long-Only (Pattern 2)
===================================================
10 semanas del año donde SOLO el lado LONG tiene alpha consistente.
El lado SHORT destruye valor en estas semanas -> NO operar shorts.

Complementa al Pattern 1 (regime_pattern_seasonal.py) que opera L+S.

Logica: se activan en pre-earnings y inicio de earnings (anticipacion alcista).
  - P2 semanas cubren 90% relacionado con temporadas de earnings.

Resultados backtested 2003-2026 (10 longs):
  - Long P&L: +$511,120 en 225 semanas (~19% del tiempo)
  - P&L/semana: +$2,272 | Sharpe: +1.29 | WR: 61%
  - Picks 1-5: +$390K (Sharpe 1.68) + Picks 6-10: +$121K (Sharpe 0.56)
  - Short P&L en mismas semanas: NEGATIVO (toxico)

Configuracion:
  - 10L (solo longs) | $50K/posicion | $500K/semana
  - Entry: Monday open | Exit: next Monday open
  - Cost: 0.3% round-trip
  - Scoring: mismo pattern multi-factor que Pattern 1 (pullback en tendencia fuerte)
"""

# ============================================================
# SEMANAS OPERABLES - SOLO LONGS (10 de 52)
# ============================================================
OPERABLE_WEEKS_LONG = {4, 5, 6, 11, 12, 25, 28, 31, 44, 50}

# Detalle por semana (solo metricas LONG):
WEEK_DETAIL_LONG = {
    4:  {'approx': 'Jan ~22', 'long_pnl': 32875,  'sharpe': 1.09, 'consistency': 57, 'cluster': 'january_effect'},
    5:  {'approx': 'Jan ~29', 'long_pnl': 52724,  'sharpe': 1.98, 'consistency': 70, 'cluster': 'january_effect'},
    6:  {'approx': 'Feb ~05', 'long_pnl': 49395,  'sharpe': 1.31, 'consistency': 65, 'cluster': 'january_effect'},
    11: {'approx': 'Mar ~12', 'long_pnl': 41191,  'sharpe': 1.98, 'consistency': 55, 'cluster': 'Q4_earnings_late'},
    12: {'approx': 'Mar ~19', 'long_pnl': 34505,  'sharpe': 2.09, 'consistency': 58, 'cluster': 'Q4_earnings_late'},
    25: {'approx': 'Jun ~18', 'long_pnl': 53928,  'sharpe': 1.69, 'consistency': 59, 'cluster': 'june_window'},
    28: {'approx': 'Jul ~09', 'long_pnl': 77788,  'sharpe': 3.38, 'consistency': 59, 'cluster': 'summer_earnings'},
    31: {'approx': 'Jul ~30', 'long_pnl': 62437,  'sharpe': 2.79, 'consistency': 59, 'cluster': 'summer_earnings'},
    44: {'approx': 'Oct ~29', 'long_pnl': 59875,  'sharpe': 2.08, 'consistency': 45, 'cluster': 'Q3_earnings_late'},
    50: {'approx': 'Dec ~10', 'long_pnl': 60456,  'sharpe': 2.13, 'consistency': 68, 'cluster': 'december_rally'},
}

# Clusters temporales:
CLUSTERS_LONG = {
    'january_effect':     {'weeks': [4, 5, 6],    'desc': 'Efecto enero / momentum principio de año'},
    'Q4_earnings_late':   {'weeks': [11, 12],      'desc': 'Cola earnings Q4 (Mar) - buenos longs'},
    'june_window':        {'weeks': [25],          'desc': 'Ventana junio pre-rebalanceo'},
    'summer_earnings':    {'weeks': [28, 31],      'desc': 'Earnings Q2 verano (Jul) - longs fuertes'},
    'Q3_earnings_late':   {'weeks': [44],          'desc': 'Cola earnings Q3 (Oct-Nov)'},
    'december_rally':     {'weeks': [50],          'desc': 'Rally diciembre pre-navidad'},
}

# ============================================================
# SCORING: usa los mismos LONG_WEIGHTS de Pattern 1
# (importar de regime_pattern_seasonal si se necesitan)
# ============================================================
# El pattern de scoring es identico: "Pullback en tendencia fuerte"
# Solo se aplica el lado LONG, sin shorts.

# ============================================================
# POSICION SIZING (solo longs)
# ============================================================
POSITION_CONFIG_LONG = {
    'n_long': 10,
    'n_short': 0,
    'position_size': 50_000,
    'total_capital': 500_000,
    'cost_pct': 0.003,
}

# ============================================================
# STATS BACKTESTED (2003-2026, 225 semanas, 10 longs)
# ============================================================
BACKTEST_STATS_LONG = {
    'period': '2003-01-03 a 2026-02-06',
    'total_weeks': 225,
    'pct_time_active': 19,
    'long_pnl': 511_120,
    'pnl_per_week': 2_272,
    'sharpe': 1.29,
    'win_rate': 61,
    'top5_pnl': 389_983,
    'top5_sharpe': 1.68,
    'extra5_pnl': 121_137,
    'extra5_sharpe': 0.56,
    'short_pnl_same_weeks': -521_825,
    'note': 'NO operar shorts en estas semanas - 10L mejor en 15/24 años (62%)',
}

# Semanas donde los picks 6-10 aportan mas valor (Sharpe > 0.3):
WEEKS_10L_STRONG = {4, 11, 28, 50}  # AÑADIR 10L
WEEKS_10L_NEUTRAL = {5, 25, 31, 44}  # neutral, 10L OK
WEEKS_5L_ONLY = {6, 12}              # picks 6-10 restan valor


def is_longonly_week(week_of_year):
    """Returns True if the given ISO week number is a long-only week."""
    return int(week_of_year) in OPERABLE_WEEKS_LONG


def get_cluster_long(week_of_year):
    """Returns the cluster name for a long-only week, or None."""
    w = int(week_of_year)
    for cname, cdata in CLUSTERS_LONG.items():
        if w in cdata['weeks']:
            return cname
    return None


if __name__ == '__main__':
    import datetime
    from regime_pattern_seasonal import OPERABLE_WEEKS

    print("=" * 100)
    print("  REGIMEN PATTERN LONG-ONLY (Pattern 2) - 10 semanas, 10 longs")
    print("=" * 100)
    print(f"\n  Semanas Long-Only: {sorted(OPERABLE_WEEKS_LONG)}")
    print(f"  Semanas Pattern 1 (L+S): {sorted(OPERABLE_WEEKS)}")
    print(f"  Total cubierto: {len(OPERABLE_WEEKS | OPERABLE_WEEKS_LONG)} semanas ({len(OPERABLE_WEEKS | OPERABLE_WEEKS_LONG)/52*100:.0f}% del año)")
    print(f"\n  Clusters:")
    for cname, cdata in CLUSTERS_LONG.items():
        weeks_str = ','.join(str(w) for w in cdata['weeks'])
        total_pnl = sum(WEEK_DETAIL_LONG[w]['long_pnl'] for w in cdata['weeks'])
        avg_sh = sum(WEEK_DETAIL_LONG[w]['sharpe'] for w in cdata['weeks']) / len(cdata['weeks'])
        print(f"    {cname:<25s}: sem [{weeks_str}] - {cdata['desc']}")
        print(f"      Long P&L: ${total_pnl:>+,} | Sharpe avg: {avg_sh:>+.2f}")

    print(f"\n  Detalle por semana:")
    print(f"  {'Sem':>4} | {'Fecha':>10} | {'Long P&L':>11} | {'Sharpe':>6} | {'Consist':>8} | Cluster")
    print(f"  {'-'*4} | {'-'*10} | {'-'*11} | {'-'*6} | {'-'*8} | {'-'*25}")
    total_long = 0
    for w in sorted(OPERABLE_WEEKS_LONG):
        d = WEEK_DETAIL_LONG[w]
        total_long += d['long_pnl']
        print(f"  {w:>4} | {d['approx']:>10} | ${d['long_pnl']:>+10,} | {d['sharpe']:>+5.2f} | {d['consistency']:>6}% | {d['cluster']}")
    print(f"  {'-'*4} | {'-'*10} | {'-'*11} | {'-'*6} | {'-'*8} |")
    print(f"  {'TOT':>4} | {'':>10} | ${total_long:>+10,} |        |         |")

    # Proximas semanas
    today = datetime.date.today()
    current_week = today.isocalendar()[1]
    all_operable = OPERABLE_WEEKS | OPERABLE_WEEKS_LONG
    upcoming = sorted([w for w in all_operable if w >= current_week])
    if not upcoming:
        upcoming = sorted(all_operable)
    print(f"\n  Hoy: {today} (semana {current_week})")
    print(f"  Proximas semanas operables (ambos patrones):")
    for w in upcoming[:8]:
        approx = datetime.date(today.year, 1, 1) + datetime.timedelta(weeks=w-1)
        regime = "L+S (P1)" if w in OPERABLE_WEEKS else "LONG (P2)"
        is_now = " <-- ESTA SEMANA" if w == current_week else ""
        print(f"    Sem {w}: ~{approx.strftime('%b %d')} [{regime}]{is_now}")
