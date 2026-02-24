"""
REGIMEN DE MERCADO: Pattern Estacional (27% del tiempo)
========================================================
14 semanas del año donde el pattern multi-factor funciona consistentemente.
Ambos lados (long + short) son rentables.

Resultados backtested 2004-2026:
  - P&L: +$748,806 en 302 semanas (27% del tiempo)
  - P&L/semana: +$2,479 | Sharpe: +1.73 | WR: 58%
  - Long: +$494,921 | Short: +$253,882
  - Ambos lados positivos -> señal fuerte

Configuracion:
  - 5L + 5S | $50K/posicion | $500K/semana
  - Entry: Monday open | Exit: next Monday open
  - Cost: 0.3% round-trip
"""

# ============================================================
# SEMANAS OPERABLES (14 de 52)
# ============================================================
OPERABLE_WEEKS = {7, 16, 19, 32, 33, 34, 35, 36, 38, 40, 41, 43, 47, 51}

# Detalle por semana:
WEEK_DETAIL = {
    7:  {'approx': 'Feb ~12',  'pnl': 51860,  'sharpe': 1.77, 'consistency': 55, 'cluster': 'Q4_earnings_tail'},
    16: {'approx': 'Apr ~16',  'pnl': 51724,  'sharpe': 2.57, 'consistency': 67, 'cluster': 'Q1_earnings_peak'},
    19: {'approx': 'May ~07',  'pnl': 39524,  'sharpe': 0.98, 'consistency': 59, 'cluster': 'Q1_earnings_tail'},
    32: {'approx': 'Aug ~06',  'pnl': 38616,  'sharpe': 1.49, 'consistency': 55, 'cluster': 'august_effect'},
    33: {'approx': 'Aug ~13',  'pnl': 69502,  'sharpe': 3.14, 'consistency': 68, 'cluster': 'august_effect'},
    34: {'approx': 'Aug ~20',  'pnl': 135749, 'sharpe': 4.42, 'consistency': 68, 'cluster': 'august_effect'},
    35: {'approx': 'Aug ~27',  'pnl': 12265,  'sharpe': 0.36, 'consistency': 50, 'cluster': 'august_effect'},
    36: {'approx': 'Sep ~03',  'pnl': 25026,  'sharpe': 0.82, 'consistency': 55, 'cluster': 'august_effect'},
    38: {'approx': 'Sep ~17',  'pnl': 48571,  'sharpe': 1.69, 'consistency': 55, 'cluster': 'sep_recovery'},
    40: {'approx': 'Oct ~01',  'pnl': 19385,  'sharpe': 0.78, 'consistency': 50, 'cluster': 'Q2_earnings_start'},
    41: {'approx': 'Oct ~08',  'pnl': 35396,  'sharpe': 0.85, 'consistency': 59, 'cluster': 'Q2_earnings_start'},
    43: {'approx': 'Oct ~22',  'pnl': 58719,  'sharpe': 1.66, 'consistency': 55, 'cluster': 'Q3_earnings_peak'},
    47: {'approx': 'Nov ~19',  'pnl': 95770,  'sharpe': 2.33, 'consistency': 59, 'cluster': 'thanksgiving_rally'},
    51: {'approx': 'Dec ~17',  'pnl': 66699,  'sharpe': 2.63, 'consistency': 55, 'cluster': 'santa_rally'},
}

# Clusters temporales identificados:
CLUSTERS = {
    'Q4_earnings_tail':   {'weeks': [7],          'desc': 'Cola de earnings Q4 (Feb)'},
    'Q1_earnings_peak':   {'weeks': [16],         'desc': 'Pico earnings Q1 (Abr)'},
    'Q1_earnings_tail':   {'weeks': [19],         'desc': 'Cola de earnings Q1 (May)'},
    'august_effect':      {'weeks': [32,33,34,35,36], 'desc': 'Efecto agosto (post-Q2 earnings + vol verano)'},
    'sep_recovery':       {'weeks': [38],         'desc': 'Recuperacion septiembre'},
    'Q2_earnings_start':  {'weeks': [40,41],      'desc': 'Inicio earnings Q3 (Oct)'},
    'Q3_earnings_peak':   {'weeks': [43],         'desc': 'Pico earnings Q3 (Oct)'},
    'thanksgiving_rally': {'weeks': [47],         'desc': 'Rally Thanksgiving (Nov)'},
    'santa_rally':        {'weeks': [51],         'desc': 'Rally Santa Claus (Dic)'},
}

# ============================================================
# PATTERN SCORING: LONG
# "Pullback en tendencia fuerte"
# ============================================================
LONG_WEIGHTS = {
    'sc_ret12w':     0.20,   # High 12w momentum (mas importante)
    'sc_ret1w_inv':  0.12,   # Recent pullback
    'sc_psar_bear':  0.12,   # PSAR BEAR
    'sc_rsi_inv':    0.10,   # Low RSI (oversold)
    'sc_bb_inv':     0.08,   # Low BB%B
    'sc_stoch_inv':  0.08,   # Low Stochastic
    'sc_st_bear':    0.08,   # SuperTrend BEAR
    'sc_margin':     0.07,   # High net margin
    'sc_vol':        0.05,   # High volatility
    'sc_beats':      0.05,   # Earnings beats (4Q)
    'sc_epsgr':      0.05,   # EPS growth YoY
}

LONG_FILTER = {
    'ret_12w_min': 0.10,          # > 10% momentum 12 semanas
    'trend_break': 'PSAR_OR_ST',  # PSAR BEAR or SuperTrend BEAR
}

# ============================================================
# PATTERN SCORING: SHORT
# "Overbought sin fundamento"
# ============================================================
SHORT_WEIGHTS = {
    'sc_psar_bull':    0.15,  # PSAR BULL (mas importante)
    'sc_ret12w_inv':   0.15,  # No long-term momentum
    'sc_ret1w':        0.12,  # Recent rally
    'sc_stoch':        0.12,  # High Stochastic
    'sc_rsi':          0.08,  # High RSI
    'sc_margin_inv':   0.08,  # Low margins
    'sc_st_bull':      0.08,  # SuperTrend BULL
    'sc_bb':           0.07,  # High BB%B
    'sc_divyield':     0.05,  # High dividend yield
    'sc_payout':       0.05,  # High payout ratio
    'sc_debt':         0.05,  # High debt/equity
}

SHORT_FILTER = {
    'psar_bull': True,        # PSAR debe ser BULL
    'ret_12w_max': 0.15,      # < 15% momentum (sin tendencia fuerte)
    'stoch_k_min': 50,        # Stochastic > 50 (zona overbought)
}

# ============================================================
# POSICION SIZING
# ============================================================
POSITION_CONFIG = {
    'n_long': 5,
    'n_short': 5,
    'position_size': 50_000,
    'total_capital': 500_000,
    'cost_pct': 0.003,
}

# ============================================================
# STATS BACKTESTED (2004-2026, 302 semanas operadas)
# ============================================================
BACKTEST_STATS = {
    'period': '2004-01-02 a 2026-02-06',
    'total_weeks': 302,
    'pct_time_active': 27,
    'total_pnl': 748_806,
    'pnl_per_week': 2_479,
    'sharpe': 1.73,
    'win_rate': 58,
    'long_pnl': 494_921,
    'short_pnl': 253_882,
    'long_sharpe': 1.13,
    'years_positive_long': '18/22',
}


def is_operable_week(week_of_year):
    """Returns True if the given ISO week number is in the operable set."""
    return int(week_of_year) in OPERABLE_WEEKS


def get_cluster(week_of_year):
    """Returns the cluster name for a given week, or None."""
    w = int(week_of_year)
    for cname, cdata in CLUSTERS.items():
        if w in cdata['weeks']:
            return cname
    return None


if __name__ == '__main__':
    import datetime
    print("=" * 100)
    print("  REGIMEN PATTERN ESTACIONAL - 14 semanas operables")
    print("=" * 100)
    print(f"\n  Semanas: {sorted(OPERABLE_WEEKS)}")
    print(f"  Tiempo activo: {BACKTEST_STATS['pct_time_active']}%")
    print(f"  P&L backtested: ${BACKTEST_STATS['total_pnl']:>+,}")
    print(f"  Sharpe: {BACKTEST_STATS['sharpe']}")
    print(f"\n  Clusters:")
    for cname, cdata in CLUSTERS.items():
        weeks_str = ','.join(str(w) for w in cdata['weeks'])
        print(f"    {cname:<25s}: sem [{weeks_str}] - {cdata['desc']}")

    # Proximas semanas operables
    today = datetime.date.today()
    current_week = today.isocalendar()[1]
    print(f"\n  Hoy: {today} (semana {current_week})")
    upcoming = sorted([w for w in OPERABLE_WEEKS if w >= current_week])
    if not upcoming:
        upcoming = sorted(OPERABLE_WEEKS)
    print(f"  Proximas semanas operables: {upcoming[:5]}")
    for w in upcoming[:5]:
        approx = datetime.date(today.year, 1, 1) + datetime.timedelta(weeks=w-1)
        is_now = " <-- ESTA SEMANA" if w == current_week else ""
        print(f"    Sem {w}: ~{approx.strftime('%b %d')}{is_now}")
