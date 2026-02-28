import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('data/regimenes_historico.csv')

REGIME_COLORS = {
    'BURBUJA': '#e91e63', 'GOLDILOCKS': '#4caf50', 'ALCISTA': '#2196f3',
    'NEUTRAL': '#ff9800', 'CAUTIOUS': '#ff5722', 'BEARISH': '#795548',
    'CRISIS': '#9c27b0', 'PANICO': '#f44336', 'CAPITULACION': '#00bcd4',
    'RECOVERY': '#8bc34a',
}
REGIME_ORDER = ['BURBUJA','GOLDILOCKS','ALCISTA','NEUTRAL','CAUTIOUS','BEARISH','RECOVERY','CRISIS','PANICO','CAPITULACION']

# Summary
summary = df.groupby('regime').agg(
    n=('regime', 'count'),
    avg_ret=('spy_ret_pct', 'mean'),
    wr=('spy_ret_pct', lambda x: (x > 0).sum() / x.notna().sum() * 100 if x.notna().sum() > 0 else 0),
    total_ret=('spy_ret_pct', 'sum'),
    avg_score=('total', 'mean'),
).reset_index()
summary['pct'] = summary['n'] / summary['n'].sum() * 100
order_map = {r: i for i, r in enumerate(REGIME_ORDER)}
summary['order'] = summary['regime'].map(order_map).fillna(99)
summary = summary.sort_values('order')

def val_class(v):
    if v > 0: return 'pos'
    if v < 0: return 'neg'
    return 'neutral'

html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Regimenes de Mercado 2001-2026</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; background: #fff; color: #222; margin: 20px; }
h1 { color: #1565c0; text-align: center; margin-bottom: 5px; }
h2 { color: #333; margin-top: 30px; margin-bottom: 10px; border-bottom: 2px solid #1565c0; padding-bottom: 5px; }
.subtitle { text-align: center; color: #666; margin-bottom: 25px; font-size: 14px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 12px; }
th { background: #1565c0; color: #fff; padding: 6px 5px; text-align: center; border: 1px solid #ccc; position: sticky; top: 0; z-index: 1; cursor: pointer; }
th:hover { background: #0d47a1; }
td { padding: 5px; text-align: center; border: 1px solid #ddd; }
tr:nth-child(even) { background: #f5f7fa; }
tr:hover { background: #e3f2fd; }
.pos { color: #2e7d32; font-weight: bold; }
.neg { color: #c62828; font-weight: bold; }
.neutral { color: #999; }
td.left { text-align: left; }
.regime-pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; color: #fff; min-width: 70px; }
.summary-table { width: auto; margin: 0 auto 25px auto; min-width: 800px; font-size: 13px; }
.summary-table td { padding: 8px 12px; }
.summary-table th { background: #455a64; padding: 8px 12px; }
.filter-bar { margin: 15px 0; text-align: center; }
.filter-btn { display: inline-block; padding: 5px 12px; margin: 3px; border-radius: 15px; font-size: 12px; font-weight: bold; color: #fff; cursor: pointer; border: 2px solid transparent; opacity: 0.7; }
.filter-btn.active { opacity: 1; border-color: #333; }
.filter-btn:hover { opacity: 1; }
.year-header { background: #263238 !important; color: #fff; font-weight: bold; font-size: 13px; }
.year-header td { padding: 8px; text-align: left; }
#searchBox { padding: 6px 12px; font-size: 13px; border: 1px solid #ccc; border-radius: 5px; width: 200px; margin-right: 15px; }
.stats-row { background: #e8eaf6 !important; font-weight: bold; }
</style>
</head>
<body>

<h1>Regimenes de Mercado - Historico Semanal</h1>
<p class="subtitle">1313 semanas | Enero 2001 - Febrero 2026 | Senal viernes/jueves &rarr; Trading lunes/viernes</p>

<h2>Resumen por Regimen</h2>
<table class="summary-table">
<tr><th>Regimen</th><th>N</th><th>% Semanas</th><th>Avg Ret SPY</th><th>Win Rate</th><th>Ret Total</th><th>Score Medio</th></tr>
"""

for _, r in summary.iterrows():
    color = REGIME_COLORS.get(r['regime'], '#666')
    html += f'<tr><td><span class="regime-pill" style="background:{color};">{r["regime"]}</span></td>'
    html += f'<td>{int(r["n"])}</td><td>{r["pct"]:.1f}%</td>'
    html += f'<td class="{val_class(r["avg_ret"])}">{r["avg_ret"]:+.2f}%</td>'
    wr_class = 'pos' if r['wr'] >= 55 else ('neg' if r['wr'] < 48 else 'neutral')
    html += f'<td class="{wr_class}">{r["wr"]:.1f}%</td>'
    html += f'<td class="{val_class(r["total_ret"])}">{r["total_ret"]:+.1f}%</td>'
    html += f'<td>{r["avg_score"]:+.1f}</td></tr>\n'

html += "</table>\n"

# Filters
html += '<div class="filter-bar">\n'
html += '<input type="text" id="searchBox" placeholder="Buscar ano o semana..." onkeyup="filterTable()">\n'
for regime in REGIME_ORDER:
    color = REGIME_COLORS.get(regime, '#666')
    html += f'<span class="filter-btn active" style="background:{color};" onclick="toggleRegime(this, \'{regime}\')">{regime}</span>\n'
html += '<br><br><button onclick="showAll()" style="padding:5px 15px;cursor:pointer;">Mostrar todos</button>\n'
html += '<button onclick="showYear(2026)" style="padding:5px 15px;cursor:pointer;margin-left:10px;">Solo 2026</button>\n'
html += '<button onclick="showYear(2025)" style="padding:5px 15px;cursor:pointer;margin-left:5px;">Solo 2025</button>\n'
html += '<button onclick="showYear(2024)" style="padding:5px 15px;cursor:pointer;margin-left:5px;">Solo 2024</button>\n'
html += '</div>\n'

# Main table
html += """
<h2>Detalle Semana a Semana</h2>
<table id="mainTable">
<tr>
<th onclick="sortTable(0)">Fecha Senal</th>
<th onclick="sortTable(1)">Ano</th>
<th onclick="sortTable(2)">Sem</th>
<th onclick="sortTable(3)">Regimen</th>
<th onclick="sortTable(4)">Score</th>
<th onclick="sortTable(5)">BDD</th>
<th onclick="sortTable(6)">BRSI</th>
<th onclick="sortTable(7)">DDP</th>
<th onclick="sortTable(8)">SPY</th>
<th onclick="sortTable(9)">MOM</th>
<th onclick="sortTable(10)">SPY $</th>
<th onclick="sortTable(11)">Dist MA200</th>
<th onclick="sortTable(12)">VIX</th>
<th onclick="sortTable(13)">Ret SPY %</th>
</tr>
"""

current_year = None
for _, row in df.sort_values('fecha_senal', ascending=False).iterrows():
    yr = int(row['year'])
    if yr != current_year:
        # Year stats
        yr_data = df[df['year'] == yr]
        yr_ret = yr_data['spy_ret_pct'].sum()
        yr_avg = yr_data['spy_ret_pct'].mean()
        yr_wr = (yr_data['spy_ret_pct'] > 0).sum() / yr_data['spy_ret_pct'].notna().sum() * 100 if yr_data['spy_ret_pct'].notna().sum() > 0 else 0
        yr_dom = yr_data['regime'].value_counts().index[0] if len(yr_data) > 0 else ''
        html += f'<tr class="year-header" data-year="{yr}"><td colspan="14">{yr} &mdash; {len(yr_data)} semanas | Ret acum: <span class="{val_class(yr_ret)}">{yr_ret:+.1f}%</span> | Avg: <span class="{val_class(yr_avg)}">{yr_avg:+.2f}%</span> | WR: {yr_wr:.0f}% | Dominante: {yr_dom}</td></tr>\n'
        current_year = yr

    regime = row['regime']
    color = REGIME_COLORS.get(regime, '#666')
    total_score = row['total'] if pd.notna(row['total']) else 0
    spy_ret = row['spy_ret_pct'] if pd.notna(row['spy_ret_pct']) else None

    html += f'<tr data-regime="{regime}" data-year="{yr}">'
    html += f'<td>{row["fecha_senal"]}</td><td>{yr}</td><td>{int(row["sem"])}</td>'
    html += f'<td><span class="regime-pill" style="background:{color};">{regime}</span></td>'
    html += f'<td class="{val_class(total_score)}"><b>{total_score:+.1f}</b></td>'

    for col in ['s_bdd', 's_brsi', 's_ddp', 's_spy', 's_mom']:
        v = row[col] if pd.notna(row[col]) else 0
        html += f'<td class="{val_class(v)}">{v:+.1f}</td>'

    spy_c = row['spy_close'] if pd.notna(row['spy_close']) else 0
    spy_d = row['spy_dist'] if pd.notna(row['spy_dist']) else 0
    vix = row['vix'] if pd.notna(row['vix']) else 0

    html += f'<td>${spy_c:.0f}</td>'
    html += f'<td class="{val_class(spy_d)}">{spy_d:+.1f}%</td>'
    html += f'<td>{"<b style=color:#c62828>" + f"{vix:.0f}" + "</b>" if vix >= 30 else f"{vix:.0f}"}</td>'

    if spy_ret is not None:
        html += f'<td class="{val_class(spy_ret)}"><b>{spy_ret:+.2f}%</b></td>'
    else:
        html += '<td class="neutral">-</td>'

    html += '</tr>\n'

html += """</table>

<script>
let activeRegimes = new Set(""" + str(REGIME_ORDER).replace("'", '"') + """);

function toggleRegime(btn, regime) {
    if (activeRegimes.has(regime)) {
        activeRegimes.delete(regime);
        btn.classList.remove('active');
    } else {
        activeRegimes.add(regime);
        btn.classList.add('active');
    }
    applyFilters();
}

function applyFilters() {
    const rows = document.querySelectorAll('#mainTable tr[data-regime]');
    const yearHeaders = document.querySelectorAll('#mainTable tr.year-header');
    const search = document.getElementById('searchBox').value.toLowerCase();

    let visibleYears = new Set();
    rows.forEach(row => {
        const regime = row.getAttribute('data-regime');
        const year = row.getAttribute('data-year');
        const text = row.textContent.toLowerCase();
        const show = activeRegimes.has(regime) && (search === '' || text.includes(search));
        row.style.display = show ? '' : 'none';
        if (show) visibleYears.add(year);
    });
    yearHeaders.forEach(h => {
        h.style.display = visibleYears.has(h.getAttribute('data-year')) ? '' : 'none';
    });
}

function filterTable() { applyFilters(); }

function showAll() {
    activeRegimes = new Set(""" + str(REGIME_ORDER).replace("'", '"') + """);
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.add('active'));
    document.getElementById('searchBox').value = '';
    applyFilters();
}

function showYear(y) {
    document.getElementById('searchBox').value = '';
    activeRegimes = new Set(""" + str(REGIME_ORDER).replace("'", '"') + """);
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.add('active'));
    const rows = document.querySelectorAll('#mainTable tr[data-regime]');
    const yearHeaders = document.querySelectorAll('#mainTable tr.year-header');
    rows.forEach(r => r.style.display = r.getAttribute('data-year') == y ? '' : 'none');
    yearHeaders.forEach(h => h.style.display = h.getAttribute('data-year') == y ? '' : 'none');
}

let sortDir = {};
function sortTable(col) {
    const table = document.getElementById('mainTable');
    const rows = Array.from(table.querySelectorAll('tr[data-regime]'));
    const dir = sortDir[col] = !(sortDir[col] || false);
    rows.sort((a, b) => {
        let va = a.cells[col].textContent.replace(/[\\$%+]/g, '').trim();
        let vb = b.cells[col].textContent.replace(/[\\$%+]/g, '').trim();
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return dir ? na - nb : nb - na;
        return dir ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    // Remove year headers
    table.querySelectorAll('.year-header').forEach(h => h.remove());
    // Re-append sorted
    rows.forEach(r => table.appendChild(r));
}
</script>

</body>
</html>"""

with open('regimenes_historico.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"OK -> regimenes_historico.html ({len(df)} semanas)")
