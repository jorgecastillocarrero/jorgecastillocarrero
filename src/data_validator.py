"""
Data Validator Module for Financial AI Assistant
Provides data quality checks, outlier detection, and consistency validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QualityReport:
    """Container for data quality report"""
    symbol: str
    total_records: int
    date_range: Tuple[str, str]
    missing_dates: int
    missing_pct: float
    outliers_count: int
    issues: List[Dict]
    score: float  # 0-100

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'total_records': self.total_records,
            'date_range': f"{self.date_range[0]} to {self.date_range[1]}",
            'missing_dates': self.missing_dates,
            'missing_pct': f"{self.missing_pct:.1f}%",
            'outliers_count': self.outliers_count,
            'issues_count': len(self.issues),
            'quality_score': f"{self.score:.0f}/100"
        }

    def summary(self) -> str:
        """Text summary for AI assistant"""
        status = "BUENA" if self.score >= 80 else "ACEPTABLE" if self.score >= 60 else "PROBLEMATICA"

        lines = [
            f"=== CALIDAD DE DATOS: {self.symbol} ===",
            f"Estado: {status} ({self.score:.0f}/100)",
            "",
            "ESTADISTICAS:",
            f"  Total registros: {self.total_records:,}",
            f"  Rango de fechas: {self.date_range[0]} a {self.date_range[1]}",
            f"  Dias faltantes: {self.missing_dates} ({self.missing_pct:.1f}%)",
            f"  Outliers detectados: {self.outliers_count}",
        ]

        if self.issues:
            lines.append("\nPROBLEMAS ENCONTRADOS:")
            for issue in self.issues[:10]:  # Limit to 10
                lines.append(f"  - {issue['type']}: {issue['description']}")
        else:
            lines.append("\nNo se encontraron problemas significativos.")

        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Container for validation result"""
    is_valid: bool
    entity: str
    checks_passed: int
    checks_failed: int
    issues: List[Dict]

    def summary(self) -> str:
        """Text summary"""
        status = "VALIDO" if self.is_valid else "INVALIDO"
        lines = [
            f"=== VALIDACION: {self.entity} ===",
            f"Estado: {status}",
            f"Checks pasados: {self.checks_passed}",
            f"Checks fallidos: {self.checks_failed}",
        ]

        if self.issues:
            lines.append("\nPROBLEMAS:")
            for issue in self.issues:
                lines.append(f"  - {issue['description']}")

        return "\n".join(lines)


class DataValidator:
    """
    Data quality validation and outlier detection.
    Uses data from local database.
    """

    def __init__(self, db_path: str = "data/financial_data.db"):
        self.db_path = db_path
        self._db = None

    @property
    def db(self):
        """Lazy load database analyzer"""
        if self._db is None:
            from src.db_analysis_tools import DatabaseAnalyzer
            self._db = DatabaseAnalyzer(self.db_path)
        return self._db

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    # =========================================================================
    # OUTLIER DETECTION
    # =========================================================================

    def detect_outliers(self, symbol: str, method: str = 'zscore',
                        threshold: float = 3.0) -> List[Dict]:
        """
        Detect price outliers.

        Args:
            symbol: Stock symbol
            method: 'zscore', 'iqr', or 'pct_change'
            threshold: Detection threshold

        Returns:
            List of outlier records
        """
        df = self.db.get_price_history(symbol, days=500)
        if df.empty:
            return []

        outliers = []

        if method == 'zscore':
            # Z-score method
            for col in ['open', 'high', 'low', 'close']:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    zscore = (df[col] - mean) / std
                    mask = abs(zscore) > threshold

                    for idx in df[mask].index:
                        row = df.loc[idx]
                        outliers.append({
                            'date': row['date'],
                            'type': f'{col}_outlier',
                            'value': row[col],
                            'zscore': zscore.loc[idx],
                            'description': f"{col} = {row[col]:.2f} (z-score: {zscore.loc[idx]:.2f})"
                        })

        elif method == 'iqr':
            # IQR method
            for col in ['open', 'high', 'low', 'close']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                mask = (df[col] < lower) | (df[col] > upper)

                for idx in df[mask].index:
                    row = df.loc[idx]
                    outliers.append({
                        'date': row['date'],
                        'type': f'{col}_iqr_outlier',
                        'value': row[col],
                        'description': f"{col} = {row[col]:.2f} (fuera de rango IQR)"
                    })

        elif method == 'pct_change':
            # Large percentage change
            df['pct_change'] = df['close'].pct_change().abs() * 100
            mask = df['pct_change'] > threshold

            for idx in df[mask].index:
                row = df.loc[idx]
                outliers.append({
                    'date': row['date'],
                    'type': 'large_move',
                    'value': row['pct_change'],
                    'description': f"Movimiento de {row['pct_change']:.1f}% en un dia"
                })

        return outliers

    def outliers_summary(self, symbol: str) -> str:
        """Generate text summary of outliers"""
        # Use pct_change method - more relevant for stocks
        outliers = self.detect_outliers(symbol, method='pct_change', threshold=10)

        if not outliers:
            return f"No se detectaron movimientos extremos (>10%) en {symbol}"

        lines = [
            f"=== OUTLIERS DETECTADOS: {symbol} ===",
            f"Metodo: Movimientos >10% en un dia\n",
            f"{'Fecha':<12} {'Tipo':<15} {'Cambio':>10}",
            "-" * 40
        ]

        for o in outliers[:20]:  # Limit to 20
            date_str = o['date'].strftime('%Y-%m-%d') if hasattr(o['date'], 'strftime') else str(o['date'])[:10]
            lines.append(f"{date_str:<12} {o['type']:<15} {o['value']:>9.1f}%")

        lines.append(f"\nTotal outliers: {len(outliers)}")

        return "\n".join(lines)

    # =========================================================================
    # MISSING DATA DETECTION
    # =========================================================================

    def find_missing_dates(self, symbol: str, days: int = 252) -> List[datetime]:
        """
        Find missing trading dates for a symbol.

        Args:
            symbol: Stock symbol
            days: Days to check

        Returns:
            List of missing dates
        """
        df = self.db.get_price_history(symbol, days=days)
        if df.empty:
            return []

        df['date'] = pd.to_datetime(df['date'])
        existing_dates = set(df['date'].dt.date)

        # Generate expected trading days (exclude weekends)
        start_date = df['date'].min()
        end_date = df['date'].max()
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

        expected_dates = set(d.date() for d in all_dates)
        missing = expected_dates - existing_dates

        return sorted(list(missing))

    def missing_dates_summary(self, symbol: str, days: int = 252) -> str:
        """Generate text summary of missing dates"""
        missing = self.find_missing_dates(symbol, days)

        lines = [
            f"=== FECHAS FALTANTES: {symbol} ===",
            f"Periodo analizado: ultimos {days} dias",
            f"Fechas faltantes: {len(missing)}",
        ]

        if missing:
            # Find gaps (consecutive missing dates)
            gaps = []
            start = missing[0]
            end = missing[0]

            for i in range(1, len(missing)):
                if (missing[i] - missing[i-1]).days <= 3:  # Allow weekends
                    end = missing[i]
                else:
                    gaps.append((start, end))
                    start = missing[i]
                    end = missing[i]
            gaps.append((start, end))

            if len(gaps) <= 10:
                lines.append("\nGaps identificados:")
                for start, end in gaps:
                    if start == end:
                        lines.append(f"  {start}")
                    else:
                        lines.append(f"  {start} a {end}")
            else:
                lines.append(f"\nDemasiados gaps para listar ({len(gaps)} gaps)")

        return "\n".join(lines)

    # =========================================================================
    # DATA CONSISTENCY
    # =========================================================================

    def check_price_consistency(self, symbol: str) -> List[Dict]:
        """
        Check for price consistency issues.

        Checks:
        - High < Low (impossible)
        - Open/Close outside High/Low range
        - Zero or negative prices
        - Unchanged prices for multiple days
        """
        df = self.db.get_price_history(symbol, days=500)
        if df.empty:
            return []

        issues = []

        # High < Low
        mask = df['high'] < df['low']
        for idx in df[mask].index:
            row = df.loc[idx]
            issues.append({
                'date': row['date'],
                'type': 'high_low_inverted',
                'description': f"High ({row['high']:.2f}) < Low ({row['low']:.2f})"
            })

        # Open outside range
        mask = (df['open'] > df['high']) | (df['open'] < df['low'])
        for idx in df[mask].index:
            row = df.loc[idx]
            issues.append({
                'date': row['date'],
                'type': 'open_out_of_range',
                'description': f"Open ({row['open']:.2f}) fuera del rango High-Low"
            })

        # Close outside range
        mask = (df['close'] > df['high']) | (df['close'] < df['low'])
        for idx in df[mask].index:
            row = df.loc[idx]
            issues.append({
                'date': row['date'],
                'type': 'close_out_of_range',
                'description': f"Close ({row['close']:.2f}) fuera del rango High-Low"
            })

        # Zero or negative prices
        for col in ['open', 'high', 'low', 'close']:
            mask = df[col] <= 0
            for idx in df[mask].index:
                row = df.loc[idx]
                issues.append({
                    'date': row['date'],
                    'type': 'invalid_price',
                    'description': f"{col} = {row[col]:.2f} (cero o negativo)"
                })

        # Unchanged close for 5+ days (suspicious for liquid stocks)
        unchanged_count = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] == df['close'].iloc[i-1]:
                unchanged_count += 1
                if unchanged_count >= 5:
                    issues.append({
                        'date': df['date'].iloc[i],
                        'type': 'stale_price',
                        'description': f"Precio sin cambios por {unchanged_count} dias consecutivos"
                    })
            else:
                unchanged_count = 0

        return issues

    def consistency_summary(self, symbol: str) -> str:
        """Generate text summary of consistency issues"""
        issues = self.check_price_consistency(symbol)

        if not issues:
            return f"No se encontraron problemas de consistencia en {symbol}"

        lines = [
            f"=== PROBLEMAS DE CONSISTENCIA: {symbol} ===\n",
            f"Total problemas: {len(issues)}",
            "",
            f"{'Fecha':<12} {'Tipo':<25} {'Descripcion'}",
            "-" * 70
        ]

        for issue in issues[:20]:
            date_str = issue['date'].strftime('%Y-%m-%d') if hasattr(issue['date'], 'strftime') else str(issue['date'])[:10]
            lines.append(f"{date_str:<12} {issue['type']:<25} {issue['description']}")

        if len(issues) > 20:
            lines.append(f"\n... y {len(issues) - 20} problemas mas")

        return "\n".join(lines)

    # =========================================================================
    # FULL QUALITY CHECK
    # =========================================================================

    def check_data_quality(self, symbol: str, days: int = 252) -> QualityReport:
        """
        Run full data quality check.

        Args:
            symbol: Stock symbol
            days: Days to analyze

        Returns:
            QualityReport object
        """
        df = self.db.get_price_history(symbol, days=days)

        if df.empty:
            return QualityReport(
                symbol=symbol,
                total_records=0,
                date_range=("N/A", "N/A"),
                missing_dates=0,
                missing_pct=100,
                outliers_count=0,
                issues=[{"type": "no_data", "description": "No hay datos para este simbolo"}],
                score=0
            )

        # Basic stats
        total_records = len(df)
        date_range = (df['date'].min().strftime('%Y-%m-%d'),
                     df['date'].max().strftime('%Y-%m-%d'))

        # Missing dates
        missing_dates = self.find_missing_dates(symbol, days)
        expected_days = len(pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='B'))
        missing_pct = len(missing_dates) / expected_days * 100 if expected_days > 0 else 0

        # Outliers
        outliers = self.detect_outliers(symbol, method='pct_change', threshold=15)

        # Consistency issues
        consistency_issues = self.check_price_consistency(symbol)

        # Combine all issues
        all_issues = []
        for o in outliers[:5]:
            all_issues.append({'type': 'outlier', 'description': o['description']})
        for c in consistency_issues[:5]:
            all_issues.append({'type': c['type'], 'description': c['description']})

        # Calculate quality score
        score = 100

        # Penalize missing data
        if missing_pct > 10:
            score -= 20
        elif missing_pct > 5:
            score -= 10
        elif missing_pct > 1:
            score -= 5

        # Penalize outliers
        outlier_pct = len(outliers) / total_records * 100 if total_records > 0 else 0
        if outlier_pct > 5:
            score -= 15
        elif outlier_pct > 2:
            score -= 10
        elif outlier_pct > 0.5:
            score -= 5

        # Penalize consistency issues
        if len(consistency_issues) > 10:
            score -= 20
        elif len(consistency_issues) > 5:
            score -= 10
        elif len(consistency_issues) > 0:
            score -= 5

        # Penalize short history
        if total_records < 50:
            score -= 20
        elif total_records < 100:
            score -= 10

        score = max(0, score)

        return QualityReport(
            symbol=symbol,
            total_records=total_records,
            date_range=date_range,
            missing_dates=len(missing_dates),
            missing_pct=missing_pct,
            outliers_count=len(outliers),
            issues=all_issues,
            score=score
        )

    # =========================================================================
    # PORTFOLIO VALIDATION
    # =========================================================================

    def validate_holdings(self, fecha: str = None) -> ValidationResult:
        """
        Validate portfolio holdings data.

        Checks:
        - All symbols exist in database
        - No duplicate entries
        - Prices are reasonable
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if not fecha:
            cursor.execute("SELECT MAX(fecha) FROM holding_diario")
            fecha = cursor.fetchone()[0]

        if not fecha:
            return ValidationResult(
                is_valid=False,
                entity=f"Holdings",
                checks_passed=0,
                checks_failed=1,
                issues=[{"description": "No hay datos de holdings"}]
            )

        cursor.execute("""
            SELECT account_code, symbol, shares, precio_entrada, currency
            FROM holding_diario WHERE fecha = ?
        """, (fecha,))
        holdings = cursor.fetchall()
        conn.close()

        checks_passed = 0
        checks_failed = 0
        issues = []

        # Check: Holdings exist
        if holdings:
            checks_passed += 1
        else:
            checks_failed += 1
            issues.append({"description": f"No hay holdings para {fecha}"})
            return ValidationResult(
                is_valid=False,
                entity=f"Holdings ({fecha})",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                issues=issues
            )

        # Check: No duplicates
        seen = set()
        for h in holdings:
            key = (h[0], h[1])  # account, symbol
            if key in seen:
                checks_failed += 1
                issues.append({"description": f"Duplicado: {h[0]}/{h[1]}"})
            else:
                seen.add(key)
        if not any("Duplicado" in i['description'] for i in issues):
            checks_passed += 1

        # Check: Valid shares
        for h in holdings:
            if h[2] == 0:  # shares
                checks_failed += 1
                issues.append({"description": f"Shares = 0 para {h[0]}/{h[1]}"})
        if not any("Shares = 0" in i['description'] for i in issues):
            checks_passed += 1

        # Check: Valid prices
        for h in holdings:
            if h[3] is None or h[3] <= 0:  # precio_entrada
                checks_failed += 1
                issues.append({"description": f"Precio invalido para {h[0]}/{h[1]}: {h[3]}"})
        if not any("Precio invalido" in i['description'] for i in issues):
            checks_passed += 1

        return ValidationResult(
            is_valid=checks_failed == 0,
            entity=f"Holdings ({fecha})",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            issues=issues
        )

    # =========================================================================
    # DATABASE STATISTICS
    # =========================================================================

    def get_database_stats(self) -> str:
        """Get overall database statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        lines = ["=== ESTADISTICAS DE BASE DE DATOS ===\n"]

        # Symbols
        cursor.execute("SELECT COUNT(*) FROM symbols")
        symbols_count = cursor.fetchone()[0]
        lines.append(f"Simbolos: {symbols_count:,}")

        # Fundamentals
        cursor.execute("SELECT COUNT(*) FROM fundamentals WHERE market_cap > 0")
        fund_count = cursor.fetchone()[0]
        lines.append(f"Con fundamentales: {fund_count:,}")

        # Price history
        cursor.execute("SELECT COUNT(*) FROM price_history")
        prices_count = cursor.fetchone()[0]
        lines.append(f"Registros de precios: {prices_count:,}")

        cursor.execute("SELECT COUNT(DISTINCT symbol_id) FROM price_history")
        symbols_with_prices = cursor.fetchone()[0]
        lines.append(f"Simbolos con precios: {symbols_with_prices:,}")

        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM price_history")
        date_range = cursor.fetchone()
        if date_range[0]:
            lines.append(f"Rango de fechas: {date_range[0]} a {date_range[1]}")

        # Holdings
        cursor.execute("SELECT COUNT(DISTINCT fecha) FROM holding_diario")
        holding_dates = cursor.fetchone()[0]
        lines.append(f"\nDias de holdings: {holding_dates:,}")

        cursor.execute("SELECT MAX(fecha) FROM holding_diario")
        last_holding = cursor.fetchone()[0]
        lines.append(f"Ultimo holding: {last_holding}")

        # Posicion
        cursor.execute("SELECT MAX(fecha) FROM posicion")
        last_posicion = cursor.fetchone()[0]
        lines.append(f"Ultima posicion: {last_posicion}")

        conn.close()

        return "\n".join(lines)


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Data Validator Test ===\n")

    validator = DataValidator()

    print("--- Database Stats ---")
    print(validator.get_database_stats())

    print("\n--- Data Quality Check (AAPL) ---")
    try:
        report = validator.check_data_quality("AAPL")
        print(report.summary())
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Outliers (AAPL) ---")
    try:
        print(validator.outliers_summary("AAPL"))
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Validate Holdings ---")
    try:
        result = validator.validate_holdings()
        print(result.summary())
    except Exception as e:
        print(f"Error: {e}")
