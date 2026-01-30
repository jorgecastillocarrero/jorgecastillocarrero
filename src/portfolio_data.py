"""
Centralized Portfolio Data Service.
Single source of truth for configuration and mappings.
Holdings and cash come from database tables (holding_diario, cash_diario, posicion).
"""

import logging
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
from sqlalchemy import text

from .database import get_db_manager
from .exchange_rate_service import get_exchange_rate_service, ExchangeRateError

logger = logging.getLogger(__name__)


# =============================================================================
# ASSET TYPE MAPPING - Single Source of Truth (Configuration)
# =============================================================================

ASSET_TYPE_MAP = {
    # CO3365 - Mensual
    'AKAM': 'Mensual', 'VRTX': 'Mensual', 'PCAR': 'Mensual', 'BDX': 'Mensual',
    'AMZN': 'Mensual', 'MCO': 'Mensual', 'HCA': 'Mensual', 'MA': 'Mensual',
    'WST': 'Mensual', 'CRM': 'Mensual',

    # La Caixa - Value
    'JD': 'Value', 'BABA': 'Value', 'IAG': 'Value', 'NESN': 'Value',
    'IAG.MC': 'Value', 'NESN.SW': 'Value',

    # La Caixa - Quant
    'ATZ': 'Quant', 'ATZ.TO': 'Quant',

    # La Caixa - Oro/Mineras
    'AEM': 'Oro/Mineras', 'AEM.TO': 'Oro/Mineras',

    # RCO951 - Oro/Mineras
    'B': 'Oro/Mineras', 'TFPM': 'Oro/Mineras', 'SSRM': 'Oro/Mineras',
    'RGLD': 'Oro/Mineras', 'KGC': 'Oro/Mineras', 'CDE': 'Oro/Mineras',
    'BVN': 'Oro/Mineras', 'WPM': 'Oro/Mineras', 'NEM': 'Oro/Mineras',
    'AGI': 'Oro/Mineras', 'SGLE.MI': 'Oro/Mineras', 'SGLE': 'Oro/Mineras',

    # RCO951 - Alpha Picks
    'EAT': 'Alpha Picks', 'EZPW': 'Alpha Picks', 'INCY': 'Alpha Picks',
    'MFC': 'Alpha Picks', 'MU': 'Alpha Picks', 'PARR': 'Alpha Picks',
    'STRL': 'Alpha Picks', 'TIGO': 'Alpha Picks', 'TTMI': 'Alpha Picks',
    'UNFI': 'Alpha Picks', 'VISN': 'Alpha Picks', 'W': 'Alpha Picks',

    # RCO951 - Quant
    'KRYS': 'Quant', 'LRCX': 'Quant', 'STX': 'Quant', 'EXEL': 'Quant',
    'ENVA': 'Quant', 'ESLT': 'Quant', 'WLDN': 'Quant', 'FIX': 'Quant',
    'GOOG': 'Quant', 'CECO': 'Quant', 'AGX': 'Quant', 'PEN': 'Quant',
    'SN': 'Quant', 'LLY': 'Quant', 'NMR': 'Quant', 'APH': 'Quant',
    'PFSI': 'Quant', 'NIC': 'Quant', 'KLAC': 'Quant', 'CLS': 'Quant',
    'PRIM': 'Quant', 'VRT': 'Quant', 'TSM': 'Quant', 'MPWR': 'Quant',
    'HRMY': 'Quant', 'CPRX': 'Quant', 'WING': 'Quant', 'YOU': 'Quant',
    'VIRT': 'Quant', 'PLMR': 'Quant', 'LTH': 'Quant', 'GMED': 'Quant',
    'ONON': 'Quant', 'GIL': 'Quant', 'SBCF': 'Quant', 'PIPR': 'Quant',
    'DLO': 'Quant', 'SEI': 'Quant', 'UI': 'Quant', 'PAHC': 'Quant',
    'HALO': 'Quant', 'EME': 'Quant', 'TGS': 'Quant', 'EVR': 'Quant',
    'SKYW': 'Quant', 'NVDA': 'Quant', 'GLDD': 'Quant', 'STC': 'Quant',
    'USAC': 'Quant', 'PJT': 'Quant', 'NMRK': 'Quant', 'AVGO': 'Quant',
    'SHAK': 'Quant', 'RCL': 'Quant', 'FUTU': 'Quant', 'SFM': 'Quant',
    'HQY': 'Quant', 'UBER': 'Quant', 'HLI': 'Quant', 'BIRK': 'Quant',
    'APP': 'Quant', 'HCI': 'Quant', 'MNDY': 'Quant', 'GWRE': 'Quant',
    'BZ': 'Quant', 'VITL': 'Quant', 'PSIX': 'Quant', 'COIN': 'Quant',
    'DOCS': 'Quant', 'DUOL': 'Quant',

    # IB - Cash/Monetario/Bonos/Futuros
    'TLT': 'Cash/Monetario/Bonos/Futuros',
}


# =============================================================================
# CURRENCY MAPPING - Single Source of Truth (Configuration)
# =============================================================================

CURRENCY_MAP = {
    'US': 'USD',
    'TO': 'CAD',
    'MC': 'EUR',
    'SW': 'CHF',
    'MI': 'EUR',
    'L': 'GBP',
    'DE': 'EUR',
    'F': 'EUR',
}

CURRENCY_SYMBOL_MAP = {
    'USD': '$',
    'CAD': 'C$',
    'EUR': '€',
    'CHF': 'CHF',
    'GBP': '£',
}


# =============================================================================
# FUTURES DATA - Reads from ib_futures_trades table
# =============================================================================
# Los datos de futuros ahora se leen directamente de la base de datos
# y el P&L se calcula automáticamente emparejando operaciones OPEN/CLOSE


# =============================================================================
# PORTFOLIO DATA SERVICE CLASS
# =============================================================================

class PortfolioDataService:
    """
    Centralized service for all portfolio data calculations.
    All data comes from database tables (holding_diario, cash_diario, posicion).
    """

    def __init__(self, db_manager=None):
        self.db = db_manager or get_db_manager()
        self._cache = {}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache = {}

    # =========================================================================
    # HOLDINGS - From holding_diario table
    # =========================================================================

    def get_holdings_for_date(self, account_code: str, fecha: date) -> dict:
        """
        Get holdings for an account on a specific date from holding_diario.

        Args:
            account_code: Account code (CO3365, RCO951, LACAIXA, IB)
            fecha: Date to get holdings for

        Returns:
            Dictionary of {symbol: {'shares': int, 'currency': str}}
        """
        cache_key = f"holdings_{account_code}_{fecha}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from sqlalchemy import text
        with self.db.get_session() as session:
            result = session.execute(text("""
                SELECT symbol, shares, currency
                FROM holding_diario
                WHERE account_code = :account AND fecha = :fecha
            """), {'account': account_code, 'fecha': fecha})

            holdings = {}
            for row in result.fetchall():
                holdings[row[0]] = {
                    'shares': row[1],
                    'currency': row[2] or 'USD'
                }

            self._cache[cache_key] = holdings
            return holdings

    def get_all_holdings_for_date(self, fecha: date) -> dict:
        """
        Get all holdings for all accounts on a specific date.

        Returns:
            Dictionary of {account_code: {symbol: {'shares': int, 'currency': str}}}
        """
        cache_key = f"all_holdings_{fecha}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from sqlalchemy import text
        with self.db.get_session() as session:
            result = session.execute(text("""
                SELECT account_code, symbol, shares, currency
                FROM holding_diario
                WHERE fecha = :fecha
            """), {'fecha': fecha})

            all_holdings = {}
            for row in result.fetchall():
                account = row[0]
                if account not in all_holdings:
                    all_holdings[account] = {}
                all_holdings[account][row[1]] = {
                    'shares': row[2],
                    'currency': row[3] or 'USD'
                }

            self._cache[cache_key] = all_holdings
            return all_holdings

    # =========================================================================
    # CASH - From cash_diario table
    # =========================================================================

    def get_cash_for_date(self, account_code: str, fecha: date) -> dict:
        """
        Get cash positions for an account on a specific date from cash_diario.

        Args:
            account_code: Account code
            fecha: Date to get cash for

        Returns:
            Dictionary of {currency: amount}
        """
        cache_key = f"cash_{account_code}_{fecha}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from sqlalchemy import text
        with self.db.get_session() as session:
            result = session.execute(text("""
                SELECT currency, saldo
                FROM cash_diario
                WHERE account_code = :account AND fecha = :fecha
            """), {'account': account_code, 'fecha': fecha})

            cash = {}
            for row in result.fetchall():
                cash[row[0]] = row[1]

            self._cache[cache_key] = cash
            return cash

    # =========================================================================
    # POSICION - From posicion table
    # =========================================================================

    def get_posicion_for_date(self, fecha: date) -> dict:
        """
        Get position summary for all accounts on a specific date from posicion table.

        Returns:
            Dictionary of {account_code: {'holding_eur': float, 'cash_eur': float, 'total_eur': float}}
        """
        cache_key = f"posicion_{fecha}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from sqlalchemy import text
        with self.db.get_session() as session:
            result = session.execute(text("""
                SELECT account_code, holding_eur, cash_eur, total_eur
                FROM posicion
                WHERE fecha = :fecha
            """), {'fecha': fecha})

            posicion = {}
            for row in result.fetchall():
                posicion[row[0]] = {
                    'holding_eur': row[1] or 0,
                    'cash_eur': row[2] or 0,
                    'total_eur': row[3] or 0
                }

            self._cache[cache_key] = posicion
            return posicion

    def get_total_for_date(self, fecha: date) -> float:
        """Get total portfolio value for a date from posicion table."""
        posicion = self.get_posicion_for_date(fecha)
        return sum(p['total_eur'] for p in posicion.values())

    def get_initial_values(self) -> dict:
        """Get initial values (31/12/2025) from posicion table."""
        return self.get_posicion_for_date(date(2025, 12, 31))

    def get_initial_total(self) -> float:
        """Get total initial value (31/12/2025) from posicion table."""
        return self.get_total_for_date(date(2025, 12, 31))

    # =========================================================================
    # EXCHANGE RATES - From price_history table
    # =========================================================================

    def get_exchange_rate(self, pair: str, target_date: date) -> Optional[float]:
        """
        Get exchange rate from database with previous-day fallback.

        Args:
            pair: Currency pair (e.g., 'EURUSD=X')
            target_date: Date for the rate

        Returns:
            Exchange rate or None if not found
        """
        cache_key = f"fx_{pair}_{target_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from sqlalchemy import text
        with self.db.get_session() as session:
            # Try target date, then fall back to previous days (market closed)
            for i in range(6):
                d = target_date - timedelta(days=i)
                result = session.execute(text("""
                    SELECT p.close FROM symbols s
                    JOIN price_history p ON s.id = p.symbol_id
                    WHERE s.code = :pair AND p.date = :fecha
                """), {'pair': pair, 'fecha': d})

                row = result.fetchone()
                if row:
                    self._cache[cache_key] = row[0]
                    return row[0]

        return None

    def get_eur_usd_rate(self, target_date: date) -> float:
        """Get EUR/USD rate for a specific date."""
        rate_service = get_exchange_rate_service(self.db)
        return rate_service.get_eur_usd(target_date)

    def get_cad_eur_rate(self, target_date: date) -> float:
        """Get CAD/EUR rate: how many EUR per 1 CAD."""
        rate_service = get_exchange_rate_service(self.db)
        return rate_service.get_cad_eur(target_date)

    def get_chf_eur_rate(self, target_date: date) -> float:
        """Get CHF/EUR rate: how many EUR per 1 CHF."""
        rate_service = get_exchange_rate_service(self.db)
        return rate_service.get_chf_eur(target_date)

    # =========================================================================
    # PRICES - From price_history table
    # =========================================================================

    def get_symbol_price(self, symbol: str, target_date: date) -> Optional[float]:
        """
        Get closing price for a symbol on a specific date with previous-day fallback.

        Args:
            symbol: Stock symbol
            target_date: Date for the price

        Returns:
            Closing price or None if not found
        """
        cache_key = f"price_{symbol}_{target_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from sqlalchemy import text
        with self.db.get_session() as session:
            # Try target date, then fall back to previous days
            for i in range(6):
                d = target_date - timedelta(days=i)
                result = session.execute(text("""
                    SELECT p.close FROM symbols s
                    JOIN price_history p ON s.id = p.symbol_id
                    WHERE s.code = :symbol AND p.date = :fecha
                """), {'symbol': symbol, 'fecha': d})

                row = result.fetchone()
                if row:
                    self._cache[cache_key] = row[0]
                    return row[0]

        return None

    def get_latest_trading_dates(self, count: int = 2) -> list[date]:
        """Get the latest trading dates from SPY data."""
        from sqlalchemy import text
        with self.db.get_session() as session:
            result = session.execute(text("""
                SELECT p.date FROM symbols s
                JOIN price_history p ON s.id = p.symbol_id
                WHERE s.code = 'SPY'
                ORDER BY p.date DESC
                LIMIT :limit
            """), {'limit': count + 5})

            today = date.today()
            dates = [row[0] for row in result.fetchall() if row[0] < today]
            return dates[:count]

    # =========================================================================
    # CALCULATIONS
    # =========================================================================

    def calculate_position_value(
        self,
        symbol: str,
        shares: int,
        target_date: date,
        target_currency: str = 'EUR'
    ) -> Optional[float]:
        """
        Calculate position value in target currency.

        Args:
            symbol: Stock symbol
            shares: Number of shares
            target_date: Date for valuation
            target_currency: Currency for result (default EUR)

        Returns:
            Position value in target currency
        """
        # Determine source currency from symbol
        if '.TO' in symbol:
            source_currency = 'CAD'
        elif '.MC' in symbol or '.MI' in symbol:
            source_currency = 'EUR'
        elif '.SW' in symbol:
            source_currency = 'CHF'
        else:
            source_currency = 'USD'

        price = self.get_symbol_price(symbol, target_date)
        if price is None:
            return None

        value_local = price * shares

        # Convert to target currency
        if source_currency == target_currency:
            return value_local

        if source_currency == 'USD' and target_currency == 'EUR':
            eur_usd = self.get_eur_usd_rate(target_date)
            return value_local / eur_usd
        elif source_currency == 'CAD' and target_currency == 'EUR':
            cad_eur = self.get_cad_eur_rate(target_date)
            return value_local * cad_eur
        elif source_currency == 'CHF' and target_currency == 'EUR':
            chf_eur = self.get_chf_eur_rate(target_date)
            return value_local * chf_eur

        return value_local

    def get_values_by_asset_type(self, fecha: date) -> dict:
        """
        Calculate portfolio values grouped by asset type for a specific date.
        All data from holding_diario (stocks, TLT) and cash_diario (cash).

        Returns:
            Dictionary with values per asset type
        """
        values = {}

        # 1. Get ALL holdings from holding_diario (includes TLT)
        all_holdings = self.get_all_holdings_for_date(fecha)

        for account, holdings in all_holdings.items():
            for symbol, data in holdings.items():
                shares = data['shares']

                # Get asset type from ASSET_TYPE_MAP
                base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                asset_type = ASSET_TYPE_MAP.get(symbol, ASSET_TYPE_MAP.get(base_symbol, 'Otros'))

                value = self.calculate_position_value(symbol, shares, fecha)
                if value:
                    if asset_type not in values:
                        values[asset_type] = 0
                    values[asset_type] += value

        # 2. Add cash from cash_diario for all accounts
        if 'Cash/Monetario/Bonos/Futuros' not in values:
            values['Cash/Monetario/Bonos/Futuros'] = 0

        eur_usd = self.get_eur_usd_rate(fecha)
        for account in ['CO3365', 'RCO951', 'IB']:
            cash_data = self.get_cash_for_date(account, fecha)
            if cash_data:
                for currency, amount in cash_data.items():
                    if currency == 'EUR':
                        values['Cash/Monetario/Bonos/Futuros'] += amount
                    elif currency == 'USD':
                        values['Cash/Monetario/Bonos/Futuros'] += amount / eur_usd

        return values

    def get_daily_comparison(self, date1: date, date2: date) -> pd.DataFrame:
        """
        Compare portfolio values between two dates by asset type.

        Returns:
            DataFrame with comparison data
        """
        values1 = self.get_values_by_asset_type(date1)
        values2 = self.get_values_by_asset_type(date2)

        all_types = set(values1.keys()) | set(values2.keys())

        data = []
        for asset_type in sorted(all_types):
            v1 = values1.get(asset_type, 0)
            v2 = values2.get(asset_type, 0)
            diff = v2 - v1
            pct = (diff / v1 * 100) if v1 > 0 else 0

            data.append({
                'Tipo': asset_type,
                'Valor Anterior': v1,
                'Valor Actual': v2,
                'Diferencia': diff,
                'Variacion %': pct,
            })

        return pd.DataFrame(data)

    def get_futures_trades_from_db(self) -> list:
        """Get raw futures trades from ib_futures_trades table."""
        with self.db.get_session() as session:
            result = session.execute(text("""
                SELECT id, symbol, expiry, trade_date, quantity, price,
                       multiplier, commission, trade_type
                FROM ib_futures_trades
                ORDER BY trade_date
            """))
            return [dict(row._mapping) for row in result.fetchall()]

    def calculate_futures_pnl(self) -> dict:
        """Calculate P&L by matching OPEN/CLOSE trades (FIFO method)."""
        trades = self.get_futures_trades_from_db()

        # Group by symbol
        by_symbol = {}
        for t in trades:
            symbol = t['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = {'opens': [], 'closes': []}
            if t['trade_type'] == 'OPEN':
                by_symbol[symbol]['opens'].append(t)
            else:
                by_symbol[symbol]['closes'].append(t)

        # Match trades and calculate P&L
        matched_trades = []
        open_positions = []
        summary_by_contract = {}

        for symbol, data in by_symbol.items():
            opens = data['opens'].copy()
            closes = data['closes'].copy()
            multiplier = opens[0]['multiplier'] if opens else 100

            symbol_pnl = 0
            symbol_ops = 0

            # Match FIFO
            for close in closes:
                if opens:
                    open_trade = opens.pop(0)
                    # Determine direction: quantity > 0 = Long, quantity < 0 = Short
                    is_long = open_trade.get('quantity', 1) > 0
                    num_contracts = abs(open_trade.get('quantity', 1))
                    # P&L = (sell - buy) * multiplier * contracts - commissions for Long
                    # P&L = (buy - sell) * multiplier * contracts - commissions for Short
                    if is_long:
                        pnl = (close['price'] - open_trade['price']) * multiplier * num_contracts
                    else:
                        pnl = (open_trade['price'] - close['price']) * multiplier * num_contracts
                    pnl -= (open_trade['commission'] + close['commission'])

                    matched_trades.append({
                        'contract': symbol,
                        'tipo': 'Largo' if is_long else 'Corto',
                        'contratos': num_contracts,
                        'buy_date': open_trade['trade_date'],
                        'buy_price': open_trade['price'],
                        'sell_date': close['trade_date'],
                        'sell_price': close['price'],
                        'pnl_usd': pnl,
                        'status': 'Cerrada'
                    })
                    symbol_pnl += pnl
                    symbol_ops += 2  # 1 open + 1 close

            # Remaining opens are open positions
            for open_trade in opens:
                is_long = open_trade.get('quantity', 1) > 0
                num_contracts = abs(open_trade.get('quantity', 1))
                open_positions.append({
                    'symbol': symbol,
                    'tipo': 'Largo' if is_long else 'Corto',
                    'contratos': num_contracts,
                    'expiry': open_trade['expiry'],
                    'trade_date': open_trade['trade_date'],
                    'price': open_trade['price'],
                    'multiplier': multiplier
                })
                symbol_ops += 1

            # Summary for this contract
            has_open = len(opens) > 0
            summary_by_contract[symbol] = {
                'realized_usd': symbol_pnl,
                'operations': symbol_ops,
                'status': 'ABIERTO' if has_open else 'CERRADO'
            }

        return {
            'matched_trades': matched_trades,
            'open_positions': open_positions,
            'by_contract': summary_by_contract
        }

    def get_futures_trades_df(self) -> pd.DataFrame:
        """Get futures trades as DataFrame (from database)."""
        pnl_data = self.calculate_futures_pnl()
        matched_trades = pnl_data['matched_trades']

        data = []
        for trade in matched_trades:
            # Format dates
            buy_dt = trade['buy_date']
            sell_dt = trade['sell_date']
            buy_str = buy_dt.strftime('%d/%m %H:%M') if hasattr(buy_dt, 'strftime') else str(buy_dt)[5:16].replace('-', '/').replace(' ', ' ')
            sell_str = sell_dt.strftime('%d/%m %H:%M') if hasattr(sell_dt, 'strftime') else str(sell_dt)[5:16].replace('-', '/').replace(' ', ' ')

            pnl = trade['pnl_usd']
            data.append({
                'Contrato': trade['contract'],
                'Tipo': trade['tipo'],
                'Contratos': trade['contratos'],
                'Entrada': buy_str,
                'Precio Entrada': f"${trade['buy_price']:,.2f}",
                'Salida': sell_str,
                'Precio Salida': f"${trade['sell_price']:,.2f}",
                'P&G': f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}",
                'Estado': trade['status'],
            })

        # Add open positions
        for pos in pnl_data['open_positions']:
            pos_dt = pos['trade_date']
            pos_str = pos_dt.strftime('%d/%m %H:%M') if hasattr(pos_dt, 'strftime') else str(pos_dt)[5:16].replace('-', '/').replace(' ', ' ')
            data.append({
                'Contrato': pos['symbol'],
                'Tipo': pos['tipo'],
                'Contratos': pos['contratos'],
                'Entrada': pos_str,
                'Precio Entrada': f"${pos['price']:,.2f}",
                'Salida': '-',
                'Precio Salida': '-',
                'P&G': '-',
                'Estado': 'ABIERTA',
            })

        return pd.DataFrame(data)

    def get_futures_summary(self) -> dict:
        """Get futures P&L summary (from database)."""
        pnl_data = self.calculate_futures_pnl()
        eur_usd = self.get_eur_usd_rate(date.today())

        # Calculate totals
        total_realized_usd = sum(c['realized_usd'] for c in pnl_data['by_contract'].values())
        total_realized_eur = total_realized_usd / eur_usd

        # Open position info
        open_positions = pnl_data['open_positions']
        if open_positions:
            pos = open_positions[0]  # First open position
            open_position = {
                'symbol': pos['symbol'],
                'description': f"Gold Futures {pos['expiry']}",
                'contracts': len(open_positions),
                'direction': 'LARGO',
                'entry_price': pos['price'],
                'current_price': 0,  # Would need live price
                'unrealized_pnl_usd': 0,
                'expiration': pos['expiry'],
            }
        else:
            open_position = {
                'symbol': '-',
                'description': 'Sin posición abierta',
                'contracts': 0,
                'direction': '-',
                'entry_price': 0,
                'current_price': 0,
                'unrealized_pnl_usd': 0,
                'expiration': '-',
            }

        return {
            'total_realized_usd': total_realized_usd,
            'total_realized_eur': total_realized_eur,
            'open_position': open_position,
            'by_contract': pnl_data['by_contract'],
        }


# Singleton instance
_service_instance = None

def get_portfolio_service(db_manager=None) -> PortfolioDataService:
    """Get or create the portfolio data service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PortfolioDataService(db_manager)
    return _service_instance
