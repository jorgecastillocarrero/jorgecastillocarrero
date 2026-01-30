"""
Daily Portfolio Tracking System.
Manages daily holdings, stock trades, and cash movements.
"""

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func

from .database import (
    get_db_manager, Symbol, PriceHistory,
    DailyHolding, StockTrade, DailyCash, CashMovement
)
from .exchange_rate_service import get_exchange_rate_service, ExchangeRateError

logger = logging.getLogger(__name__)


class DailyTrackingService:
    """Service for managing daily portfolio tracking."""

    def __init__(self, db_manager=None):
        self.db = db_manager or get_db_manager()

    # =========================================================================
    # HOLDINGS MANAGEMENT
    # =========================================================================

    def set_holdings_for_date(self, holding_date: date, holdings_data: list[dict]) -> int:
        """
        Set holdings for a specific date.

        Args:
            holding_date: Date for the holdings
            holdings_data: List of dicts with {account_code, symbol, shares, currency, entry_price}

        Returns:
            Number of holdings created
        """
        with self.db.get_session() as session:
            # Delete existing holdings for this date
            session.query(DailyHolding).filter(
                DailyHolding.holding_date == holding_date
            ).delete()

            count = 0
            for h in holdings_data:
                holding = DailyHolding(
                    holding_date=holding_date,
                    account_code=h['account_code'],
                    symbol=h['symbol'],
                    shares=h['shares'],
                    currency=h.get('currency', 'USD'),
                    entry_price=h.get('entry_price'),
                )
                session.add(holding)
                count += 1

            session.commit()
            return count

    def get_holdings_for_date(self, holding_date: date, account_code: str = None) -> list[dict]:
        """
        Get holdings for a specific date.

        If no holdings exist for the exact date, finds the most recent previous date.
        """
        with self.db.get_session() as session:
            # First try exact date
            query = session.query(DailyHolding).filter(
                DailyHolding.holding_date == holding_date
            )
            if account_code:
                query = query.filter(DailyHolding.account_code == account_code)

            holdings = query.all()

            # If no holdings, look for previous date
            if not holdings:
                prev_date = session.query(func.max(DailyHolding.holding_date)).filter(
                    DailyHolding.holding_date < holding_date
                ).scalar()

                if prev_date:
                    query = session.query(DailyHolding).filter(
                        DailyHolding.holding_date == prev_date
                    )
                    if account_code:
                        query = query.filter(DailyHolding.account_code == account_code)
                    holdings = query.all()

            return [
                {
                    'date': h.holding_date,
                    'account_code': h.account_code,
                    'symbol': h.symbol,
                    'shares': h.shares,
                    'currency': h.currency,
                    'entry_price': h.entry_price,
                }
                for h in holdings
            ]

    def propagate_holdings(self, from_date: date, to_date: date) -> int:
        """
        Copy holdings from one date to another (for days without trades).

        Returns:
            Number of holdings copied
        """
        with self.db.get_session() as session:
            # Get holdings from source date
            source_holdings = session.query(DailyHolding).filter(
                DailyHolding.holding_date == from_date
            ).all()

            if not source_holdings:
                logger.warning(f"No holdings found for {from_date}")
                return 0

            # Delete existing holdings for target date
            session.query(DailyHolding).filter(
                DailyHolding.holding_date == to_date
            ).delete()

            # Copy holdings
            count = 0
            for h in source_holdings:
                new_holding = DailyHolding(
                    holding_date=to_date,
                    account_code=h.account_code,
                    symbol=h.symbol,
                    shares=h.shares,
                    currency=h.currency,
                    entry_price=h.entry_price,
                )
                session.add(new_holding)
                count += 1

            session.commit()
            return count

    # =========================================================================
    # STOCK TRADES
    # =========================================================================

    def add_stock_trade(
        self,
        account_code: str,
        trade_date: date,
        symbol: str,
        trade_type: str,  # BUY or SELL
        shares: float,
        price: float,
        currency: str = 'USD',
        commission: float = 0,
        notes: str = None,
    ) -> int:
        """
        Record a stock trade and update holdings.

        Returns:
            Trade ID
        """
        with self.db.get_session() as session:
            amount_local = shares * price

            # Create trade record
            trade = StockTrade(
                account_code=account_code,
                trade_date=trade_date,
                symbol=symbol,
                trade_type=trade_type.upper(),
                shares=shares,
                price=price,
                currency=currency,
                commission=commission,
                amount_local=amount_local,
                notes=notes,
            )
            session.add(trade)
            session.flush()

            # Update daily holdings for this date and forward
            self._update_holdings_after_trade(
                session, account_code, trade_date, symbol,
                shares if trade_type.upper() == 'BUY' else -shares,
                price, currency
            )

            session.commit()
            return trade.id

    def _update_holdings_after_trade(
        self, session, account_code: str, trade_date: date,
        symbol: str, shares_delta: float, price: float, currency: str
    ):
        """Update holdings for trade date and all future dates."""
        # Get current holding for this symbol on this date (or previous)
        current = session.query(DailyHolding).filter(
            DailyHolding.holding_date == trade_date,
            DailyHolding.account_code == account_code,
            DailyHolding.symbol == symbol
        ).first()

        if current:
            # Update existing holding
            new_shares = current.shares + shares_delta
            if new_shares <= 0:
                session.delete(current)
            else:
                current.shares = new_shares
                # Update average price for buys
                if shares_delta > 0:
                    old_cost = (current.entry_price or price) * (current.shares - shares_delta)
                    new_cost = price * shares_delta
                    current.entry_price = (old_cost + new_cost) / new_shares
        elif shares_delta > 0:
            # New position (buy)
            new_holding = DailyHolding(
                holding_date=trade_date,
                account_code=account_code,
                symbol=symbol,
                shares=shares_delta,
                currency=currency,
                entry_price=price,
            )
            session.add(new_holding)

        # Update all future dates
        future_holdings = session.query(DailyHolding).filter(
            DailyHolding.holding_date > trade_date,
            DailyHolding.account_code == account_code,
            DailyHolding.symbol == symbol
        ).all()

        for fh in future_holdings:
            fh.shares += shares_delta
            if fh.shares <= 0:
                session.delete(fh)

    def get_trades(
        self,
        start_date: date = None,
        end_date: date = None,
        account_code: str = None,
        symbol: str = None
    ) -> list[dict]:
        """Get stock trades with optional filters."""
        with self.db.get_session() as session:
            query = session.query(StockTrade)

            if start_date:
                query = query.filter(StockTrade.trade_date >= start_date)
            if end_date:
                query = query.filter(StockTrade.trade_date <= end_date)
            if account_code:
                query = query.filter(StockTrade.account_code == account_code)
            if symbol:
                query = query.filter(StockTrade.symbol == symbol)

            trades = query.order_by(StockTrade.trade_date).all()

            return [
                {
                    'id': t.id,
                    'date': t.trade_date,
                    'account': t.account_code,
                    'symbol': t.symbol,
                    'type': t.trade_type,
                    'shares': t.shares,
                    'price': t.price,
                    'currency': t.currency,
                    'amount': t.amount_local,
                    'commission': t.commission,
                    'notes': t.notes,
                }
                for t in trades
            ]

    # =========================================================================
    # CASH MANAGEMENT
    # =========================================================================

    def set_cash_for_date(self, cash_date: date, cash_data: list[dict]) -> int:
        """
        Set cash balances for a specific date.

        Args:
            cash_date: Date for the balances
            cash_data: List of dicts with {account_code, currency, amount}

        Returns:
            Number of entries created
        """
        with self.db.get_session() as session:
            # Delete existing cash for this date
            session.query(DailyCash).filter(
                DailyCash.cash_date == cash_date
            ).delete()

            count = 0
            for c in cash_data:
                cash = DailyCash(
                    cash_date=cash_date,
                    account_code=c['account_code'],
                    currency=c['currency'],
                    amount=c['amount'],
                    amount_eur=c.get('amount_eur'),
                )
                session.add(cash)
                count += 1

            session.commit()
            return count

    def get_cash_for_date(self, cash_date: date, account_code: str = None) -> list[dict]:
        """Get cash balances for a specific date."""
        with self.db.get_session() as session:
            query = session.query(DailyCash).filter(
                DailyCash.cash_date == cash_date
            )
            if account_code:
                query = query.filter(DailyCash.account_code == account_code)

            cash = query.all()

            # If no cash for this date, look for previous
            if not cash:
                prev_date = session.query(func.max(DailyCash.cash_date)).filter(
                    DailyCash.cash_date < cash_date
                ).scalar()

                if prev_date:
                    query = session.query(DailyCash).filter(
                        DailyCash.cash_date == prev_date
                    )
                    if account_code:
                        query = query.filter(DailyCash.account_code == account_code)
                    cash = query.all()

            return [
                {
                    'date': c.cash_date,
                    'account': c.account_code,
                    'currency': c.currency,
                    'amount': c.amount,
                    'amount_eur': c.amount_eur,
                }
                for c in cash
            ]

    def add_cash_movement(
        self,
        account_code: str,
        movement_date: date,
        movement_type: str,  # DEPOSIT, WITHDRAWAL, TRANSFER_IN, TRANSFER_OUT, DIVIDEND
        amount: float,
        currency: str = 'EUR',
        counterpart_account: str = None,
        symbol: str = None,
        notes: str = None,
    ) -> int:
        """
        Record a cash movement.

        Returns:
            Movement ID
        """
        with self.db.get_session() as session:
            movement = CashMovement(
                account_code=account_code,
                movement_date=movement_date,
                movement_type=movement_type.upper(),
                amount=amount,
                currency=currency,
                counterpart_account=counterpart_account,
                symbol=symbol,
                notes=notes,
            )
            session.add(movement)
            session.commit()
            return movement.id

    def get_cash_movements(
        self,
        start_date: date = None,
        end_date: date = None,
        account_code: str = None
    ) -> list[dict]:
        """Get cash movements with optional filters."""
        with self.db.get_session() as session:
            query = session.query(CashMovement)

            if start_date:
                query = query.filter(CashMovement.movement_date >= start_date)
            if end_date:
                query = query.filter(CashMovement.movement_date <= end_date)
            if account_code:
                query = query.filter(CashMovement.account_code == account_code)

            movements = query.order_by(CashMovement.movement_date).all()

            return [
                {
                    'id': m.id,
                    'date': m.movement_date,
                    'account': m.account_code,
                    'type': m.movement_type,
                    'amount': m.amount,
                    'currency': m.currency,
                    'counterpart': m.counterpart_account,
                    'symbol': m.symbol,
                    'notes': m.notes,
                }
                for m in movements
            ]

    # =========================================================================
    # PORTFOLIO VALUATION
    # =========================================================================

    def get_portfolio_value(self, target_date: date, account_code: str = None) -> dict:
        """
        Calculate portfolio value for a specific date using daily_holdings.

        Returns:
            Dict with holdings values, cash, and total by account
        """
        with self.db.get_session() as session:
            # Get exchange rates from centralized service
            rate_service = get_exchange_rate_service(self.db)
            try:
                eur_usd = rate_service.get_eur_usd(target_date)
                cad_eur = rate_service.get_cad_eur(target_date)
                chf_eur = rate_service.get_chf_eur(target_date)
            except ExchangeRateError as e:
                logger.error(f"Could not get exchange rates for {target_date}: {e}")
                raise

            # Get holdings
            holdings = self.get_holdings_for_date(target_date, account_code)

            result = {
                'date': target_date,
                'rates': {'EUR_USD': eur_usd, 'CAD_EUR': cad_eur, 'CHF_EUR': chf_eur},
                'accounts': {},
                'total_holdings': 0,
                'total_cash': 0,
                'total': 0,
            }

            # Calculate holdings values
            for h in holdings:
                acc = h['account_code']
                if acc not in result['accounts']:
                    result['accounts'][acc] = {'holdings': {}, 'holdings_total': 0, 'cash': 0, 'total': 0}

                sym = session.query(Symbol).filter(Symbol.code == h['symbol']).first()
                if sym:
                    price = None
                    for i in range(6):
                        d = target_date - timedelta(days=i)
                        p = session.query(PriceHistory).filter(
                            PriceHistory.symbol_id == sym.id,
                            PriceHistory.date == d
                        ).first()
                        if p:
                            price = p.close
                            break

                    if price:
                        value_local = price * h['shares']
                        cur = sym.currency or h['currency'] or 'USD'

                        if cur == 'USD':
                            value_eur = value_local / eur_usd
                        elif cur == 'CAD':
                            value_eur = value_local * cad_eur
                        elif cur == 'CHF':
                            value_eur = value_local * chf_eur
                        else:
                            value_eur = value_local

                        result['accounts'][acc]['holdings'][h['symbol']] = {
                            'shares': h['shares'],
                            'price': price,
                            'currency': cur,
                            'value_local': value_local,
                            'value_eur': value_eur,
                        }
                        result['accounts'][acc]['holdings_total'] += value_eur
                        result['total_holdings'] += value_eur

            # Add cash
            cash = self.get_cash_for_date(target_date, account_code)
            for c in cash:
                acc = c['account']
                if acc not in result['accounts']:
                    result['accounts'][acc] = {'holdings': {}, 'holdings_total': 0, 'cash': 0, 'total': 0}

                if c['currency'] == 'USD':
                    cash_eur = c['amount'] / eur_usd
                else:
                    cash_eur = c['amount']

                result['accounts'][acc]['cash'] += cash_eur
                result['total_cash'] += cash_eur

            # Calculate totals
            for acc in result['accounts']:
                result['accounts'][acc]['total'] = (
                    result['accounts'][acc]['holdings_total'] +
                    result['accounts'][acc]['cash']
                )

            result['total'] = result['total_holdings'] + result['total_cash']

            return result


# Singleton
_tracking_service = None

def get_tracking_service(db_manager=None) -> DailyTrackingService:
    global _tracking_service
    if _tracking_service is None:
        _tracking_service = DailyTrackingService(db_manager)
    return _tracking_service
