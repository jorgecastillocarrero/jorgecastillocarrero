"""
Scheduler for automated daily data downloads from Yahoo Finance.
Runs at 00:01 AM Spain Time (Europe/Madrid) every day.
"""

import logging
import time
import io
import csv
from datetime import datetime, timedelta
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
import pytz

from .config import get_settings
from .database import get_db_manager, Symbol
from .yahoo_downloader import YahooDownloader
from .technical import MetricsCalculator
from .valor_actual import ValorActualCalculator
from .news_manager import NewsManager

logger = logging.getLogger(__name__)

# Spain Time zone (Europe/Madrid)
ET = pytz.timezone('Europe/Madrid')


class DailyDataUpdater:
    """Updates all financial data daily from Yahoo Finance."""

    def __init__(self):
        self.settings = get_settings()
        self.db = get_db_manager()
        self.downloader = YahooDownloader()
        self.metrics_calculator = MetricsCalculator()
        self.valor_actual = ValorActualCalculator(self.db)
        self.news_manager = NewsManager()

    def get_all_symbols(self) -> list[str]:
        """Get all symbols from database."""
        with self.db.get_session() as session:
            symbols = session.query(Symbol.code).all()
            return [s[0] for s in symbols]

    def update_missing_prices(self) -> dict:
        """
        Update prices only for symbols missing the last market day's data.
        Automatically detects the last market day from existing data.

        Returns:
            Dictionary with update results
        """
        from sqlalchemy import text
        started_at = datetime.now(ET)

        # Find the last two market days with data in the database
        with self.db.get_session() as session:
            # Get the two most recent dates with price data
            dates_result = session.execute(text('''
                SELECT DISTINCT DATE(date) as d
                FROM price_history
                ORDER BY d DESC
                LIMIT 2
            ''')).fetchall()

            if len(dates_result) < 2:
                logger.info("Not enough historical data to detect missing symbols")
                return {"total": 0, "success": 0, "failed": 0, "new_records": 0}

            target_date = str(dates_result[0][0])  # Most recent date (last market day)
            prev_date = str(dates_result[1][0])    # Previous market day

            # Get symbols that have prev_date data but not target_date data
            result = session.execute(text('''
                SELECT DISTINCT s.code
                FROM symbols s
                JOIN price_history p ON s.id = p.symbol_id
                WHERE DATE(p.date) = :prev_date
                AND NOT EXISTS (
                    SELECT 1 FROM price_history p2
                    WHERE p2.symbol_id = s.id AND DATE(p2.date) = :target_date
                )
                ORDER BY s.code
            '''), {'prev_date': prev_date, 'target_date': target_date})
            missing_symbols = [r[0] for r in result.fetchall()]

        if not missing_symbols:
            logger.info(f"No missing symbols to update for {target_date}")
            return {"total": 0, "success": 0, "failed": 0, "new_records": 0}

        logger.info(f"Updating {len(missing_symbols)} symbols missing data for {target_date} (parallel)")

        results = {
            "date": target_date,
            "total": len(missing_symbols),
            "success": 0,
            "failed": 0,
            "new_records": 0,
        }

        # Use parallel downloads (5 workers to avoid Yahoo rate limiting)
        completed = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self._download_symbol_safe, symbol): symbol
                for symbol in missing_symbols
            }

            for future in as_completed(future_to_symbol):
                symbol, count, error = future.result()

                with lock:
                    completed += 1
                    if error:
                        results["failed"] += 1
                    elif count > 0:
                        results["success"] += 1
                        results["new_records"] += count

                    if completed % 200 == 0:
                        logger.info(f"Missing update progress: {completed}/{len(missing_symbols)}")

        logger.info(f"Missing prices update completed: {results['success']} updated, {results['new_records']} new records")
        return results

    def get_symbol_map(self) -> dict:
        """Get mapping of symbol code to symbol_id."""
        from sqlalchemy import text
        with self.db.get_session() as session:
            result = session.execute(text('SELECT id, code FROM symbols'))
            return {r[1]: r[0] for r in result.fetchall()}

    def update_prices(self, batch_size: int = 100) -> dict:
        """
        Update prices for all symbols using batch download + COPY.
        This is 10-100x faster than individual downloads.

        Args:
            batch_size: Number of symbols per batch (default: 100)

        Returns:
            Dictionary with update results
        """
        from sqlalchemy import text
        started_at = datetime.now(ET)

        # Get all symbols with their IDs
        symbol_map = self.get_symbol_map()
        symbols = list(symbol_map.keys())

        logger.info(f"[{started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}] Starting COPY price update for {len(symbols)} symbols (batch={batch_size})")

        results = {
            "date": started_at.strftime('%Y-%m-%d'),
            "total": len(symbols),
            "success": 0,
            "failed": 0,
            "new_records": 0,
        }

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]

            # Batch download from Yahoo
            try:
                data = yf.download(batch, period='5d', progress=False, threads=True, auto_adjust=False)
            except Exception as e:
                results["failed"] += len(batch)
                continue

            if data.empty:
                continue

            # Prepare CSV data for COPY
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer, delimiter='\t')
            rows_written = 0

            for symbol in batch:
                if symbol not in symbol_map:
                    continue
                symbol_id = symbol_map[symbol]

                try:
                    if len(batch) == 1:
                        symbol_data = data
                    else:
                        if symbol not in data.columns.get_level_values(1):
                            continue
                        symbol_data = data.xs(symbol, axis=1, level=1)

                    if symbol_data is None or symbol_data.empty:
                        continue

                    for date_idx, row in symbol_data.iterrows():
                        if row.isna().all():
                            continue
                        writer.writerow([
                            symbol_id,
                            date_idx.date(),
                            float(row['Open']) if not row.isna()['Open'] else '',
                            float(row['High']) if not row.isna()['High'] else '',
                            float(row['Low']) if not row.isna()['Low'] else '',
                            float(row['Close']) if not row.isna()['Close'] else '',
                            float(row['Adj Close']) if not row.isna()['Adj Close'] else '',
                            int(row['Volume']) if not row.isna()['Volume'] else '',
                            datetime.now().isoformat()
                        ])
                        rows_written += 1
                    results["success"] += 1
                except Exception:
                    results["failed"] += 1
                    continue

            # Use COPY to insert (much faster than individual INSERTs)
            if rows_written > 0:
                csv_buffer.seek(0)

                with self.db.get_session() as session:
                    conn = session.connection().connection
                    cursor = conn.cursor()

                    try:
                        # Create temp table
                        cursor.execute('''
                            CREATE TEMP TABLE IF NOT EXISTS temp_prices (
                                symbol_id INTEGER,
                                date DATE,
                                open FLOAT,
                                high FLOAT,
                                low FLOAT,
                                close FLOAT,
                                adjusted_close FLOAT,
                                volume BIGINT,
                                created_at TIMESTAMP
                            ) ON COMMIT DROP
                        ''')

                        # COPY to temp table
                        cursor.copy_from(csv_buffer, 'temp_prices', sep='\t', null='')

                        # Insert from temp to real table with ON CONFLICT
                        cursor.execute('''
                            INSERT INTO price_history (symbol_id, date, open, high, low, close, adjusted_close, volume, created_at)
                            SELECT symbol_id, date, open, high, low, close, adjusted_close, volume, created_at
                            FROM temp_prices
                            ON CONFLICT (symbol_id, date) DO NOTHING
                        ''')

                        conn.commit()
                        results["new_records"] += rows_written
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"COPY error: {e}")

            # Log progress every 500 symbols
            done = min(i + batch_size, len(symbols))
            if done % 500 == 0 or done == len(symbols):
                elapsed = (datetime.now(ET) - started_at).total_seconds()
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(symbols) - done) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"Progress: {done}/{len(symbols)} ({done*100//len(symbols)}%) | "
                    f"Rate: {rate:.0f}/s | ETA: {eta:.1f}min"
                )

        # Log the operation
        with self.db.get_session() as session:
            self.db.log_download(
                session,
                operation="daily_prices",
                status="success" if results["failed"] == 0 else "partial",
                records_downloaded=results["new_records"],
                error_message=f"{results['failed']} failures" if results["failed"] else None,
                started_at=started_at.replace(tzinfo=None),
            )

        elapsed = (datetime.now(ET) - started_at).total_seconds()
        logger.info(
            f"Parallel price update completed in {elapsed/60:.1f} min: "
            f"{results['success']} updated, {results['skipped']} up-to-date, "
            f"{results['failed']} failed, {results['new_records']} new records"
        )

        return results

    def update_fundamentals(self) -> dict:
        """
        Update fundamentals for all symbols.

        Returns:
            Dictionary with update results
        """
        started_at = datetime.now(ET)
        symbols = self.get_all_symbols()

        logger.info(f"Starting fundamentals update for {len(symbols)} symbols")

        results = {
            "date": started_at.strftime('%Y-%m-%d'),
            "total": len(symbols),
            "success": 0,
            "failed": 0,
            "errors": [],
        }

        for i, symbol in enumerate(symbols, 1):
            try:
                if self.downloader.download_fundamentals(symbol):
                    results["success"] += 1
                else:
                    results["failed"] += 1

                if i % 100 == 0:
                    logger.info(f"Fundamentals progress: {i}/{len(symbols)}")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"symbol": symbol, "error": str(e)[:100]})

            if i % 50 == 0:
                time.sleep(0.5)

        with self.db.get_session() as session:
            self.db.log_download(
                session,
                operation="daily_fundamentals",
                status="success" if results["failed"] == 0 else "partial",
                records_downloaded=results["success"],
                started_at=started_at.replace(tzinfo=None),
            )

        logger.info(
            f"Fundamentals update completed: "
            f"{results['success']} success, {results['failed']} failed"
        )

        return results

    def update_metrics(self) -> dict:
        """
        Update technical metrics for all symbols (last 30 days).

        Returns:
            Dictionary with update results
        """
        started_at = datetime.now(ET)
        logger.info("Starting metrics calculation")

        # Calculate metrics for last 30 days
        from datetime import timedelta
        start_date = datetime.now() - timedelta(days=30)

        results = self.metrics_calculator.calculate_for_all_symbols(
            start_date=start_date
        )

        elapsed = (datetime.now(ET) - started_at).total_seconds()
        logger.info(
            f"Metrics calculation completed in {elapsed/60:.1f} min: "
            f"{results['success']} success, {results['skipped']} skipped, "
            f"{results['failed']} failed"
        )

        return results

    def update_news(self) -> dict:
        """
        Update news from external sources (GDELT, NewsAPI).

        Returns:
            Dictionary with update results
        """
        started_at = datetime.now(ET)
        logger.info("Starting news update")

        try:
            # Run daily news update
            results = self.news_manager.run_daily_update()
            results['success'] = True
            results['error'] = None

        except Exception as e:
            logger.error(f"Error updating news: {e}")
            results = {
                'gdelt': 0,
                'newsapi': 0,
                'total_saved': 0,
                'success': False,
                'error': str(e)
            }

        elapsed = (datetime.now(ET) - started_at).total_seconds()
        logger.info(
            f"News update completed in {elapsed:.1f}s: "
            f"{results['total_saved']} articles saved "
            f"(GDELT: {results['gdelt']}, NewsAPI: {results['newsapi']})"
        )

        return results

    def update_positions(self) -> dict:
        """
        Update portfolio positions (holding_diario, cash_diario, posicion).

        Returns:
            Dictionary with update results
        """
        from sqlalchemy import text
        started_at = datetime.now(ET)
        logger.info("Starting positions calculation")

        results = {
            'fecha': None,
            'valid': False,
            'holdings_count': 0,
            'total_eur': 0,
            'error': None
        }

        try:
            # Calculate valor actual (updates holding_diario and cash_diario)
            valor_results = self.valor_actual.calculate_valor_actual()

            results['fecha'] = valor_results.get('fecha')
            results['valid'] = valor_results.get('valid', False)
            results['holdings_count'] = valor_results.get('holdings_count', 0)
            results['total_eur'] = valor_results.get('total_eur', 0)
            results['error'] = valor_results.get('error')

            if results['valid'] and valor_results.get('by_account'):
                # Save to posicion table
                with self.db.get_session() as session:
                    fecha = results['fecha']

                    # Delete existing records for this date
                    session.execute(text(
                        "DELETE FROM posicion WHERE DATE(fecha) = :fecha"
                    ), {'fecha': fecha})

                    # Insert new records per account
                    for account, data in valor_results['by_account'].items():
                        session.execute(text("""
                            INSERT INTO posicion (fecha, account_code, holding_eur, cash_eur, total_eur)
                            VALUES (:fecha, :account, :holding, :cash, :total)
                        """), {
                            'fecha': fecha,
                            'account': account,
                            'holding': data['holdings'],
                            'cash': data['cash'],
                            'total': data['total']
                        })

                    session.commit()
                    logger.info(f"Saved posicion for {len(valor_results['by_account'])} accounts on {fecha}")

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Error updating positions: {e}")

        elapsed = (datetime.now(ET) - started_at).total_seconds()
        logger.info(f"Positions calculation completed in {elapsed:.1f}s: {results['total_eur']:,.0f} EUR")

        return results

    def run_daily_update(self):
        """Run the complete daily update (prices + fundamentals + metrics + news + positions)."""
        logger.info("=" * 60)
        logger.info("DAILY UPDATE STARTED")
        logger.info(f"Time: {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info("=" * 60)

        # Update prices
        price_results = self.update_prices()

        # Update fundamentals (weekly on Sundays, or can be run daily)
        if datetime.now(ET).weekday() == 6:  # Sunday
            fund_results = self.update_fundamentals()
        else:
            fund_results = None
            logger.info("Skipping fundamentals update (only runs on Sundays)")

        # Update technical metrics
        metrics_results = self.update_metrics()

        # Update news
        news_results = self.update_news()

        # Update portfolio positions (holding_diario, cash_diario, posicion)
        position_results = self.update_positions()

        logger.info("=" * 60)
        logger.info("DAILY UPDATE COMPLETED")
        logger.info(f"Prices: {price_results['new_records']} new records")
        if fund_results:
            logger.info(f"Fundamentals: {fund_results['success']} updated")
        logger.info(f"Metrics: {metrics_results['success']} calculated")
        logger.info(f"News: {news_results['total_saved']} articles")
        logger.info(f"Positions: {position_results['total_eur']:,.0f} EUR ({position_results['holdings_count']} holdings)")
        logger.info("=" * 60)

        return {"prices": price_results, "fundamentals": fund_results, "metrics": metrics_results, "news": news_results, "positions": position_results}


class SchedulerManager:
    """Manages scheduled data download jobs."""

    def __init__(self, blocking=False):
        self.settings = get_settings()
        if blocking:
            self.scheduler = BlockingScheduler()
        else:
            self.scheduler = BackgroundScheduler()
        self.updater = DailyDataUpdater()

        # Add event listeners
        self.scheduler.add_listener(
            self._on_job_executed, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

    def _on_job_executed(self, event):
        """Handle job execution events."""
        if event.exception:
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            logger.info(f"Job {event.job_id} executed successfully")

    def add_daily_update_job(self, hour: int = 0, minute: int = 1):
        """
        Add daily update job at specified time in Spain Time.

        Args:
            hour: Hour in ET (default: 0 = midnight)
            minute: Minute (default: 1)
        """
        self.scheduler.add_job(
            self.updater.run_daily_update,
            trigger=CronTrigger(
                hour=hour,
                minute=minute,
                timezone=ET
            ),
            id="daily_update",
            replace_existing=True,
            name=f"Daily Update at {hour:02d}:{minute:02d} ET",
        )

        logger.info(
            f"Scheduled daily update at {hour:02d}:{minute:02d} Spain Time"
        )

    def add_hourly_missing_update_job(self):
        """
        Add hourly job to complete any missing price downloads.
        Runs every hour from 01:00 to 23:00 Spain Time.
        """
        self.scheduler.add_job(
            self.updater.update_missing_prices,
            trigger=CronTrigger(
                hour='1-23',
                minute=30,
                timezone=ET
            ),
            id="hourly_missing_update",
            replace_existing=True,
            name="Hourly Missing Update (every hour at :30)",
        )

        logger.info("Scheduled hourly missing update at :30 every hour")

    def add_weekly_fundamentals_job(self, day_of_week: str = 'sun', hour: int = 1):
        """
        Add weekly fundamentals update job.

        Args:
            day_of_week: Day to run (mon, tue, wed, thu, fri, sat, sun)
            hour: Hour in ET
        """
        self.scheduler.add_job(
            self.updater.update_fundamentals,
            trigger=CronTrigger(
                day_of_week=day_of_week,
                hour=hour,
                minute=0,
                timezone=ET
            ),
            id="weekly_fundamentals",
            replace_existing=True,
            name=f"Weekly Fundamentals ({day_of_week} {hour:02d}:00 ET)",
        )

        logger.info(f"Scheduled weekly fundamentals on {day_of_week} at {hour:02d}:00 ET")

    def run_now(self, job_id: str = "daily_update"):
        """Run a specific job immediately."""
        if job_id == "daily_update":
            logger.info("Running daily update now...")
            return self.updater.run_daily_update()
        elif job_id == "fundamentals":
            logger.info("Running fundamentals update now...")
            return self.updater.update_fundamentals()
        elif job_id == "prices":
            logger.info("Running prices update now...")
            return self.updater.update_prices()
        elif job_id == "metrics":
            logger.info("Running metrics calculation now...")
            return self.updater.update_metrics()
        else:
            logger.warning(f"Unknown job: {job_id}")

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

    def get_jobs(self) -> list[dict]:
        """Get list of scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = getattr(job, 'next_run_time', None)
            if next_run:
                next_run_et = next_run.astimezone(ET)
                next_run_str = next_run_et.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                next_run_str = "Pending (starts after scheduler.start())"

            jobs.append({
                "id": job.id,
                "name": getattr(job, 'name', job.id),
                "next_run": next_run_str,
                "trigger": str(job.trigger),
            })
        return jobs


def run_scheduler_daemon():
    """Run the scheduler as a daemon with default jobs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("scheduler.log")
        ]
    )

    print("\n" + "=" * 60)
    print("FINANCIAL DATA SCHEDULER")
    print("=" * 60)
    print(f"Current time (ET): {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    scheduler = SchedulerManager(blocking=True)

    # Schedule daily update at 00:01 ET
    scheduler.add_daily_update_job(hour=0, minute=1)

    # Note: Hourly missing update job disabled - parallel downloads should complete in one run
    # scheduler.add_hourly_missing_update_job()

    # Schedule weekly fundamentals on Sunday at 01:00 ET
    scheduler.add_weekly_fundamentals_job(day_of_week='sun', hour=1)

    print("Scheduled jobs:")
    for job in scheduler.get_jobs():
        print(f"  - {job['name']}")
        print(f"    Next run: {job['next_run']}")

    print()
    print("Scheduler is running. Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.stop()
        print("\nScheduler stopped.")


# CLI functionality
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        if cmd == "--daemon":
            run_scheduler_daemon()

        elif cmd == "--run-now":
            print("\n=== Running Daily Update Now ===\n")
            updater = DailyDataUpdater()
            results = updater.run_daily_update()
            print(f"\nPrices: {results['prices']['new_records']} new records")
            if results['fundamentals']:
                print(f"Fundamentals: {results['fundamentals']['success']} updated")

        elif cmd == "--prices":
            print("\n=== Running Prices Update ===\n")
            updater = DailyDataUpdater()
            results = updater.update_prices()
            print(f"\nResults:")
            print(f"  Updated: {results['success']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Failed: {results['failed']}")
            print(f"  New records: {results['new_records']}")

        elif cmd == "--fundamentals":
            print("\n=== Running Fundamentals Update ===\n")
            updater = DailyDataUpdater()
            results = updater.update_fundamentals()
            print(f"\nResults:")
            print(f"  Success: {results['success']}")
            print(f"  Failed: {results['failed']}")

        elif cmd == "--metrics":
            print("\n=== Running Metrics Calculation ===\n")
            updater = DailyDataUpdater()
            results = updater.update_metrics()
            print(f"\nResults:")
            print(f"  Success: {results['success']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Failed: {results['failed']}")

        elif cmd == "--positions":
            print("\n=== Running Positions Calculation ===\n")
            updater = DailyDataUpdater()
            results = updater.update_positions()
            print(f"\nResults:")
            print(f"  Fecha: {results['fecha']}")
            print(f"  Valid: {results['valid']}")
            print(f"  Holdings: {results['holdings_count']}")
            print(f"  Total: {results['total_eur']:,.0f} EUR")
            if results['error']:
                print(f"  Error: {results['error']}")

        elif cmd == "--missing":
            print("\n=== Running Missing Prices Update ===\n")
            updater = DailyDataUpdater()
            results = updater.update_missing_prices()
            print(f"\nResults:")
            print(f"  Total missing: {results['total']}")
            print(f"  Updated: {results['success']}")
            print(f"  Failed: {results['failed']}")
            print(f"  New records: {results['new_records']}")

        elif cmd == "--status":
            print("\n=== Scheduler Status ===\n")
            print(f"Current time (ET): {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Current time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print()

            db = get_db_manager()
            with db.get_session() as session:
                stats = db.get_statistics(session)
                print("Database statistics:")
                print(f"  Symbols: {stats['symbols']:,}")
                print(f"  Price records: {stats['price_records']:,}")
                print(f"  Fundamentals: {stats['fundamentals']:,}")
                print(f"  Daily metrics: {stats['daily_metrics']:,}")
                print(f"  Portfolios: {stats['portfolios']:,}")

        else:
            print("Usage:")
            print("  python -m src.scheduler --daemon        # Run as daemon")
            print("  python -m src.scheduler --run-now       # Run full update now")
            print("  python -m src.scheduler --prices        # Update prices only")
            print("  python -m src.scheduler --missing       # Update missing prices only")
            print("  python -m src.scheduler --fundamentals  # Update fundamentals only")
            print("  python -m src.scheduler --metrics       # Calculate technical metrics")
            print("  python -m src.scheduler --positions     # Calculate portfolio positions")
            print("  python -m src.scheduler --status        # Show status")
    else:
        run_scheduler_daemon()
