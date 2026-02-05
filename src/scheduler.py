"""
Scheduler for automated daily data downloads from Yahoo Finance.
Runs at 00:01 AM Eastern Time (New York) every day.
"""

import logging
import time
from datetime import datetime
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
import pytz

from .config import get_settings
from .database import get_db_manager, Symbol
from .yahoo_downloader import YahooDownloader
from .technical import MetricsCalculator
from .posicion_calculator import PosicionCalculator
from .news_manager import NewsManager

logger = logging.getLogger(__name__)

# Eastern Time zone
ET = pytz.timezone('America/New_York')


class DailyDataUpdater:
    """Updates all financial data daily from Yahoo Finance."""

    def __init__(self):
        self.settings = get_settings()
        self.db = get_db_manager()
        self.downloader = YahooDownloader()
        self.metrics_calculator = MetricsCalculator()
        self.posicion_calculator = PosicionCalculator(self.db)
        self.news_manager = NewsManager()

    def get_all_symbols(self) -> list[str]:
        """Get all symbols from database."""
        with self.db.get_session() as session:
            symbols = session.query(Symbol.code).all()
            return [s[0] for s in symbols]

    def update_prices(self) -> dict:
        """
        Update prices for all symbols (incremental - only new data).

        Returns:
            Dictionary with update results
        """
        started_at = datetime.now(ET)
        symbols = self.get_all_symbols()

        logger.info(f"[{started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}] Starting daily price update for {len(symbols)} symbols")

        results = {
            "date": started_at.strftime('%Y-%m-%d'),
            "total": len(symbols),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "new_records": 0,
            "errors": [],
        }

        for i, symbol in enumerate(symbols, 1):
            try:
                # Download only recent data (last 5 days) for efficiency
                count = self.downloader.download_historical_prices(
                    symbol, period="5d", incremental=True
                )

                if count > 0:
                    results["success"] += 1
                    results["new_records"] += count
                else:
                    results["skipped"] += 1

                if i % 100 == 0:
                    logger.info(
                        f"Progress: {i}/{len(symbols)} | "
                        f"Updated: {results['success']} | "
                        f"New records: {results['new_records']}"
                    )

            except Exception as e:
                results["failed"] += 1
                error_msg = str(e)[:100]
                results["errors"].append({"symbol": symbol, "error": error_msg})
                logger.warning(f"{symbol}: Error - {error_msg}")

            # Small delay to avoid rate limiting
            if i % 50 == 0:
                time.sleep(0.5)

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
            f"Daily price update completed in {elapsed/60:.1f} min: "
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

    def update_posicion(self) -> dict:
        """
        Update posicion table for all accounts.
        Calculates portfolio values from holding_diario + prices.

        Returns:
            Dictionary with update results
        """
        started_at = datetime.now(ET)
        logger.info("Starting posicion calculation")

        try:
            # First update any missing dates
            missing_results = self.posicion_calculator.recalc_missing_dates()

            # Then update today/most recent
            today_results = self.posicion_calculator.recalc_today()

            results = {
                'missing_dates': missing_results['processed'],
                'today_updated': today_results['date'],
                'total_value': today_results['total_value'],
                'success': True,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error updating posicion: {e}")
            results = {
                'missing_dates': 0,
                'today_updated': None,
                'total_value': 0,
                'success': False,
                'error': str(e)
            }

        elapsed = (datetime.now(ET) - started_at).total_seconds()
        logger.info(
            f"Posicion calculation completed in {elapsed:.1f}s: "
            f"{results['missing_dates']} missing dates, "
            f"today={results['today_updated']}, "
            f"total={results['total_value']:,.0f} EUR"
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

    def run_daily_update(self):
        """Run the complete daily update (prices + fundamentals + metrics + news)."""
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

        # Update posicion table
        posicion_results = self.update_posicion()

        # Update news
        news_results = self.update_news()

        logger.info("=" * 60)
        logger.info("DAILY UPDATE COMPLETED")
        logger.info(f"Prices: {price_results['new_records']} new records")
        if fund_results:
            logger.info(f"Fundamentals: {fund_results['success']} updated")
        logger.info(f"Metrics: {metrics_results['success']} calculated")
        logger.info(f"Posicion: {posicion_results['total_value']:,.0f} EUR")
        logger.info(f"News: {news_results['total_saved']} articles")
        logger.info("=" * 60)

        return {"prices": price_results, "fundamentals": fund_results, "metrics": metrics_results, "posicion": posicion_results, "news": news_results}


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
        Add daily update job at specified time in Eastern Time.

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
            f"Scheduled daily update at {hour:02d}:{minute:02d} Eastern Time"
        )

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
            print("  python -m src.scheduler --fundamentals  # Update fundamentals only")
            print("  python -m src.scheduler --metrics       # Calculate technical metrics")
            print("  python -m src.scheduler --status        # Show status")
    else:
        run_scheduler_daemon()
