"""
Task Scheduler - Schedules daily video generation tasks
"""

from datetime import datetime
from typing import Callable, Dict, Any, Optional
import pytz

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskScheduler:
    """
    Schedules and manages recurring tasks for video generation.

    Supports:
    - Daily scheduling at specific times
    - Multiple timezone support
    - Job management (pause, resume, remove)
    """

    def __init__(
        self,
        timezone: str = "Asia/Kolkata",
        blocking: bool = True
    ):
        """
        Initialize task scheduler.

        Args:
            timezone: Timezone for scheduling
            blocking: Use blocking scheduler (True) or background (False)
        """
        self.timezone = pytz.timezone(timezone)

        if blocking:
            self.scheduler = BlockingScheduler(timezone=self.timezone)
        else:
            self.scheduler = BackgroundScheduler(timezone=self.timezone)

        self.jobs: Dict[str, Any] = {}

        logger.info(f"TaskScheduler initialized with timezone: {timezone}")

    def add_daily_job(
        self,
        job_id: str,
        func: Callable,
        hour: int = 6,
        minute: int = 0,
        days: list = None,
        **kwargs
    ) -> bool:
        """
        Add a daily job to the scheduler.

        Args:
            job_id: Unique job identifier
            func: Function to execute
            hour: Hour to run (24-hour format)
            minute: Minute to run
            days: Days of week to run (0=Mon, 6=Sun), None for all days
            **kwargs: Arguments to pass to function

        Returns:
            True if job added successfully
        """
        try:
            # Build day of week string
            if days:
                dow = ",".join(str(d) for d in days)
            else:
                dow = "*"  # All days

            trigger = CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week=dow,
                timezone=self.timezone
            )

            job = self.scheduler.add_job(
                func,
                trigger=trigger,
                id=job_id,
                name=job_id,
                kwargs=kwargs,
                replace_existing=True
            )

            self.jobs[job_id] = job
            logger.info(f"Added daily job: {job_id} at {hour:02d}:{minute:02d}")

            return True

        except Exception as e:
            logger.error(f"Failed to add job {job_id}: {e}")
            return False

    def add_interval_job(
        self,
        job_id: str,
        func: Callable,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        **kwargs
    ) -> bool:
        """
        Add an interval-based job.

        Args:
            job_id: Unique job identifier
            func: Function to execute
            hours: Interval hours
            minutes: Interval minutes
            seconds: Interval seconds
            **kwargs: Arguments to pass to function

        Returns:
            True if job added successfully
        """
        try:
            job = self.scheduler.add_job(
                func,
                "interval",
                id=job_id,
                name=job_id,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                kwargs=kwargs,
                replace_existing=True
            )

            self.jobs[job_id] = job
            logger.info(f"Added interval job: {job_id} every {hours}h {minutes}m {seconds}s")

            return True

        except Exception as e:
            logger.error(f"Failed to add interval job {job_id}: {e}")
            return False

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self.jobs:
                del self.jobs[job_id]
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job"""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False

    def run_job_now(self, job_id: str) -> bool:
        """Run a job immediately"""
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.now(self.timezone))
                logger.info(f"Triggered immediate run: {job_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to run job {job_id}: {e}")
            return False

    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a job"""
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                return {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time,
                    "trigger": str(job.trigger)
                }
        except Exception:
            pass
        return None

    def list_jobs(self) -> list:
        """List all scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time,
                "trigger": str(job.trigger)
            })
        return jobs

    def start(self) -> None:
        """Start the scheduler"""
        logger.info("Starting scheduler...")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped by user")
            self.stop()

    def stop(self) -> None:
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.scheduler.shutdown(wait=False)

    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self.scheduler.running
