"""
Daily Scheduler for AI History Video Generator

This script runs the history video generation pipeline on a daily schedule.
It picks the next topic from the syllabus each day and generates + uploads a video.
"""

import sys
import argparse
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.scheduler import TaskScheduler
from main import HistoryPipeline


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        return {}


def generate_history_task(upload: bool = True):
    """Task function for scheduled history video generation."""
    logger = get_logger("ScheduledTask")

    try:
        logger.info("Starting scheduled history video generation")

        pipeline = HistoryPipeline()
        results = pipeline.run_sync(
            upload=upload,
            test_mode=False
        )

        if results["success"]:
            logger.info(
                f"Scheduled video generated: Part {results['part_number']} - "
                f"{results['topic_title']}"
            )
            if results["steps"].get("upload", {}).get("url"):
                logger.info(f"Video URL: {results['steps']['upload']['url']}")
        else:
            logger.error(f"Scheduled generation failed: {results['errors']}")

    except Exception as e:
        logger.error(f"Scheduled task failed: {e}")


def main():
    """Main entry point for scheduler."""
    parser = argparse.ArgumentParser(
        description="Daily Scheduler for History Video Generation"
    )

    parser.add_argument(
        "--time",
        type=str,
        default="14:00",
        help="Time to run daily (HH:MM format, default: 14:00 = 2 PM IST)"
    )

    parser.add_argument(
        "--timezone",
        type=str,
        default="Asia/Kolkata",
        help="Timezone (default: Asia/Kolkata)"
    )

    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Don't upload to YouTube"
    )

    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run immediately, then continue with schedule"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once immediately and exit"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_level="INFO", log_file="logs/scheduler.log")
    logger = get_logger("Scheduler")

    print("\n" + "=" * 60)
    print("  AI History Video Generator - Daily Scheduler")
    print("  UPSC & State PSC - Complete History Course")
    print("=" * 60 + "\n")

    # Run once mode
    if args.once:
        print("Running history video generation once...\n")
        generate_history_task(upload=not args.no_upload)
        return 0

    # Parse time
    try:
        hour, minute = map(int, args.time.split(":"))
    except ValueError:
        print(f"Invalid time format: {args.time}. Use HH:MM")
        return 1

    # Create scheduler
    scheduler = TaskScheduler(timezone=args.timezone, blocking=True)

    # Add daily job
    scheduler.add_daily_job(
        job_id="daily_history_generation",
        func=generate_history_task,
        hour=hour,
        minute=minute,
        upload=not args.no_upload
    )

    print(f"Scheduled daily history video generation:")
    print(f"  Time: {args.time}")
    print(f"  Timezone: {args.timezone}")
    print(f"  Upload: {not args.no_upload}")
    print()

    # Show current progress
    try:
        pipeline = HistoryPipeline()
        progress = pipeline.syllabus_mgr.get_progress()
        print(f"Course Progress: {progress['completed']}/{progress['total']} "
              f"({progress['percentage']:.1f}%)")
        if progress['completed'] < progress['total']:
            next_topic = pipeline.syllabus_mgr.get_next_topic()
            if next_topic:
                print(f"Next topic: Part {next_topic.part_number} - {next_topic.title}")
        print()
    except Exception:
        pass

    # Run immediately if requested
    if args.run_now:
        print("Running immediately...\n")
        generate_history_task(upload=not args.no_upload)

    # Show next run time
    job_info = scheduler.get_job_info("daily_history_generation")
    if job_info:
        print(f"Next scheduled run: {job_info['next_run']}")

    print("\nScheduler is running. Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
