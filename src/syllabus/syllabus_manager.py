"""
Syllabus Manager for AI History Video Generator.
Loads syllabus from YAML, populates database, and manages topic sequencing.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.utils.database import Database, Topic
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyllabusManager:
    """Manages the history syllabus - loading, sequencing, and progress tracking."""

    def __init__(self, syllabus_path: str = "config/syllabus.yaml",
                 db: Database = None,
                 db_path: str = "data/history_tracker.db"):
        """
        Initialize SyllabusManager.

        Args:
            syllabus_path: Path to the syllabus YAML file.
            db: Optional Database instance (creates one if not provided).
            db_path: Path to SQLite database file.
        """
        self.syllabus_path = syllabus_path
        self.db = db or Database(db_path=db_path)
        self.syllabus_data = None

        # Load syllabus YAML
        self._load_syllabus()

        # Initialize database if empty
        if self.db.get_topic_count() == 0:
            self.initialize_topics()

    def _load_syllabus(self):
        """Load syllabus from YAML file."""
        path = Path(self.syllabus_path)
        if not path.exists():
            raise FileNotFoundError(f"Syllabus file not found: {self.syllabus_path}")

        with open(path, 'r', encoding='utf-8') as f:
            self.syllabus_data = yaml.safe_load(f)

        syllabus = self.syllabus_data.get('syllabus', {})
        logger.info(
            f"Loaded syllabus: {syllabus.get('name', 'Unknown')} "
            f"({syllabus.get('total_parts', 0)} parts)"
        )

    def initialize_topics(self) -> int:
        """
        Populate the database from the syllabus YAML.
        Only runs if database is empty.

        Returns:
            Number of topics added.
        """
        if self.db.get_topic_count() > 0:
            logger.info("Database already populated, skipping initialization")
            return 0

        syllabus = self.syllabus_data.get('syllabus', {})
        eras = syllabus.get('eras', [])
        count = 0

        for era in eras:
            era_name = era.get('name', '')
            era_hindi = era.get('hindi_name', '')
            color_theme = era.get('color_theme', '')
            sections = era.get('sections', [])

            for section in sections:
                section_name = section.get('name', '')
                topics = section.get('topics', [])

                for topic_data in topics:
                    try:
                        self.db.add_topic({
                            'part_number': topic_data['part'],
                            'era': era_name,
                            'era_hindi': era_hindi,
                            'section': section_name,
                            'title': topic_data['title'],
                            'subtopics': topic_data.get('subtopics', []),
                            'key_concepts': topic_data.get('key_concepts', []),
                            'exam_focus': topic_data.get('exam_focus', 'BOTH'),
                            'previous_year_refs': topic_data.get('previous_year_refs', []),
                            'color_theme': color_theme,
                        })
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to add Part {topic_data.get('part', '?')}: {e}")

        logger.info(f"Initialized {count} topics in database")
        return count

    def get_next_topic(self) -> Optional[Topic]:
        """
        Get the next topic to generate.
        Returns the first pending topic in part number order.
        """
        topic = self.db.get_next_pending_topic()
        if topic:
            logger.info(f"Next topic: Part {topic.part_number} - {topic.title} ({topic.era})")
        else:
            logger.info("All topics completed! Course is done.")
        return topic

    def get_topic_by_part(self, part_number: int) -> Optional[Topic]:
        """Get a specific topic by part number."""
        return self.db.get_topic_by_part(part_number)

    def get_previous_topic(self, part_number: int) -> Optional[Topic]:
        """Get the previous topic (for recap generation)."""
        if part_number <= 1:
            return None
        return self.db.get_topic_by_part(part_number - 1)

    def get_next_topic_preview(self, part_number: int) -> Optional[Topic]:
        """Get the next topic (for outro preview)."""
        total = self.syllabus_data.get('syllabus', {}).get('total_parts', 180)
        if part_number >= total:
            return None
        return self.db.get_topic_by_part(part_number + 1)

    def mark_generating(self, part_number: int) -> bool:
        """Mark a topic as currently being generated."""
        return self.db.update_topic_status(part_number, 'generating')

    def mark_completed(self, part_number: int, youtube_id: str = None,
                       youtube_url: str = None, video_path: str = None,
                       duration: float = None, **kwargs) -> bool:
        """Mark a topic as completed with upload details."""
        return self.db.mark_topic_completed(
            part_number, youtube_id=youtube_id, youtube_url=youtube_url,
            video_path=video_path, duration=duration, **kwargs
        )

    def mark_failed(self, part_number: int, error: str) -> bool:
        """Mark a topic as failed."""
        return self.db.mark_topic_failed(part_number, error)

    def get_progress(self) -> Dict[str, Any]:
        """
        Get overall course progress.

        Returns:
            Dict with keys: total, completed, failed, pending, percentage, current_era, current_part
        """
        return self.db.get_progress()

    def get_era_progress(self) -> List[Dict[str, Any]]:
        """Get progress broken down by era."""
        eras_progress = []
        syllabus = self.syllabus_data.get('syllabus', {})

        for era in syllabus.get('eras', []):
            era_name = era['name']
            parts_range = era.get('parts_range', [0, 0])

            all_topics = self.db.get_all_topics()
            era_topics = [t for t in all_topics if t.era == era_name]
            completed = sum(1 for t in era_topics if t.status == 'completed')
            total = len(era_topics)

            eras_progress.append({
                'era': era_name,
                'hindi_name': era.get('hindi_name', ''),
                'total': total,
                'completed': completed,
                'percentage': round((completed / total * 100), 1) if total > 0 else 0,
                'parts_range': parts_range,
            })

        return eras_progress

    def get_total_parts(self) -> int:
        """Get total number of parts in the syllabus."""
        return self.syllabus_data.get('syllabus', {}).get('total_parts', 180)

    def reset_failed(self) -> int:
        """Reset all failed topics back to pending."""
        return self.db.reset_failed_topics()

    def reset_stuck(self) -> int:
        """Reset stuck 'generating' topics back to pending."""
        return self.db.reset_generating_topics()

    def print_progress(self):
        """Print a formatted progress report."""
        progress = self.get_progress()
        era_progress = self.get_era_progress()

        print("\n" + "=" * 60)
        print("  HISTORY COURSE PROGRESS")
        print("=" * 60)
        print(f"  Overall: {progress['completed']}/{progress['total']} "
              f"({progress['percentage']}%)")
        print(f"  Current: Part {progress['current_part']} ({progress['current_era']})")
        print(f"  Failed: {progress['failed']} | Pending: {progress['pending']}")
        print("-" * 60)

        for era in era_progress:
            bar_len = 30
            filled = int(bar_len * era['percentage'] / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"  {era['era']:<20s} [{bar}] {era['completed']}/{era['total']}")

        print("=" * 60 + "\n")
