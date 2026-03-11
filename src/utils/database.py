"""
Database module for AI History Video Generator.
Tracks topic progress, generation logs, and video metadata.
Uses SQLAlchemy ORM with SQLite backend.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, make_transient

from src.utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Topic(Base):
    """Syllabus topic with progress tracking."""
    __tablename__ = 'topics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(Integer, unique=True, nullable=False, index=True)
    era = Column(String(50), nullable=False)           # ancient/medieval/modern
    era_hindi = Column(String(100))
    section = Column(String(200), nullable=False)
    title = Column(String(500), nullable=False)
    subtopics = Column(Text)                           # JSON list
    key_concepts = Column(Text)                        # JSON list
    exam_focus = Column(String(20))                    # PRELIMS/MAINS/BOTH
    previous_year_refs = Column(Text)                  # JSON list
    color_theme = Column(String(50))                   # ancient/medieval/modern

    # Progress tracking
    status = Column(String(20), default='pending', index=True)  # pending/generating/completed/failed
    video_path = Column(String(500))
    audio_path = Column(String(500))
    script_path = Column(String(500))
    thumbnail_path = Column(String(500))
    pdf_path = Column(String(500))
    youtube_id = Column(String(50))
    youtube_url = Column(String(300))
    duration = Column(Float)
    word_count = Column(Integer)
    error = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    generation_logs = relationship("GenerationLog", back_populates="topic", cascade="all, delete-orphan")

    def get_subtopics(self) -> List[str]:
        """Return subtopics as a Python list."""
        if self.subtopics:
            try:
                return json.loads(self.subtopics)
            except json.JSONDecodeError:
                return []
        return []

    def get_key_concepts(self) -> List[str]:
        """Return key concepts as a Python list."""
        if self.key_concepts:
            try:
                return json.loads(self.key_concepts)
            except json.JSONDecodeError:
                return []
        return []

    def get_previous_year_refs(self) -> List[str]:
        """Return PYQ refs as a Python list."""
        if self.previous_year_refs:
            try:
                return json.loads(self.previous_year_refs)
            except json.JSONDecodeError:
                return []
        return []

    def __repr__(self):
        return f"<Topic(part={self.part_number}, title='{self.title}', status='{self.status}')>"


class GenerationLog(Base):
    """Tracks each step of video generation pipeline."""
    __tablename__ = 'generation_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=False)
    part_number = Column(Integer, nullable=False)
    step = Column(String(50), nullable=False)          # script/audio/avatar/video/thumbnail/pdf/upload
    status = Column(String(20), default='running')     # running/completed/failed
    details = Column(Text)                             # JSON with step-specific info
    error = Column(Text)

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    topic = relationship("Topic", back_populates="generation_logs")

    def get_details(self) -> Dict[str, Any]:
        """Return details as a Python dict."""
        if self.details:
            try:
                return json.loads(self.details)
            except json.JSONDecodeError:
                return {}
        return {}

    def __repr__(self):
        return f"<GenerationLog(part={self.part_number}, step='{self.step}', status='{self.status}')>"


class Database:
    """Database manager for history video generation tracking."""

    def __init__(self, db_path: str = "data/history_tracker.db"):
        """Initialize database connection."""
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine and session
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        logger.info(f"Database initialized at {db_path}")

    def get_session(self):
        """Get a new database session."""
        return self.Session()

    # === Topic Operations ===

    def add_topic(self, topic_data: Dict[str, Any]) -> Topic:
        """Add a new topic to the database."""
        session = self.get_session()
        try:
            topic = Topic(
                part_number=topic_data['part_number'],
                era=topic_data['era'],
                era_hindi=topic_data.get('era_hindi', ''),
                section=topic_data['section'],
                title=topic_data['title'],
                subtopics=json.dumps(topic_data.get('subtopics', []), ensure_ascii=False),
                key_concepts=json.dumps(topic_data.get('key_concepts', []), ensure_ascii=False),
                exam_focus=topic_data.get('exam_focus', 'BOTH'),
                previous_year_refs=json.dumps(topic_data.get('previous_year_refs', []), ensure_ascii=False),
                color_theme=topic_data.get('color_theme', ''),
                status='pending'
            )
            session.add(topic)
            session.commit()
            session.refresh(topic)
            session.expunge(topic)
            make_transient(topic)
            logger.debug(f"Added topic: Part {topic.part_number} - {topic.title}")
            return topic
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add topic: {e}")
            raise
        finally:
            session.close()

    def get_topic_by_part(self, part_number: int) -> Optional[Topic]:
        """Get a topic by its part number."""
        session = self.get_session()
        try:
            topic = session.query(Topic).filter(Topic.part_number == part_number).first()
            if topic:
                session.expunge(topic)
                make_transient(topic)
            return topic
        finally:
            session.close()

    def get_next_pending_topic(self) -> Optional[Topic]:
        """Get the next pending topic in sequence."""
        session = self.get_session()
        try:
            topic = session.query(Topic).filter(
                Topic.status == 'pending'
            ).order_by(Topic.part_number).first()
            if topic:
                session.expunge(topic)
                make_transient(topic)
            return topic
        finally:
            session.close()

    def get_all_topics(self, status: str = None) -> List[Topic]:
        """Get all topics, optionally filtered by status."""
        session = self.get_session()
        try:
            query = session.query(Topic)
            if status:
                query = query.filter(Topic.status == status)
            topics = query.order_by(Topic.part_number).all()
            for t in topics:
                session.expunge(t)
                make_transient(t)
            return topics
        finally:
            session.close()

    def update_topic_status(self, part_number: int, status: str, **kwargs) -> bool:
        """Update topic status and optional fields."""
        session = self.get_session()
        try:
            topic = session.query(Topic).filter(Topic.part_number == part_number).first()
            if not topic:
                logger.warning(f"Topic not found: Part {part_number}")
                return False

            topic.status = status

            if status == 'generating':
                topic.started_at = datetime.utcnow()
            elif status == 'completed':
                topic.completed_at = datetime.utcnow()

            # Update optional fields
            for key, value in kwargs.items():
                if hasattr(topic, key):
                    setattr(topic, key, value)

            session.commit()
            logger.info(f"Updated topic Part {part_number} status to '{status}'")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update topic: {e}")
            return False
        finally:
            session.close()

    def mark_topic_completed(self, part_number: int, youtube_id: str = None,
                              youtube_url: str = None, video_path: str = None,
                              duration: float = None, **kwargs) -> bool:
        """Mark a topic as completed with upload details."""
        return self.update_topic_status(
            part_number, 'completed',
            youtube_id=youtube_id,
            youtube_url=youtube_url,
            video_path=video_path,
            duration=duration,
            **kwargs
        )

    def mark_topic_failed(self, part_number: int, error: str) -> bool:
        """Mark a topic as failed with error details."""
        return self.update_topic_status(part_number, 'failed', error=error)

    def get_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics."""
        session = self.get_session()
        try:
            total = session.query(Topic).count()
            completed = session.query(Topic).filter(Topic.status == 'completed').count()
            failed = session.query(Topic).filter(Topic.status == 'failed').count()
            pending = session.query(Topic).filter(Topic.status == 'pending').count()
            generating = session.query(Topic).filter(Topic.status == 'generating').count()

            # Determine current era
            next_topic = self.get_next_pending_topic()
            current_era = next_topic.era if next_topic else "Completed"

            return {
                'total': total,
                'completed': completed,
                'failed': failed,
                'pending': pending,
                'generating': generating,
                'percentage': round((completed / total * 100), 1) if total > 0 else 0,
                'current_era': current_era,
                'current_part': next_topic.part_number if next_topic else total,
            }
        finally:
            session.close()

    def topic_exists(self, part_number: int) -> bool:
        """Check if a topic exists in the database."""
        session = self.get_session()
        try:
            return session.query(Topic).filter(Topic.part_number == part_number).count() > 0
        finally:
            session.close()

    def get_topic_count(self) -> int:
        """Get total number of topics in database."""
        session = self.get_session()
        try:
            return session.query(Topic).count()
        finally:
            session.close()

    # === Generation Log Operations ===

    def log_step_start(self, part_number: int, step: str) -> Optional[GenerationLog]:
        """Log the start of a generation step."""
        session = self.get_session()
        try:
            topic = session.query(Topic).filter(Topic.part_number == part_number).first()
            if not topic:
                logger.warning(f"Cannot log step: topic Part {part_number} not found")
                return None
            log = GenerationLog(
                topic_id=topic.id,
                part_number=part_number,
                step=step,
                status='running',
                started_at=datetime.utcnow()
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            logger.debug(f"Started step '{step}' for Part {part_number}")
            return log
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log step start: {e}")
            raise
        finally:
            session.close()

    def log_step_complete(self, log_id: int, details: Dict[str, Any] = None) -> bool:
        """Log the completion of a generation step."""
        session = self.get_session()
        try:
            log = session.query(GenerationLog).filter(GenerationLog.id == log_id).first()
            if log:
                log.status = 'completed'
                log.completed_at = datetime.utcnow()
                if details:
                    log.details = json.dumps(details, ensure_ascii=False)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log step completion: {e}")
            return False
        finally:
            session.close()

    def log_step_failure(self, log_id: int, error: str) -> bool:
        """Log the failure of a generation step."""
        session = self.get_session()
        try:
            log = session.query(GenerationLog).filter(GenerationLog.id == log_id).first()
            if log:
                log.status = 'failed'
                log.error = error
                log.completed_at = datetime.utcnow()
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log step failure: {e}")
            return False
        finally:
            session.close()

    def get_generation_logs(self, part_number: int) -> List[GenerationLog]:
        """Get all generation logs for a specific part."""
        session = self.get_session()
        try:
            return session.query(GenerationLog).filter(
                GenerationLog.part_number == part_number
            ).order_by(GenerationLog.started_at).all()
        finally:
            session.close()

    # === Reset Operations ===

    def reset_failed_topics(self) -> int:
        """Reset all failed topics back to pending status."""
        session = self.get_session()
        try:
            count = session.query(Topic).filter(
                Topic.status == 'failed'
            ).update({'status': 'pending', 'error': None, 'started_at': None})
            session.commit()
            logger.info(f"Reset {count} failed topics to pending")
            return count
        finally:
            session.close()

    def reset_generating_topics(self) -> int:
        """Reset stuck 'generating' topics back to pending."""
        session = self.get_session()
        try:
            count = session.query(Topic).filter(
                Topic.status == 'generating'
            ).update({'status': 'pending', 'started_at': None})
            session.commit()
            logger.info(f"Reset {count} stuck topics to pending")
            return count
        finally:
            session.close()
