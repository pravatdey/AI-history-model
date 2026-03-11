"""
Topic Planner for AI History Video Generator.
Plans lesson content, determines word budgets, and handles multi-part topics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.utils.database import Topic
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LessonSection:
    """A section within a lesson."""
    name: str                        # e.g., "introduction", "main_content", "exam_corner", "summary"
    title: str                       # Display title
    word_budget: int                 # Target word count for this section
    duration_minutes: float          # Target duration
    content_hints: List[str] = field(default_factory=list)


@dataclass
class LessonPlan:
    """Complete plan for a single lesson/video."""
    part_number: int
    title: str
    era: str
    section: str
    subtopics: List[str]
    key_concepts: List[str]
    exam_focus: str
    previous_year_refs: List[str]
    color_theme: str

    # Lesson structure
    total_word_budget: int
    total_duration_minutes: float
    sections: List[LessonSection]

    # Context from adjacent parts
    previous_topic_title: Optional[str] = None
    previous_topic_summary: Optional[str] = None
    next_topic_title: Optional[str] = None
    needs_recap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'part_number': self.part_number,
            'title': self.title,
            'era': self.era,
            'section': self.section,
            'subtopics': self.subtopics,
            'key_concepts': self.key_concepts,
            'exam_focus': self.exam_focus,
            'previous_year_refs': self.previous_year_refs,
            'color_theme': self.color_theme,
            'total_word_budget': self.total_word_budget,
            'total_duration_minutes': self.total_duration_minutes,
            'previous_topic_title': self.previous_topic_title,
            'next_topic_title': self.next_topic_title,
            'needs_recap': self.needs_recap,
            'sections': [
                {
                    'name': s.name,
                    'title': s.title,
                    'word_budget': s.word_budget,
                    'duration_minutes': s.duration_minutes,
                    'content_hints': s.content_hints,
                }
                for s in self.sections
            ],
        }


class TopicPlanner:
    """Plans lesson structure and content allocation for each topic."""

    # Default configuration
    DEFAULT_DURATION_MINUTES = 20
    DEFAULT_WPM = 140                    # Words per minute for educational content

    # Lesson structure percentages
    INTRO_PCT = 0.10                     # 10% - ~2 minutes
    MAIN_CONTENT_PCT = 0.75             # 75% - ~15 minutes
    EXAM_CORNER_PCT = 0.10              # 10% - ~2 minutes
    SUMMARY_PCT = 0.05                   # 5% - ~1 minute

    def __init__(self, target_duration: float = None, words_per_minute: int = None):
        """
        Initialize TopicPlanner.

        Args:
            target_duration: Target video duration in minutes.
            words_per_minute: Speaking rate for word budget calculation.
        """
        self.target_duration = target_duration or self.DEFAULT_DURATION_MINUTES
        self.wpm = words_per_minute or self.DEFAULT_WPM
        self.total_word_budget = int(self.target_duration * self.wpm)

        logger.info(
            f"TopicPlanner initialized: {self.target_duration}min, "
            f"{self.wpm}WPM, {self.total_word_budget} words budget"
        )

    def get_word_budget(self) -> int:
        """Get total word budget for a lesson."""
        return self.total_word_budget

    def plan_lesson(self, topic: Topic,
                    previous_topic: Optional[Topic] = None,
                    next_topic: Optional[Topic] = None) -> LessonPlan:
        """
        Create a complete lesson plan for a topic.

        Args:
            topic: The topic to plan.
            previous_topic: Previous topic (for recap).
            next_topic: Next topic (for preview in outro).

        Returns:
            LessonPlan with structured sections and word budgets.
        """
        subtopics = topic.get_subtopics()
        key_concepts = topic.get_key_concepts()
        pyq_refs = topic.get_previous_year_refs()

        # Determine if recap is needed
        needs_recap = self._should_recap(topic, previous_topic)

        # Adjust word budgets if recap needed
        if needs_recap:
            recap_words = int(self.total_word_budget * 0.05)  # 5% for recap
            remaining = self.total_word_budget - recap_words
        else:
            recap_words = 0
            remaining = self.total_word_budget

        # Calculate section budgets
        intro_words = int(remaining * self.INTRO_PCT)
        main_words = int(remaining * self.MAIN_CONTENT_PCT)
        exam_words = int(remaining * self.EXAM_CORNER_PCT)
        summary_words = int(remaining * self.SUMMARY_PCT)

        # Build sections
        sections = []

        if needs_recap:
            sections.append(LessonSection(
                name="recap",
                title="Quick Recap - Previous Class",
                word_budget=recap_words,
                duration_minutes=round(recap_words / self.wpm, 1),
                content_hints=[
                    f"Brief recap of Part {previous_topic.part_number}: {previous_topic.title}",
                    "Key takeaways from last class",
                    "Connection to today's topic",
                ]
            ))

        sections.append(LessonSection(
            name="introduction",
            title="Introduction",
            word_budget=intro_words,
            duration_minutes=round(intro_words / self.wpm, 1),
            content_hints=[
                f"Welcome to Part {topic.part_number} of History Course",
                f"Today's topic: {topic.title}",
                f"Era: {topic.era}",
                f"Why this topic matters for {topic.exam_focus} exam",
                "What we'll cover today",
            ]
        ))

        # Split main content across subtopics
        if subtopics:
            words_per_subtopic = main_words // len(subtopics)
            for i, subtopic in enumerate(subtopics):
                sections.append(LessonSection(
                    name=f"main_content_{i+1}",
                    title=f"Section {i+1}: {subtopic[:50]}",
                    word_budget=words_per_subtopic,
                    duration_minutes=round(words_per_subtopic / self.wpm, 1),
                    content_hints=[
                        subtopic,
                        f"Key concepts: {', '.join(key_concepts[i:i+1]) if i < len(key_concepts) else ''}",
                        "Include dates, names, cause-effect",
                        "Connect to exam relevance",
                    ]
                ))
        else:
            sections.append(LessonSection(
                name="main_content",
                title="Main Content",
                word_budget=main_words,
                duration_minutes=round(main_words / self.wpm, 1),
                content_hints=[
                    f"Cover: {topic.title}",
                    f"Key concepts: {', '.join(key_concepts)}",
                ]
            ))

        sections.append(LessonSection(
            name="exam_corner",
            title="Exam Corner",
            word_budget=exam_words,
            duration_minutes=round(exam_words / self.wpm, 1),
            content_hints=[
                f"Exam focus: {topic.exam_focus}",
                "Previous year questions discussed",
                "Expected questions & pattern",
                "Mnemonics & memory tricks",
                f"PYQ refs: {', '.join(pyq_refs)}" if pyq_refs else "General exam tips",
            ]
        ))

        sections.append(LessonSection(
            name="summary",
            title="Summary & Key Takeaways",
            word_budget=summary_words,
            duration_minutes=round(summary_words / self.wpm, 1),
            content_hints=[
                "5 key takeaways from today's class",
                "Quick revision points",
                f"Next class preview: {next_topic.title}" if next_topic else "Course completion message",
                "Call to action - subscribe & share",
            ]
        ))

        plan = LessonPlan(
            part_number=topic.part_number,
            title=topic.title,
            era=topic.era,
            section=topic.section,
            subtopics=subtopics,
            key_concepts=key_concepts,
            exam_focus=topic.exam_focus,
            previous_year_refs=pyq_refs,
            color_theme=topic.color_theme,
            total_word_budget=self.total_word_budget,
            total_duration_minutes=self.target_duration,
            sections=sections,
            previous_topic_title=previous_topic.title if previous_topic else None,
            next_topic_title=next_topic.title if next_topic else None,
            needs_recap=needs_recap,
        )

        logger.info(
            f"Planned lesson: Part {plan.part_number} - {plan.title} "
            f"({len(sections)} sections, {plan.total_word_budget} words)"
        )

        return plan

    def _should_recap(self, topic: Topic, previous_topic: Optional[Topic]) -> bool:
        """
        Determine if the lesson should include a recap of the previous class.

        Recap is needed when:
        - Previous topic exists
        - Previous topic is in the same section (continuing a topic)
        - Part number > 1
        """
        if not previous_topic or topic.part_number <= 1:
            return False

        # Always recap if same section (topic continuation)
        if topic.section == previous_topic.section:
            return True

        # Recap if same era (provides continuity)
        if topic.era == previous_topic.era:
            return True

        # First topic of new era - recap the transition
        return False
