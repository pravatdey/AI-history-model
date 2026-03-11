"""
History Script Writer - Generates complete lesson scripts for history video classes.
Uses LLM (Groq/Ollama) to generate Hinglish educational content.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .llm_client import LLMClient
from .prompt_templates import HistoryPromptTemplates
from src.syllabus.topic_planner import TopicPlanner, LessonPlan
from src.utils.database import Topic
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScriptSegment:
    """A segment of the lesson script."""
    type: str                        # intro, recap, main_content, exam_corner, summary, subtopic
    content: str
    duration_estimate: float = 0.0   # in seconds
    title: str = ""
    # Slide data
    key_points: List[str] = field(default_factory=list)
    exam_relevance: str = ""         # PRELIMS, MAINS, BOTH
    subject_category: str = ""       # Ancient History, Medieval History, Modern History
    important_terms: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""              # Video timestamp marker


@dataclass
class LessonScript:
    """Complete lesson script with metadata."""
    title: str
    part_number: int
    era: str
    section: str
    date: str
    segments: List[ScriptSegment] = field(default_factory=list)
    total_duration: float = 0.0
    word_count: int = 0

    # Metadata
    key_points: List[str] = field(default_factory=list)
    important_terms: Dict[str, str] = field(default_factory=dict)
    practice_questions: List[str] = field(default_factory=list)
    exam_focus: str = ""
    color_theme: str = ""
    subtopics: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)

    # Adjacent topics
    previous_topic_title: Optional[str] = None
    next_topic_title: Optional[str] = None

    def get_full_script(self) -> str:
        """Get the full script as a single string."""
        parts = [seg.content for seg in self.segments]
        return "\n\n".join(parts)

    def get_script_for_tts(self) -> str:
        """Get script optimized for TTS (single continuous text)."""
        parts = []
        for segment in self.segments:
            content = segment.content
            if content and not content.rstrip().endswith(('.', '!', '?')):
                content = content.rstrip() + "."
            parts.append(content)
        return " ".join(parts)

    def get_all_key_points(self) -> List[Dict[str, Any]]:
        """Get all key points for slides and PDF notes."""
        all_points = []
        for segment in self.segments:
            if segment.key_points:
                all_points.append({
                    'section_title': segment.title,
                    'key_points': segment.key_points,
                    'exam_relevance': segment.exam_relevance,
                    'subject': segment.subject_category,
                    'timestamp': segment.timestamp,
                })
        return all_points

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for video composer compatibility."""
        return {
            'title': self.title,
            'part_number': self.part_number,
            'era': self.era,
            'section': self.section,
            'date': self.date,
            'total_duration': self.total_duration,
            'word_count': self.word_count,
            'exam_focus': self.exam_focus,
            'color_theme': self.color_theme,
            'subtopics': self.subtopics,
            'key_concepts': self.key_concepts,
            'previous_topic_title': self.previous_topic_title,
            'next_topic_title': self.next_topic_title,
            'segments': [
                {
                    'type': seg.type,
                    'content': seg.content,
                    'title': seg.title,
                    'key_points': seg.key_points,
                    'exam_relevance': seg.exam_relevance,
                    'subject_category': seg.subject_category,
                    'important_terms': seg.important_terms,
                    'timestamp': seg.timestamp,
                    'duration_estimate': seg.duration_estimate,
                }
                for seg in self.segments
            ],
            'subjects_covered': [self.era],
        }


class HistoryScriptWriter:
    """Generates history lesson scripts using LLM."""

    # Words per minute for duration estimation
    WORDS_PER_MINUTE = 140

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize HistoryScriptWriter.

        Args:
            config: Configuration dictionary (from settings.yaml).
        """
        self.config = config or {}
        llm_config = self.config.get('llm', {})

        # Initialize LLM client
        self.llm = LLMClient(
            provider=llm_config.get('provider', 'groq'),
            groq_api_key=llm_config.get('groq', {}).get('api_key', ''),
            groq_model=llm_config.get('groq', {}).get('model', 'llama-3.3-70b-versatile'),
            ollama_host=llm_config.get('ollama', {}).get('host', 'http://localhost:11434'),
            ollama_model=llm_config.get('ollama', {}).get('model', 'llama2'),
        )

        # Script settings
        script_config = self.config.get('script', {})
        self.wpm = script_config.get('words_per_minute', self.WORDS_PER_MINUTE)
        self.target_word_count = script_config.get('target_word_count', 2800)

        # Initialize topic planner
        self.planner = TopicPlanner(
            target_duration=self.config.get('video', {}).get('duration_target', 20),
            words_per_minute=self.wpm
        )

        # System prompt
        self.system_prompt = HistoryPromptTemplates.SYSTEM_PROMPT

        logger.info("HistoryScriptWriter initialized")

    def generate_lesson_script(
        self,
        topic: Topic,
        previous_topic: Optional[Topic] = None,
        next_topic: Optional[Topic] = None
    ) -> LessonScript:
        """
        Generate a complete lesson script for a topic.

        Args:
            topic: The topic to generate a script for.
            previous_topic: Previous topic (for recap).
            next_topic: Next topic (for outro preview).

        Returns:
            LessonScript with full narration and metadata.
        """
        logger.info(f"Generating script for Part {topic.part_number}: {topic.title}")

        # Create lesson plan
        plan = self.planner.plan_lesson(topic, previous_topic, next_topic)

        # Generate the main script via LLM
        raw_script = self._generate_raw_script(plan)

        # Parse script into segments
        segments = self._parse_script_into_segments(raw_script, topic)

        # Extract key points for slides
        self._extract_key_points_for_segments(segments, topic)

        # Extract important terms
        important_terms = self._extract_important_terms(raw_script)

        # Generate practice questions
        practice_questions = self._generate_practice_questions(topic, raw_script)

        # Calculate word count and duration
        word_count = sum(len(seg.content.split()) for seg in segments)
        total_duration = word_count / self.wpm * 60  # in seconds

        # Assign timestamps
        self._assign_timestamps(segments)

        # Create LessonScript
        script = LessonScript(
            title=topic.title,
            part_number=topic.part_number,
            era=topic.era,
            section=topic.section,
            date=datetime.now().strftime("%Y-%m-%d"),
            segments=segments,
            total_duration=total_duration,
            word_count=word_count,
            key_points=[kp for seg in segments for kp in seg.key_points],
            important_terms=important_terms,
            practice_questions=practice_questions,
            exam_focus=topic.exam_focus,
            color_theme=topic.color_theme,
            subtopics=topic.get_subtopics(),
            key_concepts=topic.get_key_concepts(),
            previous_topic_title=previous_topic.title if previous_topic else None,
            next_topic_title=next_topic.title if next_topic else None,
        )

        logger.info(
            f"Script generated: Part {topic.part_number} - "
            f"{word_count} words, ~{total_duration/60:.1f} minutes, "
            f"{len(segments)} segments"
        )

        return script

    def _generate_raw_script(self, plan: LessonPlan) -> str:
        """Generate raw script text via LLM."""
        prompt = HistoryPromptTemplates.get_lesson_script_prompt(
            part_number=plan.part_number,
            total_parts=180,
            title=plan.title,
            era=plan.era,
            section=plan.section,
            subtopics=plan.subtopics,
            key_concepts=plan.key_concepts,
            exam_focus=plan.exam_focus,
            previous_year_refs=plan.previous_year_refs,
            word_count=plan.total_word_budget,
            previous_topic_title=plan.previous_topic_title,
            next_topic_title=plan.next_topic_title,
            needs_recap=plan.needs_recap,
        )

        try:
            raw_script = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=plan.total_word_budget * 2,
                temperature=0.6
            )
            logger.info(f"Raw script generated: {len(raw_script.split())} words")
            return raw_script
        except Exception as e:
            logger.error(f"Failed to generate script: {e}")
            raise

    def _parse_script_into_segments(self, raw_script: str, topic: Topic) -> List[ScriptSegment]:
        """Parse raw LLM output into structured segments."""
        segments = []

        # Split by section markers
        section_pattern = r'\[SECTION:\s*(\w+)\]'
        subtopic_pattern = r'\[SUBTOPIC:\s*(.+?)\]'

        # Find all section markers
        parts = re.split(section_pattern, raw_script)

        current_type = "intro"
        current_content = ""
        subtopic_counter = 0

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Check if this is a section label
            if part.upper() in ('RECAP', 'INTRODUCTION', 'MAIN_CONTENT', 'EXAM_CORNER', 'SUMMARY'):
                # Save previous segment if any
                if current_content.strip():
                    segments.append(ScriptSegment(
                        type=current_type,
                        content=self._clean_script_text(current_content),
                        title=self._get_section_title(current_type, topic),
                        exam_relevance=topic.exam_focus,
                        subject_category=topic.era,
                    ))

                # Map section label to type
                type_map = {
                    'RECAP': 'recap',
                    'INTRODUCTION': 'intro',
                    'MAIN_CONTENT': 'main_content',
                    'EXAM_CORNER': 'exam_corner',
                    'SUMMARY': 'summary',
                }
                current_type = type_map.get(part.upper(), 'main_content')
                current_content = ""
            else:
                # Check for subtopic markers within content
                subtopic_parts = re.split(subtopic_pattern, part)

                for sp in subtopic_parts:
                    sp = sp.strip()
                    if not sp:
                        continue

                    # If this looks like a subtopic title (short text from the regex group)
                    if len(sp) < 100 and not sp.endswith('.'):
                        # Save current content
                        if current_content.strip():
                            segments.append(ScriptSegment(
                                type=current_type,
                                content=self._clean_script_text(current_content),
                                title=self._get_section_title(current_type, topic),
                                exam_relevance=topic.exam_focus,
                                subject_category=topic.era,
                            ))
                        subtopic_counter += 1
                        current_type = 'subtopic'
                        current_content = ""
                    else:
                        current_content += " " + sp

        # Don't forget the last segment
        if current_content.strip():
            segments.append(ScriptSegment(
                type=current_type,
                content=self._clean_script_text(current_content),
                title=self._get_section_title(current_type, topic),
                exam_relevance=topic.exam_focus,
                subject_category=topic.era,
            ))

        # If parsing failed (no sections found), create a single segment
        if not segments:
            logger.warning("Section markers not found, creating single segment")
            segments.append(ScriptSegment(
                type='main_content',
                content=self._clean_script_text(raw_script),
                title=topic.title,
                exam_relevance=topic.exam_focus,
                subject_category=topic.era,
            ))

        # Calculate duration estimates
        for seg in segments:
            word_count = len(seg.content.split())
            seg.duration_estimate = word_count / self.wpm * 60  # seconds

        logger.info(f"Parsed {len(segments)} segments from script")
        return segments

    def _extract_key_points_for_segments(self, segments: List[ScriptSegment], topic: Topic):
        """Extract key points for each major segment (for slides)."""
        for segment in segments:
            if segment.type in ('main_content', 'subtopic', 'exam_corner'):
                if len(segment.content) > 100:
                    try:
                        prompt = HistoryPromptTemplates.get_key_points_prompt(
                            segment.content, max_points=4
                        )
                        response = self.llm.generate(
                            prompt=prompt,
                            system_prompt="Extract key points concisely.",
                            max_tokens=300,
                            temperature=0.3
                        )
                        points = [
                            line.strip().lstrip('- •').strip()
                            for line in response.strip().split('\n')
                            if line.strip() and len(line.strip()) > 5
                        ]
                        segment.key_points = points[:4]
                    except Exception as e:
                        logger.warning(f"Failed to extract key points: {e}")
                        segment.key_points = []

    def _extract_important_terms(self, raw_script: str) -> Dict[str, str]:
        """Extract important historical terms from the script."""
        try:
            prompt = HistoryPromptTemplates.get_important_terms_prompt(raw_script, max_terms=6)
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="Extract important terms concisely.",
                max_tokens=400,
                temperature=0.3
            )

            terms = {}
            for line in response.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    term = parts[0].strip().lstrip('- •').strip()
                    definition = parts[1].strip()
                    if term and definition:
                        terms[term] = definition

            logger.info(f"Extracted {len(terms)} important terms")
            return terms
        except Exception as e:
            logger.warning(f"Failed to extract terms: {e}")
            return {}

    def _generate_practice_questions(self, topic: Topic, raw_script: str) -> List[str]:
        """Generate practice questions for PDF notes."""
        try:
            prompt = HistoryPromptTemplates.get_practice_questions_prompt(
                title=topic.title,
                content=raw_script,
                exam_focus=topic.exam_focus
            )
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="Generate exam-style practice questions.",
                max_tokens=1000,
                temperature=0.4
            )
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            logger.info(f"Generated {len(questions)} practice question lines")
            return questions
        except Exception as e:
            logger.warning(f"Failed to generate practice questions: {e}")
            return []

    def _assign_timestamps(self, segments: List[ScriptSegment]):
        """Assign video timestamps to each segment."""
        current_time = 0
        for segment in segments:
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            segment.timestamp = f"{minutes:02d}:{seconds:02d}"
            current_time += segment.duration_estimate

    def _clean_script_text(self, text: str) -> str:
        """Clean raw LLM output for TTS compatibility."""
        # Remove section/subtopic markers
        text = re.sub(r'\[SECTION:\s*\w+\]', '', text)
        text = re.sub(r'\[SUBTOPIC:\s*.+?\]', '', text)

        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'`(.+?)`', r'\1', text)

        # Remove bullet points & numbering at start of lines
        text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        text = text.strip()

        return text

    def _get_section_title(self, section_type: str, topic: Topic) -> str:
        """Get a display title for a section type."""
        titles = {
            'recap': f"Recap - Previous Class",
            'intro': f"Part {topic.part_number} - Introduction",
            'main_content': topic.title,
            'subtopic': topic.title,
            'exam_corner': "Exam Corner",
            'summary': "Summary & Key Takeaways",
        }
        return titles.get(section_type, topic.title)
