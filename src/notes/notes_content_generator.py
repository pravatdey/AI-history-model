"""
Enhanced Notes Content Generator - Uses LLM to generate comprehensive 10-15 page study notes.

Generates UPSC-quality detailed notes with:
- Full topic coverage with sub-sections
- Historical context, causes, effects
- Key provisions, judgements, timelines
- Challenges & suggestions
- Comparison tables
- Key facts & data points
- Important terms glossary
- 20 Prelims MCQs + 5 Mains descriptive questions
- Quick revision section

ROBUST: If any section fails, it uses fallback content and NEVER stops note generation.
"""

import re
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.script_generator.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes for generated note content
# ---------------------------------------------------------------------------

@dataclass
class GeneratedSection:
    """A detailed section of notes."""
    heading: str
    content: str  # paragraph text
    bullet_points: List[str] = field(default_factory=list)
    sub_sections: List[Dict[str, Any]] = field(default_factory=list)
    # sub_sections: [{'heading': str, 'points': [str], 'sub_points': {str: [str]}}]


@dataclass
class GeneratedNoteContent:
    """Complete generated content for one topic's notes."""
    title: str
    era: str
    section: str
    trigger_line: str
    what_is_it: str
    historical_background: str
    sections: List[GeneratedSection] = field(default_factory=list)
    key_provisions: List[str] = field(default_factory=list)
    causes_and_effects: Dict[str, List[str]] = field(default_factory=dict)
    timeline: List[Dict[str, str]] = field(default_factory=list)  # [{year, event}]
    key_judgements: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    comparison_table: Optional[Dict] = None
    key_facts_box: List[str] = field(default_factory=list)
    important_terms: Dict[str, str] = field(default_factory=dict)
    prelims_mcqs: List[Dict[str, Any]] = field(default_factory=list)
    # [{question, options: [a,b,c,d], answer: str, explanation: str}]
    mains_questions: List[Dict[str, str]] = field(default_factory=list)
    # [{question, hint: str}]
    quick_revision_points: List[str] = field(default_factory=list)
    additional_reading: List[str] = field(default_factory=list)
    upsc_tags: str = ""
    exam_focus: str = "BOTH"


class NotesContentGenerator:
    """
    Generates comprehensive, detailed study note content via LLM.

    Design principle: NEVER fail. If any LLM call fails, use fallback content
    derived from the script data and topic metadata. The note MUST always be generated.
    """

    def __init__(self, llm_config: Dict[str, Any] = None):
        """Initialize with LLM client."""
        config = llm_config or {}
        try:
            self.llm = LLMClient(
                provider=config.get('provider', 'groq'),
                groq_api_key=config.get('groq', {}).get('api_key', ''),
                groq_model=config.get('groq', {}).get('model', 'llama-3.3-70b-versatile'),
                ollama_host=config.get('ollama', {}).get('host', 'http://localhost:11434'),
                ollama_model=config.get('ollama', {}).get('model', 'llama2'),
            )
            logger.info("NotesContentGenerator LLM initialized")
        except Exception as e:
            logger.warning(f"LLM init failed: {e}. Will use fallback content only.")
            self.llm = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_comprehensive_notes(
        self,
        topic_title: str,
        era: str,
        section: str,
        subtopics: List[str],
        key_concepts: List[str],
        exam_focus: str,
        script_text: str = "",
        script_segments: List[Dict] = None,
        existing_key_points: List[str] = None,
        existing_terms: Dict[str, str] = None,
        existing_questions: List[str] = None,
    ) -> GeneratedNoteContent:
        """
        Generate comprehensive 10-15 page study note content.

        This method NEVER raises exceptions. All errors are handled internally
        with fallback content generation.
        """
        logger.info(f"Generating comprehensive notes for: {topic_title}")

        note = GeneratedNoteContent(
            title=topic_title,
            era=era,
            section=section,
            exam_focus=exam_focus,
            upsc_tags=f"History | {era} | {section} | {exam_focus}",
            trigger_line="",
            what_is_it="",
            historical_background="",
        )

        # Step 1: Generate detailed content sections via LLM
        self._generate_main_content(
            note, subtopics, key_concepts, script_text, script_segments or []
        )

        # Step 2: Generate trigger line and overview
        self._generate_overview(note, subtopics, key_concepts, script_text)

        # Step 3: Generate timeline
        self._generate_timeline(note, subtopics, script_text)

        # Step 4: Generate challenges & suggestions / significance
        self._generate_analysis(note, subtopics, script_text)

        # Step 5: Generate comparison table (if applicable)
        self._generate_comparison(note, subtopics, script_text)

        # Step 6: Generate key facts box
        self._generate_key_facts(note, subtopics, key_concepts, script_text)

        # Step 7: Generate important terms (enhanced from existing)
        self._generate_terms(
            note, subtopics, key_concepts, script_text,
            existing_terms or {}
        )

        # Step 8: Generate 20 Prelims MCQs
        self._generate_prelims_mcqs(
            note, subtopics, key_concepts, script_text,
            existing_questions or []
        )

        # Step 9: Generate 5 Mains questions
        self._generate_mains_questions(
            note, subtopics, key_concepts, script_text
        )

        # Step 10: Generate quick revision points
        self._generate_revision_points(note, subtopics, key_concepts)

        # Merge any existing data
        if existing_key_points:
            for kp in existing_key_points:
                if kp not in note.key_facts_box:
                    note.key_facts_box.append(kp)

        logger.info(
            f"Notes generated: {len(note.sections)} sections, "
            f"{len(note.important_terms)} terms, "
            f"{len(note.prelims_mcqs)} MCQs, "
            f"{len(note.mains_questions)} Mains Qs"
        )
        return note

    # ------------------------------------------------------------------
    # LLM call with forced fallback
    # ------------------------------------------------------------------

    def _llm_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2000,
        temperature: float = 0.4
    ) -> str:
        """Call LLM with full error handling. Returns empty string on failure."""
        if self.llm is None:
            return ""
        try:
            result = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result.strip() if result else ""
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Step 1: Main content sections
    # ------------------------------------------------------------------

    def _generate_main_content(
        self,
        note: GeneratedNoteContent,
        subtopics: List[str],
        key_concepts: List[str],
        script_text: str,
        script_segments: List[Dict],
    ):
        """Generate detailed content sections. Each subtopic becomes a section."""
        system = (
            "You are an expert Indian History professor writing comprehensive UPSC study notes. "
            "Write detailed, factual, exam-oriented content. Include dates, names, places, "
            "cause-effect relationships. Content should be dense and informative."
        )

        for i, subtopic in enumerate(subtopics):
            try:
                # Find relevant script content for this subtopic
                relevant_script = self._find_relevant_content(subtopic, script_text, script_segments)

                prompt = f"""Write detailed study notes for this history subtopic for UPSC preparation.

MAIN TOPIC: {note.title}
ERA: {note.era}
SUBTOPIC: {subtopic}
KEY CONCEPTS: {', '.join(key_concepts)}

{f'REFERENCE CONTENT:{chr(10)}{relevant_script[:2000]}' if relevant_script else ''}

Write comprehensive notes covering:
1. Introduction/Background of this subtopic (2-3 sentences)
2. Main content with 6-8 detailed bullet points covering key facts, dates, names, places
3. Each bullet point should be 2-3 lines with specific details
4. Include sub-points where needed for deeper explanation
5. Highlight UPSC-relevant aspects

OUTPUT FORMAT:
INTRO: [2-3 sentence introduction]
POINTS:
- Point 1 text here (with specific dates, names, details)
  * Sub-point if needed
  * Another sub-point
- Point 2 text here
...

Write at least 400 words of content for this subtopic."""

                response = self._llm_generate(prompt, system, max_tokens=2500, temperature=0.5)

                if response:
                    section = self._parse_section_response(subtopic, response)
                else:
                    section = self._fallback_section(subtopic, relevant_script)

                note.sections.append(section)

            except Exception as e:
                logger.warning(f"Section generation failed for '{subtopic}': {e}")
                note.sections.append(self._fallback_section(
                    subtopic, self._find_relevant_content(subtopic, script_text, script_segments)
                ))

        # If no sections were generated at all, create from script segments
        if not note.sections and script_segments:
            for seg in script_segments:
                if seg.get('content') and len(seg.get('content', '')) > 50:
                    note.sections.append(GeneratedSection(
                        heading=seg.get('title', 'Content'),
                        content=seg['content'][:500],
                        bullet_points=seg.get('key_points', []),
                    ))

    def _parse_section_response(self, heading: str, response: str) -> GeneratedSection:
        """Parse LLM response into a GeneratedSection."""
        intro = ""
        bullets = []
        sub_sections = []

        lines = response.split('\n')
        current_bullets = []
        current_sub_points = {}
        last_main_point = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('INTRO:'):
                intro = stripped.replace('INTRO:', '').strip()
            elif stripped.startswith('POINTS:'):
                continue
            elif stripped.startswith('- '):
                # Main bullet point
                point_text = stripped[2:].strip()
                if last_main_point and last_main_point not in current_sub_points:
                    current_sub_points[last_main_point] = []
                last_main_point = point_text
                current_bullets.append(point_text)
            elif stripped.startswith('* ') or stripped.startswith('  *'):
                # Sub-bullet
                sub_text = stripped.lstrip('* ').strip()
                if last_main_point:
                    if last_main_point not in current_sub_points:
                        current_sub_points[last_main_point] = []
                    current_sub_points[last_main_point].append(sub_text)
            elif not intro and len(stripped) > 30:
                # Treat as intro if we haven't found one yet
                intro = stripped

        # Build sub_sections structure
        if current_bullets:
            sub_sections = [{
                'heading': heading,
                'points': current_bullets,
                'sub_points': current_sub_points,
            }]

        return GeneratedSection(
            heading=heading,
            content=intro or f"Detailed study of {heading} - important for UPSC {response[:200] if not intro else ''}",
            bullet_points=current_bullets if not sub_sections else [],
            sub_sections=sub_sections,
        )

    def _fallback_section(self, heading: str, script_content: str = "") -> GeneratedSection:
        """Create fallback section from available content."""
        content = script_content[:500] if script_content else f"Study notes for {heading}."

        # Extract sentences as bullet points
        sentences = [s.strip() for s in re.split(r'[.!?]', content) if len(s.strip()) > 20]
        bullets = sentences[:6] if sentences else [f"Key aspects of {heading} for UPSC preparation"]

        return GeneratedSection(
            heading=heading,
            content=content[:300],
            bullet_points=bullets,
        )

    # ------------------------------------------------------------------
    # Step 2: Overview
    # ------------------------------------------------------------------

    def _generate_overview(
        self, note: GeneratedNoteContent,
        subtopics: List[str], key_concepts: List[str], script_text: str
    ):
        """Generate trigger line, what_is_it, and historical background."""
        prompt = f"""Generate an overview for UPSC study notes on this history topic.

TOPIC: {note.title}
ERA: {note.era}
SUBTOPICS: {', '.join(subtopics[:5])}
KEY CONCEPTS: {', '.join(key_concepts[:5])}

Provide:
TRIGGER: [One compelling sentence about why this topic matters for UPSC - max 30 words]
ABOUT: [3-4 sentence overview/definition of this topic - what is it, when, where, significance]
BACKGROUND: [4-5 sentence historical background/context setting the stage for this topic]"""

        response = self._llm_generate(prompt, max_tokens=800, temperature=0.4)

        if response:
            for line in response.split('\n'):
                stripped = line.strip()
                if stripped.startswith('TRIGGER:'):
                    note.trigger_line = stripped.replace('TRIGGER:', '').strip()
                elif stripped.startswith('ABOUT:'):
                    note.what_is_it = stripped.replace('ABOUT:', '').strip()
                elif stripped.startswith('BACKGROUND:'):
                    note.historical_background = stripped.replace('BACKGROUND:', '').strip()

        # Fallbacks
        if not note.trigger_line:
            note.trigger_line = (
                f"{note.title} is a crucial topic under {note.era} for UPSC {note.exam_focus} preparation."
            )
        if not note.what_is_it:
            note.what_is_it = (
                f"{note.title} covers {', '.join(subtopics[:3])}. "
                f"This topic falls under {note.section} in {note.era} and is important for "
                f"UPSC {note.exam_focus} examination."
            )
        if not note.historical_background:
            note.historical_background = (
                f"Understanding {note.title} requires knowledge of the broader context of {note.era}. "
                f"Key concepts include {', '.join(key_concepts[:4])}."
            )

    # ------------------------------------------------------------------
    # Step 3: Timeline
    # ------------------------------------------------------------------

    def _generate_timeline(
        self, note: GeneratedNoteContent,
        subtopics: List[str], script_text: str
    ):
        """Generate chronological timeline."""
        prompt = f"""Create a chronological timeline for this history topic.

TOPIC: {note.title}
ERA: {note.era}

List 8-12 important dates/events in chronological order.
FORMAT (one per line):
YEAR: event description

Only include factually accurate dates. If exact year is unknown, use approximate period (e.g., "c. 3300 BCE").
"""

        response = self._llm_generate(prompt, max_tokens=800, temperature=0.3)

        if response:
            for line in response.split('\n'):
                stripped = line.strip()
                if ':' in stripped and any(c.isdigit() for c in stripped.split(':')[0]):
                    parts = stripped.split(':', 1)
                    year = parts[0].strip().lstrip('- ')
                    event = parts[1].strip() if len(parts) > 1 else ""
                    if year and event:
                        note.timeline.append({'year': year, 'event': event})

        # Fallback: extract dates from script
        if not note.timeline and script_text:
            date_matches = re.findall(
                r'(\d{3,4}\s*(?:BCE|CE|AD|BC)?)\s*[-–:]\s*(.{10,80})',
                script_text, re.IGNORECASE
            )
            for year, event in date_matches[:8]:
                note.timeline.append({'year': year.strip(), 'event': event.strip()})

    # ------------------------------------------------------------------
    # Step 4: Analysis (challenges/significance)
    # ------------------------------------------------------------------

    def _generate_analysis(
        self, note: GeneratedNoteContent,
        subtopics: List[str], script_text: str
    ):
        """Generate challenges, significance, and way forward."""
        prompt = f"""For this history topic, provide analytical points for UPSC Mains preparation.

TOPIC: {note.title}
ERA: {note.era}

Provide:
SIGNIFICANCE:
- [6 points about historical significance, legacy, and impact]

CHALLENGES:
- [5 points about historiographical challenges, debates, or difficulties in studying this topic]

WAY_FORWARD:
- [5 points about how this topic connects to modern India, current relevance, lessons learned]

Each point should be 1-2 sentences with specific details."""

        response = self._llm_generate(prompt, max_tokens=1500, temperature=0.4)

        if response:
            current_section = None
            for line in response.split('\n'):
                stripped = line.strip()
                if 'SIGNIFICANCE' in stripped.upper():
                    current_section = 'significance'
                elif 'CHALLENGES' in stripped.upper() or 'CHALLENGE' in stripped.upper():
                    current_section = 'challenges'
                elif 'WAY_FORWARD' in stripped.upper() or 'WAY FORWARD' in stripped.upper():
                    current_section = 'suggestions'
                elif stripped.startswith('- ') or stripped.startswith('* '):
                    point = stripped.lstrip('-* ').strip()
                    if point and current_section:
                        if current_section == 'significance':
                            note.key_provisions.append(point)
                        elif current_section == 'challenges':
                            note.challenges.append(point)
                        elif current_section == 'suggestions':
                            note.suggestions.append(point)

        # Fallbacks
        if not note.key_provisions:
            note.key_provisions = [
                f"{note.title} is significant for understanding {note.era}",
                f"Important for UPSC {note.exam_focus} examination",
                f"Covers key aspects of {note.section}",
            ]
        if not note.challenges:
            note.challenges = [
                f"Limited archaeological/literary evidence for certain aspects",
                f"Historiographical debates about interpretation of sources",
                f"Need for multidisciplinary approach to understand the topic fully",
            ]
        if not note.suggestions:
            note.suggestions = [
                f"Study {note.title} in context of broader {note.era}",
                f"Focus on cause-effect relationships for Mains answers",
                f"Practice previous year questions related to this topic",
            ]

    # ------------------------------------------------------------------
    # Step 5: Comparison table
    # ------------------------------------------------------------------

    def _generate_comparison(
        self, note: GeneratedNoteContent,
        subtopics: List[str], script_text: str
    ):
        """Generate comparison table if topic has comparable elements."""
        prompt = f"""For this history topic, create a comparison table if applicable.

TOPIC: {note.title}
ERA: {note.era}
SUBTOPICS: {', '.join(subtopics[:5])}

If this topic has comparable elements (e.g., different civilizations, rulers, cultures, periods, features),
create a comparison table.

FORMAT:
HEADERS: Aspect | Item1 | Item2 | Item3
ROW: aspect_name | detail1 | detail2 | detail3
ROW: aspect_name | detail1 | detail2 | detail3
...

If no meaningful comparison exists, write: NO_COMPARISON

Create at least 5 rows if comparison is possible."""

        response = self._llm_generate(prompt, max_tokens=1000, temperature=0.3)

        if response and 'NO_COMPARISON' not in response.upper():
            headers = []
            rows = []
            for line in response.split('\n'):
                stripped = line.strip()
                if stripped.startswith('HEADERS:'):
                    headers = [h.strip() for h in stripped.replace('HEADERS:', '').split('|') if h.strip()]
                elif stripped.startswith('ROW:'):
                    row = [c.strip() for c in stripped.replace('ROW:', '').split('|') if c.strip()]
                    if row:
                        rows.append(row)

            if headers and rows:
                note.comparison_table = {
                    'headers': headers,
                    'rows': rows,
                }

    # ------------------------------------------------------------------
    # Step 6: Key facts
    # ------------------------------------------------------------------

    def _generate_key_facts(
        self, note: GeneratedNoteContent,
        subtopics: List[str], key_concepts: List[str], script_text: str
    ):
        """Generate key facts box."""
        prompt = f"""List 8-10 key facts about this history topic that UPSC aspirants must memorize.

TOPIC: {note.title}
ERA: {note.era}
KEY CONCEPTS: {', '.join(key_concepts[:5])}

Each fact should be a specific, exam-relevant piece of information:
- Include dates, names, places, numbers
- Format: FACT: [fact text]
- Each fact should be 1 line, concise but complete"""

        response = self._llm_generate(prompt, max_tokens=800, temperature=0.3)

        if response:
            for line in response.split('\n'):
                stripped = line.strip()
                if stripped.startswith('FACT:'):
                    fact = stripped.replace('FACT:', '').strip()
                    if fact:
                        note.key_facts_box.append(fact)
                elif stripped.startswith('- ') or stripped.startswith('* '):
                    fact = stripped.lstrip('-* ').strip()
                    if fact and len(fact) > 10:
                        note.key_facts_box.append(fact)

        # Fallback
        if not note.key_facts_box:
            note.key_facts_box = [
                f"{note.title} falls under {note.section} in {note.era}",
                f"Exam focus: {note.exam_focus}",
            ] + [f"Key concept: {c}" for c in key_concepts[:4]]

    # ------------------------------------------------------------------
    # Step 7: Important terms
    # ------------------------------------------------------------------

    def _generate_terms(
        self, note: GeneratedNoteContent,
        subtopics: List[str], key_concepts: List[str],
        script_text: str, existing_terms: Dict[str, str]
    ):
        """Generate comprehensive glossary of important terms."""
        # Start with existing terms
        note.important_terms = dict(existing_terms)

        prompt = f"""Generate a glossary of 12-15 important terms for this history topic.

TOPIC: {note.title}
ERA: {note.era}
KEY CONCEPTS: {', '.join(key_concepts)}
SUBTOPICS: {', '.join(subtopics[:5])}

For each term provide:
TERM: term_name | definition (10-20 words, exam-relevant)

Include:
- Historical terms, proper nouns
- Important places, archaeological sites
- Key personalities
- Technical/academic terms
- Government/administrative terms from the period"""

        response = self._llm_generate(prompt, max_tokens=1000, temperature=0.3)

        if response:
            for line in response.split('\n'):
                stripped = line.strip()
                if stripped.startswith('TERM:'):
                    content = stripped.replace('TERM:', '').strip()
                    if '|' in content:
                        parts = content.split('|', 1)
                        term = parts[0].strip()
                        defn = parts[1].strip()
                        if term and defn:
                            note.important_terms[term] = defn
                elif ':' in stripped and not stripped.startswith(('TOPIC', 'ERA', 'KEY', 'SUB')):
                    parts = stripped.split(':', 1)
                    term = parts[0].strip().lstrip('-*• ')
                    defn = parts[1].strip()
                    if term and defn and len(term) < 50 and len(defn) > 5:
                        note.important_terms[term] = defn

        # Ensure we have at least key concepts as terms
        for concept in key_concepts:
            if concept not in note.important_terms:
                note.important_terms[concept] = f"Important concept in {note.title}"

    # ------------------------------------------------------------------
    # Step 8: 20 Prelims MCQs
    # ------------------------------------------------------------------

    def _generate_prelims_mcqs(
        self, note: GeneratedNoteContent,
        subtopics: List[str], key_concepts: List[str],
        script_text: str, existing_questions: List[str]
    ):
        """Generate 20 Prelims-style MCQ questions."""
        # Generate in batches of 10 for better quality
        for batch in range(2):
            batch_num = batch + 1
            start_q = batch * 10 + 1

            prompt = f"""Generate 10 UPSC Prelims-style MCQ questions (Q{start_q}-Q{start_q+9}) on:

TOPIC: {note.title}
ERA: {note.era}
SUBTOPICS: {', '.join(subtopics)}
KEY CONCEPTS: {', '.join(key_concepts)}

{'Focus on factual questions about dates, places, events.' if batch == 0 else 'Focus on analytical questions about causes, effects, significance, and comparisons.'}

FORMAT for each question:
Q{start_q}: Question text
(a) Option A
(b) Option B
(c) Option C
(d) Option D
ANS: (correct letter)
EXP: Brief explanation (1 sentence)

RULES:
- Questions should be factually accurate
- Include 4 plausible options
- Mix easy, medium, and hard difficulty
- Cover different subtopics
- Use "Consider the following statements" format for 2-3 questions
- Use "Which of the following is/are correct" format for 2-3 questions"""

            response = self._llm_generate(
                prompt,
                system_prompt="You are a UPSC question paper setter. Create accurate, exam-standard questions.",
                max_tokens=3000,
                temperature=0.5,
            )

            if response:
                self._parse_mcqs(note, response)

        # Top up to 20 if needed using existing questions or templates
        self._topup_prelims_mcqs(note, subtopics, key_concepts, existing_questions)

    def _parse_mcqs(self, note: GeneratedNoteContent, response: str):
        """Parse MCQ response into structured format."""
        lines = response.split('\n')
        current_q = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Question line
            q_match = re.match(r'Q\d+[.:]\s*(.*)', stripped)
            if q_match:
                if current_q and current_q.get('question'):
                    note.prelims_mcqs.append(current_q)
                current_q = {
                    'question': q_match.group(1).strip(),
                    'options': [],
                    'answer': '',
                    'explanation': '',
                }
                continue

            if current_q is None:
                continue

            # Options
            opt_match = re.match(r'\(([a-d])\)\s*(.*)', stripped)
            if opt_match:
                current_q['options'].append(opt_match.group(2).strip())
                continue

            # Answer
            if stripped.upper().startswith('ANS:') or stripped.upper().startswith('ANSWER:'):
                ans = stripped.split(':', 1)[1].strip()
                # Extract just the letter
                letter_match = re.search(r'\(([a-d])\)', ans)
                if letter_match:
                    current_q['answer'] = letter_match.group(1)
                elif ans and ans[0].lower() in 'abcd':
                    current_q['answer'] = ans[0].lower()
                continue

            # Explanation
            if stripped.upper().startswith('EXP:') or stripped.upper().startswith('EXPLANATION:'):
                current_q['explanation'] = stripped.split(':', 1)[1].strip()
                continue

        # Don't forget last question
        if current_q and current_q.get('question'):
            note.prelims_mcqs.append(current_q)

    def _topup_prelims_mcqs(
        self, note: GeneratedNoteContent,
        subtopics: List[str], key_concepts: List[str],
        existing_questions: List[str]
    ):
        """Top up MCQs to reach 20."""
        needed = 20 - len(note.prelims_mcqs)
        if needed <= 0:
            return

        templates = [
            ("Consider the following statements about {topic}:\n1. Statement A\n2. Statement B\n"
             "Which of the above is/are correct?",
             ["1 only", "2 only", "Both 1 and 2", "Neither 1 nor 2"]),
            ("Which of the following is associated with {topic}?",
             ["Option A", "Option B", "Option C", "Option D"]),
            ("{topic} is significant because:",
             ["Reason A", "Reason B", "Reason C", "All of the above"]),
            ("The correct chronological order related to {topic} is:",
             ["A, B, C, D", "B, A, D, C", "C, A, B, D", "D, C, B, A"]),
        ]

        items = subtopics + key_concepts
        for i in range(needed):
            item = items[i % len(items)] if items else note.title
            tmpl_q, tmpl_opts = templates[i % len(templates)]
            note.prelims_mcqs.append({
                'question': tmpl_q.format(topic=item),
                'options': tmpl_opts,
                'answer': 'a',
                'explanation': f'Refer to the notes on {item} for detailed explanation.',
            })

    # ------------------------------------------------------------------
    # Step 9: 5 Mains questions
    # ------------------------------------------------------------------

    def _generate_mains_questions(
        self, note: GeneratedNoteContent,
        subtopics: List[str], key_concepts: List[str], script_text: str
    ):
        """Generate 5 UPSC Mains-style descriptive questions."""
        prompt = f"""Generate 5 UPSC Mains GS1 (History) style descriptive questions on:

TOPIC: {note.title}
ERA: {note.era}
SUBTOPICS: {', '.join(subtopics)}

FORMAT for each:
MAINS_Q: Question text (analytical, 150-250 word answer expected)
HINT: Brief hint on how to approach the answer (key points to cover)

TYPES to include:
1. One "Discuss" type question
2. One "Critically examine/analyze" question
3. One "Compare and contrast" question
4. One "Evaluate the significance" question
5. One "Comment on the statement" type question

Questions should test analytical thinking, not just factual recall."""

        response = self._llm_generate(
            prompt,
            system_prompt="You are a UPSC Mains question paper setter for GS Paper 1 (History).",
            max_tokens=1500,
            temperature=0.5,
        )

        if response:
            current_q = None
            for line in response.split('\n'):
                stripped = line.strip()
                if stripped.startswith('MAINS_Q:'):
                    if current_q:
                        note.mains_questions.append(current_q)
                    current_q = {
                        'question': stripped.replace('MAINS_Q:', '').strip(),
                        'hint': '',
                    }
                elif stripped.startswith('HINT:') and current_q:
                    current_q['hint'] = stripped.replace('HINT:', '').strip()

            if current_q:
                note.mains_questions.append(current_q)

        # Top up to 5
        needed = 5 - len(note.mains_questions)
        mains_templates = [
            "Discuss the historical significance of {topic} in the context of {era}. (250 words)",
            "Critically examine the impact of {topic} on Indian society and culture. (250 words)",
            "Evaluate the role of {topic} in shaping the course of {era}. (250 words)",
            "Compare and contrast the key features of {topic} with contemporary developments. (250 words)",
            "\"Understanding {topic} is essential for understanding {era}.\" Comment. (250 words)",
        ]
        for i in range(needed):
            tmpl = mains_templates[(len(note.mains_questions) + i) % len(mains_templates)]
            note.mains_questions.append({
                'question': tmpl.format(topic=note.title, era=note.era),
                'hint': f"Cover key facts, analysis, and significance of {note.title}.",
            })

    # ------------------------------------------------------------------
    # Step 10: Quick revision points
    # ------------------------------------------------------------------

    def _generate_revision_points(
        self, note: GeneratedNoteContent,
        subtopics: List[str], key_concepts: List[str]
    ):
        """Generate 15-20 quick revision one-liners."""
        prompt = f"""Generate 15-20 quick revision one-liners for last-minute exam preparation.

TOPIC: {note.title}
ERA: {note.era}
SUBTOPICS: {', '.join(subtopics)}
KEY CONCEPTS: {', '.join(key_concepts)}

Each point should be:
- One line only
- Include a specific fact (date, name, place, or number)
- Easy to memorize
- Exam-relevant

FORMAT:
- fact text here
- fact text here
..."""

        response = self._llm_generate(prompt, max_tokens=1000, temperature=0.3)

        if response:
            for line in response.split('\n'):
                stripped = line.strip()
                if stripped.startswith('- ') or stripped.startswith('* '):
                    point = stripped.lstrip('-* ').strip()
                    if point and len(point) > 10:
                        note.quick_revision_points.append(point)

        # Fallback from key_facts and terms
        if len(note.quick_revision_points) < 10:
            for fact in note.key_facts_box:
                if fact not in note.quick_revision_points:
                    note.quick_revision_points.append(fact)
            for term, defn in list(note.important_terms.items())[:5]:
                point = f"{term}: {defn}"
                if point not in note.quick_revision_points:
                    note.quick_revision_points.append(point)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_relevant_content(
        self, subtopic: str, script_text: str, script_segments: List[Dict]
    ) -> str:
        """Find relevant content for a subtopic from script data."""
        # Try to match from segments first
        subtopic_lower = subtopic.lower()
        for seg in script_segments:
            title = seg.get('title', '').lower()
            content = seg.get('content', '')
            if subtopic_lower in title or any(
                word in title for word in subtopic_lower.split()[:3]
            ):
                return content

        # Search in full script text
        if script_text:
            # Find paragraph containing subtopic keywords
            keywords = subtopic_lower.split()[:3]
            paragraphs = script_text.split('\n\n')
            for para in paragraphs:
                if any(kw in para.lower() for kw in keywords):
                    return para

        return ""
