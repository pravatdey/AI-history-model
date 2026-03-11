"""
Prompt Templates for AI History Video Generator.
History-specific LLM prompts for Hinglish lesson script generation.
"""


class HistoryPromptTemplates:
    """Templates for generating history lesson scripts via LLM."""

    # === System Prompt ===
    SYSTEM_PROMPT = """You are an expert Indian History teacher preparing video lessons for UPSC and State PSC aspirants.

Teaching Style:
- You teach in Hinglish - naturally mixing Hindi and English like a top coaching class teacher
- Use Hindi for explanations, storytelling, and connecting with students
- Use English for historical terms, proper nouns, dates, and technical vocabulary
- Your tone is engaging, confident, and story-telling oriented
- You make history come alive with vivid descriptions and interesting anecdotes
- You always connect historical events to UPSC exam relevance

Content Guidelines:
- Always mention important dates, names, and places
- Explain cause-effect relationships clearly
- Use chronological flow within each topic
- Highlight facts that are frequently asked in UPSC Prelims
- Provide analytical angles for UPSC Mains
- Include mnemonics or memory tricks where helpful
- Reference previous year questions when relevant

Language Rules:
- Primary narration in Hindi (written in Devanagari-compatible Roman script)
- Historical terms always in English (e.g., "Indus Valley Civilization", "Mauryan Empire")
- Numbers and dates in English
- Mix naturally - like "Toh friends, Ashoka ne Kalinga War ke baad Buddhism adopt kiya"
- Keep sentences clear and suitable for spoken delivery (TTS-friendly)
- Avoid complex Hindi that might confuse TTS engine
- Use simple, commonly understood Hindi words"""

    # === Main Lesson Script Prompt ===
    @staticmethod
    def get_lesson_script_prompt(
        part_number: int,
        total_parts: int,
        title: str,
        era: str,
        section: str,
        subtopics: list,
        key_concepts: list,
        exam_focus: str,
        previous_year_refs: list,
        word_count: int = 2800,
        previous_topic_title: str = None,
        next_topic_title: str = None,
        needs_recap: bool = False
    ) -> str:
        """Generate prompt for a complete lesson script."""

        subtopics_text = "\n".join(f"  {i+1}. {st}" for i, st in enumerate(subtopics))
        concepts_text = ", ".join(key_concepts) if key_concepts else "See subtopics"
        pyq_text = "\n".join(f"  - {ref}" for ref in previous_year_refs) if previous_year_refs else "  - General exam relevance"

        recap_section = ""
        if needs_recap and previous_topic_title:
            recap_section = f"""
RECAP SECTION (~140 words, ~1 minute):
- Briefly recap previous class: Part {part_number - 1} - "{previous_topic_title}"
- Mention 2-3 key points from last class
- Connect it to today's topic naturally
"""

        next_preview = ""
        if next_topic_title:
            next_preview = f'Mention: "Agle class mein hum padhenge: {next_topic_title}"'
        else:
            next_preview = 'This is the final class! Give a motivational closing message.'

        prompt = f"""Generate a complete {word_count}-word video lesson script in Hinglish for a history class.

=== LESSON DETAILS ===
Part Number: {part_number} of {total_parts}
Title: {title}
Era: {era}
Section: {section}
Exam Focus: {exam_focus}

=== SUBTOPICS TO COVER ===
{subtopics_text}

=== KEY CONCEPTS ===
{concepts_text}

=== PREVIOUS YEAR QUESTIONS ===
{pyq_text}

=== SCRIPT STRUCTURE ===
{recap_section}
INTRODUCTION (~{int(word_count * 0.10)} words, ~2 minutes):
- Greet students: "Namaste friends! Aaj ki class mein..."
- Mention Part {part_number} of {total_parts}
- Introduce today's topic with an interesting hook
- Briefly tell what we'll cover in this class
- Mention exam relevance ({exam_focus})

MAIN CONTENT (~{int(word_count * 0.75)} words, ~15 minutes):
- Cover ALL subtopics listed above in order
- For each subtopic:
  * Start with context/background
  * Give key facts with dates and names
  * Explain cause and effect
  * Highlight UPSC-relevant points
  * Use examples and analogies
- Maintain chronological flow
- Use transition phrases between subtopics
- Include interesting historical anecdotes

EXAM CORNER (~{int(word_count * 0.10)} words, ~2 minutes):
- Discuss previous year questions related to this topic
- Highlight most likely exam questions
- Give 2-3 mnemonics or memory tricks
- Tips for Prelims MCQs and Mains answer writing

SUMMARY (~{int(word_count * 0.05)} words, ~1 minute):
- "Toh friends, aaj humne seekha..."
- List 5 key takeaways
- {next_preview}
- "Like, Share, Subscribe karna mat bhulna!"

=== OUTPUT FORMAT ===
Write the complete script as continuous narration text.
Mark section transitions with [SECTION: name] tags:
[SECTION: RECAP] (if applicable)
[SECTION: INTRODUCTION]
[SECTION: MAIN_CONTENT]
[SECTION: EXAM_CORNER]
[SECTION: SUMMARY]

Within MAIN_CONTENT, mark subtopic transitions:
[SUBTOPIC: subtopic title]

=== IMPORTANT RULES ===
1. Write exactly ~{word_count} words (±10%)
2. Use Hinglish throughout - Hindi narration with English terms
3. Every important fact should have a date or name attached
4. Don't use markdown formatting (no *, #, etc.)
5. Keep sentences short and TTS-friendly
6. Don't use emojis or special characters
7. Write numbers as digits (e.g., "1526" not "fifteen twenty-six")
8. Pronunciation guide for difficult names in brackets: "Chandragupta (CHUN-dra-GUP-ta)"
"""
        return prompt

    # === Key Points Extraction Prompt ===
    @staticmethod
    def get_key_points_prompt(content: str, max_points: int = 4) -> str:
        """Generate prompt to extract key points for video slides."""
        return f"""Extract exactly {max_points} key points from this history lesson content.

CONTENT:
{content[:3000]}

RULES:
1. Each point must be concise - maximum 12 words
2. Focus on exam-relevant facts
3. Include dates and names where possible
4. Points should be in English (for slide display)
5. Each point should cover a different aspect of the topic

OUTPUT FORMAT (one per line, no numbering):
point text here
point text here
point text here
point text here"""

    # === Important Terms Extraction Prompt ===
    @staticmethod
    def get_important_terms_prompt(content: str, max_terms: int = 6) -> str:
        """Generate prompt to extract important terms for badge display."""
        return f"""Extract important historical terms from this content with brief definitions.

CONTENT:
{content[:3000]}

RULES:
1. Extract up to {max_terms} terms
2. Each term should be a proper noun, concept, or technical term
3. Definition should be 5-10 words maximum
4. Terms should be in English
5. Focus on terms that appear in UPSC exams

OUTPUT FORMAT (term: definition, one per line):
Term Name: Brief definition here
Term Name: Brief definition here"""

    # === Practice Questions Prompt ===
    @staticmethod
    def get_practice_questions_prompt(title: str, content: str,
                                      exam_focus: str = "BOTH") -> str:
        """Generate prompt for practice questions."""
        return f"""Generate practice questions for UPSC aspirants based on this history topic.

TOPIC: {title}
EXAM FOCUS: {exam_focus}

CONTENT:
{content[:3000]}

Generate:
1. 3 PRELIMS-style MCQs (4 options each, mark correct answer)
2. 2 MAINS-style questions (analytical, 150-200 word expected answers)
3. 1 One-liner factual question

OUTPUT FORMAT:
[PRELIMS MCQ 1]
Q: Question text
(a) Option A
(b) Option B
(c) Option C
(d) Option D
Answer: (correct letter)

[PRELIMS MCQ 2]
...

[MAINS Q1]
Q: Question text (150-200 words)

[MAINS Q2]
...

[ONE-LINER]
Q: Factual question
A: Brief answer"""

    # === Slide Content Generation Prompt ===
    @staticmethod
    def get_slide_content_prompt(section_content: str, section_title: str,
                                  topic_number: int, exam_focus: str) -> str:
        """Generate prompt for creating slide content from a script section."""
        return f"""Create a presentation slide content from this lesson section.

SECTION TITLE: {section_title}
TOPIC NUMBER: {topic_number}
EXAM TAG: {exam_focus}

CONTENT:
{section_content[:2000]}

Generate:
1. SLIDE_TITLE: A concise title (max 8 words)
2. KEY_POINTS: Exactly 4 bullet points (max 12 words each)
3. TERMS: Up to 4 important terms with 3-word definitions

OUTPUT FORMAT:
SLIDE_TITLE: Title here
KEY_POINTS:
- Point 1
- Point 2
- Point 3
- Point 4
TERMS:
Term1: definition
Term2: definition
Term3: definition
Term4: definition"""

    # === Mains Answer Framework Prompt ===
    @staticmethod
    def get_mains_analysis_prompt(title: str, content: str) -> str:
        """Generate a Mains-style analytical framework."""
        return f"""Create a UPSC Mains answer framework for this history topic.

TOPIC: {title}

CONTENT:
{content[:3000]}

Provide:
1. INTRODUCTION approach (2-3 lines)
2. BODY structure (3-4 key arguments with supporting evidence)
3. CONCLUSION approach (2-3 lines)
4. DIAGRAM/FLOWCHART suggestion (if applicable)
5. KEYWORDS to include for better marks

OUTPUT FORMAT:
INTRO: Approach description
BODY:
- Argument 1: Evidence
- Argument 2: Evidence
- Argument 3: Evidence
CONCLUSION: Approach description
DIAGRAM: Suggestion
KEYWORDS: keyword1, keyword2, keyword3"""

    # === Recap Generation Prompt ===
    @staticmethod
    def get_recap_prompt(previous_title: str, previous_content_summary: str) -> str:
        """Generate a brief recap of the previous lesson."""
        return f"""Generate a brief recap (100-140 words) in Hinglish of the previous history class.

PREVIOUS TOPIC: {previous_title}

SUMMARY OF PREVIOUS CLASS:
{previous_content_summary[:1500]}

RULES:
1. Start with: "Friends, pichle class mein humne padha tha..."
2. Mention 3 most important points
3. Keep it brief and engaging
4. End with a transition to the new topic
5. Use Hinglish (Hindi narration, English terms)
6. TTS-friendly language"""
