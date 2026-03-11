"""
Content Extractor - Extracts key educational content for UPSC/competitive exams
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.scraper.base_scraper import NewsArticle
from src.script_generator.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExamRelevance(Enum):
    """UPSC exam relevance categories"""
    PRELIMS = "prelims"
    MAINS = "mains"
    BOTH = "both"
    INTERVIEW = "interview"


class SubjectCategory(Enum):
    """UPSC subject categories"""
    POLITY = "Polity & Governance"
    ECONOMY = "Economy"
    ENVIRONMENT = "Environment & Ecology"
    SCIENCE_TECH = "Science & Technology"
    INTERNATIONAL = "International Relations"
    GEOGRAPHY = "Geography"
    HISTORY = "History"
    SOCIAL = "Social Issues"
    SECURITY = "Internal Security"
    ETHICS = "Ethics"
    CURRENT_AFFAIRS = "Current Affairs"


@dataclass
class KeyPoint:
    """A key point extracted from content"""
    text: str
    importance: int = 1  # 1-5 scale
    category: str = ""
    related_facts: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)  # Numbers, statistics


@dataclass
class UPSCRelevance:
    """UPSC exam relevance information"""
    subject: SubjectCategory
    exam_relevance: ExamRelevance
    syllabus_topic: str
    previous_year_questions: List[str] = field(default_factory=list)
    mains_paper: Optional[str] = None  # GS1, GS2, GS3, GS4, Essay
    prelims_topic: Optional[str] = None
    important_for: List[str] = field(default_factory=list)  # Other exams


@dataclass
class ExtractedContent:
    """Complete extracted educational content"""
    article: NewsArticle
    summary: str
    key_points: List[KeyPoint]
    upsc_relevance: UPSCRelevance
    important_terms: Dict[str, str]  # term: definition
    timeline: List[Dict[str, str]]  # date: event
    related_topics: List[str]
    practice_questions: List[str]
    image_suggestions: List[str]  # Suggestions for relevant images/maps
    static_gk_links: List[str]  # Related static GK topics


class ContentExtractor:
    """
    Extracts educational content optimized for UPSC and competitive exam preparation.
    """

    # Subject keywords for classification
    SUBJECT_KEYWORDS = {
        SubjectCategory.POLITY: [
            'parliament', 'constitution', 'judiciary', 'election', 'bill', 'act',
            'amendment', 'supreme court', 'high court', 'governor', 'president',
            'prime minister', 'cabinet', 'fundamental rights', 'directive principles',
            'lokpal', 'lokayukta', 'panchayat', 'municipality', 'federalism'
        ],
        SubjectCategory.ECONOMY: [
            'gdp', 'inflation', 'rbi', 'fiscal', 'budget', 'tax', 'gst', 'import',
            'export', 'trade', 'monetary policy', 'niti aayog', 'bank', 'finance',
            'stock market', 'investment', 'fdi', 'startup', 'msme', 'agriculture'
        ],
        SubjectCategory.ENVIRONMENT: [
            'climate', 'pollution', 'biodiversity', 'forest', 'wildlife', 'carbon',
            'renewable', 'solar', 'wind', 'conservation', 'species', 'ecosystem',
            'national park', 'sanctuary', 'wetland', 'coral', 'mangrove', 'emission'
        ],
        SubjectCategory.SCIENCE_TECH: [
            'isro', 'space', 'satellite', 'ai', 'technology', 'digital', 'cyber',
            'quantum', 'nuclear', 'research', 'innovation', 'startup', '5g', 'iot',
            'blockchain', 'drone', 'robot', 'biotech', 'vaccine', 'pharma'
        ],
        SubjectCategory.INTERNATIONAL: [
            'un', 'g20', 'g7', 'brics', 'nato', 'china', 'pakistan', 'usa',
            'russia', 'diplomat', 'treaty', 'bilateral', 'multilateral', 'summit',
            'foreign policy', 'ambassador', 'sanctions', 'war', 'peace'
        ],
        SubjectCategory.GEOGRAPHY: [
            'earthquake', 'cyclone', 'monsoon', 'river', 'mountain', 'plateau',
            'coast', 'island', 'mineral', 'soil', 'irrigation', 'dam', 'flood',
            'drought', 'landslide', 'volcano', 'ocean', 'glacier'
        ],
        SubjectCategory.HISTORY: [
            'heritage', 'archaeological', 'ancient', 'medieval', 'modern',
            'freedom', 'independence', 'colonial', 'mughal', 'british', 'revolt',
            'civilization', 'dynasty', 'monument', 'unesco'
        ],
        SubjectCategory.SOCIAL: [
            'poverty', 'education', 'health', 'women', 'child', 'welfare',
            'caste', 'tribe', 'minority', 'disability', 'elderly', 'nutrition',
            'sanitation', 'employment', 'skill', 'migration', 'population'
        ],
        SubjectCategory.SECURITY: [
            'terrorism', 'naxal', 'insurgency', 'border', 'defence', 'army',
            'navy', 'air force', 'crpf', 'bsf', 'nsa', 'nctc', 'nia', 'raw',
            'cyber security', 'internal security', 'ceasefire'
        ]
    }

    # Mains paper mapping
    MAINS_PAPER_MAP = {
        SubjectCategory.HISTORY: "GS1",
        SubjectCategory.GEOGRAPHY: "GS1",
        SubjectCategory.SOCIAL: "GS1",
        SubjectCategory.POLITY: "GS2",
        SubjectCategory.INTERNATIONAL: "GS2",
        SubjectCategory.ECONOMY: "GS3",
        SubjectCategory.ENVIRONMENT: "GS3",
        SubjectCategory.SCIENCE_TECH: "GS3",
        SubjectCategory.SECURITY: "GS3",
        SubjectCategory.ETHICS: "GS4"
    }

    def __init__(
        self,
        llm_provider: str = "groq",
        llm_config: Dict[str, Any] = None
    ):
        """Initialize content extractor with LLM support."""
        llm_config = llm_config or {}
        self.llm = LLMClient(provider=llm_provider, **llm_config)
        logger.info("ContentExtractor initialized")

    def extract_content(self, article: NewsArticle) -> ExtractedContent:
        """
        Extract comprehensive educational content from a news article.

        Args:
            article: News article to process

        Returns:
            ExtractedContent with all educational elements
        """
        logger.info(f"Extracting content from: {article.title}")

        # Determine subject category
        subject = self._classify_subject(article)

        # Extract key points using LLM
        key_points = self._extract_key_points(article)

        # Extract important terms
        important_terms = self._extract_terms(article)

        # Extract dates and timeline
        timeline = self._extract_timeline(article)

        # Determine UPSC relevance
        upsc_relevance = self._determine_upsc_relevance(article, subject)

        # Generate practice questions
        practice_questions = self._generate_practice_questions(article, subject)

        # Get related topics
        related_topics = self._get_related_topics(article, subject)

        # Get image suggestions
        image_suggestions = self._get_image_suggestions(article, subject)

        # Get static GK links
        static_gk_links = self._get_static_gk_links(article, subject)

        # Generate summary
        summary = self._generate_summary(article)

        return ExtractedContent(
            article=article,
            summary=summary,
            key_points=key_points,
            upsc_relevance=upsc_relevance,
            important_terms=important_terms,
            timeline=timeline,
            related_topics=related_topics,
            practice_questions=practice_questions,
            image_suggestions=image_suggestions,
            static_gk_links=static_gk_links
        )

    def _classify_subject(self, article: NewsArticle) -> SubjectCategory:
        """Classify article into UPSC subject category."""
        content = f"{article.title} {article.summary or ''} {article.content or ''}".lower()

        scores = {}
        for subject, keywords in self.SUBJECT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content)
            scores[subject] = score

        if scores:
            best_subject = max(scores.keys(), key=lambda k: scores[k])
            if scores[best_subject] > 0:
                return best_subject

        return SubjectCategory.CURRENT_AFFAIRS

    def _extract_key_points(self, article: NewsArticle) -> List[KeyPoint]:
        """Extract key points using LLM."""
        prompt = f"""Extract 5-7 key points from this news article for UPSC exam preparation.

Article Title: {article.title}
Content: {article.content or article.summary or ''}

For each key point, provide:
1. The main point (1-2 sentences)
2. Important dates mentioned
3. Important numbers/statistics
4. Related facts

Format each point as:
POINT: [key point text]
DATES: [comma-separated dates if any]
FIGURES: [comma-separated numbers/statistics if any]
FACTS: [related facts if any]

Focus on facts that are likely to appear in competitive exams."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3
            )

            return self._parse_key_points(response)
        except Exception as e:
            logger.warning(f"Failed to extract key points: {e}")
            # Fallback: create basic key points
            return [KeyPoint(
                text=article.summary or article.title,
                importance=3
            )]

    def _parse_key_points(self, response: str) -> List[KeyPoint]:
        """Parse LLM response into KeyPoint objects."""
        key_points = []

        # Split by POINT: markers
        point_blocks = re.split(r'POINT:\s*', response)

        for block in point_blocks[1:]:  # Skip first empty split
            lines = block.strip().split('\n')
            if not lines:
                continue

            text = lines[0].strip()
            dates = []
            figures = []
            facts = []

            for line in lines[1:]:
                if line.startswith('DATES:'):
                    dates = [d.strip() for d in line.replace('DATES:', '').split(',') if d.strip()]
                elif line.startswith('FIGURES:'):
                    figures = [f.strip() for f in line.replace('FIGURES:', '').split(',') if f.strip()]
                elif line.startswith('FACTS:'):
                    facts = [f.strip() for f in line.replace('FACTS:', '').split(',') if f.strip()]

            if text:
                key_points.append(KeyPoint(
                    text=text,
                    importance=3,
                    dates=dates,
                    figures=figures,
                    related_facts=facts
                ))

        return key_points if key_points else [KeyPoint(text="Key information from the article", importance=2)]

    def _extract_terms(self, article: NewsArticle) -> Dict[str, str]:
        """Extract important terms and their definitions."""
        prompt = f"""Extract important terms/concepts from this article that UPSC aspirants should know.

Article: {article.title}
Content: {article.content or article.summary or ''}

List 5-8 important terms with brief definitions.
Format: TERM: definition

Focus on:
- Government schemes/programs
- Organizations/bodies
- Technical terms
- Important concepts"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3
            )

            terms = {}
            for line in response.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        term = parts[0].strip().strip('-•*')
                        definition = parts[1].strip()
                        if term and definition:
                            terms[term] = definition

            return terms
        except Exception as e:
            logger.warning(f"Failed to extract terms: {e}")
            return {}

    def _extract_timeline(self, article: NewsArticle) -> List[Dict[str, str]]:
        """Extract timeline/chronological events."""
        content = f"{article.content or ''} {article.summary or ''}"

        # Find date patterns
        date_patterns = [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})'
        ]

        timeline = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:5]:  # Limit to 5 dates
                timeline.append({"date": match, "event": ""})

        return timeline

    def _determine_upsc_relevance(
        self,
        article: NewsArticle,
        subject: SubjectCategory
    ) -> UPSCRelevance:
        """Determine UPSC exam relevance."""
        content_lower = f"{article.title} {article.content or ''}".lower()

        # Determine exam relevance
        if any(kw in content_lower for kw in ['treaty', 'agreement', 'summit', 'policy', 'bill']):
            exam_relevance = ExamRelevance.BOTH
        elif any(kw in content_lower for kw in ['statistics', 'data', 'figure', 'number']):
            exam_relevance = ExamRelevance.PRELIMS
        else:
            exam_relevance = ExamRelevance.MAINS

        # Get mains paper
        mains_paper = self.MAINS_PAPER_MAP.get(subject, "GS3")

        # Other exams this is relevant for
        important_for = ["UPSC CSE"]
        if subject == SubjectCategory.ECONOMY:
            important_for.extend(["RBI Grade B", "SEBI", "NABARD"])
        elif subject == SubjectCategory.POLITY:
            important_for.extend(["State PCS", "Judicial Services"])
        elif subject == SubjectCategory.SCIENCE_TECH:
            important_for.extend(["ISRO", "DRDO"])

        return UPSCRelevance(
            subject=subject,
            exam_relevance=exam_relevance,
            syllabus_topic=subject.value,
            mains_paper=mains_paper,
            important_for=important_for
        )

    def _generate_practice_questions(
        self,
        article: NewsArticle,
        subject: SubjectCategory
    ) -> List[str]:
        """Generate practice questions for exam preparation."""
        prompt = f"""Generate 3 UPSC-style practice questions based on this news article.

Article: {article.title}
Subject: {subject.value}
Content: {article.content or article.summary or ''}

Create:
1. One Prelims-style MCQ question
2. One Mains-style analytical question (150 words)
3. One Current Affairs factual question

Format:
Q1 (Prelims): [question]
Q2 (Mains): [question]
Q3 (Current Affairs): [question]"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=600,
                temperature=0.5
            )

            questions = []
            for line in response.split('\n'):
                if line.strip().startswith('Q'):
                    questions.append(line.strip())

            return questions[:3]
        except Exception as e:
            logger.warning(f"Failed to generate questions: {e}")
            return [f"Discuss the significance of: {article.title}"]

    def _get_related_topics(
        self,
        article: NewsArticle,
        subject: SubjectCategory
    ) -> List[str]:
        """Get related static/background topics to study."""
        related_map = {
            SubjectCategory.POLITY: [
                "Constitutional Provisions", "Parliamentary Procedures",
                "Fundamental Rights", "Directive Principles"
            ],
            SubjectCategory.ECONOMY: [
                "Monetary Policy", "Fiscal Policy", "Banking System",
                "International Trade"
            ],
            SubjectCategory.ENVIRONMENT: [
                "Environmental Laws", "International Agreements",
                "Climate Change", "Biodiversity Conservation"
            ],
            SubjectCategory.SCIENCE_TECH: [
                "Space Program", "Defence Technology",
                "Digital India", "Biotechnology"
            ],
            SubjectCategory.INTERNATIONAL: [
                "India's Foreign Policy", "Regional Organizations",
                "UN System", "Bilateral Relations"
            ],
            SubjectCategory.GEOGRAPHY: [
                "Physical Geography", "Human Geography",
                "Indian Geography", "Disaster Management"
            ],
            SubjectCategory.HISTORY: [
                "Modern Indian History", "Post-Independence India",
                "Cultural Heritage", "Art & Architecture"
            ],
            SubjectCategory.SOCIAL: [
                "Social Justice", "Welfare Schemes",
                "Education Policy", "Health Policy"
            ],
            SubjectCategory.SECURITY: [
                "Internal Security Framework", "Border Management",
                "Cyber Security", "Counter-terrorism"
            ]
        }

        return related_map.get(subject, ["Current Affairs", "General Studies"])

    def _get_image_suggestions(
        self,
        article: NewsArticle,
        subject: SubjectCategory
    ) -> List[str]:
        """Suggest relevant images/diagrams for the topic."""
        suggestions = []
        content_lower = f"{article.title} {article.content or ''}".lower()

        # Geography-related
        if subject == SubjectCategory.GEOGRAPHY or any(
            kw in content_lower for kw in ['map', 'river', 'mountain', 'region', 'state', 'district']
        ):
            suggestions.append("Map of India highlighting relevant region")

        # Economy-related
        if subject == SubjectCategory.ECONOMY or any(
            kw in content_lower for kw in ['gdp', 'growth', 'statistics', 'data']
        ):
            suggestions.append("Statistical chart/graph showing trends")

        # International-related
        if subject == SubjectCategory.INTERNATIONAL:
            suggestions.append("World map highlighting countries involved")

        # Polity-related
        if subject == SubjectCategory.POLITY:
            suggestions.append("Flowchart of constitutional process")

        # Default suggestions
        if not suggestions:
            suggestions.append("Relevant infographic")
            suggestions.append("Key facts visualization")

        return suggestions

    def _get_static_gk_links(
        self,
        article: NewsArticle,
        subject: SubjectCategory
    ) -> List[str]:
        """Get links to static GK topics for background study."""
        # These are topic names that link to background knowledge
        static_links = {
            SubjectCategory.POLITY: [
                "Indian Constitution - Basic Structure",
                "Parliament and State Legislatures",
                "Judiciary System in India"
            ],
            SubjectCategory.ECONOMY: [
                "Indian Economy - Overview",
                "Five Year Plans History",
                "Financial Institutions in India"
            ],
            SubjectCategory.ENVIRONMENT: [
                "Environmental Acts in India",
                "International Environmental Conventions",
                "Protected Areas in India"
            ],
            SubjectCategory.INTERNATIONAL: [
                "India's Neighbors - Relations",
                "Important International Organizations",
                "Major Treaties and Agreements"
            ]
        }

        return static_links.get(subject, ["General Knowledge Background"])

    def _generate_summary(self, article: NewsArticle) -> str:
        """Generate a concise summary for notes."""
        if article.summary and len(article.summary) < 300:
            return article.summary

        prompt = f"""Summarize this news article in 2-3 sentences for UPSC notes.

Title: {article.title}
Content: {article.content or article.summary or ''}

Focus on: What happened, Who is involved, Why it matters for India."""

        try:
            return self.llm.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return article.summary or article.title

    def batch_extract(self, articles: List[NewsArticle]) -> List[ExtractedContent]:
        """Extract content from multiple articles."""
        results = []
        for article in articles:
            try:
                content = self.extract_content(article)
                results.append(content)
            except Exception as e:
                logger.error(f"Failed to extract content from {article.title}: {e}")

        return results
