"""
AI History Video Generator - Main Pipeline

This is the main entry point for the history video generation pipeline.
It orchestrates all components to:
1. Get next topic from syllabus
2. Generate lesson script using LLM
3. Convert script to speech (TTS)
4. Create avatar video
5. Compose final video with slides
6. Generate thumbnail
7. Generate PDF study notes
8. Upload to YouTube
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.database import Database
from src.syllabus.syllabus_manager import SyllabusManager
from src.script_generator.history_script_writer import HistoryScriptWriter
from src.tts import TTSManager
from src.avatar import AvatarGenerator
from src.video import VideoComposer, ThumbnailGenerator
from src.youtube import YouTubeUploader
from src.notes.notes_content_generator import NotesContentGenerator


class HistoryPipeline:
    """
    Main pipeline for generating history lesson videos.

    Pipeline steps:
    1. Get next topic from syllabus
    2. Generate lesson script
    3. Generate audio (TTS)
    4. Generate avatar video
    5. Compose final video
    6. Generate thumbnail
    7. Upload to YouTube (optional)
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to main configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Setup logging
        log_config = self.config.get("app", {})
        setup_logger(
            log_level=log_config.get("log_level", "INFO"),
            log_file="logs/pipeline.log"
        )
        self.logger = get_logger("HistoryPipeline")

        # Initialize database
        db_path = self.config.get("database", {}).get("path", "data/history_tracker.db")
        self.db = Database(db_path)

        # Initialize syllabus manager
        syllabus_path = self.config.get("syllabus", {}).get("path", "config/syllabus.yaml")
        self.syllabus_mgr = SyllabusManager(
            syllabus_path=syllabus_path,
            db=self.db
        )

        # Initialize script writer
        self.script_writer = HistoryScriptWriter(config=self.config)

        # Initialize TTS manager
        self.tts_manager = TTSManager(config_path=config_path)

        # Initialize avatar generator with config
        avatar_config = self.config.get("avatar", {})
        self.avatar_generator = AvatarGenerator(
            method=avatar_config.get("provider", "auto"),
            avatar_image=avatar_config.get("default_image", None),
            multitalk_config=avatar_config.get("multitalk", {}),
            sadtalker_hf_config=avatar_config.get("sadtalker_hf", {}),
            moda_config=avatar_config.get("moda", {}),
            audio2face3d_config=avatar_config.get("audio2face3d", {}),
        )

        # Initialize video composer
        self.video_composer = VideoComposer(config_path=config_path)

        # Initialize thumbnail generator
        self.thumbnail_generator = ThumbnailGenerator()

        # Initialize YouTube uploader
        self.youtube_uploader = YouTubeUploader()

        # Initialize enhanced notes content generator
        self.notes_generator = NotesContentGenerator(
            llm_config=self.config.get('llm', {})
        )

        # Output paths
        self.output_dir = Path(self.config.get("paths", {}).get("output", "output"))
        self.audio_dir = Path(self.config.get("paths", {}).get("audio", "output/audio"))
        self.video_dir = Path(self.config.get("paths", {}).get("videos", "output/videos"))
        self.thumbnail_dir = Path(self.config.get("paths", {}).get("thumbnails", "output/thumbnails"))
        self.notes_dir = Path(self.config.get("paths", {}).get("notes", "output/notes"))

        # Ensure directories exist
        for dir_path in [self.output_dir, self.audio_dir, self.video_dir,
                         self.thumbnail_dir, self.notes_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("HistoryPipeline initialized successfully")

    def _reset_unuploaded_failed_topics(self) -> int:
        """Reset failed topics that were never uploaded to YouTube back to pending."""
        from src.utils.database import Topic
        session = self.db.get_session()
        try:
            count = session.query(Topic).filter(
                Topic.status == 'failed',
                (Topic.youtube_id == None) | (Topic.youtube_id == '')
            ).update({
                'status': 'pending',
                'error': None,
                'started_at': None
            }, synchronize_session='fetch')
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to reset unuploaded topics: {e}")
            return 0
        finally:
            session.close()

    def _generate_comprehensive_notes(
        self,
        topic,
        script,
        video_duration: float = 0.0,
    ) -> Optional[str]:
        """
        Generate comprehensive 10-15 page PDF study notes.

        ROBUST: This method NEVER raises exceptions. If anything fails,
        it falls back to simpler generation methods. Notes are ALWAYS produced.
        """
        try:
            from src.notes.pdf_generator import PDFNotesGenerator, StudyNote, TopicNote
            from src.notes.notes_content_generator import NotesContentGenerator

            # Step 1: Generate enhanced content via LLM
            self.logger.info("Generating comprehensive note content via LLM...")
            note_content = self.notes_generator.generate_comprehensive_notes(
                topic_title=topic.title,
                era=topic.era,
                section=topic.section,
                subtopics=topic.get_subtopics(),
                key_concepts=topic.get_key_concepts(),
                exam_focus=topic.exam_focus,
                script_text=script.get_full_script() if script else "",
                script_segments=[s.__dict__ if hasattr(s, '__dict__') else s for s in (script.segments if script else [])],
                existing_key_points=script.key_points if script else [],
                existing_terms=script.important_terms if script else {},
                existing_questions=script.practice_questions if script else [],
            )

            # Step 2: Build TopicNote from generated content
            topic_note = TopicNote(
                title=note_content.title,
                trigger_line=note_content.trigger_line,
                what_is_it=note_content.what_is_it,
                key_provisions=note_content.key_provisions,
                sub_sections=[
                    {
                        'heading': sec.heading,
                        'points': sec.bullet_points if not sec.sub_sections else sec.sub_sections[0].get('points', []) if sec.sub_sections else [],
                        'sub_points': sec.sub_sections[0].get('sub_points', {}) if sec.sub_sections else {},
                    }
                    for sec in note_content.sections
                ],
                challenges=note_content.challenges,
                suggestions=note_content.suggestions,
                comparison_table=note_content.comparison_table,
                key_judgements=note_content.key_judgements,
                key_facts_box=note_content.key_facts_box,
                important_terms=note_content.important_terms,
                practice_questions=script.practice_questions if script else [],
                upsc_tags=note_content.upsc_tags,
                historical_background=note_content.historical_background,
                timeline=note_content.timeline,
                prelims_mcqs=note_content.prelims_mcqs,
                mains_questions=note_content.mains_questions,
                quick_revision_points=note_content.quick_revision_points,
            )

            # Step 3: Build StudyNote and generate PDF
            from datetime import datetime as dt
            study_note = StudyNote(
                title=f"Part {topic.part_number}: {topic.title}",
                date=dt.now().strftime("%B %d, %Y"),
                topics=[topic_note],
                video_duration=video_duration,
                language="Hinglish",
            )

            generator = PDFNotesGenerator(output_dir=str(self.notes_dir))
            pdf_path = generator.generate_notes(study_note)
            self.logger.info(f"Comprehensive PDF notes saved: {pdf_path}")
            return pdf_path

        except Exception as e:
            self.logger.error(f"Comprehensive notes generation failed: {e}")
            self.logger.info("Attempting fallback note generation...")

            # FALLBACK: Force-generate a simpler PDF using script data directly
            try:
                return self._fallback_generate_notes(topic, script, video_duration)
            except Exception as e2:
                self.logger.error(f"Fallback notes generation also failed: {e2}")
                return None

    def _fallback_generate_notes(
        self, topic, script, video_duration: float
    ) -> Optional[str]:
        """Fallback: generate basic PDF from script data when LLM-enhanced generation fails."""
        from src.notes.pdf_generator import PDFNotesGenerator, StudyNote, TopicNote

        segments = script.segments if script else []
        sub_sections = []
        for seg in segments:
            content = seg.content if hasattr(seg, 'content') else str(seg.get('content', ''))
            kps = seg.key_points if hasattr(seg, 'key_points') else seg.get('key_points', [])
            title = seg.title if hasattr(seg, 'title') else seg.get('title', 'Content')
            if content and len(content) > 50:
                sub_sections.append({
                    'heading': title,
                    'points': kps if kps else [s.strip() for s in content.split('.') if len(s.strip()) > 20][:5],
                    'sub_points': {},
                })

        topic_note = TopicNote(
            title=topic.title,
            trigger_line=f"{topic.title} - Important topic for UPSC {topic.exam_focus} preparation under {topic.era}.",
            what_is_it=f"{topic.title} is part of {topic.section} in {topic.era}. "
                       f"This topic covers: {', '.join(topic.get_subtopics()[:4])}.",
            key_provisions=[f"Key concept: {c}" for c in topic.get_key_concepts()[:5]],
            sub_sections=sub_sections,
            challenges=["Study in context of the broader era", "Focus on cause-effect for Mains"],
            suggestions=["Practice previous year questions", "Create timeline and flowcharts"],
            key_facts_box=[f"Part {topic.part_number} of 180", f"Era: {topic.era}", f"Section: {topic.section}"],
            important_terms=script.important_terms if script else {},
            practice_questions=script.practice_questions if script else [],
            upsc_tags=f"History | {topic.era} | {topic.section} | {topic.exam_focus}",
        )

        from datetime import datetime as dt
        study_note = StudyNote(
            title=f"Part {topic.part_number}: {topic.title}",
            date=dt.now().strftime("%B %d, %Y"),
            topics=[topic_note],
            video_duration=video_duration,
        )

        generator = PDFNotesGenerator(output_dir=str(self.notes_dir))
        return generator.generate_notes(study_note)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
            return {}

    async def run(
        self,
        upload: bool = True,
        test_mode: bool = False,
        part_number: int = None
    ) -> Dict[str, Any]:
        """
        Run the complete history video generation pipeline.

        Args:
            upload: Whether to upload to YouTube.
            test_mode: If True, uploads as private video.
            part_number: Specific part to generate (None = next pending).

        Returns:
            Dictionary with pipeline results.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting History Video Generation Pipeline")
        self.logger.info("=" * 60)

        results = {
            "success": False,
            "part_number": None,
            "topic_title": None,
            "steps": {},
            "errors": []
        }

        try:
            # === Step 1: Get next topic from syllabus ===
            self.logger.info("Step 1: Getting next topic from syllabus...")

            # Reset any stuck 'generating' or 'failed' topics (without youtube_id)
            # so pipeline retries them instead of skipping ahead
            reset_stuck = self.syllabus_mgr.reset_stuck()
            reset_failed = self._reset_unuploaded_failed_topics()
            if reset_stuck:
                self.logger.info(f"Reset {reset_stuck} stuck 'generating' topics")
            if reset_failed:
                self.logger.info(f"Reset {reset_failed} failed (not uploaded) topics")

            if part_number:
                topic = self.syllabus_mgr.get_topic_by_part(part_number)
                if not topic:
                    results["errors"].append(f"Topic for Part {part_number} not found")
                    self.logger.error(f"Part {part_number} not found in syllabus")
                    return results
            else:
                topic = self.syllabus_mgr.get_next_topic()
                if not topic:
                    self.logger.info("All 180 topics completed! Course is done.")
                    results["errors"].append("All topics completed")
                    return results

            results["part_number"] = topic.part_number
            results["topic_title"] = topic.title

            # Get adjacent topics for context
            previous_topic = self.syllabus_mgr.get_topic_by_part(topic.part_number - 1)
            next_topic = self.syllabus_mgr.get_topic_by_part(topic.part_number + 1)

            results["steps"]["topic"] = {
                "part_number": topic.part_number,
                "title": topic.title,
                "era": topic.era,
                "section": topic.section,
                "exam_focus": topic.exam_focus
            }
            self.logger.info(
                f"Topic: Part {topic.part_number}/180 - {topic.title} "
                f"[{topic.era} > {topic.section}]"
            )

            # Show progress
            progress = self.syllabus_mgr.get_progress()
            self.logger.info(
                f"Progress: {progress['completed']}/{progress['total']} "
                f"({progress['percentage']:.1f}%)"
            )

            # === Step 2: Generate lesson script ===
            self.logger.info("Step 2: Generating lesson script...")

            script = self.script_writer.generate_lesson_script(
                topic=topic,
                previous_topic=previous_topic,
                next_topic=next_topic
            )

            # Save script to file
            script_path = self.output_dir / f"part_{topic.part_number:03d}_script.txt"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(f"# Part {topic.part_number}: {topic.title}\n")
                f.write(f"# Era: {topic.era} | Section: {topic.section}\n")
                f.write(f"# Date: {script.date}\n")
                f.write(f"# Words: {script.word_count} | Duration: ~{script.total_duration/60:.1f} min\n\n")
                f.write(script.get_full_script())

            results["steps"]["script"] = {
                "word_count": script.word_count,
                "duration_estimate": f"{script.total_duration/60:.1f} min",
                "segments": len(script.segments),
                "path": str(script_path)
            }
            self.logger.info(
                f"Script generated: {script.word_count} words, "
                f"~{script.total_duration/60:.1f} min, {len(script.segments)} segments"
            )

            # === Step 3: Generate audio (TTS) ===
            self.logger.info("Step 3: Generating audio (TTS)...")

            audio_result = await self.tts_manager.generate_lesson_audio(
                script_text=script.get_script_for_tts(),
                output_dir=str(self.audio_dir),
                part_number=topic.part_number,
                language="hi"
            )

            if not audio_result.get("success"):
                error = audio_result.get("error", "TTS generation failed")
                results["errors"].append(f"TTS failed: {error}")
                self.logger.error(f"TTS failed: {error}")
                return results

            audio_path = audio_result["audio_path"]
            audio_duration = audio_result["duration"]

            results["steps"]["audio"] = {
                "duration": f"{audio_duration:.1f}s ({audio_duration/60:.1f} min)",
                "path": audio_path,
                "voice": audio_result.get("voice", "unknown")
            }
            self.logger.info(f"Audio generated: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")

            # === Step 4: Generate avatar video ===
            self.logger.info("Step 4: Generating avatar video...")

            avatar_path = str(self.video_dir / f"part_{topic.part_number:03d}_avatar.mp4")

            avatar_result = self.avatar_generator.generate(
                audio_path=audio_path,
                output_path=avatar_path
            )

            if not avatar_result.success:
                error = avatar_result.error or "Avatar generation failed"
                results["errors"].append(f"Avatar failed: {error}")
                self.logger.error(f"Avatar failed: {error}")
                return results

            results["steps"]["avatar"] = {
                "duration": f"{avatar_result.duration:.1f}s",
                "method": avatar_result.method,
                "path": avatar_path
            }
            self.logger.info(f"Avatar video generated: {avatar_result.method}")

            # === Step 5: Compose final video ===
            self.logger.info("Step 5: Composing final video...")

            final_video_path = str(
                self.video_dir / f"part_{topic.part_number:03d}_final.mp4"
            )

            # Build script_data for composer
            script_data = script.to_dict()

            composition_result = self.video_composer.compose(
                avatar_video_path=avatar_path,
                output_path=final_video_path,
                headlines=[topic.title],
                title=f"History Class - Part {topic.part_number}",
                date=script.date,
                script_data=script_data
            )

            if not composition_result.success:
                error = composition_result.error or "Video composition failed"
                results["errors"].append(f"Composition failed: {error}")
                self.logger.error(f"Composition failed: {error}")
                return results

            results["steps"]["video"] = {
                "duration": f"{composition_result.duration:.1f}s",
                "resolution": f"{composition_result.resolution[0]}x{composition_result.resolution[1]}",
                "path": final_video_path,
            }
            self.logger.info(f"Video composed: {composition_result.duration:.1f}s")

            # === Step 5b: Generate comprehensive PDF study notes ===
            self.logger.info("Step 5b: Generating comprehensive PDF study notes (10-15 pages)...")

            pdf_notes_path = self._generate_comprehensive_notes(
                topic=topic,
                script=script,
                video_duration=composition_result.duration,
            )

            if pdf_notes_path:
                results["steps"]["pdf_notes"] = {
                    "path": pdf_notes_path,
                    "status": "generated",
                }
                self.logger.info(f"Comprehensive PDF notes generated: {pdf_notes_path}")
            else:
                results["steps"]["pdf_notes"] = {"status": "fallback_used"}
                self.logger.warning("PDF notes generation used fallback")

            # === Step 6: Generate thumbnail ===
            self.logger.info("Step 6: Generating thumbnail...")

            thumbnail_path = str(
                self.thumbnail_dir / f"part_{topic.part_number:03d}_thumbnail.png"
            )

            thumbnail_bg = self.config.get("thumbnail", {}).get("background_image", "")
            if thumbnail_bg and not Path(thumbnail_bg).is_absolute():
                # Resolve relative path from project root
                project_root = Path(__file__).parent
                resolved = project_root / thumbnail_bg
                if not resolved.exists():
                    # Also check in assets/avatars
                    resolved = project_root / "assets" / "avatars" / thumbnail_bg
                thumbnail_bg = str(resolved) if resolved.exists() else thumbnail_bg

            thumbnail_result = self.thumbnail_generator.generate(
                output_path=thumbnail_path,
                title=f"Part {topic.part_number} | {topic.title}",
                date=script.date,
                background_image=thumbnail_bg if thumbnail_bg else None
            )

            results["steps"]["thumbnail"] = {
                "success": thumbnail_result.success,
                "path": str(thumbnail_path) if thumbnail_result.success else ""
            }

            if thumbnail_result.success:
                self.logger.info(f"Thumbnail generated: {thumbnail_path}")
            else:
                self.logger.warning(f"Thumbnail failed: {thumbnail_result.error}")

            # === Step 7: Upload to YouTube (if enabled) ===
            if upload:
                self.logger.info("Step 7: Uploading to YouTube...")

                # Build topic_metadata safely before calling upload
                topic_meta = {
                    "part_number": topic.part_number,
                    "total_parts": self.syllabus_mgr.get_total_parts(),
                    "topic": topic.title,
                    "era": topic.era,
                    "section": topic.section,
                    "subtopics": topic.get_subtopics(),
                }

                upload_result = self.youtube_uploader.upload_with_metadata(
                    video_path=final_video_path,
                    headlines=[
                        f"Part {topic.part_number} | {topic.title}",
                        f"{topic.era} - {topic.section}",
                    ],
                    sources=[],
                    language="hi",
                    date=script.date,
                    thumbnail_path=thumbnail_path if thumbnail_result.success else None,
                    privacy_status="private" if test_mode else "public",
                    pdf_path=pdf_notes_path,
                    topic_metadata=topic_meta,
                )

                results["steps"]["upload"] = {
                    "success": upload_result.success,
                    "video_id": upload_result.video_id,
                    "url": upload_result.video_url,
                    "error": upload_result.error
                }

                if upload_result.success:
                    self.logger.info(f"Uploaded to YouTube: {upload_result.video_url}")

                    # Mark topic as completed in syllabus
                    self.syllabus_mgr.mark_completed(
                        part_number=topic.part_number,
                        youtube_id=upload_result.video_id,
                        video_path=final_video_path
                    )
                else:
                    error = upload_result.error or "Upload failed"
                    results["errors"].append(f"Upload failed: {error}")
                    self.logger.error(f"Upload failed: {error}")
            else:
                results["steps"]["upload"] = {"skipped": True}
                self.logger.info("Upload skipped (--no-upload)")

                # Mark completed without YouTube ID
                self.syllabus_mgr.mark_completed(
                    part_number=topic.part_number,
                    video_path=final_video_path
                )

            # Pipeline complete
            results["success"] = True

            self.logger.info("=" * 60)
            self.logger.info(
                f"Pipeline completed: Part {topic.part_number} - {topic.title}"
            )

            # Show updated progress
            progress = self.syllabus_mgr.get_progress()
            self.logger.info(
                f"Course progress: {progress['completed']}/{progress['total']} "
                f"({progress['percentage']:.1f}%)"
            )
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Pipeline failed with exception: {e}", exc_info=True)
            results["errors"].append(str(e))

            # Mark topic as failed if we had one
            if results.get("part_number"):
                self.syllabus_mgr.mark_failed(results["part_number"], str(e))

        return results

    def run_sync(self, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(**kwargs))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI History Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Generate next part and upload
  python main.py --test                 # Generate and upload as private (test)
  python main.py --no-upload            # Generate video without uploading
  python main.py --part 1              # Generate specific part
  python main.py --progress             # Show course progress
        """
    )

    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Don't upload to YouTube"
    )

    parser.add_argument(
        "--test", "--private",
        action="store_true",
        help="Private mode - upload as private video (for testing/review)"
    )

    parser.add_argument(
        "--part", "-p",
        type=int,
        default=None,
        help="Generate a specific part number (default: next pending)"
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show course progress and exit"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Banner
    print("\n" + "=" * 60)
    print("  AI History Video Generator")
    print("  UPSC & State PSC - Complete History Course")
    print("=" * 60 + "\n")

    pipeline = HistoryPipeline(config_path=args.config)

    # Progress mode
    if args.progress:
        pipeline.syllabus_mgr.print_progress()
        return 0

    # Run pipeline
    results = pipeline.run_sync(
        upload=not args.no_upload,
        test_mode=args.test,
        part_number=args.part
    )

    # Print results
    print("\n" + "=" * 60)
    print("  Pipeline Results")
    print("=" * 60 + "\n")

    if results["success"]:
        print(f"Status: SUCCESS")
        print(f"Part: {results['part_number']} - {results['topic_title']}\n")

        for step, info in results["steps"].items():
            print(f"{step.upper()}:")
            if isinstance(info, dict):
                for key, value in info.items():
                    if value:
                        print(f"  {key}: {value}")
            print()

        if results["steps"].get("upload", {}).get("url"):
            print(f"YouTube URL: {results['steps']['upload']['url']}")

    else:
        print("Status: FAILED\n")
        print("Errors:")
        for error in results["errors"]:
            print(f"  - {error}")

    print("\n" + "=" * 60 + "\n")

    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
