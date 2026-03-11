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

        # Initialize avatar generator
        self.avatar_generator = AvatarGenerator()

        # Initialize video composer
        self.video_composer = VideoComposer(config_path=config_path)

        # Initialize thumbnail generator
        self.thumbnail_generator = ThumbnailGenerator()

        # Initialize YouTube uploader
        self.youtube_uploader = YouTubeUploader()

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

            # Log step start
            self.db.log_step_start(topic.part_number, "pipeline")

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
            self.db.log_step_start(topic.part_number, "script")

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

            self.db.log_step_complete(topic.id, "script", {
                "word_count": script.word_count,
                "duration": script.total_duration,
                "segments": len(script.segments)
            })

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
            self.db.log_step_start(topic.part_number, "audio")

            audio_result = await self.tts_manager.generate_lesson_audio(
                script_text=script.get_script_for_tts(),
                output_dir=str(self.audio_dir),
                part_number=topic.part_number,
                language="hi"
            )

            if not audio_result.get("success"):
                error = audio_result.get("error", "TTS generation failed")
                self.db.log_step_failure(topic.id, "audio", error)
                results["errors"].append(f"TTS failed: {error}")
                self.logger.error(f"TTS failed: {error}")
                return results

            audio_path = audio_result["audio_path"]
            audio_duration = audio_result["duration"]

            self.db.log_step_complete(topic.id, "audio", {
                "duration": audio_duration,
                "voice": audio_result.get("voice", "unknown")
            })

            results["steps"]["audio"] = {
                "duration": f"{audio_duration:.1f}s ({audio_duration/60:.1f} min)",
                "path": audio_path,
                "voice": audio_result.get("voice", "unknown")
            }
            self.logger.info(f"Audio generated: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")

            # === Step 4: Generate avatar video ===
            self.logger.info("Step 4: Generating avatar video...")
            self.db.log_step_start(topic.part_number, "avatar")

            avatar_path = str(self.video_dir / f"part_{topic.part_number:03d}_avatar.mp4")

            avatar_result = self.avatar_generator.generate(
                audio_path=audio_path,
                output_path=avatar_path
            )

            if not avatar_result.success:
                error = avatar_result.error or "Avatar generation failed"
                self.db.log_step_failure(topic.id, "avatar", error)
                results["errors"].append(f"Avatar failed: {error}")
                self.logger.error(f"Avatar failed: {error}")
                return results

            self.db.log_step_complete(topic.id, "avatar", {
                "method": avatar_result.method,
                "duration": avatar_result.duration
            })

            results["steps"]["avatar"] = {
                "duration": f"{avatar_result.duration:.1f}s",
                "method": avatar_result.method,
                "path": avatar_path
            }
            self.logger.info(f"Avatar video generated: {avatar_result.method}")

            # === Step 5: Compose final video ===
            self.logger.info("Step 5: Composing final video...")
            self.db.log_step_start(topic.part_number, "video")

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
                self.db.log_step_failure(topic.id, "video", error)
                results["errors"].append(f"Composition failed: {error}")
                self.logger.error(f"Composition failed: {error}")
                return results

            self.db.log_step_complete(topic.id, "video", {
                "duration": composition_result.duration,
                "resolution": f"{composition_result.resolution[0]}x{composition_result.resolution[1]}",
                "pdf_notes": composition_result.pdf_notes_path
            })

            results["steps"]["video"] = {
                "duration": f"{composition_result.duration:.1f}s",
                "resolution": f"{composition_result.resolution[0]}x{composition_result.resolution[1]}",
                "path": final_video_path,
                "pdf_notes": composition_result.pdf_notes_path or ""
            }
            self.logger.info(f"Video composed: {composition_result.duration:.1f}s")

            pdf_notes_path = composition_result.pdf_notes_path

            # === Step 6: Generate thumbnail ===
            self.logger.info("Step 6: Generating thumbnail...")

            thumbnail_path = str(
                self.thumbnail_dir / f"part_{topic.part_number:03d}_thumbnail.png"
            )

            thumbnail_result = self.thumbnail_generator.generate(
                output_path=thumbnail_path,
                title=f"Part {topic.part_number} | {topic.title}",
                date=script.date
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
                self.db.log_step_start(topic.part_number, "upload")

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
                    pdf_path=pdf_notes_path
                )

                results["steps"]["upload"] = {
                    "success": upload_result.success,
                    "video_id": upload_result.video_id,
                    "url": upload_result.video_url,
                    "error": upload_result.error
                }

                if upload_result.success:
                    self.db.log_step_complete(topic.id, "upload", {
                        "video_id": upload_result.video_id,
                        "url": upload_result.video_url
                    })
                    self.logger.info(f"Uploaded to YouTube: {upload_result.video_url}")

                    # Mark topic as completed in syllabus
                    self.syllabus_mgr.mark_completed(
                        part_number=topic.part_number,
                        youtube_id=upload_result.video_id,
                        video_path=final_video_path
                    )
                else:
                    error = upload_result.error or "Upload failed"
                    self.db.log_step_failure(topic.id, "upload", error)
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
            self.db.log_step_complete(topic.id, "pipeline", {
                "total_duration": composition_result.duration
            })

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
        "--test",
        action="store_true",
        help="Test mode - upload as private video"
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
