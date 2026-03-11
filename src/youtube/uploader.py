"""
YouTube Uploader - Uploads videos to YouTube with metadata
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import http.client
import httplib2

from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from .auth import YouTubeAuth
from .metadata import MetadataGenerator
from .drive_uploader import DriveUploader
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Retry configuration
MAX_RETRIES = 10
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError, http.client.NotConnected,
                        http.client.IncompleteRead, http.client.ImproperConnectionState,
                        http.client.CannotSendRequest, http.client.CannotSendHeader,
                        http.client.ResponseNotReady, http.client.BadStatusLine)
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]


@dataclass
class UploadResult:
    """Result of video upload"""
    success: bool
    video_id: str
    video_url: str
    title: str
    error: Optional[str] = None


class YouTubeUploader:
    """
    Uploads videos to YouTube with full metadata support.

    Features:
    - Resumable uploads for large files
    - Automatic retry on failure
    - Custom thumbnail upload
    - Metadata optimization
    """

    def __init__(
        self,
        auth: YouTubeAuth = None,
        metadata_generator: MetadataGenerator = None
    ):
        """
        Initialize YouTube uploader.

        Args:
            auth: YouTubeAuth instance (creates new if not provided)
            metadata_generator: MetadataGenerator instance
        """
        self.auth = auth or YouTubeAuth()
        self.metadata_gen = metadata_generator or MetadataGenerator()
        self.drive_uploader = DriveUploader(auth=self.auth)

        logger.info("YouTubeUploader initialized")

    def upload(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list = None,
        category_id: str = "25",
        privacy_status: str = "public",
        thumbnail_path: str = None,
        made_for_kids: bool = False
    ) -> UploadResult:
        """
        Upload a video to YouTube.

        Args:
            video_path: Path to video file
            title: Video title
            description: Video description
            tags: List of tags
            category_id: YouTube category ID
            privacy_status: "public", "private", or "unlisted"
            thumbnail_path: Path to thumbnail image
            made_for_kids: Whether video is made for kids

        Returns:
            UploadResult object
        """
        # Validate video file
        video_file = Path(video_path)
        if not video_file.exists():
            return UploadResult(
                success=False,
                video_id="",
                video_url="",
                title=title,
                error=f"Video file not found: {video_path}"
            )

        # Authenticate
        youtube = self.auth.get_service()
        if not youtube:
            return UploadResult(
                success=False,
                video_id="",
                video_url="",
                title=title,
                error="Failed to authenticate with YouTube"
            )

        try:
            # Prepare metadata
            body = {
                "snippet": {
                    "title": title[:100],  # YouTube limit
                    "description": description[:5000],  # YouTube limit
                    "tags": tags[:500] if tags else [],  # YouTube limit
                    "categoryId": category_id,
                    "defaultLanguage": "en",
                    "defaultAudioLanguage": "en"
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": made_for_kids,
                    "embeddable": True,
                    "publicStatsViewable": True
                }
            }

            # Create media upload
            media = MediaFileUpload(
                str(video_path),
                chunksize=10 * 1024 * 1024,  # 10MB chunks
                resumable=True,
                mimetype="video/*"
            )

            # Create upload request
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )

            logger.info(f"Starting upload: {title}")

            # Execute upload with retry
            response = self._resumable_upload(request)

            if response:
                video_id = response.get("id", "")
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                logger.info(f"Upload successful: {video_url}")

                # Upload thumbnail if provided
                if thumbnail_path and Path(thumbnail_path).exists():
                    self._upload_thumbnail(youtube, video_id, thumbnail_path)

                return UploadResult(
                    success=True,
                    video_id=video_id,
                    video_url=video_url,
                    title=title
                )
            else:
                return UploadResult(
                    success=False,
                    video_id="",
                    video_url="",
                    title=title,
                    error="Upload failed - no response"
                )

        except HttpError as e:
            error_msg = f"HTTP error {e.resp.status}: {e.content.decode()}"
            logger.error(f"Upload failed: {error_msg}")
            return UploadResult(
                success=False,
                video_id="",
                video_url="",
                title=title,
                error=error_msg
            )

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return UploadResult(
                success=False,
                video_id="",
                video_url="",
                title=title,
                error=str(e)
            )

    def _resumable_upload(self, request) -> Optional[Dict]:
        """Execute resumable upload with retry logic"""
        response = None
        error = None
        retry = 0

        while response is None:
            try:
                logger.info("Uploading file...")
                status, response = request.next_chunk()

                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Upload progress: {progress}%")

            except HttpError as e:
                if e.resp.status in RETRIABLE_STATUS_CODES:
                    error = f"Retriable HTTP error {e.resp.status}: {e.content}"
                else:
                    raise

            except RETRIABLE_EXCEPTIONS as e:
                error = f"Retriable error: {e}"

            if error:
                retry += 1
                if retry > MAX_RETRIES:
                    logger.error(f"Max retries exceeded. Last error: {error}")
                    return None

                sleep_seconds = 2 ** retry
                logger.warning(f"Error occurred, retrying in {sleep_seconds}s: {error}")
                time.sleep(sleep_seconds)
                error = None

        return response

    def _upload_thumbnail(
        self,
        youtube,
        video_id: str,
        thumbnail_path: str
    ) -> bool:
        """Upload custom thumbnail for video"""
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path)
            ).execute()

            logger.info(f"Thumbnail uploaded for video: {video_id}")
            return True

        except HttpError as e:
            # Thumbnail upload requires channel verification
            if "forbidden" in str(e).lower():
                logger.warning(
                    "Thumbnail upload failed - channel may need verification. "
                    "Visit: https://www.youtube.com/verify"
                )
            else:
                logger.error(f"Thumbnail upload failed: {e}")
            return False

    def upload_with_metadata(
        self,
        video_path: str,
        headlines: list,
        sources: list = None,
        language: str = "en",
        date: str = None,
        thumbnail_path: str = None,
        privacy_status: str = "public",
        pdf_path: str = None
    ) -> UploadResult:
        """
        Upload video with auto-generated metadata.
        If pdf_path is provided, uploads PDF to Google Drive and includes the
        download link in the video description.

        Args:
            video_path: Path to video file
            headlines: List of news headlines
            sources: List of news sources
            language: Video language
            date: Video date
            thumbnail_path: Path to thumbnail
            privacy_status: Privacy status
            pdf_path: Path to PDF study notes file (optional)

        Returns:
            UploadResult object
        """
        # Upload PDF to Google Drive and get shareable link
        pdf_link = None
        pdf_filename = None

        if pdf_path:
            from pathlib import Path as _Path
            pdf_filename = _Path(pdf_path).name
            logger.info(f"Uploading PDF study notes to Google Drive: {pdf_filename}")
            pdf_link = self.drive_uploader.upload_pdf(pdf_path, date_str=date)

            if pdf_link:
                logger.info(f"PDF uploaded to Drive: {pdf_link}")
            else:
                logger.info("Drive upload unavailable - PDF filename will be noted in description")

        # Generate metadata with PDF link
        metadata = self.metadata_gen.generate(
            headlines=headlines,
            date=date,
            language=language,
            sources=sources,
            pdf_link=pdf_link,
            pdf_filename=pdf_filename
        )

        return self.upload(
            video_path=video_path,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            category_id=metadata["category_id"],
            privacy_status=privacy_status,
            thumbnail_path=thumbnail_path,
            made_for_kids=metadata["made_for_kids"]
        )

    def get_upload_status(self, video_id: str) -> Dict[str, Any]:
        """
        Get processing status of uploaded video.

        Args:
            video_id: YouTube video ID

        Returns:
            Status information dict
        """
        youtube = self.auth.get_service()
        if not youtube:
            return {"error": "Not authenticated"}

        try:
            response = youtube.videos().list(
                part="status,processingDetails",
                id=video_id
            ).execute()

            if response.get("items"):
                item = response["items"][0]
                return {
                    "upload_status": item["status"].get("uploadStatus"),
                    "privacy_status": item["status"].get("privacyStatus"),
                    "processing_status": item.get("processingDetails", {}).get("processingStatus"),
                    "processing_progress": item.get("processingDetails", {}).get("processingProgress", {})
                }

        except Exception as e:
            logger.error(f"Failed to get upload status: {e}")

        return {"error": "Failed to get status"}


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Uploader CLI")
    parser.add_argument("--video", type=str, help="Video file to upload")
    parser.add_argument("--title", type=str, help="Video title")
    parser.add_argument("--description", type=str, default="", help="Video description")
    parser.add_argument("--tags", type=str, nargs="+", help="Video tags")
    parser.add_argument("--thumbnail", type=str, help="Thumbnail image")
    parser.add_argument("--private", action="store_true", help="Upload as private")
    parser.add_argument("--test", action="store_true", help="Test authentication only")

    args = parser.parse_args()

    uploader = YouTubeUploader()

    if args.test:
        print("\n=== Testing YouTube Authentication ===\n")
        if uploader.auth.authenticate():
            info = uploader.auth.get_channel_info()
            if info:
                print(f"Authenticated as: {info['title']}")
                print(f"Channel ID: {info['id']}")
                print("Ready to upload!")
            else:
                print("Authenticated but couldn't get channel info")
        else:
            print("Authentication failed!")

    elif args.video and args.title:
        privacy = "private" if args.private else "public"

        result = uploader.upload(
            video_path=args.video,
            title=args.title,
            description=args.description,
            tags=args.tags or [],
            privacy_status=privacy,
            thumbnail_path=args.thumbnail
        )

        if result.success:
            print(f"\nUpload successful!")
            print(f"Video ID: {result.video_id}")
            print(f"URL: {result.video_url}")
        else:
            print(f"\nUpload failed: {result.error}")

    else:
        parser.print_help()
