"""
Google Drive Uploader - Uploads PDF study notes to Google Drive and returns shareable link.

Uses the same OAuth credentials as YouTube (both are Google services).
Requires 'https://www.googleapis.com/auth/drive.file' scope in addition to YouTube scopes.
Falls back gracefully if Drive upload fails.
"""

import os
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from .auth import YouTubeAuth
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Folder name to store PDFs on Google Drive
DRIVE_FOLDER_NAME = "Current Affairs Academy - Study Notes"


class DriveUploader:
    """
    Uploads PDF files to Google Drive using the same Google OAuth credentials
    as YouTube. Returns a shareable link that can be added to the YouTube description.

    The upload is optional - if it fails (e.g., Drive scope not authorized),
    it returns None gracefully and the pipeline continues.
    """

    def __init__(self, auth: YouTubeAuth = None):
        """
        Initialize Drive uploader.

        Args:
            auth: YouTubeAuth instance (creates new if not provided)
        """
        self.auth = auth or YouTubeAuth()
        self._drive_service = None
        self._folder_id: Optional[str] = None

    def _get_drive_service(self):
        """Get authenticated Google Drive API service using existing credentials."""
        if self._drive_service:
            return self._drive_service

        try:
            # Get credentials from the YouTube auth (same Google account)
            credentials = self.auth.credentials
            if not credentials:
                # Try to authenticate
                self.auth.authenticate()
                credentials = self.auth.credentials

            if not credentials:
                logger.warning("No Google credentials available for Drive upload")
                return None

            self._drive_service = build("drive", "v3", credentials=credentials)
            logger.info("Google Drive service initialized")
            return self._drive_service

        except Exception as e:
            logger.warning(f"Failed to initialize Drive service: {e}")
            return None

    def _get_or_create_folder(self, drive_service) -> Optional[str]:
        """Get or create the Academy study notes folder on Google Drive."""
        if self._folder_id:
            return self._folder_id

        try:
            # Search for existing folder
            query = (
                f"name='{DRIVE_FOLDER_NAME}' and "
                "mimeType='application/vnd.google-apps.folder' and "
                "trashed=false"
            )
            results = drive_service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()

            files = results.get("files", [])
            if files:
                self._folder_id = files[0]["id"]
                logger.info(f"Found existing Drive folder: {DRIVE_FOLDER_NAME}")
                return self._folder_id

            # Create the folder
            folder_metadata = {
                "name": DRIVE_FOLDER_NAME,
                "mimeType": "application/vnd.google-apps.folder"
            }
            folder = drive_service.files().create(
                body=folder_metadata,
                fields="id"
            ).execute()

            self._folder_id = folder.get("id")
            logger.info(f"Created Drive folder: {DRIVE_FOLDER_NAME} (ID: {self._folder_id})")

            # Make the folder publicly viewable
            self._set_public_permission(drive_service, self._folder_id)

            return self._folder_id

        except HttpError as e:
            if "insufficientPermissions" in str(e) or "forbidden" in str(e).lower():
                logger.warning(
                    "Drive folder creation failed: insufficient permissions. "
                    "Add 'drive.file' scope to OAuth and re-authenticate."
                )
            else:
                logger.warning(f"Failed to get/create Drive folder: {e}")
            return None

        except Exception as e:
            logger.warning(f"Drive folder error: {e}")
            return None

    def _set_public_permission(self, drive_service, file_id: str) -> bool:
        """Make a Drive file/folder publicly readable."""
        try:
            permission = {
                "type": "anyone",
                "role": "reader"
            }
            drive_service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
            return True
        except Exception as e:
            logger.warning(f"Failed to set public permission: {e}")
            return False

    def upload_pdf(self, pdf_path: str, date_str: str = None) -> Optional[str]:
        """
        Upload a PDF file to Google Drive and return a shareable download link.

        Args:
            pdf_path: Local path to the PDF file
            date_str: Date string for the file name

        Returns:
            Shareable Google Drive link, or None if upload failed
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            logger.warning(f"PDF file not found for Drive upload: {pdf_path}")
            return None

        drive_service = self._get_drive_service()
        if not drive_service:
            return None

        try:
            folder_id = self._get_or_create_folder(drive_service)

            # File metadata
            file_metadata = {
                "name": pdf_file.name,
                "mimeType": "application/pdf"
            }
            if folder_id:
                file_metadata["parents"] = [folder_id]

            # Upload the PDF
            media = MediaFileUpload(
                str(pdf_path),
                mimetype="application/pdf",
                resumable=True
            )

            logger.info(f"Uploading PDF to Google Drive: {pdf_file.name}")

            uploaded_file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink, webContentLink"
            ).execute()

            file_id = uploaded_file.get("id")

            # Make the file publicly readable
            self._set_public_permission(drive_service, file_id)

            # Return the direct download link
            download_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

            logger.info(f"PDF uploaded to Drive: {download_link}")
            return download_link

        except HttpError as e:
            if "insufficientPermissions" in str(e) or "forbidden" in str(e).lower():
                logger.warning(
                    "Drive upload failed: insufficient permissions. "
                    "Re-run OAuth with drive.file scope to enable PDF uploads."
                )
            else:
                logger.warning(f"Drive upload HTTP error: {e}")
            return None

        except Exception as e:
            logger.warning(f"Drive upload failed: {e}")
            return None

    @staticmethod
    def format_description_section(pdf_link: Optional[str], pdf_filename: str = "") -> str:
        """
        Format the PDF section for inclusion in YouTube video description.

        Args:
            pdf_link: Google Drive link (or None if upload failed)
            pdf_filename: Local filename of the PDF (used as fallback)

        Returns:
            Formatted string to append to YouTube description
        """
        lines = [
            "",
            "━" * 50,
            "📄 FREE PDF STUDY NOTES - CURRENT AFFAIRS ACADEMY",
            "━" * 50,
            "",
            "✅ Download the Complete Study Notes PDF for this video:",
            "   • Full detailed coverage (more than the video!)",
            "   • In-depth analysis with background context",
            "   • Important Terms & Definitions",
            "   • UPSC Prelims & Mains relevance tags",
            "   • Quick Revision section",
            "   • 20+ Practice Questions (MCQ + Descriptive)",
            "",
        ]

        if pdf_link:
            lines.append(f"🔗 Download PDF: {pdf_link}")
        else:
            lines.append("🔗 PDF Study Notes: Available in the description (see pinned comment)")
            if pdf_filename:
                lines.append(f"   File: {pdf_filename}")

        lines.extend([
            "",
            "━" * 50,
            ""
        ])

        return "\n".join(lines)
