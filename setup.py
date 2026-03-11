"""
Setup Script for AI History Video Generator

This script helps you set up and configure the project.
Run: python setup.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_step(number, text):
    """Print a step."""
    print(f"\n[Step {number}] {text}")
    print("-" * 40)


def run_command(cmd, description=""):
    """Run a command and return success status."""
    if description:
        print(f"  {description}...")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  Python 3.9+ required. Found: {version.major}.{version.minor}")
        return False


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    if shutil.which("ffmpeg"):
        print("  FFmpeg: Installed")
        return True
    else:
        print("  FFmpeg: NOT FOUND")
        print("    Install from: https://ffmpeg.org/download.html")
        return False


def create_directories():
    """Create required directories."""
    dirs = [
        "config",
        "data",
        "logs",
        "assets/avatars",
        "assets/backgrounds",
        "assets/fonts",
        "output/audio",
        "output/videos",
        "output/thumbnails",
        "output/notes",
        "output/temp"
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}/")


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")

    if env_file.exists():
        print("  .env file already exists")
        return

    env_content = """# AI History Video Generator - Environment Variables

# Groq API Key (free at https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
    env_file.write_text(env_content)
    print("  Created .env file")
    print("  Please edit .env and add your Groq API key")


def install_dependencies():
    """Install Python dependencies."""
    print("  This may take several minutes...\n")

    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("  Failed to install dependencies")
        return False

    print("  Dependencies installed successfully")
    return True


def check_groq_api():
    """Check if Groq API key is configured."""
    api_key = os.getenv("GROQ_API_KEY", "")

    if api_key and api_key != "your_groq_api_key_here":
        print("  Groq API: Configured")
        return True
    else:
        print("  Groq API: NOT CONFIGURED")
        print("    Get free API key at: https://console.groq.com/")
        print("    Then add to .env file: GROQ_API_KEY=your_key")
        return False


def check_youtube_credentials():
    """Check if YouTube credentials exist."""
    creds_file = Path("config/client_secrets.json")

    if creds_file.exists():
        print("  YouTube credentials: Found")
        return True
    else:
        print("  YouTube credentials: NOT FOUND")
        print("    Setup instructions:")
        print("    1. Go to https://console.cloud.google.com/")
        print("    2. Create a project")
        print("    3. Enable 'YouTube Data API v3'")
        print("    4. Create OAuth 2.0 credentials (Desktop app)")
        print("    5. Download JSON and save as config/client_secrets.json")
        return False


def check_avatar():
    """Check if avatar image exists."""
    avatar_path = Path("assets/avatars/news_anchor.png")

    if avatar_path.exists():
        print("  Avatar image: Found")
        return True
    else:
        print("  Avatar image: NOT FOUND")
        print("    Place your teacher avatar image at: assets/avatars/news_anchor.png")
        print("    (512x512 or larger, clear face, front-facing)")

        # Create placeholder
        try:
            from PIL import Image, ImageDraw

            size = 512
            img = Image.new('RGB', (size, size), color='#1a1a2e')
            draw = ImageDraw.Draw(img)
            draw.ellipse([size//2-80, size//3-80, size//2+80, size//3+80], fill='#4a4a6a')
            draw.rectangle([size//2-100, size//3+80, size//2+100, size], fill='#3a3a5a')
            img.save(str(avatar_path))
            print("    Created placeholder avatar (replace with real image)")
            return True
        except ImportError:
            print("    Pillow not installed yet, skipping placeholder")
            return False


def initialize_syllabus():
    """Initialize syllabus database from YAML."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.utils.database import Database
        from src.syllabus.syllabus_manager import SyllabusManager

        db = Database("data/history_tracker.db")
        mgr = SyllabusManager(
            syllabus_path="config/syllabus.yaml",
            db=db
        )

        progress = mgr.get_progress()
        print(f"  Syllabus initialized: {progress['total']} topics loaded")
        print(f"  Completed: {progress['completed']}/{progress['total']}")
        return True
    except Exception as e:
        print(f"  Failed to initialize syllabus: {e}")
        return False


def main():
    """Main setup function."""
    print_header("AI History Video Generator - Setup")
    print("  UPSC & State PSC - Complete History Course (180 Parts)")

    # Step 1: Check Python
    print_step(1, "Checking Python version")
    if not check_python():
        print("\nPlease install Python 3.9 or higher")
        return 1

    # Step 2: Check FFmpeg
    print_step(2, "Checking FFmpeg")
    ffmpeg_ok = check_ffmpeg()

    # Step 3: Create directories
    print_step(3, "Creating directories")
    create_directories()

    # Step 4: Create .env file
    print_step(4, "Creating configuration files")
    create_env_file()

    # Step 5: Install dependencies
    print_step(5, "Installing Python dependencies")
    response = input("  Install dependencies now? [Y/n]: ").strip().lower()
    deps_ok = True
    if response != 'n':
        deps_ok = install_dependencies()
        if not deps_ok:
            print("\nFailed to install dependencies. Try manually:")
            print("  pip install -r requirements.txt")

    # Step 6: Check avatar
    print_step(6, "Checking avatar image")
    try:
        avatar_ok = check_avatar()
    except Exception:
        avatar_ok = False
        print("  Skipped (will be created on first run)")

    # Step 7: Check API configurations
    print_step(7, "Checking API configurations")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    groq_ok = check_groq_api()
    youtube_ok = check_youtube_credentials()

    # Step 8: Initialize syllabus database
    print_step(8, "Initializing syllabus database")
    if deps_ok:
        syllabus_ok = initialize_syllabus()
    else:
        syllabus_ok = False
        print("  Skipped (dependencies not installed)")

    # Summary
    print_header("Setup Summary")

    status = {
        "Python 3.9+": True,
        "FFmpeg": ffmpeg_ok,
        "Directories": True,
        "Avatar Image": avatar_ok,
        "Groq API Key": groq_ok,
        "YouTube API": youtube_ok,
        "Syllabus DB": syllabus_ok
    }

    for item, ok in status.items():
        status_str = "OK" if ok else "NEEDS SETUP"
        print(f"  {item}: {status_str}")

    print("\n" + "-" * 40)

    if all(status.values()):
        print("\n  All set! You can now run:")
        print("    python main.py --test              # Test with Part 1 (private)")
        print("    python main.py                     # Generate next part (public)")
        print("    python main.py --progress          # Show course progress")
        print("\n  Or start the daily scheduler:")
        print("    python scheduler.py                # Run at 2 PM IST daily")
        print("    python scheduler.py --once         # Run once and exit")
    else:
        print("\n  Some items need configuration.")
        print("  Please complete the setup and run this script again.")

    print("\n" + "=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
