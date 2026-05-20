"""
Google Slides Sync — detect new slides, download thumbnails, update vision_index.json.

Usage:
    python sync.py
    python sync.py --presentation-id <SLIDES_ID>

Required env vars:
    ANTHROPIC_API_KEY
    GOOGLE_SLIDES_ID              (or pass via --presentation-id)
    GOOGLE_SERVICE_ACCOUNT_JSON   (contents of service account JSON key)
                                   OR place service_account.json in this folder

The script is fully incremental — already-indexed slides are skipped.
Safe to re-run at any time.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_BASE = Path(__file__).parent
VISION_INDEX_PATH = _BASE / "vision_index.json"
SLIDES_DIR = _BASE / "slides_png"


# ── Google auth ───────────────────────────────────────────────────────────────

def _get_credentials():
    try:
        import google.oauth2.service_account as sa
        import google.auth.transport.requests as ga_req
    except ImportError:
        print("ERROR: Google libraries not installed.")
        print("       Run: pip install google-api-python-client google-auth")
        sys.exit(1)

    scopes = [
        "https://www.googleapis.com/auth/presentations.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if json_str:
        info = json.loads(json_str)
        creds = sa.Credentials.from_service_account_info(info, scopes=scopes)
    else:
        key_file = _BASE / "service_account.json"
        if not key_file.exists():
            print("ERROR: No Google credentials found.")
            print("       Set GOOGLE_SERVICE_ACCOUNT_JSON env var, or place service_account.json here.")
            sys.exit(1)
        creds = sa.Credentials.from_service_account_file(str(key_file), scopes=scopes)

    # Refresh to get a valid access token
    creds.refresh(ga_req.Request())
    return creds


# ── Slides API helpers ────────────────────────────────────────────────────────

def _list_presentation_slides(presentation_id: str, creds) -> list[dict]:
    """Return [{slide_number, page_id}] for every slide in the deck."""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("ERROR: Run: pip install google-api-python-client")
        sys.exit(1)

    service = build("slides", "v1", credentials=creds, cache_discovery=False)
    presentation = service.presentations().get(presentationId=presentation_id).execute()
    return [
        {"slide_number": str(i), "page_id": slide["objectId"]}
        for i, slide in enumerate(presentation.get("slides", []), start=1)
    ]


def _download_thumbnail(presentation_id: str, page_id: str, slide_number: int, creds) -> Path:
    """Download a slide thumbnail via the Slides API and save it to slides_png/."""
    try:
        from googleapiclient.discovery import build
        import google.auth.transport.requests as ga_req
    except ImportError:
        print("ERROR: Run: pip install google-api-python-client google-auth")
        sys.exit(1)

    service = build("slides", "v1", credentials=creds, cache_discovery=False)
    result = service.presentations().pages().getThumbnail(
        presentationId=presentation_id,
        pageObjectId=page_id,
        **{"thumbnailProperties.thumbnailSize": "LARGE"},
    ).execute()

    content_url = result["contentUrl"]
    SLIDES_DIR.mkdir(exist_ok=True)
    dest = SLIDES_DIR / f"slide-{slide_number:03d}.png"

    # The content URL may need a fresh Bearer token
    try:
        req = urllib.request.Request(content_url)
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
    except Exception:
        # Retry with explicit auth header
        creds.refresh(ga_req.Request())
        req = urllib.request.Request(content_url, headers={"Authorization": f"Bearer {creds.token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())

    return dest


# ── Main sync logic ───────────────────────────────────────────────────────────

def sync(presentation_id: str) -> int:
    """
    Sync new slides from Google Slides into vision_index.json.
    Returns number of new slides indexed.
    """
    # Load existing index to know what's already done
    existing_filenames: set[str] = set()
    if VISION_INDEX_PATH.exists():
        for entry in json.loads(VISION_INDEX_PATH.read_text()):
            fn = entry.get("filename", "")
            if fn:
                existing_filenames.add(fn)
    print(f"  Existing index  : {len(existing_filenames)} slides")

    # List all slides in the Google Slides deck
    print("  Connecting to Google Slides API...")
    creds = _get_credentials()
    all_slides = _list_presentation_slides(presentation_id, creds)
    print(f"  Deck total      : {len(all_slides)} slides")

    # Find slides not yet indexed
    new_slides = []
    for s in all_slides:
        filename = f"slide-{int(s['slide_number']):03d}.png"
        if filename not in existing_filenames:
            new_slides.append(s)

    if not new_slides:
        print("  ✓ Already up to date — nothing to do.")
        return 0

    print(f"  New slides      : {len(new_slides)}")

    # Download thumbnails for new slides
    downloaded: list[Path] = []
    for s in new_slides:
        num = int(s["slide_number"])
        local = SLIDES_DIR / f"slide-{num:03d}.png"
        if local.exists():
            downloaded.append(local)
            continue
        try:
            print(f"  Downloading slide {num:03d}...", end="", flush=True)
            path = _download_thumbnail(presentation_id, s["page_id"], num, creds)
            downloaded.append(path)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({e})")

    if not downloaded:
        print("  No new slides could be downloaded.")
        return 0

    # Run vision ingest — resume=True means it skips already-indexed slides
    from vision_ingest import run_vision_ingest
    print(f"\n  Running vision ingest on {len(downloaded)} new slides...\n")
    run_vision_ingest(str(SLIDES_DIR), resume=True)

    return len(downloaded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Google Slides → vision_index.json")
    parser.add_argument(
        "--presentation-id",
        default=os.getenv("GOOGLE_SLIDES_ID"),
        help="Google Slides presentation ID (or set GOOGLE_SLIDES_ID env var)",
    )
    args = parser.parse_args()

    if not args.presentation_id:
        print("ERROR: --presentation-id or GOOGLE_SLIDES_ID env var required.")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print(f"\n  Klimt Slide Sync\n  Deck ID: {args.presentation_id}\n")
    count = sync(args.presentation_id)
    print(f"\n  Done. {count} new slides indexed.")
