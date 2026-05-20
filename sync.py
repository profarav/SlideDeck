"""
Google Slides Sync — stable sync using page_id as the permanent slide identifier.

Every slide in Google Slides has an objectId (page_id) that never changes,
even when slides are inserted, moved, or reordered. We use page_id to:
  - Detect genuinely NEW slides (page_id not yet in index)
  - Detect MOVED slides (page_id exists but slide_number changed)
  - Avoid re-processing slides that already exist

On first run: bootstraps page_id into existing index entries by matching
position (assumes deck order hasn't changed since original export).

Usage:
    python sync.py
    python sync.py --presentation-id <SLIDES_ID>

Required env vars:
    ANTHROPIC_API_KEY
    GOOGLE_SLIDES_ID              (or pass via --presentation-id)
    GOOGLE_SERVICE_ACCOUNT_JSON   (contents of service account JSON key)
                                   OR place service_account.json in this folder
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
        print("ERROR: Run: pip install -r requirements-sync.txt")
        sys.exit(1)

    scopes = [
        "https://www.googleapis.com/auth/presentations.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if json_str:
        creds = sa.Credentials.from_service_account_info(json.loads(json_str), scopes=scopes)
    else:
        key_file = _BASE / "service_account.json"
        if not key_file.exists():
            print("ERROR: No Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON or place service_account.json here.")
            sys.exit(1)
        creds = sa.Credentials.from_service_account_file(str(key_file), scopes=scopes)

    creds.refresh(ga_req.Request())
    return creds


# ── Slides API ────────────────────────────────────────────────────────────────

def _list_deck_slides(presentation_id: str, creds) -> list[dict]:
    """
    Return the CURRENT state of the deck as:
      [{"slide_number": "1", "page_id": "g1a2b3..."}, ...]
    ordered by current position. slide_number = current position (1-based).
    """
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("ERROR: Run: pip install -r requirements-sync.txt")
        sys.exit(1)

    service = build("slides", "v1", credentials=creds, cache_discovery=False)
    presentation = service.presentations().get(presentationId=presentation_id).execute()
    return [
        {"slide_number": str(i), "page_id": slide["objectId"]}
        for i, slide in enumerate(presentation.get("slides", []), start=1)
    ]


def _download_thumbnail(presentation_id: str, page_id: str, dest: Path, creds) -> None:
    """Download a slide thumbnail at LARGE resolution and save to dest."""
    try:
        from googleapiclient.discovery import build
        import google.auth.transport.requests as ga_req
    except ImportError:
        print("ERROR: Run: pip install -r requirements-sync.txt")
        sys.exit(1)

    service = build("slides", "v1", credentials=creds, cache_discovery=False)
    result = service.presentations().pages().getThumbnail(
        presentationId=presentation_id,
        pageObjectId=page_id,
        **{"thumbnailProperties.thumbnailSize": "LARGE"},
    ).execute()

    content_url = result["contentUrl"]
    SLIDES_DIR.mkdir(exist_ok=True)

    # Try unauthenticated first (content URLs are often pre-signed)
    try:
        with urllib.request.urlopen(content_url, timeout=30) as r:
            dest.write_bytes(r.read())
    except Exception:
        # Retry with Bearer token
        creds.refresh(ga_req.Request())
        req = urllib.request.Request(content_url, headers={"Authorization": f"Bearer {creds.token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())


# ── Bootstrap: add page_id to existing entries ───────────────────────────────

def _bootstrap_page_ids(
    index_entries: list[dict],
    deck_slides: list[dict],
) -> list[dict]:
    """
    First-run only: the existing index has no page_id fields.
    Match index entries to deck slides 1:1 by current position and inject page_id.
    Safe assumption: the original PNG export was in deck order.
    """
    deck_by_num = {s["slide_number"]: s["page_id"] for s in deck_slides}
    updated = 0
    for entry in index_entries:
        if not entry.get("page_id"):
            page_id = deck_by_num.get(entry["slide_number"])
            if page_id:
                entry["page_id"] = page_id
                updated += 1
    if updated:
        print(f"  Bootstrap: added page_id to {updated} existing index entries.")
    return index_entries


# ── Main sync ─────────────────────────────────────────────────────────────────

def sync(presentation_id: str) -> int:
    """
    Sync new and moved slides from Google Slides into vision_index.json.
    Returns number of new slides indexed.
    """
    # Load existing index
    index_entries: list[dict] = []
    if VISION_INDEX_PATH.exists():
        index_entries = json.loads(VISION_INDEX_PATH.read_text())
    print(f"  Existing index  : {len(index_entries)} slides")

    # Fetch current deck state
    print("  Connecting to Google Slides API...")
    creds = _get_credentials()
    deck_slides = _list_deck_slides(presentation_id, creds)
    print(f"  Deck total      : {len(deck_slides)} slides")

    # Bootstrap page_ids into entries that don't have them yet (first run)
    needs_bootstrap = any(not e.get("page_id") for e in index_entries)
    if needs_bootstrap:
        index_entries = _bootstrap_page_ids(index_entries, deck_slides)

    # Build lookup: page_id → index entry
    by_page_id: dict[str, dict] = {e["page_id"]: e for e in index_entries if e.get("page_id")}

    # ── Detect new slides ──────────────────────────────────────────────────────
    new_deck_slides = [ds for ds in deck_slides if ds["page_id"] not in by_page_id]

    # ── Detect moved slides (same page_id, different position) ────────────────
    moved = []
    for ds in deck_slides:
        if ds["page_id"] in by_page_id:
            entry = by_page_id[ds["page_id"]]
            if entry.get("slide_number") != ds["slide_number"]:
                moved.append((entry, ds["slide_number"]))

    # Update slide_number for moved slides
    if moved:
        print(f"  Moved slides    : {len(moved)} (updating positions)")
        for entry, new_num in moved:
            old_num = entry["slide_number"]
            entry["slide_number"] = new_num
            print(f"    Slide {old_num:>4} → {new_num}")

    if not new_deck_slides:
        print("  New slides      : 0")
        if moved:
            # Still need to save the updated positions
            _save_index(index_entries, deck_slides, by_page_id)
            print("  ✓ Positions updated in index.")
        else:
            print("  ✓ Already up to date — nothing to do.")
        return 0

    print(f"  New slides      : {len(new_deck_slides)}")

    # Download thumbnails for new slides
    downloaded: list[Path] = []
    for ds in new_deck_slides:
        num = int(ds["slide_number"])
        # Use slide_number for filename — it reflects current position
        dest = SLIDES_DIR / f"slide-{num:03d}.png"

        # Avoid overwriting an existing file (another slide might be at this position)
        if dest.exists():
            # Suffix with page_id short hash to avoid collision
            short_id = ds["page_id"][-6:]
            dest = SLIDES_DIR / f"slide-{num:03d}-{short_id}.png"

        print(f"  Downloading slide {num:03d}...", end="", flush=True)
        try:
            _download_thumbnail(presentation_id, ds["page_id"], dest, creds)
            ds["_local_path"] = dest   # stash for vision ingest
            downloaded.append(dest)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({e})")

    if not downloaded:
        print("  No new slides could be downloaded.")
        return 0

    # Run vision ingest on only the new slide files
    _run_vision_ingest_on_files(downloaded, index_entries)

    # Re-read updated index (vision_ingest wrote it) and inject page_ids for new entries
    updated_entries = json.loads(VISION_INDEX_PATH.read_text())
    updated_by_page_id: dict[str, dict] = {e["page_id"]: e for e in updated_entries if e.get("page_id")}

    # The new entries from vision_ingest won't have page_id yet — inject them
    filename_to_page_id = {
        ds["_local_path"].name: ds["page_id"]
        for ds in new_deck_slides
        if "_local_path" in ds
    }
    for entry in updated_entries:
        fn = entry.get("filename", "")
        if not entry.get("page_id") and fn in filename_to_page_id:
            entry["page_id"] = filename_to_page_id[fn]
            updated_by_page_id[entry["page_id"]] = entry

    # Save final ordered index
    _save_index(updated_entries, deck_slides, updated_by_page_id)
    return len(downloaded)


def _run_vision_ingest_on_files(paths: list[Path], existing_entries: list[dict]) -> None:
    """
    Run vision ingest on a specific list of files.
    We temporarily symlink them into a staging directory so vision_ingest
    can operate on just those files without re-scanning the full slides_png dir.
    """
    import tempfile
    import shutil
    from vision_ingest import run_vision_ingest

    # Write existing entries to a temp index so resume works correctly
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_index = tmp_dir / "vision_index.json"
    tmp_index.write_text(json.dumps(existing_entries))

    # Copy new files to a staging dir so vision_ingest only sees them
    staging = tmp_dir / "staging"
    staging.mkdir()
    for p in paths:
        shutil.copy2(p, staging / p.name)

    # Override the global index path temporarily, then run
    import vision_ingest as vi
    original_path = vi.VISION_INDEX_PATH
    vi.VISION_INDEX_PATH = str(VISION_INDEX_PATH)
    try:
        run_vision_ingest(str(staging), resume=True)
    finally:
        vi.VISION_INDEX_PATH = original_path

    shutil.rmtree(tmp_dir)


def _save_index(entries: list[dict], deck_slides: list[dict], by_page_id: dict[str, dict]) -> None:
    """Save index ordered by current deck position."""
    ordered = []
    for ds in deck_slides:
        entry = by_page_id.get(ds["page_id"])
        if entry:
            ordered.append(entry)

    # Append any entries not in the deck (shouldn't happen, but safety net)
    deck_page_ids = {ds["page_id"] for ds in deck_slides}
    for entry in entries:
        if entry.get("page_id") not in deck_page_ids:
            ordered.append(entry)

    VISION_INDEX_PATH.write_text(json.dumps(ordered, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Google Slides → vision_index.json")
    parser.add_argument("--presentation-id", default=os.getenv("GOOGLE_SLIDES_ID"))
    args = parser.parse_args()

    if not args.presentation_id:
        print("ERROR: --presentation-id or GOOGLE_SLIDES_ID env var required.")
        sys.exit(1)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print(f"\n  Klimt Slide Sync\n  Deck ID: {args.presentation_id}\n")
    count = sync(args.presentation_id)
    print(f"\n  Done. {count} new slides indexed.\n")
