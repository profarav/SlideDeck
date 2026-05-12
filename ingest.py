"""
Reads Master Sheet.xlsx and writes a clean JSON index file (slide_index.json).
No vector DB or model download required.

Run once (or whenever the sheet changes):
    python ingest.py --xlsx "/path/to/Master Sheet.xlsx"
"""

import argparse
import json
import re
import openpyxl
from tags import normalize_industry, normalize_visual_style

INDEX_PATH = "./slide_index.json"

_HEADER_VALUES = {
    "slide #", "client / project", "category", "service", "appendix intro",
}


def _clean_slide_num(val) -> str:
    """Normalize slide number, handling dates Excel misparses (e.g. '7-8' → datetime)."""
    if val is None:
        return ""
    s = str(val)
    date_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if date_match:
        _, month, day = date_match.groups()
        m, d = int(month), int(day)
        return f"{m}-{d}" if m != d else str(m)
    s = re.sub(r"\.0$", "", s.strip())
    s = s.replace("–", "-")
    return s


def load_slides(xlsx_path: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.worksheets[0]

    slides: list[dict] = []
    for row in ws.iter_rows(values_only=True):
        slide_raw, col2, col3, col4, col5 = (row[i] if len(row) > i else None for i in range(5))

        if not slide_raw or not col2:
            continue
        if str(col2).strip().lower() in _HEADER_VALUES:
            continue

        slide_num = _clean_slide_num(slide_raw)
        if not slide_num:
            continue

        if col5:
            # Case study section: Client | Industry | Visual Style | Content
            client = str(col2).strip()
            industry_raw = str(col3).strip() if col3 else ""
            visual_raw = str(col4).strip() if col4 else ""
            content = str(col5).strip()
        elif col4:
            # Intro / service section: Category | Description | Visual Elements
            client = str(col2).strip()
            industry_raw = "General Agency"
            visual_raw = str(col4).strip()
            content = str(col3).strip() if col3 else ""
        else:
            continue

        if client.lower() in _HEADER_VALUES or industry_raw.lower() in _HEADER_VALUES:
            continue

        # Strip "+1", "+2" annotation suffixes added by the sheet author
        content = re.sub(r"\s*\+\d+\s*$", "", content).strip()
        visual_raw = re.sub(r"\s*\+\d+\s*$", "", visual_raw).strip()

        slides.append({
            "slide_number": slide_num,
            "client": client,
            "industry_raw": industry_raw,
            "industry": normalize_industry(f"{client} {industry_raw}"),
            "visual_style_raw": visual_raw,
            "visual_style": normalize_visual_style(visual_raw),
            "content": content,
            # Combined text blob used for TF-IDF and Claude ranking
            "text": f"{client} {industry_raw} {visual_raw} {content}",
        })

    return slides


def build_index(xlsx_path: str) -> int:
    slides = load_slides(xlsx_path)
    if not slides:
        raise ValueError("No slides parsed — check the xlsx path.")
    with open(INDEX_PATH, "w") as f:
        json.dump(slides, f, indent=2)
    return len(slides)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Master Sheet.xlsx → slide_index.json")
    parser.add_argument("--xlsx", default="/Users/aravlohe/Documents/Master Sheet.xlsx")
    args = parser.parse_args()

    count = build_index(args.xlsx)
    print(f"✓ Indexed {count} slides → {INDEX_PATH}")
