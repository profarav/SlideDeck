"""
Canonical tag lists and keyword-based normalization for industries and visual styles.
These are derived from the actual content in Master Sheet.xlsx.
"""

import re


def _contains_keyword(text: str, keyword: str) -> bool:
    """
    True if `keyword` appears in `text` as a whole token (bounded by string edges or
    non-alphanumeric chars). A plain substring test caused rampant false positives —
    "ai" matched maintain/email/detail, "hr" matched three/through, "vc" matched
    service, "action" matched transaction — which over-tagged unrelated slides. Internal
    punctuation in a keyword (e.g. "point-of-sale", "m.ink") is matched literally.
    """
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None


# ── Canonical Industry Tags ────────────────────────────────────────────────────

CANONICAL_INDUSTRIES = [
    "Finance & Wealth Management",
    "Fintech & Payments",
    "Finance & Trading",
    "Venture Capital",
    "Healthcare & HealthTech",
    "Mental Health & Wellness",
    "Consumer & E-commerce",
    "Fashion & Apparel",
    "B2B SaaS",
    "AI & Technology",
    "PropTech",
    "Branding & Design",
    "Education & EdTech",
    "Energy & Sustainability",
    "Cybersecurity",
    "HR Tech",
    "Web3 & Crypto",
    "Travel & Hospitality",
    "Sports & Athletics",
    "Data & Analytics",
    "Operations & Logistics",
    "Maritime",
    # The CLIENT is themselves an agency/studio (design, branding, marketing, growth,
    # social). Distinct from "General Agency", which means Klimt's OWN slides. Without
    # this bucket vision had nowhere to put peer agencies and scattered them into
    # "B2B SaaS" / "General Agency", which is why a design-agency prospect could not
    # find the design-agency work in the library.
    "Creative & Marketing Agency",
    "General Agency",
]

_INDUSTRY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Finance & Wealth Management",  ["wealth", "wealth mgmt", "wealth management", "evergreen"]),
    ("Fintech & Payments",           ["payments", "fintech", "banking", "point-of-sale", "clarity pay", "fasten", "savvy"]),
    ("Finance & Trading",            ["trading", "hedge fund", "equity", "long short", "1860"]),
    ("Venture Capital",              ["venture capital", "vc", "opal ventures"]),
    ("Healthcare & HealthTech",      ["healthcare", "healthtech", "health tech", "telemedicine", "medical", "health data", "clinical", "physician", "recuro", "connective health", "arlo", "sully"]),
    ("Mental Health & Wellness",     ["mental health", "wellness", "longevity", "recovery", "sobriety", "kindly", "monument", "super age"]),
    ("Consumer & E-commerce",        ["consumer", "e-comm", "ecommerce", "dtc", "retail", "brooklinen", "savage x fenty"]),
    ("Fashion & Apparel",            ["fashion", "apparel", "clothing", "luxury retail", "nonchalant", "plush", "nike"]),
    ("B2B SaaS",                     ["saas", "b2b", "invoice butler", "formspree", "fuel to fly", "codethread", "lumina", "kawin"]),
    ("AI & Technology",              ["ai", "artificial intelligence", "devtools", "enterprise ai", "video ai", "lighton", "humanfirst", "guru", "sketchpro", "pantera", "codegen"]),
    ("PropTech",                     ["proptech", "real estate", "property"]),
    ("Branding & Design",            ["branding", "logo", "identity", "stylescape", "sohva", "secludy", "sona padel"]),
    ("Education & EdTech",           ["edtech", "education", "nursing", "school", "crna", "diocese"]),
    ("Energy & Sustainability",      ["energy", "sustainability", "renewable", "solar", "wind"]),
    ("Cybersecurity",                ["cybersecurity", "security", "cyber", "cyber whyze"]),
    ("HR Tech",                      ["hr", "human resources", "hiring", "fractional", "techsta", "m.ink"]),
    ("Web3 & Crypto",                ["web3", "crypto", "blockchain", "token", "livepeer", "limitless"]),
    ("Travel & Hospitality",         ["travel", "hospitality", "event", "ticketing", "playstay"]),
    ("Sports & Athletics",           ["sports", "athletics", "fitness", "running", "atlas", "padres", "nike"]),
    ("Data & Analytics",             ["analytics", "adtech", "advertising", "cookie", "data visualization", "data platform", "data-driven"]),
    ("Operations & Logistics",       ["operations", "logistics", "ops", "ulta", "clickup"]),
    ("Maritime",                     ["maritime", "payroll", "seafare"]),
    ("General Agency",               ["general", "agency", "capabilities", "klimt"]),
]


def normalize_industry(raw: str) -> str:
    """Map a free-text industry string to the nearest canonical tag."""
    lower = raw.lower()
    for canonical, keywords in _INDUSTRY_KEYWORDS:
        if any(_contains_keyword(lower, k) for k in keywords):
            return canonical
    return "General Agency"


# ── Canonical Visual Style Tags ────────────────────────────────────────────────

CANONICAL_VISUAL_STYLES = [
    "Luxury",
    "Minimalist",
    "Corporate",
    "Futuristic",
    "Editorial",
    "Playful",
    "Technical",
    "Bold",
    "Vintage",
    "Wellness",
    "Sustainable",
    "Product",
]

_STYLE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Luxury",       ["luxury", "high-end", "premium", "sophisticated", "yacht", "gold", "premium lifestyle", "editorial photography", "sleek & visual"]),
    ("Minimalist",   ["minimalist", "minimalism", "airy", "white space", "crisp white", "geometric marks", "monochrome", "expressive minimalism", "iconic & minimalist"]),
    ("Corporate",    ["corporate", "institutional", "professional", "conservative", "structured", "business professional", "trustworthy", "traditional", "data-driven", "social proof"]),
    ("Futuristic",   ["futuristic", "dark mode", "neon", "cyber", "glowing", "high-tech", "neon accents", "gradients", "glow", "purple gradients", "futuristic & abstract"]),
    ("Editorial",    ["editorial", "fashion editorial", "streetwear", "beach photography", "sun-drenched", "high-grain", "gridded web", "high-fashion", "lifestyle"]),
    ("Playful",      ["playful", "friendly", "whimsical", "vibrant", "illustrated", "fun", "bright", "lime green", "bubblegum", "colorful", "illustration", "magic", "watercolor", "artistic"]),
    ("Technical",    ["data-heavy", "technical", "developer", "data visualization", "dashboard", "flowchart", "prompt engineering", "code blocks", "ui-focused", "app-centric", "functional", "industrial tech"]),
    ("Bold",         ["bold", "high-energy", "dynamic", "action", "high-conviction", "layered textures", "immersive", "dimensional", "action-oriented", "graphic & tech"]),
    ("Vintage",      ["vintage", "heritage", "hand-drawn", "woodcut", "classic", "retro", "tactile", "paper-textured", "classic illustration"]),
    ("Wellness",     ["calming", "soft", "pastel", "sage green", "muted peach", "empathetic", "supportive", "accessible", "clean & natural", "clinical"]),
    ("Sustainable",  ["sustainable", "nature-inspired", "recycled", "earthy", "seaweed", "jade", "coral", "nature photography", "organic"]),
    ("Product",      ["product showcase", "mobile-first", "device mockups", "product-focused", "mobile ui", "high-fidelity", "before vs. after", "comparison", "app-centric", "mobile-first"]),
]


def normalize_visual_style(raw: str) -> str:
    """Map a free-text visual style string to the nearest canonical tag."""
    lower = raw.lower()
    for canonical, keywords in _STYLE_KEYWORDS:
        if any(_contains_keyword(lower, k) for k in keywords):
            return canonical
    return "Corporate"


# ── Service Type Tags (range-based, authoritative) ────────────────────────────

# Each entry: (start_slide, end_slide, service_type, service_category)
_SLIDE_RANGE_MAP: list[tuple[int, int, str, str]] = [
    (1,   24,  "General Agency",             "General"),
    (25,  67,  "Branding",                   "Branding"),
    (70,  133, "Landing Page - SaaS",        "Landing Page"),
    (135, 148, "Landing Page - Fintech",     "Landing Page"),
    (150, 181, "Landing Page - Finance",     "Landing Page"),
    (183, 211, "Landing Page - Services",    "Landing Page"),
    (213, 217, "Landing Page - Consumer",    "Landing Page"),
    (219, 228, "Landing Page - E-commerce",  "Landing Page"),
    (230, 271, "Landing Page - Healthcare",  "Landing Page"),
    (273, 288, "Graphics",                   "Graphics"),
    (291, 332, "UX/UI Design - B2B",         "UX/UI Design"),
    (334, 351, "UX/UI Design - AI",          "UX/UI Design"),
    (353, 384, "UX/UI Design - Consumer",    "UX/UI Design"),
    (386, 394, "UX/UI Audit",                "UX/UI Design"),
    (397, 420, "Investor Deck - Finance",    "Investor Deck"),
    (422, 423, "Investor Deck - Services",   "Investor Deck"),
    (425, 461, "Investor Deck - Companies",  "Investor Deck"),
    (463, 470, "Investor Deck - Healthcare", "Investor Deck"),
    (472, 476, "Investor Deck - Sports",     "Investor Deck"),
    (479, 490, "Packaging & Print",          "Packaging & Print"),
    (492, 494, "Conference Booth",           "Conference Booth"),
    (496, 506, "One Pager",                  "One Pager"),
    (508, 511, "Email Design",               "Email Design"),
    (513, 524, "Social Post Design",         "Social & Marketing"),
    (526, 530, "Marketing Misc",             "Social & Marketing"),
    (532, 536, "Whitepaper / eBook",         "Content"),
    (538, 550, "Content Writing",            "Content"),
    (552, 558, "Sports",                     "Sports"),
    (560, 561, "Animations",                 "Animations"),
    (563, 578, "SEO & Content",              "Content"),
]

CANONICAL_SERVICE_CATEGORIES = [
    "Branding",
    "Landing Page",
    "UX/UI Design",
    "Investor Deck",
    "Graphics",
    "Packaging & Print",
    "One Pager",
    "Email Design",
    "Social & Marketing",
    "Content",
    "Animations",
    "Sports",
    "Conference Booth",
    "General",
]

# Granular list (all unique service_type values from the range map)
CANONICAL_SERVICE_TYPES = sorted({stype for _, _, stype, _ in _SLIDE_RANGE_MAP})

# Keyword fallback for slides outside the mapped ranges (579+)
_SERVICE_KEYWORD_FALLBACKS: list[tuple[str, list[str]]] = [
    ("UX/UI Design",     ["mobile app", "app ui", "ux design", "product design", "dashboard", "prototype", "wireframe"]),
    ("Landing Page",     ["website", "landing page", "web mockup", "web ui", "web design"]),
    ("Branding",         ["branding", "logo", "identity", "style guide", "brand palette", "stylescape"]),
    ("Investor Deck",    ["investor deck", "pitch deck", "one-sheet", "fundraising"]),
    ("Social & Marketing", ["social media", "ugc", "paid media", "campaign creative", "ad creative"]),
    ("Content",          ["whitepaper", "ebook", "content writing", "seo", "blog"]),
    ("Animations",       ["animation", "motion", "interactive", "motion graphics"]),
    ("Packaging & Print", ["print", "postcard", "packaging", "collateral", "sticker"]),
]


def get_service_tags(slide_number_str: str, content: str = "", visual_raw: str = "") -> tuple[str, str]:
    """
    Return (service_type, service_category) for a slide.

    Primary:  range-based lookup using the authoritative slide taxonomy.
    Fallback: keyword matching on content + visual_raw (for slides 579+ or edge cases).
    """
    nums = re.findall(r"\d+", str(slide_number_str))
    if not nums:
        return "General", "General"
    start = int(nums[0])
    for s, e, stype, scat in _SLIDE_RANGE_MAP:
        if s <= start <= e:
            return stype, scat
    # Keyword fallback
    combined = f"{content} {visual_raw}".lower()
    for scat, keywords in _SERVICE_KEYWORD_FALLBACKS:
        if any(k in combined for k in keywords):
            return scat, scat
    return "General", "General"
