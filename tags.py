"""
Canonical tag lists and keyword-based normalization for industries and visual styles.
These are derived from the actual content in Master Sheet.xlsx.
"""

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
    ("Branding & Design",            ["branding", "design", "logo", "identity", "stylescape", "sohva", "secludy", "sona padel"]),
    ("Education & EdTech",           ["edtech", "education", "nursing", "school", "crna", "diocese"]),
    ("Energy & Sustainability",      ["energy", "sustainability", "renewable", "solar", "wind"]),
    ("Cybersecurity",                ["cybersecurity", "security", "cyber", "cyber whyze"]),
    ("HR Tech",                      ["hr", "human resources", "hiring", "fractional", "techsta", "m.ink"]),
    ("Web3 & Crypto",                ["web3", "crypto", "blockchain", "token", "livepeer", "limitless"]),
    ("Travel & Hospitality",         ["travel", "hospitality", "event", "ticketing", "playstay"]),
    ("Sports & Athletics",           ["sports", "athletics", "fitness", "running", "atlas", "padres", "nike"]),
    ("Data & Analytics",             ["data", "analytics", "adtech", "advertising", "cookie"]),
    ("Operations & Logistics",       ["operations", "logistics", "ops", "ulta", "clickup"]),
    ("Maritime",                     ["maritime", "payroll", "seafare"]),
    ("General Agency",               ["general", "agency", "capabilities", "klimt"]),
]


def normalize_industry(raw: str) -> str:
    """Map a free-text industry string to the nearest canonical tag."""
    lower = raw.lower()
    for canonical, keywords in _INDUSTRY_KEYWORDS:
        if any(k in lower for k in keywords):
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
        if any(k in lower for k in keywords):
            return canonical
    return "Corporate"
