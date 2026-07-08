"""
Retrieval eval harness — a regression gate for the accuracy pipeline.

Runs a fixed set of prospect personas through retrieval and checks that expected
clients show up and forbidden ones don't. Expectations are seeded from findings
verified against the actual slide PNGs (see git b59ab2b / d0730ba):
  - the aligned front of the deck really does contain Pantera/LightOn/HumanFirst/...
  - the sheet's slide-number drift put finance firms (Ivana Asset, Hageman) under
    "Healthcare", so a healthcare persona must NOT surface them.

Modes:
  --offline  (default) deterministic: checks the filtered + grouped candidate pool.
             No API key needed.
  --full     also runs the Claude-ranked retrieve_examples and checks the top slice.

Usage:
    python eval.py
    python eval.py --full
"""

import argparse
import os

from retriever import (
    _build_filtered_pool,
    _group_into_examples,
    retrieve_examples,
    _EXCLUDE_CLIENTS,
)

# Each case: persona + clients expected to appear and clients that must not.
CASES: list[dict] = [
    {
        "name": "AI search / SaaS — landing page",
        "persona": {
            "industries": ["AI & Technology", "B2B SaaS"],
            "service_categories": ["Landing Page"],
            "search_query": "modern AI search and SaaS product landing page, clean technical UI",
            "visual_style": "Technical",
        },
        "expect": ["Pantera", "LightOn", "HumanFirst", "Guru", "CodeThread"],
        "forbid": [],
    },
    {
        "name": "Wealth management — branding + landing",
        "persona": {
            "industries": ["Finance & Wealth Management"],
            "service_categories": ["Branding", "Landing Page"],
            "search_query": "premium wealth management brand identity and website",
            "visual_style": "Luxury",
        },
        "expect": ["Evergreen Wealth"],
        "forbid": [],
    },
    {
        "name": "DTC apparel — branding",
        "persona": {
            "industries": ["Consumer & E-commerce", "Fashion & Apparel"],
            "service_categories": ["Branding"],
            "search_query": "direct-to-consumer apparel brand identity and packaging",
            "visual_style": "Editorial",
        },
        # Nonchalant is the streetwear/apparel brand — the true strong match. Ta'Da
        # (a laundromat brand) is only loosely related, so it is not an expectation.
        "expect": ["Nonchalant"],
        "forbid": [],
    },
    {
        "name": "Healthcare telemedicine — investor deck",
        "persona": {
            "industries": ["Healthcare & HealthTech"],
            "service_categories": ["Investor Deck"],
            "search_query": "digital health / telemedicine investor pitch deck",
            "visual_style": "Corporate",
        },
        "expect": [],
        # Verified finance firms the drifted sheet misfiled under Healthcare —
        # must never surface for a healthcare prospect.
        "forbid": ["Ivana Asset Solutions", "Hageman Capital", "Hageman Group", "Crestview Capital"],
    },
    {
        "name": "Fintech payments — landing page",
        "persona": {
            "industries": ["Fintech & Payments"],
            "service_categories": ["Landing Page"],
            "search_query": "fintech payments product marketing site",
            "visual_style": "Corporate",
        },
        "expect": ["ClarityPay"],
        "forbid": [],
    },
]


def _pool_clients(persona: dict) -> tuple[set[str], int]:
    pool = _build_filtered_pool(persona)
    clients = {g["client"] for g in _group_into_examples(pool)}
    return clients, len(pool)


def _check(case: dict, clients: set[str]) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []
    for e in case["expect"]:
        checks.append((f"expect  {e}", e in clients))
    for f in case["forbid"]:
        checks.append((f"forbid  {f}", f not in clients))
    leaked = sorted(c for c in clients if c in _EXCLUDE_CLIENTS)
    checks.append((f"no excluded-client leak{' ('+', '.join(leaked)+')' if leaked else ''}", not leaked))
    return checks


def run(full: bool) -> int:
    total = passed = 0
    print(f"{'MODE':<6} {'CASE':<38} RESULT")
    print("-" * 70)
    for case in CASES:
        clients, pool_n = _pool_clients(case["persona"])
        checks = _check(case, clients)

        if full:
            if not os.getenv("ANTHROPIC_API_KEY"):
                print("  --full requires ANTHROPIC_API_KEY; falling back to offline.")
                full = False
            else:
                ranked = retrieve_examples(case["persona"], n_results=6)
                top = {e["client"] for e in ranked if e.get("score", 0) >= 0.5}
                for e in case["expect"]:
                    checks.append((f"[ranked>=.5] {e}", e in top))

        case_pass = sum(1 for _, ok in checks if ok)
        total += len(checks)
        passed += case_pass
        status = "PASS" if case_pass == len(checks) else "FAIL"
        print(f"{'OFFL':<6} {case['name']:<38} {status}  ({case_pass}/{len(checks)}, pool={pool_n})")
        for label, ok in checks:
            if not ok:
                print(f"         ✗ {label}")

    score = 100.0 * passed / total if total else 0.0
    print("-" * 70)
    print(f"SCORE: {passed}/{total} checks = {score:.1f}%")
    return 0 if passed == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run Claude-ranked retrieval")
    raise SystemExit(run(full=ap.parse_args().full))
