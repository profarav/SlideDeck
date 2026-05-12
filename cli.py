#!/usr/bin/env python3
"""
CLI tool for testing the slide retrieval engine.

Usage:
    python cli.py "We're pitching to a fintech startup building a B2B payments platform"
    python cli.py "Luxury wealth management firm targeting HNW individuals" --n 5
    python cli.py --interactive
"""

import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from profiler import build_search_persona
from retriever import retrieve_case_studies


def run_query(description: str, n_results: int, verbose: bool = False):
    print(f"\n{'─' * 60}")
    print(f"  PROSPECT: {description[:80]}{'...' if len(description) > 80 else ''}")
    print(f"{'─' * 60}\n")

    print("  Profiling prospect...")
    persona = build_search_persona(description)

    print(f"  ✓ Industries matched:  {', '.join(persona['industries'])}")
    print(f"  ✓ Visual style:        {persona['visual_style']}")
    print(f"  ✓ Search query:        {persona['search_query']}")
    if verbose:
        print(f"  ✓ Reasoning:           {persona['reasoning']}")

    print(f"\n  Searching slide library...\n")
    slides = retrieve_case_studies(persona, n_results=n_results)

    if not slides:
        print("  ✗ No matching slides found.")
        return

    print(f"  {'#':<6} {'SLIDE':>8}  {'CLIENT':<22} {'INDUSTRY':<30} {'STYLE':<14} SCORE")
    print(f"  {'─'*6} {'─'*8}  {'─'*22} {'─'*30} {'─'*14} {'─'*5}")
    for i, s in enumerate(slides, 1):
        print(
            f"  {i:<6} {s['slide_number']:>8}  "
            f"{s['client'][:22]:<22} "
            f"{s['industry'][:30]:<30} "
            f"{s['visual_style']:<14} "
            f"{s['score']:.3f}"
        )

    print(f"\n  Slide numbers to pull: {', '.join(s['slide_number'] for s in slides)}\n")

    if verbose:
        print("\n  ── DETAIL ─────────────────────────────────────────────────")
        for s in slides:
            print(f"\n  [{s['slide_number']}] {s['client']} ({s['industry']})")
            print(f"  Style: {s['visual_style']} | Score: {s['score']}")
            print(f"  {s['content'][:120]}")


def interactive_mode(n_results: int, verbose: bool):
    print("\n  Klimt Slide Retrieval Engine — Interactive Mode")
    print("  Type a prospect description, or 'quit' to exit.\n")
    while True:
        try:
            desc = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break
        if not desc:
            continue
        if desc.lower() in ("quit", "exit", "q"):
            print("  Bye.")
            break
        run_query(desc, n_results, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Klimt Slide Retrieval Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("description", nargs="?", help="Prospect description string")
    parser.add_argument("--n", type=int, default=8, help="Number of slides to return (default: 8)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full slide content")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to slide_engine/.env or export it.")
        sys.exit(1)

    if args.interactive:
        interactive_mode(args.n, args.verbose)
    elif args.description:
        run_query(args.description, args.n, args.verbose)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
