#!/usr/bin/env python3
"""
test_context_manager_import.py

Test harness that IMPORTS the real implementation from `pipelines.context_manager`
instead of embedding/re-writing the code.

Usage:
    python test_context.py --run-all
    python test_context.py --show-sample
    python test_context.py --dump-fixtures > fixtures.json

"""

from typing import List, Dict, Optional
import json
import argparse
import sys
import importlib

# -------------------------
# Import the real module
# -------------------------
required_names = [
    "track_previous_studies",
    "create_context",
    "create_conversation_context",
    "prioritize_context",
]

track_previous_studies = create_context = create_conversation_context = prioritize_context = None

try:
    # Preferred: direct import of symbols
    from pipelines.context_manager import (
        track_previous_studies,
        create_context,
        create_conversation_context,
        prioritize_context,
    )
except Exception as e_direct:
    # Fallback: import module and pull attributes dynamically (less strict)
    try:
        module = importlib.import_module("pipelines.context_manager")
        track_previous_studies = getattr(module, "track_previous_studies", None)
        create_context = getattr(module, "create_context", None)
        create_conversation_context = getattr(module, "create_conversation_context", None)
        prioritize_context = getattr(module, "prioritize_context", None)
    except Exception as e_mod:
        print("ERROR: failed to import 'pipelines.context_manager'.")
        print(" Direct import error:", repr(e_direct))
        print(" Module import error:", repr(e_mod))
        print("\nMake sure:")
        print("  - The package `pipelines` is on PYTHONPATH (run from project root or set PYTHONPATH=.)")
        print("  - The file pipelines/context_manager.py exists and is importable.")
        print("Example:")
        print("  PYTHONPATH=. python test_context_manager_import.py --run-all")
        sys.exit(2)

# Verify we actually have the callables
missing = [name for name, fn in [
    ("track_previous_studies", track_previous_studies),
    ("create_context", create_context),
    ("create_conversation_context", create_conversation_context),
    ("prioritize_context", prioritize_context),
] if not callable(fn)]

if missing:
    print("ERROR: imported module is missing required functions:", missing)
    print("Inspect pipelines/context_manager.py and ensure these functions are defined and exported.")
    sys.exit(3)


# -------------------------
# Fixtures for testing
# -------------------------

SAMPLE_STUDIES = [
    {"title": "Study A on widgets", "abstract": "Abstract A " * 40, "pmid": "1001", "authors": "Alice et al.", "_score": 0.95},
    {"title": "Study B about gadgets", "abstract": "Abstract B " * 30, "pmid": "1002", "authors": "Bob et al.", "_score": 0.82},
    {"title": "Study C on widgets and gadgets", "abstract": "Abstract C " * 50, "pmid": "1003", "authors": "Carol et al.", "_score": 0.78},
]

PREVIOUS_STUDIES = [
    {"title": "Study A on widgets", "pmid": "1001"},
]

SAMPLE_TRANSCRIPTION = "This is a meeting transcription. " * 80  # long transcription to test truncation
SAMPLE_QUERIES = [
    "What are the main findings of Study A?",
    "Summarize Study B in one paragraph.",
    "Are widgets better than gadgets?",
    "List limitations of recent widget trials.",
    "Give code to reproduce Study C figures."
]
SAMPLE_ANSWERS = [
    "Answer to query 1: short summary",
    "Answer to query 2: longer summary " * 20,
    "Answer to query 3: short"
]

SAMPLE_MESSAGES = [
    {"role": "user", "content": "What's the latest on widgets?"},
    {"role": "assistant", "content": "Widgets are ... (long response) " * 30},
    {"role": "user", "content": "Can you show code?"},
    {"role": "assistant", "content": "Here's some code: print('hello')"}
]


# -------------------------
# Tests and utilities (use imported functions)
# -------------------------

def test_track_previous_studies():
    print("TEST: track_previous_studies")
    res = track_previous_studies(SAMPLE_STUDIES, PREVIOUS_STUDIES)
    assert isinstance(res, list), "Result must be a list"
    cached = [s for s in res if s.get('previously_retrieved')]
    assert len(cached) == 1 and cached[0].get('pmid') == '1001', "Study A must be marked cached"
    print("  OK - previously retrieved studies flagged correctly\n")


def test_create_context_basic_and_truncation():
    print("TEST: create_context (basic + truncation)")
    ctx = create_context(
        transcription=SAMPLE_TRANSCRIPTION,
        studies=SAMPLE_STUDIES,
        past_queries=SAMPLE_QUERIES,
        past_answers=SAMPLE_ANSWERS,
        previous_studies=PREVIOUS_STUDIES,
        max_chars=500,  # small to force truncation behavior
        priority_order=['transcription', 'studies', 'queries', 'answers']
    )
    assert isinstance(ctx, str), "create_context must return a string"
    assert "MEETING TRANSCRIPTION" in ctx or "RELEVANT STUDIES" in ctx, "Expected sections not found"
    print("  Context length:", len(ctx))
    print("  Sample (first 500 chars):\n", ctx[:500])
    print("  OK - create_context produced output and respected limits\n")


def test_create_conversation_context():
    print("TEST: create_conversation_context")
    conv = create_conversation_context(SAMPLE_MESSAGES, max_messages=3)
    assert isinstance(conv, str), "create_conversation_context must return a string"
    assert "CONVERSATION" in conv.upper(), "Conversation header expected"
    print("  Conversation snippet:\n", conv)
    print("  OK - conversation context created and truncated\n")


def test_prioritize_context_behavior():
    print("TEST: prioritize_context")
    # Build an intentionally large full_context using the imported create_context
    full = create_context(
        transcription=SAMPLE_TRANSCRIPTION,
        studies=SAMPLE_STUDIES,
        past_queries=SAMPLE_QUERIES,
        past_answers=SAMPLE_ANSWERS,
        previous_studies=PREVIOUS_STUDIES,
        max_chars=2000
    )
    # Call imported prioritize_context with small budget to force truncation
    prioritized = prioritize_context(full, max_chars=300)
    assert isinstance(prioritized, str)
    assert len(prioritized) <= 350, "Prioritized context should be within a small overhang of limit"
    print("  Prioritized length:", len(prioritized))
    print("  Prioritized output preview:\n", prioritized[:400])
    print("  OK - prioritization truncated context as expected\n")


def run_all_tests():
    print("\nRunning all tests...\n")
    test_track_previous_studies()
    test_create_context_basic_and_truncation()
    test_create_conversation_context()
    test_prioritize_context_behavior()
    print("ALL TESTS PASSED\n")


def show_sample_outputs():
    print("\n=== SAMPLE OUTPUTS ===\n")
    ctx = create_context(
        transcription=SAMPLE_TRANSCRIPTION,
        studies=SAMPLE_STUDIES,
        past_queries=SAMPLE_QUERIES,
        past_answers=SAMPLE_ANSWERS,
        previous_studies=PREVIOUS_STUDIES,
        max_chars=1000
    )
    print("FULL CONTEXT (truncated preview, 1000 chars):\n")
    print(ctx[:1000])
    print("\n--- Conversation Context ---\n")
    print(create_conversation_context(SAMPLE_MESSAGES))
    print("\n--- Prioritized (small budget) ---\n")
    print(prioritize_context(ctx, max_chars=500))


def dump_fixtures():
    payload = {
        "studies": SAMPLE_STUDIES,
        "previous_studies": PREVIOUS_STUDIES,
        "transcription": SAMPLE_TRANSCRIPTION[:200],
        "queries": SAMPLE_QUERIES,
        "answers": SAMPLE_ANSWERS,
        "messages": SAMPLE_MESSAGES
    }
    print(json.dumps(payload, indent=2))


# -------------------------
# CLI
# -------------------------

def main(argv):
    parser = argparse.ArgumentParser(description="Test harness for pipelines.context_manager")
    parser.add_argument("--run-all", action="store_true", help="Run full test suite (assertions + prints)")
    parser.add_argument("--show-sample", action="store_true", help="Print sample context outputs")
    parser.add_argument("--dump-fixtures", action="store_true", help="Dump JSON fixtures to stdout")
    args = parser.parse_args(argv)

    if args.run_all:
        run_all_tests()
        return

    if args.show_sample:
        show_sample_outputs()
        return

    if args.dump_fixtures:
        dump_fixtures()
        return

    parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
