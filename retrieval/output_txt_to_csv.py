#!/usr/bin/env python3
"""
txt_to_csv.py — Parse a RAG query output .txt file and export it to CSV.

Usage:
    python txt_to_csv.py                              # uses default input file
    python txt_to_csv.py path/to/some_query.txt       # explicit input file
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Sibling import: compare_output.py lives in the same directory as this script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compare_output import parse_file, strip_markup  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_INPUT = _HERE / "queries" / "full_corpus_3_24" / "armed_groups__all__F5_B5.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_keywords(passage_raw: str) -> str:
    matches = re.findall(r"\[\*\*(.+?)\*\*\]", passage_raw)
    seen = []
    for m in matches:
        if m not in seen:
            seen.append(m)
    return ", ".join(seen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a RAG query output .txt file and export it to CSV."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(_DEFAULT_INPUT),
        help="Path to the .txt input file (default: %(default)s)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = _HERE / input_path
    input_path = input_path.resolve()

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse the file using the shared parse_file function.
    # Note: parse_file looks for "Query:" in the header; the actual file uses
    # "Query (FAISS):", so run.query will be empty — that is acceptable since
    # the query field is not included in the CSV output.
    run = parse_file(input_path, label=input_path.stem)

    output_path = input_path.with_suffix(".csv")

    fieldnames = [
        "Doc",
        "Lines",
        "Page",
        "Keywords",
        "Description",
        "Quote",
        "Score",
        "Interview type",
        "Interviewee type",
        "location",
        "date",
        "Query type",
        "Plain text",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in run.entries:
            writer.writerow(
                {
                    "Doc": entry.source,
                    "Lines": entry.lines,
                    "Page": "",
                    "Keywords": extract_keywords(entry.passage_raw),
                    "Description": "",
                    "Quote": entry.passage_raw,
                    "Score": entry.score,
                    "Interview type": entry.meta.interview_type,
                    "Interviewee type": entry.meta.interviewee_type,
                    "location": entry.meta.location,
                    "date": entry.meta.date,
                    "Query type": entry.meta.query_type,
                    "Plain text": strip_markup(entry.passage_raw),
                }
            )

    print(f"Wrote {len(run.entries)} rows to {output_path}")


if __name__ == "__main__":
    main()
