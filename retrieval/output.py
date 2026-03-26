"""
Output writing: text results and CSV export dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import (
    OUTPUT_DIR,
    SHOW_QUERY_EXPANSION,
    COOCCURRENCE_PRIMARY_KEY,
    COOCCURRENCE_WINDOW,
)


def write_results(
    query_key: str,
    query: str,
    results: list[dict],
    expanded_query: str = "",
    require_primary: bool = False,
    apply_cooccur_boost: bool = False,
    cooccurrence_window: int = COOCCURRENCE_WINDOW,
    bm25_expand_keys: list[str] | None = None,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / f"{query_key}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        if SHOW_QUERY_EXPANSION and expanded_query:
            f.write(f"Query (FAISS):  {query}\n")
            f.write(f"Query (BM25):   {expanded_query}\n")
        else:
            f.write(f"Query: {query}\n")
        if bm25_expand_keys:
            f.write(f"BM25 forced keys: {', '.join(bm25_expand_keys)}\n")
        if require_primary or apply_cooccur_boost:
            mode = "filter+boost" if require_primary else "boost only"
            f.write(
                f"Co-occurrence:  ON  "
                f"(mode={mode}, primary='{COOCCURRENCE_PRIMARY_KEY}', "
                f"window={cooccurrence_window} chars, secondary boost=×1.5)\n"
            )
        f.write(f"Results: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"[{i}] {r['source_file']}\n")
            f.write(f"    Interview type:   {r['interview_type']}\n")
            f.write(f"    Interviewee type: {r['interviewee_type']}\n")
            f.write(f"    Location:         {r['location']}\n")
            f.write(f"    Date:             {r['date']}\n")
            f.write(f"    Query type:       {r['query_type']}\n")
            f.write(f"    Score:            {r['score']:.6f}\n")
            f.write(f"    Lines:            {r['line_start']}–{r['line_end']}\n\n")
            f.write(r["passage"])
            f.write("\n\n" + "-" * 80 + "\n\n")
