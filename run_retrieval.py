"""
Orchestration script for the qualitative interview retrieval system.

Usage:
    python run_retrieval.py

Behaviour is controlled entirely by constants in retrieval/config.py.
Set BUILD_INDEX=True to build a fresh index; False to load an existing one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from retrieval import IndexBuilder, HierarchicalRetriever, write_results
from retrieval.queries import query_d
from retrieval.config import (
    BUILD_INDEX,
    OUTPUT_DIR,
    OUTPUT_SUFFIX,
    USE_SUMMARY_WEIGHTS,
    EXPORT_CSV,
    COOCCURRENCE_FILTER,
    COOCCURRENCE_WINDOW,
    INDEX_DIR,
)

if __name__ == "__main__":
    
    _HERE = Path(__file__).parent

    builder = IndexBuilder()

    if BUILD_INDEX:
        print("Building index...")
        builder.load_and_chunk()
        print(f"  Loaded {len(builder.chunks)} chunks from {len(builder.doc_chunks)} documents.")
        builder.build_faiss_chunk_index()
        print("  FAISS chunk index built.")
        builder.build_bm25()
        print("  BM25 built.")
        if USE_SUMMARY_WEIGHTS:
            builder.generate_summaries()
            print(f"  Generated {len(builder.summaries)} summaries.")
            builder.build_faiss_summary_index()
            print("  FAISS summary index built.")
        builder.persist()
        print(f"  Index persisted to '{INDEX_DIR}/'.")
    else:
        print("Loading index...")
        builder.load()
        print(f"  Loaded {len(builder.chunks)} chunks" +
              (f", {len(builder.summaries)} summaries." if USE_SUMMARY_WEIGHTS else "."))

    retriever = HierarchicalRetriever(builder)

    _sfx = f"__{OUTPUT_SUFFIX}" if OUTPUT_SUFFIX else ""

    for query_key, query_spec in query_d.items():
        query_str, filters = query_spec[0], query_spec[1]
        # Optional third element overrides global co-occurrence defaults per query
        _opts                 = query_spec[2] if len(query_spec) > 2 else {}
        _require_primary      = _opts.get("require_primary",      COOCCURRENCE_FILTER)
        _apply_cooccur_boost  = _opts.get("apply_cooccur_boost",          False)
        _cooc_window          = _opts.get("cooccurrence_window",  COOCCURRENCE_WINDOW)
        _bm25_expand_keys     = _opts.get("bm25_expand_keys",     None)
        print(f"\nRunning query '{query_key}' — per-document mode ({len(builder.doc_chunks)} documents)...")
        all_results = []
        for fname in sorted(builder.doc_chunks.keys()):
            doc_filters = {**filters, "source_file": fname}
            doc_key = f"{query_key}__{Path(fname).stem}{_sfx}"
            results = retriever.retrieve(
                query_str, filters=doc_filters,
                require_primary=_require_primary,
                apply_cooccur_boost=_apply_cooccur_boost,
                cooccurrence_window=_cooc_window,
                bm25_expand_keys=_bm25_expand_keys,
            )
            write_results(doc_key, query_str, results, retriever.fusion.last_expanded,
                          require_primary=_require_primary, apply_cooccur_boost=_apply_cooccur_boost,
                          cooccurrence_window=_cooc_window, bm25_expand_keys=_bm25_expand_keys)
            if EXPORT_CSV:
                subprocess.run(
                    [sys.executable, str(_HERE / "retrieval" / "output_txt_to_csv.py"),
                     str(Path(OUTPUT_DIR) / f"{doc_key}.txt")],
                    check=True,
                )
            print(f"  {fname}: {len(results)} results → '{OUTPUT_DIR}/{doc_key}.txt'")
            all_results.extend(results)
        all_results.sort(key=lambda r: r["score"], reverse=True)
        combined_key = f"{query_key}__all{_sfx}"
        write_results(combined_key, query_str, all_results, retriever.fusion.last_expanded,
                      require_primary=_require_primary, apply_cooccur_boost=_apply_cooccur_boost,
                      cooccurrence_window=_cooc_window, bm25_expand_keys=_bm25_expand_keys)
        if EXPORT_CSV:
            subprocess.run(
                [sys.executable, str(_HERE / "retrieval" / "output_txt_to_csv.py"),
                 str(Path(OUTPUT_DIR) / f"{combined_key}.txt")],
                check=True,
            )
        print(f"  Combined: {len(all_results)} results → '{OUTPUT_DIR}/{combined_key}.txt'")
