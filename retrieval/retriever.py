"""
Query routing, fusion retrieval (FAISS + BM25 + RRF), and hierarchical retrieval.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .config import (
    MANUAL_QUERY_PARAMS,
    USE_CLASSIFIER,
    QUERY_TYPE,
    QUERY_TYPE_PARAMS,
    TOP_K,
    MIN_SCORE,
    RRF_K,
    USE_SUMMARY_WEIGHTS,
    COOCCURRENCE_WINDOW,
    COOCCURRENCE_PRIMARY_KEY,
    COOCCURRENCE_SECONDARY_KEYS,
    CLASSIFY_PROMPT,
)
from .models import Chunk, SummaryRecord
from .indexer import IndexBuilder
from .llm_client import embed, llm
from .vocabulary import SYNONYM_MAP, _synonym_to_regex, _expand_synonym
from .enricher import ContextEnricher, TokenHighlighter


class QueryRouter:
    def classify(self, query: str) -> dict:
        if MANUAL_QUERY_PARAMS is not None:
            return {"type": "manual", **MANUAL_QUERY_PARAMS}
        if not USE_CLASSIFIER:
            label = QUERY_TYPE if QUERY_TYPE in QUERY_TYPE_PARAMS else "analytical"
            return {"type": label, **QUERY_TYPE_PARAMS[label]}
        raw = llm(CLASSIFY_PROMPT.format(query=query), max_tokens=10).strip().lower()
        label = raw if raw in QUERY_TYPE_PARAMS else "analytical"
        return {"type": label, **QUERY_TYPE_PARAMS[label]}


class FusionRetriever:
    # Compiled once per process; depend only on module-level SYNONYM_MAP constants
    _PRIMARY_RE:   re.Pattern | None = None
    _SECONDARY_RE: re.Pattern | None = None

    def __init__(self, builder: IndexBuilder) -> None:
        self.builder = builder
        # Map chunk id → chunk for fast lookup
        self._id_to_chunk: dict[int, Chunk] = {c.id: c for c in builder.chunks}
        # Map summary filename → SummaryRecord
        self._fname_to_summary: dict[str, SummaryRecord] = {s.filename: s for s in builder.summaries}
        # Query embedding cache: query string → (query_vec, expanded_str, expanded_tokens, summary_scores)
        self._query_cache: dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Stage 3b helpers — co-occurrence re-ranking
    # ------------------------------------------------------------------

    def _get_primary_re(self) -> re.Pattern:
        """Compile (once) a regex matching any term in SYNONYM_MAP[COOCCURRENCE_PRIMARY_KEY]."""
        if FusionRetriever._PRIMARY_RE is None:
            primary_syns = SYNONYM_MAP.get(COOCCURRENCE_PRIMARY_KEY, [])
            FusionRetriever._PRIMARY_RE = re.compile(
                "|".join(r"\b" + _synonym_to_regex(s) + r"\b" for s in primary_syns),
                re.IGNORECASE,
            )
        return FusionRetriever._PRIMARY_RE

    def _get_secondary_re(self) -> re.Pattern:
        """Compile (once) a regex matching terms from all COOCCURRENCE_SECONDARY_KEYS."""
        if FusionRetriever._SECONDARY_RE is None:
            secondary_syns = [
                s
                for key in COOCCURRENCE_SECONDARY_KEYS
                for s in SYNONYM_MAP.get(key, [])
            ]
            FusionRetriever._SECONDARY_RE = re.compile(
                "|".join(r"\b" + _synonym_to_regex(s) + r"\b" for s in secondary_syns),
                re.IGNORECASE,
            )
        return FusionRetriever._SECONDARY_RE

    def _cooccurrence_score(self, chunk: Chunk, window: int) -> float:
        """Return a score multiplier based on primary/secondary topic co-occurrence.

        0.0 — no primary-topic term present            → hard-exclude when require_primary=True
        1.0 — primary present, no secondary term       → neutral pass-through
        1.5 — primary AND secondary within `window`
              characters of each other                 → proximity boost

        Primary terms:   SYNONYM_MAP[COOCCURRENCE_PRIMARY_KEY]
        Secondary terms: SYNONYM_MAP keys in COOCCURRENCE_SECONDARY_KEYS
        """
        text = chunk.text
        primary_spans = [(m.start(), m.end()) for m in self._get_primary_re().finditer(text)]
        if not primary_spans:
            return 0.0
        secondary_spans = [(m.start(), m.end()) for m in self._get_secondary_re().finditer(text)]
        if not secondary_spans:
            return 1.0
        for p_start, _ in primary_spans:
            for s_start, _ in secondary_spans:
                if abs(p_start - s_start) <= window:
                    return 1.5
        return 1.0

    def _summary_scores(self, query_vec: np.ndarray) -> dict[str, float]:
        n = self.builder.summary_index.ntotal
        scores_arr, idx_arr = self.builder.summary_index.search(query_vec, n)
        id_to_score = {int(idx): float(score) for idx, score in zip(idx_arr[0], scores_arr[0])}
        return {
            fname: max(0.0, id_to_score.get(srec.faiss_id, 0.0))
            for fname, srec in self._fname_to_summary.items()
        }

    def _expand_query(self, query: str, extra_keys: list[str] | None = None) -> str:
        """Append SYNONYM_MAP expansions to the query string for BM25.

        Canonical keys that appear literally in `query` are always expanded.
        Keys listed in `extra_keys` are expanded unconditionally, regardless of
        whether their canonical form appears in the query string — allowing
        actor-focused FAISS queries to still receive full BM25 synonym coverage
        for behavior-category terms that were deliberately omitted from the prose.
        """
        extra = []
        q_lower = query.lower()
        forced = {k.lower() for k in (extra_keys or [])}
        for canonical, synonyms in SYNONYM_MAP.items():
            if canonical.lower() in q_lower or canonical.lower() in forced:
                for syn in synonyms:
                    extra.extend(_expand_synonym(syn))
        if extra:
            return query + " " + " ".join(extra)
        return query

    def _apply_filters(self, chunks: list[Chunk], filters: dict) -> list[Chunk]:
        if not filters:
            return chunks
        result = []
        for c in chunks:
            if "source_file" in filters and c.source_file != filters["source_file"]:
                continue
            if "interview_type" in filters and c.interview_type != filters["interview_type"]:
                continue
            if "interviewee_type" in filters and c.interviewee_type != filters["interviewee_type"]:
                continue
            if "location" in filters and c.location != filters["location"]:
                continue
            if "date" in filters:
                d = filters["date"]
                if isinstance(d, tuple):
                    lo, hi = d
                    if not (lo <= c.date <= hi):
                        continue
                else:
                    if c.date != d:
                        continue
            result.append(c)
        return result

    def retrieve_faiss(self, query_vec: np.ndarray, candidate_ids: set[int], k: int) -> list[tuple[int, float]]:
        n_search = min(len(candidate_ids), self.builder.chunk_index.ntotal)
        scores_arr, idx_arr = self.builder.chunk_index.search(query_vec, n_search)
        hits = []
        for score, idx in zip(scores_arr[0], idx_arr[0]):
            if idx in candidate_ids:
                hits.append((int(idx), float(score)))
            if len(hits) >= k:
                break
        return hits

    def retrieve_bm25(self, expanded_query: str, candidate_chunks: list[Chunk], k: int) -> list[tuple[int, float]]:
        tokens = expanded_query.lower().split()
        all_scores = self.builder.bm25.get_scores(tokens)
        cid_scores = [(c.id, float(all_scores[c.id])) for c in candidate_chunks]
        cid_scores.sort(key=lambda x: x[1], reverse=True)
        return cid_scores[:k]

    def fuse_rrf(
        self,
        faiss_hits: list[tuple[int, float]],
        bm25_hits: list[tuple[int, float]],
        w_faiss: float,
        w_bm25: float,
        summary_scores: dict[str, float],
    ) -> list[tuple[int, float]]:
        rrf: dict[int, float] = {}
        for rank, (cid, _) in enumerate(faiss_hits):
            rrf[cid] = rrf.get(cid, 0.0) + w_faiss * (1.0 / (rank + RRF_K))
        for rank, (cid, _) in enumerate(bm25_hits):
            rrf[cid] = rrf.get(cid, 0.0) + w_bm25 * (1.0 / (rank + RRF_K))
        # Optionally multiply by document-level summary score
        fused = []
        for cid, score in rrf.items():
            chunk = self._id_to_chunk[cid]
            doc_boost = summary_scores.get(chunk.source_file, 0.0) if USE_SUMMARY_WEIGHTS else 1.0
            fused.append((cid, score * doc_boost))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def _extract_tokens(self, query: str, extra_keys: list[str] | None = None) -> list[str]:
        """Return canonical + synonym tokens for highlighting.

        Mirrors _expand_query: keys in `extra_keys` are included unconditionally
        so that terms forced into BM25 expansion are also highlighted in output.
        """
        tokens: list[str] = []
        q_lower = query.lower()
        forced = {k.lower() for k in (extra_keys or [])}
        for canonical, synonyms in SYNONYM_MAP.items():
            if canonical.lower() in q_lower or canonical.lower() in forced:
                tokens.append(canonical)
                tokens.extend(synonyms)
        seen: set[str] = set()
        result: list[str] = []
        for t in tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                result.append(t)
        return result

    def retrieve(
        self,
        query: str,
        params: dict,
        filters: dict,
        k: int = TOP_K,
        require_primary: bool = False,
        apply_cooccur_boost: bool = False,
        cooccurrence_window: int = COOCCURRENCE_WINDOW,
        bm25_expand_keys: list[str] | None = None,
    ) -> list[tuple[int, float]]:
        # Cache key includes bm25_expand_keys so the same query string with
        # different forced expansions produces separate cache entries.
        _cache_key = (query, tuple(sorted(bm25_expand_keys)) if bm25_expand_keys else ())
        if _cache_key not in self._query_cache:
            query_vec = embed([query])  # shape (1, dim); FAISS always uses raw query
            expanded = self._expand_query(query, extra_keys=bm25_expand_keys)
            expanded_tokens = self._extract_tokens(query, extra_keys=bm25_expand_keys)
            summary_scores = self._summary_scores(query_vec) if USE_SUMMARY_WEIGHTS else {}
            self._query_cache[_cache_key] = (query_vec, expanded, expanded_tokens, summary_scores)
        query_vec, expanded, expanded_tokens, summary_scores = self._query_cache[_cache_key]
        self.last_query_vec       = query_vec
        self.last_expanded        = expanded
        self.last_expanded_tokens = expanded_tokens
        # Filter candidate chunks
        candidate_chunks = self._apply_filters(self.builder.chunks, filters)
        candidate_ids = {c.id for c in candidate_chunks}
        faiss_hits = self.retrieve_faiss(query_vec, candidate_ids, k)
        bm25_hits = self.retrieve_bm25(expanded, candidate_chunks, k)
        fused = self.fuse_rrf(faiss_hits, bm25_hits, params["w_faiss"], params["w_bm25"], summary_scores)
        # Stage 3b — co-occurrence re-ranking
        if require_primary or apply_cooccur_boost:
            rescored = []
            for cid, score in fused:
                chunk = self._id_to_chunk[cid]
                cooc = self._cooccurrence_score(chunk, window=cooccurrence_window)
                if require_primary and cooc == 0.0:
                    continue  # hard filter: primary topic absent
                rescored.append((cid, score * max(cooc, 1.0)))  # boost only; 0.0 → 1.0 when not filtering
            rescored.sort(key=lambda x: x[1], reverse=True)
            return rescored
        return fused


class HierarchicalRetriever:
    def __init__(self, builder: IndexBuilder) -> None:
        self.builder = builder
        self.router = QueryRouter()
        self.fusion = FusionRetriever(builder)
        self.enricher = ContextEnricher(builder)
        self._id_to_chunk: dict[int, Chunk] = {c.id: c for c in builder.chunks}

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        min_score: float = MIN_SCORE,
        require_primary: bool = False,
        apply_cooccur_boost: bool = False,
        cooccurrence_window: int = COOCCURRENCE_WINDOW,
        bm25_expand_keys: list[str] | None = None,
    ) -> list[dict]:
        if filters is None:
            filters = {}
        params = self.router.classify(query)
        window = params["window"]
        ranked = self.fusion.retrieve(
            query, params, filters,
            require_primary=require_primary,
            apply_cooccur_boost=apply_cooccur_boost,
            cooccurrence_window=cooccurrence_window,
            bm25_expand_keys=bm25_expand_keys,
        )

        results = []
        covered: dict[str, set[int]] = {}  # source_file → set of covered chunk indices

        for cid, score in ranked:
            if score <= min_score:
                break
            chunk = self._id_to_chunk[cid]
            fname = chunk.source_file
            covered.setdefault(fname, set())
            if chunk.chunk_index in covered[fname]:
                continue  # anchor already covered
            passage = self.enricher.enrich(chunk, window=window)
            # Mark all indices in the enriched window as covered
            lo = chunk.chunk_index - window
            hi = chunk.chunk_index + window
            for idx in range(max(0, lo), hi + 1):
                covered[fname].add(idx)
            siblings = self.enricher.doc_chunks[fname]
            enriched_lo = max(0, lo)
            enriched_hi = min(len(siblings) - 1, hi)
            results.append({
                "score": score,
                "source_file": fname,
                "interview_type": chunk.interview_type,
                "interviewee_type": chunk.interviewee_type,
                "location": chunk.location,
                "date": chunk.date,
                "passage": passage,
                "query_type": params["type"],
                "line_start": siblings[enriched_lo].line_start,
                "line_end": siblings[enriched_hi].line_end,
            })
        if results:
            highlighter = TokenHighlighter(
                query_vec=self.fusion.last_query_vec,
                expanded_tokens=self.fusion.last_expanded_tokens,
            )
            highlighted = highlighter.highlight([r["passage"] for r in results])
            for r, h in zip(results, highlighted):
                r["passage"] = h
        return results
