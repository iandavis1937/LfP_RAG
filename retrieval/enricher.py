"""
Context enrichment and token highlighting for retrieved passages.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .config import CHUNK_OVERLAP, _DEFAULT_WINDOW, SENTENCE_MARKUP_THRESH
from .models import Chunk
from .llm_client import embed
from .vocabulary import _synonym_to_regex
from .indexer import IndexBuilder


class ContextEnricher:
    def __init__(self, builder: IndexBuilder) -> None:
        self.doc_chunks = builder.doc_chunks

    @staticmethod
    def _find_overlap(a: str, b: str, max_overlap: int = CHUNK_OVERLAP + 50) -> int:
        limit = min(len(a), len(b), max_overlap)
        for n in range(limit, 0, -1):
            if a.endswith(b[:n]):
                return n
        return 0

    def enrich(self, chunk: Chunk, window: int = _DEFAULT_WINDOW) -> str:
        siblings = self.doc_chunks.get(chunk.source_file, [])
        lo = max(0, chunk.chunk_index - window)
        hi = chunk.chunk_index + window + 1
        window_chunks = siblings[lo:hi]
        if not window_chunks:
            return ""
        anchor_pos = chunk.chunk_index - lo
        parts = [window_chunks[0].text]
        for prev, curr in zip(window_chunks, window_chunks[1:]):
            overlap = ContextEnricher._find_overlap(prev.text, curr.text)
            parts.append(curr.text[overlap:])
        parts[anchor_pos] = "|---[\n" + parts[anchor_pos] + "\n]---|"
        return "\n".join(parts)


class TokenHighlighter:
    _SENT_RE = re.compile(r'(?<=[.?!])(?=\s)')

    def __init__(self, query_vec: np.ndarray, expanded_tokens: list[str]) -> None:
        self.query_vec = query_vec          # (1, dim), L2-normalized
        self.expanded_tokens = expanded_tokens
        if expanded_tokens:
            ordered = sorted(set(expanded_tokens), key=len, reverse=True)
            pattern = '|'.join(r'\b' + _synonym_to_regex(t) + r'\b' for t in ordered)
            self._bm25_re: Optional[re.Pattern] = re.compile(pattern, re.IGNORECASE)
        else:
            self._bm25_re = None

    def highlight(self, passages: list[str]) -> list[str]:
        # Split each passage into sentences directly
        split = [self._split_line(p) for p in passages]
        flat = [s for sents in split for s in sents]
        if not flat:
            return passages
        vecs = embed(flat)   # single batched API call
        idx = 0
        results = []
        for sents in split:
            n = len(sents)
            sent_vecs = vecs[idx:idx + n]
            idx += n
            if not sents:
                results.append('')
                continue
            scores = (sent_vecs @ self.query_vec.T).flatten().tolist()
            selected, is_fallback = self._select_faiss(scores)
            marked = []
            for i, sent in enumerate(sents):
                ws = sent[:len(sent) - len(sent.lstrip())]
                s = self._apply_bm25(sent.lstrip())
                if i in selected:
                    marked.append(ws + (f'[~{s}~]' if is_fallback else f'[[{s}]]'))
                else:
                    marked.append(ws + s)
            results.append(''.join(marked))
        return results

    @classmethod
    def _split_line(cls, line: str) -> list[str]:
        parts = cls._SENT_RE.split(line)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _select_faiss(scores: list[float]) -> tuple[set[int], bool]:
        """Return (selected_indices, is_fallback).

        Every sentence scoring > SENTENCE_MARKUP_THRESH is selected (no cap). If none clear the
        threshold, only the single top-scoring sentence is selected and
        is_fallback=True, signalling the caller to use [~sentence~] markup.
        """
        if not scores:
            return set(), False
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        above = {i for i, s in enumerate(scores) if s > SENTENCE_MARKUP_THRESH}
        if above:
            return above, False          # all sentences clearing threshold
        return {indexed[0][0]}, True     # fallback: top scorer only

    def _apply_bm25(self, text: str) -> str:
        if self._bm25_re is None:
            return text
        return self._bm25_re.sub(lambda m: f'[**{m.group(0)}**]', text)
