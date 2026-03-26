"""
Index building and loading: chunking, FAISS, BM25, and summary generation.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from .config import (
    CORPUS_DIR,
    INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    USE_SUMMARY_WEIGHTS,
    SUMMARY_SEG_TOKENS,
    SUMMARY_PROMPT,
)
from .models import Chunk, SummaryRecord
from .llm_client import embed, count_tokens, truncate_to_tokens, llm


def parse_metadata(filename: str) -> dict:
    """Split stem on '_' into [interview_type, interviewee_type, location, date]."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return {"interview_type": "unknown", "interviewee_type": "unknown",
                "location": "unknown", "date": "unknown"}
    return {
        "interview_type": parts[0],
        "interviewee_type": parts[1],
        "location": parts[2],
        "date": parts[3],
    }


class IndexBuilder:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.doc_chunks: dict[str, list[Chunk]] = {}
        self.summaries: list[SummaryRecord] = []
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_index: Optional[faiss.IndexFlatIP] = None
        self.summary_index: Optional[faiss.IndexFlatIP] = None

    def load_and_chunk(self) -> None:
        splitter = RecursiveCharacterTextSplitter(
            separators=[r"\nI[12]:", "\n\n", "\n", " "],
            is_separator_regex=True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        global_id = 0
        for fpath in sorted(Path(CORPUS_DIR).iterdir()):
            if not fpath.is_file():
                continue
            meta = parse_metadata(fpath.name)
            text = fpath.read_text(encoding="utf-8", errors="replace")
            docs = splitter.create_documents([text])
            file_chunks = []
            for ci, doc in enumerate(docs):
                piece = doc.page_content
                start_char = doc.metadata.get("start_index", 0)
                end_char = start_char + len(piece)
                line_start = text[:start_char].count("\n") + 1
                line_end = text[:end_char].count("\n") + 1
                c = Chunk(
                    id=global_id,
                    text=piece,
                    source_file=fpath.name,
                    chunk_index=ci,
                    line_start=line_start,
                    line_end=line_end,
                    **meta,
                )
                self.chunks.append(c)
                file_chunks.append(c)
                global_id += 1
            self.doc_chunks[fpath.name] = file_chunks

    def build_faiss_chunk_index(self) -> None:
        vecs = embed([c.text for c in self.chunks])
        dim = vecs.shape[1]
        self.chunk_index = faiss.IndexFlatIP(dim)
        self.chunk_index.add(vecs)

    def build_bm25(self) -> None:
        corpus = [c.text.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def generate_summaries(self) -> None:
        for fid, (fname, chunks) in enumerate(self.doc_chunks.items()):
            full_text = "\n".join(c.text for c in chunks)
            summary_text = self._summarize_document(full_text)
            self.summaries.append(SummaryRecord(filename=fname, summary_text=summary_text, faiss_id=fid))

    def _summarize_document(self, text: str) -> str:
        if count_tokens(text) <= SUMMARY_SEG_TOKENS:
            return llm(SUMMARY_PROMPT.format(text=text), max_tokens=600)
        # Map: split into token-limited segments, summarize each
        # Use count_tokens / truncate_to_tokens to avoid direct tiktoken dependency
        total_tokens = count_tokens(text)
        segments = []
        start_tok = 0
        remaining = text
        while start_tok < total_tokens:
            seg = truncate_to_tokens(remaining, SUMMARY_SEG_TOKENS)
            segments.append(seg)
            seg_tok_len = count_tokens(seg)
            # Advance `remaining` by stripping the segment we just consumed
            # Re-encode the rest by character approximation: drop the seg text from front
            remaining = remaining[len(seg):]
            start_tok += seg_tok_len
            if not remaining:
                break
        partials = [llm(SUMMARY_PROMPT.format(text=seg), max_tokens=400) for seg in segments]
        # Reduce: combine partial summaries
        combined = "\n\n".join(partials)
        return llm(SUMMARY_PROMPT.format(text=combined), max_tokens=600)

    def build_faiss_summary_index(self) -> None:
        vecs = embed([s.summary_text for s in self.summaries])
        dim = vecs.shape[1]
        self.summary_index = faiss.IndexFlatIP(dim)
        self.summary_index.add(vecs)

    def persist(self) -> None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.chunk_index, f"{INDEX_DIR}/chunk.index")
        with open(f"{INDEX_DIR}/bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(f"{INDEX_DIR}/chunk_store.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)
        if USE_SUMMARY_WEIGHTS:
            faiss.write_index(self.summary_index, f"{INDEX_DIR}/summary.index")
            with open(f"{INDEX_DIR}/summaries.json", "w", encoding="utf-8") as f:
                json.dump([{"filename": s.filename, "summary_text": s.summary_text, "faiss_id": s.faiss_id}
                           for s in self.summaries], f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        self.chunk_index = faiss.read_index(f"{INDEX_DIR}/chunk.index")
        with open(f"{INDEX_DIR}/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(f"{INDEX_DIR}/chunk_store.json", encoding="utf-8") as f:
            self.chunks = [Chunk.from_dict(d) for d in json.load(f)]
        if USE_SUMMARY_WEIGHTS:
            self.summary_index = faiss.read_index(f"{INDEX_DIR}/summary.index")
            with open(f"{INDEX_DIR}/summaries.json", encoding="utf-8") as f:
                self.summaries = [SummaryRecord(**d) for d in json.load(f)]
        # Rebuild doc_chunks grouping
        for c in self.chunks:
            self.doc_chunks.setdefault(c.source_file, []).append(c)
