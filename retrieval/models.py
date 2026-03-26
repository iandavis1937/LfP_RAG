"""
Data classes for the retrieval system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    id: int
    text: str
    source_file: str
    chunk_index: int
    interview_type: str
    interviewee_type: str
    location: str
    date: str
    line_start: int = 0
    line_end: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        d.setdefault("line_start", 0)
        d.setdefault("line_end", 0)
        return cls(**d)


@dataclass
class SummaryRecord:
    filename: str
    summary_text: str
    faiss_id: int
