#!/usr/bin/env python3
"""
compare_output.py — Compare RAG query result files and generate an HTML report.

Usage:
    python compare_output.py                         # compare all .txt files in queries/
    python compare_output.py file1.txt file2.txt     # compare specific files
    python compare_output.py -o report.html          # custom output path
"""

import argparse
import html
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Meta:
    interview_type: str = "unknown"
    interviewee_type: str = "unknown"
    location: str = "unknown"
    date: str = "unknown"
    query_type: str = "unknown"


@dataclass
class Entry:
    rank: int
    source: str
    score: float
    meta: Meta
    passage_raw: str
    lines: str = ""


@dataclass
class RunData:
    label: str
    filename: str
    query: str
    result_count: int
    entries: list = field(default_factory=list)


@dataclass
class AppearanceInfo:
    rank: int
    score: float
    passage_raw: str
    meta: Meta
    lines: str = ""


@dataclass
class ChunkInfo:
    source: str
    chunk_key: str
    full_text: str = ""
    appearances: dict = field(default_factory=dict)  # {run_label: AppearanceInfo}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_file(path: Path, label: str) -> RunData:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n-{80}\n", content)

    # Parse file header from block 0
    header_block = blocks[0]
    query = ""
    result_count = 0
    for line in header_block.splitlines():
        if line.startswith("Query:"):
            query = line[len("Query:"):].strip()
        elif line.startswith("Results:"):
            try:
                result_count = int(line[len("Results:"):].strip())
            except ValueError:
                pass

    run = RunData(label=label, filename=path.name, query=query,
                  result_count=result_count)

    # Entry blocks: block 0 contains entry [1], blocks 1..N-2 contain entries [2..N],
    # last block is usually a trailing newline
    entry_blocks = []
    # From block 0, extract everything from the first "[1]" line onward
    eq_line = "=" * 80
    if eq_line in header_block:
        after_header = header_block[header_block.index(eq_line) + len(eq_line):]
        entry_blocks.append(after_header.strip())
    # Remaining blocks (skip last if it's just whitespace)
    for b in blocks[1:]:
        stripped = b.strip()
        if stripped:
            entry_blocks.append(stripped)

    for block in entry_blocks:
        entry = _parse_entry_block(block)
        if entry is not None:
            run.entries.append(entry)

    return run


def _parse_entry_block(block: str) -> Entry | None:
    lines = block.splitlines()
    if not lines:
        return None

    # Find entry header line: [N] filename.txt
    header_match = None
    header_idx = 0
    for i, line in enumerate(lines):
        m = re.match(r"^\[(\d+)\]\s+(.+)", line.strip())
        if m:
            header_match = m
            header_idx = i
            break
    if header_match is None:
        return None

    rank = int(header_match.group(1))
    source = header_match.group(2).strip()

    # Parse metadata lines (up to 8 key: value lines after header)
    meta = Meta()
    meta_end_idx = header_idx + 1
    meta_map = {
        "interview type": "interview_type",
        "interviewee type": "interviewee_type",
        "location": "location",
        "date": "date",
        "query type": "query_type",
        "score": None,   # handled separately
        "lines": None,   # handled separately
    }
    score = 0.0
    lines_str = ""
    for i in range(header_idx + 1, min(header_idx + 12, len(lines))):
        line = lines[i].strip()
        if ":" in line:
            key, _, val = line.partition(":")
            key_lower = key.strip().lower()
            val = val.strip()
            if key_lower in meta_map:
                attr = meta_map[key_lower]
                if attr is not None:
                    setattr(meta, attr, val)
                elif key_lower == "score":
                    try:
                        score = float(val)
                    except ValueError:
                        pass
                elif key_lower == "lines":
                    lines_str = val
                meta_end_idx = i + 1
        elif line == "":
            if i > header_idx + 1:
                meta_end_idx = i
                break

    # Passage is everything after the metadata block (skip leading blank lines)
    passage_lines = lines[meta_end_idx:]
    while passage_lines and passage_lines[0].strip() == "":
        passage_lines = passage_lines[1:]
    passage_raw = "\n".join(passage_lines).rstrip()

    return Entry(rank=rank, source=source, score=score, meta=meta,
                 passage_raw=passage_raw, lines=lines_str)


# ---------------------------------------------------------------------------
# Chunk key (for identity comparison across runs)
# ---------------------------------------------------------------------------

def strip_markup(passage_raw: str) -> str:
    """Strip all RAG markup and return normalized plain text."""
    text = passage_raw
    # Strip triple-bracket BM25 inside FAISS: [[[**word**]]]
    for _ in range(5):
        prev = text
        text = re.sub(r"\[\[\[(\*\*)?([^\[\]]*?)(\*\*)?\]\]\]", r"\2", text)
        if text == prev:
            break
    # Strip FAISS double-brackets [[sentence]]
    text = re.sub(r"\[\[(.+?)\]\]", r"\1", text, flags=re.DOTALL)
    # Strip fallback FAISS markers [~sentence~]
    text = re.sub(r"\[~(.+?)~\]", r"\1", text, flags=re.DOTALL)
    # Strip BM25 markers [**term**]
    text = re.sub(r"\[\*\*(.+?)\*\*\]", r"\1", text)
    # Strip any remaining bare ** bold markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Strip anchor chunk delimiters |---[anchor]---|
    text = re.sub(r"\|---\[(.+?)\]---\|", r"\1", text)
    # Strip stray brackets
    text = text.replace("[", "").replace("]", "")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_chunk_key(passage_raw: str) -> str:
    """Return first 120 chars of stripped text as a dict key."""
    return strip_markup(passage_raw)[:120]


# ---------------------------------------------------------------------------
# Unification and grouping
# ---------------------------------------------------------------------------

def unify_runs(runs: list[RunData]) -> dict:
    """Build master chunk dict keyed on (source, chunk_key)."""
    chunks: dict[tuple, ChunkInfo] = {}
    for run in runs:
        seen_keys: dict[tuple, int] = {}  # key -> best rank seen
        for entry in run.entries:
            key = (entry.source, make_chunk_key(entry.passage_raw))
            if key in seen_keys:
                # Keep the entry with lower rank (= higher relevance)
                if entry.rank < seen_keys[key]:
                    seen_keys[key] = entry.rank
                    chunks[key].appearances[run.label] = AppearanceInfo(
                        rank=entry.rank, score=entry.score,
                        passage_raw=entry.passage_raw, meta=entry.meta,
                        lines=entry.lines)
                continue
            seen_keys[key] = entry.rank
            if key not in chunks:
                chunks[key] = ChunkInfo(source=entry.source, chunk_key=key[1],
                                        full_text=strip_markup(entry.passage_raw))
            chunks[key].appearances[run.label] = AppearanceInfo(
                rank=entry.rank, score=entry.score,
                passage_raw=entry.passage_raw, meta=entry.meta,
                lines=entry.lines)
    return chunks


def _parse_line_start(lines_str: str) -> int:
    """Extract the start line number from a 'N–M' string; returns maxint if absent."""
    m = re.match(r"(\d+)", lines_str)
    return int(m.group(1)) if m else 10 ** 9


def group_chunks(chunks: dict, run_labels: list[str], sort_by: str = "line") -> dict:
    run_set = set(run_labels)
    groups = {"all": [], "some": [], "one": []}
    for chunk in chunks.values():
        present = set(chunk.appearances.keys())
        if present == run_set:
            groups["all"].append(chunk)
        elif len(present) >= 2:
            groups["some"].append(chunk)
        else:
            groups["one"].append(chunk)

    if sort_by == "line":
        def line_key(c):
            return min(_parse_line_start(a.lines) for a in c.appearances.values())
        for lst in groups.values():
            lst.sort(key=line_key)
    else:
        def avg_score(c):
            scores = [a.score for a in c.appearances.values()]
            return sum(scores) / len(scores) if scores else 0

        def max_score(c):
            return max((a.score for a in c.appearances.values()), default=0)

        groups["all"].sort(key=avg_score, reverse=True)
        groups["some"].sort(key=lambda c: (len(c.appearances), max_score(c)), reverse=True)
        groups["one"].sort(key=max_score, reverse=True)

    return groups


def merge_overlapping_chunks(chunks: dict) -> tuple[dict, int]:
    """Merge ChunkInfos where one passage completely contains another.

    When chunk A's stripped text contains chunk B's stripped text (or vice
    versa), they represent the same interview passage retrieved at different
    context window widths. Merging them causes the combined entry to be
    counted as shared across runs rather than unique to each.

    Chunks that share a run label are never merged: both passages were
    independently retrieved within the same run, so they remain distinct.

    Returns the updated chunks dict and the number of merges performed.
    """
    # Group chunk keys by source file
    by_source: dict[str, list] = {}
    for key, chunk in chunks.items():
        by_source.setdefault(chunk.source, []).append(key)

    to_absorb: dict = {}   # absorbed_key -> key_to_keep

    for keys in by_source.values():
        for i, key_a in enumerate(keys):
            if key_a in to_absorb:
                continue
            chunk_a = chunks[key_a]
            for key_b in keys[i + 1:]:
                if key_b in to_absorb:
                    continue
                chunk_b = chunks[key_b]
                # Never merge if the same run produced both chunks
                if set(chunk_a.appearances) & set(chunk_b.appearances):
                    continue
                # Check containment (compare full stripped texts)
                if chunk_a.full_text in chunk_b.full_text or chunk_b.full_text in chunk_a.full_text:
                    # Keep the larger passage as representative
                    if len(chunk_a.full_text) >= len(chunk_b.full_text):
                        keep, absorb = key_a, key_b
                    else:
                        keep, absorb = key_b, key_a
                    to_absorb[absorb] = keep

    # Apply merges: copy appearances from absorbed chunks into their keeper
    for absorb_key, keep_key in to_absorb.items():
        for run_label, appearance in chunks[absorb_key].appearances.items():
            chunks[keep_key].appearances.setdefault(run_label, appearance)

    merged = {k: v for k, v in chunks.items() if k not in to_absorb}
    return merged, len(to_absorb)


# ---------------------------------------------------------------------------
# Passage HTML rendering
# ---------------------------------------------------------------------------

def render_passage_html(passage_raw: str) -> str:
    """Convert raw passage text (with markup) to safe HTML."""
    # First, strip deeply nested triple-bracket BM25
    text = passage_raw
    for _ in range(5):
        prev = text
        text = re.sub(r"\[\[\[(\*\*)?([^\[\]]*?)(\*\*)?\]\]\]", r"\2", text)
        if text == prev:
            break

    # We need to convert markup → HTML before escaping.
    # Strategy: replace markup tokens with placeholder strings, escape, then
    # replace placeholders with HTML tags.
    OPEN_BM25       = "\x00BM25_OPEN\x00"
    CLOSE_BM25      = "\x00BM25_CLOSE\x00"
    OPEN_FAISS      = "\x00FAISS_OPEN\x00"
    CLOSE_FAISS     = "\x00FAISS_CLOSE\x00"
    OPEN_FAISS_FB   = "\x00FAISS_FB_OPEN\x00"
    CLOSE_FAISS_FB  = "\x00FAISS_FB_CLOSE\x00"

    # Replace FAISS [[...]] — may span newlines
    def replace_faiss(m):
        return OPEN_FAISS + m.group(1) + CLOSE_FAISS
    text = re.sub(r"\[\[(.+?)\]\]", replace_faiss, text, flags=re.DOTALL)

    # Replace fallback FAISS [~...~] — may span newlines
    def replace_faiss_fb(m):
        return OPEN_FAISS_FB + m.group(1) + CLOSE_FAISS_FB
    text = re.sub(r"\[~(.+?)~\]", replace_faiss_fb, text, flags=re.DOTALL)

    # Replace BM25 [**...**]
    def replace_bm25(m):
        return OPEN_BM25 + m.group(1) + CLOSE_BM25
    text = re.sub(r"\[\*\*(.+?)\*\*\]", replace_bm25, text)

    # Now HTML-escape (safe for Spanish chars, angle brackets, etc.)
    text = html.escape(text)

    # Swap placeholders for HTML tags
    text = text.replace(html.escape(OPEN_BM25),      '<mark class="bm25">')
    text = text.replace(html.escape(CLOSE_BM25),     '</mark>')
    text = text.replace(html.escape(OPEN_FAISS),     '<span class="faiss">')
    text = text.replace(html.escape(CLOSE_FAISS),    '</span>')
    text = text.replace(html.escape(OPEN_FAISS_FB),  '<span class="faiss-fallback">')
    text = text.replace(html.escape(CLOSE_FAISS_FB), '</span>')

    # Newlines → <br>
    text = text.replace("\n", "<br>\n")
    return text


# ---------------------------------------------------------------------------
# Run label derivation
# ---------------------------------------------------------------------------

def derive_labels(paths: list[Path]) -> list[str]:
    """Strip shared prefix and .txt suffix to create short run labels."""
    stems = [p.stem for p in paths]
    if len(stems) < 2:
        return stems

    # Find longest common prefix
    prefix = stems[0]
    for s in stems[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break

    labels = [s[len(prefix):].lstrip("_") or s for s in stems]

    # Ensure uniqueness
    seen: dict[str, int] = {}
    result = []
    for lbl in labels:
        if lbl in seen:
            seen[lbl] += 1
            result.append(f"{lbl}_{seen[lbl]}")
        else:
            seen[lbl] = 0
            result.append(lbl)
    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

RUN_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
]

CSS = """
* { box-sizing: border-box; }
body { font-family: Georgia, serif; margin: 0; padding: 0; background: #f5f5f5; color: #222; }
h1 { margin: 0; }
h2 { margin-top: 1.5em; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
h3 { margin: 0 0 6px 0; font-size: 1em; }

/* Sticky nav */
nav { position: sticky; top: 0; background: #2c3e50; color: white; padding: 8px 20px;
      display: flex; align-items: center; gap: 16px; z-index: 100; font-size: 0.9em; flex-wrap: wrap; }
nav a { color: #aed6f1; text-decoration: none; white-space: nowrap; }
nav a:hover { color: white; text-decoration: underline; }
nav .nav-title { font-weight: bold; color: white; font-size: 1em; margin-right: 8px; }

/* Main container */
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }

/* Coverage visualization */
.viz-scroll { overflow-x: auto; }
.viz-sources { display: flex; gap: 32px; align-items: flex-start; padding: 12px 0; }
.viz-source { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.viz-source-label { font-size: 0.78em; color: #555; text-align: center; max-width: 180px;
                    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.viz-legend { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 14px 0; font-size: 0.82em; }
.viz-legend-item { display: flex; align-items: center; gap: 5px; }
.viz-legend-swatch { width: 10px; height: 18px; border-radius: 2px; flex-shrink: 0; }

/* Run cards */
.run-cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.run-card { border-left: 6px solid #ccc; background: white; padding: 12px 16px;
            border-radius: 4px; min-width: 200px; max-width: 300px;
            box-shadow: 0 1px 3px rgba(0,0,0,.1); flex: 1; }
.run-card .run-label-badge { display: inline-block; color: white; padding: 2px 8px;
                              border-radius: 3px; font-weight: bold; margin-bottom: 6px; }
.run-card p { margin: 3px 0; font-size: 0.85em; color: #555; }
.run-card .query-text { font-style: italic; color: #333; }

/* Summary table */
.table-wrapper { overflow-x: auto; margin: 16px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.85em; background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); }
th { background: #2c3e50; color: white; padding: 8px 10px; text-align: left;
     white-space: nowrap; }
td { padding: 6px 10px; vertical-align: top; border-bottom: 1px solid #eee; }
tr:last-child td { border-bottom: none; }
tr.in-all  { background: #d4edda; }
tr.in-some { background: #fff3cd; }
tr.in-one  { background: #f8f9fa; }
.rank-score { font-weight: bold; }
.absent { color: #bbb; font-style: italic; font-size: 0.9em; }
.passage-preview { max-width: 360px; white-space: nowrap; overflow: hidden;
                   text-overflow: ellipsis; color: #555; font-size: 0.9em; }

/* Chunk cards */
.chunk-card { background: white; border: 1px solid #ddd; border-radius: 6px;
              margin: 12px 0; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.chunk-card.in-all  { border-left: 5px solid #28a745; }
.chunk-card.in-some { border-left: 5px solid #ffc107; }
.chunk-card.in-one  { border-left: 5px solid #6c757d; }
.chunk-header { padding: 10px 14px; background: #f8f9fa; display: flex;
                align-items: center; gap: 10px; flex-wrap: wrap; }
.source-tag { font-weight: bold; font-size: 0.9em; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 3px;
         font-size: 0.78em; font-weight: bold; color: white; }
.badge-all  { background: #28a745; }
.badge-some { background: #e0a800; }
.badge-one  { background: #6c757d; }
.diff-warn  { background: #dc3545; color: white; padding: 2px 6px;
              border-radius: 3px; font-size: 0.78em; }

/* Passage grid */
.passage-grid { display: grid; gap: 1px; background: #ddd; }
.passage-col { background: white; padding: 12px; }
.run-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.run-label-sm { color: white; padding: 2px 8px; border-radius: 3px;
                font-weight: bold; font-size: 0.85em; }
.run-meta { font-size: 0.8em; color: #666; margin-bottom: 8px; }
.not-retrieved { color: #bbb; font-style: italic; padding: 20px 0; }

/* Passage text */
details summary { cursor: pointer; font-size: 0.85em; color: #0066cc;
                  padding: 4px 0; user-select: none; }
details summary:hover { color: #003d80; }
.passage-text { margin-top: 8px; font-size: 0.88em; line-height: 1.65;
                border-top: 1px solid #eee; padding-top: 8px; }
mark.bm25          { background: #fff9c4; font-weight: 600; padding: 0 1px; border-radius: 2px; }
span.faiss         { background: #e3f2fd; padding: 0 2px; border-radius: 2px; }
span.faiss-fallback { background: #f3e8ff; padding: 0 2px; border-radius: 2px;
                      border-bottom: 1px dashed #9b59b6; }

/* Toggle buttons */
.toggle-btns { margin: 8px 0; font-size: 0.82em; }
.toggle-btns button { background: none; border: 1px solid #aaa; border-radius: 3px;
                       padding: 2px 8px; cursor: pointer; color: #555; margin-right: 4px; }
.toggle-btns button:hover { background: #eee; }

/* Section intro */
.section-intro { font-size: 0.88em; color: #555; margin: 4px 0 12px 0; }
.count-badge { display: inline-block; background: #555; color: white;
               border-radius: 10px; padding: 1px 8px; font-size: 0.82em; margin-left: 4px; }
"""

JS = """
function toggleSection(id, open) {
  document.querySelectorAll('#' + id + ' details')
    .forEach(function(d) { d.open = open; });
}
"""


def _color(run_labels: list[str], label: str) -> str:
    idx = run_labels.index(label) % len(RUN_COLORS)
    return RUN_COLORS[idx]


def _passage_preview(passage_raw: str, length: int = 80) -> str:
    key = make_chunk_key(passage_raw)
    return html.escape(key[:length])


def _texts_differ(appearances: dict, run_labels: list[str]) -> bool:
    keys = [make_chunk_key(appearances[l].passage_raw)
            for l in run_labels if l in appearances]
    return len(set(keys)) > 1


def build_coverage_svg(chunks: dict, run_labels: list[str]) -> str:
    """Build an SVG line-coverage visualization grouped by source file.

    For each source file, renders one narrow vertical bar per run. Colored
    segments within each bar mark the line ranges of retrieved passages.
    Returns an HTML string (one SVG per source file) or empty string if no
    line data is available.
    """
    BAR_W    = 10    # px width of each run bar
    BAR_GAP  = 3     # px gap between adjacent run bars
    VIZ_H    = 300   # px height of the bar area
    PAD_TOP  = 8     # px above the bars
    TICK_W   = 36    # px reserved on the left for line-number tick labels
    PAD_R    = 8     # px padding to the right of the last bar
    MIN_SEG  = 3     # minimum segment height in px

    # Collect: source -> run_label -> [(line_start, line_end), ...]
    coverage: dict[str, dict[str, list]] = {}
    for chunk in chunks.values():
        for run_label, app in chunk.appearances.items():
            if not app.lines:
                continue
            m = re.match(r"(\d+)\D+(\d+)", app.lines)
            if not m:
                continue
            ls, le = int(m.group(1)), int(m.group(2))
            coverage.setdefault(chunk.source, {}).setdefault(run_label, []).append((ls, le))

    if not coverage:
        return ""

    n_runs   = len(run_labels)
    bar_span = n_runs * BAR_W + (n_runs - 1) * BAR_GAP
    svg_w    = TICK_W + bar_span + PAD_R
    svg_h    = PAD_TOP + VIZ_H + 4   # 4px margin below bars

    svgs = []
    for source in sorted(coverage):
        run_data = coverage[source]

        all_starts = [ls for segs in run_data.values() for ls, _ in segs]
        all_ends   = [le for segs in run_data.values() for _, le in segs]
        min_line   = min(all_starts)
        max_line   = max(all_ends)
        span       = max_line - min_line or 1

        def to_y(line: int) -> float:
            return PAD_TOP + (line - min_line) / span * VIZ_H

        p = []
        p.append(f'<svg width="{svg_w}" height="{svg_h}" '
                 f'xmlns="http://www.w3.org/2000/svg">')

        # Faint horizontal guide lines and tick labels at 0%, 25%, 50%, 75%, 100%
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            line_num = round(min_line + frac * span)
            y = PAD_TOP + frac * VIZ_H
            p.append(f'<line x1="{TICK_W}" y1="{y:.1f}" '
                     f'x2="{TICK_W + bar_span}" y2="{y:.1f}" '
                     f'stroke="#e0e0e0" stroke-width="1"/>')
            p.append(f'<text x="{TICK_W - 4}" y="{y + 3.5:.1f}" '
                     f'font-size="9" fill="#aaa" text-anchor="end" '
                     f'font-family="monospace">{line_num}</text>')

        # One bar per run
        for ri, run_label in enumerate(run_labels):
            x     = TICK_W + ri * (BAR_W + BAR_GAP)
            color = RUN_COLORS[ri % len(RUN_COLORS)]

            # Background track (full extent of retrieved lines)
            p.append(f'<rect x="{x}" y="{PAD_TOP}" width="{BAR_W}" '
                     f'height="{VIZ_H}" fill="#ebebeb" rx="2"/>')

            # Colored segments for each retrieved passage
            for ls, le in run_data.get(run_label, []):
                y1    = to_y(ls)
                seg_h = max(to_y(le) - y1, MIN_SEG)
                p.append(f'<rect x="{x}" y="{y1:.1f}" width="{BAR_W}" '
                         f'height="{seg_h:.1f}" fill="{color}" '
                         f'rx="1" opacity="0.85"/>')

        p.append('</svg>')
        svgs.append({
            "svg": "".join(p),
            "label": html.escape(source),
            "range": f"lines {min_line}–{max_line}",
        })

    if not svgs:
        return ""

    # Legend
    legend_items = "".join(
        f'<span class="viz-legend-item">'
        f'<span class="viz-legend-swatch" style="background:{RUN_COLORS[i % len(RUN_COLORS)]}"></span>'
        f'{html.escape(lbl)}</span>'
        for i, lbl in enumerate(run_labels)
    )
    legend = f'<div class="viz-legend">{legend_items}</div>'

    # Source panels
    panels = "".join(
        f'<div class="viz-source">'
        f'{s["svg"]}'
        f'<span class="viz-source-label" title="{s["label"]}">'
        f'{s["label"]}<br><span style="color:#aaa">{s["range"]}</span></span>'
        f'</div>'
        for s in svgs
    )

    return (f'<h2 id="coverage">Line Coverage</h2>'
            f'<p class="section-intro">Each column is one run. Colored segments '
            f'mark the line ranges of retrieved passages within the source file.</p>'
            f'{legend}'
            f'<div class="viz-scroll"><div class="viz-sources">{panels}</div></div>')


def build_html(runs: list[RunData], groups: dict, chunks: dict,
               run_labels: list[str], sort_by: str = "line") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_all  = len(groups["all"])
    n_some = len(groups["some"])
    n_one  = len(groups["one"])

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Run Comparison — {html.escape(ts)}</title>
<style>{CSS}</style>
<script>{JS}</script>
</head>
<body>
<nav>
  <span class="nav-title">RAG Comparison</span>
  <a href="#runs">Runs</a>
  <a href="#coverage">Line Coverage</a>
  <a href="#summary">Summary Table</a>
  <a href="#in-all">In All Runs ({n_all})</a>
  <a href="#in-some">In Some Runs ({n_some})</a>
  <a href="#unique">Unique ({n_one})</a>
</nav>
<div class="container">
<h1 id="runs">RAG Query Comparison Report</h1>
<p style="color:#666;font-size:.9em">Generated: {html.escape(ts)} &nbsp;|&nbsp; {len(runs)} runs compared &nbsp;|&nbsp; Sorted by: {"line number" if sort_by == "line" else "relevance score"}</p>
""")

    # Run cards
    parts.append('<div class="run-cards">')
    for run in runs:
        color = _color(run_labels, run.label)
        parts.append(f"""
<div class="run-card" style="border-left-color:{color}">
  <span class="run-label-badge" style="background:{color}">{html.escape(run.label)}</span>
  <p><strong>File:</strong> {html.escape(run.filename)}</p>
  <p class="query-text">{html.escape(run.query)}</p>
  <p><strong>Results returned:</strong> {run.result_count}</p>
</div>""")
    parts.append('</div>')

    # Line coverage visualization
    coverage_html = build_coverage_svg(chunks, run_labels)
    if coverage_html:
        parts.append(coverage_html)

    # Summary table
    parts.append(f'<h2 id="summary">Summary Matrix</h2>')
    parts.append('<p class="section-intro">Each row is a distinct passage chunk. '
                 'Green = retrieved in all runs; yellow = some runs; gray = one run only.</p>')
    parts.append('<div class="table-wrapper"><table>')
    # Header row
    parts.append('<thead><tr><th>Source File</th><th>Passage (start)</th>')
    for lbl in run_labels:
        color = _color(run_labels, lbl)
        parts.append(f'<th style="background:{color}">{html.escape(lbl)}</th>')
    parts.append('</tr></thead><tbody>')

    for group_key, row_class in [("all", "in-all"), ("some", "in-some"), ("one", "in-one")]:
        for chunk in groups[group_key]:
            parts.append(f'<tr class="{row_class}">')
            parts.append(f'<td>{html.escape(chunk.source)}</td>')
            parts.append(f'<td class="passage-preview">{_passage_preview(next(iter(chunk.appearances.values())).passage_raw)}</td>')
            for lbl in run_labels:
                if lbl in chunk.appearances:
                    a = chunk.appearances[lbl]
                    lines_cell = f' &nbsp; <small>{html.escape(a.lines)}</small>' if a.lines else ""
                    parts.append(f'<td class="rank-score">#{a.rank} &nbsp; <small>{a.score:.6f}</small>{lines_cell}</td>')
                else:
                    parts.append('<td class="absent">—</td>')
            parts.append('</tr>')

    parts.append('</tbody></table></div>')

    # Detailed sections
    for group_key, section_id, section_title, badge_class, badge_label, intro in [
        ("all",  "in-all",  "In All Runs",   "badge-all",  "All runs",
         "These passages were retrieved regardless of parameter settings — highest-confidence results."),
        ("some", "in-some", "In Some Runs",  "badge-some", "Some runs",
         "These passages appear in multiple but not all runs — parameter-sensitive results."),
        ("one",  "unique",  "Unique to One Run", "badge-one", "1 run",
         "These passages were retrieved in only one run — unique to a specific parameter configuration."),
    ]:
        n = len(groups[group_key])
        parts.append(f'<h2 id="{section_id}">{section_title} <span class="count-badge">{n}</span></h2>')
        parts.append(f'<p class="section-intro">{intro}</p>')
        if n == 0:
            parts.append('<p style="color:#999;font-style:italic">None.</p>')
            continue
        parts.append(f'<div class="toggle-btns">'
                     f'<button onclick="toggleSection(\'{section_id}\', true)">Expand all</button>'
                     f'<button onclick="toggleSection(\'{section_id}\', false)">Collapse all</button>'
                     f'</div>')

        for chunk in groups[group_key]:
            present_labels = [l for l in run_labels if l in chunk.appearances]
            differ = _texts_differ(chunk.appearances, present_labels)
            diff_badge = ' <span class="diff-warn">text differs</span>' if differ else ""

            parts.append(f'<div class="chunk-card {group_key if group_key != "one" else "in-one"}">')
            parts.append(f'<div class="chunk-header">'
                         f'<span class="source-tag">{html.escape(chunk.source)}</span>'
                         f'<span class="badge {badge_class}">{badge_label}</span>'
                         f'{diff_badge}'
                         f'</div>')

            # Grid: one column per run (or just present runs for "one" group)
            display_labels = run_labels if group_key != "one" else present_labels
            cols = len(display_labels)
            parts.append(f'<div class="passage-grid" style="grid-template-columns: repeat({cols}, 1fr)">')

            for lbl in display_labels:
                color = _color(run_labels, lbl)
                if lbl in chunk.appearances:
                    a = chunk.appearances[lbl]
                    passage_html = render_passage_html(a.passage_raw)
                    parts.append(f"""
<div class="passage-col">
  <div class="run-header">
    <span class="run-label-sm" style="background:{color}">{html.escape(lbl)}</span>
  </div>
  <div class="run-meta">Rank #{a.rank} &nbsp;|&nbsp; Score: {a.score:.6f}{f" &nbsp;|&nbsp; Lines: {html.escape(a.lines)}" if a.lines else ""}</div>
  <details>
    <summary>Show passage</summary>
    <div class="passage-text">{passage_html}</div>
  </details>
</div>""")
                else:
                    parts.append(f"""
<div class="passage-col">
  <div class="run-header">
    <span class="run-label-sm" style="background:{color}">{html.escape(lbl)}</span>
  </div>
  <div class="not-retrieved">Not retrieved in this run</div>
</div>""")

            parts.append('</div>')  # passage-grid
            parts.append('</div>')  # chunk-card

    parts.append('</div></body></html>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare RAG query result files and generate an HTML report.")
    parser.add_argument("files", nargs="*",
                        help="Result .txt files to compare. Defaults to all .txt "
                             "files in queries/compare/ relative to this script.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML path. Defaults to queries/compare/comparison_TIMESTAMP.html")
    parser.add_argument("--sort", "-s", choices=["line", "rank"], default="line",
                        help="Order passages by source line number (default) or relevance rank.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    queries_dir = script_dir / "queries/compare/"

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(queries_dir.glob("*.txt"))
        if not paths:
            sys.exit(f"No .txt files found in {queries_dir}")

    for p in paths:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    if len(paths) < 2:
        sys.exit("Need at least 2 files to compare.")

    labels = derive_labels(paths)

    print(f"Comparing {len(paths)} files:")
    for p, lbl in zip(paths, labels):
        print(f"  [{lbl}] {p.name}")

    runs = [parse_file(p, lbl) for p, lbl in zip(paths, labels)]

    for run in runs:
        print(f"  {run.label}: parsed {len(run.entries)} entries "
              f"(file says {run.result_count})")

    chunks = unify_runs(runs)
    chunks, n_merged = merge_overlapping_chunks(chunks)
    if n_merged:
        print(f"  Merged {n_merged} overlapping chunk pair(s) across runs")
    groups = group_chunks(chunks, labels, sort_by=args.sort)

    print(f"\nChunks: {len(groups['all'])} in all runs, "
          f"{len(groups['some'])} in some runs, "
          f"{len(groups['one'])} unique to one run")

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = queries_dir / f"comparison_{ts}.html"

    html_content = build_html(runs, groups, chunks, labels, sort_by=args.sort)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
