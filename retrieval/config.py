"""
Configuration constants for the qualitative interview retrieval system.
"""

from __future__ import annotations
from pathlib import Path

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------------

CORPUS_DIR    = str(_HERE / ".." / "transcripts" / "test")  # input directory; each transcript should be one text file
BUILD_INDEX   = True
INDEX_DIR     = str(_HERE / ".." / "intermediate" / "index_storage")
OUTPUT_DIR    = str(_HERE / ".." / "output" / "queries" / "full_corpus_3_24")
OUTPUT_SUFFIX = "F5_B5"  # ""; appended to every output filename as __{suffix}; e.g. "w2_bm25" → armed_groups__all__w2_bm25.txt
EXPORT_CSV    = True  # Set True to also write a .csv alongside each query .txt output
EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "gpt-4o-mini"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 20
MIN_SCORE     = 0.005  # minimum RRF score; passages below this threshold are excluded
RRF_K         = 60
SUMMARY_SEG_TOKENS = 3000
USE_SUMMARY_WEIGHTS  = False  # multiply RRF scores by document summary relevance; disable to treat all docs equally
USE_CLASSIFIER       = False  # set True to classify query type via LLM; if False, use QUERY_TYPE or MANUAL_QUERY_PARAMS
QUERY_TYPE          = "analytical"  # used when USE_CLASSIFIER=False and MANUAL_QUERY_PARAMS is None; one of QUERY_TYPE_PARAMS keys
MANUAL_QUERY_PARAMS = {"w_faiss": 0.5, "w_bm25": 0.5, "window": 1}   # e.g. None or {"w_faiss": 0.6, "w_bm25": 0.4, "window": 7}; overrides QUERY_TYPE when set
SHOW_QUERY_EXPANSION = True   # prefix output files with the FAISS query and full BM25-expanded query
SENTENCE_MARKUP_THRESH = 0.3

# Co-occurrence boost/filter — see Stage 3b in query_topics3_README.md
COOCCURRENCE_FILTER          = False           # global default; per-query third element overrides this
COOCCURRENCE_WINDOW          = 300             # character radius for secondary-topic proximity boost
COOCCURRENCE_PRIMARY_KEY     = "armed groups"  # SYNONYM_MAP key whose terms are most heavily-weighted (or filtered for)
COOCCURRENCE_SECONDARY_KEYS  = [               # SYNONYM_MAP keys that contribute the additive boost
    "violence", "extort", "threats", "restrict access",
]
# Chunks that contain both a primary-topic term AND a secondary-topic term within
# COOCCURRENCE_WINDOW characters receive a ×1.5 score boost.
# When COOCCURRENCE_FILTER is True (or a per-query "require_primary" override is set),
# chunks that contain no primary-topic term are dropped after fusion.
# Per-query override: add a third element to any query_d tuple, e.g.:
#   "armed_groups": (query_str, filters, {"require_primary": True, "cooccurrence_window": 200})


QUERY_TYPE_PARAMS = {
    "factual":    {"w_faiss": 0.4, "w_bm25": 0.6,"window": 5},
    "opinion":    {"w_faiss": 0.7, "w_bm25": 0.3, "window": 5},
    "analytical": {"w_faiss": 0.5, "w_bm25": 0.5, "window": 5},
    "contextual": {"w_faiss": 0.6, "w_bm25": 0.4, "window": 5},
}

_DEFAULT_WINDOW: int = (
    MANUAL_QUERY_PARAMS["window"] if MANUAL_QUERY_PARAMS is not None
    else QUERY_TYPE_PARAMS.get(QUERY_TYPE, QUERY_TYPE_PARAMS["analytical"])["window"]
)

SUMMARY_PROMPT = """You are analyzing a qualitative interview transcript. Write one short paragraph per topic that is clearly present in the text. Omit any topic that is not discussed. Do not invent content.

Topics to address if present:
(1) Government impact on conservation, economy, or communities
(2) Armed group impact on people, territory, or resources
(3) NGO or authority impact on communities or environment
(4) Inter-actor relationships (e.g., between government, armed groups, NGOs, and communities)

Text:
{text}
"""

CLASSIFY_PROMPT = """Classify the following query into exactly one of these categories: factual, opinion, analytical, contextual.

- factual: asks for a specific fact, event, date, or named entity
- opinion: asks what people think, feel, or believe
- analytical: asks for causes, mechanisms, or comparisons
- contextual: asks for background, setting, or broader context

Return only the single classification word, nothing else.

Query: {query}
"""
