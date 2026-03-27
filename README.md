# retrieval/ — Architecture and Design Decisions

Originally implemented as `query_topics3.py`. Refactored into the `retrieval/`
package. Replaces LlamaIndex + Ollama with OpenAI and a custom retrieval stack
incorporating fusion retrieval, hierarchical indices, context enrichment, and
adaptive query routing.

## Quickstart

```bash
# Clone repo
git clone https://github.com/iandavis1937/LfP_RAG

# Make a dedicated virtual environment
python -m venv LfP_RAG_env
source LfP_RAG_env/bin/activate

# Install libraries
pip install --upgrade pip
pip install -r requirements.txt

# Insert .env file with CO_API_KEY and OPENAI_API_KEY

# Run
python3 run_retrieval.py
```

---

## Parameters

All parameters are defined in `retrieval/config.py` unless noted otherwise.


| Parameter | Default | Usage |
|---|---|---|
| `BUILD_INDEX` | `True` | `True` rebuilds and persists the index from `CORPUS_DIR`; `False` loads from `INDEX_DIR`.<br> For analytical reproducibility, the index should be treated as a versioned artifact: if corpus files are added, corrected, or re-translated, a rebuild is required or subsequent queries silently draw on a stale evidence base.<br> Indexes built before line-number support was added will load without error (line numbers default to `0`), but will show `Lines: 0–0` in output. Set `BUILD_INDEX = True` once to regenerate the index with correct line numbers. |
| `CORPUS_DIR` | `"txt/test"` | Directory of `.txt` interview files to index.<br> Interviews absent from this directory are invisible to every query regardless of their substantive relevance — a quiet form of corpus truncation that will not produce an error. Confirm the directory contains the intended full corpus before running a build for analytical purposes. |
| `INDEX_DIR` | `"index_storage"` | Directory for persisted index files.<br> Useful for maintaining separate named indices when comparing runs across corpus versions or parameter configurations. |
| `OUTPUT_DIR` | `"queries"` | Directory for query result `.txt` files. |
| `EMBED_MODEL` | `"text-embedding-3-small"` | OpenAI embedding model used for chunks, summaries, and query vectors.<br> The model must be the same at index time and query time; a mismatch produces no error but silently corrupts similarity scores. If the model is changed, the index must be fully rebuilt. Larger models (e.g. `text-embedding-3-large`) improve semantic precision on nuanced thematic queries at higher API cost. For this corpus, the practical difference matters most for abstract analytical queries; factual/entity queries are dominated by BM25 regardless. |
| `LLM_MODEL` | `"gpt-4o-mini"` | OpenAI chat model used for document summarization and query classification. Is not used when `USE_CLASSIFIER=False` and `USE_SUMMARY_WEIGHTS=False` (the current defaults).<br> When either flag is enabled, the choice of model matters for classification accuracy and summary quality respectively. |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk.<br> Smaller chunks improve retrieval precision (each chunk is more topically uniform) but increase the risk of splitting a key observation mid-sentence and losing its framing. Larger chunks preserve more conversational context within each retrieved unit but dilute the relevance signal — a chunk containing one relevant sentence surrounded by 600 characters of preamble may rank lower than it deserves. For interview transcripts, 800 chars is roughly 1–2 speaker turns; the optimal value depends on how densely relevant content is distributed across the question guide. Changes require a full index rebuild. |
| `CHUNK_OVERLAP` | `100` | Characters shared between consecutive chunks; used by the indexer and by overlap-trimming in `enrich()`.<br> Higher overlap reduces the probability that a key passage straddling a chunk boundary is missed, at the cost of increased index size and more deduplication work in `enrich()`. At 100 chars (~1 sentence), most boundary effects are mitigated for this corpus. Changes require a full index rebuild. |
| `TOP_K` | `20` | Number of candidate chunks retrieved by each retriever (FAISS and BM25) before fusion.<br> Higher values increase recall — a passage that ranks 25th in FAISS but 1st in BM25 will be captured at `TOP_K=30` but missed at `TOP_K=20`. Lower values reduce noise and speed up enrichment. For a 79-interview corpus with ~1,400 chunks, `TOP_K=20` means each retriever considers roughly 1.4% of chunks; raising to 40 doubles recall coverage at modest cost. |
| `MIN_SCORE` | `0.001` | Minimum RRF score; results below this are discarded.<br> The default is very permissive and mainly excludes passages that appeared in neither retriever's top-K list. Raising this threshold (e.g. to `0.01`) trims low-confidence results that passed fusion on weak signals from both retrievers, producing a shorter but higher-precision output. Useful when the raw result count is too large for manual review. |
| `USE_SUMMARY_WEIGHTS` | `False` | `True` multiplies each chunk's RRF score by its document's summary cosine similarity.<br> When enabled, results skew toward interviews whose overall content is topically relevant to the query, suppressing chunks from interviews that mention query terms only incidentally. The methodological implication is that enabling this flag trades some recall for precision at the document level — appropriate for focused thematic queries, but potentially excluding relevant incidental mentions in off-topic interviews. Requires summary generation at index time. |
| `USE_CLASSIFIER` | `False` | `True` calls the LLM to classify the query type; ignored when `MANUAL_QUERY_PARAMS` is set.<br> Adds per-query API latency and cost. Useful when running a heterogeneous batch of queries (mixing factual, opinion, and analytical types) and the appropriate weight balance genuinely varies. For a homogeneous query set, setting `QUERY_TYPE` directly is faster and produces identical results without the classification overhead or risk of misclassification. |
| `QUERY_TYPE` | `"analytical"` | Active when `USE_CLASSIFIER=False` and `MANUAL_QUERY_PARAMS=None`; one of `factual`, `opinion`, `analytical`, `contextual`.<br> Sets options for the BM25/FAISS weight ratio for the LLM to choose (see Stage 1 table below). The choice matters most for queries that are either entity-heavy (favor `factual`, boosting BM25) or abstractly phrased (favor `opinion`, boosting FAISS). For this corpus, `analytical` (equal weights) is the appropriate default for most research questions; switching to `factual` is advisable when querying for specific institutional names or program acronyms that the embedding model may not encode precisely. |
| `MANUAL_QUERY_PARAMS` | `{"w_faiss": 0.99, "w_bm25": 0.01, "window": 5}` | Directly sets fusion weights and context window; set to `None` to use `QUERY_TYPE` or the classifier.<br> For example, w_faiss = 1 and w_bm25 = 0 makes retrieval purely semantic — domain proper nouns receive almost no BM25 boost. This is appropriate for open-ended thematic queries but will underperform for named-entity lookups. The `window` parameter independently controls context enrichment radius: smaller values (e.g. `±2`) return tighter excerpts centered on the retrieved chunk; larger values (e.g. `±8`) return broader conversational context, which is useful when the key evidence is framed by the interviewer's preceding question. |
| `RRF_K` | `60` | Smoothing constant in the RRF denominator; higher values reduce the rank-gap between top and lower results.<br> At `k=60` (a well-validated default), rank differences are moderately smoothed — a chunk at rank 1 scores roughly 3× a chunk at rank 60. Lower values (e.g. `k=10`) amplify that gap, making top-ranked results dominate more strongly. For this corpus, `k=60` is appropriate; tuning this parameter is rarely necessary unless the retrieval ranking feels either too flat (raise k) or too winner-takes-all (lower k). |
| `SUMMARY_SEG_TOKENS` | `3000` | Maximum tokens per segment in map-reduce summarization. Applies only when `USE_SUMMARY_WEIGHTS=True`.<br> Smaller values cause longer transcripts to be split into more segments for the map phase, increasing API calls but improving coverage of content distributed across a long interview. Larger values risk the model attending unevenly to late content in long segments. For this corpus, most transcripts fall within a single pass at 3,000 tokens; this parameter primarily affects the longest FGD transcripts. |
| `COOCCURRENCE_FILTER` | `False` | Global default for the Stage 3b hard filter (`require_primary`). `True` drops any chunk that contains no primary-topic term after fusion. The proximity boost is controlled separately via `apply_boost` and does not require this flag to be set.<br> Can be overridden per query via the third element of the `query_d` tuple: `{"require_primary": True, "apply_boost": True, "cooccurrence_window": 200}`. The global constant sets the default only for `require_primary`; `apply_boost` defaults to `False` for all queries unless set explicitly in the per-query element. |
| `COOCCURRENCE_WINDOW` | `150` | Character radius for the secondary-topic proximity boost in Stage 3b. A primary-topic term and a secondary-topic term must both appear within this many characters of each other for the ×1.5 boost to apply.<br> 150 characters is roughly two short sentences — tight enough to require the primary actor and the secondary behavior to appear in close textual proximity, loose enough not to penalise sentences split across a clause boundary. Increase to ~300 for mechanism queries where causal framing takes several words to connect actor and behavior; decrease to ~75 for queries where the relevant claim is typically packed into a single clause. |
| `COOCCURRENCE_PRIMARY_KEY` | `"armed groups"` | The `SYNONYM_MAP` key whose terms must appear in a chunk for it to pass the Stage 3b hard filter. Changing this re-purposes the co-occurrence filter for a different anchor topic (e.g., `"government"`) without modifying the method logic. |
| `COOCCURRENCE_SECONDARY_KEYS` | `["violence", "extort", …]` | List of `SYNONYM_MAP` keys whose terms are checked for proximity to the primary-topic term. A chunk scores ×1.5 if any secondary term appears within `COOCCURRENCE_WINDOW` characters of any primary term. Keys not listed here neither contribute to the boost nor to filtering. Adjust when extending `SYNONYM_MAP` with new behavior categories. |
| `bm25_expand_keys` *(per-query)* | `None` | Per-query list of `SYNONYM_MAP` keys to force into BM25 expansion regardless of whether their canonical form appears in the query string. Set in the third element of a `query_d` entry. Use when the FAISS query string is deliberately actor-focused and omits behavior vocabulary that BM25 should still match. Has no effect on FAISS. See Stage 2b. |

---

## Design Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Question augmentation (HyPE/HyDE) | Dropped | API cost not justified; synonym expansion + fusion retrieval considered sufficient |
| 2 | Summary index scope | Soft weighting — boost chunks from top-K summaries (opt-in via `USE_SUMMARY_WEIGHTS`) | More robust for cross-document queries; avoids hard cutoff discarding relevant chunks; disabled by default so chunk-level signals alone determine ranking |
| 3 | Context enrichment assembly | Query-time from chunk store | Flexible; avoids storing redundant data at index time |
| 4 | LLM reranking | Dropped | Cost and latency not justified; fusion + adaptive routing considered sufficient |
| 5 | Highlighting signal — BM25 | Regex term matching on expanded query tokens | Free (no API call); multi-word phrases matched longest-first to prevent sub-phrase double-marking |
| 6 | Highlighting signal — FAISS | Sentence-level cosine similarity to query embedding | Single batched `embed()` call across all result passages; all sentences scoring >0.25 are marked with no cap; if none clear the threshold, only the single top-scoring sentence is marked (fallback) |
| 7 | Highlight markup | `[**term**]` (BM25), `[[sentence]]` (FAISS above threshold), `[~sentence~]` (FAISS fallback) | Plain-text readable; markdown-compatible bold inside BM25 brackets; `[~..~]` signals that no sentence cleared the 0.25 threshold — only the top scorer is marked; nesting is valid (`[[... [**term**]]]`) |

---

## Module Structure

```
qual/
├── run_retrieval.py          ← entry point: build or load index, run query_d, write output
└── retrieval/
    ├── __init__.py           ← re-exports public surface (IndexBuilder, HierarchicalRetriever, …)
    ├── config.py             ← all constants, QUERY_TYPE_PARAMS, _DEFAULT_WINDOW, prompt strings
    ├── vocabulary.py         ← SYNONYM_MAP, _synonym_to_regex(), _expand_synonym()
    ├── models.py             ← Chunk dataclass, SummaryRecord dataclass
    ├── llm_client.py         ← OpenAI client singleton, embed(), llm(), count_tokens(), truncate_to_tokens()
    ├── indexer.py
    │   ├── parse_metadata()
    │   └── class IndexBuilder
    │       ├── load_and_chunk()
    │       ├── build_faiss_chunk_index()
    │       ├── generate_summaries()          ← only called if USE_SUMMARY_WEIGHTS
    │       ├── build_faiss_summary_index()   ← only called if USE_SUMMARY_WEIGHTS
    │       └── persist() / load()            ← summary files conditional on USE_SUMMARY_WEIGHTS
    ├── retriever.py
    │   ├── class QueryRouter
    │   │   └── classify(query) → QueryType
    │   ├── class FusionRetriever
    │   │   ├── retrieve_faiss()
    │   │   ├── retrieve_bm25()
    │   │   ├── fuse_rrf(faiss_hits, bm25_hits, weights) → ranked chunks
    │   │   ├── _cooccurrence_score(chunk, window) → 0.0 | 1.0 | 1.5   ← Stage 3b
    │   │   ├── _get_primary_re()                                        ← compiled once, cached
    │   │   └── _get_secondary_re()                                      ← compiled once, cached
    │   └── class HierarchicalRetriever
    │       └── retrieve(query) → final passages (with highlighting applied)
    ├── enricher.py
    │   ├── class ContextEnricher
    │   │   └── enrich(chunk, chunk_store, window=5) → passage
    │   └── class TokenHighlighter
    │       ├── highlight(passages) → annotated passages  ← batched embed() call
    │       ├── _split_line(line) → sentences
    │       ├── _select_faiss(scores) → set of sentence indices
    │       └── _apply_bm25(text) → marked text           ← uses regex compiled once in __init__
    ├── output.py             ← write_results(), CSV export dispatch
    └── queries.py            ← query_d dict (domain query definitions — edit this file to add/change queries)
```

---

## Indexing Pipeline

### Step 1 — Load and chunk
Read files from `txt/test/` (configured via `CORPUS_DIR`). Split with `RecursiveCharacterTextSplitter`
(`add_start_index=True`). Assign each chunk a sequential `chunk_index` within its source document.
For each chunk, record its **start and end line numbers** in the source file: the splitter exposes
the character offset of each chunk (`start_index`), and line numbers are derived by counting
newlines before that offset (`text[:start_index].count("\n") + 1`). Line numbers are 1-based and
cover the full text of the chunk (not the enriched passage — that is computed at query time from
the window endpoints).

Parse the following metadata for each document at load time and attach it to every
chunk produced from that document:

Filenames follow the convention `{interview_type}_{interviewee_type}_{location}_{date}.txt`
(e.g. `KII_NGO_Caqueta_2024-03.txt`). Parse metadata by splitting on `_`:

| Field | Position | Example |
|---|---|---|
| `interview_type` | 0 | `KII` |
| `interviewee_type` | 1 | `NGO` |
| `location` | 2 | `Caqueta` |
| `date` | 3 (strip `.txt`) | `2024-03` |

If a filename does not match this pattern, fall back to reading the first lines
of the file. Unresolvable fields are stored as `"unknown"`.

### Step 2 — Build chunk-level indices
- **FAISS**: embed raw chunk text with `text-embedding-3-small`.
  Build a `IndexFlatIP` (inner product / cosine) index. **Note:** `IndexFlatIP`
  computes raw inner product — cosine similarity is only equivalent if vectors are
  L2-normalized before indexing and querying. `text-embedding-3-small` returns
  normalized vectors, but normalization must be applied explicitly if the embedding
  model is ever swapped.
- **BM25**: build a `BM25Okapi` index from raw chunk texts (tokenized).

### Step 3 — Generate topic-focused summaries and build summary index *(skipped if `USE_SUMMARY_WEIGHTS = False`)*
For each document, generate a structured summary focused on the following topics,
where present in the interview:

- Impact of **government** on conservation, the economy, and communities
- Impact of **armed groups** (paramilitaries, insurgents, dissidents) on conservation,
  the economy, and communities
- Impact of **NGOs and other authorities** on conservation, the economy, and communities
- Relationships and interactions **among** these actors

Use **map-reduce chunked summarization** to handle documents that exceed the token
limit, rather than simple truncation:

1. **Map**: split the document into token-limited segments. Call GPT-4o-mini on each
   segment with the topic-focused prompt, producing a partial summary per segment.
2. **Reduce**: concatenate the partial summaries and call GPT-4o-mini once more with
   the same topic-focused prompt to produce a single coherent final summary.

If the document fits within the token limit in one pass, skip the map step and call
GPT-4o-mini directly. Each final summary is a short paragraph per topic (omitting
topics not present in the document). Concatenate all topic paragraphs for a document into a single string before
embedding. One vector is produced per document and added to the summary FAISS
index. This is the level-1 hierarchical index.

### Step 4 — Persist
Save to `index_storage/`:
- Chunk FAISS index and BM25 corpus — pickled
- Chunk store (including all chunk metadata) — JSON (`chunk_store.json`); each entry now includes
  `line_start` and `line_end` in addition to the original fields
- Summary FAISS index and `summaries.json` — only written if `USE_SUMMARY_WEIGHTS = True`

A `build_index_flag` controls whether this phase runs or loads from disk.

---

## Query Pipeline

```
query
  → Stage 1: classify query type (QueryRouter)
  → Stage 2: summary-level soft weighting (optional — only if USE_SUMMARY_WEIGHTS)
  → Stage 2b: query expansion via synonym map (+ forced keys via bm25_expand_keys)
  → Stage 2c: metadata pre-filtering (optional)
  → Stage 3: fusion retrieval — BM25 + FAISS with weighted RRF (FusionRetriever)
             soft summary weights applied as a score multiplier (if USE_SUMMARY_WEIGHTS)
  → Stage 3b: co-occurrence re-ranking (optional — only if COOCCURRENCE_FILTER, or per-query require_primary or apply_boost)
              hard filter (require_primary): drop chunks with no primary-topic term
              proximity boost (apply_boost):  ×1.5 if primary + secondary term within COOCCURRENCE_WINDOW chars
              both flags are independent: boost-only mode ranks all results but lifts co-occurring passages
  → Stage 4: context enrichment ±5 (ContextEnricher)
  → Stage 5: score cutoff (MIN_SCORE) + deduplication
  → Stage 6: passage highlighting (TokenHighlighter)
  → output
```

### Stage 1 — Query classification / adaptive routing
Three mutually exclusive modes, checked in priority order:

| Priority | Condition | Behaviour |
|---|---|---|
| 1 | `MANUAL_QUERY_PARAMS` is set | Use the supplied `w_faiss`, `w_bm25`, `window` directly; no LLM call |
| 2 | `USE_CLASSIFIER = False` (default) | Look up `QUERY_TYPE` (default `"analytical"`) in `QUERY_TYPE_PARAMS`; no LLM call |
| 3 | `USE_CLASSIFIER = True` | GPT-4o-mini classifies the query and selects the matching row below |

Predefined type parameters:

| Type | Description | BM25 weight | FAISS weight | Window |
|---|---|---|---|---|
| `factual` | Who/what/when | 0.6 | 0.4 | ±5 |
| `opinion` | What do people think/feel | 0.3 | 0.7 | ±5 |
| `analytical` | Why/how patterns | 0.5 | 0.5 | ±5 |
| `contextual` | What was said around X | 0.4 | 0.6 | ±5 |

### Stage 2 — Hierarchical retrieval, level 1 (summaries) *(skipped if `USE_SUMMARY_WEIGHTS = False`)*
When enabled, embed the query and search the summary FAISS index. Because each
summary is structured around the four topics from Step 3 of the indexing pipeline
(government impact, armed group impact, NGO/authority impact, inter-actor
relationships), the resulting cosine similarity reflects topical relevance rather
than generic document similarity. This per-document `summary_score` is passed to
Stage 3 as a score multiplier — chunks from more topically relevant documents score
higher, but no documents are hard-excluded. When disabled, the summary index is not
searched and the multiplier is fixed at 1.0, so chunk-level retrieval signals alone
determine ranking.

### Stage 2b — Query expansion via domain synonym map
Before retrieval, expand the query by appending synonyms for matched key terms.
The synonym map is defined in `retrieval/vocabulary.py`. Synonyms are appended
to the BM25 query string only — FAISS always receives the raw query prose, preserving
semantic precision in the embedding space.

**Standard expansion** matches `SYNONYM_MAP` canonical keys that appear literally in
the query string. This is sufficient when the query prose contains the relevant
vocabulary (e.g. a query that includes the phrase "armed groups" triggers expansion
of that entire synonym list).

**Forced expansion via `bm25_expand_keys`** is required when the FAISS query string
is deliberately actor-focused and omits behavior vocabulary. In that case, canonical
keys for behavior categories (e.g. `"violence"`, `"restrict access"`) will not appear
in the query string and standard expansion will silently skip their synonym lists —
leaving BM25 without lexical coverage for terms like `"assassination"`, `"checkpoint"`,
or `"curfew"`. Setting `bm25_expand_keys` in the per-query retrieval opts forces those
synonym lists into BM25 expansion unconditionally:

```python
"armed_groups": (
    # Actor-focused prose — no behavior vocabulary, so FAISS embedding stays sharp
    "Armed groups...operate outside state authority, exercise territorial control...",
    {},
    {
        "bm25_expand_keys": [        # forced regardless of query string content
            "armed groups",          # actor synonyms (also in query, but explicit for clarity)
            "violence",              # assassination, killing, kidnapping, arson, ...
            "extort",                # vacuna, cobro, cuota, extorsión, ...
            "threats",               # amenaza, ultimátum, intimidación, ...
            "restrict access",       # checkpoint, roadblock, curfew, retén, ...
            "displacement",          # desplazamiento, fled, abandoned land, ...
        ],
    },
),
```

The same keys are passed to `_extract_tokens` so that terms forced into BM25 are
also highlighted in the output passages, keeping the `[**term**]` markup consistent
with what BM25 actually matched.

Example map (to be extended as needed):

| Canonical term | Synonyms |
|---|---|
| armed groups | paramilitaries, guerrillas, insurgents, dissidents, FARC, ELN, bacrim, illegal armed actors, grupos armados |
| deforestation | forest loss, tree clearing, land clearing, forest degradation, tala, deforestación |
| conservation | forest protection, protected areas, reserves, environmental protection, conservación |
| government | state, authorities, municipality, alcaldía, gobernación, institutions |
| NGO | organization, foundation, fundación, civil society, implementing partner |
| community | village, vereda, resguardo, residents, settlers, colonos, campesinos |
| illegal economy | drug trafficking, coca, narcotrafficking, mining, ganadería ilegal, illegal crops |

Expansion is additive — the original query terms are retained. Synonyms are
inserted as a space-separated string appended to the query before tokenization
for BM25 only. The raw query is passed to FAISS without expansion, preserving
semantic precision in the embedding space.

Synonym entries support a `[a|b|c]` shorthand for alternation. For BM25
expansion, each variant is generated as a separate plain string (e.g.
`[forest|jungle] loss` → `"forest loss"`, `"jungle loss"`). For highlighting,
the same pattern is compiled directly into the regex as `(?:forest|jungle) loss`,
so a single match covers all variants.

**Grammatical forms and query readability**

BM25 performs no stemming — `"killed"`, `"killing"`, and `"kill"` are three
distinct tokens. Without expansion, a query containing only `"kill"` misses
chunks where the corpus uses `"killed"`. Including surface form variants is
therefore correct and necessary for recall.

The alternation syntax keeps this compact: `"kill[s|ed|ing|ings|]"` expands to
`["kills","killed","killing","killings","kill"]`. The trailing `|` (empty
string) generates the base form. Do **not** use a trailing `?` instead — e.g.
`"kill[s|ed|ing|ings]?"` — because `_expand_synonym` treats `?` as a literal
character, producing broken tokens like `"killings?"` that match nothing in the
corpus.

The prose query string is unaffected by this syntax — `_expand_query` appends
synonyms internally and the original query text is never modified. The
alternation patterns appear only in SYNONYM_MAP and are never shown to the
researcher.

**Cost of synonym length — FAISS vs. BM25**
The two retrievers have asymmetric sensitivity to the number of synonyms.

*FAISS:* The query string is embedded as a single vector. Appending many synonyms
shifts the embedding centroid toward the average meaning of all terms. A short,
focused query produces a vector that points sharply at a semantic neighborhood;
a long synonym list produces a centroid that may land in a blurry region close to
nothing in particular. For this reason, synonym expansion is deliberately *not*
applied to the FAISS query — the raw query string is embedded as-is. Synonyms
intended to improve FAISS recall should instead be incorporated into the query
prose itself, written as a coherent sentence (HyDE-style) rather than appended as
a list.

*BM25:* Expansion has low cost — BM25 is robust to query length. The main risk is
precision loss from low-IDF terms (`"violence"`, `"illegal"`, `"conflict"`) that
appear across many documents and drag scores toward generic political language
rather than specific testimony. Keep SYNONYM_MAP entries specific; avoid adding
high-frequency generic terms.

### Stage 2c — Metadata pre-filtering (optional)
Before fusion retrieval, optionally restrict the candidate chunk pool to chunks
matching specified metadata values. Filters are passed as a dict in `query_d`
alongside the query string and are applied by checking chunk metadata in the
chunk store. All filter fields are optional; omitting a field imposes no
restriction on that dimension.

Filterable fields:

| Field | Example values |
|---|---|
| `source_file` | `"Diana.Navarro.Mayor.Pto.Rico_trad_ok.txt"` |
| `interview_type` | `"KII"`, `"FGD"` |
| `interviewee_type` | `"NGO"`, `"farmer"`, `"government official"` |
| `location` | `"Caqueta"`, `"Putumayo"` |
| `date` | `"2024-03"`, or a range `("2023-01", "2024-06")` |

`source_file` restricts retrieval to chunks from a single transcript file. It is
checked first, before all other fields. When the per-document loop in
`run_retrieval.py` is active (see below), this filter is injected automatically
for each document; it does not need to be set manually in `query_d`.

The filtered chunk subset is passed as the candidate pool to both BM25 and FAISS
in Stage 3. BM25 is re-scoped to the filtered subset; FAISS uses an ID selector
to restrict search to matching chunk indices.

### Stage 3 — Fusion retrieval (BM25 + FAISS)
`FusionRetriever.retrieve()` first resolves all query-level values that are
invariant across documents, then runs the per-document retrieval steps.

**Query cache** (`FusionRetriever._query_cache`)
On the first call for a given query string, four values are computed and stored
in a dict keyed by the query string:

| Cached value | How produced |
|---|---|
| `query_vec` | `embed([query])` — one API call |
| `expanded` | `_expand_query(query)` — synonym-expanded string for BM25 |
| `expanded_tokens` | `_extract_tokens(query)` — canonical + synonym tokens for highlighting |
| `summary_scores` | `_summary_scores(query_vec)` — per-document cosine scores (only if `USE_SUMMARY_WEIGHTS`) |

On every subsequent call with the same query string (e.g., each iteration of
the per-document loop in `run_retrieval.py`), all four values are read from the cache
with no API calls. For a 76-document corpus with one query string this reduces
`embed()` calls from 76 to 1.

The per-call work — `_apply_filters`, `retrieve_faiss`, `retrieve_bm25`,
`fuse_rrf` — still runs fresh on each call, as it depends on the per-document
candidate pool.

The cache is not persisted between runs; it lives for the lifetime of the
`FusionRetriever` instance.

Run both retrievers against the filtered candidate pool:

- **FAISS**: top-N chunks by cosine similarity to the query embedding.
- **BM25**: top-N chunks by BM25Okapi score.

Merge with **weighted Reciprocal Rank Fusion**:
```
rrf_score(chunk) = w_faiss * (1 / (rank_faiss + 60)) + w_bm25 * (1 / (rank_bm25 + 60))
```
Weights `w_faiss` and `w_bm25` are set by the query type from Stage 1. Chunks
appearing in only one list use that list's term only. If `USE_SUMMARY_WEIGHTS`
is `True`, the final `rrf_score` is multiplied by the document's `summary_score`
from Stage 2; otherwise the multiplier is 1.0.

### Stage 3b — Co-occurrence re-ranking *(skipped if `COOCCURRENCE_FILTER = False` and no per-query `require_primary` or `apply_boost`)*

Applied after `fuse_rrf` and before context enrichment. Operates on the chunk text
only (not the enriched passage). Two independent flags control its behavior:
`require_primary` (hard filter) and `apply_boost` (proximity boost). Either or both
may be active; they are set globally via `COOCCURRENCE_FILTER` and per-query via the
optional third element of each `query_d` entry.

#### What it does

Each fused chunk is scored by `_cooccurrence_score(chunk, window)`, which searches the
raw chunk text for terms drawn from two tiers of `SYNONYM_MAP`:

| Tier | Config key | Role |
|---|---|---|
| Primary | `COOCCURRENCE_PRIMARY_KEY` (default: `"armed groups"`) | Anchor — presence required when `require_primary=True`; proximity target for boost |
| Secondary | `COOCCURRENCE_SECONDARY_KEYS` (default: violence, extortion, threats, …) | Additive boost signal — secondary term within `COOCCURRENCE_WINDOW` chars of a primary term |

The raw multiplier from `_cooccurrence_score` is:

| Primary present | Secondary within window | Raw multiplier |
|---|---|---|
| No | — | `0.0` |
| Yes | No | `1.0` |
| Yes | Yes | `1.5` |

How that multiplier is applied depends on which flags are active:

| `require_primary` | `apply_boost` | Chunk with no primary | Chunk with primary only | Chunk with primary + secondary |
|---|---|---|---|---|
| `False` | `False` | Stage 3b skipped entirely | Stage 3b skipped entirely | Stage 3b skipped entirely |
| `False` | `True` | Score × 1.0 (retained unchanged) | Score × 1.0 (retained unchanged) | Score × 1.5 (boosted) |
| `True` | *(any)* | Dropped (hard filter) | Score × 1.0 (retained) | Score × 1.5 (boosted) |

When `apply_boost=True` without `require_primary`, the `0.0` multiplier for
primary-absent chunks is floored to `1.0` (`max(cooc, 1.0)`), so those passages pass
through at their original RRF score rather than being zeroed out.

Chunks dropped by the hard filter are removed before context enrichment and before
the `MIN_SCORE` cutoff, so they will not appear in output under any score threshold.

#### Enabling per query

`COOCCURRENCE_FILTER` sets the global default for `require_primary` only; `apply_boost`
defaults to `False` and must always be set explicitly. To configure either flag for a
specific query, add a third element to the `query_d` tuple:

```python
query_d = {
    # Filter + boost: only return passages that name an armed group; rank co-occurring passages higher
    "armed_groups_strict": (
        "Armed groups... carry out violence...",
        {},
        {"require_primary": True, "apply_boost": True, "cooccurrence_window": 150},
    ),
    # Boost only: retain all passages but lift those with tight actor+behavior co-occurrence
    "armed_groups_broad": (
        "Armed groups... carry out violence...",
        {},
        {"apply_boost": True, "cooccurrence_window": 150},
    ),
    # No co-occurrence logic
    "deforestation": (
        "Deforestation and forest clearing...",
        {},
        # no third element → uses COOCCURRENCE_FILTER global default (False)
    ),
}
```

#### Qualitative methodological implications

**What the hard filter achieves.** Standard RRF fusion retrieves the highest-scoring
chunks on a combined semantic and lexical basis. For a query like the `armed_groups`
prompt, this means chunks about community fear, state checkpoints, or environmental
insecurity can rank highly because they match behavioral vocabulary without any armed
group actor being named. The hard filter (`require_primary=True`) enforces an
actor-presence requirement: every returned passage must contain at least one term from
the primary-topic synonym list. From a qualitative standpoint, this raises the
evidentiary standard for inclusion — a passage that describes an outcome without
attributing it to an armed group is not returned, regardless of its relevance score.

**What the proximity boost achieves.** A chunk may mention an armed group in one
sentence and describe a killing in a sentence three paragraphs later. Co-occurrence
within a narrow character window (150 chars ≈ 1–2 sentences) distinguishes passages
where the connection is directly stated — "the guerrillas killed two community members"
— from passages where the two concepts appear incidentally in the same chunk. Passages
with tight co-occurrence are ranked higher, surfacing the most directly attributive
testimony ahead of passages with looser contextual association.

**Choosing a mode.** The two flags support distinct analytical goals:

| Goal | Recommended mode |
|---|---|
| *Quote extraction* — assembling attributive testimony for a paper | `require_primary=True, apply_boost=True` — only passages that name the actor; co-occurring passages ranked first |
| *Thematic scoping* — identifying which interviews discuss the topic, including oblique references | `apply_boost=True` only — all passages retained; co-occurring passages floated to the top without discarding indirect evidence |
| *Exploratory retrieval* — no assumptions about how the topic is expressed | Both flags `False` — standard RRF ranking |

**Output indicator.** When either flag is active, the output file header includes a
`Co-occurrence` line identifying the mode (`filter+boost` or `boost only`), primary
key, window size, and boost factor, so the retrieval configuration is legible from the
output alone without opening the script.

---

### Stage 4 — Context enrichment ±5
For each chunk in the final ranked list, retrieve neighbors at `chunk_index ± 5` from the same document (assembled
from the chunk store at query time). Before joining, trim the leading overlap from each successive chunk — the
longest suffix of chunk N that equals a prefix of chunk N+1, up to `CHUNK_OVERLAP + 50` characters — to prevent the
shared text from appearing twice at each boundary. Concatenate the de-overlapped chunks into a single enriched passage.

The **line range of the enriched passage** is derived from the window endpoints: `line_start` of the
first chunk in the window and `line_end` of the last chunk in the window. These are taken directly
from the chunk store and added to the result record alongside the passage text.

### Stage 5 — Deduplication and score cutoff
Iterate over the ranked list in score order. Stop immediately when a passage's
score falls below `MIN_SCORE` (default `0.001`) — all subsequent passages will also
be below the threshold. For each passage above the threshold, suppress it if its
anchor chunk index already falls within the covered window of a higher-ranked
passage from the same document. Track covered indices in a per-document set as
passages are accepted.

This prevents the same stretch of transcript from appearing multiple times in the
output under different anchor chunks, and excludes low-signal results entirely.

### Stage 6 — Passage highlighting (TokenHighlighter)
After the final result list is assembled, annotate each enriched passage with two
layers of markup before writing to disk:

**BM25 term highlighting — `[**term**]`**
Wraps verbatim occurrences (case-insensitive, whole-word) of:
- The canonical keys from `SYNONYM_MAP` that matched the query (e.g. `armed groups`)
- Their full synonym lists (e.g. `paramilitaries`, `illegal armed actors`, ...)

Only domain terms from `SYNONYM_MAP` are highlighted — raw query words are not
used, avoiding stopword noise. Entries using `[a|b|c]` alternation syntax are
compiled to `(?:a|b|c)` regex groups. Multi-word phrases and alternation groups
are sorted longest-first before compilation, preventing double-marking of
sub-phrases.

**FAISS sentence highlighting — `[[sentence]]`**
Each passage is split on sentence-ending punctuation. All sentences across all
result passages are embedded in a **single batched `embed()` call**. Per
passage, all sentences scoring above a cosine similarity threshold of 0.25 to the
query embedding are wrapped in `[[...]]` with no cap. If no sentence clears the
threshold, only the single top-scoring sentence is wrapped in `[~...~]` to signal
that it is a fallback selection. This identifies semantic hot-spots within the
broader enriched context window.

**Markup nesting:** FAISS sentence markup is applied first; BM25 token markup
runs inside each sentence afterward. A term appearing in a semantically
relevant sentence renders as `[[text [**term**] text]]`.

---

## Output

### Per-document mode (current default)
`run_retrieval.py` iterates over every document in `builder.doc_chunks` and runs
`retriever.retrieve()` once per document, injecting `{"source_file": fname}`
into the base filters from `query_d` (defined in `retrieval/queries.py`). Each
document produces a separate output file named:

```
queries/{query_key}__{stem}.txt
```

where `{stem}` is the document filename without its `.txt` extension (e.g.,
`armed_groups_conservation__Diana.Navarro.Mayor.Pto.Rico_trad_ok.txt`). The
double underscore separates the query key from the document stem and avoids
collision with the single underscores used in metadata-structured filenames.

`TOP_K` and `MIN_SCORE` apply within each document's chunk pool rather than
across the whole corpus, so results are the top-scoring passages from that
document only. Documents with fewer chunks than `TOP_K` will naturally return
fewer results.

**API calls:** `FusionRetriever` caches the query embedding, synonym expansion,
and summary scores on the first call for a given query string (see Stage 3).
Subsequent per-document calls reuse the cache, so the full corpus loop in
`run_retrieval.py` issues only one `embed()` call regardless of document count.

### Single-file mode (previous default)
To revert to a single output file per query across the full corpus, replace
the per-document loop with the original call:

```python
results = retriever.retrieve(query_str, filters=filters)
write_results(query_key, query_str, results)
```

### Output fields (both modes)
Each result entry in a `.txt` file contains:
- Source filename
- Interview type (KII / FGD)
- Interviewee type (community leader, government official, NGO staff, farmer, etc.)
- Location
- Date
- Query type
- Final fusion score
- **Lines** — 1-based start and end line numbers of the enriched passage within the source file
  (e.g. `Lines: 42–118`), enabling direct navigation to the passage in the original transcript
- Enriched passage text (with `[**term**]` and `[[sentence]]` markup)
- Separator line

---

## Dependencies
```
openai
faiss-cpu
rank_bm25
numpy
langchain-text-splitters
python-dotenv
tiktoken
```

---

## Paths
| Purpose | Path | Config location |
|---|---|---|
| Entry point | `qual/run_retrieval.py` | — |
| Query definitions | `retrieval/queries.py` | — |
| All constants | `retrieval/config.py` | — |
| Input corpus | `txt/test/` | `CORPUS_DIR` in `config.py` |
| Index storage | `index_storage/` | `INDEX_DIR` in `config.py` |
| Query output | `queries/{query_key}.txt` | `OUTPUT_DIR` in `config.py` |
