"""
Query definitions: query_d maps query_key → (query_string, filters, [retrieval_opts]).

Each entry is a 2- or 3-element tuple:
  [0] query_string    — prose passed to FAISS as-is. For armed-group queries, keep this
                        actor-focused (who they are, not what they do) so the embedding
                        points at actor-centered passages. Behavior coverage is handled by
                        bm25_expand_keys and the co-occurrence boost.
  [1] filters         — dict restricting the candidate chunk pool (all fields optional):
                          "source_file"      : str         — single transcript file
                          "interview_type"   : str         — e.g. "KII", "FGD"
                          "interviewee_type" : str         — e.g. "NGO", "community"
                          "location"         : str         — e.g. "caqueta"
                          "date"             : str | tuple — exact "2024-03" or range ("2023-01","2024-06")
  [2] retrieval_opts  — optional dict; all keys optional, defaults shown:
                          "require_primary"     : bool      — drop chunks with no primary-topic term (default: COOCCURRENCE_FILTER)
                          "apply_cooccur_boost"         : bool      — ×1.5 score for primary+secondary proximity (default: False)
                          "cooccurrence_window" : int       — character radius for proximity check (default: COOCCURRENCE_WINDOW)
                          "bm25_expand_keys"    : list[str] — SYNONYM_MAP keys to force into BM25 expansion
                                                              regardless of query string content (default: None).
                                                              Required when query string is actor-focused and
                                                              omits behavior vocabulary BM25 should still match.
"""

from __future__ import annotations

# armed_groups note: "exploit resources", "form alliances", "govern" removed from
# active query to avoid overlap with other topics.
query_d = {
    "armed_groups": (
        "Armed groups, illegal groups, groups outside the law, insurgents, paramilitaries, \
        and criminal organizations carry out \
        intimidation, extortion, violence, kidnappings, assassinations, and disappearances, \
        disrupt public order, cause insecurity, operate outside state authority, and defy the law.",
        {},
        {
            "apply_cooccur_boost":  True,
            "require_primary": False,
            "cooccurrence_window":  300,
            "bm25_expand_keys": [
                "armed groups",
                "violence",
                "extort",
                "threats",
                "restrict access",
            ],
        },
    ),

    # armed_groups: Removed "exploit resources,";"form alliances";"govern" to avoid catching too many overlaps with other topics
    #
    # "armed_groups": (
    #     "Armed groups, illegal groups, groups outside the law, insurgents, paramilitaries, and criminal organizations\
    #     carry out violence, assassinations, kindappings, disappearances, arson, looting, and assaults.\
    #     These groups cause insecurity and fear, restrict access to certain areas, extort, demand vacunas,\
    #     issue threats and ultimatums, impose checkpoints and curfews, impose rules, and defy norms and laws.",
    #     {},
    # ),
    #
    # --- Commented-out examples showing available options ---
    #
    # Location filter (retrieval_opts carry over unchanged):
    # "armed_groups_caqueta": (
    #     "Armed groups... operate outside state authority...",
    #     {"location": "caqueta"},
    #     {"apply_cooccur_boost": True, "cooccurrence_window": 300,
    #      "bm25_expand_keys": ["armed groups","violence","extort","threats","restrict access","displacement"]},
    # ),
    #
    # Interviewee-type filter:
    # "armed_groups_community": (
    #     "Armed groups... operate outside state authority...",
    #     {"interviewee_type": "community"},
    #     {"apply_cooccur_boost": True, "cooccurrence_window": 300,
    #      "bm25_expand_keys": ["armed groups","violence","extort","threats","restrict access","displacement"]},
    # ),
    #
    # Date range filter:
    # "armed_groups_recent": (
    #     "Armed groups... operate outside state authority...",
    #     {"date": ("2023-01", "2024-12")},
    #     {"apply_cooccur_boost": True, "cooccurrence_window": 300,
    #      "bm25_expand_keys": ["armed groups","violence","extort","threats","restrict access","displacement"]},
    # ),
    #
    # Filter + boost — require actor term present; also boost tight co-occurrences:
    # "armed_groups_strict": (
    #     "Armed groups... operate outside state authority...",
    #     {},
    #     {"require_primary": True, "apply_cooccur_boost": True, "cooccurrence_window": 150,
    #      "bm25_expand_keys": ["armed groups","violence","extort","threats","restrict access","displacement"]},
    # ),
}
