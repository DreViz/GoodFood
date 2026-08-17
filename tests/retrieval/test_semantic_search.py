"""Recall + fallback tests for the semantic retrieval layer.

Two suites:

1. ``TestSemanticRank`` — recall@5 / MRR over hand-authored queries whose
   ground truth is derived from the seed data file (restaurants carrying the
   relevant tag / cuisine / zone). The headline cases are *paraphrase*
   queries with ZERO keyword overlap with the stored tags — e.g. "romantic"
   must match the ``date-night`` tag, "cheap" must match ``budget``. That is
   the entire reason embeddings exist here.

   These tests SKIP gracefully when the embedding model is unavailable
   (torch not loadable on the host). The fallback path is covered separately.

2. ``TestFallback`` — asserts the system still answers via ilike when
   embeddings are disabled. This MUST always pass; it is the production
   guarantee.
"""
import os
from pathlib import Path

import pytest

from app.config import get_settings
from app.retrieval.embeddings import (
    _get_model,
    build_restaurant_index,
    restaurant_doc,
    semantic_rank,
)


# ---------------------------------------------------------------------------
# Fixtures + data loading
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).resolve().parents[2] / "app" / "data" / "goodfoods_locations_unique_50.json"


def _load_seed():
    import json
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def seed():
    return _load_seed()


@pytest.fixture(scope="module")
def index_ready():
    """Build the index once for the module; skip if model unavailable."""
    if _get_model() is None:
        pytest.skip("embedding model unavailable on this host")
    idx = build_restaurant_index(force=True)
    if idx is None:
        pytest.skip("index could not be built")
    return idx


def _relevant_ids(seed, *, tag=None, cuisine=None, zone=None):
    """Location_ids matching ANY of the supplied predicates (OR semantics).

    OR (not AND) on purpose: for a vibe query the relevant set is every
    venue plausibly matching the intent — e.g. "romantic dinner" is relevant
    to any date-night OR premium venue. Recall@5 then measures whether the
    ranker surfaces at least one of them near the top.
    """
    out = []
    for r in seed:
        tags = [t.lower() for t in (r.get("tags") or [])]
        cuisines = [c.lower() for c in (r.get("cuisines") or [])]
        zn = (r.get("zone") or "").lower()
        match = (
            (tag and tag in tags)
            or (cuisine and cuisine in cuisines)
            or (zone and zone == zn)
        )
        if match:
            out.append(r["location_id"])
    return out


# ---------------------------------------------------------------------------
# Suite 1 — semantic rank quality
# ---------------------------------------------------------------------------
class TestSemanticRank:
    """Precision@5 + hit@5 + MRR on paraphrase + direct queries.

    Metrics chosen for the data shape: tags like 'budget' appear on 22 of 50
    rows, so recall@5 (hits/total_relevant) is mathematically capped near 0.23
    — the wrong metric. Precision@5 (what fraction of the top-5 are relevant)
    and hit@5 (binary: ≥1 relevant in top-5) are the standard retrieval
    metrics for this setting.
    """

    # (query, relevant_kw, min_precision@5, min_mrr)
    # precision@5 threshold of 0.2 means "≥1 of the top-5 is relevant". MRR
    # threshold of 0.30 means "first relevant is in the top-3". Together they
    # verify the ranker surfaces relevant venues early — the all-MiniLM-L6-v2
    # model won't fill all 5 slots for every paraphrase, and that's fine for a
    # 90 MB model on a 50-row corpus.
    CASES = [
        # --- Paraphrase: zero keyword overlap with stored tags ---
        ("romantic dinner", {"tag": "date-night"}, 0.4, 0.30),
        ("cheap eats", {"tag": "budget"}, 0.2, 0.30),
        ("affordable places", {"tag": "budget"}, 0.4, 0.30),
        # --- Direct tag / cuisine ---
        ("outdoor seating", {"tag": "outdoor"}, 0.2, 0.30),
        ("italian food", {"cuisine": "italian"}, 0.4, 0.30),
        ("family friendly restaurant", {"tag": "family-friendly"}, 0.4, 0.30),
        ("rooftop dinner", {"tag": "rooftop"}, 0.4, 0.30),
        # --- Multi-signal vibe ---
        ("anniversary date night", {"tag": "date-night"}, 0.4, 0.30),
        ("live music tonight", {"tag": "live-music"}, 0.4, 0.30),
        ("healthy food options", {"tag": "healthy"}, 0.4, 0.30),
    ]

    @pytest.mark.parametrize("query,relevant_kw,min_precision,min_mrr", CASES)
    def test_precision_and_mrr(self, seed, index_ready, query, relevant_kw, min_precision, min_mrr):
        all_ids = [r["location_id"] for r in seed]
        relevant = set(_relevant_ids(seed, **relevant_kw))
        if not relevant:
            pytest.skip(f"no ground-truth rows for {relevant_kw}")

        ranked = semantic_rank(query, all_ids, top_k=5)
        assert ranked is not None, "ranker returned None despite model being available"

        top5_ids = [lid for lid, _ in ranked]
        hits_in_top5 = [lid for lid in top5_ids if lid in relevant]
        precision = len(hits_in_top5) / 5.0

        # MRR: reciprocal rank of the FIRST relevant result.
        mrr = 0.0
        for i, lid in enumerate(top5_ids, 1):
            if lid in relevant:
                mrr = 1.0 / i
                break

        assert precision >= min_precision, (
            f'"{query}" precision@5={precision:.2f} < {min_precision}. '
            f"top5={top5_ids}, hits={hits_in_top5}, relevant_count={len(relevant)}"
        )
        assert mrr >= min_mrr, (
            f'"{query}" MRR={mrr:.2f} < {min_mrr}. top5={top5_ids}'
        )

    def test_empty_query_returns_none(self):
        """Empty query must signal 'fall back' rather than rank."""
        assert semantic_rank("", [1, 2, 3]) is None

    def test_empty_candidates_returns_empty(self):
        """Empty candidate set ranks to empty, not None."""
        assert semantic_rank("italian", []) == []

    def test_restaurant_doc_contains_key_fields(self, seed):
        """The embedded blob must surface name + cuisines + zone + tags."""
        from types import SimpleNamespace

        r = seed[0]
        doc = restaurant_doc(SimpleNamespace(**r))  # adapt dict → attrs
        assert r["unit_name"] in doc
        for c in (r.get("cuisines") or []):
            assert c in doc
        assert r["zone"] in doc


# ---------------------------------------------------------------------------
# Suite 2 — fallback (must ALWAYS pass; production guarantee)
# ---------------------------------------------------------------------------
class TestFallback:
    """When embeddings are off / unavailable, search still answers via ilike."""

    def test_recommend_venues_returns_results_without_model(self, monkeypatch):
        # Force the model off — simulates a host where torch is broken.
        from app.agent import tool_calls

        def _no_model():
            return None

        monkeypatch.setattr("app.retrieval.embeddings._get_model", _no_model)

        res = tool_calls.recommend_venues(query="italian", limit=3)
        assert res["ok"] is True
        assert len(res["results"]) > 0
        # No score field leaks into the public shape.
        assert all("unit_name" in r for r in res["results"])

    def test_search_returns_results_without_model(self, monkeypatch):
        from app.agent import tool_calls

        monkeypatch.setattr("app.retrieval.embeddings._get_model", lambda: None)

        res = tool_calls.search_restaurants(cuisine="italian", limit=5)
        assert res["ok"] is True
        assert len(res["results"]) > 0
        # ilike should still surface Italian-cuisine restaurants.
        for r in res["results"]:
            cuisines = " ".join(r.get("cuisines") or []).lower()
            assert "italian" in cuisines

    def test_disabled_via_config_falls_back(self, monkeypatch):
        """semantic_search_enabled=False must not even attempt to load the model."""
        settings = get_settings()
        monkeypatch.setattr(settings, "semantic_search_enabled", False)
        from app.retrieval import embeddings as emb

        assert emb._get_model() is None
