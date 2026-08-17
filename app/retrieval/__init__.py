"""Retrieval layer for the GoodFoods agent.

Hybrid retrieval: exact SQL predicates for hard constraints (zone, price,
rating) + sentence-embedding cosine rank for soft semantics (cuisine, tags,
amenities, free-text vibe). See SEMANTIC_SEARCH_PLAN.md.
"""

from app.retrieval.embeddings import (
    build_restaurant_index,
    semantic_rank,
    restaurant_doc,
    _get_model,
)

__all__ = [
    "build_restaurant_index",
    "semantic_rank",
    "restaurant_doc",
    "_get_model",
]
