"""Sentence-embedding index + cosine ranker for restaurant retrieval.

Hybrid retrieval pipeline (see SEMANTIC_SEARCH_PLAN.md):
  - hard filters (zone, price, rating) stay as SQL predicates (exact)
  - soft semantics (cuisine, tags, amenities, free-text vibe) ranked via
    cosine similarity over all-MiniLM-L6-v2 embeddings (384-dim)

Design:
  - Lazy model load via ``_get_model()`` singleton. Returns ``None`` on any
    failure (dep missing, model download failed, disabled by config). Callers
    treat ``None`` as "fall back to ilike keyword search" — the system must
    never hard-fail because embeddings are unavailable.
  - Index is a ``(matrix, location_ids)`` pair cached as a module-level
    singleton; persisted to ``app/data/restaurant_embeddings.npy`` for fast
    restart. Rebuild is idempotent and triggered lazily on first query.
  - One document per restaurant: name + cuisines + zone + tags + amenities +
    price + rating flattened to a natural-language blob. Menu and opening
    hours are deliberately skipped (menu adds noise; hours aren't semantic).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence, Tuple

from app.config import get_settings
from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant

logger = logging.getLogger(__name__)

# Lazy, thread-safe module-level singletons.
_model = None
_model_lock = threading.Lock()

# (embeddings_matrix, location_ids) — built once, reused.
_index: Optional[Tuple["object", List[int]]] = None  # matrix is np.ndarray
_index_lock = threading.Lock()


def _get_model():
    """Return a cached SentenceTransformer model, or ``None`` if unavailable.

    ``None`` is the sentinel for "ranking unavailable — fall back to ilike".
    Failures covered: config kill-switch, dep not installed, model load error.
    Never raises.
    """
    global _model
    settings = get_settings()
    if not settings.semantic_search_enabled:
        return None

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover - dep optional
            logger.warning(
                "[semantic] sentence-transformers not available (%s); "
                "falling back to ilike search.", e
            )
            return None

        try:
            _model = SentenceTransformer(settings.semantic_model)
            logger.info("[semantic] loaded model '%s'", settings.semantic_model)
        except Exception as e:  # pragma: no cover - network/model errors
            logger.warning(
                "[semantic] failed to load model '%s' (%s); falling back to ilike.",
                settings.semantic_model, e,
            )
            _model = None
        return _model


def _join_list(value) -> str:
    """Flatten a JSONB list (or scalar) to a comma-joined string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def restaurant_doc(r: Restaurant) -> str:
    """Build the natural-language blob embedded per restaurant.

    Pure function over row attributes — unit-testable without a DB by passing
    any object exposing the same attributes. Deterministic + rebuildable.
    """
    parts: List[str] = []
    if r.unit_name:
        parts.append(r.unit_name + ".")
    cuisines = _join_list(getattr(r, "cuisines", None))
    if cuisines:
        parts.append(f"Cuisines: {cuisines}.")
    if r.zone:
        parts.append(f"Zone: {r.zone}.")
    tags = _join_list(getattr(r, "tags", None))
    if tags:
        parts.append(f"Tags: {tags}.")
    amenities = _join_list(getattr(r, "amenities", None))
    if amenities:
        parts.append(f"Amenities: {amenities}.")
    if getattr(r, "avg_price_per_person", None) is not None:
        parts.append(f"Price: ${r.avg_price_per_person} avg.")
    if getattr(r, "rating", None) is not None:
        parts.append(f"Rating: {r.rating}.")
    return " ".join(parts)


def build_restaurant_index(force: bool = False):
    """Build (or return cached) the embeddings matrix + location_id list.

    Returns ``(matrix, location_ids)`` or ``None`` if unavailable (model
    missing, no restaurants, disabled). Idempotent: returns the cached index
    on repeated calls unless ``force=True``.

    Persistence: the matrix + ids are pickled to
    ``settings.semantic_index_path`` so cold starts skip the embedding pass.
    A staleness guard re-embeds if the DB row count diverges from the cached
    id list (rows added/removed).
    """
    global _index
    settings = get_settings()

    model = _get_model()
    if model is None:
        return None

    if _index is not None and not force:
        return _index

    with _index_lock:
        if _index is not None and not force:
            return _index

        session = SessionLocal()
        try:
            rows = session.query(Restaurant).order_by(Restaurant.location_id).all()
        finally:
            session.close()

        if not rows:
            logger.warning("[semantic] no restaurants to index.")
            return None

        location_ids = [r.location_id for r in rows]

        # Restore a cache that matches the current row set.
        if not force:
            cached = _load_cache(settings.semantic_index_path)
            if cached is not None:
                cached_ids, cached_matrix = cached
                if cached_ids == location_ids:
                    logger.info("[semantic] restored cached index (%d rows).", len(location_ids))
                    _index = (cached_matrix, cached_ids)
                    return _index
                logger.info("[semantic] cache stale (row set changed); rebuilding.")

        docs = [restaurant_doc(r) for r in rows]
        try:
            import numpy as np  # type: ignore
            matrix = model.encode(
                docs,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            matrix = np.asarray(matrix, dtype="float32")
        except Exception as e:
            logger.warning("[semantic] embedding failed (%s); index not built.", e)
            return None

        try:
            _save_cache(settings.semantic_index_path, location_ids, matrix)
        except Exception as e:  # pragma: no cover - best-effort persist
            logger.warning("[semantic] failed to persist index cache (%s).", e)

        _index = (matrix, location_ids)
        logger.info("[semantic] built index over %d restaurants.", len(location_ids))
        return _index


def _load_cache(path: str):
    """Return ``(location_ids, matrix)`` from disk, or ``None``."""
    if not path or not os.path.exists(path):
        return None
    try:
        import numpy as np  # type: ignore
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True)
        payload = data.item() if data.dtype == object else data
        if not isinstance(payload, dict):
            return None
        ids = payload.get("location_ids")
        matrix = payload.get("embeddings")
        if ids is None or matrix is None:
            return None
        return (list(ids), np.asarray(matrix, dtype="float32"))
    except Exception as e:
        logger.warning("[semantic] cache load failed (%s); will rebuild.", e)
        return None


def _save_cache(path: str, location_ids: Sequence[int], matrix) -> None:
    """Persist ``(location_ids, embeddings)`` to ``path`` as a pickled .npy."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import numpy as np  # type: ignore
    payload = {"location_ids": list(location_ids), "embeddings": np.asarray(matrix)}
    np.save(path, payload, allow_pickle=True)


def semantic_rank(
    query_text: str,
    candidate_location_ids: Sequence[int],
    top_k: int = 5,
) -> Optional[List[Tuple[int, float]]]:
    """Cosine-rank candidate restaurants against ``query_text``.

    Args:
        query_text: free-form user intent (e.g. "cozy anniversary dinner").
            If empty, returns ``None`` so the caller keeps SQL-order.
        candidate_location_ids: the SQL-filtered candidate set (hard filters
            already applied). Ranked within this set.
        top_k: number of results to return.

    Returns:
        ``[(location_id, score), ...]`` sorted desc by cosine similarity,
        length <= top_k. Returns ``None`` when ranking is unavailable (model
        missing / index not built / empty query) so callers fall back to
        ilike order. Returns ``[]`` only when the candidate set is empty.
    """
    if not query_text or not query_text.strip():
        return None
    if not candidate_location_ids:
        return []

    index = build_restaurant_index()
    if index is None:
        return None
    matrix, all_ids = index

    model = _get_model()
    if model is None:
        return None

    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    # Map candidate location_ids -> matrix row indices. Unknown ids are
    # dropped (they were deleted after the index was built).
    id_to_row = {lid: i for i, lid in enumerate(all_ids)}
    rows = []
    cand_ids = []
    for lid in candidate_location_ids:
        i = id_to_row.get(lid)
        if i is not None:
            rows.append(i)
            cand_ids.append(lid)
    if not rows:
        return []

    # Embed the query (normalized) and cosine-rank against the candidate
    # slice. Vectors are L2-normalized at build time, so dot product == cosine.
    try:
        q = model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )
        q = np.asarray(q, dtype="float32")[0]            # (384,)
        cand_matrix = matrix[rows]                        # (N, 384)
        scores = cand_matrix @ q                          # (N,)
    except Exception as e:
        logger.warning("[semantic] ranking failed (%s); falling back.", e)
        return None

    order = np.argsort(-scores)[:top_k]
    ranked = [(cand_ids[i], float(scores[i])) for i in order]
    return ranked
