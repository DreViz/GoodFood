"""Live demo of semantic restaurant retrieval.

Usage:
    python -m scripts.demo_semantic_search "cozy italian for two"
    python -m scripts.demo_semantic_search "romantic anniversary dinner"
    python -m scripts.demo_semantic_search "cheap family lunch south"

Prints the top-5 ranked restaurants with cosine scores. Demonstrates the
hybrid retrieval pipeline (SQL hard filters + embedding cosine rank) for
interviews / on-screen demos. Falls back to ilike order if the embedding
model is unavailable, with a clear notice.
"""
import sys

from app.agent.tool_calls import recommend_venues, search_restaurants
from app.retrieval.embeddings import _get_model

# Windows consoles default to cp1252 and choke on the ★ glyph or accented
# restaurant names (Café). Force UTF-8 so the demo prints cleanly everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


def _print_results(title: str, res: dict) -> None:
    print(f"\n=== {title} ===")
    if not res.get("ok"):
        print(f"  error: {res.get('error')}")
        return
    rows = res.get("results", [])
    if not rows:
        print("  (no results)")
        return
    for i, r in enumerate(rows, 1):
        tags = ", ".join(r.get("tags") or []) or "-"
        cuisines = ", ".join(r.get("cuisines") or []) or "-"
        print(
            f"  {i}. {r['unit_name']}  [{r['zone']}]  "
            f"${r.get('avg_price_per_person')}  ★{r.get('rating')}\n"
            f"     cuisines: {cuisines}  |  tags: {tags}"
        )


def main() -> int:
    vibe = " ".join(sys.argv[1:]).strip()
    if not vibe:
        vibe = "cozy italian for two"

    model = _get_model()
    if model is None:
        print(
            "[demo] Embedding model unavailable (torch/sentence-transformers not "
            "loadable). Showing ilike fallback results instead.\n"
        )

    print(f'Query: "{vibe}"')

    # Path 1: pure recommendation (no hard filters) — ranks ALL 50 by vibe.
    _print_results("recommend_venues (all 50, ranked by vibe)", recommend_venues(query=vibe, limit=5))

    # Path 2: hybrid — a soft cuisine filter narrows the candidate set, then
    # the vibe ranks within it. Demonstrates SQL pre-filter + semantic rank.
    _print_results(
        'search_restaurants_by_filters (cuisine=italian + vibe)',
        search_restaurants(cuisine="italian", query=vibe, limit=5),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
