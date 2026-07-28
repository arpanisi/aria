"""Unit tests for the pure (non-SQLite) logic in arxiv_index.py."""
from __future__ import annotations

import math

import pytest

from scripts.retrieval.arxiv_index import (
    arxiv_categories,
    category_matches,
    literature_slate_diversity,
    select_diverse_literature_slate,
)


def test_arxiv_categories_splits_on_whitespace() -> None:
    assert arxiv_categories("cs.LG stat.ML  stat.AP") == ["cs.LG", "stat.ML", "stat.AP"]


def test_arxiv_categories_empty_string() -> None:
    assert arxiv_categories("") == []


def test_category_matches_exact_and_subcategory() -> None:
    prefixes = ["stat.ME"]
    assert category_matches("stat.ME", prefixes) is True
    # category_matches intentionally supports c == prefix or c.startswith(f"{prefix}."),
    # so a deeper sub-level under the same prefix still matches.
    assert category_matches("stat.ME.sub", prefixes) is True
    assert category_matches("cs.LG", prefixes) is False


def test_category_matches_prefix_dot_form() -> None:
    # category_matches checks c == prefix or c.startswith(f"{prefix}.")
    assert category_matches("stat.ME", ["stat"]) is True
    assert category_matches("statistics", ["stat"]) is False


def test_literature_slate_diversity_uniform_categories_has_max_entropy() -> None:
    results = [{"categories": "cs.LG"}, {"categories": "stat.ML"}, {"categories": "stat.AP"}, {"categories": "stat.ME"}]
    diversity = literature_slate_diversity(results)
    assert diversity["distinct_category_count"] == 4
    assert diversity["category_entropy"] == pytest.approx(math.log(4, 2), rel=1e-6)


def test_literature_slate_diversity_single_category_has_zero_entropy() -> None:
    results = [{"categories": "cs.LG"}, {"categories": "cs.LG"}, {"categories": "cs.LG"}]
    diversity = literature_slate_diversity(results)
    assert diversity["distinct_category_count"] == 1
    assert diversity["category_entropy"] == 0.0


def test_literature_slate_diversity_empty_results() -> None:
    diversity = literature_slate_diversity([])
    assert diversity["distinct_category_count"] == 0
    assert diversity["category_entropy"] == 0.0


def test_select_diverse_literature_slate_returns_all_if_under_top_k() -> None:
    candidates = [{"categories": "cs.LG", "score": 1.0}, {"categories": "cs.LG", "score": 2.0}]
    assert select_diverse_literature_slate(candidates, top_k=5) == candidates


def test_select_diverse_literature_slate_spreads_across_categories() -> None:
    # Two cs.LG candidates rank higher than the one stat.ME candidate by score,
    # but a top_k=2 slate should still pick the stat.ME one for diversity
    # rather than taking the top two cs.LG candidates.
    candidates = [
        {"categories": "cs.LG", "score": 10.0, "intent_rrf_score": 0.5},
        {"categories": "cs.LG", "score": 9.0, "intent_rrf_score": 0.4},
        {"categories": "stat.ME", "score": 1.0, "intent_rrf_score": 0.1},
    ]
    selected = select_diverse_literature_slate(candidates, top_k=2)
    categories_selected = {c["categories"] for c in selected}
    assert categories_selected == {"cs.LG", "stat.ME"}
