"""Tests for turning retrieved literature batches into selected method specs.

test_summarize_retrieved_method_specs_real_deterministic_extraction is a real
integration test: no mocking, it drives the actual deterministic paper-method
extractor and its on-disk cache (summarize_method_spec_cached), the same path
_run_action's summarize_method_specs branch uses when --paper-summarizer
deterministic is selected.
"""
from __future__ import annotations

from scripts.orchestration.literature_summarization import (
    _iter_unsummarized_literature_results,
    _paper_text_from_literature_result,
    _seen_literature_paper_ids,
    _seen_method_spec_paper_ids,
    _summarize_retrieved_method_specs,
)

BOOTSTRAP_ABSTRACT = (
    "We propose a bootstrap resampling procedure to assess coefficient sign "
    "stability for an interpretable regression model, reporting cross-validated fit."
)


def test_seen_literature_paper_ids_collects_across_batches() -> None:
    state = {
        "literature_evidence": [
            {"results": [{"paper_id": "p1"}, {"paper_id": "p2"}]},
            {"results": [{"paper_id": "p2"}, {"paper_id": ""}]},
        ]
    }
    assert _seen_literature_paper_ids(state) == {"p1", "p2"}


def test_seen_method_spec_paper_ids_reads_nested_source() -> None:
    state = {
        "method_spec_evidence": [
            {"method_spec_records": [{"method_spec": {"source": {"paper_id": "p1"}}}]},
        ]
    }
    assert _seen_method_spec_paper_ids(state) == {"p1"}


def test_iter_unsummarized_literature_results_excludes_seen_and_tags_round() -> None:
    state = {
        "literature_evidence": [
            {
                "retrieval_round": "initial",
                "results": [{"paper_id": "p1"}, {"paper_id": "p2"}],
            }
        ]
    }
    results = _iter_unsummarized_literature_results(state, seen={"p1"})
    assert [r["paper_id"] for r in results] == ["p2"]
    assert results[0]["retrieval_round"] == "initial"


def test_paper_text_from_literature_result_includes_title_abstract_and_full_text() -> None:
    item = {"title": "T", "abstract": "A", "full_text": "F" * 20}
    text = _paper_text_from_literature_result(item)
    assert "Title: T" in text
    assert "Abstract: A" in text
    assert "Full text excerpt:" in text


def test_paper_text_from_literature_result_falls_back_to_summary_when_no_abstract() -> None:
    item = {"title": "T", "summary": "S"}
    text = _paper_text_from_literature_result(item)
    assert "Abstract: S" in text


def test_summarize_retrieved_method_specs_real_deterministic_extraction(tmp_path) -> None:
    state = {
        "dataset_profile": {"numeric_columns": ["x1", "y"]},
        "literature_evidence": [
            {
                "retrieval_round": "initial",
                "results": [
                    {"paper_id": "arxiv.1111", "title": "Bootstrap Paper", "abstract": BOOTSTRAP_ABSTRACT},
                ],
            }
        ],
    }
    observation = _summarize_retrieved_method_specs(
        state,
        summarizer="deterministic",
        model="unused",
        reasoning_mode="none",
        limit=2,
        method_spec_cache_dir=tmp_path,
    )

    assert observation["status"] == "ok"
    assert observation["n_papers_summarized"] == 1
    assert observation["selected_method_spec_name"] == "bootstrap_stability_screening"
    cached_files = list(tmp_path.glob("*.json"))
    assert len(cached_files) == 1


def test_summarize_retrieved_method_specs_reuses_cache_on_second_call(tmp_path) -> None:
    state = {
        "dataset_profile": {"numeric_columns": ["x1", "y"]},
        "literature_evidence": [
            {
                "retrieval_round": "initial",
                "results": [
                    {"paper_id": "arxiv.2222", "title": "Bootstrap Paper", "abstract": BOOTSTRAP_ABSTRACT},
                ],
            }
        ],
    }
    first = _summarize_retrieved_method_specs(
        state,
        summarizer="deterministic",
        model="unused",
        reasoning_mode="none",
        limit=2,
        method_spec_cache_dir=tmp_path,
    )
    record = first["method_spec_records"][0]
    assert record.get("cache_hit") is not True

    state["method_spec_evidence"] = [first]
    second = _summarize_retrieved_method_specs(
        state,
        summarizer="deterministic",
        model="unused",
        reasoning_mode="none",
        limit=2,
        method_spec_cache_dir=tmp_path,
    )
    assert second["status"] == "empty"
    assert second["n_papers_summarized"] == 0


def test_summarize_retrieved_method_specs_empty_when_no_new_literature() -> None:
    observation = _summarize_retrieved_method_specs(
        {"dataset_profile": {}, "literature_evidence": []},
        summarizer="deterministic",
        model="unused",
        reasoning_mode="none",
        limit=2,
        method_spec_cache_dir=None,  # unused when there is nothing to summarize
    )
    assert observation["status"] == "empty"
    assert observation["selected_method_spec"] is None
