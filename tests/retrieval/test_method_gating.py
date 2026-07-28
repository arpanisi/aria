"""Unit tests for the method-gate eligibility filter and text-matching primitives."""
from __future__ import annotations

from scripts.retrieval.method_gating import (
    _contains_phrase,
    _phrase_variants,
    _tokens,
    link_method_terms,
)


def test_contains_phrase_requires_word_boundary() -> None:
    assert _contains_phrase("we used cross validation here", "cross validation") is True
    assert _contains_phrase("crossvalidationtest", "cross validation") is False


def test_contains_phrase_empty_phrase_is_false() -> None:
    assert _contains_phrase("anything", "") is False


def test_phrase_variants_pluralizes_and_singularizes() -> None:
    assert "categorical variable" in _phrase_variants("categorical variables")
    assert "categorical variables" in _phrase_variants("categorical variable")


def test_tokens_drops_stopwords_and_short_tokens() -> None:
    tokens = _tokens("This is a test of the tokenizer with stopwords")
    assert "is" not in tokens
    assert "the" not in tokens
    assert "with" not in tokens
    assert "test" in tokens
    assert "tokenizer" in tokens


def test_link_method_terms_eligible_when_vocabulary_matches() -> None:
    text = "We propose a statistical model for tabular data with cross validation."
    result = link_method_terms({}, text, dataset_profile={}, data_evidence=[])
    assert result["eligible_for_method_guidance"] is True
    assert "profile_or_design" in result["matched_method_groups"] or "model_or_estimation" in result["matched_method_groups"]


def test_link_method_terms_ineligible_for_unrelated_text() -> None:
    text = "This paper is about deep sea coral reef biodiversity in the Pacific."
    result = link_method_terms({}, text, dataset_profile={}, data_evidence=[])
    assert result["eligible_for_method_guidance"] is False


def test_link_method_terms_adds_mixed_effects_vocab_for_repeated_measures() -> None:
    profile_with_repeat = {"repeated_measures": {"detected": True}}
    text = "We use a mixed effects model for longitudinal repeated measures data."
    result = link_method_terms({}, text, dataset_profile=profile_with_repeat, data_evidence=[])
    assert result["eligible_for_method_guidance"] is True
