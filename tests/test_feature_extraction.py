"""
Unit tests for Comprehensive Prompt Feature Extraction (Topic 7).
"""

import pytest
from core.routing.feature_extractor import (
    FeatureExtractor,
    PromptFeatures,
    QuestionType,
)


def test_feature_extraction_completeness():
    extractor = FeatureExtractor()
    prompt = (
        "Why does GraphRAG eliminate context bleed compared to flat vector embeddings? "
        "Explain step-by-step with mathematical reasoning."
    )
    features = extractor.extract(prompt, token_count=25)

    assert isinstance(features, PromptFeatures)
    assert features.token_count == 25
    assert features.char_count == len(prompt)
    assert features.word_count > 0
    assert features.char_to_token_ratio > 0
    assert features.type_token_ratio > 0.5
    assert features.sentence_count >= 1
    assert features.reasoning_density_score > 0.3
    assert "why" in features.detected_reasons or "step-by-step" in features.detected_reasons


def test_question_type_classification():
    extractor = FeatureExtractor()

    causal = extractor.extract("Why does the model hallucinate?", 10)
    assert causal.question_type == QuestionType.CAUSAL.value

    process = extractor.extract("How does a BPE tokenizer split bytes?", 10)
    assert process.question_type == QuestionType.PROCESS.value

    comparative = extractor.extract("Compare and contrast GraphRAG versus vector RAG.", 10)
    assert comparative.question_type == QuestionType.COMPARATIVE.value

    directive = extractor.extract("Optimize this SQL query for high concurrency.", 10)
    assert directive.question_type == QuestionType.DIRECTIVE.value


def test_ml_feature_vector_dimension_and_bounds():
    extractor = FeatureExtractor()
    prompt = "Optimize this python function: ```def foo(): pass```"
    features = extractor.extract(prompt, token_count=15)

    vector = features.to_feature_vector()
    names = features.feature_names()

    # Exact 12-dimensional feature vector for ML models
    assert len(vector) == 12
    assert len(names) == 12

    # All normalized features must be bounded in [0.0, 1.0]
    for idx, val in enumerate(vector):
        assert 0.0 <= val <= 1.0, f"Feature '{names[idx]}' value {val} out of bounds!"
