"""
Unit tests for Lightweight Intent Classification:
Rule Engine, TF-IDF + Logistic Regression, Distilled Transformers, and Hybrid Ensemble.
"""

import pytest
from core.routing.classifiers import (
    RuleIntentClassifier,
    TFIDFLogisticIntentClassifier,
    DistilledTransformerClassifier,
    HybridEnsembleIntentClassifier,
)


def test_rule_classifier_fast_path():
    classifier = RuleIntentClassifier()

    # 1. Error / traceback trigger
    err_prompt = "TypeError: cannot unpack non-iterable NoneType object in line 42."
    res_err = classifier.predict(err_prompt)
    assert res_err.matched_intent == "DEBUGGING"
    assert res_err.confidence >= 0.90

    # 2. Code block trigger
    code_prompt = "```python\ndef solve(): pass\n```"
    res_code = classifier.predict(code_prompt)
    assert res_code.matched_intent == "CODE_GENERATION"
    assert res_code.confidence >= 0.90

    # 3. Simple greeting
    greet_prompt = "Hello there!"
    res_greet = classifier.predict(greet_prompt)
    assert res_greet.matched_intent == "FACTUAL_LOOKUP"
    assert res_greet.confidence >= 0.95


def test_tfidf_logistic_classifier_training_and_inference():
    classifier = TFIDFLogisticIntentClassifier()

    # Test code generation prompt
    code_query = "Implement binary search tree insertion in C++."
    res_code = classifier.predict(code_query)
    assert res_code.predicted_intent in ["CODE_GENERATION", "EXPLANATION"]
    assert res_code.confidence > 0.3
    assert len(res_code.probabilities) == 5
    # Probabilities must sum to ~1.0
    assert abs(sum(res_code.probabilities.values()) - 1.0) < 0.05
    assert res_code.latency_ms < 10.0


def test_distilled_transformer_semantic_scorer():
    scorer = DistilledTransformerClassifier()

    arch_query = "Architect a distributed sharded cache with fault tolerance and consensus."
    res = scorer.predict(arch_query)
    assert res.predicted_intent == "SYSTEM_ARCHITECTURE"
    assert res.confidence > 0.25
    assert "SYSTEM_ARCHITECTURE" in res.intent_scores
    assert res.latency_ms < 5.0


def test_hybrid_ensemble_classifier():
    ensemble = HybridEnsembleIntentClassifier()

    # Test ambiguous / conceptual prompt
    prompt = "Explain why GraphRAG eliminates context bleed in long-term AI memory."
    res = ensemble.classify(prompt)

    assert res.consensus_intent == "EXPLANATION"
    assert res.confidence > 0.3
    assert res.fast_path_triggered is False
    assert "rule_result" in res.to_dict()
    assert "tfidf_result" in res.to_dict()
    assert "transformer_result" in res.to_dict()
    assert res.total_latency_ms < 15.0


def test_hybrid_ensemble_fast_path_override():
    ensemble = HybridEnsembleIntentClassifier()

    code_prompt = (
        "Here is my code:\n"
        "```python\n"
        "def quicksort(arr): return arr\n"
        "```"
    )
    res = ensemble.classify(code_prompt)

    assert res.consensus_intent == "CODE_GENERATION"
    assert res.fast_path_triggered is True
    assert res.confidence >= 0.90
