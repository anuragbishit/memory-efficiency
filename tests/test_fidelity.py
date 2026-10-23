"""
Unit and Integration Tests for Output Answer Fidelity (Module 10).
Validates LLM execution, semantic similarity, ROUGE-L, entity preservation,
AST code equivalence, and end-to-end A/B test pipeline.
"""

import pytest
from core.pipeline import LocalPreProcessingPipeline
from core.fidelity import (
    LLMClient,
    LLMResponse,
    DeterministicFrontierEngine,
    FidelityEvaluator,
    FidelityMetrics,
    ABFidelityTester,
    ABTestResult,
)


def test_deterministic_frontier_simulation_engine():
    engine = DeterministicFrontierEngine()

    # Conversational Code prompt
    out_code = engine.generate("Hello, could you help me sort an array in python? Thanks!")
    assert "sorted(" in out_code
    assert ".sort()" in out_code
    assert "Timsort" in out_code

    # Factual prompt
    out_fact = engine.generate("What is the capital of France?")
    assert "Paris" in out_fact

    # Hindi multilingual prompt
    out_hindi = engine.generate("नमस्ते, क्या आप मुझे ग्राफ डेटा संरचना के बारे में बता सकते हैं?")
    assert "ग्राफ" in out_hindi

    # Custom prompt
    out_custom = engine.generate("Explain quantum superposition and entanglement.")
    assert "superposition" in out_custom.lower() or "entanglement" in out_custom.lower()


def test_fidelity_evaluator_identical_texts():
    evaluator = FidelityEvaluator()
    sample = "In Python, use sorted(iterable) or list.sort() with O(n log n) complexity."

    metrics: FidelityMetrics = evaluator.evaluate(sample, sample)
    assert metrics.semantic_similarity_pct == 100.0
    assert metrics.rouge_l_f1_pct == 100.0
    assert metrics.key_entity_retention_pct == 100.0
    assert metrics.composite_fidelity_pct == 100.0
    assert metrics.intelligence_loss_pct == 0.0
    assert "[PASSED]" in metrics.examiner_verdict


def test_fidelity_evaluator_code_ast():
    evaluator = FidelityEvaluator()
    code_a = "```python\ndef solve(x: int) -> int:\n    return x * 2\n```"
    code_b = "```python\ndef solve(x: int) -> int:\n    # Optimized multiplication\n    return x * 2\n```"

    equiv = evaluator.verify_code_equivalence(code_a, code_b)
    assert equiv is True


def test_end_to_end_ab_fidelity_tester():
    tester = ABFidelityTester()
    raw_prompt = (
        "Hello there, could you please kindly assist me in understanding how to sort an array "
        "in Python? I was wondering if you might be able to explain it simply for me. "
        "Thank you so much in advance, I really appreciate your help!"
    )

    result: ABTestResult = tester.run_ab_test(raw_prompt)

    # 1. Verification of Token & Economic Savings
    assert result.raw_prompt_tokens > result.compressed_prompt_tokens
    assert result.prompt_tokens_saved > 0
    assert result.prompt_token_savings_pct > 30.0

    # 2. Verification of Intelligence Preservation
    assert result.metrics.composite_fidelity_pct >= 95.0
    assert result.metrics.intelligence_loss_pct == 0.0
    assert result.metrics.key_entity_retention_pct >= 95.0
    assert result.metrics.semantic_similarity_pct >= 95.0

    # 3. Verification of Examiner Verdict
    assert "Zero intelligence" in result.metrics.examiner_verdict or "PASSED" in result.metrics.examiner_verdict


def test_pipeline_fidelity_integration():
    pipeline = LocalPreProcessingPipeline()
    prompt = "Please explain the capital of France and its history. Thank you!"

    # Direct helper test
    ab_res = pipeline.run_fidelity_ab_test(prompt)
    assert isinstance(ab_res, ABTestResult)
    assert ab_res.raw_prompt_tokens > 0
    assert ab_res.metrics.composite_fidelity_pct >= 90.0

    # Pipeline process with flag
    payload = pipeline.process(prompt, run_ab_fidelity=True)
    assert payload.ab_fidelity is not None
    assert "composite_fidelity_pct" in payload.ab_fidelity["metrics"]
