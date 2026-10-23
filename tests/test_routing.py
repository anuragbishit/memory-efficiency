"""
Unit tests for Weighted Intent Recognition & Dynamic Routing (Pillar 2).
"""

import pytest
from core.routing.intent_engine import (
    WeightedIntentRouter,
    ComplexityTier,
    TargetRuntime,
)
from core.routing.feature_extractor import FeatureExtractor


def test_simple_factual_query_routes_to_local_slm_low_tier():
    router = WeightedIntentRouter()
    prompt = "What is the capital of France?"
    decision = router.route(prompt)

    assert decision.tier == ComplexityTier.LOW
    assert decision.target_runtime == TargetRuntime.LOCAL_SLM
    assert decision.compute_budget_tokens == 256
    assert decision.graph_retrieval_depth_hops == 0
    assert decision.complexity_score < 0.35


def test_code_heavy_and_constraint_query_routes_to_cloud_api_high_tier():
    router = WeightedIntentRouter()
    prompt = (
        "Architect a distributed, thread-safe cache system in Python. "
        "You must strictly optimize for O(1) reads and ensure concurrency locks. "
        "Analyze the tradeoffs between lock striping vs Read-Write locks step-by-step. "
        "Provide full code implementation with class structures:\n"
        "```python\n"
        "class ShardedCache:\n"
        "    def __init__(self, shards: int):\n"
        "        pass\n"
        "```"
    )
    decision = router.route(prompt)

    assert decision.tier == ComplexityTier.HIGH
    assert decision.target_runtime == TargetRuntime.CLOUD_API
    assert decision.compute_budget_tokens == 4096
    assert decision.graph_retrieval_depth_hops >= 2
    assert decision.complexity_score >= 0.70
    assert "code_score" in decision.feature_breakdown


def test_medium_reasoning_query():
    router = WeightedIntentRouter()
    prompt = "Explain why GraphRAG improves multi-hop reasoning over standard text-chunk RAG."
    decision = router.route(prompt)

    assert decision.tier in [ComplexityTier.MEDIUM, ComplexityTier.HIGH]
    assert decision.triage_latency_ms < 10.0
