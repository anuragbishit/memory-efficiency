"""
Unit tests for Dynamic Model Routing and Cost-Latency Optimization (Topic 8).
"""

import pytest
from core.routing.optimization import (
    CostLatencyOptimizer,
    RoutingStrategy,
    DEFAULT_MODEL_REGISTRY,
    RuntimeType,
    ModelCapabilityTier,
)


def test_local_model_zero_cost():
    local_model = DEFAULT_MODEL_REGISTRY["local/llama-3.2-3b-q4"]
    assert local_model.runtime_type == RuntimeType.LOCAL_EDGE
    cost = local_model.calculate_cost_usd(input_tokens=1500, output_tokens=500)
    assert cost == 0.0


def test_cloud_model_commercial_pricing():
    gpt4o = DEFAULT_MODEL_REGISTRY["cloud/gpt-4o"]
    assert gpt4o.runtime_type == RuntimeType.CLOUD_API
    # 1,000,000 in ($2.50) + 1,000,000 out ($10.00) = $12.50
    cost = gpt4o.calculate_cost_usd(input_tokens=1_000_000, output_tokens=1_000_000)
    assert round(cost, 2) == 12.50

    gpt4o_mini = DEFAULT_MODEL_REGISTRY["cloud/gpt-4o-mini"]
    cost_mini = gpt4o_mini.calculate_cost_usd(input_tokens=100_000, output_tokens=50_000)
    assert cost_mini > 0.0
    assert cost_mini < cost


def test_cost_minimization_strategy_favors_local():
    optimizer = CostLatencyOptimizer(default_strategy=RoutingStrategy.COST_MINIMIZATION)
    # Simple query with low complexity
    report = optimizer.optimize(
        complexity_score=0.20,
        prompt_tokens=50,
        raw_prompt_tokens=100,
        compute_budget_tokens=256,
        strategy=RoutingStrategy.COST_MINIMIZATION,
    )

    assert report.runtime_type == "LOCAL_EDGE"
    assert report.estimated_query_cost_usd == 0.0
    assert report.cost_reduction_pct == 100.0
    assert report.immediate_savings_usd > 0.0


def test_quality_maximization_strategy_routes_high_complexity_to_cloud():
    optimizer = CostLatencyOptimizer(default_strategy=RoutingStrategy.QUALITY_MAXIMIZATION)
    # Complex multi-hop system design query
    report = optimizer.optimize(
        complexity_score=0.85,
        prompt_tokens=200,
        raw_prompt_tokens=350,
        compute_budget_tokens=2048,
        strategy=RoutingStrategy.QUALITY_MAXIMIZATION,
    )

    assert report.runtime_type == "CLOUD_API"
    assert report.capability_tier == "EXPERT"
    assert report.selected_model_id in ["cloud/gpt-4o", "cloud/claude-3-5-sonnet"]


def test_balanced_strategy_tradeoffs():
    optimizer = CostLatencyOptimizer(default_strategy=RoutingStrategy.BALANCED)

    # Low complexity -> Local
    rep_low = optimizer.optimize(
        complexity_score=0.15,
        prompt_tokens=30,
        raw_prompt_tokens=60,
        compute_budget_tokens=256,
    )
    assert rep_low.runtime_type == "LOCAL_EDGE"

    # High complexity -> Cloud
    rep_high = optimizer.optimize(
        complexity_score=0.75,
        prompt_tokens=150,
        raw_prompt_tokens=250,
        compute_budget_tokens=2048,
    )
    assert rep_high.runtime_type == "CLOUD_API"
    assert rep_high.immediate_savings_usd >= 0.0
