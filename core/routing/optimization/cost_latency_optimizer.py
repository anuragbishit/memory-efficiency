"""
Dynamic Model Routing and Cost-Latency Optimization Engine.
Applies Pareto-optimal trade-off analysis across cost ($), latency (ms),
and reasoning capability to select optimal local or cloud runtimes.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .model_registry import (
    ModelMetadata,
    RuntimeType,
    ModelCapabilityTier,
    DEFAULT_MODEL_REGISTRY,
)


class RoutingStrategy(str, Enum):
    """User-selectable optimization strategies."""
    COST_MINIMIZATION = "COST_MINIMIZATION"       # Maximizes financial savings ($0.00 local bias)
    BALANCED = "BALANCED"                         # Balances capability, budget, and latency
    LATENCY_MINIMIZATION = "LATENCY_MINIMIZATION" # Minimizes time-to-first-token & generation
    QUALITY_MAXIMIZATION = "QUALITY_MAXIMIZATION" # Prioritizes expert reasoning and full fidelity


@dataclass
class CostLatencyOptimizationReport:
    """Comprehensive financial and latency optimization verdict."""
    selected_model_id: str
    selected_model_name: str
    runtime_type: str
    capability_tier: str
    strategy_used: str

    # Financial Metrics ($ USD)
    estimated_query_cost_usd: float
    baseline_unoptimized_cost_usd: float  # Raw prompt sent to GPT-4o
    immediate_savings_usd: float
    cost_reduction_pct: float
    projected_10k_queries_savings_usd: float

    # Latency Metrics (ms)
    estimated_ttft_ms: float
    estimated_total_latency_ms: float

    # Optimization Rationale
    routing_rationales: List[str]
    model_comparison_table: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostLatencyOptimizer:
    """
    Cost-Latency Multi-Objective Optimizer.
    Evaluates Pareto frontiers across financial billing and generation latency
    to dynamically dispatch prompts to optimal local or cloud endpoints.
    """

    BASELINE_BENCHMARK_MODEL_ID = "cloud/gpt-4o"

    def __init__(
        self,
        registry: Dict[str, ModelMetadata] | None = None,
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    ):
        self.registry = registry or DEFAULT_MODEL_REGISTRY
        self.default_strategy = default_strategy

    def _score_capability_match(
        self,
        model: ModelMetadata,
        complexity_score: float
    ) -> float:
        """
        Calculates alignment between model capability tier and query complexity.
        Heavily penalizes under-powered models when complexity is High (>= 0.70).
        High-capability models are never penalized for being more capable than needed.
        """
        tier_weights = {
            ModelCapabilityTier.ENTRY: 0.25,
            ModelCapabilityTier.INTERMEDIATE: 0.55,
            ModelCapabilityTier.ADVANCED: 0.80,
            ModelCapabilityTier.EXPERT: 1.00,
        }
        model_cap = tier_weights.get(model.capability_tier, 0.5)

        # High complexity tasks (>= 0.70) strictly require ADVANCED or EXPERT runtimes
        if complexity_score >= 0.70 and model_cap < 0.80:
            return -5.0  # Inadmissible (blocks under-powered edge SLMs from high-complexity tasks)

        # Under-capacity penalty
        if model_cap < complexity_score - 0.10:
            return -3.0

        # When model satisfies or exceeds query complexity
        if model_cap >= complexity_score:
            # Expert models provide max fidelity
            return 1.0 if model.capability_tier == ModelCapabilityTier.EXPERT else 0.90

        # Sub-threshold fallback
        return max(0.0, 1.0 - (complexity_score - model_cap))

    def optimize(
        self,
        complexity_score: float,
        prompt_tokens: int,
        raw_prompt_tokens: int,
        compute_budget_tokens: int,
        strategy: Optional[RoutingStrategy] = None,
        intent_category: Optional[str] = None,
    ) -> CostLatencyOptimizationReport:
        """
        Calculates Pareto utility across all registered models and selects the optimal runtime.
        Enforces Minor Project Architecture:
          - LOW / MEDIUM complexity (< 0.70) -> Sovereign Local Edge SLM ($0.00 cost)
          - HIGH complexity (>= 0.70) -> Commercial Cloud API (GPT-4o / Claude 3.5 Sonnet)
        """
        strat = strategy or self.default_strategy

        # Baseline: Uncompressed raw prompt executed on commercial GPT-4o
        baseline_model = self.registry.get(self.BASELINE_BENCHMARK_MODEL_ID)
        if baseline_model:
            baseline_cost = baseline_model.calculate_cost_usd(raw_prompt_tokens, compute_budget_tokens)
        else:
            baseline_cost = 0.001

        # Determine target runtime candidate pool based on architectural tiers and strategy
        if strat == RoutingStrategy.COST_MINIMIZATION:
            # Strictly prefer local edge whenever complexity <= 0.80
            allowed_runtime = RuntimeType.LOCAL_EDGE if complexity_score <= 0.80 else RuntimeType.CLOUD_API
        elif strat == RoutingStrategy.QUALITY_MAXIMIZATION:
            # High complexity gets expert frontier cloud models
            allowed_runtime = RuntimeType.CLOUD_API if complexity_score >= 0.50 else RuntimeType.LOCAL_EDGE
        else:  # BALANCED / LATENCY
            # Standard Stage 4 Pipeline: Low/Medium -> Local SLM, High -> Cloud API
            allowed_runtime = RuntimeType.LOCAL_EDGE if complexity_score < 0.70 else RuntimeType.CLOUD_API

        # Filter candidate models
        candidate_models = [m for m in self.registry.values() if m.runtime_type == allowed_runtime]
        if not candidate_models:
            candidate_models = list(self.registry.values())

        # Expected output token estimation (short queries don't generate max budget)
        expected_output = min(compute_budget_tokens, 20 if complexity_score < 0.35 else (100 if complexity_score < 0.70 else compute_budget_tokens))

        model_evaluations: List[Tuple[float, ModelMetadata, float, float]] = []
        table_rows: List[Dict[str, Any]] = []

        # Strategy weights: (w_cost, w_latency, w_capability)
        strategy_weights = {
            RoutingStrategy.COST_MINIMIZATION: (0.70, 0.10, 0.20),
            RoutingStrategy.BALANCED: (0.35, 0.25, 0.40),
            RoutingStrategy.LATENCY_MINIMIZATION: (0.15, 0.60, 0.25),
            RoutingStrategy.QUALITY_MAXIMIZATION: (0.05, 0.10, 0.85),
        }
        w_cost, w_lat, w_cap = strategy_weights.get(strat, (0.35, 0.25, 0.40))

        for model in self.registry.values():
            cost_usd = model.calculate_cost_usd(prompt_tokens, expected_output)
            total_lat = model.estimate_total_latency_ms(expected_output)
            cap_match = self._score_capability_match(model, complexity_score)

            # Preference bonus for the designated runtime tier
            runtime_bonus = 2.0 if model.runtime_type == allowed_runtime else -2.0

            # Quality strategy bonus for Expert models when complexity is high
            if strat == RoutingStrategy.QUALITY_MAXIMIZATION and complexity_score >= 0.70:
                if model.capability_tier == ModelCapabilityTier.EXPERT:
                    runtime_bonus += 3.0

            norm_cost = min(1.0, cost_usd / 0.015) if cost_usd > 0 else 0.0
            norm_lat = min(1.0, total_lat / 3000.0)

            utility = (
                (-w_cost * norm_cost) +
                (-w_lat * norm_lat) +
                (w_cap * cap_match) +
                runtime_bonus
            )

            if model in candidate_models:
                model_evaluations.append((utility, model, cost_usd, total_lat))

            table_rows.append({
                "model_id": model.model_id,
                "display_name": model.display_name,
                "runtime": model.runtime_type.value,
                "cost_usd": cost_usd,
                "total_lat_ms": total_lat,
                "utility": round(utility, 3),
            })

        # Select model with maximum utility within candidate pool
        best_eval = max(model_evaluations, key=lambda x: x[0])
        best_model = best_eval[1]
        best_cost = best_eval[2]
        best_lat = best_eval[3]

        immediate_saved = max(0.0, round(baseline_cost - best_cost, 6))
        reduction_pct = round((immediate_saved / baseline_cost) * 100, 2) if baseline_cost > 0 else 0.0
        projected_10k = round(immediate_saved * 10_000, 2)

        rationales: List[str] = []
        if best_model.runtime_type == RuntimeType.LOCAL_EDGE:
            rationales.append(
                f"Selected local edge SLM '{best_model.display_name}': 100% free ($0.00) execution on local consumer hardware."
            )
            rationales.append("Zero network round-trip overhead; full data privacy sovereign execution.")
        else:
            rationales.append(
                f"High task complexity ({complexity_score} / 1.0) warrants cloud dispatch to '{best_model.display_name}'."
            )
            rationales.append(
                f"Payload pre-compression saved {round(raw_prompt_tokens - prompt_tokens, 1)} tokens prior to upstream API billing."
            )

        rationales.append(
            f"Strategy '{strat.value}' achieved {reduction_pct}% cost reduction vs baseline uncompressed GPT-4o."
        )

        return CostLatencyOptimizationReport(
            selected_model_id=best_model.model_id,
            selected_model_name=best_model.display_name,
            runtime_type=best_model.runtime_type.value,
            capability_tier=best_model.capability_tier.value,
            strategy_used=strat.value,
            estimated_query_cost_usd=best_cost,
            baseline_unoptimized_cost_usd=baseline_cost,
            immediate_savings_usd=immediate_saved,
            cost_reduction_pct=reduction_pct,
            projected_10k_queries_savings_usd=projected_10k,
            estimated_ttft_ms=best_model.total_ttft_ms,
            estimated_total_latency_ms=best_lat,
            routing_rationales=rationales,
            model_comparison_table=table_rows,
        )
