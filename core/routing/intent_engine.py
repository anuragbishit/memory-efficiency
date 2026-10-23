"""
Weighted Intent Recognition and Dynamic Routing Engine (Pillar 2).
Calculates query complexity scores, determines compute budgets,
and routes tasks between Local Edge SLMs and Upstream Cloud APIs.
"""

import time
from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from core.tokenization.tokenizer import SubwordTokenizer
from .feature_extractor import FeatureExtractor, PromptFeatures
from .classifiers import (
    HybridEnsembleIntentClassifier,
    HybridEnsembleResult,
)
from .optimization import (
    CostLatencyOptimizer,
    CostLatencyOptimizationReport,
    RoutingStrategy,
)


class ComplexityTier(str, Enum):
    """Query complexity tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TargetRuntime(str, Enum):
    """Dynamic execution destination."""
    LOCAL_SLM = "LOCAL_SLM"       # e.g., Quantized GGUF model via llama.cpp
    CLOUD_API = "CLOUD_API"       # e.g., Upstream GPT-4 / Claude Cloud API


@dataclass
class IntentRouteDecision:
    """Complete routing decision payload."""
    tier: ComplexityTier
    complexity_score: float
    target_runtime: TargetRuntime
    selected_model_id: str
    selected_model_name: str
    compute_budget_tokens: int
    graph_retrieval_depth_hops: int
    triage_latency_ms: float
    feature_breakdown: Dict[str, float]
    rationales: List[str]
    ensemble_classification: Dict[str, Any]
    cost_latency_optimization: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tier"] = self.tier.value
        data["target_runtime"] = self.target_runtime.value
        return data


class WeightedIntentRouter:
    """
    Weighted Intent Router (Pillar 2).
    Evaluates lexical, structural, and semantic signals using a hybrid ensemble
    of Rule Engines, TF-IDF + Logistic Regression, and Distilled Transformers.
    Dynamically optimizes model routing against commercial cost and latency constraints.
    """

    def __init__(
        self,
        tokenizer: SubwordTokenizer | None = None,
        feature_extractor: FeatureExtractor | None = None,
        ensemble_classifier: HybridEnsembleIntentClassifier | None = None,
        cost_optimizer: CostLatencyOptimizer | None = None,
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        weight_code: float = 0.30,
        weight_reasoning: float = 0.30,
        weight_constraint: float = 0.20,
        weight_syntax: float = 0.20,
        low_threshold: float = 0.35,
        high_threshold: float = 0.70,
    ):
        self.tokenizer = tokenizer or SubwordTokenizer()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.ensemble_classifier = ensemble_classifier or HybridEnsembleIntentClassifier()
        self.cost_optimizer = cost_optimizer or CostLatencyOptimizer(default_strategy=default_strategy)
        self.default_strategy = default_strategy

        # Configurable feature weights (must sum to 1.0)
        total_w = weight_code + weight_reasoning + weight_constraint + weight_syntax
        self.w_code = weight_code / total_w
        self.w_reasoning = weight_reasoning / total_w
        self.w_constraint = weight_constraint / total_w
        self.w_syntax = weight_syntax / total_w

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def route(self, text: str, encoding_name: str = "cl100k_base") -> IntentRouteDecision:
        """
        Evaluates prompt complexity using continuous feature scoring and
        hybrid ensemble classification (Rule Engine + TF-IDF + Distilled Transformer).
        """
        start_time = time.perf_counter()

        token_count = self.tokenizer.count_tokens(text, encoding_name)
        features: PromptFeatures = self.feature_extractor.extract(text, token_count)

        # 1. Run Hybrid Ensemble Classifier (Rules + TF-IDF/Logistic + Distilled Transformer)
        ensemble_res: HybridEnsembleResult = self.ensemble_classifier.classify(text)

        # 2. Compute weighted complexity score with peak signal amplification
        base_score = (
            (self.w_code * features.code_presence_score) +
            (self.w_reasoning * features.reasoning_density_score) +
            (self.w_constraint * features.constraint_density_score) +
            (self.w_syntax * features.syntactic_complexity_score)
        )

        # Domain peak signal: Strong presence in code, reasoning, or constraints elevates tier
        peak_signal = max(
            features.code_presence_score,
            features.reasoning_density_score,
            features.constraint_density_score
        )
        complexity_score = round(min(1.0, max(base_score, peak_signal * 0.70)), 3)

        # If ensemble predicts a heavy technical intent with high confidence, elevate complexity score
        if ensemble_res.consensus_intent in ["CODE_GENERATION", "SYSTEM_ARCHITECTURE", "DEBUGGING"]:
            complexity_score = max(complexity_score, round(0.70 * ensemble_res.confidence, 3))
        elif ensemble_res.consensus_intent == "EXPLANATION":
            complexity_score = max(complexity_score, round(0.40 * ensemble_res.confidence, 3))

        rationales: List[str] = []
        rationales.append(
            f"Ensemble consensus: {ensemble_res.consensus_intent} (Confidence: {round(ensemble_res.confidence*100, 1)}%, FastPath: {ensemble_res.fast_path_triggered})"
        )

        # Decision Tiers
        if complexity_score < self.low_threshold:
            tier = ComplexityTier.LOW
            runtime = TargetRuntime.LOCAL_SLM
            budget_tokens = 256
            retrieval_depth = 0
            rationales.append("Query is simple/factual; low compute allocation sufficient.")
        elif complexity_score < self.high_threshold:
            tier = ComplexityTier.MEDIUM
            runtime = TargetRuntime.LOCAL_SLM
            budget_tokens = 1024
            retrieval_depth = 1
            rationales.append("Query involves moderate context or single-step logic.")
        else:
            tier = ComplexityTier.HIGH
            runtime = TargetRuntime.CLOUD_API
            budget_tokens = 4096
            retrieval_depth = 2
            rationales.append("High complexity detected (multi-step reasoning, strict constraints, or code blocks).")

        if features.has_code_block:
            rationales.append("Active programming code markers detected in payload.")
        if features.detected_constraints:
            rationales.append(f"Constraints detected: {', '.join(features.detected_constraints)}")
        if features.detected_reasons:
            rationales.append(f"Reasoning signals: {', '.join(features.detected_reasons)}")
        if features.primary_script != "Latin" and features.primary_script != "None":
            rationales.append(
                f"Non-Latin script detected ({features.primary_script}, frag ratio {features.fragmentation_ratio}x). "
                f"Suggest o200k_base encoding to minimize BPE fragmentation."
            )
            # Increase compute budget by 1.5x for non-Latin to prevent premature truncation
            budget_tokens = int(budget_tokens * 1.5)

        # 3. Cost-Latency Multi-Objective Model Optimization (Topic 8)
        opt_report: CostLatencyOptimizationReport = self.cost_optimizer.optimize(
            complexity_score=complexity_score,
            prompt_tokens=token_count,
            raw_prompt_tokens=token_count,
            compute_budget_tokens=budget_tokens,
            strategy=self.default_strategy,
            intent_category=ensemble_res.consensus_intent,
        )

        # If optimizer picks local edge, ensure runtime is LOCAL_SLM; else CLOUD_API
        if opt_report.runtime_type == "LOCAL_EDGE":
            runtime = TargetRuntime.LOCAL_SLM
        else:
            runtime = TargetRuntime.CLOUD_API

        rationales.extend(opt_report.routing_rationales[:2])

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return IntentRouteDecision(
            tier=tier,
            complexity_score=complexity_score,
            target_runtime=runtime,
            selected_model_id=opt_report.selected_model_id,
            selected_model_name=opt_report.selected_model_name,
            compute_budget_tokens=budget_tokens,
            graph_retrieval_depth_hops=retrieval_depth,
            triage_latency_ms=latency_ms,
            feature_breakdown={
                "code_score": features.code_presence_score,
                "reasoning_score": features.reasoning_density_score,
                "constraint_score": features.constraint_density_score,
                "syntactic_score": features.syntactic_complexity_score,
            },
            rationales=rationales,
            ensemble_classification=ensemble_res.to_dict(),
            cost_latency_optimization=opt_report.to_dict(),
        )
