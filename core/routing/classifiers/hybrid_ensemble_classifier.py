"""
Hybrid Ensemble Intent Classifier.
Blends Rule Engines, TF-IDF + Logistic Regression, and Distilled Transformers
to produce high-confidence consensus triage for consumer-side prompt routing.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from .rule_classifier import RuleIntentClassifier, RuleClassificationResult
from .tfidf_logistic_classifier import TFIDFLogisticIntentClassifier, TFIDFLogisticResult
from .distilled_transformer_classifier import DistilledTransformerClassifier, DistilledTransformerResult
from .training_data import INTENT_CLASSES


@dataclass
class HybridEnsembleResult:
    """Consensus output from the hybrid ensemble intent classifier."""
    consensus_intent: str
    confidence: float
    ensemble_probabilities: Dict[str, float]
    fast_path_triggered: bool
    rule_result: Dict[str, Any]
    tfidf_result: Dict[str, Any]
    transformer_result: Dict[str, Any]
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridEnsembleIntentClassifier:
    """
    Ensemble Intent Classifier.
    Combines:
      - Rule Engine (0.1ms deterministic fast-path)
      - TF-IDF + Logistic Regression (Statistical NLP baseline)
      - Distilled Transformer Semantic Scorer (Dense semantic prototype alignment)
    """

    def __init__(
        self,
        weight_rule: float = 0.20,
        weight_tfidf: float = 0.50,
        weight_transformer: float = 0.30,
        fast_path_threshold: float = 0.90,
    ):
        self.rule_classifier = RuleIntentClassifier()
        self.tfidf_classifier = TFIDFLogisticIntentClassifier()
        self.transformer_classifier = DistilledTransformerClassifier()

        self.w_rule = weight_rule
        self.w_tfidf = weight_tfidf
        self.w_transformer = weight_transformer
        self.fast_path_threshold = fast_path_threshold

    def classify(self, text: str) -> HybridEnsembleResult:
        """Runs the hybrid triage pipeline across all three classification paradigms."""
        start = time.perf_counter()

        # 1. Rule Engine Pass (Fast-path)
        rule_res: RuleClassificationResult = self.rule_classifier.predict(text)

        # 2. TF-IDF + Logistic Regression Pass
        tfidf_res: TFIDFLogisticResult = self.tfidf_classifier.predict(text)

        # 3. Distilled Transformer Pass
        transformer_res: DistilledTransformerResult = self.transformer_classifier.predict(text)

        # Check for fast-path override (e.g. unambiguous code block or stacktrace)
        if rule_res.matched_intent and rule_res.confidence >= self.fast_path_threshold:
            total_latency = round((time.perf_counter() - start) * 1000, 3)
            return HybridEnsembleResult(
                consensus_intent=rule_res.matched_intent,
                confidence=rule_res.confidence,
                ensemble_probabilities=tfidf_res.probabilities,
                fast_path_triggered=True,
                rule_result=rule_res.to_dict(),
                tfidf_result=tfidf_res.to_dict(),
                transformer_result=transformer_res.to_dict(),
                total_latency_ms=total_latency,
            )

        # Blend probabilities across classifiers
        ensemble_probs: Dict[str, float] = {}
        for c in INTENT_CLASSES:
            # Rule probability: if matched, give confidence; else baseline
            p_rule = 0.8 if rule_res.matched_intent == c else 0.05
            p_tfidf = tfidf_res.probabilities.get(c, 0.2)
            p_trans = transformer_res.intent_scores.get(c, 0.2)

            blended = (
                (self.w_rule * p_rule) +
                (self.w_tfidf * p_tfidf) +
                (self.w_transformer * p_trans)
            )
            ensemble_probs[c] = round(blended, 4)

        # Normalize blended distribution
        total_p = sum(ensemble_probs.values())
        if total_p > 0:
            ensemble_probs = {k: round(v / total_p, 4) for k, v in ensemble_probs.items()}

        consensus_intent = max(ensemble_probs, key=ensemble_probs.get)
        confidence = ensemble_probs[consensus_intent]
        total_latency = round((time.perf_counter() - start) * 1000, 3)

        return HybridEnsembleResult(
            consensus_intent=consensus_intent,
            confidence=confidence,
            ensemble_probabilities=ensemble_probs,
            fast_path_triggered=False,
            rule_result=rule_res.to_dict(),
            tfidf_result=tfidf_res.to_dict(),
            transformer_result=transformer_res.to_dict(),
            total_latency_ms=total_latency,
        )
