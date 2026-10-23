"""
Output Answer Fidelity & Empirical A/B Quality Testing Package.
Verifies that prompt compression strips only fluff while preserving 100% of LLM intelligence.
"""

from .llm_client import LLMClient, LLMResponse, DeterministicFrontierEngine
from .evaluator import FidelityEvaluator, FidelityMetrics
from .ab_tester import ABFidelityTester, ABTestResult

__all__ = [
    "LLMClient",
    "LLMResponse",
    "DeterministicFrontierEngine",
    "FidelityEvaluator",
    "FidelityMetrics",
    "ABFidelityTester",
    "ABTestResult",
]
