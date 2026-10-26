"""
Core pre-processing framework
"""

from .pipeline import LocalPreProcessingPipeline, ProcessedPromptPayload
from .editor import (
    TokenEfficientCodeEditor,
    EditTier,
    EditOperation,
    ParsedInstruction,
    CodeEditResult,
)
from .fidelity import (
    ABFidelityTester,
    ABTestResult,
    FidelityEvaluator,
    FidelityMetrics,
    LLMClient,
    LLMResponse,
)

__all__ = [
    "LocalPreProcessingPipeline",
    "ProcessedPromptPayload",
    "TokenEfficientCodeEditor",
    "EditTier",
    "EditOperation",
    "ParsedInstruction",
    "CodeEditResult",
    "ABFidelityTester",
    "ABTestResult",
    "FidelityEvaluator",
    "FidelityMetrics",
    "LLMClient",
    "LLMResponse",
]
