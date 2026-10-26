"""
Core Editor Module: Token-Efficient Direct Code Editing System.
Implements the 3-Tier Architecture:
  Tier 1: Direct Zero-Token Edit (AST & Regex) - 0 LLM calls
  Tier 2: Targeted Retrieval (Micro-context snippet) - Minimal tokens
  Tier 3: Deep Reasoning (Multi-file / complex logic) - Escalated context
"""

from core.editor.models import EditTier, EditOperation, ParsedInstruction, CodeEditResult
from core.editor.intent_classifier import EditIntentClassifier
from core.editor.direct_engine import DirectEditEngine
from core.editor.targeted_retrieval import TargetedRetrievalEngine
from core.editor.patch_engine import MinimalPatchEngine
from core.editor.orchestrator import TokenEfficientCodeEditor

__all__ = [
    "EditTier",
    "EditOperation",
    "ParsedInstruction",
    "CodeEditResult",
    "EditIntentClassifier",
    "DirectEditEngine",
    "TargetedRetrievalEngine",
    "MinimalPatchEngine",
    "TokenEfficientCodeEditor",
]
