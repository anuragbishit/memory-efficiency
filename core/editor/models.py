"""
Data models and schemas for Token-Efficient Direct Code Editing.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
import uuid


class EditTier(str, Enum):
    """Execution tiers based on complexity and token requirements."""
    DIRECT_EDIT = "DIRECT_EDIT"               # 0 LLM Calls, 0 Tokens (Regex / AST / Exact Search)
    TARGETED_RETRIEVAL = "TARGETED_RETRIEVAL" # 1 LLM Call, Minimal Tokens (Local snippet extraction)
    DEEP_REASONING = "DEEP_REASONING"         # Multi-step LLM, Full context for complex refactoring/bugs


class EditOperation(str, Enum):
    """Specific edit operations recognized by the intent classifier."""
    ASSIGNMENT_CHANGE = "ASSIGNMENT_CHANGE"   # e.g., "Change a = 10 to a = 12"
    LITERAL_REPLACE = "LITERAL_REPLACE"       # e.g., "Replace 'localhost' with 'api.prod.com'"
    VARIABLE_RENAME = "VARIABLE_RENAME"       # e.g., "Rename total_count to items_sum"
    FUNCTION_EDIT = "FUNCTION_EDIT"           # e.g., "Update timeout in fetch_data"
    GENERAL_EDIT = "GENERAL_EDIT"             # Complex semantic edit or bug fix


@dataclass
class ParsedInstruction:
    """Structured representation of a parsed user edit instruction."""
    raw_instruction: str
    tier: EditTier
    operation: EditOperation
    target_variable: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    target_function: Optional[str] = None
    target_file: Optional[str] = None
    confidence: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        d["operation"] = self.operation.value
        return d


@dataclass
class CodeEditResult:
    """Comprehensive result of a code editing action."""
    success: bool
    tier: EditTier
    operation: str
    original_code: str
    modified_code: str
    diff: str
    modified_lines: List[int] = field(default_factory=list)
    llm_calls_made: int = 0
    tokens_consumed: int = 0
    baseline_tokens: int = 0
    tokens_saved: int = 0
    token_savings_pct: float = 0.0
    syntax_valid: bool = True
    validation_message: str = "Valid syntax"
    latency_ms: float = 0.0
    edit_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value if isinstance(self.tier, EditTier) else str(self.tier)
        return d
