"""
Intermediate Representation (IR) Schema for Compact Prompts (Topic 6).
Defines language-neutral, token-minimized intermediate formats
(Tagged-KV, Compact-DSL, and Minimal-JSON) to eliminate conversational entropy.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
import json


class IRFormat(str, Enum):
    """Serialization formats for Prompt Intermediate Representation."""
    TAGGED_KV = "tagged_kv"      # [TAG: value] format (Optimal token density in BPE)
    COMPACT_DSL = "compact_dsl"  # ACTION(target) | REQ(...) pipeline format
    MINIMAL_JSON = "minimal_json"# Canonical minified JSON format


class TaskIntentType(str, Enum):
    """Categorized atomic task intent."""
    OPTIMIZE = "OPTIMIZE"
    EXPLAIN = "EXPLAIN"
    ARCHITECT = "ARCHITECT"
    DEBUG = "DEBUG"
    GENERATE = "GENERATE"
    ANALYZE = "ANALYZE"
    LOOKUP = "LOOKUP"
    GENERAL = "GENERAL"


@dataclass
class PromptIR:
    """
    Standardized Intermediate Representation of a Prompt Payload.
    Isolates task action, target entity, inputs, constraints, and requested format.
    """
    intent: TaskIntentType
    action: str
    target: str
    constraints: List[str] = field(default_factory=list)
    code_snippets: List[Dict[str, str]] = field(default_factory=list)  # [{"lang": "python", "code": "..."}]
    context_anchors: List[str] = field(default_factory=list)
    output_format: Optional[str] = None
    original_tokens: int = 0
    ir_tokens: int = 0
    token_savings_pct: float = 0.0

    def serialize(self, format_type: IRFormat = IRFormat.TAGGED_KV) -> str:
        """Serializes the IR into the requested minimal-token representation."""
        if format_type == IRFormat.TAGGED_KV:
            return self._serialize_tagged_kv()
        elif format_type == IRFormat.COMPACT_DSL:
            return self._serialize_compact_dsl()
        elif format_type == IRFormat.MINIMAL_JSON:
            return self._serialize_minimal_json()
        return self._serialize_tagged_kv()

    def _serialize_tagged_kv(self) -> str:
        """Emits compact [TAG: VALUE] representation (maximizes BPE token efficiency)."""
        lines = [
            f"[TASK: {self.intent.value} -> {self.action} {self.target}]".strip()
        ]

        if self.constraints:
            constraints_str = "; ".join(self.constraints)
            lines.append(f"[CONSTRAINTS: {constraints_str}]")

        if self.context_anchors:
            lines.append(f"[CONTEXT: {', '.join(self.context_anchors)}]")

        for snippet in self.code_snippets:
            lang = snippet.get("lang", "")
            code = snippet.get("code", "")
            lines.append(f"[INPUT_CODE: {lang}\n{code}\n]")

        if self.output_format:
            lines.append(f"[OUTPUT_FORMAT: {self.output_format}]")

        return "\n".join(lines)

    def _serialize_compact_dsl(self) -> str:
        """Emits minimal functional pipeline DSL: DO(target) | WITH(constraints)."""
        parts = [f"{self.intent.value}({self.target})"]
        if self.constraints:
            parts.append(f"CONSTRAIN({'; '.join(self.constraints)})")
        for snippet in self.code_snippets:
            parts.append(f"CODE({snippet.get('lang', '')}: {snippet.get('code', '').strip()})")
        return " | ".join(parts)

    def _serialize_minimal_json(self) -> str:
        """Emits minified, unspaced JSON."""
        payload = {
            "intent": self.intent.value,
            "action": self.action,
            "target": self.target,
            "constraints": self.constraints,
            "code": self.code_snippets,
            "context": self.context_anchors,
        }
        if self.output_format:
            payload["format"] = self.output_format
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["intent"] = self.intent.value
        return data
