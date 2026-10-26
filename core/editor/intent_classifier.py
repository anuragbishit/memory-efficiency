"""
Intent Classifier for Code Edits.
Categorizes user edit requests into:
  - DIRECT_EDIT (0 LLM Calls, deterministic AST/Regex execution)
  - TARGETED_RETRIEVAL (1 LLM Call, minimal localized context extraction)
  - DEEP_REASONING (Multi-step LLM reasoning for ambiguous bugs/architectural refactoring)
"""

import re
from typing import Optional, Dict, Any, Tuple
from core.editor.models import EditTier, EditOperation, ParsedInstruction


class EditIntentClassifier:
    """
    Evaluates natural language instructions to determine whether an AI model
    is actually needed or if ordinary deterministic code execution suffices.
    """

    # Patterns for Direct Assignment edits (e.g. "Change a = 10 to a = 12", "Update port from 8000 to 8080")
    ASSIGNMENT_PATTERNS = [
        # "Change a = 10 to a = 12" or "Change a = 10 to 12"
        re.compile(
            r"^(?:please\s+)?(?:change|update|set|modify)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+(?:[^\n]*?[^\s])?)\s+to\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?([^\s]+(?:[^\n]*?[^\s])?)$",
            re.IGNORECASE
        ),
        # "Change port from 8000 to 8080"
        re.compile(
            r"^(?:please\s+)?(?:change|update|modify)\s+([A-Za-z_][A-Za-z0-9_]*)\s+from\s+([^\s]+(?:[^\n]*?[^\s])?)\s+to\s+([^\s]+(?:[^\n]*?[^\s])?)$",
            re.IGNORECASE
        ),
        # "Update MAX_RETRIES to 5"
        re.compile(
            r"^(?:please\s+)?(?:update|set|change)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:to|=)\s+([^\s]+(?:[^\n]*?[^\s])?)$",
            re.IGNORECASE
        ),
    ]

    # Patterns for Literal replacement (e.g., "Replace 'localhost' with 'api.production.com'")
    LITERAL_REPLACE_PATTERNS = [
        re.compile(
            r"^(?:please\s+)?replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]$",
            re.IGNORECASE
        ),
        re.compile(
            r"^(?:please\s+)?replace\s+(\S+)\s+with\s+(\S+)$",
            re.IGNORECASE
        ),
        re.compile(
            r"^(?:please\s+)?substitute\s+['\"]?(.+?)['\"]?\s+for\s+['\"]?(.+?)['\"]?$",
            re.IGNORECASE
        ),
    ]

    # Patterns for Variable / Symbol Rename (e.g., "Rename old_count to total_items")
    RENAME_PATTERNS = [
        re.compile(
            r"^(?:please\s+)?rename\s+(?:variable\s+|symbol\s+|identifier\s+)?([A-Za-z_][A-Za-z0-9_]*)\s+to\s+([A-Za-z_][A-Za-z0-9_]*)$",
            re.IGNORECASE
        ),
    ]

    # Targeted retrieval keywords (identifying a specific function/class/block)
    TARGETED_FUNCTION_PATTERNS = [
        re.compile(r"(?:in|inside|for)\s+(?:function|def|method)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
        re.compile(r"(?:update|optimize|change|refactor)\s+(?:function|method)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
        re.compile(r"(?:in|inside)\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE),
    ]

    # Deep reasoning keywords (multi-step bug fixing, architecture, unknown runtime errors)
    DEEP_REASONING_KEYWORDS = [
        "fix bug", "runtime error", "exception", "traceback", "crash",
        "why is this failing", "debug", "architect", "multi-file",
        "investigate", "memory leak", "race condition", "deadlock"
    ]

    def classify(self, instruction: str, code_context: Optional[str] = None) -> ParsedInstruction:
        """
        Classifies an edit instruction and returns structured metadata.
        """
        cleaned = instruction.strip()

        # 1. Check for Variable Rename (0 LLM Calls)
        for pat in self.RENAME_PATTERNS:
            m = pat.match(cleaned)
            if m:
                old_var, new_var = m.group(1), m.group(2)
                return ParsedInstruction(
                    raw_instruction=cleaned,
                    tier=EditTier.DIRECT_EDIT,
                    operation=EditOperation.VARIABLE_RENAME,
                    target_variable=old_var,
                    old_value=old_var,
                    new_value=new_var,
                    confidence=0.98,
                    reasoning=f"Exact symbol rename '{old_var}' -> '{new_var}'. Zero LLM tokens required."
                )

        # 2. Check for Assignment Value Change (0 LLM Calls)
        for i, pat in enumerate(self.ASSIGNMENT_PATTERNS):
            m = pat.match(cleaned)
            if m:
                groups = m.groups()
                if len(groups) == 3:
                    target_var, old_val, new_val = groups[0], groups[1].strip("'\""), groups[2].strip("'\"")
                elif len(groups) == 2:
                    target_var, new_val = groups[0], groups[1].strip("'\"")
                    old_val = None
                else:
                    continue

                return ParsedInstruction(
                    raw_instruction=cleaned,
                    tier=EditTier.DIRECT_EDIT,
                    operation=EditOperation.ASSIGNMENT_CHANGE,
                    target_variable=target_var,
                    old_value=old_val,
                    new_value=new_val,
                    confidence=0.99,
                    reasoning=f"Direct assignment modification for variable '{target_var}'. Zero LLM tokens required."
                )

        # 3. Check for Literal String / Token Replacement (0 LLM Calls)
        for pat in self.LITERAL_REPLACE_PATTERNS:
            m = pat.match(cleaned)
            if m:
                old_val, new_val = m.group(1), m.group(2)
                return ParsedInstruction(
                    raw_instruction=cleaned,
                    tier=EditTier.DIRECT_EDIT,
                    operation=EditOperation.LITERAL_REPLACE,
                    old_value=old_val,
                    new_value=new_val,
                    confidence=0.97,
                    reasoning=f"Direct literal replacement '{old_val}' -> '{new_val}'. Zero LLM tokens required."
                )

        # 4. Check for Deep Reasoning triggers (Runtime bugs, architectural investigations)
        lowered = cleaned.lower()
        if any(keyword in lowered for keyword in self.DEEP_REASONING_KEYWORDS):
            return ParsedInstruction(
                raw_instruction=cleaned,
                tier=EditTier.DEEP_REASONING,
                operation=EditOperation.GENERAL_EDIT,
                confidence=0.90,
                reasoning="Instruction requires diagnostic deduction, root-cause analysis, or multi-file context."
            )

        # 5. Check for Targeted Retrieval (Scoped function or class edit)
        for pat in self.TARGETED_FUNCTION_PATTERNS:
            m = pat.search(cleaned)
            if m:
                target_func = m.group(1)
                return ParsedInstruction(
                    raw_instruction=cleaned,
                    tier=EditTier.TARGETED_RETRIEVAL,
                    operation=EditOperation.FUNCTION_EDIT,
                    target_function=target_func,
                    confidence=0.92,
                    reasoning=f"Scoped function edit for '{target_func}'. Only the target function block needs retrieval."
                )

        # 6. Default Fallback based on text length & heuristics
        if len(cleaned.split()) <= 8 and ("to" in lowered or "=" in lowered):
            # Ambiguous short edit -> Targeted retrieval
            return ParsedInstruction(
                raw_instruction=cleaned,
                tier=EditTier.TARGETED_RETRIEVAL,
                operation=EditOperation.GENERAL_EDIT,
                confidence=0.75,
                reasoning="Short localized edit instruction. Retrieve only the relevant code section."
            )

        return ParsedInstruction(
            raw_instruction=cleaned,
            tier=EditTier.DEEP_REASONING,
            operation=EditOperation.GENERAL_EDIT,
            confidence=0.80,
            reasoning="Open-ended instruction requiring deep reasoning or broader context."
        )
