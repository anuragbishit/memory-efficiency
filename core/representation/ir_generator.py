"""
Prompt Intermediate Representation (IR) Generator (Topic 6).
Transforms unstructured conversational prompts into standardized, high-density IR formats.
Bypasses tokenizer subword bloat, eliminates context bleed, and strips conversational entropy.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from core.tokenization.tokenizer import SubwordTokenizer
from core.representation.ir import PromptIR, IRFormat, TaskIntentType


class PromptIRGenerator:
    """
    Translates raw/compressed natural language prompts into
    dense, structured Prompt Intermediate Representations (IR).
    """

    INTENT_ACTION_MAP = [
        (re.compile(r"\b(optimize|speed\s+up|refactor|improve)\b", re.IGNORECASE), TaskIntentType.OPTIMIZE, "optimize"),
        (re.compile(r"\b(architect|design|build\s+architecture)\b", re.IGNORECASE), TaskIntentType.ARCHITECT, "architect"),
        (re.compile(r"\b(debug|fix|resolve\s+issue|troubleshoot)\b", re.IGNORECASE), TaskIntentType.DEBUG, "debug"),
        (re.compile(r"\b(explain|why|how\s+does|what\s+is|describe)\b", re.IGNORECASE), TaskIntentType.EXPLAIN, "explain"),
        (re.compile(r"\b(analyze|compare|evaluate|contrast)\b", re.IGNORECASE), TaskIntentType.ANALYZE, "analyze"),
        (re.compile(r"\b(generate|write|implement|create)\b", re.IGNORECASE), TaskIntentType.GENERATE, "implement"),
        (re.compile(r"\b(what\s+is\s+the\s+capital|where\s+is|lookup|find)\b", re.IGNORECASE), TaskIntentType.LOOKUP, "lookup"),
    ]

    FENCED_CODE_PATTERN = re.compile(r"```([a-zA-Z0-9_-]*)\n([\s\S]*?)```")

    CONSTRAINT_EXTRACTOR = re.compile(
        r"\b((?:must\s+(?:not\s+)?[^,.;]+|should\s+(?:not\s+)?[^,.;]+|strictly\s+[^,.;]+|without\s+using\s+[^,.;]+|cannot\s+[^,.;]+|only\s+use\s+[^,.;]+|o\([^)]+\)\s+[^,.;]*|time\s+complexity\s+[^,.;]+|space\s+complexity\s+[^,.;]+))\b",
        re.IGNORECASE
    )

    def __init__(self, tokenizer: SubwordTokenizer | None = None):
        self.tokenizer = tokenizer or SubwordTokenizer()

    def _extract_code_snippets(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Extracts code blocks and returns cleaned text alongside code dictionaries."""
        snippets = []

        def replacer(match: re.Match) -> str:
            lang = match.group(1).strip() or "generic"
            code = match.group(2).strip()
            snippets.append({"lang": lang, "code": code})
            return f"__CODE_REF_{len(snippets)-1}__"

        cleaned_text = self.FENCED_CODE_PATTERN.sub(replacer, text)
        return cleaned_text, snippets

    def _extract_constraints(self, text: str) -> List[str]:
        """Identifies explicit boundary conditions and constraints."""
        matches = self.CONSTRAINT_EXTRACTOR.findall(text)
        cleaned_constraints = []
        for m in matches:
            cleaned = re.sub(r"\s+", " ", m).strip(" ,;.")
            if len(cleaned) > 4 and cleaned not in cleaned_constraints:
                cleaned_constraints.append(cleaned)
        return cleaned_constraints

    def _detect_intent_and_action(self, text: str) -> Tuple[TaskIntentType, str]:
        """Classifies the primary atomic task and action verb."""
        for pattern, intent, action in self.INTENT_ACTION_MAP:
            if pattern.search(text):
                return intent, action
        return TaskIntentType.GENERAL, "process"

    def _extract_target_topic(self, text: str, action: str) -> str:
        """Extracts the core conceptual topic or target entity being operated on."""
        # Strip code references and punctuation
        sanitized = re.sub(r"__CODE_REF_\d+__", "", text)
        sanitized = re.sub(r"[,.?!;:]+", " ", sanitized)

        # Look for phrases following the action verb
        action_match = re.search(rf"\b{action}\b\s+([^,.;]+)", sanitized, re.IGNORECASE)
        if action_match:
            candidate = action_match.group(1).strip()
            # Truncate to first 6 meaningful words
            words = [w for w in candidate.split() if len(w) > 1][:6]
            return " ".join(words)

        # Fallback to general noun chunk preview
        words = [w for w in sanitized.split() if len(w) > 2][:6]
        return " ".join(words) if words else "input request"

    def generate_ir(
        self,
        prompt_text: str,
        encoding_name: str = "cl100k_base",
        format_type: IRFormat = IRFormat.TAGGED_KV
    ) -> PromptIR:
        """
        Parses input prompt text and compiles it into a standardized PromptIR instance.
        Calculates exact token reduction compared to raw prompt.
        """
        if not prompt_text or not prompt_text.strip():
            return PromptIR(
                intent=TaskIntentType.GENERAL,
                action="none",
                target="empty",
                original_tokens=0,
                ir_tokens=0,
                token_savings_pct=0.0
            )

        raw_tokens = self.tokenizer.count_tokens(prompt_text, encoding_name)

        # Step 1: Extract code snippets
        prose_text, snippets = self._extract_code_snippets(prompt_text)

        # Step 2: Extract explicit constraints
        constraints = self._extract_constraints(prose_text)

        # Step 3: Detect Intent and Action Verb
        intent, action = self._detect_intent_and_action(prose_text)

        # Step 4: Extract Target Subject
        target = self._extract_target_topic(prose_text, action)

        # Step 5: Detect Context Anchors
        context_anchors = []
        if "graph" in prompt_text.lower():
            context_anchors.append("Graph Architecture")
        if "thread-safe" in prompt_text.lower() or "concurrency" in prompt_text.lower():
            context_anchors.append("Concurrency")
        if "memory" in prompt_text.lower():
            context_anchors.append("Memory System")

        prompt_ir = PromptIR(
            intent=intent,
            action=action,
            target=target,
            constraints=constraints,
            code_snippets=snippets,
            context_anchors=context_anchors,
            output_format="code_and_explanation" if snippets else "structured_analysis",
            original_tokens=raw_tokens,
        )

        serialized_ir = prompt_ir.serialize(format_type)
        ir_tokens = self.tokenizer.count_tokens(serialized_ir, encoding_name)
        prompt_ir.ir_tokens = ir_tokens

        saved = max(0, raw_tokens - ir_tokens)
        prompt_ir.token_savings_pct = (
            round((saved / raw_tokens) * 100, 2) if raw_tokens > 0 else 0.0
        )

        return prompt_ir
