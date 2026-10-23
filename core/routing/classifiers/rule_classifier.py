"""
Deterministic Rule-Based Intent Classifier.
Fast-path expert heuristic engine providing sub-0.1ms immediate triage for distinct prompt patterns.
"""

import re
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class RuleClassificationResult:
    """Output from the deterministic rule classifier."""
    matched_intent: Optional[str]
    confidence: float
    matched_rule_description: Optional[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleIntentClassifier:
    """
    Expert Heuristic Rule Engine.
    Executes fast pattern matching for unambiguous signals like code blocks,
    exception traces, architectural keywords, and trivial lookups.
    """

    EXPLICIT_RULES: List[Tuple[re.Pattern, str, float, str]] = [
        # 1. Error / Exception debugging rules
        (re.compile(r"\b(traceback|typeerror|valueerror|indexerror|keyerror|nullpointerexception|oomkilled|syntaxerror|exception\b|debug\s+why|fix\s+this\s+error)\b", re.IGNORECASE), "DEBUGGING", 0.95, "Explicit error stack trace or debug trigger detected"),

        # 2. Code blocks present
        (re.compile(r"```[a-zA-Z0-9_-]*\n[\s\S]*?```"), "CODE_GENERATION", 0.92, "Fenced programming code block detected"),

        # 3. Direct code generation imperatives
        (re.compile(r"\b(write\s+a\s+(python|c\+\+|java|javascript|sql|react)\s+(function|script|code|algorithm)|implement\s+(dijkstra|quicksort|lru|linked\s+list|binary\s+search))\b", re.IGNORECASE), "CODE_GENERATION", 0.95, "Direct programming implementation imperative"),

        # 4. System architecture triggers
        (re.compile(r"\b(architect|distributed\s+(cache|system|database)|microservices|consensus\s+protocol|event-driven|graphrag\s+pipeline)\b", re.IGNORECASE), "SYSTEM_ARCHITECTURE", 0.90, "Enterprise system architecture domain terms detected"),

        # 5. Explanatory & Causal triggers
        (re.compile(r"\b(explain\s+(why|how|the\s+difference)|how\s+does\s+[a-zA-Z\s]+\s+work|why\s+does\s+[a-zA-Z\s]+\s+(eliminate|cause|work))\b", re.IGNORECASE), "EXPLANATION", 0.88, "Conceptual explanation / causal query structure"),

        # 6. Simple greetings and trivial lookups
        (re.compile(r"^(hello|hi|hey|good\s+(morning|afternoon|evening))\s*(there|everyone|all)?[\s,!.]*$", re.IGNORECASE), "FACTUAL_LOOKUP", 0.98, "Conversational greeting fast-path"),
        (re.compile(r"\b(what\s+is\s+the\s+capital\s+of|who\s+invented|what\s+is\s+the\s+speed\s+of\s+light)\b", re.IGNORECASE), "FACTUAL_LOOKUP", 0.95, "Simple factual lookup pattern"),
    ]

    def predict(self, text: str) -> RuleClassificationResult:
        """Evaluates fast-path heuristics against the prompt."""
        start = time.perf_counter()

        for pattern, intent, conf, desc in self.EXPLICIT_RULES:
            if pattern.search(text):
                latency = round((time.perf_counter() - start) * 1000, 3)
                return RuleClassificationResult(
                    matched_intent=intent,
                    confidence=conf,
                    matched_rule_description=desc,
                    latency_ms=latency,
                )

        latency = round((time.perf_counter() - start) * 1000, 3)
        return RuleClassificationResult(
            matched_intent=None,
            confidence=0.0,
            matched_rule_description=None,
            latency_ms=latency,
        )
