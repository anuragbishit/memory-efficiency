"""
Rule-Based Text Compression and Context-Aware Stopword Removal Engine (Topic 4).
Applies syntactic redundancy stripping, pleonasm compaction, and whitelisted
stopword filtering without mutating question semantics, code blocks, or constraints.
"""

import re
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass, asdict
from core.compression.rules import SEMANTIC_ANCHORS, LOW_INFO_FILLERS

# Pleonasm and Wordy Construction Compactor:
# Maps bloated, multi-word English idioms to their atomic semantic equivalent
WORDY_PHRASE_MAPPINGS: List[Tuple[re.Pattern, str]] = [
    # 1. Code & Technical Query Request Frames (Prunes imperative prompt framing)
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?write\s+(?:a\s+)?code\s+in\s+([a-zA-Z0-9+#]+)\s+to\b", re.IGNORECASE), r"write \1 code to"),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?write\s+(?:a\s+)?(program|script|function|algorithm)\s+in\s+([a-zA-Z0-9+#]+)\s+to\b", re.IGNORECASE), r"write \2 \1 to"),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?write\s+code\s+in\s+([a-zA-Z0-9+#]+)\s+to\b", re.IGNORECASE), r"write \1 code to"),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?write\s+a\s+(code|program|script|function|algorithm)\s+to\b", re.IGNORECASE), r"write \1 to"),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?write\s+a\s+(code|program|script|function|algorithm)\s+for\b", re.IGNORECASE), r"write \1 for"),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:give|provide)\s+(?:me\s+)?(?:a\s+)?(?:code|program|script)\s+in\s+([a-zA-Z0-9+#]+)\s+to\b", re.IGNORECASE), r"write \1 code to"),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:give|provide)\s+(?:me\s+)?(?:a\s+)?(?:code|program|script)\s+(?:for|to)\b", re.IGNORECASE), "code for "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?provide\s+(?:a\s+)?solution\s+(?:for|to)\b", re.IGNORECASE), "solve "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:create|generate|develop)\s+a\s+(code|program|script|function)\s+to\b", re.IGNORECASE), r"create \1 to"),
    (re.compile(r"\bhow\s+(?:can\s+i|do\s+i)\s+write\s+(?:a\s+)?code\s+in\s+([a-zA-Z0-9+#]+)\s+to\b", re.IGNORECASE), r"how to code \1 to"),
    (re.compile(r"\bhow\s+(?:can\s+i|do\s+i)\s+write\s+(?:a\s+)?code\s+to\b", re.IGNORECASE), "how to code to "),
    (re.compile(r"\bhow\s+(?:can\s+i|do\s+i|should\s+i)\s+\b", re.IGNORECASE), "how to "),

    # 2. Action Verb + Indefinite Determiner Compaction (Safely eliminates non-semantic articles)
    (re.compile(r"\b(write|create|implement|build|print|display|develop|generate|design|solve|reverse|sort|traverse|compute|calculate)\s+an?\s+", re.IGNORECASE), r"\1 "),

    # 3. Pleonasms & Verbose Connectors
    (re.compile(r"\band\s+also\b", re.IGNORECASE), "and"),
    (re.compile(r"\bas\s+well\s+as\b", re.IGNORECASE), "and"),
    (re.compile(r"\balong\s+with\b", re.IGNORECASE), "and"),
    (re.compile(r"\bin\s+addition\s+to\b", re.IGNORECASE), "and"),
    (re.compile(r"\btogether\s+with\b", re.IGNORECASE), "with"),
    (re.compile(r"\bsuch\s+as\s+for\s+example\b", re.IGNORECASE), "such as"),
    (re.compile(r"\bfor\s+example\s+such\s+as\b", re.IGNORECASE), "for example"),
    (re.compile(r"\bdue\s+to\s+the\s+reason\s+that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bfor\s+the\s+reason\s+that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bin\s+order\s+that\b", re.IGNORECASE), "so that"),
    (re.compile(r"\bin\s+order\s+to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bso\s+as\s+to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bwith\s+the\s+aim\s+of\b", re.IGNORECASE), "to"),
    (re.compile(r"\bwith\s+the\s+intention\s+of\b", re.IGNORECASE), "to"),
    (re.compile(r"\bthe\s+question\s+as\s+to\s+whether\b", re.IGNORECASE), "whether"),
    (re.compile(r"\bas\s+to\s+whether\b", re.IGNORECASE), "whether"),
    (re.compile(r"\bregardless\s+of\s+the\s+fact\s+that\b", re.IGNORECASE), "although"),
    (re.compile(r"\bin\s+the\s+case\s+that\b", re.IGNORECASE), "if"),
    (re.compile(r"\bunder\s+the\s+condition\s+that\b", re.IGNORECASE), "if"),
    (re.compile(r"\bon\s+the\s+assumption\s+that\b", re.IGNORECASE), "assuming"),
    (re.compile(r"\bin\s+light\s+of\s+the\s+fact\s+that\b", re.IGNORECASE), "since"),

    # 4. Empty Nominalizations & Abstractions
    (re.compile(r"\bthe\s+concept\s+of\b", re.IGNORECASE), ""),
    (re.compile(r"\bthe\s+process\s+of\b", re.IGNORECASE), ""),
    (re.compile(r"\bthe\s+notion\s+of\b", re.IGNORECASE), ""),
    (re.compile(r"\bthe\s+meaning\s+of\b", re.IGNORECASE), ""),
    (re.compile(r"\bthe\s+topic\s+of\b", re.IGNORECASE), ""),
    (re.compile(r"\bthe\s+subject\s+of\b", re.IGNORECASE), ""),
    (re.compile(r"\bwhat\s+(?:is|are)\s+the\s+differences?\s+between\b", re.IGNORECASE), "difference between"),
    (re.compile(r"\bwhat\s+is\s+the\s+definition\s+of\b", re.IGNORECASE), "define"),
    (re.compile(r"\bthe\s+difference\s+between\b", re.IGNORECASE), "difference between"),
    (re.compile(r"\bis\s+defined\s+as\b", re.IGNORECASE), "is"),
    (re.compile(r"\bcan\s+be\s+defined\s+as\b", re.IGNORECASE), "is"),
    (re.compile(r"\bis\s+referred\s+to\s+as\b", re.IGNORECASE), "is"),
    (re.compile(r"\bis\s+known\s+as\b", re.IGNORECASE), "is"),
    (re.compile(r"\bplays\s+a\s+role\s+in\b", re.IGNORECASE), "affects"),
    (re.compile(r"\bis\s+responsible\s+for\b", re.IGNORECASE), "handles"),
    (re.compile(r"\bpertaining\s+to\b", re.IGNORECASE), "regarding"),
    (re.compile(r"\bin\s+relation\s+to\b", re.IGNORECASE), "regarding"),
    (re.compile(r"\bin\s+connection\s+with\b", re.IGNORECASE), "regarding"),
    (re.compile(r"\bwith\s+respect\s+to\b", re.IGNORECASE), "regarding"),
    (re.compile(r"\bin\s+terms\s+of\b", re.IGNORECASE), "regarding"),
    (re.compile(r"\bin\s+the\s+context\s+of\b", re.IGNORECASE), "in"),
    (re.compile(r"\ba\s+variety\s+of\b", re.IGNORECASE), "various"),
    (re.compile(r"\ba\s+wide\s+range\s+of\b", re.IGNORECASE), "various"),
    (re.compile(r"\ball\s+of\s+the\b", re.IGNORECASE), "all"),
    (re.compile(r"\beach\s+and\s+every\b", re.IGNORECASE), "each"),
    (re.compile(r"\bone\s+and\s+the\s+same\b", re.IGNORECASE), "the same"),
    (re.compile(r"\bfirst\s+and\s+foremost\b", re.IGNORECASE), "first"),
    (re.compile(r"\bany\s+and\s+all\b", re.IGNORECASE), "all"),
    (re.compile(r"\b(?:in\s+a\s+)?step\s+by\s+step(?:\s+manner)?\b", re.IGNORECASE), "step-by-step"),
    (re.compile(r"\bin\s+a\s+detailed\s+(?:manner|way)\b", re.IGNORECASE), "in detail"),
    (re.compile(r"\bwith\s+an?\s+example\b", re.IGNORECASE), "with example"),

    # 5. Explanations & Inquiries Framing
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?tell\s+me\s+about\b", re.IGNORECASE), "explain "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?tell\s+me\s+how\s+to\b", re.IGNORECASE), "how to "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?tell\s+me\s+what\s+is\b", re.IGNORECASE), "what is "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?tell\s+me\s+why\b", re.IGNORECASE), "why "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?tell\s+me\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?show\s+me\s+how\s+to\b", re.IGNORECASE), "how to "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:give|show)\s+me\s+an?\s+example\s+of\b", re.IGNORECASE), "example of "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?provide\s+an?\s+example\s+of\b", re.IGNORECASE), "example of "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?explain\s+to\s+me\s+what\s+is\b", re.IGNORECASE), "what is "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?explain\s+to\s+me\b", re.IGNORECASE), "explain "),
    (re.compile(r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?explain\s+about\b", re.IGNORECASE), "explain "),
    (re.compile(r"\bi\s+(?:want|would\s+like)\s+to\s+know\s+what\s+is\b", re.IGNORECASE), "what is "),
    (re.compile(r"\bi\s+(?:want|would\s+like)\s+to\s+know\s+how\s+to\b", re.IGNORECASE), "how to "),
    (re.compile(r"\bi\s+(?:want|would\s+like)\s+to\s+know\s+about\b", re.IGNORECASE), "explain "),
    (re.compile(r"\bi\s+(?:want|would\s+like)\s+to\s+know\b", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+(?:want|would\s+like)\s+to\s+learn\s+how\s+to\b", re.IGNORECASE), "how to "),
    (re.compile(r"\bi\s+(?:want|would\s+like)\s+to\s+learn\s+about\b", re.IGNORECASE), "explain "),
    (re.compile(r"\bi\s+(?:want|would\s+like)\s+to\s+learn\b", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+am\s+trying\s+to\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:can|could)\s+you\s+explain\b", re.IGNORECASE), "explain"),
    (re.compile(r"\bgive\s+me\s+some\s+information\s+about\b", re.IGNORECASE), "explain "),
    (re.compile(r"\bgive\s+me\s+details\s+about\b", re.IGNORECASE), "explain "),
    (re.compile(r"\bgive\s+me\s+an\s+overview\s+of\b", re.IGNORECASE), "overview of "),

    # 6. Directives & Politeness
    (re.compile(r"\bplease\s+make\s+sure\s+(?:that\s+)?\b", re.IGNORECASE), "ensure "),
    (re.compile(r"\bmake\s+sure\s+(?:that\s+)?\b", re.IGNORECASE), "ensure "),
    (re.compile(r"\bbe\s+sure\s+to\b", re.IGNORECASE), "ensure to "),
    (re.compile(r"\bkeep\s+in\s+mind\s+that\b", re.IGNORECASE), "note: "),
    (re.compile(r"\bdon't\s+forget\s+to\b", re.IGNORECASE), "ensure to "),
    (re.compile(r"\byou\s+need\s+to\b", re.IGNORECASE), "must "),
    (re.compile(r"\byou\s+have\s+to\b", re.IGNORECASE), "must "),

    # 7. Original Verbose verb phrases
    (re.compile(r"\bmake\s+an\s+attempt\s+to\b", re.IGNORECASE), "attempt to"),
    (re.compile(r"\btake\s+into\s+consideration\b", re.IGNORECASE), "consider"),
    (re.compile(r"\bgive\s+consideration\s+to\b", re.IGNORECASE), "consider"),
    (re.compile(r"\bconduct\s+an\s+investigation\s+(of|on)\b", re.IGNORECASE), "investigate"),
    (re.compile(r"\bperform\s+an\s+analysis\s+(of|on)\b", re.IGNORECASE), "analyze"),
    (re.compile(r"\bprovide\s+an\s+explanation\s+(of|for)\b", re.IGNORECASE), "explain"),
    (re.compile(r"\bhave\s+a\s+discussion\s+about\b", re.IGNORECASE), "discuss"),
    (re.compile(r"\bcarry\s+out\s+an\s+evaluation\s+of\b", re.IGNORECASE), "evaluate"),
    (re.compile(r"\bmake\s+a\s+decision\b", re.IGNORECASE), "decide"),
    (re.compile(r"\bhas\s+the\s+ability\s+to\b", re.IGNORECASE), "can"),
    (re.compile(r"\bis\s+able\s+to\b", re.IGNORECASE), "can"),

    # 8. Original Verbose transition and constraint phrases
    (re.compile(r"\bit\s+is\s+(crucial|essential|mandatory|necessary)\s+that\b", re.IGNORECASE), "must"),
    (re.compile(r"\bit\s+is\s+required\s+to\b", re.IGNORECASE), "must"),
    (re.compile(r"\bin\s+spite\s+of\s+the\s+fact\s+that\b", re.IGNORECASE), "although"),
    (re.compile(r"\bas\s+a\s+consequence\s+of\b", re.IGNORECASE), "because"),
    (re.compile(r"\bwith\s+the\s+exception\s+of\b", re.IGNORECASE), "except"),
    (re.compile(r"\bprior\s+to\s+the\s+start\s+of\b", re.IGNORECASE), "before"),
    (re.compile(r"\bprior\s+to\b", re.IGNORECASE), "before"),
    (re.compile(r"\bsubsequent\s+to\b", re.IGNORECASE), "after"),
    (re.compile(r"\bin\s+the\s+event\s+that\b", re.IGNORECASE), "if"),
    (re.compile(r"\bat\s+the\s+present\s+time\b", re.IGNORECASE), "now"),
    (re.compile(r"\bin\s+the\s+near\s+future\b", re.IGNORECASE), "soon"),
    (re.compile(r"\bfor\s+the\s+purpose\s+of\b", re.IGNORECASE), "for"),
    (re.compile(r"\ba\s+large\s+number\s+of\b", re.IGNORECASE), "many"),
    (re.compile(r"\ba\s+small\s+number\s+of\b", re.IGNORECASE), "few"),

    # 9. Original Tautologies / Pleonasms (redundant paired adjectives)
    (re.compile(r"\badvance\s+planning\b", re.IGNORECASE), "planning"),
    (re.compile(r"\bend\s+result\b", re.IGNORECASE), "result"),
    (re.compile(r"\bfinal\s+outcome\b", re.IGNORECASE), "outcome"),
    (re.compile(r"\bpast\s+history\b", re.IGNORECASE), "history"),
    (re.compile(r"\bsummarize\s+briefly\b", re.IGNORECASE), "summarize"),
    (re.compile(r"\bcompletely\s+eliminate\b", re.IGNORECASE), "eliminate"),
]

# Strict Question & Intent Whitelist (Must NEVER be pruned like standard stopwords)
INTENT_QUESTION_MARKERS: Set[str] = {
    "why", "how", "what", "which", "where", "when", "who", "whom", "whose"
}

# Non-essential functional stopwords safe to prune in AGGRESSIVE mode
PRUNABLE_FUNCTION_STOPWORDS: Set[str] = {
    "the", "a", "an", "very", "extremely", "basically", "actually",
    "literally", "honestly", "really", "totally", "simply", "just",
    "somewhat", "perhaps", "maybe", "obviously", "clearly", "its", "our", "their"
}


@dataclass
class RuleCompressionReport:
    """Report detailing phrase compaction and context-aware stopword removal."""
    original_text: str
    compressed_text: str
    phrases_compacted: int
    stopwords_pruned: int
    rules_applied: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleBasedTextCompressor:
    """
    Rule-Based Text Compressor (Topic 4).
    Deterministically transforms wordy phrases into compact canonical forms
    and prunes low-entropy stopwords with semantic anchor guarding.
    """

    def __init__(self):
        self.mappings = WORDY_PHRASE_MAPPINGS

    def compact_wordy_phrases(self, text: str) -> Tuple[str, int, List[str]]:
        """Replaces verbose idioms and pleonasms with atomic keywords."""
        compacted_count = 0
        applied: List[str] = []

        for pattern, replacement in self.mappings:
            matches = list(pattern.finditer(text))
            if matches:
                for match in matches:
                    matched_str = match.group(0).strip()
                    substituted = pattern.sub(replacement, match.group(0)).strip()
                    compacted_count += 1
                    if substituted:
                        applied.append(f"Compacted '{matched_str}' -> '{substituted}'")
                    else:
                        applied.append(f"Pruned redundant phrase '{matched_str}'")
                text = pattern.sub(replacement, text)

        return text, compacted_count, applied

    def prune_context_aware_stopwords(
        self,
        text: str,
        aggressive: bool = False
    ) -> Tuple[str, int, List[str]]:
        """
        Prunes filler words and low-entropy function words while strictly preserving:
          1. Question words (why, how, what)
          2. Negations (not, never, without)
          3. Boundary constraints (must, only, strict)
        """
        words = text.split()
        pruned_words: List[str] = []
        removed_count = 0
        applied: List[str] = []

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word).lower()

            # ALWAYS preserve question intent markers
            if clean_word in INTENT_QUESTION_MARKERS:
                pruned_words.append(word)
                continue

            # ALWAYS preserve semantic anchors (negations, constraints, bounds)
            if clean_word in SEMANTIC_ANCHORS:
                pruned_words.append(word)
                continue

            # Prune conversational fillers
            if clean_word in LOW_INFO_FILLERS:
                removed_count += 1
                applied.append(f"Pruned filler '{clean_word}'")
                continue

            # In aggressive mode, prune non-essential determiners (the, a, an)
            if aggressive and clean_word in PRUNABLE_FUNCTION_STOPWORDS:
                removed_count += 1
                applied.append(f"Pruned stopword '{clean_word}'")
                continue

            pruned_words.append(word)

        return " ".join(pruned_words), removed_count, applied

    def compress(self, text: str, aggressive: bool = False) -> RuleCompressionReport:
        """Runs the combined rule-based compaction and context-aware stopword pass."""
        compacted_text, n_phrases, rules_p = self.compact_wordy_phrases(text)
        final_text, n_words, rules_w = self.prune_context_aware_stopwords(
            compacted_text, aggressive=aggressive
        )

        return RuleCompressionReport(
            original_text=text,
            compressed_text=final_text,
            phrases_compacted=n_phrases,
            stopwords_pruned=n_words,
            rules_applied=rules_p + rules_w,
        )
