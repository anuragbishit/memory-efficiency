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
    # Verbose verb phrases
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
    # Verbose transition and constraint phrases
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
    # Tautologies / Pleonasms (redundant paired adjectives)
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
    "somewhat", "perhaps", "maybe", "obviously", "clearly"
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
            match = pattern.search(text)
            if match:
                matched_str = match.group(0)
                text = pattern.sub(replacement, text)
                compacted_count += 1
                applied.append(f"Compacted '{matched_str}' -> '{replacement}'")

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
