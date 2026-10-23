"""
Comprehensive Prompt Feature Extraction Engine (Topic 7).
Extracts lexical, syntactic, structural, domain-complexity, cross-lingual,
and statistical entropy signals into structured representations and ML-ready vectors.
"""

import re
import math
import unicodedata
from enum import Enum
from typing import Dict, Any, List, Set, Tuple
from collections import Counter
from dataclasses import dataclass, asdict

# Analytical and multi-step reasoning triggers
REASONING_PATTERNS = [
    r"\bexplain\b",
    r"\bwhy\b",
    r"\bhow\s+(does|do|can|is)\b",
    r"\bstep[\s-]by[\s-]step\b",
    r"\bcompare\s+(and\s+contrast)?\b",
    r"\bversus\b|\bvs\.?\b",
    r"\bdifference\s+between\b",
    r"\btrade[\s-]?offs?\b",
    r"\bderive\b",
    r"\bmathematical(ly)?\b",
    r"\bprove\b",
    r"\broot[\s-]cause\b",
    r"\banalyze\b",
    r"\bevaluate\b",
    r"\barchitect(ure)?\b",
    r"\balgorithm(ic)?\b",
    r"\bmulti[\s-]hop\b",
    r"\breason(ing)?\b",
    r"\bdesign\s+pattern\b",
    r"\bconcurrency\b",
    r"\boptimize\s+for\b",
]

# Strict constraint indicators
CONSTRAINT_PATTERNS = [
    r"\bmust\s+(not|be|have)?\b",
    r"\bshould\s+(not)?\b",
    r"\bonly\s+use\b",
    r"\bdo\s+not\s+use\b",
    r"\bwithout\s+using\b",
    r"\bstrict(ly)?\b",
    r"\brequirement(s)?\b",
    r"\btime\s+complexity\b",
    r"\bspace\s+complexity\b",
    r"\bo\([^\)]+\)",
    r"\blimit(ed)?\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\bthread[\s-]safe\b",
]

# Programming language and code presence patterns
CODE_MARKERS = [
    r"```[a-zA-Z0-9_-]*",
    r"\bdef\s+[a-zA-Z_]\w*\s*\(",
    r"\bfunction\s+[a-zA-Z_]\w*\s*\(",
    r"\bclass\s+[a-zA-Z_]\w*",
    r"\bimport\s+[a-zA-Z_]\w*",
    r"\bfrom\s+[a-zA-Z_]\w*\s+import\b",
    r"\bpublic\s+static\s+void\b",
    r"\bconst\s+[a-zA-Z_]\w*\s*=",
    r"\blet\s+[a-zA-Z_]\w*\s*=",
    r"\bvar\s+[a-zA-Z_]\w*\s*=",
    r"[{}\[\];]{2,}",
    r"=>",
    r"\breturn\b",
]

# Imperative prompt command verbs
IMPERATIVE_VERBS: Set[str] = {
    "optimize", "write", "implement", "create", "build", "develop",
    "fix", "debug", "refactor", "explain", "analyze", "derive",
    "calculate", "evaluate", "compare", "find", "show", "summarize"
}

# Negation words
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "without", "none", "neither", "nor", "cannot", "can't", "don't"
}

# Politeness / Fluff tokens
FLUFF_WORDS: Set[str] = {
    "hello", "hi", "hey", "please", "kindly", "thanks", "thank",
    "wondering", "appreciate", "basically", "actually", "literally", "simply", "just"
}


class QuestionType(str, Enum):
    """Categorized interrogative question intent."""
    FACTUAL = "FACTUAL"          # What is, Who is, Where is
    PROCESS = "PROCESS"          # How to, How does
    CAUSAL = "CAUSAL"            # Why, Root cause
    COMPARATIVE = "COMPARATIVE"  # Compare, Difference between, Versus
    DIRECTIVE = "DIRECTIVE"      # Imperative instruction without question mark
    NONE = "NONE"


@dataclass
class PromptFeatures:
    """
    Comprehensive numerical and categorical feature vector representing a prompt.
    Captures lexical, syntactic, structural, domain-specific, and entropy signals.
    """
    # 1. Lexical & Statistical
    token_count: int
    char_count: int
    word_count: int
    char_to_token_ratio: float
    type_token_ratio: float       # Vocabulary diversity (unique / total words)
    sentence_count: int
    avg_sentence_length: float

    # 2. Syntactic & Intent
    question_type: str
    imperative_command_count: int
    has_code_block: bool
    code_presence_score: float
    detected_code_language: str

    # 3. Domain Complexity & Constraints
    reasoning_density_score: float
    constraint_density_score: float
    negation_count: int
    syntactic_complexity_score: float
    detected_reasons: List[str]
    detected_constraints: List[str]

    # 4. Cross-Lingual & Information Density
    primary_script: str
    fragmentation_ratio: float
    fluff_density_score: float
    shannon_entropy: float
    token_perplexity: float
    lexical_density_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_feature_vector(self) -> List[float]:
        """
        Emits a fixed-length 12-dimensional normalized float vector
        suitable for Scikit-Learn classifiers and ML intent routing.
        """
        return [
            min(1.0, self.token_count / 200.0),          # Length normalized
            self.type_token_ratio,                       # TTR
            min(1.0, self.char_to_token_ratio / 6.0),    # Char/token
            self.code_presence_score,                    # Code score
            self.reasoning_density_score,                 # Reasoning score
            self.constraint_density_score,                # Constraint score
            self.syntactic_complexity_score,              # Syntax score
            min(1.0, self.fragmentation_ratio / 4.0),     # Frag ratio
            min(1.0, self.shannon_entropy / 8.0),         # Entropy
            min(1.0, self.token_perplexity / 100.0),      # Perplexity
            self.lexical_density_pct / 100.0,             # Lexical density
            self.fluff_density_score,                     # Fluff density
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "token_count_norm",
            "type_token_ratio",
            "char_to_token_norm",
            "code_score",
            "reasoning_score",
            "constraint_score",
            "syntax_score",
            "frag_ratio_norm",
            "entropy_norm",
            "perplexity_norm",
            "lexical_density_norm",
            "fluff_score",
        ]


class FeatureExtractor:
    """
    Comprehensive Prompt Feature Extraction Engine (Topic 7).
    Computes 20+ multi-dimensional signals across lexical, structural,
    code, and information-theoretic properties in CPU memory (< 1 ms).
    """

    def __init__(self):
        self.reasoning_regexes = [re.compile(p, re.IGNORECASE) for p in REASONING_PATTERNS]
        self.constraint_regexes = [re.compile(p, re.IGNORECASE) for p in CONSTRAINT_PATTERNS]
        self.code_regexes = [re.compile(p, re.IGNORECASE) for p in CODE_MARKERS]

    def _classify_question_type(self, text: str) -> QuestionType:
        """Determines the interrogative classification of the prompt."""
        lower = text.lower().strip()
        if re.search(r"\b(compare|difference\s+between|versus|\bvs\b)\b", lower):
            return QuestionType.COMPARATIVE
        if re.search(r"\bwhy\b", lower) or "root cause" in lower:
            return QuestionType.CAUSAL
        if re.search(r"\bhow\s+(to|do|does|can|is)\b", lower):
            return QuestionType.PROCESS
        if re.search(r"\bwhat\s+(is|are)\b|\bwho\s+is\b|\bwhere\s+is\b", lower):
            return QuestionType.FACTUAL
        if any(lower.startswith(v) for v in IMPERATIVE_VERBS):
            return QuestionType.DIRECTIVE
        if "?" in text:
            return QuestionType.FACTUAL
        return QuestionType.NONE

    def _compute_entropy_and_perplexity(self, words: List[str]) -> Tuple[float, float]:
        """Calculates Shannon entropy and perplexity over tokenized words."""
        if not words:
            return 0.0, 1.0

        n = len(words)
        counts = Counter(words)
        entropy = 0.0
        for count in counts.values():
            p_x = count / n
            entropy -= p_x * math.log2(p_x)
        perplexity = math.pow(2, entropy)
        return round(entropy, 3), round(perplexity, 3)

    def extract(self, text: str, token_count: int) -> PromptFeatures:
        """
        Extracts the full multi-dimensional feature representation from the prompt.
        Backwards-compatible with all existing intent router attributes.
        """
        if not text or not text.strip():
            return PromptFeatures(
                token_count=0,
                char_count=0,
                word_count=0,
                char_to_token_ratio=0.0,
                type_token_ratio=0.0,
                sentence_count=0,
                avg_sentence_length=0.0,
                question_type=QuestionType.NONE.value,
                imperative_command_count=0,
                has_code_block=False,
                code_presence_score=0.0,
                detected_code_language="none",
                reasoning_density_score=0.0,
                constraint_density_score=0.0,
                negation_count=0,
                syntactic_complexity_score=0.0,
                primary_script="None",
                fragmentation_ratio=0.0,
                fluff_density_score=0.0,
                shannon_entropy=0.0,
                token_perplexity=1.0,
                lexical_density_pct=0.0,
                detected_reasons=[],
                detected_constraints=[],
            )

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        frag_ratio = round(token_count / word_count, 2) if word_count > 0 else 0.0
        char_to_token = round(char_count / token_count, 2) if token_count > 0 else 0.0

        # Vocabulary diversity (Type-Token Ratio)
        unique_words = len(set(w.lower() for w in words))
        ttr = round(unique_words / word_count, 3) if word_count > 0 else 0.0

        # Sentence metrics
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = max(1, len(sentences))
        avg_sentence_len = round(word_count / sentence_count, 2)

        # Script detection
        devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        cjk_chars = sum(1 for c in text if "\u4E00" <= c <= "\u9FFF")
        cyrillic_chars = sum(1 for c in text if "\u0400" <= c <= "\u04FF")

        if devanagari_chars > 3:
            primary_script = "Devanagari"
        elif arabic_chars > 3:
            primary_script = "Arabic"
        elif cjk_chars > 3:
            primary_script = "CJK"
        elif cyrillic_chars > 3:
            primary_script = "Cyrillic"
        else:
            primary_script = "Latin"

        # Code block & syntax detection
        has_code_block = "```" in text or "`" in text
        code_matches = sum(1 for rgx in self.code_regexes if rgx.search(text))
        code_score = min(1.0, (1.0 if has_code_block else 0.0) * 0.5 + (code_matches * 0.25))

        lang_match = re.search(r"```([a-zA-Z0-9_-]+)", text)
        detected_lang = lang_match.group(1).lower() if lang_match else ("python" if "def " in text or "import " in text else "none")

        # Question type & Imperative commands
        q_type = self._classify_question_type(text)
        imperatives = sum(1 for w in words if re.sub(r"[^\w]", "", w).lower() in IMPERATIVE_VERBS)

        # Reasoning density
        detected_reasons: List[str] = []
        for rgx in self.reasoning_regexes:
            match = rgx.search(text)
            if match:
                detected_reasons.append(match.group(0).lower())
        reasoning_score = min(1.0, len(detected_reasons) / 3.0)

        # Constraint density & Negations
        detected_constraints: List[str] = []
        for rgx in self.constraint_regexes:
            match = rgx.search(text)
            if match:
                detected_constraints.append(match.group(0).lower())
        constraint_score = min(1.0, len(detected_constraints) / 2.5)

        negation_count = sum(1 for w in words if re.sub(r"[^\w]", "", w).lower() in NEGATION_WORDS)

        # Syntactic complexity
        length_score = min(1.0, token_count / 150.0)
        sentence_complexity = min(1.0, avg_sentence_len / 25.0)
        syntactic_score = round(0.6 * length_score + 0.4 * sentence_complexity, 3)

        # Information density & fluff metrics
        fluff_count = sum(1 for w in words if re.sub(r"[^\w]", "", w).lower() in FLUFF_WORDS)
        fluff_score = round(min(1.0, fluff_count / max(1, word_count)), 3)

        content_words = [w for w in words if len(w) > 3 and re.sub(r"[^\w]", "", w).lower() not in FLUFF_WORDS]
        lexical_density = round((len(content_words) / word_count) * 100, 2) if word_count > 0 else 0.0

        entropy, perplexity = self._compute_entropy_and_perplexity([w.lower() for w in words])

        return PromptFeatures(
            token_count=token_count,
            char_count=char_count,
            word_count=word_count,
            char_to_token_ratio=char_to_token,
            type_token_ratio=ttr,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_len,
            question_type=q_type.value,
            imperative_command_count=imperatives,
            has_code_block=has_code_block,
            code_presence_score=round(code_score, 3),
            detected_code_language=detected_lang,
            reasoning_density_score=round(reasoning_score, 3),
            constraint_density_score=round(constraint_score, 3),
            negation_count=negation_count,
            syntactic_complexity_score=round(syntactic_score, 3),
            primary_script=primary_script,
            fragmentation_ratio=frag_ratio,
            fluff_density_score=fluff_score,
            shannon_entropy=entropy,
            token_perplexity=perplexity,
            lexical_density_pct=lexical_density,
            detected_reasons=detected_reasons,
            detected_constraints=detected_constraints,
        )
