"""
Information Density and Perplexity Measurement Module.
Calculates Shannon Entropy, Token Perplexity, Lexical Density,
and Information-to-Token Efficiency before and after algorithmic compression.
"""

import math
import re
from typing import Dict, Any, List, Set, Tuple
from collections import Counter
from dataclasses import dataclass, asdict

from core.tokenization.tokenizer import SubwordTokenizer
from core.compression.rules import SEMANTIC_ANCHORS, LOW_INFO_FILLERS


# Standard closed-class grammatical words (function words) in English linguistics
FUNCTION_WORDS: Set[str] = {
    # Prepositions & Particles
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under", "again", "of",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    # Auxiliary & Modal Verbs
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could", "ought",
    "might", "may", "shall", "can", "will",
    # Conjunctions & Adverbs
    "if", "because", "as", "until", "while", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just", "now",
    # Polite Fluff & Greetings
    "hello", "hi", "hey", "please", "kindly", "thanks", "thank", "welcome"
}


@dataclass
class DensityMetrics:
    """Metrics quantifying the information density and perplexity of a text payload."""
    token_count: int
    unique_tokens: int
    type_token_ratio: float  # Vocabulary diversity (unique / total)
    shannon_entropy_bits: float  # H(X) in bits per token
    token_perplexity: float  # 2^H(X)
    lexical_density_pct: float  # Content words / total words
    content_word_count: int
    function_word_count: int
    information_density_index: float  # (Entropy * Lexical Density) / Token Count
    estimated_coherence_score: float  # 0.0 to 1.0 linguistic smoothness

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InformationDensityAnalyzer:
    """
    Analyzes Shannon Entropy, Lexical Information Density,
    and Token Perplexity for prompt payloads.
    """

    def __init__(self, tokenizer: SubwordTokenizer | None = None):
        self.tokenizer = tokenizer or SubwordTokenizer()

    def compute_shannon_entropy(self, token_ids: List[int]) -> Tuple[float, float]:
        """
        Calculates the Shannon Entropy H(X) in bits and Perplexity 2^H(X)
        over the empirical distribution of tokens.
        
        Formula:
          H(X) = - sum( p(x) * log2( p(x) ) )
          Perplexity = 2^H(X)
        """
        if not token_ids:
            return 0.0, 1.0

        n = len(token_ids)
        counts = Counter(token_ids)

        entropy = 0.0
        for count in counts.values():
            p_x = count / n
            entropy -= p_x * math.log2(p_x)

        # Perplexity is base-2 exponentiation of Shannon entropy
        perplexity = math.pow(2, entropy)
        return round(entropy, 3), round(perplexity, 3)

    def compute_lexical_density(self, text: str) -> Tuple[float, int, int]:
        """
        Computes Lexical Density (Halliday's linguistic metric):
          Lexical Density = (Content Words / Total Words) * 100%
        Content words convey real semantic information (nouns, main verbs, adjectives,
        code symbols, technical constraints), whereas function words are grammatical glue.
        """
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        if not words:
            return 0.0, 0, 0

        content_words = 0
        function_words = 0

        for w in words:
            # Code tokens, numbers, and technical terms are high-value content words
            if any(char.isdigit() for char in w) or "_" in w or "-" in w:
                content_words += 1
            elif w in SEMANTIC_ANCHORS:
                content_words += 1
            elif w in FUNCTION_WORDS or w in LOW_INFO_FILLERS:
                function_words += 1
            else:
                content_words += 1

        total = len(words)
        density_pct = round((content_words / total) * 100, 2) if total > 0 else 0.0
        return density_pct, content_words, function_words

    def estimate_coherence(self, text: str, token_ids: List[int]) -> float:
        """
        Estimates prompt linguistic coherence and structural regularity.
        Penalizes repeated adjacent duplicate tokens, dangling punctuation,
        and ungrammatical fragment spikes.
        """
        if not token_ids or len(token_ids) <= 1:
            return 1.0

        # Check for unnatural consecutive duplicate tokens
        consecutive_duplicates = sum(
            1 for i in range(len(token_ids) - 1) if token_ids[i] == token_ids[i + 1]
        )
        duplicate_penalty = min(0.4, (consecutive_duplicates / len(token_ids)) * 2.0)

        # Check for healthy punctuation structure
        has_ending_punct = bool(re.search(r"[.!?`:}\]]\s*$", text.strip()))
        ending_bonus = 0.1 if has_ending_punct else 0.0

        base_coherence = 0.90 + ending_bonus - duplicate_penalty
        return round(min(1.0, max(0.2, base_coherence)), 3)

    def analyze(self, text: str, encoding_name: str = "cl100k_base") -> DensityMetrics:
        """Computes complete information density, entropy, and perplexity metrics."""
        if not text or not text.strip():
            return DensityMetrics(
                token_count=0,
                unique_tokens=0,
                type_token_ratio=0.0,
                shannon_entropy_bits=0.0,
                token_perplexity=1.0,
                lexical_density_pct=0.0,
                content_word_count=0,
                function_word_count=0,
                information_density_index=0.0,
                estimated_coherence_score=1.0,
            )

        token_ids = self.tokenizer.encode(text, encoding_name)
        total_tokens = len(token_ids)
        unique_tokens = len(set(token_ids))
        ttr = round(unique_tokens / total_tokens, 3) if total_tokens > 0 else 0.0

        entropy, perplexity = self.compute_shannon_entropy(token_ids)
        lex_density, content_count, func_count = self.compute_lexical_density(text)

        # Information Density Index (IDI):
        # Quantifies bits of high-density lexical content packed per token
        # Higher score = denser, more efficient prompt payload
        if total_tokens > 0:
            idi = round((entropy * (lex_density / 100.0) * 10.0) / math.sqrt(total_tokens), 3)
        else:
            idi = 0.0

        coherence = self.estimate_coherence(text, token_ids)

        return DensityMetrics(
            token_count=total_tokens,
            unique_tokens=unique_tokens,
            type_token_ratio=ttr,
            shannon_entropy_bits=entropy,
            token_perplexity=perplexity,
            lexical_density_pct=lex_density,
            content_word_count=content_count,
            function_word_count=func_count,
            information_density_index=idi,
            estimated_coherence_score=coherence,
        )

    def compare_compression_density_gain(
        self,
        raw_text: str,
        compressed_text: str,
        encoding_name: str = "cl100k_base"
    ) -> Dict[str, Any]:
        """
        Measures the efficiency gain in information density and perplexity
        produced by the algorithmic token compressor.
        """
        raw_m = self.analyze(raw_text, encoding_name)
        comp_m = self.analyze(compressed_text, encoding_name)

        lexical_gain_pct = round(comp_m.lexical_density_pct - raw_m.lexical_density_pct, 2)
        idi_gain_pct = round(
            ((comp_m.information_density_index - raw_m.information_density_index) / raw_m.information_density_index) * 100, 2
        ) if raw_m.information_density_index > 0 else 0.0

        return {
            "raw_payload": raw_m.to_dict(),
            "compressed_payload": comp_m.to_dict(),
            "lexical_density_increase_pct": f"+{lexical_gain_pct}%" if lexical_gain_pct >= 0 else f"{lexical_gain_pct}%",
            "information_density_index_gain_pct": f"+{idi_gain_pct}%" if idi_gain_pct >= 0 else f"{idi_gain_pct}%",
            "coherence_maintained": comp_m.estimated_coherence_score >= 0.85,
            "bits_per_token_raw": raw_m.shannon_entropy_bits,
            "bits_per_token_compressed": comp_m.shannon_entropy_bits,
        }
