"""
Token Analysis and Multilingual Inflation Measurement Module.
Analyzes subword fragmentation, token-to-word ratios, and multilingual
token inflation across cl100k_base and o200k_base encodings.
"""

from typing import Dict, Any, List
import unicodedata
from .tokenizer import SubwordTokenizer


class TokenAnalyzer:
    """
    Analyzes token-level efficiency, subword fragmentation,
    and cross-lingual token inflation.
    """

    def __init__(self, tokenizer: SubwordTokenizer | None = None):
        self.tokenizer = tokenizer or SubwordTokenizer()

    def analyze_fragmentation(self, text: str, encoding_name: str | None = None) -> Dict[str, Any]:
        """
        Calculates word-to-subword fragmentation metrics.
        Higher fragmentation means the tokenizer cannot represent words
        as atomic units, leading to token bloat.
        """
        words = text.strip().split()
        total_words = len(words)
        token_count = self.tokenizer.count_tokens(text, encoding_name)
        token_details = self.tokenizer.inspect_tokens(text, encoding_name)

        multibyte_tokens = sum(1 for t in token_details if t.is_multibyte)
        tokens_per_word = round(token_count / total_words, 2) if total_words > 0 else 0.0

        return {
            "total_characters": len(text),
            "total_words": total_words,
            "total_tokens": token_count,
            "tokens_per_word": tokens_per_word,
            "multibyte_token_count": multibyte_tokens,
            "multibyte_ratio": round(multibyte_tokens / token_count, 3) if token_count > 0 else 0.0,
            "avg_bytes_per_token": round(
                sum(t.byte_length for t in token_details) / token_count, 2
            ) if token_count > 0 else 0.0
        }

    def measure_multilingual_inflation(
        self,
        reference_english: str,
        target_language_text: str,
        target_lang_name: str = "Target Language"
    ) -> Dict[str, Any]:
        """
        Computes the token inflation factor of a non-English language relative to
        an equivalent English sentence across cl100k_base and o200k_base.
        
        Formula: Inflation Factor = (Target Tokens) / (English Tokens)
        """
        en_cl = self.tokenizer.count_tokens(reference_english, "cl100k_base")
        en_o2 = self.tokenizer.count_tokens(reference_english, "o200k_base")

        tgt_cl = self.tokenizer.count_tokens(target_language_text, "cl100k_base")
        tgt_o2 = self.tokenizer.count_tokens(target_language_text, "o200k_base")

        inflation_cl = round(tgt_cl / en_cl, 2) if en_cl > 0 else 0.0
        inflation_o2 = round(tgt_o2 / en_o2, 2) if en_o2 > 0 else 0.0

        # Improvement from cl100k to o200k on target language
        o2_improvement_pct = round(((en_cl - tgt_o2) / tgt_cl) * 100, 2) if tgt_cl > 0 else 0.0

        return {
            "target_language": target_lang_name,
            "english_baseline": {
                "text": reference_english,
                "cl100k_tokens": en_cl,
                "o200k_tokens": en_o2
            },
            "target_language_metrics": {
                "text": target_language_text,
                "cl100k_tokens": tgt_cl,
                "o200k_tokens": tgt_o2
            },
            "inflation_factors": {
                "cl100k_base_inflation_multiplier": f"{inflation_cl}x",
                "o200k_base_inflation_multiplier": f"{inflation_o2}x"
            },
            "o200k_token_reduction_over_cl100k_pct": round(
                ((tgt_cl - tgt_o2) / tgt_cl) * 100, 2
            ) if tgt_cl > 0 else 0.0
        }
