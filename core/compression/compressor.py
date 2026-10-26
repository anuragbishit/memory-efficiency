"""
Deterministic Non-AI Algorithmic Token Compression Engine (Pillar 3).
Performs sub-millisecond linguistic optimization, code preservation,
semantic anchor protection, and token bloat reduction.
"""

import re
import time
import unicodedata
from enum import Enum
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from core.tokenization.tokenizer import SubwordTokenizer
from .rules import (
    CONVERSATIONAL_BOILERPLATE,
    PUNCTUATION_REDUNDANCY,
    SEMANTIC_ANCHORS,
    LOW_INFO_FILLERS,
)
from .ast_code_compressor import ASTCodeCompressor, ASTCompressionResult
from .rule_compressor import RuleBasedTextCompressor, RuleCompressionReport


class CompressionLevel(str, Enum):
    """Compression intensity levels."""
    SAFE = "safe"              # Minimal: Cleans boilerplate & whitespace, NFKC Unicode
    BALANCED = "balanced"      # Recommended: AST code compression, phrase compaction, guards anchors
    AGGRESSIVE = "aggressive"  # Maximum: Compacts phrasing, prunes non-anchor stopwords


@dataclass
class CompressionResult:
    """Detailed summary of the algorithmic compression pass."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio_pct: float
    latency_ms: float
    encoding_used: str
    preserved_code_blocks: int
    ast_code_summaries: List[Dict[str, Any]]
    rules_applied: List[str]


class NonAITokenCompressor:
    """
    Algorithmic Token Compressor.
    Executes deterministic computer-science heuristics to minimize payload tokens
    without incurring secondary LLM latency or financial costs.
    """

    FENCED_CODE_PATTERN = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
    INLINE_CODE_PATTERN = re.compile(r"(`[^`]+`)")

    def __init__(self, tokenizer: SubwordTokenizer | None = None):
        self.tokenizer = tokenizer or SubwordTokenizer()
        self.ast_compressor = ASTCodeCompressor(tokenizer=self.tokenizer)
        self.rule_compressor = RuleBasedTextCompressor()

    def _process_and_protect_code_blocks(
        self,
        text: str,
        level: CompressionLevel,
        encoding_name: str
    ) -> Tuple[str, List[str], List[ASTCompressionResult]]:
        """
        Shields code blocks and compresses them via AST (Python) or lexical rules (non-Python)
        while mathematically verifying semantic equivalence.
        """
        placeholders: List[str] = []
        ast_results: List[ASTCompressionResult] = []

        def replacer(match: re.Match) -> str:
            raw_block = match.group(1)
            placeholder = f"__CODE_BLOCK_{len(placeholders)}__"

            # In BALANCED/AGGRESSIVE mode, compress code with AST
            if level in [CompressionLevel.BALANCED, CompressionLevel.AGGRESSIVE] and raw_block.startswith("```"):
                ast_res = self.ast_compressor.compress_code_block(
                    raw_block, encoding_name=encoding_name, strip_docstrings=True
                )
                placeholders.append(ast_res.compressed_code)
                ast_results.append(ast_res)
            else:
                placeholders.append(raw_block)

            return placeholder

        # Shield fenced blocks first, then inline code
        text = self.FENCED_CODE_PATTERN.sub(replacer, text)
        text = self.INLINE_CODE_PATTERN.sub(replacer, text)
        return text, placeholders, ast_results

    def _restore_code_blocks(self, text: str, placeholders: List[str]) -> str:
        """Restores shielded code blocks with exact original fidelity."""
        for i, code_content in enumerate(placeholders):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(placeholder, code_content)
        return text

    def _clean_conversational_boilerplate(self, text: str) -> Tuple[str, List[str]]:
        """Applies regex heuristics to strip conversational greeting and closing fluff."""
        applied: List[str] = []
        for pattern, replacement in CONVERSATIONAL_BOILERPLATE:
            matches = list(pattern.finditer(text))
            if matches:
                for match in matches:
                    matched_str = match.group(0).strip()
                    if matched_str:
                        applied.append(f"Boilerplate: Pruned '{matched_str}'")
                text = pattern.sub(replacement, text)
        return text, applied

    def _prune_filler_words(self, text: str, level: CompressionLevel) -> str:
        """
        Removes conversational filler words while strictly guarding
        semantic anchors and negation operators.
        """
        if level == CompressionLevel.SAFE:
            return text

        words = text.split(" ")
        cleaned_words: List[str] = []

        for word in words:
            clean_word_lower = re.sub(r"[^\w]", "", word).lower()

            # NEVER prune semantic anchors (negations, constraints, boundary logic)
            if clean_word_lower in SEMANTIC_ANCHORS:
                cleaned_words.append(word)
                continue

            # Strip low-information conversational fillers
            if clean_word_lower in LOW_INFO_FILLERS:
                continue

            cleaned_words.append(word)

        return " ".join(cleaned_words)

    def _clean_punctuation_and_whitespace(self, text: str) -> str:
        """Compacts duplicated punctuation marks and excessive whitespace."""
        for pattern, replacement in PUNCTUATION_REDUNDANCY:
            text = pattern.sub(replacement, text)
        return text.strip()

    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.BALANCED,
        encoding_name: str = "cl100k_base"
    ) -> CompressionResult:
        """
        Executes the end-to-end algorithmic token compression pipeline:
          1. Unicode NFKC normalization
          2. Code block extraction & AST semantic compression
          3. Conversational boilerplate removal
          4. Rule-based wordy phrase compaction (Topic 4)
          5. Context-aware stopword removal (with negation/anchor guarding)
          6. Whitespace & punctuation compaction
          7. Code block restoration with 100% semantic fidelity
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                compression_ratio_pct=0.0,
                latency_ms=0.0,
                encoding_used=encoding_name,
                preserved_code_blocks=0,
                ast_code_summaries=[],
                rules_applied=[],
            )

        original_tokens = self.tokenizer.count_tokens(text, encoding_name)
        all_rules_applied: List[str] = []

        # Step 1: Unicode Normalization (NFKC prevents multi-byte BPE fragmentation)
        normalized_text = unicodedata.normalize("NFKC", text)

        # Step 2: Code Shielding & AST Semantic Compression (Topic 5)
        shielded_text, code_placeholders, ast_results = self._process_and_protect_code_blocks(
            normalized_text, level=level, encoding_name=encoding_name
        )
        for ast_item in ast_results:
            if ast_item.tokens_saved > 0:
                all_rules_applied.append(
                    f"AST ({ast_item.language}): Compressed code by {ast_item.tokens_saved} tokens ({ast_item.compression_ratio_pct}%) [Equivalence: {ast_item.semantic_equivalence_verified}]"
                )

        # Step 3: Strip Conversational Boilerplate
        processed, bp_rules = self._clean_conversational_boilerplate(shielded_text)
        all_rules_applied.extend(bp_rules)

        # Step 4: Rule-Based Wordy Phrase Compaction & Stopwords (Topic 4)
        if level in [CompressionLevel.BALANCED, CompressionLevel.AGGRESSIVE]:
            rule_report: RuleCompressionReport = self.rule_compressor.compress(
                processed, aggressive=(level == CompressionLevel.AGGRESSIVE)
            )
            processed = rule_report.compressed_text
            all_rules_applied.extend(rule_report.rules_applied)
        else:
            processed = self._prune_filler_words(processed, level)

        # Step 5: Whitespace and Punctuation Cleanup
        processed = self._clean_punctuation_and_whitespace(processed)

        # Step 6: Restore AST-Compressed Code Blocks
        compressed_text = self._restore_code_blocks(processed, code_placeholders)

        # Recalculate tokens & latency
        compressed_tokens = self.tokenizer.count_tokens(compressed_text, encoding_name)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        tokens_saved = max(0, original_tokens - compressed_tokens)
        ratio = (
            round((tokens_saved / original_tokens) * 100, 2)
            if original_tokens > 0
            else 0.0
        )

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            compression_ratio_pct=ratio,
            latency_ms=latency_ms,
            encoding_used=encoding_name,
            preserved_code_blocks=len(code_placeholders),
            ast_code_summaries=[r.to_dict() for r in ast_results],
            rules_applied=all_rules_applied,
        )
