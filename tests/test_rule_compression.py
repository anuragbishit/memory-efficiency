"""
Unit tests for Rule-Based Text Compression and Stopword Removal (Topic 4).
"""

import pytest
from core.compression.rule_compressor import RuleBasedTextCompressor


def test_pleonasm_and_wordy_phrase_compaction():
    compressor = RuleBasedTextCompressor()
    text = (
        "Please make an attempt to perform an analysis on this dataset "
        "at the present time in order to decide the final outcome."
    )

    report = compressor.compress(text, aggressive=False)

    assert "make an attempt to" not in report.compressed_text
    assert "attempt to" in report.compressed_text
    assert "perform an analysis on" not in report.compressed_text
    assert "analyze" in report.compressed_text
    assert "at the present time" not in report.compressed_text
    assert "now" in report.compressed_text
    assert report.phrases_compacted >= 3


def test_question_words_and_negations_never_pruned():
    compressor = RuleBasedTextCompressor()
    # Critical test: Standard stopword removers delete 'why', 'how', 'not'. Ours MUST keep them!
    text = "Explain why this algorithm cannot achieve O(1) space and how to optimize it without extra memory."

    report = compressor.compress(text, aggressive=True)

    compressed_lower = report.compressed_text.lower()
    assert "why" in compressed_lower
    assert "how" in compressed_lower
    assert "cannot" in compressed_lower
    assert "without" in compressed_lower
    assert "o(1)" in compressed_lower or "o (1)" in compressed_lower


def test_filler_stopword_pruning():
    compressor = RuleBasedTextCompressor()
    text = "Basically you should simply and actually consider using a graph structure."

    report = compressor.compress(text, aggressive=False)

    assert "basically" not in report.compressed_text.lower()
    assert "actually" not in report.compressed_text.lower()
    assert "simply" not in report.compressed_text.lower()
    assert "consider using a graph structure" in report.compressed_text.lower()
    assert report.stopwords_pruned >= 3
