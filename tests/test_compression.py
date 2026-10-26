"""
Unit tests for Non-AI Algorithmic Token Compression (Pillar 3).
"""

import pytest
from core.compression.compressor import NonAITokenCompressor, CompressionLevel


def test_boilerplate_stripping():
    compressor = NonAITokenCompressor()
    prompt = "Hello, could you please kindly help me to write a quicksort algorithm? Thanks in advance!"
    res = compressor.compress(prompt, level=CompressionLevel.BALANCED)

    assert "hello" not in res.compressed_text.lower()
    assert "please" not in res.compressed_text.lower()
    assert "kindly" not in res.compressed_text.lower()
    assert "thanks in advance" not in res.compressed_text.lower()
    assert "quicksort algorithm" in res.compressed_text.lower()
    assert res.tokens_saved > 0
    assert res.compression_ratio_pct > 20.0


def test_strict_code_preservation():
    compressor = NonAITokenCompressor()
    prompt = (
        "Please help me optimize this function:\n"
        "```python\n"
        "def compute_fibonacci(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)\n"
        "```\n"
        "Thanks!"
    )
    res = compressor.compress(prompt, level=CompressionLevel.BALANCED)

    # The code block inside triple backticks must be byte-for-byte intact
    expected_code = (
        "```python\n"
        "def compute_fibonacci(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)\n"
        "```"
    )
    assert expected_code in res.compressed_text
    assert res.preserved_code_blocks == 1


def test_semantic_anchor_and_negation_protection():
    compressor = NonAITokenCompressor()
    # Crucial test: Negations and constraint words MUST NOT be removed
    prompt = "You must not use recursion and do not import external libraries. Only use O(1) space."
    res = compressor.compress(prompt, level=CompressionLevel.BALANCED)

    compressed_lower = res.compressed_text.lower()
    assert "not" in compressed_lower
    assert "must" in compressed_lower
    assert "only" in compressed_lower
    assert "o(1)" in compressed_lower or "o (1)" in compressed_lower


def test_punctuation_and_whitespace_cleanup():
    compressor = NonAITokenCompressor()
    prompt = "What is graph defragmentation????    Are nodes merged!!!!\n\n\n\nTell me."
    res = compressor.compress(prompt, level=CompressionLevel.BALANCED)

    assert "????" not in res.compressed_text
    assert "!!!!" not in res.compressed_text
    assert "\n\n\n\n" not in res.compressed_text
    assert "?" in res.compressed_text
    assert "!" in res.compressed_text


def test_compression_speed_latency():
    compressor = NonAITokenCompressor()
    prompt = "Could you please explain how GraphRAG avoids context bleed compared to flat vector embeddings?"
    res = compressor.compress(prompt)

    # Non-AI compression must execute in under 10 milliseconds
    assert res.latency_ms < 10.0


def test_user_query_compression():
    compressor = NonAITokenCompressor()
    prompt = "write a code in python to print a fibonnaci series and also define its variant ?"
    res = compressor.compress(prompt, level=CompressionLevel.BALANCED)

    assert "write python code to print fibonnaci series and define its variant?" == res.compressed_text
    assert res.tokens_saved > 0
    assert res.compression_ratio_pct > 20.0
    assert len(res.rules_applied) >= 3
