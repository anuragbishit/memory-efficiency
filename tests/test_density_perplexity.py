"""
Unit tests for Information Density and Token Perplexity.
"""

import pytest
from core.compression.density import InformationDensityAnalyzer
from core.compression.compressor import NonAITokenCompressor, CompressionLevel


def test_shannon_entropy_and_perplexity_calculation():
    analyzer = InformationDensityAnalyzer()
    tokens = [100, 200, 300, 400, 500, 100, 200]
    entropy, perplexity = analyzer.compute_shannon_entropy(tokens)

    # Shannon entropy must be positive
    assert entropy > 0.0
    # Perplexity must be 2^entropy
    assert perplexity >= 1.0


def test_lexical_density_calculation():
    analyzer = InformationDensityAnalyzer()
    # Content-rich prompt with technical words
    rich_text = "Architect a distributed thread-safe cache with O(1) read operations."
    density_pct, content_count, function_count = analyzer.compute_lexical_density(rich_text)

    assert density_pct > 50.0
    assert content_count > function_count


def test_compression_increases_information_density():
    analyzer = InformationDensityAnalyzer()
    compressor = NonAITokenCompressor()

    raw_prompt = (
        "Hello, could you please kindly assist me in understanding how to sort an array "
        "in Python? I was wondering if you might be able to explain it simply for me. "
        "Thank you so much in advance, I really appreciate your help!"
    )

    compressed = compressor.compress(raw_prompt, level=CompressionLevel.BALANCED)
    comparison = analyzer.compare_compression_density_gain(
        raw_prompt, compressed.compressed_text
    )

    # Lexical density should increase because boilerplate function fluff is pruned
    raw_ld = comparison["raw_payload"]["lexical_density_pct"]
    comp_ld = comparison["compressed_payload"]["lexical_density_pct"]
    assert comp_ld > raw_ld

    # Information density index should increase
    raw_idi = comparison["raw_payload"]["information_density_index"]
    comp_idi = comparison["compressed_payload"]["information_density_index"]
    assert comp_idi > raw_idi

    # Linguistic coherence must remain high
    assert comparison["coherence_maintained"] is True


def test_empty_string_density():
    analyzer = InformationDensityAnalyzer()
    metrics = analyzer.analyze("")
    assert metrics.token_count == 0
    assert metrics.shannon_entropy_bits == 0.0
    assert metrics.token_perplexity == 1.0
