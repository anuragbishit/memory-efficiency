"""
Unit tests for Subword Tokenization (BPE, cl100k_base, o200k_base).
"""

import pytest
from core.tokenization.tokenizer import SubwordTokenizer, EducationalBPE
from core.tokenization.analyzer import TokenAnalyzer


def test_tokenizer_initialization():
    tokenizer = SubwordTokenizer()
    assert tokenizer.default_encoding_name == "cl100k_base"
    assert "cl100k_base" in tokenizer.SUPPORTED_ENCODINGS
    assert "o200k_base" in tokenizer.SUPPORTED_ENCODINGS


def test_token_encode_decode_roundtrip():
    tokenizer = SubwordTokenizer()
    sample = "Optimizing AI Architecture for Long-Term Memory and Efficiency."

    for enc in ["cl100k_base", "o200k_base"]:
        tokens = tokenizer.encode(sample, enc)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens, enc)
        assert decoded == sample


def test_token_inspection():
    tokenizer = SubwordTokenizer()
    sample = "Defines a directed graph G=(V,E)"
    details = tokenizer.inspect_tokens(sample, "cl100k_base")

    assert len(details) > 0
    # Reconstructed text from token details must match original
    reconstructed = "".join(t.text for t in details)
    assert reconstructed == sample


def test_compare_encodings():
    tokenizer = SubwordTokenizer()
    sample = "Multilingual tokenization comparison: नमस्ते दुनिया, Hello World, Bonjour le monde."
    comparison = tokenizer.compare_encodings(sample)

    assert "cl100k_base" in comparison["encodings"]
    assert "o200k_base" in comparison["encodings"]
    assert comparison["input_characters"] == len(sample)
    assert comparison["encodings"]["cl100k_base"]["token_count"] > 0
    assert comparison["encodings"]["o200k_base"]["token_count"] > 0


def test_multilingual_token_inflation():
    analyzer = TokenAnalyzer()
    en = "The artificial intelligence system optimizes memory and tokens."
    # Hindi translation of the same sentence
    hi = "कृत्रिम बुद्धिमत्ता प्रणाली मेमोरी और टोकन को अनुकूलित करती है।"

    report = analyzer.measure_multilingual_inflation(en, hi, "Hindi")
    assert report["english_baseline"]["cl100k_tokens"] > 0
    assert report["target_language_metrics"]["cl100k_tokens"] > 0
    # Non-English under cl100k should exhibit noticeable token inflation
    assert report["target_language_metrics"]["cl100k_tokens"] > report["english_baseline"]["cl100k_tokens"]


def test_educational_bpe_training():
    bpe = EducationalBPE(vocab_size_target=260)
    corpus = "low lower newest widest lowest lower"
    merges = bpe.train(corpus)
    # Must learn at least one merge from repeated substrings
    assert len(merges) > 0
