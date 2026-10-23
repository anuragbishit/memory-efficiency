"""
Unit tests for Multilingual Token Fragmentation Analysis.
"""

import pytest
from core.tokenization.fragmentation import (
    MultilingualFragmentationAnalyzer,
    PARALLEL_MULTILINGUAL_CORPUS,
)


def test_script_detection_accuracy():
    analyzer = MultilingualFragmentationAnalyzer()

    latin_text = "Standard English prompt for system architecture."
    hindi_text = "दीर्घकालिक मेमोरी और संदर्भ प्रबंधन प्रणाली।"
    arabic_text = "كيف تعمل معمارية الذاكرة طويلة المدى؟"
    cyrillic_text = "Архитектура долговременной памяти."

    assert analyzer.analyze(latin_text).detected_primary_script == "Latin"
    assert "Devanagari" in analyzer.analyze(hindi_text).detected_primary_script
    assert analyzer.analyze(arabic_text).detected_primary_script == "Arabic"
    assert analyzer.analyze(cyrillic_text).detected_primary_script == "Cyrillic"


def test_fragmentation_ratio_indic_vs_english():
    analyzer = MultilingualFragmentationAnalyzer()

    en = "How does graph-based memory eliminate context bleed?"
    hi = "ग्राफ-आधारित मेमोरी संदर्भ के बहाव को कैसे समाप्त करती है?"

    en_metrics = analyzer.analyze(en)
    hi_metrics = analyzer.analyze(hi)

    # English has modest fragmentation (~1.0 to 1.8 tokens/word)
    assert en_metrics.cl100k_fragmentation_ratio < 2.0
    # Hindi under cl100k suffers heavy subword fragmentation (> 2.2 tokens/word)
    assert hi_metrics.cl100k_fragmentation_ratio > 2.2
    assert hi_metrics.is_heavily_fragmented is True
    assert hi_metrics.fragmentation_penalty_score > 0.3


def test_o200k_mitigates_fragmentation():
    analyzer = MultilingualFragmentationAnalyzer()
    hi = "कृत्रिम बुद्धिमत्ता प्रणाली मेमोरी और टोकन को अनुकूलित करती है।"

    metrics = analyzer.analyze(hi)
    # o200k_base must have significantly lower fragmentation ratio than cl100k_base
    assert metrics.o200k_tokens < metrics.cl100k_tokens
    assert metrics.o200k_fragmentation_ratio < metrics.cl100k_fragmentation_ratio


def test_cross_lingual_parity_benchmark():
    analyzer = MultilingualFragmentationAnalyzer()
    parity_report = analyzer.evaluate_cross_lingual_parity()

    assert "english_baseline" in parity_report
    assert "languages" in parity_report
    assert len(parity_report["languages"]) >= 7

    # Non-English languages should show their cost multipliers relative to English
    for lang_entry in parity_report["languages"]:
        assert "cl100k_cost_multiplier_vs_en" in lang_entry
        assert "o200k_cost_multiplier_vs_en" in lang_entry
