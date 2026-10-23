"""
Multilingual Token Fragmentation Analysis Module.
Analyzes Unicode script distribution, word-to-subword fragmentation,
byte-to-token inefficiencies, and cross-lingual billing parity across BPE encodings.
"""

import unicodedata
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from core.tokenization.tokenizer import SubwordTokenizer


@dataclass
class ScriptBreakdown:
    """Distribution of character scripts detected in a text."""
    script_name: str
    char_count: int
    percentage: float


@dataclass
class FragmentationMetrics:
    """Detailed fragmentation analytics for a given text payload."""
    text: str
    detected_primary_script: str
    script_breakdown: List[Dict[str, Any]]
    total_characters: int
    total_utf8_bytes: int
    total_words: int
    cl100k_tokens: int
    o200k_tokens: int
    cl100k_fragmentation_ratio: float  # tokens / words
    o200k_fragmentation_ratio: float   # tokens / words
    cl100k_bytes_per_token: float
    o200k_bytes_per_token: float
    is_heavily_fragmented: bool
    fragmentation_penalty_score: float  # 0.0 (low/English-like) to 1.0 (severe)


# Parallel test corpus representing identical semantic queries in diverse scripts
PARALLEL_MULTILINGUAL_CORPUS = {
    "English (Latin)": (
        "How does graph-based memory eliminate context bleed and hallucinations in long-term AI interactions?"
    ),
    "Hindi (Devanagari)": (
        "ग्राफ-आधारित मेमोरी दीर्घकालिक एआई इंटरैक्शन में संदर्भ के बहाव और मतिभ्रम को कैसे समाप्त करती है?"
    ),
    "Bengali (Bengali)": (
        "গ্রাফ-ভিত্তিক মেমরি কীভাবে দীর্ঘমেয়াদী এআই ইন্টারঅ্যাকশনে প্রসঙ্গ রক্তপাত এবং হ্যালুসিনেশন দূর করে?"
    ),
    "Spanish (Latin)": (
        "¿Cómo elimina la memoria basada en grafos la pérdida de contexto y las alucinaciones en las interacciones de IA a largo plazo?"
    ),
    "French (Latin)": (
        "Comment la mémoire basée sur des graphes élimine-t-elle la perte de contexte et les hallucinations dans les interactions d'IA à long terme ?"
    ),
    "Russian (Cyrillic)": (
        "Как память на основе графов устраняет утечку контекста и галлюцинации в долгосрочных взаимодействиях с ИИ?"
    ),
    "Arabic (Arabic)": (
        "كيف تقضي الذاكرة المستندة إلى الرسوم البيانية على تسرب السياق والهلوسة في تفاعلات الذكاء الاصطناعي طويلة المدى؟"
    ),
    "Chinese (Hanzi/CJK)": (
        "基于图谱的记忆如何在长期AI交互中消除上下文泄露和幻觉？"
    ),
}


class MultilingualFragmentationAnalyzer:
    """
    Multilingual Token Fragmentation Analyzer.
    Quantifies the subword penalty and economic disparity imposed on non-Latin scripts
    by standard BPE tokenizers (e.g. cl100k_base vs o200k_base).
    """

    def __init__(self, tokenizer: SubwordTokenizer | None = None):
        self.tokenizer = tokenizer or SubwordTokenizer()

    @staticmethod
    def identify_char_script(char: str) -> str:
        """Determines the Unicode script category of a character."""
        if not char.strip():
            return "Whitespace"
        if char.isnumeric():
            return "Numeric"
        if unicodedata.category(char).startswith("P"):
            return "Punctuation"

        name = unicodedata.name(char, "").upper()
        if "DEVANAGARI" in name:
            return "Devanagari (Hindi/Sanskrit)"
        elif "BENGALI" in name:
            return "Bengali"
        elif "ARABIC" in name:
            return "Arabic"
        elif "CYRILLIC" in name:
            return "Cyrillic"
        elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
            return "CJK (East Asian)"
        elif "LATIN" in name:
            return "Latin"
        else:
            return "Other"

    def analyze_script_distribution(self, text: str) -> List[ScriptBreakdown]:
        """Calculates percentage breakdown of linguistic scripts in the text."""
        if not text:
            return []

        counts: Dict[str, int] = {}
        total = 0
        for char in text:
            if char.strip() and not unicodedata.category(char).startswith("P"):
                script = self.identify_char_script(char)
                counts[script] = counts.get(script, 0) + 1
                total += 1

        if total == 0:
            return []

        breakdown = [
            ScriptBreakdown(
                script_name=script,
                char_count=count,
                percentage=round((count / total) * 100, 2),
            )
            for script, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ]
        return breakdown

    def analyze(self, text: str) -> FragmentationMetrics:
        """
        Computes detailed token fragmentation metrics across cl100k and o200k.
        Evaluates fragmentation ratio F = tokens / words and economic penalty.
        """
        if not text or not text.strip():
            return FragmentationMetrics(
                text="",
                detected_primary_script="None",
                script_breakdown=[],
                total_characters=0,
                total_utf8_bytes=0,
                total_words=0,
                cl100k_tokens=0,
                o200k_tokens=0,
                cl100k_fragmentation_ratio=0.0,
                o200k_fragmentation_ratio=0.0,
                cl100k_bytes_per_token=0.0,
                o200k_bytes_per_token=0.0,
                is_heavily_fragmented=False,
                fragmentation_penalty_score=0.0,
            )

        words = text.strip().split()
        total_words = len(words)
        char_count = len(text)
        utf8_bytes = len(text.encode("utf-8"))

        cl100k_tokens = self.tokenizer.count_tokens(text, "cl100k_base")
        o200k_tokens = self.tokenizer.count_tokens(text, "o200k_base")

        cl_frag_ratio = round(cl100k_tokens / total_words, 2) if total_words > 0 else 0.0
        o2_frag_ratio = round(o200k_tokens / total_words, 2) if total_words > 0 else 0.0

        cl_bytes_per_token = round(utf8_bytes / cl100k_tokens, 2) if cl100k_tokens > 0 else 0.0
        o2_bytes_per_token = round(utf8_bytes / o200k_tokens, 2) if o200k_tokens > 0 else 0.0

        script_breakdown = self.analyze_script_distribution(text)
        primary_script = script_breakdown[0].script_name if script_breakdown else "Unknown"

        # Baseline English usually has fragmentation ratio around 1.2 - 1.5 tokens/word.
        # Ratios > 2.2 indicate severe subword fragmentation (typical in Devanagari/Bengali/Arabic).
        is_heavily_fragmented = cl_frag_ratio >= 2.2

        # Penalty score normalized from 0.0 to 1.0 (where 1.3 is 0.0, and >= 4.0 is 1.0)
        penalty = min(1.0, max(0.0, (cl_frag_ratio - 1.3) / 2.7))

        return FragmentationMetrics(
            text=text,
            detected_primary_script=primary_script,
            script_breakdown=[asdict(sb) for sb in script_breakdown],
            total_characters=char_count,
            total_utf8_bytes=utf8_bytes,
            total_words=total_words,
            cl100k_tokens=cl100k_tokens,
            o200k_tokens=o200k_tokens,
            cl100k_fragmentation_ratio=cl_frag_ratio,
            o200k_fragmentation_ratio=o2_frag_ratio,
            cl100k_bytes_per_token=cl_bytes_per_token,
            o200k_bytes_per_token=o2_bytes_per_token,
            is_heavily_fragmented=is_heavily_fragmented,
            fragmentation_penalty_score=round(penalty, 3),
        )

    def evaluate_cross_lingual_parity(
        self,
        custom_corpus: Dict[str, str] | None = None
    ) -> Dict[str, Any]:
        """
        Runs a comprehensive cross-lingual parity benchmark comparing
        identical semantic payloads across 8 world languages.
        Calculates exact token inflation multipliers and API cost disparities.
        """
        corpus = custom_corpus or PARALLEL_MULTILINGUAL_CORPUS
        english_text = corpus["English (Latin)"]
        en_cl = self.tokenizer.count_tokens(english_text, "cl100k_base")
        en_o2 = self.tokenizer.count_tokens(english_text, "o200k_base")
        en_words = len(english_text.split())

        comparisons = []
        for lang, text in corpus.items():
            metrics = self.analyze(text)
            cl_inflation = round(metrics.cl100k_tokens / en_cl, 2) if en_cl > 0 else 1.0
            o2_inflation = round(metrics.o200k_tokens / en_o2, 2) if en_o2 > 0 else 1.0
            o2_savings_over_cl = round(
                ((metrics.cl100k_tokens - metrics.o200k_tokens) / metrics.cl100k_tokens) * 100, 2
            ) if metrics.cl100k_tokens > 0 else 0.0

            comparisons.append({
                "language": lang,
                "script": metrics.detected_primary_script,
                "words": metrics.total_words,
                "cl100k_tokens": metrics.cl100k_tokens,
                "o200k_tokens": metrics.o200k_tokens,
                "cl100k_frag_ratio": metrics.cl100k_fragmentation_ratio,
                "o200k_frag_ratio": metrics.o200k_fragmentation_ratio,
                "cl100k_cost_multiplier_vs_en": f"{cl_inflation}x",
                "o200k_cost_multiplier_vs_en": f"{o2_inflation}x",
                "o200k_efficiency_gain_pct": o2_savings_over_cl,
                "is_severely_fragmented": metrics.is_heavily_fragmented,
            })

        return {
            "english_baseline": {
                "words": en_words,
                "cl100k_tokens": en_cl,
                "o200k_tokens": en_o2,
                "cl100k_frag_ratio": round(en_cl / en_words, 2),
            },
            "languages": comparisons,
        }
