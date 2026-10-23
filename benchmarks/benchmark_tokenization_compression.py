"""
Empirical Benchmark Suite for Tokenization, Algorithmic Compression, and Intent Routing.
Runs evaluation across real-world prompts (conversational, code-heavy, reasoning, multilingual).
"""

import os
import sys
import time
import json
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Enable UTF-8 console output for cross-lingual scripts on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.pipeline import LocalPreProcessingPipeline
from core.compression.compressor import CompressionLevel


BENCHMARK_PROMPTS = [
    {
        "id": "PROMPT_01_CONVERSATIONAL",
        "category": "Conversational & Polite Fluff",
        "text": (
            "Hello there, could you please kindly assist me in understanding how to sort an array "
            "in Python? I was wondering if you might be able to explain it simply for me. "
            "Thank you so much in advance, I really appreciate your help!"
        )
    },
    {
        "id": "PROMPT_02_CODE_AND_CONSTRAINTS",
        "category": "Code Optimization & Strict Constraints",
        "text": (
            "Please make an attempt to perform an analysis on this function at the present time. "
            "You must not use any recursion, and you cannot import external third-party libraries. "
            "It must strictly achieve O(1) space complexity.\n"
            "```python\n"
            "def find_duplicates(nums: list[int]) -> list[int]:\n"
            '    """Identifies all duplicate integers within a list in linear time."""\n'
            "    # Initialize tracking structures\n"
            "    seen = set()\n"
            "    dups = []\n"
            "    for x in nums:\n"
            "        if x in seen:\n"
            "            dups.append(x)\n"
            "        seen.add(x)\n"
            "    return dups\n"
            "```\n"
            "Thanks a lot!"
        )
    },
    {
        "id": "PROMPT_03_MULTIHOP_REASONING",
        "category": "Deep Analytical & Architectural Reasoning",
        "text": (
            "Analyze the root cause of context bleed in traditional vector-chunk RAG architectures. "
            "Compare and contrast how GraphRAG resolves relational ambiguity using atomic entity nodes "
            "and typed edges. Explain the step-by-step mathematical tradeoffs between vector similarity "
            "search and multi-hop graph traversal."
        )
    },
    {
        "id": "PROMPT_04_MULTILINGUAL_HINDI",
        "category": "Multilingual BPE Token Fragmentation (Hindi)",
        "text": (
            "नमस्ते, क्या आप मुझे बता सकते हैं कि कंप्यूटर साइंस में ग्राफ डेटा संरचना क्या होती है "
            "और यह फ्लैट वेक्टर एम्बेडिंग से कैसे बेहतर है? कृपया विस्तार से समझाइए।"
        )
    },
    {
        "id": "PROMPT_05_SIMPLE_FACTUAL",
        "category": "Low Complexity / Factual Lookup",
        "text": "What is the capital of France?"
    },
    {
        "id": "PROMPT_06_SYSTEM_DESIGN",
        "category": "Distributed Systems & Concurrency",
        "text": (
            "Architect a distributed sharded cache system supporting thread-safe writes, "
            "high concurrency, and consistent hashing. Must handle node failure and data replication "
            "without losing consistency. Provide class structure."
        )
    }
]


def run_benchmark():
    print("=" * 95)
    print(" " * 20 + "MINOR PROJECT: EMPIRICAL BENCHMARK SUITE")
    print(" " * 15 + "Tokenization, Non-AI Compression & Weighted Intent Routing")
    print("=" * 95)

    pipeline = LocalPreProcessingPipeline(
        default_encoding="cl100k_base",
        compression_level=CompressionLevel.BALANCED
    )

    total_raw_tokens = 0
    total_compressed_tokens = 0
    total_latency = 0.0
    total_cost_usd = 0.0
    total_baseline_cost_usd = 0.0

    results_table: List[Dict[str, Any]] = []

    for item in BENCHMARK_PROMPTS:
        pid = item["id"]
        cat = item["category"]
        text = item["text"]

        processed = pipeline.process(text)

        total_raw_tokens += processed.raw_tokens
        total_compressed_tokens += processed.compressed_tokens
        total_latency += processed.total_pipeline_latency_ms

        cost_info = processed.cost_latency_optimization
        total_cost_usd += cost_info.get("estimated_query_cost_usd", 0.0)
        total_baseline_cost_usd += cost_info.get("baseline_unoptimized_cost_usd", 0.0)

        route = processed.routing_decision

        results_table.append({
            "id": pid,
            "category": cat,
            "raw_tokens": processed.raw_tokens,
            "compressed_tokens": processed.compressed_tokens,
            "reduction_pct": processed.token_reduction_pct,
            "complexity_score": route["complexity_score"],
            "tier": route["tier"],
            "model": route.get("selected_model_name", route["target_runtime"])[:24],
            "cost_usd": f"${cost_info.get('estimated_query_cost_usd', 0.0):.5f}",
            "savings_pct": f"{cost_info.get('cost_reduction_pct', 0.0)}%",
            "latency_ms": processed.total_pipeline_latency_ms,
        })

    print(f"\n{'ID':<26} | {'Category':<16} | {'Tokens':<9} | {'Model Selected':<24} | {'Cost ($)':<9} | {'Cost Saved':<10} | {'Latency':<7}")
    print("-" * 110)

    for r in results_table:
        token_str = f"{r['raw_tokens']} -> {r['compressed_tokens']}"
        lat_str = f"{r['latency_ms']}ms"
        print(f"{r['id']:<26} | {r['category'][:16]:<16} | {token_str:<9} | {r['model']:<24} | {r['cost_usd']:<9} | {r['savings_pct']:<10} | {lat_str:<7}")

    overall_token_reduction = round(
        ((total_raw_tokens - total_compressed_tokens) / total_raw_tokens) * 100, 2
    ) if total_raw_tokens > 0 else 0.0
    overall_cost_reduction = round(
        ((total_baseline_cost_usd - total_cost_usd) / total_baseline_cost_usd) * 100, 2
    ) if total_baseline_cost_usd > 0 else 0.0
    net_dollar_savings = max(0.0, round(total_baseline_cost_usd - total_cost_usd, 5))
    avg_latency = round(total_latency / len(BENCHMARK_PROMPTS), 3)

    print("-" * 110)
    print(f"TOTAL PAYLOAD TOKENS PROCESSED:      {total_raw_tokens} tokens")
    print(f"TOTAL COMPRESSED TOKENS:             {total_compressed_tokens} tokens ({overall_token_reduction}% token compression)")
    print(f"ACTUAL OPTIMIZED FINANCIAL COST:     ${total_cost_usd:.5f} USD")
    print(f"BASELINE UNOPTIMIZED (RAW GPT-4o):   ${total_baseline_cost_usd:.5f} USD")
    print(f"NET MONETARY SAVINGS:                ${net_dollar_savings:.5f} USD ({overall_cost_reduction}% cost reduction)")
    print(f"PROJECTED SAVINGS PER 10,000 QUERIES:${net_dollar_savings * 10000:,.2f} USD")
    print(f"AVERAGE PRE-PROCESSING LATENCY:      {avg_latency} ms (Pure CPU, Zero Neural Overhead)")
    print("=" * 110)

    # Save benchmark results to JSON
    with open("benchmarks/latest_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_prompts": len(BENCHMARK_PROMPTS),
                "total_raw_tokens": total_raw_tokens,
                "total_compressed_tokens": total_compressed_tokens,
                "net_token_reduction_pct": overall_token_reduction,
                "actual_cost_usd": total_cost_usd,
                "baseline_cost_usd": total_baseline_cost_usd,
                "cost_reduction_pct": overall_cost_reduction,
                "net_dollar_savings_usd": net_dollar_savings,
                "projected_10k_savings_usd": net_dollar_savings * 10000,
                "avg_latency_ms": avg_latency
            },
            "details": results_table
        }, f, indent=2)

    print("\n[+] Benchmark metrics successfully recorded to 'benchmarks/latest_benchmark_results.json'")


if __name__ == "__main__":
    run_benchmark()
