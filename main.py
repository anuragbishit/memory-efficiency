"""
Interactive Demonstration CLI for Minor Project:
Optimizing AI Architecture for Long-Term Memory and Efficiency.
Covers:
  - Subword Tokenization (BPE, cl100k_base, o200k_base)
  - Non-AI Algorithmic Token Compression
  - Weighted Intent Recognition and Dynamic Routing
"""

import sys
import os
import argparse
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Enable UTF-8 console output for cross-lingual scripts on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.pipeline import LocalPreProcessingPipeline
from core.compression.compressor import CompressionLevel
from core.tokenization.tokenizer import SubwordTokenizer
from core.tokenization.analyzer import TokenAnalyzer
from core.tokenization.fragmentation import MultilingualFragmentationAnalyzer
from core.compression.density import InformationDensityAnalyzer
from core.representation import PromptIRGenerator, IRFormat
from core.routing import RoutingStrategy, DEFAULT_MODEL_REGISTRY
from benchmarks.benchmark_tokenization_compression import BENCHMARK_PROMPTS, run_benchmark


def print_banner():
    banner = r"""
================================================================================
   OPTIMIZING AI ARCHITECTURE FOR LONG-TERM MEMORY AND EFFICIENCY
   Consumer-Side Pre-Processing, Compression & Intent Routing Layer
   B.Tech Minor Project | Shivalik College of Engineering, Dehradun
================================================================================
"""
    print(banner)


def format_token_chips(details, max_display=12):
    """Formats subword tokens visually as chips/fragments."""
    chips = []
    for d in details[:max_display]:
        text_repr = repr(d.text)[1:-1]
        chips.append(f"[{d.token_id}: '{text_repr}']")
    if len(details) > max_display:
        chips.append(f"... (+{len(details) - max_display} more tokens)")
    return " ".join(chips)


def process_and_display(pipeline: LocalPreProcessingPipeline, prompt_text: str, encoding: str = "cl100k_base"):
    print("\n" + "=" * 80)
    print(" INPUT PAYLOAD (RAW)")
    print("=" * 80)
    print(prompt_text.strip())

    result = pipeline.process(prompt_text, encoding_name=encoding)
    route = result.routing_decision
    subword = result.subword_comparison
    frag = result.multilingual_fragmentation
    density = result.information_density_gain

    print("\n" + "=" * 80)
    print(" COMPRESSED PAYLOAD (NON-AI DETERMINISTIC OPTIMIZATION)")
    print("=" * 80)
    print(result.compressed_prompt.strip())

    print("\n" + "=" * 80)
    print(" METRICS & EFFICIENCY SUMMARY")
    print("=" * 80)
    print(f"  * Encoding Engine:             {result.encoding_used}")
    print(f"  * Raw Token Count:             {result.raw_tokens} tokens")
    print(f"  * Compressed Token Count:      {result.compressed_tokens} tokens")
    print(f"  * Tokens Saved:                {result.tokens_saved} tokens ({result.token_reduction_pct}%)")
    print(f"  * Pre-Processing Latency:      {result.total_pipeline_latency_ms} ms")

    print("\n" + "=" * 80)
    print(" MULTILINGUAL TOKEN FRAGMENTATION & SCRIPT ANALYSIS")
    print("=" * 80)
    print(f"  * Primary Script Detected:     {frag['detected_primary_script']}")
    print(f"  * Total Words / Chars:         {frag['total_words']} words | {frag['total_characters']} characters ({frag['total_utf8_bytes']} UTF-8 bytes)")
    print(f"  * cl100k Fragmentation Ratio:  {frag['cl100k_fragmentation_ratio']} tokens/word ({frag['cl100k_bytes_per_token']} bytes/token)")
    print(f"  * o200k Fragmentation Ratio:   {frag['o200k_fragmentation_ratio']} tokens/word ({frag['o200k_bytes_per_token']} bytes/token)")
    print(f"  * Fragmentation Severity:      {'[!] HEAVILY FRAGMENTED' if frag['is_heavily_fragmented'] else 'NORMAL / OPTIMAL'}")
    print(f"  * Subword Penalty Score:       {frag['fragmentation_penalty_score']} / 1.000")

    print("\n" + "=" * 80)
    print(" INFORMATION DENSITY & TOKEN PERPLEXITY")
    print("=" * 80)
    raw_p = density["raw_payload"]
    comp_p = density["compressed_payload"]
    print(f"  * Shannon Entropy H(X):        {raw_p['shannon_entropy_bits']} bits -> {comp_p['shannon_entropy_bits']} bits/token")
    print(f"  * Empirical Token Perplexity:  {raw_p['token_perplexity']} -> {comp_p['token_perplexity']}")
    print(f"  * Lexical Density:             {raw_p['lexical_density_pct']}% -> {comp_p['lexical_density_pct']}% ({density['lexical_density_increase_pct']})")
    print(f"  * Information Density Index:   {raw_p['information_density_index']} -> {comp_p['information_density_index']} ({density['information_density_index_gain_pct']})")
    print(f"  * Linguistic Coherence:        Score: {comp_p['estimated_coherence_score']} | Preserved: {density['coherence_maintained']}")

    print("\n" + "=" * 80)
    print(" SUBWORD BPE VOCABULARY COMPARISON (cl100k_base vs o200k_base)")
    print("=" * 80)
    cl_info = subword["encodings"]["cl100k_base"]
    o2_info = subword["encodings"]["o200k_base"]
    print(f"  * cl100k_base (GPT-4/3.5):     {cl_info['token_count']} tokens | {cl_info['char_to_token_ratio']} chars/token")
    print(f"  * o200k_base  (GPT-4o/Omni):   {o2_info['token_count']} tokens | {o2_info['char_to_token_ratio']} chars/token")
    print(f"  * BPE Vocabulary Advantage:    {subword['o200k_vs_cl100k_delta_percent']}% token shift with o200k_base")

    # Subword breakdown preview
    details = pipeline.tokenizer.inspect_tokens(result.compressed_prompt, encoding)
    print(f"  * Subword Token Fragments:     {format_token_chips(details)}")

    if result.ast_code_summaries:
        print("\n" + "=" * 80)
        print(" AST CODE COMPRESSION & SEMANTIC EQUIVALENCE (TOPIC 5)")
        print("=" * 80)
        for idx, ast_s in enumerate(result.ast_code_summaries, 1):
            print(f"  * Code Block [{idx}]:           Language: {ast_s['language'].upper()} | AST Parsed: {ast_s['ast_parsed']}")
            print(f"  * Semantic Equivalence:        >>> {ast_s['semantic_equivalence_verified']} (Zero Logic Mutation) <<<")
            print(f"  * Code Token Shift:            {ast_s['raw_tokens']} -> {ast_s['compressed_tokens']} tokens (Saved {ast_s['tokens_saved']} tokens, {ast_s['compression_ratio_pct']}%)")

    if result.rules_applied:
        print("\n" + "=" * 80)
        print(" RULE-BASED COMPRESSION & STOPWORD TRANSFORMS (TOPIC 4)")
        print("=" * 80)
        for rule_msg in result.rules_applied[:6]:
            print(f"  * {rule_msg}")
        if len(result.rules_applied) > 6:
            print(f"  * ... (+{len(result.rules_applied) - 6} additional heuristic transforms)")

    print("\n" + "=" * 80)
    print(" INTERMEDIATE REPRESENTATION (IR) FOR COMPACT PROMPTS (TOPIC 6)")
    print("=" * 80)
    ir_data = result.prompt_ir
    print(f"  * Atomic Intent:               {ir_data['intent']} -> Action: {ir_data['action']} | Target: {ir_data['target']}")
    if ir_data.get("constraints"):
        print(f"  * Extracted Constraints:       {'; '.join(ir_data['constraints'])}")
    print(f"  * IR Token Shift:              {ir_data['original_tokens']} raw -> {ir_data['ir_tokens']} IR tokens ({ir_data['token_savings_pct']}% savings)")
    print("  * Serialized Tagged-KV IR:")
    # Pretty print indented IR lines
    generator = PromptIRGenerator()
    reconstructed_ir = generator.generate_ir(result.compressed_prompt, encoding_name=encoding)
    for line in reconstructed_ir.serialize(IRFormat.TAGGED_KV).split("\n"):
        print(f"      {line}")

    print("\n" + "=" * 80)
    print(" MULTI-DIMENSIONAL PROMPT FEATURE VECTOR (TOPIC 7)")
    print("=" * 80)
    extracted_features = pipeline.router.feature_extractor.extract(prompt_text, result.raw_tokens)
    print(f"  * Interrogative Type:          {extracted_features.question_type}")
    print(f"  * Imperative Commands:         {extracted_features.imperative_command_count}")
    print(f"  * Type-Token Ratio (TTR):      {extracted_features.type_token_ratio} (Vocabulary Diversity)")
    print(f"  * Fluff / Filler Density:      {extracted_features.fluff_density_score}")
    print(f"  * 12-D Normalized ML Vector:   {[round(x, 3) for x in result.ml_feature_vector]}")

    print("\n" + "=" * 80)
    print(" WEIGHTED INTENT RECOGNITION & ROUTING VERDICT (PILLAR 2)")
    print("=" * 80)
    ens = route.get("ensemble_classification", {})
    if ens:
        print(f"  * Ensemble Intent Consensus:   >>> {ens.get('consensus_intent')} <<< (Confidence: {round(ens.get('confidence', 0)*100, 1)}%)")
        print(f"  * Fast-Path Fast Triage:       {'[!] TRIGGERED (Deterministic Rule Match)' if ens.get('fast_path_triggered') else 'Statistical / Transformer Blended Consensus'}")

        tfidf_info = ens.get("tfidf_result", {})
        if tfidf_info.get("probabilities"):
            top_prob = sorted(tfidf_info["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
            prob_str = ", ".join(f"{k}: {round(v*100, 1)}%" for k, v in top_prob)
            print(f"  * TF-IDF + Logistic Prob:      {prob_str}")

        trans_info = ens.get("transformer_result", {})
        if trans_info.get("intent_scores"):
            top_trans = sorted(trans_info["intent_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
            trans_str = ", ".join(f"{k}: {round(v*100, 1)}%" for k, v in top_trans)
            print(f"  * Distilled Transformer Match: {trans_str} ({trans_info.get('distilled_model_name')})")

    print(f"  * Complexity Score:            {route['complexity_score']} / 1.000")
    print(f"  * Complexity Tier:             {route['tier']}")
    print(f"  * Target Model Destination:    >>> {route.get('selected_model_name', route['target_runtime'])} <<<")
    print(f"  * Compute Token Budget:        {route['compute_budget_tokens']} max tokens")
    print(f"  * Graph Retrieval Depth:       {route['graph_retrieval_depth_hops']} hop(s) subgraph traversal")

    cost_opt = result.cost_latency_optimization
    if cost_opt:
        print("\n" + "=" * 80)
        print(" DYNAMIC MODEL ROUTING & COST-LATENCY OPTIMIZATION (TOPIC 8)")
        print("=" * 80)
        print(f"  * Selected Execution Engine:   {cost_opt['selected_model_name']} ({cost_opt['runtime_type']})")
        print(f"  * Optimization Strategy:       {cost_opt['strategy_used']}")
        print(f"  * Estimated Query Cost:        ${cost_opt['estimated_query_cost_usd']:.6f} USD")
        print(f"  * Baseline Raw GPT-4o Cost:    ${cost_opt['baseline_unoptimized_cost_usd']:.6f} USD")
        print(f"  * Immediate Financial Savings: ${cost_opt['immediate_savings_usd']:.6f} USD ({cost_opt['cost_reduction_pct']}% reduction)")
        print(f"  * Projected 10k Query Savings: ${cost_opt['projected_10k_queries_savings_usd']:,.2f} USD")
        print(f"  * Estimated Generation Latency:{cost_opt['estimated_total_latency_ms']} ms (TTFT: {cost_opt['estimated_ttft_ms']} ms)")
        print("  * Pareto Optimization Rationales:")
        for rat in cost_opt.get("routing_rationales", []):
            print(f"      - {rat}")

    print("\n" + "=" * 80)
    print(" FEATURE SIGNALS & COMPUTE DECISION RATIONALES")
    print("=" * 80)
    for k, v in route["feature_breakdown"].items():
        print(f"  * {k:<20}: {v}")
    print("  * Decision Rationales:")
    for r in route["rationales"]:
        print(f"      - {r}")
    print("=" * 80 + "\n")


def interactive_menu():
    print_banner()
    pipeline = LocalPreProcessingPipeline(
        default_encoding="cl100k_base",
        compression_level=CompressionLevel.BALANCED
    )

    while True:
        print("\nSelect an option:")
        print(" [1] Enter custom prompt for real-time analysis")
        print(" [2] Run preset benchmark prompt (Conversational, Code, Multilingual, Reasoning)")
        print(" [3] Run full empirical benchmark suite across all test cases")
        print(" [4] Run 8-Language Cross-Lingual Parity & Cost Disparity Benchmark")
        print(" [5] Exit")

        choice = input("\nEnter choice [1-5]: ").strip()

        if choice == "1":
            user_text = input("\nEnter prompt: ").strip()
            if user_text:
                process_and_display(pipeline, user_text)
        elif choice == "2":
            print("\nAvailable Presets:")
            for idx, p in enumerate(BENCHMARK_PROMPTS, 1):
                print(f"  [{idx}] {p['category']}")
            p_choice = input(f"Select preset [1-{len(BENCHMARK_PROMPTS)}]: ").strip()
            try:
                idx = int(p_choice) - 1
                if 0 <= idx < len(BENCHMARK_PROMPTS):
                    process_and_display(pipeline, BENCHMARK_PROMPTS[idx]["text"])
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
        elif choice == "3":
            run_benchmark()
        elif choice == "4":
            print("\n" + "=" * 95)
            print(" 8-LANGUAGE CROSS-LINGUAL TOKEN FRAGMENTATION & ECONOMIC PARITY BENCHMARK")
            print("=" * 95)
            frag_analyzer = MultilingualFragmentationAnalyzer(pipeline.tokenizer)
            report = frag_analyzer.evaluate_cross_lingual_parity()
            base = report["english_baseline"]
            print(f"English Baseline: {base['words']} words | {base['cl100k_tokens']} tokens (cl100k) | Frag Ratio: {base['cl100k_frag_ratio']} tokens/word\n")

            print(f"{'Language':<22} | {'Script':<20} | {'Tokens (cl)':<11} | {'Tokens (o2)':<11} | {'Cost vs EN':<10} | {'o2 Gain':<9} | {'Severely Frag?'}")
            print("-" * 95)
            for item in report["languages"]:
                flag_str = "[!] YES" if item["is_severely_fragmented"] else "No"
                print(
                    f"{item['language'][:22]:<22} | "
                    f"{item['script'][:20]:<20} | "
                    f"{item['cl100k_tokens']:<11} | "
                    f"{item['o200k_tokens']:<11} | "
                    f"{item['cl100k_cost_multiplier_vs_en']:<10} | "
                    f"{item['o200k_efficiency_gain_pct']}%{'':<3} | "
                    f"{flag_str}"
                )
            print("=" * 95)
            print("Insight: Non-Latin scripts pay up to 4x more per query in cl100k_base.")
            print("         o200k_base dramatically restores parity by expanding non-Latin vocabulary allocations.")
            print("=" * 95)
        elif choice == "5":
            print("Exiting. Thank you!")
            break
        else:
            print("Invalid option. Please choose 1 to 5.")


def main():
    parser = argparse.ArgumentParser(description="AI Architecture Tokenization, Compression & Intent Routing Layer")
    parser.add_argument("--prompt", type=str, help="Single prompt string to process")
    parser.add_argument("--benchmark", action="store_true", help="Run empirical benchmark suite")
    parser.add_argument("--encoding", type=str, default="cl100k_base", choices=["cl100k_base", "o200k_base"])

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.prompt:
        pipeline = LocalPreProcessingPipeline(default_encoding=args.encoding)
        process_and_display(pipeline, args.prompt, encoding=args.encoding)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
