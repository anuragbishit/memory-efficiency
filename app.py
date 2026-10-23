"""
Interactive Streamlit Web Dashboard for Minor Project:
Optimizing AI Architecture for Long-Term Memory and Efficiency.
Provides real-time interactive testing for all 9 architectural modules.
"""

import sys
import os
import streamlit as st

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.pipeline import LocalPreProcessingPipeline
from core.compression.compressor import CompressionLevel
from core.routing.optimization.cost_latency_optimizer import RoutingStrategy
from core.representation import IRFormat
from benchmarks.benchmark_tokenization_compression import BENCHMARK_PROMPTS

st.set_page_config(
    page_title="AI Architecture Optimizer - Minor Project",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    .badge-local {
        background-color: #2e7d32;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-cloud {
        background-color: #1565c0;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ AI Architecture Optimization Engine")
st.caption("B.Tech Minor Project | Consumer-Side Pre-Processing, Non-AI Compression & Dynamic Intent Routing")

# Sidebar Controls
st.sidebar.header("⚙️ Pipeline Configuration")

encoding_choice = st.sidebar.selectbox(
    "Tokenizer Vocabulary (Module 1)",
    ["cl100k_base (GPT-4/3.5)", "o200k_base (GPT-4o/Omni)"],
    index=0
)
enc_name = "cl100k_base" if "cl100k" in encoding_choice else "o200k_base"

comp_level_str = st.sidebar.select_slider(
    "Compression Aggressiveness (Module 4 & 5)",
    options=["CONSERVATIVE", "BALANCED", "AGGRESSIVE"],
    value="BALANCED"
)
comp_level = CompressionLevel[comp_level_str]

strategy_str = st.sidebar.selectbox(
    "Routing Optimization Strategy (Module 9)",
    ["BALANCED", "COST_MINIMIZATION", "LATENCY_MINIMIZATION", "QUALITY_MAXIMIZATION"],
    index=0
)
routing_strat = RoutingStrategy[strategy_str]

# Preset selection
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Preset Benchmark Prompts")
preset_options = ["Custom Input"] + [p["category"] for p in BENCHMARK_PROMPTS]
selected_preset = st.sidebar.selectbox("Choose a sample test case:", preset_options)

default_text = "Could you please help me write a Python function that calculates fibonacci numbers? In order to achieve good performance, do not use recursion without memoization."

if selected_preset != "Custom Input":
    for p in BENCHMARK_PROMPTS:
        if p["category"] == selected_preset:
            default_text = p["text"]
            break

# Input Section
st.subheader("1. Input Prompt Payload")
user_input = st.text_area(
    "Enter raw user prompt (supports Python code blocks, Hindi/multilingual scripts, or complex constraints):",
    value=default_text,
    height=140
)

col_run, col_clear = st.columns([1, 5])
run_button = col_run.button("🚀 Process Payload", type="primary")

@st.cache_resource
def get_pipeline():
    return LocalPreProcessingPipeline()

pipeline = get_pipeline()

if run_button or user_input:
    with st.spinner("Processing through 9-stage local pipeline..."):
        pipeline.compression_level = comp_level
        pipeline.default_strategy = routing_strat
        result = pipeline.process(user_input, encoding_name=enc_name)

    # Top KPI Metrics
    st.markdown("---")
    st.subheader("📊 Executive Efficiency & Cost Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Raw Tokens", f"{result.raw_tokens}")
    with m2:
        st.metric("Compressed Tokens", f"{result.compressed_tokens}", f"-{result.tokens_saved} ({result.token_reduction_pct}%)")
    with m3:
        st.metric("Pre-Processing Latency", f"{result.total_pipeline_latency_ms:.1f} ms")
    with m4:
        opt = result.cost_latency_optimization
        st.metric("Query Cost", f"${opt['estimated_query_cost_usd']:.6f}")
    with m5:
        st.metric("Financial Savings", f"{opt['cost_reduction_pct']:.1f}%", f"${opt['immediate_savings_usd']:.5f}")

    # Main Tabs for Module Inspection
    tab_overview, tab_tokens, tab_compression, tab_ir_features, tab_routing = st.tabs([
        "🎯 Routing & Cost Verdict",
        "🔤 Tokenization & Fragmentation",
        "✂️ Compression & AST",
        "🧩 Prompt IR & 12-D ML Features",
        "🧠 Intent Ensemble & Model Registry"
    ])

    with tab_overview:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Selected Execution Target")
            opt = result.cost_latency_optimization
            runtime_badge = "🟢 LOCAL EDGE SLM" if opt["runtime_type"] == "LOCAL_EDGE" else "🔵 CLOUD FRONTIER API"
            st.markdown(f"#### **{opt['selected_model_name']}**")
            st.markdown(f"**Runtime Type:** `{runtime_badge}`")
            st.markdown(f"**Capability Tier:** `{opt['capability_tier']}`")
            st.markdown(f"**Estimated TTFT:** `{opt['estimated_ttft_ms']} ms` | **Total Latency:** `{opt['estimated_total_latency_ms']} ms`")
            st.markdown(f"**Immediate Savings per Query:** `${opt['immediate_savings_usd']:.6f} USD`")
            st.markdown(f"**Projected 10,000 Queries Savings:** `${opt['projected_10k_queries_savings_usd']:,.2f} USD`")

        with c2:
            st.markdown("### Pareto Optimization Rationales")
            for r in opt.get("routing_rationales", []):
                st.info(f"💡 {r}")

        st.markdown("### Alternative Candidate Models Evaluated")
        if "model_comparison_table" in opt:
            st.dataframe(opt["model_comparison_table"], use_container_width=True)

    with tab_tokens:
        st.markdown("### Module 1: Subword Tokenization & Vocabulary Delta")
        subw = result.subword_comparison["encodings"]
        col_tok1, col_tok2 = st.columns(2)
        with col_tok1:
            st.write("**cl100k_base (GPT-4/3.5)**")
            st.write(f"• Token count: `{subw['cl100k_base']['token_count']}` tokens")
            st.write(f"• Ratio: `{subw['cl100k_base']['char_to_token_ratio']} chars/token`")
        with col_tok2:
            st.write("**o200k_base (GPT-4o/Omni)**")
            st.write(f"• Token count: `{subw['o200k_base']['token_count']}` tokens")
            st.write(f"• Ratio: `{subw['o200k_base']['char_to_token_ratio']} chars/token`")

        st.markdown("### Module 2: Multilingual Token Fragmentation")
        frag = result.multilingual_fragmentation
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Primary Script", frag["detected_primary_script"])
        f2.metric("Total Words", frag["total_words"])
        f3.metric("cl100k Ratio", f"{frag['cl100k_fragmentation_ratio']} tok/word")
        f4.metric("Severity", "HEAVY" if frag["is_heavily_fragmented"] else "OPTIMAL")

        st.markdown("### Module 3: Information Density & Perplexity")
        dens = result.information_density_gain
        d1, d2, d3, d4 = st.columns(4)
        raw_p = dens["raw_payload"]
        comp_p = dens["compressed_payload"]
        d1.metric("Entropy H(X)", f"{comp_p['shannon_entropy_bits']:.2f} bits", f"{comp_p['shannon_entropy_bits'] - raw_p['shannon_entropy_bits']:.2f}")
        d2.metric("Token Perplexity", f"{comp_p['token_perplexity']:.1f}")
        d3.metric("Lexical Density", f"{comp_p['lexical_density_pct']:.1f}%", dens["lexical_density_increase_pct"])
        d4.metric("Density Index (IDI)", f"{comp_p['information_density_index']:.2f}", dens["information_density_index_gain_pct"])

    with tab_compression:
        col_orig, col_comp = st.columns(2)
        with col_orig:
            st.markdown("#### Original Payload")
            st.text_area("Raw Text", value=result.raw_prompt, height=220, disabled=True)
        with col_comp:
            st.markdown("#### Compressed Payload (Non-AI)")
            st.text_area("Cleaned Text", value=result.compressed_prompt, height=220, disabled=True)

        st.markdown("### Module 4: Heuristic Rules Applied")
        if result.rules_applied:
            for rule in result.rules_applied:
                st.markdown(f"- `{rule}`")
        else:
            st.write("No redundant stopwords or conversational fluff detected.")

        st.markdown("### Module 5: AST Code Compression")
        if result.ast_code_summaries:
            for s in result.ast_code_summaries:
                st.success(f"✅ AST Validated for `{s['language'].upper()}` | Semantic Equivalence: **Verified (Zero Logic Mutation)** | Saved `{s['tokens_saved']}` tokens ({s['compression_ratio_pct']}%)")
        else:
            st.info("No code blocks detected in current prompt payload.")

    with tab_ir_features:
        st.markdown("### Module 6: Intermediate Representation (IR)")
        ir_data = result.prompt_ir
        col_ir1, col_ir2 = st.columns(2)
        with col_ir1:
            st.markdown(f"**Atomic Intent:** `{ir_data['intent']}`")
            st.markdown(f"**Action:** `{ir_data['action']}`")
            st.markdown(f"**Target:** `{ir_data['target']}`")
            if ir_data.get("constraints"):
                st.markdown(f"**Extracted Constraints:** `{ir_data['constraints']}`")
        with col_ir2:
            st.markdown(f"**IR Token Count:** `{ir_data['ir_tokens']}` (vs {ir_data['original_tokens']} raw)")

        generator = pipeline.ir_generator
        rec_ir = generator.generate_ir(result.compressed_prompt, encoding_name=enc_name)
        st.code(rec_ir.serialize(IRFormat.TAGGED_KV), language="yaml")

        st.markdown("### Module 7: 12-Dimensional ML Feature Vector")
        labels = [
            "Length Log", "TTR Diversity", "Lexical Density", "Code Block",
            "Code Ratio", "Constraint Density", "Syntactic Complexity",
            "Question Score", "Imperative Score", "Reasoning Depth",
            "Technical Vocabulary", "Fluff Density"
        ]
        vector_data = dict(zip(labels, [round(x, 3) for x in result.ml_feature_vector]))
        st.bar_chart(vector_data)

    with tab_routing:
        st.markdown("### Module 8: Lightweight Intent Classification Ensemble")
        routing = result.routing_decision
        ens = routing.get("ensemble_classification", {})

        col_ens1, col_ens2 = st.columns(2)
        with col_ens1:
            st.markdown(f"#### Consensus Intent: **{ens.get('consensus_intent')}**")
            st.markdown(f"**Confidence:** `{round(ens.get('confidence', 0) * 100, 1)}%`")
            st.markdown(f"**Fast-Path Deterministic Match:** `{'Yes (< 0.1ms)' if ens.get('fast_path_triggered') else 'No (Ensemble blended)'}`")
            st.markdown(f"**Complexity Score:** `{routing['complexity_score']:.3f} / 1.000` (`{routing['tier']}` Tier)")

        with col_ens2:
            st.markdown("#### Intent Ensemble Probabilities")
            if "ensemble_probabilities" in ens:
                st.bar_chart(ens["ensemble_probabilities"])
