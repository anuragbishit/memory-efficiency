# Optimizing AI Architecture for Long-Term Memory and Efficiency
## Consumer-Side Pre-Processing: Subword Tokenization, Non-AI Compression & Weighted Intent Routing

**Minor Project** — Bachelor of Technology in Computer Science & Engineering (AI & ML)  
**Institution:** Shivalik College of Engineering, Dehradun  

---

### Overview
This repository contains the production implementation of the **Consumer-Side Pre-Processing Layer** (Stages 1, 3, and 4 from the Minor Project Architecture). 

It sits between the user interface and LLM runtimes (Local Edge SLMs and Upstream Cloud APIs) to eliminate token bloat, mitigate multilingual BPE token fragmentation, and dynamically triage execution complexity without secondary neural model latency or cost.

---

### Core Modules Implemented

#### 1. Subword Tokenization & Multilingual Fragmentation (`core/tokenization/`)
* **Engines:** Full Byte-Pair Encoding (BPE) support for **`cl100k_base`** (GPT-4 / GPT-3.5) and **`o200k_base`** (GPT-4o / Omni).
* **Token Inspector:** Decodes individual token IDs, raw byte representations, subword splits, and multi-byte indicators.
* **Multilingual Fragmentation Analyzer (`fragmentation.py`):**
  * Unicode script detection (Latin, Devanagari, Bengali, Arabic, Cyrillic, CJK).
  * Word-to-token fragmentation ratio $F = \frac{N_{\text{tokens}}}{N_{\text{words}}}$.
  * Subword penalty scoring and economic cost multiplier benchmarks across 8 languages.
  * Shows how `cl100k_base` inflates Hindi by **5.6x** and Bengali by **7.3x** vs English, and how `o200k_base` recovers **up to 68% token efficiency**.
* **Educational BPE:** Pure-Python reference merge learner for academic verification and viva demonstrations.

#### 2. Non-AI Token Compression, AST Code Preservation & Density (`core/compression/`)
* **Zero-Neural Overhead:** Runs in **< 1 millisecond** using deterministic linguistic heuristics.
* **AST Code Compression & Semantic Preservation (`ast_code_compressor.py` - Topic 5):**
  * Parses Python code blocks into an Abstract Syntax Tree (`ast.parse`).
  * Strips redundant docstrings and non-essential comments while retaining function signatures, arguments, assignments, and control flow.
  * Re-generates minimal, canonical code via `ast.unparse()`.
  * Verifies mathematical AST semantic equivalence (`_verify_ast_equivalence()`) to guarantee zero logic mutation.
  * Provides language-agnostic lexical comment pruning for non-Python blocks (JS, TS, SQL, C++, Go).
* **Rule-Based Text Compression & Stopword Removal (`rule_compressor.py` - Topic 4):**
  * Compacts multi-word pleonasms and wordy constructions (*"make an attempt to perform an analysis on"* $\rightarrow$ *"attempt to analyze"*, *"at the present time"* $\rightarrow$ *"now"*).
  * Context-aware stopword removal strictly guarding Question Markers (*why, how, what*), Negations (*not, never, without*), and Constraints (*must, only, strict, O(1)*).
* **Information Density & Perplexity Analyzer (`density.py` - Topic 3):**
  * **Shannon Entropy:** $H(X) = -\sum p(t_i) \log_2 p(t_i)$ measuring bits per token.
  * **Empirical Token Perplexity:** $\text{PPL} = 2^{H(X)}$ tracking token distribution focus.
  * **Lexical Density:** $\frac{\text{Content Words}}{\text{Total Words}} \times 100\%$ tracking technical signal density.
  * **Information Density Index (IDI):** Measures how effectively information is packed per token post-compression.
  * **Linguistic Coherence:** Evaluates structural transitions to guarantee prompts remain smooth and grammatical.

#### 3. Intermediate Representation (IR) for Compact Prompts (`core/representation/` - Topic 6)
* **Tagged-KV, Compact-DSL, and Minimal-JSON IR:** Translates unstructured natural language prompts into language-neutral, schema-driven intermediate slots (`[TASK: ...]`, `[CONSTRAINTS: ...]`, `[INPUT_CODE: ...]`, `[CONTEXT: ...]`).
* **Entropy Elimination:** Eliminates conversational filler, resolves prompt ambiguity, and guides downstream LLM attention directly to verified constraints and entities.

#### 4. Comprehensive Prompt Feature Extraction (`core/routing/feature_extractor.py` - Topic 7)
* **20+ Multi-Dimensional Signals:**
  * **Lexical:** Token count, character-to-token ratio, Type-Token Ratio (TTR vocabulary richness), sentence length.
  * **Syntactic:** Question classification (`FACTUAL`, `PROCESS`, `CAUSAL`, `COMPARATIVE`, `DIRECTIVE`), imperative verb counts.
  * **Domain Complexity:** Code block density, reasoning density, explicit constraint density, negation counts.
  * **Information-Theoretic:** Shannon Entropy, empirical token perplexity, lexical density, fluff density.
* **12-Dimensional ML-Ready Vector:** Emits normalized vector $\in [0.0, 1.0]^{12}$ for machine-learning triage and clustering classifiers.

#### 5. Lightweight Intent Classification & Hybrid Ensemble (`core/routing/classifiers/`)
* **Tri-Paradigm Hybrid Ensemble:**
  * **Deterministic Rule Engine (`rule_classifier.py`):** Sub-0.1ms immediate fast-path for unambiguous patterns (fenced code blocks, exception stack traces, trivial greetings).
  * **TF-IDF + Logistic Regression (`tfidf_logistic_classifier.py`):** N-gram statistical classification emitting calibrated probabilities over 5 core technical intents (`FACTUAL_LOOKUP`, `EXPLANATION`, `CODE_GENERATION`, `SYSTEM_ARCHITECTURE`, `DEBUGGING`).
  * **Distilled Transformer Semantic Scorer (`distilled_transformer_classifier.py`):** Subword-aware dense semantic prototype alignment modeled after DistilBERT / MiniLM running in < 2ms on CPU.
  * **Consensus Blender (`hybrid_ensemble_classifier.py`):** Integrates rules, statistical probabilities, and distilled transformer scores to deliver stable routing decisions.
* **Complexity-Driven Dynamic Routing (`intent_engine.py`):**
  * **LOW Tier (< 0.35):** Factual queries / greetings $\rightarrow$ `LOCAL_SLM` (256-384 tokens, 0-hop retrieval).
  * **MEDIUM Tier (0.35 - 0.70):** Single-step logic / conceptual tasks $\rightarrow$ `LOCAL_SLM` (1024 tokens, 1-hop subgraph traversal).
  * **HIGH Tier (> 0.70):** Code generation / multi-hop architecture / bug fixing $\rightarrow$ `CLOUD_API` (4096 tokens, deep graph extraction).

---

### Quick Start

#### 1. Run Unit Tests (40 Test Cases)
```bash
python -m pytest tests -v
```

#### 2. Run Empirical Benchmarks
```bash
python benchmarks/benchmark_tokenization_compression.py
```

#### 3. Run Interactive CLI
```bash
python main.py
```

#### 4. Run CLI on Custom Prompt via Argument
```bash
python main.py --prompt "Could you please kindly help me optimize this function? You must not use recursion: def fib(n): return n"
```

---

### Empirical Benchmark Summary

| Category | Input Tokens | Compressed Tokens | Savings (%) | Tier | Destination | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Conversational Fluff | 47 | 22 | **53.19%** | LOW | LOCAL_SLM | ~0.5 ms |
| Code & Strict Constraints | 93 | 87 | **6.45%** | HIGH | CLOUD_API | ~0.6 ms |
| Multi-Hop Reasoning | 53 | 53 | **0.00%** | HIGH | CLOUD_API | ~0.5 ms |
| Multilingual (Hindi) | 155 | 129 | **16.77%** | LOW | LOCAL_SLM | ~0.8 ms |
| Simple Factual | 7 | 7 | **0.00%** | LOW | LOCAL_SLM | ~0.1 ms |
| Distributed System Design | 35 | 35 | **0.00%** | MEDIUM | LOCAL_SLM | ~0.3 ms |

*Note: On Hindi text, upgrading the tokenizer from `cl100k_base` (155 tokens) to `o200k_base` yields an additional **67.1% token reduction (down to 51 tokens)**!*
