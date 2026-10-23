# Optimizing AI Architecture for Long-Term Memory and Efficiency

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)](tests/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](app.py)
[![License](https://img.shields.io/badge/license-Academic%20Research-purple.svg)](README.md)

**A B.Tech Minor Project in Computer Science and Engineering (Artificial Intelligence & Machine Learning)**  
**Department of Computer Science & Engineering, Shivalik College of Engineering, Dehradun**  
*Academic Session: 2026*

* **Supervised By:** Ms. Swati Kashyap (Assistant Professor, Department of CSE)
* **Submitted By:**
  1. Arpit Bisht (230410101034)
  2. Anurag Kumar Singh (230410101032)
  3. Amandeep Kumar (230410101016)
  4. Anurag Yadav (230410101033)

---

## 📌 Executive Summary

Commercial Large Language Model APIs (e.g., OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet) monetize via linear per-token billing. Because providers earn more revenue when prompts are verbose, they lack financial incentive to compress user payloads. Furthermore, standard Byte-Pair Encoding (BPE) vocabularies severely fragment non-Latin scripts (e.g., Hindi suffers a **5.6× token inflation** vs English for identical semantic content).

This repository contains the complete production-grade implementation of the **Consumer-Side Pre-Processing, Algorithmic Non-AI Compression, and Dynamic Intent Routing Pipeline**. Operating entirely on local client hardware before network requests are dispatched, this system intercepts raw user prompts and achieves:
- **Up to 53% prompt token reduction** without secondary neural network calls or summarization latency.
- **Sub-1 millisecond deterministic execution** using Abstract Syntax Tree (AST) preservation and rule heuristics.
- **Zero code mutation guarantee** verified mathematically via structural AST diffing.
- **Over 95% financial cost reduction** via multi-objective Pareto model routing (biasing simple/medium queries to \$0.00 local edge SLMs and reserving expensive frontier cloud APIs exclusively for high-complexity tasks).

---

## 🏗️ End-to-End System Architecture

```text
                                   [ User Prompt Input ]
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      │                                                                             │
      ▼                                                                             ▼
[Module 7] Feature Extractor                                          [Module 1] Subword Tokenizer
• 20+ Lexical/Syntactic Signals                                       • BPE cl100k_base vs o200k_base
• Interrogative Categorizer                                           • Subword Inspection & Bytes
• 12-D ML-Ready Feature Vector                                        • Offline Local Disk Cache
      │                                                                             │
      ▼                                                                             ▼
[Module 8] Lightweight Intent Ensemble                                [Module 2] Multilingual Fragmentation
• Deterministic Rule Fast-Path (<0.1ms)                               • Unicode Script Identification
• Sublinear TF-IDF + Logistic Classifier                              • Fragmentation Ratio F = Tokens/Words
• Distilled Transformer Semantic Prototype Scorer                     • 8-Language Economic Parity Benchmark
      │                                                                             │
      └──────────────────────────────────────┬──────────────────────────────────────┘
                                             │
                                             ▼
                        [Module 4 & 5] Non-AI Compression Engine
                        ├── AST Code Compressor: Docstring & comment stripper with
                        │   zero-mutation AST equivalence verification (Python, C/C++, Java,
                        │   Rust, Go, JS/TS, C#, SQL, Bash, Ruby, and custom languages)
                        └── Rule-Based Text Compressor: Pleonasm compaction, fluff stripping,
                            and strict Semantic Anchor Whitelist (protects NOT, MUST, O(1))
                                             │
                                             ▼
                        [Module 3] Density & Perplexity Analyzer
                        • Shannon Entropy: H(X) = -Σ P(t) log2 P(t) (bits/token)
                        • Empirical Token Perplexity: 2^H(X)
                        • Lexical Density % & Information Density Index (IDI)
                                             │
                                             ▼
                        [Module 6] Prompt Intermediate Representation (IR)
                        • Schema Serialization: Tagged-KV, Compact-DSL, Minimal-JSON
                        • Compiles natural language into atomic Task, Constraints, Context
                                             │
                                             ▼
                        [Module 9] Dynamic Routing & Cost-Latency Optimizer
                        • Multi-Objective Pareto Utility: U(M) = -w_c*C - w_l*L + w_q*Q
                        • Model Registry: Local GGUF Edge SLMs ($0.00) vs Cloud Frontier APIs
                        • Strategies: COST_MINIMIZATION, BALANCED, LATENCY, QUALITY
                                             │
                                             ▼
                      [ Final Optimized Payload Execution ]
               ├── Tier LOW / MEDIUM ──> Local Edge SLM (Llama 3.2, Qwen 2.5 Coder) [$0.00]
               └── Tier HIGH         ──> Cloud Frontier API (GPT-4o, Claude 3.5) [Compressed]
```

---

## 📦 The 9 Architectural Modules

### Module 1: Subword Tokenization (`core/tokenization/tokenizer.py`)
- **Dual BPE Engines:** Production BPE encoders supporting `cl100k_base` (GPT-4 / GPT-3.5) and `o200k_base` (GPT-4o / Omni).
- **Subword Inspection:** Token ID mapping, byte-level decomposition, and multi-byte sequence analysis.
- **Offline High-Availability Cache:** Pre-cached `.tiktoken` binaries mapped by SHA-1 hash to eliminate network dependencies.
- **Educational BPE:** Pure-Python reference merge-pair learner for viva demonstrations and pedagogical verification.

### Module 2: Multilingual Token Fragmentation (`core/tokenization/fragmentation.py`)
- **Unicode Script Identification:** Classifies Latin, Devanagari, Bengali, Arabic, Cyrillic, CJK, and Greek text blocks.
- **Fragmentation Metric:** Computes the word-to-token ratio $F = \frac{N_{\text{tokens}}}{N_{\text{words}}}$.
- **Cross-Lingual Parity Benchmark:** Quantifies the commercial billing disparity across 8 languages, proving that `o200k_base` recovers up to **68% token parity** for Indic languages over `cl100k_base`.

### Module 3: Perplexity & Information Density (`core/compression/density.py`)
- **Shannon Entropy:** Quantifies information bits packed per subword token:
  $$H(X) = -\sum_{i=1}^{n} P(t_i) \log_2 P(t_i)$$
- **Token Perplexity:** $\text{PPL} = 2^{H(X)}$ tracking distribution spread.
- **Lexical Density %:** $\frac{\text{Content Words}}{\text{Total Words}} \times 100\%$ tracking technical signal vs function filler.
- **Information Density Index (IDI):** Composite metric assessing high-density lexical content per token.
- **Coherence Estimator:** Evaluates transition structure to guarantee grammatical integrity after compression.

### Module 4: Rule-Based Text Compression (`core/compression/rule_compressor.py`)
- **Pleonasm Compactor:** Compacts multi-word wordy phrases (*"in order to"* $\rightarrow$ *"to"*, *"at the present time"* $\rightarrow$ *"now"*).
- **Politeness / Fluff Stripper:** Prunes conversational preambles (*"Could you please kindly help me"* $\rightarrow$ *"help me"*).
- **Semantic Anchor Whitelist:** Strictly preserves negations (*not, never, without*), interrogative question words (*how, why, what*), and algorithmic boundary constraints (*must, strictly, O(1)*).

### Module 5: AST Preservation for Code Compression (`core/compression/ast_code_compressor.py`)
- **Python AST Transformation:** Utilizes Python's `ast.NodeTransformer` to safely strip docstrings and comments from functions, classes, and modules without mutating operational logic.
- **Mathematical Equivalence Verification:** `_verify_ast_equivalence()` inspects structural AST signatures, function arguments, and control-flow branches to verify $100\%$ equivalence.
- **Universal Multi-Language Support:** Lexical comment and blank-line compaction for:
  - **C-Style Syntax Family (`//` and `/* */`):** C, C++, C#, Java, JavaScript, TypeScript, Go, Rust, Kotlin, Swift, Dart, Scala, Zig, PHP, Solidity.
  - **Hash-Style Family (`#`):** Bash, Shell, PowerShell, Ruby, Perl, R, Julia, YAML, Dockerfile.
  - **Dash-Style Family (`--`):** SQL, Lua, Haskell, Ada, VHDL.
  - **Semicolon Family (`;`):** Assembly (x86, ARM), Lisp, Clojure.
  - **Universal Fallback:** Automatically detects comment syntax for any custom or unknown programming language.

### Module 6: Intermediate Representation (IR) (`core/representation/`)
- **Schema-Driven Slot Generation:** Translates verbose natural language into atomic, machine-readable representations.
- **Multi-Format Serialization:**
  - `TAGGED_KV`: Fast, human-readable slot notation (`[TASK: ...]`, `[CONSTRAINTS: ...]`, `[CODE: ...]`).
  - `COMPACT_DSL`: Minimal space-delimited command format.
  - `MINIMAL_JSON`: Machine-to-machine structured JSON.

### Module 7: Prompt Feature Extraction (`core/routing/feature_extractor.py`)
- **20+ Extracted Signals:** Lexical diversity (TTR), character-to-token ratio, imperative verb counts, interrogative classification (`FACTUAL`, `PROCESS`, `CAUSAL`, `COMPARATIVE`, `DIRECTIVE`), fluff density, and constraint scores.
- **12-Dimensional Normalized ML Vector:** Emits a calibrated feature vector $\mathbf{x} \in [0.0, 1.0]^{12}$ ready for machine-learning clustering and classifier models.

### Module 8: Lightweight Intent Classification (`core/routing/classifiers/`)
- **Deterministic Rule Fast-Path (`rule_classifier.py`):** Sub-0.01ms instantaneous classification for unambiguous patterns (code blocks, stack traces, greetings).
- **TF-IDF + Multinomial Logistic Regression (`tfidf_logistic_classifier.py`):** Sublinear TF-IDF feature weighting with calibrated class probabilities over 5 core intents (`FACTUAL_LOOKUP`, `EXPLANATION`, `CODE_GENERATION`, `SYSTEM_ARCHITECTURE`, `DEBUGGING`).
- **Distilled Transformer Scorer (`distilled_transformer_classifier.py`):** Semantic prototype similarity matcher modeled after DistilBERT/MiniLM running on CPU in $< 2\text{ ms}$.
- **Consensus Blender (`hybrid_ensemble_classifier.py`):** Weighted ensemble blending rules, statistical probabilities, and dense embeddings.

### Module 9: Dynamic Routing & Cost-Latency Optimizer (`core/routing/optimization/`)
- **Hardware Model Registry:** Comprehensive catalog including:
  - **Local Edge SLMs (\$0.00 cost):** Llama 3.2 1B (Q4), Llama 3.2 3B (Q4), Qwen 2.5 Coder 7B (Q4).
  - **Cloud Commercial APIs:** OpenAI GPT-4o Mini, GPT-4o Frontier, Claude 3.5 Haiku, Claude 3.5 Sonnet.
- **Pareto Multi-Objective Utility Function:**
  $$U(M) = -w_c \cdot C(M) - w_l \cdot L(M) + w_q \cdot Q(M)$$
- **4 Optimization Strategies:** `COST_MINIMIZATION`, `BALANCED`, `LATENCY_MINIMIZATION`, `QUALITY_MAXIMIZATION`.
- **Financial Projections:** Live calculation of immediate query savings and 10,000-query commercial projections.

---

## 🚀 Installation & Environment Setup

### Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Arpit-codex/Ai-efficiency-.git
cd Ai-efficiency-
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 How to Run & Test

### Interface 1: Interactive Streamlit Web Dashboard
Launch the full visual dashboard to test prompts, inspect token chips, view AST diffs, and inspect dynamic cost routing:
```bash
streamlit run app.py
```
*Open your browser at **`http://localhost:8501`**.*

### Interface 2: Interactive Terminal CLI
Run the command-line interface with interactive numerical menus:
```bash
python main.py
```

### Interface 3: Single Prompt Execution via CLI
Test any custom prompt directly from your terminal:
```bash
python main.py --prompt "Could you please kindly help me write a Python function that calculates fibonacci numbers? In order to achieve good performance, do not use recursion without memoization."
```

### Interface 4: Run Empirical Benchmark Suite
Run the 6-category benchmark (Conversational, Algorithmic Code, Multilingual, Reasoning, Architecture, Fast-Path):
```bash
python main.py --benchmark
```

### Interface 5: Run 8-Language Cross-Lingual Parity Benchmark
Evaluate cross-lingual token inflation across English, Hindi, Bengali, Arabic, Spanish, French, Chinese, and Russian:
```bash
python main.py --multilingual
```

### Interface 6: Automated Test Suite (49 Tests)
Run all 49 unit tests verifying mathematical, AST, and classification integrity:
```bash
python -m pytest tests -v
```

---

## 📊 Empirical Results & Performance Benchmarks

In an empirical evaluation across 6 diverse prompt categories, the pipeline achieved the following results:

| Category | Raw Tokens | Compressed Tokens | Token Savings (%) | Assigned Tier | Target Engine | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conversational Fluff** | 47 | 22 | **53.19%** | LOW | Local Llama 3.2 1B (Q4) | 0.4 ms |
| **Simple Factual Lookup** | 31 | 18 | **41.94%** | LOW | Local Llama 3.2 1B (Q4) | 0.3 ms |
| **Algorithmic Code (AST)** | 145 | 94 | **35.17%** | HIGH | Cloud GPT-4o (Frontier) | 1.1 ms |
| **Multilingual (Indic/Hindi)** | 78 | 58 | **25.64%** | MEDIUM | Local Llama 3.2 3B (Q4) | 0.6 ms |
| **Multi-Hop Architecture** | 184 | 142 | **22.83%** | HIGH | Cloud Claude 3.5 Sonnet | 1.4 ms |
| **Deterministic Fast-Path** | 12 | 12 | **0.00%** | LOW | Local Llama 3.2 1B (Q4) | 0.08 ms |

### Financial Impact
- **Baseline Cost (Uncompressed on GPT-4o):** \$0.10217 USD (across sample batch)
- **Optimized Cost (Pre-processed & Routed):** \$0.00494 USD
- **Immediate Cost Reduction:** **95.17% savings**
- **Projected Savings per 10,000 Queries:** **\$972.30 USD saved**

---

## 📁 Repository Directory Structure

```text
Ai-efficiency-/
├── .cache/
│   └── tiktoken/              # Pre-cached BPE dictionaries (cl100k_base, o200k_base)
├── benchmarks/
│   ├── benchmark_tokenization_compression.py  # 6-category empirical benchmark runner
│   └── latest_benchmark_results.json          # Serialized benchmark metrics
├── core/
│   ├── compression/
│   │   ├── ast_code_compressor.py             # AST docstring & universal comment stripper
│   │   ├── compressor.py                      # Unified Non-AI Token Compressor orchestrator
│   │   ├── density.py                         # Shannon Entropy, Perplexity, Lexical Density
│   │   ├── rule_compressor.py                 # Pleonasm compactor & stopword pruner
│   │   └── rules.py                           # Heuristic regex rules & semantic whitelist
│   ├── pipeline.py                            # Unified LocalPreProcessingPipeline
│   ├── representation/
│   │   ├── ir.py                              # PromptIR schema & serializers (KV, DSL, JSON)
│   │   └── ir_generator.py                    # Atomic intent & slot extractor
│   ├── routing/
│   │   ├── classifiers/
│   │   │   ├── distilled_transformer_classifier.py # Semantic prototype similarity matcher
│   │   │   ├── hybrid_ensemble_classifier.py       # Consensus blender with fast-path override
│   │   │   ├── rule_classifier.py                  # Sub-0.1ms deterministic rule fast-path
│   │   │   ├── tfidf_logistic_classifier.py        # TF-IDF + Multinomial Logistic Regression
│   │   │   └── training_data.py                    # Domain intent training corpus
│   │   ├── feature_extractor.py               # 20+ signals & 12-D normalized ML vector
│   │   ├── intent_engine.py                   # Weighted intent router & compute budgeting
│   │   └── optimization/
│   │       ├── cost_latency_optimizer.py      # Pareto utility optimizer (4 strategies)
│   │       └── model_registry.py              # Hardware model catalog (Local vs Cloud)
│   └── tokenization/
│       ├── analyzer.py                        # Fragmentation & multi-encoding comparison
│       ├── fragmentation.py                   # Unicode script detector & 8-language parity
│       └── tokenizer.py                       # Production BPE & Educational BPE
├── tests/                                     # 49 automated unit tests across 11 modules
├── app.py                                     # Interactive Streamlit Web Dashboard
├── main.py                                    # Interactive Console CLI
├── requirements.txt                           # Project Python dependencies
├── .gitignore                                 # Git ignore patterns
└── README.md                                  # Complete academic & architectural documentation
```

---

## 🔬 Academic Citation & Authors

This project was developed as part of the B.Tech Minor Project requirements at **Shivalik College of Engineering, Dehradun**:

```bibtex
@misc{bisht2026optimizing,
  title={Optimizing AI Architecture for Long-Term Memory and Efficiency: Consumer-Side Pre-Processing, AST Compression, and Dynamic Model Routing},
  author={Bisht, Arpit and Singh, Anurag Kumar and Kumar, Amandeep and Yadav, Anurag},
  year={2026},
  institution={Shivalik College of Engineering, Dehradun},
  note={Under the supervision of Ms. Swati Kashyap}
}
```
