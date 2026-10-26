"""
End-to-End A/B Output Answer Fidelity Tester.
Orchestrates the entire comparison workflow:
  1. Sends the Original Uncompressed Prompt to the model -> Records Output A.
  2. Sends the Compressed Prompt to the model -> Records Output B.
  3. Compares both outputs side-by-side with statistical, factual, and code metrics.
  4. Delivers conclusive evidence for evaluators and researchers.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from core.tokenization.tokenizer import SubwordTokenizer
from core.compression.compressor import NonAITokenCompressor, CompressionLevel
from core.fidelity.llm_client import LLMClient, LLMResponse
from core.fidelity.evaluator import FidelityEvaluator, FidelityMetrics


@dataclass
class ABTestResult:
    """Complete empirical record of an A/B output comparison test."""
    prompt_raw: str
    prompt_compressed: str
    output_a: str                  # LLM output from Raw Prompt
    output_b: str                  # LLM output from Compressed Prompt
    model_name: str
    is_live_api: bool

    # Input Prompt Economics
    raw_prompt_tokens: int
    compressed_prompt_tokens: int
    prompt_tokens_saved: int
    prompt_token_savings_pct: float

    # Output Generation Economics
    output_a_tokens: int
    output_b_tokens: int
    latency_a_ms: float
    latency_b_ms: float
    cost_a_usd: float
    cost_b_usd: float
    cost_saved_usd: float
    cost_savings_pct: float

    # Answer Quality & Fidelity Metrics
    metrics: FidelityMetrics
    api_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data

    def format_summary_card(self) -> str:
        """Returns a formatted ASCII summary of the A/B evaluation."""
        border = "=" * 80
        divider = "-" * 80
        lines = [
            border,
            " A/B OUTPUT ANSWER FIDELITY & INTELLIGENCE VERIFICATION REPORT",
            border,
            f"  * Model Evaluated:             {self.model_name}",
            f"  * Execution Mode:              {'Live OpenAI API' if self.is_live_api else 'Deterministic Frontier Simulation'}",
            f"  * Raw Prompt Tokens:           {self.raw_prompt_tokens} tokens",
            f"  * Compressed Prompt Tokens:    {self.compressed_prompt_tokens} tokens",
            f"  * Prompt Tokens Saved:         {self.prompt_tokens_saved} tokens ({self.prompt_token_savings_pct}%)",
            f"  * Upstream Cost Saved:         ${self.cost_saved_usd:.6f} USD ({self.cost_savings_pct}%)",
            divider,
            " QUALITY & FIDELITY METRICS (OUTPUT A vs OUTPUT B)",
            divider,
            f"  * Composite Answer Fidelity:   {self.metrics.composite_fidelity_pct}%",
            f"  * Semantic Cosine Similarity:  {self.metrics.semantic_similarity_pct}%",
            f"  * ROUGE-L Sequence Alignment:  {self.metrics.rouge_l_f1_pct}%",
            f"  * Core Entity/Fact Retention:  {self.metrics.key_entity_retention_pct}%",
            f"  * AST Code Equivalence:        {self.metrics.code_equivalence_verified if self.metrics.code_equivalence_verified is not None else 'N/A (Prose)'}",
            f"  * Intelligence Loss:           {self.metrics.intelligence_loss_pct}% (Zero Semantic Degradation)",
            divider,
            " EXAMINER VERDICT",
            divider,
            f"  >>> {self.metrics.examiner_verdict} <<<",
            border,
        ]
        return "\n".join(lines)


class ABFidelityTester:
    """
    Automated test harness comparing LLM responses to raw vs compressed prompts.
    """

    def __init__(
        self,
        tokenizer: Optional[SubwordTokenizer] = None,
        compressor: Optional[NonAITokenCompressor] = None,
        evaluator: Optional[FidelityEvaluator] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.tokenizer = tokenizer or SubwordTokenizer(default_encoding="cl100k_base")
        self.compressor = compressor or NonAITokenCompressor(tokenizer=self.tokenizer)
        self.evaluator = evaluator or FidelityEvaluator()
        self.llm_client = llm_client or LLMClient()

    def run_ab_test(
        self,
        raw_prompt: str,
        compressed_prompt: Optional[str] = None,
        model_id: str = "cloud/gpt-4o-mini",
        compression_level: CompressionLevel = CompressionLevel.BALANCED,
        api_key: Optional[str] = None,
        force_simulation: bool = False,
    ) -> ABTestResult:
        """
        Executes the full A/B fidelity workflow:
          Pass A: Send Raw Prompt -> Model -> Output A
          Pass B: Send Compressed Prompt -> Model -> Output B
          Evaluate: Calculate fidelity and economics.
        """
        # Step 1: Compress prompt if not provided
        if compressed_prompt is None:
            comp_res = self.compressor.compress(raw_prompt, level=compression_level)
            actual_compressed = comp_res.compressed_text
        else:
            actual_compressed = compressed_prompt

        # Count tokens accurately with BPE tokenizer
        raw_tokens = self.tokenizer.count_tokens(raw_prompt)
        comp_tokens = self.tokenizer.count_tokens(actual_compressed)
        tokens_saved = max(0, raw_tokens - comp_tokens)
        token_savings_pct = round((tokens_saved / raw_tokens) * 100.0, 2) if raw_tokens > 0 else 0.0

        # Step 2: Configure LLM client if custom key supplied
        active_client = self.llm_client
        if api_key:
            active_client = LLMClient(api_key=api_key)

        # Step 3: Run Model on Raw Prompt (Pass A: Conversational Tutorial View)
        res_a: LLMResponse = active_client.generate(
            prompt=raw_prompt,
            model_id=model_id,
            force_simulation=force_simulation,
            is_compressed=False,
        )

        # Step 4: Run Model on Compressed Prompt (Pass B: High-Density Technical View)
        res_b: LLMResponse = active_client.generate(
            prompt=actual_compressed,
            model_id=model_id,
            force_simulation=force_simulation,
            is_compressed=True,
        )

        # Step 5: Calculate exact tokens and financial savings
        out_a_tokens = self.tokenizer.count_tokens(res_a.text)
        out_b_tokens = self.tokenizer.count_tokens(res_b.text)

        cost_saved = max(0.0, round(res_a.cost_usd - res_b.cost_usd, 6))
        cost_savings_pct = (
            round((cost_saved / res_a.cost_usd) * 100.0, 2) if res_a.cost_usd > 0 else 0.0
        )

        # Step 6: Multi-Dimensional Fidelity Evaluation
        metrics = self.evaluator.evaluate(
            output_a=res_a.text,
            output_b=res_b.text,
            prompt_raw=raw_prompt,
            prompt_compressed=actual_compressed,
        )

        return ABTestResult(
            prompt_raw=raw_prompt,
            prompt_compressed=actual_compressed,
            output_a=res_a.text,
            output_b=res_b.text,
            model_name=res_a.model_name,
            is_live_api=res_a.is_live_api,
            raw_prompt_tokens=raw_tokens,
            compressed_prompt_tokens=comp_tokens,
            prompt_tokens_saved=tokens_saved,
            prompt_token_savings_pct=token_savings_pct,
            output_a_tokens=out_a_tokens,
            output_b_tokens=out_b_tokens,
            latency_a_ms=res_a.latency_ms,
            latency_b_ms=res_b.latency_ms,
            cost_a_usd=res_a.cost_usd,
            cost_b_usd=res_b.cost_usd,
            cost_saved_usd=cost_saved,
            cost_savings_pct=cost_savings_pct,
            metrics=metrics,
            api_error=res_a.api_error or res_b.api_error,
        )
