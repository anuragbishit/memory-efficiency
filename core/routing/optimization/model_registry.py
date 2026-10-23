"""
Model Registry and Runtime Specification Module.
Maintains operational metadata, commercial token pricing, latency profiles,
and capability tiers across Local Edge SLMs and Upstream Cloud APIs.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


class RuntimeType(str, Enum):
    """Target execution runtime environment."""
    LOCAL_EDGE = "LOCAL_EDGE"     # Consumer hardware (CPU/GPU/NPU via GGUF)
    CLOUD_API = "CLOUD_API"       # Commercial SaaS endpoints (OpenAI/Anthropic)


class ModelCapabilityTier(str, Enum):
    """Reasoning and domain capability tiers."""
    ENTRY = "ENTRY"               # Simple lookups, greetings, basic factual tasks
    INTERMEDIATE = "INTERMEDIATE" # Explanations, single-step tasks, light code
    ADVANCED = "ADVANCED"         # Fast cloud models, moderate system analysis
    EXPERT = "EXPERT"             # Multi-hop reasoning, complex distributed architecture


@dataclass
class ModelMetadata:
    """Comprehensive specifications for an individual model runtime."""
    model_id: str
    display_name: str
    runtime_type: RuntimeType
    capability_tier: ModelCapabilityTier
    pricing_input_per_1m_usd: float    # Cost per 1,000,000 input tokens ($)
    pricing_output_per_1m_usd: float   # Cost per 1,000,000 output tokens ($)
    avg_network_latency_ms: float      # Roundtrip network latency
    avg_compute_ttft_ms: float         # Compute Time-To-First-Token
    generation_throughput_tps: float   # Tokens per second generation rate
    context_window_tokens: int
    quantization_format: Optional[str] = None
    min_ram_required_gb: float = 0.0

    @property
    def total_ttft_ms(self) -> float:
        """Total time to first token (network + compute)."""
        return self.avg_network_latency_ms + self.avg_compute_ttft_ms

    def calculate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Calculates exact financial cost for a prompt execution in USD."""
        if self.runtime_type == RuntimeType.LOCAL_EDGE:
            return 0.0  # Runs entirely on consumer-owned hardware

        cost_in = (input_tokens / 1_000_000.0) * self.pricing_input_per_1m_usd
        cost_out = (output_tokens / 1_000_000.0) * self.pricing_output_per_1m_usd
        return round(cost_in + cost_out, 6)

    def estimate_total_latency_ms(self, output_tokens: int) -> float:
        """Estimates total execution latency including generation time."""
        gen_time_sec = output_tokens / max(1.0, self.generation_throughput_tps)
        return round(self.total_ttft_ms + (gen_time_sec * 1000.0), 2)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["runtime_type"] = self.runtime_type.value
        data["capability_tier"] = self.capability_tier.value
        data["total_ttft_ms"] = self.total_ttft_ms
        return data


# Standard Model Registry: Local Edge SLMs + Cloud APIs
DEFAULT_MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    # 1. Local Edge SLMs (Zero API Cost, Sovereign, Zero Network Latency)
    "local/llama-3.2-1b-q4": ModelMetadata(
        model_id="local/llama-3.2-1b-q4",
        display_name="Llama 3.2 1B Instruct (GGUF Q4_K_M)",
        runtime_type=RuntimeType.LOCAL_EDGE,
        capability_tier=ModelCapabilityTier.ENTRY,
        pricing_input_per_1m_usd=0.0,
        pricing_output_per_1m_usd=0.0,
        avg_network_latency_ms=0.0,
        avg_compute_ttft_ms=45.0,
        generation_throughput_tps=55.0,
        context_window_tokens=8192,
        quantization_format="Q4_K_M",
        min_ram_required_gb=2.0,
    ),
    "local/llama-3.2-3b-q4": ModelMetadata(
        model_id="local/llama-3.2-3b-q4",
        display_name="Llama 3.2 3B Instruct (GGUF Q4_K_M)",
        runtime_type=RuntimeType.LOCAL_EDGE,
        capability_tier=ModelCapabilityTier.INTERMEDIATE,
        pricing_input_per_1m_usd=0.0,
        pricing_output_per_1m_usd=0.0,
        avg_network_latency_ms=0.0,
        avg_compute_ttft_ms=85.0,
        generation_throughput_tps=38.0,
        context_window_tokens=8192,
        quantization_format="Q4_K_M",
        min_ram_required_gb=4.5,
    ),
    "local/qwen-2.5-coder-7b-q4": ModelMetadata(
        model_id="local/qwen-2.5-coder-7b-q4",
        display_name="Qwen 2.5 Coder 7B (GGUF Q4_K_M)",
        runtime_type=RuntimeType.LOCAL_EDGE,
        capability_tier=ModelCapabilityTier.INTERMEDIATE,
        pricing_input_per_1m_usd=0.0,
        pricing_output_per_1m_usd=0.0,
        avg_network_latency_ms=0.0,
        avg_compute_ttft_ms=130.0,
        generation_throughput_tps=28.0,
        context_window_tokens=16384,
        quantization_format="Q4_K_M",
        min_ram_required_gb=8.0,
    ),

    # 2. Upstream Commercial Cloud APIs
    "cloud/gpt-4o-mini": ModelMetadata(
        model_id="cloud/gpt-4o-mini",
        display_name="OpenAI GPT-4o Mini",
        runtime_type=RuntimeType.CLOUD_API,
        capability_tier=ModelCapabilityTier.ADVANCED,
        pricing_input_per_1m_usd=0.15,
        pricing_output_per_1m_usd=0.60,
        avg_network_latency_ms=220.0,
        avg_compute_ttft_ms=80.0,
        generation_throughput_tps=95.0,
        context_window_tokens=128000,
    ),
    "cloud/claude-3-5-haiku": ModelMetadata(
        model_id="cloud/claude-3-5-haiku",
        display_name="Anthropic Claude 3.5 Haiku",
        runtime_type=RuntimeType.CLOUD_API,
        capability_tier=ModelCapabilityTier.ADVANCED,
        pricing_input_per_1m_usd=0.80,
        pricing_output_per_1m_usd=4.00,
        avg_network_latency_ms=250.0,
        avg_compute_ttft_ms=90.0,
        generation_throughput_tps=85.0,
        context_window_tokens=200000,
    ),
    "cloud/gpt-4o": ModelMetadata(
        model_id="cloud/gpt-4o",
        display_name="OpenAI GPT-4o (Omni Frontier)",
        runtime_type=RuntimeType.CLOUD_API,
        capability_tier=ModelCapabilityTier.EXPERT,
        pricing_input_per_1m_usd=2.50,
        pricing_output_per_1m_usd=10.00,
        avg_network_latency_ms=280.0,
        avg_compute_ttft_ms=120.0,
        generation_throughput_tps=110.0,
        context_window_tokens=128000,
    ),
    "cloud/claude-3-5-sonnet": ModelMetadata(
        model_id="cloud/claude-3-5-sonnet",
        display_name="Anthropic Claude 3.5 Sonnet",
        runtime_type=RuntimeType.CLOUD_API,
        capability_tier=ModelCapabilityTier.EXPERT,
        pricing_input_per_1m_usd=3.00,
        pricing_output_per_1m_usd=15.00,
        avg_network_latency_ms=310.0,
        avg_compute_ttft_ms=140.0,
        generation_throughput_tps=75.0,
        context_window_tokens=200000,
    ),
}
