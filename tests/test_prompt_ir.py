"""
Unit tests for Prompt Intermediate Representation (IR) Engine (Topic 6).
"""

import pytest
from core.representation import PromptIRGenerator, PromptIR, IRFormat, TaskIntentType


def test_ir_generation_and_intent_detection():
    generator = PromptIRGenerator()
    prompt = (
        "Could you please optimize this quicksort function so that it strictly runs in O(n log n) time?\n"
        "```python\n"
        "def quicksort(arr): return arr\n"
        "```"
    )

    prompt_ir = generator.generate_ir(prompt)

    assert prompt_ir.intent == TaskIntentType.OPTIMIZE
    assert prompt_ir.action == "optimize"
    assert len(prompt_ir.code_snippets) == 1
    assert prompt_ir.code_snippets[0]["lang"] == "python"
    assert any("o(n log n)" in c.lower() or "strictly" in c.lower() for c in prompt_ir.constraints)


def test_ir_serialization_formats():
    prompt_ir = PromptIR(
        intent=TaskIntentType.ARCHITECT,
        action="architect",
        target="distributed cache",
        constraints=["strictly thread-safe", "O(1) read latency"],
        code_snippets=[{"lang": "python", "code": "class Cache: pass"}],
        context_anchors=["Concurrency"],
        output_format="code_and_explanation"
    )

    tagged_kv = prompt_ir.serialize(IRFormat.TAGGED_KV)
    assert "[TASK: ARCHITECT -> architect distributed cache]" in tagged_kv
    assert "[CONSTRAINTS: strictly thread-safe; O(1) read latency]" in tagged_kv
    assert "[INPUT_CODE: python" in tagged_kv

    compact_dsl = prompt_ir.serialize(IRFormat.COMPACT_DSL)
    assert "ARCHITECT(distributed cache)" in compact_dsl
    assert "CONSTRAIN(" in compact_dsl

    minimal_json = prompt_ir.serialize(IRFormat.MINIMAL_JSON)
    assert '"intent":"ARCHITECT"' in minimal_json
    assert '"target":"distributed cache"' in minimal_json


def test_ir_token_reduction():
    generator = PromptIRGenerator()
    verbose_prompt = (
        "Hello, could you please kindly assist me in explaining how GraphRAG works? "
        "I was wondering if you might be able to compare it with flat vector embeddings. "
        "Thank you so much in advance!"
    )

    prompt_ir = generator.generate_ir(verbose_prompt, format_type=IRFormat.TAGGED_KV)

    # IR should achieve substantial token reduction over verbose polite fluff
    assert prompt_ir.ir_tokens < prompt_ir.original_tokens
    assert prompt_ir.token_savings_pct > 25.0
