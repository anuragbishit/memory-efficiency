"""
Orchestrator for the Token-Efficient Code Editing System.
Coordinates the Intent Classifier, Direct Edit Engine, Targeted Retrieval,
Patch Engine, and Subword Tokenizer to achieve maximum token savings.
"""

import time
import os
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from core.tokenization.tokenizer import SubwordTokenizer
from core.editor.models import EditTier, EditOperation, ParsedInstruction, CodeEditResult
from core.editor.intent_classifier import EditIntentClassifier
from core.editor.direct_engine import DirectEditEngine
from core.editor.targeted_retrieval import TargetedRetrievalEngine
from core.editor.patch_engine import MinimalPatchEngine, EditSnapshot


class TokenEfficientCodeEditor:
    """
    Main orchestration engine for token-efficient code editing.
    Dramatically reduces or completely eliminates LLM token consumption by:
      - Bypassing LLMs entirely for direct variable, literal, and symbol edits (0 tokens).
      - Extracting only micro-snippets for targeted edits (up to 95% token savings).
      - Escalating to deep reasoning only when strictly necessary.
    """

    def __init__(self, tokenizer: Optional[SubwordTokenizer] = None):
        self.tokenizer = tokenizer or SubwordTokenizer(default_encoding="cl100k_base")
        self.classifier = EditIntentClassifier()
        self.direct_engine = DirectEditEngine()
        self.retrieval_engine = TargetedRetrievalEngine()
        self.patch_engine = MinimalPatchEngine()

    def edit_code(
        self,
        code: str,
        instruction: str,
        language: str = "python",
        file_path: Optional[str] = None,
        force_tier: Optional[EditTier] = None
    ) -> CodeEditResult:
        """
        Executes a token-optimized edit on code based on a natural language instruction.
        """
        t0 = time.perf_counter()

        # 1. Baseline Token Calculation (What a naive LLM assistant consumes)
        # Naive approach reads the whole file + instruction, then regenerates the whole file
        instruction_tokens = self.tokenizer.count_tokens(instruction)
        file_tokens = self.tokenizer.count_tokens(code)
        # Naive prompt: instruction + whole file; Naive completion: whole file
        baseline_tokens = instruction_tokens + (2 * file_tokens)

        # 2. Intent Classification & Routing
        parsed: ParsedInstruction = self.classifier.classify(instruction, code_context=code)
        tier = force_tier or parsed.tier

        modified_code = code
        modified_lines: List[int] = []
        llm_calls = 0
        tokens_consumed = 0
        success = False
        status_msg = ""
        error_msg = None

        # 3. Execution Path based on Tier
        if tier == EditTier.DIRECT_EDIT:
            # FAST-PATH: Direct deterministic edit (0 LLM Calls, 0 LLM Tokens)
            success, modified_code, modified_lines, status_msg = self.direct_engine.apply_edit(
                code, parsed, language=language
            )
            llm_calls = 0
            tokens_consumed = 0

            # If direct edit failed because of mismatch or validation, stop immediately
            if not success:
                if "mismatch" in status_msg.lower():
                    # Definite value/symbol mismatch detected: do NOT modify code, do NOT escalate
                    error_msg = status_msg
                elif parsed.target_variable and parsed.target_function:
                    tier = EditTier.TARGETED_RETRIEVAL
                    status_msg = f"Direct edit could not locate '{parsed.target_variable}'. Escalating to targeted snippet search."

        if tier == EditTier.TARGETED_RETRIEVAL:
            # MEDIUM-PATH: Extract targeted function/snippet
            target_name = parsed.target_function or parsed.target_variable or ""
            snippet_ctx = self.retrieval_engine.extract_function_or_block(code, target_name, language=language)

            if snippet_ctx:
                # LLM only processes the minimal snippet rather than the whole file
                snip_token_count = self.tokenizer.count_tokens(snippet_ctx.snippet_code)
                tokens_consumed = instruction_tokens + (2 * snip_token_count)
                llm_calls = 1

                # If an assignment or rename was requested inside the snippet, apply it
                sub_success, edited_snippet, sub_lines, sub_msg = self.direct_engine.apply_edit(
                    snippet_ctx.snippet_code, parsed, language=language
                )
                if sub_success:
                    modified_code = self.retrieval_engine.splice_snippet(
                        code, snippet_ctx.start_line, snippet_ctx.end_line, edited_snippet
                    )
                    modified_lines = [snippet_ctx.start_line + l - 1 for l in sub_lines]
                    success = True
                    status_msg = f"Targeted edit applied inside '{target_name}' (lines {snippet_ctx.start_line}-{snippet_ctx.end_line})."
                else:
                    success = False
                    status_msg = f"Targeted retrieval located block, but sub-edit failed: {sub_msg}"
            else:
                # Could not find snippet block; fallback to direct regex
                success, modified_code, modified_lines, status_msg = self.direct_engine.apply_edit(
                    code, parsed, language=language
                )
                tokens_consumed = 0
                llm_calls = 0

        elif tier == EditTier.DEEP_REASONING:
            # COMPLEX-PATH: Deep reasoning (Full context required)
            llm_calls = 1
            tokens_consumed = baseline_tokens
            success = False
            status_msg = "Deep reasoning required. Context formulated for LLM inference."

        # 4. Token Savings Metrics & Patch Handling
        if not success:
            modified_code = code
            modified_lines = []
            tokens_consumed = 0
            tokens_saved = 0
            savings_pct = 0.0
            diff = ""
        else:
            tokens_saved = max(0, baseline_tokens - tokens_consumed)
            savings_pct = round((tokens_saved / max(1, baseline_tokens)) * 100, 2)
            diff = self.patch_engine.compute_diff(code, modified_code, filename=file_path or "source.py")

        syntax_ok, val_msg = self.patch_engine.validate_syntax(modified_code, language=language)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 6. Snapshot for Rollback
        result = CodeEditResult(
            success=success,
            tier=tier,
            operation=parsed.operation.value,
            original_code=code,
            modified_code=modified_code,
            diff=diff,
            modified_lines=modified_lines,
            llm_calls_made=llm_calls,
            tokens_consumed=tokens_consumed,
            baseline_tokens=baseline_tokens,
            tokens_saved=tokens_saved,
            token_savings_pct=savings_pct,
            syntax_valid=syntax_ok,
            validation_message=f"{status_msg} | {val_msg}" if success else (status_msg or val_msg),
            latency_ms=latency_ms,
            error=error_msg
        )

        if success:
            self.patch_engine.create_snapshot(
                edit_id=result.edit_id,
                original_code=code,
                modified_code=modified_code,
                file_path=file_path
            )

        return result

    def edit_file(
        self,
        file_path: str,
        instruction: str,
        save: bool = True
    ) -> CodeEditResult:
        """
        Loads code from disk, performs token-efficient edit, and optionally saves back.
        """
        path = Path(file_path)
        if not path.exists():
            return CodeEditResult(
                success=False,
                tier=EditTier.DIRECT_EDIT,
                operation="FILE_IO",
                original_code="",
                modified_code="",
                diff="",
                error=f"File not found: {file_path}"
            )

        original_code = path.read_text(encoding="utf-8", errors="replace")
        ext = path.suffix.lstrip(".").lower()
        language = "python" if ext in ("py", "pyw") else ext

        result = self.edit_code(
            code=original_code,
            instruction=instruction,
            language=language,
            file_path=str(path)
        )

        if result.success and save:
            path.write_text(result.modified_code, encoding="utf-8")

        return result

    def rollback(self, edit_id: str, file_path: Optional[str] = None) -> bool:
        """
        Restores code to pre-edit state. If file_path provided, writes back to disk.
        """
        original = self.patch_engine.rollback(edit_id)
        if original is None:
            return False

        if file_path:
            path = Path(file_path)
            if path.exists():
                path.write_text(original, encoding="utf-8")

        return True
