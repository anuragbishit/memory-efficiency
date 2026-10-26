"""
Targeted Retrieval Engine (Tier 2: Micro-Context Extractor).
Instead of sending an entire file (1,000+ tokens) to an LLM,
this engine extracts only the specific function, class, or 5-10 line
code window where the edit is needed, drastically minimizing tokenization.
"""

import ast
import re
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CodeSnippetContext:
    """Extracted micro-context block for targeted editing."""
    target_name: str
    snippet_code: str
    start_line: int
    end_line: int
    total_file_lines: int
    snippet_tokens: int
    full_file_tokens: int
    tokens_saved: int
    token_reduction_pct: float


class TargetedRetrievalEngine:
    """
    Extracts the minimal relevant code snippet required for semantic edits
    and accurately splices the edited snippet back into the original file.
    """

    def extract_function_or_block(
        self,
        code: str,
        target_name: str,
        language: str = "python"
    ) -> Optional[CodeSnippetContext]:
        """
        Locates and extracts only the target function/class block.
        """
        lines = code.splitlines(keepends=True)
        total_lines = len(lines)

        start_line = None
        end_line = None

        # 1. AST-based block extraction for Python
        if language.lower() in ("python", "py"):
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.name.lower() == target_name.lower():
                            start_line = node.lineno
                            end_line = getattr(node, "end_lineno", node.lineno + 10)
                            break
            except Exception:
                pass

        # 2. Heuristic boundary extraction (fallback for non-Python or unparseable blocks)
        if start_line is None:
            func_pattern = re.compile(
                rf"^\s*(?:def|async\s+def|class|function|public|private)?\s*{re.escape(target_name)}\b",
                re.IGNORECASE
            )
            for i, line in enumerate(lines):
                if func_pattern.search(line) or re.search(rf"\b{re.escape(target_name)}\b", line):
                    start_line = i + 1
                    # Extract roughly 10 lines or until indentation resets
                    end_line = min(total_lines, start_line + 12)
                    break

        if start_line is None or end_line is None:
            return None

        # Extract snippet
        snippet_lines = lines[start_line - 1:end_line]
        snippet_code = "".join(snippet_lines)

        # Token metrics (word-based approximation if tokenizer not injected)
        full_tokens = max(1, len(code.split()))
        snip_tokens = max(1, len(snippet_code.split()))
        tokens_saved = max(0, full_tokens - snip_tokens)
        reduction_pct = round((tokens_saved / full_tokens) * 100, 2)

        return CodeSnippetContext(
            target_name=target_name,
            snippet_code=snippet_code,
            start_line=start_line,
            end_line=end_line,
            total_file_lines=total_lines,
            snippet_tokens=snip_tokens,
            full_file_tokens=full_tokens,
            tokens_saved=tokens_saved,
            token_reduction_pct=reduction_pct
        )

    def splice_snippet(
        self,
        original_code: str,
        start_line: int,
        end_line: int,
        edited_snippet: str
    ) -> str:
        """
        Splices an edited micro-snippet back into the parent file at exact line boundaries.
        Preserves all code before and after the snippet.
        """
        lines = original_code.splitlines(keepends=True)
        before = lines[:start_line - 1]
        after = lines[end_line:]

        # Ensure edited snippet ends with a newline if appropriate
        if edited_snippet and not edited_snippet.endswith("\n"):
            edited_snippet += "\n"

        return "".join(before) + edited_snippet + "".join(after)
