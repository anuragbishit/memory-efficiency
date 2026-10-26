"""
Direct Edit Engine (Tier 1: Zero-Token Local Code Modifier).
Applies deterministic, zero-LLM edits to source code files using:
  - AST-guided line-range targeting (preserves surrounding whitespace & comments)
  - Word-boundary identifier substitution (safe variable rename)
  - Exact literal search-and-replace across multi-language files
"""

import ast
import re
import time
from typing import Tuple, List, Optional, Dict, Any
from core.editor.models import EditTier, EditOperation, ParsedInstruction, CodeEditResult


class DirectEditEngine:
    """
    Executes code modifications locally without invoking an LLM.
    Guarantees 0 token consumption for direct, deterministic edits.
    """

    def apply_edit(
        self,
        code: str,
        parsed: ParsedInstruction,
        language: str = "python"
    ) -> Tuple[bool, str, List[int], str]:
        """
        Applies direct edit based on parsed instruction.
        Returns:
            (success, modified_code, modified_lines, status_message)
        """
        if parsed.operation == EditOperation.ASSIGNMENT_CHANGE:
            return self._apply_assignment_change(
                code,
                target_var=parsed.target_variable,
                old_val=parsed.old_value,
                new_val=parsed.new_value,
                language=language
            )

        elif parsed.operation == EditOperation.VARIABLE_RENAME:
            return self._apply_variable_rename(
                code,
                old_var=parsed.target_variable or parsed.old_value,
                new_var=parsed.new_value,
                language=language
            )

        elif parsed.operation == EditOperation.LITERAL_REPLACE:
            return self._apply_literal_replace(
                code,
                old_literal=parsed.old_value,
                new_literal=parsed.new_value
            )

        return False, code, [], f"Unsupported direct edit operation: {parsed.operation}"

    def _apply_assignment_change(
        self,
        code: str,
        target_var: Optional[str],
        old_val: Optional[str],
        new_val: Optional[str],
        language: str = "python"
    ) -> Tuple[bool, str, List[int], str]:
        """
        Modifies a variable assignment preserving exact comments and formatting.
        Uses AST targeting for Python and regex fallback for any language.
        """
        if not target_var or new_val is None:
            return False, code, [], "Missing target variable or new value."

        lines = code.splitlines(keepends=True)
        modified_lines: List[int] = []

        # 1. Try AST-guided line localization for Python
        ast_matched = False
        target_lineno = None
        if language.lower() in ("python", "py"):
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == target_var:
                                target_lineno = node.lineno
                                ast_matched = True
                                break
                    elif isinstance(node, ast.AnnAssign):
                        if isinstance(node.target, ast.Name) and node.target.id == target_var:
                            target_lineno = node.lineno
                            ast_matched = True
                            break
                    if ast_matched:
                        break
            except Exception:
                ast_matched = False

        # 2. If AST identified the line, target that specific line
        if ast_matched and target_lineno is not None and 1 <= target_lineno <= len(lines):
            line_idx = target_lineno - 1
            line = lines[line_idx]

            # Match assignment pattern on that specific line: target_var = <val>
            pattern = re.compile(
                rf"^(\s*{re.escape(target_var)}\s*(?::\s*[^=]+)?=\s*)(.+?)(\s*(?:#.*)?\r?\n?$)"
            )
            m = pattern.match(line)
            if m:
                curr_val = m.group(2).strip().strip("'\"")
                # Strict check: If old_val was specified by user, it MUST match the current value in code!
                if old_val is not None:
                    clean_old = old_val.strip().strip("'\"")
                    if clean_old != curr_val:
                        return False, code, [], (
                            f"Value mismatch for variable '{target_var}': instruction specified "
                            f"'{target_var} = {clean_old}', but source code currently has '{target_var} = {curr_val}'. "
                            f"Modification rejected to prevent unintended changes."
                        )

                prefix = m.group(1)
                suffix = m.group(3)
                lines[line_idx] = f"{prefix}{new_val}{suffix}"
                modified_lines.append(target_lineno)
                return True, "".join(lines), modified_lines, f"AST-targeted: set '{target_var} = {new_val}' on line {target_lineno}."

        # 3. Regex Fallback across lines (Language Agnostic: Python, JS, C++, etc.)
        pattern = re.compile(
            rf"^(\s*(?:(?:let|var|const|val|int|float|double|auto)\s+)?{re.escape(target_var)}\s*(?::\s*[^=]+)?=\s*)(.+?)(\s*(?:[;#].*)?\r?\n?$)"
        )
        found_var = False
        actual_val = None
        for i, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                found_var = True
                curr_val = m.group(2).strip().strip("'\"")
                actual_val = curr_val
                # If old_val was specified, check if it matches
                if old_val is not None:
                    clean_old = old_val.strip().strip("'\"")
                    if clean_old != curr_val:
                        return False, code, [], (
                            f"Value mismatch for variable '{target_var}': instruction specified "
                            f"'{target_var} = {clean_old}', but source code currently has '{target_var} = {curr_val}'. "
                            f"Modification rejected to prevent unintended changes."
                        )

                prefix = m.group(1)
                suffix = m.group(3)
                lines[i] = f"{prefix}{new_val}{suffix}"
                modified_lines.append(i + 1)
                return True, "".join(lines), modified_lines, f"Regex-targeted: updated '{target_var} = {new_val}' on line {i + 1}."

        if found_var:
            return False, code, [], f"Value mismatch for variable '{target_var}': source code has '{target_var} = {actual_val}'."

        return False, code, [], f"Variable '{target_var}' not found in source code."

    def _apply_variable_rename(
        self,
        code: str,
        old_var: Optional[str],
        new_var: Optional[str],
        language: str = "python"
    ) -> Tuple[bool, str, List[int], str]:
        """
        Safely renames an identifier across a file using word boundaries.
        """
        if not old_var or not new_var:
            return False, code, [], "Old and new variable names must be specified."

        # Verify old_var exists as a whole word
        pattern = re.compile(rf"\b{re.escape(old_var)}\b")
        if not pattern.search(code):
            return False, code, [], f"Symbol mismatch: variable '{old_var}' not found in source code. Rename rejected."

        lines = code.splitlines(keepends=True)
        modified_lines: List[int] = []
        new_lines: List[str] = []

        for i, line in enumerate(lines):
            if pattern.search(line):
                new_line = pattern.sub(new_var, line)
                new_lines.append(new_line)
                modified_lines.append(i + 1)
            else:
                new_lines.append(line)

        modified_code = "".join(new_lines)
        return True, modified_code, modified_lines, f"Renamed symbol '{old_var}' -> '{new_var}' across {len(modified_lines)} lines."

    def _apply_literal_replace(
        self,
        code: str,
        old_literal: Optional[str],
        new_literal: Optional[str]
    ) -> Tuple[bool, str, List[int], str]:
        """
        Replaces an exact literal or substring across the codebase.
        """
        if old_literal is None or new_literal is None:
            return False, code, [], "Old and new literals must be provided."

        if old_literal not in code:
            return False, code, [], f"Value mismatch: literal '{old_literal}' not found in source code. Modification rejected."

        lines = code.splitlines(keepends=True)
        modified_lines: List[int] = []
        new_lines: List[str] = []

        for i, line in enumerate(lines):
            if old_literal in line:
                new_lines.append(line.replace(old_literal, new_literal))
                modified_lines.append(i + 1)
            else:
                new_lines.append(line)

        return True, "".join(new_lines), modified_lines, f"Replaced literal across {len(modified_lines)} lines."
