"""
Minimal Patch & Validation Engine.
Handles:
  - Unified diff generation (difflib)
  - AST / Syntax validity verification
  - Rollback and history snapshots for safe, reversible edits
"""

import ast
import difflib
import time
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EditSnapshot:
    """Historical snapshot for instant rollback."""
    edit_id: str
    original_code: str
    modified_code: str
    timestamp: float = field(default_factory=time.time)
    file_path: Optional[str] = None


class MinimalPatchEngine:
    """
    Computes unified diffs, verifies syntax integrity before saving,
    and maintains rollback checkpoints.
    """

    def __init__(self):
        self._history: Dict[str, EditSnapshot] = {}
        self._history_order: List[str] = []

    def compute_diff(self, original_code: str, modified_code: str, filename: str = "code.py") -> str:
        """
        Generates standard unified diff showing exact modified lines.
        """
        orig_lines = original_code.splitlines()
        mod_lines = modified_code.splitlines()

        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm=""
            )
        )
        return "\n".join(diff_lines)

    def validate_syntax(self, code: str, language: str = "python") -> Tuple[bool, str]:
        """
        Verifies that modified code is structurally valid and compiles without error.
        """
        if language.lower() in ("python", "py"):
            try:
                ast.parse(code)
                return True, "AST syntax check passed successfully."
            except SyntaxError as e:
                return False, f"SyntaxError at line {e.lineno}: {e.msg}"
            except Exception as e:
                return False, f"Validation error: {str(e)}"

        # Non-Python generic validation (basic brace and bracket balance)
        stack = []
        pairs = {')': '(', '}': '{', ']': '['}
        for ch in code:
            if ch in "({[":
                stack.append(ch)
            elif ch in ")}]":
                if not stack or stack[-1] != pairs[ch]:
                    return False, f"Unbalanced delimiter: '{ch}'"
                stack.pop()

        return True, "Delimiter and structural balance check passed."

    def create_snapshot(
        self,
        edit_id: str,
        original_code: str,
        modified_code: str,
        file_path: Optional[str] = None
    ) -> EditSnapshot:
        """
        Saves snapshot to rollback registry.
        """
        snapshot = EditSnapshot(
            edit_id=edit_id,
            original_code=original_code,
            modified_code=modified_code,
            file_path=file_path
        )
        self._history[edit_id] = snapshot
        self._history_order.append(edit_id)
        return snapshot

    def rollback(self, edit_id: str) -> Optional[str]:
        """
        Restores code to the pre-edit state using snapshot ID.
        """
        if edit_id in self._history:
            return self._history[edit_id].original_code
        return None

    def rollback_latest(self) -> Optional[Tuple[str, str]]:
        """
        Rolls back the most recent edit. Returns (edit_id, original_code).
        """
        if not self._history_order:
            return None
        last_id = self._history_order.pop()
        snapshot = self._history.pop(last_id, None)
        if snapshot:
            return snapshot.edit_id, snapshot.original_code
        return None

    def get_history_summary(self) -> List[Dict[str, Any]]:
        """Returns list of past snapshots."""
        return [
            {
                "edit_id": s.edit_id,
                "timestamp": s.timestamp,
                "file_path": s.file_path,
                "lines_original": len(s.original_code.splitlines()),
                "lines_modified": len(s.modified_code.splitlines()),
            }
            for s in self._history.values()
        ]
