"""
Unit Tests for Token-Efficient Direct Code Editing System (Topic 11).
Verifies:
  - Intent classification into Direct, Targeted, and Deep tiers
  - Direct zero-token editing (0 LLM calls, 100% token savings)
  - AST-guided line editing preserving formatting and comments
  - Variable rename and literal replacement
  - Targeted retrieval and micro-context extraction
  - Unified diff generation and AST syntax validation
  - Snapshot rollback
"""

import pytest
from core.editor import (
    TokenEfficientCodeEditor,
    EditIntentClassifier,
    DirectEditEngine,
    TargetedRetrievalEngine,
    MinimalPatchEngine,
    EditTier,
    EditOperation,
)
from core.tokenization.tokenizer import SubwordTokenizer


SAMPLE_PYTHON_CODE = """# Server Configuration
host = "127.0.0.1"
port = 8000  # Default HTTP port
max_retries = 3
debug_mode = False

def calculate_metrics(count: int) -> int:
    multiplier = 2
    total = count * multiplier
    return total

def fetch_data(timeout: int = 10):
    url = "http://localhost:5000/api"
    retries = max_retries
    return url
"""


class TestEditIntentClassifier:
    def setup_method(self):
        self.classifier = EditIntentClassifier()

    def test_classify_assignment_change(self):
        inst = "Change a = 10 to a = 12"
        res = self.classifier.classify(inst)
        assert res.tier == EditTier.DIRECT_EDIT
        assert res.operation == EditOperation.ASSIGNMENT_CHANGE
        assert res.target_variable == "a"
        assert res.old_value == "10"
        assert res.new_value == "12"

    def test_classify_assignment_from_to(self):
        inst = "Change port from 8000 to 8080"
        res = self.classifier.classify(inst)
        assert res.tier == EditTier.DIRECT_EDIT
        assert res.operation == EditOperation.ASSIGNMENT_CHANGE
        assert res.target_variable == "port"
        assert res.old_value == "8000"
        assert res.new_value == "8080"

    def test_classify_variable_rename(self):
        inst = "Rename variable old_counter to total_items"
        res = self.classifier.classify(inst)
        assert res.tier == EditTier.DIRECT_EDIT
        assert res.operation == EditOperation.VARIABLE_RENAME
        assert res.old_value == "old_counter"
        assert res.new_value == "total_items"

    def test_classify_literal_replace(self):
        inst = "Replace 'http://localhost' with 'https://api.example.com'"
        res = self.classifier.classify(inst)
        assert res.tier == EditTier.DIRECT_EDIT
        assert res.operation == EditOperation.LITERAL_REPLACE
        assert res.old_value == "http://localhost"
        assert res.new_value == "https://api.example.com"

    def test_classify_targeted_function(self):
        inst = "Update timeout in function fetch_data"
        res = self.classifier.classify(inst)
        assert res.tier == EditTier.TARGETED_RETRIEVAL
        assert res.operation == EditOperation.FUNCTION_EDIT
        assert res.target_function == "fetch_data"

    def test_classify_deep_reasoning(self):
        inst = "Fix bug where memory leak occurs during async requests"
        res = self.classifier.classify(inst)
        assert res.tier == EditTier.DEEP_REASONING


class TestDirectEditEngine:
    def setup_method(self):
        self.engine = DirectEditEngine()
        self.classifier = EditIntentClassifier()

    def test_apply_assignment_change_preserves_comments(self):
        inst = self.classifier.classify("Change port = 8000 to 9000")
        success, modified, lines, msg = self.engine.apply_edit(SAMPLE_PYTHON_CODE, inst)
        assert success is True
        assert "port = 9000  # Default HTTP port" in modified
        assert "# Server Configuration" in modified
        assert "host = \"127.0.0.1\"" in modified

    def test_apply_variable_rename(self):
        inst = self.classifier.classify("Rename multiplier to factor")
        success, modified, lines, msg = self.engine.apply_edit(SAMPLE_PYTHON_CODE, inst)
        assert success is True
        assert "factor = 2" in modified
        assert "total = count * factor" in modified
        assert "multiplier" not in modified

    def test_apply_literal_replace(self):
        inst = self.classifier.classify("Replace 'http://localhost:5000/api' with 'https://api.prod.com'")
        success, modified, lines, msg = self.engine.apply_edit(SAMPLE_PYTHON_CODE, inst)
        assert success is True
        assert 'url = "https://api.prod.com"' in modified


class TestTargetedRetrievalEngine:
    def setup_method(self):
        self.retriever = TargetedRetrievalEngine()

    def test_extract_function_block(self):
        ctx = self.retriever.extract_function_or_block(SAMPLE_PYTHON_CODE, "calculate_metrics")
        assert ctx is not None
        assert ctx.target_name == "calculate_metrics"
        assert "def calculate_metrics" in ctx.snippet_code
        assert ctx.snippet_tokens < ctx.full_file_tokens
        assert ctx.tokens_saved > 0

    def test_splice_snippet(self):
        ctx = self.retriever.extract_function_or_block(SAMPLE_PYTHON_CODE, "calculate_metrics")
        assert ctx is not None
        edited_snippet = "def calculate_metrics(count: int) -> int:\n    return count * 10\n"
        spliced = self.retriever.splice_snippet(
            SAMPLE_PYTHON_CODE, ctx.start_line, ctx.end_line, edited_snippet
        )
        assert "return count * 10" in spliced
        assert 'host = "127.0.0.1"' in spliced
        assert "def fetch_data" in spliced


class TestPatchEngineAndRollback:
    def setup_method(self):
        self.patcher = MinimalPatchEngine()

    def test_diff_and_syntax_validation(self):
        orig = "x = 1\n"
        mod = "x = 2\n"
        diff = self.patcher.compute_diff(orig, mod)
        assert "-x = 1" in diff
        assert "+x = 2" in diff

        valid, msg = self.patcher.validate_syntax(mod)
        assert valid is True

        # Invalid Python syntax
        invalid_code = "x = = 2"
        valid_bad, msg_bad = self.patcher.validate_syntax(invalid_code)
        assert valid_bad is False
        assert "SyntaxError" in msg_bad

    def test_snapshot_and_rollback(self):
        orig = "a = 10"
        mod = "a = 20"
        self.patcher.create_snapshot("edit-1", orig, mod)

        restored = self.patcher.rollback("edit-1")
        assert restored == orig


class TestTokenEfficientCodeEditor:
    def setup_method(self):
        self.tokenizer = SubwordTokenizer()
        self.editor = TokenEfficientCodeEditor(tokenizer=self.tokenizer)

    def test_zero_token_direct_edit(self):
        """Verify that changing a value directly consumes 0 LLM tokens with 100% savings."""
        # In our sample code, edit max_retries = 3 to 5
        res_real = self.editor.edit_code(
            code=SAMPLE_PYTHON_CODE,
            instruction="Change max_retries = 3 to 5"
        )
        assert res_real.success is True
        assert res_real.tier == EditTier.DIRECT_EDIT
        assert res_real.llm_calls_made == 0
        assert res_real.tokens_consumed == 0
        assert res_real.baseline_tokens > 0
        assert res_real.tokens_saved == res_real.baseline_tokens
        assert res_real.token_savings_pct == 100.0
        assert res_real.syntax_valid is True
        assert "max_retries = 5" in res_real.modified_code
        assert res_real.latency_ms >= 0

    def test_value_mismatch_rejection(self):
        """Verify that when the expected old value does not match current code, the edit is safely rejected."""
        sample = "a = 20\nb = 25\n"
        res = self.editor.edit_code(sample, "Change a = 10 to a = 13")
        assert res.success is False
        assert "Value mismatch" in res.validation_message
        assert res.modified_code == sample
        assert res.diff == ""

    def test_variable_not_found_rejection(self):
        """Verify that attempting to change a non-existent variable is safely rejected."""
        sample = "x = 5\n"
        res = self.editor.edit_code(sample, "Change non_existent_var = 1 to 2")
        assert res.success is False
        assert "not found" in res.validation_message.lower()
        assert res.modified_code == sample

    def test_direct_edit_rollback(self):
        res = self.editor.edit_code(
            code=SAMPLE_PYTHON_CODE,
            instruction="Change port = 8000 to 9090"
        )
        assert res.success is True
        assert "port = 9090" in res.modified_code

        # Rollback using edit_id
        restored = self.editor.patch_engine.rollback(res.edit_id)
        assert restored == SAMPLE_PYTHON_CODE
