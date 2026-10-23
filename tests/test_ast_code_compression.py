"""
Unit tests for AST Code Compression and Preservation Engine (Topic 5).
"""

import ast
import pytest
from core.compression.ast_code_compressor import ASTCodeCompressor


def test_ast_python_docstring_stripping():
    compressor = ASTCodeCompressor()
    python_code = (
        "def fibonacci(n: int) -> int:\n"
        '    """Calculates the n-th Fibonacci number recursively."""\n'
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
    )

    unparsed, parsed_ok, is_equiv = compressor.compress_python_ast(python_code, strip_docstrings=True)

    assert parsed_ok is True
    assert is_equiv is True
    assert "Calculates the n-th Fibonacci number recursively." not in unparsed
    assert "def fibonacci(n: int) -> int:" in unparsed
    assert "return fibonacci(n - 1) + fibonacci(n - 2)" in unparsed


def test_ast_semantic_equivalence_verification():
    compressor = ASTCodeCompressor()
    orig = "def add(x, y):\n    '''Doc'''\n    return x + y\n"
    tree1 = ast.parse(orig)
    tree2 = ast.parse("def add(x, y):\n    return x + y\n")

    assert compressor._verify_ast_equivalence(tree1, tree2) is True


def test_compress_code_block_fenced():
    compressor = ASTCodeCompressor()
    fenced_block = (
        "```python\n"
        "class CacheManager:\n"
        '    """Thread-safe cache manager class."""\n'
        "    def __init__(self, capacity: int):\n"
        "        self.capacity = capacity\n"
        "        self.store = {}\n"
        "\n"
        "    def get(self, key: str):\n"
        '        """Retrieve value."""\n'
        "        return self.store.get(key)\n"
        "```"
    )

    res = compressor.compress_code_block(fenced_block, strip_docstrings=True)

    assert res.ast_parsed is True
    assert res.semantic_equivalence_verified is True
    assert res.tokens_saved > 0
    assert "Thread-safe cache manager class." not in res.compressed_code
    assert "class CacheManager:" in res.compressed_code
    assert "def get(self, key: str):" in res.compressed_code


def test_generic_non_python_code_compression():
    compressor = ASTCodeCompressor()
    js_code = (
        "```javascript\n"
        "// Non-essential comment\n"
        "function computeHash(data) {\n"
        "    /* Multi-line\n"
        "       comment */\n"
        "    return crypto.createHash('sha256').update(data).digest('hex');\n"
        "}\n"
        "```"
    )

    res = compressor.compress_code_block(js_code)

    assert res.language == "javascript"
    assert "Non-essential comment" not in res.compressed_code
    assert "Multi-line" not in res.compressed_code
    assert "function computeHash(data)" in res.compressed_code
    assert res.tokens_saved > 0


def test_multi_language_support_rust_cpp_csharp_java():
    compressor = ASTCodeCompressor()
    rust_code = (
        "```rust\n"
        "// Prune this regular comment\n"
        "// MUST run in O(1) time complexity\n"
        "pub fn search_index(id: u64) -> Option<Item> {\n"
        "    /* multi-line comment */\n"
        "    INDEX.get(&id).cloned()\n"
        "}\n"
        "```"
    )
    res = compressor.compress_code_block(rust_code)
    assert res.language == "rust"
    assert "Prune this regular comment" not in res.compressed_code
    assert "MUST run in O(1) time complexity" in res.compressed_code
    assert "pub fn search_index" in res.compressed_code
    assert res.tokens_saved > 0


def test_multi_language_support_bash_ruby_hash_family():
    compressor = ASTCodeCompressor()
    bash_code = (
        "```bash\n"
        "# Regular boilerplate comment\n"
        "# MUST preserve this constraint\n"
        "export API_KEY='token'\n"
        "curl -X POST https://api.openai.com\n"
        "```"
    )
    res = compressor.compress_code_block(bash_code)
    assert res.language == "bash"
    assert "Regular boilerplate comment" not in res.compressed_code
    assert "MUST preserve this constraint" in res.compressed_code
    assert "curl -X POST" in res.compressed_code


def test_multi_language_support_sql_lua_dash_family():
    compressor = ASTCodeCompressor()
    sql_code = (
        "```sql\n"
        "-- Regular query comment\n"
        "-- MUST use cluster index\n"
        "SELECT id, username FROM users WHERE is_active = 1;\n"
        "```"
    )
    res = compressor.compress_code_block(sql_code)
    assert res.language == "sql"
    assert "Regular query comment" not in res.compressed_code
    assert "MUST use cluster index" in res.compressed_code
    assert "SELECT id, username" in res.compressed_code


def test_universal_unknown_programming_language():
    compressor = ASTCodeCompressor()
    # Completely custom or esoteric programming language tag
    custom_code = (
        "```customlang\n"
        "// A comment in custom language\n"
        "PROCESS item INTO stream WITH priority=HIGH;\n"
        "```"
    )
    res = compressor.compress_code_block(custom_code)
    assert res.language == "customlang"
    assert "A comment in custom language" not in res.compressed_code
    assert "PROCESS item INTO stream" in res.compressed_code
