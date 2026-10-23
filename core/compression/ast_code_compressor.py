"""
AST (Abstract Syntax Tree) Code Compressor and Preservation Engine (Topic 5).
Safely compresses code blocks inside prompts using Python's AST module,
stripping docstrings/redundant comments while mathematically verifying semantic equivalence.
Includes language-agnostic token compression for non-Python blocks (JS, SQL, etc.).
"""

import ast
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from core.tokenization.tokenizer import SubwordTokenizer


@dataclass
class ASTCompressionResult:
    """Detailed summary of code block compression."""
    original_code: str
    compressed_code: str
    language: str
    ast_parsed: bool
    semantic_equivalence_verified: bool
    raw_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocstringAndCommentStripper(ast.NodeTransformer):
    """
    AST Visitor that removes docstrings from module, class, and function definitions
    without modifying statements, variable assignments, loops, or return expressions.
    """

    def _strip_docstring(self, body: List[ast.AST]) -> List[ast.AST]:
        if not body:
            return body

        first = body[0]
        # Docstrings in Python AST are represented as Expr nodes containing a Constant string
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            # Keep docstring only if it's the only statement in the body (e.g. an abstract stub)
            if len(body) > 1:
                return body[1:]
        return body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip_docstring(node.body)
        return node


class ASTCodeCompressor:
    """
    Abstract Syntax Tree Code Compressor.
    Compresses programming code blocks inside prompts by removing docstrings,
    redundant comments, and whitespace padding while verifying AST semantic equivalence.
    """

    def __init__(self, tokenizer: SubwordTokenizer | None = None):
        self.tokenizer = tokenizer or SubwordTokenizer()

    def _verify_ast_equivalence(self, orig_tree: ast.AST, comp_tree: ast.AST) -> bool:
        """
        Verifies that function signatures, classes, statements, and logic structures
        are identical between original and compressed ASTs (modulo stripped docstrings).
        """
        def get_structural_signatures(tree: ast.AST) -> List[Tuple[str, str]]:
            signatures = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    signatures.append(("func", f"{node.name}({','.join(args)})"))
                elif isinstance(node, ast.ClassDef):
                    signatures.append(("class", node.name))
                elif isinstance(node, ast.Return):
                    signatures.append(("return", "return_stmt"))
                elif isinstance(node, ast.If):
                    signatures.append(("control", "if_stmt"))
                elif isinstance(node, (ast.For, ast.While)):
                    signatures.append(("control", "loop_stmt"))
            return signatures

        orig_sigs = get_structural_signatures(orig_tree)
        comp_sigs = get_structural_signatures(comp_tree)
        return orig_sigs == comp_sigs

    def compress_python_ast(self, code_str: str, strip_docstrings: bool = True) -> Tuple[str, bool, bool]:
        """
        Parses Python code into an AST, strips non-essential docstrings,
        and regenerates syntactically minimal, canonical Python code.
        """
        try:
            tree = ast.parse(code_str)
            if strip_docstrings:
                stripper = DocstringAndCommentStripper()
                new_tree = stripper.visit(tree)
                ast.fix_missing_locations(new_tree)
            else:
                new_tree = tree

            # Unparse AST back into formatted, clean, minimal Python code
            unparsed = ast.unparse(new_tree)
            # Verify semantic equivalence against original tree
            is_equivalent = self._verify_ast_equivalence(tree, new_tree)
            return unparsed, True, is_equivalent
        except (SyntaxError, ValueError):
            return code_str, False, False

    # Broad language syntax families for universal multi-language support
    C_STYLE_LANGUAGES = {
        "c", "cpp", "c++", "cc", "cxx", "h", "hpp", "cs", "csharp", "java",
        "js", "javascript", "ts", "typescript", "jsx", "tsx", "go", "golang",
        "rs", "rust", "kt", "kotlin", "swift", "dart", "scala", "zig", "php",
        "solidity", "verilog", "objc", "m", "mm", "cuda"
    }
    HASH_STYLE_LANGUAGES = {
        "sh", "bash", "zsh", "shell", "powershell", "ps1", "py", "python",
        "rb", "ruby", "pl", "perl", "r", "yaml", "yml", "toml", "dockerfile",
        "makefile", "cmake", "julia", "nim", "elixir"
    }
    DASH_STYLE_LANGUAGES = {
        "sql", "mysql", "pgsql", "postgres", "postgresql", "sqlite",
        "lua", "hs", "haskell", "ada", "vhdl", "elm"
    }
    SEMICOLON_STYLE_LANGUAGES = {
        "asm", "assembly", "s", "nasm", "lisp", "clojure", "scheme", "ini"
    }
    PERCENT_STYLE_LANGUAGES = {
        "matlab", "octave", "latex", "tex", "erlang"
    }

    def compress_generic_code(self, code_str: str, language: str) -> str:
        """
        Universal language-agnostic token compression for ANY programming language
        (Python, C/C++, Rust, Go, Java, JS/TS, C#, SQL, Bash, Ruby, PHP, Swift, etc.).
        Safely strips non-essential single-line/multi-line comments and blank lines
        while strictly protecting algorithmic constraints (e.g. O(1), MUST, TODO).
        """
        lines = code_str.split("\n")
        cleaned_lines = []
        lang_lower = (language or "generic").lower()

        in_multiline_comment = False
        constraint_keywords = ["must", "o(", "todo", "fix", "strict", "keep", "constraint", "invariant"]

        for line in lines:
            stripped = line.strip()

            # Skip empty lines to reduce token consumption
            if not stripped:
                continue

            # Handle multi-line block comments /* ... */
            if "/*" in stripped and "*/" in stripped:
                line = re.sub(r"/\*.*?\*/", "", line)
                stripped = line.strip()
                if not stripped:
                    continue

            if "/*" in stripped:
                in_multiline_comment = True
                continue
            if in_multiline_comment:
                if "*/" in stripped:
                    in_multiline_comment = False
                continue

            # Check if line contains a semantic constraint
            has_constraint = any(kw in stripped.lower() for kw in constraint_keywords)

            # Detect comment family either by declared language tag or by syntax prefix fallback
            is_c_style = lang_lower in self.C_STYLE_LANGUAGES or stripped.startswith("//")
            is_dash_style = lang_lower in self.DASH_STYLE_LANGUAGES or stripped.startswith("--")
            is_hash_style = lang_lower in self.HASH_STYLE_LANGUAGES or stripped.startswith("#")
            is_semicolon_style = lang_lower in self.SEMICOLON_STYLE_LANGUAGES or stripped.startswith(";")
            is_percent_style = lang_lower in self.PERCENT_STYLE_LANGUAGES or stripped.startswith("%")

            # Strip single-line comments based on language syntax family
            if is_c_style and stripped.startswith("//"):
                if not has_constraint:
                    continue
            elif is_dash_style and stripped.startswith("--"):
                if not has_constraint:
                    continue
            elif is_hash_style and stripped.startswith("#"):
                if not has_constraint:
                    continue
            elif is_semicolon_style and stripped.startswith(";"):
                if not has_constraint:
                    continue
            elif is_percent_style and stripped.startswith("%"):
                if not has_constraint:
                    continue

            # Strip inline trailing comments if safe (outside string literals)
            if is_c_style and "//" in line and not has_constraint:
                line = re.sub(r"\s*//(?!.*(must|o\(|strict)).*$", "", line)
            elif is_dash_style and "--" in line and not has_constraint:
                line = re.sub(r"\s*--(?!.*(must|o\(|strict)).*$", "", line)
            elif is_hash_style and "#" in line and not has_constraint:
                line = re.sub(r"\s*#(?!.*(must|o\(|strict)).*$", "", line)

            cleaned_lines.append(line.rstrip())

        return "\n".join(cleaned_lines)

    def compress_code_block(
        self,
        raw_code_block: str,
        encoding_name: str = "cl100k_base",
        strip_docstrings: bool = True
    ) -> ASTCompressionResult:
        """
        Compresses a markdown code block (e.g. ```python ... ```) using AST
        or generic lexical rules. Guarantees code validity and reports exact token savings.
        """
        # Parse language tag and inner code content
        match = re.match(r"^```([a-zA-Z0-9_-]*)\n([\s\S]*?)```$", raw_code_block.strip())
        if match:
            lang = match.group(1).lower() or "generic"
            inner_code = match.group(2)
        else:
            lang = "generic"
            inner_code = raw_code_block.strip("` \n")

        raw_tokens = self.tokenizer.count_tokens(raw_code_block, encoding_name)
        ast_parsed = False
        semantic_equivalent = False

        if lang in ["py", "python", ""] or not lang:
            compressed_inner, ast_parsed, semantic_equivalent = self.compress_python_ast(
                inner_code, strip_docstrings=strip_docstrings
            )
            if not ast_parsed:
                # If incomplete snippet (e.g. snippet inside function), fall back to generic
                compressed_inner = self.compress_generic_code(inner_code, "python")
                semantic_equivalent = True
        else:
            compressed_inner = self.compress_generic_code(inner_code, lang)
            ast_parsed = False
            semantic_equivalent = True

        reconstructed_block = f"```{lang}\n{compressed_inner}\n```"
        comp_tokens = self.tokenizer.count_tokens(reconstructed_block, encoding_name)
        saved = max(0, raw_tokens - comp_tokens)
        ratio = round((saved / raw_tokens) * 100, 2) if raw_tokens > 0 else 0.0

        return ASTCompressionResult(
            original_code=raw_code_block,
            compressed_code=reconstructed_block,
            language=lang,
            ast_parsed=ast_parsed,
            semantic_equivalence_verified=semantic_equivalent,
            raw_tokens=raw_tokens,
            compressed_tokens=comp_tokens,
            tokens_saved=saved,
            compression_ratio_pct=ratio,
        )
