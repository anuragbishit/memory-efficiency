"""
Subword Tokenization Module
Supports Byte-Pair Encoding (BPE), OpenAI cl100k_base, and o200k_base.
Includes detailed token breakdown, multi-encoding comparison, and educational BPE reference mechanics.
"""

import os
import unicodedata
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import tiktoken

# Ensure local offline cache directory is configured
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_DEFAULT_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache", "tiktoken")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", _DEFAULT_CACHE_DIR)


@dataclass
class TokenDetail:
    """Detailed metadata for an individual token."""
    token_id: int
    raw_bytes: bytes
    text: str
    byte_length: int
    is_multibyte: bool


class SubwordTokenizer:
    """
    Subword Tokenizer utilizing production Byte-Pair Encoding (BPE).
    Primary support for:
      - cl100k_base (GPT-4, GPT-3.5-Turbo, text-embedding-ada-002)
      - o200k_base (GPT-4o, Omni models)
    """

    SUPPORTED_ENCODINGS = ["cl100k_base", "o200k_base"]
    _cached_encoders: Dict[str, tiktoken.Encoding] = {}

    def __init__(self, default_encoding: str = "cl100k_base"):
        if default_encoding not in self.SUPPORTED_ENCODINGS:
            raise ValueError(
                f"Unsupported encoding '{default_encoding}'. "
                f"Supported encodings: {self.SUPPORTED_ENCODINGS}"
            )
        self.default_encoding_name = default_encoding

    def get_encoder(self, encoding_name: str | None = None) -> tiktoken.Encoding:
        """Lazily retrieves and caches the requested BPE encoding."""
        name = encoding_name or self.default_encoding_name
        if name not in self.SUPPORTED_ENCODINGS:
            raise ValueError(f"Unknown encoding '{name}'. Must be one of {self.SUPPORTED_ENCODINGS}")

        if name not in self._cached_encoders:
            self._cached_encoders[name] = tiktoken.get_encoding(name)

        return self._cached_encoders[name]

    def count_tokens(self, text: str, encoding_name: str | None = None) -> int:
        """Returns the exact number of subword tokens for a given string."""
        if not text:
            return 0
        encoder = self.get_encoder(encoding_name)
        return len(encoder.encode(text))

    def encode(self, text: str, encoding_name: str | None = None) -> List[int]:
        """Encodes text into a list of integer token IDs."""
        if not text:
            return []
        encoder = self.get_encoder(encoding_name)
        return encoder.encode(text)

    def decode(self, token_ids: List[int], encoding_name: str | None = None) -> str:
        """Decodes a list of token IDs back into a Unicode string."""
        if not token_ids:
            return ""
        encoder = self.get_encoder(encoding_name)
        return encoder.decode(token_ids)

    def inspect_tokens(self, text: str, encoding_name: str | None = None) -> List[TokenDetail]:
        """
        Tokenizes text and returns detailed token fragments,
        revealing exactly where subword splits, whitespace preservation,
        and multi-byte encodings occur.
        """
        if not text:
            return []

        encoder = self.get_encoder(encoding_name)
        token_ids = encoder.encode(text)
        details: List[TokenDetail] = []

        for tid in token_ids:
            raw_bytes = encoder.decode_single_token_bytes(tid)
            # Safely represent text representation for display
            decoded_repr = raw_bytes.decode("utf-8", errors="replace")
            details.append(
                TokenDetail(
                    token_id=tid,
                    raw_bytes=raw_bytes,
                    text=decoded_repr,
                    byte_length=len(raw_bytes),
                    is_multibyte=len(raw_bytes) > 1,
                )
            )
        return details

    def compare_encodings(self, text: str) -> Dict[str, Any]:
        """
        Compares tokenization characteristics between cl100k_base and o200k_base.
        Useful for measuring vocabulary efficiency and token inflation.
        """
        results = {
            "input_characters": len(text),
            "input_bytes": len(text.encode("utf-8")),
            "encodings": {}
        }

        for enc_name in self.SUPPORTED_ENCODINGS:
            tokens = self.encode(text, enc_name)
            char_to_token_ratio = (
                round(len(text) / len(tokens), 3) if tokens else 0.0
            )
            bytes_per_token = (
                round(len(text.encode("utf-8")) / len(tokens), 3) if tokens else 0.0
            )

            results["encodings"][enc_name] = {
                "token_count": len(tokens),
                "char_to_token_ratio": char_to_token_ratio,
                "bytes_per_token": bytes_per_token,
                "token_ids": tokens[:15]  # first 15 for preview
            }

        cl_count = results["encodings"]["cl100k_base"]["token_count"]
        o2_count = results["encodings"]["o200k_base"]["token_count"]

        if cl_count > 0:
            diff_percent = round(((o2_count - cl_count) / cl_count) * 100, 2)
        else:
            diff_percent = 0.0

        results["o200k_vs_cl100k_delta_percent"] = diff_percent
        return results


class EducationalBPE:
    """
    Pure Python Reference Implementation of Byte-Pair Encoding (BPE).
    Demonstrates the core algorithmic principle of iterative pair merges
    for project presentation and theoretical verification.
    """

    def __init__(self, vocab_size_target: int = 50):
        self.vocab_size_target = vocab_size_target
        self.merges: Dict[Tuple[bytes, bytes], bytes] = {}
        self.vocab: List[bytes] = []

    def train(self, corpus: str) -> List[Tuple[bytes, bytes]]:
        """
        Learns BPE merges from a text corpus by iteratively counting
        consecutive byte pairs and merging the most frequent pair.
        """
        raw_bytes = corpus.encode("utf-8")
        tokens = [bytes([b]) for b in raw_bytes]

        num_merges = self.vocab_size_target - 256
        if num_merges <= 0:
            return []

        learned_merges = []
        for _ in range(num_merges):
            stats: Dict[Tuple[bytes, bytes], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                stats[pair] = stats.get(pair, 0) + 1

            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                # No repeated pairs left
                break

            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            learned_merges.append(best_pair)

            # Re-tokenize with the new merged token
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return learned_merges
