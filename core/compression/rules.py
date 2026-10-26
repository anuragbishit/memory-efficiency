"""
Heuristic rules, regex patterns, and semantic anchor whitelists for Non-AI Token Compression.
"""

import re
from typing import Set, List, Tuple

# Conversational boilerplate patterns that can be deterministically pruned or simplified
CONVERSATIONAL_BOILERPLATE: List[Tuple[re.Pattern, str]] = [
    # Polite English greetings and openings
    (re.compile(r"\b(hello|hi|hey|good\s+(morning|afternoon|evening|day))\s*(there|everyone|all)?[\s,!.]*", re.IGNORECASE), ""),
    (re.compile(r"\b(can\s+you\s+(please\s+)?|could\s+you\s+(please\s+)?|would\s+you\s+(mind\s+)?)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(please\s+kindly|kindly\s+please|please|kindly)\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+(am\s+looking\s+to|want\s+to|would\s+like\s+to|was\s+wondering\s+if\s+you\s+(could|might\s+be\s+able\s+to))\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+need\s+you\s+to\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\b(assist\s+me\s+in|help\s+me\s+to|help\s+me\s+with|assist\s+me\s+and|help\s+me\s+and|assist\s+me|help\s+me)\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\band\s+(can\s+you\s+|could\s+you\s+)?(please\s+)?(tell|show|give|provide)\s+(me|us)\b\s*(about)?\s*", re.IGNORECASE), "and "),
    (re.compile(r"\b(can\s+you\s+|could\s+you\s+)?(please\s+)?(tell|show|give|provide)\s+(me|us)\b\s*(about)?\s*", re.IGNORECASE), ""),
    (re.compile(r"\b(explain\s+to\s+(me|us)|share\s+with\s+(me|us))\b\s*(about)?\s*", re.IGNORECASE), "explain "),
    (re.compile(r"\b(as\s+an\s+ai|as\s+a\s+language\s+model)[,\s]*", re.IGNORECASE), ""),

    # Prompt Request Framing & Meta-queries
    (re.compile(r"\bi\s+want\s+to\s+know\s+what\s+is\b", re.IGNORECASE), "what is "),
    (re.compile(r"\bi\s+want\s+to\s+know\s+how\s+to\b", re.IGNORECASE), "how to "),
    (re.compile(r"\bi\s+want\s+to\s+know\s+about\b", re.IGNORECASE), "explain "),
    (re.compile(r"\bi\s+(want|would\s+like)\s+to\s+(know|learn)\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\bi\s+am\s+trying\s+to\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\b(?:can|could)\s+you\s+explain\b", re.IGNORECASE), "explain"),

    # Redundant determiners & Capability clauses
    (re.compile(r"\b(the\s+some|a\s+some|some\s+of\s+the)\b", re.IGNORECASE), "some"),
    (re.compile(r"\b(where|so\s+that|in\s+order\s+that)\s+(i|we|you)\s+can\b", re.IGNORECASE), "to"),
    (re.compile(r"\bhow\s+(i|we|you)\s+can\b", re.IGNORECASE), "how to"),
    (re.compile(r"\b(wonderfull?|awesome|fantastic|superb)\b\s*", re.IGNORECASE), ""),

    # Multilingual Openings (Hindi, Spanish, French)
    (re.compile(r"\b(नमस्ते|नमस्कार|प्रणाम)\b[\s,!.]*", re.IGNORECASE), ""),
    (re.compile(r"\b(कृपया|कृप्या)\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\bक्या\s+आप\s+(मुझे\s+)?(बता\s+सकते\s+हैं\s+कि\s*)?", re.IGNORECASE), ""),
    (re.compile(r"\bऔर\s+यह\s+भी\s+(बताइए\s+कि\s*)?", re.IGNORECASE), "और "),
    (re.compile(r"\b(hola|buenos\s+d[ií]as|buenas\s+tardes)\b[\s,!.]*", re.IGNORECASE), ""),
    (re.compile(r"\b(por\s+favor|hazme\s+el\s+favor\s+de)\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\b(bonjour|salut)\b[\s,!.]*", re.IGNORECASE), ""),
    (re.compile(r"\b(s['\s]il\s+vous\s+pla[iî]t)\b\s*", re.IGNORECASE), ""),

    # Verbose prepositional and transition phrases
    (re.compile(r"\bin\s+order\s+to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bdue\s+to\s+the\s+fact\s+that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bat\s+this\s+point\s+in\s+time\b", re.IGNORECASE), "now"),
    (re.compile(r"\bfor\s+the\s+purpose\s+of\b", re.IGNORECASE), "for"),
    (re.compile(r"\bwith\s+regard\s+to\b", re.IGNORECASE), "regarding"),
    (re.compile(r"\bin\s+the\s+event\s+that\b", re.IGNORECASE), "if"),
    (re.compile(r"\bas\s+a\s+matter\s+of\s+fact\b", re.IGNORECASE), "actually"),

    # Closing fluff and gratitude boilerplate (English & Multilingual)
    (re.compile(r"[,.\s]*(?:thank\s+you|thanks|i\s+appreciate)[\s\S]*$", re.IGNORECASE), ""),
    (re.compile(r"\blet\s+me\s+know\s+what\s+you\s+think[\s.!]*$", re.IGNORECASE), ""),
    (re.compile(r"[,\s]*(धन्यवाद|बहुत\s+धन्यवाद)[\s.!]*$", re.IGNORECASE), ""),
    (re.compile(r"[,\s]*(muchas\s+gracias|gracias)[\s.!]*$", re.IGNORECASE), ""),
    (re.compile(r"[,\s]*(merci\s+beaucoup|merci)[\s.!]*$", re.IGNORECASE), ""),
]

# Punctuation & Whitespace cleanup patterns
PUNCTUATION_REDUNDANCY: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\?{2,}"), "?"),
    (re.compile(r"!{2,}"), "!"),
    (re.compile(r"\.{4,}"), "..."),
    (re.compile(r"\s+([?!.,;:])"), r"\1"),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]

# Strict Whitelist: Words that must NEVER be stripped under any compression level
SEMANTIC_ANCHORS: Set[str] = {
    # Negations (Meaning inversion hazard)
    "not", "no", "never", "none", "neither", "nor", "cannot", "can't",
    "won't", "don't", "doesn't", "didn't", "shouldn't", "wouldn't",
    "couldn't", "isn't", "aren't", "wasn't", "weren't", "haven't",
    "hasn't", "hadn't", "without", "except", "excluding", "against",

    # Constraints and Boundary enforcement
    "must", "should", "shall", "always", "only", "strict", "strictly",
    "require", "required", "ensure", "mandatory", "limit", "bounded",
    "maximum", "minimum", "max", "min", "at least", "at most", "exact",
    "exactly", "constant", "atomic", "isolated",

    # Logic & Relations
    "if", "then", "else", "and", "or", "xor", "implies", "iff",
    "because", "therefore", "thus", "hence", "where", "while",
    "depends_on", "implements", "inherits", "extends",

    # Mathematical / Complexity symbols
    "o(1)", "o(n)", "o(log n)", "o(n log n)", "o(n^2)", "o(2^n)",
}

# Low-information conversational fillers for BALANCED / AGGRESSIVE mode
LOW_INFO_FILLERS: Set[str] = {
    "basically", "actually", "literally", "honestly", "really",
    "totally", "simply", "just", "kind of", "sort of", "somewhat",
    "quite", "perhaps", "maybe", "obviously", "clearly", "essentially",
    "definitely", "certainly", "surely", "undoubtedly", "absolutely",
    "frankly", "truthfully",
    "wonderful", "wonderfull", "awesome", "amazing", "superb"
}
