"""
Training dataset for Lightweight Intent Classification (TF-IDF & Logistic Regression).
Contains curated samples across 5 technical intent classes for prompt routing.
"""

INTENT_CLASSES = [
    "FACTUAL_LOOKUP",
    "EXPLANATION",
    "CODE_GENERATION",
    "SYSTEM_ARCHITECTURE",
    "DEBUGGING",
]

INTENT_TRAINING_CORPUS = [
    # 1. FACTUAL_LOOKUP (Tier: LOW -> LOCAL_SLM)
    ("What is the capital of France?", "FACTUAL_LOOKUP"),
    ("Who invented the Python programming language?", "FACTUAL_LOOKUP"),
    ("Define what an API stands for.", "FACTUAL_LOOKUP"),
    ("What is the speed of light in vacuum?", "FACTUAL_LOOKUP"),
    ("Hello there, how are you today?", "FACTUAL_LOOKUP"),
    ("What is the ASCII value of character A?", "FACTUAL_LOOKUP"),
    ("List the primary colors.", "FACTUAL_LOOKUP"),
    ("What is the boiling point of water at sea level?", "FACTUAL_LOOKUP"),
    ("Tell me the date of the Apollo 11 moon landing.", "FACTUAL_LOOKUP"),
    ("What is a compiler?", "FACTUAL_LOOKUP"),

    # 2. EXPLANATION (Tier: MEDIUM -> LOCAL_SLM)
    ("Explain how a binary search tree maintains sorted order.", "EXPLANATION"),
    ("Why does GraphRAG eliminate context bleed compared to standard text-chunk RAG?", "EXPLANATION"),
    ("How does Byte-Pair Encoding subword tokenization work step by step?", "EXPLANATION"),
    ("Explain the difference between synchronous and asynchronous execution in Python.", "EXPLANATION"),
    ("How does backpropagation update weights in a neural network?", "EXPLANATION"),
    ("Why are BPE tokenizers biased against non-English scripts?", "EXPLANATION"),
    ("Explain the concept of virtual memory and page tables in operating systems.", "EXPLANATION"),
    ("How does an atomic transaction maintain ACID properties in relational databases?", "EXPLANATION"),
    ("What is the difference between TCP and UDP protocols?", "EXPLANATION"),
    ("Explain how post-training quantization converts FP16 weights to INT4.", "EXPLANATION"),

    # 3. CODE_GENERATION (Tier: HIGH -> CLOUD_API)
    ("Write a Python function to sort an array using quicksort with O(n log n) average time.", "CODE_GENERATION"),
    ("Implement a thread-safe LRU cache in Python with O(1) get and put operations.", "CODE_GENERATION"),
    ("Write code to reverse a singly linked list in-place.", "CODE_GENERATION"),
    ("Create a React hook for debouncing search input queries.", "CODE_GENERATION"),
    ("Implement Dijkstra's shortest path algorithm using a min-heap priority queue in Python.", "CODE_GENERATION"),
    ("Write a recursive function to calculate the Fibonacci series with memoization.", "CODE_GENERATION"),
    ("Implement a Trie data structure with insert, search, and startsWith methods in C++.", "CODE_GENERATION"),
    ("Generate a Python script to parse CSV files and compute column summary statistics.", "CODE_GENERATION"),
    ("Write a binary search algorithm in Python that returns the index of target element.", "CODE_GENERATION"),
    ("Implement an asynchronous HTTP client pool using aiohttp in Python.", "CODE_GENERATION"),

    # 4. SYSTEM_ARCHITECTURE (Tier: HIGH -> CLOUD_API)
    ("Architect a distributed sharded cache system supporting thread-safe writes and replication.", "SYSTEM_ARCHITECTURE"),
    ("Design a high-throughput event-driven microservices architecture using Apache Kafka.", "SYSTEM_ARCHITECTURE"),
    ("How should we architect a multi-region database with strong consistency and low latency?", "SYSTEM_ARCHITECTURE"),
    ("Design an end-to-end GraphRAG pipeline with dynamic entity resolution and knowledge graph defragmentation.", "SYSTEM_ARCHITECTURE"),
    ("Architect a scalable real-time analytics streaming pipeline processing 100k events per second.", "SYSTEM_ARCHITECTURE"),
    ("Design a fault-tolerant distributed consensus protocol using Raft or Paxos.", "SYSTEM_ARCHITECTURE"),
    ("Architect a long-term AI memory store using atomic nodes and typed edges to prevent context bleed.", "SYSTEM_ARCHITECTURE"),
    ("Design an edge computing topology for running quantized Small Language Models on consumer hardware.", "SYSTEM_ARCHITECTURE"),
    ("How to architect an API gateway supporting rate limiting, circuit breaking, and JWT auth?", "SYSTEM_ARCHITECTURE"),
    ("Design a distributed object storage architecture with erasure coding and data deduplication.", "SYSTEM_ARCHITECTURE"),

    # 5. DEBUGGING (Tier: HIGH -> CLOUD_API)
    ("Fix this Python code: TypeError: unsupported operand type(s) for +: 'int' and 'str'.", "DEBUGGING"),
    ("Debug why my async loop is deadlocking when acquiring redis locks concurrently.", "DEBUGGING"),
    ("Resolve this memory leak: my PyTorch training loop runs out of CUDA memory after 50 epochs.", "DEBUGGING"),
    ("Why is this SQL query resulting in a slow sequential table scan instead of using the index?", "DEBUGGING"),
    ("Fix this recursion error: RecursionError: maximum recursion depth exceeded in comparison.", "DEBUGGING"),
    ("Debug this null pointer exception in Java when accessing user session attributes.", "DEBUGGING"),
    ("Resolve this CORS header error when making fetch requests from localhost to my FastAPI backend.", "DEBUGGING"),
    ("Troubleshoot why my Docker container exits with code 137 OOMKilled during build.", "DEBUGGING"),
    ("Fix the race condition in this multithreaded bank account balance transfer code.", "DEBUGGING"),
    ("Why does my BPE tokenizer throw an out of bounds token ID error during decoding?", "DEBUGGING"),
]
