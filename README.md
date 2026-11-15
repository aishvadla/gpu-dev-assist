# gpu-dev-assist
Domain-specific RAG assistant for GPU programming documentation, with retrieval evaluation


## Phase 1 — Core RAG pipeline
- [ ] Document ingestion — Load one public Intel oneAPI GPU programming/optimization guide; retain document name, section and page metadata.
- [ ] Chunking — Split content into sensible chunks; preserve technical identifiers and avoid arbitrarily breaking code blocks.
- [ ] Embeddings — Generate dense embeddings using nomic-embed-text through Ollama.
- [ ] Vector database — Store embeddings and metadata in FAISS or Chroma; persist the index so you don't rebuild on every query.
- [ ] Semantic retrieval — Retrieve top-k relevant chunks from user queries.
- [ ] LLM answer generation — Pass retrieved context into an LLM using a prompt that forbids unsupported claims.
- [ ] Source citations — Show source document and page/section in generated answers, tied to retrieved evidence.
- [ ] Insufficient-evidence handling — Return an explicit 'I don't have enough information' response when sources cannot support the answer.

## Phase 2 — Evaluation and engineering
- [ ] Evaluation dataset — Create about 25–30 domain questions, including answerable and unanswerable examples, with source-grounded reference evidence.
- [ ] Retrieval metrics — Measure Recall@k or Hit@k and MRR with labeled relevant source chunks.
- [ ] Answer quality — Evaluate answer correctness, citation support/faithfulness, and abstention on unanswerable questions.
- [ ] Failure analysis — Inspect examples where retrieval misses exact symbols, code syntax, or critical context.
- [ ] Baseline comparison — Compare a basic dense retriever with one meaningful improvement, such as BM25 + dense hybrid retrieval.
- [ ] API and tests — Add FastAPI querying, basic input handling, and pytest coverage of ingestion/retrieval and API behaviors.
- [ ] Reproducibility — Pin dependencies, document setup, use environment variables for configuration, and make indexing repeatable.
- [ ] GitHub README — Include architecture, data sources, sample questions and outputs, metrics, limitations, and setup instructions.
- [ ] Demo — Provide a brief screen recording or GIF showing a question, cited answer, and unanswerable query.

## Phase 3 — Optional differentiation
- [ ] Reranking — Add a cross-encoder or other reranker and measure improvement against the baseline.
- [ ] Agentic tools — Add tool selection for documentation retrieval, public code search, or structured troubleshooting.
- [ ] More documentation — Expand to other public SYCL, compiler, or tuning documents; reindex and evaluate.
- [ ] Deployment — Deploy the API and optionally add a simple UI, authentication, or resource limits.
- [ ] Observability — Capture query latency, retrieval diagnostics, LLM token usage and failure categories.


