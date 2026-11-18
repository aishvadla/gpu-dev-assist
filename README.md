# gpu-dev-assist
Domain-specific RAG assistant for GPU programming documentation, with retrieval evaluation


## Phase 1 — Core RAG pipeline
- [x] Document ingestion: Intel oneAPI GPU Optimization Guide (2025.0) and Programming Guide (2025.1); source file and page kept on every chunk
- [x] Chunking: recursive character splitting within pages (1000 / 200)
- [x] Noise filtering: drop title pages and table-of-contents chunks (measured improvement; see docs/experiment_log.md)
- [x] Chunk size check: verify chunks fit the embedding model's 256-token input limit
- [x] Embeddings — Generate dense embeddings via sentence-transformers
- [x] Vector database — Store embeddings and metadata in Chroma; persist the index so you don't rebuild on every query.
- [x] Semantic retrieval — Retrieve top-k relevant chunks from user queries with source, page and distance
- [x] LLM answer generation — Pass retrieved context into an LLM using a prompt that forbids unsupported claims.
- [x] Source citations — Show source document and page/section in generated answers, tied to retrieved evidence.
- [x] Insufficient-evidence handling — Return an explicit 'I don't have enough information' response when sources cannot support the answer.
- [x] Unit tests: chunking, ingestion (no duplicates, metadata kept) and generation (citation mapping, refusals), with the LLM mocked
- [x] CI: GitHub Actions runs the tests on every push

## Phase 2 — Evaluation and engineering
- [ ] Evaluation dataset — Create about 25–30 domain questions, including answerable and unanswerable examples, with source-grounded reference evidence.
- [ ] Retrieval metrics — Measure Recall@k or Hit@k and MRR with labeled relevant source chunks.
- [ ] Answer quality — Evaluate answer correctness, citation support/faithfulness, and abstention on unanswerable questions.
- [ ] Failure analysis — Inspect examples where retrieval misses exact symbols, code syntax, or critical context.
- [ ] Baseline comparison — Compare a basic dense retriever with one meaningful improvement, such as BM25 + dense hybrid retrieval.
- [ ] API and tests — Add FastAPI querying, basic input handling, and pytest coverage of ingestion/retrieval and API behaviors.
- [ ] Reproducibility — Pin dependencies, document setup, use environment variables for configuration, and make indexing repeatable.
- [ ] Docker: containerized API
- [ ] Deployment — Deploy the API and optionally add a simple UI, authentication, or resource limits.
- [ ] GitHub README — Include architecture, data sources, sample questions and outputs, metrics, limitations, and setup instructions.
- [ ] Demo — Provide a brief screen recording or GIF showing a question, cited answer, and unanswerable query.

## Phase 3
- [ ] Reranking — Add a cross-encoder or other reranker and measure improvement against the baseline.
- [ ] Structure-aware chunking: preserve technical identifiers and avoid arbitrarily breaking code blocks.
- [ ] More documentation — Expand to other public SYCL, compiler, or tuning documents; reindex and evaluate.
- [ ] Agentic tools — Add tool selection for documentation retrieval, public code search, or structured troubleshooting.
- [ ] Observability — Capture query latency, retrieval diagnostics, LLM token usage and failure categories.


