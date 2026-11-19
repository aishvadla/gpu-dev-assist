# GPU DevAssist: experiment log

A running record of setups, results and decisions. The README's evaluation and design sections are written from this file.

---

## Setup (baseline)

| Item | Value |
|---|---|
| Corpus | Intel oneAPI GPU Optimization Guide 2025.0 (405 pages) + Intel oneAPI Programming Guide 2025.1 (129 pages) |
| Loader | LangChain `PyPDFLoader`, one document per page (keeps page metadata) |
| Chunking | `RecursiveCharacterTextSplitter`, 1000 characters, 200 overlap, split within pages |
| Chunks | 1,585 after noise filtering |
| Embeddings | `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token input limit); was `all-MiniLM-L6-v2` until 2026-11-18 |
| Vector store | Chroma, persistent, cosine distance |
| LLM | Claude Haiku via the Anthropic API |
| Retrieval | Top 6 by cosine similarity |

---

## 2026-11-17: Retrieval noise from title pages and table of contents

**Observation.** Query "what is oneapi programming model?" (top 6) returned the guide's title page (#2) and table of contents (#3). These chunks share the query's words but contain no usable content. The page 6 overview, one of the best answers, fell to #7, just outside the top 6.

**Also observed.** A one-letter typo in the query ("programmig") reordered the near-tied results (distances 0.335–0.368), which is enough to move a relevant chunk across the cutoff.

**Change.** Filter chunks before embedding:
- shorter than 100 characters (titles, stray headers)
- 3 or more table-of-contents lines (5+ consecutive dots)

**Result (2026-11-18).**

| | Before filter | After filter |
|---|---|---|
| Title page / table of contents in top 5 | Yes (#2, #3) | Gone |
| Page 6 overview | #7 (outside top 6) | #5 |
| Best match distance | 0.232 | 0.193 |

**Takeaway.** A single query is unreliable evidence; retrieval quality needs a question set and Recall@k. Covered by unit tests in `tests/test_chunking_embedding.py`.

---

## 2026-11-18: First generation results with citations

Prompt design: numbered sources in the prompt; every factual sentence must cite `[n]`; code maps `[n]` back to source and page; exact refusal sentence when evidence is missing.

| Question | Outcome | Notes |
|---|---|---|
| What is the oneAPI programming model? | Answered, every sentence cited | Cited pages 5, 6, 7, 46, the same pages retrieval ranked highest. To do: spot-check page 46. |
| When should I use shared local memory in a SYCL kernel? | **False refusal** | The optimization guide has a full section on shared local memory. Diagnosing: retrieval miss or over-strict prompt? |
| What is NVIDIA's CUDA warp size? | Partial answer with caveat | Not truly unanswerable: page 60 mentions NVIDIA's 32 work-items per sub-group. The model cited it and said the docs don't confirm it. No fabrication. |

**Takeaways for the evaluation set.**
1. Verify "unanswerable" questions are really absent from the corpus (search the PDFs) before adding them.
2. Score three outcomes separately: answered, refused, partial with caveat.

---

## 2026-11-18: Embedding model input limit (MiniLM → bge-small)

**Observation.** Chunks were longer than the embedding model could read. With 1000-character chunks, token counts (using the model's own tokenizer) were median 238, max 524. `all-MiniLM-L6-v2` reads at most 256 tokens, so 664 of 1,585 chunks (42%) were truncated before embedding. The full text is still stored and sent to the LLM, but anything past the limit couldn't contribute to retrieval.

**Options considered.**
- Smaller chunks (~600 characters): fits MiniLM, but nearly doubles the chunk count and splits explanations more often.
- An embedding model with a longer input limit: keeps chunks intact.

**Change.** Switched to `BAAI/bge-small-en-v1.5` (512-token limit, same 384 dimensions). Chunk size kept at 1000 so only one variable changed. Index fully rebuilt (embeddings from different models can't be mixed).

**Result.**

| | all-MiniLM-L6-v2 | bge-small-en-v1.5 |
|---|---|---|
| Input limit | 256 tokens | 512 tokens |
| Chunks over the limit | 664 of 1,585 (42%) | 2 of 1,585 (0%) |
| "What is the oneAPI programming model?" | Answered, cited (p. 5, 6, 7, 46) | Answered, cited (p. 4, 5, 6, 7, 46) |
| "When should I use shared local memory in a SYCL kernel?" | False refusal | Still a false refusal |
| "What is NVIDIA's CUDA warp size?" | Partial answer citing p. 60, with caveat | Exact refusal |

**Takeaways.**
1. Truncation is fixed, but it was **not** the cause of the shared local memory false refusal. Diagnosis pending (retrieval miss vs over-strict prompt).
2. Changing the embedding model changed generation outcomes (the CUDA question), so evaluation must cover retrieval and answers together.

---

## Open questions

- [ ] Shared local memory: are the relevant pages in the top 10 for that query? (retrieval vs prompt)
- [x] How many chunks exceed the embedding model's input limit? 42% with MiniLM; fixed by switching to bge-small (see 2026-11-18).
- [ ] Does BM25 or hybrid retrieval fix exact-phrase queries like "shared local memory"?

---

## Design decisions

| Decision | Reason |
|---|---|
| Public documentation only; PDFs not committed | Redistribution terms; sources listed in `data/sources.md` |
| Chunk within pages | Every chunk keeps its page number for citations |
| bge-small-en-v1.5 for embeddings | 512-token limit fits 1000-character chunks; MiniLM truncated 42% of them |
| Noise filter before embedding | Title and contents pages were taking top-k slots (see 2026-11-17) |
| Full index rebuild on each ingestion | `add` skips existing IDs, so tuned chunking would leave stale chunks |
| The model cites `[n]`; code maps to pages | The model never writes page numbers, so it can't invent them |
| Exact refusal sentence | Refusals can be checked with code, not a judge |
| Generation errors are not swallowed | The answer is the product; the API layer will return a proper error |