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
| Retrieval | Top 6; dense (cosine), BM25 (`rank_bm25`) and hybrid (reciprocal rank fusion) available |

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
| What is NVIDIA's CUDA warp size? | Partial answer with caveat | Answerable after all: p. 196 maps a SYCL sub-group to a CUDA warp of size 32 (found later via BM25). Here the model cited p. 60 and said the docs didn't confirm it. No fabrication. |

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

## 2026-11-18: Shared local memory false refusal diagnosed (retrieval miss)

**Diagnosis.** For "When should I use shared local memory in a SYCL kernel?", dense retrieval's top 10 were almost entirely about **Unified Shared Memory** (`malloc_shared`, shared allocations; pages 128, 142, 143, 145). The pages that actually discuss shared local memory / SLM (9, 16, 40, 57, 64–72, 98) were absent. The model refused correctly: it was given six chunks about a different topic.

**Cause.** "Shared local memory" (work-group scratchpad, `local_accessor`) and "Unified Shared Memory" (pointer-based host/device allocations) are unrelated concepts that share a word. A general-purpose embedding model treats them as similar; a GPU engineer doesn't.

**BM25 check (same chunks).**

| | Dense (bge-small) | BM25 |
|---|---|---|
| SLM pages in top 10 | 0 | 6 (98, 16, 72, 57, 280, 64) |
| SLM pages in top 5 | 0 | 3 (98, 16, 72) |
| Other results | USM pages | Code chunks (64, 72) and partial matches (327, 395, 151) |

**Takeaway.** Dense retrieval understands meaning but confuses similar-sounding domain terms; BM25 matches exact terms but can't tell explanation from code or partial matches. Next: hybrid retrieval.

**Debugging note.** The first diagnosis run returned nothing: a notebook resolved the relative path `data/vector_store` from the wrong folder and silently created a new, empty store. Fix: always check `collection.count()` (1,585) before trusting results; default store path to be built from the project root.

---

## 2026-11-18: Dense vs BM25 vs hybrid (RRF), first comparison

**Setup.** Same 1,585 chunks. BM25 via `rank_bm25` with a simple lowercase tokenizer. Hybrid = reciprocal rank fusion (k = 60, 20 candidates from each method). Top 6 sent to the LLM. All three retrievers share one interface (`query_retrieve`), so the generator is unchanged.

| Question | Dense | BM25 | Hybrid |
|---|---|---|---|
| What is the oneAPI programming model? | Answered (p. 7, 6, 5, 4, 46) | Answered, some noise (p. 184, 277) | Answered; lost p. 7 |
| When should I use shared local memory in a SYCL kernel? | False refusal (USM pages) | Answered (p. 98, 16, 72, 57) | Answered (p. 57, 98) |
| What is NVIDIA's CUDA warp size? | False refusal | Answered: 32 (p. 196) | Answered: 32 (p. 196) |
| How does multi-head attention work? (not in the docs) | Refused | Refused | Refused |
| **Correct behavior** | **2 / 4** | **4 / 4** | **4 / 4** |

**Findings.**
1. Hybrid fixed both dense failures, and no retriever produced an answer for the out-of-scope question: grounding held even when retrieval returned unrelated pages.
2. The CUDA warp size question is answerable: p. 196 maps a SYCL sub-group to a CUDA warp of size 32. Dense missed it; BM25 matched the rare term "warp". Second time an assumed-unanswerable question had an answer in the corpus.
3. Hybrid's SLM top 6 still contained Unified Shared Memory pages (142, 142, 128) inherited from dense. It answered correctly because enough good chunks got in, not because ranking was clean. Candidate fix: reranking.
4. These questions all contain distinctive terms, which favors BM25. No conclusion until the evaluation set, including paraphrased questions.
5. **Answers vary between runs.** With identical retrieved pages, the wording and cited sources changed between two runs. Retrieval metrics are deterministic; answer-level scores are noisy and should be reported as such (or averaged over repeated runs).

---

## 2026-11-19: Retrieval evaluation on 33 answerable questions

**Setup.** 40-question evaluation set (`data/eval/questions.jsonl`): 33 answerable with verified gold pages, 7 unanswerable (used later for refusal accuracy). Each retriever returns its top 10 chunks; pages are de-duplicated in rank order. Metrics: Hit@5, Recall@5, MRR@10.

| Retriever | Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|
| Dense (bge-small) | 0.697 | 0.626 | 0.551 |
| BM25 | 0.818 | 0.705 | **0.712** |
| **Hybrid (RRF)** | **0.879** | **0.783** | 0.691 |

**Hit@5 by question type** (n per type in brackets; one question = 10–14 points for most types):

| Type | Dense | BM25 | Hybrid |
|---|---|---|---|
| Conceptual (7) | **1.00** | 0.86 | 0.86 |
| Domain term (1) | 0.00 | 1.00 | 1.00 |
| Exact term (10) | 0.60 | 0.80 | **1.00** |
| Paraphrased (8) | 0.62 | 0.75 | 0.75 |
| Reasoning (7) | 0.71 | 0.86 | 0.86 |

**Findings.**
1. Hybrid finds a relevant page for the most questions (29 of 33). BM25 has the best MRR: when it finds the page, it ranks it higher; fusion sometimes pulls a dense-only result above it.
2. Dense is best on conceptual questions (7/7) and weakest on exact identifiers: it missed `warp` (q003), `sycl-ls` (q010) and `local_mem_size` (q022), and the SLM/USM confusion (q002).
3. Hybrid fixed every exact-term miss (10/10).
4. Paraphrased questions are the hardest for all methods (best 0.75), as intended.

**Hybrid's 4 misses.**
- **q024 (prefetching):** BM25 missed entirely: the question says "prefetching", the pages say "prefetch", and the tokenizer does no stemming. Hybrid found it at rank 7. Candidate fix: stemming in BM25.
- **q036 (implicit scaling, paraphrased):** all three retrieved pages from the right chapter (297–319) but not p. 327. Checked: those pages don't state the answer, so this is a page-level miss, not a missing label.
- **q034 (bank conflicts, paraphrased):** the paraphrase avoids the term "bank conflict" by design; no method connected it to p. 66.
- **q021 (SIMD32 → SIMD16 SLM bug):** all three missed p. 65.

**Caveats.** Small sample: per-type differences of one question are not meaningful. Gold labels found by keyword search and manual review; a relevant page with entirely different wording could be unlabeled, which affects Recall@5 most.

---

## 2026-11-19: Generation evaluation (hybrid retriever, 40 questions)

**Setup.** Hybrid retriever, top 6 chunks, Claude Haiku with the citation prompt. Refusals checked with code (exact refusal sentence). Citation hit: at least one cited page is a gold page. Faithfulness: an LLM judge splits the answer into claims and checks each against the retrieved chunks.

| Metric | Result |
|---|---|
| Answered when answerable | 31 / 33 |
| Refused when unanswerable | **7 / 7** (no fabricated answers, including the 3 in-domain near-misses) |
| Citation hit (answered questions) | 0.94 (29 / 31) |
| Mean faithfulness | 0.99 |

**Every generation failure traces back to retrieval.**
- **False refusals, q021 and q034:** the same questions hybrid retrieval missed. With no relevant chunk in context, the model refused instead of guessing: the intended behavior.
- **Citation misses, q024 and q036:** also hybrid retrieval misses (q024's gold page was at rank 7). The model answered from neighboring pages it was given, and the judge found those answers supported by them.
- **Faithfulness below 1.0, q013 (0.88) and q020 (0.92):** one claim each was not found in the retrieved text (e.g. "create the queue once outside the loop", which is stated on p. 109, a page that wasn't retrieved). Minor: correct advice, but drawn from beyond the given context.

**Not captured by the metrics: completeness.** q002 (shared local memory) was answered, cited correctly and fully faithful, but narrow: it covered histograms and reductions and said the sources give no general rule. The chunk with the general explanation (p. 65) wasn't in the top 6. Citation hit and faithfulness check that the answer is supported, not that it is complete.

**Takeaway.** The generation layer is reliable: it refuses when evidence is missing and stays grounded. The bottleneck is retrieval, so further work should go there (BM25 stemming, reranking).

**Limitations.** The judge is from the same model family as the generator (possible bias). Answers vary between runs, so generation metrics are from a single run. Citation hit counts a correct citation to an unlabeled page as a miss.

---

## 2026-11-19: Snowball stemming in BM25

**Hypothesis.** q024 asks about "prefetching"; the gold pages say "prefetch". BM25 matched exact tokens only, so it missed the page, and hybrid found it only at rank 7. Stemming should fix it.

**Change.** `keyword_retrieve.tokenize` stems tokens with the Snowball English stemmer (`snowballstemmer`). Identifiers such as `local_mem_size` stay single tokens. Unit tests in `tests/test_keyword_retrieve.py`. The dense retriever is unaffected (its subword tokenizer is a different mechanism; the embedding model was already handling q024 at rank 3).

**Retrieval, before → after** (33 answerable questions)

| Retriever | Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|
| Dense | 0.697 → 0.697 | 0.626 → 0.626 | 0.551 → 0.551 |
| BM25 | 0.818 → **0.879** | 0.705 → **0.758** | 0.712 → 0.672 |
| Hybrid | 0.879 → 0.848 | 0.783 → 0.753 | 0.691 → **0.723** |

**Hit@5 by type after stemming** (dense / BM25 / hybrid): conceptual 1.00 / 0.86 / 1.00; exact term 0.60 / 0.90 / 0.90; reasoning 0.71 / 0.86 / 0.86; paraphrased 0.62 / 0.88 / 0.62; domain term 0 / 1 / 1.

**Per question**
- **Hypothesis confirmed:** q024 hybrid rank 7 → **1**; BM25 not in top 10 → rank 5.
- **BM25:** fixed q009, q024, q035; broke q011 (now rank 7). Net +2.
- **Hybrid:** fixed q024; newly missed q007 (`queue.submit`) and q037 (EU active time, rank 7). Net −1.
- **Fusion demotes BM25-only hits.** On q034 and q037, stemmed BM25 ranked the gold page 5th and 3rd, dense didn't rank it, and RRF (which rewards agreement) pushed it to 7th or out of the top 10.

**Generation rerun** (hybrid, after stemming): answered 31/33, refused 7/7, citation hit 0.90 (28/31; was 0.94), faithfulness 0.97 (was 0.99). False refusals q021, q034; citation misses q007, q036, q037: all hybrid retrieval misses, as before. Six answers had one or two unsupported claims, mostly inferences beyond the retrieved text; the lowest (q037, 0.78) answered from the wrong pages and filled the gap with general knowledge. Part of the faithfulness change is run-to-run variation.

**Decision.** Keep stemming: it is correct, fixed the diagnosed miss, and improved BM25 and hybrid MRR. Keep hybrid as the default, but don't claim it beats BM25: the difference is one question of 33, within noise. Hybrid is best or tied on 4 of 5 types and has the best MRR; real queries are vaguer than these, so the semantic signal is a safeguard. Next: a larger evaluation set, then weighted fusion favouring BM25.

---

## 2026-11-19: User-style questions (prediction recorded before running)

**Why.** After stemming, BM25 and hybrid tied (29 vs 28 of 33). The stated reason for keeping hybrid, that it handles vaguer real-world questions better, had not been tested. The existing questions mostly reuse the documents' vocabulary.

**Added.** q041–q050, type `user_style`: questions worded the way a developer would type them, avoiding the documents' terms (e.g. "a noticeable delay before the first kernel runs" for JIT compilation; "merging three small GPU steps" for kernel fusion). Gold pages verified against the PDFs with quoted excerpts. Written without running any retriever on them.

**Prediction (written before running).** Hybrid will beat stemmed BM25 on Hit@5 for the `user_style` questions. If BM25 matches or beats hybrid, the default switches to BM25.

**Result: prediction confirmed.**

| Retriever | user_style Hit@5 | All 43: Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| Dense | 0.50 | 0.651 | 0.539 | 0.491 |
| BM25 (stemmed) | 0.50 | 0.791 | 0.651 | 0.584 |
| Hybrid | **0.70** | **0.814** | **0.671** | **0.666** |

First relevant rank per question (dense / BM25 / hybrid; "-" = not in top 10):
q041 - / 3 / 5 · q042 5 / 2 / 1 · q043 1 / 2 / 1 · q044 - / - / - · q045 2 / - / 3 · q046 - / 9 / - · q047 1 / - / 1 · q048 - / - / - · q049 - / 2 / 4 · q050 4 / 1 / 1

- **Complementary failures.** Hybrid kept every question BM25 had in the top 5 and added q045 (CPU device selection) and q047 (GPU debugger), which only dense found. This is what fusion is for.
- **Missed by all:** q044 (kernel fusion), q046 (GEMM padding), q048 (host USM allocation). Candidates for a reranker or query rewriting.
- **Generation on all 50 questions:** answered 40/43, refused 7/7, citation hit 0.88 (35/40), faithfulness 0.98. False refusals q021, q034, q044; citation misses q007, q036, q037, q046, q048. All 8 failures are exactly hybrid's 8 retrieval misses. Lowest faithfulness: q048 (0.67), a citation miss where the model filled gaps with general knowledge.

**Decision.** Hybrid stays the default, now with evidence: best overall on all three metrics over 43 questions and best on user-style wording. Caveat: 10 questions, written by the author (before any retriever was run on them).

---

## Open questions

- [x] Shared local memory: retrieval miss, not the prompt. Dense retrieval returned Unified Shared Memory pages (see 2026-11-18).
- [x] How many chunks exceed the embedding model's input limit? 42% with MiniLM; fixed by switching to bge-small (see 2026-11-18).
- [x] Does BM25 or hybrid retrieval fix exact-phrase queries like "shared local memory"? Yes on the first 4 questions (hybrid 4/4 vs dense 2/4); to confirm on the evaluation set.
- [ ] Gold pages and reference answer for the SLM question (q002).
- [ ] Does a reranker remove the USM pages from hybrid's SLM results?
- [x] How do the three retrievers compare on paraphrased questions? After stemming: BM25 0.88, dense and hybrid 0.62 (see stemming entry).
- [ ] Does weighting BM25 higher in the fusion close the paraphrased gap without losing conceptual questions?
- [x] Does hybrid beat BM25 on vaguer, user-style questions? Yes: 0.70 vs 0.50 Hit@5 on 10 questions (prediction recorded before running).
- [ ] Do a reranker or query rewriting fix the questions all methods miss (q044, q046, q048)?

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
| Hybrid retrieval with reciprocal rank fusion | Dense confused domain terms (SLM vs USM) and missed rare terms ("warp"); RRF combines rankings without normalizing incompatible scores |
| Snowball stemming in BM25 | Fixed "prefetching" vs "prefetch" (q024); BM25 Hit@5 0.82 → 0.88 |
| Hybrid stays the default over stemmed BM25 | Tied on document-worded questions; on user-style questions hybrid 0.70 vs BM25 0.50 (prediction recorded first); best overall on all metrics |
| One retriever interface (`query_retrieve`) | Dense, BM25 and hybrid are interchangeable, so the generator and evaluation don't change |
| Retrievers passed in, not built inside (dependency injection) | One shared store and model; fusion testable with fake retrievers |
| "Unanswerable" evaluation questions verified by searching the PDFs | Two assumed-unanswerable questions turned out to be answerable |