# Evaluation

How well does GPU DevAssist find the right documentation, and are its answers grounded in it? This document covers the evaluation set, the metrics, the results for three retrieval strategies, answer-level results, and an analysis of every failure.

**Summary.** Hybrid retrieval (dense + BM25 with reciprocal rank fusion) found a relevant page in the top 5 for 29 of 33 answerable questions (Hit@5 0.88, vs 0.70 for dense alone). On answers, the system refused all 7 unanswerable questions, answered 31 of 33 answerable ones, and its answers were 99% supported by the retrieved text. Every answer-level failure traces back to a retrieval miss.

---

## 1. Evaluation set

[`data/eval/questions.jsonl`](../data/eval/questions.jsonl): 40 questions over the two source documents.

| Type | Count | What it tests |
|---|---|---|
| Exact term | 10 | API names, tools and attributes (`sycl-ls`, `intel::reqd_sub_group_size`, `local_mem_size`) |
| Conceptual | 7 | Explanations of concepts |
| Reasoning | 7 | "Why" questions that need an explanation, not a definition |
| Paraphrased | 8 | Phrased the way a developer would ask, avoiding the documents' wording |
| Domain term | 1 | Terms easily confused (shared local memory vs Unified Shared Memory) |
| Unanswerable | 7 | Not covered by the documents, including in-domain near-misses |

**How the labels were made**
- **Gold pages are verified against the source PDFs.** Each answerable question carries a quoted excerpt, and a script confirmed the excerpt appears on the labeled page.
- **Additional relevant pages** were found by searching both PDFs for each question's key terms and reading the surrounding text. A page was added only if it directly answers the question (e.g. code that merely uses an attribute was not counted). 11 questions have more than one relevant page; one spans both documents.
- **Unanswerable questions are verified absent** by searching both PDFs for their key terms. This mattered: two questions first assumed to be unanswerable (including "What is NVIDIA's CUDA warp size?") turned out to be answered in the documents. Three unanswerable questions are deliberate near-misses that tempt a model to answer from general knowledge: converting HIP kernels with hipify, CUDA stream priorities, and NCCL all-reduce. The documents cover neighboring topics (CUDA migration, multiple queues, multi-GPU) but not these.

---

## 2. Metrics

**Retrieval.** Each retriever returns its top 10 chunks; chunks are reduced to an ordered, de-duplicated list of (document, page).

| Metric | Definition |
|---|---|
| **Hit@5** | 1 if any gold page is in the top 5, else 0; averaged over questions |
| **Recall@5** | Gold pages found in the top 5 ÷ all gold pages for the question |
| **MRR@10** | 1 ÷ rank of the first gold page within the top 10 (0 if none) |

**Answers** (hybrid retriever, top 6 chunks sent to the LLM)

| Metric | How it's computed |
|---|---|
| **Answer / refusal accuracy** | Code: is the response the exact refusal sentence? |
| **Citation hit** | Code: does at least one cited page match a gold page? |
| **Faithfulness** | LLM judge: the answer is split into claims, each checked against the retrieved chunks; supported ÷ total |

Retrieval metrics are deterministic. Answer metrics vary between runs and come from a single run.

---

## 3. Retrieval results

| Retriever | Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|
| Dense (`bge-small-en-v1.5`) | 0.697 | 0.626 | 0.551 |
| BM25 | 0.818 | 0.705 | **0.712** |
| **Hybrid (RRF, k = 60)** | **0.879** | **0.783** | 0.691 |

**Hit@5 by question type**

| Type (n) | Dense | BM25 | Hybrid |
|---|---|---|---|
| Conceptual (7) | **1.00** | 0.86 | 0.86 |
| Exact term (10) | 0.60 | 0.80 | **1.00** |
| Reasoning (7) | 0.71 | 0.86 | 0.86 |
| Paraphrased (8) | 0.62 | 0.75 | 0.75 |
| Domain term (1) | 0.00 | 1.00 | 1.00 |

With 7–10 questions per type, one question moves a score by 10–14 points; per-type figures are indicative only.

**Observations**
1. **Hybrid finds the most** (29 / 33). **BM25 ranks highest when it finds** (best MRR); fusion sometimes places a dense-only result above BM25's top hit.
2. **Dense is strongest on conceptual questions and weakest on exact identifiers.** It missed `warp` (only one page in either document contains the word), `sycl-ls` and `local_mem_size`. Hybrid found a relevant page for all 10 exact-term questions.
3. **Dense confused domain terms.** For shared local memory, its top 10 were almost all Unified Shared Memory pages (`malloc_shared`); BM25 placed 6 shared-local-memory pages in its top 10.
4. **Paraphrased questions are hardest for every method** (best 0.75), as intended.

### Hybrid's four misses

| Question | What happened |
|---|---|
| q024: GPU prefetching | BM25 missed entirely: the question says "prefetching", the pages say "prefetch", and the tokenizer does no stemming. Hybrid found it at rank 7. **Fix: stemming.** |
| q036: implicit scaling (paraphrased) | All three methods retrieved pages from the right chapter (297–319) but not p. 327. Those pages were checked and don't state the answer: a genuine page-level miss, not a missing label. |
| q034: bank conflicts (paraphrased) | The paraphrase avoids the term "bank conflict" by design; no method connected it to p. 66. |
| q021: SIMD32 → SIMD16 SLM bug | All three methods missed p. 65. |

---

## 4. Answer results

| Metric | Result |
|---|---|
| Answered when answerable | 31 / 33 |
| Refused when unanswerable | **7 / 7** |
| Citation hit (answered questions) | 0.94 (29 / 31) |
| Mean faithfulness | 0.99 |

**Every answer-level failure traces back to retrieval**
- **False refusals (q021, q034)** are two of hybrid's four retrieval misses. With no relevant chunk in context, the model refused rather than guessed: the intended behavior.
- **Citation misses (q024, q036)** are the other two. The model answered from the neighboring pages it was given, and the judge found those answers supported by them.
- **No fabricated answers:** all 7 unanswerable questions were refused, including the three in-domain near-misses.
- **Faithfulness below 1.0** occurred twice, one claim each: correct advice drawn from beyond the given context (e.g. "create the queue once outside the loop", stated on p. 109, which wasn't retrieved).

**Not captured by these metrics: completeness.** The shared local memory question was answered, correctly cited and fully faithful, but narrow: it covered histograms and reductions and noted the sources gave no general rule. The page with the general explanation (p. 65) wasn't in the top 6. Citation hit and faithfulness check that an answer is supported, not that it is complete.

**Conclusion.** The generation layer is reliable: it refuses when evidence is missing and stays grounded when it isn't. Improvements should target retrieval.

---

## 5. Experiments that shaped the system

| Change | Measurement |
|---|---|
| Noise filter for title and table-of-contents chunks | A relevant page moved from rank 7 to rank 5; title and contents pages left the top 5 |
| `all-MiniLM-L6-v2` → `bge-small-en-v1.5` | Chunks over the embedding model's input limit: 42% → 0% |
| Dense → hybrid retrieval | Hit@5 0.70 → 0.88 |

Details and the reasoning behind each: [experiment_log.md](experiment_log.md).

---

## 6. Limitations

- **Small set:** 40 questions; one or two questions per category can change conclusions.
- **Label completeness:** gold pages were found by keyword search and manual review. A relevant page with entirely different wording may be unlabeled, which affects Recall@5 most and can count a correct citation as a citation miss.
- **Judge bias:** the faithfulness judge is from the same model family as the generator.
- **Run-to-run variation:** answer metrics come from one run.
- **Completeness and correctness** against reference answers are not yet scored.

## 7. Reproduce

```bash
uv run gpu-dev-assist ingest
uv run gpu-dev-assist eval retrieval
uv run gpu-dev-assist eval generation
```

Per-question results are written to `data/eval/results/` (not committed).
