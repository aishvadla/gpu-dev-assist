# Evaluation

How well does GPU DevAssist find the right documentation, and are its answers grounded in it? This document covers the evaluation set, the metrics, the results for three retrieval strategies, answer-level results, and an analysis of every failure.

**Summary.** Over 43 answerable questions, hybrid retrieval found a relevant page in the top 5 for 35 (Hit@5 0.81), stemmed BM25 for 34 (0.79) and dense alone for 28 (0.65). On questions that reuse the documents' vocabulary, BM25 and hybrid tie; on 10 questions worded the way developers actually ask, hybrid found 7 and BM25 5, a result predicted and recorded before the run. Hybrid is the default. On answers, the system refused all 7 unanswerable questions, answered 40 of 43 answerable ones, and 98% of answer claims were supported by the retrieved text. All 8 answer-level failures are exactly hybrid's 8 retrieval misses.

---

## 1. Evaluation set

[`data/eval/questions.jsonl`](../data/eval/questions.jsonl): 50 questions over the two source documents (43 answerable, 7 unanswerable).

| Type | Count | What it tests |
|---|---|---|
| Exact term | 10 | API names, tools and attributes (`sycl-ls`, `intel::reqd_sub_group_size`, `local_mem_size`) |
| Conceptual | 7 | Explanations of concepts |
| Reasoning | 7 | "Why" questions that need an explanation, not a definition |
| Paraphrased | 8 | Phrased the way a developer would ask, avoiding the documents' wording |
| Domain term | 1 | Terms easily confused (shared local memory vs Unified Shared Memory) |
| User style | 10 | Worded as a developer would type it, avoiding the documents' terms ("a delay before the first kernel runs" for JIT compilation). Added later to test a specific hypothesis (section 3) |
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
All 43 answerable questions:

| Retriever | Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|
| Dense (`bge-small-en-v1.5`) | 0.651 | 0.539 | 0.491 |
| BM25 (stemmed) | 0.791 | 0.651 | 0.584 |
| **Hybrid (RRF, k = 60), default** | **0.814** | **0.671** | **0.666** |

The original 33 questions alone: dense 0.697 / 0.626 / 0.551; BM25 **0.879** / **0.758** / 0.672; hybrid 0.848 / 0.753 / **0.723**.

BM25 uses Snowball stemming (added after the first evaluation; before/after in section 5).

**Hit@5 by question type**

| Type (n) | Dense | BM25 | Hybrid |
|---|---|---|---|
| Conceptual (7) | **1.00** | 0.86 | **1.00** |
| Exact term (10) | 0.60 | **0.90** | **0.90** |
| Reasoning (7) | 0.71 | **0.86** | **0.86** |
| Paraphrased (8) | 0.62 | **0.88** | 0.62 |
| Domain term (1) | 0.00 | 1.00 | 1.00 |
| **User style (10)** | 0.50 | 0.50 | **0.70** |

With 7–10 questions per type, one question moves a score by 10–14 points; per-type figures are indicative only.

**Observations**
1. **Dense alone is clearly weakest.** It missed exact identifiers (`warp`, which appears on one page in either document; `sycl-ls`; `local_mem_size`) and confused domain terms: for shared local memory, its top 10 were almost all Unified Shared Memory pages (`malloc_shared`), while BM25 placed 6 shared-local-memory pages in its top 10.
2. **On the original 33 questions, stemmed BM25 and hybrid tie.** BM25 finds one more question in the top 5 (29 vs 28); hybrid ranks its hits higher (MRR 0.72 vs 0.67). Recall@5 is the same. Those questions mostly reuse the documents' vocabulary, which favours BM25.
3. **Fusion can demote a correct BM25 result.** Reciprocal rank fusion rewards agreement between the retrievers. On two paraphrased questions (q034 bank conflicts, q037 EU active time), BM25 ranked the gold page 5th and 3rd, but dense didn't rank it at all, and hybrid pushed it to 7th or out of the top 10. This is why hybrid trails BM25 on paraphrased questions.
4. **On user-style questions, hybrid wins, as predicted.** To test whether hybrid helps on vaguer wording, 10 questions were added and the prediction ("hybrid beats BM25 on Hit@5") was recorded before running. Hybrid found 7, BM25 5, dense 5. The two retrievers fail on *different* questions: hybrid kept every question BM25 found in the top 5 and added the two that only dense found (q045, running on the CPU without code changes; q047, pausing a kernel to inspect threads). This is the case for fusion. Three questions were missed by every method (q044 kernel fusion, q046 matrix padding, q048 host USM allocation).
5. **Decision: hybrid is the default.** Best overall Hit@5, Recall@5 and MRR on all 43 questions, best on user-style questions, and best or tied on every type except paraphrased. Ten questions is a small sample, so the conclusion is "hybrid is more robust", not "hybrid is better by a fixed margin".

### Hybrid's misses on the original 33 questions

| Question | What happened |
|---|---|
| q007: `queue.submit` in the vector-add example | No method ranked p. 9 in its top 5 (dense placed it 9th). |
| q021: SIMD32 → SIMD16 SLM bug | All three methods missed p. 65. |
| q034: bank conflicts (paraphrased) | The paraphrase avoids the term "bank conflict" by design. Stemmed BM25 found p. 66 at rank 5; dense didn't, and fusion pushed it out. |
| q036: implicit scaling (paraphrased) | All three methods retrieved pages from the right chapter (297–319) but not p. 327. Those pages were checked and don't state the answer: a genuine page-level miss, not a missing label. |
| q037: EU active time (paraphrased) | BM25 ranked p. 395 3rd; fusion placed it 7th. |

---

## 4. Answer results

| Metric | Result |
|---|---|
| Answered when answerable | 40 / 43 |
| Refused when unanswerable | **7 / 7** |
| Citation hit (answered questions) | 0.88 (35 / 40) |
| Mean faithfulness | 0.98 |

**Every answer-level failure traces back to retrieval**
- **The 8 answer-level failures are exactly hybrid's 8 retrieval misses** (5 on the original questions, 3 on the user-style ones).
- **False refusals (q021, q034, q044):** with no relevant chunk in context, the model refused rather than guessed: the intended behavior.
- **Citation misses (q007, q036, q037, q046, q048):** the model answered from the neighboring pages it was given.
- **No fabricated answers:** all 7 unanswerable questions were refused, including the three in-domain near-misses.
- **Faithfulness below 1.0** occurred in 6 of 40 answers, one or two claims each, mostly reasonable inferences that go beyond the retrieved text. The lowest scores come from citation misses: answering from the wrong pages, the model fills gaps with general knowledge (q048, 0.67; q046, 0.86). Faithfulness also varies between runs: q037 scored 0.78 in one run and 0.90 in the next.

**Not captured by these metrics: completeness.** The shared local memory question was answered, correctly cited and fully faithful, but narrow: it covered histograms and reductions and noted the sources gave no general rule. The page with the general explanation (p. 65) wasn't in the top 6. Citation hit and faithfulness check that an answer is supported, not that it is complete.

**Conclusion.** The generation layer is reliable: it refuses when evidence is missing and stays grounded when it isn't. Improvements should target retrieval.

---

## 5. Experiments that shaped the system

| Change | Measurement |
|---|---|
| Noise filter for title and table-of-contents chunks | A relevant page moved from rank 7 to rank 5; title and contents pages left the top 5 |
| `all-MiniLM-L6-v2` → `bge-small-en-v1.5` | Chunks over the embedding model's input limit: 42% → 0% |
| Dense → hybrid retrieval | Hit@5 0.70 → 0.88 |
| User-style questions (prediction recorded first) | Hybrid 0.70 vs BM25 0.50 vs dense 0.50 Hit@5; hybrid confirmed as default |
| Snowball stemming in BM25 | BM25 Hit@5 0.82 → 0.88 (fixed q009, q024 "prefetching" vs "prefetch", q035; broke q011). Hybrid 0.88 → 0.85 (fixed q024, which rose from rank 7 to 1; lost q007 and q037), MRR 0.69 → 0.72 |

Details and the reasoning behind each: [experiment_log.md](experiment_log.md).

---

## 6. Limitations

- **Small set:** 50 questions; one or two questions per category can change conclusions. The 10 user-style questions were written by the project author, who knows the system's weaknesses; they were written before any retriever was run on them.
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
