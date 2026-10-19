# RAG Retrieval: Baseline Dense Retrieval vs. Cross-Encoder Re-Ranking

**Course:** LMA · **Author:** Ameya Purohit · **Date:** October 2026
**Code:** `rag_reranking.ipynb` (this repository) · **Data and results:** `results/`

> *Template: the section structure and measured tables come from the committed run. The prose under each heading is to be written by the author. Facts and supporting notes are in `report_notes.md`.*

---

## 1. Introduction and Experimental Setup

*[Write 3–5 sentences: the aim of the comparison; the chosen document and why it is a hard retrieval target; how it was chunked; which models were used.]*

| Component | Configuration |
| --- | --- |
| Document | NIST SP 800-63B (2017, updated 2020), PDF pages 12–80 |
| Corpus | 254 chunks, ≤ 200 tokens each, 40-token overlap within a section |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2`, FAISS exact inner product on normalised vectors |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2`, applied to the same Top 10 |
| LLM | `gemini-3.1-flash-lite`, temperature 0, identical prompt, Top 3 chunks |
| Evaluation | 16 document-grounded queries in five failure categories |

## 2. Architecture Comparison: Bi-Encoder vs. Cross-Encoder

*[Write one paragraph: independent encoding and cosine similarity vs. joint self-attention over the concatenated query and passage; what can be precomputed in each case; the resulting accuracy/cost trade-off. Optionally refer to the Q05 score spread (cosine 0.489–0.559 vs. CE −0.81 to 6.49).]*

## 3. Qualitative and Rank-Shift Analysis

### 3.1 Overall Outcome

| Measure | Result |
| --- | --- |
| Critical chunk in baseline Top 3 | 9 / 16 |
| Critical chunk in re-ranked Top 3 | 14 / 16 |
| Critical chunk in dense Top 10 (upper bound for re-ranking) | 15 / 16 |
| Outcomes | 8 both correct · 6 rescued · 1 degraded · 1 missed by both |

*[Write 2–3 sentences interpreting these counts, including the caveat about the small query set.]*

### 3.2 Failure Cases Restored by Re-Ranking

| Query | Category | Critical chunk | Rank shift | Answer 1 (baseline) | Answer 2 (re-ranked) |
| --- | --- | --- | --- | --- | --- |
| Q01 Are password hints allowed? | Keyword trap | C046, §5.1.1.2 | #6 → #3 | No answer from context | Correct (SHALL NOT permit hints) |
| Q02 How often should users be forced to change passwords? | Keyword trap | C198, §10.2.1 | #6 → #1 | No answer from context | Correct (only on evidence of compromise) |
| Q05 How long is an out-of-band secret valid? | Exception | C066, §5.1.3.2 | #4 → #1 | No answer from context | Correct (10 minutes) |
| Q08 Which authenticators are permitted at AAL3? | Exception / level limit | C028, §4.3.1 | #10 → #2 | Incomplete (no list) | Lists permitted combinations |
| Q15 Can push notifications be used for out-of-band authentication? | Buried detail | C064, §5.1.3.2 | #5 → #1 | No answer from context | Correct (MAY send push notification) |
| Q04 Can SMS be used to deliver one-time codes? | Keyword trap | C061, §5.1.3.1 | #8 → #2 | No answer from context | Partially correct (omits RESTRICTED status) |

*[For two to four of these cases, write a short paragraph each: why the baseline ranked the critical chunk low (e.g. vocabulary mismatch "password" vs. "memorized secret"), what the cross-encoder changed, and how Answer 1 and Answer 2 differ. State whether Answer 1 hallucinated, omitted a clause, or misstated a rule.]*

### 3.3 Negative and Unresolved Cases

| Query | Observation |
| --- | --- |
| Q13 Can a password alone satisfy AAL2? | Re-ranking demoted the critical chunk (#2 → #8); the answer stayed correct because the rule is repeated in C005/C017 |
| Q03 What is the minimum password length? | Critical chunk at dense rank 11, outside the Top 10; both pipelines failed to answer |
| Q04 (restriction clause) | "PSTN … is RESTRICTED" (C068) at dense rank 31; never reachable by the re-ranker |

*[Write one paragraph: what these cases show about the limits of re-ranking (recall is bounded by stage 1; the cross-encoder can also err, e.g. confusing assurance levels).]*

## 4. Latency Profiling

Measured on CPU (12 logical cores) with models loaded and warmed up; 16 queries × 20 repetitions.

| Stage | Median (ms) | p95 (ms) |
| --- | --- | --- |
| Pipeline 1 retrieval (query embedding + FAISS Top 10 + Top 3) | 8.2 | 11.2 |
| Re-ranking step (cross-encoder on 10 pairs + sort) | 183.3 | 250.5 |
| Pipeline 2 retrieval (dense + re-ranking) | 191.6 | 261.6 |
| LLM generation, Answer 1 (n = 16) | 1648.7 | 2670.0 |
| LLM generation, Answer 2 (n = 16) | 1433.8 | 1948.8 |

*[Write 2–3 sentences: the relative cost of re-ranking (~23× slower retrieval) against its absolute cost compared with LLM generation.]*

## 5. Scaling: Why Not Apply the Cross-Encoder to the Whole Collection?

| Setting | Per-query cost |
| --- | --- |
| Cross-encoder on all 254 chunks (measured) | 4.78 s |
| Cross-encoder on 10 candidates (measured) | 0.18 s |
| Cross-encoder on 1,000,000 chunks (linear extrapolation) | ≈ 18,800 s (≈ 5.2 h) |

*[Write one paragraph: the per-pair, non-cacheable cost of cross-encoding vs. precomputed bi-encoder vectors and sub-linear vector search; why two-stage retrieval is the practical compromise, and what it costs (recall bounded by stage 1).]*

## 6. Conclusion and Limitations

*[Write 3–5 sentences summarising the findings. Limitations to mention: 16 author-written queries on a single document; key-phrase relevance labels; small MiniLM models; chunk-boundary effects (Q08); results describe the 2017/2020 edition of the standard.]*
