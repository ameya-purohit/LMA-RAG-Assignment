# Report notes: Baseline vs. Re-Ranking (NIST SP 800-63B)

> **These are working notes, not the report.** The assignment requires the report to be written in your own words. Below are the measured facts, tables and an outline to write from. Every number comes from the committed run (`results/`, `rag_reranking.ipynb`). Interpretations are marked **[interpretation]** so you can check, agree or disagree before using them.

## 0. Setup facts (for a short "Setup" paragraph)

| Item | Value |
| --- | --- |
| Document | NIST SP 800-63B (2017, updated 2020), PDF pages 12–80 indexed (front matter skipped) |
| Chunking | sentence-packed, ≤200 bi-encoder tokens, ≤40-token overlap within a section, new chunk at every heading; **254 chunks** (median 167 tokens) |
| Bi-encoder | `all-MiniLM-L6-v2` (384-d), normalised vectors, FAISS `IndexFlatIP` (exact cosine) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2`, scores all 10 candidates, sorted descending |
| LLM | `gemini-3.1-flash-lite`, temperature 0, same prompt template, top-3 chunks in each pipeline's rank order |
| Queries | 16 hand-written queries in 5 categories; relevance = chunk contains a key phrase from the answer sentence |
| Hardware | CPU only (12 logical cores, torch uses 6 threads), no GPU |

Why `flash-lite` and not `gemini-3.8-flash`: the free-tier key allows only about 20 requests per day for 3.8-flash, and one run needs 32 calls.

## 1. Architecture comparison: points to explain

- **Bi-encoder:** query and chunk are encoded **separately** into one vector each, and relevance is the cosine between the two vectors. Chunk vectors are computed **once, offline**, so at query time only one forward pass (the query) plus a vector search is needed.
  - Consequence: the whole chunk is compressed into 384 numbers *before* the question is known. Word-level interactions between query and chunk ("hint" ↔ "SHALL NOT permit … store a hint") can only be captured indirectly.
- **Cross-encoder:** the query and chunk are concatenated (`[CLS] q [SEP] chunk [SEP]`) and passed through the transformer **together**, so self-attention links every query token to every chunk token. The output is one relevance score per pair. Nothing can be precomputed, so there is one forward pass per (query, chunk) pair.
- Trade-off in one line: bi-encoder = cheap and scalable but coarse; cross-encoder = expensive per pair but more precise. Hence retrieve-then-rerank.
- Possible evidence from this run: in Q05 the cosine scores of the 10 candidates are packed into **0.489–0.559**, while the CE scores spread from **−0.81 to 6.49**, with a clear gap between the critical chunk (6.49) and the next one (1.96). **[interpretation]** The cosine scores barely separate the candidates; the joint model does.

## 2. Rank-shift analysis

### 2.1 Overall (16 queries)

| | Count |
| --- | --- |
| Critical chunk in the **baseline** top 3 | **9/16** |
| Critical chunk in the **re-ranked** top 3 | **14/16** |
| Critical chunk in the dense top 10 at all (best any reranker could do) | 15/16 |
| Outcomes | 8 both hit · **6 rescued by re-ranking** · **1 made worse by re-ranking** · 1 missed by both |

Small, hand-made query set → these are descriptive counts, not a statistically meaningful improvement rate.

### 2.2 Failure cases where re-ranking restored relevance (meets the "at least two" requirement)

| Query (category) | Critical chunk (section, PDF page) | Dense rank → CE rank | CE score (next best) | Answer 1 (baseline) | Answer 2 (re-ranked) |
| --- | --- | --- | --- | --- | --- |
| Q01 "Are password hints allowed?" (keyword trap) | C046 (§5.1.1.2, p.24) | **#6 → #3** | −3.91 (#1 was −2.98) | "The provided context does not answer this question." | Correct: verifiers SHALL NOT permit storing a "hint" accessible to an unauthenticated claimant [C046] |
| Q02 "How often should users be forced to change their passwords?" (keyword trap) | C198 (§10.2.1, p.64) | **#6 → #1** | −0.50 (−2.33) | "does not answer" | Correct: not periodically; only on user request or evidence of compromise [C198] |
| Q04 "Can SMS text messages be used to deliver one-time codes?" (keyword trap) | C061 (§5.1.3.1, p.28) | **#8 → #2** | −9.78 (−9.32) | "does not answer" | Says *yes* via SIM-authenticated OOB [C061], but **omits that PSTN/SMS is RESTRICTED** (§5.1.3.3, C068, dense rank 31) — see 2.4 |
| Q05 "How long is an out-of-band authentication secret valid?" (exception / specific limit) | C066 (§5.1.3.2, p.29) | **#4 → #1** | 6.49 (1.96) | "does not answer" | Correct: invalid if not completed within **10 minutes**, accepted only once [C066] |
| Q08 "Which authenticators are permitted at AAL3?" (exception / level-specific) | C028 (§4.3.1, p.20) | **#10 → #2** | 7.58 (8.24 for C018, an *AAL2* chunk) | General AAL3 requirements (hardware-based, FIPS 140 levels) but **no list of permitted authenticator combinations** | Lists the permitted combinations [C028] (5 of the 6: the 6th is in C029, cut off by the chunk boundary) |
| Q15 "Can push notifications on a smartphone be used for out-of-band authentication?" (buried detail) | C064 (§5.1.3.2, p.29) | **#5 → #1** | 6.28 (2.91) | "does not answer" | Correct: verifier MAY send a push notification [C064] |

What the baseline failures looked like (factual): in all six cases the baseline LLM **did not hallucinate**. In 5/6 it said the context does not answer, and in Q08 it gave true but incomplete information. So the cost of the missing chunk was **omission / non-answer**, not invented rules. **[interpretation]** That is partly due to the strict "use only the context" prompt; a less constrained prompt might have filled the gap from the model's prior knowledge.

Full top-10 tables for each case (all 10 dense ranks, cosine, CE score, new rank) are printed in notebook §8.2 and stored in `results/rankings_top10.csv`.

Example detail for Q01 (to quote as a rank-shift example): dense top 3 = C198 (§10.2.1 usability), C243 (Appendix A heading), C244 (A.1 Introduction). All three use the word "password" heavily, and none states the rule. The rule chunk C046 says "memorized secret", not "password". **[interpretation]** This is the keyword-trap pattern: lexical/topic overlap with informative sections beats the normative sentence that uses different terminology.

### 2.3 Case where re-ranking made the ranking worse (report this honestly)

- **Q13 "Can a password alone satisfy AAL2?"**: C018 (§4.2.1, "At AAL2, authentication SHALL occur by the use of either a multi-factor authenticator or a combination of two single-factor authenticators") moved **#2 → #8** (CE −2.18). The CE put C022 (§4.2.2, about phone-unlock PINs, CE 0.52) at #1.
- Answer impact: **none.** Both answers say "No", because C005/C017 (still in the top 3) repeat that AAL2 needs two factors.
- **[interpretation]** The cross-encoder may be matching "PIN/password" + "AAL2" on the surface. Redundancy in the document hid the ranking error.
- A similar level confusion shows up in Q08: the CE's #1 there is the **AAL2** permitted-types chunk C018 (8.24), above the AAL3 one.

### 2.4 Cases re-ranking could not fix (limitations)

- **Q03 "What is the minimum password length?"**: the normative rule (C043, "at least 8 characters … if chosen by the subscriber") is at **dense rank 11**, so it is outside the top 10 and the CE never sees it. Both answers: "does not answer". The dense top 10 is dominated by Appendix A ("A.2 Length", "password length").
- **Q04 (SMS)**: re-ranking surfaced a relevant chunk, but the key **restriction** ("Use of the PSTN for out-of-band verification is RESTRICTED", C068) is at **dense rank 31**. As a result Answer 2 reads like an unqualified "yes": an **omitted essential clause**, caused by first-stage recall rather than ranking.
- **[interpretation]** Re-ranking improves precision within the candidate set; it cannot fix recall. Possible remedies (not implemented): larger k, hybrid BM25 + dense, query expansion (password → memorized secret, SMS → PSTN).

### 2.5 Both pipelines fine (8 queries)

Q06, Q07, Q09–Q12, Q14, Q16. Several are negation queries ("SHALL NOT…", Q09–Q12) where the critical chunk was already at dense rank 1–2. **[interpretation]** For this document, the negation queries were not where the bi-encoder failed; the bigger problem was vocabulary mismatch. Re-ranking often still moved the critical chunk up within the top 3 (Q07, Q11, Q12, Q14: #2 → #1), which changes passage order but not the answers.

## 3. Latency profiling (CPU, models loaded, warmed up; 16 queries × 20 repeats = 320 measurements)

| Stage | Median ms | p95 ms |
| --- | --- | --- |
| Pipeline 1 retrieval: query embedding + FAISS top-10 + take top-3 | **8.2** | 11.2 |
| Re-ranking step only: cross-encoder on 10 pairs + sort | 183.3 | 250.5 |
| Pipeline 2 retrieval: dense + re-ranking | **191.6** | 261.6 |
| LLM generation, Answer 1 (Gemini API, n=16) | 1648.7 | 2670.0 |
| LLM generation, Answer 2 (Gemini API, n=16) | 1433.8 | 1948.8 |

- Re-ranking makes retrieval about **23× slower** (8 → 192 ms), but that adds only ~0.18 s to a request whose LLM call takes ~1.4–1.7 s.
- Summing medians, end-to-end is ≈ 0.008 + 1.65 ≈ 1.66 s (baseline) vs ≈ 0.19 + 1.43 ≈ 1.63 s (re-rank). The re-rank pipeline comes out *slightly faster* only because of LLM/API variance. The ~240 ms difference between the two LLM medians is larger than the whole re-ranking cost. **[interpretation]** In this setup the LLM call dominates latency, and re-ranking overhead is within its noise.
- The two LLM rows differ because of API/network variance, not the pipeline (same model, same prompt template).
- Measurement boundaries are defined in notebook §9. Model loading and corpus embedding (6.4 s, one-off) are excluded.

## 4. Scaling question: measured numbers to support your explanation

| | Value |
| --- | --- |
| CE over **all 254 chunks** for one query | **4.78 s** (18.8 ms/pair, batch 32) |
| CE over 10 candidates | 0.18 s |
| Linear extrapolation, 10,000 chunks | ~188 s per query |
| Linear extrapolation, 1,000,000 chunks | ~18,800 s ≈ **5.2 hours per query** |
| Bi-encoder: one-off corpus embedding at 1M chunks (extrapolated) | ~7 h **once**, offline; query time stays ~ms with an (approximate) vector index |

Points to explain in your own words:

- CE cost is **per query × per document**, nothing can be cached because the score depends on the pair, and it grows linearly with collection size.
- Bi-encoder document vectors are computed once. Search over them is a matrix product / ANN lookup, sub-linear with indexes like HNSW/IVF.
- So: use the cheap model to narrow millions of chunks down to k ≈ 10–100, then the expensive model only on those. The cost is that recall is capped by stage 1 (see Q03/Q04).

## 5. Limitations to state

- 16 queries on one document, written by me after reading it and refined while exploring retrieval. They are targeted stress tests, not a representative sample.
- Relevance = key-phrase containment chosen by me. The `answer_check` regex is only a screen; the answers above were read manually.
- Small MiniLM encoders. Gemini output can vary between uncached runs. Answers are cached (`results/llm_cache.json`) so the notebook reproduces this run.
- Chunking artifacts: a list split across chunks (Q08: C028/C029).
- The source edition was withdrawn in 2025 (superseded by SP 800-63B-4).

## 6. Suggested report outline (keep it concise)

1. **Setup** (3–4 sentences): document and why it is hard, chunking, models.
2. **Architecture comparison** (1 short paragraph, or a 2-column table in your words).
3. **Rank shift and answer comparison**: the table in 2.2 (pick 2–4 cases), one sentence each on what Answer 1 got wrong. Then the honest negatives: Q13 (worse ranking, same answer) and Q03/Q04 (recall limit).
4. **Latency**: the table in §3 plus one or two sentences.
5. **Scaling**: 1 paragraph using the §4 numbers.
6. (Optional) **Limitations**: 2–3 bullets.

## 7. Check these yourself before writing (so you can defend them)

- [ ] Open notebook §8.2 and read the top-10 table for at least Q01, Q05 and Q08. Can you explain why each critical chunk was low in the dense ranking?
- [ ] Read C068 in `results/chunks.jsonl` and decide for yourself whether Answer 2 for Q04 is misleading.
- [ ] Why can't the cross-encoder fix Q03? (Answer: C043 is at dense rank 11; only the top 10 are re-ranked.)
- [ ] What would change with k = 20? (Untested here. Say so if you mention it.)
