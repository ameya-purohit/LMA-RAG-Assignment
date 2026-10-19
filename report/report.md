# RAG Retrieval: Baseline Dense Retrieval vs. Cross-Encoder Re-Ranking

**Course:** LMA · **Author:** Ameya Purohit · **Date:** 19 October 2026
**Code:** [`rag_reranking.ipynb`](../rag_reranking.ipynb) · **Raw results:** [`results/`](../results/)

---

## 1. Introduction and Experimental Setup

This report compares two retrieval workflows for retrieval-augmented generation (RAG) over the same document:

- **Pipeline 1** is a dense bi-encoder baseline.
- **Pipeline 2** re-ranks the same baseline candidates with a cross-encoder.

The aim is to find out when the re-ranking step changes which passages reach the LLM, and whether that changes the answer.

The document is **NIST SP 800-63B**, *Digital Identity Guidelines: Authentication and Lifecycle Management* (2017 edition, updated 2020). It is an 80-page normative standard and a hard retrieval target for three reasons:

- Its rules depend on precise modal verbs (SHALL, SHALL NOT, SHOULD, MAY).
- Requirements differ across three assurance levels (AAL1–AAL3).
- It uses formal terms where users would use everyday ones: *memorized secret* for "password", *PSTN* for "SMS", *biometric* for "fingerprint".

Informative sections, such as Appendix A on password strength, discuss the same topics as the normative rules and act as natural distractors.

| Component | Configuration |
| --- | --- |
| Corpus | PDF pages 12–80. Running headers removed; section titles taken from the PDF outline. 254 chunks of ≤ 200 tokens, sentence-aligned, up to 40 tokens of overlap within a section, every chunk tagged with an ID, section and page. |
| Bi-encoder | `all-MiniLM-L6-v2` (384-d). Normalised vectors in a FAISS exact inner-product index, so the score is cosine similarity. |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2`, applied to the same Top 10 candidates. |
| Selection | Baseline: dense ranks 1–3. Re-ranked: top 3 by cross-encoder score, sorted from highest to lowest. |
| LLM | `gemini-3.1-flash-lite`, temperature 0. The prompt is identical except for the three context chunks. The model must answer only from the context and cite chunk IDs. |
| Evaluation | 16 queries written from the document, covering keyword traps, exceptions, negations, level-specific limits and buried details. |

A chunk counts as **critical** for a query if it contains the sentence that answers it. This is checked automatically by matching a key phrase taken from that sentence. The notebook also asserts that both pipelines share an identical Top 10 list, that the baseline keeps the dense order, that all 10 candidates are re-scored and sorted correctly, and that both pipelines pass exactly three chunks to the LLM.

## 2. Architecture Comparison: Bi-Encoder vs. Cross-Encoder

A **bi-encoder** encodes the query and each passage *independently* into one fixed-size vector, and relevance is the cosine similarity between the two vectors. Passage vectors depend only on the passage, so they are computed once, offline, and stored in a vector index. At query time the system embeds only the query and runs a fast nearest-neighbour search. The cost is that each passage is compressed into a single vector *before* the question is known. Fine-grained interactions between specific query words and passage words, such as negation, an exception clause, or a synonym, are captured only indirectly.

A **cross-encoder** concatenates the query and the passage (`[CLS] query [SEP] passage [SEP]`) and processes them *jointly*. Self-attention connects every query token to every passage token in every layer, and the model outputs one relevance score for the pair. This makes it more precise at judging whether a passage actually answers the question. However, the score depends on the pair, so nothing can be precomputed, and every candidate passage needs a full forward pass at query time.

The difference was visible in the experiment. For Q05, the 10 candidates' cosine similarities lay in a narrow band (0.489–0.559), while their cross-encoder scores ranged from −0.81 to 6.49. The critical chunk stood clearly apart at 6.49; the next-best was 1.96.

## 3. Qualitative and Rank-Shift Analysis

### 3.1 Overall Outcome

| Measure | Result |
| --- | --- |
| Critical chunk in baseline Top 3 | 9 / 16 |
| Critical chunk in re-ranked Top 3 | 14 / 16 |
| Critical chunk anywhere in dense Top 10 (upper bound for re-ranking) | 15 / 16 |
| Outcome per query | 8 correct in both · 6 rescued by re-ranking · 1 degraded · 1 missed by both |

Re-ranking moved the critical chunk into the context for six queries and removed it for one. These are descriptive counts from a small, purposely difficult query set. They show the kinds of failure re-ranking can fix, not a general accuracy rate.

### 3.2 Failure Cases Restored by Re-Ranking

| Query | Category | Critical chunk | Rank shift | Answer 1 (baseline) | Answer 2 (re-ranked) |
| --- | --- | --- | --- | --- | --- |
| Q01 Are password hints allowed? | Keyword trap | C046, §5.1.1.2 | #6 → #3 | "Context does not answer" | Correct: verifiers SHALL NOT permit storing a hint |
| Q02 How often should users be forced to change their passwords? | Keyword trap | C198, §10.2.1 | #6 → #1 | "Context does not answer" | Correct: not periodically; only on user request or evidence of compromise |
| Q05 How long is an out-of-band authentication secret valid? | Exception / specific limit | C066, §5.1.3.2 | #4 → #1 | "Context does not answer" | Correct: invalid if not completed within 10 minutes |
| Q08 Which authenticators are permitted at AAL3? | Level-specific rule | C028, §4.3.1 | #10 → #2 | General AAL3 requirements only; omits the list of permitted authenticators | Lists the permitted combinations |
| Q15 Can push notifications be used for out-of-band authentication? | Buried detail | C064, §5.1.3.2 | #5 → #1 | "Context does not answer" | Correct: verifier MAY send a push notification |
| Q04 Can SMS be used to deliver one-time codes? | Keyword trap | C061, §5.1.3.1 | #8 → #2 | "Context does not answer" | Partially correct: omits that SMS is RESTRICTED |

**Keyword trap (Q01, Q02).** The user asks about "passwords", but the normative rules say "memorized secret". The baseline's top three for Q01 were:

- C198 (usability guidance, §10.2.1)
- C243 (Appendix A heading)
- C244 (Appendix A.1 introduction)

All three use the word "password" heavily, and none contains the rule. The rule chunk C046 ("Memorized secret verifiers SHALL NOT permit the subscriber to store a 'hint'…") was ranked 6th. The cross-encoder reads the query and passage together, so it matched the question's intent to the specific prohibition and promoted C046 to 3rd. Answer 1 said the context did not answer the question; Answer 2 stated the rule and cited C046. Q02 follows the same pattern: Appendix A chunks on password length and complexity filled the baseline Top 3, while the chunk stating when a change may be forced was promoted from 6th to 1st.

**Exception / specific limit (Q05).** Eight of the 10 candidates come from the out-of-band authentication sections, and their cosine similarities are nearly identical. The chunk with the 10-minute validity limit was 4th, just outside the context. The cross-encoder ranked it 1st with a wide score margin. Without it, the baseline answer correctly declined to answer, but the user received no information.

**Level-specific rule (Q08).** The list of permitted AAL3 authenticators (§4.3.1) was 10th, behind general chunks about all three assurance levels (including the AAL2 permitted-authenticator list). The baseline answer was not wrong, but it described hardware and FIPS 140 requirements instead of answering the question. This is an omission of the essential clause. Re-ranking promoted §4.3.1 to 2nd, and Answer 2 listed the permitted combinations. The list continues into the next chunk (C029), so Answer 2 gives five of the six combinations, which is a chunk-boundary effect.

**Effect on the answers.** In none of the six cases did the baseline LLM hallucinate a rule. It either stated that the context did not answer the question (five cases) or gave true but incomplete information (Q08). With a strict "answer only from the context" prompt, missing the critical chunk shows up as **omission**, not invention. Re-ranking turned these into correct, cited answers in five of the six cases. The exception is Q04, discussed in Section 3.3.

### 3.3 Negative and Unresolved Cases

**Re-ranking degraded the ranking (Q13: "Can a password alone satisfy AAL2?").** The chunk stating the AAL2 rule (C018: authentication SHALL use a multi-factor authenticator or two single-factor authenticators) fell from 2nd to 8th. The cross-encoder preferred C022, which mentions PINs and AAL2 but is about device unlocking. The answer was unaffected, because the two-factor requirement is repeated in C005 and C017, which stayed in the Top 3. A similar level confusion occurred in Q08: the cross-encoder's top candidate was the *AAL2* permitted-authenticator chunk. Cross-encoders are more precise but not infallible, especially when chunks differ only in a level number.

**First-stage recall limits re-ranking (Q03, Q04).**

- **Q03 "What is the minimum password length?"** The rule ("at least 8 characters in length if chosen by the subscriber", C043) was ranked 11th by the bi-encoder, behind Appendix A's discussion of password length. It never entered the Top 10, so both pipelines answered that the context does not contain the answer.
- **Q04 (SMS).** Re-ranking surfaced a relevant chunk, but the essential restriction ("Use of the PSTN for out-of-band verification is RESTRICTED", C068) was ranked 31st. Answer 2 therefore described SMS as usable without its RESTRICTED status. That is an omitted clause which re-ranking could not fix.

A re-ranker can only reorder what the first stage retrieves. Its benefit is bounded by the bi-encoder's recall at k = 10.

**Queries where both pipelines succeeded.** The remaining eight queries succeeded in both pipelines. They include all four negation queries (Q09–Q12), whose "SHALL NOT" chunks the bi-encoder already ranked 1st or 2nd. On this document, the bi-encoder's main weakness was vocabulary mismatch and near-duplicate sections, not negation.

## 4. Latency Profiling

Timings are wall-clock (`time.perf_counter`) on a CPU-only laptop (12 logical cores), with models loaded and warmed up. Each of the 16 queries was run 20 times, giving 320 measurements per retrieval stage. Model loading and the one-off corpus embedding (6.4 s) are excluded. LLM latency is one Gemini API call per answer.

| Stage | Median (ms) | p95 (ms) |
| --- | --- | --- |
| Pipeline 1 retrieval: query embedding + FAISS Top 10 + Top 3 | 8.2 | 11.2 |
| Re-ranking step: cross-encoder on 10 pairs + sort | 183.3 | 250.5 |
| Pipeline 2 retrieval: dense retrieval + re-ranking | 191.6 | 261.6 |
| LLM generation, Answer 1 (n = 16) | 1648.7 | 2670.0 |
| LLM generation, Answer 2 (n = 16) | 1433.8 | 1948.8 |

Re-ranking makes retrieval about **23 times slower** (8.2 ms → 191.6 ms), because the cross-encoder runs ten full transformer passes while the baseline encodes one query. In absolute terms it adds about 0.18 s to a request whose LLM call takes 1.4–1.7 s. The difference between the two LLM medians (about 0.2 s, caused by API variance) is about as large as the entire re-ranking cost. In this setup, generation dominates end-to-end latency, and re-ranking is affordable.

## 5. Scaling: Why Not Apply the Cross-Encoder to the Whole Collection?

| Setting | Per-query cost |
| --- | --- |
| Cross-encoder on 10 candidates (measured) | 0.18 s |
| Cross-encoder on all 254 chunks (measured) | 4.78 s (18.8 ms per pair) |
| Cross-encoder on 1,000,000 chunks (linear extrapolation) | ≈ 18,800 s ≈ 5.2 hours |

A cross-encoder scores a (query, passage) pair, so its cost is one transformer forward pass per passage per query. Nothing can be computed ahead of time, because the score depends on the query, and the cost grows linearly with the collection size. Even this single 254-chunk document takes 4.8 s per query; a realistic collection of a million passages would take hours per question.

A bi-encoder moves almost all of its cost offline. Passage embeddings are computed once (and are reused by every query), and at query time only the query is encoded. The search is a vector similarity lookup. Query encoding plus exact search over 254 vectors took 8.2 ms here, and approximate nearest-neighbour indexes such as HNSW or IVF keep it fast and sub-linear for millions of vectors (a general property, not measured in this experiment).

The two-stage design combines the strengths of both: the cheap bi-encoder narrows the collection to a few candidates, and the expensive cross-encoder spends its compute only where precision matters. The cost of this design is that the final result can be no better than the first stage's recall, as Q03 and Q04 show.

## 6. Conclusion and Limitations

On NIST SP 800-63B, cross-encoder re-ranking raised the number of queries whose critical passage reached the LLM from 9 to 14 out of 16. In five of the six rescued cases, a non-answer or incomplete answer became a correct, cited answer; in the sixth (SMS), the answer improved but remained incomplete. The failures it fixed were mainly vocabulary mismatches between user language and the standard's terminology, and specific limits hidden among near-identical neighbouring chunks. Re-ranking also demoted a relevant chunk once (without harming the answer), and it could not help when the critical passage was outside the dense Top 10. The extra cost was about 0.18 s per query, small next to LLM generation, while cross-encoding the whole collection would not scale.

**Limitations.**

- The evaluation uses 16 queries on a single document. I wrote them after reading the document and refined them while exploring retrieval, so they are targeted stress tests, not a random sample. The counts are descriptive and not statistically significant.
- Relevance was judged by key-phrase containment, and answers were checked manually.
- Both encoders are small MiniLM models, and chunk boundaries split at least one list (Q08).
- Gemini answers can vary between uncached runs; answers are cached in `results/llm_cache.json`, so re-running the notebook reproduces this run.
- The standard's 2017/2020 edition has since been superseded by SP 800-63B-4.

---
