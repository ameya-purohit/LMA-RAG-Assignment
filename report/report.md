# Baseline Retrieval vs. Cross-Encoder Re-Ranking in a RAG Pipeline

Course: LMA\
Author: Ameya Purohit\
Date: 19 October 2026\
Code and results: <https://github.com/ameya-purohit/LMA-RAG-Assignment>

## 1. Setup

For this assignment I built two retrieval pipelines over the same PDF and compared them. Both pipelines first use a bi-encoder to find the 10 chunks most similar to the question. The baseline then simply passes the top 3 of those 10 to the LLM. The second pipeline re-scores all 10 with a cross-encoder, re-sorts them, and passes its own top 3 to the LLM. Everything else, including the LLM, its settings and the prompt, is kept the same, so any difference in the answers comes only from which 3 chunks were chosen.

I used NIST Special Publication 800-63B, the US government guideline on digital authentication (2017 edition, updated in 2020). It is an 80-page technical standard, and I chose it because it is genuinely hard to search. Its rules depend on exact wording such as "shall", "shall not" and "should". Many rules are repeated with small changes for three assurance levels (AAL1, AAL2 and AAL3). It also uses formal terms where a normal user would use everyday words. For example, it says "memorized secret" instead of "password" and "PSTN" (the public telephone network) instead of "SMS". On top of that, the appendix on password strength talks about the same topics as the actual rules, so it acts as a natural distractor.

| Part | What I used |
| --- | --- |
| Document | NIST SP 800-63B, pages 12 to 80 (the cover, contents and errata pages were skipped) |
| Chunks | 254 chunks of at most 200 tokens, split at sentence boundaries, with a small overlap between neighbouring chunks |
| Bi-encoder | all-MiniLM-L6-v2, with a FAISS index using cosine similarity |
| Cross-encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Gemini 3.1 Flash Lite, temperature 0, told to answer only from the given chunks |
| Test questions | 16 questions written from the document |

Each chunk has a number (chunk 0 to chunk 253) and remembers which section and page of the PDF it came from, so every result below can be traced back to the original text. For each test question, I found the sentence in the document that actually answers it. I call the chunk containing that sentence the "key chunk". The question is then simply whether the key chunk reaches the LLM.

## 2. Bi-Encoder vs. Cross-Encoder

A bi-encoder turns the question and each chunk into vectors separately, without either one ever seeing the other. Relevance is just the cosine similarity between the two vectors. Since the chunk vectors do not depend on the question, they can all be computed once in advance and stored in an index. At question time, only the question needs to be encoded, which makes search very fast. The downside is that each chunk is squeezed into a single vector before the question is known. Small but important details, such as a "not", an exception, or a synonym, can easily get lost.

A cross-encoder reads the question and the chunk together as one input. Inside the transformer, every word of the question can attend to every word of the chunk, and the model outputs a single relevance score for that pair. This makes it much better at judging whether a chunk really answers the question. The price is that nothing can be prepared in advance, because the score depends on the question, and every candidate chunk needs its own full pass through the model.

I could see this difference in my results. For the question about how long an out-of-band authentication code stays valid (Q05), the cosine similarities of the 10 candidates were all squeezed between 0.489 and 0.559. The cross-encoder scores for the same 10 chunks ranged from -0.81 to 6.49, and the key chunk was clearly ahead at 6.49, with the next best at only 1.96.

## 3. Rank Shifts and Answer Comparison

### 3.1 Overall Results

| | Baseline | Re-ranking |
| --- | --- | --- |
| Key chunk reached the LLM (in the top 3) | 9 out of 16 | 14 out of 16 |

Out of the 16 questions:

- 8 worked in both pipelines.
- 6 failed in the baseline but were fixed by re-ranking.
- 1 got worse with re-ranking.
- 1 failed in both.

For 15 of the 16 questions, the key chunk was somewhere in the bi-encoder's top 10, which is the best re-ranking could possibly do. Since I only used 16 questions, and I picked them to be difficult, these counts show the kinds of mistakes re-ranking can fix rather than a general accuracy figure.

### 3.2 Failure Cases Fixed by Re-Ranking

| Question | Type | Key chunk (section) | Rank before | Rank after | Baseline answer | Re-ranked answer |
| --- | --- | --- | --- | --- | --- | --- |
| Q01: Are password hints allowed? | Keyword trap | Chunk 46 (Section 5.1.1.2) | 6 | 3 | Said the context does not answer | Correct: a hint that can be seen before logging in is not allowed |
| Q02: How often should users be forced to change their passwords? | Keyword trap | Chunk 198 (Section 10.2.1) | 6 | 1 | Said the context does not answer | Correct: not on a schedule, only if the user asks or there is evidence of compromise |
| Q05: How long is an out-of-band authentication secret valid? | Exception / specific limit | Chunk 66 (Section 5.1.3.2) | 4 | 1 | Said the context does not answer | Correct: 10 minutes |
| Q08: Which authenticators are permitted at AAL3? | Level-specific rule | Chunk 28 (Section 4.3.1) | 10 | 2 | Gave general AAL3 rules but left out the actual list | Gave the list of allowed combinations |
| Q15: Can push notifications be used for out-of-band authentication? | Buried detail | Chunk 64 (Section 5.1.3.2) | 5 | 1 | Said the context does not answer | Correct: yes, the verifier may send a push notification |
| Q04: Can SMS be used to deliver one-time codes? | Keyword trap | Chunk 61 (Section 5.1.3.1) | 8 | 2 | Said the context does not answer | Only partly correct (see Section 3.3) |

**Keyword trap (Q01 and Q02).** The question says "password", but the rule in the document says "memorized secret". For Q01, the baseline's top 3 were chunk 198 (usability advice), chunk 243 (the start of the appendix on password strength) and chunk 244 (the appendix introduction). All three use the word "password" a lot, but none of them contains the rule. The chunk with the actual rule, chunk 46, says that verifiers "SHALL NOT permit the subscriber to store a hint that is accessible to an unauthenticated claimant". The bi-encoder ranked it only 6th. Because the cross-encoder reads the question and chunk together, it recognised that chunk 46 answers the question and moved it up to 3rd. The baseline answer said the context did not contain the answer, while the re-ranked answer stated the rule correctly. Q02 followed the same pattern. The baseline's top 3 were all appendix chunks about password length and complexity, while the chunk saying when a password change can be forced moved from 6th to 1st.

**Exception / specific limit (Q05).** Eight of the 10 candidates came from the out-of-band authentication sections, and their cosine scores were almost the same. The chunk with the 10-minute limit came 4th, just outside the top 3. The cross-encoder put it 1st by a wide margin. Without it, the baseline answer correctly said it could not find the answer, but that still left the user with no information.

**Level-specific rule (Q08).** The list of authenticators allowed at AAL3 (Section 4.3.1) was ranked 10th, behind general chunks about all three levels, including the similar list for AAL2. The baseline answer was not wrong, but it talked about hardware and FIPS 140 requirements and never gave the list that was asked for, so it missed the essential part. Re-ranking moved Section 4.3.1 up to 2nd, and the answer then listed the allowed combinations. It gave five of the six, because the sixth was cut off into the next chunk, which is a side effect of where the chunk boundary fell.

**Did the baseline hallucinate?** No. In all six cases the baseline LLM either said the context did not contain the answer (five cases) or gave correct but incomplete information (Q08). Because the prompt told the model to use only the given chunks, a missing key chunk led to a missing answer, not a made-up rule. Re-ranking turned these into correct answers that cite their source in five of the six cases. Q04 is the exception, and I discuss it below.

### 3.3 Cases Where Re-Ranking Did Not Help

**Re-ranking made one ranking worse (Q13: "Can a password alone satisfy AAL2?").** The chunk with the AAL2 rule (chunk 18), which says authentication must use a multi-factor authenticator or two single-factor authenticators, dropped from 2nd to 8th. The cross-encoder preferred chunk 22, which covers other AAL2 requirements, including the rule that unlocking a phone with a PIN does not count as a factor, but does not say that two factors are needed. Luckily, the answer was still correct, because the same two-factor requirement also appears in chunks 5 and 17, which stayed in the top 3. I saw a similar mix-up in Q08, where the cross-encoder's first choice was the AAL2 list instead of the AAL3 one. So the cross-encoder is more accurate, but it can still confuse chunks that differ only in the level number.

**Re-ranking cannot fix what the first stage misses (Q03 and Q04).**

- For Q03 ("What is the minimum password length?"), the rule says passwords chosen by the user must be "at least 8 characters" long. That rule is in chunk 43, but the bi-encoder ranked it 11th, behind the appendix chunks about password length. Since only the top 10 are re-ranked, the cross-encoder never saw it, and both pipelines said they could not answer.
- For Q04 (SMS), re-ranking did bring a relevant chunk into the top 3. However, the most important rule, that using the phone network for this "is RESTRICTED", is in chunk 68, which the bi-encoder ranked 31st. As a result, the re-ranked answer said SMS can be used and left out the restriction.

A re-ranker can only re-order what it is given, so it can never be better than the first stage's top 10.

**Questions that worked in both pipelines.** The other eight questions worked in both pipelines. These include all four of my negation questions (Q09 to Q12), where the bi-encoder already ranked the "shall not" chunk 1st or 2nd. So for this document, the bi-encoder's real weakness was different wording between the question and the document, and near-duplicate sections, rather than negation.

## 4. Latency

I timed each stage on my laptop's CPU (12 logical cores) after the models were loaded and warmed up. Each of the 16 questions was run 20 times, giving 320 measurements per stage. Loading the models and embedding the document (6.4 seconds, done only once) are not included. The LLM time is one Gemini API call per answer.

| Stage | Median time | 95th percentile |
| --- | --- | --- |
| Baseline retrieval (encode question, search top 10, keep top 3) | 8.2 ms | 11.2 ms |
| Re-ranking step only (cross-encoder on 10 chunks, then sort) | 183.3 ms | 250.5 ms |
| Full re-ranking retrieval (search plus re-ranking) | 191.6 ms | 261.6 ms |
| LLM answer, baseline pipeline (16 calls) | 1648.7 ms | 2670.0 ms |
| LLM answer, re-ranking pipeline (16 calls) | 1433.8 ms | 1948.8 ms |

Re-ranking makes retrieval about 23 times slower (8.2 ms against 191.6 ms), because the cross-encoder needs ten full model passes, while the baseline only encodes the question once. In real terms, though, it adds only about 0.18 seconds to a request where the LLM call alone takes about 1.4 to 1.7 seconds. The gap between the two LLM medians, about 0.2 seconds, is caused just by API variation and is roughly as large as the whole re-ranking cost. So in this setup, the LLM dominates the total time, and re-ranking is cheap enough to be worth it.

## 5. Why Not Use the Cross-Encoder on Everything?

| Setting | Time per question |
| --- | --- |
| Cross-encoder on the 10 candidates (measured) | 0.18 seconds |
| Cross-encoder on all 254 chunks (measured) | 4.78 seconds (about 18.8 ms per chunk) |
| Cross-encoder on 1,000,000 chunks (estimated by scaling up) | about 18,800 seconds, which is about 5.2 hours |

The cross-encoder scores a question and a chunk together, so every chunk needs a full model pass for every new question. None of this work can be done ahead of time, because the score depends on the question. The cost therefore grows directly with the number of chunks. Even for my one 254-chunk document it took almost 5 seconds per question, and a real collection with a million chunks would take hours for a single question.

The bi-encoder does most of its work in advance. Every chunk is embedded once and reused for every question, so at question time it only encodes the question and looks up the nearest vectors. Here that took 8.2 ms in total. For very large collections, approximate search indexes such as HNSW keep this lookup fast, although I did not measure that myself.

This is why the two are combined. The fast bi-encoder narrows thousands or millions of chunks down to a handful, and the slow but accurate cross-encoder is only used on those few. The trade-off is that the final result can never be better than what the first stage finds, as Q03 and Q04 showed.

## 6. Conclusion and Limitations

On NIST SP 800-63B, re-ranking increased the number of questions where the key chunk reached the LLM from 9 to 14 out of 16. In five of the six fixed cases, a missing or incomplete answer became a correct one, and in the sixth (SMS) the answer improved but still left out an important restriction. The problems it fixed were mostly cases where the question used different words from the document, or where the specific rule was hidden among very similar chunks. Re-ranking also pushed one useful chunk down once, though without hurting the answer, and it could not help when the key chunk was not in the top 10 at all. It added only about 0.18 seconds per question, which is small compared to the LLM, whereas running the cross-encoder over the whole collection would not scale.

Limitations:

- I only tested 16 questions on one document. I wrote them myself after reading the document, so they are targeted stress tests rather than a random sample, and the numbers are not statistically significant.
- I decided whether a chunk was relevant by checking whether it contained a key phrase from the answering sentence, and I read all the answers myself.
- Both encoders are small models, and in at least one case (Q08) a chunk boundary split a list in two.
- Gemini's answers can change slightly between runs. I saved the answers from this run so that the notebook reproduces them exactly.
- This edition of the standard has since been replaced by a newer version (SP 800-63B-4).
