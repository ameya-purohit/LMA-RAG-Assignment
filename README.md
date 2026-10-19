# RAG Assignment: Baseline Dense Retrieval vs. Cross-Encoder Re-Ranking

All code is in one notebook: **[`rag_reranking.ipynb`](rag_reranking.ipynb)**. It compares two retrieval pipelines that run over the same chunked PDF and share the same Top-10 candidates:

| | Pipeline 1 – Baseline | Pipeline 2 – Re-ranking |
| --- | --- | --- |
| Stage 1 | Bi-encoder `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` (cosine) → Top 10 | same Top 10 |
| Stage 2 | dense ranks 1–3 | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs → sort descending → Top 3 |
| Answer | `gemini-3.1-flash-lite`, temperature 0, identical prompt → Answer 1 | same → Answer 2 |

## Document

[NIST SP 800-63B](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf), *Digital Identity Guidelines: Authentication and Lifecycle Management* (June 2017, updated 2020-03-02). It is an 80-page normative standard with many `SHALL NOT` rules, exceptions and level-specific (AAL1/2/3) limits. It is a US-government work in the public domain, and a copy is included in `data/`.
NIST withdrew this edition on 2025-08-01 and replaced it with SP 800-63B-4. The answers here describe the 2017/2020 text only.

The notebook indexes PDF pages 12–80 (Section 1 through Appendix A) and skips the cover, withdrawal notice, table of contents and errata.

## Repository layout

```text
rag_reranking.ipynb   # the full implementation, executed, with outputs
data/                 # the source PDF
results/              # outputs written by the notebook (the run discussed in the report)
  chunks.jsonl            # the exact corpus: chunk_id, section, pages, text
  rankings_top10.csv      # for every query: all 10 candidates with dense rank, cosine, CE score, re-rank rank
  rank_shift_summary.csv  # per query: critical chunk rank shifts and whether it reached the top 3
  answers.json            # Answer 1 / Answer 2 for every query
  llm_cache.json          # cached Gemini responses (lets the notebook re-run without new API calls)
  latency.csv, scaling.csv
report/report.md      # report template: structure + measured tables (prose written by the author)
report/report_notes.md  # factual notes and observations supporting the report
requirements.txt
```

## Setup and running

Requires Python 3.12. No GPU is needed: the embedding and re-ranking models run on CPU, and only answer generation calls an API.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -r requirements.txt

# Gemini API key (https://aistudio.google.com/apikey). Never commit it: .env is git-ignored.
echo 'GEMINI_API_KEY=your-key-here' > .env

# Run the notebook end to end (or open it in Jupyter / VS Code and choose "Run All").
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 rag_reranking.ipynb
```

* The models (~180 MB in total) are downloaded from Hugging Face on the first run.
* **LLM answers are cached** in `results/llm_cache.json`, keyed by model + generation settings + full prompt. A re-run reproduces the committed answers without API calls, and works even without a key. Delete that file to regenerate the answers.
* Without a key and without the cache, retrieval, re-ranking, rank analysis and latency still run, and answer generation is skipped and reported as such.
* **Model choice:** free-tier keys allow only about 20 requests per day for `gemini-3.8-flash`, and one run needs 32 (16 queries × 2 answers). The notebook therefore uses `gemini-3.1-flash-lite`. To use another model, change `LLM_MODEL` in the first code cell. The notebook retries on HTTP 429/503 with backoff, and reports calls that still fail (or hit a daily quota) as failures.
* Retrieval and re-ranking are deterministic. Gemini answers can differ slightly between uncached runs even at temperature 0.
