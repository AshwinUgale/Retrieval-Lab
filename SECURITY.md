# Security policy

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository, or contact the
maintainer directly, rather than filing a public issue. We aim to acknowledge
within a few days.

## Threat model & trust boundaries

Retrieval Lab is a developer tool you run locally or in your own CI. A few things
are worth understanding before you run it.

- **It reads your corpus and query files** (`docs.jsonl`, `queries.jsonl`) as data,
  not code. Gold spans are verified against their `quoted_text` and fail closed on a
  mismatch. Treat the input files as you would any data you load.
- **Optional extras download models.** `[real-embed]` / `[rerank]` fetch model
  weights from the Hugging Face Hub, and `[ann]` builds a native HNSW index. These are
  opt-in; the core and the full test suite run offline with the deterministic embedder.
- **The HTML report is self-contained** — inline CSS, no scripts, no external resources,
  and every value is HTML-escaped — so it is safe to open locally and share.

If you find a way to make the tool emit a confident but wrong verdict (e.g. bypass the
gold-verification or min-sample gates), that's a security-relevant honesty bug — please
report it.
