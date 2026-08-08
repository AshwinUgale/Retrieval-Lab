# Contributing to Retrieval Lab

Thanks for your interest! Retrieval Lab is a focused, honest tool — a corpus-specific
retrieval benchmark with **deterministic, stage-level failure attribution** — and
contributions that keep it *trustworthy and simple* are very welcome. This guide gets
you productive fast.

New to the codebase? The `src/retrieval_lab/` modules map to the pipeline stages:
`text.py` (chunkers) · `embedding` (embedders) · `pipeline.py` (dense/sparse/hybrid) ·
`scoring.py` (rerankers) · `gold.py` (span-level gold) · `attribution.py` (stage
attribution) · `metrics.py` · `sweep.py` (the config sweep) · `report.py` (HTML) ·
`cli.py`.

## Development setup

```bash
git clone https://github.com/AshwinUgale/Retrieval-Lab
cd Retrieval-Lab
pip install -e ".[dev]"          # editable install + pytest & ruff
pytest -q                        # the suite (all green)
retrieval-lab demo               # keyless end-to-end demo (no model download / API key)
```

Before opening a PR, run what CI runs:

```bash
ruff check src tests             # lint
ruff format src tests            # auto-format (CI checks this)
pytest -q                        # unit + attribution + pipeline tests
```

The heavy model paths are optional extras: `[real-embed]` (E5/BGE via
sentence-transformers), `[rerank]` (cross-encoder), `[ann]` (HNSW). The **core and the
whole test suite run offline and keyless** via the deterministic test embedder — please
keep it that way (see Guidelines).

## Good first contributions

- **A new chunker** — highest-value, smallest surface. Add a class next to
  `FixedSizeChunker` / `RecursiveChunker` / `ParentChildChunker` and wire it into the CLI
  `--chunkers` parser. Ideas: sliding-window/overlap, sentence-based, markdown/structure-aware.
- **A new reranker** — follow `LexicalReranker` / `CrossEncoderReranker` (e.g. an MMR
  diversity reranker).
- **A dataset importer** — follow `import-squad` / `import-beir` in `cli.py` (e.g. MS MARCO,
  Natural Questions, or a generic CSV importer). Answer-span datasets map directly to
  source-span gold; passage-level datasets import as coarser gold — say which.
- **An embedder backend** — behind a new extra (e.g. an API embedder), lazily imported like
  the E5/BGE path so the core stays dependency-free.
- **Docs & examples** — a worked corpus, a report walkthrough, clearer authoring docs.

Browse issues labeled **`good first issue`** and **`help wanted`**. For anything larger
than a single chunker/reranker/importer, open an issue first so we agree on the shape.

## Guidelines

- **Keep the core dependency-free** (NumPy + stdlib). Real models (sentence-transformers,
  hnswlib) go behind extras and are imported lazily; the demo and tests must run offline.
- **Fail closed and stay honest.** Gold offsets that don't match their `quoted_text` are
  rejected; below `--min-sample` the tool withholds a verdict; attribution is only emitted
  when the pipeline is stage-decomposable. New code should preserve these gates — a
  benchmark that reports a confident wrong number is worse than one that says "I can't tell."
- **Determinism.** Same inputs + seed → same result. Rankings, attribution, and the report
  must be reproducible; thread the seed, don't reach for global RNG state.
- **Gold is a source-document span**, never a chunk id — this is what keeps evaluation stable
  across chunking strategies. Don't add metrics or attribution that assume fixed chunk ids.
- **Every public component and metric gets a test.**
- **Update `CHANGELOG.md`** under the top section for user-visible changes.

## Pull request checklist

- [ ] `pytest -q` passes and new behavior has a test
- [ ] `ruff check src tests` and `ruff format src tests` are clean
- [ ] `retrieval-lab demo` still runs green (offline)
- [ ] `CHANGELOG.md` updated (for user-visible changes)
- [ ] docs / README touched if the change is user-facing
- [ ] the PR description says *what* and *why* in a sentence or two

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you
agree to uphold it — be kind, be constructive.
