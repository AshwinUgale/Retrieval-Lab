"""Command-line interface (spec §I.12).

    retrieval-lab run --corpus docs.jsonl --queries queries.jsonl \\
        --embed-models det --chunkers fixed,recursive --retrieval dense,hybrid \\
        --rerank none,lexical --top-k 5 --candidate-n 50 --budget-tokens 2000,4000 \\
        --json out.json
    retrieval-lab explain --json out.json --query-id Q17
    retrieval-lab pareto  --json out.json

``run`` executes a sweep and writes machine-readable JSON; ``explain`` and ``pareto`` read
that JSON so they operate on a prior run. The process exits non-zero for CI when the baseline
is broken or a ``--fail-under`` quality gate is missed (spec §0: a CI-usable exit code).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from retrieval_lab.chunking import FixedSizeChunker, RecursiveChunker
from retrieval_lab.embedding import DeterministicEmbedder, EmbeddingCache
from retrieval_lab.gold import OffsetVerificationError, load_documents, load_queries
from retrieval_lab.report import read_json, render_explain, render_pareto, render_report, write_json
from retrieval_lab.retrieval import LexicalReranker
from retrieval_lab.sweep import SweepSpec, run_sweep

EXIT_OK = 0
EXIT_QUALITY_GATE = 1
EXIT_INPUT_ERROR = 2


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _make_embedder(name: str, cache: EmbeddingCache):
    if name in ("det", "det-hash"):
        return DeterministicEmbedder(dim=2048, name=name, cache=cache)
    if name in ("e5", "bge"):
        from retrieval_lab.embedding import bge_embedder, e5_embedder

        factory = e5_embedder if name == "e5" else bge_embedder
        emb = factory(cache=cache)
        # Re-key under the short CLI name so config ids stay readable.
        emb.name = name
        return emb
    raise ValueError(f"unknown embed model {name!r} (use det, e5, or bge)")


def _make_chunker(spec: str):
    kind, _, size = spec.partition(":")
    chunk_size = int(size) if size else 400
    if kind == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size)
    if kind == "recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    raise ValueError(f"unknown chunker {spec!r} (use fixed[:size] or recursive[:size])")


def _make_reranker(name: str):
    if name == "none":
        return None, None
    if name == "lexical":
        return "lexical", LexicalReranker()
    if name == "ce":
        from retrieval_lab.retrieval import CrossEncoderReranker

        return "ce", CrossEncoderReranker()
    raise ValueError(f"unknown reranker {name!r} (use none, lexical, or ce)")


def _build_spec(args: argparse.Namespace) -> SweepSpec:
    cache = EmbeddingCache()
    embedders = {n: _make_embedder(n, cache) for n in _csv(args.embed_models)}
    chunkers = {c: _make_chunker(c) for c in _csv(args.chunkers)}
    rerankers = dict(_make_reranker(r) for r in _csv(args.rerank))
    budgets: list[int | None] = [None]
    if args.budget_tokens:
        budgets += [int(b) for b in _csv(args.budget_tokens)]
    return SweepSpec(
        embedders=embedders,
        chunkers=chunkers,
        retrieval_modes=tuple(_csv(args.retrieval)),
        rerankers=rerankers,
        top_k=args.top_k,
        candidate_n=args.candidate_n,
        budgets=tuple(budgets),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        documents = load_documents(args.corpus)
        queries = load_queries(args.queries, documents, strict=True)
    except OffsetVerificationError as exc:
        print(f"error: gold verification failed (fail closed): {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except (OSError, ValueError) as exc:
        print(f"error: could not load inputs: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if not queries:
        print("error: no queries loaded (all refused or empty).", file=sys.stderr)
        return EXIT_INPUT_ERROR

    spec = _build_spec(args)
    sweep = run_sweep(documents, queries, spec, min_sample=args.min_sample, seed=args.seed)
    print(render_report(sweep, top=args.top))
    if args.json:
        write_json(sweep, args.json)
        print(f"\nWrote {args.json}")

    if sweep.validity.baseline_broken:
        print("\nCI: baseline broken (zero recall under all configs).", file=sys.stderr)
        return EXIT_QUALITY_GATE
    if args.fail_under is not None:
        best = sweep.best()
        if best is None or best.hit_rate < args.fail_under:
            got = 0.0 if best is None else best.hit_rate
            print(f"\nCI: best hit@k {got:.2f} < --fail-under {args.fail_under:.2f}.",
                  file=sys.stderr)
            return EXIT_QUALITY_GATE
    return EXIT_OK


def _cmd_explain(args: argparse.Namespace) -> int:
    sweep = read_json(args.json)
    print(render_explain(sweep, args.query_id))
    return EXIT_OK


def _cmd_pareto(args: argparse.Namespace) -> int:
    sweep = read_json(args.json)
    print(render_pareto(sweep))
    return EXIT_OK


def _cmd_geometry(args: argparse.Namespace) -> int:
    print("geometry lenses (hubness / anisotropy / cross-model mismatch) arrive in a later "
          "phase; see PROJECTS-TECHNICAL-SPEC.md §I.7.", file=sys.stderr)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="sweep configs over a corpus + query set")
    run.add_argument("--corpus", required=True, help="documents JSONL ({id, text, meta?})")
    run.add_argument("--queries", required=True, help="queries JSONL (with source-span gold)")
    run.add_argument("--embed-models", default="det", help="csv: det,e5,bge")
    run.add_argument("--chunkers", default="fixed", help="csv: fixed[:size],recursive[:size]")
    run.add_argument("--retrieval", default="hybrid", help="csv: dense,sparse,hybrid")
    run.add_argument("--rerank", default="none", help="csv: none,lexical,ce")
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--candidate-n", type=int, default=50)
    run.add_argument("--budget-tokens", default="", help="csv of token budgets, e.g. 2000,4000")
    run.add_argument("--min-sample", type=int, default=20, help="min queries for a verdict")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--top", type=int, default=None, help="show only the top N configs")
    run.add_argument("--fail-under", type=float, default=None,
                     help="exit non-zero if the best hit@k is below this")
    run.add_argument("--json", default=None, help="write full results here")
    run.set_defaults(func=_cmd_run)

    explain = sub.add_parser("explain", help="per-stage attribution for one query")
    explain.add_argument("--json", required=True, help="a results JSON from `run`")
    explain.add_argument("--query-id", required=True)
    explain.set_defaults(func=_cmd_explain)

    pareto = sub.add_parser("pareto", help="Pareto frontier over quality x tokens")
    pareto.add_argument("--json", required=True, help="a results JSON from `run`")
    pareto.set_defaults(func=_cmd_pareto)

    geometry = sub.add_parser("geometry", help="embedding-space diagnostics (later phase)")
    geometry.add_argument("--corpus", required=False)
    geometry.add_argument("--embed-model", required=False)
    geometry.set_defaults(func=_cmd_geometry)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
