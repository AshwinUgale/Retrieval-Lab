"""Embedding-space geometry lenses — risk indicators, never verdicts (spec §I.7).

These describe pathologies of an embedding space that can *explain why* recall is low, and
are reported as context — never promoted to a per-query cause:

- **Hubness** — the skewness of the k-occurrence distribution (how often each point appears
  in others' k-NN lists). A few "hub" vectors that are everyone's neighbour distort
  retrieval; high positive skew flags it.
- **Anisotropy** — the mean cosine of random pairs. In an isotropic space this is ~0; a large
  positive value means all vectors sit in a narrow cone, so cosine separates them poorly.
- **Cross-model mismatch** — how far the query vectors sit from the corpus vectors
  (centroid cosine + each query's nearest-corpus similarity). Low values mean queries land
  in a different region than any document — a retrieval risk, especially when queries and
  passages are embedded by different models or without the right role prefix.

All inputs are assumed L2-normalized (the embedder guarantees it); cosine is a dot product.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from retrieval_lab.embedding.base import l2_normalize


def _skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / sd**3)


def hubness(vectors: np.ndarray, k: int = 10) -> dict:
    """k-occurrence skewness (spec §I.5). Higher positive skew ⇒ more hub-dominated."""
    v = l2_normalize(np.asarray(vectors, dtype=float))
    n = len(v)
    if n < 3:
        return {"k": 0, "skewness": 0.0, "n": n}
    k = min(k, n - 1)
    sim = v @ v.T
    np.fill_diagonal(sim, -np.inf)  # exclude self
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        for j in np.argpartition(-sim[i], k - 1)[:k]:
            counts[j] += 1
    return {"k": k, "skewness": _skewness(counts), "n": n}


def anisotropy(vectors: np.ndarray) -> dict:
    """Mean random-pair cosine and an isotropy score ``1 - mean`` (spec §I.5)."""
    v = l2_normalize(np.asarray(vectors, dtype=float))
    n = len(v)
    if n < 2:
        return {"mean_random_cosine": 0.0, "isotropy": 1.0, "n": n}
    sim = v @ v.T
    off = (sim.sum() - np.trace(sim)) / (n * (n - 1))  # mean of off-diagonal
    mean_cos = float(off)
    return {"mean_random_cosine": mean_cos, "isotropy": 1.0 - mean_cos, "n": n}


def cross_model_mismatch(query_vectors: np.ndarray, corpus_vectors: np.ndarray) -> dict:
    """How far queries sit from the corpus: centroid cosine + mean nearest-corpus similarity."""
    q = l2_normalize(np.asarray(query_vectors, dtype=float))
    c = l2_normalize(np.asarray(corpus_vectors, dtype=float))
    if len(q) == 0 or len(c) == 0:
        return {"centroid_cosine": 0.0, "mean_query_to_corpus_nn": 0.0, "n_query": len(q)}
    q_centroid = l2_normalize(q.mean(axis=0))
    c_centroid = l2_normalize(c.mean(axis=0))
    centroid_cosine = float(q_centroid @ c_centroid)
    nn = (q @ c.T).max(axis=1)  # each query's best corpus similarity
    return {
        "centroid_cosine": centroid_cosine,
        "mean_query_to_corpus_nn": float(nn.mean()),
        "n_query": int(len(q)),
    }


@dataclass
class GeometryReport:
    hubness: dict
    anisotropy: dict
    cross_model_mismatch: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def geometry_report(
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray | None = None,
    k: int = 10,
) -> GeometryReport:
    """Compute the geometry lenses over a corpus (and queries, if given)."""
    hub = hubness(corpus_vectors, k=k)
    aniso = anisotropy(corpus_vectors)
    mismatch = None
    if query_vectors is not None and len(query_vectors) > 0:
        mismatch = cross_model_mismatch(query_vectors, corpus_vectors)

    notes: list[str] = ["These are risk indicators, not per-query verdicts (spec §I.7)."]
    if hub["skewness"] > 1.0:
        notes.append(f"hubness skew {hub['skewness']:.2f} is high — a few hub vectors may "
                     "dominate neighbourhoods.")
    if aniso["mean_random_cosine"] > 0.3:
        notes.append(f"anisotropy {aniso['mean_random_cosine']:.2f} is high — vectors share a "
                     "narrow cone, so cosine separates them weakly.")
    if mismatch and mismatch["mean_query_to_corpus_nn"] < 0.2:
        notes.append("queries land far from every document (low nearest-corpus similarity) — "
                     "possible query/passage embedding mismatch.")
    return GeometryReport(hubness=hub, anisotropy=aniso, cross_model_mismatch=mismatch,
                          notes=notes)


def render_geometry(report: GeometryReport) -> str:
    lines = ["Embedding-space geometry (risk indicators, not verdicts):", ""]
    h = report.hubness
    lines.append(
        f"  hubness      : k-occurrence skew {h['skewness']:.3f}  (k={h['k']}, n={h['n']})"
    )
    a = report.anisotropy
    lines.append(f"  anisotropy   : mean random-pair cosine {a['mean_random_cosine']:.3f}  "
                 f"(isotropy {a['isotropy']:.3f})")
    if report.cross_model_mismatch:
        m = report.cross_model_mismatch
        lines.append(f"  query/corpus : centroid cosine {m['centroid_cosine']:.3f}, "
                     f"mean nearest-corpus sim {m['mean_query_to_corpus_nn']:.3f}")
    lines.append("")
    lines.extend(f"  - {n}" for n in report.notes)
    return "\n".join(lines)
