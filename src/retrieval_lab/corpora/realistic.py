"""A constructed-but-realistic corpus: documentation for a fictional product (spec §I.13).

Unlike the tiny ``build_basic_corpus`` (where every config trivially scores 1.00), this set
is sized and shaped like real documentation, and its queries are authored so that different
configurations genuinely diverge and queries fail at *different* DAG stages:

- **fragmentation / chunk-size** — long procedural answers split across small chunks (a
  small ``fixed`` chunker fragments them; a large chunker keeps them whole).
- **rare-term (BM25 wins)** — questions keyed on rare exact tokens (error codes, flags) that
  a keyless subword embedder dilutes but BM25's IDF nails.
- **paraphrase (dense contributes)** — questions sharing subword/morphological overlap with
  the answer but few exact tokens (the dense advantage is strongest with a real embedding
  model; with the keyless subword embedder it is milder).
- **distractors** — several docs share vocabulary with a query but only one answers it.
- **multi-span** — an answer that requires two separate spans (an ``EvidenceSet``).
- **multi-alternative** — a question answerable from either of two docs (a ``GoldAnswer``
  with two alternatives).

Everything is offline and keyless; gold is located by substring, so offsets and
``quoted_text`` are exact by construction. This is a *constructed* example — labelled as
such — not scraped real data (a real-dataset importer is a separate, later feature).
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval_lab.corpora.constructed import span_in
from retrieval_lab.gold import EvidenceSet, GoldAnswer, Query
from retrieval_lab.models import Document

# --------------------------------------------------------------------------------------
# Corpus: docs for "Cirrus", a fictional hosted vector database.
# --------------------------------------------------------------------------------------

_DOCS: dict[str, str] = {
    "overview": (
        "Cirrus is a hosted vector database for semantic search and retrieval-augmented "
        "generation. It stores high-dimensional embeddings, indexes them for fast "
        "approximate nearest-neighbor search, and returns the most similar records for a "
        "query vector. Cirrus is offered as a fully managed cloud service with regional "
        "deployments."
    ),
    "install": (
        "Install the Cirrus command-line tool with the package manager for your platform. "
        "On macOS and Linux you can run the install script, which downloads the binary and "
        "adds it to your path. After installation, verify the version by running cirrus "
        "version, which prints the client and server build identifiers."
    ),
    "quickstart": (
        "To get started, create a collection and upsert a few vectors. First authenticate "
        "the client with an API key. Then create a collection, insert your embeddings, and "
        "run a query. The quickstart guide walks through indexing a small dataset and "
        "issuing your first similarity search in under five minutes."
    ),
    "collections": (
        "A collection is a named group of vectors that share the same dimension and distance "
        "metric. To create a collection you must supply two required fields: a unique "
        "collection name, and the vector dimension that every record in the collection will "
        "use. An optional description and metadata schema may also be provided at creation "
        "time. The dimension is fixed once the collection is created and cannot be changed."
    ),
    "auth-api-keys": (
        "The simplest way to authenticate a request is with an API key. Create an API key in "
        "the console, then pass it in the Authorization header of every request as a bearer "
        "token. API keys are scoped to a single project and can be granted read-only or "
        "read-write permissions. Treat API keys as secrets and never commit them to source "
        "control."
    ),
    "auth-oauth": (
        "For user-facing applications you can authenticate a request using OAuth 2.0 instead "
        "of a static key. Register an OAuth client, direct the user through the consent "
        "screen, and exchange the authorization code for a short-lived access token. The "
        "access token is then presented as a bearer token, exactly like an API key, but it "
        "expires and can be refreshed."
    ),
    "auth-rotation": (
        "Rotating an API key is a four-step procedure that avoids any downtime. First, create "
        "a second API key in the console alongside the existing one. Second, deploy the new "
        "key to all of your services and confirm that traffic is flowing using it. Third, "
        "monitor the old key's usage in the dashboard until it drops to zero requests. "
        "Fourth, once the old key is idle, revoke it permanently from the console. Never "
        "revoke the old key before the new key is fully deployed, or in-flight requests will "
        "fail with an authentication error."
    ),
    "config-file": (
        "The Cirrus client reads settings from a configuration file named cirrus.toml in the "
        "working directory. The file specifies the endpoint URL, the default collection, and "
        "the request timeout in seconds. Values in the configuration file are overridden by "
        "environment variables, which are in turn overridden by command-line flags."
    ),
    "config-env": (
        "Every configuration setting can also be supplied through an environment variable. "
        "The endpoint is set with CIRRUS_ENDPOINT, the API key with CIRRUS_API_KEY, and the "
        "request timeout with CIRRUS_TIMEOUT. Environment variables are convenient in "
        "containerized deployments where writing a configuration file is awkward."
    ),
    "index-hnsw": (
        "The default index type is HNSW, a graph-based approximate nearest-neighbor index. "
        "HNSW offers excellent recall at low latency and is a good choice for most workloads. "
        "Its build is incremental, so new vectors can be added without rebuilding the whole "
        "index. HNSW uses more memory than a quantized index because it stores the full graph."
    ),
    "index-ivf": (
        "For very large collections that must minimize memory, Cirrus offers an IVF index, "
        "which partitions vectors into clusters and searches only the closest clusters. IVF "
        "uses less memory than HNSW but requires a training step over a representative sample "
        "before it can be queried, and its recall depends on how many clusters are probed."
    ),
    "index-params": (
        "HNSW exposes two tuning parameters. The parameter M controls the number of "
        "neighbors per node in the graph; larger M improves recall at the cost of memory and "
        "build time. The parameter ef_search controls how many candidates are examined at "
        "query time; larger ef_search improves recall at the cost of latency. Increasing "
        "ef_search is the usual first step when recall is too low."
    ),
    "distance-metrics": (
        "Cirrus supports three distance metrics: cosine similarity, dot product, and squared "
        "Euclidean distance. Cosine is the default and is appropriate for normalized "
        "embeddings. The metric is chosen when the collection is created and applies to every "
        "query against that collection."
    ),
    "upsert": (
        "Adding vectors is done with the upsert operation, which inserts new records or "
        "replaces existing ones with the same identifier. Each record carries an id, a vector "
        "of the collection's dimension, and an optional JSON metadata object. Upserts are "
        "acknowledged once the record is durably written, but it may take a moment before the "
        "record becomes visible to queries as the index catches up."
    ),
    "query-topk": (
        "A similarity search returns the top_k records nearest to the query vector, ordered "
        "by score. You choose top_k per request; larger values return more results but cost "
        "more latency. Each result includes the record id, its similarity score, and, if "
        "requested, its stored metadata."
    ),
    "filtering": (
        "Queries can be constrained with a metadata filter so that only records matching a "
        "predicate are considered. Filters support equality, ranges, and set membership over "
        "metadata fields. Filtering is applied during the search, so a filtered query still "
        "returns the top_k nearest records that satisfy the predicate."
    ),
    "hybrid-search": (
        "Hybrid search combines dense vector similarity with sparse keyword matching, then "
        "fuses the two rankings so that results strong in either signal rise to the top. "
        "Hybrid search is most useful when queries contain rare exact terms, such as product "
        "codes or identifiers, that a dense embedding alone tends to overlook."
    ),
    "reranking": (
        "An optional reranking stage re-scores the shortlist returned by first-stage "
        "retrieval using a cross-encoder that reads the query and each candidate together. "
        "Reranking improves precision at the top of the list but adds latency, so it is "
        "applied only to a small candidate set rather than the whole collection."
    ),
    "rate-limits": (
        "Each project is subject to a request rate limit measured in requests per second. "
        "When the limit is exceeded, Cirrus rejects further requests with error code CX-429 "
        "until the rate falls back within the allowance. Clients should back off and retry "
        "with jitter when they receive this code."
    ),
    "errors-client": (
        "Client errors indicate a problem with the request itself. Error code CX-400 means "
        "the request body was malformed. Error code CX-401 means authentication failed, "
        "usually a missing or invalid API key. Error code CX-404 means the referenced "
        "collection or record does not exist."
    ),
    "errors-server": (
        "Server errors indicate a problem on the Cirrus side. Error code CX-500 is a generic "
        "internal error and is safe to retry after a short delay. Error code CX-503 means a "
        "region is temporarily unavailable, typically during a failover, and requests should "
        "be retried against the same endpoint once the failover completes."
    ),
    "quotas": (
        "Every plan includes quotas on the number of collections, the total vector count, and "
        "the monthly query volume. Exceeding a storage quota blocks further upserts but leaves "
        "queries working. Quotas can be raised by upgrading the plan or by requesting an "
        "increase through support."
    ),
    "replication": (
        "Collections can be replicated across multiple nodes to improve availability and read "
        "throughput. Adding read replicas spreads query load, so a collection that is slow "
        "under heavy read traffic often responds faster once replicas are added. Writes still "
        "go through a single primary, so replication does not increase write throughput."
    ),
    "backups": (
        "Cirrus takes automatic daily snapshots of every collection and retains them for "
        "thirty days. A snapshot can be restored into a new collection without affecting the "
        "original. Restores run in the background and the new collection becomes queryable "
        "once the restore completes."
    ),
    "regions": (
        "Cirrus runs in several geographic regions, and a project's data resides in the "
        "region chosen at project creation. Placing the database in the same region as your "
        "application reduces query latency. Data is not automatically replicated across "
        "regions; cross-region replication must be configured explicitly."
    ),
    "monitoring": (
        "The dashboard reports query latency percentiles, request throughput, and index "
        "memory usage for each collection. Latency is shown as p50, p95, and p99 so that tail "
        "latency is visible. Alerts can be configured to fire when a metric crosses a "
        "threshold you specify."
    ),
    "troubleshoot-latency": (
        "If queries are slow, the most common causes are a low ef_search value, an "
        "under-replicated collection under heavy read load, or querying across regions. Raise "
        "ef_search first, then add read replicas if the collection is read-bound, and confirm "
        "that the client and the database are in the same region."
    ),
}


def build_realistic_corpus() -> tuple[dict[str, Document], list[Query]]:
    """Return ``(documents, queries)`` for the realistic Cirrus documentation corpus."""
    documents = {doc_id: Document(id=doc_id, text=text) for doc_id, text in _DOCS.items()}

    def one(needle: str, source_id: str) -> GoldAnswer:
        return GoldAnswer((EvidenceSet((span_in(documents[source_id], needle),)),))

    def q(qid: str, text: str, gold: GoldAnswer, kind: str) -> Query:
        return Query(id=qid, text=text, gold=gold, meta={"kind": kind})

    queries: list[Query] = [
        # --- rare-term: exact error codes that BM25 nails and dense tends to blur ---
        q("Q01", "what does error code CX-429 mean",
          one("error code CX-429", "rate-limits"), "rare-term"),
        q("Q02", "what is CX-503",
          one("Error code CX-503 means a region is temporarily unavailable, typically during "
              "a failover, and requests should be retried against the same endpoint once the "
              "failover completes.", "errors-server"), "rare-term"),
        q("Q03", "meaning of CX-401 authentication",
          one("Error code CX-401 means authentication failed, usually a missing or invalid "
              "API key.", "errors-client"), "rare-term"),
        # --- fragmentation / chunk-size: long procedures that split under small chunks ---
        q("Q04", "what is the full procedure to rotate an API key without downtime",
          one("Rotating an API key is a four-step procedure that avoids any downtime. First, "
              "create a second API key in the console alongside the existing one. Second, "
              "deploy the new key to all of your services and confirm that traffic is flowing "
              "using it. Third, monitor the old key's usage in the dashboard until it drops to "
              "zero requests. Fourth, once the old key is idle, revoke it permanently from the "
              "console.", "auth-rotation"), "fragmentation"),
        q("Q05", "how do I fix slow queries",
          one("If queries are slow, the most common causes are a low ef_search value, an "
              "under-replicated collection under heavy read load, or querying across regions. "
              "Raise ef_search first, then add read replicas if the collection is read-bound, "
              "and confirm that the client and the database are in the same region.",
              "troubleshoot-latency"), "fragmentation"),
        # --- paraphrase: subword overlap, few exact tokens (favors the dense branch) ---
        q("Q06", "how do I make the database faster for reads",
          one("Adding read replicas spreads query load, so a collection that is slow under "
              "heavy read traffic often responds faster once replicas are added.",
              "replication"), "paraphrase"),
        q("Q07", "which indexing option uses the least memory",
          one("For very large collections that must minimize memory, Cirrus offers an IVF "
              "index, which partitions vectors into clusters and searches only the closest "
              "clusters.", "index-ivf"), "paraphrase"),
        # --- distractors: many docs mention "index" / "recall"; only one answers ---
        q("Q08", "which parameter should I raise first when recall is too low",
          one("Increasing ef_search is the usual first step when recall is too low.",
              "index-params"), "distractor"),
        q("Q09", "what is the default distance metric",
          one("Cosine is the default and is appropriate for normalized embeddings.",
              "distance-metrics"), "distractor"),
        q("Q10", "what does top_k control in a search",
          one("A similarity search returns the top_k records nearest to the query vector, "
              "ordered by score.", "query-topk"), "distractor"),
        # --- multi-span: the answer needs two separate required spans ---
        q("Q11", "what two fields are required to create a collection",
          GoldAnswer((EvidenceSet((
              span_in(documents["collections"], "a unique collection name"),
              span_in(documents["collections"], "the vector dimension that every record in "
                      "the collection will use"),
          )),)), "multi-span"),
        # --- multi-alternative: answerable from either the API-key or the OAuth doc ---
        q("Q12", "how can I authenticate a request",
          GoldAnswer((
              EvidenceSet((span_in(documents["auth-api-keys"],
                          "The simplest way to authenticate a request is with an API key."),)),
              EvidenceSet((span_in(documents["auth-oauth"],
                          "you can authenticate a request using OAuth 2.0 instead of a static "
                          "key"),)),
          )), "multi-alternative"),
        # --- straightforward hits (should pass under most configs) ---
        q("Q13", "what is Cirrus",
          one("Cirrus is a hosted vector database for semantic search and "
              "retrieval-augmented generation.", "overview"), "easy"),
        q("Q14", "how do I add or replace vectors",
          one("Adding vectors is done with the upsert operation, which inserts new records or "
              "replaces existing ones with the same identifier.", "upsert"), "easy"),
        q("Q15", "how do I restrict a query to matching metadata",
          one("Queries can be constrained with a metadata filter so that only records "
              "matching a predicate are considered.", "filtering"), "easy"),
        q("Q16", "which environment variable sets the API key",
          one("the API key with CIRRUS_API_KEY", "config-env"), "rare-term"),
        q("Q17", "when should I use hybrid search",
          one("Hybrid search is most useful when queries contain rare exact terms, such as "
              "product codes or identifiers, that a dense embedding alone tends to overlook.",
              "hybrid-search"), "paraphrase"),
        q("Q18", "how long are backups kept",
          one("Cirrus takes automatic daily snapshots of every collection and retains them "
              "for thirty days.", "backups"), "easy"),
    ]
    return documents, queries


def dump_realistic_corpus_jsonl(directory: str | Path) -> tuple[Path, Path]:
    """Write the realistic corpus to ``docs.jsonl`` + ``queries.jsonl``. Returns the paths."""
    from retrieval_lab.gold import query_to_dict

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    documents, queries = build_realistic_corpus()

    docs_path = directory / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as fh:
        for doc in documents.values():
            fh.write(json.dumps({"id": doc.id, "text": doc.text}) + "\n")

    queries_path = directory / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as fh:
        for query in queries:
            fh.write(json.dumps(query_to_dict(query)) + "\n")

    return docs_path, queries_path
