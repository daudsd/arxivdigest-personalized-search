#!/usr/bin/env python3
"""Pool top-10 candidates from BM25, dense, and hybrid into a single set per query.

Standard TREC pooling: take top-10 from each ranker, deduplicate.
Result is up to 30 unique candidates per query (avg ~23 in practice).

Output: data/candidates/candidates_pooled.jsonl
        one line per query: {qid, qstr, hits: [{doc_id, score, source}]}

The score field is the max score across rankers for that doc.
The source field lists which rankers retrieved the doc (e.g. ["bm25", "dense"]).

Usage:
  python3 scripts/pool_candidates.py
"""
import json
from pathlib import Path

candidates_dir = Path(__file__).parent / "data" / "candidates"
TOP_K = 10


def load_candidates(path, top_k):
    entries = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            entries[e["qid"]] = {
                "qstr": e["qstr"],
                "hits": e["hits"][:top_k]
            }
    return entries


def main():
    bm25 = load_candidates(candidates_dir / "candidates_bm25.jsonl", TOP_K)
    dense = load_candidates(candidates_dir / "candidates_dense.jsonl", TOP_K)
    hybrid = load_candidates(candidates_dir / "candidates_hybrid.jsonl", TOP_K)

    all_qids = sorted(set(bm25) | set(dense) | set(hybrid))
    results = []
    pool_sizes = []

    for qid in all_qids:
        qstr = (bm25.get(qid) or dense.get(qid) or hybrid.get(qid))["qstr"]

        # Merge hits: track max score and which rankers retrieved each doc
        doc_info = {}
        for ranker, entries in [("bm25", bm25), ("dense", dense), ("hybrid", hybrid)]:
            for hit in entries.get(qid, {}).get("hits", []):
                did = hit["doc_id"]
                if did not in doc_info:
                    doc_info[did] = {"score": hit["score"], "sources": [ranker]}
                else:
                    doc_info[did]["score"] = max(doc_info[did]["score"], hit["score"])
                    doc_info[did]["sources"].append(ranker)

        # Sort by number of rankers retrieving the doc (desc), then by max score (desc)
        hits = sorted(
            [{"doc_id": did, "score": info["score"], "sources": info["sources"]}
             for did, info in doc_info.items()],
            key=lambda x: (len(x["sources"]), x["score"]),
            reverse=True
        )

        pool_sizes.append(len(hits))
        results.append({"qid": qid, "qstr": qstr, "hits": hits})

    out_path = candidates_dir / "candidates_pooled.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} pooled queries to {out_path}")
    print(f"Pool size: min={min(pool_sizes)}  max={max(pool_sizes)}  "
          f"avg={sum(pool_sizes)/len(pool_sizes):.1f}")

    # Source overlap stats
    only_one = sum(
        sum(1 for h in r["hits"] if len(h["sources"]) == 1)
        for r in results
    )
    in_all_three = sum(
        sum(1 for h in r["hits"] if len(h["sources"]) == 3)
        for r in results
    )
    total_docs = sum(len(r["hits"]) for r in results)
    print(f"Docs retrieved by only 1 ranker: {only_one} ({100*only_one/total_docs:.1f}%)")
    print(f"Docs retrieved by all 3 rankers: {in_all_three} ({100*in_all_three/total_docs:.1f}%)")


if __name__ == "__main__":
    main()
