#!/usr/bin/env python3
"""Step 4: Evaluate retrieval using qrels from LLM judge.

Computes nDCG@10, MAP, MRR for each retrieval mode.

Usage:
  pip install pytrec-eval-terrier
  python3 scripts/evaluate.py --mode bm25
  python3 scripts/evaluate.py --mode bm25 hybrid   # compare multiple
"""
import argparse
import json
from pathlib import Path

data_dir = Path(__file__).parent / "data"
candidates_dir = data_dir / "candidates"
qrels_dir_default = data_dir / "qrels" / "llama3.3-70b"


def load_qrels(path):
    """Returns {qid: {doc_id: score}}"""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, doc_id, score = parts
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def load_run(path, personalized=False, expand_users=None):
    """Returns {qid: {doc_id: score}} from candidates jsonl.
    If expand_users is provided, expands each qid into qid_uid keys for all users."""
    run = {}
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            if personalized and "uid" in q:
                key = f"{q['qid']}_u{q['uid']}"
                run[key] = {h["doc_id"]: h["score"] for h in q["hits"]}
            elif expand_users:
                for uid in expand_users:
                    key = f"{q['qid']}_u{uid}"
                    run[key] = {h["doc_id"]: h["score"] for h in q["hits"]}
            else:
                run[q["qid"]] = {h["doc_id"]: h["score"] for h in q["hits"]}
    return run


def evaluate(qrels, run):
    import pytrec_eval
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut_10", "map", "recip_rank"}
    )
    results = evaluator.evaluate(run)
    metrics = {"ndcg@10": 0, "map": 0, "mrr": 0}
    n = len(results)
    for scores in results.values():
        metrics["ndcg@10"] += scores["ndcg_cut_10"]
        metrics["map"] += scores["map"]
        metrics["mrr"] += scores["recip_rank"]
    return {k: v / n for k, v in metrics.items()}


def load_filter(path):
    """Returns set of (uid, qid) pairs with score=2 from query_profile_scores.jsonl"""
    pairs = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["score"] == 2:
                pairs.add((str(r["uid"]), r["qid"]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", nargs="+", default=["bm25"],
                        help="One or more modes to evaluate (e.g. bm25 hybrid)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing qrels (default: scripts/data)")
    parser.add_argument("--filter-queries", default=None,
                        help="Path to query_profile_scores.jsonl; keep only score=2 pairs")
    args = parser.parse_args()

    filter_pairs = None
    if args.filter_queries:
        filter_pairs = load_filter(args.filter_queries)
        print(f"Filter: {len(filter_pairs)} (uid, qid) pairs with score=2")

    qrels_dir = Path(args.data_dir) if args.data_dir else qrels_dir_default

    print(f"{'Mode':<22} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 54)
    for mode in args.mode:
        # personalized_bm25 / personalized_dense / personalized_hybrid
        if mode.startswith("personalized_emb_"):
            base = mode.split("personalized_emb_", 1)[1]
            qrels_path = qrels_dir / f"qrels_personalized_{base}_emb.txt"
            candidates_path = candidates_dir / f"candidates_personalized_{base}_emb.jsonl"
            personalized = True
        elif mode.startswith("personalized_"):
            base = mode.split("_", 1)[1]
            qrels_path = qrels_dir / f"qrels_personalized_{base}.txt"
            candidates_path = candidates_dir / f"candidates_personalized_{base}.jsonl"
            personalized = True
        elif mode.startswith("baseline_useraware_"):
            base = mode.split("baseline_useraware_", 1)[1]
            qrels_path = qrels_dir / f"qrels_baseline_useraware_{base}.txt"
            candidates_path = candidates_dir / f"candidates_{base}.jsonl"
            personalized = True  # keys are qid_uid format
            # Get the 10 sampled user IDs from qrels file
            expand_users = set()
            if qrels_path.exists():
                with open(qrels_path) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 4:
                            uid = parts[0].split("_u")[-1]
                            expand_users.add(uid)
        else:
            qrels_path = qrels_dir / f"qrels_{mode}.txt"
            candidates_path = candidates_dir / f"candidates_{mode}.jsonl"
            personalized = False

        if not qrels_path.exists():
            print(f"{mode:<20}  qrels not found: {qrels_path}")
            continue
        if not candidates_path.exists():
            print(f"{mode:<20}  candidates not found: {candidates_path}")
            continue

        qrels = load_qrels(qrels_path)
        if mode.startswith("baseline_useraware_"):
            run = load_run(candidates_path, expand_users=expand_users)
        else:
            run = load_run(candidates_path, personalized=personalized)

        # Apply query-profile filter if provided
        if filter_pairs and personalized:
            def matches_filter(qid_uid):
                # key format: "citeseerx-q1_u2"
                uid = qid_uid.split("_u")[-1]
                qid = qid_uid[:qid_uid.rfind("_u")]
                return (uid, qid) in filter_pairs
            run = {k: v for k, v in run.items() if matches_filter(k)}
            print(f"  {mode}: {len(run)} pairs after filtering")
        elif filter_pairs and not personalized:
            filter_qids = {qid for uid, qid in filter_pairs}
            run = {k: v for k, v in run.items() if k in filter_qids}
            print(f"  {mode}: {len(run)} queries after filtering")

        # Only evaluate queries that have qrels
        run = {qid: docs for qid, docs in run.items() if qid in qrels}

        metrics = evaluate(qrels, run)
        print(f"{mode:<22} {metrics['ndcg@10']:>10.4f} {metrics['map']:>10.4f} {metrics['mrr']:>10.4f}")


if __name__ == "__main__":
    main()
