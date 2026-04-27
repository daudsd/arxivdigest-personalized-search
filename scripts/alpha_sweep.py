#!/usr/bin/env python3
"""Alpha sweep: evaluate personalized reranking across different alpha values.

Re-runs reranking on existing candidates using different alpha values and
evaluates against existing qrels. No ES or LLM calls needed.

Usage:
  python3 scripts/alpha_sweep.py --base bm25
  python3 scripts/alpha_sweep.py --base bm25 dense hybrid
  python3 scripts/alpha_sweep.py --base bm25 dense hybrid --plot
  python3 scripts/alpha_sweep.py --base bm25 --alphas 0.1 0.3 0.5 0.7 0.9
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytrec_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))

data_dir = Path(__file__).parent / "data"
candidates_dir = data_dir / "candidates"
cache_dir = data_dir / "cache"
qrels_dir = data_dir / "qrels" / "llama3.3-70b"


def rerank(hits, doc_texts, user_keywords, alpha):
    doc_ids = [h["doc_id"] for h in hits if h["doc_id"] in doc_texts]
    if not doc_ids:
        return {h["doc_id"]: h["score"] for h in hits}

    corpus = [doc_texts[d] for d in doc_ids]
    user_text = " ".join(user_keywords)

    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus + [user_text])
    sims = cosine_similarity(tfidf[-1], tfidf[:-1])[0]

    bm25_scores = np.array([h["score"] for h in hits if h["doc_id"] in doc_texts])
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-9)

    combined = alpha * bm25_norm + (1 - alpha) * sims
    return {doc_id: float(score) for doc_id, score in zip(doc_ids, combined)}


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qid, _, doc_id, score = parts
                qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def evaluate(qrels, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "map", "recip_rank"})
    results = evaluator.evaluate(run)
    n = len(results)
    return {
        "ndcg@10": sum(r["ndcg_cut_10"] for r in results.values()) / n,
        "map":     sum(r["map"]          for r in results.values()) / n,
        "mrr":     sum(r["recip_rank"]   for r in results.values()) / n,
    }


def sweep(base, alphas):
    candidates_path = candidates_dir / f"candidates_personalized_{base}.jsonl"
    qrels_path = qrels_dir / f"qrels_personalized_{base}.txt"

    if not candidates_path.exists():
        print(f"Missing: {candidates_path}")
        return
    if not qrels_path.exists():
        print(f"Missing: {qrels_path}")
        return

    with open(candidates_path) as f:
        entries = [json.loads(l) for l in f]

    # Pre-build doc_texts per query from stored hits (title+abstract already in hits? No — rebuild from text)
    # hits only have doc_id and score, so we reconstruct doc_texts from the candidates file
    # The candidates file doesn't store text — we need to load it differently.
    # Instead, re-derive doc_texts from the stored keywords and scores by using the
    # original base candidates + stored text. Since we don't have text here, we use
    # the fact that rerank() only needs doc_texts. We store a lightweight version:
    # load doc texts from a pre-built cache if available, else skip text and use score-only.

    # Check if doc_text cache exists
    cache_path = cache_dir / f"doc_texts_{base}.json"
    if not cache_path.exists():
        print(f"Doc text cache not found at {cache_path}.")
        print("Run: python3 scripts/alpha_sweep.py --build-cache --base {base}")
        return

    with open(cache_path) as f:
        doc_texts = json.load(f)

    qrels = load_qrels(qrels_path)

    print(f"\n{'':>6}  {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}   (base={base})")
    print("-" * 46)

    results = []
    for alpha in alphas:
        run = {}
        for e in entries:
            qid_uid = f"{e['qid']}_u{e['uid']}"
            if qid_uid not in qrels:
                continue
            scores = rerank(e["hits"], doc_texts, e["keywords"], alpha)
            run[qid_uid] = scores

        run = {qid: docs for qid, docs in run.items() if qid in qrels}
        if not run:
            continue

        m = evaluate(qrels, run)
        m["alpha"] = alpha
        results.append(m)
        print(f"α={alpha:.1f}   {m['ndcg@10']:>10.4f} {m['map']:>10.4f} {m['mrr']:>10.4f}")

    return results


def plot_sweep(all_results, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"bm25": "#2196F3", "dense": "#FF5722", "hybrid": "#4CAF50"}
    labels = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}

    for base, results in all_results.items():
        if not results:
            continue
        alphas = [r["alpha"] for r in results]
        ndcg = [r["ndcg@10"] for r in results]
        ax.plot(alphas, ndcg, marker="o", color=colors[base], label=labels[base], linewidth=2)

    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel("nDCG@10", fontsize=12)
    ax.set_xticks([round(a * 0.1, 1) for a in range(11)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


def build_cache(base, es_url, index_name):
    """Fetch and cache doc texts from ES for all doc_ids in candidates file."""
    from elasticsearch import Elasticsearch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from arxivdigest.core.config import config_elasticsearch, elastic_index_name

    es = Elasticsearch(hosts=[es_url or config_elasticsearch.get("url", "http://localhost:9200")])
    index = index_name or elastic_index_name

    candidates_path = candidates_dir / f"candidates_personalized_{base}.jsonl"
    with open(candidates_path) as f:
        entries = [json.loads(l) for l in f]

    all_doc_ids = list({h["doc_id"] for e in entries for h in e["hits"]})
    print(f"Fetching {len(all_doc_ids)} unique docs from ES...")

    doc_texts = {}
    batch_size = 500
    for i in range(0, len(all_doc_ids), batch_size):
        batch = all_doc_ids[i:i + batch_size]
        resp = es.mget(index=index, body={"ids": batch})
        for d in resp["docs"]:
            if d.get("found"):
                doc_texts[d["_id"]] = f"{d['_source'].get('title','')} {d['_source'].get('abstract','')}"
        if (i // batch_size + 1) % 10 == 0:
            print(f"  {min(i+batch_size, len(all_doc_ids))}/{len(all_doc_ids)}")

    cache_path = cache_dir / f"doc_texts_{base}.json"
    with open(cache_path, "w") as f:
        json.dump(doc_texts, f)
    print(f"Cached {len(doc_texts)} docs to {cache_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", nargs="+", choices=["bm25", "dense", "hybrid"], default=["bm25"])
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[round(i * 0.1, 1) for i in range(11)])
    parser.add_argument("--plot", action="store_true", help="Save nDCG@10 vs alpha line plot")
    parser.add_argument("--build-cache", action="store_true", help="Fetch and cache doc texts from ES")
    parser.add_argument("--es-url", default=None)
    parser.add_argument("--index", default=None)
    args = parser.parse_args()

    if args.build_cache:
        for base in args.base:
            build_cache(base, args.es_url, args.index)
        return

    all_results = {}
    for base in args.base:
        all_results[base] = sweep(base, args.alphas)

    if args.plot:
        out_path = data_dir / "figures" / "alpha_sweep.pdf"
        plot_sweep(all_results, out_path)


if __name__ == "__main__":
    main()
