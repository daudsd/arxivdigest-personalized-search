#!/usr/bin/env python3
"""Personalized retrieval: rerank candidates using user keyword overlap.

For each (query, user) pair, combines retrieval score with profile similarity.
Two similarity methods are supported:
  tfidf    : TF-IDF cosine similarity between doc text and user keywords
  embedding: cosine similarity between doc embedding and user profile embedding

Input:
  data/candidates_{base}.jsonl — base retrieval candidates (bm25/dense/hybrid)
  data/user_profiles.json      — {user_id: [keyword, ...]}

Output:
  data/candidates_personalized_{base}.jsonl          (tfidf, default)
  data/candidates_personalized_{base}_emb.jsonl      (embedding)

Usage:
  python3 scripts/retrieve_candidates_personalized.py --base bm25
  python3 scripts/retrieve_candidates_personalized.py --base bm25 --sim-method embedding
"""
import argparse
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

data_dir = Path(__file__).parent / "data"
candidates_dir = data_dir / "candidates"
profiles_dir = data_dir / "profiles"


def select_users(profiles, n):
    """Pick n users with most diverse/representative profiles."""
    # Sort by number of unique keywords, take every k-th for diversity
    sorted_users = sorted(profiles.items(), key=lambda x: len(x[1]), reverse=True)
    step = max(1, len(sorted_users) // n)
    return dict(sorted_users[::step][:n])


def fetch_docs(es, index, doc_ids):
    resp = es.mget(index=index, body={"ids": doc_ids})
    return {
        d["_id"]: {
            "text": f"{d['_source'].get('title','')} {d['_source'].get('abstract','')}",
            "vector": d["_source"].get("abstract_vector", None)
        }
        for d in resp["docs"] if d.get("found")
    }


def rerank_tfidf(hits, docs, user_keywords, alpha=0.5):
    """Rerank using TF-IDF cosine similarity between doc text and user keywords."""
    doc_ids = [h["doc_id"] for h in hits if h["doc_id"] in docs]
    if not doc_ids:
        return hits

    corpus = [docs[d]["text"] for d in doc_ids]
    user_text = " ".join(user_keywords)

    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus + [user_text])
    sims = cosine_similarity(tfidf[-1], tfidf[:-1])[0]

    bm25_scores = np.array([h["score"] for h in hits if h["doc_id"] in docs])
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-9)

    combined = alpha * bm25_norm + (1 - alpha) * sims
    ranked = sorted(zip(doc_ids, combined), key=lambda x: x[1], reverse=True)
    return [{"doc_id": d, "score": float(s)} for d, s in ranked]


def rerank_embedding(hits, docs, user_keywords, profile_vector, alpha=0.5):
    """Rerank using cosine similarity between doc embedding and user profile embedding."""
    doc_ids = [h["doc_id"] for h in hits
               if h["doc_id"] in docs and docs[h["doc_id"]]["vector"] is not None]
    if not doc_ids:
        return hits

    doc_vecs = np.array([docs[d]["vector"] for d in doc_ids])
    profile_vec = np.array(profile_vector).reshape(1, -1)

    # Cosine similarity
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
    profile_norm = np.linalg.norm(profile_vec) + 1e-9
    sims = (doc_vecs / doc_norms) @ (profile_vec / profile_norm).T
    sims = sims.flatten()

    bm25_scores = np.array([h["score"] for h in hits if h["doc_id"] in docs
                            and docs[h["doc_id"]]["vector"] is not None])
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-9)

    combined = alpha * bm25_norm + (1 - alpha) * sims
    ranked = sorted(zip(doc_ids, combined), key=lambda x: x[1], reverse=True)
    return [{"doc_id": d, "score": float(s)} for d, s in ranked]


def rerank(hits, docs, user_keywords, alpha=0.5, sim_method="tfidf", profile_vector=None):
    if sim_method == "embedding" and profile_vector is not None:
        return rerank_embedding(hits, docs, user_keywords, profile_vector, alpha)
    return rerank_tfidf(hits, docs, user_keywords, alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--sim-method", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--n-users", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--index", default=None)
    parser.add_argument("--es-url", default=None)
    args = parser.parse_args()

    from arxivdigest.core.config import config_elasticsearch, elastic_index_name
    es = Elasticsearch(hosts=[args.es_url or config_elasticsearch.get("url", "http://localhost:9200")])
    index = args.index or elastic_index_name

    # Load embedding model if needed
    embed_model = None
    if args.sim_method == "embedding":
        from sentence_transformers import SentenceTransformer
        print("Loading all-MiniLM-L6-v2 for profile encoding...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(candidates_dir / f"candidates_{args.base}.jsonl") as f:
        queries = [json.loads(l) for l in f]

    with open(profiles_dir / "user_profiles.json") as f:
        all_profiles = json.load(f)

    users = select_users(all_profiles, args.n_users)
    print(f"Selected {len(users)} users")
    print(f"Sim method: {args.sim_method}")
    print(f"Processing {len(queries)} queries x {len(users)} users = {len(queries)*len(users)} pairs")

    # Pre-compute profile vectors for embedding mode
    profile_vectors = {}
    if embed_model is not None:
        print("Encoding user profiles...")
        for uid, keywords in users.items():
            profile_text = " ".join(keywords)
            profile_vectors[uid] = embed_model.encode(profile_text).tolist()
        print("Done.")

    results = []
    for i, q in enumerate(queries):
        doc_ids = [h["doc_id"] for h in q["hits"]]
        if not doc_ids:
            continue
        docs = fetch_docs(es, index, doc_ids)

        for uid, keywords in users.items():
            reranked = rerank(
                q["hits"], docs, keywords,
                alpha=args.alpha,
                sim_method=args.sim_method,
                profile_vector=profile_vectors.get(uid)
            )
            results.append({
                "qid": q["qid"],
                "uid": uid,
                "qstr": q["qstr"],
                "keywords": keywords,
                "hits": reranked[:args.top_k]
            })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} queries done")

    suffix = "_emb" if args.sim_method == "embedding" else ""
    out = candidates_dir / f"candidates_personalized_{args.base}{suffix}.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {out}")


if __name__ == "__main__":
    main()
