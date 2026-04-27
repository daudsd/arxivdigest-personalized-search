#!/usr/bin/env python3
"""Step 2: Retrieve top-k candidate articles from Elasticsearch for each TREC query.

Supports three retrieval modes:
  --mode bm25    : BM25 text search only
  --mode dense   : kNN dense vector search only (requires hybrid index with abstract_vector)
  --mode hybrid  : BM25 + dense vector search (requires hybrid index with abstract_vector)

Output: data/candidates.jsonl  — one line per query:
  {"qid": "...", "qstr": "...", "hits": [{"doc_id": "...", "score": ...}, ...]}
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent.parent))


def bm25_query(qstr, top_k):
    return {
        "size": top_k,
        "_source": ["title", "abstract"],
        "query": {
            "multi_match": {
                "query": qstr,
                "fields": ["title^2", "abstract"],
                "type": "best_fields"
            }
        }
    }


def dense_query(vector, top_k):
    return {
        "size": top_k,
        "_source": ["title", "abstract"],
        "query": {
            "knn": {
                "field": "abstract_vector",
                "query_vector": vector,
                "num_candidates": top_k * 5
            }
        }
    }


def hybrid_query(qstr, vector, top_k):
    return {
        "size": top_k,
        "_source": ["title", "abstract"],
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": qstr,
                            "fields": ["title^2", "abstract"],
                            "boost": 0.5
                        }
                    },
                    {
                        "knn": {
                            "field": "abstract_vector",
                            "query_vector": vector,
                            "num_candidates": top_k * 5,
                            "boost": 0.5
                        }
                    }
                ]
            }
        }
    }


def get_query_vector(qstr):
    from scripts.index_articles_hybrid import get_embedding_model
    model = get_embedding_model()
    return model.encode([qstr])[0].tolist()


def retrieve(es, index, queries, mode, top_k):
    results = []
    for i, (qid, qstr) in enumerate(queries):
        if mode in ("hybrid", "dense"):
            vector = get_query_vector(qstr)
            body = hybrid_query(qstr, vector, top_k) if mode == "hybrid" else dense_query(vector, top_k)
        else:
            body = bm25_query(qstr, top_k)

        resp = es.search(index=index, body=body)
        hits = [
            {"doc_id": h["_id"], "score": h["_score"]}
            for h in resp["hits"]["hits"]
        ]
        results.append({"qid": qid, "qstr": qstr, "hits": hits})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(queries)} queries done")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--index", default=None)
    parser.add_argument("--es-url", default=None)
    parser.add_argument("--queries", default=None, help="Path to queries CSV (default: data/trec_citeseerx_queries.csv)")
    args = parser.parse_args()

    from arxivdigest.core.config import config_elasticsearch, elastic_index_name

    es_url = args.es_url or config_elasticsearch.get("url", "http://localhost:9200")
    index = args.index or elastic_index_name

    queries_path = args.queries or Path(__file__).parent / "data" / "queries" / "trec_citeseerx_queries.csv"
    with open(queries_path) as f:
        queries = [(row["qid"], row["qstr"]) for row in csv.DictReader(f)]

    print(f"Loaded {len(queries)} queries")
    print(f"Retrieving top-{args.top_k} from index '{index}' using {args.mode}...")

    es = Elasticsearch(hosts=[es_url])
    results = retrieve(es, index, queries, args.mode, args.top_k)

    out_path = Path(__file__).parent / "data" / "candidates" / f"candidates_{args.mode}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
