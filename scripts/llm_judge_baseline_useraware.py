#!/usr/bin/env python3
"""Run user-aware LLM judge on non-personalized candidates for the 10 sampled users.

For each (query, user, doc) triple, scores relevance using the user-aware prompt
(same prompt as llm_judge_personalized.py). This allows fair comparison between
non-personalized and personalized systems under the same judge.

Input:  data/candidates/{base}.jsonl          (top-20 per query)
        data/profiles/user_profiles.json       (671 user profiles)
Output: data/qrels/llama3.3-70b/qrels_baseline_useraware_{base}.txt
        TREC format: qid_uid 0 doc_id score

Usage:
  python3 scripts/llm_judge_baseline_useraware.py --base bm25 \\
    --api-base https://ollama.ux.uis.no --api-key sk-... --resume
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent.parent))

data_dir = Path(__file__).parent / "data"

PROMPT = """Given a user with research interests in: {keywords}

Rate the relevance of the document to their query on a scale of 0 to 3.
0 = not relevant
1 = marginally relevant
2 = relevant
3 = highly relevant

Query: {query}
Title: {title}
Abstract: {abstract}

Reply with a single digit (0, 1, 2, or 3):"""


def fetch_docs(es, index, doc_ids):
    resp = es.mget(index=index, body={"ids": doc_ids})
    return {
        d["_id"]: {"title": d["_source"].get("title", ""),
                   "abstract": d["_source"].get("abstract", "")}
        for d in resp["docs"] if d.get("found")
    }


def call_gemini(api_key, model, prompt, retries=8):
    import urllib.error
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 5, "temperature": 0}
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            try:
                return max(0, min(3, int(text[0])))
            except (ValueError, IndexError):
                return 0
        except Exception as e:
            wait = 60 if hasattr(e, 'code') and e.code == 429 else 5 * (attempt + 1)
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def call_ollama(base_url, api_key, model, prompt, retries=5):
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 3}
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["message"]["content"].strip()
            try:
                return max(0, min(3, int(text[0])))
            except (ValueError, IndexError):
                return 0
        except Exception:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def select_users(profiles, n=10):
    sorted_users = sorted(profiles.items(), key=lambda x: len(x[1]), reverse=True)
    step = max(1, len(sorted_users) // n)
    return dict(sorted_users[::step][:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--backend", choices=["ollama", "gemini"], default="ollama")
    parser.add_argument("--api-base", default="https://ollama.ux.uis.no")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="llama3.3:70b")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index", default=None)
    parser.add_argument("--es-url", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from arxivdigest.core.config import config_elasticsearch, elastic_index_name
    es = Elasticsearch(hosts=[args.es_url or config_elasticsearch.get("url", "http://localhost:9200")])
    index = args.index or elastic_index_name

    # Load non-personalized candidates
    candidates_path = data_dir / "candidates" / f"candidates_{args.base}.jsonl"
    with open(candidates_path) as f:
        queries = [json.loads(l) for l in f]

    # Load same 10 users as personalized evaluation
    with open(data_dir / "profiles" / "user_profiles.json") as f:
        all_profiles = json.load(f)
    users = select_users(all_profiles, 10)
    print(f"Users: {list(users.keys())}")

    qrels_path = data_dir / "qrels" / "llama3.3-70b" / f"qrels_baseline_useraware_{args.base}.txt"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and qrels_path.exists():
        with open(qrels_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4:
                    done.add((parts[0], parts[2]))
        print(f"Resuming: {len(done)} pairs already scored")

    total = len(queries) * len(users) * args.top_k
    scored = len(done)
    print(f"Total triples: ~{total} | Remaining: ~{total - scored}")

    with open(qrels_path, "a" if args.resume else "w") as out:
        for i, q in enumerate(queries):
            qid, qstr = q["qid"], q["qstr"]
            doc_ids = [h["doc_id"] for h in q["hits"][:args.top_k]]
            if not doc_ids:
                continue

            docs = fetch_docs(es, index, doc_ids)

            for uid, keywords in users.items():
                qid_uid = f"{qid}_u{uid}"
                keywords_str = ", ".join(keywords[:10])

                for doc_id in doc_ids:
                    if (qid_uid, doc_id) in done:
                        continue
                    if doc_id not in docs:
                        continue

                    prompt = PROMPT.format(
                        keywords=keywords_str,
                        query=qstr,
                        title=docs[doc_id]["title"],
                        abstract=docs[doc_id]["abstract"][:800]
                    )
                    rel = call_gemini(args.api_key, args.model, prompt) if args.backend == "gemini" else call_ollama(args.api_base, args.api_key, args.model, prompt)
                    out.write(f"{qid_uid} 0 {doc_id} {rel}\n")
                    out.flush()
                    scored += 1
                    if scored % 50 == 0:
                        print(f"  {scored}/{total} scored", flush=True)

    print(f"Done. Saved to {qrels_path}")


if __name__ == "__main__":
    main()
