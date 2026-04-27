#!/usr/bin/env python3
"""Personalized LLM judge: score (query, user, document) triples.

Uses a user-aware prompt that considers the user's research interests.
Output qrel key is qid_uid to keep per-user judgments separate.

Input:  data/candidates_personalized.jsonl
Output: data/qrels_personalized.txt  (TREC format: qid_uid 0 doc_id score)

Usage:
  python3 scripts/llm_judge_personalized.py \
    --api-base https://ollama.ux.uis.no \
    --api-key sk-... \
    --model llama3.3:70b \
    --resume
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent.parent))

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
        d["_id"]: {"title": d["_source"].get("title", ""), "abstract": d["_source"].get("abstract", "")}
        for d in resp["docs"] if d.get("found")
    }


def call_gemini(api_key, model, prompt, retries=8):
    import time
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


def call_nim(api_base, api_key, model, prompt, retries=5):
    import time
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            try:
                return max(0, min(3, int(text[0])))
            except (ValueError, IndexError):
                return 0
        except Exception:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def call_ollama(base_url, api_key, model, prompt, retries=5):
    import time
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
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--sim-method", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--backend", choices=["ollama", "nim", "gemini"], default="ollama")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="llama3.3:70b")
    parser.add_argument("--top-k", type=int, default=10, help="Docs to judge per pair")
    parser.add_argument("--index", default=None)
    parser.add_argument("--es-url", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from arxivdigest.core.config import config_elasticsearch, elastic_index_name
    es = Elasticsearch(hosts=[args.es_url or config_elasticsearch.get("url", "http://localhost:9200")])
    index = args.index or elastic_index_name

    data_dir = Path(__file__).parent / "data"
    suffix = "_emb" if args.sim_method == "embedding" else ""
    candidates_path = data_dir / "candidates" / f"candidates_personalized_{args.base}{suffix}.jsonl"
    qrels_path = data_dir / "qrels" / "llama3.3-70b" / f"qrels_personalized_{args.base}{suffix}.txt"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    with open(candidates_path) as f:
        entries = [json.loads(l) for l in f]

    done = set()
    if args.resume and qrels_path.exists():
        with open(qrels_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4:
                    done.add((parts[0], parts[2]))
        print(f"Resuming: {len(done)} pairs already scored")

    total = sum(min(len(e["hits"]), args.top_k) for e in entries)
    scored = 0

    with open(qrels_path, "a" if args.resume else "w") as out:
        for e in entries:
            qid_uid = f"{e['qid']}_u{e['uid']}"
            keywords_str = ", ".join(e["keywords"][:10])
            hits = e["hits"][:args.top_k]
            doc_ids = [h["doc_id"] for h in hits if (qid_uid, h["doc_id"]) not in done]
            if not doc_ids:
                continue

            docs = fetch_docs(es, index, doc_ids)
            for doc_id in doc_ids:
                if doc_id not in docs:
                    continue
                prompt = PROMPT.format(
                    keywords=keywords_str,
                    query=e["qstr"],
                    title=docs[doc_id]["title"],
                    abstract=docs[doc_id]["abstract"][:800]
                )
                if args.backend == "nim":
                    api_base = args.api_base or "https://integrate.api.nvidia.com/v1"
                    rel = call_nim(api_base, args.api_key, args.model, prompt)
                elif args.backend == "gemini":
                    rel = call_gemini(args.api_key, args.model, prompt)
                else:
                    api_base = args.api_base or "https://ollama.ux.uis.no"
                    rel = call_ollama(api_base, args.api_key, args.model, prompt)
                out.write(f"{qid_uid} 0 {doc_id} {rel}\n")
                out.flush()
                scored += 1
                if scored % 50 == 0:
                    print(f"  {scored}/{total} scored", flush=True)

    print(f"Done. Saved to {qrels_path}")


if __name__ == "__main__":
    main()
