#!/usr/bin/env python3
"""Step 3: LLM-as-judge relevance scoring (UMBRELA-style).

Scores each (query, document) pair 0-3:
  0 = not relevant
  1 = marginally relevant
  2 = relevant
  3 = highly relevant

Output: data/qrels_{mode}.txt  (TREC qrels format: qid 0 doc_id score)

Usage:
  # Free local model (default):
  python3 scripts/llm_judge.py --mode bm25

  # Specific HF model:
  python3 scripts/llm_judge.py --mode bm25 --model Qwen/Qwen2.5-0.5B-Instruct

  # OpenAI:
  OPENAI_API_KEY=sk-... python3 scripts/llm_judge.py --mode bm25 --backend openai --model gpt-4o-mini

  # NVIDIA NIM:
  python3 scripts/llm_judge.py --mode bm25 --backend nim --model qwen/qwen3.5-122b-a10b --api-key nvapi-...
"""
import argparse
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent.parent))

PROMPT = """Rate the relevance of the document to the query on a scale of 0 to 3.
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


def make_local_scorer(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    def score(query, title, abstract):
        messages = [{"role": "user", "content": PROMPT.format(
            query=query, title=title, abstract=abstract[:800])}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=3, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        result = tok.decode(new_tokens, skip_special_tokens=True).strip()
        try:
            return max(0, min(3, int(result[0])))
        except (ValueError, IndexError):
            return 0

    return score


def make_gemini_scorer(model_name, api_key, delay=0.5):
    import urllib.request
    import urllib.error
    import time

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }

    def score(query, title, abstract, retries=8):
        import json
        payload = json.dumps({
            "contents": [{"parts": [{"text": PROMPT.format(
                query=query, title=title, abstract=abstract[:800])}]}]
        }).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                time.sleep(delay)
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                try:
                    return max(0, min(3, int(text[0])))
                except (ValueError, IndexError):
                    return 0
            except urllib.error.HTTPError as e:
                wait = 60 if e.code == 429 else 5 * (attempt + 1)
                if attempt < retries - 1:
                    print(f"  HTTP {e.code}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except Exception:
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

    return score


def make_nim_scorer(model_name, api_key, api_base="https://api.us-west-2.modal.direct/v1", delay=1.0):
    import urllib.request
    import time

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    def score(query, title, abstract, retries=8):
        import json
        import urllib.error
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": PROMPT.format(
                query=query, title=title, abstract=abstract[:800])}],
            "max_tokens": 10,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                time.sleep(delay)
                text = data["choices"][0]["message"]["content"].strip()
                try:
                    return max(0, min(3, int(text[0])))
                except (ValueError, IndexError):
                    return 0
            except urllib.error.HTTPError as e:
                wait = 60 if e.code == 429 else 5 * (attempt + 1)
                if attempt < retries - 1:
                    print(f"  HTTP {e.code}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except Exception:
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

    return score


def make_openai_scorer(model_name, api_base=None, api_key=None):
    import urllib.request
    import urllib.error
    import time

    base = (api_base or "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/api/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def score(query, title, abstract, retries=5):
        import json
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": PROMPT.format(
                query=query, title=title, abstract=abstract[:800])}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 3}
        }).encode()
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

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="bm25")
    parser.add_argument("--backend", choices=["local", "openai", "nim", "gemini"], default="local")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--es-url", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between NIM requests (default: 1.0)")
    parser.add_argument("--model-dir", default=None,
                        help="Override qrels subfolder name (default: derived from --model)")
    args = parser.parse_args()

    from arxivdigest.core.config import config_elasticsearch, elastic_index_name
    es = Elasticsearch(hosts=[args.es_url or config_elasticsearch.get("url", "http://localhost:9200")])
    index = args.index or elastic_index_name

    if args.backend == "local":
        score_fn = make_local_scorer(args.model)
    elif args.backend == "nim":
        score_fn = make_nim_scorer(args.model, args.api_key, args.api_base or "https://api.us-west-2.modal.direct/v1", delay=args.delay)
    elif args.backend == "gemini":
        score_fn = make_gemini_scorer(args.model, args.api_key, delay=args.delay)
    else:
        score_fn = make_openai_scorer(args.model, args.api_base, args.api_key)

    model_dir = args.model_dir or args.model.replace("/", "_").replace(":", "_")
    data_dir = Path(__file__).parent / "data" / "qrels" / model_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = Path(__file__).parent / "data" / "candidates" / f"candidates_{args.mode}.jsonl"
    qrels_path = data_dir / f"qrels_{args.mode}.txt"

    with open(candidates_path) as f:
        queries = [json.loads(l) for l in f]

    done = set()
    if args.resume and qrels_path.exists():
        with open(qrels_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4:
                    done.add((parts[0], parts[2]))
        print(f"Resuming: {len(done)} pairs already scored")

    total = sum(len(q["hits"]) for q in queries)
    scored = 0

    with open(qrels_path, "a" if args.resume else "w") as out:
        for q in queries:
            qid, qstr = q["qid"], q["qstr"]
            doc_ids = [h["doc_id"] for h in q["hits"] if (qid, h["doc_id"]) not in done]
            if not doc_ids:
                continue

            docs = fetch_docs(es, index, doc_ids)
            for doc_id in doc_ids:
                if doc_id not in docs:
                    continue
                rel = score_fn(qstr, docs[doc_id]["title"], docs[doc_id]["abstract"])
                out.write(f"{qid} 0 {doc_id} {rel}\n")
                out.flush()
                scored += 1
                if scored % 50 == 0:
                    print(f"  {scored}/{total} scored")

    print(f"Done. Saved to {qrels_path}")


if __name__ == "__main__":
    main()
