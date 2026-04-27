#!/usr/bin/env python3
"""Score (user, query) pairs by how likely a user with a given profile would issue them.

Scale:
  0 = unlikely   (query unrelated to user's interests)
  1 = possible    (some overlap)
  2 = very likely (query directly relevant to user's interests)

Input:  data/candidates/candidates_personalized_bm25.jsonl
        (contains uid, keywords, qid, qstr — same users/queries across all bases)
Output: data/query_profile_scores.jsonl
        one line per (uid, qid): {"uid": ..., "qid": ..., "qstr": ..., "score": 0|1|2}

Usage:
  python3 scripts/score_query_profile.py \
    --api-base https://ollama.ux.uis.no --api-key <key> --model llama3.3:70b --resume
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

data_dir = Path(__file__).parent / "data"
candidates_dir = data_dir / "candidates"

PROMPT = """Given a user with research interests in: {keywords}

How likely is it that this user would issue the following search query?
0 = unlikely (the query is unrelated to their interests)
1 = possible (some overlap with their interests)
2 = very likely (the query is directly relevant to their interests)

Query: {query}

Reply with a single digit (0, 1, or 2):"""


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
                return max(0, min(2, int(text[0])))
            except (ValueError, IndexError):
                return 0
        except Exception:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="https://ollama.ux.uis.no")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="llama3.3:70b")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load unique (uid, qid) pairs from bm25 personalized candidates
    # (same users and queries across all bases — no need to repeat)
    candidates_path = candidates_dir / "candidates_personalized_bm25.jsonl"
    with open(candidates_path) as f:
        entries = [json.loads(l) for l in f]

    # Deduplicate to one entry per (uid, qid)
    seen = set()
    pairs = []
    for e in entries:
        key = (e["uid"], e["qid"])
        if key not in seen:
            seen.add(key)
            pairs.append({"uid": e["uid"], "qid": e["qid"],
                          "qstr": e["qstr"], "keywords": e["keywords"]})

    out_path = data_dir / "query_profile_scores.jsonl"

    done = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["uid"], r["qid"]))
        print(f"Resuming: {len(done)} pairs already scored")

    pairs = [p for p in pairs if (p["uid"], p["qid"]) not in done]
    total = len(pairs) + len(done)
    scored = len(done)

    print(f"Total pairs: {total} | Remaining: {len(pairs)}")

    with open(out_path, "a" if args.resume else "w") as out:
        for p in pairs:
            keywords_str = ", ".join(p["keywords"][:10])
            prompt = PROMPT.format(keywords=keywords_str, query=p["qstr"])
            score = call_ollama(args.api_base, args.api_key, args.model, prompt)
            out.write(json.dumps({
                "uid": p["uid"], "qid": p["qid"],
                "qstr": p["qstr"], "score": score
            }) + "\n")
            out.flush()
            scored += 1
            if scored % 100 == 0:
                print(f"  {scored}/{total} scored")

    print(f"Done. Saved to {out_path}")

    # Print summary statistics
    with open(out_path) as f:
        results = [json.loads(l) for l in f]

    from collections import Counter, defaultdict
    dist = Counter(r["score"] for r in results)
    print(f"\nScore distribution: {dict(sorted(dist.items()))}")
    print(f"Score=2 (very likely): {dist[2]} pairs "
          f"({100*dist[2]/len(results):.1f}%)")

    # Per-user stats for score=2
    per_user = defaultdict(int)
    for r in results:
        if r["score"] == 2:
            per_user[r["uid"]] += 1

    if per_user:
        counts = list(per_user.values())
        print(f"\nQueries per user (score=2):")
        print(f"  min={min(counts)}  max={max(counts)}  "
              f"avg={sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    main()
