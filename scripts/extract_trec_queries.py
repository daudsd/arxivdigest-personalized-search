#!/usr/bin/env python3
"""Step 1: Extract CiteSeerX queries from TREC OpenSearch data.

Downloads trecos.tar.gz, parses CiteSeerX queries.json (NDJSON),
and saves a clean queries.csv with columns: qid, qstr
"""
import csv
import io
import json
import tarfile
import urllib.request
from pathlib import Path

URL = "https://github.com/living-labs/trec-os-data/raw/master/trecos.tar.gz"
OUT = Path(__file__).parent / "data" / "queries" / "trec_citeseerx_queries.csv"


def main():
    print("Downloading trecos.tar.gz...")
    with urllib.request.urlopen(URL) as resp:
        data = resp.read()

    queries = {}  # qid -> qstr (deduplicate across years)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if "citeseerx/queries.json" in member.name:
                f = tar.extractfile(member)
                for line in f:
                    line = line.strip()
                    if line:
                        q = json.loads(line)
                        queries[q["qid"]] = q["qstr"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "qstr"])
        for qid, qstr in sorted(queries.items()):
            writer.writerow([qid, qstr])

    print(f"Saved {len(queries)} queries to {OUT}")


if __name__ == "__main__":
    main()
