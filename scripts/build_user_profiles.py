#!/usr/bin/env python3
"""Build user profiles from user_topics in the DB.

Output: data/user_profiles.json  — {user_id: [keyword, ...]}
Only includes users with >= min_topics topics.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from arxivdigest.core.config import config_sql
    import mysql.connector

    conn = mysql.connector.connect(**config_sql)
    cur = conn.cursor()
    cur.execute("""
        SELECT ut.user_id, t.topic
        FROM user_topics ut
        JOIN topics t ON ut.topic_id = t.topic_id
        ORDER BY ut.user_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    profiles = {}
    for user_id, topic in rows:
        profiles.setdefault(user_id, []).append(topic)

    raw_profiles = profiles

    # Keep only users with >= 3 topics, filter to short clean keywords (<=4 words)
    profiles = {}
    for uid, topics in raw_profiles.items():
        clean = [t for t in topics if len(t.split()) <= 4]
        if len(clean) >= 3:
            profiles[str(uid)] = clean[:20]  # cap at 20 keywords per user

    out = Path(__file__).parent / "data" / "profiles" / "user_profiles.json"
    with open(out, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"Saved {len(profiles)} user profiles to {out}")
    # Show sample
    sample_id = next(iter(profiles))
    print(f"Sample user {sample_id}: {profiles[sample_id]}")


if __name__ == "__main__":
    main()
