#!/usr/bin/env python3
"""Generate all thesis figures."""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

figures_dir = Path("/Users/daud/Desktop/university/DAT620/MSc_2026_Daud_Sadiq/figures")
data_dir = Path(__file__).parent / "data"


def fig_score_distribution():
    import pytrec_eval
    from collections import Counter
    from pathlib import Path

    qrels_dir = data_dir / "qrels" / "llama3.3-70b"
    systems = [
        ("BM25",         "qrels_bm25.txt"),
        ("Dense",        "qrels_dense.txt"),
        ("Hybrid",       "qrels_hybrid.txt"),
        ("Pers. BM25",   "qrels_personalized_bm25.txt"),
        ("Pers. Dense",  "qrels_personalized_dense.txt"),
        ("Pers. Hybrid", "qrels_personalized_hybrid.txt"),
    ]

    colors_nonpers = ["#2196F3", "#FF5722", "#4CAF50"]
    colors_pers    = ["#1565C0", "#BF360C", "#1B5E20"]
    score_labels   = ["0", "1", "2", "3"]
    x = np.arange(4)
    w = 0.13

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, fname) in enumerate(systems):
        path = qrels_dir / fname
        if not path.exists():
            continue
        scores = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4:
                    scores.append(int(parts[3]))
        c = Counter(scores)
        total = len(scores)
        vals = [100 * c[s] / total for s in [0, 1, 2, 3]]
        color = colors_nonpers[i] if i < 3 else colors_pers[i - 3]
        offset = (i - 2.5) * w
        ax.bar(x + offset, vals, w, label=name, color=color, alpha=0.85)

    ax.set_xlabel("Relevance score")
    ax.set_ylabel("Proportion of judgments (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(score_labels)
    ax.legend(fontsize=8, ncol=2)
    ax.axvline(2.5, color="#aaa", linestyle="--", linewidth=0.8)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, "Non-personalized", ha="center", fontsize=8, color="#555")
    ax.text(3.0, ax.get_ylim()[1] * 0.95, "Personalized",     ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = figures_dir / "score_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_judgment_illustration():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, color, fontsize=8.5):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    box(0.2, 2.1, 1.2, 0.8, "Query", "#555")
    box(2.0, 3.6, 1.5, 0.65, "BM25", "#2196F3")
    box(2.0, 2.2, 1.5, 0.65, "Dense", "#FF5722")
    box(2.0, 0.8, 1.5, 0.65, "Hybrid", "#4CAF50")
    arrow(1.4, 2.5, 2.0, 3.93)
    arrow(1.4, 2.5, 2.0, 2.53)
    arrow(1.4, 2.5, 2.0, 1.13)
    box(4.1, 2.2, 1.4, 0.65, "Candidates", "#9C27B0")
    arrow(3.5, 3.93, 4.1, 2.53)
    arrow(3.5, 2.53, 4.1, 2.53)
    arrow(3.5, 1.13, 4.1, 2.53)
    box(6.2, 3.6, 1.6, 0.65, "Topical Judge", "#607D8B")
    arrow(5.5, 2.53, 6.2, 3.93, color="#607D8B")
    box(8.3, 3.6, 1.4, 0.65, "Non-pers.\nResults", "#607D8B")
    arrow(7.8, 3.93, 8.3, 3.93)
    box(5.5, 0.8, 1.4, 0.65, "User Profile", "#FF9800")
    box(6.2, 1.8, 1.6, 0.65, "Personalized\nReranking", "#FF9800")
    arrow(5.5, 2.53, 6.2, 2.13, color="#FF9800")
    arrow(6.2, 1.13, 6.2, 1.8, color="#FF9800")
    box(8.3, 1.8, 1.4, 0.65, "User-aware\nJudge", "#E91E63")
    arrow(7.8, 2.13, 8.3, 2.13)
    ax.text(7.05, 4.45, "Topical qrels", ha="center", fontsize=8, color="#607D8B", style="italic")
    ax.text(7.05, 1.35, "User-aware qrels", ha="center", fontsize=8, color="#E91E63", style="italic")
    fig.tight_layout()
    out = figures_dir / "judgment_illustration.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_system_overview():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, color="#4A90D9", fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    box(0.2, 2.1, 1.4, 0.8, "Query", "#555")
    box(2.0, 3.5, 1.6, 0.7, "BM25", "#2196F3", 8)
    box(2.0, 2.15, 1.6, 0.7, "Dense\n(kNN)", "#FF5722", 8)
    box(2.0, 0.8, 1.6, 0.7, "Hybrid\n(BM25+kNN)", "#4CAF50", 8)
    box(4.3, 2.15, 1.4, 0.7, "Candidates\n(top-20)", "#9C27B0", 8)
    box(4.3, 3.5, 1.4, 0.7, "User\nProfile", "#FF9800", 8)
    box(6.3, 2.15, 1.5, 0.7, "Personalized\nReranking", "#607D8B", 8)
    box(8.3, 2.15, 1.4, 0.7, "Ranked\nResults", "#555", 8)
    arrow(1.6, 2.5, 2.0, 3.85)
    arrow(1.6, 2.5, 2.0, 2.5)
    arrow(1.6, 2.5, 2.0, 1.15)
    arrow(3.6, 3.85, 4.3, 2.5)
    arrow(3.6, 2.5, 4.3, 2.5)
    arrow(3.6, 1.15, 4.3, 2.5)
    arrow(5.7, 2.5, 6.3, 2.5)
    arrow(5.7, 3.85, 6.3, 2.7)
    arrow(7.8, 2.5, 8.3, 2.5)
    ax.text(2.8, 4.35, "Retrieval", ha="center", fontsize=8, color="#555", style="italic")
    ax.text(5.0, 1.5, "alpha controls\nblend", ha="center", fontsize=7, color="#555", style="italic")
    fig.tight_layout()
    out = figures_dir / "system_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_query_lengths():
    with open(data_dir / "queries" / "trec_citeseerx_queries.csv") as f:
        queries = list(csv.DictReader(f))
    lengths = [len(q["qstr"].split()) for q in queries]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lengths, bins=range(1, max(lengths) + 2), color="#2196F3",
            edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Query length (words)")
    ax.set_ylabel("Number of queries")
    ax.axvline(np.mean(lengths), color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(lengths):.1f}")
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, min(max(lengths)+1, 16)))
    fig.tight_layout()
    out = figures_dir / "query_length_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_profile_sizes():
    with open(data_dir / "profiles" / "user_profiles.json") as f:
        profiles = json.load(f)
    sizes = sorted([len(v) for v in profiles.values()])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(sizes, bins=range(1, 22), color="#4CAF50", edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Number of keywords per user profile")
    ax.set_ylabel("Number of users")
    ax.axvline(np.mean(sizes), color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(sizes):.1f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = figures_dir / "profile_size_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_full_vs_filtered():
    # All systems: user-aware judge, full query set vs filtered (score=2) subset
    systems  = ["BM25", "Dense", "Hybrid",
                "Pers.\nBM25\n(TF-IDF)", "Pers.\nDense\n(TF-IDF)", "Pers.\nHybrid\n(TF-IDF)",
                "Pers.\nBM25\n(Emb)",    "Pers.\nDense\n(Emb)",    "Pers.\nHybrid\n(Emb)"]
    full     = [0.4396, 0.4589, 0.4276, 0.6102, 0.5760, 0.5991, 0.4875, 0.4712, 0.4821]
    filtered = [0.8625, 0.9279, 0.8343, 0.9626, 0.9680, 0.9616, 0.9113, 0.9294, 0.9140]

    x = np.arange(len(systems))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - w/2, full,     w, label="Full query set (user-aware judge)",  color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, filtered, w, label="Filtered subset (user-aware judge)", color="#FF5722", alpha=0.85)
    ax.set_ylabel("nDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.axvline(2.5, color="#aaa", linestyle="--", linewidth=0.8)
    ax.axvline(5.5, color="#aaa", linestyle="--", linewidth=0.8)
    ax.text(1.0, 1.04, "Baseline",     ha="center", fontsize=8, color="#555")
    ax.text(4.0, 1.04, "TF-IDF Pers.", ha="center", fontsize=8, color="#555")
    ax.text(7.0, 1.04, "Emb. Pers.",   ha="center", fontsize=8, color="#555")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = figures_dir / "full_vs_filtered.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_tfidf_vs_emb():
    # TF-IDF vs Embedding similarity under user-aware judge, full query set
    bases    = ["BM25 base", "Dense base", "Hybrid base"]
    baseline = [0.4396, 0.4589, 0.4276]
    tfidf    = [0.6102, 0.5760, 0.5991]
    emb      = [0.4875, 0.4712, 0.4821]

    x = np.arange(len(bases))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w,  baseline, w, label="No reranking (baseline)", color="#9E9E9E", alpha=0.85)
    ax.bar(x,      tfidf,    w, label="TF-IDF similarity",       color="#2196F3", alpha=0.85)
    ax.bar(x + w,  emb,      w, label="Embedding similarity",    color="#FF5722", alpha=0.85)
    ax.set_ylabel("nDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels(bases)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = figures_dir / "tfidf_vs_emb.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_per_user():
    users        = ["u2",   "u102",  "u230",  "u326",  "u362",  "u403",  "u596",  "u617",  "u636",  "u766"]
    bm25_scores  = [0.9502, 0.9676,  0.9711,  0.9519,  0.9081,  0.9509,  0.9614,  0.9996,  0.9152,  0.9415]
    dense_scores = [0.9629, 0.9705,  0.9707,  0.9643,  0.9836,  0.9926,  0.9633,  0.9772,  0.9804,  0.9532]
    n_queries    = [39,     90,      114,     10,      3,       14,      125,     8,       5,       18]

    x = np.arange(len(users))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, bm25_scores,  w, label="Pers. BM25 (TF-IDF)",  color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, dense_scores, w, label="Pers. Dense (TF-IDF)", color="#FF5722", alpha=0.85)
    ax.set_ylabel("nDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{u}\n(n={n})" for u, n in zip(users, n_queries)], fontsize=8)
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=9)
    ax.set_xlabel("User (n = queries in filtered subset)")
    fig.tight_layout()
    out = figures_dir / "per_user_ndcg.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig_alpha_sweep():
    alphas      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bm25_ndcg   = [0.6754, 0.6491, 0.6282, 0.6201, 0.6163, 0.6140, 0.6128, 0.6119, 0.6112, 0.6106, 0.6102]
    dense_ndcg  = [0.6278, 0.6016, 0.5875, 0.5825, 0.5801, 0.5786, 0.5776, 0.5771, 0.5767, 0.5763, 0.5760]
    hybrid_ndcg = [0.6672, 0.6365, 0.6159, 0.6085, 0.6049, 0.6029, 0.6015, 0.6007, 0.6001, 0.5997, 0.5991]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, bm25_ndcg,   marker="o", color="#2196F3", label="BM25",   linewidth=2)
    ax.plot(alphas, dense_ndcg,  marker="o", color="#FF5722", label="Dense",  linewidth=2)
    ax.plot(alphas, hybrid_ndcg, marker="o", color="#4CAF50", label="Hybrid", linewidth=2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("nDCG@10")
    ax.set_xticks(alphas)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "alpha_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    figures_dir.mkdir(exist_ok=True)
    fig_score_distribution()
    fig_judgment_illustration()
    fig_system_overview()
    fig_query_lengths()
    fig_profile_sizes()
    fig_full_vs_filtered()
    fig_tfidf_vs_emb()
    fig_per_user()
    fig_alpha_sweep()
    print("All figures generated.")
