import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_FILES = {
    "BM25": "results/test_bm25.json",
    "Base MiniLM": "results/test_base_vs_hardneg.json",
    "Hard-neg MiniLM": "results/test_base_vs_hardneg.json",
    "Hybrid": "results/test_hybrid_dense125.json",
    "Hybrid + Reranker": "results/test_frozen_hybrid_reranker3.json",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    bm25 = load_json(RESULT_FILES["BM25"])
    dense = load_json(RESULT_FILES["Base MiniLM"])
    hybrid = load_json(RESULT_FILES["Hybrid"])
    reranker = load_json(RESULT_FILES["Hybrid + Reranker"])

    systems = [
        "BM25",
        "Base MiniLM",
        "Hard-neg MiniLM",
        "Hybrid",
        "Hybrid + Reranker",
    ]

    mrr = [
        bm25["metrics"]["MRR"],
        dense["base"]["metrics"]["MRR"],
        dense["fine_tuned"]["metrics"]["MRR"],
        hybrid["metrics"]["MRR"],
        reranker["metrics"]["MRR"],
    ]

    recall10 = [
        bm25["metrics"]["Recall@10"],
        dense["base"]["metrics"]["Recall@10"],
        dense["fine_tuned"]["metrics"]["Recall@10"],
        hybrid["metrics"]["Recall@10"],
        reranker["metrics"]["Recall@10"],
    ]

    x = range(len(systems))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar([i - width / 2 for i in x], mrr, width, label="MRR")
    ax.bar([i + width / 2 for i in x], recall10, width, label="Recall@10")

    ax.set_title("Frozen Test Performance on ToolRet Subset")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems, rotation=15, ha="right")
    ax.set_ylim(0, 0.7)
    ax.legend()

    for i, value in enumerate(mrr):
        ax.text(
            i - width / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for i, value in enumerate(recall10):
        ax.text(
            i + width / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = Path("assets/test_results.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved figure to {output_path}")


def plot_validation_ablation():
    bm25 = load_json("results/validation_bm25.json")
    dense = load_json("results/validation_base_vs_hardneg.json")
    hybrid = load_json("results/validation_hybrid_dense125.json")
    reranker = load_json("results/validation_toolret_reranker3.json")

    systems = [
        "BM25",
        "Base MiniLM",
        "Hard-neg MiniLM",
        "Hybrid",
        "Hybrid + Reranker",
    ]

    mrr = [
        bm25["metrics"]["MRR"],
        dense["base"]["metrics"]["MRR"],
        dense["fine_tuned"]["metrics"]["MRR"],
        hybrid["metrics"]["MRR"],
        reranker["metrics"]["MRR"],
    ]

    ndcg10 = [
        bm25["metrics"]["nDCG@10"],
        dense["base"]["metrics"]["nDCG@10"],
        dense["fine_tuned"]["metrics"]["nDCG@10"],
        hybrid["metrics"]["nDCG@10"],
        reranker["metrics"]["nDCG@10"],
    ]

    x = range(len(systems))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar([i - width / 2 for i in x], mrr, width, label="MRR")
    ax.bar([i + width / 2 for i in x], ndcg10, width, label="nDCG@10")

    ax.set_title("Validation Ablation: Effect of Each Retrieval Stage")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems, rotation=15, ha="right")
    ax.set_ylim(0, 0.8)
    ax.legend()

    for i, value in enumerate(mrr):
        ax.text(
            i - width / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for i, value in enumerate(ndcg10):
        ax.text(
            i + width / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = Path("assets/validation_ablation.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved figure to {output_path}")
if __name__ == "__main__":
    main()
    plot_validation_ablation()