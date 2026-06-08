import json
from pathlib import Path
from statistics import mean

from evalscope.benchmarks.pruning.statistics_selector import select_indices_from_statistics


EVALS_ROOT = Path(r"C:\Users\gudug\OneDrive\Desktop\Cerebras-task\Evals\Part 1")
MODELS = ["gpt-oss-120b", "kimi-k2.5", "minimax-m2.5"]


def read_scores(path: Path, metric: str):
    scores = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            idx = int(row["index"])
            val = row["sample_score"]["score"]["value"][metric]
            scores[idx] = float(val)
    return scores


def validate_benchmark(name: str, file_prefix: str, metric: str, stats_path: str, prune_ratio: float):
    keep = set(select_indices_from_statistics(stats_path, prune_ratio=prune_ratio))

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Pruned samples: {len(keep)}")

    full_means = []
    pruned_means = []

    for model in MODELS:
        path = EVALS_ROOT / "reviews" / f"{file_prefix}__{model}.jsonl"
        scores = read_scores(path, metric)

        full_score = mean(scores.values())
        pruned_score = mean(scores[i] for i in keep if i in scores)

        full_means.append(full_score)
        pruned_means.append(pruned_score)

        print(
            f"{model:15s} full={full_score:.4f} "
            f"pruned={pruned_score:.4f} "
            f"delta={pruned_score - full_score:+.4f}"
        )

    full_rank = sorted(zip(MODELS, full_means), key=lambda x: x[1], reverse=True)
    pruned_rank = sorted(zip(MODELS, pruned_means), key=lambda x: x[1], reverse=True)

    print("\nFull ranking:")
    print(" > ".join(m for m, _ in full_rank))

    print("Pruned ranking:")
    print(" > ".join(m for m, _ in pruned_rank))

    print("Rank preserved:", [m for m, _ in full_rank] == [m for m, _ in pruned_rank])


def main():
    validate_benchmark(
        name="LiveCodeBench v5",
        file_prefix="live_code_bench_v5",
        metric="pass",
        stats_path="analysis/lcb_statistics.jsonl",
        prune_ratio=0.20,
    )

    validate_benchmark(
        name="AA-LCR",
        file_prefix="aa_lcr",
        metric="acc",
        stats_path="analysis/aalcr_statistics.jsonl",
        prune_ratio=0.25,
    )


if __name__ == "__main__":
    main()
