import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


ROOT = Path(r"C:\Users\gudug\OneDrive\Desktop\Cerebras-task\Evals\Part 1")

MODELS = ["gpt-oss-120b", "kimi-k2.5", "minimax-m2.5"]

REVIEWS = {
    model: ROOT / "reviews" / f"live_code_bench_v5__{model}.jsonl"
    for model in MODELS
}

OUTPUT = Path("analysis/lcb_statistics.jsonl")


def read_lcb_pass_scores(path: Path) -> Dict[int, float]:
    scores = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index = row["index"]
            value = row["sample_score"]["score"]["value"]["pass"]
            scores[index] = float(value)

    return scores


def difficulty_bucket(difficulty: float) -> str:
    if difficulty == 1.0:
        return "easy_all_pass"
    if difficulty >= 2 / 3:
        return "medium_two_pass"
    if difficulty >= 1 / 3:
        return "hard_one_pass"
    return "very_hard_none_pass"


def main() -> None:
    model_scores: Dict[str, Dict[int, float]] = {
        model: read_lcb_pass_scores(path)
        for model, path in REVIEWS.items()
    }

    all_indices = sorted(set().union(*[scores.keys() for scores in model_scores.values()]))

    rows: List[dict] = []

    for index in all_indices:
        passes = [model_scores[model].get(index, 0.0) for model in MODELS]

        difficulty = mean(passes)
        disagreement = pstdev(passes)

        rows.append(
            {
                "benchmark": "live_code_bench_v5",
                "index": index,
                "scores": dict(zip(MODELS, passes)),
                "difficulty": difficulty,
                "difficulty_bucket": difficulty_bucket(difficulty),
                "disagreement": disagreement,
                "is_disagreement_sample": disagreement > 0,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT}")

    print("\nDifficulty buckets:")
    counts = {}
    for row in rows:
        counts[row["difficulty_bucket"]] = counts.get(row["difficulty_bucket"], 0) + 1
    for bucket, count in sorted(counts.items()):
        print(f"  {bucket}: {count}")

    disagreement_count = sum(1 for row in rows if row["is_disagreement_sample"])
    print(f"\nDisagreement samples: {disagreement_count}/{len(rows)}")


if __name__ == "__main__":
    main()