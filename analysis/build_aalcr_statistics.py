import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


ROOT = Path(r"C:\Users\gudug\OneDrive\Desktop\Cerebras-task\Evals\Part 1")

MODELS = ["gpt-oss-120b", "kimi-k2.5", "minimax-m2.5"]

REVIEWS = {
    model: ROOT / "reviews" / f"aa_lcr__{model}.jsonl"
    for model in MODELS
}

OUTPUT = Path("analysis/aalcr_statistics.jsonl")


def read_aalcr_scores(path: Path) -> Dict[int, dict]:
    rows = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index = row["index"]
            sample_score = row["sample_score"]

            acc = float(sample_score["score"]["value"]["acc"])
            metadata = sample_score.get("sample_metadata", {})

            rows[index] = {
                "acc": acc,
                "input_tokens": int(metadata.get("input_tokens", 0) or 0),
                "question": metadata.get("question", ""),
                "data_source_urls": metadata.get("data_source_urls", ""),
            }

    return rows


def context_bucket(tokens: int) -> str:
    if tokens < 32_000:
        return "ctx_lt_32k"
    if tokens < 64_000:
        return "ctx_32k_64k"
    if tokens < 96_000:
        return "ctx_64k_96k"
    return "ctx_96k_plus"


def difficulty_bucket(difficulty: float) -> str:
    if difficulty == 1.0:
        return "easy_all_correct"
    if difficulty >= 2 / 3:
        return "medium_two_correct"
    if difficulty >= 1 / 3:
        return "hard_one_correct"
    return "very_hard_none_correct"


def main() -> None:
    model_rows: Dict[str, Dict[int, dict]] = {
        model: read_aalcr_scores(path)
        for model, path in REVIEWS.items()
    }

    all_indices = sorted(set().union(*[rows.keys() for rows in model_rows.values()]))

    output_rows: List[dict] = []

    for index in all_indices:
        accs = [model_rows[model].get(index, {}).get("acc", 0.0) for model in MODELS]

        first = next(
            model_rows[model][index]
            for model in MODELS
            if index in model_rows[model]
        )

        difficulty = mean(accs)
        disagreement = pstdev(accs)
        tokens = first["input_tokens"]

        output_rows.append(
            {
                "benchmark": "aa_lcr",
                "index": index,
                "scores": dict(zip(MODELS, accs)),
                "difficulty": difficulty,
                "difficulty_bucket": difficulty_bucket(difficulty),
                "disagreement": disagreement,
                "is_disagreement_sample": disagreement > 0,
                "input_tokens": tokens,
                "context_bucket": context_bucket(tokens),
                "question": first["question"],
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")

    print("\nDifficulty buckets:")
    diff_counts = {}
    for row in output_rows:
        diff_counts[row["difficulty_bucket"]] = diff_counts.get(row["difficulty_bucket"], 0) + 1
    for bucket, count in sorted(diff_counts.items()):
        print(f"  {bucket}: {count}")

    print("\nContext buckets:")
    ctx_counts = {}
    for row in output_rows:
        ctx_counts[row["context_bucket"]] = ctx_counts.get(row["context_bucket"], 0) + 1
    for bucket, count in sorted(ctx_counts.items()):
        print(f"  {bucket}: {count}")

    disagreement_count = sum(1 for row in output_rows if row["is_disagreement_sample"])
    print(f"\nDisagreement samples: {disagreement_count}/{len(output_rows)}")


if __name__ == "__main__":
    main()