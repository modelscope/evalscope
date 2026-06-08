import argparse
import json
from pathlib import Path
from statistics import mean


def find_jsonl_files(root: Path):
    return list(root.rglob("*.jsonl"))


def extract_score_value(sample_score: dict) -> float | None:
    score = sample_score.get("score", {})
    value = score.get("value", {})

    for key in ("acc", "pass"):
        if key in value:
            return float(value[key])

    return None


def load_scores(result_dir: Path):
    scores = []

    for path in find_jsonl_files(result_dir):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sample_score = row.get("sample_score")
                if not isinstance(sample_score, dict):
                    continue

                value = extract_score_value(sample_score)
                if value is not None:
                    scores.append(value)

    return scores


def summarize(scores):
    if not scores:
        return {
            "n": 0,
            "mean": None,
        }

    return {
        "n": len(scores),
        "mean": mean(scores),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare full and pruned EvalScope run directories."
    )
    parser.add_argument("--full", required=True, help="Path to full EvalScope results directory.")
    parser.add_argument("--pruned", required=True, help="Path to pruned EvalScope results directory.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional go/no-go threshold for mean score.",
    )

    args = parser.parse_args()

    full_dir = Path(args.full)
    pruned_dir = Path(args.pruned)

    full_scores = load_scores(full_dir)
    pruned_scores = load_scores(pruned_dir)

    full_summary = summarize(full_scores)
    pruned_summary = summarize(pruned_scores)

    print("\nEvalScope run comparison")
    print("=" * 80)

    print(f"Full run:   n={full_summary['n']}, mean={full_summary['mean']}")
    print(f"Pruned run: n={pruned_summary['n']}, mean={pruned_summary['mean']}")

    if full_summary["mean"] is not None and pruned_summary["mean"] is not None:
        delta = pruned_summary["mean"] - full_summary["mean"]
        abs_delta = abs(delta)

        print(f"Delta:      {delta:+.4f}")
        print(f"Abs delta:  {abs_delta:.4f}")

    if args.threshold is not None:
        print("\nGo / no-go threshold")
        print("-" * 80)

        full_go = (
            full_summary["mean"] is not None
            and full_summary["mean"] >= args.threshold
        )
        pruned_go = (
            pruned_summary["mean"] is not None
            and pruned_summary["mean"] >= args.threshold
        )

        print(f"Threshold:  {args.threshold:.4f}")
        print(f"Full:       {'GO' if full_go else 'NO-GO'}")
        print(f"Pruned:     {'GO' if pruned_go else 'NO-GO'}")
        print(f"Agreement:  {'YES' if full_go == pruned_go else 'NO'}")


if __name__ == "__main__":
    main()
