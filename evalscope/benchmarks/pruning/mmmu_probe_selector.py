import json
from collections import defaultdict


def select_mmmu_probe(
    statistics_path: str,
    probe_size: int = 50,
):
    with open(statistics_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    selected = []
    selected_ids = set()

    buckets = defaultdict(list)

    for row in rows:
        key = (
            row["subject"],
            row["img_type_primary"],
            row["question_type"],
        )

        buckets[key].append(row)

    # coverage-first selection
    for bucket_rows in buckets.values():
        bucket_rows = sorted(
            bucket_rows,
            key=lambda x: (
                -x["encoder_stress_score"],
                x["index"],
            ),
        )

        candidate = bucket_rows[0]

        selected.append(candidate)
        selected_ids.add(candidate["id"])

    remaining = [
        r
        for r in rows
        if r["id"] not in selected_ids
    ]

    remaining = sorted(
        remaining,
        key=lambda x: (
            -x["encoder_stress_score"],
            x["index"],
        ),
    )

    for row in remaining:
        if len(selected) >= probe_size:
            break

        selected.append(row)

    return [r["id"] for r in selected[:probe_size]]


if __name__ == "__main__":
    ids = select_mmmu_probe(
        "analysis/mmmu_statistics.jsonl",
        probe_size=50,
    )

    print("Selected:", len(ids))
