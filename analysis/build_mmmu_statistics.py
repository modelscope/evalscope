import json
from collections import Counter
from pathlib import Path
from typing import List


ROOT = Path(r"C:\Users\gudug\OneDrive\Desktop\Cerebras-task\Evals\MMMU\reviews\glm-4.5v-fp8")
OUTPUT = Path("analysis/mmmu_statistics.jsonl")


ENCODER_STRESS_TYPES = {
    "Tables",
    "Charts",
    "Diagrams",
    "Scientific Figures",
    "Maps",
    "Medical Images",
    "Chemical Structures",
    "Geometry Diagrams",
    "Sheet Music",
}


def normalize_img_type(raw: str) -> List[str]:
    if not raw:
        return ["unknown"]

    cleaned = (
        raw.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    values = [x.strip() for x in cleaned.split(",") if x.strip()]
    return values or ["unknown"]


def encoder_stress_score(img_types: List[str], difficulty: str, question_type: str) -> float:
    score = 0.0

    for img_type in img_types:
        if img_type in ENCODER_STRESS_TYPES:
            score += 2.0
        else:
            score += 1.0

    if difficulty.lower() == "hard":
        score += 1.0
    elif difficulty.lower() == "medium":
        score += 0.5

    if question_type == "open":
        score += 0.5

    if len(img_types) > 1:
        score += 0.5

    return score


def main() -> None:
    rows = []

    for path in sorted(ROOT.glob("*.jsonl")):
        subject = path.stem.replace("mmmu_", "")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                idx = int(row["index"])
                sample_score = row["sample_score"]
                metadata = sample_score.get("sample_metadata", {})

                img_types = normalize_img_type(metadata.get("img_type", ""))
                difficulty = metadata.get("topic_difficulty", "unknown")
                question_type = metadata.get("question_type", "unknown")
                subfield = metadata.get("subfield", "unknown")
                acc = float(sample_score["score"]["value"].get("acc", 0.0))

                rows.append(
                    {
                        "benchmark": "mmmu",
                        "subject": subject,
                        "index": idx,
                        "id": metadata.get("id", ""),
                        "img_types": img_types,
                        "img_type_primary": img_types[0],
                        "question_type": question_type,
                        "subfield": subfield,
                        "topic_difficulty": difficulty,
                        "acc": acc,
                        "encoder_stress_score": encoder_stress_score(
                            img_types, difficulty, question_type
                        ),
                    }
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT}")

    print("\nSubjects:")
    for k, v in Counter(r["subject"] for r in rows).most_common():
        print(f"  {k}: {v}")

    print("\nImage types:")
    for k, v in Counter(r["img_type_primary"] for r in rows).most_common():
        print(f"  {k}: {v}")

    print("\nDifficulty:")
    for k, v in Counter(r["topic_difficulty"] for r in rows).most_common():
        print(f"  {k}: {v}")

    print("\nQuestion type:")
    for k, v in Counter(r["question_type"] for r in rows).most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
