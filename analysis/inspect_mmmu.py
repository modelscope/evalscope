import json
from pathlib import Path
from pprint import pprint

ROOT = Path(r"C:\Users\gudug\OneDrive\Desktop\Cerebras-task\Evals\MMMU")

FILES = [
    ROOT / "predictions" / "glm-4.5v-fp8" / "mmmu_Accounting.jsonl",
    ROOT / "reviews" / "glm-4.5v-fp8" / "mmmu_Accounting.jsonl",
]

for path in FILES:
    print("\n" + "=" * 100)
    print(path)

    with path.open("r", encoding="utf-8") as f:
        row = json.loads(next(f))

    print("\nTOP LEVEL KEYS")
    pprint(list(row.keys()))

    if "metadata" in row:
        print("\nMETADATA KEYS")
        pprint(list(row["metadata"].keys()))
        print("\nMETADATA SAMPLE")
        pprint(row["metadata"])

    if "sample_score" in row:
        print("\nSAMPLE SCORE")
        pprint(row["sample_score"])
