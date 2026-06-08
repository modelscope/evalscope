import json
from pathlib import Path
from pprint import pprint

ROOT = Path(r"C:\Users\gudug\OneDrive\Desktop\Cerebras-task\Evals")

FILES = [
    ROOT / "Part 1" / "predictions" / "live_code_bench_v5__gpt-oss-120b.jsonl",
    ROOT / "Part 1" / "reviews" / "live_code_bench_v5__gpt-oss-120b.jsonl",
    ROOT / "Part 1" / "predictions" / "aa_lcr__gpt-oss-120b.jsonl",
    ROOT / "Part 1" / "reviews" / "aa_lcr__gpt-oss-120b.jsonl",
]

for path in FILES:
    print("\n" + "=" * 100)
    print(path)

    with open(path, "r", encoding="utf-8") as f:
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

    if "model_output" in row:
        print("\nMODEL OUTPUT KEYS")
        pprint(list(row["model_output"].keys()))