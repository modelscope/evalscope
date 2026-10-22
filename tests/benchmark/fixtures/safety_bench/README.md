`official_eva.json` contains the unmodified MIT-licensed official `eva.py` source from
[thu-coai/SafetyBench@960833692a1ec82b191ff3e902daca7993c35cc1](https://github.com/thu-coai/SafetyBench/blob/960833692a1ec82b191ff3e902daca7993c35cc1/eva.py).
It is stored as a JSON string so formatting tools cannot change the independent scoring oracle.
The upstream license is retained in `LICENSE`. Tests generate synthetic question/answer fixtures locally.

`api_smoke_2026-10-22.json` records a manual run against the real DashScope `qwen-plus` API.
Each configuration selected the first two original records per category (14 questions each, 42 total).
It retains original IDs, gold/predicted 0-based options, extraction and termination states, and official scores.
All request prompts and the deterministic extraction stage matched the upstream generation example;
the official `eva.py` matched per-sample correctness, category scores, and overall scores. Report values
are rounded to four decimal places. The example's random fallback was deliberately not applied.

41 responses ended normally and one reached the 1,024-token output budget. Six replies were unparseable,
including the truncated reply; each scored zero without reducing the denominator. No transport errors occurred.
These are integration smoke results, not leaderboard scores. The automated suite never calls the API.

The `report_regression_validation` entry records the subsequent report fixes: all 24,970 configuration
rows retain their count and overall scores through serialization, and another six real API requests
have matching run, report, reloaded-report, and Web-list sample counts.
