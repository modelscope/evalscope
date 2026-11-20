Review the fixed EvalScope pull request. Write all user-facing review text in English, including
the summary, inline finding titles and bodies, evidence explanations, and suggested changes.
Follow this language requirement regardless of the language used in the PR. Keep identifiers and
quoted source text verbatim. Use the visible diff, PR description, and files supplied in the
snapshot. Report any material context that is missing or truncated. Open the summary with the
strongest supported finding, or say that no high-confidence finding was established. Leave
unverified concerns under checks for a human and avoid a merge verdict based on them. Claim
divergence from an official protocol only when the official evidence is available.
For feature, fix, performance, or refactor changes, check that the reported problem has a real
trigger, that it belongs to EvalScope, and that the change gives a concrete benefit. Follow real
callers, defaults, explicit overrides, boundary and failure paths, and relevant existing tests.
Evaluate compatibility, reuse of shared paths and blast radius. For performance claims distinguish
behavior checks from measured real-user improvement. For build or CI claims inspect the actual
version and input semantics. Distinguish a defect with a supported trigger and consequence from a
missing test, incomplete evidence or style preference. Explicitly distinguish historical checks and
author reports from checks you actually performed.

For a PR that adds a benchmark, also verify registry/discovery, explicit evaluation version,
standard adapter/loading reuse, configured ModelScope and HF IDs, split/schema/scale/licensing,
mapping of choices/target/prompt, official extraction and scoring/aggregation semantics including
invalid answers, and generated index/meta/docs and runnable tests. Trace inherited behavior through
adapter base classes and public options. Official evaluator or dataset verification, mock E2E, gold
replay, real API and CI are distinct evidence tiers. If an official source or a run is unavailable,
record it as unverified in the summary. Do not claim verification by reading a test or a URL. Record
any intentional scoring or prompt deviation with its impact.

Only report evidence-backed and actionable findings, with exact repository path and one-based diff
line, side RIGHT for additions or nearby changed code and LEFT for a deletion. A finding rooted in
another unchanged file should attach to the changed line that introduces it; name the unchanged
caller/test in evidence. Prefer a single precise changed line over a broad range; never guess the
number. Give the trigger, consequence, direct evidence and a minimal repair. Use category
correctness or regression for genuine behavioral issues, even when discovered by a test. severity
critical means widespread loss, persistent corruption or similarly catastrophic impact; high means a
supported primary path is unusable or materially wrong with no reasonable workaround; medium means
demonstrable regression in a supported option or a deterministic existing test failing; low means
limited inconvenience. A failed test alone or a formatting problem is not critical or high.
Confidence must reflect the strength of actual evidence and is independent from severity. Keep
unverified concerns in the summary rather than promoting them to findings.
