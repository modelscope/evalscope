# Design QA

## References and scope

- Approved desktop direction: `/Users/yunlin/.codex/generated_images/01a084e3-705f-72e3-97d2-a1fc101393cc/exec-7378f081-96a9-430f-9950-d7fcdbe001bf.png`
- Implemented reference route: `http://127.0.0.1:4321/evalscope/`
- Browser inspection: Codex in-app browser, 2026-09-09.

## Checks

- The implemented home hero preserves the warm-paper background, oversized ink headline, blue technical label, gold-and-blue measurement line, dark primary CTA, and real dashboard evidence in the first view.
- The real dashboard image, not generated UI, is legible at desktop width and includes its provenance caption.
- The navigation, language links and CTA retain the `/evalscope` base path; benchmark query parameters produce filtered results. The catalog showed 2 GSM8K results for `q=gsm8k&category=LLM` and 46 LLM + Reasoning results after interaction.
- The responsive CSS reduces the header to a mobile menu below 960px, collapses cards to a single column below 900px, and reduces catalog columns below 620px. Static checks cover all 16 locale routes plus 404.
- No P1/P2 visual regression was found relative to the approved hierarchy. The deliberate content reduction removes duplicate screenshot panels and decorative density, rather than restoring omitted mockup sections.

final result: passed
