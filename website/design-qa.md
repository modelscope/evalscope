# Design QA

## References and scope

- Approved visual direction: `/Users/yunlin/.codex/generated_images/01a084e3-705f-72e3-97d2-a1fc101393cc/exec-7378f081-96a9-430f-9950-d7fcdbe001bf.png`
- Local prototype: `http://127.0.0.1:4321/evalscope/`
- Browser inspection: Codex in-app browser, 2026-09-09, on Home, Evaluation, Agent & Harness, and Get Started.

## Checks

- Recompared the approved reference with the local Home capture: warm paper ground, ink display type, technical blue labels, gold measurement marks, registration line, fine ruler ticks, and dark evidence layer are present without restoring the removed decorative excess.
- Header and footer use the supplied `docs/en/_static/images/evalscope_icon.png` asset plus editable `EvalScope` text; no legacy wordmark image is displayed.
- Each evidence-led route has a distinct primary source: Evaluation uses the qwen-plus GSM8K result, Agent uses the real Details view, Performance uses Overview and Charts, Visualization uses those three distinct views, and pages without a genuine dashboard view use code or artifact evidence rather than a fabricated screenshot.
- Replaced the malformed inline terminal blocks with the reusable `CodeSnippet` component. It preserves command line breaks, labels the snippet, uses a macOS-style toolbar and exposes a working Copy control. The local browser click completed without navigation or console-visible failure.
- Browser review confirms the Agent loop, real evidence caption, drift-boundary labels, Evaluation modality/contract path, and four bounded Get Started prompts render in the intended information order.
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of 16 locale routes plus 404.
- Responsive breakpoints preserve the compact mobile menu and collapse dense card/timeline layouts at 960px, 900px, and 620px. The initial 390px/768px behavior is covered by the responsive layout rules; a later visual polish pass can tune only breakpoint spacing if needed.

final result: passed
