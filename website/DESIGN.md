# EvalScope website design system

## Intent

The site makes EvalScope's measurement surface legible in under three minutes: first a concrete promise, then the core mechanism, one real evidence item, and one next action. It is a separate GitHub Pages surface, not the Python package, dashboard, or documentation site.

## Tokens

- Paper `#F7F5EF`; ink `#101820`; rule `#D8D5CC`
- Measurement gold `#DFA72D`; evidence blue `#2F5CFF`; exploring violet `#7668FF`
- Inter Variable for reading; IBM Plex Mono fallback stack for measurements and terminal snippets
- Gold measurement lines, blue evidence nodes and small technical labels are deliberate accents. Decoration density is reduced from the design exploration so content remains primary.

## Page structure

Every route uses four to five sections: promise, mechanism, real evidence, and one CTA. The eight English routes have `/zh/` mirrors. The catalog reads the current branch's `evalscope/benchmarks/_meta/*.json` at build time and stays intentionally separate from product code.

## Claims and evidence

- Shipped capabilities use `Available Today`.
- Research concepts use `Exploring`.
- Papers use `External Research`.
- Screenshots are all labelled `Real API demo snapshot · 2026-09-09`.
- Evaluation snapshot: qwen-plus, GSM8K, 5 samples, Accuracy 100%, 5.188s, 36.85 tok/s.
- Agent snapshot: 1 sample, 5 model requests, 3/3 tool calls, 6213 tokens, final answer 18.
- Performance snapshot: 24 requests, 1536 output tokens, 20.24s, Best RPS 2.08, lowest latency 1.713s, concurrency 4 throughput 133.22 tok/s and 100% success.
- These are small real API demos, not broad model or vendor performance claims.

## Motion and accessibility

Only links/buttons, first-view movement and evidence-path emphasis are animated. `prefers-reduced-motion` disables motion. High contrast, semantic headings, visible focus, a skip link and responsive one-column layouts are required.
