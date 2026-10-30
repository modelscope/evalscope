# Copyright (c) Alibaba, Inc. and its affiliates.
"""Simulated tools, task rubrics and fixtures vendored from the official AgentHarm sources.

Source: UKGovernmentBEIS/inspect_evals, commit ``d1adf47db09ade66824c1b8468b90be6fb5161c9``,
directory ``src/inspect_evals/agentharm``, redistributed under the upstream MIT license with its
additional AI safety/security use clause (see the ``LICENSE`` file next to this module). Full
provenance, the file mapping and every intentional deviation are recorded in ``SOURCES.md``.

AgentHarm is not published as a standalone package: its official code ships only inside the
inspect_evals monorepo, whose declaration layer imports ``inspect_ai`` at module level and which
requires Python >= 3.11, so it cannot be consumed the way :mod:`evalscope.benchmarks.agentdojo`
consumes its pinned upstream distribution. These sources are therefore a pinned snapshot rather
than a verbatim mirror: the Inspect decorators and types were removed, each grader's semantic
model call became a yielded :class:`~evalscope.third_party.agentharm.types.SemanticCheck` so the
native JudgeExecutor owns all judge I/O, and file-reading tools resolve bundled package resources
instead of the working directory. Because the sources are adapted, a textual diff against upstream
is not clean; fidelity is verified by AST-normalised comparison instead.

The modules here depend only on ``evalscope.api`` and never import ``evalscope.benchmarks``.
``manifest.json`` whitelists exactly the tools, rubrics and fixtures the released public and
validation tasks reference, so an unknown name cannot import an arbitrary module;
``data_manifest.json`` pins the official data files by sha256. The tools are plain in-memory
simulations: they send no messages, run no commands and contact no external services, which is why
the rollout executes them in-process instead of inside a code sandbox. See
:mod:`evalscope.benchmarks.agentharm.agentharm_adapter` for the EvalScope-side glue.
"""
