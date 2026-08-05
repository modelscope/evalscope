# Copyright (c) Alibaba, Inc. and its affiliates.
"""Simulated API classes vendored from the official ACEBench repository.

Source: https://github.com/ACEBench/ACEBench, ``model_inference/multi_step/scenarios{en,zh}``,
redistributed under the upstream MIT license (see the ``LICENSE`` file next to this module).
The modules under ``en/`` and ``zh/`` are reproduced verbatim, with only the ``BaseApi`` import
rewritten to be relative. ACEBench is not published as a package, and its agent categories are
graded on the state these classes end up in, so they have to be executed as-is for the reported
scores to mean the same thing as the official ones.

Because they are unmodified upstream sources, they are excluded from this repository's formatters
(see ``.pre-commit-config.yaml``) so they stay diffable against upstream.

The classes are plain in-memory state machines: they touch no files, network or subprocesses, which
is why the rollout executes them in-process instead of inside a code sandbox. Model output never
reaches ``eval``; see :mod:`evalscope.benchmarks.acebench.rollout` for the dispatch and
:mod:`evalscope.benchmarks.acebench.scenarios` for instantiation and state capture.
"""
