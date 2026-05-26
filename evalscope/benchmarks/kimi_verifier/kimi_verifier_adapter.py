# Copyright (c) Alibaba, Inc. and its affiliates.
"""kimi_verifier: synthetic-probe benchmark for Kimi K2 parameter compliance.

This benchmark does NOT load any external dataset. It synthesizes a small
suite of probe requests against the vendor API to check that immutable
decoding parameters (``temperature``, ``top_p``, ``presence_penalty``,
``frequency_penalty``, ``n``) are enforced per the Kimi K2 specification:
default values must be ACCEPTED, any other value must be REJECTED with 400.

Adapted from Kimi-Vendor-Verifier's ``verify_params.py``. Each probe is one
Sample; ``_on_inference`` makes one chat completion call with a per-sample
``extra_body`` and catches BadRequestError as the success signal when a
reject was expected.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, VendorVerifierAdapter
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model.generate_config import GenerateConfig
from evalscope.api.model.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from .param_spec import IMMUTABLE_PARAMS, thinking_extra_body

logger = get_logger()

_PROBE_INPUT = "Say 'OK' and nothing else."
_THINK_MODES = ('kimi', 'opensource', 'none')

KIMI_VERIFIER_DESCRIPTION = """
## Overview

Kimi-Vendor-Verifier is a pre-flight compliance check for Kimi K2 / K2-Thinking deployments. It sends synthetic probe requests to verify that the vendor API correctly **rejects** non-default values of immutable decoding parameters (``temperature``, ``top_p``, ``presence_penalty``, ``frequency_penalty``, ``n``) and **accepts** their defaults. A vendor that silently accepts wrong values risks producing degraded model output that does not match official Moonshot AI behavior. Adapted from [Kimi-Vendor-Verifier/verify_params.py](https://github.com/MoonshotAI/Kimi-Vendor-Verifier/blob/main/verify_params.py).

## Task Description

- **Task Type**: API parameter-compliance probing (deployment health check)
- **Input**: A minimal chat message plus a single test parameter and thinking-mode `extra_body`
- **Output**: Whether the vendor accepted (HTTP 200) or rejected (HTTP 400) the request
- **Dataset**: Fully synthetic — no external dataset is downloaded; probes are generated in code from the K2 spec

## Key Features

- Synthetic probe set: one ``no_param`` sanity probe + 5 default-value (accept) probes + 5 wrong-value (reject) probes per (subset × thinking) combination
- Three subsets covering all common Kimi deployment shapes:
    - ``kimi`` — official Moonshot SaaS API (``extra_body = {"thinking": {"type": ...}}``); thinking on/off
    - ``opensource`` — vLLM / SGLang / KTransformers chat-template hook (``extra_body = {"chat_template_kwargs": {"thinking": ...}}``); thinking on/off
    - ``none`` — non-hybrid model; no thinking parameter sent
- HTTP 400 responses are treated as the success signal when a reject was expected
- Single small request per probe; total cost is negligible compared to a full benchmark

## Evaluation Notes

- Default configuration uses **0-shot** synthetic probes
- Metrics: **param_immutable_reject_rate**, **param_default_accept_rate**
- A correctly-deployed Kimi K2 vendor should report both metrics at **1.0**; anything less indicates a parameter-enforcement gap
- For non-Kimi models, expect ``param_immutable_reject_rate = 0`` (no K2 spec to enforce) and ``param_default_accept_rate = 1.0`` (sensible defaults accepted)
- Select subset via ``dataset_args={'kimi_verifier': {'subset_list': ['kimi']}}`` (or ``opensource`` / ``none``)
"""


@register_benchmark(
    BenchmarkMeta(
        name='kimi_verifier',
        pretty_name='Kimi-Vendor-Verifier (Param Compliance)',
        description=KIMI_VERIFIER_DESCRIPTION,
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT],
        dataset_id='kimi_verifier',  # synthetic; placeholder
        metric_list=[
            'param_immutable_reject_rate',
            'param_default_accept_rate',
        ],
        aggregation='mean',
        subset_list=list(_THINK_MODES),
        eval_split='test',
    )
)
class KimiVerifierAdapter(VendorVerifierAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False

    # --------------------------------------------------------------------
    # Synthetic dataset
    # --------------------------------------------------------------------

    def load(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        """Synthesize probe samples. No external dataset is consulted."""
        by_subset: Dict[str, List[Sample]] = {mode: [] for mode in self.subset_list}

        for mode in self.subset_list:
            if mode not in _THINK_MODES:
                logger.warning(f'kimi_verifier: unknown subset {mode!r}, skipping')
                continue
            # 'none' subset doesn't have a thinking concept
            thinking_options = [False] if mode == 'none' else [False, True]
            for thinking in thinking_options:
                by_subset[mode].extend(self._build_probes_for(mode, thinking))

        datasets: Dict[str, MemoryDataset] = {}
        for mode, samples in by_subset.items():
            ds = MemoryDataset(samples, name=f'kimi_verifier:{mode}')
            ds.reindex()
            datasets[mode] = ds
        return DatasetDict(datasets), None

    @classmethod
    def _build_probes_for(cls, think_mode: str, thinking: bool) -> List[Sample]:
        """Build all probes for one (think_mode, thinking) combination.

        Mirrors Kimi-Vendor-Verifier ``verify_params.py``:
          - one "no_param" probe (only the thinking extra_body) → expect ACCEPT
          - per ParamSpec, default value → expect ACCEPT
          - per ParamSpec, ``wrong_value`` → expect REJECT (skipped when
            wrong_value happens to equal the default)
        """
        probes: List[Sample] = [
            # no_param: base-API sanity check (catches "vendor always 400s")
            cls._make_probe(
                think_mode=think_mode,
                thinking=thinking,
                param_name=None,
                test_value=None,
                expected_reject=False,
            )
        ]
        for spec in IMMUTABLE_PARAMS:
            default_value = spec.think_default if thinking else spec.non_think_default
            probes.append(cls._make_probe(
                think_mode=think_mode,
                thinking=thinking,
                param_name=spec.name,
                test_value=default_value,
                expected_reject=False,
            ))
            if spec.wrong_value != default_value:
                probes.append(cls._make_probe(
                    think_mode=think_mode,
                    thinking=thinking,
                    param_name=spec.name,
                    test_value=spec.wrong_value,
                    expected_reject=True,
                ))
        return probes

    @staticmethod
    def _make_probe(
        *,
        think_mode: str,
        thinking: bool,
        param_name: Optional[str],
        test_value: Any,
        expected_reject: bool,
    ) -> Sample:
        return Sample(
            input=[ChatMessageUser(content=_PROBE_INPUT)],
            target='',
            subset_key=think_mode,
            metadata={
                'think_mode': think_mode,
                'thinking': thinking,
                'param_name': param_name,
                'test_value': test_value,
                'expected_reject': expected_reject,
            },
        )

    # --------------------------------------------------------------------
    # Inference: per-sample extra_body and BadRequestError capture
    # --------------------------------------------------------------------

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        """One chat completion per probe; capture HTTP 400 as expected reject signal."""
        meta = sample.metadata
        extra_body = thinking_extra_body(meta['thinking'], meta['think_mode'])

        cfg_kwargs: Dict[str, Any] = {
            'max_tokens': 16,
            'extra_body': extra_body if extra_body else None,
        }
        # no_param probe sends ONLY the thinking extra_body — no immutable param
        if meta['param_name'] is not None:
            cfg_kwargs[meta['param_name']] = meta['test_value']
        cfg = GenerateConfig(**cfg_kwargs)

        try:
            return model.generate(input=sample.input, config=cfg)
        except Exception as e:
            # Both raw httpx-level 400s and openai.BadRequestError surface here.
            # We treat any error as a "rejected" response; the score logic
            # decides whether that was expected or a failure.
            return ModelOutput.from_content(
                content='',
                stop_reason='stop',
                error=f'{type(e).__name__}: {e}',
            )

    # --------------------------------------------------------------------
    # Scoring
    # --------------------------------------------------------------------

    def match_score(self, original_prediction, filtered_prediction, reference, task_state: TaskState) -> Score:
        score = Score(
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
        )
        meta = task_state.metadata
        expected_reject = bool(meta.get('expected_reject', False))
        was_rejected = bool(task_state.output.error)

        passed = (was_rejected == expected_reject)
        score.value = {
            'passed': int(passed),
            'expected_reject': int(expected_reject),
            'was_rejected': int(was_rejected),
        }
        score.metadata = {
            'param_name': meta.get('param_name'),
            'test_value': meta.get('test_value'),
            'thinking': meta.get('thinking'),
            'think_mode': meta.get('think_mode'),
            'error': task_state.output.error,
        }
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        reject_total = reject_correct = 0
        accept_total = accept_correct = 0

        for ss in sample_scores:
            v = ss.score.value or {}
            expected_reject = int(v.get('expected_reject', 0))
            passed = int(v.get('passed', 0))
            if expected_reject:
                reject_total += 1
                reject_correct += passed
            else:
                accept_total += 1
                accept_correct += passed

        reject_rate = reject_correct / reject_total if reject_total else 0.0
        accept_rate = accept_correct / accept_total if accept_total else 0.0

        return [
            AggScore(metric_name='param_immutable_reject_rate', score=reject_rate, num=reject_total, metadata={}),
            AggScore(metric_name='param_default_accept_rate', score=accept_rate, num=accept_total, metadata={}),
        ]
