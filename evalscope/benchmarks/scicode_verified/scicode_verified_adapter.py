# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501

import base64
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetHub, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageSpec
from evalscope.constants import DEFAULT_EVALSCOPE_CACHE_DIR, HubType, Tags
from evalscope.utils.logger import get_logger

from ..scicode.prompt_templates import INITIAL_PROMPT
from ..scicode.scicode_adapter import SciCodeAdapter

logger = get_logger()

# Steps the official SciCode harness never generates nor scores: their reference code is injected instead, so
# that later steps build on it (same as the SciCode-Verified harness).
SKIPPED_STEPS = ('13.6', '62.1', '76.3')
HERE = Path(__file__).parent
GOLD_DIR = HERE / 'gold'
OFFICIAL_DIR = HERE / 'official'
SCICODE_DOCKER_DIR = HERE.parent / 'scicode' / 'docker'
IMPORT_RE = re.compile(r'^\s*(import .*|from .*\s+import\s+.*)', flags=re.MULTILINE)
ENV_LINE_RE = re.compile(r'^SCV_ENV (\S+) (PASS|FAIL) ([0-9.]+)s$', flags=re.MULTILINE)

# Pinned grading environments of the SciCode-Verified multi-environment OR grading (analysis/or_gain.json):
# '2024' = numpy 1.26.4 / scipy 1.13.1, '2025' = numpy 2.4.2 / scipy 1.17.1. Both venvs live in one sandbox image
# (Python 3.11; the reference '2025' env is Python 3.12). matplotlib is installed in both (problem 80).
GRADING_ENVS = (('2024', '/opt/env2024/bin/python'), ('2025', '/opt/env2025/bin/python'))
ENV_TIMEOUT = 1800  # seconds per environment, as the SciCode-Verified harness
DOCKERFILE = """FROM python:3.11-slim

RUN python -m venv /opt/env2024 && /opt/env2024/bin/pip install --no-cache-dir \\
    numpy==1.26.4 scipy==1.13.1 h5py sympy "matplotlib<3.10"
RUN python -m venv /opt/env2025 && /opt/env2025/bin/pip install --no-cache-dir \\
    numpy==2.4.2 scipy==1.17.1 h5py sympy matplotlib

WORKDIR /workspace

COPY process_data.py test_util.py test_data.h5 /workspace/
ENV PYTHONPATH="/workspace"
"""

# Runs the composed grading script under each environment, pass if any (short-circuits on the first pass). The
# script is passed base64-encoded; executed by the sandbox's base interpreter.
OR_WRAPPER = """import base64, os, subprocess, sys, tempfile, time
_src = base64.b64decode('{b64}').decode()
_fd, _path = tempfile.mkstemp(suffix='.py', dir='/tmp'); os.write(_fd, _src.encode()); os.close(_fd)
_env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
_last = None
for _label, _py in {envs!r}:
    _t0 = time.time()
    try:
        _p = subprocess.run([_py, _path], capture_output=True, text=True, timeout={timeout}, env=_env,
                            cwd='/workspace' if os.path.isdir('/workspace') else None)
        _ok, _out = _p.returncode == 0, (_p.stdout or '') + (_p.stderr or '')
    except subprocess.TimeoutExpired:
        _ok, _out = False, 'Timeout: step exceeded {timeout} s'
    print('SCV_ENV', _label, 'PASS' if _ok else 'FAIL', f'{{time.time() - _t0:.1f}}s', flush=True)
    if _ok:
        sys.exit(0)
    _last = (_label, _out)
print('--- last environment output (' + _last[0] + '):')
print(_last[1][-4000:])
sys.exit(1)
"""


@register_benchmark(
    BenchmarkMeta(
        name='scicode_verified',
        pretty_name='SciCode-Verified',
        tags=[Tags.CODING],
        description="""
## Overview

SciCode-Verified is a human-verified correction of the SciCode test benchmark (arXiv 2608.04975). A domain-expert audit of the 65 SciCode test problems found 192 defects that reject correct solutions (non-reproducible or wrong gold answers, over-tight tolerances, under-specified or self-contradictory specifications), touching 155 of the 287 scored subproblems. SciCode-Verified fixes them, keeping the scientific task unchanged.

## Task Description

- **Task Type**: Scientific Code Generation
- **Input**: Multi-step scientific problem with subproblems
- **Output**: Python code solving each subproblem
- **Domains**: Physics, Math, Material Science, Biology, Chemistry

## Protocol (defaults reproduce the SciCode-Verified harness)

- 64 problems (problem 2 dropped: its specification admits no verifiable answer), corrected text and test targets (`test_data_cleaned.h5`), 287 scored subproblems.
- `prompt_mode='official'` (default): one single-turn prompt per subproblem with the official SciCode templates (previous step descriptions + previously extracted code). `prompt_mode='multi_turn'`: evalscope's SciCode conversation (one chat, one turn per subproblem).
- Steps 13.6, 62.1 and 76.3 are not generated nor scored: their reference code is used (official harness).
- Extracted code: first ```python block, import lines removed (official harness); dependencies imported once.
- `grading_envs='or'` (default): each step runs under numpy 1.26.4 / scipy 1.13.1 and, if it fails, numpy 2.4.2 / scipy 1.17.1; it passes if either passes. 1800 s per environment. `grading_envs='2024'` / `'2025'`: one environment only.
- The dataset is loaded from the Hugging Face hub (`shhu2001/SciCode-Verified`) unless a local path is given; `problem_ids` restricts the run to some problems.

## Evaluation Notes

- **Sandbox environment required** for code execution
- Two metrics tracked:
  - `main_problem_pass_rate`: Complete problem solved
  - `subproblem_pass_rate`: Individual subproblem success rate (scored subproblems only)
- The SciCode-Verified paper reports results with and without scientist background (`provide_background`) separately.
""",
        dataset_id='shhu2001/SciCode-Verified',
        hf_dataset_id='shhu2001/SciCode-Verified',
        metric_list=['main_problem_pass_rate', 'subproblem_pass_rate'],
        primary_metric='main_problem_pass_rate',
        eval_split='test',
        evaluation_version='v1.0',
        review_timeout=2 * ENV_TIMEOUT + 60,
        prompt_template=INITIAL_PROMPT,
        extra_params={
            'provide_background': {
                'type': 'bool',
                'value': False,
                'description': "Include scientific background information written by scientists for the problem in the model's prompt.",
            },
            'prompt_mode': {
                'type': 'str',
                'value': 'official',
                'description': "'official': single-turn prompts with the official SciCode templates (SciCode-Verified harness); 'multi_turn': one conversation, one turn per subproblem (evalscope scicode).",
            },
            'grading_envs': {
                'type': 'str',
                'value': 'or',
                'description': "'or': pass if the step passes under numpy 1.26.4/scipy 1.13.1 or numpy 2.4.2/scipy 1.17.1 (SciCode-Verified); '2024' or '2025': that environment only.",
            },
            'problem_ids': {
                'type': 'list[str]',
                'value': [],
                'description': 'Evaluate only these problem ids (e.g. ["5", "12"]); empty: all problems.',
            },
        },
        sandbox_config={
            'image': 'scicode-verified-benchmark:latest',
            'tools_config': {'shell_executor': {}, 'python_executor': {}},
        },
    )
)
class SciCodeVerifiedAdapter(SciCodeAdapter):
    """SciCode on the SciCode-Verified data, reproducing the SciCode-Verified harness by default."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docker_path = Path(DEFAULT_EVALSCOPE_CACHE_DIR) / 'scicode_verified' / 'docker'
        self.problem_ids = {str(p) for p in (self.extra_params.get('problem_ids') or [])}
        self.prompt_mode = self.extra_params.get('prompt_mode', 'official')
        if self.prompt_mode not in ('official', 'multi_turn'):
            raise ValueError(f'prompt_mode must be official or multi_turn, got {self.prompt_mode!r}')
        mode = str(self.extra_params.get('grading_envs', 'or'))
        self.grading_envs = [e for e in GRADING_ENVS if mode == 'or' or e[0] == mode]
        if not self.grading_envs:
            raise ValueError(f"grading_envs must be 'or', '2024' or '2025', got {mode!r}")
        template = 'multistep_template.txt' if self.provide_background else 'background_comment_template.txt'
        self.official_template = (OFFICIAL_DIR / template).read_text()

    @property
    def dataset_hub(self) -> str:
        """SciCode-Verified is only published on the Hugging Face hub: use it unless a local path is configured."""
        hub = super().dataset_hub
        return hub if hub == HubType.LOCAL else HubType.HUGGINGFACE

    @dataset_hub.setter
    def dataset_hub(self, value: str):
        self._task_config.dataset_hub = value

    def sample_filter(self, sample: Sample) -> bool:
        return not self.problem_ids or str(sample.metadata['problem_id']) in self.problem_ids

    def load(self):
        """Assemble the sandbox build context: grading helpers, the two-environment Dockerfile, the cleaned data."""
        self.docker_path.mkdir(parents=True, exist_ok=True)
        # the image tag hashes the whole context: drop files the current recipe does not write
        for stale in self.docker_path.iterdir():
            if stale.name not in ('Dockerfile', 'process_data.py', 'test_util.py', 'test_data.h5'):
                stale.unlink()
        for name in ('process_data.py', 'test_util.py'):
            shutil.copyfile(SCICODE_DOCKER_DIR / name, self.docker_path / name)
        (self.docker_path / 'Dockerfile').write_text(DOCKERFILE)
        source_path = DatasetHub(
            data_id_or_path=self.dataset_id,
            data_source=self.dataset_hub,
            revision=self.dataset_revision,
            force_redownload=self.force_redownload,
            cache_dir=self.dataset_dir,
        ).download_file('test_data_cleaned.h5')
        target = self.docker_path / 'test_data.h5'
        src = Path(source_path).resolve()
        if not target.exists() or target.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, target)
        # Skip SciCodeAdapter.load (it downloads the original test_data.h5).
        return super(SciCodeAdapter, self).load()

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_imports(code: str) -> str:
        return IMPORT_RE.sub('', code)

    @classmethod
    def _extract(cls, response: str) -> str:
        """Code of a response exactly as the official harness: first ```python block (else first ``` block, else the
        whole response), not whitespace-stripped, import lines removed."""
        if not response:
            return ''
        if '```' in response:
            if '```python' in response:
                response = response.split('```python')[1].split('```')[0]
            else:
                response = response.split('```')[1].split('```')[0]
        return cls._strip_imports(response)

    def _gold_code(self, step_number: str) -> str:
        return self._strip_imports((GOLD_DIR / f'{step_number}.txt').read_text())

    def _step_text(self, subproblem: Dict[str, Any]) -> str:
        text = subproblem['step_description_prompt']
        if self.provide_background:
            text += '\n' + subproblem['step_background']
        return text

    def _official_prompt(self, sample: Sample, k: int, prev_codes: List[str]) -> str:
        """Official SciCode prompt for step k (as SciCode-Verified's build_prompt: full previous code)."""
        steps = sample.metadata['sub_steps']
        lines: List[str] = []
        for i in range(k):
            lines += [self._step_text(steps[i]), prev_codes[i], '------']
        cur = steps[k]
        return self.official_template.format(
            problem_steps_str='\n\n'.join(lines[:-1]) if lines else '',
            next_step_str=f"{self._step_text(cur)}\n\n{cur['function_header']}\n\n{cur['return_line']}",
            dependencies=sample.metadata['required_dependencies'],
        )

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs: Any) -> TaskState:
        """One assistant message per step; skipped steps get the reference code without a model call.

        official: each step is a fresh single-turn request; the history keeps (prompt, answer) pairs for display
        and for code extraction. multi_turn: evalscope's SciCode conversation.
        """
        history: list = [] if self.prompt_mode == 'official' else self.initialize_history(sample)
        prev_codes: List[str] = []
        last_output = None
        for k, subproblem in enumerate(sample.metadata['sub_steps']):
            if subproblem['step_number'] in SKIPPED_STEPS:
                logger.info(f'Inserting reference code for subproblem {subproblem["step_number"]} (not scored).')
                code = self._gold_code(subproblem['step_number'])
                if self.prompt_mode == 'official':
                    prompt = ChatMessageUser(content=self._official_prompt(sample, k, prev_codes))
                else:
                    prompt = ChatMessageUser(content=self.prompt_template.format(**subproblem))
                history += [prompt, ModelOutput.from_content(model=model.name, content=f'```python\n{code}\n```').message]
                prev_codes.append(code)
                continue
            if self.prompt_mode == 'official':
                logger.info(f'Submitting subproblem {subproblem["step_number"]} to model (official prompt)...')
                prompt = ChatMessageUser(content=self._official_prompt(sample, k, prev_codes))
                last_output = model.generate(input=[prompt], tools=sample.tools)
            else:
                prompt = self.build_turn_prompt(sample, history, k)
                if isinstance(prompt, str):
                    prompt = ChatMessageUser(content=prompt)
                last_output = model.generate(input=history + [prompt], tools=sample.tools)
            history += [prompt, last_output.message]
            prev_codes.append(self._extract(last_output.message.text))
            self.on_turn_complete(sample, k, last_output, history)
        return TaskState(
            model=model.name,
            sample=sample,
            messages=history,
            output=self.build_final_output(model, sample, history, last_output),
            completed=True,
        )

    # ------------------------------------------------------------------ #
    # Grading                                                              #
    # ------------------------------------------------------------------ #

    def _step_codes(self, state: TaskState) -> List[str]:
        answers = [m for m in state.messages if m.role == 'assistant']
        assert len(answers) == len(state.metadata['sub_steps'])
        return [self._extract(m.text) for m in answers]

    def _compose_code(self, subproblem: Dict[str, Any], state: TaskState) -> str:
        """Dependencies + steps 1..k code (imports stripped, as the official harness) + step k's tests, wrapped
        to run under the grading environments (pass if any)."""
        from ..scicode.util import subproblem_str_to_int

        step = subproblem['step_number']
        k = subproblem_str_to_int(step)
        sections = [state.metadata['required_dependencies'], *self._step_codes(state)[:k],
                    'from test_util import are_dicts_close, cmp_tuple_or_list',
                    'from process_data import process_hdf5_to_tuple',
                    f"targets = process_hdf5_to_tuple('{step}', {len(subproblem['test_cases'])})"]
        for i, test_case in enumerate(subproblem['test_cases']):
            sections += [f'target = targets[{i}]', test_case]
        script = '\n\n'.join(sections)
        script = script.replace('from scicode.compare.cmp import cmp_tuple_or_list', '')
        script = script.replace('from scicode.compare.cmp import are_dicts_close', '')
        b64 = base64.b64encode(script.encode()).decode()
        return OR_WRAPPER.format(b64=b64, envs=list(self.grading_envs), timeout=ENV_TIMEOUT)

    def _verify_subproblem(self, subproblem: Dict[str, Any], state: TaskState) -> Score:
        """Grade one step (SciCodeAdapter) and record, from the wrapper's output, the verdict and grading time of
        each environment (`per_env`, same shape as SciCode-Verified's score.json), their total, and the environments
        that hit the time limit (`timed_out_envs`; `timed_out`: the step failed and an environment hit the limit)."""
        score = super()._verify_subproblem(subproblem=subproblem, state=state)
        result = (score.metadata or {}).get('execution_result') or {}
        output = str(result.get('output') or '') + str(result.get('error') or '')
        envs = ENV_LINE_RE.findall(output)
        score.metadata = dict(score.metadata or {})
        score.metadata['per_env'] = {label: verdict == 'PASS' for label, verdict, _ in envs}
        score.metadata['grading_seconds'] = {label: float(sec) for label, _, sec in envs}
        score.metadata['grading_seconds_total'] = round(sum(float(sec) for _, _, sec in envs), 1)
        score.metadata['timed_out_envs'] = [label for label, _, sec in envs if float(sec) >= ENV_TIMEOUT]
        passed = any(score.metadata['per_env'].values())
        score.metadata['timed_out'] = (not passed) and bool(score.metadata['timed_out_envs'])
        return score

    def match_score(self, original_prediction, filtered_prediction, reference, task_state: TaskState) -> Score:
        """Score only the generated subproblems (the skipped steps are excluded, as in the official harness)."""
        result: Dict[str, float] = {}
        meta: Dict[str, Any] = {}
        for subproblem in task_state.metadata['sub_steps']:
            if subproblem['step_number'] in SKIPPED_STEPS:
                continue
            score = self._verify_subproblem(subproblem=subproblem, state=task_state)
            for k, v in score.value.items():
                result[k] = v
                meta[k] = score.metadata
        main_problem = bool(result) and all(v == 1 for v in result.values())
        return Score(
            value={'main_problem': 1.0 if main_problem else 0.0, **result},
            metadata=meta,
            prediction=original_prediction,
            extracted_prediction=filtered_prediction,
            main_score_name='main_problem',
        )

    def get_sandbox_image_spec(self) -> DockerImageSpec:
        """Sandbox image with the two pinned grading environments and the SciCode-Verified test data."""
        return DockerImageSpec(
            name_prefix='scicode-verified-benchmark',
            context_dir=self.docker_path.as_posix(),
            dockerfile=(self.docker_path / 'Dockerfile').as_posix(),
            cache_key_parts=['sandbox', 'scicode-verified-benchmark:latest'],
        )
