# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.code_utils import (
    END_TOKENS,
    IMPORT_HELPER,
    default_extract_helper,
    extract_code_from_freeform_completion,
)
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='multiple_mbpp',
        pretty_name='MultiPL-E MBPP',
        tags=[Tags.CODING],
        description='This multilingual MBPP was from MultiPL-E. 18 languages were implemented and tested. '
        '**Sandbox environment is needed for execution to safely run and evaluate the generated code, please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for more details.**',  # noqa: E501
        dataset_id='evalscope/MultiPL-E',
        subset_list=[
            'mbpp-cpp',
            'mbpp-ts',
            'mbpp-sh',
            'mbpp-cs',
            'mbpp-go',
            'mbpp-java',
            'mbpp-lua',
            'mbpp-js',
            'mbpp-php',
            'mbpp-pl',
            'mbpp-rkt',
            'mbpp-r',
            'mbpp-rs',
            'mbpp-scala',
            'mbpp-swift',
            'mbpp-rb',
            'mbpp-d',
            'mbpp-jl',
        ],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template='{prompt}',
        review_timeout=30,
        sandbox_config={
            'image': 'volcengine/sandbox-fusion:server-20250609',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {},
                'multi_code_executor': {}  # Multi-language code executor
            },
            'memory_limit': '4g',
            'cpu_limit': '4.0',
        },
    )
)
class MultiPLEMBPPAdapter(DefaultDataAdapter):
    """
    MultiPL-E MBPP adapter using the new data processing framework.
    Assumptions:
    - Each subset is a single language suite.
    - Records contain: 'prompt', 'tests', optional 'stop_tokens', 'language', and id: 'task_id' or 'name'.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        return Sample(
            input=record['prompt'],
            target='',
            metadata={
                'tests': record['tests'],
                'stop_tokens': record.get('stop_tokens', []),
                'task_id': record.get('name', record.get('task_id')),
                'language': record.get('language'),
                'doctests': record.get('doctests', ''),
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Freeform SFT prompt:
        - Fence the given prompt with the language derived from metadata.language ("mbpp-<lang>").
        - Add a short instruction requesting full code without a Main entrypoint.
        """
        extract_lang, _ = self._normalize_languages(sample.metadata.get('language'))
        instruction = (
            'Please complete the above code according to the requirements in the docstring. '
            'Write the complete code and wrap it in markdown fenced code. The code should not contain `Main` function.'
        )
        return f'```{extract_lang}\n{sample.input}\n```\n\n{instruction}'

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """SFT extraction: extract fenced code, stop at language end tokens, remove entrypoints, append tests."""
        extract_lang, _ = self._normalize_languages(task_state.metadata.get('language'))

        # Choose extraction strategy per language
        use_func_extractor_langs = {'typescript', 'go', 'perl', 'racket', 'lua', 'julia', 'd', 'js', 'php', 'r', 'ruby'}
        if extract_lang in use_func_extractor_langs:
            code = default_extract_helper(prediction, extract_lang)
            code = self._remove_main(code, extract_lang)
        else:
            code, _ = extract_code_from_freeform_completion(prediction, extract_lang, first_block_only=True)
            code = self._trim_by_stop_tokens(code, task_state.metadata.get('stop_tokens', []))
            code = self._remove_main(code, extract_lang)

        # Prepend import helpers
        # import_helper = IMPORT_HELPER.get(extract_lang, [])
        # full_code = '\n'.join(import_helper) + '\n' + code

        # Language-specific post-processing (e.g., consolidate Go packages/imports)
        full_code = self._postprocess_full_code(code, extract_lang)

        # Append dataset tests
        full_code = f'{full_code}\n{task_state.metadata.get("tests", "")}'

        # Java runner expects Main
        if extract_lang == 'java':
            full_code = full_code.replace('class Problem', 'class Main')

        return full_code

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Run code in sandbox and return pass/fail."""
        if not self.use_sandbox:
            raise RuntimeError(
                'MultiPL-E MBPP requires sandboxed code execution for safety. Enable use_sandbox in TaskConfig.'
            )

        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        _, run_language = self._normalize_languages(task_state.metadata.get('language'))

        res = self.execute_code_in_sandbox(
            code=filtered_prediction,
            timeout=self.review_timeout,
            language=run_language,
        )
        passed = res.get('status') == 'success'
        score.value = {'acc': passed}
        score.metadata = {
            'task_id': task_state.metadata.get('task_id'),
            'timeout': self.review_timeout,
            'execution_result': res,
            'run_language': run_language,
        }
        score.main_score_name = 'acc'
        return score

    # Helpers

    @staticmethod
    def _normalize_languages(base: str) -> Tuple[str, str]:
        """
        Parse metadata.language (expected "mbpp-<lang>") and normalize to:
        - extract_lang: language key for code fencing/extraction and END_TOKENS
        - run_lang: sandbox-supported runtime key
        Falls back conservatively and logs warnings on unknown languages.
        """

        # Normalize for extraction (code fences and end tokens)
        extract_map = {
            'ts': 'typescript',
            'js': 'js',
            'rkt': 'racket',
            'rb': 'ruby',
            'rs': 'rust',
            'jl': 'julia',
            'sh': 'bash',
            'd': 'd',
            'go_test.go': 'go',
            'pl': 'perl',
        }
        extract_lang = extract_map.get(base, base)

        # Map to sandbox runtime keys
        run_map = {
            'go': 'go_test',  # prefer test runner
            'go_test.go': 'go_test',
            'cs': 'csharp',
            'js': 'nodejs',
            'd': 'd_ut',
            'kotlin': 'kotlin_script',
            'verilog': 'verilog',
            'sh': 'bash',
            'rs': 'rust',
            'rb': 'ruby',
            'jl': 'julia',
            'rkt': 'racket',
            'jsnode': 'nodejs',
            'typescript': 'ts',
            'pl': 'perl',
        }
        run_lang = run_map.get(base, base)

        return extract_lang, run_lang

    @classmethod
    def _trim_by_stop_tokens(cls, s: str, stop_tokens: List[str]) -> str:
        if not stop_tokens:
            return s
        for st in stop_tokens:
            if not st:
                continue
            if st.startswith('re:'):
                pattern = re.compile(st[3:].strip())
                match = pattern.search(s)
                if match:
                    s = s[:match.start()]
            else:
                index = s.find(st)
                if index != -1:
                    s = s[:index]
        return s

    @classmethod
    def _stop_after_end_token(cls, s: str, language: str) -> str:
        tokens = END_TOKENS.get(language, [])
        for et in tokens:
            index = s.find(et)
            if index != -1:
                return s[:index] + et
        return s

    @classmethod
    def _remove_main(cls, code: str, language: str) -> str:
        main_tokens: List[str] = []
        if language == 'd':
            main_tokens = ['void main']
        elif language == 'csharp':
            main_tokens = ['public static void Main']
        for token in main_tokens:
            index = code.find(token)
            if index != -1:
                return code[:index]
        return code

    @staticmethod
    def _postprocess_full_code(code: str, language: str) -> str:
        """
        Post-process the constructed full code for language-specific adjustments.

        - go: consolidate packages and imports into a single header, then place code body.
        """
        if language == 'go':
            # Collect all package declarations
            packages = set(re.findall(r'package\s+(\w+)', code))
            # Strip package lines from body
            code_body = re.sub(r'package\s+\w+\s*', '', code)

            # Collect imports: single and multi-form
            single_imports = re.findall(r'import\s+(".*?")', code_body)
            multi_block = re.findall(r'import\s*\((.*?)\)', code_body, flags=re.DOTALL)
            multi_imports: List[str] = []
            for blk in multi_block:
                multi_imports.extend([ln.strip() for ln in blk.split('\n') if ln.strip()])

            imports = {imp for imp in (single_imports + multi_imports) if imp}
            # Strip import lines from body
            code_body = re.sub(r'import\s+(".*?")', '', code_body)
            code_body = re.sub(r'import\s*\((.*?)\)', '', code_body, flags=re.DOTALL)

            # Rebuild header: packages + imports
            pkg_hdr = '\n'.join(f'package {p}' for p in packages) if packages else ''
            imp_hdr = '\n'.join(f'import {imp}' for imp in sorted(imports))
            hdr = '\n\n'.join([s for s in [pkg_hdr, imp_hdr] if s])

            # Final code layout
            code = f'{hdr}\n\n{code_body}'.strip()

        return code
