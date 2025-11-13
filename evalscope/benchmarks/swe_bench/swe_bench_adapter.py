import json
import re
import traceback
from typing import Any, Dict, List

from evalscope.api.benchmark import AgentAdapter, BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report import Category, Report, Subset
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_verified',
        pretty_name='SWE-bench_Verified',
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT, Tags.CODING],
        description='SWE-bench Verified is a subset of 500 samples from the SWE-bench test set, '
        'which have been human-validated for quality. SWE-bench is a dataset that tests systems\' '
        'ability to solve GitHub issues automatically. '
        'Need to run `pip install swebench==4.1.0` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)',
        dataset_id='princeton-nlp/SWE-bench_Verified',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='Please solve the following coding issue:\n\n{question}',
        extra_params={
            'build_docker_images': True,
            'pull_remote_images_if_available': True,
        }
    )
)
class SWEBenchVerifiedAdapter(AgentAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import('swebench', package='swebench==4.1.0', raise_error=True, feature_name=self.pretty_name)

        self.build_docker_images: bool = self.extra_params.get('build_docker_images', True)
        self.pull_remote_images_if_available: bool = self.extra_params.get('pull_remote_images_if_available', True)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['problem_statement'],
            metadata={
                'instance_id': record['instance_id'],
                'base_commit': record['base_commit'],
                'patch': record['patch'],
                'PASS_TO_PASS': json.loads(record['PASS_TO_PASS']),
                'FAIL_TO_PASS': json.loads(record['FAIL_TO_PASS']),
                'test_patch': record['test_patch'],
                'version': record['version'],
                'repo': record['repo'],
                'environment_setup_commit': record['environment_setup_commit'],
                'hints_text': record['hints_text'],
                'created_at': record['created_at'],
            }
        )

    def _post_process_samples(self):
        """build images in post process samples"""

        from .build_images import build_images

        if self.build_docker_images:
            samples = self.test_dataset[self.default_subset]

            # build images
            id_to_docker_image_map = build_images(
                samples=samples,
                force_rebuild=False,
                max_workers=4,
                use_remote_images=self.pull_remote_images_if_available,
            )

            # update metadata with docker image
            for sample in samples:
                instance_id = sample.metadata['instance_id']
                sample.metadata['docker_image'] = id_to_docker_image_map[instance_id]

        super()._post_process_samples()


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_verified_mini',
        pretty_name='SWE-bench_Verified_mini',
        tags=[Tags.FUNCTION_CALLING, Tags.AGENT, Tags.CODING],
        description='SWEBench-verified-mini is a subset of SWEBench-verified '
        'that uses 50 instead of 500 datapoints, requires 5GB instead of 130GB '
        'of storage and has approximately the same distribution of performance, '
        'test pass rates and difficulty as the original dataset. '
        'Need to run `pip install swebench==4.1.0` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)',
        dataset_id='evalscope/swe-bench-verified-mini',
        metric_list=['acc'],
        eval_split='test',
    )
)
class SWEBenchVerifiedMiniAdapter(SWEBenchVerifiedAdapter):
    ...
