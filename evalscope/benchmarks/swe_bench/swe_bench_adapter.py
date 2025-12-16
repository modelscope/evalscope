import json

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import FieldSpec, RemoteDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_verified',
        pretty_name='SWE-bench_Verified',
        tags=[Tags.CODING],
        description='SWE-bench Verified is a subset of 500 samples from the SWE-bench test set, '
        'which have been human-validated for quality. SWE-bench is a dataset that tests systems\' '
        'ability to solve GitHub issues automatically. '
        'Need to run `pip install swebench==4.1.0` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)',
        dataset_id='princeton-nlp/SWE-bench_Verified',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'inference_dataset_id': {
                'type': 'str',
                'description': 'Oracle dataset ID used to fetch inference context.',
                'value': 'princeton-nlp/SWE-bench_oracle'
            },
            'build_docker_images': {
                'type': 'bool',
                'description': 'Build Docker images locally for each sample.',
                'value': True
            },
            'pull_remote_images_if_available': {
                'type': 'bool',
                'description': 'Attempt to pull existing remote Docker images before building.',
                'value': True
            },
            'force_arch': {
                'type': 'str',
                'description': 'Optionally force the docker images to be pulled/built for a specific architecture.',
                'value': '',
                'choices': ['', 'arm64', 'x86_64']
            }
        }
    )
)
class SWEBenchVerifiedAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import('swebench', package='swebench==4.1.0', raise_error=True, feature_name=self.pretty_name)

        self.build_docker_images: bool = self.extra_params.get('build_docker_images', True)
        self.pull_remote_images_if_available: bool = self.extra_params.get('pull_remote_images_if_available', True)
        self.inference_dataset_id: str = self.extra_params.get('inference_dataset_id', 'princeton-nlp/SWE-bench_oracle')
        self.force_arch: str = self.extra_params.get('force_arch', '')

    def load(self):
        logger.info(f'Loading oracle dataset: {self.inference_dataset_id}')
        loader = RemoteDataLoader(
            data_id_or_path=self.inference_dataset_id,
            split='test',
            sample_fields=FieldSpec(input='problem_statement', metadata=[
                'instance_id',
                'text',
            ])
        )
        infer_dataset = loader.load()
        self.infer_samples = {s.metadata['instance_id']: s.metadata for s in infer_dataset}

        return super().load()

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=self.infer_samples[record['instance_id']]['text'],
            metadata={
                'problem_statement': record['problem_statement'],
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
        """Build images in post process samples"""
        from .build_images import build_images

        if self.build_docker_images:
            samples = self.test_dataset[self.default_subset]

            # build images
            id_to_docker_image_map = build_images(
                samples=samples,
                force_rebuild=False,
                max_workers=4,
                use_remote_images=self.pull_remote_images_if_available,
                force_arch=self.force_arch,
            )

            # Replace docker_image_from_id function with authoritative source
            def get_docker_image(instance_id: str) -> str:
                return id_to_docker_image_map.get(instance_id, '')

            docker_image_from_id = get_docker_image
        else:
            from .utils import get_remote_docker_image_from_id

            docker_image_from_id = get_remote_docker_image_from_id

        # update metadata with docker image
        for sample in samples:
            instance_id = sample.metadata['instance_id']
            sample.metadata['docker_image'] = docker_image_from_id(instance_id)

        super()._post_process_samples()

    def extract_answer(self, prediction: str, task_state) -> str:
        """Extract the final answer from the model output."""
        from swebench.inference.make_datasets.utils import extract_diff

        return extract_diff(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        from .utils import eval_instance

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        result = eval_instance(
            instance=task_state.metadata,
            pred=filtered_prediction,
            timeout=1800,
            log_dir=self._task_config.work_dir,
        )

        score.value = {'acc': float(result.get('resolved', 0.0))}
        score.metadata = result
        return score


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_verified_mini',
        pretty_name='SWE-bench_Verified_mini',
        tags=[Tags.CODING],
        description='SWEBench-verified-mini is a subset of SWEBench-verified '
        'that uses 50 instead of 500 datapoints, requires 5GB instead of 130GB '
        'of storage and has approximately the same distribution of performance, '
        'test pass rates and difficulty as the original dataset. '
        'Need to run `pip install swebench==4.1.0` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)',
        dataset_id='evalscope/swe-bench-verified-mini',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'build_docker_images': {
                'type': 'bool',
                'description': 'Build Docker images locally for each sample.',
                'value': True
            },
            'pull_remote_images_if_available': {
                'type': 'bool',
                'description': 'Attempt to pull existing remote Docker images before building.',
                'value': True
            },
            'inference_dataset_id': {
                'type': 'str',
                'description': 'Oracle dataset ID used to fetch inference context.',
                'value': 'princeton-nlp/SWE-bench_oracle'
            },
            'force_arch': {
                'type': 'str',
                'description': 'Optionally force the docker images to be pulled/built for a specific architecture.',
                'value': '',
                'choices': ['', 'arm64', 'x86_64']
            }
        }
    )
)
class SWEBenchVerifiedMiniAdapter(SWEBenchVerifiedAdapter):
    ...


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_lite',
        pretty_name='SWE-bench_Lite',
        tags=[Tags.CODING],
        description='SWE-bench Lite is subset of SWE-bench, a dataset that tests systems\' '
        'ability to solve GitHub issues automatically. The dataset collects 300 test '
        'Issue-Pull Request pairs from 11 popular Python. Evaluation is performed by '
        'unit test verification using post-PR behavior as the reference solution. '
        'Need to run `pip install swebench==4.1.0` before evaluating. '
        '[Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html)',
        dataset_id='princeton-nlp/SWE-bench_Lite',
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        extra_params={
            'build_docker_images': {
                'type': 'bool',
                'description': 'Build Docker images locally for each sample.',
                'value': True
            },
            'pull_remote_images_if_available': {
                'type': 'bool',
                'description': 'Attempt to pull existing remote Docker images before building.',
                'value': True
            },
            'inference_dataset_id': {
                'type': 'str',
                'description': 'Oracle dataset ID used to fetch inference context.',
                'value': 'princeton-nlp/SWE-bench_oracle'
            },
            'force_arch': {
                'type': 'str',
                'description': 'Optionally force the docker images to be pulled/built for a specific architecture.',
                'value': '',
                'choices': ['', 'arm64', 'x86_64']
            }
        }
    )
)
class SWEBenchLiteAdapter(SWEBenchVerifiedAdapter):
    ...
