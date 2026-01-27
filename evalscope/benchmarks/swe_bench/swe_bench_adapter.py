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
        description="""
## Overview

SWE-bench Verified is a human-validated subset of 500 samples from SWE-bench, designed to test systems' ability to automatically resolve real-world GitHub issues. Each sample represents a genuine bug fix or feature implementation from popular Python repositories.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing
- **Input**: GitHub issue description with repository context
- **Output**: Code patch (diff format) that resolves the issue
- **Repositories**: 12 popular Python projects (Django, Flask, Requests, etc.)

## Key Features

- 500 human-validated Issue-Pull Request pairs
- Real-world bugs from production Python repositories
- Evaluation via unit test verification
- Docker-based isolated execution environments
- Tests both bug understanding and code modification skills

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically for each repository
- Timeout of 1800 seconds (30 min) per instance
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Supports both local image building and remote image pulling
""",
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
        description="""
## Overview

SWE-bench Verified Mini is a compact subset of SWE-bench Verified, containing 50 carefully selected samples that maintain the same distribution of performance, test pass rates, and difficulty as the full dataset while requiring only 5GB of storage instead of 130GB.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing
- **Input**: GitHub issue description with repository context
- **Output**: Code patch (diff format) that resolves the issue
- **Size**: 50 samples (vs 500 in full Verified set)

## Key Features

- Representative 50-sample subset of SWE-bench Verified
- Same difficulty distribution as full dataset
- Dramatically reduced storage requirements (5GB vs 130GB)
- Ideal for quick evaluation and development iteration
- Maintains statistical validity for benchmarking

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup
- Good for rapid prototyping and initial model assessment
""",
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
        description="""
## Overview

SWE-bench Lite is a focused subset of SWE-bench containing 300 Issue-Pull Request pairs from 11 popular Python repositories. It provides a more accessible entry point for evaluating automated software engineering capabilities.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing
- **Input**: GitHub issue description with repository context
- **Output**: Code patch (diff format) that resolves the issue
- **Size**: 300 carefully selected test instances

## Key Features

- 300 test Issue-Pull Request pairs
- 11 popular Python repositories covered
- Real-world bugs with verified solutions
- Evaluation via unit test verification
- More manageable than full SWE-bench while still challenging

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically for each repository
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Popular benchmark variant for initial model comparison
""",
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
