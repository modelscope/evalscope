import copy
import os
from collections import defaultdict
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DataAdapter, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, LocalDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric.scorer import AggScore, SampleScore
from evalscope.api.registry import get_benchmark, register_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import DataCollection, Tags
from evalscope.report.generator import ReportGenerator
from evalscope.report.report import Report
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name=DataCollection.NAME,
        dataset_id='',  # dataset_id need to be set
        description='Custom Data collection, mixing multiple evaluation datasets for '
        'a unified evaluation, aiming to use less data to achieve a more comprehensive '
        'assessment of the model\'s capabilities. '
        '[Usage Reference](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)',
        tags=[Tags.CUSTOM],
        metric_list=['acc'],
        eval_split='test',
        prompt_template='',
    )
)
class DataCollectionAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        """
        Data adapter for collection dataset.
        """
        super().__init__(**kwargs)

    def load(self):
        # Try to load dataset from local disk
        dataset_name_or_path = self.dataset_id
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download

            # Load dataset from remote
            logger.info(f'Loading dataset from modelscope: > dataset_name: {dataset_name_or_path}')
            # download dataset snapshot
            dataset_path = dataset_snapshot_download(dataset_name_or_path, allow_file_pattern='*.jsonl')

        dataset = LocalDataLoader(
            data_id_or_path=dataset_path,
            split=self.eval_split,
            sample_fields=self.record_to_sample,
            subset='test',  # NOTE: using hardcoded test subset
            limit=self.limit,
            repeats=self.repeats,
            shuffle=self.shuffle,
        ).load()

        test_dataset = DatasetDict({self.default_subset: dataset})

        return test_dataset, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object. Every record is a DatasetEntry.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        from evalscope.collections import DatasetEntry

        entry = DatasetEntry.model_validate(record)
        sample = Sample.model_validate(entry.prompt)

        record_without_prompt = copy.deepcopy(record)
        del record_without_prompt['prompt']
        sample.metadata[DataCollection.INFO] = record_without_prompt  # keep all metadata
        return sample

    def _post_process_samples(self):
        """Post process of each sample"""
        self._initialize_adapters()

    def _initialize_adapters(self):
        """Init adapters for each dataset and create dataset id map"""
        self.dataset_adapters: Dict[str, DataAdapter] = {}
        self.dataset_name_map = defaultdict(lambda: defaultdict(list))

        # load dataset args
        dataset_args = copy.deepcopy(self._task_config.dataset_args)

        # Iterate through each sample in the dataset
        dataset = self.test_dataset[self.default_subset]
        for sample in dataset:
            collection_info = sample.metadata.get(DataCollection.INFO, {})
            dataset_name = collection_info.get('dataset_name', '')
            subset_name = collection_info.get('subset_name', '')
            # create id mapping
            self.dataset_name_map[dataset_name][subset_name].append(sample.id)

            # update dataset args
            cur_dataset_args = dataset_args.get(dataset_name, {})

            # Initialize dataset adapter
            if dataset_name not in self.dataset_adapters:
                config = TaskConfig(dataset_args={dataset_name: cur_dataset_args})
                self.dataset_adapters[dataset_name] = get_benchmark(dataset_name, config=config)

    def _get_adapter(self, metadata: Dict[str, Any]) -> DataAdapter:
        collection_info = metadata.get(DataCollection.INFO, {})
        dataset_name = collection_info.get('dataset_name', '')
        return self.dataset_adapters.get(dataset_name)

    def run_inference(self, model, sample, output_dir, **kwargs) -> TaskState:
        data_adapter = self._get_adapter(sample.metadata)
        if not data_adapter:
            raise ValueError(f'No data adapter found for sample: {sample}')

        return data_adapter.run_inference(model, sample, output_dir, **kwargs)

    def calculate_metrics(self, task_state) -> SampleScore:
        data_adapter = self._get_adapter(task_state.metadata)
        if not data_adapter:
            raise ValueError(f'No data adapter found for task state: {task_state}')

        return data_adapter.calculate_metrics(task_state)

    def aggregate_scores(self, sample_scores: List[SampleScore]):
        import pandas as pd
        from tabulate import tabulate

        data = []
        for sample_score in sample_scores:
            collection_info = sample_score.sample_metadata[DataCollection.INFO]
            main_score = sample_score.score.main_value
            main_metric = sample_score.score.main_score_name

            # use main score
            data.append(
                dict(
                    task_type=collection_info['task_type'],
                    categories=tuple(collection_info['categories']),
                    dataset_name=collection_info['dataset_name'],
                    subset_name=collection_info['subset_name'],
                    tags=collection_info['tags'],
                    sample_id=sample_score.sample_id,
                    metric=main_metric,
                    score=main_score
                )
            )

        df = pd.DataFrame(data)

        def aggregate_and_sort(df, group_by_cols):
            # aggregate by group_by_cols, and calculate average_score and count
            report_df = df.groupby(group_by_cols) \
                .agg(average_score=('score', 'mean'), count=('score', 'size')) \
                .reset_index()
            report_df['average_score'] = report_df['average_score'].round(4)
            report_df = report_df.sort_values(by='count', ascending=False) \
                .to_dict(orient='records')
            return report_df

        # multi-level aggregation
        subset_report_df = aggregate_and_sort(df, ['task_type', 'metric', 'dataset_name', 'subset_name'])
        dataset_report_df = aggregate_and_sort(df, ['task_type', 'metric', 'dataset_name'])
        task_report_df = aggregate_and_sort(df, ['task_type', 'metric'])

        # explode tags to multiple rows
        df_exploded_tags = df.explode('tags')
        tag_report_df = aggregate_and_sort(df_exploded_tags, ['tags', 'metric'])

        # process multi-level categories
        df_categories = df.copy()
        # multi-level aggregation for categories
        max_depth = df_categories['categories'].apply(len).max()
        for level in range(max_depth):
            df_categories[f'category{level}'] = df_categories['categories'].apply(
                lambda x: x[level] if len(x) > level else ''
            )
        category_report_df = aggregate_and_sort(
            df_categories, [f'category{level}' for level in range(max_depth)] + ['metric']
        )

        # convert to dict format
        report_dict = {
            'subset_level': subset_report_df,
            'dataset_level': dataset_report_df,
            'task_level': task_report_df,
            'tag_level': tag_report_df,
            'category_level': category_report_df,
        }

        # record report
        for level, data in report_dict.items():
            table = tabulate(data, headers='keys', tablefmt='pretty', showindex=False)
            logger.info(f'{level} Report:\n{table}')

        return df

    def generate_report(self, scores, model_name, output_dir, **kwargs) -> Report:
        df = scores[self.default_subset]
        report = ReportGenerator.gen_collection_report(df, self.name, model_name)
        return report
