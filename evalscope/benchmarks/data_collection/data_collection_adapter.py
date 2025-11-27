import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

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
        pretty_name='Data-Collection',
        dataset_id='',  # dataset_id need to be set
        description='Custom Data collection, mixing multiple evaluation datasets for '
        'a unified evaluation, aiming to use less data to achieve a more comprehensive '
        'assessment of the model\'s capabilities. '
        '[Usage Reference](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html)',
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

        # Iterate through each sample in the dataset
        dataset = self.test_dataset[self.default_subset]
        for sample in dataset:
            collection_info = sample.metadata.get(DataCollection.INFO, {})
            dataset_name = collection_info.get('dataset_name', '')
            subset_name = collection_info.get('subset_name', '')
            # create id mapping
            self.dataset_name_map[dataset_name][subset_name].append(sample.id)

            # Initialize dataset adapter
            if dataset_name not in self.dataset_adapters:
                config = copy.deepcopy(self._task_config)
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
        # Build sample-level dataframe (includes per-sample weight)
        df = self._build_sample_dataframe(sample_scores)

        # Compute all reports from sample-level data; macro is hierarchical where applicable
        subset_report_df = self._group_and_compute(df, ['task_type', 'dataset_name', 'subset_name'])
        # Only keep micro_avg. for subset level (drop macro_avg. and weighted_avg.)
        subset_report_df = [{
            k: v
            for k, v in row.items()
            if k not in ('macro_avg.', 'weighted_avg.')
        }
                            for row in subset_report_df]  # noqa
        dataset_report_df = self._group_and_compute(df, ['task_type', 'dataset_name'], macro_child='subset_name')
        task_report_df = self._group_and_compute(df, ['task_type'], macro_child='subset_name')
        tag_report_df = self._build_tag_level_report(df)
        category_report_df = self._build_category_level_report(df)

        report_dict = {
            'subset_level': subset_report_df,
            'dataset_level': dataset_report_df,
            'task_level': task_report_df,
            'tag_level': tag_report_df,
            'category_level': category_report_df,
            'df': df,
        }
        return report_dict

    def generate_report(self, scores, model_name, output_dir, **kwargs) -> Report:
        import json
        from tabulate import tabulate

        df_dict = scores[self.default_subset]
        detailed_dict = {}
        # Log all levels as pretty tables
        for level, data_item in df_dict.items():
            if level == 'df':
                continue
            table = tabulate(data_item, headers='keys', tablefmt='pretty', showindex=False)
            detailed_dict[level] = data_item
            logger.info(f'{level} Report:\n{table}')

        # Save detailed report dataframe as JSON
        with open(os.path.join(output_dir, DataCollection.REPORT_NAME), 'w') as f:
            json.dump(detailed_dict, f, indent=2)

        report = ReportGenerator.gen_collection_report(df_dict['df'], self.name, model_name)
        return report

    # ---------------------------------------------------------------------
    # Aggregation helpers
    # ---------------------------------------------------------------------

    def _build_sample_dataframe(self, sample_scores: List[SampleScore]):
        """Convert sample scores into a flat dataframe used for all aggregations."""
        import pandas as pd

        records: List[Dict[str, Any]] = []
        for sample_score in sample_scores:
            collection_info = sample_score.sample_metadata[DataCollection.INFO]
            main_score = sample_score.score.main_value
            main_metric = sample_score.score.main_score_name
            sample_weight = float(collection_info.get('weight', 1.0))

            # Each row represents one sample
            records.append({
                'task_type': collection_info['task_type'],
                'categories': tuple(collection_info['categories']),
                'dataset_name': collection_info['dataset_name'],
                'subset_name': collection_info['subset_name'],
                'tags': collection_info['tags'],
                'sample_id': sample_score.sample_id,
                'metric': main_metric,
                'score': main_score,
                'sample_weight': sample_weight,
            })
        # NOTE: All sample weights are assumed (as per new requirement) to sum to ~1 globally.
        return pd.DataFrame(records)

    def _group_and_compute(self, df, group_cols, macro_child: Optional[str] = None):
        """
        Generic aggregation using per-sample weights.
        - weighted_avg.: sum(score * sample_weight) / sum(sample_weight) (fallback to unweighted mean)
        - micro_avg.: unweighted mean over samples in the group
        - macro_avg.: mean of child-group micro_avg. if macro_child provided (hierarchical),
                      otherwise equals micro_avg.
        """
        import pandas as pd

        rows = []
        grouped = df.groupby(group_cols)
        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys, )
            base = {col: key for col, key in zip(group_cols, keys)}

            scores = g['score']
            # Use provided sample weights if present; otherwise default to 1.0
            weights = g['sample_weight'] if 'sample_weight' in g.columns else pd.Series([1.0] * len(g), index=g.index)
            total_w = float(weights.sum())

            if total_w == 0.0:
                weighted = float(scores.mean())
            else:
                weighted = float((scores * weights).sum() / total_w)

            micro = float(scores.mean())

            # Hierarchical macro: average of child group micro means (e.g., subsets under a dataset)
            if macro_child and (macro_child in g.columns) and (macro_child not in group_cols):
                child_groups = g.groupby(macro_child)
                child_micros = [float(cg['score'].mean()) for _, cg in child_groups]
                macro = float(pd.Series(child_micros).mean()) if child_micros else micro
            else:
                macro = micro

            base['micro_avg.'] = round(micro, 4)
            base['macro_avg.'] = round(macro, 4)
            base['weighted_avg.'] = round(weighted, 4)
            base['count'] = int(len(g))
            rows.append(base)

        rows_sorted = sorted(rows, key=lambda r: r['count'], reverse=True)
        return rows_sorted

    def _build_tag_level_report(self, df):
        """
        Tag-level report using per-sample weights.
        Macro is the mean of subset-level micro averages.
        """
        df_exploded = df.explode('tags')
        return self._group_and_compute(df_exploded, ['tags'], macro_child='subset_name')

    def _build_category_level_report(self, df):
        """
        Category-level hierarchical aggregation using sample-level weights.
        Macro is the mean of subset-level micro averages.
        """
        df_categories = df.copy()
        max_depth = df_categories['categories'].apply(len).max()
        for level in range(max_depth):
            col = f'category{level}'
            df_categories[col] = df_categories['categories'].apply(lambda x, i=level: x[i] if len(x) > i else '')
        category_cols = [f'category{level}' for level in range(max_depth)]
        return self._group_and_compute(df_categories, category_cols, macro_child='subset_name')
