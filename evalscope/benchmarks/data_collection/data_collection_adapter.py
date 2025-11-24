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
        # Build base per-sample dataframe
        df = self._build_sample_dataframe(sample_scores)

        # NEW: subset-level aggregation base (atomic weighted unit)
        df_subset = self._aggregate_subset_level(df)

        # Subset-level (unchanged logic: simple per-sample mean inside each subset)
        subset_report_df = aggregate_and_sort(
            df_group=df,
            group_by_cols=['task_type', 'metric', 'dataset_name', 'subset_name'],
        )

        # Dataset-level: weighted over subset means (subset weights)
        dataset_report_df = aggregate_and_sort_weighted(
            df_group=df_subset,
            group_by_cols=['task_type', 'metric', 'dataset_name'],
            score_col='average_score',
            weight_col='dataset_weight',
            count_col='count',
        )

        # Task-level: weighted over all subsets
        task_report_df = self._build_task_level_report(df_subset)

        # Tag-level: weighted over subset-level tag means
        tag_report_df = self._build_tag_level_report(df)

        # Category-level: weighted over subset-level category means
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
            dataset_weight = float(collection_info.get('weight', 1.0))  # now treated as subset weight

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
                'dataset_weight': dataset_weight,
                'sample_weight': dataset_weight,
            })

        return pd.DataFrame(records)

    def _aggregate_subset_level(self, df):
        """
        Aggregate raw sample dataframe to subset-level (dataset + subset) units.
        Each unit has: average_score (simple mean of samples), count (sample count), dataset_weight (subset weight).
        """
        return (
            df.groupby(['task_type', 'metric', 'dataset_name', 'subset_name'], as_index=False).agg(
                average_score=('score', 'mean'),
                count=('score', 'size'),
                dataset_weight=('dataset_weight', 'first'),
            )
        )

    def _build_task_level_report(self, df_subset):
        """
        Build task-level report weighted over subset units.
        df_subset is the subset-level aggregated dataframe.
        """
        task_report_df = aggregate_and_sort_weighted(
            df_group=df_subset,
            group_by_cols=['task_type', 'metric'],
            score_col='average_score',
            weight_col='dataset_weight',
            count_col='count',
        )
        return task_report_df

    def _build_tag_level_report(self, df):
        """
        Build tag-level report weighted over subset units derived from exploded tags.
        """
        df_exploded = df.explode('tags')
        df_tag_subset = (
            df_exploded.groupby(['tags', 'metric', 'dataset_name', 'subset_name'], as_index=False).agg(
                average_score=('score', 'mean'),
                count=('score', 'size'),
                dataset_weight=('dataset_weight', 'first'),
            )
        )
        tag_report_df = aggregate_and_sort_weighted(
            df_group=df_tag_subset,
            group_by_cols=['tags', 'metric'],
            score_col='average_score',
            weight_col='dataset_weight',
            count_col='count',
        )
        return tag_report_df

    def _build_category_level_report(self, df):
        """
        Build category-level report using subset-weighted aggregation over hierarchical categories.
        """
        df_categories = df.copy()
        max_depth = df_categories['categories'].apply(len).max()
        for level in range(max_depth):
            col = f'category{level}'
            df_categories[col] = df_categories['categories'].apply(lambda x, i=level: x[i] if len(x) > i else '')

        category_cols = [f'category{level}' for level in range(max_depth)]
        df_cat_subset = (
            df_categories.groupby(category_cols + ['metric', 'dataset_name', 'subset_name'], as_index=False).agg(
                average_score=('score', 'mean'),
                count=('score', 'size'),
                dataset_weight=('dataset_weight', 'first'),
            )
        )

        category_report_df = aggregate_and_sort_weighted(
            df_group=df_cat_subset,
            group_by_cols=category_cols + ['metric'],
            score_col='average_score',
            weight_col='dataset_weight',
            count_col='count',
        )
        return category_report_df


def aggregate_and_sort(df_group, group_by_cols):
    """
    Subset-level aggregation (unweighted).
    Changes:
      - Remove weighted_average_score.
      - Add weight column (dataset_weight per subset, first sample fallback 1.0).
    """
    import pandas as pd

    # Include dataset_weight to expose subset weight
    agg_dict = {
        'average_score': ('score', 'mean'),
        'count': ('score', 'size'),
        'weight': ('dataset_weight', 'first'),
    }
    # Graceful fallback if dataset_weight column missing
    missing_weight = 'dataset_weight' not in df_group.columns
    if missing_weight:
        df_group = df_group.copy()
        df_group['dataset_weight'] = 1.0
    report_df: pd.DataFrame = df_group.groupby(group_by_cols).agg(**agg_dict).reset_index()
    report_df['average_score'] = report_df['average_score'].round(4)
    # Sort by count desc
    report_df = report_df.sort_values(by='count', ascending=False)
    return report_df.to_dict(orient='records')


def aggregate_and_sort_weighted(
    df_group,
    group_by_cols: List[str],
    score_col: str,
    weight_col: str,
    count_col: str,
) -> List[Dict[str, Any]]:
    """
    Aggregate with two means:
      weighted_average_score: weighted by provided weights.
      sample_average_score: weighted by sample counts (count_col).
    Input df_group is already aggregated at subset level (atomic unit) with per-subset mean in score_col.
    """
    import pandas as pd

    def weighted_mean(group_df: 'pd.DataFrame') -> float:
        weights = group_df[weight_col]
        scores = group_df[score_col]
        total_weight = float(weights.sum())
        if total_weight == 0.0:
            return float(scores.mean())
        return float((scores * weights).sum() / total_weight)

    def sample_count_mean(group_df: 'pd.DataFrame') -> float:
        counts = group_df[count_col]
        scores = group_df[score_col]
        total = float(counts.sum())
        if total == 0.0:
            return float(scores.mean())
        return float((scores * counts).sum() / total)

    grouped = df_group.groupby(group_by_cols)
    rows: List[Dict[str, Any]] = []
    for keys, group_df in grouped:
        if not isinstance(keys, tuple):
            keys = (keys, )
        base: Dict[str, Any] = {col: key for col, key in zip(group_by_cols, keys)}
        base['weighted_average_score'] = round(weighted_mean(group_df), 4)
        base['sample_average_score'] = round(sample_count_mean(group_df), 4)
        base['count'] = int(group_df[count_col].sum())
        rows.append(base)

    rows_sorted = sorted(rows, key=lambda r: r['count'], reverse=True)
    return rows_sorted
