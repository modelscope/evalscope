import pandas as pd
from pandas import DataFrame
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from evalscope.api.metric.semantics import MetricIdentity, MetricSelector
from evalscope.constants import DataCollection
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.metrics.semantics.identity import canonicalize_producer_identity
from evalscope.metrics.semantics.resolver import attribute_metric_roles
from evalscope.report.report import *

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.metric import AggScore
    from evalscope.api.metric.semantics import MetricSemantics


class ReportGenerator:

    @staticmethod
    def gen_collection_report(
        df: DataFrame,
        all_dataset_name: str,
        model_name: str,
    ) -> Report:
        """Build a collection report."""
        resolver = get_semantics_resolver()
        metrics_list = []
        for metric_name, group_metric in df.groupby('metric', sort=False):
            categories = []
            for category_name, group_category in group_metric.groupby('categories'):
                subsets = []
                for (dataset_name, subset_name), group_subset in group_category.groupby(['dataset_name',
                                                                                         'subset_name']):
                    avg_score = group_subset['score'].mean()
                    num = group_subset['score'].count()
                    subsets.append(Subset(name=f'{dataset_name}/{subset_name}', score=float(avg_score), num=int(num)))
                categories.append(Category(name=category_name, subsets=subsets))
            identity = canonicalize_producer_identity(metric_name, 'mean')
            semantics = resolver.resolve(all_dataset_name, identity).semantics
            metrics_list.append(Metric(identity=identity, categories=categories, semantics=semantics))
        return Report(
            name=DataCollection.NAME,
            metrics=metrics_list,
            dataset_name=all_dataset_name,
            model_name=model_name,
            primary_metric_identity=None,
        )

    @staticmethod
    def generate_report(
        score_dict: Dict[str, List['AggScore']],
        model_name: str,
        data_adapter: 'DataAdapter',
    ) -> Report:
        """
        Generate a report for a specific dataset based on provided subset scores.

        Args:
            score_dict: A mapping from subset names to aggregated scores.
            ```
            {
                'subset_name': [
                    AggScore(metric_name='accuracy', aggregation='mean', score=0.3389, num=100),
                    AggScore(metric_name='f1', aggregation='mean', score=0.3389, num=100),
                ],
                ...
            }
            ```
            model_name: Name written into the report.
            data_adapter: Adapter providing benchmark metadata and primary metric selection.
        Returns:
            Report: A structured report object containing metrics, categories, and subsets.

            >>> report = ReportGenerator.generate_report(score_dict, 'Model', data_adapter)
        """  # noqa: E501

        dataset_name = data_adapter.name
        category_map = data_adapter.category_map
        report_name = f'{model_name}@{dataset_name}'

        def flatten_subset() -> DataFrame:
            """
            Flatten subset score map to a DataFrame.

            Example:
                        name  score  num   categories      metric_name
            0       ARC-Easy    0.5    2    [default]  AverageAccuracy
            1  ARC-Challenge    0.5    2    [default]  AverageAccuracy
            """
            subsets = []
            for subset_name, agg_scores in score_dict.items():
                for agg_score_item in agg_scores:
                    categories = category_map.get(subset_name, ['default'])
                    identity = agg_score_item.identity

                    if isinstance(categories, str):
                        categories = [categories]
                    subsets.append(
                        dict(
                            name=subset_name,
                            score=agg_score_item.score,
                            num=agg_score_item.num,
                            identity_key=identity.key,
                            identity=identity,
                            categories=tuple(categories)
                        )
                    )
            df = pd.DataFrame(subsets)
            return df

        df = flatten_subset()

        if df.empty:
            raise ValueError(
                f'No scores were collected for dataset "{dataset_name}". '
                'Please check that samples are not all filtered out and that the aggregation step produces results.'
            )
        if 'identity_key' not in df.columns:
            raise KeyError(
                f'Column "identity_key" is missing from the score DataFrame for dataset "{dataset_name}". '
                f'Available columns: {list(df.columns)}'
            )

        identities = list({identity.key: identity for identity in df['identity']}.values())
        semantics_by_identity, primary_identity = ReportGenerator._resolve_semantics(
            benchmark_name=dataset_name,
            identities=identities,
            selector=data_adapter.primary_metric,
        )

        metrics_list = []
        for identity_key, group_metric in df.groupby('identity_key', sort=False):
            categories = []
            for category_name, group_category in group_metric.groupby('categories'):
                subsets = []
                for _, row in group_category.iterrows():
                    subsets.append(Subset(name=row['name'], score=row['score'], num=row['num']))

                categories.append(Category(name=category_name, subsets=subsets))

            identity = group_metric.iloc[0]['identity']
            semantics = semantics_by_identity[identity_key]
            metric = Metric(
                identity=identity,
                categories=categories,
                semantics=semantics,
            )
            metrics_list.append(metric)

        report = Report(
            name=report_name,
            metrics=metrics_list,
            dataset_name=dataset_name,
            model_name=model_name,
            dataset_description=data_adapter.description,
            dataset_pretty_name=data_adapter.pretty_name,
            primary_metric_identity=primary_identity,
        )
        return report

    @staticmethod
    def _resolve_semantics(
        benchmark_name: str,
        identities: List[MetricIdentity],
        selector: Optional[MetricSelector],
    ) -> Tuple[Dict[str, 'MetricSemantics'], Optional[MetricIdentity]]:
        """Resolve the semantics of every metric this report will contain.

        An undeclared metric degrades to a diagnostic, which shows the stored value without
        claiming a direction or unit and logs where to declare it. Aggregation and dynamic axes
        are already structured in each identity, so resolution never parses a final-name string.

        Args:
            benchmark_name: Benchmark name the scores belong to.
            identities: Distinct identities emitted by aggregation.
            selector: Structured primary selector from benchmark metadata.

        Returns:
            Identity key -> semantics mapping and the uniquely selected primary identity.
        """
        resolver = get_semantics_resolver()

        semantics_by_identity: Dict[str, 'MetricSemantics'] = {}
        for identity in identities:
            resolved = resolver.resolve(benchmark_name, identity)
            resolved.log_audit_messages()
            semantics_by_identity[identity.key] = resolved.semantics
        return attribute_metric_roles(identities, semantics_by_identity, selector)
