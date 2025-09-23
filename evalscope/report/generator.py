import pandas as pd
from pandas import DataFrame
from typing import TYPE_CHECKING

from evalscope.constants import DataCollection
from evalscope.report.report import *

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter
    from evalscope.api.metric import AggScore


class ReportGenerator:

    @staticmethod
    def gen_collection_report(df: DataFrame, all_dataset_name: str, model_name: str) -> Report:
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
            metrics_list.append(Metric(name=metric_name, categories=categories))
        return Report(
            name=DataCollection.NAME, metrics=metrics_list, dataset_name=all_dataset_name, model_name=model_name
        )

    @staticmethod
    def generate_report(
        score_dict: Dict[str, List['AggScore']],
        model_name: str,
        data_adapter: 'DataAdapter',
        add_aggregation_name: bool = True
    ) -> Report:
        """
        Generate a report for a specific dataset based on provided subset scores.

        Args:
            subset_score_map (dict): A mapping from subset names to a list of score dictionaries.
            ```
            {
                'subset_name': [
                    AggScore={'metric_name': 'AverageAccuracy', 'score': 0.3389, 'num': 100},
                    AggScore={'metric_name': 'WeightedAverageAccuracy', 'score': 0.3389, 'num': 100}
                ],
                ...
            }
            ```
            data_adapter (DataAdapter): An adapter object for data handling.

        Returns:
            Report: A structured report object containing metrics, categories, and subsets.

            >>> report = gen_report(subset_score_map, "My Report", data_adapter, dataset_name="Dataset", model_name="Model")
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
                    if add_aggregation_name and agg_score_item.aggregation_name:
                        metric_name = f'{agg_score_item.aggregation_name}_{agg_score_item.metric_name}'
                    else:
                        metric_name = agg_score_item.metric_name

                    if isinstance(categories, str):
                        categories = [categories]
                    subsets.append(
                        dict(
                            name=subset_name,
                            score=agg_score_item.score,
                            num=agg_score_item.num,
                            metric_name=metric_name,
                            categories=tuple(categories)
                        )
                    )
            df = pd.DataFrame(subsets)
            return df

        df = flatten_subset()

        metrics_list = []
        for metric_name, group_metric in df.groupby('metric_name', sort=False):
            categories = []
            for category_name, group_category in group_metric.groupby('categories'):
                subsets = []
                for _, row in group_category.iterrows():
                    subsets.append(Subset(name=row['name'], score=row['score'], num=row['num']))

                categories.append(Category(name=category_name, subsets=subsets))

            metrics_list.append(Metric(name=metric_name, categories=categories))

        report = Report(
            name=report_name,
            metrics=metrics_list,
            dataset_name=dataset_name,
            model_name=model_name,
            dataset_description=data_adapter.description,
            dataset_pretty_name=data_adapter.pretty_name
        )
        return report
