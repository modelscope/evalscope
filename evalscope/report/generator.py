import pandas as pd
from pandas import DataFrame
from typing import TYPE_CHECKING

from evalscope.constants import DataCollection
from evalscope.report.utils import *

if TYPE_CHECKING:
    from evalscope.benchmarks import DataAdapter


class ReportGenerator:

    @staticmethod
    def gen_report(subset_score_map: dict, model_name: str, data_adapter: 'DataAdapter', **kwargs) -> Report:
        """
        Generate a report for a specific dataset based on provided subset scores.

        Args:
            subset_score_map (dict): A mapping from subset names to a list of score dictionaries.
                    {
                        'subset_name': [
                            {'metric_name': 'AverageAccuracy', 'score': 0.3389, 'num': 100},
                            {'metric_name': 'WeightedAverageAccuracy', 'score': 0.3389, 'num': 100}
                        ],
                        ...
                    }
            report_name (str): The name of the report to generate.
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
            for subset_name, scores in subset_score_map.items():
                for score_item in scores:
                    categories = category_map.get(subset_name, ['default'])
                    if isinstance(categories, str):
                        categories = [categories]
                    subsets.append(
                        dict(
                            name=subset_name,
                            score=score_item['score'],
                            num=score_item['num'],
                            metric_name=score_item['metric_name'],
                            categories=tuple(categories)))
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
            dataset_pretty_name=data_adapter.pretty_name)
        return report

    @staticmethod
    def gen_collection_report(df: DataFrame, all_dataset_name: str, model_name: str) -> Report:
        categories = []
        for category_name, group_category in df.groupby('categories'):
            subsets = []
            for (dataset_name, subset_name), group_subset in group_category.groupby(['dataset_name', 'subset_name']):
                avg_score = group_subset['score'].mean()
                num = group_subset['score'].count()
                subsets.append(Subset(name=f'{dataset_name}/{subset_name}', score=float(avg_score), num=int(num)))

            categories.append(Category(name=category_name, subsets=subsets))
        return Report(
            name=DataCollection.NAME,
            metrics=[Metric(name='Average', categories=categories)],
            dataset_name=all_dataset_name,
            model_name=model_name)
