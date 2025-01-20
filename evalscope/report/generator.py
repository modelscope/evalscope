import pandas as pd
from pandas import DataFrame

from evalscope.constants import DataCollection
from evalscope.report.utils import *


class ReportGenerator:

    @staticmethod
    def gen_report(subset_score_map: dict, report_name: str, **kwargs) -> Report:
        """
        Generate report for specific dataset.
        subset_score_map: e.g. {subset_name: [{'metric_name': 'AverageAccuracy', 'score': 0.3389, 'num': 100}, {'metric_name': 'WeightedAverageAccuracy', 'score': 0.3389, 'num': 100}]}
        category_map: e.g. {'subset_name': ['category_name1', 'category_name2'], ...}
        metric_list: e.g. [{'object': AverageAccuracy, 'name': 'AverageAccuracy'}, {'object': 'WeightedAverageAccuracy', 'name': 'WeightedAverageAccuracy'}]
        """  # noqa: E501

        dataset_name = kwargs.get('dataset_name', None)
        model_name = kwargs.get('model_name', None)
        category_map = kwargs.get('category_map', {})

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
        for metric_name, group_metric in df.groupby('metric_name'):
            categories = []
            for category_name, group_category in group_metric.groupby('categories'):
                subsets = []
                for _, row in group_category.iterrows():
                    subsets.append(Subset(name=row['name'], score=row['score'], num=row['num']))

                categories.append(Category(name=category_name, subsets=subsets))

            metrics_list.append(Metric(name=metric_name, categories=categories))

        report = Report(name=report_name, metrics=metrics_list, dataset_name=dataset_name, model_name=model_name)
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
