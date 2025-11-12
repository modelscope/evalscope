# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from typing import List, Optional, Union

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.metric.scorer import AggScore, Score
from evalscope.api.registry import get_metric, register_benchmark
from evalscope.constants import Tags
from evalscope.metrics import mean
from evalscope.utils.function_utils import thread_safe
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='evalmuse',
        pretty_name='EvalMuse',
        dataset_id='AI-ModelScope/T2V-Eval-Prompts',
        description='EvalMuse Text-to-Image Benchmark. Used for evaluating the quality '
        'and semantic alignment of finely generated images',
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['EvalMuse'],
        metric_list=['FGA_BLIP2Score'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class EvalMuseAdapter(Text2ImageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.metric_list
                   ) == 1 and self.metric_list[0] == 'FGA_BLIP2Score', 'Only FGA_BLIP2Score is supported for EvalMuse'

    @thread_safe
    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        # Get prediction and prompt from task state
        image_path = task_state.metadata.get('image_path', original_prediction)

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=image_path,
            prediction=image_path,
        )

        # Calculate scores for each configured metric
        try:
            metric_name = self.metric_list[0]
            metric_cls = get_metric(metric_name)
            metric_func = metric_cls()  # Initialize with parameters
            metric_score = metric_func(image_path, task_state.metadata)[0]

            for k, v in metric_score.items():
                score.value[f'{metric_name}:{k}'] = v.cpu().item()
        except Exception as e:
            logger.error(f'Error calculating metric {metric_name}: {e}')
            score.value[metric_name] = 0
            score.metadata[metric_name] = f'error: {str(e)}'

        return score

    def aggregate_scores(self, sample_scores) -> List[AggScore]:
        new_items = defaultdict(list)
        agg_list = []
        for sample_score in sample_scores:
            for metric_name, value in sample_score.score.value.items():
                metrics_prefix = metric_name.split(':')[0]
                category = metric_name.rpartition('(')[-1].split(')')[0]
                category = category.split('-')[0].lower()  # remove the suffix if exists
                new_items[f'{metrics_prefix}:{category}'].append(value)

        for k, v in new_items.items():
            agg_list.append(AggScore(metric_name=k, score=mean(v), num=len(v)))

        return agg_list
