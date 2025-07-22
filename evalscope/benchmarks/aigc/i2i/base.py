import threading
from typing import List, Optional, Union

from evalscope.benchmarks import DataAdapter
from evalscope.metrics import mean, metric_registry
from evalscope.utils.logger import get_logger

logger = get_logger()


class I2IBaseAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = None
        self._lock = threading.Lock()

    def _init_metrics(self):
        """Lazy initialization of metrics with thread safety"""
        if self.metrics is not None:
            return

        with self._lock:
            # Double-check pattern to avoid race conditions
            if self.metrics is not None:
                return

            logger.info(f'Initializing metrics: {self.metric_list}')
            self.metrics = {m: metric_registry.get(m).object() for m in self.metric_list}

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        # dummy prompt for general i2i
        return self.gen_prompt_data(prompt=input_d.get('prompt', ''), id=input_d.get('id', 0))

    def get_gold_answer(self, input_d: dict) -> str:
        # dummy gold answer for general i2i
        return input_d.get('prompt', '')

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        # dummy parse pred result for general i2i
        return result or raw_input_d.get('image_path', '')

    def match(self, gold: str, pred: str) -> dict:
        # Initialize metrics only once before first use
        self._init_metrics()

        # dummy match for general i2i
        # pred is the image path, gold is the prompt
        res = {}
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(images=[pred], texts=[gold])[0][0]
            if isinstance(score, dict):
                for k, v in score.items():
                    res[f'{metric_name}_{k}'] = v.cpu().item()
            else:
                res[metric_name] = score.cpu().item()
        return res

    def compute_metric(self, review_res_list: Union[List[dict], List[List[dict]]], **kwargs) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: List[dict]

        """
        items = super().compute_dict_metric(review_res_list, **kwargs)
        return [{'metric_name': k, 'score': mean(v), 'num': len(v)} for k, v in items.items()]
