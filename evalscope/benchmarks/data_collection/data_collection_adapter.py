import math
import os
import re
from typing import Any, Optional

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import DEFAULT_DATASET_CACHE_DIR, EvalType, HubType
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


@Benchmark.register(
    name='data_collection',
    dataset_id='',  # dataset_id need to be set
    subset_list=['default'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template='',
)
class DataCollectionAdapter(DataAdapter):

    def __init__(self, **kwargs):
        """
        Data adapter for collection dataset.
        """
        super().__init__(**kwargs)

    def load(self,
             dataset_name_or_path: str = None,
             subset_list: list = None,
             work_dir: Optional[str] = DEFAULT_DATASET_CACHE_DIR,
             datasets_hub: str = HubType.MODELSCOPE,
             **kwargs) -> dict:
        """
        Load the dataset. Remote and local datasets are supported.
        """
        dataset_name_or_path = os.path.expanduser(dataset_name_or_path or self.dataset_id)
        subset_list = subset_list or self.subset_list

        # Try to load dataset from local disk
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset = jsonl_to_list(dataset_name_or_path)
            if len(dataset) == 0:
                raise ValueError(f'Local dataset is empty: {dataset_name_or_path}')
        else:
            from modelscope import dataset_snapshot_download

            # Load dataset from remote
            logger.info(f'Loading dataset from {datasets_hub}: > dataset_name: {dataset_name_or_path}')

            dataset_path = dataset_snapshot_download(
                dataset_name_or_path, cache_dir=work_dir, allow_file_pattern='*.jsonl')
            # find the jsonl file
            dataset_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jsonl')]
            dataset = jsonl_to_list(dataset_files[0])

        return dataset

    def get_gold_answer(self, input_d: Any) -> Any:
        return super().get_gold_answer(input_d)

    def match(self, gold: Any, pred: Any) -> Any:
        return super().match(gold, pred)

    def parse_pred_result(self, result: Any, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> Any:
        return super().parse_pred_result(result, raw_input_d, eval_type)
