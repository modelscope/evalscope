from typing import Any, Dict, Iterator

from evalscope.api.dataset import DatasetSource
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('open_decisions')
class OpenDecisionsDatasetPlugin(DatasetPluginBase):
    """Replay the OpenDecisions benchmark as System One decision requests.

    Loads the same public dataset used by ``evalscope eval --datasets open_decisions``
    and yields complete request bodies, so ``--api systemone`` can measure a
    decision service without a hand-built ``line_by_line`` file.
    """

    dataset_source = DatasetSource(dataset_id='evalscope/OpenDecisions')

    def build_messages(self) -> Iterator[Dict[str, Any]]:
        dataset = self.load_hub_dataset(split='test', subset='default')
        for record in dataset:
            question = {key: value for key, value in record['question'].items() if value is not None}
            yield {'state': record['state'], 'questions': {'answer': question}}
