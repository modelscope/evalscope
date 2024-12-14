import datasets
from collections import defaultdict
from datasets import DatasetDict
from modelscope import MsDataset
from mteb import AbsTask
from typing import List

from evalscope.backend.rag_eval.cmteb.tasks import CLS_CUSTOM, CLS_DICT, CLS_RETRIEVAL
from evalscope.constants import HubType

__all__ = ['TaskBase']


class TaskBase:

    @staticmethod
    def get_tasks(task_names, **kwargs) -> List[AbsTask]:

        return [TaskBase.get_task(task_name, **kwargs) for task_name in task_names]

    @staticmethod
    def get_task(task_name, **kwargs) -> AbsTask:

        if task_name in CLS_CUSTOM:
            task_cls = CLS_CUSTOM[task_name]
        elif task_name in CLS_DICT:
            task_cls = CLS_DICT[task_name]
            task_cls.load_data = load_data
        else:
            from mteb.overview import TASKS_REGISTRY

            task_cls = TASKS_REGISTRY[task_name]
            if task_cls.metadata.type != 'Retrieval':
                task_cls.load_data = load_data

        # init task instance
        task_instance = task_cls(**kwargs)
        return task_instance


def load_data(self, **kwargs):
    """Load dataset from the hub, compatible with ModelScope and Hugging Face."""
    if self.data_loaded:
        return

    limits = kwargs.get('limits', None)
    hub = kwargs.get('hub', HubType.MODELSCOPE)
    name = self.metadata_dict.get('name')
    path = self.metadata_dict['dataset'].get('path')

    assert path is not None, 'Path must be specified in dataset'

    # Loading the dataset based on the source hub
    if hub == HubType.MODELSCOPE:
        import re

        path = re.sub(r'^mteb/', 'MTEB/', path)
        dataset = MsDataset.load(path)
    else:
        dataset = datasets.load_dataset(**self.metadata_dict['dataset'])  # type: ignore

    if limits is not None:
        dataset = {split: dataset[split].select(range(min(limits, len(dataset[split])))) for split in dataset.keys()}

    if name in CLS_RETRIEVAL:
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            dataset,
            path,
            self.metadata_dict['eval_splits'],
        )

    self.dataset = dataset
    self.dataset_transform()
    self.data_loaded = True


def load_retrieval_data(dataset, dataset_name: str, eval_splits: list) -> tuple:
    eval_split = eval_splits[0]
    qrels = MsDataset.load(dataset_name + '-qrels')[eval_split]

    corpus = {e['id']: {'text': e['text']} for e in dataset['corpus']}
    queries = {e['id']: e['text'] for e in dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['qid']][e['pid']] = e['score']

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs
