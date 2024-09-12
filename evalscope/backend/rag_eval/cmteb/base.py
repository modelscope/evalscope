from collections import defaultdict

from datasets import DatasetDict
from modelscope import MsDataset
import datasets
from .tasks import CLS_DICT, CLS_RETRIEVAL

__all__ = ["TaskBase", "INSTRUCTIONS"]

INSTRUCTIONS = {
    "fiqa": "Instruction: Given a financial question, retrieve user replies that best answer the question. Query: ",
    "dbpedia": "Instruction: Given a query, retrieve relevant entity descriptions from DBPedia. Query: ",
    "CmedqaRetrieval": "Instruction: 为这个医疗问题检索相关回答。 Query: ",
    "nfcorpus": "Instruction: Given a question, retrieve relevant documents that best answer the question. Query: ",
    "touche2020": "Instruction: Given a question, retrieve detailed and persuasive arguments that answer the question. Query: ",
    "CovidRetrieval": "Instruction: 为这个问题检索相关政策回答。 Query: ",
    "scifact": "Instruction: Given a scientific claim, retrieve documents that support or refute the claim. Query: ",
    "scidocs": "Instruction: Given a scientific paper title, retrieve paper abstracts that are cited by the given paper. Query: ",
    "nq": "Instruction: Given a question, retrieve Wikipedia passages that answer the question. Query: ",
    "T2Retrieval": "Instruction: 为这个问题检索相关段落。 Query: ",
    "VideoRetrieval": "Instruction: 为这个电影标题检索相关段落。 Query: ",
    "DuRetrieval": "Instruction: 为这个问题检索相关百度知道回答。 Query: ",
    "MMarcoRetrieval": "Instruction: 为这个查询检索相关段落。 Query: ",
    "hotpotqa": "Instruction: Given a multi-hop question, retrieve documents that can help answer the question. Query: ",
    "quora": "Instruction: Given a question, retrieve questions that are semantically equivalent to the given question. Query: ",
    "climate-fever": "Instruction: Given a claim about climate change, retrieve documents that support or refute the claim. Query: ",
    "arguana": "Instruction: Given a claim, find documents that refute the claim. Query: ",
    "fever": "Instruction: Given a claim, retrieve documents that support or refute the claim. Query: ",
    "trec-covid": "Instruction: Given a query on COVID-19, retrieve documents that answer the query. Query: ",
    "msmarco": "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: ",
    "EcomRetrieval": "Instruction: 为这个查询检索相关商品标题。 Query: ",
    "MedicalRetrieval": "Instruction: 为这个医学问题检索相关回答。 Query: ",
}


class TaskBase:

    @staticmethod
    def get_tasks(task_names, instructions: dict = {}, **kwargs):
        INSTRUCTIONS.update(instructions)

        return [TaskBase.get_task(task_name, **kwargs) for task_name in task_names]

    @staticmethod
    def get_task(task_name, **kwargs):

        if task_name not in INSTRUCTIONS:
            INSTRUCTIONS.update({task_name: None})

        if task_name not in CLS_DICT:
            from mteb.overview import TASKS_REGISTRY

            task_cls = TASKS_REGISTRY[task_name]
            if task_cls.metadata.type != "Retrieval":
                task_cls.load_data = load_data
        else:
            task_cls = CLS_DICT[task_name]
            task_cls.load_data = load_data
        # init task instance
        task_instance = task_cls()
        return task_instance


def load_data(self, **kwargs):
    """Load dataset from the hub, compatible with ModelScope and Hugging Face."""
    if self.data_loaded:
        return

    limits = kwargs.get("limits", None)
    hub = kwargs.get("hub", "modelscope")
    name = self.metadata_dict.get("name")
    path = self.metadata_dict["dataset"].get("path")

    assert path is not None, "Path must be specified in dataset"

    # Loading the dataset based on the source hub
    if hub == "modelscope":
        import re

        path = re.sub(r"^mteb/", "MTEB/", path)
        dataset = MsDataset.load(path)
    else:
        dataset = datasets.load_dataset(**self.metadata_dict["dataset"])  # type: ignore

    if limits is not None:
        dataset = {
            split: dataset[split].select(range(min(limits, len(dataset[split]))))
            for split in dataset.keys()
        }

    if name in CLS_RETRIEVAL:
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            dataset,
            path,
            self.metadata_dict["eval_splits"],
        )

    self.dataset = dataset
    self.dataset_transform()
    self.data_loaded = True


def load_retrieval_data(dataset, dataset_name: str, eval_splits: list) -> tuple:
    eval_split = eval_splits[0]
    qrels = MsDataset.load(dataset_name + "-qrels")[eval_split]

    corpus = {e["id"]: {"text": e["text"]} for e in dataset["corpus"]}
    queries = {e["id"]: e["text"] for e in dataset["queries"]}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e["qid"]][e["pid"]] = e["score"]

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs
