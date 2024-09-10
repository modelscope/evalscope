from collections import defaultdict

from datasets import DatasetDict
from modelscope import MsDataset
from mteb import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

def load_retrieval_data(dataset_name, eval_splits):
    eval_split = eval_splits[0]
    dataset = MsDataset.load(dataset_name)
    qrels = MsDataset.load(dataset_name + '-qrels')[eval_split]

    corpus = {e['id']: {'text': e['text']} for e in dataset['corpus']}
    queries = {e['id']: e['text'] for e in dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['qid']][e['pid']] = e['score']

    corpus = DatasetDict({eval_split:corpus})
    queries = DatasetDict({eval_split:queries})
    relevant_docs = DatasetDict({eval_split:relevant_docs})
    return corpus, queries, relevant_docs


class T2Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="T2Retrieval",
        reference="https://arxiv.org/abs/2304.03679",
        description='T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/T2Retrieval',
            'revision': None,
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True

class MMarcoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMarcoRetrieval",
        reference="https://github.com/unicamp-dl/mMARCO",
        description='mMARCO is a multilingual version of the MS MARCO passage ranking dataset',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/MMarcoRetrieval',
            'revision': None,
        }
    )
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True

class DuRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DuRetrieval",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        description='A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/DuRetrieval',
            'revision': None,
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True

class CovidRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CovidRetrieval",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        description='COVID-19 news articles',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/CovidRetrieval',
            'revision': None,
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True

class CmedqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CmedqaRetrieval",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        description='Online medical consultation text',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/CmedqaRetrieval',
            'revision': None,
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True

class EcomRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EcomRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        description='Passage retrieval dataset collected from Alibaba search engine systems in ecom domain',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/EcomRetrieval',
            'revision': None,
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True

class MedicalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MedicalRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        description='Passage retrieval dataset collected from Alibaba search engine systems in medical domain',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/MedicalRetrieval',
            'revision': None,
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True

class VideoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VideoRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        description='Passage retrieval dataset collected from Alibaba search engine systems in video domain',
        type='Retrieval',
        category='s2p',
        eval_splits=['dev'],
        eval_langs=['zh'],
        main_score='ndcg_at_10',
        dataset={
            'path': 'C-MTEB/VideoRetrieval',
            'revision': None,
        }
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.metadata.dataset['path'],
                                                                            self.metadata.eval_splits)
        self.data_loaded = True