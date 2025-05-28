from mteb import AbsTaskRetrieval
from mteb import HFDataLoader as CustomDataLoader
from mteb.abstasks.TaskMetadata import TaskMetadata
from typing import Optional


class CustomRetrieval(AbsTaskRetrieval):
    metadata: TaskMetadata
    ignore_identical_ids: bool = True

    def __init__(self, dataset_path: Optional[str] = 'custom_eval/text/retrieval', **kwargs):
        self.metadata = TaskMetadata(
            name='CustomRetrieval',
            description='CustomRetrieval Task',
            reference=None,
            dataset={
                'path': dataset_path,
                'revision': 'v1',
            },
            type='Retrieval',
            category='s2p',
            modalities=['text'],
            eval_splits=['test'],
            eval_langs=['cmn-Hans'],
            main_score='recall_at_5',
            date=None,
            domains=None,
            task_subtypes=None,
            license=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation='',
            descriptive_stats={},
        )
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict['dataset']['path']

        for split in kwargs.get('eval_splits', self.metadata_dict['eval_splits']):
            corpus, queries, qrels = CustomDataLoader(
                data_folder=dataset_path,
                streaming=False,
                keep_in_memory=False,
            ).load(split=split)
            # Conversion from DataSet
            queries = {query['id']: query['text'] for query in queries}
            corpus = {doc['id']: {'text': doc['text']} for doc in corpus}
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )

        self.data_loaded = True
