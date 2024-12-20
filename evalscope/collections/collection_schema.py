class CollectionSchema:

    def __init__(self):
        self.datasets = []

    def register_dataset(self, name, evaluator, weight=1, task_type='', tags=''):
        dataset_info = {'name': name, 'evaluator': evaluator, 'weight': weight, 'task_type': task_type, 'tags': tags}
        self.datasets.append(dataset_info)

    def get_evaluator(self, name):
        for dataset in self.datasets:
            if dataset['name'] == name:
                return dataset['evaluator']
        return None

    def get_datasets(self):
        return self.datasets
