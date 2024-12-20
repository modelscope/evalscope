import random


class DatasetSampler:

    def __init__(self, collection_schema):
        self.collection_schema = collection_schema
        self.datasets = collection_schema['datasets']
        self.total_weight = sum(dataset['weight'] for dataset in self.datasets)

    def sample_dataset(self):
        rand_value = random.uniform(0, self.total_weight)
        cumulative_weight = 0
        for dataset in self.datasets:
            cumulative_weight += dataset['weight']
            if rand_value <= cumulative_weight:
                return dataset['name']
        return None


# 示例使用
collection_schema = {
    'collection_name':
    'math',
    'datasets': [
        {
            'name': 'gsm8k',
            'weight': 1,
            'task_type': 'math',
            'tags': 'en,math'
        },
        {
            'name': 'competition_math',
            'weight': 2,
            'task_type': 'math',
            'tags': 'en,math'
        },
        # 可以继续添加其他数据集
    ]
}

sampler = DatasetSampler(collection_schema)

# 采样数据集
for _ in range(10):
    sampled_dataset = sampler.sample_dataset()
    print(f"Sampled dataset: {sampled_dataset}")
