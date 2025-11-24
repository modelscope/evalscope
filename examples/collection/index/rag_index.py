# This script creates a collection schema for RAG index evaluation
# and samples data from the defined datasets using a weighted sampler.

# 1. Define the collection schema
from evalscope.collections import CollectionSchema, DatasetInfo

schema = CollectionSchema(name='rag_index', datasets=[
    DatasetInfo(name='chinese_simpleqa', weight=0.3, task_type='knowledge', tags=['rag']),
    DatasetInfo(name='aa_lcr', weight=0.3, task_type='long_context', tags=['rag']),
    DatasetInfo(name='ifeval', weight=0.4, task_type='instruction_following', tags=['rag']),
])

schema.dump_json('examples/collection/index/rag_index.json')

# 2. Sample data from the collection schema using a weighted sampler
from evalscope.collections.sampler import WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

sampler = WeightedSampler(schema)
sampled_data = sampler.sample(count=10)
dump_jsonl_data(sampled_data, 'examples/collection/index/rag_index_sampled.jsonl')

# 3. Evaluate the sampled data using a RAG index benchmark