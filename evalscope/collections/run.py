# 导入必要的模块
from evalscope.collections.collection_schema import CollectionSchema
from evalscope.collections.data_generator import generate_mixed_dataset, save_to_jsonl
from evalscope.collections.data_sampler import DatasetSampler
from evalscope.collections.evaluators import EvaluatorCollection


# 假设 Gsm8kEvaluator 和 CompetitionMathEvaluator 已定义
class Gsm8kEvaluator:

    def evaluate(self, sample):
        # 实现评估逻辑
        return {'score': 1.0}


class CompetitionMathEvaluator:

    def evaluate(self, sample):
        # 实现评估逻辑
        return {'score': 2.0}


# 创建集合架构
schema = CollectionSchema()
schema.register_dataset('gsm8k', Gsm8kEvaluator(), weight=1, task_type='math', tags='en,math')
schema.register_dataset('competition_math', CompetitionMathEvaluator(), weight=2, task_type='math', tags='en,math')

# 创建评估器集合
evaluator_collection = EvaluatorCollection(schema)
evaluator_collection.add_evaluator('gsm8k')
evaluator_collection.add_evaluator('competition_math')

# 示例数据
samples = [{
    'id': 1,
    'row_data': {
        'prompt': '一艘大船运了6次货，一艘小船运了9次货，大船每次运30吨，小船每次运12吨，大船和小船一共运了多少吨货？',
        'answer': ['288.0']
    },
    'tags': 'zh_cn,math',
    'task_type': 'math23k',
    'source': 'gsm8k'
}, {
    'id': 2,
    'row_data': {
        'prompt': '0.054-(-0.045)=',
        'answer': ['0.0990']
    },
    'tags': 'en,math',
    'task_type': 'math401',
    'source': 'competition_math'
}]

# 生成混合数据集
mixed_data = generate_mixed_dataset(evaluator_collection, samples)

# 保存为JSONL文件
save_to_jsonl(mixed_data, 'mixed_dataset.jsonl')
