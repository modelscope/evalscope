import re
from typing import Any, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import LLMJudge, Metric, mean, metric_registry
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

GRADER_TEMPLATE = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

以下是【未尝试】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
    - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
    - 预测答案“3520”和“3600”均为【错误】。
    - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题“菱镁矿的主要化学成分是什么？”标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()  # noqa E501

SUBSET_LIST = ['中华文化', '人文与社会科学', '工程、技术与应用科学', '生活、艺术与文化', '社会', '自然与自然科学']


@Benchmark.register(
    name='chinese_simpleqa',
    pretty_name='Chinese SimpleQA',
    subset_list=SUBSET_LIST,
    dataset_id='AI-ModelScope/Chinese-SimpleQA',
    metric_list=['is_correct', 'is_incorrect', 'is_not_attempted'],
    few_shot_num=0,
    train_split=None,
    eval_split='train')
class ChineseSimpleQAAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # register metrics
        metric_registry.register(Metric(name='is_correct', object=mean))
        metric_registry.register(Metric(name='is_incorrect', object=mean))
        metric_registry.register(Metric(name='is_not_attempted', object=mean))

        # whether to use LLM as a judge
        self.llm_as_a_judge = True

    def load(self, **kwargs):
        kwargs['subset_list'] = ['default']
        data_dict = super().load(**kwargs)
        return self.reformat_subset(data_dict, subset_key='primary_category', format='{}')

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        question = input_d['question']
        return self.gen_prompt_data(question)

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d['answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, **kwargs) -> str:
        return result.strip()

    def match(self, gold: str, pred: str) -> float:
        # simple match
        logger.warning(f'Please use LLMJudge to match the result for {self.name}')
        is_correct = 1 if gold.lower().strip() == pred.lower().strip() else 0
        is_incorrect = not is_correct
        is_not_attempted = 0
        return {
            'is_correct': is_correct,
            'is_incorrect': is_incorrect,
            'is_not_attempted': is_not_attempted,
        }

    def llm_match(self, gold: Any, pred: Any, judge: LLMJudge, **kwargs) -> dict:
        raw_input = kwargs.get('raw_input', None)
        question = raw_input['question']
        # get grading response
        prompt = GRADER_TEMPLATE.format(question=question, target=gold, predicted_answer=pred)
        system_prompt = '你是一个智能助手，请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。'
        grading_response = judge(prompt, system_prompt)
        # parse grading response
        match = re.search(r'(A|B|C)', grading_response)
        res = match.group(0) if match else 'C'
        return {
            'is_correct': 1 if res == 'A' else 0,
            'is_incorrect': 1 if res == 'B' else 0,
            'is_not_attempted': 1 if res == 'C' else 0,
            'judge_response': grading_response,
        }

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [{'is_correct': 1, 'is_incorrect': 0, 'is_not_attempted': 0}, ...]
        """
        # zip dict answers
        res_dict = super().compute_dict_metric(review_res_list, **kwargs)

        return super().compute_metric(res_dict, **kwargs)
