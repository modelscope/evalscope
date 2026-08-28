import json
from typing import Any, Dict, List, Literal, Sequence, Type

from pydantic import BaseModel, Field, model_validator

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags

SUBSET_LIST = [
    'global_economics_and_finance',
    'global_healthcare_and_medicine',
    'global_industry',
    'global_law',
    'global_natural_sciences',
    'cn_economics_and_finance',
    'cn_healthcare_and_medicine',
    'cn_industry',
    'cn_law',
    'cn_natural_sciences',
]

_DOMAIN_SUBSETS = {
    ('global', 'Economics and Finance'): 'global_economics_and_finance',
    ('global', 'Economics & Finance'): 'global_economics_and_finance',
    ('global', 'Healthcare and Medicine'): 'global_healthcare_and_medicine',
    ('global', 'Industry'): 'global_industry',
    ('global', 'Law'): 'global_law',
    ('global', 'Natural Sciences'): 'global_natural_sciences',
    ('cn', '经济金融'): 'cn_economics_and_finance',
    ('cn', '医疗健康'): 'cn_healthcare_and_medicine',
    ('cn', '工业'): 'cn_industry',
    ('cn', '法律'): 'cn_law',
    ('cn', '自然科学'): 'cn_natural_sciences',
}

_JUDGE_PROMPT = """## 角色与核心任务

**角色：** 你是一名公正、精确且严格的AI响应评估裁判。

**核心任务：** 根据详细的评分标准（Rubric），对大型语言模型的回复（modelResponse）进行逐项评估。你需要判断模型回复是否符合评分标准中的具体描述。

**评估原则：**

1. **寻找直接证据：** 评估必须严格依据模型回复中**实际存在**的文本证据。不能进行主观猜测或过度解读。只有明确指出的内容才算数。

2. **二元判断（是/否）：** 每一个 Rubric 项的评估结果只有两种：
   - **命中 (是)**：模型回复中确实包含或命中了rubric描述的内容或特征。
   - **未命中 (否)**：模型回复中没有包含或没有命中rubric描述的内容或特征。

   *注意：这一逻辑通用于正分项（得分点）和负分项（扣分点）。只要rubric里的描述发生了，就是“命中/是”。*

3. **评分规则：**
   - **正向得分项（rubricWeight > 0）**：输出“是”代表得到该项分数，输出“否”代表不得分（0分）。
   - **负向扣分项（rubricWeight < 0）**：输出“是”代表需要扣分（扣除对应分值），输出“否”代表不扣分（0分）。

---

## 评分步骤

请保持冷静和专注，严格遵循以下步骤：

**步骤一：理解上下文**
仔细阅读用户问题（prompt）、模型回复（modelResponse）、评分标准（rubric）。

**步骤二：判断是否命中**
对照评分标准（rubric）的描述，检查模型回复：
- 如果回复中**出现**了rubric描述的情况（无论是好的行为还是坏的错误），状态为 **“命中”**，结论输出 **“是”**。
- 如果回复中**未出现**rubric描述的情况，状态为 **“未命中”**，结论输出 **“否”**。

**步骤三：自我反思与格式化**
- 检查证据是否充分支持你的“是/否”判断。
- 对每条 Rubric 输出 `rubric_id`、`status` 和 `justification`，其中 `status` 必须是“是”或“否”。
- 将所有逐项结果放入单个 JSON 对象的 `results` 列表中。

---

## 输入信息

### 用户问题（prompt）
{prompt}

---

### AI回复（modelResponse）
{model_response}

---

### 评分项（Rubrics）
{rubrics}

---

请逐条评估所有Rubric。"""

_DESCRIPTION = """
## Overview

$OneMillion-Bench ($1M-Bench) evaluates how well language models and agents complete economically valuable,
expert-level professional work. The public release contains 400 bilingual tasks written and reviewed by domain
experts across finance, healthcare, industry, law, and natural science.

## Task Description

- **Task Type**: Open-ended professional question answering with rubric-based LLM judging
- **Input**: A realistic, context-heavy professional request in Chinese or English
- **Output**: A complete free-form professional analysis or deliverable
- **Domain**: Economics and finance, healthcare and medicine, industry, law, and natural sciences

## Key Features

- 400 zero-shot tasks, balanced across Chinese and global tracks and five professional domains (40 tasks per
  language-domain subset)
- Each task has 11-37 expert-authored criteria covering factual information, analytical reasoning, instruction
  following, and structure and formatting
- Every task includes both positive criteria and negative penalties, with rubric weights ranging from -20 to 12 in
  the hosted release
- Samples are exposed as ten language-domain subsets so both the paper's language tracks and domain breakdowns are
  visible in EvalScope reports

## Evaluation Notes

- An LLM judge is required. Configure `judge.strategy='llm'` (or `'auto'`) and `judge.models`; the official harness
  currently recommends Gemini 3.1 Pro Preview, but judge identity affects absolute scores and is not hard-coded here
- All rubrics for one response are judged together in one request using the official binary hit/miss instructions
- `expert_score` is the weighted sum of hit rubrics divided by the sum of positive weights, clipped to `[0, 1]`;
  `pass_rate` is 1 when `expert_score >= 0.7`, otherwise 0
- Judge replies must contain every rubric exactly once. Malformed replies and transport failures are excluded instead
  of being silently converted to zero scores
- The official study compares vanilla models, search-enabled models, and deep-research agents separately. This native
  adapter performs the benchmark's one-turn generation path; results from external tool-using agents are comparable
  only when their final responses are evaluated under the same judge configuration
- Tasks often require long, cited reports. Configure sufficiently large generation and judge `max_tokens` values;
  with one judge and one repeat, a full run performs 400 generation calls and 400 judge calls

Resources: [Paper](https://arxiv.org/abs/2603.07980) |
[GitHub](https://github.com/humanlaya/OneMillion-Bench) |
[Dataset](https://modelscope.cn/datasets/evalscope/OneMillion-Bench)
"""


class RubricJudgment(BaseModel):
    rubric_id: int
    status: Literal[
        '是', '否', 'Yes', 'No', 'yes', 'no', 'Y', 'N', 'YES', 'NO', 'true', 'false', 'True', 'False', '命中', '未命中'
    ]
    justification: str


def _grading_result_model(expected_ids: Sequence[int]) -> Type[BaseModel]:
    expected = set(expected_ids)

    class GradingResult(BaseModel):
        results: List[RubricJudgment] = Field(description='One judgment for every rubric in the prompt.')

        @model_validator(mode='after')
        def validate_rubric_ids(self) -> 'GradingResult':
            actual = [result.rubric_id for result in self.results]
            if len(actual) != len(set(actual)):
                raise ValueError('rubric_id values must be unique')
            if set(actual) != expected:
                raise ValueError(f'rubric_id values must exactly match {sorted(expected)}')
            return self

    return GradingResult


def _format_rubrics(rubrics: Sequence[Dict[str, Any]]) -> str:
    return '\n\n'.join(
        f'**Rubric {rubric["rubric_number"]}**\n'
        f'rubricDetail: {rubric["rubric_detail"]}\n'
        f'rubricWeight: {int(rubric["rubric_weight"]):+d}分'
        for rubric in rubrics
    )


def _is_hit(status: str) -> bool:
    return status in {'是', 'Yes', 'yes', 'Y', 'YES', 'true', 'True', '命中'}


@register_benchmark(
    BenchmarkMeta(
        name='one_million_bench',
        pretty_name='$OneMillion-Bench',
        dataset_id='evalscope/OneMillion-Bench',
        tags=[Tags.AGENT, Tags.KNOWLEDGE, Tags.QA, Tags.REASONING, Tags.MULTI_LINGUAL],
        description=_DESCRIPTION,
        paper_url='https://arxiv.org/abs/2603.07980',
        subset_list=SUBSET_LIST,
        eval_split='test',
        metric_list=['expert_score', 'pass_rate'],
        primary_metric='expert_score',
        evaluation_version='v1.0',
    )
)
class OneMillionBenchAdapter(DefaultDataAdapter):
    """Adapter for the official weighted-rubric $OneMillion-Bench evaluation."""

    scoring_policy = ScoringPolicy.JUDGE_ONLY

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reformat_subset = True
        self.category_map = {subset: 'CN' if subset.startswith('cn_') else 'Global' for subset in SUBSET_LIST}

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        rubrics = record.get('rubrics')
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError('$OneMillion-Bench record must contain a non-empty rubrics list.')

        tags = record.get('tags') or {}
        topics = tags.get('topics') or []
        if not topics:
            raise ValueError(f'$OneMillion-Bench record {record.get("id")!r} has no domain topic.')

        language = str(record.get('language', '')).strip()
        domain = str(topics[0]).strip()
        subset = _DOMAIN_SUBSETS.get((language, domain))
        if subset is None:
            raise ValueError(f'Unsupported $OneMillion-Bench language-domain pair: {(language, domain)!r}.')

        question = str(record['question'])
        messages = []
        system_prompt = str(record.get('system_prompt') or '').strip()
        if system_prompt:
            messages.append(ChatMessageSystem(content=system_prompt))
        messages.append(ChatMessageUser(content=question))

        return Sample(
            input=messages,
            target=json.dumps(rubrics, ensure_ascii=False),
            subset_key=subset,
            metadata={
                'id': record.get('id'),
                'case_id': record.get('case_id'),
                'language': language,
                'domain': domain,
                'topics': topics,
                'time_sensitivity': tags.get('time_sensitivity'),
                'question': question,
            },
        )

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        rubrics = json.loads(context.reference)
        rubric_ids = [int(rubric['rubric_number']) for rubric in rubrics]
        contract = OutputContract(schema_model=_grading_result_model(rubric_ids))

        def request(
            case: JudgeCase,
            placement: Any,
            completed_cases: Sequence[CaseVerdict],
            judge_context: JudgeContext,
        ) -> JudgeRequest:
            metadata = judge_context.task_state.metadata or {}
            prompt = _JUDGE_PROMPT.format(
                prompt=metadata['question'],
                model_response=judge_context.original_prediction,
                rubrics=_format_rubrics(case.metadata['rubrics']),
            )
            return JudgeRequest(messages=[ChatMessageUser(content=prompt + case.output_contract.instruction())])

        def reduce(case_verdicts: Sequence[CaseVerdict], judge_context: JudgeContext) -> ReducedVerdict:
            judgments = case_verdicts[0].value.results
            verdict_by_id = {judgment.rubric_id: judgment for judgment in judgments}
            positive_weight = sum(int(rubric['rubric_weight']) for rubric in rubrics if rubric['rubric_weight'] > 0)
            if positive_weight <= 0:
                raise ValueError('$OneMillion-Bench requires at least one positive-weight rubric.')

            rubric_scores = []
            raw_score = 0
            for rubric in rubrics:
                rubric_id = int(rubric['rubric_number'])
                weight = int(rubric['rubric_weight'])
                judgment = verdict_by_id[rubric_id]
                awarded = weight if _is_hit(judgment.status) else 0
                raw_score += awarded
                rubric_scores.append(
                    {
                        'rubric_id': rubric_id,
                        'status': judgment.status,
                        'weight': weight,
                        'score': awarded,
                        'justification': judgment.justification,
                    }
                )

            expert_score = min(max(raw_score / positive_weight, 0.0), 1.0)
            return ReducedVerdict(
                value={
                    'expert_score': expert_score,
                    'pass_rate': float(expert_score >= 0.7),
                },
                metadata={
                    'raw_score': raw_score,
                    'max_score': positive_weight,
                    'rubric_scores': rubric_scores,
                },
            )

        return JudgeDefinition.workflow(
            cases=[JudgeCase(case_id='rubrics', output_contract=contract, metadata={'rubrics': rubrics})],
            request=request,
            reduce=reduce,
            main_score_name='expert_score',
        )
