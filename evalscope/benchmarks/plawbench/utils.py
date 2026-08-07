# flake8: noqa: E501
"""Prompts and rubric grading helpers for PLawBench.

Prompts are ported verbatim from the official release
(https://github.com/skylenage/PLawbench): ``prompt_generate.py`` for the task prompts and
``prompt_zh.py`` for the judge prompts.
"""

import json
import re
from typing import Any, Dict, List, Tuple

from evalscope.utils.logger import get_logger

logger = get_logger()

# Rubric grading protocol used by each task, stored in the ``judge_type`` dataset column.
JUDGE_TYPE_CASE_ANALYSIS = 'case_analysis'
JUDGE_TYPE_LEGAL_QA = 'legal_qa'
JUDGE_TYPE_DOCUMENT_GENERATION = 'document_generation'

# Number of follow-up questions the consultation judge considers. The official release only
# open-sources the 'mid' difficulty consultations, whose prompt asks for 10-25 questions.
CONSULTATION_MAX_QUESTIONS = '25'

# Case-analysis rubric sections: dataset rubric tag -> judge output key -> EvalScope metric name.
RUBRIC_TAG_TO_JUDGE_KEY = {
    '结论得分': '结论',
    '案情简述得分': '案件事实',
    '分析过程得分': '推理过程',
    '法条依据得分': '法条依据',
}
JUDGE_KEY_TO_METRIC = {
    '结论': 'conclusion_acc',
    '案件事实': 'fact_acc',
    '推理过程': 'reasoning_acc',
    '法条依据': 'law_acc',
}

TASK_INSTRUCTIONS: Dict[str, str] = {
    JUDGE_TYPE_LEGAL_QA: """
## 角色

你是一名具有十年以上执业经验的律师，精通中国现行法律法规与司法实践。你擅长根据当事人为你提供的片面事实，还原事件全貌，识别当事人可能的隐瞒与偏差、补齐证据链与时间线，为后续法律分析/诉讼策略做准备。

## 核心要求

1. 中立、不诱导、不站队，避免使用“畜生/家暴男/你肯定”等定性词。
2. 可核实、可取证：尽量问“时间、地点、人物、行为、金额、证据、来源”。
3. 覆盖面完整：包括事件各方面细节、签订协议细节、双方过错与风险点等。
4. 先关键后细节：先问最影响案件走向的问题（安全/证据/财产归属/债务性质/抚养安排），再问补充细节。

一次输出多问：每次输出 10–25 个问题，用编号列表；问题应短、明确、单一指向。输出问题数多于或少于都将被视为错误。

5. 发现矛盾或缺口：必须追加“核对矛盾”的澄清问题（例如时间前后不一致、金额来源不明、协议真实性不明）。
6. 隐瞒检测：对可能被省略的不利事实，必须加入对等追问（如：是否互殴、是否先动手、是否有报警记录、是否存在婚外情、是否共同还贷、是否共同经营、是否存在共同签字借款等）。
7. 输出格式：仅输出问题清单，不要任何前后缀。

## 回答示例

1. XXX是否持有电焊作业对应的特种作业操作资格证书？
2. XX与XXX关于“XXXX”的约定中，双方属于雇佣关系还是承揽关系？
3.2000元款项转给XXX个人微信后，有无书面约定该转账系“代公司收款”及后续补开公司收据的相关凭证？
3. 案涉茶叶外包装是否标注委托方为X岭公司、受托方为XXXX加工厂？
5.......
6.......
......

""",
    JUDGE_TYPE_DOCUMENT_GENERATION: """
你是一名法律实务专家，熟悉法律文书的撰写规范。我将为你提供你的客户（当事人）的陈述或者其他信息，你需要根据现有的信息，生成一篇法律文书。文书类型和具体要求如下：1. 文书类型
请根据以下选项生成相应类型的法律文书（请确保文书内容符合该类型的特定要求）：
（1 ）起诉状
（2）答辩状
（3）裁判文书
2. 文书要求
（1）需要自动根据每题的模板进行撰写；如未有对应案由的模板，则采用中国大陆通用的文书格式撰写。（2）你所撰写的法律文书需要在2500-3000。(3) 生成结果输出文书即可，不要包括模拟律师回答之类的其他废话！
（3）你需要根据提示词，自行选择以上三种文书类型进行生成。（4）客户的法律素养不高，陈述不一定都是正确的。你可能需要在识别陷阱的基础上撰写合法合理合规的文书。
（5）你需要在法律文书中引用真实有效的法律法条。
""",
    JUDGE_TYPE_CASE_ANALYSIS: """
## 角色

你是一名具有十年以上执业经验的法律实务专家，精通中国现行法律法规与司法实践。你擅长将复杂的法律问题分解为清晰的逻辑模块，并严格依据“结论先行、事实为重、推理严密、依据支撑”的专业风格进行解答。

## 核心要求

1. 严格顺序：回答必须按照以下四部分顺序展开，并使用对应标题：
【结论】
【案件事实】
【推理过程】
【法条依据】
2. 内容规范：
结论：直接、明确，针对提问的核心争议点给出肯定或否定的判断。
案件事实：基于用户提供的案情，简明、客观地摘录与法律判断相关的事实要素，不添加假设或推测。
推理过程：逐步展示如何从事实链接到法条，并最终推导出结论的逻辑链条，可分层、分点表述。
法条依据：引用中国大陆现行有效的法律、司法解释、行政法规等，注明发文机关、名称及具体条、款、项。给出法条原文。## 模板
【结论】
[此处给出清晰明确的法律判断]
【案件事实】
（仅摘录与结论相关的关键事实，保持客观精简）
【推理过程】
1. ......；
2. ......；
3. ......。
【法条依据】
● 《XXX法》第X条第X款（发文机关：XXX）
● 《最高人民法院关于XXX的解释》第X条
（如有多个依据，按效力层级或相关性排列）

## 回答示例

【结论】
夫妻财产关系应适用澳门特别行政区域（即双方约定的共同常居地）的法律。
房产一（内地A市）属夫妻共同财产。
房产二（内地B市）因系政策性房改福利且源于R某父母的单位权利，依法认定为R某个人财产。
【案件事实】
关键事实1：L某（澳门特别行政区域居民）与R某（内地居民）2013年在澳门特别行政区域登记结婚。 2016 年双方在该地区协议离婚。
关键事实2：签署《婚前协议记录》，约定：①“共同常居地”为澳门；② 夫妻财产制度适用“一般共同财产制”。
关键事实3： 房产一系L某婚前全款购买；登记在L某个人名下（内地城市 A）。
关键事实4：房产二系婚姻存续期间由R某父母购买；通过政策性房改渠道，实际仅支付1万余元，市场价值远高于支付价；涉及R某父母的单位福利或工龄政策（内地城市 B）；登记在R某名下。L某曾签署委托书，同意同意产权登记在R某名下。
【推理过程】
第一步，确定夫妻财产关系的法律适用问题。
《中华人民共和国涉外民事关系法律适用法》第二十四条规定：“夫妻财产关系，当事人可以协议选择适用一方当事人经常居所地法律、国籍国法律或者主要财产所在地法律。当事人没有选择的，适用共同经常居所地法律；没有共同经常居所地的，适用共同国籍国法律。”
本案中，L某与R某签署《婚前协议记录》，约定：①“共同常居地”为澳门；② 夫妻财产制度适用“一般共同财产制”，因此两人的夫妻财产关系应以澳门特别行政区的实体法作为准据法。
第二步，确定财产分割的法律适用问题。
双方在婚前协议中明示选择共同常居地和一般共同财产制。
《澳门民法典》允许夫妻协议选择夫妻财产适用法律，并在第一千六百零九条规定：“夫妻采用之财产制为一般共同财产制时，共同财产由夫妻现在及将来拥有之一切财产组成，但被法律排除之财产除外。双方在婚前协议中明示选择共同常居地和一般共同财产制，因此适用一般共同财产制。
第三步，确定房产一归属。
《澳门民法典》第一千六百零九条规定：“夫妻采用之财产制为一般共同财产制时，共同财产由夫妻现在及将来拥有之一切财产组成，但被法律排除之财产除外。
适用澳门“一般共同财产制”下的处理逻辑：房产一系L某婚前以个人资金购买并登记，不属于被法律排除的财产，因此属于夫妻共同财产。
第四步：确定房产二归属。
房产二性质特殊：根据房产性质、对价、协议签署情况分析：该房产低价来源于政策性福利、R某父母的单位工龄权益（家庭福利派生利益）。
委托书性质： L某曾签署委托书同意房产登记在R某名下，表明其对产权归属的认可。L某作为一个理性人，对自身在上房产二中享有的权益应尽审慎注意义务，更应知道其出具委托书的法律后果。
因此，房产二符合《澳门民法典》第一千六百零一十条关于“一、下列者不属共同拥有之财产：ａ)_附有规定不可由夫妻拥有之条款而被赠予或死因处分之财产，即使该等财产系计入特留份范围亦然……”的规定，属于R某个人财产。
【法条依据】
《中华人民共和国涉外民事关系法律适用法》第24条：夫妻财产关系适用共同经常居所地法律或协议选择的法律。
《澳门民法典》第一千六百零九条规定：“夫妻采用之财产制为一般共同财产制时，共同财产由夫妻现在及将来拥有之一切财产组成，但被法律排除之财产除外。
《澳门民法典》第一千六百零一十条关于“一、下列者不属共同拥有之财产：ａ)_附有规定不可由夫妻拥有之条款而被赠予或死因处分之财产，即使该等财产系计入特留份范围亦然……”
""",
}

# The consultation and drafting tasks share a question-only user prompt.
_QUESTION_USER_PROMPT = """
## 问题
{question}
"""

USER_PROMPTS: Dict[str, str] = {
    JUDGE_TYPE_CASE_ANALYSIS: """
## 案例
{context}

## 问题
{question}
""",
    JUDGE_TYPE_LEGAL_QA: _QUESTION_USER_PROMPT,
    JUDGE_TYPE_DOCUMENT_GENERATION: _QUESTION_USER_PROMPT,
}

JUDGE_SYSTEM_PROMPTS: Dict[str, str] = {
    JUDGE_TYPE_LEGAL_QA: """
## 角色
你是一名评估员，你的任务是对照一份标有得分项的标准答案，在律师对当事人提问收集信息的过程中，对律师的表现进行打分，以评估律师在面对当事人时的理解和业务能力。

## 输入和输出格式
输入包含两部分：
1、待评分律师的提问问题清单（10-25条，已进行编号）
2、标准答案的rubric（总分N分；每个要点为“（+X分）要点描述”）
3、案情描述

输出格式固定为：
总分为「得分」/「总分」，得分率为「百分比」%
分析如下：
-要点1:「得分情况」，理由为：.......。
-要点2:「得分情况」，理由为：......。
-......
```json
    {
        "total_points": 模型得分,
        "max_points": 该题总分,
    },
```

## 评分规则
1、你无须对答案本身的准确性进行判断，只需严格按照标准答案，判断律师追问的问题中有没有相同或本质相同的内容，并计算最终得分。
2、只对前 <<num>> 条问题进行评价，多余部分的问题一律不考虑。
3、不得补充新问题、提供法律建议、评价当事人、复述长篇背景；只做对照与打分说明。
4、对标准rubric中的每个“问题点”，在待评分问题清单中寻找最佳匹配，按照匹配程度给分，如果完全覆盖关键要素，语义一致，给出100%的分数；如果命中核心主题，但缺少关键限定，例如标准要“报警记录及住院病历、诊断证明”，只问“有无报警”而未问病历，可以酌情给分；如果未覆盖语义关键点，则不给分。

""",
    JUDGE_TYPE_DOCUMENT_GENERATION: """
## 角色和任务
你是一名评估员，你的任务是对照一份标有得分项的标准答案，对法律文书的写作情况进行打分。

## 输入和输出格式
输入包含两部分：
1、待评分的法律文书。
2、标准rubric

输出格式固定为：
总分为「得分」/「总分」，得分率为「百分比」%
分析如下：
-要点1:「得分情况」，理由为：.......。
-要点2:「得分情况」，理由为：......。
-......
```json
    {
        "total_points": 模型得分,
        "max_points": 该题总分,
    },
```

## 评分规则
1、你无须对答案本身的准确性进行判断，只需严格按照标准答案，判断模型的回答中有没有相同或本质相同的内容，并计算最终得分。
2、不得更改答案、提供法律建议、评价当事人、复述长篇背景；只做对照与打分说明。
3、对标准rubric中的每个“问题点”，在待评分问题清单中寻找最佳匹配，按照匹配程度给分，如果完全覆盖关键要素，语义一致，给出100%的分数；如果命中核心主题，但缺少关键限定，例如标准要“报警记录及住院病历、诊断证明”，只问“有无报警”而未问病历，可以酌情给分；如果未覆盖语义关键点，则不给分。
""",
    JUDGE_TYPE_CASE_ANALYSIS: """
## 角色设定

你是一名严谨的法律实务评分专家，擅长根据精确的评分标准对法律实务问题的回答进行逐项检查。你严格遵守"提及即得分，未提及则扣分"的原则，不做主观推断。

## 核心任务

根据提供的详细评分细则（rubric），对法律实务题回答进行精确评分。回答分为四个部分：结论、法条依据、案件事实总结、推理过程。每个部分可能有独立的评分细则。

## 输入信息结构

**问题内容**：简要的案件描述和提问问题。

**评分细则（rubric）**：分为多个部分，每个部分的格式可能类似示例：

结论评分细则

法条依据评分细则

案件事实评分细则

推理过程评分细则

其他部分评分细则

**待评分回答**：需要评分的文本，其中明确标有“结论”、“法条依据”、“案件事实”和“推理过程”四个部分。


## 评分原则

1.  **严格字面匹配**：以rubric中的表述为准，检查回答中是否有相同或实质相同的表述。

2.  **独立评分**：每个得分项独立计算，不考虑其他项。

3.  **不推断不补充**：仅根据回答中明确提及的内容评分，不进行推理或补充。

4.  **明确加分项扣分项**：严格按照分数进行加分和扣分操作。

5.  **酌情扣分**：在回答结果没有完全覆盖得分点时，根据已经答出的要点酌情给部分结果分。


## 评分流程

### 步骤1：分割回答

将回答分割为四个部分：

*   结论

*   法条依据即大前提

*   案件事实即小前提

*   推理过程


### 步骤2：分别评分

对每个部分，使用对应的rubric进行评分。每个部分的评分步骤：

1.  **解析rubric**：将rubric分解为得分项和扣分规则。

2.  **内容检查**：检查该部分回答中是否提及rubric要求的每个内容。


### 步骤3：整体结构评分（如果有）

如果rubric中包含对整体结构的评分，则根据整体结构的要求进行评分。

### 步骤4：汇总与反馈

## 输出格式

要求输出以下四个部分的详细评分，以及总分和整体反馈。

```json
{
  "score_details": {
    "法条依据": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [
        {
          "rubric_item": "具体得分项描述",
          "max_points": ...,
          "points_awarded": ...,
          "mentions": [...],
          "rationale": "评分理由"
        }
      ]
    },
    "案件事实": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [...]
    },
    "推理过程": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [...]
    },
    "结论": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [...]
    }
  },
  "total_score": {
    "total_awarded": ...,
    "total_max": ...,
    "percentage": ...
  },
  "overall_feedback": {

    "strengths": [ ],


    "weaknesses": [ ],


    "suggestions": [ ]

  }
}

```""",
}

# The consultation and drafting judges share a single-total verdict prompt.
_TOTAL_POINTS_JUDGE_PROMPT = """
# 问答内容
<<conversation>>

# 评分细则
<<rubric_item>>

# 该题总分
<<score>>

# 输出格式
请按照指令，先分析，再给出得分，务必在回答的结尾按照以下json格式输出：
```json
    {
        "total_points": 模型得分,
        "max_points": 该题总分,
    },
```

"""

JUDGE_USER_PROMPTS: Dict[str, str] = {
    JUDGE_TYPE_LEGAL_QA: _TOTAL_POINTS_JUDGE_PROMPT,
    JUDGE_TYPE_DOCUMENT_GENERATION: _TOTAL_POINTS_JUDGE_PROMPT,
    JUDGE_TYPE_CASE_ANALYSIS: """
# 问题内容
<<conversation>>

# 评分细则
<<rubric_item>>

# 输出格式
请务必在回答的结尾按照以下json格式输出：

```json
{
  "score_details": {
    "法条依据": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [
        {
          "rubric_item": "具体得分项描述",
          "max_points": ...,
          "points_awarded": ...,
          "mentions": [...],
          "rationale": "评分理由"
        }
      ]
    },
    "案件事实": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [...]
    },
    "推理过程": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [...]
    },
    "结论": {
      "total_points": ...,
      "max_points": ...,
      "breakdown": [...]
    }
  },
  "total_score": {
    "total_awarded": ...,
    "total_max": ...,
    "percentage": ...
  },
  "overall_feedback": {

    "strengths": [ ],


    "weaknesses": [ ],


    "suggestions": [ ]

  }
}

```
""",
}


def _iter_json_objects(text: str) -> List[str]:
    """Return every ``{...}`` block in ``text``, in order of appearance.

    Brace matching is string-aware so that braces inside rubric quotations do not
    truncate the object, which a regex cannot handle for the nested judge verdicts.
    A trailing block that is never closed is still returned so that :func:`_repair_json`
    can recover a verdict truncated at the judge's token limit.
    """
    objects: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}' and depth > 0:
            depth -= 1
            if depth == 0:
                objects.append(text[start:index + 1])
    if depth > 0 and start >= 0:
        objects.append(text[start:])
    return objects


def _repair_json(text: str) -> str:
    """Close unterminated strings, arrays, and objects in a judge verdict.

    Judges regularly emit a verdict whose ``overall_feedback`` array is left unclosed, or
    stop mid-object when they hit the token limit. Balancing the containers recovers the
    ``score_details`` payload that was already emitted instead of discarding the response.
    """
    output: List[str] = []
    stack: List[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            output.append(char)
        elif char in '{[':
            stack.append(char)
            output.append(char)
        elif char in '}]':
            opener = '{' if char == '}' else '['
            if opener not in stack:
                continue  # Stray closer with no matching opener.
            # Implicitly close containers opened inside the one being closed.
            while stack[-1] != opener:
                output.append('}' if stack.pop() == '{' else ']')
            stack.pop()
            output.append(char)
        else:
            output.append(char)

    if in_string:
        output.append('"')
    while stack:
        output.append('}' if stack.pop() == '{' else ']')
    return ''.join(output)


def parse_json_to_dict(text: str) -> Dict[str, Any]:
    """Extract the judge verdict object from a judge response.

    Judges are instructed to append the verdict as a fenced JSON block at the end of the
    response, so the last parsable top-level object wins.
    """
    if not text:
        return {}

    candidates = _iter_json_objects(text) or [text.strip()]
    for candidate in reversed(candidates):
        for variant in (candidate, _repair_json(candidate)):
            # Judges frequently emit trailing commas before a closing brace / bracket.
            cleaned = re.sub(r',(\s*[}\]])', r'\1', variant)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    logger.warning(f'Failed to parse PLawBench judge JSON: {text[-200:]!r}')
    return {}


def _to_float(value: Any) -> float:
    """Coerce a judge-reported number to float, returning 0.0 for unusable values."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r'-?\d+(?:\.\d+)?', value)
        if match:
            return float(match.group(0))
    return 0.0


def parse_rubric_sections(rubrics: str) -> List[Dict[str, Any]]:
    """Parse the JSON-encoded ``case_analysis`` rubric into its four scored sections."""
    sections = json.loads(rubrics)
    if not isinstance(sections, list):
        raise ValueError('PLawBench case_analysis rubric must be a JSON array.')
    return sections


def build_conversation(prompt: str, response: str) -> str:
    """Render the judged conversation exactly as the official evaluator does."""
    return f'user: {prompt}\n\nassistant: {response}\n\n'


def score_total_points(judge_json: Dict[str, Any], max_points: float) -> Tuple[float, Dict[str, Any]]:
    """Score a single-total rubric (``legal_qa`` / ``document_generation``).

    The dataset's ``max_points`` is authoritative; the judge only contributes the awarded
    points, which are clamped into ``[0, max_points]``.
    """
    if 'total_points' not in judge_json:
        raise ValueError(f'Judge response is missing "total_points": {judge_json}')

    awarded = _to_float(judge_json['total_points'])
    awarded = min(max(awarded, 0.0), max_points)
    acc = awarded / max_points if max_points > 0 else 0.0
    details = {
        'awarded_points': awarded,
        'max_points': max_points,
        'judge_reported_max_points': _to_float(judge_json.get('max_points')),
    }
    return acc, details


def score_case_analysis(judge_json: Dict[str, Any],
                        rubric_sections: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Score the four ``case_analysis`` rubric sections.

    Section denominators come from the dataset rubric rather than the judge output, so a
    judge that mis-reports ``max_points`` cannot distort the metric.
    """
    score_details = judge_json.get('score_details')
    if not isinstance(score_details, dict):
        raise ValueError(f'Judge response is missing "score_details": {judge_json}')

    section_max: Dict[str, float] = {}
    for section in rubric_sections:
        judge_key = RUBRIC_TAG_TO_JUDGE_KEY.get(section.get('tags', ''))
        if judge_key is None:
            continue
        section_max[judge_key] = section_max.get(judge_key, 0.0) + _to_float(section.get('points'))

    missing = [key for key in section_max if key not in score_details]
    if missing:
        raise ValueError(f'Judge response is missing rubric sections {missing}: {list(score_details)}')

    # ``acc`` is inserted first so that it stays the report's primary metric, which the
    # dashboard derives from the first metric in report order.
    values: Dict[str, float] = {'acc': 0.0}
    breakdown: Dict[str, Any] = {}
    total_awarded = 0.0
    total_max = 0.0
    for judge_key, max_points in section_max.items():
        section_result = score_details[judge_key]
        if not isinstance(section_result, dict) or 'total_points' not in section_result:
            raise ValueError(f'Judge response section "{judge_key}" is missing "total_points".')
        awarded = min(max(_to_float(section_result['total_points']), 0.0), max_points)
        total_awarded += awarded
        total_max += max_points
        values[JUDGE_KEY_TO_METRIC[judge_key]] = awarded / max_points if max_points > 0 else 0.0
        breakdown[judge_key] = {'awarded_points': awarded, 'max_points': max_points}

    values['acc'] = total_awarded / total_max if total_max > 0 else 0.0
    details = {
        'awarded_points': total_awarded,
        'max_points': total_max,
        'sections': breakdown,
    }
    return values, details
