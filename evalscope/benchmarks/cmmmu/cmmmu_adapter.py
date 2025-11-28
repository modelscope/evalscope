from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    '设计',
    '音乐',
    '艺术',
    '艺术理论',
    '经济',
    '会计',
    '金融',
    '管理',
    '营销',
    '物理',
    '地理',
    '化学',
    '生物',
    '数学',
    '临床医学',
    '公共卫生',
    '基础医学',
    '诊断学与实验室医学',
    '制药',
    '历史',
    '心理学',
    '文献学',
    '社会学',
    '计算机科学',
    '电子学',
    '机械工程',
    '能源和电力',
    '材料',
    '建筑学',
    '农业',
]

DOMAIN_CAT2SUB_CAT = {
    '艺术与设计': ['艺术', '艺术理论', '设计', '音乐'],
    '商业': ['会计', '经济', '金融', '管理', '营销'],
    '科学': ['生物', '化学', '地理', '数学', '物理'],
    '健康与医学': ['基础医学', '临床医学', '诊断学与实验室医学', '制药', '公共卫生'],
    '人文社会科学': ['历史', '文献学', '社会学', '心理学'],
    '技术与工程': ['农业', '建筑学', '计算机科学', '电子学', '能源和电力', '材料', '机械工程'],
}

PROMPT = {
    'task_instructions': [
        '请回答以下多项选择题，并选出正确选项。这些题目可能包括单选和多选题型。如果所提供的信息不足以确定一个明确的答案，那么请根据可用的数据和你的判断来选择最可能正确的选项。',
        '请回答以下判断题，并根据题目描述和所给的信息来判断问题中陈述的对错。如果信息不完整或不足以作出绝对判断，请运用你的逻辑推理和现有信息来做出最可能的判断。',
        '请回答以下填空题，并根据题目的要求和所提供的信息来给出最恰当的答案。如果信息不足以确切回答，那么请依据现有的数据和你的推理能力来填写最合理的答案。',
    ],
    'multi_choice_example_format': ['问题：{}\n选项：\n{}\n正确答案：\n'],
    'T/F_example_format': ['问题：{}\n正确答案：\n'],
    'short_ans_example_format': ['问题：{}\n正确答案：\n'],
}


@register_benchmark(
    BenchmarkMeta(
        name='cmmmu',
        pretty_name='CMMMU',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA, Tags.CHINESE],
        description=
        'CMMMU includes manually collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, like its companion, MMMU. These questions span 30 subjects and comprise 39 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures.',  # noqa: E501
        dataset_id='lmms-lab/CMMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='val',
    )
)
class CMMMUAdapter(VisionLanguageAdapter):
    MAX_IMAGES: int = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reformat_subset = True
        self.category_map = {sub_cat: cat for cat, sub_cats in DOMAIN_CAT2SUB_CAT.items() for sub_cat in sub_cats}

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list = self.create_content_list(record)

        metadata = {
            'id': record['id'],
            'type': record['type'],
            'source_type': record['source_type'],
            'analysis': record['analysis'],
            'distribution': record['distribution'],
            'difficulty_level': record['difficulty_level'],
            'subcategory': record['subcategory'],
            'category': record['category'],
            'subfield': record['subfield'],
            'img_type': record['img_type'],
            'answer': record['answer'],
            'option1': record.get('option1', ''),
            'option2': record.get('option2', ''),
            'option3': record.get('option3', ''),
            'option4': record.get('option4', ''),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['answer'],
            subset_key=record['subcategory'],
            metadata=metadata,
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        from .utils import (
            get_fill_blank_prediction,
            get_multi_choice_info,
            get_multi_choice_prediction,
            get_TF_prediction,
        )

        doc = task_state.metadata
        if doc['type'] == '选择':
            index2ans, all_choices = get_multi_choice_info([doc[f'option{i}'] for i in range(1, 5)])
            parsed_pred = get_multi_choice_prediction(prediction, all_choices, index2ans)
        elif doc['type'] == '判断':
            parsed_pred = get_TF_prediction(prediction)
        else:
            parsed_pred = get_fill_blank_prediction(prediction, doc['answer'])

        task_state.metadata['parsed_pred'] = parsed_pred
        return str(parsed_pred)

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        from .utils import eval_cmmmu

        correct = eval_cmmmu(task_state.metadata)

        return Score(
            extracted_prediction=filtered_prediction, prediction=original_prediction, value={'acc': int(correct)}
        )

    def create_content_list(self, record: Dict[str, Any]) -> List[Content]:
        """
        Create a list of content elements and a list of answers from a record.
        Images are inserted at their <image x> placeholder positions in the text.

        Args:
            record (dict): The record containing question, images, and options.

        Returns:
            List[Content]: A list of content elements (text and images).
        """
        question = record['question']
        task_instructions = PROMPT['task_instructions']

        if record['type'] == '选择':
            formatted_options = ''
            start_chr = 'A'
            for i in range(1, 5):
                formatted_options += f"({start_chr}) {record[f'option{i}']}\n"
                start_chr = chr(ord(start_chr) + 1)

            current_example_template = PROMPT['multi_choice_example_format'][0]
            current_example = current_example_template.format(question, formatted_options)
            final_input_prompt = task_instructions[0] + '\n\n' + current_example

        elif record['type'] == '判断':
            current_example_template = PROMPT['T/F_example_format'][0]
            current_example = current_example_template.format(question)
            final_input_prompt = task_instructions[1] + '\n\n' + current_example

        else:  # For fill in the blanks questions.
            current_example_template = PROMPT['short_ans_example_format'][0]
            current_example = current_example_template.format(question)
            final_input_prompt = task_instructions[2] + '\n\n' + current_example

        # Prepare image map
        image_map: Dict[int, str] = {}
        for i in range(1, CMMMUAdapter.MAX_IMAGES + 1):
            final_input_prompt = final_input_prompt.replace(f'<img="{record[f"image_{i}_filename"]}">', f'<image {i}>')
            image = record.get(f'image_{i}')
            if image:
                image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
                image_map[i] = image_base64

        # Parse and replace image placeholders
        content_list = self._parse_text_with_images(final_input_prompt, image_map)
        return content_list
