import json
import os

os.environ['BUILD_DOC'] = '1'  # To avoid some heavy dependencies
from collections import defaultdict
from tqdm import tqdm
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import (
    AgentAdapter,
    DataAdapter,
    DefaultDataAdapter,
    ImageEditAdapter,
    MultiChoiceAdapter,
    Text2ImageAdapter,
    VisionLanguageAdapter,
)
from . import DESCRIPTION_JSON_PATH, load_description_json


# Language dictionaries for dataset markdown generation
def get_dataset_detail_locale_dict(category: str):
    """Get dataset detail locale dictionary with category"""
    return {
        'back_to_top': {
            'zh': '返回目录',
            'en': 'Back to Top'
        },
        'toc_title': {
            'zh': f'{category}评测集',
            'en': f'{category} Benchmarks'
        },
        'dataset_name': {
            'zh': '数据集名称',
            'en': 'Dataset Name'
        },
        'dataset_id': {
            'zh': '数据集ID',
            'en': 'Dataset ID'
        },
        'description': {
            'zh': '数据集介绍',
            'en': 'Description'
        },
        'task_categories': {
            'zh': '任务类别',
            'en': 'Task Categories'
        },
        'evaluation_metrics': {
            'zh': '评估指标',
            'en': 'Evaluation Metrics'
        },
        'aggregation_methods': {
            'zh': '聚合方法',
            'en': 'Aggregation Methods'
        },
        'requires_llm_judge': {
            'zh': '是否需要LLM Judge',
            'en': 'Requires LLM Judge'
        },
        'default_shots': {
            'zh': '默认提示方式',
            'en': 'Default Shots'
        },
        'subsets': {
            'zh': '数据集子集',
            'en': 'Subsets'
        },
        'eval_split': {
            'zh': '评测数据集划分',
            'en': 'Evaluation Split'
        },
        'supported_output_formats': {
            'zh': '支持输出格式',
            'en': 'Supported Output Formats'
        },
        'review_timeout': {
            'zh': '评测超时时间（秒）',
            'en': 'Review Timeout (seconds)'
        },
        'extra_parameters': {
            'zh': '额外参数',
            'en': 'Extra Parameters'
        },
        'sandbox_config': {
            'zh': '沙箱配置',
            'en': 'Sandbox Configuration'
        },
        'system_prompt': {
            'zh': '系统提示词',
            'en': 'System Prompt'
        },
        'prompt_template': {
            'zh': '提示模板',
            'en': 'Prompt Template'
        },
        'yes': {
            'zh': '是',
            'en': 'Yes'
        },
        'no': {
            'zh': '否',
            'en': 'No'
        },
        'no_description': {
            'zh': '暂无详细介绍',
            'en': 'No detailed description available'
        }
    }

def get_document_locale_dict(category: str):
    """Get document locale dictionary with category"""
    return {
        'title': {
            'zh': f'{category}评测集',
            'en': f'{category} Benchmarks'
        },
        'intro': {
            'zh': f'以下是支持的{category}评测集列表，点击数据集标准名称可跳转详细信息。',
            'en': f'Below is the list of supported {category} benchmarks. Click on a benchmark name to jump to details.'
        },
        'dataset_name': {
            'zh': '数据集名称',
            'en': 'Benchmark Name'
        },
        'pretty_name': {
            'zh': '标准名称',
            'en': 'Pretty Name'
        },
        'task_categories': {
            'zh': '任务类别',
            'en': 'Task Categories'
        },
        'details_title': {
            'zh': '数据集详情',
            'en': 'Benchmark Details'
        }
    }

def wrap_key_words(keywords: Union[str, List[str]]) -> str:
    """
    将关键词列表转换为Markdown格式的字符串
    
    Args:
        keywords (list[str]): 关键词列表
        
    Returns:
        str: 格式化的Markdown字符串
    """

    # 使用逗号分隔关键词，并添加反引号格式化
    if isinstance(keywords, str):
        return f'`{keywords}`'
    return ', '.join(sorted([f'`{keyword}`' for keyword in keywords]))

def process_dictionary(data: dict) -> str:
    """
    json.dumps的包装函数，处理字典格式化为Markdown代码块
    Args:
        data (dict): 要格式化的字典
    """
    json_str = json.dumps(data, ensure_ascii=False, indent=4)
    return f'```json\n{json_str}\n```'

def format_description(description: str) -> str:
    """
    将可能包含换行的描述格式化为正确的 Markdown 引用块。
    每一行都会加上 '  > ' 前缀，保证在同一个列表项下展示为多行引用。
    """
    if not description:
        return ''
    lines = description.strip('\n').splitlines()
    return '  > ' + '\n  > '.join(line if line.strip() else '' for line in lines)

def get_dataset_detail_locale(category: str, lang: str) -> Dict[str, str]:
    """Get localized strings for dataset details"""
    locale_dict = get_dataset_detail_locale_dict(category)
    return {k: v[lang] for k, v in locale_dict.items()}

def get_document_locale(category: str, lang: str) -> Dict[str, str]:
    """Get localized strings for document structure"""
    locale_dict = get_document_locale_dict(category)
    return {k: v[lang] for k, v in locale_dict.items()}

# Load translations once (if available)
_description_translations: dict = {}
try:
    _description_translations = load_description_json(DESCRIPTION_JSON_PATH)
except Exception:
    _description_translations = {}

def generate_dataset_markdown(data_adapter: DataAdapter, category: str, lang: str = 'zh') -> str:
    """
    Generate a well-formatted Markdown benchmark introduction based on a DataAdapter instance
    
    Args:
        data_adapter (DataAdapter): Dataset adapter instance
        category (str): Category name (e.g., 'LLM', 'AIGC')
        lang (str): Language code ('zh' for Chinese, 'en' for English)
        
    Returns:
        str: Formatted Markdown string
    """
    # Get localized text
    text = get_dataset_detail_locale(category, lang)
    name = data_adapter.name
    pretty_name = data_adapter.pretty_name or name
    dataset_id = data_adapter.dataset_id
    # Original English description as baseline
    base_description = data_adapter.description or text['no_description']
    # Use translated Chinese if available and lang == zh
    if lang == 'zh':
        zh_record = _description_translations.get(name)
        if isinstance(zh_record, dict):
            translated = zh_record.get('zh') or ''
            if translated.strip():
                description = translated.strip()
            else:
                description = base_description
        else:
            description = base_description
    else:
        description = base_description
    
    # Format dataset ID links
    if dataset_id.startswith(('http://', 'https://')):
        dataset_id_md = f'[{os.path.basename(dataset_id)}]({dataset_id})'
    elif '/' in dataset_id:  # ModelScope format ID
        dataset_id_md = f'[{dataset_id}](https://modelscope.cn/datasets/{dataset_id}/summary)'
    else:
        dataset_id_md = dataset_id
    
    # Build details section
    details = [
        f'### {pretty_name}',
        '',
        f'[{text["back_to_top"]}](#{text["toc_title"].lower().replace(" ", "-")})',
        f'- **{text["dataset_name"]}**: `{name}`',
        f'- **{text["dataset_id"]}**: {dataset_id_md}',
        f'- **{text["description"]}**:\n{format_description(description)}',
        f'- **{text["task_categories"]}**: {wrap_key_words(data_adapter.tags)}',
        f'- **{text["evaluation_metrics"]}**: {wrap_key_words(data_adapter.metric_list)}',
        f'- **{text["aggregation_methods"]}**: {wrap_key_words(data_adapter.aggregation)}',
        f'- **{text["requires_llm_judge"]}**: {text["yes"] if data_adapter._use_llm_judge else text["no"]}',
        f'- **{text["default_shots"]}**: {data_adapter.few_shot_num}-shot'
    ]
    
    # Add dataset subsets
    if data_adapter.eval_split:
        details.append(f'- **{text["eval_split"]}**: {wrap_key_words(data_adapter.eval_split)}')
    if data_adapter.subset_list:
        details.append(f'- **{text["subsets"]}**: {wrap_key_words(data_adapter.subset_list)}')

    # Add technical information
    technical_info = []
    if data_adapter.review_timeout is not None:
        technical_info.append(f'- **{text["review_timeout"]}**: {data_adapter.review_timeout}')
    
    # Add extra parameters
    extra_params = data_adapter._benchmark_meta.extra_params
    if extra_params:
        technical_info.append(f'- **{text["extra_parameters"]}**: \n{process_dictionary(extra_params)}')

    sandbox_config = data_adapter.sandbox_config
    if sandbox_config:
        technical_info.append(f'- **{text["sandbox_config"]}**: \n{process_dictionary(sandbox_config)}')

    # Add prompt templates
    if data_adapter.system_prompt:
        technical_info.append(f'- **{text["system_prompt"]}**:\n<details><summary>View</summary>\n\n````text\n{data_adapter.system_prompt}\n````\n\n</details>')
    if data_adapter.prompt_template:
        technical_info.append(f'- **{text["prompt_template"]}**:\n<details><summary>View</summary>\n\n````text\n{data_adapter.prompt_template}\n````\n\n</details>')

    return '\n'.join(details + [''] + technical_info + [''])

def generate_full_documentation(adapters: list[DataAdapter], category: str, lang: str = 'zh') -> str:
    """
    Generate complete Markdown documentation with index and all benchmark details
    
    Args:
        adapters (list[DataAdapter]): List of DataAdapter instances
        category (str): Category name (e.g., 'LLM', 'AIGC')
        lang (str): Language code ('zh' for Chinese, 'en' for English)
        
    Returns:
        str: Complete Markdown document
    """
    # Get localized text
    text = get_document_locale(category, lang)
    
    # Generate index
    index = [
        f'# {text["title"]}',
        '',
        f'{text["intro"]}',
        '',
        f'| {text["dataset_name"]} | {text["pretty_name"]} | {text["task_categories"]} |',
        '|------------|----------|----------|',
    ]
    
    for adapter in adapters:
        name = adapter.name
        pretty_name = adapter.pretty_name or name
        link_name = pretty_name.lower().replace(' ', '-').replace('.', '').replace("'", '').replace('+', '').replace('*', '')
        tags = wrap_key_words(adapter.tags)
        index.append(f'| `{name}` | [{pretty_name}](#{link_name}) | {tags} |')
    
    # Generate details section
    details = [
        '',
        '---',
        '',
        f'## {text["details_title"]}',
        ''
    ]

    for i, adapter in enumerate(adapters):
        details.append(generate_dataset_markdown(adapter, category, lang))
        if i < len(adapters) - 1:
            details.append('---')
            details.append('')
    
    return '\n'.join(index + details)

def get_adapters():
    from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark

    print('Getting DataAdapters...')
    # 获取所有DataAdapter实例
    adapters = defaultdict(list)
    for benchmark in tqdm(BENCHMARK_REGISTRY.values(), desc='Loading Benchmarks'):
        adapter = get_benchmark(benchmark.name)
        if isinstance(adapter, (Text2ImageAdapter, ImageEditAdapter)):
            adapters['aigc'].append(adapter)
        elif isinstance(adapter, (VisionLanguageAdapter,)):
            adapters['vlm'].append(adapter)
        elif isinstance(adapter, AgentAdapter):
            adapters['agent'].append(adapter)
        else:
            adapters['llm'].append(adapter)

    return adapters

def generate_docs(category: str, adapter_list:List[DataAdapter]):
    """
    Generate documentation for a specific category
    
    Args:
        category (str): Category name (e.g., 'llm', 'aigc')
        adapter_list (List[DataAdapter]): List of adapters for this category
    """
    category_upper = category.upper()
    adapter_list.sort(key=lambda x: x.name)  # 按名称排序

    print(f'Generating documentation for {category_upper}...')
    markdown_doc = generate_full_documentation(adapter_list, category_upper, 'zh')
    markdown_doc_en = generate_full_documentation(adapter_list, category_upper, 'en')
    
    # 输出到文件
    with open(f'docs/zh/get_started/supported_dataset/{category}.md', 'w', encoding='utf-8') as f:
        f.write(markdown_doc)

    with open(f'docs/en/get_started/supported_dataset/{category}.md', 'w', encoding='utf-8') as f:
        f.write(markdown_doc_en)

    print(f'{category_upper} Done')

if __name__ == '__main__':

    adapter_dict = get_adapters()
    for category, adapters in adapter_dict.items():
        generate_docs(category, adapters)