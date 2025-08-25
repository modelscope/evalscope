import json
import os
from tqdm import tqdm
from typing import Any, Dict, List

from evalscope.api.benchmark import DataAdapter

# Language dictionaries for dataset markdown generation
DATASET_DETAIL_LOCALE = {
    'back_to_top': {
        'zh': '返回目录',
        'en': 'Back to Top'
    },
    'toc_title': {
        'zh': 'LLM评测集',
        'en': 'LLM Benchmarks'
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
        'zh': '数据集描述',
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
    'requires_llm_judge': {
        'zh': '需要LLM Judge',
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
    'supported_output_formats': {
        'zh': '支持输出格式',
        'en': 'Supported Output Formats'
    },
    'extra_parameters': {
        'zh': '额外参数',
        'en': 'Extra Parameters'
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
        'zh': '暂无详细描述',
        'en': 'No detailed description available'
    }
}

DOCUMENT_LOCALE = {
    'title': {
        'zh': 'LLM评测集',
        'en': 'LLM Benchmarks'
    },
    'intro': {
        'zh': '以下是支持的LLM评测集列表，点击数据集标准名称可跳转详细信息。',
        'en': 'Below is the list of supported LLM benchmarks. Click on a benchmark name to jump to details.'
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

def wrap_key_words(keywords: list[str]) -> str:
    """
    将关键词列表转换为Markdown格式的字符串
    
    Args:
        keywords (list[str]): 关键词列表
        
    Returns:
        str: 格式化的Markdown字符串
    """

    # 使用逗号分隔关键词，并添加反引号格式化
    return ', '.join(sorted([f'`{keyword}`' for keyword in keywords]))

def process_dictionary(data: dict) -> str:
    """
    json.dumps的包装函数，处理字典格式化为Markdown代码块
    Args:
        data (dict): 要格式化的字典
    """
    return json.dumps(data, ensure_ascii=False, indent=4)
    
def get_dataset_detail_locale(lang: str) -> Dict[str, str]:
    """Get localized strings for dataset details"""
    return {k: v[lang] for k, v in DATASET_DETAIL_LOCALE.items()}

def get_document_locale(lang: str) -> Dict[str, str]:
    """Get localized strings for document structure"""
    return {k: v[lang] for k, v in DOCUMENT_LOCALE.items()}

def generate_dataset_markdown(data_adapter: DataAdapter, lang: str = 'zh') -> str:
    """
    Generate a well-formatted Markdown benchmark introduction based on a DataAdapter instance
    
    Args:
        data_adapter (DataAdapter): Dataset adapter instance
        lang (str): Language code ('zh' for Chinese, 'en' for English)
        
    Returns:
        str: Formatted Markdown string
    """
    # Get localized text
    text = get_dataset_detail_locale(lang)
    
    # Get basic information
    name = data_adapter.name
    pretty_name = data_adapter.pretty_name or name
    dataset_id = data_adapter.dataset_id
    description = data_adapter.description or text['no_description']
    
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
        f'- **{text["description"]}**:  \n  > {description}',
        f'- **{text["task_categories"]}**: {wrap_key_words(data_adapter.tags)}',
        f'- **{text["evaluation_metrics"]}**: {wrap_key_words(data_adapter.metric_list)}',
        f'- **{text["requires_llm_judge"]}**: {text["yes"] if data_adapter._use_llm_judge else text["no"]}',
        f'- **{text["default_shots"]}**: {data_adapter.few_shot_num}-shot'
    ]
    
    # Add dataset subsets
    if data_adapter.subset_list:
        details.append(f'- **{text["subsets"]}**: {wrap_key_words(data_adapter.subset_list)}')

    # Add technical information
    technical_info = [
        f'- **{text["supported_output_formats"]}**: {wrap_key_words(data_adapter.output_types)}',
    ]
    
    # Add extra parameters
    extra_params = data_adapter.extra_params
    if extra_params:
        technical_info.append(f'- **{text["extra_parameters"]}**: \n```json\n{process_dictionary(extra_params)}\n```')

    # Add prompt templates
    if data_adapter.system_prompt:
        technical_info.append(f'- **{text["system_prompt"]}**: \n```text\n{data_adapter.system_prompt}\n```')
    if data_adapter.prompt_template:
        technical_info.append(f'- **{text["prompt_template"]}**: \n```text\n{data_adapter.prompt_template}\n```')

    return '\n'.join(details + [''] + technical_info + [''])

def generate_full_documentation(adapters: list[DataAdapter], lang: str = 'zh') -> str:
    """
    Generate complete Markdown documentation with index and all benchmark details
    
    Args:
        adapters (list[DataAdapter]): List of DataAdapter instances
        lang (str): Language code ('zh' for Chinese, 'en' for English)
        
    Returns:
        str: Complete Markdown document
    """
    # Get localized text
    text = get_document_locale(lang)
    
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
        link_name = pretty_name.lower().replace(' ', '-').replace('.', '').replace("'", '')
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
        details.append(generate_dataset_markdown(adapter, lang))
        if i < len(adapters) - 1:
            details.append('---')
            details.append('')
    
    return '\n'.join(index + details)

if __name__ == '__main__':
    # 示例用法
    from evalscope.api.registry import BENCHMARK_REGISTRY, get_benchmark
    
    aigc_benchmarks = ['evalmuse', 'genai_bench', 'general_t2i', 'general_i2i', 'hpdv2', 'tifa160', 'data_collection']
    # 获取所有DataAdapter实例
    adapters: List[DataAdapter] = []
    for benchmark in tqdm(BENCHMARK_REGISTRY.values()):
        if benchmark.name not in aigc_benchmarks:
            adapters.append(get_benchmark(benchmark.name))

    adapters.sort(key=lambda x: x.name)  # 按名称排序
    
    # 生成完整文档
    markdown_doc = generate_full_documentation(adapters, 'zh')
    markdown_doc_en = generate_full_documentation(adapters, 'en')
    
    # 输出到文件
    with open('docs/zh/get_started/supported_dataset/llm.md', 'w', encoding='utf-8') as f:
        f.write(markdown_doc)

    with open('docs/en/get_started/supported_dataset/llm.md', 'w', encoding='utf-8') as f:
        f.write(markdown_doc_en)
        
    print('Done')