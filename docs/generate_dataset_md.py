import json
from tqdm import tqdm

from evalscope.benchmarks import DataAdapter


def wrap_key_words(keywords: list[str]) -> str:
    """
    将关键词列表转换为Markdown格式的字符串
    
    Args:
        keywords (list[str]): 关键词列表
        
    Returns:
        str: 格式化的Markdown字符串
    """
    if not keywords:
        return '无'
    
    # 使用逗号分隔关键词，并添加反引号格式化
    return ', '.join(sorted([f'`{keyword}`' for keyword in keywords]))

def process_dictionary(data: dict) -> str:
    """
    json.dumps的包装函数，处理字典格式化为Markdown代码块
    Args:
        data (dict): 要格式化的字典
    """
    return json.dumps(data, ensure_ascii=False, indent=4)
    

def generate_dataset_markdown(data_adapter: DataAdapter) -> str:
    """
    根据DataAdapter实例生成美观的Markdown数据集介绍
    
    Args:
        data_adapter (DataAdapter): 数据集适配器实例
        
    Returns:
        str: 格式化的Markdown字符串
    """
    # 获取基础信息
    name = data_adapter.name
    pretty_name = data_adapter.pretty_name or name
    dataset_id = data_adapter.dataset_id
    description = data_adapter.description or '暂无详细描述'
    
    # 生成锚点链接
    anchor = f"<a id=\"{name}\"></a>"
    
    # 处理数据集ID的链接格式
    if dataset_id.startswith(('http://', 'https://')):
        dataset_id_md = f'[{dataset_id}]({dataset_id})'
    elif '/' in dataset_id:  # ModelScope格式的ID
        dataset_id_md = f'[{dataset_id}](https://modelscope.cn/datasets/{dataset_id}/summary)'
    else:
        dataset_id_md = dataset_id
    
    # 构建详情部分
    details = [
        f'### {pretty_name}',
        '',
        f'[返回目录](#llm评测集)',
        f'- **数据集名称**: `{name}`',
        f'- **数据集ID**: {dataset_id_md}',
        f'- **数据集描述**:  \n  > {description}',
        f'- **任务类别**: {wrap_key_words(data_adapter.tags)}',
        f'- **评估指标**: {wrap_key_words(data_adapter.metric_list)}',
        f"- **需要LLM Judge**: {'是' if data_adapter.llm_as_a_judge else '否'}",
        f'- **默认提示方式**: {data_adapter.few_shot_num} Few-shot'
    ]
    
    # 添加数据集子集信息
    if data_adapter.subset_list:
        details.append(f'- **数据集子集**: {wrap_key_words(data_adapter.subset_list)}')

    # 添加其他技术信息
    technical_info = [
        f'- **支持输出格式**: {wrap_key_words(data_adapter.output_types)}',
    ]
    
    # 添加额外参数信息
    extra_params = data_adapter.config_kwargs.get('extra_params', {})
    if extra_params:
        technical_info.append(f'- **额外参数**: \n```json\n{process_dictionary(extra_params)}\n```')

    # 添加提示模板
    if data_adapter.system_prompt:
        technical_info.append(f'- **系统提示词**: `{data_adapter.system_prompt}`')
    if data_adapter.prompt_template:
        technical_info.append(f'- **提示模板**: \n```text\n{data_adapter.prompt_template}\n```')

    return '\n'.join(details + [''] + technical_info + [''])


def generate_full_documentation(adapters: list[DataAdapter]) -> str:
    """
    生成完整的Markdown文档，包含索引和所有数据集详情
    
    Args:
        adapters (list[DataAdapter]): DataAdapter实例列表
        
    Returns:
        str: 完整的Markdown文档
    """
    # 生成索引
    index = [
        '# LLM评测集',
        '',
        '以下是支持的LLM评测集列表，点击数据集标准名称可跳转详细信息。',
        '',
        '| 数据集名称 | 标准名称 | 任务类别 |',
        '|------------|----------|----------|',
    ]
    
    for adapter in adapters:
        name = adapter.name
        pretty_name = adapter.pretty_name or name
        tags = wrap_key_words(adapter.tags)
        index.append(f'| `{name}` | [{pretty_name}](#{pretty_name.lower()}) | {tags} |')
    
    # 生成详情部分
    details = [
        '',
        '---',
        '',
        '## 数据集详情',
        ''
    ]

    for i, adapter in enumerate(adapters):
        details.append(generate_dataset_markdown(adapter))
        if i < len(adapters) - 1:
            details.append('---')
            details.append('')
    
    return '\n'.join(index + details)


if __name__ == '__main__':
    # 示例用法
    from evalscope.benchmarks.benchmark import BENCHMARK_MAPPINGS
    
    aigc_benchmarks = ['evalmuse', 'genai_bench', 'general_t2i', 'hpdv2', 'tifa160', 'data_collection']
    # 获取所有DataAdapter实例
    adapters = []
    for benchmark in tqdm(BENCHMARK_MAPPINGS.values()):
        if benchmark.name not in aigc_benchmarks:
            adapters.append(benchmark.get_data_adapter())

    adapters.sort(key=lambda x: x.name)  # 按名称排序
    
    # 生成完整文档
    markdown_doc = generate_full_documentation(adapters)
    
    # 输出到文件
    with open('docs/zh/get_started/supported_dataset/native/llm.md', 'w', encoding='utf-8') as f:
        f.write(markdown_doc)

    print('Markdown文档已生成: llm.md')