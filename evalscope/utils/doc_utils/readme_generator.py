# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Utility functions for generating README documentation for benchmarks.

This module provides tools to automatically generate comprehensive README.md
files for benchmark datasets, including metadata, statistics, examples,
and usage instructions.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from evalscope.utils.logger import get_logger

logger = get_logger()

# Standard README template with all sections
README_TEMPLATE = '''# {pretty_name}

{description}

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `{name}` |
| **Dataset ID** | {dataset_id_link} |
| **Paper** | {paper_link} |
| **Tags** | {tags} |
| **Metrics** | {metrics} |
| **Default Shots** | {few_shot_num}-shot |
| **Evaluation Split** | `{eval_split}` |
{extra_overview_rows}
## Data Statistics

{statistics_section}

## Sample Example

{sample_example_section}

## Prompt Template

{prompt_template_section}
{extra_params_section}{sandbox_config_section}
{usage_section}

'''


def _format_dataset_link(dataset_id: str) -> str:
    """Format dataset ID as a clickable link."""
    if not dataset_id:
        return 'N/A'
    if dataset_id.startswith(('http://', 'https://')):
        return f'[{os.path.basename(dataset_id)}]({dataset_id})'
    elif '/' in dataset_id:
        # ModelScope format: org/name
        return f'[{dataset_id}](https://modelscope.cn/datasets/{dataset_id}/summary)'
    return f'`{dataset_id}`'


def _format_link(url: Optional[str], display_text: str = 'Link') -> str:
    """Format a URL as a markdown link."""
    if url:
        return f'[{display_text}]({url})'
    return 'N/A'


def _format_tags(tags: Union[str, List[str]]) -> str:
    """Format tags as inline code."""
    if not tags:
        return 'N/A'
    if isinstance(tags, str):
        return f'`{tags}`'
    return ', '.join(f'`{t}`' for t in sorted(tags))


def _format_metrics(metric_list: List[Union[str, Dict]]) -> str:
    """Format metrics list."""
    if not metric_list:
        return 'N/A'

    formatted = []
    for m in metric_list:
        if isinstance(m, str):
            formatted.append(f'`{m}`')
        elif isinstance(m, dict):
            # Handle dict format like {'acc': {'numeric': True}}
            for key in m.keys():
                formatted.append(f'`{key}`')
    return ', '.join(formatted)


def _format_sample_count(count: Union[int, str]) -> str:
    """Format sample count with thousand separators."""
    if isinstance(count, int):
        return f'{count:,}'
    return str(count)


def _format_prompt_template_section(
    prompt_template: Optional[str],
    system_prompt: Optional[str] = None,
    few_shot_template: Optional[str] = None,
) -> str:
    """Format the prompt template section."""
    sections = []

    if system_prompt:
        sections.append('**System Prompt:**')
        sections.append('```text')
        sections.append(system_prompt)
        sections.append('```')
        sections.append('')

    if prompt_template:
        sections.append('**Prompt Template:**')
        sections.append('```text')
        sections.append(prompt_template)
        sections.append('```')
    else:
        sections.append('*No prompt template defined.*')

    if few_shot_template:
        sections.append('')
        sections.append('<details>')
        sections.append('<summary>Few-shot Template</summary>')
        sections.append('')
        sections.append('```text')
        sections.append(few_shot_template)
        sections.append('```')
        sections.append('')
        sections.append('</details>')

    return '\n'.join(sections)


def _format_extra_overview_rows(
    train_split: Optional[str] = None,
    aggregation: Optional[str] = None,
) -> str:
    """Format additional overview table rows."""
    rows = []
    if train_split:
        rows.append(f'| **Train Split** | `{train_split}` |')
    if aggregation and aggregation != 'mean':
        rows.append(f'| **Aggregation** | `{aggregation}` |')
    if rows:
        return '\n'.join(rows) + '\n\n'
    return '\n'


def _format_extra_params_section(extra_params: Optional[Dict] = None) -> str:
    """Format the extra parameters section."""
    if not extra_params:
        return ''

    lines = ['', '## Extra Parameters', '']
    lines.append('| Parameter | Type | Default | Description |')
    lines.append('|-----------|------|---------|-------------|')

    for param_name, param_spec in extra_params.items():
        if isinstance(param_spec, dict) and 'description' in param_spec:
            param_type = param_spec.get('type', 'any')
            param_value = param_spec.get('value', 'N/A')
            param_desc = param_spec.get('description', '')
            choices = param_spec.get('choices')
            if choices:
                param_desc += f' Choices: {choices}'
            lines.append(f'| `{param_name}` | `{param_type}` | `{param_value}` | {param_desc} |')
        else:
            lines.append(f'| `{param_name}` | - | `{param_spec}` | - |')

    return '\n'.join(lines) + '\n'


def _format_sandbox_config_section(sandbox_config: Optional[Dict] = None) -> str:
    """Format the sandbox configuration section."""
    import json

    if not sandbox_config:
        return ''

    lines = ['', '## Sandbox Configuration', '']
    lines.append('This benchmark requires a sandbox environment for code execution.')
    lines.append('')
    lines.append('```json')
    lines.append(json.dumps(sandbox_config, indent=2, ensure_ascii=False))
    lines.append('```')

    return '\n'.join(lines) + '\n'


def _format_usage_section(
    name: str,
    subsets: Optional[List[str]] = None,
    sandbox_config: Optional[Dict] = None,
    extra_params: Optional[Dict] = None,
) -> str:
    """
    Format the usage section with CLI and Python examples.

    Args:
        name: Benchmark name
        subsets: List of subset names (if any)
        sandbox_config: Sandbox configuration (if any)
        extra_params: Extra parameters configuration (if any)

    Returns:
        Formatted usage section markdown string
    """
    # Build dataset_args content for Python
    dataset_args_comments = []
    if subsets and len(subsets) > 1:
        subset_example = subsets[:3] if len(subsets) > 3 else subsets
        dataset_args_comments.append(
            f'            # subset_list: {subset_example}  # optional, evaluate specific subsets'
        )
    if extra_params:
        dataset_args_comments.append('            # extra_params: {}  # uses default extra parameters')

    # Build CLI command
    cli_lines = [
        'evalscope eval \\',
        '    --model YOUR_MODEL \\',
        '    --api-url OPENAI_API_COMPAT_URL \\',
        '    --api-key EMPTY_TOKEN \\',
        f'    --datasets {name} \\',
    ]
    if sandbox_config:
        cli_lines.append('    --use-sandbox \\')
    cli_lines.append('    --limit 10  # Remove this line for formal evaluation')

    # Build Python code
    python_dataset_args = ''
    if dataset_args_comments:
        python_dataset_args = f'''    dataset_args={{
        '{name}': {{
{chr(10).join(dataset_args_comments)}
        }}
    }},
'''

    python_use_sandbox = '    use_sandbox=True,\n' if sandbox_config else ''

    python_code = f'''from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['{name}'],
{python_use_sandbox}{python_dataset_args}    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)'''

    return f'''## Usage

### Using CLI

```bash
{chr(10).join(cli_lines)}
```

### Using Python

```python
{python_code}
```
'''


def _format_statistics_section_from_dict(statistics: Optional[Dict]) -> str:
    """Format the statistics section from a dictionary."""
    if not statistics or not statistics.get('total_samples'):
        return '*Statistics not available.*'

    lines = []
    total_samples = statistics.get('total_samples', 'N/A')
    prompt_length = statistics.get('prompt_length', {})

    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.append(f'| Total Samples | {_format_sample_count(total_samples)} |')

    if prompt_length:
        lines.append(f'| Prompt Length (Mean) | {prompt_length.get("mean", "N/A")} chars |')
        lines.append(
            f'| Prompt Length (Min/Max) | {prompt_length.get("min", "N/A")} / {prompt_length.get("max", "N/A")} chars |'
        )

    # Subset statistics
    subset_stats = statistics.get('subset_stats', [])
    if subset_stats and len(subset_stats) > 1:
        lines.extend(['', '**Per-Subset Statistics:**', ''])
        lines.append('| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |')
        lines.append('|--------|---------|-------------|------------|------------|')
        for s in subset_stats:
            sample_count = s.get('sample_count', 'N/A')
            lines.append(
                f'| `{s.get("name", "N/A")}` | {_format_sample_count(sample_count)} | '
                f'{s.get("prompt_length_mean", "N/A")} | {s.get("prompt_length_min", "N/A")} | '
                f'{s.get("prompt_length_max", "N/A")} |'
            )

    # Multimodal statistics
    multimodal = statistics.get('multimodal')
    if multimodal:
        lines.append('')
        lines.append(_format_multimodal_statistics_from_dict(multimodal))

    return '\n'.join(lines)


def _format_multimodal_statistics_from_dict(multimodal: Dict) -> str:
    """Format multimodal statistics section from a dictionary."""
    lines = []

    # Image statistics
    if multimodal.get('has_images') and multimodal.get('image'):
        image_stats = multimodal['image']
        lines.append('**Image Statistics:**')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        lines.append(f'| Total Images | {_format_sample_count(image_stats.get("count_total", 0))} |')

        count_per_sample = image_stats.get('count_per_sample', {})
        if count_per_sample:
            lines.append(
                f'| Images per Sample | min: {count_per_sample.get("min", "N/A")}, '
                f'max: {count_per_sample.get("max", "N/A")}, '
                f'mean: {count_per_sample.get("mean", "N/A")} |'
            )

        resolution_range = image_stats.get('resolution_range')
        if resolution_range and (resolution_range.get('min') or resolution_range.get('max')):
            lines.append(
                f'| Resolution Range | {resolution_range.get("min", "N/A")} - '
                f'{resolution_range.get("max", "N/A")} |'
            )

        formats = image_stats.get('formats', [])
        if formats:
            lines.append(f'| Formats | {", ".join(formats)} |')
        lines.append('')

    # Audio statistics
    if multimodal.get('has_audio') and multimodal.get('audio'):
        audio_stats = multimodal['audio']
        lines.append('**Audio Statistics:**')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        lines.append(f'| Total Audio Files | {_format_sample_count(audio_stats.get("count_total", 0))} |')

        count_per_sample = audio_stats.get('count_per_sample', {})
        if count_per_sample:
            lines.append(
                f'| Audio per Sample | min: {count_per_sample.get("min", "N/A")}, '
                f'max: {count_per_sample.get("max", "N/A")}, '
                f'mean: {count_per_sample.get("mean", "N/A")} |'
            )

        duration = audio_stats.get('duration')
        if duration and duration.get('mean') is not None:
            lines.append(
                f'| Duration (sec) | min: {duration.get("min", "N/A")}, '
                f'max: {duration.get("max", "N/A")}, '
                f'mean: {duration.get("mean", "N/A")} |'
            )

        sample_rates = audio_stats.get('sample_rates', [])
        if sample_rates:
            lines.append(f'| Sample Rates | {", ".join(map(str, sample_rates))} Hz |')

        formats = audio_stats.get('formats', [])
        if formats:
            lines.append(f'| Formats | {", ".join(formats)} |')
        lines.append('')

    # Video statistics
    if multimodal.get('has_video') and multimodal.get('video'):
        video_stats = multimodal['video']
        lines.append('**Video Statistics:**')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        lines.append(f'| Total Videos | {_format_sample_count(video_stats.get("count_total", 0))} |')

        count_per_sample = video_stats.get('count_per_sample', {})
        if count_per_sample:
            lines.append(
                f'| Videos per Sample | min: {count_per_sample.get("min", "N/A")}, '
                f'max: {count_per_sample.get("max", "N/A")}, '
                f'mean: {count_per_sample.get("mean", "N/A")} |'
            )

        duration = video_stats.get('duration')
        if duration and duration.get('mean') is not None:
            lines.append(
                f'| Duration (sec) | min: {duration.get("min", "N/A")}, '
                f'max: {duration.get("max", "N/A")}, '
                f'mean: {duration.get("mean", "N/A")} |'
            )

        formats = video_stats.get('formats', [])
        if formats:
            lines.append(f'| Formats | {", ".join(formats)} |')
        lines.append('')

    return '\n'.join(lines)


def _format_sample_example_section_from_dict(sample_example: Optional[Dict]) -> str:
    """Format the sample example section from a dictionary."""
    import json as json_module

    if not sample_example or not sample_example.get('data'):
        return '*Sample example not available.*'

    lines = []
    subset = sample_example.get('subset')
    if subset:
        lines.append(f'**Subset**: `{subset}`')
        lines.append('')
    lines.append('```json')
    lines.append(json_module.dumps(sample_example.get('data', {}), ensure_ascii=False, indent=2))
    lines.append('```')
    if sample_example.get('truncated'):
        lines.append('')
        lines.append('*Note: Some content was truncated for display.*')

    return '\n'.join(lines)


def generate_readme_from_dict(
    name: str,
    meta: Dict[str, Any],
    statistics: Optional[Dict[str, Any]] = None,
    sample_example: Optional[Dict[str, Any]] = None,
    lang: str = 'en',
) -> str:
    """
    Generate complete README content for a benchmark from dictionary data.

    This is the primary function for generating README from persisted JSON data.

    Args:
        name: Benchmark name
        meta: Benchmark metadata dictionary
        statistics: Data statistics dictionary
        sample_example: Sample example dictionary
        lang: Language code ('en' or 'zh') - reserved for future i18n support

    Returns:
        Formatted README markdown string
    """
    # Extract subsets from statistics if available
    subsets = None
    if statistics and statistics.get('subset_stats'):
        subsets = [s.get('name') for s in statistics.get('subset_stats', []) if s.get('name')]

    return README_TEMPLATE.format(
        pretty_name=meta.get('pretty_name') or name,
        description=meta.get('description') or '*No description available.*',
        name=name,
        dataset_id_link=_format_dataset_link(meta.get('dataset_id', '')),
        paper_link=_format_link(meta.get('paper_url'), 'Paper'),
        tags=_format_tags(meta.get('tags', [])),
        metrics=_format_metrics(meta.get('metrics', [])),
        few_shot_num=meta.get('few_shot_num', 0),
        eval_split=meta.get('eval_split') or 'N/A',
        extra_overview_rows=_format_extra_overview_rows(
            meta.get('train_split'),
            meta.get('aggregation'),
        ),
        statistics_section=_format_statistics_section_from_dict(statistics),
        sample_example_section=_format_sample_example_section_from_dict(sample_example),
        prompt_template_section=_format_prompt_template_section(
            meta.get('prompt_template'),
            meta.get('system_prompt'),
            meta.get('few_shot_prompt_template'),
        ),
        extra_params_section=_format_extra_params_section(meta.get('extra_params')),
        sandbox_config_section=_format_sandbox_config_section(meta.get('sandbox_config')),
        usage_section=_format_usage_section(
            name,
            subsets=subsets,
            sandbox_config=meta.get('sandbox_config'),
            extra_params=meta.get('extra_params'),
        ),
    )


# Export utility functions for reuse
__all__ = [
    'generate_readme_from_dict',
    '_format_tags',
    '_format_sample_count',
]
