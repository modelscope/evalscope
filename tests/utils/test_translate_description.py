"""Unit tests for fenced code block protection in README translation."""

from types import SimpleNamespace
from unittest.mock import Mock

import evalscope.utils.doc_utils.translate_description as translate_description
from evalscope.utils.doc_utils.translate_description import _protect_code_blocks, _restore_code_blocks, translate_readme

ENGLISH_MD = """# AI2D

## Overview

AI2D evaluates diagram understanding.

```json
{"choices": ["Antenna", "Thorax"]}
```

## Usage

```bash
evalscope eval --datasets ai2d
```
"""


def _completion(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _client_sequence(contents: list[str]):
    create = Mock(side_effect=[_completion(content) for content in contents])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _client_returning(content: str):
    return _client_sequence([content])


class TestTranslateReadme:
    def test_localizes_usage_hints_without_changing_code_or_data(self) -> None:
        english = '''# Example

```bash
evalscope eval \\
    --datasets cc_ocr_v2 \\
    --limit 10  # Remove this line for formal evaluation
```

```python
text = '# Remove this line for formal evaluation'
config = {
    # subset_list: ['test']  # optional, evaluate specific subsets
    # extra_params: {}  # uses default extra parameters
    'limit': 10,  # Remove this line for formal evaluation
}
```

```json
{"comment": "# Remove this line for formal evaluation"}
```
'''
        translated = '# Example\n\n[CODE_BLOCK_0]\n\n[CODE_BLOCK_1]\n\n[CODE_BLOCK_2]\n'
        client = _client_returning(translated)
        result = translate_readme(english, 'example', client=client)

        assert client.chat.completions.create.call_args.kwargs['model'] == 'qwen3.8-flash'
        assert '--limit 10  # 正式评估时请删除此行' in result
        assert "# subset_list: ['test']  # 可选，用于评估特定子集" in result
        assert '# extra_params: {}  # 使用默认额外参数' in result
        assert "'limit': 10,  # 正式评估时请删除此行" in result
        assert "text = '# Remove this line for formal evaluation'" in result
        assert '{"comment": "# Remove this line for formal evaluation"}' in result
        assert '[CODE_BLOCK_' not in result

    def test_preserves_code_blocks(self):
        json_block = '```json\n{"choices": ["Antenna", "Thorax"]}\n```'
        bash_block = '```bash\nevalscope eval --datasets ai2d\n```'
        translated = '# AI2D\n\n## 概述\n\nAI2D 评估图表理解。\n\n[CODE_BLOCK_0]\n\n## 用法\n\n[CODE_BLOCK_1]\n'

        result = translate_readme(ENGLISH_MD, 'ai2d', client=_client_returning(translated))

        assert json_block in result
        assert bash_block in result
        assert '概述' in result
        assert '[CODE_BLOCK_' not in result

    def test_rejects_translation_that_dropped_placeholder(self):
        mangled = ENGLISH_MD.replace('Antenna', '触角').replace('AI2D evaluates diagram understanding.', '评估')

        result = translate_readme(ENGLISH_MD, 'ai2d', client=_client_returning(mangled))

        assert result == ''

    def test_preserves_inline_code_with_nested_backticks(self):
        inline_code = '` ```mswea_bash_command ``` `'
        english = f'# Example\n\nExpect one {inline_code} block per turn.\n'
        translated = '# Example\n\n每轮输出一个 [CODE_BLOCK_0] 代码块。\n'

        result = translate_readme(english, 'example', client=_client_returning(translated))

        assert result == '# Example\n\n每轮输出一个 ` ```mswea_bash_command ``` ` 代码块。'

    def test_localizes_standard_headings(self):
        english = '# Example\n\n## Overview\n\nOverview text.\n\n## Agentic Mode\n\nAgent instructions.\n'
        translated = '# Example\n\n## Overview\n\n概述文本。\n\n## Agentic Mode\n\n代理说明。\n'

        result = translate_readme(english, 'example', client=_client_returning(translated))

        assert result == '# Example\n\n## 概述\n\n概述文本。\n\n## Agentic 模式\n\n代理说明。'

    def test_localizes_generated_labels(self):
        translated = (
            '# Example\n\n| Metric | Value |\n| Total Samples | 300 |\n'
            '| Prompt Length (Mean) | 10 chars |\n| Prompt Length (Min/Max) | 1 / 20 chars |\n'
            '| **Default Shots** | 0-shot |\n| **默认 Shots** | 0-shot |\n\n[sandbox guide](https://example.com)\n'
            '[sandbox documentation](https://example.com)\n'
            '[usage documentation](https://example.com)\n<summary>Few-shot Template</summary>\n'
        )
        expected = (
            '# Example\n\n| 指标 | 值 |\n| 总样本数 | 300 |\n'
            '| 提示词长度（平均值） | 10 字符 |\n| 提示词长度（最小/最大值） | 1 / 20 字符 |\n'
            '| **默认示例数** | 0-shot |\n| **默认示例数** | 0-shot |\n\n[沙箱指南](https://example.com)\n'
            '[沙箱文档](https://example.com)\n[使用文档](https://example.com)\n<summary>少样本模板</summary>'
        )

        result = translate_readme('# Example\n', 'example', client=_client_returning(translated))

        assert result == expected

    def test_preserves_generated_labels_inside_code_blocks(self):
        english = '# Example\n\n```markdown\n## Overview\n\n| Metric | Value |\n[sandbox guide](https://example.com)\n<summary>Few-shot Template</summary>\n```\n'
        translated = '# Example\n\n[CODE_BLOCK_0]\n'

        result = translate_readme(english, 'example', client=_client_returning(translated))

        assert result == english.strip()

    def test_stops_split_translation_after_failed_section(self, monkeypatch):
        monkeypatch.setattr(translate_description, '_MAX_TRANSLATION_CHARS', 1)
        english = '# Example\n\n## Overview\n\nEnglish overview.\n\n## Usage\n\nUse English.\n'
        client = _client_sequence(['# Example', '', '## 使用方法\n\n使用。'])

        result = translate_readme(english, 'example', client=client)

        assert result == ''
        assert client.chat.completions.create.call_count == 2

    def test_splits_long_documents_by_section(self, monkeypatch):
        monkeypatch.setattr(translate_description, '_MAX_TRANSLATION_CHARS', 1)
        english = '# Example\n\n## Overview\n\nEnglish overview.\n\n## Usage\n\nUse English.\n'
        client = _client_sequence(['# Example', '## 概述\n\n概述。', '## 使用方法\n\n使用。'])

        result = translate_readme(english, 'example', client=client)

        assert result == '# Example\n\n## 概述\n\n概述。\n\n## 使用方法\n\n使用。'
        assert client.chat.completions.create.call_count == 3

    def test_does_not_split_headings_inside_code_blocks(self, monkeypatch):
        monkeypatch.setattr(translate_description, '_MAX_TRANSLATION_CHARS', 1)
        english = '# Example\n\n## Overview\n\n```text\n## not a section\n```\n\n## Usage\n\nUse English.\n'
        client = _client_sequence(['# Example', '## 概述\n\n[CODE_BLOCK_0]', '## 使用方法\n\n使用。'])

        result = translate_readme(english, 'example', client=client)

        assert result == '# Example\n\n## 概述\n\n```text\n## not a section\n```\n\n## 使用方法\n\n使用。'
        assert client.chat.completions.create.call_count == 3


class TestProtectCodeBlocks:
    def test_roundtrip(self):
        protected, blocks = _protect_code_blocks(ENGLISH_MD)

        assert len(blocks) == 2
        assert '[CODE_BLOCK_0]' in protected
        assert 'Antenna' not in protected

        assert _restore_code_blocks(protected, blocks) == ENGLISH_MD
