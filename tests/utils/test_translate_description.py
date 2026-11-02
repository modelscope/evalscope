"""Unit tests for fenced code block protection in README translation."""

from types import SimpleNamespace
from unittest.mock import Mock

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


def _client_returning(content: str):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=completion))))


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


class TestProtectCodeBlocks:
    def test_roundtrip(self):
        protected, blocks = _protect_code_blocks(ENGLISH_MD)

        assert len(blocks) == 2
        assert '[CODE_BLOCK_0]' in protected
        assert 'Antenna' not in protected

        assert _restore_code_blocks(protected, blocks) == ENGLISH_MD
