"""Unit tests for fenced code block protection in README translation."""

from types import SimpleNamespace

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
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))


class TestTranslateReadme:
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
