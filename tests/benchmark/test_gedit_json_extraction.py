"""gedit: the payload is the last JSON object, even when the reply shows an example first."""

import json

from evalscope.benchmarks.image_edit.gedit.utils import mllm_output_to_dict, write_entry_to_json_file


def test_example_object_before_the_answer_is_ignored() -> None:
    text = 'Example: {"score": 0, "reasoning": "template"}. Answer: {"score": 4, "reasoning": "ok"}'

    assert mllm_output_to_dict(text) == {'score': [4], 'reasoning': 'ok'}


def test_delimited_payload_still_parses() -> None:
    assert mllm_output_to_dict('||V^=^V||{"score": 4}||V^=^V||') == {'score': [4]}


def test_unquoted_keys_are_repaired() -> None:
    assert mllm_output_to_dict('The answer is {score: 4, reasoning: ok}') == {'score': [4], 'reasoning': 'ok'}


def test_list_of_scores_fallback_is_kept() -> None:
    assert mllm_output_to_dict('[6, 0]') == {'score': [6, 0], 'reasoning': 'System: output is simply a list of scores'}


def test_bare_score_fallback_is_kept() -> None:
    assert mllm_output_to_dict('7') == {'score': [7], 'reasoning': 'System: output is simply a number'}


def test_guessed_score_fallback_is_kept() -> None:
    parsed = mllm_output_to_dict('cannot parse this', give_up_parsing=True)

    assert isinstance(parsed, dict)
    assert 0 <= parsed['score'][0] <= 10
    assert parsed['reasoning'].startswith('guess_if_cannot_parse | ')


def test_unparseable_output_returns_false() -> None:
    assert mllm_output_to_dict('cannot parse this') is False


def test_write_entry_uses_the_last_object(tmp_path) -> None:
    output = tmp_path / 'scores.json'

    written = write_entry_to_json_file(
        'Example: {"score": 0}. Answer: {"score": 8}', 'sample-1', 'prompt', None, str(output)
    )

    assert written is True
    assert json.loads(output.read_text(encoding='utf-8')) == {'sample-1': {'score': 8, 'prompt_input': 'prompt'}}
