"""ArxivRollBench prompts must carry the text each task format asks the model to reorder or complete."""
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _prompt(record: dict) -> str:
    adapter = get_benchmark('arxivrollbench', TaskConfig(model='mock', datasets=['arxivrollbench']))
    return adapter.format_prompt_template(adapter.record_to_sample(record))


def test_cloze_prompt_includes_masked_paragraph_and_candidates() -> None:
    # Field layout of the published `*_c` subsets: no `shuffled_text`, the passage lives elsewhere.
    record = {
        'text_with_holes': 'We test the method. <|MaskedSetence|> The results agree with theory.',
        'text_candidates': '**A**: It is fast.\n**B**: It is exact.\n**C**: It is robust.',
        'A': 'ABC',
        'B': 'BAC',
        'C': 'CAB',
        'D': 'CBA',
        'label': 'Selection 2',
    }
    prompt = _prompt(record)
    assert record['text_with_holes'] in prompt
    assert record['text_candidates'] in prompt
    assert _target(record) == 'B'


def test_sequencing_and_prediction_prompts_keep_their_text() -> None:
    sequencing = {'shuffled_text': '**A**: First.\n**B**: Second.', 'A': 'AB', 'B': 'BA', 'C': 'AB', 'D': 'BA',
                  'label': 'Selection 1'}
    prediction = {'context': 'The proof is complete.', 'A': 'x', 'B': 'y', 'C': 'z', 'D': 'w', 'label': 'c'}
    assert sequencing['shuffled_text'] in _prompt(sequencing)
    assert prediction['context'] in _prompt(prediction)
    assert _target(sequencing) == 'A'
    assert _target(prediction) == 'C'


def _target(record: dict) -> str:
    adapter = get_benchmark('arxivrollbench', TaskConfig(model='mock', datasets=['arxivrollbench']))
    return adapter.record_to_sample(record).target
