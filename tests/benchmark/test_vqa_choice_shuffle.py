from base64 import b64decode

import pytest

from evalscope.api.dataset.loader import DictDataLoader
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ContentImage, ContentText
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize('name', ['science_qa', 'a_okvqa'])
@pytest.mark.parametrize('shuffle', [False, True])
def test_visible_choice_matches_scored_label(name: str, shuffle: bool) -> None:
    choices = ['red', 'blue', 'green', 'yellow']
    # Fixed PNG bytes keep the content-derived shuffle seed reproducible.
    image = b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP8z4AdMOEQH6QSAM1BAQ/oQeJvAAAAAElFTkSuQmCC'
    )
    record = {
        'question': 'Which color is shown?',
        'hint': 'Look at the square.',
        'choices': choices,
        'image': {'bytes': image},
        'answer': 0,
        'correct_choice_idx': 0,
    }
    config = TaskConfig(datasets=[name], dataset_args={name: {'shuffle_choices': shuffle}}, seed=7)
    adapter = get_benchmark(name, config)
    sample = DictDataLoader(
        [record], sample_fields=adapter.record_to_sample, shuffle_choices=adapter.shuffle_choices, seed=config.seed
    ).load()[0]
    original_image = sample.input[0].content[1]
    sample.input = adapter.process_sample_messages_input(sample, 'default')
    text, picture = sample.input[-1].content
    assert isinstance(text, ContentText)
    assert isinstance(picture, ContentImage) and picture == original_image
    assert 'Which color is shown?' in text.text
    if name == 'science_qa':
        assert 'Context: Look at the square.' in text.text
    if shuffle:
        assert sample.target != 'A'
    visible = text.text.splitlines()[-len(choices):]
    assert visible == [f'{letter}) {choice}' for letter, choice in zip('ABCD', sample.choices)]
    for line in visible:
        letter, choice = line.split(') ', 1)
        state = TaskState(
            model='mock', sample=sample, output=ModelOutput.from_content('mock', f'ANSWER: {letter}'), completed=True
        )
        assert adapter.calculate_metrics(state).score.main_value == float(choice == 'red')
