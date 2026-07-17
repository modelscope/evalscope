from evalscope.api.dataset import Sample
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.bbh.bbh_adapter import SUBSET_LIST
from evalscope.benchmarks.bbh.cot_prompts import COT_PROMPTS


def test_all_bbh_subsets_have_cot_prompts() -> None:
    assert set(COT_PROMPTS) == set(SUBSET_LIST)
    assert all(prompt.strip() for prompt in COT_PROMPTS.values())


def test_bbh_adapter_uses_imported_cot_prompt() -> None:
    adapter = get_benchmark('bbh')
    formatted = adapter.format_fewshot_template('fallback', Sample(input='Question?', subset_key='navigate'))

    assert formatted.startswith(COT_PROMPTS['navigate'].strip())
    assert 'fallback' not in formatted
    assert 'Q: Question?' in formatted
