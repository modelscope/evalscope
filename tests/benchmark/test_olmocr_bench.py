"""Unit tests for the olmOCR-Bench scoring rules and adapter logic.

The rule classes are a 1:1 port of the official olmocr bench implementation, so the assertions
here double as fidelity checks: examples are taken from (or shaped like) the released bench data.
"""
import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.benchmarks.olmocr_bench.olmocr_bench_adapter import (
    PROMPT_TEMPLATE,
    SUBSET_LIST,
    UNSUPPORTED_SUBSETS,
    OlmocrBenchAdapter,
)
from evalscope.benchmarks.olmocr_bench.table_parsing import parse_html_tables, parse_markdown_tables
from evalscope.benchmarks.olmocr_bench.unit_tests import (
    BaselineTest,
    RepeatDetector,
    TableTest,
    TextOrderTest,
    TextPresenceTest,
    ValidationError,
    load_single_test,
    normalize_text,
)


def make_presence_test(**overrides) -> TextPresenceTest:
    data = {
        'pdf': 'long_tiny_text/14a_pg1.pdf',
        'page': 1,
        'id': '14a_pg1_text_01',
        'type': 'present',
        'max_diffs': 1,
        'text': 'The Aftonian deposits consist of ancient soil profiles.',
    }
    data.update(overrides)
    return TextPresenceTest(**data)


def make_task_state(metadata: dict) -> TaskState:
    return TaskState(model='test-model', sample=Sample(input='transcribe this page', metadata=metadata))


class TestNormalizeText:

    def test_collapses_whitespace_and_strips_markdown_emphasis(self) -> None:
        assert normalize_text('**bold** and _italic_') == 'bold and italic'
        assert normalize_text('a\n\nb\tc') == 'a b c'

    def test_replaces_fancy_unicode_with_ascii(self) -> None:
        assert normalize_text('‘quoted’ – dash µm') == "'quoted' - dash μm"

    def test_none_stays_none(self) -> None:
        assert normalize_text(None) is None


class TestTextPresence:

    def test_present_passes_when_text_is_transcribed(self) -> None:
        test = make_presence_test()
        passed, _ = test.run('The Aftonian deposits consist of ancient soil profiles that may also include peat.')
        assert passed

    def test_present_fails_when_text_is_missing(self) -> None:
        test = make_presence_test()
        passed, _ = test.run('Completely unrelated transcription.')
        assert not passed

    def test_max_diffs_allows_fuzzy_matches(self) -> None:
        test = make_presence_test(max_diffs=5)
        passed, _ = test.run('The Aftonian deposits consist 0f ancient soil profilez that may also include peat.')
        assert passed

    def test_absent_passes_when_text_is_excluded(self) -> None:
        # Real headers_footers rule shape: page numbers must not survive transcription
        test = make_presence_test(type='absent', text='Page 3 of 42', case_sensitive=False)
        assert test.run('Chapter One\n\nIt was the best of times.')[0]
        assert not test.run('It was the best of times. Page 3 of 42')[0]

    def test_first_n_constrains_the_search_window(self) -> None:
        test = make_presence_test(text='Encyclopaedia Britannica', first_n=50)
        assert test.run('Encyclopaedia Britannica, vol. 1. ' + 'filler ' * 200)[0]
        assert not test.run('filler ' * 200 + ' Encyclopaedia Britannica')[0]


class TestTextOrder:

    def make_order_test(self, **overrides) -> TextOrderTest:
        data = {
            'pdf': 'multi_column/abc_pg1.pdf',
            'page': 1,
            'id': 'abc_pg1_order_01',
            'type': 'order',
            'max_diffs': 2,
            'before': 'Results and Discussion',
            'after': 'Materials and Methods',
        }
        data.update(overrides)
        return TextOrderTest(**data)

    def test_passes_when_before_precedes_after(self) -> None:
        test = self.make_order_test()
        assert test.run('Results and Discussion\n\nWe observe...\n\nMaterials and Methods\n\nSamples...')[0]

    def test_fails_when_order_is_reversed(self) -> None:
        test = self.make_order_test()
        assert not test.run('Materials and Methods\n\nSamples...\n\nResults and Discussion\n\nWe observe...')[0]

    def test_fails_when_either_span_is_missing(self) -> None:
        test = self.make_order_test()
        assert not test.run('Results and Discussion only')[0]

    def test_rejects_max_diffs_over_half_the_span(self) -> None:
        # Official rule: max_diffs must not exceed len(span) // 2 ('Materials and Methods' -> 10)
        with pytest.raises(ValidationError):
            self.make_order_test(max_diffs=11)


class TestTableRules:

    MD_TABLE = (
        '| Name | Age | City |\n'
        '|------|-----|------|\n'
        '| Alice | 30 | Springfield |\n'
        '| Bob | 41 | Shelbyville |\n'
    )

    def make_table_test(self, **overrides) -> TableTest:
        data = {
            'pdf': 'table_tests/188_pg1.pdf',
            'page': 1,
            'id': '188_pg1_table_01',
            'type': 'table',
            'max_diffs': 2,
            'cell': 'Bob',
            'up': 'Alice',
        }
        data.update(overrides)
        return TableTest(**data)

    def test_markdown_table_cell_relation(self) -> None:
        test = self.make_table_test()
        assert test.run(self.MD_TABLE)[0]

    def test_wrong_relation_fails(self) -> None:
        test = self.make_table_test(up='Shelbyville')
        assert not test.run(self.MD_TABLE)[0]

    def test_top_heading_relation(self) -> None:
        test = self.make_table_test(up='', top_heading='Age')
        assert test.run(self.MD_TABLE)[0]

    def test_no_table_fails(self) -> None:
        test = self.make_table_test()
        assert not test.run('No tables here')[0]

    def test_html_table_with_rowspan(self) -> None:
        html = (
            '<table>'
            '<tr><th rowspan="2">Region</th><th>2023</th></tr>'
            '<tr><td>2024</td></tr>'
            '<tr><td>North</td><td>1,204</td></tr>'
            '</table>'
        )
        test = self.make_table_test(cell='1,204', left='North', top_heading='2023', up='')
        assert test.run(html)[0]

    def test_markdown_and_html_parsers_agree_on_simple_tables(self) -> None:
        md_tables = parse_markdown_tables(self.MD_TABLE)
        html = '<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>'
        html_tables = parse_html_tables(html)
        assert md_tables[0].cell_text[(0, 0)] == 'Name'
        assert html_tables[0].cell_text[(0, 0)] == 'Name'
        assert 'Springfield' in md_tables[0].cell_text.values()


class TestBaseline:

    def make_baseline_test(self, **overrides) -> BaselineTest:
        data = {'pdf': 'headers_footers/blank_pg2.pdf', 'page': 2, 'id': 'blank_pg2_base_01', 'type': 'baseline'}
        data.update(overrides)
        return BaselineTest(**data)

    def test_blank_page_check(self) -> None:
        test = self.make_baseline_test(max_length=5)
        assert test.run('')[0]
        assert not test.run('This page has plenty of real content.')[0]

    def test_blank_page_check_skips_image_alt_tags(self) -> None:
        test = self.make_baseline_test(max_length=5, max_length_skips_image_alt_tags=True)
        assert test.run('![A large scanned figure description](figure_1.png)')[0]

    def test_empty_content_fails_without_max_length(self) -> None:
        assert not self.make_baseline_test().run('!!! ...')[0]

    def test_trailing_repetition_fails(self) -> None:
        # The official detector only looks at n-grams up to 5 characters, so the repeated unit
        # must have a period <= 5 (a longer period such as 'the end ' is invisible to it).
        test = self.make_baseline_test(max_repeats=5)
        assert not test.run('Introduction. ' + 'the ' * 50)[0]

    def test_disallowed_characters_fail(self) -> None:
        assert not self.make_baseline_test().run('Regular text with 中文 characters.')[0]


class TestRepeatDetector:

    def test_counts_trailing_repeats(self) -> None:
        detector = RepeatDetector(max_ngram_size=3)
        detector.add_letters('abab')
        assert detector.ngram_repeats() == [1, 2, 1]

    def test_no_repeats(self) -> None:
        detector = RepeatDetector(max_ngram_size=3)
        detector.add_letters('abc')
        assert detector.ngram_repeats() == [1, 1, 1]


class TestLoadSingleTest:

    def test_dispatches_by_type(self) -> None:
        assert isinstance(load_single_test({
            'pdf': 'a.pdf', 'page': 1, 'id': 'x1', 'type': 'present', 'text': 'hello',
        }), TextPresenceTest)
        assert isinstance(load_single_test({
            'pdf': 'a.pdf', 'page': 1, 'id': 'x2', 'type': 'order', 'before': 'a', 'after': 'b',
        }), TextOrderTest)

    def test_accepts_a_json_line(self) -> None:
        test = load_single_test('{"pdf": "a.pdf", "page": 1, "id": "x3", "type": "baseline"}')
        assert isinstance(test, BaselineTest)

    def test_math_type_is_rejected_with_a_clear_message(self) -> None:
        with pytest.raises(ValidationError, match='math'):
            load_single_test({'pdf': 'a.pdf', 'page': 1, 'id': 'x4', 'type': 'math', 'math': 'x^2'})

    def test_duplicate_ids_are_not_the_loaders_concern(self) -> None:
        # load_single_test validates one rule at a time; duplicate-id detection is the official
        # CLI loader's job and is unnecessary here because pages are grouped, not appended.
        first = load_single_test({'pdf': 'a.pdf', 'page': 1, 'id': 'dup', 'type': 'baseline'})
        second = load_single_test({'pdf': 'b.pdf', 'page': 1, 'id': 'dup', 'type': 'baseline'})
        assert first.id == second.id


class TestAdapterContract:

    def make_adapter(self) -> OlmocrBenchAdapter:
        return OlmocrBenchAdapter.__new__(OlmocrBenchAdapter)

    def test_prompt_matches_the_official_no_anchoring_prompt(self) -> None:
        # Verbatim official prompt (olmocr/bench/prompts.py); changes here invalidate score
        # comparability, so the adapter must keep it byte-identical.
        assert PROMPT_TEMPLATE == (
            'Below is the image of one page of a PDF document. '
            'Just return the plain text representation of this document as if you were reading it '
            'naturally.\n'
            'Turn equations into a LaTeX representation, and tables into markdown format. '
            'Remove the headers and footers, but keep references and footnotes.\n'
            'Read any natural handwriting.\n'
            'This is likely one page out of several in the document, so be sure to preserve any '
            'sentences that come from the previous page, or continue onto the next page, exactly '
            'as they are.\n'
            'If there is no text at all that you think you should read, you can output null.\n'
            'Do not hallucinate.'
        )

    def test_math_only_subsets_are_excluded(self) -> None:
        assert set(UNSUPPORTED_SUBSETS) == {'arxiv_math', 'old_scans_math'}
        assert not set(UNSUPPORTED_SUBSETS) & set(SUBSET_LIST)

    def test_extract_answer_maps_null_to_empty(self) -> None:
        adapter = self.make_adapter()
        assert adapter.extract_answer('null', None) == ''
        assert adapter.extract_answer('  Null ', None) == ''
        assert adapter.extract_answer('# Heading\n\nBody text', None) == '# Heading\n\nBody text'

    def test_match_score_counts_passed_rules(self) -> None:
        adapter = self.make_adapter()
        sample_metadata = {
            'pdf': 'long_tiny_text/14a_pg1.pdf',
            'page': 1,
            'tests': [
                {'pdf': 'long_tiny_text/14a_pg1.pdf', 'page': 1, 'id': 't1', 'type': 'present',
                 'max_diffs': 1, 'text': 'ancient soil profiles'},
                {'pdf': 'long_tiny_text/14a_pg1.pdf', 'page': 1, 'id': 't2', 'type': 'present',
                 'max_diffs': 1, 'text': 'a phrase that was never transcribed'},
            ],
        }
        task_state = make_task_state(sample_metadata)

        score = adapter.match_score(
            original_prediction='The deposits consist of ancient soil profiles.',
            filtered_prediction='The deposits consist of ancient soil profiles.',
            reference='',
            task_state=task_state,
        )
        assert score.value['tests_passed'] == 1.0
        assert score.value['tests_total'] == 2.0
        assert score.value['pass_rate'] == 0.5
        assert score.main_score_name == 'pass_rate'
        assert len(score.metadata['failed_tests']) == 1

    def test_aggregate_uses_test_level_pass_rate(self) -> None:
        from evalscope.api.metric import SampleScore, Score

        adapter = self.make_adapter()

        def make_sample_score(passed: int, total: int, sample_id: int) -> SampleScore:
            score = Score(extracted_prediction='', prediction='')
            score.value = {'pass_rate': passed / total, 'tests_passed': float(passed),
                           'tests_total': float(total)}
            return SampleScore(score=score, sample_id=sample_id)

        # 4/5 + 0/1 must aggregate to 4/6 at the test level, not to the page mean (0.4 + 0.0) / 2
        aggregated = adapter.aggregate_scores([make_sample_score(4, 5, 1), make_sample_score(0, 1, 2)])
        assert len(aggregated) == 1
        assert aggregated[0].metric_name == 'pass_rate'
        assert aggregated[0].score == pytest.approx(4 / 6)
        assert aggregated[0].num == 6

    def test_empty_metadata_scores_zero(self) -> None:
        adapter = self.make_adapter()
        task_state = make_task_state({})
        score = adapter.match_score('anything', 'anything', '', task_state)
        assert score.value == {'pass_rate': 0.0, 'tests_passed': 0, 'tests_total': 0}
