from evalscope.api.metric import SampleScore, Score
from evalscope.api.metric.semantics import MetricIdentity
from evalscope.benchmarks.hallusion_bench.hallusion_bench_adapter import HallusionBenchAdapter
from evalscope.benchmarks.locomo.locomo_adapter import LoCoMoAdapter
from evalscope.benchmarks.longmemeval.longmemeval_adapter import LongMemEvalAdapter
from evalscope.benchmarks.openai_mrcr.openai_mrcr_adapter import OPENAI_MRCR_BINS, OpenAIMRCRAdapter


def _sample(metric_name: str, value: float, **metadata) -> SampleScore:
    return SampleScore(
        score=Score(value={metric_name: value}, main_score_name=metric_name, metadata=metadata),
        sample_id=len(metadata),
        sample_metadata=metadata,
    )


def test_hallusion_aggregation_emits_structured_overall_answer_identity() -> None:
    adapter = HallusionBenchAdapter.__new__(HallusionBenchAdapter)
    scores = [_sample(
        'accuracy',
        1.0,
        category='VD',
        subcategory='VD1',
        set_id='1',
        figure_id='1',
        question_id='1',
    )]

    identities = {score.identity for score in adapter.aggregate_scores(scores)}
    assert MetricIdentity(
        name='accuracy',
        aggregation='mean',
        dimensions={
            'level': 'overall',
            'target': 'answer'
        },
    ) in identities


def test_longmemeval_aggregation_reads_canonical_sample_key() -> None:
    adapter = LongMemEvalAdapter.__new__(LongMemEvalAdapter)
    scores = [_sample('accuracy', 1.0, question_type='single-session-user', is_abstention=False)]

    identities = {score.identity for score in adapter.aggregate_scores(scores)}
    assert MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'overall'}) in identities


def test_locomo_aggregation_moves_question_type_out_of_name() -> None:
    adapter = LoCoMoAdapter.__new__(LoCoMoAdapter)
    scores = [_sample('f1', 0.75, category=1)]

    identities = {score.identity for score in adapter.aggregate_scores(scores)}
    assert MetricIdentity(name='f1', aggregation='mean', dimensions={'scope': 'overall'}) in identities
    assert any(identity.name == 'f1' and 'question_type' in identity.dimensions for identity in identities)


def test_openai_mrcr_aggregation_moves_token_range_out_of_name() -> None:
    adapter = OpenAIMRCRAdapter.__new__(OpenAIMRCRAdapter)
    scores = [_sample('mrcr_score', 0.5, bin_index=0)]

    identities = {score.identity for score in adapter.aggregate_scores(scores)}
    minimum, maximum = OPENAI_MRCR_BINS[0]
    assert MetricIdentity(name='mrcr_score', aggregation='mean', dimensions={'scope': 'overall'}) in identities
    assert MetricIdentity(
        name='mrcr_score',
        aggregation='mean',
        dimensions={
            'min_tokens': minimum,
            'max_tokens': maximum
        },
    ) in identities
