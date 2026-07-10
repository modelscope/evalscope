from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Dict, Tuple


class MetricScope(str, Enum):
    REQUEST = 'request'
    TRACE = 'trace'
    WORKLOAD = 'workload'


class MetricProvenance(str, Enum):
    MEASURED = 'measured'
    SERVER_REPORTED = 'server_reported'
    DERIVED = 'derived'
    ESTIMATED = 'estimated'


class MetricDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    unit: str
    scope: MetricScope
    provenance: MetricProvenance
    aggregation: Tuple[str, ...] = ('average', 'percentile')
    applicability: str = 'successful non-warmup requests with required values'


METRICS: Dict[str, MetricDefinition] = {
    'latency': MetricDefinition(
        id='latency', name='Latency', unit='s', scope=MetricScope.REQUEST, provenance=MetricProvenance.MEASURED
    ),
    'ttft': MetricDefinition(
        id='ttft',
        name='Time to first token',
        unit='ms',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.MEASURED
    ),
    'tpot': MetricDefinition(
        id='tpot',
        name='Time per output token',
        unit='ms',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.DERIVED
    ),
    'itl': MetricDefinition(
        id='itl',
        name='Inter-token latency',
        unit='ms',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.MEASURED
    ),
    'arrival_lag': MetricDefinition(
        id='arrival_lag',
        name='Arrival lag',
        unit='ms',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.MEASURED
    ),
    'input_tokens': MetricDefinition(
        id='input_tokens',
        name='Input tokens',
        unit='tokens',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.SERVER_REPORTED
    ),
    'output_tokens': MetricDefinition(
        id='output_tokens',
        name='Output tokens',
        unit='tokens',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.SERVER_REPORTED
    ),
    'input_throughput': MetricDefinition(
        id='input_throughput',
        name='Input throughput',
        unit='tokens/s',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.DERIVED
    ),
    'output_throughput': MetricDefinition(
        id='output_throughput',
        name='Output throughput',
        unit='tokens/s',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.DERIVED
    ),
    'total_throughput': MetricDefinition(
        id='total_throughput',
        name='Total throughput',
        unit='tokens/s',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.DERIVED
    ),
    'decode_throughput': MetricDefinition(
        id='decode_throughput',
        name='Decode throughput',
        unit='tokens/s',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.DERIVED
    ),
    'decoded_tokens_per_iter': MetricDefinition(
        id='decoded_tokens_per_iter',
        name='Decoded tokens per iteration',
        unit='tokens',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.ESTIMATED
    ),
    'approx_spec_acceptance_rate': MetricDefinition(
        id='approx_spec_acceptance_rate',
        name='Approximate speculative acceptance rate',
        unit='ratio',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.ESTIMATED
    ),
    'draft_acceptance_rate': MetricDefinition(
        id='draft_acceptance_rate',
        name='Draft acceptance rate',
        unit='ratio',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.SERVER_REPORTED
    ),
    'cache_hit_rate': MetricDefinition(
        id='cache_hit_rate',
        name='KV cache hit rate',
        unit='ratio',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.DERIVED
    ),
    'first_turn_ttft': MetricDefinition(
        id='first_turn_ttft',
        name='First-turn TTFT',
        unit='ms',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.MEASURED
    ),
    'subsequent_turn_ttft': MetricDefinition(
        id='subsequent_turn_ttft',
        name='Subsequent-turn TTFT',
        unit='ms',
        scope=MetricScope.REQUEST,
        provenance=MetricProvenance.MEASURED
    ),
}
