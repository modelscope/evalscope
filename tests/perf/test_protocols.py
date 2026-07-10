from evalscope.perf.config.models import GenerationConfig, TargetConfig
from evalscope.perf.domain.observation import TokenSource, TransportEvent
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.protocols.openai_chat import OpenAIChatProtocol
from evalscope.perf.protocols.openai_completions import OpenAICompletionsProtocol
from evalscope.perf.protocols.openai_embedding import OpenAIEmbeddingProtocol
from evalscope.perf.protocols.openai_responses import OpenAIResponsesProtocol


def _event(kind, data=None, timestamp=1.0, status_code=None):
    return TransportEvent(kind=kind, data=data, timestamp=timestamp, status_code=status_code)


def test_chat_non_stream_usage_and_text() -> None:
    protocol = OpenAIChatProtocol(TargetConfig(model='model'))
    request = protocol.build_request(SingleTurnItem(messages='hello'), GenerationConfig(stream=False))
    state = protocol.new_state(request)
    for event in [
        _event('request_start', timestamp=1),
        _event('response_start', timestamp=1.1, status_code=200),
        _event(
            'json',
            {
                'choices': [{
                    'message': {
                        'content': 'world'
                    }
                }],
                'usage': {
                    'prompt_tokens': 2,
                    'completion_tokens': 1
                }
            },
            timestamp=2,
        ),
        _event('response_end', timestamp=2),
    ]:
        protocol.consume_event(state, event)
    assert state.success
    assert state.generated_text == 'world'
    assert state.first_token_time == 2
    assert state.prompt_token_source == TokenSource.SERVER_REPORTED
    assert state.completion_token_source == TokenSource.SERVER_REPORTED


def test_responses_stream_events() -> None:
    protocol = OpenAIResponsesProtocol(TargetConfig(model='model', protocol='openai_responses'))
    request = protocol.build_request(SingleTurnItem(messages='hello'), GenerationConfig())
    state = protocol.new_state(request)
    protocol.consume_event(state, _event('request_start', timestamp=1))
    protocol.consume_event(
        state,
        _event('sse', 'data: {"type":"response.output_text.delta","delta":"ok"}', timestamp=1.5),
    )
    protocol.consume_event(
        state,
        _event(
            'sse',
            'data: {"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":1}}}',
            timestamp=2,
        ),
    )
    protocol.consume_event(state, _event('response_end', timestamp=2))
    assert state.generated_text == 'ok'
    assert state.prompt_tokens == 3
    assert state.completion_tokens == 1


def test_embedding_protocol_has_no_generation_fields() -> None:
    protocol = OpenAIEmbeddingProtocol(TargetConfig(model='model', protocol='openai_embedding'))
    request = protocol.build_request(SingleTurnItem(messages=['a', 'b']), GenerationConfig(max_tokens=99))
    assert request.body == {'model': 'model', 'input': ['a', 'b']}


def test_completions_protocol_accepts_token_ids() -> None:
    protocol = OpenAICompletionsProtocol(TargetConfig(model='model', protocol='openai_completions'))
    request = protocol.build_request(SingleTurnItem(messages=[1, 2, 3]), GenerationConfig(stream=False))
    assert request.url.endswith('/completions')
    assert request.body['prompt'] == [1, 2, 3]
