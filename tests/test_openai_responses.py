import pytest
from openai.types.responses import Response
from types import SimpleNamespace

from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.model import GenerateConfig
from evalscope.models.openai_responses import OpenAIResponsesAPI
from evalscope.models.utils.openai_responses import (
    chat_choices_from_openai_response,
    openai_response_messages,
    response_text_from_dict,
    response_usage_from_dict,
)
from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_responses_api import OpenAIResponsesPlugin, _extract_sse_data


def test_openai_response_messages_normalize_multimodal_content():
    messages = openai_response_messages([
        ChatMessageUser(
            content=[
                ContentText(text='Describe the image.'),
                ContentImage(image='https://example.com/cat.png', detail='low'),
            ]
        )
    ])

    assert messages == [{
        'role': 'user',
        'content': [
            {
                'type': 'input_text',
                'text': 'Describe the image.'
            },
            {
                'type': 'input_image',
                'image_url': 'https://example.com/cat.png',
                'detail': 'low'
            },
        ],
    }]


def test_openai_responses_provider_builds_native_request():
    api = OpenAIResponsesAPI(
        model_name='gpt-4.1-mini',
        base_url='https://api.openai.com/v1/responses',
        api_key='dummy',
    )
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return Response(
            id='resp_test',
            created_at=0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model='gpt-4.1-mini',
            object='response',
            output=[],
            parallel_tool_calls=True,
            temperature=0,
            tool_choice='auto',
            tools=[],
            top_p=1,
            status='completed',
        )

    api.client.responses.create = fake_create
    api._valid_params = {
        'input',
        'max_output_tokens',
        'model',
        'stream',
        'temperature',
        'tool_choice',
        'tools',
    }

    output = api.generate(
        input=[ChatMessageUser(content='hello')],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(stream=True, max_tokens=16, temperature=0, retries=1),
    )

    assert api.base_url == 'https://api.openai.com/v1'
    assert captured['model'] == 'gpt-4.1-mini'
    assert captured['input'] == [{
        'role': 'user',
        'content': 'hello',
    }]
    assert captured['stream'] is True
    assert captured['max_output_tokens'] == 16
    assert 'messages' not in captured
    assert output.model == 'gpt-4.1-mini'


def test_openai_responses_provider_requires_responses_client():
    api = SimpleNamespace(client=SimpleNamespace())

    with pytest.raises(RuntimeError, match='openai>=1.56.0'):
        OpenAIResponsesAPI._validate_responses_client(api)


def test_openai_responses_perf_build_request_and_endpoint_resolution():
    args = Arguments(
        model='gpt-4.1-mini',
        api='openai_responses',
        url='https://api.openai.com/v1',
        number=1,
        parallel=1,
        stream=True,
        max_tokens=64,
    )
    plugin = OpenAIResponsesPlugin(args)

    request = plugin.build_request([{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}])

    assert args.url == 'https://api.openai.com/v1/responses'
    assert request['model'] == 'gpt-4.1-mini'
    assert request['stream'] is True
    assert request['max_output_tokens'] == 64
    assert request['input'] == [{
        'role': 'user',
        'content': [{
            'type': 'input_text',
            'text': 'hello'
        }],
    }]


def test_openai_responses_perf_warns_only_for_multiple_choices(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        'evalscope.perf.plugin.api.openai_responses_api.logger.warning',
        lambda msg: warnings.append(msg),
    )

    args = Arguments(
        model='gpt-4.1-mini',
        api='openai_responses',
        url='https://api.openai.com/v1/responses',
        number=1,
        parallel=1,
        n_choices=1,
    )
    OpenAIResponsesPlugin(args).build_request('hello')
    assert warnings == []

    args = Arguments(
        model='gpt-4.1-mini',
        api='openai_responses',
        url='https://api.openai.com/v1/responses',
        number=1,
        parallel=1,
        n_choices=2,
    )
    OpenAIResponsesPlugin(args).build_request('hello')
    assert warnings == ['OpenAI Responses API does not support n_choices > 1; ignoring --n-choices.']


def test_openai_responses_perf_defaults_to_responses_endpoint_and_rejects_tokenized_prompt():
    args = Arguments(
        model='gpt-4.1-mini',
        api='openai_responses',
        number=1,
        parallel=1,
    )

    assert args.url == 'http://127.0.0.1:8877/v1/responses'

    with pytest.raises(ValueError, match='--tokenize-prompt is not supported'):
        Arguments(
            model='gpt-4.1-mini',
            api='openai_responses',
            number=1,
            parallel=1,
            tokenize_prompt=True,
            tokenizer_path='dummy',
        )


def test_openai_responses_perf_parse_stream_events():
    args = Arguments(
        model='gpt-4.1-mini',
        api='openai_responses',
        url='https://api.openai.com/v1/responses',
        number=1,
        parallel=1,
    )
    plugin = OpenAIResponsesPlugin(args)

    responses = [
        {
            'type': 'response.output_text.delta',
            'delta': 'hel',
        },
        {
            'type': 'response.output_text.delta',
            'delta': 'lo',
        },
        {
            'type': 'response.completed',
            'response': {
                'usage': {
                    'input_tokens': 3,
                    'output_tokens': 2,
                }
            },
        },
    ]

    assert plugin.parse_responses(responses, request='{}') == (3, 2)
    assert plugin._collect_output_text(responses)[0] == ['hel', 'lo']


def test_openai_responses_extracts_semantic_sse_data():
    message = 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hi"}'

    assert _extract_sse_data(message) == '{"type":"response.output_text.delta","delta":"hi"}'


def test_response_dict_text_and_usage_parsing():
    payload = {
        'output': [{
            'type': 'message',
            'content': [{
                'type': 'output_text',
                'text': 'hello'
            }],
        }],
        'usage': {
            'input_tokens': 3,
            'output_tokens': 1,
            'total_tokens': 4,
        },
    }

    assert response_text_from_dict(payload) == 'hello'
    assert response_usage_from_dict(payload) == (3, 1)


def test_chat_choices_from_openai_response_object():
    response = SimpleNamespace(
        id='resp_1',
        model='gpt-4.1-mini',
        status='completed',
        incomplete_details=None,
        output=[SimpleNamespace(
            type='message',
            content=[
                SimpleNamespace(type='output_text', text='hello'),
            ],
        )],
    )

    choices = chat_choices_from_openai_response(response, tools=[])

    assert len(choices) == 1
    assert choices[0].message.text == 'hello'
    assert choices[0].stop_reason == 'stop'
