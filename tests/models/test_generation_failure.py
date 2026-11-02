"""Tests for generation-failure classification (issue #1992).

Categories are derived from the openai exception type and machine-readable error
codes — never from message text; anything else stays UNKNOWN.
"""
import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

from evalscope.constants import GenerationFailureCategory
from evalscope.models.utils.openai import classify_openai_error, openai_handle_bad_request


def api_error(cls, status_code, body=None, message='provider error'):
    request = httpx.Request('POST', 'http://test/v1/chat/completions')
    response = httpx.Response(status_code=status_code, request=request)
    return cls(message, response=response, body=body)


def test_auth_error_maps_to_auth_failed():
    failure = classify_openai_error(api_error(AuthenticationError, 401))
    assert failure.category == GenerationFailureCategory.AUTH_FAILED
    assert failure.is_retryable is False
    assert failure.hint


def test_permission_denied_maps_to_auth_failed():
    failure = classify_openai_error(api_error(PermissionDeniedError, 403))
    assert failure.category == GenerationFailureCategory.AUTH_FAILED


def test_rate_limit_maps_to_rate_limited_and_retryable():
    failure = classify_openai_error(api_error(RateLimitError, 429))
    assert failure.category == GenerationFailureCategory.RATE_LIMITED
    assert failure.is_retryable is True


def test_connection_error_and_timeout_are_distinct():
    request = httpx.Request('POST', 'http://test/v1/chat/completions')
    connection = classify_openai_error(APIConnectionError(request=request))
    assert connection.category == GenerationFailureCategory.CONNECTION_FAILED
    assert connection.is_retryable is True

    timeout = classify_openai_error(APITimeoutError(request))
    assert timeout.category == GenerationFailureCategory.TIMEOUT
    assert timeout.is_retryable is True


def test_context_length_exceeded_maps_to_context_overflow():
    failure = classify_openai_error(
        api_error(BadRequestError, 400, body={'message': 'too long', 'code': 'context_length_exceeded'})
    )
    assert failure.category == GenerationFailureCategory.CONTEXT_OVERFLOW
    assert failure.is_retryable is False


def test_content_filter_code_maps_to_content_filtered():
    failure = classify_openai_error(
        api_error(BadRequestError, 400, body={'message': 'blocked', 'code': 'content_policy_violation'})
    )
    assert failure.category == GenerationFailureCategory.CONTENT_FILTERED


def test_unrecognized_bad_request_stays_unknown():
    failure = classify_openai_error(
        api_error(BadRequestError, 400, body={'message': 'bad param', 'code': 'invalid_parameter_error'})
    )
    assert failure.category == GenerationFailureCategory.UNKNOWN
    assert failure.hint == ''


def test_unknown_exception_type_stays_unknown():
    failure = classify_openai_error(ValueError('something else'))
    assert failure.category == GenerationFailureCategory.UNKNOWN


def test_handle_bad_request_attaches_error_info():
    e = api_error(BadRequestError, 400, body={'message': 'too long', 'code': 'context_length_exceeded'})
    output = openai_handle_bad_request('test-model', e)

    assert output is not None
    assert output.error_info is not None
    assert output.error_info.category == GenerationFailureCategory.CONTEXT_OVERFLOW
    assert output.error_info.message == 'too long'
    # The plain string field stays untouched for existing consumers.
    assert output.error is None


def test_error_info_survives_serialization_roundtrip():
    e = api_error(BadRequestError, 400, body={'message': 'too long', 'code': 'context_length_exceeded'})
    output = openai_handle_bad_request('test-model', e)

    restored = output.model_dump()
    assert restored['error_info']['category'] == 'context_overflow'
    # Old result files (error_info absent) still load unchanged.
    assert restored['error'] is None
