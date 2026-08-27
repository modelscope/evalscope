from typing import Any, Type

from flask import Response, jsonify

from .api_models import ApiResponseModel


def json_response(model_type: Type[ApiResponseModel], payload: Any, status: int = 200) -> Response:
    """Validate and serialize one successful JSON API response."""
    model = payload if isinstance(payload, model_type) else model_type.model_validate(payload)
    response = jsonify(model.model_dump(mode='json', by_alias=True, exclude_unset=True))
    response.status_code = status
    return response
