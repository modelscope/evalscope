# Copyright (c) Alibaba, Inc. and its affiliates.
"""Endpoint listing the plugins the operator enabled for this service."""

from flask import Blueprint

from evalscope.plugins import load_plugins
from evalscope.service.api_models import PluginsResponse

from ..responses import json_response
from ..utils import enabled_plugins

bp_plugins = Blueprint('plugins', __name__, url_prefix='/api/v1/plugins')


@bp_plugins.route('', methods=['GET'])
def list_plugins():
    """Return enabled plugins with their package versions and registered names.

    The dashboard offers the model APIs (``registrations['model_api']``) and perf API
    protocols (``registrations['perf_api']``) listed here next to the built-in ones.
    """
    plugins = [info.model_dump(mode='json') for info in load_plugins(enabled_plugins())]
    return json_response(PluginsResponse, {'plugins': plugins})
