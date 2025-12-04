# Copyright (c) Alibaba, Inc. and its affiliates.
"""EvalScope Flask Service."""

from .app import app, create_app, run_service

__all__ = ['app', 'create_app', 'run_service']
