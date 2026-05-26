"""HTTP reverse-proxy bridge.

Listens on a local port and translates inbound Anthropic / OpenAI requests
into calls on EvalScope's ``Model.generate_async``, then translates the
``ModelOutput`` back into the agent's expected response format.  Each
request is keyed by ``trial_id`` so multiple concurrent samples can share
a single proxy instance.
"""

from .server import ModelProxyServer, TrialSession

__all__ = ['ModelProxyServer', 'TrialSession']
