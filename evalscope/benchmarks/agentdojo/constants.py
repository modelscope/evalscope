"""Constants for the AgentDojo benchmark; importable without the upstream ``agentdojo`` package."""

from typing import Tuple

# Pinned AgentDojo task-suite version and upstream package version.
BENCHMARK_VERSION = 'v1.2.2'
UPSTREAM_VERSION = '0.1.35'

# Supported task suites, in the fixed evaluation and index order.
SUITES: Tuple[str, ...] = ('workspace', 'travel', 'banking', 'slack')

# Attack name for benign runs (no injections are written).
BENIGN_ATTACK = 'none'

# All supported attack names, matched case-sensitively. DoS and manual attacks are intentionally excluded.
SUPPORTED_ATTACKS: Tuple[str, ...] = (
    'important_instructions',
    'important_instructions_no_user_name',
    'important_instructions_no_model_name',
    'important_instructions_no_names',
    'important_instructions_wrong_model_name',
    'important_instructions_wrong_user_name',
    'tool_knowledge',
    'direct',
    'ignore_previous',
    'system_message',
    'injecagent',
    BENIGN_ATTACK,
)

DEFAULT_ATTACK = 'important_instructions'

# Upstream fallback value for the ``{model}`` placeholder in injection templates.
DEFAULT_MODEL_NAME = 'the AI language model'

AGENTDOJO_INSTALL_HINT = "pip install 'evalscope[agentdojo]'"

# Keys under ``Sample.metadata`` / ``TaskState.metadata``.
SAMPLE_META_KEY = 'agentdojo'
RUN_RECORD_META_KEY = 'agentdojo_run'
