"""Shared Node.js probe and nodesource apt installer for Node-based runners.

Four runners (``claude-code``, ``codex``, ``opencode``, ``gemini-cli``) all
need the same two-step sequence when Node.js is absent: install apt
prerequisites (curl, ca-certificates, gnupg) via apt-get, then pull the
nodesource setup script and install ``nodejs``.  This module extracts that
common path so the logic lives in one place.
"""

from typing import TYPE_CHECKING

from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.agent import AgentEnvironment

logger = get_logger()


async def node_present(env: 'AgentEnvironment') -> bool:
    """Return ``True`` when both ``node`` and ``npm`` are on PATH."""
    probe = await env.exec(['bash', '-c', 'command -v node && command -v npm'])
    return probe.returncode == 0


async def ensure_node_via_apt(
    env: 'AgentEnvironment',
    *,
    node_setup_url: str,
    timeout_s: float,
    runner_name: str,
) -> None:
    """Ensure Node.js and npm are available, installing via nodesource if needed.

    No-ops when ``node`` and ``npm`` are already on PATH.  Raises
    ``RuntimeError`` when any install step fails (apt prereqs or nodesource).

    Args:
        env: The agent execution environment.
        node_setup_url: URL of the nodesource distribution setup script
            (e.g. ``https://deb.nodesource.com/setup_22.x``).
        timeout_s: Wall-clock budget (seconds) for each sub-command.
        runner_name: Runner class name used in log and error messages.
    """
    if await node_present(env):
        return
    logger.info(
        f'{runner_name}.setup: installing Node.js via {node_setup_url} '
        f'(one-shot per sample; use a pre-built image for faster iteration).'
    )
    prep = await env.exec(
        [
            'bash',
            '-c',
            'set -e; export DEBIAN_FRONTEND=noninteractive; '
            'apt-get update -qq && '
            'apt-get install -y --no-install-recommends curl ca-certificates gnupg',
        ],
        timeout=timeout_s,
    )
    if prep.returncode != 0:
        raise RuntimeError(
            f'{runner_name}.setup: apt prerequisite install failed (rc={prep.returncode}). '
            f'This runner expects a Debian/Ubuntu-based image with network access, or a '
            f'base image where Node.js is already installed (e.g. node:22-slim). '
            f'stderr={prep.stderr.strip()[-1000:]!r}'
        )
    node = await env.exec(
        [
            'bash',
            '-c',
            'set -e; export DEBIAN_FRONTEND=noninteractive; '
            f'curl -fsSL {node_setup_url} | bash - && '
            'apt-get install -y --no-install-recommends nodejs',
        ],
        timeout=timeout_s,
    )
    if node.returncode != 0:
        raise RuntimeError(
            f'{runner_name}.setup: Node.js install failed (rc={node.returncode}). '
            f'stderr={node.stderr.strip()[-1000:]!r}'
        )


__all__ = ['ensure_node_via_apt', 'node_present']
