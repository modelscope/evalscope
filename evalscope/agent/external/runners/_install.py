"""Installation helpers shared by external runners."""

from typing import TYPE_CHECKING, List, Optional

from evalscope.agent.skills import install_skills_command, skills_from_sample_metadata
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.agent import AgentEnvironment
    from .base import ExternalAgentTask

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
    """Ensure Node.js and npm are available, installing via nodesource if needed."""
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


async def install_task_skills(
    env: 'AgentEnvironment',
    task: 'ExternalAgentTask',
    *,
    home_dir: Optional[str],
    native_install_paths: Optional[List[str]] = None,
    runner_name: str,
) -> None:
    """Copy task skills into paths visible to the wrapped agent CLI."""
    skills = skills_from_sample_metadata(task.metadata)
    if not skills.enabled or not skills.sandbox_dir:
        return

    install_paths = _resolve_install_paths(
        list(skills.install_paths or []) + list(native_install_paths or []),
        home_dir=home_dir,
    )
    command = install_skills_command(skills.sandbox_dir, install_paths)
    if not command:
        return

    result = await env.exec(['bash', '-lc', command], timeout=60)
    if result.returncode != 0:
        detail = ((result.stderr or result.stdout or '').strip() or f'rc={result.returncode}')[-1000:]
        raise RuntimeError(f'{runner_name} failed to install skills: {detail}')


def _resolve_install_paths(install_paths: List[str], *, home_dir: Optional[str]) -> List[str]:
    resolved: List[str] = []
    seen = set()
    for path in install_paths:
        if not path:
            continue
        resolved_path = path.replace('$HOME', home_dir) if home_dir else path
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        resolved.append(resolved_path)
    return resolved


__all__ = ['ensure_node_via_apt', 'install_task_skills', 'node_present']
