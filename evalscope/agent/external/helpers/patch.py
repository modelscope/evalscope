"""Extract a unified-diff patch from an :class:`AgentEnvironment`.

Used by SWE-bench-style benchmarks running through an external CLI
agent: the agent edits files in the sandbox without emitting the
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel that the native
AgentLoop relies on, so the adapter recovers the patch by diffing the
working tree against ``HEAD`` after the run finishes.
"""

from evalscope.api.agent import AgentEnvironment
from evalscope.utils.logger import get_logger

logger = get_logger()


async def extract_patch(env: AgentEnvironment, cwd: str) -> str:
    """Return the agent's tracked-file changes as a unified ``git diff HEAD``.

    Mirrors the SWE-bench ``mini-swe-agent`` Submission step (``git diff
    -- <files>``): only changes to files already tracked by the repo are
    captured. Untracked files (runtime artifacts like Redis AOF, build
    output, log files) are intentionally excluded — the eval container's
    ``git reset --hard {base_commit}`` does not delete them either, so
    including them in the patch causes ``git apply`` to fail with
    ``already exists in working directory``.

    Returns an empty string if the working tree is clean or the diff
    cannot be produced; the caller passes that through to ``match_score``
    which scores it as a failed attempt.

    Trade-off: brand-new source files created by the agent are not
    captured. SWE-bench tasks are predominantly modify-existing; for
    benchmarks needing new-file support, the adapter should override
    this with a pathspec-restricted ``git diff`` over the source tree.
    """
    diff_result = await env.exec(['git', 'diff', 'HEAD'], cwd=cwd)
    if diff_result.returncode != 0:
        logger.warning(
            f'extract_patch: `git diff HEAD` failed in {cwd!r} '
            f'(returncode={diff_result.returncode}): {diff_result.stderr.strip()!r}'
        )
        return ''

    return diff_result.stdout or ''
