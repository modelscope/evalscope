"""SWE-bench image selection and engine-independent official harness execution."""

import platform
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
from evalscope.api.agent import prepare_environment
from evalscope.api.sandbox import (
    SandboxConfigError,
    SandboxEngine,
    merge_sandbox_config_dicts,
    resolve_task_engine,
    sandbox_image_arch,
    task_sandbox_env_kwargs,
)
from evalscope.utils import get_logger
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner

if TYPE_CHECKING:
    from swebench.harness.test_spec.test_spec import TestSpec

    from evalscope.config import TaskConfig

logger = get_logger()
DEFAULT_DOCKERHUB_USERNAME = 'swebench'
EVAL_COMMAND_TIMEOUT_S = 600
SwebenchArch = Literal['arm64', 'x86_64']
SwebenchForceArch = Literal['', 'arm64', 'x86_64']


def _is_arm_machine() -> bool:
    return platform.machine().lower() in {'aarch64', 'arm64'}


def _requires_x86(instance_id: str) -> bool:
    from swebench.harness.constants import USE_X86

    return instance_id in USE_X86


def resolve_swebench_arch(
    instance_id: str, force_arch: SwebenchForceArch = '', engine: SandboxEngine = SandboxEngine.DOCKER
) -> SwebenchArch:
    """Respect the engine's fixed architecture, otherwise preserve local selection."""
    fixed = sandbox_image_arch(engine)
    if fixed:
        return fixed
    if force_arch:
        return force_arch
    if _is_arm_machine() and not _requires_x86(instance_id):
        return 'arm64'
    return 'x86_64'


def get_remote_docker_image_from_id(
    instance_id: str,
    dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME,
    force_arch: SwebenchForceArch = '',
    engine: SandboxEngine = SandboxEngine.DOCKER,
) -> str:
    """Return the official SWE-bench v3+ per-instance image reference."""
    updated_instance_id = instance_id.replace('__', '_1776_')
    arch = resolve_swebench_arch(instance_id, force_arch, engine)
    username = dockerhub_username or DEFAULT_DOCKERHUB_USERNAME
    return f'{username}/sweb.eval.{arch}.{updated_instance_id}:latest'


def _eval_sandbox_config(task_config: Optional['TaskConfig'], test_spec: 'TestSpec') -> Dict[str, Any]:
    from swebench.harness.constants import DOCKER_WORKDIR

    required: Dict[str, Any] = {
        'image': test_spec.instance_image_key,
        'command': 'tail -f /dev/null',
        'working_dir': DOCKER_WORKDIR,
        'platform': test_spec.platform,
        'tools_config': {'shell_executor': {}},
    }
    cap_add = test_spec.docker_specs.get('run_args', {}).get('cap_add', [])
    if cap_add:
        required['cap_add'] = list(cap_add)
    defaults = task_config.sandbox.default_config if task_config and task_config.sandbox else {}
    if resolve_task_engine(task_config) is SandboxEngine.DOCKER:
        # The original Docker harness imposed no resource limits unless requested.
        defaults = {'memory_limit': None, 'cpu_limit': None, **defaults}
    return merge_sandbox_config_dicts(defaults, required)


@dataclass
class _HarnessRun:
    output: str
    timed_out: bool
    apply_cmd: str
    runtime: float


async def _run_harness(
    env: EnclaveAgentEnvironment, instance_id: str, patch_text: str, eval_script: str, timeout: int
) -> _HarnessRun:
    try:
        from swebench.harness.constants import APPLY_PATCH_FAIL, DOCKER_PATCH, DOCKER_WORKDIR
        from swebench.harness.run_evaluation import GIT_APPLY_CMDS

        with tempfile.TemporaryDirectory() as staging:
            Path(staging, PurePosixPath(DOCKER_PATCH).name).write_text(patch_text)
            Path(staging, 'eval.sh').write_text(eval_script)
            await env.put_dir(staging, str(PurePosixPath(DOCKER_PATCH).parent))
        for cmd in GIT_APPLY_CMDS:
            last = await env.exec(['bash', '-c', f'{cmd} {DOCKER_PATCH}'], cwd=DOCKER_WORKDIR)
            if last.returncode == 0:
                break
        else:
            raise RuntimeError(f'{instance_id} {APPLY_PATCH_FAIL}:\n{last.stdout}{last.stderr}')
        diff_cmd = ['bash', '-c', 'git -c core.fileMode=false diff']
        before = await env.exec(diff_cmd, cwd=DOCKER_WORKDIR)
        logger.info(f'Git diff before:\n{before.stdout.strip()}')
        started = time.monotonic()
        result = await env.exec(['bash', '-c', '/bin/bash /tmp/eval.sh 2>&1'], cwd=DOCKER_WORKDIR, timeout=timeout)
        runtime = time.monotonic() - started
        if not result.timed_out:
            after = await env.exec(diff_cmd, cwd=DOCKER_WORKDIR)
            logger.info(f'Git diff after:\n{after.stdout.strip()}')
            if before.stdout.strip() != after.stdout.strip():
                logger.info('Git diff changed after running eval script')
        return _HarnessRun(result.stdout + result.stderr, result.timed_out, cmd, runtime)
    finally:
        await env.close()


def eval_instance(
    instance: dict,
    pred: str,
    *,
    task_config: Optional['TaskConfig'],
    timeout: int = 1800,
    log_dir: str = 'outputs',
    dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME,
    force_arch: SwebenchForceArch = '',
) -> dict:
    """Evaluate a patch in a clean sandbox following the official harness steps."""
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION, LOG_TEST_OUTPUT
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance_id = instance[KEY_INSTANCE_ID]
    env_kwargs = task_sandbox_env_kwargs(task_config)
    engine = env_kwargs['engine']
    arch = resolve_swebench_arch(instance_id, force_arch, engine)
    test_spec = make_test_spec(instance, namespace=dockerhub_username or DEFAULT_DOCKERHUB_USERNAME, arch=arch)
    patch_text = pred or ''
    instance_log_dir = Path(log_dir) / 'swebench_log' / instance_id
    instance_log_dir.mkdir(parents=True, exist_ok=True)
    (instance_log_dir / 'patch.diff').write_text(patch_text)
    (instance_log_dir / 'eval.sh').write_text(test_spec.eval_script)
    env = EnclaveAgentEnvironment(
        **env_kwargs,
        sandbox_config=_eval_sandbox_config(task_config, test_spec),
        timeout=EVAL_COMMAND_TIMEOUT_S,
        expected_duration=timeout,
    )
    prepare_environment(env, sample_id=instance_id)
    report: Dict[str, Any] = {}
    completed = False
    run = None
    try:
        run = AsyncioLoopRunner.run(_run_harness(env, instance_id, patch_text, test_spec.eval_script, timeout))
        output_path = instance_log_dir / LOG_TEST_OUTPUT
        output_path.write_text(
            run.output + (f'\n\nTimeout error: {timeout} seconds exceeded.' if run.timed_out else '')
        )
        if run.timed_out:
            raise TimeoutError(f'{instance_id} Test timed out after {timeout} seconds.')
        prediction = {KEY_PREDICTION: patch_text, KEY_INSTANCE_ID: instance_id, KEY_MODEL: ''}
        report = get_eval_report(
            test_spec=test_spec, prediction=prediction, test_log_path=output_path, include_tests_status=True
        )
        completed = True
    except SandboxConfigError:
        raise
    except Exception as exc:
        report['error'] = f'Error in evaluating model for {instance_id}: {exc}\n{traceback.format_exc()}'
        logger.error(report['error'])
    resolved = completed and bool(report.get(instance_id, {}).get('resolved', False))
    logger.info(
        f'SWE-bench eval {instance_id} engine={engine.value} apply={run.apply_cmd if run else None} '
        f'runtime={run.runtime if run else None} resolved={resolved}'
    )
    return {'completed': completed, 'resolved': resolved, 'report': report}
