import ast
import json
import re
import subprocess
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from evalscope.agent.environments.enclave import EnclaveAgentEnvironment
from evalscope.api.sandbox import SandboxConfigError, merge_sandbox_config_dicts, task_sandbox_env_kwargs
from evalscope.utils.asyncio_runtime import AsyncioLoopRunner
from evalscope.utils.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from evalscope.config import TaskConfig

PRO_REPO_URL = 'https://github.com/scaleapi/SWE-bench_Pro-os.git'
PINNED_COMMIT = 'ca10a60'
DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'evalscope' / 'swe_bench_pro' / 'SWE-bench_Pro-os'
DEFAULT_DOCKERHUB_USERNAME = 'jefzda'


def get_dockerhub_image_uri(instance_id: str, repo: str, dockerhub_username: str = DEFAULT_DOCKERHUB_USERNAME) -> str:
    """Construct DockerHub image URI for an SWE-bench_Pro instance.

    Mirrors helper_code/image_uri.py:get_dockerhub_image_uri from SWE-bench_Pro-os.
    """
    if not repo or '/' not in repo:
        raise ValueError(f"Invalid repo {repo!r}; expected 'org/name' format.")
    repo_base, repo_name_only = repo.lower().split('/')
    hsh = instance_id.replace('instance_', '')

    if instance_id == 'instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan':
        repo_name_only = 'element-web'
    elif 'element-hq' in repo.lower() and 'element-web' in repo.lower():
        repo_name_only = 'element'
        if hsh.endswith('-vnan'):
            hsh = hsh[:-5]
    elif hsh.endswith('-vnan'):
        hsh = hsh[:-5]

    tag = f'{repo_base}.{repo_name_only}-{hsh}'
    if len(tag) > 128:
        tag = tag[:128]
    return f'{dockerhub_username}/sweap-images:{tag}'


def _run_git(args: List[str], cwd: Path) -> None:
    result = subprocess.run(['git', *args], cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} (cwd={cwd}) failed: {result.stderr.strip()}')


def ensure_pro_repo(user_path: str = '') -> str:
    """Return a local path to the SWE-bench_Pro-os repo at the pinned commit.

    If user_path is provided, trust it as-is. Otherwise clone (or reuse) the repo
    at DEFAULT_CACHE_DIR and check out PINNED_COMMIT for reproducibility.
    """
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(f'swe_bench_pro_repo_path={user_path!r} does not exist or is not a directory.')
        return str(p)

    dst = DEFAULT_CACHE_DIR
    manual_hint = (
        f'Auto-clone failed. Manually run:\n'
        f'  git clone {PRO_REPO_URL} {dst}\n'
        f'  git -C {dst} checkout {PINNED_COMMIT}\n'
        f"and pass extra_params.swe_bench_pro_repo_path='{dst}'."
    )

    try:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Cloning {PRO_REPO_URL} to {dst}...')
            result = subprocess.run(['git', 'clone', PRO_REPO_URL, str(dst)], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f'git clone failed: {result.stderr.strip()}')
        else:
            logger.info(f'Reusing existing SWE-bench_Pro-os clone at {dst}.')

        _run_git(['fetch', '--quiet', 'origin'], cwd=dst)
        _run_git(['checkout', '--quiet', PINNED_COMMIT], cwd=dst)
    except Exception as e:
        raise RuntimeError(f'{e}\n\n{manual_hint}') from e

    return str(dst)


def load_instance_resources(repo_path: str, instance_id: str) -> Dict[str, str]:
    """Read run_script.sh, parser.py, base/instance Dockerfile for an instance."""
    repo = Path(repo_path)
    paths = {
        'run_script': repo / 'run_scripts' / instance_id / 'run_script.sh',
        'parser_script': repo / 'run_scripts' / instance_id / 'parser.py',
        'base_dockerfile': repo / 'dockerfiles' / 'base_dockerfile' / instance_id / 'Dockerfile',
        'instance_dockerfile': repo / 'dockerfiles' / 'instance_dockerfile' / instance_id / 'Dockerfile',
    }
    out = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing {key} for {instance_id}: {path}. '
                f'Is the SWE-bench_Pro-os clone at the pinned commit ({PINNED_COMMIT})?'
            )
        out[key] = path.read_text()
    return out


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch
    sections = re.split(r'(?=^diff --git )', patch, flags=re.MULTILINE)
    kept: List[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r'^Binary files .* differ$', section, re.MULTILINE):
            continue
        if re.search(r'^GIT binary patch$', section, re.MULTILINE):
            continue
        kept.append(section)
    return ''.join(kept)


def _parse_test_list(value: Any) -> List[str]:
    """fail_to_pass / pass_to_pass / selected_test_files_to_run come as Python-list strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def build_entry_script(metadata: Dict[str, Any], resources: Dict[str, str]) -> str:
    """Generate the entrypoint shell script run inside the container."""
    before_repo_set_cmd = (metadata.get('before_repo_set_cmd') or '').strip().split('\n')[-1]
    selected_test_files = ','.join(_parse_test_list(metadata.get('selected_test_files_to_run')))
    base_commit = metadata['base_commit']

    env_cmds: List[str] = []
    for dockerfile_content in (resources['base_dockerfile'], resources['instance_dockerfile']):
        for line in dockerfile_content.split('\n'):
            line = line.strip()
            if line.startswith('ENV'):
                env_cmds.append(line.replace('ENV', 'export', 1))
    env_block = '\n'.join(env_cmds)

    return f"""
{env_block}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""


def eval_instance(
    metadata: Dict[str, Any],
    pred: str,
    resources: Dict[str, str],
    image: str,
    *,
    task_config: Optional['TaskConfig'],
    log_dir: str = 'outputs',
    timeout: int = 3600,
    sandbox_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the SWE-bench_Pro evaluation for one instance via ms-enclave.

    ``sandbox_config`` is a base ms_enclave ``DockerSandboxConfig`` dict
    (memory_limit / cpu_limit / platform / network_enabled / ...). Stage-required
    fields (``image``, ``working_dir``) are always injected and cannot be overridden.

    Returns dict: {completed, resolved, report}.
    """
    instance_id = metadata['instance_id']
    workspace_root = Path(log_dir) / 'swe_bench_pro_log' / instance_id
    workspace = workspace_root / 'workspace'
    workspace.mkdir(parents=True, exist_ok=True)

    cleaned_patch = strip_binary_hunks(pred or '')
    if cleaned_patch and not cleaned_patch.endswith('\n'):
        cleaned_patch += '\n'
    files = {
        'patch.diff': cleaned_patch,
        'run_script.sh': resources['run_script'],
        'parser.py': resources['parser_script'],
        'entryscript.sh': build_entry_script(metadata, resources),
    }
    for name, content in files.items():
        (workspace / name).write_text(content if content is not None else '')

    test_patch_dir = workspace_root / 'tests'
    test_patch_dir.mkdir(parents=True, exist_ok=True)
    (test_patch_dir / 'test_patch.patch').write_text(metadata.get('test_patch') or '')

    completed = False
    resolved = False
    output_json: Any = None
    error: str = ''

    defaults = {'tools_config': {'shell_executor': {}}}
    stage_required = {
        'image': image,
        'working_dir': '/app',
        'pull_progress': True,
        'pull_progress_interval': 5.0,
    }
    cfg_dict = merge_sandbox_config_dicts(defaults, sandbox_config, stage_required)
    env = EnclaveAgentEnvironment(
        **task_sandbox_env_kwargs(task_config),
        sandbox_config=cfg_dict,
        timeout=120,
        expected_duration=timeout,
    )

    async def _run() -> Tuple[str, Optional[str]]:
        try:
            await env.prepare()
            await env.put_dir(workspace, '/workspace')
            await env.put_dir(test_patch_dir, '/tests')
            run = await env.exec(['bash', '/workspace/entryscript.sh'], timeout=timeout)
            if run.timed_out:
                raise TimeoutError(f'{instance_id} Test timed out after {timeout} seconds.')
            for name in ('stdout.log', 'stderr.log'):
                log = await env.exec(['cat', f'/workspace/{name}'])
                if log.returncode == 0:
                    (workspace / name).write_text(log.stdout)
            output = await env.exec(['cat', '/workspace/output.json'])
            if output.returncode != 0:
                raise RuntimeError(f'Could not read /workspace/output.json: {output.stderr}')
            return f'{run.stdout}\n{run.stderr}', output.stdout
        finally:
            await env.close()

    try:
        logger.info(f'Running container for {instance_id} on image {image}...')
        container_log, output_text = AsyncioLoopRunner.run(_run(), timeout=timeout + 60)
        try:
            (workspace_root / 'container.log').write_text(container_log)
        except Exception:
            pass

        output_path = workspace / 'output.json'
        if output_text is not None:
            output_path.write_text(output_text)
            output_json = json.loads(output_text)
            completed = True

    except SandboxConfigError:
        raise
    except Exception as e:
        error = f'{e}\n{traceback.format_exc()}'
        logger.error(f'Error evaluating {instance_id}: {error}')

    f2p = _parse_test_list(metadata.get('fail_to_pass'))
    p2p = _parse_test_list(metadata.get('pass_to_pass'))
    report: Dict[str, Any] = {'f2p': f2p, 'p2p': p2p}

    if completed and isinstance(output_json, dict):
        tests = output_json.get('tests') or []
        passed = {t.get('name') for t in tests if t.get('status') == 'PASSED'}
        required = set(f2p) | set(p2p)
        resolved = bool(required) and required.issubset(passed)
        report['tests'] = tests
        report['passed_count'] = len(passed)
        report['required_count'] = len(required)
    if error:
        report['error'] = error

    return {'completed': completed, 'resolved': resolved, 'report': report}
