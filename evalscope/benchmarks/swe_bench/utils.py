import platform
import traceback
from pathlib import Path, PurePosixPath

from evalscope.utils import get_logger

logger = get_logger()


def get_remote_docker_image_from_id(instance_id: str) -> str:
    """Image name format as found on DockerHub since swebench v3.0"""
    # NOTE: The swebench library contains this logic within `make_test_spec`,
    # but this module would require significant refactoring to use it.
    updated_instance_id = instance_id.replace('__', '_1776_')
    if platform.machine() in {'aarch64', 'arm64'}:
        from swebench.harness.constants import USE_X86  # type: ignore

        # use arm64 unless explicitly specified
        arch = 'arm64' if instance_id not in USE_X86 else 'x86_64'
    else:
        arch = 'x86_64'
    return f'swebench/sweb.eval.{arch}.{updated_instance_id}:latest'


GIT_APPLY_CMDS = [
    'git apply --verbose',
    'git apply --verbose --reject',
    'patch --batch --fuzz=5 -p1 -i',
]


def eval_instance(instance: dict, pred: str, timeout: int = 1800, log_dir: str = 'outputs'):
    from docker.client import DockerClient
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        APPLY_PATCH_PASS,
        DOCKER_PATCH,
        DOCKER_USER,
        DOCKER_WORKDIR,
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
        LOG_TEST_OUTPUT,
        UTF8,
    )
    from swebench.harness.docker_utils import cleanup_container, copy_to_container, exec_run_with_timeout
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

    from .build_images import build_container

    # Build + start instance container (instance image should already be built)
    container = None
    eval_completed = False
    report = {}
    try:
        test_spec: TestSpec = make_test_spec(instance, namespace='swebench')
        instance_id = test_spec.instance_id
        pred = {
            KEY_PREDICTION: pred,
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: '',
        }
        log_dir = Path(log_dir) / 'swebench_log' / instance_id

        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Starting evaluation for {instance_id} in log dir {log_dir}...')

        client = DockerClient.from_env()
        container = build_container(test_spec, client=client)
        container.start()
        logger.info(f'Container for {instance_id} started: {container.id}')

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / 'patch.diff')
        patch_file.write_text(pred[KEY_PREDICTION] or '')
        logger.info(f'Intermediate patch for {instance_id} written to {patch_file}, now applying to container...')
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        # Attempt to apply patch to container
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f'{git_apply_cmd} {DOCKER_PATCH}',
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f'{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}')
                applied_patch = True
                break
            else:
                logger.info(f'Failed to apply patch to container: {git_apply_cmd}')
        if not applied_patch:
            logger.info(f'{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}')
            raise Exception(f'{instance_id} {APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}')

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run('git -c core.fileMode=false diff', workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()
        )
        logger.info(f'Git diff before:\n{git_diff_output_before}')

        eval_file = Path(log_dir / 'eval.sh')
        eval_file.write_text(test_spec.eval_script)
        logger.info(f'Eval script for {instance_id} written to {eval_file}; copying to container...')
        copy_to_container(container, eval_file, PurePosixPath('/eval.sh'))

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, '/bin/bash /eval.sh', timeout)
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_path, 'w') as f:
            f.write(test_output)
            logger.info(f'Test output for {instance_id} written to {test_output_path}')
            if timed_out:
                f.write(f'\n\nTimeout error: {timeout} seconds exceeded.')
                raise TimeoutError(f'{instance_id} Test timed out after {timeout} seconds.')

        # Get git diff after running eval script (ignore permission changes)
        git_diff_output_after = (
            container.exec_run('git -c core.fileMode=false diff', workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()
        )

        # Check if git diff changed after running eval script
        logger.info(f'Git diff after:\n{git_diff_output_after}')
        if git_diff_output_after != git_diff_output_before:
            logger.info('Git diff changed after running eval script')

        # Get report from test output
        logger.info(f'Grading answer for {instance_id}...')
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(f'report: {report}\n'
                    f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}")
        eval_completed = True
    except Exception as e:
        error_msg = (f'Error in evaluating model for {instance_id}: {e}\n'
                     f'{traceback.format_exc()}')
        logger.error(error_msg)
        report['error'] = error_msg
    finally:
        # Remove instance container
        cleanup_container(client, container, logger)
        return {
            'completed': eval_completed,
            'resolved': report.get(instance_id, {}).get('resolved', False),
            'report': report,
        }
