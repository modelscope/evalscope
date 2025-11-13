import platform


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
