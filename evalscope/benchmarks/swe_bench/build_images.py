import traceback
from typing import TYPE_CHECKING, Dict, List, Literal

from evalscope.api.dataset import Dataset
from evalscope.utils.function_utils import run_in_threads_with_progress
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from swebench.harness.test_spec.test_spec import TestSpec

logger = get_logger()


def build_images(
    samples: Dataset,
    max_workers: int = 4,
    force_rebuild: bool = False,
    use_remote_images: bool = True,
    force_arch: Literal['', 'arm64', 'x86_64'] = '',
) -> Dict[str, str]:
    """This function uses the swe_bench library to build the docker images for the SWE-bench dataset.

    It can also try to pull images from a registry before building them locally.

    Args:
        samples (Dataset): The dataset to build the images for
        max_workers (int): The maximum number of workers to use for building images. Defaults to 4.
        force_rebuild (bool, optional): Whether to force a rebuild of the images. Defaults to False.
        use_remote_images (bool, optional): Whether to try pulling images from Docker Hub before building locally. Defaults to True. See https://hub.docker.com/u/swebench
        force_arch (str, optional): Optionally force the docker images to be pulled/built for a specific architecture. Defaults to "".
    """  # noqa: E501
    from docker.client import DockerClient  # type: ignore
    from swebench.harness.constants import LATEST, SWEbenchInstance  # type: ignore
    from swebench.harness.docker_build import build_instance_images  # type: ignore
    from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

    extra_build_instance_images_kwargs = {'tag': LATEST, 'env_image_tag': LATEST}

    # Code copied from the swe_bench repository
    docker_client = DockerClient.from_env()

    def _get_available_docker_images() -> List[str]:
        return [tag for image in docker_client.images.list() for tag in image.tags]

    # The swebench library requires a huggingface version of the code to be loaded in order to build the images.
    # We load the dataset and then use the library to build the images.
    samples_hf: List[SWEbenchInstance] = [s.metadata for s in samples]

    # We also keep a mapping from instance_ids to the name of the docker image
    id_to_docker_image: Dict[str, str] = {}

    # Note that remote images are named eg "sphinx-doc_1776_sphinx-11502"
    namespace = 'swebench' if use_remote_images else None

    for swebench_instance in samples_hf:
        test_spec = make_test_spec(swebench_instance, namespace=namespace)
        test_spec.arch = force_arch or test_spec.arch
        docker_image_name = test_spec.instance_image_key
        id_to_docker_image[swebench_instance['instance_id']] = docker_image_name

    # Get list of locally available Docker images
    available_docker_images = _get_available_docker_images()
    samples_to_build_images_for = [
        s for s in samples_hf if id_to_docker_image[s['instance_id']] not in available_docker_images
    ]

    # Try to pull images from Docker Hub first if requested
    if use_remote_images and len(samples_to_build_images_for) > 0:
        logger.info(f'Attempting to pull {len(samples_to_build_images_for)} SWE-BENCH images from Docker Hub')
        successfully_pulled: List[str] = []

        def _pull_image(sample) -> str:
            """Pull a single SWE-Bench image; returns instance_id on success."""
            instance_id: str = sample['instance_id']
            image_name: str = id_to_docker_image[instance_id]
            image_base_name: str = image_name.split(':')[0]
            logger.info(f'Pulling {image_name}...')
            for line in docker_client.api.pull(image_name, stream=True, decode=True):
                status = line.get('status')
                progress = line.get('progress')
                if progress:
                    logger.info(f'{image_name} {status} {progress}')
                elif status:
                    logger.info(f'{image_name} {status}')
            docker_client.api.tag(image_name, image_base_name, 'latest')
            logger.info(f'Successfully pulled {image_name}')
            return instance_id

        def _on_error(sample, exc: Exception) -> None:
            image_name = id_to_docker_image[sample['instance_id']]
            logger.warning(f'Failed to pull {image_name}: {exc}')

        def _on_result(sample, instance_id: str) -> None:
            successfully_pulled.append(instance_id)

        run_in_threads_with_progress(
            items=samples_to_build_images_for,
            worker=_pull_image,
            desc='Pulling SWE-Bench images',
            max_workers=max_workers,
            log_interval=30,
            on_result=_on_result,
            on_error=_on_error,
            filter_none_results=True,
        )
        logger.info(f'Pulled {len(successfully_pulled)} images from Docker Hub')

        # Remove successfully pulled images from the build list
        samples_to_build_images_for = [
            s for s in samples_to_build_images_for if s['instance_id'] not in successfully_pulled
        ]

        # Update available images list
        available_docker_images = _get_available_docker_images()

    # Build any remaining images locally
    if len(samples_to_build_images_for) > 0:
        logger.warning('BUILDING SWE-BENCH IMAGES. NOTE: This can take a long time.')
        build_instance_images(
            client=docker_client,
            dataset=samples_hf,
            force_rebuild=force_rebuild,
            max_workers=max_workers,
            **extra_build_instance_images_kwargs,
        )

    # Check that all the images were built
    available_docker_images = _get_available_docker_images()
    missing_images = [
        id_to_docker_image[s['instance_id']]
        for s in samples_hf
        if id_to_docker_image[s['instance_id']] not in available_docker_images
    ]
    assert len(missing_images) == 0, (f'Not all images were built: {missing_images}, {id_to_docker_image}')

    return id_to_docker_image


def build_container(
    test_spec: 'TestSpec',
    client,
):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
    """
    from swebench.harness.constants import DOCKER_USER
    from swebench.harness.docker_utils import cleanup_container

    # Build corresponding instance image
    container = None
    try:
        # Create the container
        logger.info(f'Creating container for {test_spec.instance_id}...')

        # Define arguments for running the container
        run_args = test_spec.docker_specs.get('run_args', {})
        cap_add = run_args.get('cap_add', [])

        container = client.containers.create(
            image=test_spec.instance_image_key,
            user=DOCKER_USER,
            detach=True,
            command='tail -f /dev/null',
            platform=test_spec.platform,
            cap_add=cap_add,
        )
        logger.info(f'Container for {test_spec.instance_id} created: {container.id}')
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f'Error creating container for {test_spec.instance_id}: {e}')
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise e
