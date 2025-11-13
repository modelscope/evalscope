from typing import Dict, List, Literal

from evalscope.api.dataset import Dataset, Sample
from evalscope.utils.logger import get_logger

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

    extra_build_instance_images_kwargs = {'tag': LATEST}

    # Code copied from the swe_bench repository
    docker_client = DockerClient.from_env()

    def _get_available_docker_images() -> List[str]:
        return [tag for image in docker_client.images.list() for tag in image.tags]

    # The swebench library requires a huggingface version of the code to be loaded in order to build the images.
    # We load the dataset and then use the library to build the images.
    samples_hf: List[SWEbenchInstance] = [sample_to_hf(s) for s in samples]

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
        successfully_pulled = []

        for sample in samples_to_build_images_for:
            instance_id = sample['instance_id']
            image_name = id_to_docker_image[instance_id]
            # Extract just the image name without the tag
            image_base_name = image_name.split(':')[0]

            try:
                logger.info(f'Pulling {image_name}...')
                docker_client.images.pull(image_name)
                # Tag the pulled image with the expected name
                docker_client.api.tag(image_name, image_base_name, 'latest')
                successfully_pulled.append(instance_id)
                logger.info(f'Successfully pulled {image_name}')
            except Exception as e:
                logger.warning(f'Failed to pull {image_name}: {e}')

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


def sample_to_hf(sample: Sample) -> Dict[str, str]:
    assert sample.metadata is not None
    return {
        'problem_statement': str(sample.input),
        'base_commit': sample.metadata['base_commit'],
        'instance_id': sample.metadata['instance_id'],
        'patch': sample.metadata['patch'],
        'PASS_TO_PASS': sample.metadata['PASS_TO_PASS'],
        'FAIL_TO_PASS': sample.metadata['FAIL_TO_PASS'],
        'test_patch': sample.metadata['test_patch'],
        'version': sample.metadata['version'],
        'repo': sample.metadata['repo'],
        'environment_setup_commit': sample.metadata['environment_setup_commit'],
        'hints_text': sample.metadata['hints_text'],
        'created_at': sample.metadata['created_at'],
    }
