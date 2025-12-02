import os

from evalscope.constants import VisualizerType
from evalscope.perf.arguments import Arguments
from evalscope.utils.logger import get_logger

logger = get_logger()


class StepCounter:
    """Global step counter for ClearML logging."""

    def __init__(self):
        self._step = 0

    def increment(self) -> int:
        """Increment and return current step."""
        current = self._step
        self._step += 1
        return current

    def reset(self):
        """Reset counter to 0."""
        self._step = 0


# Global step counter instance
_clearml_step_counter = StepCounter()


def init_visualizer(args: Arguments) -> None:
    # Initialize wandb and swanlab
    visualizer = args.visualizer
    if visualizer is None:
        if args.wandb_api_key is not None:
            visualizer = VisualizerType.WANDB
            logger.warning('--wandb-api-key is deprecated. Please use `--visualizer wandb` instead.')
        elif args.swanlab_api_key is not None:
            visualizer = VisualizerType.SWANLAB
            logger.warning('--swanlab-api-key is deprecated. Please use `--visualizer swanlab` instead.')
    args.visualizer = visualizer
    if visualizer == VisualizerType.WANDB:
        init_wandb(args)
    elif visualizer == VisualizerType.SWANLAB:
        init_swanlab(args)
    elif visualizer == VisualizerType.CLEARML:
        init_clearml(args)


def maybe_log_to_visualizer(args: Arguments, message: dict) -> None:
    visualizer = args.visualizer
    if visualizer == VisualizerType.WANDB:
        import wandb
        wandb.log(message)
    elif visualizer == VisualizerType.SWANLAB:
        import swanlab
        swanlab.log(message)
    elif visualizer == VisualizerType.CLEARML:
        from clearml import Task
        task = Task.current_task()
        if task:
            logger_instance = task.get_logger()
            step = _clearml_step_counter.increment()
            # Log metrics as scalars with auto-incrementing step
            for key, value in message.items():
                if isinstance(value, (int, float)):
                    logger_instance.report_scalar(title='Performance Metrics', series=key, value=value, iteration=step)


def init_wandb(args: Arguments) -> None:
    """
    Initialize WandB for logging.
    """
    import datetime
    try:
        import wandb
    except ImportError:
        raise RuntimeError('Cannot import wandb. Please install it with command: \n pip install wandb')
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_DIR'] = args.outputs_dir
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = args.name if args.name else f'{args.model_id}_{current_time}'

    logging_config = _get_sanitized_config(args)

    if args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)
    wandb.init(project='perf_benchmark', name=name, config=logging_config)


def init_swanlab(args: Arguments) -> None:
    """
    Initialize SwanLab for logging.
    """
    import datetime
    try:
        import swanlab
    except ImportError:
        raise RuntimeError('Cannot import swanlab. Please install it with command: \n pip install swanlab')
    os.environ['SWANLAB_LOG_DIR'] = args.outputs_dir
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = args.name if args.name else f'{args.model_id}_{current_time}'
    swanlab.config.update({'framework': '📏evalscope'})

    logging_config = _get_sanitized_config(args)

    init_kwargs = {
        'project': os.getenv('SWANLAB_PROJ_NAME', 'perf_benchmark'),
        'name': name,
        'config': logging_config,
        'mode': 'local' if args.swanlab_api_key == 'local' else None
    }

    workspace = os.getenv('SWANLAB_WORKSPACE')
    if workspace:
        init_kwargs['workspace'] = workspace

    if isinstance(args.swanlab_api_key, str) and not args.swanlab_api_key == 'local':
        swanlab.login(api_key=args.swanlab_api_key)
    swanlab.init(**init_kwargs)


def init_clearml(args: Arguments) -> None:
    """
    Initialize ClearML for logging.
    """
    import datetime
    try:
        from clearml import Task
    except ImportError:
        raise RuntimeError('Cannot import clearml. Please install it with command: \n pip install clearml')

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    task_name = args.name if args.name else f'{args.model_id}_{current_time}'
    project_name = os.getenv('CLEARML_PROJECT_NAME', 'perf_benchmark')

    logging_config = _get_sanitized_config(args)

    # Reset step counter for new task
    _clearml_step_counter.reset()

    # Initialize ClearML Task
    task = Task.init(
        project_name=project_name, task_name=task_name, task_type=Task.TaskTypes.testing, output_uri=args.outputs_dir
    )

    # Connect configuration parameters
    task.connect(logging_config)

    logger.info(f'ClearML Task initialized: {project_name}/{task_name}')


def _get_sanitized_config(args: Arguments) -> dict:
    """
    Get configuration dict with sensitive information removed.

    Args:
        args: Arguments object containing configuration

    Returns:
        Dict with sensitive keys removed
    """
    config = args.to_dict()
    sensitive_keys = ['api_key', 'wandb_api_key', 'swanlab_api_key']
    for key in sensitive_keys:
        config.pop(key, None)
    return config
