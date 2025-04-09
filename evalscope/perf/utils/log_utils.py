import os

from evalscope.perf.arguments import Arguments


def init_wandb(args: Arguments) -> None:
    """
    Initialize WandB for logging.
    """
    # Initialize wandb if the api key is provided
    import datetime
    try:
        import wandb
    except ImportError:
        raise RuntimeError('Cannot import wandb. Please install it with command: \n pip install wandb')
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_DIR'] = args.outputs_dir

    wandb.login(key=args.wandb_api_key)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = args.name if args.name else f'{args.model_id}_{current_time}'
    wandb.init(project='perf_benchmark', name=name, config=args.to_dict())


def init_swanlab(args: Arguments) -> None:
    import datetime
    try:
        import swanlab
    except ImportError:
        raise RuntimeError('Cannot import swanlab. Please install it with command: \n pip install swanlab')
    os.environ['SWANLAB_LOG_DIR'] = args.outputs_dir
    if not args.swanlab_api_key == 'local':
        swanlab.login(api_key=args.swanlab_api_key)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = args.name if args.name else f'{args.model_id}_{current_time}'
    swanlab.config.update({'framework': '📏evalscope'})
    swanlab.init(
        project='perf_benchmark',
        name=name,
        config=args.to_dict(),
        mode='local' if args.swanlab_api_key == 'local' else None)
