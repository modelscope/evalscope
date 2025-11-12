"""
Main application module for the Evalscope dashboard.
"""
import argparse

from evalscope.utils.logger import configure_logging
from .arguments import add_argument
from .ui import create_app_ui
from .utils.env_utils import setup_env


def create_app(args: argparse.Namespace):
    """
    Create and launch the Evalscope dashboard application.

    Args:
        args: Command line arguments.
    """
    configure_logging(debug=args.debug)

    setup_env(args)

    demo = create_app_ui(args)

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        debug=args.debug,
        allowed_paths=args.allowed_paths,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    create_app(args)
