# flake8: noqa
import os


def setup_env(args):
    compat_dsw_gradio(args)


def compat_dsw_gradio(args) -> None:
    if ('JUPYTER_NAME' in os.environ) and ('dsw-'
                                           in os.environ['JUPYTER_NAME']) and ('GRADIO_ROOT_PATH' not in os.environ):
        os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/{args.server_port}"
