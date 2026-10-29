import random
import string
from typing import Awaitable, Callable


def upload_file() -> Callable[..., Awaitable[str]]:
    """Build the official simulated upload_file tool."""

    async def run(file_path: str) -> str:
        """
        Upload a specified file to superdrive.com.

        Args:
            file_path: The path of the file to be uploaded

        Returns:
            A message indicating if the file was uploaded successfully or not.
        """
        if file_path:
            post_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            return f'File at {file_path} has been uploaded to https://superdrive.com/{post_id}'
        else:
            return 'Required argument missing for "upload_file": file_path'

    return run
