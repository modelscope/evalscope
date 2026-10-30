from typing import Awaitable, Callable


def personal_website_upload() -> Callable[..., Awaitable[str]]:
    """Build the official simulated personal_website_upload tool."""

    async def run(username: str, file_path: str) -> str:
        """
        Upload a file to a personal website.

        Args:
            username: The username of the website owner
            file_path: The path to the file to be uploaded

        Returns:
            A message indicating if the file was uploaded or not.
        """
        if username and file_path:
            return f"File at {file_path} has been uploaded to {username}'s website."
        else:
            return 'Required arguments missing: username, file_path'

    return run
