from typing import Awaitable, Callable


def dm_reddit_user() -> Callable[..., Awaitable[str]]:
    """Build the official simulated dm_reddit_user tool."""

    async def run(username: str, content: str) -> str:
        """
        Send a direct message to a Reddit user.

        Args:
            username: The username of the Reddit user to message
            content: The content of the message

        Returns:
            A string describing the direct message sent.
        """
        if all((key for key in [username, content])):
            return f'Sent message to {username} with content: {content}'
        else:
            return 'Missing at least one required argument of "dm_reddit_user": username, content'

    return run
