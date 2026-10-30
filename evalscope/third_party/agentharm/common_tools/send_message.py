from typing import Awaitable, Callable


def send_message() -> Callable[..., Awaitable[str]]:
    """Build the official simulated send_message tool."""

    async def run(id: str, content: str) -> str:
        """
        Send a message to a user.

        Args:
            id: The username of the recipient (starting with '@')
            content: Message content

        Returns:
            A message indicating if the message was sent successfully or not.
        """
        if id and content:
            return f'Message has been sent successfully to the conversation with ID: {id}'
        else:
            return 'Missing at least one required argument of send_message: id, content'

    return run
