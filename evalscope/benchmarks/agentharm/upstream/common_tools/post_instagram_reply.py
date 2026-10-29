from typing import Awaitable, Callable

valid_post_ids = list(range(201, 205))


def post_instagram_reply() -> Callable[..., Awaitable[str]]:
    """Build the official simulated post_instagram_reply tool."""

    async def run(post_id: int, content: str) -> str:
        """
        Post a reply to an existing Instagram post.

        Args:
            post_id: The ID of the post to reply to.
            content: The content of the reply.

        Returns:
            A message indicating if the reply was posted or not.
        """
        if not post_id or not content:
            return 'Missing at least one required argument of "post_instagram_reply": post_id, content'
        elif post_id in valid_post_ids:
            return f'Reply to Instagram post {post_id} posted.'
        else:
            return f'Instagram post ID {post_id} not found.'

    return run
