from typing import Awaitable, Callable


def upload_tiktok_video() -> Callable[..., Awaitable[str]]:
    """Build the official simulated upload_tiktok_video tool."""

    async def run(video_path: str, description: str, tags: list[str]) -> str:
        """
        Upload a video to TikTok with description and tags.

        Args:
            video_path: The file path of the video to be uploaded
            description: The description for the TikTok video
            tags: A list of tags for the TikTok video

        Returns:
            A message indicating if the video was uploaded successfully or not.
        """
        required_args = [video_path, description, tags]
        missing_args = [str(arg) for arg in required_args if not arg]
        if missing_args:
            return f'Missing required arguments: {", ".join(missing_args)}'
        return f"""Video uploaded with description "{description}" and tags {tags}. Video path: {video_path}"""

    return run
