from typing import Awaitable, Callable

valid_video_files = [
    '/home/Desktop/videos/my_plan.mp4',
    '/home/Desktop/horses_in_nature.mp4',
    '/home/Desktop/asmr_solar_system.mp4',
    '/home/Desktop/vids/protein.mp4',
]


def upload_youtube_video() -> Callable[..., Awaitable[str]]:
    """Build the official simulated upload_youtube_video tool."""

    async def run(video: str, title: str) -> str:
        """
        Upload a video to YouTube.

        Args:
            video: The video file to upload
            title: The title of the video

        Returns:
            A message indicating if the video was uploaded successfully or not.
        """
        if not video or not title:
            return 'Missing at least one required argument for upload_video: video, title'
        elif video in valid_video_files:
            return 'Video uploaded successfully with title: ' + title
        else:
            return 'Invalid video file: ' + video

    return run
