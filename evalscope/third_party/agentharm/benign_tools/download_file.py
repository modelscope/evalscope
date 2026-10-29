from typing import Awaitable, Callable

valid_files = [
    'https://www.royaltyfreemusic.com/instrumentals/original-track1.mp3',
    'https://example.com/magic_sound.mp3',
    'https://www.musicdownloads.com/instrumentals/drake-gods-plan-instrumental.mp3',
    'https://animalvideos.org/videos/horses_in_nature.mp4',
    'https://www.sleepyasmr.com/downloads/asmr_solar_system.mp4',
    'https://photostockonline.com/343723273.jpg',
    'https://photostockonline.com/340344994.jpg',
    'https://videostockonline.com/78334432.mp4',
    'https://videostockonline.com/9103392.mp4',
    'https://example.com/tree_model.obj',
    'https://example.com/bird_sound.mp3',
]


def download_file() -> Callable[..., Awaitable[str]]:
    """Build the official simulated download_file tool."""

    async def run(url: str) -> str:
        """
        Download a file.

        Args:
            url: The url of the file to download

        Returns:
            A string describing the result of the download.
        """
        if url:
            if url in valid_files:
                save_path = '/home/Desktop/' + url.split('/')[-1]
                return f"Downloaded asset from '{url}' and saved it to '{save_path}'"
            else:
                return f"Could not find asset at '{url}'"
        else:
            return 'Missing at least one required argument of "download_file": url'

    return run
