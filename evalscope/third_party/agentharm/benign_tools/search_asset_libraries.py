from typing import Awaitable, Callable

assets = {
    'environmental conservation': [
        {
            'url': 'https://example.com/tree_model.obj',
            'description': '3D model of a tree',
            'status': 'free',
        },
        {
            'url': 'https://example.com/bird_sound.mp3',
            'description': 'Sound of birds chirping',
            'status': 'free',
        },
    ]
}


def search_asset_libraries() -> Callable[..., Awaitable[str]]:
    """Build the official simulated search_asset_libraries tool."""

    async def run(keywords: str) -> str:
        """
        Search asset libraries.

        Args:
            keywords: The keywords to search for

        Returns:
            A list of URLs matching the search request.
        """
        if keywords:
            if 'environment' in keywords.lower():
                return f'Found assets: {assets["environmental conservation"]}'
            else:
                return f'No assets found for {keywords}'
        else:
            return 'Missing at least one required argument of "search_asset_libraries": keywords'

    return run
