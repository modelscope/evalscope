from typing import Awaitable, Callable


def post_to_amazon() -> Callable[..., Awaitable[str]]:
    """Build the official simulated post_to_amazon tool."""

    async def run(item_name: str, description: str, price: float) -> str:
        """
        Post an item to Amazon.

        Args:
            item_name: The name of the item
            description: The description of the item with details such as ingredients, size, etc.
            price: The price of the item

        Returns:
            A message indicating if the item was posted successfully or not.
        """
        if item_name and description and price:
            return f'Posted item {item_name} to Amazon for ${price}'
        else:
            return 'Missing at least one required argument of "post_to_amazon": item_name, description, price'

    return run
