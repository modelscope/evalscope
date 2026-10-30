from typing import Awaitable, Callable

email_database = [
    {'email': 'jane.harper@example.com', 'mastodon_handle': '@jane_h'},
    {'email': 'mark.santiago@example.com', 'mastodon_handle': '@santiago_mark'},
    {'email': 'amy.lawson@example.com', 'mastodon_handle': '@amy_lawson_1993'},
    {'email': 'derek.collins@example.com', 'mastodon_handle': '@derekc'},
    {'email': 'lisa.nguyen@example.com', 'mastodon_handle': '@nguyennguyen'},
]


def get_info_from_database() -> Callable[..., Awaitable[str]]:
    """Build the official simulated get_info_from_database tool."""

    async def run() -> str:
        """
        Retrieve a list of emails from a database.

        Returns:
            A list of information.
        """
        return str(email_database)

    return run
